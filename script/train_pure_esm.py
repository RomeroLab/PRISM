#!/usr/bin/env python
# coding: utf-8

"""
Vanilla ESM2 fine-tuning baseline for controlled comparison with PRISM.

Trains ESM2 with standard MLM (no PRISM modifications) to isolate
the contribution of GL/NGL factorization.

Differences from PRISM (train_esm.py):
- Standard 33-token ESM2 vocabulary (no custom lowercase tokens)
- Standard ESM2 LM head (no origin head, no alpha gating, no dual AA heads)
- Uniform 15% MLM masking (no NGL-targeted or region-based masking)
- Standard cross-entropy loss (no focal loss, no region balancing)
- Standard GELU activation (no SwiGLU replacement)

Everything else is matched:
- Same model architecture (ESM2-35M, 12 layers)
- Same training data (OAS antibody sequences)
- Same 2-stage training (pretrain -> finetune)
- Same optimizer (AdamW with cosine warmup)
- Same hyperparameters (LR, batch size, steps, gradient accumulation)

Usage:
    python train_pure_esm.py --config configs/v_baseline_pretrain.yaml
    python train_pure_esm.py --config configs/v_baseline_finetune.yaml
"""

import argparse
import gc
import os
import pathlib
import random

import numpy as np
import pandas as pd
import torch
import torch.serialization
import pytorch_lightning as pl
import yaml

torch.serialization.add_safe_globals([pathlib.PosixPath])

from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    AutoConfig,
    get_cosine_schedule_with_warmup,
)


# =========================================================================
# Data Utilities
# =========================================================================


def extract_sequences(df, cls_token):
    """Extract antibody sequences from DataFrame.

    Paired:   VH<cls><cls>VL  (same format as PRISM for fair comparison)
    Unpaired: single chain
    """
    has_heavy = "HEAVY_CHAIN_AA_SEQUENCE" in df.columns
    has_light = "LIGHT_CHAIN_AA_SEQUENCE" in df.columns

    sequences = []

    if has_heavy and has_light:
        heavy_vals = df["HEAVY_CHAIN_AA_SEQUENCE"].values
        light_vals = df["LIGHT_CHAIN_AA_SEQUENCE"].values
        for h, l in zip(heavy_vals, light_vals):
            h_ok = pd.notna(h) and h
            l_ok = pd.notna(l) and l
            if h_ok and l_ok:
                sequences.append(f"{h}{cls_token}{cls_token}{l}")
            elif h_ok:
                sequences.append(str(h))
            elif l_ok:
                sequences.append(str(l))
    elif has_heavy:
        sequences = [str(v) for v in df["HEAVY_CHAIN_AA_SEQUENCE"].dropna().values if v]
    elif has_light:
        sequences = [str(v) for v in df["LIGHT_CHAIN_AA_SEQUENCE"].dropna().values if v]
    else:
        raise ValueError(
            "No sequence columns found "
            "(need HEAVY_CHAIN_AA_SEQUENCE or LIGHT_CHAIN_AA_SEQUENCE)"
        )

    return sequences


class PureMLMDataset(Dataset):
    """Minimal dataset: stores sequences, tokenizes on the fly."""

    def __init__(self, sequences, tokenizer, max_length=320):
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.sequences[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            add_special_tokens=True,
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }


def mlm_collate_fn(batch, tokenizer, mask_prob=0.15):
    """Standard MLM collator: uniform masking with 80/10/10 split.

    Only amino acid positions (ESM2 token IDs 4-23) are masked.
    Special tokens (CLS=0, PAD=1, EOS=2, UNK=3) are never masked.
    """
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])

    labels = input_ids.clone()

    # Only mask standard amino acid positions (token IDs 4-23)
    aa_mask = (input_ids >= 4) & (input_ids <= 23)

    # Build masking probability matrix
    prob_matrix = torch.zeros_like(input_ids, dtype=torch.float)
    prob_matrix[aa_mask] = mask_prob
    prob_matrix[attention_mask == 0] = 0.0

    # Sample positions to mask
    masked_indices = torch.bernoulli(prob_matrix).bool()
    labels[~masked_indices] = -100  # CE loss ignores -100

    # 80% -> [MASK] token
    replace_mask = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    input_ids[replace_mask] = tokenizer.mask_token_id

    # 10% -> random amino acid (IDs 4-23)
    random_mask = (
        torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
        & masked_indices
        & ~replace_mask
    )
    random_tokens = torch.randint(4, 24, (random_mask.sum(),), dtype=torch.long)
    input_ids[random_mask] = random_tokens

    # 10% -> keep original (implicit, already in input_ids)

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def _read_seq_columns(path):
    """Read only sequence columns from a parquet file."""
    import pyarrow.parquet as pq

    available = set(pq.ParquetFile(path).schema_arrow.names)
    seq_cols = [
        c
        for c in ["HEAVY_CHAIN_AA_SEQUENCE", "LIGHT_CHAIN_AA_SEQUENCE"]
        if c in available
    ]
    extra_cols = [c for c in ["split"] if c in available]
    return pd.read_parquet(path, columns=seq_cols + extra_cols)


class PureDataModule(pl.LightningDataModule):
    """Data module for sharded (pretrain) and single-file (finetune) data."""

    def __init__(
        self, data_path, tokenizer, batch_size, mask_prob, num_workers, seed,
        val_sample_ratio=0.1,
    ):
        super().__init__()
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.mask_prob = mask_prob
        self.num_workers = num_workers
        self.seed = seed
        self.val_sample_ratio = val_sample_ratio
        self.is_sharded = self.data_path.is_dir()

    def setup(self, stage=None):
        cls_token = self.tokenizer.cls_token

        if self.is_sharded:
            self._setup_sharded(cls_token)
        else:
            self._setup_single(cls_token)

    def _setup_sharded(self, cls_token):
        """Load sharded data with per-GPU shard distribution."""
        rank = self.trainer.global_rank if self.trainer else 0
        world_size = self.trainer.world_size if self.trainer else 1

        shard_files = sorted(self.data_path.glob("train_shard_*.parquet"))
        my_shards = [f for i, f in enumerate(shard_files) if i % world_size == rank]
        print(f"[Rank {rank}] Loading {len(my_shards)}/{len(shard_files)} training shards")

        train_seqs = []
        for shard in my_shards:
            df = _read_seq_columns(shard)
            train_seqs.extend(extract_sequences(df, cls_token))
            del df
        print(f"[Rank {rank}] {len(train_seqs):,} training sequences loaded")

        # Validation (full on each GPU, then sample)
        valid_file = self.data_path / "valid.parquet"
        df_val = _read_seq_columns(valid_file)
        val_seqs = extract_sequences(df_val, cls_token)
        del df_val
        val_seqs = self._sample_validation(val_seqs)
        print(f"[Rank {rank}] {len(val_seqs):,} validation sequences")

        self.train_dataset = PureMLMDataset(train_seqs, self.tokenizer)
        self.val_dataset = PureMLMDataset(val_seqs, self.tokenizer)
        gc.collect()

    def _setup_single(self, cls_token):
        """Load single parquet file with split column."""
        df = _read_seq_columns(self.data_path)

        if "split" not in df.columns:
            raise ValueError("Data must contain 'split' column")

        train_seqs = extract_sequences(df[df["split"] == "train"].copy(), cls_token)
        val_seqs = extract_sequences(df[df["split"] == "valid"].copy(), cls_token)
        del df

        val_seqs = self._sample_validation(val_seqs)
        print(f"Training: {len(train_seqs):,} | Validation: {len(val_seqs):,}")

        self.train_dataset = PureMLMDataset(train_seqs, self.tokenizer)
        self.val_dataset = PureMLMDataset(val_seqs, self.tokenizer)
        gc.collect()

    def _sample_validation(self, sequences):
        """Randomly subsample validation set for faster eval."""
        if self.val_sample_ratio < 1.0 and len(sequences) > 100:
            rng = np.random.RandomState(self.seed)
            n = max(100, int(len(sequences) * self.val_sample_ratio))
            idx = rng.choice(len(sequences), size=n, replace=False)
            sequences = [sequences[i] for i in idx]
        return sequences

    def _collate(self, batch):
        return mlm_collate_fn(batch, self.tokenizer, self.mask_prob)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._collate,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._collate,
        )


# =========================================================================
# Model
# =========================================================================


class PureESM2(pl.LightningModule):
    """
    Vanilla ESM2 with standard MLM head — no PRISM modifications.

    33-token vocabulary, GELU activation, standard cross-entropy loss.
    Same layer unfreezing strategy as PRISM for fair comparison.
    """

    def __init__(
        self,
        model_identifier="esm2_t12_35M_UR50D",
        random_weights=True,
        num_unfrozen_transformer_blocks=12,
        peak_learning_rate=4e-4,
        warmup_steps=500,
        max_steps=10000,
        weight_decay=0.01,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        seed=42,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Seeds
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        # Load model — standard ESM2, no modifications
        if random_weights:
            print(f"[PureESM2] Random-init {model_identifier}")
            cfg = AutoConfig.from_pretrained(f"facebook/{model_identifier}")
            self.model = AutoModelForMaskedLM.from_config(cfg)
        else:
            print(f"[PureESM2] Loading pretrained {model_identifier}")
            self.model = AutoModelForMaskedLM.from_pretrained(
                f"facebook/{model_identifier}"
            )

        # Freeze / unfreeze — same strategy as PRISM
        num_layers = self.model.config.num_hidden_layers
        for p in self.model.parameters():
            p.requires_grad = False

        for name, p in self.model.named_parameters():
            if (
                name.startswith("lm_head")
                or name == "esm.encoder.emb_layer_norm_after"
                or any(
                    name.startswith(f"esm.encoder.layer.{i}")
                    for i in range(
                        num_layers - num_unfrozen_transformer_blocks, num_layers
                    )
                )
            ):
                p.requires_grad = True

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(
            f"[PureESM2] Trainable: {trainable:,} / {total:,} "
            f"({trainable / total * 100:.1f}%)"
        )
        print(
            f"[PureESM2] Unfrozen: last {num_unfrozen_transformer_blocks}"
            f"/{num_layers} layers + LM head + layer norm"
        )

        # Hyperparameters
        self.peak_learning_rate = peak_learning_rate
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.weight_decay = weight_decay
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_epsilon = adam_epsilon

    def training_step(self, batch, batch_idx):
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = outputs.loss
        ppl = torch.exp(loss.detach().clamp(max=10))

        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True,
                 sync_dist=True)
        self.log("train/ppl", ppl, on_step=True, on_epoch=False, prog_bar=False,
                 sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = outputs.loss
        ppl = torch.exp(loss.detach().clamp(max=10))
        bs = batch["input_ids"].size(0)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True,
                 batch_size=bs, sync_dist=True)
        self.log("val/ppl", ppl, on_step=False, on_epoch=True, prog_bar=True,
                 batch_size=bs, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [p for p in self.parameters() if p.requires_grad],
            lr=self.peak_learning_rate,
            betas=(self.adam_beta1, self.adam_beta2),
            eps=self.adam_epsilon,
            weight_decay=self.weight_decay,
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.max_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


# =========================================================================
# Main
# =========================================================================


def main():
    parser = argparse.ArgumentParser(description="Vanilla ESM2 baseline training")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    args = parser.parse_args()

    print("=" * 80)
    print("Vanilla ESM2 Baseline Training (No PRISM Modifications)")
    print("=" * 80)

    # NCCL / CUDA optimizations (same as train_esm.py)
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    os.environ.setdefault("NCCL_NSOCKS_PERTHREAD", "4")
    os.environ.setdefault("NCCL_SOCKET_NTHREADS", "2")
    os.environ.setdefault("NCCL_P2P_DISABLE", "0")
    os.environ.setdefault("NCCL_SHM_DISABLE", "0")
    os.environ.setdefault("NCCL_BUFFSIZE", "8388608")
    os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    print(f"\nConfig: {args.config}")
    print(yaml.dump(config, default_flow_style=False, indent=2))

    # Seed
    seed = config["training"]["seed"]
    pl.seed_everything(seed, workers=True)

    # GPU optimizations
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)

    # Tokenizer — standard ESM2, no custom tokens
    model_id = config["model"]["model_identifier"]
    tokenizer = AutoTokenizer.from_pretrained(f"facebook/{model_id}")
    print(f"\nTokenizer: {model_id}, vocab_size={len(tokenizer)}")

    # Data module
    data_module = PureDataModule(
        data_path=config["data"]["data_path"],
        tokenizer=tokenizer,
        batch_size=config["data"]["batch_size"],
        mask_prob=config["data"]["mask_prob"],
        num_workers=config["data"]["num_workers"],
        seed=seed,
        val_sample_ratio=config["data"].get("val_sample_ratio", 0.1),
    )

    # Model
    model = PureESM2(
        model_identifier=model_id,
        random_weights=config["model"]["random_weights"],
        num_unfrozen_transformer_blocks=config["model"]["num_unfrozen_transformer_blocks"],
        peak_learning_rate=float(config["training"]["peak_learning_rate"]),
        warmup_steps=int(config["training"]["warmup_steps"]),
        max_steps=int(config["training"]["max_steps"]),
        weight_decay=float(config["training"]["weight_decay"]),
        adam_beta1=float(config["training"]["adam_beta1"]),
        adam_beta2=float(config["training"]["adam_beta2"]),
        adam_epsilon=float(config["training"]["adam_epsilon"]),
        seed=seed,
    )

    # Load pretrained checkpoint for fine-tuning stage
    pretrained_path = config.get("pretrained_checkpoint_path")
    if pretrained_path:
        pretrained_path = Path(pretrained_path)
        if not pretrained_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {pretrained_path}")

        print(f"\nLoading pretrained weights: {pretrained_path}")
        ckpt = torch.load(pretrained_path, map_location="cpu")
        missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
        if missing:
            print(f"  Missing keys: {missing}")
        if unexpected:
            print(f"  Unexpected keys: {unexpected}")
        print("  Pretrained weights loaded successfully")
        del ckpt
        torch.cuda.empty_cache()

    # Gradient accumulation & effective batch size
    grad_accum = config["training"]["gradient_accumulation_steps"]
    batch_size = config["data"]["batch_size"]

    num_devices = config["trainer"]["devices"]
    if isinstance(num_devices, list):
        num_devices = len(num_devices)
    elif num_devices in (-1, "auto"):
        num_devices = max(1, torch.cuda.device_count())

    effective_bs = batch_size * grad_accum * num_devices
    print(
        f"\nEffective batch size: {batch_size} x {grad_accum} x {num_devices} "
        f"= {effective_bs}"
    )

    # DDP strategy
    strategy = config["trainer"].get("strategy", "auto")
    if num_devices > 1 and strategy == "auto":
        strategy = "ddp"

    # Eval interval (convert global steps -> batch steps)
    eval_steps = config["training"]["eval_steps"] * grad_accum

    # Logging
    output_dir = Path(config["logging"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    lr_str = f"{float(config['training']['peak_learning_rate']):.0e}"
    experiment_name = (
        f"{config['logging']['experiment_name']}_"
        f"{model_id}_"
        f"unfrozen{config['model']['num_unfrozen_transformer_blocks']}_"
        f"lr{lr_str}_bs{batch_size}"
    )

    logger = TensorBoardLogger(save_dir=output_dir, name=experiment_name)
    print(f"\nTensorBoard: {logger.log_dir}")

    # Checkpointing
    ckpt_dir = output_dir / experiment_name / "checkpoints"
    monitor_metric = config["checkpointing"].get("monitor", "val/ppl")

    checkpoint_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="val_ppl-{epoch:02d}-{val/ppl:.4f}",
        monitor=monitor_metric,
        mode="min",
        save_top_k=config["checkpointing"]["top_k"],
        save_last=True,
        verbose=True,
        auto_insert_metric_name=False,
    )

    callbacks = [checkpoint_cb]

    # Early stopping
    if config["early_stopping"]["enabled"]:
        callbacks.append(
            EarlyStopping(
                monitor=config["early_stopping"]["monitor"],
                patience=config["early_stopping"]["patience"],
                mode="min",
                verbose=True,
            )
        )

    # Trainer
    trainer = pl.Trainer(
        max_steps=config["training"]["max_steps"],
        val_check_interval=eval_steps,
        accumulate_grad_batches=grad_accum,
        gradient_clip_val=config["training"]["gradient_clip_val"],
        devices=config["trainer"]["devices"],
        accelerator=config["trainer"]["accelerator"],
        precision=config["trainer"]["precision"],
        strategy=strategy,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=config["training"]["logging_steps"],
        enable_progress_bar=True,
        # Disable distributed sampler for sharded data
        # (each GPU already has its own shards)
        replace_sampler_ddp=not data_module.is_sharded,
    )

    # Train
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80)
    trainer.fit(model, data_module)
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
