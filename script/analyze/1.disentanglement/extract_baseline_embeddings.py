#!/usr/bin/env python
"""
Extract per-residue embeddings from the PRISM-less (Pure ESM2 fine-tuned) checkpoint.

Loads a Lightning checkpoint (strips "model." prefix) into a HuggingFace ESM-2
model, then runs residue-level embedding extraction on train/val/test splits.

Usage:
    CUDA_VISIBLE_DEVICES=4 python extract_baseline_embeddings.py \
        --checkpoint outputs/ESM2_baseline_finetune_.../checkpoints/val_ppl-53-1.5111.ckpt \
        --input_dir data/unpaired_OAS/linear_probe_data \
        --output_dir data/unpaired_OAS/linear_probe_data \
        --splits train val test \
        --batch_size 128
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.serialization
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer

torch.serialization.add_safe_globals([pathlib.PosixPath])

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def load_baseline_model(checkpoint_path: str, model_identifier: str, device: torch.device):
    """Load Pure ESM2 weights from Lightning checkpoint into HuggingFace AutoModel."""
    logger.info(f"Loading baseline checkpoint: {checkpoint_path}")
    config = AutoConfig.from_pretrained(f"facebook/{model_identifier}")
    model = AutoModel.from_config(config)
    tokenizer = AutoTokenizer.from_pretrained(f"facebook/{model_identifier}")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]

    # Lightning wraps keys as "model.esm.*".
    # AutoModel returns EsmModel (not EsmForMaskedLM), so keys must NOT have
    # the "esm." prefix. Strip both "model." and "esm.".
    cleaned_for_base = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith("model."):
            nk = nk[len("model."):]
        if nk.startswith("esm."):
            nk = nk[len("esm."):]
        # Drop LM head / contact head - AutoModel has neither
        if nk.startswith("lm_head.") or nk.startswith("contact_head."):
            continue
        cleaned_for_base[nk] = v

    missing, unexpected = model.load_state_dict(cleaned_for_base, strict=False)
    if missing:
        logger.warning(f"Missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        logger.warning(f"Unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    model.eval()
    model.to(device)
    del ckpt, state_dict, cleaned_for_base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info(f"Baseline model loaded on {device}")
    return model, tokenizer


@torch.no_grad()
def extract_embeddings(
    sequences: List[str],
    model,
    tokenizer,
    device: torch.device,
    batch_size: int,
    max_length: int = 1024,
    desc: str = "Embedding",
) -> List[np.ndarray]:
    """Extract per-residue last-hidden embeddings. Skips CLS/EOS."""
    out: List[np.ndarray] = []
    for i in tqdm(range(0, len(sequences), batch_size), desc=desc):
        batch = sequences[i : i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        hidden = model(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state  # (B, L, D)

        for j, seq in enumerate(batch):
            L = len(seq)
            emb = hidden[j, 1 : L + 1, :].cpu().float().numpy().astype(np.float16)
            if emb.shape[0] != L:
                if emb.shape[0] < L:
                    pad = np.zeros((L - emb.shape[0], emb.shape[1]), dtype=np.float16)
                    emb = np.vstack([emb, pad])
                else:
                    emb = emb[:L]
            out.append(emb)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to Lightning .ckpt")
    parser.add_argument("--model_identifier", default="esm2_t12_35M_UR50D")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--embed_prefix", default="embed_baseline",
                        help="Output column prefix; will produce {prefix}_h and {prefix}_l")
    args = parser.parse_args()

    device = torch.device(args.device)
    model, tokenizer = load_baseline_model(args.checkpoint, args.model_identifier, device)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        in_path = input_dir / f"{split}_linear.pkl"
        if not in_path.exists():
            logger.warning(f"Skipping {split}: {in_path} not found")
            continue

        logger.info(f"\n{'='*60}\nProcessing split: {split}\n{'='*60}")
        df = pd.read_pickle(in_path)
        logger.info(f"Loaded {len(df)} rows from {in_path}")

        heavy_seqs = df["HEAVY_CHAIN_AA_SEQUENCE"].tolist()
        light_seqs = df["LIGHT_CHAIN_AA_SEQUENCE"].tolist()

        emb_h = extract_embeddings(
            heavy_seqs, model, tokenizer, device, args.batch_size, desc=f"{split}-H"
        )
        emb_l = extract_embeddings(
            light_seqs, model, tokenizer, device, args.batch_size, desc=f"{split}-L"
        )

        out_df = df.copy()
        out_df[f"{args.embed_prefix}_h"] = emb_h
        out_df[f"{args.embed_prefix}_l"] = emb_l

        out_path = output_dir / f"{split}_linear_baseline.pkl"
        out_df.to_pickle(out_path)
        logger.info(f"Saved: {out_path} (added {args.embed_prefix}_h/_l)")

    logger.info("\nAll splits complete.")


if __name__ == "__main__":
    main()
