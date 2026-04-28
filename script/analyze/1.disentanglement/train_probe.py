#!/usr/bin/env python
"""
Linear Probing Script for Residue-Level Germline/Non-Germline Classification

This script trains a single linear layer to classify amino acid residues as
Germline (0) or Non-Germline (1) using pre-extracted embeddings from PLMs.

Usage (Single Model):
    python train_probe.py \
        --train_path data/unpaired_OAS/linear_probe_data/train_linear_embeddings.pkl \
        --val_path data/unpaired_OAS/linear_probe_data/val_linear_embeddings.pkl \
        --test_path data/unpaired_OAS/linear_probe_data/test_linear_embeddings.pkl \
        --embedding_col_prefix embed_esm2_35m \
        --input_dim 480 \
        --batch_size 64 \
        --lr 1e-3 \
        --epochs 50

Usage (Ablation Mode - Train on Multiple Embedding Types):
    python train_probe.py \
        --ablation_mode \
        --train_path data/unpaired_OAS/linear_probe_data/train_linear_ablations.pkl \
        --val_path data/unpaired_OAS/linear_probe_data/val_linear_ablations.pkl \
        --test_path data/unpaired_OAS/linear_probe_data/test_linear_ablations.pkl \
        --input_dim 480 \
        --batch_size 64 \
        --epochs 50 \
        --log_dir runs/linear_probe_ablation

    In ablation mode, the script automatically detects all ablation embedding columns
    (e.g., embed_evo_ab_ablation1_h, embed_evo_ab_ablation2_h, embed_evo_ab_ablation3_h)
    and trains a separate probe for each.

Output (Ablation Mode):
    - runs/linear_probe_ablation/ablation1/best_model.pt
    - runs/linear_probe_ablation/ablation2/best_model.pt
    - runs/linear_probe_ablation/ablation3/best_model.pt
    - runs/linear_probe_ablation/ablation_results.json  (aggregated results)
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ==============================================================================
# Ablation Mode Configuration
# ==============================================================================

@dataclass
class AblationEmbeddingConfig:
    """Configuration for a single ablation embedding type."""
    name: str  # e.g., "ablation1", "ablation2", "ablation3"
    embedding_col_prefix: str  # e.g., "embed_evo_ab_ablation1"
    input_dim: int  # Embedding dimension


def detect_ablation_embeddings(
    df: pd.DataFrame,
    pattern: str = r"embed_evo_ab_(ablation\d+)_h"
) -> List[AblationEmbeddingConfig]:
    """
    Auto-detect ablation embedding columns in a DataFrame.

    Args:
        df: DataFrame with embedding columns
        pattern: Regex pattern to match ablation embedding columns.
                 Default matches: embed_evo_ab_ablation1_h, embed_evo_ab_ablation2_h, etc.

    Returns:
        List of AblationEmbeddingConfig objects sorted by ablation number
    """
    configs = []
    embed_cols = [c for c in df.columns if c.startswith("embed_") and c.endswith("_h")]

    for col in embed_cols:
        match = re.search(pattern, col)
        if match:
            ablation_name = match.group(1)  # e.g., "ablation1"
            # Derive the prefix by removing "_h" suffix
            prefix = col[:-2]  # e.g., "embed_evo_ab_ablation1"

            # Get embedding dimension from first non-null entry
            sample_embed = df[col].dropna().iloc[0] if df[col].dropna().shape[0] > 0 else None
            if sample_embed is not None and hasattr(sample_embed, 'shape'):
                input_dim = sample_embed.shape[-1]
            else:
                logger.warning(f"Could not determine input_dim for {col}, using default 480")
                input_dim = 480

            configs.append(AblationEmbeddingConfig(
                name=ablation_name,
                embedding_col_prefix=prefix,
                input_dim=input_dim,
            ))

    # Sort by ablation number
    configs.sort(key=lambda x: int(re.search(r'\d+', x.name).group()) if re.search(r'\d+', x.name) else 0)

    logger.info(f"Detected {len(configs)} ablation embedding types:")
    for cfg in configs:
        logger.info(f"  - {cfg.name}: prefix={cfg.embedding_col_prefix}, dim={cfg.input_dim}")

    return configs


def get_default_ablation_prefixes() -> List[str]:
    """
    Return default ablation embedding prefixes for manual specification.
    """
    return [
        "embed_evo_ab_ablation1",
        "embed_evo_ab_ablation2",
        "embed_evo_ab_ablation3",
    ]


# ==============================================================================
# Utility Functions
# ==============================================================================

def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_mut_positions(mut_str: str) -> List[int]:
    """
    Parse mutation string to get positions (1-indexed).

    Args:
        mut_str: Mutation string like 'S31T;S35T;A50G'

    Returns:
        List of 1-indexed positions with mutations
    """
    if pd.isna(mut_str) or mut_str == '' or str(mut_str) == 'nan':
        return []

    muts = str(mut_str).split(';')
    positions = []
    for m in muts:
        m = m.strip()
        if m:
            # Format is like 'S31T' - extract the number between first and last char
            pos_str = ''.join(c for c in m[1:-1] if c.isdigit())
            if pos_str:
                positions.append(int(pos_str))
    return positions


def create_labels(seq_len: int, mut_positions: List[int]) -> np.ndarray:
    """
    Create binary labels: 0=germline, 1=non-germline.

    Args:
        seq_len: Length of the sequence
        mut_positions: List of 1-indexed mutation positions

    Returns:
        Binary labels array of shape (seq_len,)
    """
    labels = np.zeros(seq_len, dtype=np.int64)
    for pos in mut_positions:
        if 1 <= pos <= seq_len:  # 1-indexed to 0-indexed
            labels[pos - 1] = 1
    return labels


# ==============================================================================
# Dataset
# ==============================================================================

def is_zero_embedding(embed: np.ndarray) -> bool:
    """Check if embedding is all zeros (fallback for failed extraction)."""
    return np.allclose(embed, 0, atol=1e-8)


class ResidueProbeDataset(Dataset):
    """
    Dataset for residue-level linear probing.

    Each sample contains concatenated H+L embeddings and labels.
    Automatically filters out samples with zero embeddings (failed extractions).
    """

    def __init__(
        self,
        data_path: str,
        embedding_col_prefix: str,
        filter_zero_embeddings: bool = True,
    ):
        """
        Args:
            data_path: Path to pickle file with embeddings
            embedding_col_prefix: Prefix for embedding columns (e.g., 'embed_esm2_35m')
                For most models: columns are {prefix}_h and {prefix}_l
                For Sapiens: columns are embed_sapiens_h and embed_sapiens_l (pass 'embed_sapiens')
            filter_zero_embeddings: If True, skip samples where H or L embeddings are all zeros
        """
        logger.info(f"Loading dataset from: {data_path}")
        self.df = pd.read_pickle(data_path)

        # Determine embedding column names
        # Try standard format first: {prefix}_h, {prefix}_l
        self.embedding_col_h = f"{embedding_col_prefix}_h"
        self.embedding_col_l = f"{embedding_col_prefix}_l"

        # Check if columns exist, if not try alternative naming (for Sapiens)
        if self.embedding_col_h not in self.df.columns:
            # Try without additional suffix (Sapiens format: embed_sapiens_h already has _h)
            alt_h = embedding_col_prefix.replace("_h", "") + "_h" if "_h" not in embedding_col_prefix else embedding_col_prefix
            alt_l = embedding_col_prefix.replace("_l", "") + "_l" if "_l" not in embedding_col_prefix else embedding_col_prefix.replace("_h", "_l")

            # Check available embed columns
            embed_cols = [c for c in self.df.columns if c.startswith("embed_")]
            logger.info(f"Available embedding columns: {embed_cols}")

            # Auto-detect H/L columns
            h_cols = [c for c in embed_cols if c.endswith("_h")]
            l_cols = [c for c in embed_cols if c.endswith("_l")]

            if len(h_cols) == 1 and len(l_cols) == 1:
                self.embedding_col_h = h_cols[0]
                self.embedding_col_l = l_cols[0]
                logger.info(f"Auto-detected columns: H={self.embedding_col_h}, L={self.embedding_col_l}")
            else:
                raise ValueError(
                    f"Column {embedding_col_prefix}_h not found. "
                    f"Available columns: {embed_cols}"
                )

        # Final validation
        if self.embedding_col_h not in self.df.columns:
            raise ValueError(f"Column {self.embedding_col_h} not found in data")
        if self.embedding_col_l not in self.df.columns:
            raise ValueError(f"Column {self.embedding_col_l} not found in data")

        original_len = len(self.df)

        # Filter out samples with zero embeddings
        if filter_zero_embeddings:
            valid_indices = []
            skipped_h = 0
            skipped_l = 0
            for idx in range(len(self.df)):
                row = self.df.iloc[idx]
                embed_h = row[self.embedding_col_h]
                embed_l = row[self.embedding_col_l]

                h_is_zero = is_zero_embedding(embed_h)
                l_is_zero = is_zero_embedding(embed_l)

                if h_is_zero:
                    skipped_h += 1
                if l_is_zero:
                    skipped_l += 1

                if not h_is_zero and not l_is_zero:
                    valid_indices.append(idx)

            if len(valid_indices) < original_len:
                self.df = self.df.iloc[valid_indices].reset_index(drop=True)
                logger.warning(
                    f"Filtered out {original_len - len(valid_indices)} samples with zero embeddings "
                    f"(H: {skipped_h}, L: {skipped_l})"
                )

        logger.info(f"Loaded {len(self.df)} samples (original: {original_len})")
        logger.info(f"Using embedding columns: {self.embedding_col_h}, {self.embedding_col_l}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample with concatenated H+L embeddings and labels.

        Returns:
            embeddings: Tensor of shape (L_h + L_l, dim)
            labels: Tensor of shape (L_h + L_l,) with 0/1 values
        """
        row = self.df.iloc[idx]

        # Get embeddings
        embed_h = row[self.embedding_col_h]  # (L_h, dim)
        embed_l = row[self.embedding_col_l]  # (L_l, dim)

        # Convert to float32 tensors
        embed_h = torch.from_numpy(embed_h.astype(np.float32))
        embed_l = torch.from_numpy(embed_l.astype(np.float32))

        # Get sequence lengths
        h_len = len(row['HEAVY_CHAIN_AA_SEQUENCE'])
        l_len = len(row['LIGHT_CHAIN_AA_SEQUENCE'])

        # Parse mutations and create labels
        h_muts = parse_mut_positions(row['hc_mut_codes'])
        l_muts = parse_mut_positions(row['lc_mut_codes'])

        labels_h = create_labels(h_len, h_muts)
        labels_l = create_labels(l_len, l_muts)

        labels_h = torch.from_numpy(labels_h)
        labels_l = torch.from_numpy(labels_l)

        # Concatenate H + L
        embeddings = torch.cat([embed_h, embed_l], dim=0)  # (L_h + L_l, dim)
        labels = torch.cat([labels_h, labels_l], dim=0)    # (L_h + L_l,)

        return embeddings, labels

    def compute_class_weights(self) -> torch.Tensor:
        """
        Compute class weights based on class frequencies.

        Returns:
            Tensor of shape (2,) with weights [w_0, w_1]
        """
        logger.info("Computing class weights from training data...")

        total_gl = 0
        total_ngl = 0

        for idx in tqdm(range(len(self.df)), desc="Computing class weights"):
            row = self.df.iloc[idx]
            h_len = len(row['HEAVY_CHAIN_AA_SEQUENCE'])
            l_len = len(row['LIGHT_CHAIN_AA_SEQUENCE'])

            h_muts = parse_mut_positions(row['hc_mut_codes'])
            l_muts = parse_mut_positions(row['lc_mut_codes'])

            h_labels = create_labels(h_len, h_muts)
            l_labels = create_labels(l_len, l_muts)

            total_gl += (h_labels == 0).sum() + (l_labels == 0).sum()
            total_ngl += (h_labels == 1).sum() + (l_labels == 1).sum()

        # Compute weight for positive class (minority)
        # weight = n_negative / n_positive
        pos_weight = total_gl / total_ngl

        logger.info(f"Class 0 (Germline): {total_gl:,} residues")
        logger.info(f"Class 1 (Non-Germline): {total_ngl:,} residues")
        logger.info(f"Imbalance ratio: {pos_weight:.2f}")

        # Return weights as [1.0, pos_weight]
        return torch.tensor([1.0, pos_weight], dtype=torch.float32)


# ==============================================================================
# Collator
# ==============================================================================

class ResidueProbeCollator:
    """
    Collator for batching variable-length sequences.

    Pads embeddings with 0 and labels with -100 (ignore_index).
    """

    def __init__(self, pad_label: int = -100):
        self.pad_label = pad_label

    def __call__(
        self,
        batch: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Collate a batch of samples.

        Args:
            batch: List of (embeddings, labels) tuples

        Returns:
            padded_embeddings: (B, max_len, dim)
            padded_labels: (B, max_len) with -100 for padding
            attention_mask: (B, max_len) with 1 for valid, 0 for padding
        """
        embeddings_list = [item[0] for item in batch]
        labels_list = [item[1] for item in batch]

        # Pad embeddings with 0
        padded_embeddings = pad_sequence(
            embeddings_list,
            batch_first=True,
            padding_value=0.0
        )  # (B, max_len, dim)

        # Pad labels with -100
        padded_labels = pad_sequence(
            labels_list,
            batch_first=True,
            padding_value=self.pad_label
        )  # (B, max_len)

        # Create attention mask
        lengths = torch.tensor([len(e) for e in embeddings_list])
        max_len = padded_embeddings.size(1)
        attention_mask = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)
        attention_mask = attention_mask.long()  # (B, max_len)

        return padded_embeddings, padded_labels, attention_mask


# ==============================================================================
# Model
# ==============================================================================

class LinearProbe(nn.Module):
    """
    Simple linear probe for residue classification.

    Strictly a single Linear layer with no hidden layers, activations, or dropout.
    """

    def __init__(self, input_dim: int, num_classes: int = 2):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input embeddings of shape (B, L, dim)

        Returns:
            Logits of shape (B, L, num_classes)
        """
        return self.linear(x)


# ==============================================================================
# Trainer
# ==============================================================================

class LinearProbeTrainer:
    """Trainer for linear probe experiments."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader],
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        log_dir: str,
        epochs: int,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.epochs = epochs

        # Setup logging
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

        # Best model tracking
        self.best_val_prauc = 0.0
        self.best_model_path = self.log_dir / "best_model.pt"

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]")
        for batch_idx, (embeddings, labels, attention_mask) in enumerate(pbar):
            embeddings = embeddings.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            logits = self.model(embeddings)  # (B, L, 2)

            # Reshape for loss computation
            B, L, C = logits.shape
            logits_flat = logits.view(-1, C)  # (B*L, 2)
            labels_flat = labels.view(-1)      # (B*L,)

            # Compute loss (CrossEntropy ignores -100)
            loss = self.criterion(logits_flat, labels_flat)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            # Collect predictions for metrics (only non-padded)
            with torch.no_grad():
                probs = torch.softmax(logits, dim=-1)[..., 1]  # (B, L)
                mask = labels != -100

                for b in range(B):
                    valid_mask = mask[b]
                    if valid_mask.sum() > 0:
                        all_preds.extend(probs[b][valid_mask].cpu().numpy())
                        all_labels.extend(labels[b][valid_mask].cpu().numpy())

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Compute metrics
        avg_loss = total_loss / len(self.train_loader)
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        prauc = average_precision_score(all_labels, all_preds)
        f1 = f1_score(all_labels, (all_preds > 0.5).astype(int), zero_division=0)

        return {"loss": avg_loss, "prauc": prauc, "f1": f1}

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, desc: str = "Eval") -> Dict[str, float]:
        """Evaluate on a given loader."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        pbar = tqdm(loader, desc=desc)
        for embeddings, labels, attention_mask in pbar:
            embeddings = embeddings.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            logits = self.model(embeddings)  # (B, L, 2)

            # Reshape for loss computation
            B, L, C = logits.shape
            logits_flat = logits.view(-1, C)
            labels_flat = labels.view(-1)

            loss = self.criterion(logits_flat, labels_flat)
            total_loss += loss.item()

            # Collect predictions
            probs = torch.softmax(logits, dim=-1)[..., 1]  # (B, L)
            mask = labels != -100

            for b in range(B):
                valid_mask = mask[b]
                if valid_mask.sum() > 0:
                    all_preds.extend(probs[b][valid_mask].cpu().numpy())
                    all_labels.extend(labels[b][valid_mask].cpu().numpy())

        # Compute metrics
        avg_loss = total_loss / len(loader)
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        prauc = average_precision_score(all_labels, all_preds)
        f1 = f1_score(all_labels, (all_preds > 0.5).astype(int), zero_division=0)

        return {"loss": avg_loss, "prauc": prauc, "f1": f1}

    def train(self) -> None:
        """Run the full training loop."""
        logger.info(f"Starting training for {self.epochs} epochs")
        logger.info(f"Log directory: {self.log_dir}")

        for epoch in range(self.epochs):
            # Train
            train_metrics = self.train_epoch(epoch)

            # Validate
            val_metrics = self.evaluate(self.val_loader, desc=f"Epoch {epoch+1} [Val]")

            # Log to TensorBoard
            self.writer.add_scalar("Loss/train", train_metrics["loss"], epoch)
            self.writer.add_scalar("Loss/val", val_metrics["loss"], epoch)
            self.writer.add_scalar("PR-AUC/train", train_metrics["prauc"], epoch)
            self.writer.add_scalar("PR-AUC/val", val_metrics["prauc"], epoch)
            self.writer.add_scalar("F1/train", train_metrics["f1"], epoch)
            self.writer.add_scalar("F1/val", val_metrics["f1"], epoch)

            # Print summary
            logger.info(
                f"Epoch {epoch+1}/{self.epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f}, PR-AUC: {train_metrics['prauc']:.4f}, F1: {train_metrics['f1']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f}, PR-AUC: {val_metrics['prauc']:.4f}, F1: {val_metrics['f1']:.4f}"
            )

            # Save best model based on validation PR-AUC
            if val_metrics["prauc"] > self.best_val_prauc:
                self.best_val_prauc = val_metrics["prauc"]
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_prauc": self.best_val_prauc,
                    },
                    self.best_model_path
                )
                logger.info(f"  -> Saved new best model (Val PR-AUC: {self.best_val_prauc:.4f})")

        logger.info(f"Training complete. Best Val PR-AUC: {self.best_val_prauc:.4f}")
        self.writer.close()

    def test(self) -> Dict[str, float]:
        """Run final test evaluation using best model."""
        if self.test_loader is None:
            logger.warning("No test loader provided")
            return {}

        # Load best model
        logger.info(f"Loading best model from {self.best_model_path}")
        checkpoint = torch.load(self.best_model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        # Evaluate on test set
        test_metrics = self.evaluate(self.test_loader, desc="Final Test")

        logger.info("=" * 60)
        logger.info("FINAL TEST RESULTS")
        logger.info("=" * 60)
        logger.info(f"Test Loss: {test_metrics['loss']:.4f}")
        logger.info(f"Test PR-AUC: {test_metrics['prauc']:.4f}")
        logger.info(f"Test F1: {test_metrics['f1']:.4f}")
        logger.info("=" * 60)

        # Log to TensorBoard
        self.writer.add_scalar("Loss/test", test_metrics["loss"], 0)
        self.writer.add_scalar("PR-AUC/test", test_metrics["prauc"], 0)
        self.writer.add_scalar("F1/test", test_metrics["f1"], 0)

        return test_metrics


# ==============================================================================
# Single Probe Training Function (for ablation mode)
# ==============================================================================

def train_single_probe(
    train_path: str,
    val_path: str,
    test_path: Optional[str],
    embedding_col_prefix: str,
    input_dim: int,
    log_dir: str,
    device: torch.device,
    batch_size: int = 64,
    lr: float = 1e-3,
    epochs: int = 50,
    num_workers: int = 4,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Train a single linear probe on a specific embedding type.

    Args:
        train_path: Path to training pickle file
        val_path: Path to validation pickle file
        test_path: Path to test pickle file (optional)
        embedding_col_prefix: Prefix for embedding columns (e.g., 'embed_evo_ab_ablation1')
        input_dim: Dimension of input embeddings
        log_dir: Directory for logs and checkpoints
        device: PyTorch device
        batch_size: Batch size for training
        lr: Learning rate
        epochs: Number of training epochs
        num_workers: Number of data loader workers
        seed: Random seed

    Returns:
        Dictionary containing training results:
        {
            "embedding_prefix": str,
            "input_dim": int,
            "best_val_prauc": float,
            "test_prauc": float (if test set provided),
            "test_f1": float (if test set provided),
            "test_loss": float (if test set provided),
            "log_dir": str,
        }
    """
    # Set seed for reproducibility
    set_seed(seed)

    logger.info(f"\n{'='*60}")
    logger.info(f"Training probe for: {embedding_col_prefix}")
    logger.info(f"{'='*60}")
    logger.info(f"  Input dim: {input_dim}")
    logger.info(f"  Log dir: {log_dir}")

    # Create datasets
    try:
        train_dataset = ResidueProbeDataset(
            data_path=train_path,
            embedding_col_prefix=embedding_col_prefix,
        )

        val_dataset = ResidueProbeDataset(
            data_path=val_path,
            embedding_col_prefix=embedding_col_prefix,
        )

        test_dataset = None
        if test_path:
            test_dataset = ResidueProbeDataset(
                data_path=test_path,
                embedding_col_prefix=embedding_col_prefix,
            )
    except ValueError as e:
        logger.error(f"Failed to load dataset for {embedding_col_prefix}: {e}")
        return {
            "embedding_prefix": embedding_col_prefix,
            "input_dim": input_dim,
            "error": str(e),
        }

    # Compute class weights from training data
    class_weights = train_dataset.compute_class_weights().to(device)

    # Create data loaders
    collator = ResidueProbeCollator(pad_label=-100)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True if device.type == "cuda" else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True if device.type == "cuda" else False,
    )

    test_loader = None
    if test_dataset:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collator,
            pin_memory=True if device.type == "cuda" else False,
        )

    # Create model
    model = LinearProbe(input_dim=input_dim, num_classes=2)
    logger.info(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create optimizer
    optimizer = AdamW(model.parameters(), lr=lr)

    # Create loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)

    # Create trainer
    trainer = LinearProbeTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        log_dir=log_dir,
        epochs=epochs,
    )

    # Train
    trainer.train()

    # Prepare results
    results = {
        "embedding_prefix": embedding_col_prefix,
        "input_dim": input_dim,
        "best_val_prauc": trainer.best_val_prauc,
        "log_dir": log_dir,
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
    }

    # Test (if test set provided)
    if test_loader:
        test_metrics = trainer.test()
        results["test_prauc"] = test_metrics.get("prauc", 0.0)
        results["test_f1"] = test_metrics.get("f1", 0.0)
        results["test_loss"] = test_metrics.get("loss", 0.0)

    # Cleanup
    del model, trainer, train_loader, val_loader, test_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Linear Probing for Residue-Level GL/NGL Classification"
    )

    # Data paths
    parser.add_argument(
        "--train_path",
        type=str,
        required=True,
        help="Path to training pickle file"
    )
    parser.add_argument(
        "--val_path",
        type=str,
        required=True,
        help="Path to validation pickle file"
    )
    parser.add_argument(
        "--test_path",
        type=str,
        default=None,
        help="Path to test pickle file (optional)"
    )

    # Embedding configuration (single model mode)
    parser.add_argument(
        "--embedding_col_prefix",
        type=str,
        default=None,
        help="Prefix for embedding columns (e.g., 'embed_esm2_35m'). Required for single model mode."
    )
    parser.add_argument(
        "--input_dim",
        type=int,
        default=None,
        help="Dimension of input embeddings. Required for single model mode, auto-detected in ablation mode."
    )

    # Ablation mode arguments
    parser.add_argument(
        "--ablation_mode",
        action="store_true",
        help="Enable ablation mode to train probes for all detected ablation embeddings"
    )
    parser.add_argument(
        "--ablation_prefixes",
        type=str,
        nargs="+",
        default=None,
        help="Specific embedding prefixes to train on in ablation mode. "
             "If not provided, auto-detects all ablation embeddings."
    )
    parser.add_argument(
        "--ablation_pattern",
        type=str,
        default=r"embed_evo_ab_(ablation\d+)_h",
        help="Regex pattern to match ablation embedding columns (default: embed_evo_ab_(ablation\\d+)_h)"
    )

    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Logging
    parser.add_argument(
        "--log_dir",
        type=str,
        default="runs/linear_probe",
        help="Directory for TensorBoard logs and checkpoints. "
             "In ablation mode, creates subdirectories for each ablation model."
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )

    # Misc
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loader workers"
    )

    args = parser.parse_args()

    # Validate arguments
    if args.ablation_mode:
        logger.info("Running in ABLATION MODE (sequential training for multiple embeddings)")
    else:
        if args.embedding_col_prefix is None or args.input_dim is None:
            parser.error("--embedding_col_prefix and --input_dim are required in single model mode. "
                        "Use --ablation_mode to auto-detect embeddings.")

    # Set seed
    set_seed(args.seed)

    # Setup device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # =========================================================================
    # ABLATION MODE: Train multiple probes sequentially
    # =========================================================================
    if args.ablation_mode:
        # Load training data to detect embeddings
        logger.info(f"Loading training data to detect ablation embeddings...")
        train_df = pd.read_pickle(args.train_path)

        # Detect or use specified ablation embeddings
        if args.ablation_prefixes:
            # Use manually specified prefixes
            logger.info(f"Using manually specified prefixes: {args.ablation_prefixes}")
            ablation_configs = []
            for prefix in args.ablation_prefixes:
                # Extract ablation name from prefix
                match = re.search(r'(ablation\d+)', prefix)
                name = match.group(1) if match else prefix.replace("embed_", "").replace("_", "-")

                # Get input dim from data
                h_col = f"{prefix}_h"
                if h_col in train_df.columns:
                    sample_embed = train_df[h_col].dropna().iloc[0]
                    input_dim = sample_embed.shape[-1] if hasattr(sample_embed, 'shape') else args.input_dim or 480
                else:
                    input_dim = args.input_dim or 480
                    logger.warning(f"Could not find {h_col}, using input_dim={input_dim}")

                ablation_configs.append(AblationEmbeddingConfig(
                    name=name,
                    embedding_col_prefix=prefix,
                    input_dim=input_dim,
                ))
        else:
            # Auto-detect ablation embeddings
            ablation_configs = detect_ablation_embeddings(train_df, args.ablation_pattern)

        if not ablation_configs:
            logger.error("No ablation embeddings found! Check your data or use --ablation_prefixes.")
            sys.exit(1)

        # Create base log directory
        base_log_dir = Path(args.log_dir)
        base_log_dir.mkdir(parents=True, exist_ok=True)

        # Train probes for each ablation embedding sequentially
        all_results: List[Dict[str, Any]] = []

        for i, ablation_cfg in enumerate(ablation_configs):
            logger.info(f"\n{'#'*70}")
            logger.info(f"ABLATION {i+1}/{len(ablation_configs)}: {ablation_cfg.name}")
            logger.info(f"{'#'*70}")

            # Create separate log directory for this ablation
            ablation_log_dir = str(base_log_dir / ablation_cfg.name)

            # Train probe
            results = train_single_probe(
                train_path=args.train_path,
                val_path=args.val_path,
                test_path=args.test_path,
                embedding_col_prefix=ablation_cfg.embedding_col_prefix,
                input_dim=ablation_cfg.input_dim,
                log_dir=ablation_log_dir,
                device=device,
                batch_size=args.batch_size,
                lr=args.lr,
                epochs=args.epochs,
                num_workers=args.num_workers,
                seed=args.seed,
            )

            # Add ablation name to results
            results["ablation_name"] = ablation_cfg.name
            all_results.append(results)

            # Print intermediate summary
            if "error" not in results:
                logger.info(f"\n[{ablation_cfg.name}] Best Val PR-AUC: {results['best_val_prauc']:.4f}")
                if "test_prauc" in results:
                    logger.info(f"[{ablation_cfg.name}] Test PR-AUC: {results['test_prauc']:.4f}")

        # =====================================================================
        # Aggregate and save results
        # =====================================================================
        logger.info(f"\n{'='*70}")
        logger.info("ABLATION STUDY RESULTS SUMMARY")
        logger.info(f"{'='*70}")

        # Print results table
        logger.info(f"\n{'Model':<20} {'Val PR-AUC':<12} {'Test PR-AUC':<12} {'Test F1':<10}")
        logger.info("-" * 60)
        for r in all_results:
            if "error" in r:
                logger.info(f"{r['ablation_name']:<20} ERROR: {r['error']}")
            else:
                val_prauc = r.get('best_val_prauc', 0.0)
                test_prauc = r.get('test_prauc', 'N/A')
                test_f1 = r.get('test_f1', 'N/A')
                test_prauc_str = f"{test_prauc:.4f}" if isinstance(test_prauc, float) else test_prauc
                test_f1_str = f"{test_f1:.4f}" if isinstance(test_f1, float) else test_f1
                logger.info(f"{r['ablation_name']:<20} {val_prauc:<12.4f} {test_prauc_str:<12} {test_f1_str:<10}")

        # Save aggregated results to JSON
        results_path = base_log_dir / "ablation_results.json"
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        logger.info(f"\nAggregated results saved to: {results_path}")

        # Also save as CSV for easy analysis
        results_df = pd.DataFrame(all_results)
        csv_path = base_log_dir / "ablation_results.csv"
        results_df.to_csv(csv_path, index=False)
        logger.info(f"Results CSV saved to: {csv_path}")

    # =========================================================================
    # SINGLE MODEL MODE: Train one probe (original behavior)
    # =========================================================================
    else:
        # Create datasets
        train_dataset = ResidueProbeDataset(
            data_path=args.train_path,
            embedding_col_prefix=args.embedding_col_prefix,
        )

        val_dataset = ResidueProbeDataset(
            data_path=args.val_path,
            embedding_col_prefix=args.embedding_col_prefix,
        )

        test_dataset = None
        if args.test_path:
            test_dataset = ResidueProbeDataset(
                data_path=args.test_path,
                embedding_col_prefix=args.embedding_col_prefix,
            )

        # Compute class weights from training data
        class_weights = train_dataset.compute_class_weights().to(device)

        # Create data loaders
        collator = ResidueProbeCollator(pad_label=-100)

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collator,
            pin_memory=True if device.type == "cuda" else False,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collator,
            pin_memory=True if device.type == "cuda" else False,
        )

        test_loader = None
        if test_dataset:
            test_loader = DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                collate_fn=collator,
                pin_memory=True if device.type == "cuda" else False,
            )

        # Create model
        model = LinearProbe(input_dim=args.input_dim, num_classes=2)
        logger.info(f"Model: {model}")
        logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Create optimizer
        optimizer = AdamW(model.parameters(), lr=args.lr)

        # Create loss function with class weights
        criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)
        logger.info(f"Class weights: {class_weights.cpu().numpy()}")

        # Create trainer
        trainer = LinearProbeTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            log_dir=args.log_dir,
            epochs=args.epochs,
        )

        # Train
        trainer.train()

        # Test (if test set provided)
        if test_loader:
            trainer.test()

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
