#!/usr/bin/env python
"""
Linear Probe Inference Script for GL/NGL Classification

This script runs inference on trained linear probes for multiple PLMs and
saves per-residue probabilities to calculate F1-score and PR-AUC.

Standard Mode (6 PLMs):
    python inference_linear_probes.py \
        --output_path results/linear_probe_predictions.pkl

Ablation Mode (4 models: PRISM Full + 3 ablations):
    python inference_linear_probes.py \
        --ablation_mode \
        --ablation_data_file data/unpaired_OAS/linear_probe_data/test_linear_ablations.pkl \
        --original_embed_file data/unpaired_OAS/linear_probe_data/test_linear_evo_ab.pkl \
        --output_path results/linear_probe_predictions_ablation.pkl

Output:
    A pickle file containing a DataFrame with columns:
    - Original test data columns
    - {model}_prob_h, {model}_prob_l: Per-residue NGL probabilities for each model
    - {model}_pred_h, {model}_pred_l: Per-residue binary predictions (threshold=0.5)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ==============================================================================
# Model Configuration
# ==============================================================================

MODEL_CONFIGS = {
    "evo_ab": {
        "checkpoint_path": "runs/linear_probe/best_model.pt",
        "embedding_prefix": "embed_evo_ab",
        "input_dim": 480,
        "data_suffix": "evo_ab",
    },
    "esm2_35m": {
        "checkpoint_path": "runs/linear_probe/esm2_35m/best_model.pt",
        "embedding_prefix": "embed_esm2_35m",
        "input_dim": 480,
        "data_suffix": "esm2_35m",
    },
    "esm2_650m": {
        "checkpoint_path": "runs/linear_probe/esm2_650m/best_model.pt",
        "embedding_prefix": "embed_esm2_650m",
        "input_dim": 1280,
        "data_suffix": "esm2_650m",
    },
    "ablang2": {
        "checkpoint_path": "runs/linear_probe/ablang2/best_model.pt",
        "embedding_prefix": "embed_ablang2",
        "input_dim": 480,
        "data_suffix": "ablang2",
    },
    "antiberty": {
        "checkpoint_path": "runs/linear_probe/antiberty/best_model.pt",
        "embedding_prefix": "embed_antiberty",
        "input_dim": 512,
        "data_suffix": "antiberty",
    },
    "sapiens": {
        "checkpoint_path": "runs/linear_probe/sapiens/best_model.pt",
        "embedding_prefix": "embed_sapiens",
        "input_dim": 128,
        "data_suffix": "sapiens",
    },
}

# Ablation model configurations for 2x2 factorial design:
# ┌─────────────────────┬───────────────────┬───────────────────┐
# │                     │ Multihead (Full)  │ Simple LM Head    │
# ├─────────────────────┼───────────────────┼───────────────────┤
# │ With Pretraining    │ PRISM Full (best) │ Ablation 2        │
# ├─────────────────────┼───────────────────┼───────────────────┤
# │ No Pretraining      │ Ablation 1        │ Ablation 3        │
# └─────────────────────┴───────────────────┴───────────────────┘
ABLATION_MODEL_CONFIGS = {
    "evo_ab": {
        "checkpoint_path": "runs/linear_probe/best_model.pt",
        "embedding_prefix": "embed_evo_ab",
        "input_dim": 480,
        "display_name": "PRISM Full\n(Multihead + Pretrain)",
    },
    "ablation1": {
        "checkpoint_path": "runs/linear_probe_ablation/ablation1/best_model.pt",
        "embedding_prefix": "embed_evo_ab_ablation1",
        "input_dim": 480,
        "display_name": "Ablation 1\n(Multihead + No Pretrain)",
    },
    "ablation2": {
        "checkpoint_path": "runs/linear_probe_ablation/ablation2/best_model.pt",
        "embedding_prefix": "embed_evo_ab_ablation2",
        "input_dim": 480,
        "display_name": "Ablation 2\n(Simple + Pretrain)",
    },
    "ablation3": {
        "checkpoint_path": "runs/linear_probe_ablation/ablation3/best_model.pt",
        "embedding_prefix": "embed_evo_ab_ablation3",
        "input_dim": 480,
        "display_name": "Ablation 3\n(Simple + No Pretrain)",
    },
}


# ==============================================================================
# Utility Functions
# ==============================================================================

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
        if 1 <= pos <= seq_len:
            labels[pos - 1] = 1
    return labels


def is_zero_embedding(embed: np.ndarray) -> bool:
    """Check if embedding is all zeros (fallback for failed extraction)."""
    return np.allclose(embed, 0, atol=1e-8)


# ==============================================================================
# Model
# ==============================================================================

class LinearProbe(nn.Module):
    """Simple linear probe for residue classification."""

    def __init__(self, input_dim: int, num_classes: int = 2):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


# ==============================================================================
# Inference
# ==============================================================================

def load_model(checkpoint_path: str, input_dim: int, device: torch.device) -> LinearProbe:
    """Load a trained linear probe model."""
    model = LinearProbe(input_dim=input_dim, num_classes=2)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def run_inference_for_model(
    model: LinearProbe,
    df_embed: pd.DataFrame,
    df_base: pd.DataFrame,
    embed_prefix: str,
    device: torch.device,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Run inference for a single model.

    Returns:
        probs_h, probs_l: Per-residue NGL probabilities for heavy/light chains
        preds_h, preds_l: Per-residue binary predictions
    """
    probs_h_list = []
    probs_l_list = []
    preds_h_list = []
    preds_l_list = []

    embed_col_h = f"{embed_prefix}_h"
    embed_col_l = f"{embed_prefix}_l"

    with torch.no_grad():
        for idx in range(len(df_embed)):
            embed_h = df_embed.iloc[idx][embed_col_h]
            embed_l = df_embed.iloc[idx][embed_col_l]

            # Handle zero embeddings (failed extraction)
            if is_zero_embedding(embed_h):
                h_len = len(df_base.iloc[idx]['HEAVY_CHAIN_AA_SEQUENCE'])
                probs_h_list.append(np.full(h_len, np.nan))
                preds_h_list.append(np.full(h_len, -1, dtype=np.int64))
            else:
                embed_h_tensor = torch.from_numpy(embed_h.astype(np.float32)).to(device)
                logits_h = model(embed_h_tensor)
                probs_h = torch.softmax(logits_h, dim=-1)[:, 1].cpu().numpy()
                preds_h = (probs_h > 0.5).astype(np.int64)
                probs_h_list.append(probs_h)
                preds_h_list.append(preds_h)

            if is_zero_embedding(embed_l):
                l_len = len(df_base.iloc[idx]['LIGHT_CHAIN_AA_SEQUENCE'])
                probs_l_list.append(np.full(l_len, np.nan))
                preds_l_list.append(np.full(l_len, -1, dtype=np.int64))
            else:
                embed_l_tensor = torch.from_numpy(embed_l.astype(np.float32)).to(device)
                logits_l = model(embed_l_tensor)
                probs_l = torch.softmax(logits_l, dim=-1)[:, 1].cpu().numpy()
                preds_l = (probs_l > 0.5).astype(np.int64)
                probs_l_list.append(probs_l)
                preds_l_list.append(preds_l)

    return probs_h_list, probs_l_list, preds_h_list, preds_l_list


def compute_metrics(
    df: pd.DataFrame,
    model_name: str,
) -> Dict[str, float]:
    """
    Compute F1 and PR-AUC for a model's predictions.

    Args:
        df: DataFrame with predictions and labels
        model_name: Name of the model

    Returns:
        Dictionary with metrics
    """
    all_probs = []
    all_labels = []
    all_preds = []

    prob_h_col = f"{model_name}_prob_h"
    prob_l_col = f"{model_name}_prob_l"
    pred_h_col = f"{model_name}_pred_h"
    pred_l_col = f"{model_name}_pred_l"

    for idx in range(len(df)):
        row = df.iloc[idx]

        # Heavy chain
        probs_h = row[prob_h_col]
        preds_h = row[pred_h_col]
        h_len = len(row['HEAVY_CHAIN_AA_SEQUENCE'])
        h_muts = parse_mut_positions(row['hc_mut_codes'])
        labels_h = create_labels(h_len, h_muts)

        # Skip if embeddings failed (NaN probs)
        if not np.any(np.isnan(probs_h)):
            all_probs.extend(probs_h)
            all_labels.extend(labels_h)
            all_preds.extend(preds_h)

        # Light chain
        probs_l = row[prob_l_col]
        preds_l = row[pred_l_col]
        l_len = len(row['LIGHT_CHAIN_AA_SEQUENCE'])
        l_muts = parse_mut_positions(row['lc_mut_codes'])
        labels_l = create_labels(l_len, l_muts)

        if not np.any(np.isnan(probs_l)):
            all_probs.extend(probs_l)
            all_labels.extend(labels_l)
            all_preds.extend(preds_l)

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    # Compute metrics
    prauc = average_precision_score(all_labels, all_probs)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)

    # Class statistics
    n_total = len(all_labels)
    n_ngl = all_labels.sum()
    n_gl = n_total - n_ngl

    return {
        "model": model_name,
        "pr_auc": prauc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "n_total_residues": n_total,
        "n_ngl": int(n_ngl),
        "n_gl": int(n_gl),
        "ngl_ratio": n_ngl / n_total if n_total > 0 else 0,
    }


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Linear Probe Inference for GL/NGL Classification"
    )
    parser.add_argument(
        "--base_data_path",
        type=str,
        default="data/unpaired_OAS/linear_probe_data/test_linear.pkl",
        help="Path to base test data (with labels)",
    )
    parser.add_argument(
        "--embed_data_dir",
        type=str,
        default="data/unpaired_OAS/linear_probe_data",
        help="Directory containing embedding pickle files",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="runs/linear_probe",
        help="Directory containing model checkpoints",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/unpaired_OAS/linear_probe_data/test_predictions.pkl",
        help="Path to save predictions",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=list(MODEL_CONFIGS.keys()),
        help="Models to run inference on",
    )

    # Ablation mode arguments
    parser.add_argument(
        "--ablation_mode",
        action="store_true",
        help="Enable ablation mode: run inference on 4 ablation models (PRISM Full + 3 ablations)",
    )
    parser.add_argument(
        "--ablation_data_file",
        type=str,
        default="data/unpaired_OAS/linear_probe_data/test_linear_ablations.pkl",
        help="Path to ablation embeddings pickle file (contains ablation1/2/3 embeddings)",
    )
    parser.add_argument(
        "--original_embed_file",
        type=str,
        default=None,
        help="Path to original PRISM Full embeddings (embed_evo_ab_h/l). "
             "If not provided, will try test_linear_evo_ab.pkl in same directory.",
    )

    args = parser.parse_args()

    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Store metrics for summary
    all_metrics = []

    # =========================================================================
    # ABLATION MODE: Single file with multiple embedding types
    # =========================================================================
    if args.ablation_mode:
        logger.info("Running in ABLATION MODE (4 models)")

        # Load ablation embeddings file
        ablation_data_path = Path(args.ablation_data_file)
        if not ablation_data_path.exists():
            raise FileNotFoundError(f"Ablation data file not found: {ablation_data_path}")

        logger.info(f"Loading ablation embeddings from: {ablation_data_path}")
        df_embed = pd.read_pickle(ablation_data_path)
        logger.info(f"Loaded {len(df_embed)} samples")

        # Check available embedding columns
        embed_cols = [c for c in df_embed.columns if c.startswith("embed_") and c.endswith("_h")]
        logger.info(f"Available embedding columns: {embed_cols}")

        # Load original PRISM Full embeddings if needed
        if "embed_evo_ab_h" not in df_embed.columns:
            logger.info("PRISM Full embeddings (embed_evo_ab_h) not found, loading from separate file...")

            if args.original_embed_file:
                original_path = Path(args.original_embed_file)
            else:
                original_path = ablation_data_path.parent / "test_linear_evo_ab.pkl"

            if original_path.exists():
                logger.info(f"Loading original embeddings from: {original_path}")
                df_original = pd.read_pickle(original_path)

                if "embed_evo_ab_h" in df_original.columns:
                    # Merge embedding columns
                    df_embed = df_embed.reset_index(drop=True)
                    df_original = df_original.reset_index(drop=True)

                    if len(df_embed) == len(df_original):
                        for col in ["embed_evo_ab_h", "embed_evo_ab_l"]:
                            if col in df_original.columns:
                                df_embed[col] = df_original[col]
                        logger.info("✓ Merged PRISM Full embeddings")
                    else:
                        logger.warning(f"Row count mismatch: ablation={len(df_embed)}, original={len(df_original)}")
            else:
                logger.warning(f"Original embeddings file not found: {original_path}")

        # Load base test data (for labels)
        logger.info(f"Loading base test data from: {args.base_data_path}")
        df_base = pd.read_pickle(args.base_data_path)
        df_result = df_base.copy()

        # Run inference for each ablation model
        ablation_model_names = list(ABLATION_MODEL_CONFIGS.keys())

        for model_name in ablation_model_names:
            config = ABLATION_MODEL_CONFIGS[model_name]
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing model: {model_name} ({config['display_name'].replace(chr(10), ' ')})")
            logger.info(f"{'='*60}")

            # Check if embedding columns exist
            embed_col_h = f"{config['embedding_prefix']}_h"
            if embed_col_h not in df_embed.columns:
                logger.warning(f"Embedding column {embed_col_h} not found, skipping")
                continue

            # Construct checkpoint path
            checkpoint_path = Path(config["checkpoint_path"])
            if not checkpoint_path.exists():
                logger.warning(f"Checkpoint not found: {checkpoint_path}, skipping")
                continue

            # Load model
            logger.info(f"Loading model from: {checkpoint_path}")
            model = load_model(str(checkpoint_path), config["input_dim"], device)

            # Run inference
            logger.info("Running inference...")
            probs_h, probs_l, preds_h, preds_l = run_inference_for_model(
                model=model,
                df_embed=df_embed,
                df_base=df_base,
                embed_prefix=config["embedding_prefix"],
                device=device,
            )

            # Add predictions to result DataFrame
            df_result[f"{model_name}_prob_h"] = probs_h
            df_result[f"{model_name}_prob_l"] = probs_l
            df_result[f"{model_name}_pred_h"] = preds_h
            df_result[f"{model_name}_pred_l"] = preds_l

            # Compute metrics
            metrics = compute_metrics(df_result, model_name)
            metrics["display_name"] = config["display_name"].replace('\n', ' ')
            all_metrics.append(metrics)

            logger.info(f"  PR-AUC: {metrics['pr_auc']:.4f}")
            logger.info(f"  F1: {metrics['f1']:.4f}")
            logger.info(f"  Precision: {metrics['precision']:.4f}")
            logger.info(f"  Recall: {metrics['recall']:.4f}")

    # =========================================================================
    # STANDARD MODE: Multiple files, 6 PLMs
    # =========================================================================
    else:
        logger.info("Running in STANDARD MODE (6 PLMs)")

        # Load base test data
        logger.info(f"Loading base test data from: {args.base_data_path}")
        df_base = pd.read_pickle(args.base_data_path)
        logger.info(f"Loaded {len(df_base)} samples")

        # Initialize result DataFrame (copy of base)
        df_result = df_base.copy()

        # Run inference for each model
        for model_name in args.models:
            if model_name not in MODEL_CONFIGS:
                logger.warning(f"Unknown model: {model_name}, skipping")
                continue

            config = MODEL_CONFIGS[model_name]
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing model: {model_name}")
            logger.info(f"{'='*60}")

            # Update paths with base directories
            checkpoint_path = Path(args.model_dir) / Path(config["checkpoint_path"]).relative_to("runs/linear_probe")
            if model_name == "evo_ab":
                checkpoint_path = Path(args.model_dir) / "best_model.pt"

            embed_data_path = Path(args.embed_data_dir) / f"test_linear_{config['data_suffix']}.pkl"

            # Check paths exist
            if not checkpoint_path.exists():
                logger.warning(f"Checkpoint not found: {checkpoint_path}, skipping")
                continue
            if not embed_data_path.exists():
                logger.warning(f"Embedding data not found: {embed_data_path}, skipping")
                continue

            # Load model
            logger.info(f"Loading model from: {checkpoint_path}")
            model = load_model(str(checkpoint_path), config["input_dim"], device)

            # Load embedding data
            logger.info(f"Loading embeddings from: {embed_data_path}")
            df_embed = pd.read_pickle(embed_data_path)

            # Run inference
            logger.info("Running inference...")
            probs_h, probs_l, preds_h, preds_l = run_inference_for_model(
                model=model,
                df_embed=df_embed,
                df_base=df_base,
                embed_prefix=config["embedding_prefix"],
                device=device,
            )

            # Add predictions to result DataFrame
            df_result[f"{model_name}_prob_h"] = probs_h
            df_result[f"{model_name}_prob_l"] = probs_l
            df_result[f"{model_name}_pred_h"] = preds_h
            df_result[f"{model_name}_pred_l"] = preds_l

            # Compute metrics
            metrics = compute_metrics(df_result, model_name)
            all_metrics.append(metrics)

            logger.info(f"  PR-AUC: {metrics['pr_auc']:.4f}")
            logger.info(f"  F1: {metrics['f1']:.4f}")
            logger.info(f"  Precision: {metrics['precision']:.4f}")
            logger.info(f"  Recall: {metrics['recall']:.4f}")

    # Save results
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_result.to_pickle(output_path)
    logger.info(f"\nSaved predictions to: {output_path}")

    # Print summary table
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY OF RESULTS")
    logger.info("=" * 80)

    if all_metrics:
        df_metrics = pd.DataFrame(all_metrics)
        df_metrics = df_metrics.sort_values("pr_auc", ascending=False)

        # Pretty print
        print("\n")
        print(f"{'Model':<12} {'PR-AUC':>10} {'F1':>10} {'Precision':>10} {'Recall':>10}")
        print("-" * 56)
        for _, row in df_metrics.iterrows():
            print(
                f"{row['model']:<12} "
                f"{row['pr_auc']:>10.4f} "
                f"{row['f1']:>10.4f} "
                f"{row['precision']:>10.4f} "
                f"{row['recall']:>10.4f}"
            )
        print("-" * 56)
        print(f"\nTotal residues: {df_metrics.iloc[0]['n_total_residues']:,}")
        print(f"NGL residues: {df_metrics.iloc[0]['n_ngl']:,} ({df_metrics.iloc[0]['ngl_ratio']*100:.2f}%)")
        print(f"GL residues: {df_metrics.iloc[0]['n_gl']:,}")

        # Save metrics summary
        metrics_path = output_path.parent / "linear_probe_metrics_summary.csv"
        df_metrics.to_csv(metrics_path, index=False)
        logger.info(f"\nSaved metrics summary to: {metrics_path}")

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
