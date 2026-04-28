#!/usr/bin/env python
# coding: utf-8
"""
PRISM GL/NGL Discrimination CLI

Evaluates a PRISM model's Origin Head ability to discriminate germline (GL)
from non-germline (NGL) residues. Computes per-residue classification metrics
(accuracy, F1, AUROC, PR-AUC) and per-sequence summary statistics.

Uses the prism.pretrained() API for model loading and predict_origin() for inference.

Usage:
    # Linear probe test data (paired sequences with germline alignments)
    python inference_gl_ngl_discrimination.py --checkpoint path/to/checkpoint.ckpt \
        --data_path data/unpaired_OAS/linear_probe_data/test_linear.pkl \
        --output_path results.csv --max_sequences 100

    # Heavy chain only
    python inference_gl_ngl_discrimination.py --checkpoint path/to/checkpoint.ckpt \
        --data_path test_linear.pkl --heavy_only

    # With per-residue output
    python inference_gl_ngl_discrimination.py --checkpoint path/to/checkpoint.ckpt \
        --data_path test_linear.pkl --save_per_residue
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
import prism


def parse_mutation_codes(mut_codes: str) -> List[int]:
    """Parse mutation codes like 'A40D;A61T;M83I' into 0-indexed positions."""
    if pd.isna(mut_codes) or mut_codes == "" or mut_codes is None:
        return []
    positions = []
    for mut in str(mut_codes).split(";"):
        mut = mut.strip()
        if not mut:
            continue
        try:
            pos_str = "".join(c for c in mut[1:-1] if c.isdigit())
            if pos_str:
                positions.append(int(pos_str) - 1)
        except (ValueError, IndexError):
            continue
    return positions


def compute_ngl_mask_from_alignment(seq: str, germline_seq: str) -> List[int]:
    """Compute NGL mask by comparing sequence to germline alignment."""
    mask = []
    for s, g in zip(seq, germline_seq):
        if g in ("X", "-"):
            mask.append(0)
        elif s != g:
            mask.append(1)
        else:
            mask.append(0)
    return mask


def extract_sequences_and_labels(
    df: pd.DataFrame, heavy_only: bool = False
) -> Tuple[List[str], List[List[int]]]:
    """Extract sequences and NGL labels from linear probe data format.

    Returns:
        sequences: List of uppercase amino acid sequences
        ngl_masks: List of binary masks (0=GL, 1=NGL) per sequence
    """
    sequences = []
    ngl_masks = []

    for _, row in df.iterrows():
        hc_seq = row["HEAVY_CHAIN_AA_SEQUENCE"]
        hc_gl = row.get("HEAVY_CHAIN_AA_GERMLINE_ALIGNMENT", None)

        # Compute HC NGL mask
        if pd.notna(hc_gl):
            hc_mask = compute_ngl_mask_from_alignment(hc_seq, hc_gl)
        else:
            hc_mut_pos = parse_mutation_codes(row.get("hc_mut_codes", ""))
            hc_mask = [1 if i in hc_mut_pos else 0 for i in range(len(hc_seq))]

        if heavy_only:
            sequences.append(hc_seq.upper())
            ngl_masks.append(hc_mask)
            continue

        lc_seq = row["LIGHT_CHAIN_AA_SEQUENCE"]
        lc_gl = row.get("LIGHT_CHAIN_AA_GERMLINE_ALIGNMENT", None)

        if pd.notna(lc_gl):
            lc_mask = compute_ngl_mask_from_alignment(lc_seq, lc_gl)
        else:
            lc_mut_pos = parse_mutation_codes(row.get("lc_mut_codes", ""))
            lc_mask = [1 if i in lc_mut_pos else 0 for i in range(len(lc_seq))]

        combined_seq = hc_seq.upper() + lc_seq.upper()
        combined_mask = hc_mask + lc_mask

        sequences.append(combined_seq)
        ngl_masks.append(combined_mask)

    return sequences, ngl_masks


def compute_metrics(all_probs: np.ndarray, all_labels: np.ndarray) -> dict:
    """Compute classification metrics from probabilities and binary labels."""
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_recall_curve,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.metrics import auc as sk_auc

    preds = (all_probs > 0.5).astype(int)

    accuracy = accuracy_score(all_labels, preds)
    f1 = f1_score(all_labels, preds, zero_division=0)
    precision = precision_score(all_labels, preds, zero_division=0)
    recall = recall_score(all_labels, preds, zero_division=0)

    # AUROC (requires both classes present)
    if len(np.unique(all_labels)) > 1:
        roc_auc = roc_auc_score(all_labels, all_probs)
        prec_curve, rec_curve, _ = precision_recall_curve(all_labels, all_probs)
        pr_auc = sk_auc(rec_curve, prec_curve)
    else:
        roc_auc = float("nan")
        pr_auc = float("nan")

    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "n_total": len(all_labels),
        "n_gl": int((all_labels == 0).sum()),
        "n_ngl": int((all_labels == 1).sum()),
        "ngl_fraction": float(all_labels.mean()),
    }


def main():
    parser = argparse.ArgumentParser(
        description="PRISM GL/NGL Discrimination Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to PRISM checkpoint (.ckpt)"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to test data (PKL with germline alignments)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output CSV path (default: <data>_gl_ngl_metrics.csv)",
    )
    parser.add_argument(
        "--max_sequences", type=int, default=None, help="Limit to N sequences (for testing)"
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Device: auto, cuda, cpu (default: auto)"
    )
    parser.add_argument(
        "--heavy_only", action="store_true", help="Evaluate on heavy chain only"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for inference (default: 32)"
    )
    parser.add_argument(
        "--save_per_residue",
        action="store_true",
        help="Also save per-residue predictions CSV",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("PRISM GL/NGL Discrimination Evaluation")
    print("=" * 70)

    # Resolve device
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    # Load data
    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"[ERROR] Data file not found: {data_path}")
        sys.exit(1)

    print(f"  Data: {data_path}")
    if data_path.suffix in (".pkl", ".pickle"):
        df = pd.read_pickle(data_path)
    elif data_path.suffix == ".csv":
        df = pd.read_csv(data_path)
    elif data_path.suffix == ".parquet":
        df = pd.read_parquet(data_path)
    else:
        raise ValueError(f"Unsupported file format: {data_path.suffix}")

    # Validate format
    required_cols = ["HEAVY_CHAIN_AA_SEQUENCE", "LIGHT_CHAIN_AA_SEQUENCE"]
    for col in required_cols:
        if col not in df.columns:
            print(f"[ERROR] Required column '{col}' not found.")
            print(f"  Available: {list(df.columns)}")
            sys.exit(1)

    print(f"  Total sequences: {len(df)}")

    if args.max_sequences:
        df = df.head(args.max_sequences)
        print(f"  Limited to {len(df)} sequences")

    # Extract sequences and ground truth NGL labels
    sequences, ngl_masks = extract_sequences_and_labels(df, heavy_only=args.heavy_only)
    chain_mode = "heavy only" if args.heavy_only else "heavy+light concatenated"
    print(f"  Chain mode: {chain_mode}")

    total_residues = sum(len(m) for m in ngl_masks)
    total_ngl = sum(sum(m) for m in ngl_masks)
    print(f"  Total residues: {total_residues:,}")
    print(f"  NGL residues: {total_ngl:,} ({100*total_ngl/total_residues:.1f}%)")

    # Load model
    print(f"\n  Loading checkpoint: {args.checkpoint}")
    t0 = time.time()
    model = prism.pretrained(args.checkpoint, device=device)
    print(f"  Model loaded in {time.time() - t0:.1f}s")

    # Run origin prediction
    print(f"\n{'='*70}")
    print("Running Origin Head prediction...")
    print(f"{'='*70}")

    all_probs = []
    all_labels = []
    per_seq_metrics = []
    per_residue_rows = []

    t0 = time.time()
    for i, (seq, ngl_mask) in enumerate(
        tqdm(zip(sequences, ngl_masks), total=len(sequences), desc="Predicting")
    ):
        result = model.predict_origin(seq, batch_size=args.batch_size)

        # origin_probs includes special tokens; extract sequence part
        # predict_origin returns logits for all positions including CLS/EOS
        origin_probs = result["origin_probs"]
        # Skip CLS (position 0) and EOS (last), take seq_len positions
        seq_len = len(seq)
        # origin_probs shape is [total_tokens]; seq tokens are [1:seq_len+1]
        if len(origin_probs) > seq_len:
            probs = origin_probs[1 : seq_len + 1]
        else:
            probs = origin_probs[:seq_len]

        # Align lengths
        min_len = min(len(probs), len(ngl_mask))
        probs = probs[:min_len]
        labels = np.array(ngl_mask[:min_len])

        all_probs.extend(probs.tolist())
        all_labels.extend(labels.tolist())

        # Per-sequence metrics
        if len(labels) > 0 and len(np.unique(labels)) > 1:
            seq_metrics = compute_metrics(probs, labels)
        else:
            preds_seq = (probs > 0.5).astype(int)
            seq_metrics = {
                "accuracy": float((preds_seq == labels).mean()) if len(labels) > 0 else 0.0,
                "f1": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "roc_auc": float("nan"),
                "pr_auc": float("nan"),
                "n_total": len(labels),
                "n_gl": int((labels == 0).sum()),
                "n_ngl": int((labels == 1).sum()),
                "ngl_fraction": float(labels.mean()) if len(labels) > 0 else 0.0,
            }
        seq_metrics["seq_idx"] = i
        per_seq_metrics.append(seq_metrics)

        # Per-residue output
        if args.save_per_residue:
            for j in range(min_len):
                per_residue_rows.append(
                    {
                        "seq_idx": i,
                        "residue_idx": j,
                        "amino_acid": seq[j],
                        "origin_prob": float(probs[j]),
                        "ngl_label": int(labels[j]),
                        "pred_ngl": int(probs[j] > 0.5),
                    }
                )

    elapsed = time.time() - t0

    # Aggregate metrics
    all_probs_arr = np.array(all_probs)
    all_labels_arr = np.array(all_labels)

    overall_metrics = compute_metrics(all_probs_arr, all_labels_arr)

    print(f"\n{'='*70}")
    print("RESULTS (all residues pooled)")
    print(f"{'='*70}")
    print(f"  Accuracy:  {overall_metrics['accuracy']:.4f}")
    print(f"  F1:        {overall_metrics['f1']:.4f}")
    print(f"  Precision: {overall_metrics['precision']:.4f}")
    print(f"  Recall:    {overall_metrics['recall']:.4f}")
    print(f"  ROC-AUC:   {overall_metrics['roc_auc']:.4f}")
    print(f"  PR-AUC:    {overall_metrics['pr_auc']:.4f}")
    print(f"  Residues:  {overall_metrics['n_total']:,} "
          f"({overall_metrics['n_gl']:,} GL, {overall_metrics['n_ngl']:,} NGL)")
    print(f"  Time:      {elapsed:.1f}s ({elapsed/len(sequences):.3f}s/seq)")

    # Per-sequence summary
    seq_df = pd.DataFrame(per_seq_metrics)
    print(f"\n  Per-sequence metric averages:")
    for col in ["accuracy", "f1", "roc_auc", "pr_auc"]:
        vals = seq_df[col].dropna()
        if len(vals) > 0:
            print(f"    {col:>10}: {vals.mean():.4f} +/- {vals.std():.4f}")

    # Save per-sequence metrics
    if args.output_path:
        output_path = Path(args.output_path)
    else:
        output_path = data_path.parent / f"{data_path.stem}_gl_ngl_metrics.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Add overall metrics as first row with seq_idx=-1
    overall_row = {**overall_metrics, "seq_idx": -1}
    output_df = pd.concat([pd.DataFrame([overall_row]), seq_df], ignore_index=True)
    output_df.to_csv(output_path, index=False)
    print(f"\n  Metrics saved to: {output_path}")

    # Per-residue output
    if args.save_per_residue and per_residue_rows:
        per_res_path = output_path.with_suffix(".per_residue.csv")
        per_res_df = pd.DataFrame(per_residue_rows)
        per_res_df.to_csv(per_res_path, index=False)
        print(f"  Per-residue predictions saved to: {per_res_path}")

    print(f"\n{'='*70}")
    print("Done!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
