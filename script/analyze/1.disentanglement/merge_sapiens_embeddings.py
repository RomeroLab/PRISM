#!/usr/bin/env python
"""
Merge Sapiens H and L embedding files into a single file.

Sapiens uses separate models for Heavy and Light chains, producing separate files:
  - train_linear_sapiens_h.pkl (contains embed_sapiens_h)
  - train_linear_sapiens_l.pkl (contains embed_sapiens_l)

This script merges them into a single file:
  - train_linear_sapiens.pkl (contains embed_sapiens_h and embed_sapiens_l)

Usage:
    python merge_sapiens_embeddings.py --data_dir data/unpaired_OAS/linear_probe_data
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def merge_sapiens_files(data_dir: Path, prefix: str) -> str:
    """
    Merge sapiens_h and sapiens_l files into a single file.

    Args:
        data_dir: Directory containing the embedding files
        prefix: File prefix (e.g., 'train_linear', 'val_linear', 'test_linear')

    Returns:
        Path to merged file
    """
    h_file = data_dir / f"{prefix}_sapiens_h.pkl"
    l_file = data_dir / f"{prefix}_sapiens_l.pkl"
    output_file = data_dir / f"{prefix}_sapiens.pkl"

    if not h_file.exists():
        logger.warning(f"Heavy chain file not found: {h_file}")
        return None

    if not l_file.exists():
        logger.warning(f"Light chain file not found: {l_file}")
        return None

    logger.info(f"Loading {h_file.name}...")
    df_h = pd.read_pickle(h_file)

    logger.info(f"Loading {l_file.name}...")
    df_l = pd.read_pickle(l_file)

    # Verify same number of rows
    if len(df_h) != len(df_l):
        logger.error(f"Row count mismatch: H={len(df_h)}, L={len(df_l)}")
        return None

    # Get embedding columns
    h_embed_col = [c for c in df_h.columns if c.startswith('embed_sapiens_h')]
    l_embed_col = [c for c in df_l.columns if c.startswith('embed_sapiens_l')]

    if not h_embed_col:
        logger.error("No embed_sapiens_h column found in H file")
        return None
    if not l_embed_col:
        logger.error("No embed_sapiens_l column found in L file")
        return None

    h_embed_col = h_embed_col[0]
    l_embed_col = l_embed_col[0]

    # Start with H file (has all base columns + H embeddings)
    result_df = df_h.copy()

    # Add L embeddings from L file
    result_df[l_embed_col] = df_l[l_embed_col].values

    # Rename columns to match expected format: embed_sapiens_h, embed_sapiens_l
    # (they should already be named correctly, but verify)
    logger.info(f"Embedding columns in merged file: {[c for c in result_df.columns if 'embed' in c]}")

    # Save merged file
    logger.info(f"Saving merged file to: {output_file}")
    result_df.to_pickle(output_file)

    logger.info(f"Merged {len(result_df)} samples with columns: embed_sapiens_h, embed_sapiens_l")

    return str(output_file)


def main():
    parser = argparse.ArgumentParser(
        description="Merge Sapiens H and L embedding files"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing sapiens embedding files"
    )
    parser.add_argument(
        "--prefixes",
        type=str,
        nargs="+",
        default=["train_linear", "val_linear", "test_linear"],
        help="File prefixes to merge (default: train_linear, val_linear, test_linear)"
    )

    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    if not data_dir.is_dir():
        logger.error(f"Directory not found: {data_dir}")
        sys.exit(1)

    merged_files = []
    for prefix in args.prefixes:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {prefix}")
        logger.info(f"{'='*60}")

        output = merge_sapiens_files(data_dir, prefix)
        if output:
            merged_files.append(output)

    logger.info(f"\n{'='*60}")
    logger.info("Done!")
    logger.info(f"{'='*60}")
    logger.info(f"Merged files created: {len(merged_files)}")
    for f in merged_files:
        logger.info(f"  - {Path(f).name}")


if __name__ == "__main__":
    main()
