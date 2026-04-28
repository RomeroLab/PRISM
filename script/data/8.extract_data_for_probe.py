"""
Script to create a balanced benchmark dataset for Linear Probing evaluation.

This script performs cluster-based splitting to prevent data leakage and
stratified sampling based on NGL (non-germline) mutation counts.

Usage:
    python 8.extract_data_for_probe.py <input_path> <output_dir>

Example:
    python 8.extract_data_for_probe.py annotated_data_final/paired_with_clusters_filtered_ngl.pkl ./linear_probe_data
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path


def assign_ngl_bin(ngl_count: int) -> str:
    """Assign NGL count to a mutation load bin."""
    if ngl_count <= 5:
        return "low"
    elif ngl_count <= 15:
        return "mid"
    else:
        return "high"


def compute_ngl_count(row: pd.Series) -> int:
    """Compute total NGL count from heavy and light chain mutations."""
    return row["num_ngl_muts_hc"] + row["num_ngl_muts_lc"]


def split_clusters(df: pd.DataFrame, random_state: int = 42) -> tuple:
    """
    Split unique cluster_ids into train/val/test sets (80:10:10).

    Returns:
        Tuple of (train_clusters, val_clusters, test_clusters)
    """
    unique_clusters = df["cluster_id"].unique()

    # First split: 80% train, 20% temp (val + test)
    train_clusters, temp_clusters = train_test_split(
        unique_clusters,
        test_size=0.2,
        random_state=random_state
    )

    # Second split: 50% val, 50% test from temp (i.e., 10% each of total)
    val_clusters, test_clusters = train_test_split(
        temp_clusters,
        test_size=0.5,
        random_state=random_state
    )

    return set(train_clusters), set(val_clusters), set(test_clusters)


def stratified_sample_from_pool(
    pool_df: pd.DataFrame,
    target_total: int,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Sample sequences from a pool with stratification by NGL bin.

    Attempts to sample equally from Low/Mid/High bins. If a bin lacks
    enough data, takes all available and fills from others.

    Args:
        pool_df: DataFrame pool to sample from
        target_total: Total number of sequences to sample
        random_state: Random seed for reproducibility

    Returns:
        Sampled DataFrame
    """
    np.random.seed(random_state)

    # Split pool by bins
    bins = ["low", "mid", "high"]
    bin_dfs = {b: pool_df[pool_df["ngl_bin"] == b].copy() for b in bins}
    bin_counts = {b: len(bin_dfs[b]) for b in bins}

    # Target per bin (equal split)
    target_per_bin = target_total // 3
    remainder = target_total % 3

    sampled_dfs = []
    deficit = 0

    # First pass: sample up to target_per_bin from each bin
    for i, b in enumerate(bins):
        current_target = target_per_bin + (1 if i < remainder else 0)
        available = bin_counts[b]

        if available >= current_target:
            sampled = bin_dfs[b].sample(n=current_target, random_state=random_state)
        else:
            sampled = bin_dfs[b]  # Take all available
            deficit += current_target - available

        sampled_dfs.append(sampled)

    # Second pass: fill deficit from bins with remaining data
    if deficit > 0:
        already_sampled_indices = pd.concat(sampled_dfs).index
        remaining_pool = pool_df[~pool_df.index.isin(already_sampled_indices)]

        if len(remaining_pool) >= deficit:
            extra_samples = remaining_pool.sample(n=deficit, random_state=random_state)
        else:
            extra_samples = remaining_pool  # Take all remaining

        sampled_dfs.append(extra_samples)

    result = pd.concat(sampled_dfs, ignore_index=True)
    return result


def print_summary(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    """Print a summary report of the dataset splits."""
    print("\n" + "=" * 60)
    print("DATASET SUMMARY REPORT")
    print("=" * 60)

    for name, df in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
        print(f"\n{name} Set:")
        print(f"  Total sequences: {len(df)}")
        print(f"  Unique clusters: {df['cluster_id'].nunique()}")
        print(f"  Columns preserved: {len(df.columns)}")
        print(f"  Bin distribution:")
        for bin_name in ["low", "mid", "high"]:
            count = len(df[df["ngl_bin"] == bin_name])
            pct = count / len(df) * 100 if len(df) > 0 else 0
            print(f"    - {bin_name.capitalize():>4}: {count:>6} ({pct:>5.1f}%)")

        # NGL count statistics (using computed ngl_count = num_ngl_muts_hc + num_ngl_muts_lc)
        print(f"  NGL count stats (total = HC + LC):")
        print(f"    - Mean: {df['ngl_count'].mean():.2f}")
        print(f"    - Median: {df['ngl_count'].median():.1f}")
        print(f"    - Min: {df['ngl_count'].min()}, Max: {df['ngl_count'].max()}")
        print(f"  Heavy chain NGL: mean={df['num_ngl_muts_hc'].mean():.2f}")
        print(f"  Light chain NGL: mean={df['num_ngl_muts_lc'].mean():.2f}")

    # Verify no cluster leakage
    train_clusters = set(train_df["cluster_id"])
    val_clusters = set(val_df["cluster_id"])
    test_clusters = set(test_df["cluster_id"])

    print("\n" + "-" * 60)
    print("DATA LEAKAGE CHECK:")
    train_val_overlap = train_clusters & val_clusters
    train_test_overlap = train_clusters & test_clusters
    val_test_overlap = val_clusters & test_clusters

    print(f"  Train-Val cluster overlap: {len(train_val_overlap)} clusters")
    print(f"  Train-Test cluster overlap: {len(train_test_overlap)} clusters")
    print(f"  Val-Test cluster overlap: {len(val_test_overlap)} clusters")

    if not (train_val_overlap or train_test_overlap or val_test_overlap):
        print("   No data leakage detected!")
    else:
        print("   WARNING: Data leakage detected!")

    print("=" * 60 + "\n")


def main():
    # Parse command line arguments
    if len(sys.argv) != 3:
        print("Usage: python 8.extract_data_for_probe.py <input_path> <output_dir>")
        print("Example: python 8.extract_data_for_probe.py annotated_data_final/paired_with_clusters_filtered_ngl.pkl ./linear_probe_data")
        sys.exit(1)

    input_path = sys.argv[1]
    output_dir = sys.argv[2]

    # Validate input file
    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from: {input_path}")

    # Load dataframe
    try:
        df = pd.read_pickle(input_path)
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        sys.exit(1)

    # Validate required columns
    required_columns = [
        "HEAVY_CHAIN_AA_SEQUENCE",
        "LIGHT_CHAIN_AA_SEQUENCE",
        "num_ngl_muts_hc",
        "num_ngl_muts_lc",
        "cluster_id"
    ]
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)

    print(f"Loaded {len(df)} sequences with {df['cluster_id'].nunique()} unique clusters")

    # Compute total NGL count (heavy + light chain mutations)
    df["ngl_count"] = df.apply(compute_ngl_count, axis=1)

    # Assign NGL bins
    df["ngl_bin"] = df["ngl_count"].apply(assign_ngl_bin)

    print("\nOriginal bin distribution:")
    for bin_name in ["low", "mid", "high"]:
        count = len(df[df["ngl_bin"] == bin_name])
        print(f"  {bin_name.capitalize()}: {count} ({count/len(df)*100:.1f}%)")

    # Split clusters (80:10:10)
    print("\nSplitting clusters (80:10:10)...")
    train_clusters, val_clusters, test_clusters = split_clusters(df, random_state=42)

    print(f"  Train clusters: {len(train_clusters)}")
    print(f"  Val clusters: {len(val_clusters)}")
    print(f"  Test clusters: {len(test_clusters)}")

    # Create pool DataFrames based on cluster split
    train_pool = df[df["cluster_id"].isin(train_clusters)].copy()
    val_pool = df[df["cluster_id"].isin(val_clusters)].copy()
    test_pool = df[df["cluster_id"].isin(test_clusters)].copy()

    print(f"\nPool sizes after cluster split:")
    print(f"  Train pool: {len(train_pool)}")
    print(f"  Val pool: {len(val_pool)}")
    print(f"  Test pool: {len(test_pool)}")

    # Stratified sampling
    print("\nPerforming stratified sampling...")

    train_target = 40000
    val_target = 5000
    test_target = 5000

    # Adjust targets if pools are smaller
    if len(train_pool) < train_target:
        print(f"  Warning: Train pool ({len(train_pool)}) smaller than target ({train_target})")
        train_target = len(train_pool)
    if len(val_pool) < val_target:
        print(f"  Warning: Val pool ({len(val_pool)}) smaller than target ({val_target})")
        val_target = len(val_pool)
    if len(test_pool) < test_target:
        print(f"  Warning: Test pool ({len(test_pool)}) smaller than target ({test_target})")
        test_target = len(test_pool)

    train_df = stratified_sample_from_pool(train_pool, train_target, random_state=42)
    val_df = stratified_sample_from_pool(val_pool, val_target, random_state=42)
    test_df = stratified_sample_from_pool(test_pool, test_target, random_state=42)

    # Print summary
    print_summary(train_df, val_df, test_df)

    # Save outputs
    train_output = output_path / "train_linear.pkl"
    val_output = output_path / "val_linear.pkl"
    test_output = output_path / "test_linear.pkl"

    train_df.to_pickle(train_output)
    val_df.to_pickle(val_output)
    test_df.to_pickle(test_output)

    print(f"Saved outputs to {output_dir}:")
    print(f"  - {train_output}")
    print(f"  - {val_output}")
    print(f"  - {test_output}")


if __name__ == "__main__":
    main()
