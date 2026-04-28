#!/usr/bin/env python
# coding: utf-8

"""
Filter Unpaired OAS by P90 Thresholds and Save
===============================================

This script:
1. Loads P90 thresholds from visualization output
2. Filters heavy and light chains by P90 threshold
3. Saves filtered data as chunked pickle files (max 30GB per chunk)

Memory-efficient: processes data in chunks to avoid loading everything into RAM.
"""

import os
import gc
import numpy as np
import pandas as pd
import dask.dataframe as dd
from tqdm.auto import tqdm

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

STAGE_2_HEAVY_DIR = "./stage_2_heavy_parquet"
STAGE_2_LIGHT_DIR = "./stage_2_light_parquet"
THRESHOLD_FILE = "./img/p90_thresholds.npz"
OUTPUT_DIR = "./"

# ═══════════════════════════════════════════════════════════════════
# FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def load_thresholds(threshold_file):
    """
    Load P90 thresholds from npz file.

    These thresholds are calculated from Naive B-cells only
    (matching the paired antibody approach).

    Args:
        threshold_file: Path to p90_thresholds.npz

    Returns:
        dict with thresholds
    """
    print(f"\n{'='*70}")
    print("Loading Naive B-Cell P90 Thresholds")
    print('='*70)

    if not os.path.exists(threshold_file):
        print(f"  ERROR: Threshold file not found: {threshold_file}")
        print(f"  Run visualize_unpaired_statistics.py first!")
        return None

    data = np.load(threshold_file)
    thresholds = {
        'hc_p90': float(data['hc_p90']),
        'lc_p90': float(data['lc_p90']),
        'hc_p95': float(data['hc_p95']),
        'lc_p95': float(data['lc_p95']),
        'hc_mean': float(data['hc_mean']),
        'lc_mean': float(data['lc_mean']),
        'hc_median': float(data['hc_median']),
        'lc_median': float(data['lc_median'])
    }

    print(f"  Heavy Chain P90: {thresholds['hc_p90']:.2f}")
    print(f"  Light Chain P90: {thresholds['lc_p90']:.2f}")

    return thresholds


def filter_and_save_chunked(ddf, threshold, chain_name, output_prefix, max_size_gb=30):
    """
    Filter sequences by P90 threshold and save as chunked pickle files.

    Args:
        ddf: Dask DataFrame
        threshold: P90 threshold value
        chain_name: "Heavy" or "Light"
        output_prefix: Prefix for output files
        max_size_gb: Maximum size per pickle chunk in GB

    Returns:
        Number of sequences retained
    """
    print(f"\n{'='*70}")
    print(f"Filtering and Saving {chain_name} Chain")
    print('='*70)

    # Filter by threshold
    print(f"  Filtering sequences with num_ngl_muts > {threshold:.2f}...")
    ddf_filtered = ddf[ddf['num_ngl_muts'] > threshold]

    # Count before and after
    n_before = len(ddf)
    n_after = len(ddf_filtered)
    pct_retained = 100 * n_after / n_before if n_before > 0 else 0

    print(f"    Before filtering: {n_before:,} sequences")
    print(f"    After filtering:  {n_after:,} sequences")
    print(f"    Retained: {pct_retained:.2f}%")

    if n_after == 0:
        print(f"    WARNING: No sequences passed the filter!")
        return 0

    # Use fixed row estimation based on typical antibody sequence data
    # Typical row is ~500-800 bytes with all columns
    print(f"\n  Using fixed chunk size estimation...")
    estimated_bytes_per_row = 700  # Conservative estimate
    max_bytes_per_chunk = max_size_gb * 1024**3
    rows_per_chunk = int(max_bytes_per_chunk / estimated_bytes_per_row)

    print(f"    Target: {max_size_gb} GB per chunk")
    print(f"    Estimated rows per chunk: ~{rows_per_chunk:,}")

    # Save chunks
    print(f"\n  Saving filtered data in chunks...")

    # Compute filtered data in chunks to avoid memory issues
    chunk_counter = 0
    rows_processed = 0

    # Get number of partitions
    n_partitions = ddf_filtered.npartitions
    print(f"    Processing {n_partitions} partitions...")

    # Accumulate rows until we reach chunk size
    accumulated_dfs = []
    accumulated_rows = 0

    for partition_idx in tqdm(range(n_partitions), desc="  Partitions"):
        # Compute one partition at a time
        partition_df = ddf_filtered.get_partition(partition_idx).compute()

        if len(partition_df) == 0:
            del partition_df
            continue

        accumulated_dfs.append(partition_df)
        accumulated_rows += len(partition_df)
        del partition_df  # No longer need reference after appending

        # Check if we should write a chunk
        if accumulated_rows >= rows_per_chunk:
            # Combine accumulated dataframes
            chunk_df = pd.concat(accumulated_dfs, ignore_index=True)

            # Save chunk
            chunk_path = f"{output_prefix}_chunk_{chunk_counter}.pkl"
            chunk_df.to_pickle(chunk_path)

            size_mb = os.path.getsize(chunk_path) / (1024**2)
            print(f"    Saved {chunk_path}: {len(chunk_df):,} rows, {size_mb:.2f} MB")

            # Reset accumulator
            accumulated_dfs = []
            accumulated_rows = 0
            chunk_counter += 1
            rows_processed += len(chunk_df)

            # Free memory explicitly
            del chunk_df
            gc.collect()  # Force garbage collection

    # Save remaining data
    if accumulated_dfs:
        chunk_df = pd.concat(accumulated_dfs, ignore_index=True)
        chunk_path = f"{output_prefix}_chunk_{chunk_counter}.pkl"
        chunk_df.to_pickle(chunk_path)

        size_mb = os.path.getsize(chunk_path) / (1024**2)
        print(f"    Saved {chunk_path}: {len(chunk_df):,} rows, {size_mb:.2f} MB")

        rows_processed += len(chunk_df)
        chunk_counter += 1

    print(f"\n  ✓ Saved {chunk_counter} chunk(s) with {rows_processed:,} total rows")

    return n_after


def save_all_data_chunked(ddf, chain_name, output_prefix, max_size_gb=30):
    """
    Save all data (unfiltered) as chunked pickle files.

    Args:
        ddf: Dask DataFrame
        chain_name: "Heavy" or "Light"
        output_prefix: Prefix for output files
        max_size_gb: Maximum size per pickle chunk in GB

    Returns:
        Number of sequences saved
    """
    print(f"\n{'='*70}")
    print(f"Saving All {chain_name} Chain Data (Unfiltered)")
    print('='*70)

    n_total = len(ddf)
    print(f"    Total sequences: {n_total:,}")

    # Use fixed row estimation based on typical antibody sequence data
    print(f"\n  Using fixed chunk size estimation...")
    estimated_bytes_per_row = 700  # Conservative estimate
    max_bytes_per_chunk = max_size_gb * 1024**3
    rows_per_chunk = int(max_bytes_per_chunk / estimated_bytes_per_row)

    print(f"    Target: {max_size_gb} GB per chunk")
    print(f"    Estimated rows per chunk: ~{rows_per_chunk:,}")

    # Save chunks
    print(f"\n  Saving data in chunks...")

    chunk_counter = 0
    rows_processed = 0

    # Get number of partitions
    n_partitions = ddf.npartitions
    print(f"    Processing {n_partitions} partitions...")

    # Accumulate rows until we reach chunk size
    accumulated_dfs = []
    accumulated_rows = 0

    for partition_idx in tqdm(range(n_partitions), desc="  Partitions"):
        # Compute one partition at a time
        partition_df = ddf.get_partition(partition_idx).compute()

        if len(partition_df) == 0:
            del partition_df
            continue

        accumulated_dfs.append(partition_df)
        accumulated_rows += len(partition_df)
        del partition_df  # No longer need reference after appending

        # Check if we should write a chunk
        if accumulated_rows >= rows_per_chunk:
            # Combine accumulated dataframes
            chunk_df = pd.concat(accumulated_dfs, ignore_index=True)

            # Save chunk
            chunk_path = f"{output_prefix}_chunk_{chunk_counter}.pkl"
            chunk_df.to_pickle(chunk_path)

            size_mb = os.path.getsize(chunk_path) / (1024**2)
            print(f"    Saved {chunk_path}: {len(chunk_df):,} rows, {size_mb:.2f} MB")

            # Reset accumulator
            accumulated_dfs = []
            accumulated_rows = 0
            chunk_counter += 1
            rows_processed += len(chunk_df)

            # Free memory explicitly
            del chunk_df
            gc.collect()  # Force garbage collection

    # Save remaining data
    if accumulated_dfs:
        chunk_df = pd.concat(accumulated_dfs, ignore_index=True)
        chunk_path = f"{output_prefix}_chunk_{chunk_counter}.pkl"
        chunk_df.to_pickle(chunk_path)

        size_mb = os.path.getsize(chunk_path) / (1024**2)
        print(f"    Saved {chunk_path}: {len(chunk_df):,} rows, {size_mb:.2f} MB")

        rows_processed += len(chunk_df)
        chunk_counter += 1

    print(f"\n  ✓ Saved {chunk_counter} chunk(s) with {rows_processed:,} total rows")

    return n_total


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Filter unpaired OAS data by P90 threshold and save as chunked pickle files"
    )
    parser.add_argument(
        '--save-all',
        action='store_true',
        help="Also save unfiltered data (all sequences)"
    )
    parser.add_argument(
        '--max-size-gb',
        type=float,
        default=30.0,
        help="Maximum size per pickle chunk in GB (default: 30)"
    )
    parser.add_argument(
        '--use-p95',
        action='store_true',
        help="Use P95 threshold instead of P90"
    )

    args = parser.parse_args()

    print("="*70)
    print("FILTER BY P90 AND SAVE CHUNKED PICKLE FILES")
    print("="*70)

    # Check if data exists
    if not os.path.exists(STAGE_2_HEAVY_DIR):
        print(f"\nERROR: Heavy chain data not found: {STAGE_2_HEAVY_DIR}")
        return

    if not os.path.exists(STAGE_2_LIGHT_DIR):
        print(f"\nERROR: Light chain data not found: {STAGE_2_LIGHT_DIR}")
        return

    # Load thresholds
    thresholds = load_thresholds(THRESHOLD_FILE)
    if thresholds is None:
        return

    # Select threshold level
    if args.use_p95:
        hc_threshold = thresholds['hc_p95']
        lc_threshold = thresholds['lc_p95']
        threshold_name = "p95"
        print(f"\n  Using P95 thresholds:")
    else:
        hc_threshold = 3
        lc_threshold = 2
        threshold_name = "p90"
        print(f"\n  Using P90 thresholds:")

    print(f"    Heavy: {hc_threshold:.2f}")
    print(f"    Light: {lc_threshold:.2f}")

    # Load data
    print(f"\nLoading data...")
    ddf_heavy = dd.read_parquet(STAGE_2_HEAVY_DIR)
    ddf_light = dd.read_parquet(STAGE_2_LIGHT_DIR)

    # Filter and save heavy chain
    n_heavy_filtered = filter_and_save_chunked(
        ddf_heavy,
        hc_threshold,
        "Heavy",
        os.path.join(OUTPUT_DIR, f"unpaired_HEAVY_filtered_{threshold_name}"),
        max_size_gb=args.max_size_gb
    )

    # Filter and save light chain
    n_light_filtered = filter_and_save_chunked(
        ddf_light,
        lc_threshold,
        "Light",
        os.path.join(OUTPUT_DIR, f"unpaired_LIGHT_filtered_{threshold_name}"),
        max_size_gb=args.max_size_gb
    )

    # Optionally save all data
    if args.save_all:
        print(f"\n{'='*70}")
        print("SAVING ALL DATA (UNFILTERED)")
        print('='*70)

        save_all_data_chunked(
            ddf_heavy,
            "Heavy",
            os.path.join(OUTPUT_DIR, "unpaired_HEAVY_all"),
            max_size_gb=args.max_size_gb
        )

        save_all_data_chunked(
            ddf_light,
            "Light",
            os.path.join(OUTPUT_DIR, "unpaired_LIGHT_all"),
            max_size_gb=args.max_size_gb
        )

    # Summary
    print(f"\n{'='*70}")
    print("COMPLETE")
    print('='*70)
    print(f"  Filtered data saved to:")
    print(f"    - unpaired_HEAVY_filtered_{threshold_name}_chunk_*.pkl")
    print(f"    - unpaired_LIGHT_filtered_{threshold_name}_chunk_*.pkl")

    if args.save_all:
        print(f"\n  All data saved to:")
        print(f"    - unpaired_HEAVY_all_chunk_*.pkl")
        print(f"    - unpaired_LIGHT_all_chunk_*.pkl")


if __name__ == "__main__":
    main()
