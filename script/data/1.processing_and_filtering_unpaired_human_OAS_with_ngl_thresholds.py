#!/usr/bin/env python
# coding: utf-8

"""
Processing and Filtering Unpaired Human OAS Data with NGL Thresholds
=====================================================================

This script processes unpaired antibody sequences from OAS database with:
1. Duplicate sequence filtering
2. Missing conserved cysteine filtering
3. Heavily fragmented sequence filtering
4. Non-canonical amino acid filtering
5. Non-human source filtering
6. Naive B-cell p90 threshold filtering

Memory-optimized for:
- RAM < 100GB
- Disk < 4TB
- Multiprocessing enabled
- Progress tracking with tqdm
- Chunked output (max 30GB per chunk)
"""

import glob, re, json, gzip, os, subprocess, sys
from pathlib import Path
import pandas as pd
import numpy as np
import dask.dataframe as dd
import argparse
import concurrent.futures
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as spstats
from matplotlib.patches import Patch
from tqdm.auto import tqdm
from dask.diagnostics import ProgressBar

# Register tqdm for pandas
tqdm.pandas(desc="Processing", bar_format="{l_bar}{bar:15}{r_bar}")

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

OUTPUT_DIR = "./"
IMG_DIR = os.path.join(OUTPUT_DIR, "img")
STAGE_1_HEAVY_DIR = os.path.join(OUTPUT_DIR, "stage_1_heavy_parquet")
STAGE_1_LIGHT_DIR = os.path.join(OUTPUT_DIR, "stage_1_light_parquet")
STAGE_2_HEAVY_DIR = os.path.join(OUTPUT_DIR, "stage_2_heavy_parquet")
STAGE_2_LIGHT_DIR = os.path.join(OUTPUT_DIR, "stage_2_light_parquet")

# Create directories
for d in [IMG_DIR, STAGE_1_HEAVY_DIR, STAGE_1_LIGHT_DIR]:
    os.makedirs(d, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════
# AMINO ACID & MUTATION HELPERS
# ═══════════════════════════════════════════════════════════════════

VALID_RE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")
AA = set("ACDEFGHIKLMNPQRSTVWY")

def mutation_codes_from_alignment(seq_aln: str, gl_aln: str):
    """
    Compute mutation codes and count of non-germline residues.
    Positions are 1-based, counted over ungapped germline positions.
    Returns (codes_list, num_mutations).
    """
    if not isinstance(seq_aln, str) or not isinstance(gl_aln, str):
        return [], 0
    if len(seq_aln) != len(gl_aln):
        return [], 0

    pos = 0
    codes = []
    for s, g in zip(seq_aln, gl_aln):
        if g != '-':
            pos += 1
        if s in AA and g in AA and s != g:
            codes.append(f"{g}{pos}{s}")
    return codes, len(codes)

def join_codes(codes):
    return ";".join(codes) if codes else ""

# ═══════════════════════════════════════════════════════════════════
# FILTERING HELPERS
# ═══════════════════════════════════════════════════════════════════

def is_heavily_fragmented(status: str) -> bool:
    """Check if ANARCI status indicates heavy fragmentation"""
    s = str(status)
    if "Deletions" not in s:
        return False
    deleted = {int(n) for n in re.findall(r'\d+', s)}
    n_term_block = set(range(1, 18))
    c_term_block = set(range(122, 129))
    return n_term_block.issubset(deleted) or c_term_block.issubset(deleted)

def has_missing_cysteine(status: str) -> bool:
    """Check if sequence has missing conserved cysteine"""
    return "missing conserved cysteine" in str(status).lower()

def has_non_canonical_aa(seq: str) -> bool:
    """Check if sequence contains non-canonical amino acids"""
    if not isinstance(seq, str):
        return True
    return VALID_RE.match(seq) is None

# ═══════════════════════════════════════════════════════════════════
# STAGE 0: LOAD URL LIST
# ═══════════════════════════════════════════════════════════════════

def load_urls():
    """Load URLs from bulk_download.sh"""
    try:
        with open("bulk_download.sh", "r") as f:
            urls = [
                line.strip().split(" ")[-1]
                for line in f
                if line.strip().startswith("wget") and line.strip().endswith(".csv.gz")
            ]
        print(f"[URLs] Loaded {len(urls)} URLs from bulk_download.sh")
        return urls
    except FileNotFoundError:
        print("[ERROR] bulk_download.sh file not found. Exiting.")
        sys.exit(1)

# ═══════════════════════════════════════════════════════════════════
# STAGE 1: DOWNLOAD, FILTER, CALCULATE NGLs (MULTIPROCESSING)
# ═══════════════════════════════════════════════════════════════════

# No more queue - delete files immediately after processing

def process_single_csv(url):
    """
    Download and process a single CSV file.
    Applies filters: non-human, missing cysteine, fragmented, non-canonical AA.
    Calculates NGL mutations.
    CSV files are deleted IMMEDIATELY after processing to prevent memory/disk buildup.
    """
    import gc
    fname = os.path.basename(url)
    pid = os.getpid()

    try:
        # Check if output already exists (skip if already processed)
        chain_type = "HEAVY" if "heavy" in fname.lower() else "LIGHT"
        output_dir = STAGE_1_HEAVY_DIR if chain_type == "HEAVY" else STAGE_1_LIGHT_DIR
        output_path = os.path.join(output_dir, f"{Path(fname).stem}.parquet")

        if os.path.exists(output_path):
            return f"ALREADY_PROCESSED: {fname}"

        # Download only if file doesn't exist
        if not os.path.exists(fname):
            subprocess.run(["wget", "-q", url, "-O", fname], check=True)

        # Filter 1: Non-human species (and extract BType)
        try:
            with gzip.open(fname, "rt") as f:
                meta_line = f.readline().strip()
            start, end = meta_line.find("{"), meta_line.rfind("}")
            meta = json.loads(meta_line[start:end + 1].replace('""', '"')) if start != -1 else {}

            if meta.get("Species", "").lower() != "human":
                print(f"  [PID {pid}] SKIP (Non-human): {fname}")
                if os.path.exists(fname):
                    os.remove(fname)
                return f"SKIPPED (Non-human): {fname}"

            # Extract BType from metadata (will be added to all sequences from this file)
            btype = meta.get("BType", pd.NA)  # Use pd.NA for missing values
        except Exception as e:
            print(f"  [PID {pid}] FAIL (Metadata error): {fname} ({e})")
            if os.path.exists(fname):
                os.remove(fname)
            return f"FAILED (Metadata): {fname}"

        # Load CSV in chunks to reduce memory peak
        # First read a small chunk to check columns
        try:
            chunk_iter = pd.read_csv(fname, skiprows=1, compression='gzip',
                                     chunksize=50000, low_memory=True)
            first_chunk = next(chunk_iter)
        except Exception as e:
            print(f"  [PID {pid}] FAIL (CSV read error): {fname} ({e})")
            if os.path.exists(fname):
                os.remove(fname)
            gc.collect()
            return f"FAILED (CSV Read): {fname}"

        # Check required columns (unpaired OAS format)
        required_cols = [
            "sequence_alignment_aa", "germline_alignment_aa",
            "ANARCI_status", "sequence"  # "sequence" is the nucleotide sequence
        ]
        if not all(col in first_chunk.columns for col in required_cols):
            # Print available columns for debugging
            print(f"  [PID {pid}] SKIP (Missing cols): {fname}")
            print(f"  [PID {pid}]   Available columns: {', '.join(first_chunk.columns[:10])}...")
            del first_chunk
            if os.path.exists(fname):
                os.remove(fname)
            gc.collect()
            return f"SKIPPED (Missing Cols): {fname}"

        # Process chunks and filter incrementally
        filtered_chunks = []

        # Process first chunk
        chunk = first_chunk
        del first_chunk

        # Apply filters to chunk
        mask = (~chunk["ANARCI_status"].apply(has_missing_cysteine) &
                ~chunk["sequence_alignment_aa"].apply(has_non_canonical_aa) &
                ~chunk["ANARCI_status"].apply(is_heavily_fragmented))
        chunk = chunk[mask].copy()
        del mask

        if not chunk.empty:
            # Calculate mutations for this chunk
            mut_results = chunk.apply(
                lambda r: mutation_codes_from_alignment(
                    r["sequence_alignment_aa"], r["germline_alignment_aa"]
                ),
                axis=1
            )
            chunk["mut_codes"] = mut_results.map(lambda t: join_codes(t[0]))
            chunk["num_ngl_muts"] = mut_results.map(lambda t: t[1]).astype("Int64")
            chunk["BType"] = btype  # Add BType from metadata
            del mut_results
            filtered_chunks.append(chunk)
        del chunk
        gc.collect()

        # Process remaining chunks
        for chunk in chunk_iter:
            mask = (~chunk["ANARCI_status"].apply(has_missing_cysteine) &
                    ~chunk["sequence_alignment_aa"].apply(has_non_canonical_aa) &
                    ~chunk["ANARCI_status"].apply(is_heavily_fragmented))
            chunk = chunk[mask].copy()
            del mask

            if not chunk.empty:
                mut_results = chunk.apply(
                    lambda r: mutation_codes_from_alignment(
                        r["sequence_alignment_aa"], r["germline_alignment_aa"]
                    ),
                    axis=1
                )
                chunk["mut_codes"] = mut_results.map(lambda t: join_codes(t[0]))
                chunk["num_ngl_muts"] = mut_results.map(lambda t: t[1]).astype("Int64")
                chunk["BType"] = btype  # Add BType from metadata
                del mut_results
                filtered_chunks.append(chunk)
            del chunk
            gc.collect()

        # Combine filtered chunks
        if not filtered_chunks:
            if os.path.exists(fname):
                os.remove(fname)
            gc.collect()
            return f"EMPTY (Filter): {fname}"

        df = pd.concat(filtered_chunks, ignore_index=True)
        del filtered_chunks
        gc.collect()

        # Select output columns (unpaired OAS format)
        output_cols = [
            'sequence_alignment_aa', 'germline_alignment_aa',
            'sequence',  # nucleotide sequence
            'mut_codes', 'num_ngl_muts',
            'BType'  # B-cell type from metadata
        ]

        # Also save sequence_aa if available (ungapped amino acid sequence)
        if 'sequence_aa' in df.columns:
            df_final = df[output_cols + ['sequence_aa']].copy()
        else:
            df_final = df[output_cols].copy()

        # Explicitly delete df BEFORE saving to free memory
        del df
        gc.collect()

        # Save to Stage 1 Parquet with fast compression
        df_final.to_parquet(output_path, index=False, compression='snappy', engine='pyarrow')

        # Explicitly free memory after save
        del df_final
        gc.collect()

        # DELETE CSV IMMEDIATELY after successful processing
        if os.path.exists(fname):
            os.remove(fname)

        return f"SUCCESS: {fname}"

    except Exception as e:
        print(f"  [PID {pid}] UNHANDLED ERROR ({fname}): {e}")
        if os.path.exists(fname):
            os.remove(fname)
        return f"FAILED (Unhandled): {fname}"

# CSV files are now deleted immediately in process_single_csv()
# No separate cleanup function needed

def run_stage_1(urls, max_workers):
    """Run Stage 1 parallel processing"""
    import gc
    import signal
    import sys

    print("\n" + "="*70)
    print("STAGE 1: DOWNLOAD, FILTER & CALCULATE NGLs")
    print("="*70)
    print(f"  Workers: {max_workers}")
    print(f"  Files: {len(urls)}")

    # Track counts directly instead of storing all results
    success_count = 0
    skip_count = 0
    fail_count = 0
    empty_count = 0
    already_processed = 0

    # Use context manager with proper cleanup
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)

    try:
        # Process files and clean up in batches
        with tqdm(total=len(urls), desc="Processing Files", unit="file") as pbar:
            for i, result in enumerate(executor.map(process_single_csv, urls)):
                # Track stats directly (no list storage)
                if result.startswith("SUCCESS"):
                    success_count += 1
                elif result.startswith("ALREADY_PROCESSED"):
                    already_processed += 1
                elif result.startswith("SKIP"):
                    skip_count += 1
                elif result.startswith("FAIL"):
                    fail_count += 1
                elif result.startswith("EMPTY"):
                    empty_count += 1

                pbar.update(1)

                # VERY aggressive garbage collection every 5 files
                if (i + 1) % 5 == 0:
                    gc.collect()

                # Progress update every 100 files
                if (i + 1) % 100 == 0:
                    print(f"\n  [PROGRESS] Processed {i+1}/{len(urls)} files | New: {success_count} | Cached: {already_processed}")
                    pbar.set_postfix({"New": success_count, "Cached": already_processed})

    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Shutting down gracefully...")
        executor.shutdown(wait=False, cancel_futures=True)
        sys.exit(1)

    finally:
        # Always cleanup executor properly to avoid zombies
        executor.shutdown(wait=True)
        gc.collect()

    print(f"\n  ✓ Stage 1 Complete:")
    print(f"    SUCCESS (newly processed): {success_count}/{len(urls)}")
    print(f"    ALREADY PROCESSED (skipped): {already_processed}/{len(urls)}")
    print(f"    SKIPPED (other): {skip_count}/{len(urls)}")
    print(f"    FAILED:  {fail_count}/{len(urls)}")
    print(f"    EMPTY:   {empty_count}/{len(urls)}")
    print(f"    TOTAL GOOD: {success_count + already_processed}/{len(urls)}")

    # Return summary (not full results list)
    return {
        "success": success_count,
        "already_processed": already_processed,
        "skipped": skip_count,
        "failed": fail_count,
        "empty": empty_count
    }

# ═══════════════════════════════════════════════════════════════════
# STAGE 2: GLOBAL FILTERING WITH DASK
# ═══════════════════════════════════════════════════════════════════

def run_stage_2_dedup(chain_type):
    """
    Stage 2: Remove duplicates using seqkit rmdup.

    Requires: seqkit (already installed)

    seqkit rmdup is extremely fast and memory-efficient for exact deduplication:
    - Hash-based deduplication: O(n) time complexity
    - Low memory footprint (~2-5GB constant)
    - Multi-threaded (8 cores)
    - Faster than MMseqs2 for exact matches

    Process:
    1. Convert parquet files to FASTA (vectorized, chunked metadata)
    2. Run seqkit rmdup -s to deduplicate by sequence
    3. Parse kept sequence IDs from deduplicated FASTA
    4. Write deduplicated sequences back to parquet
    """
    print("\n" + "="*70)
    print(f"STAGE 2: GLOBAL DEDUPLICATION ({chain_type})")
    print("="*70)

    input_dir = STAGE_1_HEAVY_DIR if chain_type == "HEAVY" else STAGE_1_LIGHT_DIR
    output_dir = STAGE_2_HEAVY_DIR if chain_type == "HEAVY" else STAGE_2_LIGHT_DIR
    temp_dir = f"./temp_dedup_{chain_type.lower()}"

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    parquet_files = glob.glob(f"{input_dir}/*.parquet")
    if not os.path.exists(input_dir) or not parquet_files:
        print(f"  No parquet files found in {input_dir}. Skipping.")
        return None

    print(f"  Found {len(parquet_files)} parquet files")
    print(f"  Using MMseqs2 Linclust for memory-efficient deduplication")

    # Step 1: Convert parquet to FASTA (save metadata to parquet to avoid huge dict)
    fasta_file = os.path.join(temp_dir, "sequences.fasta")
    metadata_dir = os.path.join(temp_dir, "metadata")
    os.makedirs(metadata_dir, exist_ok=True)

    # Check if FASTA already exists (skip if it does)
    if os.path.exists(fasta_file) and os.path.getsize(fasta_file) > 0:
        print(f"  [1/3] Converting parquet to FASTA...")
        print(f"    ✓ FASTA file already exists: {fasta_file}")
        print(f"    Skipping Step 1 (to regenerate, delete {temp_dir})")
        
        # Count sequences in existing FASTA (use grep, much faster)
        result = subprocess.run(['grep', '-c', '^>', fasta_file], capture_output=True, text=True, check=True)
        seq_id = int(result.stdout.strip())
        print(f"    Total sequences in FASTA: {seq_id:,}")
    else:
        print(f"  [1/3] Converting parquet to FASTA...")
        print(f"    Processing {len(parquet_files)} files...")
        seq_id = 0
        metadata_chunks = []
        metadata_chunk_id = 0

        with open(fasta_file, 'w', buffering=16*1024*1024) as fasta:  # 16MB buffer
            for pq_file in tqdm(parquet_files, desc="  Files"):
                df = pd.read_parquet(pq_file, engine='pyarrow')

                # Filter valid sequences
                valid = df['sequence_alignment_aa'].notna() & (df['sequence_alignment_aa'].str.len() > 0)
                df = df[valid].copy()

                if len(df) == 0:
                    continue

                # Add seq_id column
                df['seq_id'] = range(seq_id, seq_id + len(df))

                # Write FASTA (vectorized)
                fasta.write(''.join([f">{sid}\n{seq}\n"
                    for sid, seq in zip(df['seq_id'].values, df['sequence_alignment_aa'].values)]))

                # Accumulate metadata
                metadata_chunks.append(df)

                # Write metadata chunk every 1M rows to limit memory
                if sum(len(c) for c in metadata_chunks) > 1000000:
                    meta_df = pd.concat(metadata_chunks, ignore_index=True)
                    meta_file = os.path.join(metadata_dir, f"meta_{metadata_chunk_id}.parquet")
                    meta_df.to_parquet(meta_file, engine='pyarrow', index=False)
                    metadata_chunks = []
                    metadata_chunk_id += 1

                seq_id += len(df)
                del df

        # Write remaining metadata
        if metadata_chunks:
            meta_df = pd.concat(metadata_chunks, ignore_index=True)
            meta_file = os.path.join(metadata_dir, f"meta_{metadata_chunk_id}.parquet")
            meta_df.to_parquet(meta_file, engine='pyarrow', index=False)

        print(f"    Total sequences: {seq_id:,}")

    # Step 2: Deduplicate using seqkit (faster than MMseqs2 for exact duplicates)
    dedup_fasta = os.path.join(temp_dir, "sequences_dedup.fasta")

    # Check if deduplicated FASTA already exists (skip if it does)
    if os.path.exists(dedup_fasta):
        print(f"  [2/3] Running seqkit rmdup for sequence deduplication...")
        print(f"    ✓ Deduplicated FASTA already exists: {dedup_fasta}")
        print(f"    Skipping Step 2 (to regenerate, delete {dedup_fasta})")
    else:
        print(f"  [2/3] Running seqkit rmdup for sequence deduplication...")
        # Run seqkit rmdup with 8 threads
        result = subprocess.run([
            'seqkit', 'rmdup',
            '-s',                    # By sequence (not ID)
            '-j', '8',               # 8 threads
            '-o', dedup_fasta,       # Output file
            fasta_file
        ], check=True, capture_output=True, text=True)

        print(f"    ✓ Deduplication complete")
        print(f"    Output: {result.stderr.strip()}")

    # Step 3: Parse deduplicated FASTA to get kept sequence IDs
    print(f"  [3/3] Extracting unique sequence IDs from deduplicated FASTA...")
    cluster_reps = set()

    # Optimized: use grep to extract headers (much faster than Python line-by-line)
    print(f"    Using grep to extract sequence IDs (faster)...")
    result = subprocess.run(
        ['grep', '^>', dedup_fasta],
        capture_output=True,
        text=True,
        check=True
    )

    # Parse headers
    for line in result.stdout.strip().split('\n'):
        if line:
            seq_id_str = line[1:].strip()  # Remove '>'
            cluster_reps.add(int(seq_id_str))

    print(f"    Unique sequences: {len(cluster_reps):,}")
    if seq_id > 0:
        print(f"    Deduplication rate: {100*(1 - len(cluster_reps)/seq_id):.1f}%")

    # Step 4: Write deduplicated sequences to parquet
    print(f"  [3/3] Writing deduplicated sequences to parquet...")

    # Convert cluster_reps set to DataFrame for fast merge (much faster than .isin())
    print(f"    Preparing cluster representatives for merge...")
    df_reps = pd.DataFrame({'seq_id': list(cluster_reps), 'keep': True})
    df_reps['seq_id'] = df_reps['seq_id'].astype('int64')  # Ensure int64 for merge
    print(f"    Representatives ready: {len(df_reps):,} sequences")

    # Read metadata chunks and merge with cluster representatives
    print(f"    Reading metadata from {metadata_dir}...")
    metadata_files = glob.glob(f"{metadata_dir}/meta_*.parquet")

    chunk_counter = 0
    for meta_file in tqdm(metadata_files, desc="  Processing metadata"):
        df_meta = pd.read_parquet(meta_file)
        df_meta['seq_id'] = df_meta['seq_id'].astype('int64')  # Ensure int64

        # Use merge instead of .isin() (much faster!)
        df_dedup = df_meta.merge(df_reps[['seq_id']], on='seq_id', how='inner')

        # Free df_meta immediately after merge (don't need it anymore)
        del df_meta

        if len(df_dedup) > 0:
            # Drop seq_id column (internal use only)
            df_dedup = df_dedup.drop(columns=['seq_id'])

            # Write chunk
            chunk_file = os.path.join(output_dir, f"part.{chunk_counter}.parquet")
            df_dedup.to_parquet(chunk_file, engine='pyarrow', compression='snappy', index=False)
            chunk_counter += 1

        # Free df_dedup immediately after write
        del df_dedup

    print(f"  ✓ Stage 2 Complete: Saved to {output_dir}")
    print(f"    Total unique sequences: {len(cluster_reps):,}")
    print(f"    Output chunks: {chunk_counter}")

    # Cleanup
    print(f"  Cleaning up temporary files...")
    subprocess.run(['rm', '-rf', temp_dir], check=True)

    return output_dir

# ═══════════════════════════════════════════════════════════════════
# STAGE 3: COMBINE HEAVY & LIGHT, CALCULATE STATISTICS
# ═══════════════════════════════════════════════════════════════════

def combine_and_analyze():
    """
    Calculate statistics for unpaired heavy and light chains separately.
    No need to combine or load all data into memory - just compute stats!
    """
    print("\n" + "="*70)
    print("STAGE 3: CALCULATE STATISTICS (UNPAIRED)")
    print("="*70)

    # Heavy chain statistics
    print("\n  [1/2] Heavy Chain Statistics")
    print("  " + "-"*66)
    try:
        ddf_heavy = dd.read_parquet(STAGE_2_HEAVY_DIR)

        # Count total sequences
        n_heavy = len(ddf_heavy)
        print(f"    Total sequences: {n_heavy:,}")

        # Calculate NGL mutation statistics (without loading into memory)
        if 'num_ngl_muts' in ddf_heavy.columns:
            ngl_stats = ddf_heavy['num_ngl_muts'].describe().compute()
            print(f"    NGL mutations per sequence:")
            print(f"      Mean:   {ngl_stats['mean']:.2f}")
            print(f"      Median: {ngl_stats['50%']:.2f}")
            print(f"      P90:    {ddf_heavy['num_ngl_muts'].quantile(0.90).compute():.2f}")
            print(f"      Max:    {ngl_stats['max']:.0f}")
    except Exception as e:
        print(f"    ERROR: {e}")

    # Light chain statistics
    print("\n  [2/2] Light Chain Statistics")
    print("  " + "-"*66)
    try:
        ddf_light = dd.read_parquet(STAGE_2_LIGHT_DIR)

        # Count total sequences
        n_light = len(ddf_light)
        print(f"    Total sequences: {n_light:,}")

        # Calculate NGL mutation statistics (without loading into memory)
        if 'num_ngl_muts' in ddf_light.columns:
            ngl_stats = ddf_light['num_ngl_muts'].describe().compute()
            print(f"    NGL mutations per sequence:")
            print(f"      Mean:   {ngl_stats['mean']:.2f}")
            print(f"      Median: {ngl_stats['50%']:.2f}")
            print(f"      P90:    {ddf_light['num_ngl_muts'].quantile(0.90).compute():.2f}")
            print(f"      Max:    {ngl_stats['max']:.0f}")
    except Exception as e:
        print(f"    ERROR: {e}")

    print("\n  ✓ Stage 3 Complete")
    print("="*70)

    return True  # Success indicator

# ═══════════════════════════════════════════════════════════════════
# STAGE 4: VISUALIZATION & P90 THRESHOLD CALCULATION
# ═══════════════════════════════════════════════════════════════════

def calculate_naive_p90_and_visualize(df_heavy, df_light):
    """
    Calculate p90 thresholds from the overall distribution (treating all as "naive" baseline).
    Generate all visualizations matching the paired notebook.
    """
    print("\n" + "="*70)
    print("STAGE 4: CALCULATE P90 THRESHOLDS & VISUALIZE")
    print("="*70)

    metrics_hc = "num_ngl_muts_hc"
    metrics_lc = "num_ngl_muts_lc"

    # For unpaired data, we don't have BType, so we calculate p90 from entire distribution
    # This represents the "naive" baseline for filtering

    print("  Calculating p90 thresholds from full distributions...")

    vals_hc = df_heavy[metrics_hc].dropna().astype(float).values
    vals_lc = df_light[metrics_lc].dropna().astype(float).values

    hc_p90 = float(np.percentile(vals_hc, 90)) if vals_hc.size else np.nan
    lc_p90 = float(np.percentile(vals_lc, 90)) if vals_lc.size else np.nan

    print(f"    Heavy Chain p90: {hc_p90:.2f}")
    print(f"    Light Chain p90: {lc_p90:.2f}")

    # Save thresholds
    threshold_path = os.path.join(IMG_DIR, "naive_p90_thresholds.npz")
    np.savez(
        threshold_path,
        num_ngl_muts_hc=hc_p90,
        num_ngl_muts_lc=lc_p90
    )
    print(f"  ✓ Saved thresholds to {threshold_path}")

    # Generate visualizations
    print("\n  Generating visualizations...")

    # 1. Distribution plots for HC and LC
    for vals, metric, chain in [(vals_hc, metrics_hc, "HC"), (vals_lc, metrics_lc, "LC")]:
        if vals.size == 0:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        # Histogram with KDE
        axes[0].hist(vals, bins=40, alpha=0.7, color="C0", edgecolor="black", density=True)
        sns.kdeplot(vals, ax=axes[0], color="red", linewidth=2)
        axes[0].set_title(f"Distribution of {chain} NGLs (All Sequences)", fontsize=11)
        axes[0].set_xlabel("Mutations", fontsize=10)
        axes[0].set_ylabel("Density", fontsize=10)

        # Percentile lines
        percentiles = [50, 90, 95, 99]
        q_vals = np.percentile(vals, percentiles)

        for p, q in zip(percentiles, q_vals):
            axes[0].axvline(q, color="k", linestyle="--", linewidth=1)
            axes[0].text(
                q, axes[0].get_ylim()[1]*0.9, f"q{p}",
                rotation=90, va="top", ha="right", fontsize=7,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.7)
            )

        # Q-Q plot
        spstats.probplot(vals, dist="norm", plot=axes[1])
        axes[1].set_title(f"Q-Q Plot of {chain} NGLs", fontsize=11)

        plt.tight_layout()
        plt.savefig(os.path.join(IMG_DIR, f"dist_{metric}_of_ngls.png"))
        plt.savefig(os.path.join(IMG_DIR, f"dist_{metric}_of_ngls.svg"))
        plt.close()

    print(f"  ✓ Saved distribution plots to {IMG_DIR}")

    # 2. Summary statistics
    print("\n  Summary Statistics:")
    print(f"    Heavy Chain - Mean: {vals_hc.mean():.2f}, Std: {vals_hc.std():.2f}, Median: {np.median(vals_hc):.2f}")
    print(f"    Light Chain - Mean: {vals_lc.mean():.2f}, Std: {vals_lc.std():.2f}, Median: {np.median(vals_lc):.2f}")

    return hc_p90, lc_p90

# ═══════════════════════════════════════════════════════════════════
# STAGE 5: FILTER BY P90 THRESHOLDS
# ═══════════════════════════════════════════════════════════════════

def filter_by_p90(df_heavy, df_light, hc_thresh, lc_thresh):
    """
    Filter sequences exceeding p90 thresholds.
    Matches the paired notebook's filtering logic.
    """
    print("\n" + "="*70)
    print("STAGE 5: FILTER BY P90 THRESHOLDS")
    print("="*70)
    print(f"  HC threshold: {hc_thresh:.2f}")
    print(f"  LC threshold: {lc_thresh:.2f}")

    # Filter heavy chains
    hc_before = len(df_heavy)
    df_heavy_filtered = df_heavy[df_heavy["num_ngl_muts_hc"] > hc_thresh].copy()
    print(f"  Heavy chains: {len(df_heavy_filtered)}/{hc_before} retained ({len(df_heavy_filtered)/hc_before*100:.2f}%)")

    # Filter light chains
    lc_before = len(df_light)
    df_light_filtered = df_light[df_light["num_ngl_muts_lc"] > lc_thresh].copy()
    print(f"  Light chains: {len(df_light_filtered)}/{lc_before} retained ({len(df_light_filtered)/lc_before*100:.2f}%)")

    return df_heavy_filtered, df_light_filtered

# ═══════════════════════════════════════════════════════════════════
# STAGE 6: SAVE CHUNKED PICKLE FILES
# ═══════════════════════════════════════════════════════════════════

def save_chunked_pickle(df, basename, max_size_gb=30):
    """
    Save DataFrame as chunked pickle files, each <= max_size_gb.
    """
    print(f"\n  Saving {basename} as chunked pickle files...")

    # Estimate size per row
    sample_size = df.head(1000).memory_usage(deep=True).sum()
    bytes_per_row = sample_size / min(1000, len(df))
    max_bytes = max_size_gb * 1024**3
    rows_per_chunk = int(max_bytes / bytes_per_row)

    print(f"    Estimated {bytes_per_row:.0f} bytes/row, {rows_per_chunk} rows/chunk")

    chunk_paths = []
    total_rows = len(df)
    num_chunks = (total_rows + rows_per_chunk - 1) // rows_per_chunk

    for i in range(num_chunks):
        start_idx = i * rows_per_chunk
        end_idx = min((i + 1) * rows_per_chunk, total_rows)
        chunk = df.iloc[start_idx:end_idx]

        chunk_path = f"{basename}_chunk_{i+1:03d}_of_{num_chunks:03d}.pkl"
        chunk.to_pickle(chunk_path)

        size_mb = os.path.getsize(chunk_path) / (1024**2)
        chunk_paths.append(chunk_path)
        print(
            f"    Saved {chunk_path}: {len(chunk)} rows, {size_mb:.2f} MB")

    print(f"  ✓ Saved {num_chunks} chunk(s) for {basename}")
    return chunk_paths

# ═══════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Process unpaired antibody sequences from OAS with comprehensive filtering"
    )
    parser.add_argument(
        '--n-cpus',
        type=int,
        default=4,
        help="Number of parallel CPU cores for Stage 1 (default: 4). "
             "WARNING: Each worker loads ~5-10GB uncompressed data from gzip files. "
             "For 100GB RAM limit, use --n-cpus 6-8 max."
    )
    parser.add_argument(
        '--skip-download',
        action='store_true',
        help="Skip Stage 1 (download & initial filtering) if already completed"
    )
    parser.add_argument(
        '--skip-stage2',
        action='store_true',
        help="Skip Stage 2 (deduplication) if already completed"
    )
    parser.add_argument(
        '--start-from-stage',
        type=int,
        choices=[1, 2, 3],
        help="Start from specific stage (1=download, 2=dedup, 3=combine). Skips all earlier stages."
    )
    parser.add_argument(
        '--skip-p90-filter',
        action='store_true',
        help="Skip final p90 threshold filtering"
    )

    args = parser.parse_args()

    print("\n" + "="*70)
    print("UNPAIRED OAS PROCESSING PIPELINE")
    print("="*70)
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Image directory: {IMG_DIR}")
    print(f"  CPUs: {args.n_cpus}")

    # Memory warning
    if args.n_cpus > 8:
        estimated_peak_gb = args.n_cpus * 10
        print(f"\n  ⚠️  WARNING: High CPU count detected!")
        print(f"     Each worker loads ~5-10GB from gzip decompression.")
        print(f"     Peak memory estimate: {args.n_cpus} workers × 10GB = ~{estimated_peak_gb}GB")
        print(f"     For <100GB RAM, recommend --n-cpus 6-8")
        response = input(f"     Continue with {args.n_cpus} CPUs? [y/N]: ")
        if response.lower() not in ['y', 'yes']:
            print("Exiting. Rerun with --n-cpus 6-8")
            sys.exit(0)

    # Determine which stages to run
    start_stage = args.start_from_stage if args.start_from_stage else 1
    skip_stage1 = args.skip_download or (start_stage > 1)
    skip_stage2 = args.skip_stage2 or (start_stage > 2)

    # Stage 1: Download and filter
    if not skip_stage1:
        urls = load_urls()
        run_stage_1(urls, max_workers=args.n_cpus)
    else:
        print(f"\n[INFO] Skipping Stage 1 (starting from stage {start_stage})")

    # Stage 2: Deduplicate
    if not skip_stage2:
        run_stage_2_dedup("HEAVY")
        run_stage_2_dedup("LIGHT")
    else:
        print(f"\n[INFO] Skipping Stage 2 (starting from stage {start_stage})")

    # Stage 3: Calculate statistics (memory-efficient, no data loading)
    result = combine_and_analyze()
    if not result:
        print("[ERROR] Failed to calculate statistics. Exiting.")
        sys.exit(1)

    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    print(f"  Heavy chain data: {STAGE_2_HEAVY_DIR}")
    print(f"  Light chain data: {STAGE_2_LIGHT_DIR}")
    print(f"\n  Use these files for downstream analysis (clustering, training, etc.)")
    print(f"  Data is already deduplicated and filtered!")

    print("\n  Note: For visualization and p90 filtering, use a separate script")
    print(f"  that processes data in chunks to avoid memory issues.")

if __name__ == "__main__":
    main()
