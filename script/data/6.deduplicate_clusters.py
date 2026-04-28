#!/usr/bin/env python
# coding: utf-8

"""
Compress Antibody Datasets via Cluster-based Deduplication (Final Optimized)
============================================================================

Workflow:
1. Extract Metadata: Read chunks one by one -> Save (Index, ClusterID, Length) to temp parquet.
2. Global Filtering: Load all metadata -> Filter by CDR3 Cluster (Longest wins) -> Filter by Whole Seq Cluster (Longest wins).
3. Reconstruct: Reload chunks -> Keep only surviving indices -> Save to new compressed chunks.
4. Logging: Save detailed reduction statistics to 'compression_log.txt'.

Memory Logic: Never loads full sequence strings of more than one chunk at a time.
"""

import os
import sys
import gc
import glob
import shutil
import time
import datetime
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

INPUT_ROOT = "./clustered_data"
OUTPUT_ROOT = "./compressed_data"
TEMP_META_DIR = "./temp_metadata"
LOG_FILE = "compression_log.txt"

# Target directories (Must match your folder structure)
TARGET_DIRS = {
    "Heavy": os.path.join(INPUT_ROOT, "heavy_chunks"),
    "Light": os.path.join(INPUT_ROOT, "light_chunks")
}

# The column name containing the amino acid sequence
SEQ_COL = 'sequence_alignment_aa'

# ═══════════════════════════════════════════════════════════════════
# STEP 1: EXTRACT METADATA
# ═══════════════════════════════════════════════════════════════════

def extract_metadata(file_path, seq_col):
    """
    Reads a pickle file, calculates sequence lengths, and saves minimal metadata.
    Returns: (path_to_parquet, record_count)
    """
    try:
        df = pd.read_pickle(file_path)
        
        # Check required columns
        req_cols = ['cdr3_cluster_id', 'whole_seq_cluster_id', seq_col]
        if not all(col in df.columns for col in req_cols):
            return None, 0

        # Create lightweight DataFrame
        meta_df = pd.DataFrame()
        meta_df['original_idx'] = df.index
        
        # Optimize memory using categories
        meta_df['cdr3_cluster_id'] = df['cdr3_cluster_id'].astype('category')
        meta_df['whole_seq_cluster_id'] = df['whole_seq_cluster_id'].astype('category')
        
        # Calculate length (fill NA with 0)
        meta_df['seq_len'] = df[seq_col].str.len().fillna(0).astype('int32')
        
        # Save to parquet (fast I/O)
        os.makedirs(TEMP_META_DIR, exist_ok=True)
        filename = os.path.basename(file_path).replace('.pkl', '_meta.parquet')
        save_path = os.path.join(TEMP_META_DIR, filename)
        
        meta_df.to_parquet(save_path, index=False)
        
        count = len(meta_df)
        del df, meta_df
        gc.collect()
        
        return save_path, count

    except Exception as e:
        tqdm.write(f"[!] Error extracting metadata from {file_path}: {e}")
        return None, 0

# ═══════════════════════════════════════════════════════════════════
# STEP 2: GLOBAL FILTERING
# ═══════════════════════════════════════════════════════════════════

def get_surviving_indices(meta_files):
    """
    Loads all metadata, performs 2-stage deduplication (CDR3 -> Whole).
    Returns: (survivors_dict, stats_tuple)
    """
    print(f"  > Loading {len(meta_files)} metadata files...")
    
    dfs = []
    # Merge Metadata
    for f in tqdm(meta_files, desc="  Merging Meta", unit="file", leave=False):
        original_pkl_name = os.path.basename(f).replace('_meta.parquet', '.pkl')
        sub_df = pd.read_parquet(f)
        # Track source file
        sub_df['source_file'] = original_pkl_name
        sub_df['source_file'] = sub_df['source_file'].astype('category')
        dfs.append(sub_df)
        
    full_meta = pd.concat(dfs, ignore_index=True)
    total_records = len(full_meta)
    del dfs
    gc.collect()
    
    print(f"  > Total loaded sequences: {total_records:,}")

    # -------------------------------------------------------
    # Filter 1: CDR3 Cluster Deduplication
    # -------------------------------------------------------
    print("  > [Filter 1] Deduplicating by CDR3 Cluster (Longest wins)...")
    start_time = time.time()
    
    # Sort by length descending (crucial for 'longest wins')
    full_meta = full_meta.sort_values(by='seq_len', ascending=False)
    
    # Drop duplicates keeping first (longest)
    full_meta = full_meta.drop_duplicates(subset=['cdr3_cluster_id'], keep='first')
    
    after_cdr3 = len(full_meta)
    print(f"    - Reduced: {total_records:,} -> {after_cdr3:,} "
          f"(Dropped {total_records - after_cdr3:,} | {time.time()-start_time:.1f}s)")
    
    # -------------------------------------------------------
    # Filter 2: Whole Sequence Cluster Deduplication
    # -------------------------------------------------------
    print("  > [Filter 2] Deduplicating by Whole Seq Cluster (Longest wins)...")
    start_time = time.time()
    
    # Sort again to ensure stability
    full_meta = full_meta.sort_values(by='seq_len', ascending=False)
    
    full_meta = full_meta.drop_duplicates(subset=['whole_seq_cluster_id'], keep='first')
    
    after_whole = len(full_meta)
    print(f"    - Reduced: {after_cdr3:,} -> {after_whole:,} "
          f"(Dropped {after_cdr3 - after_whole:,} | {time.time()-start_time:.1f}s)")
    
    # -------------------------------------------------------
    # Indexing Survivors
    # -------------------------------------------------------
    print("  > Indexing surviving sequences...")
    survivors = {}
    
    # Group by source file for efficient reconstruction
    grouped = full_meta.groupby('source_file', observed=True)
    for source, group in tqdm(grouped, desc="  Indexing", total=len(grouped), unit="file"):
        if not group.empty:
            survivors[source] = set(group['original_idx'].values)
            
    del full_meta
    gc.collect()
    
    # Prepare stats
    stats = (total_records, after_cdr3, after_whole)
    
    return survivors, stats

# ═══════════════════════════════════════════════════════════════════
# STEP 3: RECONSTRUCT CHUNKS
# ═══════════════════════════════════════════════════════════════════

def reconstruct_chunks(survivors_dict, input_dir, output_dir):
    """
    Reloads original pickles, filters by surviving indices, and saves.
    """
    os.makedirs(output_dir, exist_ok=True)
    pkl_files = sorted(glob.glob(os.path.join(input_dir, "*.pkl")))
    
    for pkl_path in tqdm(pkl_files, desc="  Reconstructing", unit="chunk"):
        file_name = os.path.basename(pkl_path)
        indices_to_keep = survivors_dict.get(file_name, set())
        
        # If no sequences survived in this chunk, skip creating file
        if not indices_to_keep:
            continue
            
        try:
            # Load original
            df = pd.read_pickle(pkl_path)
            
            # Filter
            compressed_df = df.loc[df.index.isin(indices_to_keep)].copy()
            
            # Save if not empty
            if not compressed_df.empty:
                out_path = os.path.join(output_dir, file_name)
                compressed_df.to_pickle(out_path)
            
            del df, compressed_df
            gc.collect()
            
        except Exception as e:
            tqdm.write(f"[!] Error reconstructing {file_name}: {e}")

# ═══════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════

def main():
    print("="*80)
    print("ANTIBODY DATA COMPRESSION PIPELINE (FINAL)")
    print("Strategy: Meta-Extract -> Global Filter (CDR3 then Whole) -> Reconstruct")
    print("="*80)

    # Clean start for temp files
    if os.path.exists(TEMP_META_DIR):
        shutil.rmtree(TEMP_META_DIR)
    os.makedirs(TEMP_META_DIR)
    
    # Prepare Log
    log_messages = []
    log_messages.append(f"Run Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_messages.append("="*60)
    
    # Process Heavy then Light
    target_order = ["Heavy", "Light"]
    
    for chain_type in target_order:
        input_dir = TARGET_DIRS.get(chain_type)
        if not input_dir or not os.path.exists(input_dir):
            print(f"Skipping {chain_type}: Directory not found ({input_dir})")
            continue
            
        output_dir = os.path.join(OUTPUT_ROOT, f"{chain_type}_chunks")
        print(f"\n[{chain_type} CHAIN PROCESSING]")
        print(f"Input:  {input_dir}")
        print(f"Output: {output_dir}")
        
        pkl_files = sorted(glob.glob(os.path.join(input_dir, "*.pkl")))
        if not pkl_files:
            print("No pickle files found.")
            continue

        # [1] Extract
        print("\n[Step 1/3] Extracting Metadata...")
        meta_files = []
        for pkl in tqdm(pkl_files, desc="  Extracting", unit="chunk"):
            meta_path, _ = extract_metadata(pkl, SEQ_COL)
            if meta_path:
                meta_files.append(meta_path)
        
        if not meta_files:
            continue

        # [2] Filter & Stats
        print("\n[Step 2/3] Global Deduplication & Stats...")
        survivors_dict, stats = get_surviving_indices(meta_files)
        
        initial, after_cdr3, final = stats
        ratio = (final / initial * 100) if initial > 0 else 0
        
        # Log formatting
        log_entry = (
            f"[{chain_type} Chain]\n"
            f"  1. Initial Sequences:          {initial:15,}\n"
            f"  2. After CDR3 Deduplication:   {after_cdr3:15,}\n"
            f"  3. After WholeSeq (Final):     {final:15,}\n"
            f"  --------------------------------------------------\n"
            f"  * Compression Ratio:           {ratio:.2f}% kept ({(100-ratio):.2f}% removed)"
        )
        log_messages.append(log_entry)
        log_messages.append("-" * 60)
        
        # [3] Reconstruct
        print("\n[Step 3/3] Reconstructing Compressed Chunks...")
        reconstruct_chunks(survivors_dict, input_dir, output_dir)
        
        # Cleanup intermediate files for this chain
        for f in meta_files:
            if os.path.exists(f): os.unlink(f)
            
    # Final Cleanup
    if os.path.exists(TEMP_META_DIR):
        shutil.rmtree(TEMP_META_DIR)

    # Write Log File
    with open(LOG_FILE, "w") as f:
        f.write("\n".join(log_messages))

    print("\n" + "="*80)
    print(f"ALL DONE! Output saved to: {OUTPUT_ROOT}")
    print(f"Statistics Log: {LOG_FILE}")
    print("="*80)
    
    # Print Final Summary to Console
    print("\n--- FINAL SUMMARY ---")
    for msg in log_messages:
        print(msg)

if __name__ == "__main__":
    main()