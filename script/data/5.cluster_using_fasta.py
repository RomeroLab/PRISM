#!/usr/bin/env python
# coding: utf-8

"""
Cluster Antibody Sequences Using FASTA Workflow
================================================

Memory-efficient FASTA-based clustering pipeline.

Workflow:
1. Write whole seq FASTA from pickle chunks (one chunk at a time)
2. Run ANARCI on whole seq FASTA → extract CDR3 → write CDR3 FASTA
3. Run MMseqs2 on CDR3 FASTA (100% identity)
4. Run MMseqs2 on whole seq FASTA (95% identity)
5. Map cluster IDs back to pickle chunks

Memory usage: Max ~30GB (one pickle chunk at a time)
"""

import os
import sys
import gc
import glob
import pickle
import pandas as pd
import subprocess
from pathlib import Path
from tqdm.auto import tqdm
import argparse
import threading
import time
import multiprocessing as mp
from functools import partial

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

PAIRED_DATA_PATH = "../merged_antibody_sequences.pkl"
UNPAIRED_HEAVY_PATTERN = "./unpaired_HEAVY_filtered_p90_chunk_*.pkl"
UNPAIRED_LIGHT_PATTERN = "./unpaired_LIGHT_filtered_p90_chunk_*.pkl"

OUTPUT_DIR = "./clustered_data"
FASTA_DIR = "./temp_fasta"

# MMseqs2 parameters
MMSEQS_THREADS = 8
CDR3_IDENTITY = 1.00
WHOLE_SEQ_IDENTITY = 0.95
MMSEQS_MEMORY_LIMIT = "200G"  # Set to ~60% of available RAM (200GB -> 120GB)

# ANARCI parameters
ANARCI_PROCESSES = 8
ANARCI_CHUNK_SIZE = 10000  # Process 1M sequences per chunk to avoid memory issues

# ═══════════════════════════════════════════════════════════════════
# STEP 1: WRITE FASTA FROM PICKLE CHUNKS
# ═══════════════════════════════════════════════════════════════════

def write_fasta_from_pickles(pickle_files, output_fasta, seq_column, id_prefix, skip_if_exists=True):
    """
    Write FASTA file from pickle chunks without loading all into memory.
    Uses fast file writing with temporary chunk FASTA files and cat.

    Args:
        pickle_files: List of pickle file paths
        output_fasta: Output FASTA file path
        seq_column: Column name containing sequences
        id_prefix: Prefix for sequence IDs (e.g., 'heavy', 'light', 'paired_hc')
        skip_if_exists: If True, skip if output file already exists
    """
    if skip_if_exists and os.path.exists(output_fasta):
        # Quick check - just verify file exists and has non-zero size
        file_size = os.path.getsize(output_fasta)
        if file_size > 0:
            file_size_gb = file_size / (1024**3)
            print(f"  FASTA exists: {output_fasta} ({file_size_gb:.2f} GB)")
            return -1  # Return -1 to indicate file exists but we didn't count
        else:
            print(f"  FASTA exists but is empty, regenerating...")
            os.unlink(output_fasta)

    print(f"  Writing FASTA: {output_fasta}")

    temp_fastas = []
    total_seqs = 0

    # Write each chunk to temporary FASTA
    for pkl_file in tqdm(pickle_files, desc="    Chunks"):
        df = pd.read_pickle(pkl_file)
        chunk_name = Path(pkl_file).stem

        temp_fasta = f"{output_fasta}.{chunk_name}.tmp"
        temp_fastas.append(temp_fasta)

        # Fast writing using list comprehension and join
        fasta_lines = []
        for idx, seq in enumerate(df[seq_column]):
            if pd.notna(seq) and len(seq) > 0:
                seq_id = f"{id_prefix}_{chunk_name}_seq_{idx}"
                fasta_lines.append(f">{seq_id}\n{seq}\n")
                total_seqs += 1

        with open(temp_fasta, 'w') as f:
            f.write(''.join(fasta_lines))

        del df, fasta_lines
        gc.collect()

    # Concatenate all temp FASTA files using cat (much faster than Python)
    print(f"    Concatenating {len(temp_fastas)} temp files...")
    subprocess.run(['cat'] + temp_fastas, stdout=open(output_fasta, 'w'), check=True)

    # Remove temp files
    subprocess.run(['rm', '-f'] + temp_fastas, check=True, capture_output=True)

    print(f"    Wrote {total_seqs:,} sequences")
    return total_seqs


def write_fasta_from_single_pickle(pickle_file, output_fasta, heavy_col, light_col, skip_if_exists=True):
    """Write paired sequences as separate heavy/light FASTA entries."""
    heavy_fasta = output_fasta + '_heavy.fasta'
    light_fasta = output_fasta + '_light.fasta'

    # Quick check if both files exist and have non-zero size
    if skip_if_exists and os.path.exists(heavy_fasta) and os.path.exists(light_fasta):
        heavy_size = os.path.getsize(heavy_fasta)
        light_size = os.path.getsize(light_fasta)

        if heavy_size > 0 and light_size > 0:
            heavy_size_mb = heavy_size / (1024**2)
            light_size_mb = light_size / (1024**2)
            print(f"  Paired FASTA exists:")
            print(f"    Heavy: {heavy_fasta} ({heavy_size_mb:.2f} MB)")
            print(f"    Light: {light_fasta} ({light_size_mb:.2f} MB)")
            return -1
        else:
            print(f"  Paired FASTA exists but one or both are empty, regenerating...")
            if os.path.exists(heavy_fasta):
                os.unlink(heavy_fasta)
            if os.path.exists(light_fasta):
                os.unlink(light_fasta)

    print(f"  Writing paired FASTA: {output_fasta}")

    df = pd.read_pickle(pickle_file)
    total_seqs = 0

    with open(heavy_fasta, 'w') as fh, open(light_fasta, 'w') as fl:
        for idx, row in df.iterrows():
            hc_seq = row[heavy_col]
            lc_seq = row[light_col]

            if pd.notna(hc_seq) and len(hc_seq) > 0:
                fh.write(f">paired_hc_seq_{idx}\n{hc_seq}\n")
                total_seqs += 1

            if pd.notna(lc_seq) and len(lc_seq) > 0:
                fl.write(f">paired_lc_seq_{idx}\n{lc_seq}\n")
                total_seqs += 1

    del df
    gc.collect()

    print(f"    Wrote {total_seqs:,} sequences (heavy + light)")
    return total_seqs


# ═══════════════════════════════════════════════════════════════════
# STEP 2: EXTRACT CDR3 WITH ANARCI
# ═══════════════════════════════════════════════════════════════════

def monitor_anarci_progress(output_file, stop_event, start_time):
    """Background thread to monitor ANARCI output file size."""
    last_size = 0
    last_update = start_time
    while not stop_event.is_set():
        current_time = time.time()
        elapsed = current_time - start_time

        if os.path.exists(output_file):
            current_size = os.path.getsize(output_file)
            if current_size > last_size:
                size_mb = current_size / (1024**2)
                size_gb = current_size / (1024**3)
                print(f"    ANARCI progress: {size_gb:.2f} GB written (elapsed: {elapsed/60:.1f} min)    ")
                last_size = current_size
                last_update = current_time
            elif current_time - last_update > 30:  # No update for 30 seconds
                print(f"    ANARCI still running (elapsed: {elapsed/60:.1f} min, waiting for output...)    ")
                last_update = current_time
        else:
            if current_time - last_update > 30:
                print(f"    ANARCI loading input file (elapsed: {elapsed/60:.1f} min)...    ")
                last_update = current_time

        time.sleep(10)  # Check every 10 seconds


def split_fasta_by_size(input_fasta, seqs_per_chunk):
    """Split large FASTA into chunks of fixed size for parallel processing."""
    chunk_dir = os.path.join(FASTA_DIR, "anarci_chunks")
    os.makedirs(chunk_dir, exist_ok=True)

    base_name = os.path.basename(input_fasta).replace('.fasta', '')

    # Check if chunks already exist
    existing_chunks = sorted(glob.glob(os.path.join(chunk_dir, f"{base_name}_chunk_*.fasta")))
    if existing_chunks:
        # Verify chunks have non-zero size
        valid_chunks = [c for c in existing_chunks if os.path.getsize(c) > 0]
        if valid_chunks:
            print(f"  Found {len(valid_chunks)} existing chunks, skipping split...")
            return valid_chunks
        else:
            print(f"  Found {len(existing_chunks)} chunks but all are empty, re-splitting...")
            # Clean up empty chunks
            for chunk in existing_chunks:
                os.unlink(chunk)

    print(f"  Splitting into chunks of {seqs_per_chunk:,} sequences each...")

    chunk_files = []
    chunk_idx = 0
    seq_count = 0
    current_chunk = os.path.join(chunk_dir, f"{base_name}_chunk_{chunk_idx:04d}.fasta")
    f_out = open(current_chunk, 'w')
    chunk_files.append(current_chunk)

    total_seqs = 0
    with open(input_fasta, 'r') as f_in:
        for line in f_in:
            if line.startswith('>'):
                total_seqs += 1
                seq_count += 1
                if seq_count > seqs_per_chunk:
                    f_out.close()
                    chunk_idx += 1
                    current_chunk = os.path.join(chunk_dir, f"{base_name}_chunk_{chunk_idx:04d}.fasta")
                    f_out = open(current_chunk, 'w')
                    chunk_files.append(current_chunk)
                    seq_count = 1
                    if chunk_idx % 100 == 0:
                        print(f"    Created {chunk_idx} chunks ({total_seqs:,} sequences)...")
            f_out.write(line)

    f_out.close()
    print(f"  Created {len(chunk_files)} chunks ({total_seqs:,} total sequences)")
    return chunk_files


def process_anarci_chunk(chunk_file, chain_type):
    """Process one FASTA chunk with ANARCI (for multiprocessing)."""
    chunk_name = os.path.basename(chunk_file)
    anarci_output_base = chunk_file.replace('.fasta', '_anarci')
    cdr3_output = chunk_file.replace('.fasta', '_cdr3.fasta')

    print(f"    [{chunk_name}] Starting ANARCI...")

    # Run ANARCI with CSV output (requires --outfile parameter)
    # ANARCI appends _{chain_type}.csv to the output filename
    try:
        result = subprocess.run(
            ['ANARCI', '-i', chunk_file, '--scheme', 'imgt', '--csv', '--outfile', anarci_output_base],
            stderr=subprocess.PIPE, text=True,
            timeout=1800  # 30 min timeout per chunk
        )

        if result.returncode != 0:
            stderr_msg = result.stderr[:500] if result.stderr else "No stderr"
            print(f"    [{chunk_name}] FAILED: return code {result.returncode}")
            print(f"    [{chunk_name}] STDERR: {stderr_msg}")
            return 0, cdr3_output

    except Exception as e:
        print(f"    [{chunk_name}] FAILED: {e}")
        return 0, cdr3_output

    print(f"    [{chunk_name}] ANARCI complete, parsing CDR3...")

    # ANARCI creates output files with specific suffixes:
    # Heavy: _H.csv, Kappa light: _KL.csv, Lambda light: _LL.csv
    # For light chains, need to check for both KL and LL
    if chain_type == 'L':
        # Check for kappa light (KL) or lambda light (LL)
        anarci_output_kl = f"{anarci_output_base}_KL.csv"
        anarci_output_ll = f"{anarci_output_base}_LL.csv"
        if os.path.exists(anarci_output_kl):
            anarci_output = anarci_output_kl
        elif os.path.exists(anarci_output_ll):
            anarci_output = anarci_output_ll
        else:
            print(f"    [{chunk_name}] WARNING: No light chain output found (checked KL and LL)")
            open(cdr3_output, 'w').close()
            return 0, cdr3_output
    else:
        anarci_output = f"{anarci_output_base}_{chain_type}.csv"
        if not os.path.exists(anarci_output):
            print(f"    [{chunk_name}] WARNING: ANARCI output not found: {anarci_output}")
            open(cdr3_output, 'w').close()
            return 0, cdr3_output

    # Parse ANARCI CSV output and extract CDR3 (positions 105-117 in IMGT)
    cdr3_count = 0
    with open(anarci_output, 'r') as f_in, open(cdr3_output, 'w') as f_out:
        for line in f_in:
            if line.startswith('Id,') or line.startswith('#') or not line.strip():
                continue

            parts = line.strip().split(',')
            if len(parts) < 130:  # Need at least positions up to 128
                continue

            seq_id = parts[0]

            # CSV columns: Id, domain_no, hmm_species, chain_type, ..., then positions start at column 13
            # Positions 105-117 in IMGT are at column indices 13+104 to 13+116 = 117 to 129
            cdr3_aas = []
            for i in range(117, 130):  # Column indices for positions 105-117
                if i < len(parts):
                    aa = parts[i].strip()
                    if aa and aa != '-':
                        cdr3_aas.append(aa)

            cdr3 = ''.join(cdr3_aas)
            # Write all CDR3 sequences (no length filter)
            if cdr3:  # Only skip if completely empty
                f_out.write(f">{seq_id}\n{cdr3}\n")
                cdr3_count += 1

    # Cleanup ANARCI output
    if os.path.exists(anarci_output):
        os.unlink(anarci_output)

    print(f"    [{chunk_name}] Complete: {cdr3_count:,} CDR3s extracted")
    return cdr3_count, cdr3_output


def extract_cdr3_with_anarci(input_fasta, output_cdr3_fasta, chain_type='H', skip_if_exists=True):
    """
    Run ANARCI on whole sequence FASTA and extract CDR3 to new FASTA.
    Processes in parallel chunks to avoid memory issues.

    Args:
        input_fasta: Input FASTA with whole sequences
        output_cdr3_fasta: Output FASTA with CDR3 sequences only
        chain_type: 'H' for heavy, 'L' for light
        skip_if_exists: If True, skip if output CDR3 FASTA already exists
    """
    # Check if CDR3 FASTA already exists and has non-zero size
    if skip_if_exists and os.path.exists(output_cdr3_fasta):
        file_size = os.path.getsize(output_cdr3_fasta)
        if file_size > 0:
            file_size_mb = file_size / (1024**2)
            print(f"  CDR3 FASTA exists: {output_cdr3_fasta} ({file_size_mb:.2f} MB)")
            return -1  # Return -1 to indicate exists (skip counting for speed)
        else:
            print(f"  Existing CDR3 FASTA is empty, regenerating...")
            os.unlink(output_cdr3_fasta)

    print(f"  Running ANARCI ({chain_type}) in parallel on {input_fasta}...")

    # Split FASTA by chunk size (not number of chunks)
    chunk_files = split_fasta_by_size(input_fasta, ANARCI_CHUNK_SIZE)

    # Process chunks in parallel batches
    n_parallel = ANARCI_PROCESSES
    print(f"  Processing {len(chunk_files)} chunks in batches of {n_parallel}...")

    total_cdr3_count = 0
    all_results = []

    # Process in batches to control memory
    for batch_start in range(0, len(chunk_files), n_parallel):
        batch_end = min(batch_start + n_parallel, len(chunk_files))
        batch = chunk_files[batch_start:batch_end]

        print(f"  Processing batch {batch_start//n_parallel + 1}/{(len(chunk_files) + n_parallel - 1)//n_parallel} "
              f"(chunks {batch_start+1}-{batch_end} / {len(chunk_files)})...")

        with mp.Pool(processes=len(batch)) as pool:
            process_func = partial(process_anarci_chunk, chain_type=chain_type)
            batch_results = pool.map(process_func, batch)
            all_results.extend(batch_results)

        # Print batch progress
        batch_cdr3 = sum(r[0] for r in batch_results)
        total_cdr3_count += batch_cdr3
        print(f"    Batch complete: {batch_cdr3:,} CDR3s extracted, {total_cdr3_count:,} total so far")

    # Combine all results
    print(f"  Combining all results...")
    with open(output_cdr3_fasta, 'w') as f_out:
        for cdr3_count, cdr3_file in all_results:
            if os.path.exists(cdr3_file) and os.path.getsize(cdr3_file) > 0:
                subprocess.run(['cat', cdr3_file], stdout=f_out, check=True)
                os.unlink(cdr3_file)

    # Cleanup chunk files
    for chunk_file in chunk_files:
        if os.path.exists(chunk_file):
            os.unlink(chunk_file)

    print(f"    Extracted {total_cdr3_count:,} CDR3 sequences total")
    return total_cdr3_count


# ═══════════════════════════════════════════════════════════════════
# STEP 3: RUN MMSEQS2 CLUSTERING
# ═══════════════════════════════════════════════════════════════════

def run_mmseqs_linclust(input_fasta, output_tsv, identity, coverage=0.8, threads=8,
                        memory_limit=None, skip_if_exists=True):
    """
    Run MMseqs2 Linclust and return TSV with cluster assignments.

    Args:
        input_fasta: Input FASTA file
        output_tsv: Output TSV file with cluster assignments
        identity: Sequence identity threshold (0.0-1.0)
        coverage: Coverage threshold (0.0-1.0)
        threads: Number of threads to use
        memory_limit: Memory limit string (e.g., "120G", "200G"). Set to ~60% of available RAM.
        skip_if_exists: If True, skip if output TSV already exists and is non-empty
    """
    # Check if output already exists and is valid (size check is instant)
    if skip_if_exists and os.path.exists(output_tsv):
        file_size = os.path.getsize(output_tsv)
        if file_size > 0:
            file_size_mb = file_size / (1024**2)
            print(f"  ✓ Skipping: {output_tsv} already exists ({file_size_mb:.2f} MB)")
            return output_tsv
        else:
            print(f"  Cluster TSV exists but is empty, regenerating...")
            os.unlink(output_tsv)

    print(f"  Running MMseqs2 Linclust (id={identity}, cov={coverage})...")

    base = input_fasta.replace('.fasta', '')
    tmp_dir = f"{base}_mmseqs_tmp"
    os.makedirs(tmp_dir, exist_ok=True)

    db = f"{base}_DB"
    clu_db = f"{base}_clu"

    # Create DB
    print(f"    Creating database...")
    subprocess.run(['mmseqs', 'createdb', input_fasta, db], check=True)

    # Build clustering command with memory limit
    cluster_cmd = [
        'mmseqs', 'linclust', db, clu_db, tmp_dir,
        '--min-seq-id', str(identity),
        '-c', str(coverage),
        '--threads', str(threads),
        '--cov-mode', '1',
        '-v', '3'
    ]

    # Add memory limit if specified (recommended: ~60% of available RAM)
    if memory_limit:
        cluster_cmd.extend(['--split-memory-limit', memory_limit])
        print(f"    Memory limit: {memory_limit}")

    # Cluster
    print(f"    Clustering with {threads} threads...")
    subprocess.run(cluster_cmd, check=True)

    # Export to TSV
    print(f"    Generating TSV...")
    subprocess.run(['mmseqs', 'createtsv', db, db, clu_db, output_tsv], check=True)

    # Cleanup MMseqs2 intermediate files
    cleanup_patterns = [
        tmp_dir,
        db, f"{db}.index", f"{db}.lookup", f"{db}.source", f"{db}.dbtype",
        f"{db}_h", f"{db}_h.index", f"{db}_h.dbtype",
        clu_db, f"{clu_db}.index", f"{clu_db}.dbtype"
    ]
    for pattern in cleanup_patterns:
        if os.path.exists(pattern):
            if os.path.isdir(pattern):
                subprocess.run(['rm', '-rf', pattern], capture_output=True)
            else:
                os.unlink(pattern)

    # Count clusters
    num_lines = int(subprocess.check_output(['wc', '-l', output_tsv]).split()[0])
    print(f"    ✓ Done: {num_lines:,} cluster assignments")

    return output_tsv


# ═══════════════════════════════════════════════════════════════════
# STEP 4: MAP CLUSTERS BACK TO PICKLE CHUNKS
# ═══════════════════════════════════════════════════════════════════

def split_tsv_by_chunk(tsv_file, output_dir, id_prefix):
    """
    Split a large TSV file into per-chunk TSV files using awk (single pass, very fast).
    This is much faster than grep per chunk.

    Args:
        tsv_file: Path to the large cluster TSV file
        output_dir: Directory to write per-chunk TSV files
        id_prefix: Prefix used in FASTA IDs (e.g., "unpaired_heavy")

    Returns:
        dict: {chunk_name: split_tsv_path}
    """
    if not tsv_file or not os.path.exists(tsv_file):
        return {}

    os.makedirs(output_dir, exist_ok=True)
    base_name = Path(tsv_file).stem

    print(f"    Splitting {tsv_file} by chunk pattern...")

    # Use awk to split by chunk pattern in a single pass
    # Extract chunk name from seq_id pattern: {id_prefix}_{chunk_name}_seq_{idx}
    awk_script = f'''
    BEGIN {{ FS="\\t"; OFS="\\t" }}
    {{
        # seq_id is in column 2
        seq_id = $2
        # Find pattern: look for id_prefix, then extract chunk_name before _seq_
        if (match(seq_id, /^{id_prefix}_(.+)_seq_[0-9]+$/, arr)) {{
            chunk = arr[1]
            print $0 >> "{output_dir}/{base_name}_" chunk ".tsv"
        }}
    }}
    '''

    # Run awk
    subprocess.run(['awk', awk_script, tsv_file], check=True)

    # Find generated files
    split_files = glob.glob(os.path.join(output_dir, f"{base_name}_*.tsv"))
    chunk_to_file = {}
    for f in split_files:
        # Extract chunk name from filename
        fname = Path(f).stem  # e.g., "all_heavy_whole_clusters_unpaired_HEAVY_filtered_p90_chunk_0"
        # Remove base_name prefix
        chunk_name = fname[len(base_name)+1:]  # e.g., "unpaired_HEAVY_filtered_p90_chunk_0"
        chunk_to_file[chunk_name] = f

    print(f"    Split into {len(chunk_to_file)} chunk files")
    return chunk_to_file


def load_cluster_array_from_split_tsv(split_tsv, id_prefix, chunk_name, num_rows):
    """
    Load cluster array from a pre-split per-chunk TSV file.
    Much faster since the file only contains relevant data.
    """
    if not split_tsv or not os.path.exists(split_tsv):
        return [None] * num_rows

    pattern = f"{id_prefix}_{chunk_name}_seq_"
    prefix_len = len(pattern)

    clusters = [None] * num_rows

    with open(split_tsv, 'r') as f:
        for line in f:
            tab_pos = line.find('\t')
            if tab_pos == -1:
                continue
            cluster_id = line[:tab_pos]
            seq_id = line[tab_pos+1:].rstrip('\n')

            if seq_id.startswith(pattern):
                try:
                    idx = int(seq_id[prefix_len:])
                    if 0 <= idx < num_rows:
                        clusters[idx] = cluster_id
                except ValueError:
                    pass

    return clusters


def load_paired_clusters_from_tsv(tsv_file, id_prefix, num_rows):
    """
    Load cluster array for paired data using grep (paired data is smaller).
    Pattern: {id_prefix}_seq_{idx}
    """
    if not tsv_file or not os.path.exists(tsv_file):
        return [None] * num_rows

    pattern = f"{id_prefix}_seq_"
    prefix_len = len(pattern)

    clusters = [None] * num_rows

    # Use grep to extract only paired data lines
    try:
        proc = subprocess.Popen(
            ['grep', '-F', pattern, tsv_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=65536
        )

        for line in proc.stdout:
            tab_pos = line.find('\t')
            if tab_pos == -1:
                continue
            cluster_id = line[:tab_pos]
            seq_id = line[tab_pos+1:].rstrip('\n')

            if seq_id.startswith(pattern):
                try:
                    idx = int(seq_id[prefix_len:])
                    if 0 <= idx < num_rows:
                        clusters[idx] = cluster_id
                except ValueError:
                    pass

        proc.wait()
    except Exception as e:
        print(f"    Warning: Failed to load paired clusters: {e}")

    return clusters


def add_clusters_to_pickles(pickle_files, cdr3_tsv, whole_tsv,
                            seq_column, id_prefix, output_dir):
    """
    Add cluster IDs to pickle files and save to output directory.
    Strategy: Split TSV once, then process each chunk from its small split file.

    Args:
        pickle_files: List of input pickle files
        cdr3_tsv: Path to CDR3 cluster TSV file (or None)
        whole_tsv: Path to whole sequence cluster TSV file
        seq_column: Column name with sequences
        id_prefix: Prefix used in FASTA IDs
        output_dir: Where to save updated pickles
    """
    print(f"  Adding clusters to pickle files...")
    os.makedirs(output_dir, exist_ok=True)

    # Create temp directory for split files
    split_dir = os.path.join(output_dir, "_split_tsv_temp")
    os.makedirs(split_dir, exist_ok=True)

    # Split TSV files by chunk (single pass each)
    cdr3_splits = {}
    if cdr3_tsv and os.path.exists(cdr3_tsv):
        print(f"  [1/3] Splitting CDR3 TSV...")
        cdr3_splits = split_tsv_by_chunk(cdr3_tsv, split_dir, id_prefix)

    print(f"  [2/3] Splitting whole sequence TSV...")
    whole_splits = split_tsv_by_chunk(whole_tsv, split_dir, id_prefix)

    # Process each pickle file using its split TSV
    print(f"  [3/3] Processing pickle files...")
    for pkl_file in tqdm(pickle_files, desc="    Chunks"):
        chunk_name = Path(pkl_file).stem

        df = pd.read_pickle(pkl_file)
        num_rows = len(df)

        # Load from split files (small, fast)
        cdr3_split_file = cdr3_splits.get(chunk_name)
        whole_split_file = whole_splits.get(chunk_name)

        df['cdr3_cluster_id'] = load_cluster_array_from_split_tsv(
            cdr3_split_file, id_prefix, chunk_name, num_rows)
        df['whole_seq_cluster_id'] = load_cluster_array_from_split_tsv(
            whole_split_file, id_prefix, chunk_name, num_rows)

        # Save
        output_file = os.path.join(output_dir, Path(pkl_file).name)
        df.to_pickle(output_file)

        del df
        gc.collect()

    # Cleanup split files
    print(f"  Cleaning up temp split files...")
    subprocess.run(['rm', '-rf', split_dir], capture_output=True)


# ═══════════════════════════════════════════════════════════════════
# MAIN WORKFLOW
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Cluster antibody sequences using FASTA workflow")
    parser.add_argument('--threads', type=int, default=64, help="MMseqs2 threads")
    parser.add_argument('--anarci-processes', type=int, default=64, help="ANARCI parallel processes")
    parser.add_argument('--skip-anarci', action='store_true', help="Skip ANARCI CDR3 extraction")
    parser.add_argument('--memory-limit', type=str, default="120G",
                        help="MMseqs2 memory limit (e.g., '120G', '200G'). Set to ~60%% of available RAM. Default: 120G")
    args = parser.parse_args()

    global MMSEQS_THREADS, ANARCI_PROCESSES, MMSEQS_MEMORY_LIMIT
    MMSEQS_THREADS = args.threads
    ANARCI_PROCESSES = args.anarci_processes
    MMSEQS_MEMORY_LIMIT = args.memory_limit

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FASTA_DIR, exist_ok=True)

    print("="*70)
    print("ANTIBODY SEQUENCE CLUSTERING (FASTA-BASED)")
    print("="*70)

    # Get input files
    heavy_pickles = sorted(glob.glob(UNPAIRED_HEAVY_PATTERN))
    light_pickles = sorted(glob.glob(UNPAIRED_LIGHT_PATTERN))

    print(f"\nFound {len(heavy_pickles)} heavy chunks, {len(light_pickles)} light chunks")
    print(f"Paired data: {PAIRED_DATA_PATH}")

    # ===================================================================
    # STEP 1: WRITE ALL FASTA FILES (PAIRED + UNPAIRED)
    # ===================================================================
    print("\n" + "="*70)
    print("STEP 1: WRITING FASTA FILES")
    print("="*70)

    # Unpaired heavy
    print("\n[1a] Writing unpaired heavy FASTA...")
    unpaired_heavy_fasta = os.path.join(FASTA_DIR, "unpaired_heavy.fasta")
    write_fasta_from_pickles(heavy_pickles, unpaired_heavy_fasta,
                             'sequence_alignment_aa', 'unpaired_heavy')

    # Unpaired light
    print("\n[1b] Writing unpaired light FASTA...")
    unpaired_light_fasta = os.path.join(FASTA_DIR, "unpaired_light.fasta")
    write_fasta_from_pickles(light_pickles, unpaired_light_fasta,
                             'sequence_alignment_aa', 'unpaired_light')

    # Paired heavy + light
    print("\n[1c] Writing paired FASTA...")
    paired_base = os.path.join(FASTA_DIR, "paired")
    paired_hc_fasta = f"{paired_base}_heavy.fasta"
    paired_lc_fasta = f"{paired_base}_light.fasta"
    write_fasta_from_single_pickle(PAIRED_DATA_PATH, paired_base,
                                   'HEAVY_CHAIN_AA_SEQUENCE',
                                   'LIGHT_CHAIN_AA_SEQUENCE')

    # Combine paired + unpaired for clustering
    print("\n[1d] Combining paired + unpaired FASTA...")
    all_heavy_fasta = os.path.join(FASTA_DIR, "all_heavy.fasta")
    all_light_fasta = os.path.join(FASTA_DIR, "all_light.fasta")

    # Check if combined files already exist and are valid (non-empty)
    heavy_exists = os.path.exists(all_heavy_fasta) and os.path.getsize(all_heavy_fasta) > 0
    light_exists = os.path.exists(all_light_fasta) and os.path.getsize(all_light_fasta) > 0

    if heavy_exists and light_exists:
        heavy_size_gb = os.path.getsize(all_heavy_fasta) / (1024**3)
        light_size_gb = os.path.getsize(all_light_fasta) / (1024**3)
        print(f"  Combined FASTA already exists:")
        print(f"    Heavy: {all_heavy_fasta} ({heavy_size_gb:.2f} GB)")
        print(f"    Light: {all_light_fasta} ({light_size_gb:.2f} GB)")
    else:
        # Use cat to combine (fast)
        if not heavy_exists:
            print(f"  Creating combined heavy FASTA...")
            subprocess.run(['cat', unpaired_heavy_fasta, paired_hc_fasta],
                           stdout=open(all_heavy_fasta, 'w'), check=True)
            print(f"  ✓ Combined heavy FASTA: {all_heavy_fasta}")

        if not light_exists:
            print(f"  Creating combined light FASTA...")
            subprocess.run(['cat', unpaired_light_fasta, paired_lc_fasta],
                           stdout=open(all_light_fasta, 'w'), check=True)
            print(f"  ✓ Combined light FASTA: {all_light_fasta}")

    # ===================================================================
    # STEP 2: EXTRACT CDR3 WITH ANARCI
    # ===================================================================
    print("\n" + "="*70)
    print("STEP 2: EXTRACTING CDR3 WITH ANARCI")
    print("="*70)

    all_heavy_cdr3_fasta = os.path.join(FASTA_DIR, "all_heavy_cdr3.fasta")
    all_light_cdr3_fasta = os.path.join(FASTA_DIR, "all_light_cdr3.fasta")

    if not args.skip_anarci:
        print("\n[2a] Extracting heavy chain CDR3...")
        hc_cdr3_count = extract_cdr3_with_anarci(all_heavy_fasta, all_heavy_cdr3_fasta, 'H')
        # -1 means file exists (skipped), 0 means extraction failed
        if hc_cdr3_count == 0:
            print("\n" + "="*70)
            print("ERROR: Heavy chain CDR3 extraction failed!")
            print("="*70)
            print(f"Check ANARCI output: {all_heavy_fasta.replace('.fasta', '_anarci.txt')}")
            sys.exit(1)

        print("\n[2b] Extracting light chain CDR3...")
        lc_cdr3_count = extract_cdr3_with_anarci(all_light_fasta, all_light_cdr3_fasta, 'L')
        # -1 means file exists (skipped), 0 means extraction failed
        if lc_cdr3_count == 0:
            print("\n" + "="*70)
            print("ERROR: Light chain CDR3 extraction failed!")
            print("="*70)
            print(f"Check ANARCI output: {all_light_fasta.replace('.fasta', '_anarci.txt')}")
            sys.exit(1)
    else:
        print("\n[2] Skipping ANARCI (--skip-anarci)")

    # ===================================================================
    # STEP 3: CLUSTER ALL SEQUENCES (PAIRED + UNPAIRED TOGETHER)
    # ===================================================================
    print("\n" + "="*70)
    print("STEP 3: CLUSTERING (PAIRED + UNPAIRED TOGETHER)")
    print("="*70)

    # Heavy chain clustering
    print("\n[3a] Clustering heavy chains...")
    hc_cdr3_tsv = None
    if (not args.skip_anarci and os.path.exists(all_heavy_cdr3_fasta) and
        os.path.getsize(all_heavy_cdr3_fasta) > 0):
        hc_cdr3_tsv = os.path.join(FASTA_DIR, "all_heavy_cdr3_clusters.tsv")
        run_mmseqs_linclust(all_heavy_cdr3_fasta, hc_cdr3_tsv, CDR3_IDENTITY,
                           coverage=0.8, threads=MMSEQS_THREADS,
                           memory_limit=MMSEQS_MEMORY_LIMIT)
    else:
        if not args.skip_anarci:
            print("  Skipping CDR3 clustering (no CDR3 sequences extracted)")

    hc_whole_tsv = os.path.join(FASTA_DIR, "all_heavy_whole_clusters.tsv")
    run_mmseqs_linclust(all_heavy_fasta, hc_whole_tsv, WHOLE_SEQ_IDENTITY,
                       coverage=0.8, threads=MMSEQS_THREADS,
                       memory_limit=MMSEQS_MEMORY_LIMIT)

    # Light chain clustering
    print("\n[3b] Clustering light chains...")
    lc_cdr3_tsv = None
    if (not args.skip_anarci and os.path.exists(all_light_cdr3_fasta) and
        os.path.getsize(all_light_cdr3_fasta) > 0):
        lc_cdr3_tsv = os.path.join(FASTA_DIR, "all_light_cdr3_clusters.tsv")
        run_mmseqs_linclust(all_light_cdr3_fasta, lc_cdr3_tsv, CDR3_IDENTITY,
                           coverage=0.8, threads=MMSEQS_THREADS,
                           memory_limit=MMSEQS_MEMORY_LIMIT)
    else:
        if not args.skip_anarci:
            print("  Skipping CDR3 clustering (no CDR3 sequences extracted)")

    lc_whole_tsv = os.path.join(FASTA_DIR, "all_light_whole_clusters.tsv")
    run_mmseqs_linclust(all_light_fasta, lc_whole_tsv, WHOLE_SEQ_IDENTITY,
                       coverage=0.8, threads=MMSEQS_THREADS,
                       memory_limit=MMSEQS_MEMORY_LIMIT)

    # ===================================================================
    # STEP 4: MAP CLUSTERS BACK TO PICKLE FILES
    # ===================================================================
    print("\n" + "="*70)
    print("STEP 4: MAPPING CLUSTERS TO PICKLE FILES")
    print("="*70)

    # Map to unpaired heavy chunks
    print("\n[4a] Mapping clusters to unpaired heavy chunks...")
    add_clusters_to_pickles(heavy_pickles, hc_cdr3_tsv, hc_whole_tsv,
                           'sequence_alignment_aa', 'unpaired_heavy',
                           os.path.join(OUTPUT_DIR, 'heavy_chunks'))

    print("✓ Unpaired heavy complete!")

    # Map to unpaired light chunks
    print("\n[4b] Mapping clusters to unpaired light chunks...")
    add_clusters_to_pickles(light_pickles, lc_cdr3_tsv, lc_whole_tsv,
                           'sequence_alignment_aa', 'unpaired_light',
                           os.path.join(OUTPUT_DIR, 'light_chunks'))

    print("✓ Unpaired light complete!")

    # Map to paired data (use grep since paired data is smaller)
    print("\n[4c] Mapping clusters to paired data...")
    df = pd.read_pickle(PAIRED_DATA_PATH)
    num_rows = len(df)

    # Heavy chain cluster IDs (using 'paired_hc' as prefix)
    print("    Loading paired heavy chain clusters...")
    df['hc_cdr3_cluster_id'] = load_paired_clusters_from_tsv(
        hc_cdr3_tsv, 'paired_hc', num_rows)
    df['hc_whole_seq_cluster_id'] = load_paired_clusters_from_tsv(
        hc_whole_tsv, 'paired_hc', num_rows)
    gc.collect()

    # Light chain cluster IDs (using 'paired_lc' as prefix)
    print("    Loading paired light chain clusters...")
    df['lc_cdr3_cluster_id'] = load_paired_clusters_from_tsv(
        lc_cdr3_tsv, 'paired_lc', num_rows)
    df['lc_whole_seq_cluster_id'] = load_paired_clusters_from_tsv(
        lc_whole_tsv, 'paired_lc', num_rows)
    gc.collect()

    output_paired = os.path.join(OUTPUT_DIR, "paired_with_clusters.pkl")
    df.to_pickle(output_paired)
    print(f"  ✓ Saved: {output_paired}")

    del df
    gc.collect()

    print("✓ Paired data complete!")

    # ===================================================================
    # CLEANUP
    # ===================================================================
    print("\n" + "="*70)
    print("CLEANUP")
    print("="*70)

    print(f"  Removing temporary FASTA directory: {FASTA_DIR}")
    subprocess.run(['rm', '-rf', FASTA_DIR], check=True, capture_output=True)

    print("\n" + "="*70)
    print("✓ CLUSTERING COMPLETE!")
    print("="*70)
    print(f"  Heavy chunks: {OUTPUT_DIR}/heavy_chunks/")
    print(f"  Light chunks: {OUTPUT_DIR}/light_chunks/")
    print(f"  Paired data:  {output_paired}")


if __name__ == "__main__":
    main()
