#!/usr/bin/env python
# coding: utf-8

"""
Cluster Paired and Unpaired Antibody Sequences (Chunk-Based)
==============================================================

Memory-efficient version that processes pickle chunks one at a time.

Workflow:
1. Process each unpaired pickle chunk: load → extract CDR3 → save with CDR3 column
2. Combine all sequences into FASTA files for clustering
3. Run MMseqs2 clustering (CDR3 100% + whole seq 95%)
4. Map cluster IDs back to chunks and save final results

This approach never loads more than ~30GB at once (1 pickle chunk).
"""

import os
import sys
import gc
import glob
import pandas as pd
import subprocess
import tempfile
from pathlib import Path
from tqdm.auto import tqdm
import argparse
from multiprocessing import Pool

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

PAIRED_DATA_PATH = "../merged_antibody_sequences.pkl"
UNPAIRED_HEAVY_PATTERN = "./unpaired_HEAVY_filtered_p90_chunk_*.pkl"
UNPAIRED_LIGHT_PATTERN = "./unpaired_LIGHT_filtered_p90_chunk_*.pkl"

OUTPUT_DIR = "./clustered_data"
TEMP_DIR = "./temp_clustering"

# MMseqs2 parameters
MMSEQS_THREADS = 8
CDR3_IDENTITY = 1.00
WHOLE_SEQ_IDENTITY = 0.95

# ANARCI parameters
USE_ANARCI = True
ANARCI_BATCH_SIZE = 10000
ANARCI_N_PROCESSES = 4

# ═══════════════════════════════════════════════════════════════════
# DEPENDENCY CHECKS
# ═══════════════════════════════════════════════════════════════════

def check_mmseqs2():
    try:
        result = subprocess.run(['mmseqs', 'version'], capture_output=True, text=True)
        print(f"✓ Found MMseqs2: {result.stdout.strip()}")
        return True
    except FileNotFoundError:
        print("✗ MMseqs2 not found! Install with: conda install -c bioconda mmseqs2")
        return False


def check_anarci():
    try:
        subprocess.run(['ANARCI', '--help'], capture_output=True, text=True)
        print(f"✓ Found ANARCI")
        return True
    except FileNotFoundError:
        print("✗ ANARCI not found! Install with: pip install anarci")
        return False


# ═══════════════════════════════════════════════════════════════════
# CDR3 EXTRACTION
# ═══════════════════════════════════════════════════════════════════

def process_anarci_batch(args):
    """Process a batch of sequences with ANARCI (for multiprocessing)."""
    sequences, chain_type, batch_id = args
    cdr3_results = []

    with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
        temp_fasta = f.name
        for idx, seq in enumerate(sequences):
            if pd.notna(seq) and len(seq) > 0:
                f.write(f">seq_{idx}\n{seq}\n")

    try:
        result = subprocess.run(
            ['ANARCI', '-i', temp_fasta, '--scheme', 'imgt'],
            capture_output=True, text=True, timeout=600
        )

        seq_cdr3_map = {}
        current_seq_id = None
        numbering = {}

        for line in result.stdout.split('\n'):
            if line.startswith('>'):
                if current_seq_id is not None and numbering:
                    cdr3 = ''.join([numbering.get(i, '-') for i in range(105, 118)])
                    cdr3 = cdr3.replace('-', '')
                    seq_cdr3_map[current_seq_id] = cdr3 if len(cdr3) >= 5 else None
                current_seq_id = line.split()[0][1:]
                numbering = {}
            elif line.strip() and not line.startswith('#') and not line.startswith('//'):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        pos = int(parts[0])
                        aa = parts[1]
                        if aa != '-':
                            numbering[pos] = aa
                    except (ValueError, IndexError):
                        continue

        if current_seq_id is not None and numbering:
            cdr3 = ''.join([numbering.get(i, '-') for i in range(105, 118)])
            cdr3 = cdr3.replace('-', '')
            seq_cdr3_map[current_seq_id] = cdr3 if len(cdr3) >= 5 else None

        for idx in range(len(sequences)):
            cdr3_results.append(seq_cdr3_map.get(f"seq_{idx}", None))

    except Exception as e:
        print(f"    Warning: ANARCI failed for batch {batch_id}: {e}")
        cdr3_results = [None] * len(sequences)
    finally:
        try:
            os.unlink(temp_fasta)
        except:
            pass

    return cdr3_results


def extract_cdr3_with_anarci_parallel(sequences, chain_type='H', batch_size=10000, n_processes=4):
    """Extract CDR3 with ANARCI using multiprocessing."""
    seq_list = list(sequences)
    batches = []
    for i in range(0, len(seq_list), batch_size):
        batch = seq_list[i:i+batch_size]
        batches.append((batch, chain_type, i//batch_size))

    all_cdr3 = []
    with Pool(processes=n_processes) as pool:
        for batch_results in tqdm(pool.imap(process_anarci_batch, batches),
                                   total=len(batches), desc=f"    ANARCI ({chain_type})"):
            all_cdr3.extend(batch_results)

    return all_cdr3


def extract_cdr3_heuristic(sequence):
    """Heuristic CDR3 extraction (fallback)."""
    if pd.isna(sequence) or len(sequence) < 50:
        return None

    AA = set("ACDEFGHIKLMNPQRSTVWY")
    c_positions = [i for i, aa in enumerate(sequence) if aa == 'C']
    wg_positions = [i for i in range(len(sequence) - 1) if sequence[i:i+2] == 'WG']

    valid_c = [pos for pos in c_positions if 70 <= pos <= 110]
    valid_wg = [pos for pos in wg_positions if 90 <= pos <= 130]

    if valid_c and valid_wg:
        c_pos = max([c for c in valid_c if c < min(valid_wg)])
        wg_pos = min(valid_wg)
        cdr3 = sequence[c_pos:wg_pos+1]
        if 5 <= len(cdr3) <= 35:
            return cdr3
    return None


# ═══════════════════════════════════════════════════════════════════
# CHUNK PROCESSING
# ═══════════════════════════════════════════════════════════════════

def process_unpaired_chunk(pickle_file, chain_type, output_dir, use_anarci=True):
    """
    Process one unpaired pickle chunk: load, extract CDR3, save.

    Returns: path to processed chunk file
    """
    print(f"\n  Processing chunk: {os.path.basename(pickle_file)}")

    # Load chunk
    df = pd.read_pickle(pickle_file)
    print(f"    Loaded: {len(df):,} sequences")

    # Extract CDR3
    if use_anarci:
        try:
            chain_letter = 'H' if chain_type == 'heavy' else 'L'
            cdr3_list = extract_cdr3_with_anarci_parallel(
                df['sequence_alignment_aa'],
                chain_type=chain_letter,
                batch_size=ANARCI_BATCH_SIZE,
                n_processes=ANARCI_N_PROCESSES
            )
            df['cdr3_seq'] = cdr3_list
        except Exception as e:
            print(f"    ⚠ ANARCI failed: {e}, using heuristic")
            df['cdr3_seq'] = df['sequence_alignment_aa'].apply(extract_cdr3_heuristic)
    else:
        df['cdr3_seq'] = df['sequence_alignment_aa'].apply(extract_cdr3_heuristic)

    cdr3_count = df['cdr3_seq'].notna().sum()
    print(f"    CDR3 extracted: {cdr3_count:,} / {len(df):,} ({100*cdr3_count/len(df):.1f}%)")

    # Add unique sequence IDs
    chunk_id = os.path.basename(pickle_file).replace('.pkl', '')
    df['seq_id'] = [f"{chunk_id}_seq_{i}" for i in range(len(df))]
    df['source'] = 'unpaired'
    df['chunk_file'] = os.path.basename(pickle_file)

    # Save processed chunk
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{chunk_id}_with_cdr3.pkl")
    df.to_pickle(output_file)
    print(f"    Saved: {output_file}")

    del df
    gc.collect()

    return output_file


def process_paired_data(paired_file, output_dir, use_anarci=True):
    """Process paired data: load, extract CDR3 for both chains, save."""
    print(f"\n  Processing paired data: {paired_file}")

    df = pd.read_pickle(paired_file)
    print(f"    Loaded: {len(df):,} paired sequences")

    df['paired_index'] = df.index
    df['source'] = 'paired'

    # Extract heavy chain CDR3
    print(f"    Extracting heavy chain CDR3...")
    if use_anarci:
        try:
            hc_cdr3 = extract_cdr3_with_anarci_parallel(
                df['HEAVY_CHAIN_AA_SEQUENCE'],
                chain_type='H',
                batch_size=ANARCI_BATCH_SIZE,
                n_processes=ANARCI_N_PROCESSES
            )
            df['hc_cdr3_seq'] = hc_cdr3
        except Exception as e:
            print(f"    ⚠ ANARCI failed for HC: {e}")
            df['hc_cdr3_seq'] = df['HEAVY_CHAIN_AA_SEQUENCE'].apply(extract_cdr3_heuristic)
    else:
        df['hc_cdr3_seq'] = df['HEAVY_CHAIN_AA_SEQUENCE'].apply(extract_cdr3_heuristic)

    # Extract light chain CDR3
    print(f"    Extracting light chain CDR3...")
    if use_anarci:
        try:
            lc_cdr3 = extract_cdr3_with_anarci_parallel(
                df['LIGHT_CHAIN_AA_SEQUENCE'],
                chain_type='L',
                batch_size=ANARCI_BATCH_SIZE,
                n_processes=ANARCI_N_PROCESSES
            )
            df['lc_cdr3_seq'] = lc_cdr3
        except Exception as e:
            print(f"    ⚠ ANARCI failed for LC: {e}")
            df['lc_cdr3_seq'] = df['LIGHT_CHAIN_AA_SEQUENCE'].apply(extract_cdr3_heuristic)
    else:
        df['lc_cdr3_seq'] = df['LIGHT_CHAIN_AA_SEQUENCE'].apply(extract_cdr3_heuristic)

    hc_count = df['hc_cdr3_seq'].notna().sum()
    lc_count = df['lc_cdr3_seq'].notna().sum()
    print(f"    HC CDR3: {hc_count:,} / {len(df):,} ({100*hc_count/len(df):.1f}%)")
    print(f"    LC CDR3: {lc_count:,} / {len(df):,} ({100*lc_count/len(df):.1f}%)")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "paired_with_cdr3.pkl")
    df.to_pickle(output_file)
    print(f"    Saved: {output_file}")

    del df
    gc.collect()

    return output_file


# ═══════════════════════════════════════════════════════════════════
# FASTA GENERATION
# ═══════════════════════════════════════════════════════════════════

def write_fasta_from_chunks(chunk_files, seq_column, id_column, output_fasta, cdr3_only=False):
    """Write FASTA file from multiple chunk files using efficient iteration."""
    print(f"  Writing FASTA: {output_fasta}")

    total_seqs = 0
    with open(output_fasta, 'w') as f:
        for chunk_file in tqdm(chunk_files, desc="    Chunks"):
            df = pd.read_pickle(chunk_file)

            if cdr3_only:
                df = df[df[seq_column].notna()]

            # Efficient vectorized operation
            for seq_id, seq in zip(df[id_column], df[seq_column]):
                if pd.notna(seq) and len(seq) > 0:
                    f.write(f">{seq_id}\n{seq}\n")
                    total_seqs += 1

            del df
            gc.collect()

    print(f"    Wrote {total_seqs:,} sequences")
    return total_seqs


# ═══════════════════════════════════════════════════════════════════
# CLUSTERING
# ═══════════════════════════════════════════════════════════════════

def run_mmseqs_linclust(input_fasta, output_prefix, identity, coverage=0.8, threads=8):
    """Run MMseqs2 Linclust with progress output."""
    print(f"  Running MMseqs2 Linclust (id={identity}, cov={coverage})...")

    tmp_dir = f"{output_prefix}_tmp"
    os.makedirs(tmp_dir, exist_ok=True)

    db = f"{output_prefix}_DB"

    print(f"    Creating database...")
    subprocess.run(['mmseqs', 'createdb', input_fasta, db], check=True)

    clu_db = f"{output_prefix}_clu"
    print(f"    Clustering with {threads} threads...")
    subprocess.run([
        'mmseqs', 'linclust', db, clu_db, tmp_dir,
        '--min-seq-id', str(identity),
        '-c', str(coverage),
        '--threads', str(threads),
        '--cov-mode', '1',
        '-v', '3'  # Verbosity for progress
    ], check=True)

    tsv_output = f"{output_prefix}_clusters.tsv"
    print(f"    Generating TSV output...")
    subprocess.run(['mmseqs', 'createtsv', db, db, clu_db, tsv_output], check=True)

    print(f"    Cleaning up temporary files...")
    subprocess.run(['rm', '-rf', tmp_dir, db, db + '.index', clu_db, clu_db + '.index'],
                   check=True, capture_output=True)

    print(f"  ✓ Done: {tsv_output}")
    return tsv_output


def parse_cluster_tsv(tsv_path):
    """Parse MMseqs2 TSV to get {seq_id: cluster_id} mapping using efficient pandas read."""
    # Much faster than line-by-line reading for large files
    df = pd.read_csv(tsv_path, sep='\t', header=None, names=['cluster_id', 'seq_id'])
    clusters = dict(zip(df['seq_id'], df['cluster_id']))
    del df
    gc.collect()
    return clusters


def add_cluster_ids_to_chunks(chunk_files, cluster_map, cluster_col_name):
    """Add cluster IDs to chunk files in-place."""
    print(f"  Adding {cluster_col_name} to chunks...")

    for chunk_file in tqdm(chunk_files, desc="    Chunks"):
        df = pd.read_pickle(chunk_file)
        df[cluster_col_name] = df['seq_id'].map(cluster_map)
        df.to_pickle(chunk_file)
        del df
        gc.collect()


# ═══════════════════════════════════════════════════════════════════
# MAIN WORKFLOW
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Cluster antibody sequences (chunk-based)")
    parser.add_argument('--threads', type=int, default=8, help="MMseqs2 threads")
    parser.add_argument('--no-anarci', action='store_true', help="Use heuristic CDR3")
    parser.add_argument('--anarci-processes', type=int, default=4, help="ANARCI parallel processes")
    parser.add_argument('--keep-chunks', action='store_true',
                        help="Keep individual pickle chunks as final output (saves memory)")
    args = parser.parse_args()

    global MMSEQS_THREADS, USE_ANARCI, ANARCI_N_PROCESSES
    MMSEQS_THREADS = args.threads
    ANARCI_N_PROCESSES = args.anarci_processes
    USE_ANARCI = not args.no_anarci

    # Check dependencies
    if not check_mmseqs2():
        sys.exit(1)
    if USE_ANARCI and not check_anarci():
        print("⚠ ANARCI not found, using heuristic")
        USE_ANARCI = False

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)

    # STEP 1: Process paired data
    print("\n" + "="*70)
    print("STEP 1: PROCESS PAIRED DATA")
    print("="*70)
    paired_processed = process_paired_data(PAIRED_DATA_PATH, OUTPUT_DIR, USE_ANARCI)

    # STEP 2: Process unpaired heavy chunks
    print("\n" + "="*70)
    print("STEP 2: PROCESS UNPAIRED HEAVY CHUNKS")
    print("="*70)
    heavy_files = sorted(glob.glob(UNPAIRED_HEAVY_PATTERN))
    print(f"Found {len(heavy_files)} heavy chain chunks")

    heavy_processed = []
    for hf in heavy_files:
        processed = process_unpaired_chunk(hf, 'heavy', OUTPUT_DIR, USE_ANARCI)
        heavy_processed.append(processed)

    # STEP 3: Process unpaired light chunks
    print("\n" + "="*70)
    print("STEP 3: PROCESS UNPAIRED LIGHT CHUNKS")
    print("="*70)
    light_files = sorted(glob.glob(UNPAIRED_LIGHT_PATTERN))
    print(f"Found {len(light_files)} light chain chunks")

    light_processed = []
    for lf in light_files:
        processed = process_unpaired_chunk(lf, 'light', OUTPUT_DIR, USE_ANARCI)
        light_processed.append(processed)

    # STEP 4: Cluster heavy chains
    print("\n" + "="*70)
    print("STEP 4: CLUSTER HEAVY CHAINS")
    print("="*70)

    # Write FASTAs
    hc_cdr3_fasta = os.path.join(TEMP_DIR, "heavy_cdr3.fasta")
    hc_whole_fasta = os.path.join(TEMP_DIR, "heavy_whole.fasta")

    # For paired: use HEAVY_CHAIN_AA_SEQUENCE and hc_cdr3_seq
    # For unpaired: use sequence_alignment_aa and cdr3_seq
    # We need to handle both...

    print("\n  Writing heavy chain FASTAs...")
    print("    Note: Combining paired and unpaired sequences")

    # Heavy CDR3 clustering
    print("\n  [1/2] Heavy CDR3 clustering")
    with open(hc_cdr3_fasta, 'w') as f:
        # Paired heavy
        df_p = pd.read_pickle(paired_processed)
        df_p_hc = df_p[df_p['hc_cdr3_seq'].notna()].copy()
        df_p_hc['seq_id'] = ['paired_hc_' + str(i) for i in df_p_hc['paired_index']]
        for _, row in df_p_hc.iterrows():
            f.write(f">{row['seq_id']}\n{row['hc_cdr3_seq']}\n")
        del df_p, df_p_hc
        gc.collect()

        # Unpaired heavy
        for hf in tqdm(heavy_processed, desc="    Unpaired chunks"):
            df_u = pd.read_pickle(hf)
            df_u_cdr3 = df_u[df_u['cdr3_seq'].notna()]
            for _, row in df_u_cdr3.iterrows():
                f.write(f">{row['seq_id']}\n{row['cdr3_seq']}\n")
            del df_u, df_u_cdr3
            gc.collect()

    hc_cdr3_tsv = run_mmseqs_linclust(hc_cdr3_fasta, os.path.join(TEMP_DIR, "hc_cdr3"),
                                      CDR3_IDENTITY, 1.0, MMSEQS_THREADS)
    hc_cdr3_clusters = parse_cluster_tsv(hc_cdr3_tsv)

    # Heavy whole sequence clustering
    print("\n  [2/2] Heavy whole sequence clustering")
    with open(hc_whole_fasta, 'w') as f:
        # Paired
        df_p = pd.read_pickle(paired_processed)
        df_p_hc = df_p[df_p['HEAVY_CHAIN_AA_SEQUENCE'].notna()].copy()
        df_p_hc['seq_id'] = ['paired_hc_' + str(i) for i in df_p_hc['paired_index']]
        for _, row in df_p_hc.iterrows():
            f.write(f">{row['seq_id']}\n{row['HEAVY_CHAIN_AA_SEQUENCE']}\n")
        del df_p, df_p_hc
        gc.collect()

        # Unpaired
        for hf in tqdm(heavy_processed, desc="    Unpaired chunks"):
            df_u = pd.read_pickle(hf)
            for _, row in df_u.iterrows():
                if pd.notna(row['sequence_alignment_aa']):
                    f.write(f">{row['seq_id']}\n{row['sequence_alignment_aa']}\n")
            del df_u
            gc.collect()

    hc_whole_tsv = run_mmseqs_linclust(hc_whole_fasta, os.path.join(TEMP_DIR, "hc_whole"),
                                       WHOLE_SEQ_IDENTITY, 0.8, MMSEQS_THREADS)
    hc_whole_clusters = parse_cluster_tsv(hc_whole_tsv)

    # STEP 5: Cluster light chains (similar to heavy)
    print("\n" + "="*70)
    print("STEP 5: CLUSTER LIGHT CHAINS")
    print("="*70)

    lc_cdr3_fasta = os.path.join(TEMP_DIR, "light_cdr3.fasta")
    lc_whole_fasta = os.path.join(TEMP_DIR, "light_whole.fasta")

    print("\n  [1/2] Light CDR3 clustering")
    with open(lc_cdr3_fasta, 'w') as f:
        df_p = pd.read_pickle(paired_processed)
        df_p_lc = df_p[df_p['lc_cdr3_seq'].notna()].copy()
        df_p_lc['seq_id'] = ['paired_lc_' + str(i) for i in df_p_lc['paired_index']]
        for _, row in df_p_lc.iterrows():
            f.write(f">{row['seq_id']}\n{row['lc_cdr3_seq']}\n")
        del df_p, df_p_lc
        gc.collect()

        for lf in tqdm(light_processed, desc="    Unpaired chunks"):
            df_u = pd.read_pickle(lf)
            df_u_cdr3 = df_u[df_u['cdr3_seq'].notna()]
            for _, row in df_u_cdr3.iterrows():
                f.write(f">{row['seq_id']}\n{row['cdr3_seq']}\n")
            del df_u, df_u_cdr3
            gc.collect()

    lc_cdr3_tsv = run_mmseqs_linclust(lc_cdr3_fasta, os.path.join(TEMP_DIR, "lc_cdr3"),
                                      CDR3_IDENTITY, 1.0, MMSEQS_THREADS)
    lc_cdr3_clusters = parse_cluster_tsv(lc_cdr3_tsv)

    print("\n  [2/2] Light whole sequence clustering")
    with open(lc_whole_fasta, 'w') as f:
        df_p = pd.read_pickle(paired_processed)
        df_p_lc = df_p[df_p['LIGHT_CHAIN_AA_SEQUENCE'].notna()].copy()
        df_p_lc['seq_id'] = ['paired_lc_' + str(i) for i in df_p_lc['paired_index']]
        for _, row in df_p_lc.iterrows():
            f.write(f">{row['seq_id']}\n{row['LIGHT_CHAIN_AA_SEQUENCE']}\n")
        del df_p, df_p_lc
        gc.collect()

        for lf in tqdm(light_processed, desc="    Unpaired chunks"):
            df_u = pd.read_pickle(lf)
            for _, row in df_u.iterrows():
                if pd.notna(row['sequence_alignment_aa']):
                    f.write(f">{row['seq_id']}\n{row['sequence_alignment_aa']}\n")
            del df_u
            gc.collect()

    lc_whole_tsv = run_mmseqs_linclust(lc_whole_fasta, os.path.join(TEMP_DIR, "lc_whole"),
                                       WHOLE_SEQ_IDENTITY, 0.8, MMSEQS_THREADS)
    lc_whole_clusters = parse_cluster_tsv(lc_whole_tsv)

    # STEP 6: Add cluster IDs back to data
    print("\n" + "="*70)
    print("STEP 6: ADDING CLUSTER IDs TO DATA")
    print("="*70)

    # Add to paired data
    print("  Processing paired data...")
    df_paired = pd.read_pickle(paired_processed)
    df_paired['seq_id_hc'] = ['paired_hc_' + str(i) for i in df_paired['paired_index']]
    df_paired['seq_id_lc'] = ['paired_lc_' + str(i) for i in df_paired['paired_index']]
    df_paired['hc_cdr3_cluster_id'] = df_paired['seq_id_hc'].map(hc_cdr3_clusters)
    df_paired['hc_cluster_id'] = df_paired['seq_id_hc'].map(hc_whole_clusters)
    df_paired['lc_cdr3_cluster_id'] = df_paired['seq_id_lc'].map(lc_cdr3_clusters)
    df_paired['lc_cluster_id'] = df_paired['seq_id_lc'].map(lc_whole_clusters)

    output_paired = os.path.join(OUTPUT_DIR, "paired_with_clusters.pkl")
    df_paired.to_pickle(output_paired)
    print(f"  Saved: {output_paired}")
    del df_paired
    gc.collect()

    # Add to unpaired heavy
    print("  Processing unpaired heavy chunks...")
    for hf in tqdm(heavy_processed, desc="    Chunks"):
        df = pd.read_pickle(hf)
        df['cdr3_cluster_id'] = df['seq_id'].map(hc_cdr3_clusters)
        df['whole_seq_cluster_id'] = df['seq_id'].map(hc_whole_clusters)
        df.to_pickle(hf)
        del df
        gc.collect()

    # Add to unpaired light
    print("  Processing unpaired light chunks...")
    for lf in tqdm(light_processed, desc="    Chunks"):
        df = pd.read_pickle(lf)
        df['cdr3_cluster_id'] = df['seq_id'].map(lc_cdr3_clusters)
        df['whole_seq_cluster_id'] = df['seq_id'].map(lc_whole_clusters)
        df.to_pickle(lf)
        del df
        gc.collect()

    # STEP 7: Save final outputs
    print("\n" + "="*70)
    print("STEP 7: SAVING FINAL OUTPUTS")
    print("="*70)

    if args.keep_chunks:
        # Keep chunks as-is (most memory efficient)
        print("  Keeping processed chunks as final output (memory-efficient mode)")

        # Just copy chunks to output directory
        output_heavy_dir = os.path.join(OUTPUT_DIR, "unpaired_heavy_chunks")
        os.makedirs(output_heavy_dir, exist_ok=True)
        for hf in heavy_processed:
            subprocess.run(['cp', hf, output_heavy_dir], check=True)
        print(f"  ✓ Heavy chunks: {output_heavy_dir} ({len(heavy_processed)} files)")

        output_light_dir = os.path.join(OUTPUT_DIR, "unpaired_light_chunks")
        os.makedirs(output_light_dir, exist_ok=True)
        for lf in light_processed:
            subprocess.run(['cp', lf, output_light_dir], check=True)
        print(f"  ✓ Light chunks: {output_light_dir} ({len(light_processed)} files)")

        output_heavy = output_heavy_dir
        output_light = output_light_dir
    else:
        # Combine into single parquet files
        print("  Combining unpaired heavy chunks into single parquet...")
        dfs = []
        for hf in tqdm(heavy_processed, desc="    Loading"):
            dfs.append(pd.read_pickle(hf))
        df_heavy_final = pd.concat(dfs, ignore_index=True)
        del dfs
        gc.collect()

        output_heavy = os.path.join(OUTPUT_DIR, "unpaired_heavy_with_clusters.parquet")
        df_heavy_final.to_parquet(output_heavy, index=False)
        print(f"  ✓ Saved: {output_heavy} ({len(df_heavy_final):,} sequences)")
        del df_heavy_final
        gc.collect()

        print("\n  Combining unpaired light chunks into single parquet...")
        dfs = []
        for lf in tqdm(light_processed, desc="    Loading"):
            dfs.append(pd.read_pickle(lf))
        df_light_final = pd.concat(dfs, ignore_index=True)
        del dfs
        gc.collect()

        output_light = os.path.join(OUTPUT_DIR, "unpaired_light_with_clusters.parquet")
        df_light_final.to_parquet(output_light, index=False)
        print(f"  ✓ Saved: {output_light} ({len(df_light_final):,} sequences)")
        del df_light_final
        gc.collect()

    # Cleanup
    print(f"\n  Cleaning up {TEMP_DIR}...")
    subprocess.run(['rm', '-rf', TEMP_DIR], check=True, capture_output=True)

    print("\n" + "="*70)
    print("✓ CLUSTERING COMPLETE!")
    print("="*70)
    print(f"  Paired: {output_paired}")
    print(f"  Heavy:  {output_heavy}")
    print(f"  Light:  {output_light}")


if __name__ == "__main__":
    main()
