#!/usr/bin/env python
# coding: utf-8

"""
Annotate Antibody Sequences with Genes & Region Masks (Fix: Ungap Sequences)
===========================================================================

Fix Applied:
- Removes '-' (gaps) from sequences before passing to ANARCI.
  Reason: Input is already aligned ('sequence_alignment_aa'), which confuses ANARCI's germline detection.
"""

import os
import sys
import gc
import re
import shutil
import pandas as pd
import subprocess
import argparse
import multiprocessing as mp
from collections import Counter
from tqdm.auto import tqdm
import glob

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

INPUT_ROOT = "./compressed_data"
OUTPUT_ROOT = "./annotated_data_final"
TEMP_DIR = "./temp_annotation_ungapped"

ANARCI_CHUNK_SIZE = 10000 

REGION_TO_ID = {
    'fr1': '0', 'cdr1': '1', 'fr2': '2', 'cdr2': '3', 
    'fr3': '4', 'cdr3': '5', 'fr4': '6'
}

REGIONS_RANGES = {
    'fr1': (1, 26), 'cdr1': (27, 38), 'fr2': (39, 55),
    'cdr2': (56, 65), 'fr3': (66, 104), 'cdr3': (105, 117), 'fr4': (118, 128)
}

# ═══════════════════════════════════════════════════════════════════
# PARSING LOGIC
# ═══════════════════════════════════════════════════════════════════

def clean_gene_name(gene_str):
    if pd.isna(gene_str) or gene_str in ["", "-"]: return None
    return gene_str.split('*')[0]

def get_region_id(imgt_pos_str):
    match = re.match(r"(\d+)", imgt_pos_str)
    if not match: return None
    pos = int(match.group(1))
    for region, (start, end) in REGIONS_RANGES.items():
        if start <= pos <= end: return REGION_TO_ID[region]
    return None

def parse_anarci_csv_mask(csv_file):
    results = {}
    try:
        if os.path.getsize(csv_file) == 0: return results
        with open(csv_file, 'r') as f: lines = f.readlines()
        if not lines: return results

        header_map = {}
        data_start_idx = 0
        for i, line in enumerate(lines):
            if line.startswith('Id,'):
                parts = line.strip().split(',')
                for col_idx, part in enumerate(parts):
                    header_map[col_idx] = part.strip()
                data_start_idx = i + 1
                break
        if not header_map: return results

        pos_cols = []
        meta_indices = {}
        for idx, name in header_map.items():
            if name in ['Id', 'v_gene', 'j_gene']: meta_indices[name] = idx
            elif re.match(r"^\d", name):
                rid = get_region_id(name)
                if rid: pos_cols.append((idx, rid))
        pos_cols.sort(key=lambda x: x[0])

        for line in lines[data_start_idx:]:
            if not line.strip() or line.startswith('#'): continue
            parts = line.strip().split(',')
            if len(parts) <= max(header_map.keys()): continue

            seq_id = parts[meta_indices['Id']]
            v_gene = clean_gene_name(parts[meta_indices.get('v_gene', -1)])
            j_gene = clean_gene_name(parts[meta_indices.get('j_gene', -1)])

            mask_chars = []
            for col_idx, rid in pos_cols:
                if col_idx < len(parts):
                    aa = parts[col_idx].strip()
                    if aa and aa not in ['-', '.', '*']:
                        mask_chars.append(rid)
            mask_string = "".join(mask_chars)
            
            results[seq_id] = {'v_gene': v_gene, 'j_gene': j_gene, 'region_mask': mask_string}
    except: pass
    return results

# ═══════════════════════════════════════════════════════════════════
# EXECUTION LOGIC
# ═══════════════════════════════════════════════════════════════════

def split_fasta(input_fasta, seqs_per_chunk):
    chunk_dir = os.path.join(TEMP_DIR, "chunks")
    os.makedirs(chunk_dir, exist_ok=True)
    base_name = os.path.basename(input_fasta).replace('.fasta', '')
    
    chunks, chunk_idx, seq_cnt = [], 0, 0
    curr_path = os.path.join(chunk_dir, f"{base_name}_{chunk_idx}.fasta")
    f_out = open(curr_path, 'w')
    chunks.append(curr_path)
    
    with open(input_fasta, 'r') as f_in:
        for line in f_in:
            if line.startswith('>'):
                seq_cnt += 1
                if seq_cnt > seqs_per_chunk:
                    f_out.close()
                    chunk_idx += 1
                    curr_path = os.path.join(chunk_dir, f"{base_name}_{chunk_idx}.fasta")
                    f_out = open(curr_path, 'w')
                    chunks.append(curr_path)
                    seq_cnt = 1
            f_out.write(line)
    f_out.close()
    return chunks

def process_chunk(chunk_file):
    out_base = chunk_file.replace('.fasta', '_anarci')
    try:
        subprocess.run(['ANARCI', '-i', chunk_file, '--scheme', 'imgt', '--csv', '--assign_germline', '--outfile', out_base],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=1800)
    except: return {}, []

    results = {}
    csv_files = []
    for suffix in ['_H.csv', '_KL.csv', '_LL.csv']:
        path = out_base + suffix
        if os.path.exists(path):
            results.update(parse_anarci_csv_mask(path))
            csv_files.append(path)
    return results, csv_files

def annotate_dataframe(df, seq_col, processes=32):
    # 최적화된 방식: 거대한 파일을 쓰지 않고, DF를 바로 잘라서 작은 파일들로 만듦
    
    # 1. Chunk Directory 준비
    pid = os.getpid()
    chunk_dir = os.path.join(TEMP_DIR, f"proc_{pid}_chunks")
    os.makedirs(chunk_dir, exist_ok=True)
    
    total_seqs = len(df)
    chunk_paths = []
    
    # 2. DataFrame을 Slicing해서 바로 작은 FASTA로 저장 (메모리/IO 효율 증대)
    # tqdm을 달아서 "파일 준비 과정"도 눈으로 확인 가능하게 함
    num_chunks = (total_seqs + ANARCI_CHUNK_SIZE - 1) // ANARCI_CHUNK_SIZE
    
    print(f"  > Preparing {num_chunks} chunks for ANARCI...")
    
    for i in range(0, total_seqs, ANARCI_CHUNK_SIZE):
        chunk_df = df.iloc[i : i + ANARCI_CHUNK_SIZE]
        chunk_filename = os.path.join(chunk_dir, f"chunk_{i//ANARCI_CHUNK_SIZE}.fasta")
        
        with open(chunk_filename, 'w') as f:
            # 벡터화된 문자열 처리 (Loop보다 빠름)
            # 1. 인덱스와 서열 추출
            # 2. Gap 제거 (-와 . 제거)
            # 3. FASTA 포맷팅
            
            # 여기서 loop를 돌리는 게 가장 안전함 (메모리 스파이크 방지)
            buffer = []
            for idx, seq in zip(chunk_df.index, chunk_df[seq_col]):
                if pd.notna(seq):
                    # 문자열 변환 및 gap 제거
                    clean_seq = str(seq).replace('-', '').replace('.', '')
                    if clean_seq:
                        buffer.append(f">{idx}\n{clean_seq}\n")
            
            f.write("".join(buffer))
        
        chunk_paths.append(chunk_filename)
            
    # 3. 병렬 실행
    print(f"  > Running ANARCI on {len(chunk_paths)} chunks with {processes} cores...")
    all_data = {}
    all_csv_files = []
    if chunk_paths:
        with mp.Pool(processes) as pool:
            # imap_unordered가 결과를 나오는대로 즉시 줘서 진행상황 보기에 더 좋음
            results = list(tqdm(pool.imap_unordered(process_chunk, chunk_paths),
                               total=len(chunk_paths), desc="  > ANARCI Progress", leave=False))

        for r, csv_files in results:
            all_data.update(r)
            all_csv_files.extend(csv_files)

    # 5. Map back
    print("  > Mapping results back to DataFrame...")
    v_genes, j_genes, masks = [], [], []
    # 리스트 컴프리헨션으로 속도 최적화
    # get 메서드보다 직접 접근이 빠를 수 있으나 안전을 위해 get 유지
    for idx in df.index:
        key = str(idx) # DataFrame 인덱스가 정수라면 문자열로 변환 필요 (ANARCI ID는 문자열)
        if key in all_data:
            entry = all_data[key]
            v_genes.append(entry['v_gene'])
            j_genes.append(entry['j_gene'])
            masks.append(entry['region_mask'])
        else:
            v_genes.append(None)
            j_genes.append(None)
            masks.append(None)

    # 6. Validation Check: If first 10 v_genes or j_genes are all None, raise error
    check_count = min(10, len(v_genes))
    if check_count > 0:
        first_v_genes = v_genes[:check_count]
        first_j_genes = j_genes[:check_count]

        v_all_none = all(v is None for v in first_v_genes)
        j_all_none = all(j is None for j in first_j_genes)

        if v_all_none or j_all_none:
            failed_genes = []
            if v_all_none:
                failed_genes.append("v_gene")
            if j_all_none:
                failed_genes.append("j_gene")
            raise RuntimeError(
                f"ANARCI FAILED: First {check_count} sequences have all None for {', '.join(failed_genes)}. "
                f"This indicates ANARCI is not properly extracting gene information. "
                f"Check if sequences are valid antibody sequences and ANARCI is correctly installed."
            )

        # Also print statistics for debugging
        v_success = sum(1 for v in v_genes if v is not None)
        j_success = sum(1 for j in j_genes if j is not None)
        print(f"  > Gene extraction stats: v_gene={v_success}/{len(v_genes)} ({100*v_success/len(v_genes):.1f}%), "
              f"j_gene={j_success}/{len(j_genes)} ({100*j_success/len(j_genes):.1f}%)")

    # 7. Cleanup CSV files and chunk directory after all validation passed
    for csv_file in all_csv_files:
        if os.path.exists(csv_file):
            os.unlink(csv_file)
    if os.path.exists(chunk_dir):
        shutil.rmtree(chunk_dir)

    return v_genes, j_genes, masks

# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--processes', type=int, default=32)
    args = parser.parse_args()

    if os.path.exists(TEMP_DIR): shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    stats_v = Counter()
    stats_j = Counter()

    print("="*70)
    print("ANTIBODY ANNOTATION PIPELINE (UNGAPPED INPUT)")
    print("="*70)

    # File Collection (Same as before)
    heavy_dir = os.path.join(INPUT_ROOT, "Heavy_chunks")
    heavy_files = sorted(glob.glob(os.path.join(heavy_dir, "*.pkl")))
    
    light_dir = os.path.join(INPUT_ROOT, "Light_chunks")
    light_files = sorted(glob.glob(os.path.join(light_dir, "*.pkl")))
    
    root_files = [f for f in glob.glob(os.path.join(INPUT_ROOT, "*.pkl")) if os.path.isfile(f)]

    all_tasks = []
    for f in heavy_files: all_tasks.append((f, 'heavy', 'Heavy_chunks'))
    for f in light_files: all_tasks.append((f, 'light', 'Light_chunks'))
    for f in root_files: all_tasks.append((f, 'paired', ''))

    print(f"Tasks: {len(heavy_files)} Heavy, {len(light_files)} Light, {len(root_files)} Paired.")

    for pkl_path, ftype, subdir in tqdm(all_tasks, desc="Processing"):
        try:
            fname = os.path.basename(pkl_path)
            
            if subdir:
                out_path = os.path.join(OUTPUT_ROOT, subdir, fname)
            else:
                out_path = os.path.join(OUTPUT_ROOT, fname)
            
            if os.path.exists(out_path):
                continue
                
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            df = pd.read_pickle(pkl_path)
            
            if ftype == 'heavy':
                if 'sequence_alignment_aa' in df.columns:
                    v, j, m = annotate_dataframe(df, 'sequence_alignment_aa', args.processes)
                    df['v_gene'], df['j_gene'], df['region_mask'] = v, j, m
                    stats_v.update([x for x in v if x])
                    stats_j.update([x for x in j if x])

            elif ftype == 'light':
                if 'sequence_alignment_aa' in df.columns:
                    v, j, m = annotate_dataframe(df, 'sequence_alignment_aa', args.processes)
                    df['v_gene'], df['j_gene'], df['region_mask'] = v, j, m
                    stats_v.update([x for x in v if x])
                    stats_j.update([x for x in j if x])

            elif ftype == 'paired':
                if 'HEAVY_CHAIN_AA_SEQUENCE' in df.columns:
                    v, j, m = annotate_dataframe(df, 'HEAVY_CHAIN_AA_SEQUENCE', args.processes)
                    df['v_gene_heavy'], df['j_gene_heavy'], df['region_mask_heavy'] = v, j, m
                    stats_v.update([x for x in v if x])
                    stats_j.update([x for x in j if x])
                
                if 'LIGHT_CHAIN_AA_SEQUENCE' in df.columns:
                    v, j, m = annotate_dataframe(df, 'LIGHT_CHAIN_AA_SEQUENCE', args.processes)
                    df['v_gene_light'], df['j_gene_light'], df['region_mask_light'] = v, j, m
                    stats_v.update([x for x in v if x])
                    stats_j.update([x for x in j if x])

            df.to_pickle(out_path)
            del df
            gc.collect()

        except Exception as e:
            print(f"[ERROR] Failed {pkl_path}: {e}")

    print("\n" + "="*60)
    print(f"Total Unique V-Genes: {len(stats_v)}")
    print(f"Total Unique J-Genes: {len(stats_j)}")
    print("Top 10 V-Genes:", stats_v.most_common(10))
    print("="*60)
    
    if os.path.exists(TEMP_DIR): shutil.rmtree(TEMP_DIR)

if __name__ == "__main__":
    main()