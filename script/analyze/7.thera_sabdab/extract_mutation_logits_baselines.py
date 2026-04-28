#!/usr/bin/env python
# coding: utf-8
"""
Extract amino acid logits at mutation positions using baseline pLM models.

This script extracts the full 20-AA logit distribution at each mutation position
identified in therasabdab_germline.csv using masked prediction.

Models supported:
- ESM-2 35M (facebook/esm2_t12_35M_UR50D)
- ESM-2 650M (facebook/esm2_t33_650M_UR50D)
- AbLang2 (paired mode)
- AntiBERTy
- Sapiens (heavy/light specific)

Output format (CSV):
    Therapeutic, chain, position, germline_aa, mutated_aa, model, A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y

Usage:
    conda run -n devant python extract_mutation_logits_baselines.py \
        --data_path ../../data/therasabdab_germline.csv \
        --output_path ../../data/therasabdab_baseline_logits.csv \
        --models esm2_35m esm2_650m
"""

import argparse
import re
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

# Standard amino acids
AMINO_ACIDS = list('ACDEFGHIKLMNPQRSTVWY')


def parse_mutations(mutation_str: str) -> List[Tuple[int, str, str]]:
    """
    Parse mutation string into list of (position, germline_aa, mutated_aa).

    Args:
        mutation_str: Comma-separated mutations like "Q3K,S36N,E51D"

    Returns:
        List of (position, germline_aa, mutated_aa) tuples
    """
    if pd.isna(mutation_str) or not mutation_str or mutation_str == '':
        return []

    mutations = []
    for mut in mutation_str.split(','):
        mut = mut.strip()
        if not mut:
            continue

        # Parse format: {germline_aa}{position}{mutated_aa}
        # Handle insertion codes like "111A" -> position "111A"
        match = re.match(r'^([A-Z])(\d+[A-Za-z]?)([A-Z])$', mut)
        if match:
            germline_aa = match.group(1)
            position_str = match.group(2)
            mutated_aa = match.group(3)

            # Convert position to integer (ignore insertion code for now)
            try:
                position = int(re.match(r'(\d+)', position_str).group(1))
                mutations.append((position, germline_aa, mutated_aa))
            except (ValueError, AttributeError):
                continue

    return mutations


def get_aa_logits_esm2(
    model,
    tokenizer,
    sequence: str,
    positions: List[int],
    device: str,
    aa_to_idx: Dict[str, int]
) -> Dict[int, Dict[str, float]]:
    """
    Get logits for all 20 AAs at specified positions using ESM-2 masked prediction.

    Args:
        model: ESM-2 model
        tokenizer: ESM-2 tokenizer
        sequence: Full amino acid sequence
        positions: List of 0-indexed positions to extract logits for
        device: torch device
        aa_to_idx: Amino acid to token ID mapping

    Returns:
        Dict mapping position -> {AA: logit}
    """
    if not positions:
        return {}

    results = {}

    # Tokenize sequence
    inputs = tokenizer(
        sequence,
        return_tensors="pt",
        add_special_tokens=True,
        truncation=True,
        max_length=1024
    )
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    # Process each position
    for pos in positions:
        token_pos = pos + 1  # +1 for [CLS] token

        if token_pos >= input_ids.shape[1] - 1:  # -1 for [EOS]
            continue

        # Create masked input
        masked_input = input_ids.clone()
        masked_input[0, token_pos] = tokenizer.mask_token_id

        # Get logits
        with torch.no_grad():
            outputs = model(input_ids=masked_input, attention_mask=attention_mask)
            logits = outputs.logits[0, token_pos]  # [vocab_size]

        # Extract logits for each amino acid
        aa_logits = {}
        for aa in AMINO_ACIDS:
            idx = aa_to_idx.get(aa)
            if idx is not None:
                aa_logits[aa] = logits[idx].item()
            else:
                aa_logits[aa] = float('nan')

        results[pos] = aa_logits

    return results


def evaluate_esm2_model(
    model_name: str,
    model_id: str,
    df: pd.DataFrame,
    device: str,
    batch_size: int = 32
) -> pd.DataFrame:
    """
    Evaluate ESM-2 model and extract logits at mutation positions.

    Args:
        model_name: Display name for the model
        model_id: HuggingFace model identifier
        df: DataFrame with sequence and mutation data
        device: torch device
        batch_size: Number of masked positions to process in parallel

    Returns:
        DataFrame with columns: Therapeutic, chain, position, germline_aa, mutated_aa, model, A-Y logits
    """
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    print(f"\n{'='*60}")
    print(f"Loading {model_name} ({model_id})...")
    print('='*60)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForMaskedLM.from_pretrained(model_id)
    model.to(device)
    model.eval()

    # Build AA vocabulary
    aa_to_idx = {aa: tokenizer.convert_tokens_to_ids(aa) for aa in AMINO_ACIDS}

    print(f"  Model loaded on {device}")
    print(f"  Batch size: {batch_size}")

    all_results = []

    for idx in tqdm(range(len(df)), desc=f"Processing with {model_name}"):
        row = df.iloc[idx]
        therapeutic = row['Therapeutic']

        # Process heavy chain mutations
        heavy_seq = row.get('HeavySequence', '')
        mutations_heavy = parse_mutations(row.get('mutations_heavy', ''))

        if heavy_seq and heavy_seq != 'na' and mutations_heavy:
            positions = [m[0] - 1 for m in mutations_heavy]  # Convert to 0-indexed
            valid_positions = [p for p in positions if 0 <= p < len(heavy_seq)]

            logits = get_aa_logits_esm2(model, tokenizer, heavy_seq, valid_positions, device, aa_to_idx)

            for (imgt_pos, germ_aa, mut_aa), pos in zip(mutations_heavy, positions):
                if pos in logits:
                    result = {
                        'Therapeutic': therapeutic,
                        'chain': 'heavy',
                        'position': imgt_pos,
                        'germline_aa': germ_aa,
                        'mutated_aa': mut_aa,
                        'model': model_name,
                    }
                    result.update(logits[pos])
                    all_results.append(result)

        # Process light chain mutations
        light_seq = row.get('LightSequence', '')
        mutations_light = parse_mutations(row.get('mutations_light', ''))

        if light_seq and light_seq != 'na' and mutations_light:
            positions = [m[0] - 1 for m in mutations_light]  # Convert to 0-indexed
            valid_positions = [p for p in positions if 0 <= p < len(light_seq)]

            logits = get_aa_logits_esm2(model, tokenizer, light_seq, valid_positions, device, aa_to_idx)

            for (imgt_pos, germ_aa, mut_aa), pos in zip(mutations_light, positions):
                if pos in logits:
                    result = {
                        'Therapeutic': therapeutic,
                        'chain': 'light',
                        'position': imgt_pos,
                        'germline_aa': germ_aa,
                        'mutated_aa': mut_aa,
                        'model': model_name,
                    }
                    result.update(logits[pos])
                    all_results.append(result)

    # Clean up
    del model
    if device == 'cuda':
        torch.cuda.empty_cache()

    print(f"  Extracted logits for {len(all_results)} mutation positions")
    return pd.DataFrame(all_results)


def evaluate_ablang2_model(df: pd.DataFrame, device: str, batch_size: int = 32) -> pd.DataFrame:
    """Evaluate AbLang2 paired model and extract logits at mutation positions."""
    try:
        import ablang2
    except ImportError:
        print("WARNING: ablang2 not installed, skipping...")
        return pd.DataFrame()

    print(f"\n{'='*60}")
    print("Loading AbLang2 (paired mode)...")
    print(f"  Batch size: {batch_size}")
    print('='*60)

    ablang = ablang2.pretrained(model_to_use="ablang2-paired", random_init=False, device=device)
    model = ablang.AbLang
    tokenizer = ablang.tokenizer
    model.eval()

    # Build AA vocabulary
    aa_to_idx = {}
    for aa in AMINO_ACIDS:
        try:
            token_id = tokenizer.aa_to_id.get(aa, None)
            if token_id is not None:
                aa_to_idx[aa] = token_id
        except:
            pass

    if not aa_to_idx:
        for i, aa in enumerate(AMINO_ACIDS):
            aa_to_idx[aa] = i + 4

    print(f"  Model loaded on {device}")

    all_results = []

    for idx in tqdm(range(len(df)), desc="Processing with AbLang2"):
        row = df.iloc[idx]
        therapeutic = row['Therapeutic']

        heavy_seq = row.get('HeavySequence', '')
        light_seq = row.get('LightSequence', '')

        if not heavy_seq or heavy_seq == 'na' or not light_seq or light_seq == 'na':
            continue

        paired_seq = f"{heavy_seq}|{light_seq}"
        heavy_len = len(heavy_seq)

        # Process heavy chain mutations
        mutations_heavy = parse_mutations(row.get('mutations_heavy', ''))
        for imgt_pos, germ_aa, mut_aa in mutations_heavy:
            pos = imgt_pos - 1  # 0-indexed
            if pos < 0 or pos >= len(heavy_seq):
                continue

            # Tokenize and mask
            tokenized = tokenizer([paired_seq], pad=True, w_extra_tkns=False, device=device)

            with torch.no_grad():
                outputs = model(tokenized)
                logits = outputs[0, pos]  # Position in heavy chain

            result = {
                'Therapeutic': therapeutic,
                'chain': 'heavy',
                'position': imgt_pos,
                'germline_aa': germ_aa,
                'mutated_aa': mut_aa,
                'model': 'AbLang2',
            }
            for aa in AMINO_ACIDS:
                aa_idx = aa_to_idx.get(aa)
                if aa_idx is not None and aa_idx < logits.shape[0]:
                    result[aa] = logits[aa_idx].item()
                else:
                    result[aa] = float('nan')
            all_results.append(result)

        # Process light chain mutations
        mutations_light = parse_mutations(row.get('mutations_light', ''))
        for imgt_pos, germ_aa, mut_aa in mutations_light:
            pos = imgt_pos - 1  # 0-indexed
            if pos < 0 or pos >= len(light_seq):
                continue

            # Position in paired sequence: heavy_len + 1 (separator) + pos
            token_pos = heavy_len + 1 + pos

            tokenized = tokenizer([paired_seq], pad=True, w_extra_tkns=False, device=device)

            with torch.no_grad():
                outputs = model(tokenized)
                if token_pos < outputs.shape[1]:
                    logits = outputs[0, token_pos]
                else:
                    continue

            result = {
                'Therapeutic': therapeutic,
                'chain': 'light',
                'position': imgt_pos,
                'germline_aa': germ_aa,
                'mutated_aa': mut_aa,
                'model': 'AbLang2',
            }
            for aa in AMINO_ACIDS:
                aa_idx = aa_to_idx.get(aa)
                if aa_idx is not None and aa_idx < logits.shape[0]:
                    result[aa] = logits[aa_idx].item()
                else:
                    result[aa] = float('nan')
            all_results.append(result)

    print(f"  Extracted logits for {len(all_results)} mutation positions")
    return pd.DataFrame(all_results)


def evaluate_antiberty_model(df: pd.DataFrame, device: str, batch_size: int = 32) -> pd.DataFrame:
    """Evaluate AntiBERTy model and extract logits at mutation positions."""
    try:
        from antiberty import AntiBERTyRunner
    except ImportError:
        print("WARNING: antiberty not installed, skipping...")
        return pd.DataFrame()

    print(f"\n{'='*60}")
    print("Loading AntiBERTy...")
    print(f"  Batch size: {batch_size}")
    print('='*60)

    runner = AntiBERTyRunner()
    model = runner.model.to(device)
    tokenizer = runner.tokenizer
    model.eval()

    aa_to_idx = {aa: tokenizer.convert_tokens_to_ids(aa) for aa in AMINO_ACIDS}

    print(f"  Model loaded on {device}")

    all_results = []

    for idx in tqdm(range(len(df)), desc="Processing with AntiBERTy"):
        row = df.iloc[idx]
        therapeutic = row['Therapeutic']

        # Process heavy chain
        heavy_seq = row.get('HeavySequence', '')
        mutations_heavy = parse_mutations(row.get('mutations_heavy', ''))

        if heavy_seq and heavy_seq != 'na' and mutations_heavy:
            spaced_seq = ' '.join(list(heavy_seq))

            for imgt_pos, germ_aa, mut_aa in mutations_heavy:
                pos = imgt_pos - 1
                if pos < 0 or pos >= len(heavy_seq):
                    continue

                tokens = tokenizer(
                    spaced_seq,
                    return_tensors="pt",
                    add_special_tokens=True,
                    truncation=True,
                    max_length=1024
                )
                input_ids = tokens['input_ids'].to(device)
                attention_mask = tokens['attention_mask'].to(device)

                # Mask position
                token_pos = pos + 1  # +1 for CLS
                masked_input = input_ids.clone()
                masked_input[0, token_pos] = tokenizer.mask_token_id

                with torch.no_grad():
                    outputs = model(input_ids=masked_input, attention_mask=attention_mask)
                    if hasattr(outputs, 'logits'):
                        logits = outputs.logits[0, token_pos]
                    elif hasattr(outputs, 'prediction_logits'):
                        logits = outputs.prediction_logits[0, token_pos]
                    else:
                        continue

                result = {
                    'Therapeutic': therapeutic,
                    'chain': 'heavy',
                    'position': imgt_pos,
                    'germline_aa': germ_aa,
                    'mutated_aa': mut_aa,
                    'model': 'AntiBERTy',
                }
                for aa in AMINO_ACIDS:
                    aa_idx = aa_to_idx.get(aa)
                    if aa_idx is not None:
                        result[aa] = logits[aa_idx].item()
                    else:
                        result[aa] = float('nan')
                all_results.append(result)

        # Process light chain
        light_seq = row.get('LightSequence', '')
        mutations_light = parse_mutations(row.get('mutations_light', ''))

        if light_seq and light_seq != 'na' and mutations_light:
            spaced_seq = ' '.join(list(light_seq))

            for imgt_pos, germ_aa, mut_aa in mutations_light:
                pos = imgt_pos - 1
                if pos < 0 or pos >= len(light_seq):
                    continue

                tokens = tokenizer(
                    spaced_seq,
                    return_tensors="pt",
                    add_special_tokens=True,
                    truncation=True,
                    max_length=1024
                )
                input_ids = tokens['input_ids'].to(device)
                attention_mask = tokens['attention_mask'].to(device)

                token_pos = pos + 1
                masked_input = input_ids.clone()
                masked_input[0, token_pos] = tokenizer.mask_token_id

                with torch.no_grad():
                    outputs = model(input_ids=masked_input, attention_mask=attention_mask)
                    if hasattr(outputs, 'logits'):
                        logits = outputs.logits[0, token_pos]
                    elif hasattr(outputs, 'prediction_logits'):
                        logits = outputs.prediction_logits[0, token_pos]
                    else:
                        continue

                result = {
                    'Therapeutic': therapeutic,
                    'chain': 'light',
                    'position': imgt_pos,
                    'germline_aa': germ_aa,
                    'mutated_aa': mut_aa,
                    'model': 'AntiBERTy',
                }
                for aa in AMINO_ACIDS:
                    aa_idx = aa_to_idx.get(aa)
                    if aa_idx is not None:
                        result[aa] = logits[aa_idx].item()
                    else:
                        result[aa] = float('nan')
                all_results.append(result)

    del model
    if device == 'cuda':
        torch.cuda.empty_cache()

    print(f"  Extracted logits for {len(all_results)} mutation positions")
    return pd.DataFrame(all_results)


def evaluate_sapiens_model(df: pd.DataFrame, device: str, batch_size: int = 32) -> pd.DataFrame:
    """Evaluate Sapiens model and extract logits at mutation positions."""
    from transformers import RobertaForMaskedLM, RobertaTokenizer

    # Force CPU for stability
    device = 'cpu'
    max_seq_len = 143

    print(f"\n{'='*60}")
    print("Loading Sapiens (heavy + light models)...")
    print(f"  Batch size: {batch_size}")
    print('='*60)

    tokenizer = RobertaTokenizer.from_pretrained("prihodad/biophi-sapiens1-tokenizer")
    heavy_model = RobertaForMaskedLM.from_pretrained("prihodad/biophi-sapiens1-vh")
    light_model = RobertaForMaskedLM.from_pretrained("prihodad/biophi-sapiens1-vl")

    heavy_model.to(device)
    light_model.to(device)
    heavy_model.eval()
    light_model.eval()

    aa_to_idx = {aa: tokenizer.convert_tokens_to_ids(aa) for aa in AMINO_ACIDS}

    print(f"  Models loaded on {device} (forced CPU for stability)")

    all_results = []

    for idx in tqdm(range(len(df)), desc="Processing with Sapiens"):
        row = df.iloc[idx]
        therapeutic = row['Therapeutic']

        # Process heavy chain
        heavy_seq = row.get('HeavySequence', '')
        mutations_heavy = parse_mutations(row.get('mutations_heavy', ''))

        if heavy_seq and heavy_seq != 'na' and len(heavy_seq) <= max_seq_len and mutations_heavy:
            for imgt_pos, germ_aa, mut_aa in mutations_heavy:
                pos = imgt_pos - 1
                if pos < 0 or pos >= len(heavy_seq):
                    continue

                tokens = tokenizer(
                    heavy_seq,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=max_seq_len + 2
                )
                input_ids = tokens['input_ids'].to(device)
                attention_mask = tokens['attention_mask'].to(device)

                token_pos = pos + 1
                masked_input = input_ids.clone()
                masked_input[0, token_pos] = tokenizer.mask_token_id

                with torch.no_grad():
                    outputs = heavy_model(input_ids=masked_input, attention_mask=attention_mask)
                    logits = outputs.logits[0, token_pos]

                result = {
                    'Therapeutic': therapeutic,
                    'chain': 'heavy',
                    'position': imgt_pos,
                    'germline_aa': germ_aa,
                    'mutated_aa': mut_aa,
                    'model': 'Sapiens',
                }
                for aa in AMINO_ACIDS:
                    aa_idx = aa_to_idx.get(aa)
                    if aa_idx is not None:
                        result[aa] = logits[aa_idx].item()
                    else:
                        result[aa] = float('nan')
                all_results.append(result)

        # Process light chain
        light_seq = row.get('LightSequence', '')
        mutations_light = parse_mutations(row.get('mutations_light', ''))

        if light_seq and light_seq != 'na' and len(light_seq) <= max_seq_len and mutations_light:
            for imgt_pos, germ_aa, mut_aa in mutations_light:
                pos = imgt_pos - 1
                if pos < 0 or pos >= len(light_seq):
                    continue

                tokens = tokenizer(
                    light_seq,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=max_seq_len + 2
                )
                input_ids = tokens['input_ids'].to(device)
                attention_mask = tokens['attention_mask'].to(device)

                token_pos = pos + 1
                masked_input = input_ids.clone()
                masked_input[0, token_pos] = tokenizer.mask_token_id

                with torch.no_grad():
                    outputs = light_model(input_ids=masked_input, attention_mask=attention_mask)
                    logits = outputs.logits[0, token_pos]

                result = {
                    'Therapeutic': therapeutic,
                    'chain': 'light',
                    'position': imgt_pos,
                    'germline_aa': germ_aa,
                    'mutated_aa': mut_aa,
                    'model': 'Sapiens',
                }
                for aa in AMINO_ACIDS:
                    aa_idx = aa_to_idx.get(aa)
                    if aa_idx is not None:
                        result[aa] = logits[aa_idx].item()
                    else:
                        result[aa] = float('nan')
                all_results.append(result)

    del heavy_model, light_model

    print(f"  Extracted logits for {len(all_results)} mutation positions")
    return pd.DataFrame(all_results)


def main():
    parser = argparse.ArgumentParser(
        description='Extract amino acid logits at mutation positions using baseline pLMs'
    )
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to therasabdab_germline.csv')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Output path for logits CSV')
    parser.add_argument('--models', type=str, nargs='+',
                        default=['esm2_35m', 'esm2_650m'],
                        choices=['esm2_35m', 'esm2_650m', 'ablang2', 'antiberty', 'sapiens'],
                        help='Models to run')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for processing multiple positions (default: 32)')

    args = parser.parse_args()

    print("=" * 80)
    print("Extract Mutation Position Logits - Baseline Models")
    print("=" * 80)

    # Load data
    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"ERROR: Input file not found: {data_path}")
        sys.exit(1)

    print(f"\nLoading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"  Loaded {len(df)} rows")

    # Count total mutations
    total_mut_heavy = df['mutations_heavy'].dropna().apply(lambda x: len(x.split(',')) if x else 0).sum()
    total_mut_light = df['mutations_light'].dropna().apply(lambda x: len(x.split(',')) if x else 0).sum()
    print(f"  Total mutations: {total_mut_heavy} heavy + {total_mut_light} light = {total_mut_heavy + total_mut_light}")

    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")

    model_configs = {
        'esm2_35m': ('ESM2_35M', 'facebook/esm2_t12_35M_UR50D'),
        'esm2_650m': ('ESM2_650M', 'facebook/esm2_t33_650M_UR50D'),
    }

    all_results = []

    for model_key in args.models:
        try:
            if model_key in model_configs:
                model_name, model_id = model_configs[model_key]
                result_df = evaluate_esm2_model(model_name, model_id, df, device, args.batch_size)
            elif model_key == 'ablang2':
                result_df = evaluate_ablang2_model(df, device, args.batch_size)
            elif model_key == 'antiberty':
                result_df = evaluate_antiberty_model(df, device, args.batch_size)
            elif model_key == 'sapiens':
                result_df = evaluate_sapiens_model(df, device, args.batch_size)
            else:
                print(f"WARNING: Unknown model {model_key}, skipping...")
                continue

            if len(result_df) > 0:
                all_results.append(result_df)

        except Exception as e:
            print(f"ERROR: Failed to evaluate {model_key}: {e}")
            import traceback
            traceback.print_exc()

    # Combine all results
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)

        # Reorder columns
        cols = ['Therapeutic', 'chain', 'position', 'germline_aa', 'mutated_aa', 'model'] + AMINO_ACIDS
        final_df = final_df[cols]

        # Save
        if args.output_path:
            output_path = Path(args.output_path)
        else:
            output_path = data_path.parent / f"{data_path.stem}_baseline_logits.csv"

        output_path.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_csv(output_path, index=False)

        print(f"\n{'='*80}")
        print(f"Results saved to: {output_path}")
        print(f"Total rows: {len(final_df)}")
        print(f"Models: {final_df['model'].unique().tolist()}")
        print("=" * 80)
    else:
        print("\nNo results to save.")


if __name__ == "__main__":
    main()
