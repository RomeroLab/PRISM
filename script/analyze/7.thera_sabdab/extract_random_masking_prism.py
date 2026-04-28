#!/usr/bin/env python
# coding: utf-8
"""
Experiment 1: Automatic Recovery (Stratified Random Masking) for PRISM.

This experiment tests general masked prediction ability with stratified sampling:
- 5 random positions from FR regions
- 5 random positions from CDR regions

Key analyses:
1. GL position accuracy: How well does the model predict at germline (non-mutated) positions?
2. NGL position accuracy: How well does the model predict at non-germline (mutated) positions?
3. Forced-GL mode: When constrained to germline domain, accuracy at GL positions
4. Forced-NGL mode: When constrained to NGL domain, accuracy at NGL positions

Usage:
    python extract_random_masking_prism.py \
        --data_path data/therasabdab_germline.csv \
        --config configs/config_esm2_v34.1b_paired_cdr_ngl_focus.yaml \
        --checkpoint outputs/.../best.ckpt \
        --gene_vocab_json data/unpaired_OAS/annotated_data_final/gene_vocabulary.json \
        --output_path data/controllable_generation/exp1_random_masking_prism.csv \
        --batch_size 128
"""

import argparse
import json
import random
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Standard amino acids
AMINO_ACIDS = list('ACDEFGHIKLMNPQRSTVWY')

# Import prism modules
try:
    import prism
    from prism import SFT_ESM2
    from prism.multimodal_io import GeneVocabulary
except ImportError:
    print("[Error] Could not import prism. Please ensure the package is installed: pip install -e .")
    sys.exit(1)

# IMGT region mapping
FR_REGION_IDS = {'0', '2', '4', '6'}   # FR1, FR2, FR3, FR4
CDR_REGION_IDS = {'1', '3', '5'}       # CDR1, CDR2, CDR3


def load_model(checkpoint_path: str, gene_vocab_path: str, device: str):
    """Load PRISM model from checkpoint using prism.pretrained() API."""
    print(f"  Loading checkpoint: {checkpoint_path}")

    prism_model = prism.pretrained(
        checkpoint_path,
        device=str(device),
        gene_vocab_path=gene_vocab_path,
    )

    return prism_model.model, prism_model.gene_vocab


def get_token_ids(tokenizer) -> Tuple[List[int], List[int]]:
    """Get token IDs for uppercase and lowercase amino acids."""
    uppercase_ids = []
    lowercase_ids = []

    for aa in AMINO_ACIDS:
        upper_id = tokenizer.convert_tokens_to_ids(aa.upper())
        lower_id = tokenizer.convert_tokens_to_ids(aa.lower())

        if upper_id != tokenizer.unk_token_id:
            uppercase_ids.append(upper_id)
        if lower_id != tokenizer.unk_token_id:
            lowercase_ids.append(lower_id)

    return uppercase_ids, lowercase_ids


def apply_domain_constraint(
    logits: torch.Tensor,
    mode: str,
    uppercase_ids: List[int],
    lowercase_ids: List[int]
) -> torch.Tensor:
    """
    Apply domain constraint to logits using whitelist approach.

    Args:
        logits: Raw logits tensor [vocab_size]
        mode: 'natural', 'forced_gl', or 'forced_ngl'
        uppercase_ids: Token IDs for uppercase (GL) amino acids
        lowercase_ids: Token IDs for lowercase (NGL) amino acids

    Returns:
        Constrained logits tensor
    """
    constrained = logits.clone()

    if mode == 'forced_gl':
        # Force germline: only uppercase AA tokens should remain valid
        constrained[:] = float('-inf')
        for idx in uppercase_ids:
            constrained[idx] = logits[idx]
    elif mode == 'forced_ngl':
        # Force non-germline: only lowercase AA tokens should remain valid
        constrained[:] = float('-inf')
        for idx in lowercase_ids:
            constrained[idx] = logits[idx]
    # 'natural' mode: no constraint

    return constrained


def select_stratified_positions(
    region_mask: str,
    germline_seq: str,
    mature_seq: str,
    n_fr: int = 5,
    n_cdr: int = 5,
    seed: int = None
) -> Tuple[List[Dict], List[Dict]]:
    """
    Select stratified random positions from FR and CDR regions.

    Returns:
        Tuple of (gl_positions, ngl_positions) where each is a list of dicts with
        position info including whether it's a germline or mutation site.
    """
    if seed is not None:
        random.seed(seed)

    gl_positions = []  # Germline positions (sequence matches germline)
    ngl_positions = []  # Non-germline positions (mutations)

    for i, (r, g_aa, m_aa) in enumerate(zip(region_mask, germline_seq, mature_seq)):
        if g_aa == '-' or m_aa == '-':
            continue  # Skip gaps

        is_fr = r in FR_REGION_IDS
        is_cdr = r in CDR_REGION_IDS
        is_mutation = (g_aa.upper() != m_aa.upper())

        pos_info = {
            'position': i,
            'region_id': r,
            'region_type': 'FR' if is_fr else 'CDR' if is_cdr else 'unknown',
            'germline_aa': g_aa.upper(),
            'mature_aa': m_aa.upper(),
            'is_mutation_site': is_mutation
        }

        if is_mutation:
            ngl_positions.append(pos_info)
        else:
            gl_positions.append(pos_info)

    # Sample from GL positions (stratified by region)
    gl_fr = [p for p in gl_positions if p['region_type'] == 'FR']
    gl_cdr = [p for p in gl_positions if p['region_type'] == 'CDR']

    sampled_gl_fr = random.sample(gl_fr, min(n_fr, len(gl_fr))) if gl_fr else []
    sampled_gl_cdr = random.sample(gl_cdr, min(n_cdr, len(gl_cdr))) if gl_cdr else []

    # Sample from NGL positions (stratified by region)
    ngl_fr = [p for p in ngl_positions if p['region_type'] == 'FR']
    ngl_cdr = [p for p in ngl_positions if p['region_type'] == 'CDR']

    sampled_ngl_fr = random.sample(ngl_fr, min(n_fr, len(ngl_fr))) if ngl_fr else []
    sampled_ngl_cdr = random.sample(ngl_cdr, min(n_cdr, len(ngl_cdr))) if ngl_cdr else []

    return (sampled_gl_fr + sampled_gl_cdr, sampled_ngl_fr + sampled_ngl_cdr)


def prepare_masked_sequence(
    sequence: str,
    position: int,
    mask_token: str = '<mask>'
) -> str:
    """Prepare single chain sequence with masked position."""
    seq_list = list(sequence)
    seq_list[position] = mask_token
    return ''.join(seq_list)


def extract_predictions_for_position(
    model,
    tokenizer,
    sequence: str,
    position: int,
    v_gene: str,
    j_gene: str,
    region_mask: str,
    gene_vocab: Optional[GeneVocabulary],
    uppercase_ids: List[int],
    lowercase_ids: List[int],
    device: str
) -> Dict:
    """Extract predictions for a single masked position on a single chain."""

    # Prepare masked sequence
    masked_seq = prepare_masked_sequence(sequence.upper(), position)

    # Tokenize
    tokens = tokenizer(
        masked_seq,
        return_tensors='pt',
        add_special_tokens=True,
        truncation=True,
        max_length=1024
    )
    input_ids = tokens['input_ids'].to(device)
    attention_mask = tokens['attention_mask'].to(device)

    # Find mask position in tokens (add 1 for CLS token)
    token_pos = position + 1
    if token_pos >= input_ids.shape[1] - 1:
        return None

    # Apply mask
    masked_input = input_ids.clone()
    masked_input[0, token_pos] = tokenizer.mask_token_id

    # Prepare gene IDs (1D tensors with single value)
    v_gene_ids_tensor = None
    j_gene_ids_tensor = None
    if gene_vocab is not None:
        v_gene_ids_tensor = torch.tensor([gene_vocab.encode(v_gene)], dtype=torch.long, device=device)
        j_gene_ids_tensor = torch.tensor([gene_vocab.encode(j_gene)], dtype=torch.long, device=device)

    # Prepare region mask
    region_ids_tensor = None
    if region_mask:
        seq_len = input_ids.shape[1]
        region_ids = torch.zeros(1, seq_len, dtype=torch.long, device=device)
        for i, char in enumerate(region_mask):
            if i + 1 < seq_len - 1:
                region_ids[0, i + 1] = int(char) if char.isdigit() else 0
        region_ids_tensor = region_ids

    # Forward pass using internal multihead methods
    alpha_val = None
    with torch.no_grad():
        use_multihead = getattr(model, 'use_multihead_architecture', False)
        # [FIX] Check use_prism_architecture flag, not just method existence
        # v34.1b has use_multihead=True but use_prism_architecture=False
        # It should use _forward_multihead (with origin conditioning), not _forward_multihead_prism
        use_prism = getattr(model, 'use_prism_architecture', False)

        if use_multihead and use_prism and hasattr(model, '_forward_multihead_prism'):
            # PRISM v2 architecture (v38+): Separate GL and NGL heads
            forward_kwargs = {
                'input_ids': masked_input,
                'attention_mask': attention_mask,
            }
            if v_gene_ids_tensor is not None:
                forward_kwargs['v_gene_ids'] = v_gene_ids_tensor
                forward_kwargs['j_gene_ids'] = j_gene_ids_tensor
            if region_ids_tensor is not None:
                forward_kwargs['region_ids'] = region_ids_tensor

            logits_gl, logits_ngl, logits_mut, alpha, trust, logits_final, _ = \
                model._forward_multihead_prism(**forward_kwargs)

            if alpha is not None:
                alpha_val = alpha[0, token_pos, 0].item()
            if logits_final is not None:
                logits = logits_final[0, token_pos].cpu()
            else:
                logits = logits_ngl[0, token_pos].cpu() if logits_ngl is not None else torch.zeros(53)

        elif use_multihead and hasattr(model, '_forward_multihead'):
            # Standard multihead architecture (v34.1b): AA head with origin conditioning
            forward_kwargs = {
                'input_ids': masked_input,
                'attention_mask': attention_mask,
            }
            if v_gene_ids_tensor is not None:
                forward_kwargs['v_gene_ids'] = v_gene_ids_tensor
                forward_kwargs['j_gene_ids'] = j_gene_ids_tensor
            if region_ids_tensor is not None:
                forward_kwargs['region_ids'] = region_ids_tensor

            logits_aa, _, logits_mut, alpha, logits_final, _ = model._forward_multihead(**forward_kwargs)

            if alpha is not None:
                alpha_val = alpha[0, token_pos, 0].item()
            logits = logits_final[0, token_pos].cpu() if logits_final is not None else logits_aa[0, token_pos].cpu()

        else:
            # Fallback to standard forward
            outputs = model.esm2(input_ids=masked_input, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state
            lm_logits = model.esm2.lm_head(hidden_states)
            logits = lm_logits[0, token_pos].cpu()

    # Natural prediction (no constraint)
    pred_natural_id = torch.argmax(logits).item()
    pred_natural = tokenizer.convert_ids_to_tokens(pred_natural_id)

    # Forced-GL prediction
    gl_logits = apply_domain_constraint(logits, 'forced_gl', uppercase_ids, lowercase_ids)
    pred_gl_id = torch.argmax(gl_logits).item()
    pred_forced_gl = tokenizer.convert_ids_to_tokens(pred_gl_id)

    # Forced-NGL prediction
    ngl_logits = apply_domain_constraint(logits, 'forced_ngl', uppercase_ids, lowercase_ids)
    pred_ngl_id = torch.argmax(ngl_logits).item()
    pred_forced_ngl = tokenizer.convert_ids_to_tokens(pred_ngl_id)

    return {
        'pred_natural': pred_natural,
        'pred_forced_gl': pred_forced_gl,
        'pred_forced_ngl': pred_forced_ngl,
        'alpha': alpha_val
    }


def main():
    parser = argparse.ArgumentParser(description='Random masking experiment for PRISM')
    parser.add_argument('--data_path', type=str, default='data/therasabdab_germline.csv')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--gene_vocab_json', type=str, required=True)
    parser.add_argument('--output_path', type=str, default='data/controllable_generation/exp1_random_masking_prism.csv')
    parser.add_argument('--n_fr_positions', type=int, default=5)
    parser.add_argument('--n_cdr_positions', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 80)
    print("EXPERIMENT 1: RANDOM MASKING (PRISM)")
    print("=" * 80)

    # Load data
    print(f"\nLoading data from: {args.data_path}")
    df = pd.read_csv(args.data_path)
    print(f"  Loaded {len(df)} antibodies")

    # Load model with gene vocabulary
    print(f"\nLoading model from: {args.checkpoint}")
    model, gene_vocab = load_model(args.checkpoint, args.gene_vocab_json, args.device)
    tokenizer = model.tokenizer
    print("  Model loaded successfully")
    print(f"  Gene vocabulary size: {len(gene_vocab) if gene_vocab else 0}")

    # Get token IDs
    uppercase_ids, lowercase_ids = get_token_ids(tokenizer)
    print(f"  Uppercase token IDs: {len(uppercase_ids)}")
    print(f"  Lowercase token IDs: {len(lowercase_ids)}")

    # Process antibodies
    results = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing antibodies"):
        therapeutic = row.get('Therapeutic', f'antibody_{idx}')

        for chain in ['heavy', 'light']:
            # Get sequences
            mature_seq = row.get(f'{chain.capitalize()}Sequence', row.get(f'{chain}_sequence', ''))
            germline_seq = row.get(f'germline_{chain}', '')
            region_mask = row.get(f'region_mask_{chain}', '')

            # Check for NaN or empty values
            if pd.isna(mature_seq) or pd.isna(germline_seq) or pd.isna(region_mask):
                continue
            if not mature_seq or not germline_seq or not region_mask:
                continue

            # Ensure they are strings
            mature_seq = str(mature_seq)
            germline_seq = str(germline_seq)
            region_mask = str(region_mask)

            if len(mature_seq) != len(germline_seq) or len(mature_seq) != len(region_mask):
                continue

            # Get gene annotations
            v_gene_heavy = row.get('v_gene_heavy', '')
            j_gene_heavy = row.get('j_gene_heavy', '')
            v_gene_light = row.get('v_gene_light', '')
            j_gene_light = row.get('j_gene_light', '')

            heavy_seq = row.get('HeavySequence', row.get('heavy_sequence', ''))
            light_seq = row.get('LightSequence', row.get('light_sequence', ''))
            region_mask_heavy = row.get('region_mask_heavy', '')
            region_mask_light = row.get('region_mask_light', '')

            if not heavy_seq or not light_seq:
                continue

            # Select stratified positions
            gl_positions, ngl_positions = select_stratified_positions(
                region_mask, germline_seq, mature_seq,
                n_fr=args.n_fr_positions, n_cdr=args.n_cdr_positions,
                seed=args.seed + idx
            )

            # Get chain-specific gene IDs and region mask
            if chain == 'heavy':
                v_gene = row.get('v_gene_heavy', '')
                j_gene = row.get('j_gene_heavy', '')
                sequence = mature_seq
            else:
                v_gene = row.get('v_gene_light', '')
                j_gene = row.get('j_gene_light', '')
                sequence = mature_seq

            # Process GL positions
            for pos_info in gl_positions:
                preds = extract_predictions_for_position(
                    model, tokenizer,
                    sequence, pos_info['position'],
                    v_gene, j_gene, region_mask,
                    gene_vocab, uppercase_ids, lowercase_ids, args.device
                )

                if preds is None:
                    continue

                ground_truth = pos_info['mature_aa']  # At GL positions, mature == germline

                results.append({
                    'Therapeutic': therapeutic,
                    'chain': chain,
                    'position': pos_info['position'],
                    'position_type': 'GL',  # Germline position
                    'region_id': pos_info['region_id'],
                    'region_type': pos_info['region_type'],
                    'germline_aa': pos_info['germline_aa'],
                    'ground_truth_aa': ground_truth,
                    'is_mutation_site': False,
                    'pred_natural': preds['pred_natural'],
                    'pred_forced_gl': preds['pred_forced_gl'],
                    'pred_forced_ngl': preds['pred_forced_ngl'],
                    'alpha': preds['alpha'],
                    'is_correct_natural': preds['pred_natural'].upper() == ground_truth.upper(),
                    'is_correct_forced_gl': preds['pred_forced_gl'].upper() == ground_truth.upper(),
                    'is_correct_forced_ngl': preds['pred_forced_ngl'].upper() == ground_truth.upper(),
                })

            # Process NGL positions
            for pos_info in ngl_positions:
                preds = extract_predictions_for_position(
                    model, tokenizer,
                    sequence, pos_info['position'],
                    v_gene, j_gene, region_mask,
                    gene_vocab, uppercase_ids, lowercase_ids, args.device
                )

                if preds is None:
                    continue

                ground_truth = pos_info['mature_aa']  # At NGL positions, this is the mutation

                results.append({
                    'Therapeutic': therapeutic,
                    'chain': chain,
                    'position': pos_info['position'],
                    'position_type': 'NGL',  # Non-germline (mutation) position
                    'region_id': pos_info['region_id'],
                    'region_type': pos_info['region_type'],
                    'germline_aa': pos_info['germline_aa'],
                    'ground_truth_aa': ground_truth,
                    'is_mutation_site': True,
                    'pred_natural': preds['pred_natural'],
                    'pred_forced_gl': preds['pred_forced_gl'],
                    'pred_forced_ngl': preds['pred_forced_ngl'],
                    'alpha': preds['alpha'],
                    'is_correct_natural': preds['pred_natural'].upper() == ground_truth.upper(),
                    'is_correct_forced_gl': preds['pred_forced_gl'].upper() == pos_info['germline_aa'].upper(),  # For GL mode, compare to germline
                    'is_correct_forced_ngl': preds['pred_forced_ngl'].upper() == ground_truth.upper(),
                })

    # Save results
    results_df = pd.DataFrame(results)

    output_dir = Path(args.output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(args.output_path, index=False)
    print(f"\nResults saved to: {args.output_path}")
    print(f"  Total positions: {len(results_df)}")
    print(f"  GL positions: {len(results_df[results_df['position_type'] == 'GL'])}")
    print(f"  NGL positions: {len(results_df[results_df['position_type'] == 'NGL'])}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    for pos_type in ['GL', 'NGL']:
        subset = results_df[results_df['position_type'] == pos_type]
        if len(subset) == 0:
            continue

        print(f"\n{pos_type} Positions (n={len(subset)}):")
        print(f"  Natural accuracy: {subset['is_correct_natural'].mean()*100:.1f}%")

        if pos_type == 'GL':
            print(f"  Forced-GL accuracy: {subset['is_correct_forced_gl'].mean()*100:.1f}%")
        else:
            print(f"  Forced-NGL accuracy: {subset['is_correct_forced_ngl'].mean()*100:.1f}%")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
