#!/usr/bin/env python
# coding: utf-8

"""
Benchmark Developability Baselines - Perplexity Calculation for Antibody Sequences.

This script calculates per-sequence perplexity (PPL) using various protein language models
as a proxy for sequence naturalness/developability.

Method (Pseudo-Perplexity):
    For each sequence position i:
        1. Mask position i
        2. Get log P(true_token | context) from model
    PPL = exp(-mean(log_probs))

Lower PPL indicates the sequence is more "natural" according to the model.

Supported Models:
    - ESM-2 (35M, 150M, 650M, 3B)
    - AbLang2 (paired antibody model)
    - AntiBERTy (antibody-specific BERT)
    - Sapiens (humanness predictor)

Key Optimizations:
    - Batch processing across sequences (GPU)
    - Position-wise batching for memory efficiency
    - Multiprocessing for CPU-bound models

Usage:
    python benchmark_developability_baselines.py \
        --data_path data/ginkgo/developability_data.csv \
        --output_path data/ginkgo/developability_data_with_ppl.csv \
        --models esm2_35m esm2_650m ablang2 \
        --batch_size 32

Author: DevAnt-LM Team
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# Core Perplexity Calculation Functions
# =============================================================================

def calculate_batch_perplexity_esm2(
    model,
    tokenizer,
    sequences: List[str],
    device: str,
    batch_size: int = 32
) -> List[float]:
    """
    Calculate perplexity for a batch of sequences using ESM-2 model.

    Uses efficient position-wise batching: for each position, process all sequences
    that have an amino acid at that position.

    Args:
        model: ESM-2 model
        tokenizer: ESM-2 tokenizer
        sequences: List of sequences
        device: torch device
        batch_size: Number of sequences to process at once per position

    Returns:
        List of perplexity values (one per sequence)
    """
    model.eval()
    model = model.to(device)

    mask_token_id = tokenizer.mask_token_id
    all_special_ids = tokenizer.all_special_ids

    tokens = tokenizer(
        sequences,
        return_tensors="pt",
        add_special_tokens=True,
        padding=True,
        truncation=True,
        max_length=1024,
    )
    all_input_ids = tokens['input_ids'].to(device)
    all_attention_mask = tokens['attention_mask'].to(device)

    N, L = all_input_ids.shape
    log_probs_tensor = torch.zeros(N, L, dtype=torch.float32, device=device)

    # Standard 20 AAs occupy ESM-2 ids 4..23
    aa_mask = (all_input_ids >= 4) & (all_input_ids <= 23)

    for pos in range(L):
        if not aa_mask[:, pos].any():
            continue

        for batch_start in range(0, N, batch_size):
            batch_end = min(batch_start + batch_size, N)

            batch_input_ids = all_input_ids[batch_start:batch_end]
            batch_attention_mask = all_attention_mask[batch_start:batch_end]
            original_tokens = batch_input_ids[:, pos].clone()

            masked_input = batch_input_ids.clone()
            masked_input[:, pos] = mask_token_id

            with torch.no_grad():
                outputs = model(input_ids=masked_input, attention_mask=batch_attention_mask)
                pos_logits = outputs.logits[:, pos, :].clone()
                pos_logits[:, all_special_ids] = -float("inf")
                log_probs = F.log_softmax(pos_logits, dim=-1)

                log_prob_original = log_probs.gather(1, original_tokens.unsqueeze(1)).squeeze(1)
                current_aa_mask = aa_mask[batch_start:batch_end, pos]
                log_prob_original = log_prob_original * current_aa_mask.float()
                log_probs_tensor[batch_start:batch_end, pos] = log_prob_original

    # Calculate perplexity for each sequence
    perplexities = []
    for i in range(N):
        seq_aa_mask = aa_mask[i]
        seq_log_probs = log_probs_tensor[i][seq_aa_mask]

        if len(seq_log_probs) > 0:
            mean_log_prob = seq_log_probs.mean().item()
            ppl = np.exp(-mean_log_prob)
        else:
            ppl = float('inf')

        perplexities.append(ppl)

    return perplexities


def calculate_batch_perplexity_ablang2(
    ablang_model,
    heavy_sequences: List[str],
    light_sequences: List[str],
    device: str,
    batch_size: int = 32
) -> List[float]:
    """
    Pseudo-PPL for paired antibody sequences using AbLang2-paired.

    AbLang2 is trained on "<HEAVY>|<LIGHT>"; `<`, `>`, `|` are real vocab
    tokens. PPL is averaged over AA positions only (all special tokens
    excluded), matching the library's own `pseudo_log_likelihood`.
    Special token ids are suppressed in logits so the softmax norm stays
    over the AA vocabulary.
    """
    ablang_model.AbLang.eval()
    tokenizer = ablang_model.tokenizer

    mask_token = tokenizer.mask_token
    special_ids = list(tokenizer.all_special_tokens)

    N = len(heavy_sequences)
    perplexities = []

    for idx in tqdm(range(N), desc="AbLang2 PPL", leave=False):
        heavy = heavy_sequences[idx]
        light = light_sequences[idx]

        if not heavy or not light:
            perplexities.append(float('inf'))
            continue

        # Library-format string: wrapped with start/end tokens
        paired_seq = f"<{heavy}>|<{light}>"

        try:
            input_ids = tokenizer([paired_seq], pad=True, w_extra_tkns=False, device=device)

            # Score only non-special positions (exclude <, >, |, pad, mask, X)
            is_special = torch.isin(input_ids[0], torch.tensor(special_ids, device=input_ids.device))
            seq_indices = (~is_special).nonzero(as_tuple=True)[0].tolist()

            if not seq_indices:
                perplexities.append(float('inf'))
                continue

            masked_inputs = []
            true_tokens = []
            for i in seq_indices:
                masked = input_ids.clone()
                true_tokens.append(input_ids[0, i].item())
                masked[0, i] = mask_token
                masked_inputs.append(masked)

            masked_batch = torch.cat(masked_inputs, dim=0).to(device)

            log_probs_list = []
            with torch.no_grad():
                for i in range(0, masked_batch.size(0), batch_size):
                    batch_input = masked_batch[i:i+batch_size]
                    outputs = ablang_model.AbLang(batch_input).clone()
                    outputs[:, :, special_ids] = -float("inf")
                    log_softmax = F.log_softmax(outputs, dim=-1)

                    for j in range(batch_input.size(0)):
                        pos = seq_indices[i + j]
                        token_id = true_tokens[i + j]
                        log_prob = log_softmax[j, pos, token_id].item()
                        log_probs_list.append(log_prob)

            if log_probs_list:
                ppl = float(np.exp(-np.mean(log_probs_list)))
            else:
                ppl = float('inf')

        except Exception as e:
            if idx == 0:
                print(f"  Error: {e}")
            ppl = float('inf')

        perplexities.append(ppl)

    return perplexities


def calculate_batch_perplexity_antiberty(
    model,
    tokenizer,
    heavy_sequences: List[str],
    light_sequences: List[str],
    device: str,
    batch_size: int = 32,
) -> List[float]:
    """
    Pseudo-PPL using AntiBERTy.

    AntiBERTy is a single-chain model: heavy and light chains are tokenized
    and scored independently; the combined log-prob list is averaged to a
    single PPL per antibody. Special token ids are suppressed in logits.
    """
    model.eval()
    model = model.to(device)

    aa_to_idx = {aa: tokenizer.convert_tokens_to_ids(aa) for aa in 'ACDEFGHIKLMNPQRSTVWY'}
    mask_token_id = tokenizer.mask_token_id
    all_special_ids = tokenizer.all_special_ids

    def _chain_log_probs(chain_seq: str) -> List[float]:
        if not chain_seq:
            return []
        spaced = ' '.join(list(chain_seq))
        tokens = tokenizer(
            spaced, return_tensors="pt", add_special_tokens=True,
            truncation=True, max_length=1024,
        )
        input_ids = tokens['input_ids'].to(device)
        attention_mask = tokens['attention_mask'].to(device)

        logs = []
        for pos, aa in enumerate(chain_seq):
            if aa not in aa_to_idx:
                continue
            token_pos = pos + 1  # +1 for [CLS]
            masked = input_ids.clone()
            masked[0, token_pos] = mask_token_id

            with torch.no_grad():
                outputs = model(input_ids=masked, attention_mask=attention_mask)
                if hasattr(outputs, 'prediction_logits') and outputs.prediction_logits is not None:
                    logits = outputs.prediction_logits
                elif hasattr(outputs, 'logits') and outputs.logits is not None:
                    logits = outputs.logits
                else:
                    continue
                pos_logits = logits[0, token_pos, :].clone()
                pos_logits[all_special_ids] = -float("inf")
                log_probs = F.log_softmax(pos_logits, dim=-1)
                logs.append(log_probs[aa_to_idx[aa]].item())
        return logs

    perplexities = []
    for heavy, light in tqdm(zip(heavy_sequences, light_sequences),
                             total=len(heavy_sequences), desc="AntiBERTy PPL", leave=False):
        combined_logs = _chain_log_probs(heavy) + _chain_log_probs(light)
        if combined_logs:
            ppl = float(np.exp(-np.mean(combined_logs)))
        else:
            ppl = float('inf')
        perplexities.append(ppl)

    return perplexities


def calculate_batch_perplexity_sapiens(
    heavy_model,
    light_model,
    tokenizer,
    heavy_sequences: List[str],
    light_sequences: List[str],
    device: str = 'cpu',
) -> List[float]:
    """
    Pseudo-PPL using Sapiens (separate VH/VL models).

    VH and VL have different `max_position_embeddings` (typically 146 / 130).
    Usable residue length = max_position_embeddings - pad_token_id - 1 - 2
    (RoBERTa position IDs start at pad+1; 2 slots for <s> and </s>).
    Forced to CPU for stability. Special token ids are suppressed in logits.
    """
    heavy_model.eval()
    light_model.eval()

    def _chain_max_residues(cfg):
        return cfg.max_position_embeddings - cfg.pad_token_id - 1 - 2

    heavy_max_residues = _chain_max_residues(heavy_model.config)
    light_max_residues = _chain_max_residues(light_model.config)

    aa_to_idx = {aa: tokenizer.convert_tokens_to_ids(aa) for aa in 'ACDEFGHIKLMNPQRSTVWY'}
    mask_token_id = tokenizer.mask_token_id
    all_special_ids = tokenizer.all_special_ids

    def _chain_log_probs(chain_model, chain_seq: str, max_residues: int) -> List[float]:
        if not chain_seq or len(chain_seq) > max_residues:
            return []
        tokens = tokenizer(
            chain_seq, return_tensors="pt", padding=False,
            truncation=True, max_length=max_residues + 2,
        )
        input_ids = tokens['input_ids'].to(device)
        attention_mask = tokens['attention_mask'].to(device)

        logs = []
        for pos, aa in enumerate(chain_seq):
            if aa not in aa_to_idx:
                continue
            token_pos = pos + 1  # +1 for <s>
            masked = input_ids.clone()
            masked[0, token_pos] = mask_token_id
            with torch.no_grad():
                outputs = chain_model(input_ids=masked, attention_mask=attention_mask)
                pos_logits = outputs.logits[0, token_pos, :].clone()
                pos_logits[all_special_ids] = -float("inf")
                log_probs = F.log_softmax(pos_logits, dim=-1)
                logs.append(log_probs[aa_to_idx[aa]].item())
        return logs

    perplexities = []
    for heavy, light in tqdm(zip(heavy_sequences, light_sequences),
                             total=len(heavy_sequences), desc="Sapiens PPL", leave=False):
        combined = (
            _chain_log_probs(heavy_model, heavy, heavy_max_residues)
            + _chain_log_probs(light_model, light, light_max_residues)
        )
        if combined:
            ppl = float(np.exp(-np.mean(combined)))
        else:
            ppl = float('inf')
        perplexities.append(ppl)

    return perplexities


# =============================================================================
# Model Evaluation Functions
# =============================================================================

def evaluate_esm2_ppl(
    model_name: str,
    model_id: str,
    sequences: List[str],
    device: str,
    batch_size: int = 32
) -> List[float]:
    """
    Evaluate ESM-2 model perplexity on sequences.
    """
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    print(f"\n{'='*60}")
    print(f"Loading {model_name} ({model_id})...")
    print('='*60)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForMaskedLM.from_pretrained(model_id)
    model.to(device)
    model.eval()

    print(f"  Model loaded on {device}")
    print(f"  Vocab size: {len(tokenizer)}")
    print(f"  Processing {len(sequences)} sequences with batch_size={batch_size}")

    perplexities = calculate_batch_perplexity_esm2(
        model, tokenizer, sequences, device, batch_size
    )

    # Statistics
    valid_ppls = [p for p in perplexities if p != float('inf')]
    if valid_ppls:
        print(f"\n  PPL Statistics for {model_name}:")
        print(f"    Mean: {np.mean(valid_ppls):.4f}")
        print(f"    Median: {np.median(valid_ppls):.4f}")
        print(f"    Min: {np.min(valid_ppls):.4f}")
        print(f"    Max: {np.max(valid_ppls):.4f}")

    # Clean up
    del model
    if device == 'cuda':
        torch.cuda.empty_cache()

    return perplexities


def evaluate_ablang2_ppl(
    heavy_sequences: List[str],
    light_sequences: List[str],
    device: str,
    batch_size: int = 16
) -> List[float]:
    """
    Evaluate AbLang2 paired model perplexity.
    """
    try:
        import ablang2
    except ImportError:
        print("WARNING: ablang2 not installed, skipping...")
        return [float('inf')] * len(heavy_sequences)

    print(f"\n{'='*60}")
    print("Loading AbLang2 (paired mode)...")
    print('='*60)

    ablang = ablang2.pretrained(
        model_to_use="ablang2-paired",
        random_init=False,
        device=device,
    )

    print(f"  Model loaded on {device}")
    print(f"  Processing {len(heavy_sequences)} paired sequences")

    # Pass the full ablang object (contains AbLang model and tokenizer)
    perplexities = calculate_batch_perplexity_ablang2(
        ablang, heavy_sequences, light_sequences, device, batch_size
    )

    # Statistics
    valid_ppls = [p for p in perplexities if p != float('inf')]
    if valid_ppls:
        print(f"\n  PPL Statistics for AbLang2:")
        print(f"    Mean: {np.mean(valid_ppls):.4f}")
        print(f"    Median: {np.median(valid_ppls):.4f}")
        print(f"    Min: {np.min(valid_ppls):.4f}")
        print(f"    Max: {np.max(valid_ppls):.4f}")

    return perplexities


def evaluate_antiberty_ppl(
    heavy_sequences: List[str],
    light_sequences: List[str],
    device: str,
    batch_size: int = 32,
) -> List[float]:
    """AntiBERTy PPL (single-chain model, heavy/light scored independently)."""
    try:
        from antiberty import AntiBERTyRunner
    except ImportError:
        print("WARNING: antiberty not installed, skipping...")
        return [float('inf')] * len(heavy_sequences)

    print(f"\n{'='*60}")
    print("Loading AntiBERTy...")
    print('='*60)

    runner = AntiBERTyRunner()
    model = runner.model.to(device)
    tokenizer = runner.tokenizer
    model.eval()

    print(f"  Model loaded on {device}")
    print(f"  Processing {len(heavy_sequences)} paired sequences (heavy+light scored separately)")

    perplexities = calculate_batch_perplexity_antiberty(
        model, tokenizer, heavy_sequences, light_sequences, device, batch_size
    )

    # Statistics
    valid_ppls = [p for p in perplexities if p != float('inf')]
    if valid_ppls:
        print(f"\n  PPL Statistics for AntiBERTy:")
        print(f"    Mean: {np.mean(valid_ppls):.4f}")
        print(f"    Median: {np.median(valid_ppls):.4f}")
        print(f"    Min: {np.min(valid_ppls):.4f}")
        print(f"    Max: {np.max(valid_ppls):.4f}")

    del model
    if device == 'cuda':
        torch.cuda.empty_cache()

    return perplexities


def evaluate_sapiens_ppl(
    heavy_sequences: List[str],
    light_sequences: List[str],
    device: str = 'cpu'
) -> List[float]:
    """
    Evaluate Sapiens model perplexity (forced CPU).
    """
    from transformers import RobertaForMaskedLM, RobertaTokenizer

    # Force CPU for Sapiens stability
    device = 'cpu'

    print(f"\n{'='*60}")
    print("Loading Sapiens (heavy + light models)...")
    print('='*60)

    tokenizer = RobertaTokenizer.from_pretrained("prihodad/biophi-sapiens1-tokenizer")
    heavy_model = RobertaForMaskedLM.from_pretrained("prihodad/biophi-sapiens1-vh")
    light_model = RobertaForMaskedLM.from_pretrained("prihodad/biophi-sapiens1-vl")

    heavy_model.to(device)
    light_model.to(device)
    heavy_model.eval()
    light_model.eval()

    print(f"  Models loaded on {device} (forced CPU)")
    print(f"  Processing {len(heavy_sequences)} paired sequences")

    perplexities = calculate_batch_perplexity_sapiens(
        heavy_model, light_model, tokenizer,
        heavy_sequences, light_sequences, device
    )

    # Statistics
    valid_ppls = [p for p in perplexities if p != float('inf')]
    if valid_ppls:
        print(f"\n  PPL Statistics for Sapiens:")
        print(f"    Mean: {np.mean(valid_ppls):.4f}")
        print(f"    Median: {np.median(valid_ppls):.4f}")
        print(f"    Min: {np.min(valid_ppls):.4f}")
        print(f"    Max: {np.max(valid_ppls):.4f}")

    del heavy_model, light_model

    return perplexities


# =============================================================================
# Efficient Batch Processing for ESM2 (Optimized Version)
# =============================================================================

def calculate_ppl_esm2_optimized(
    model,
    tokenizer,
    sequences: List[str],
    device: str,
    batch_size: int = 64,
    positions_per_batch: int = 8
) -> List[float]:
    """
    Optimized perplexity calculation for ESM-2.

    Uses chunked position processing to balance memory and speed.
    Instead of processing one position at a time across all sequences,
    we process multiple positions for a chunk of sequences.

    Args:
        model: ESM-2 model
        tokenizer: ESM-2 tokenizer
        sequences: List of sequences
        device: torch device
        batch_size: Number of sequences per chunk
        positions_per_batch: Number of positions to process per forward pass

    Returns:
        List of perplexity values
    """
    model.eval()
    mask_token_id = tokenizer.mask_token_id
    all_special_ids = tokenizer.all_special_ids

    all_perplexities = []

    # Process sequences in chunks
    for chunk_start in tqdm(range(0, len(sequences), batch_size),
                            desc="Processing chunks", leave=False):
        chunk_end = min(chunk_start + batch_size, len(sequences))
        chunk_sequences = sequences[chunk_start:chunk_end]

        # Tokenize chunk
        tokens = tokenizer(
            chunk_sequences,
            return_tensors="pt",
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=512
        )
        input_ids = tokens['input_ids'].to(device)
        attention_mask = tokens['attention_mask'].to(device)

        B, L = input_ids.shape

        # Initialize log probs storage
        log_probs_tensor = torch.zeros(B, L, dtype=torch.float32, device=device)

        # AA mask (positions 4-23 are standard AAs)
        aa_mask = (input_ids >= 4) & (input_ids <= 23)

        # Process positions in groups
        for pos in range(L):
            if not aa_mask[:, pos].any():
                continue

            # Get original tokens
            original_tokens = input_ids[:, pos].clone()

            # Create masked input
            masked_input = input_ids.clone()
            masked_input[:, pos] = mask_token_id

            with torch.no_grad():
                outputs = model(
                    input_ids=masked_input,
                    attention_mask=attention_mask
                )
                pos_logits = outputs.logits[:, pos, :].clone()
                pos_logits[:, all_special_ids] = -float("inf")
                log_probs = F.log_softmax(pos_logits, dim=-1)

                log_prob_original = log_probs.gather(1, original_tokens.unsqueeze(1)).squeeze(1)
                log_prob_original = log_prob_original * aa_mask[:, pos].float()
                log_probs_tensor[:, pos] = log_prob_original

        # Calculate perplexity for each sequence in chunk
        for i in range(B):
            seq_aa_mask = aa_mask[i]
            seq_log_probs = log_probs_tensor[i][seq_aa_mask]

            if len(seq_log_probs) > 0:
                mean_log_prob = seq_log_probs.mean().item()
                ppl = np.exp(-mean_log_prob)
            else:
                ppl = float('inf')

            all_perplexities.append(ppl)

    return all_perplexities


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Calculate perplexity for developability assessment using baseline models'
    )
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to input CSV file with antibody sequences')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Output path for results CSV')
    parser.add_argument('--models', type=str, nargs='+',
                        default=['esm2_35m', 'esm2_650m','ablang2', 'antiberty', 'sapiens'],
                        choices=['esm2_35m', 'esm2_150m', 'esm2_650m', 'esm2_3b',
                                'ablang2', 'antiberty', 'sapiens'],
                        help='Models to evaluate')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for processing')
    parser.add_argument('--heavy_col', type=str, default='vh_protein_sequence',
                        help='Column name for heavy chain variable region')
    parser.add_argument('--light_col', type=str, default='vl_protein_sequence',
                        help='Column name for light chain variable region')
    parser.add_argument('--concatenate', action='store_true',
                        help='Concatenate heavy+light chains for single-sequence models')

    args = parser.parse_args()

    print("=" * 80)
    print("Developability Baseline Benchmark - Perplexity Calculation")
    print("=" * 80)
    print(f"\nMethod: Pseudo-perplexity (masked language model)")
    print(f"  PPL = exp(-mean(log P(token_i | context)))")
    print(f"  Lower PPL = more 'natural' sequence")
    print("=" * 80)

    # Check input file
    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"\nERROR: Input file not found: {data_path}")
        sys.exit(1)

    # Load data
    print(f"\nLoading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"  Loaded {len(df)} rows")
    print(f"  Columns: {df.columns.tolist()[:10]}...")  # Show first 10

    # Validate sequence columns
    if args.heavy_col not in df.columns:
        print(f"\nERROR: Heavy chain column '{args.heavy_col}' not found")
        print(f"Available columns: {df.columns.tolist()}")
        sys.exit(1)

    if args.light_col not in df.columns:
        print(f"\nERROR: Light chain column '{args.light_col}' not found")
        print(f"Available columns: {df.columns.tolist()}")
        sys.exit(1)

    # Get sequences
    heavy_sequences = df[args.heavy_col].fillna('').tolist()
    light_sequences = df[args.light_col].fillna('').tolist()

    # Concatenated sequences for single-sequence models
    if args.concatenate:
        concatenated_sequences = [h + l for h, l in zip(heavy_sequences, light_sequences)]
    else:
        concatenated_sequences = heavy_sequences  # Default to heavy only for ESM2

    print(f"\nSequence statistics:")
    print(f"  Heavy chain: {len(heavy_sequences)} sequences")
    print(f"    Mean length: {np.mean([len(s) for s in heavy_sequences if s]):.1f}")
    print(f"  Light chain: {len(light_sequences)} sequences")
    print(f"    Mean length: {np.mean([len(s) for s in light_sequences if s]):.1f}")

    # Set device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")

    # Model configurations
    model_configs = {
        'esm2_35m': ('ESM-2 35M', 'facebook/esm2_t12_35M_UR50D'),
        'esm2_150m': ('ESM-2 150M', 'facebook/esm2_t30_150M_UR50D'),
        'esm2_650m': ('ESM-2 650M', 'facebook/esm2_t33_650M_UR50D'),
        'esm2_3b': ('ESM-2 3B', 'facebook/esm2_t36_3B_UR50D'),
    }

    # Evaluate each model
    for model_key in args.models:
        try:
            if model_key in model_configs:
                model_name, model_id = model_configs[model_key]

                # For ESM2, use concatenated heavy+light sequences
                sequences_to_use = [h + l for h, l in zip(heavy_sequences, light_sequences)]

                perplexities = evaluate_esm2_ppl(
                    model_name, model_id, sequences_to_use, device, args.batch_size
                )

            elif model_key == 'ablang2':
                perplexities = evaluate_ablang2_ppl(
                    heavy_sequences, light_sequences, device, args.batch_size
                )

            elif model_key == 'antiberty':
                # AntiBERTy is single-chain: score heavy/light independently, combined PPL
                perplexities = evaluate_antiberty_ppl(
                    heavy_sequences, light_sequences, device, args.batch_size
                )

            elif model_key == 'sapiens':
                perplexities = evaluate_sapiens_ppl(
                    heavy_sequences, light_sequences, device
                )

            else:
                print(f"WARNING: Unknown model {model_key}, skipping...")
                continue

            # Add column to dataframe
            col_name = f'{model_key}_ppl'
            df[col_name] = perplexities
            print(f"\n  Added column: {col_name}")

        except Exception as e:
            print(f"\nERROR: Failed to evaluate {model_key}: {e}")
            import traceback
            traceback.print_exc()
            df[f'{model_key}_ppl'] = float('inf')

    # Save results
    if args.output_path:
        output_path = Path(args.output_path)
    else:
        output_path = data_path.parent / f"{data_path.stem}_with_ppl.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\n{'='*80}")
    print(f"Results saved to: {output_path}")
    print(f"New PPL columns: {[c for c in df.columns if '_ppl' in c]}")
    print("=" * 80)

    # Print summary table
    print("\nPPL Summary Table:")
    print("-" * 60)
    ppl_cols = [c for c in df.columns if '_ppl' in c]
    if ppl_cols:
        summary_data = []
        for col in ppl_cols:
            valid = df[col].replace([np.inf, -np.inf], np.nan).dropna()
            if len(valid) > 0:
                summary_data.append({
                    'Model': col.replace('_ppl', ''),
                    'Mean': f"{valid.mean():.2f}",
                    'Median': f"{valid.median():.2f}",
                    'Min': f"{valid.min():.2f}",
                    'Max': f"{valid.max():.2f}",
                    'Valid': len(valid)
                })

        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()
