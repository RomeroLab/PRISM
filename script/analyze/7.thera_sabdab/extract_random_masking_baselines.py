#!/usr/bin/env python
# coding: utf-8
"""
Experiment 1: Random Masking Baselines.

Extract predictions from baseline models (ESM2, AbLang2, AntiBERTy, Sapiens)
at stratified random positions (GL and NGL) for comparison with PRISM.

Uses the same loading methods as benchmark_baselines_fixed.py:
- ESM2: HuggingFace transformers
- AbLang2: ablang2 package (ablang2.pretrained)
- AntiBERTy: antiberty package (AntiBERTyRunner)
- Sapiens: HuggingFace with prihodad/biophi-sapiens1-vh/vl

Usage:
    python extract_random_masking_baselines.py \
        --data_path data/therasabdab_germline.csv \
        --exp1_prism data/controllable_generation/exp1_random_masking_prism.csv \
        --output_path data/controllable_generation/exp1_random_masking_baselines.csv \
        --batch_size 32
"""

import argparse
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Standard amino acids
AMINO_ACIDS = list('ACDEFGHIKLMNPQRSTVWY')


# =============================================================================
# ESM2 Models
# =============================================================================

def load_esm2(model_id: str, device: str):
    """Load ESM2 model via HuggingFace."""
    from transformers import AutoTokenizer, EsmForMaskedLM

    print(f"  Loading ESM2 from {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = EsmForMaskedLM.from_pretrained(model_id)
    model.eval()
    model.to(device)

    # Build AA vocab
    aa_to_id = {}
    for aa in AMINO_ACIDS:
        token_id = tokenizer.convert_tokens_to_ids(aa)
        if token_id != tokenizer.unk_token_id:
            aa_to_id[aa] = token_id

    return model, tokenizer, aa_to_id


def predict_esm2(model, tokenizer, sequence: str, position: int,
                 aa_to_id: Dict[str, int], device: str) -> Optional[Dict]:
    """Get prediction from ESM2 at masked position."""
    seq_list = list(sequence.upper())
    seq_list[position] = tokenizer.mask_token
    masked_seq = ''.join(seq_list)

    tokens = tokenizer(
        masked_seq,
        return_tensors='pt',
        add_special_tokens=True,
        truncation=True,
        max_length=512
    )
    input_ids = tokens['input_ids'].to(device)
    attention_mask = tokens['attention_mask'].to(device)

    # Find mask position
    mask_positions = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)
    if len(mask_positions[1]) == 0:
        return None
    token_pos = mask_positions[1][0].item()

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[0, token_pos].cpu()

    aa_logits = {aa: logits[idx].item() for aa, idx in aa_to_id.items()}
    predicted_aa = max(aa_logits, key=aa_logits.get)

    return {'predicted_aa': predicted_aa, 'logits': aa_logits}


# =============================================================================
# AbLang2 Model
# =============================================================================

def load_ablang2(device: str):
    """Load AbLang2 using the ablang2 package."""
    import ablang2

    print("  Loading AbLang2 (paired mode)...")
    ablang = ablang2.pretrained(
        model_to_use="ablang2-paired",
        random_init=False,
        device=device,
    )
    model = ablang.AbLang
    tokenizer = ablang.tokenizer
    model.eval()

    # Build AA vocab from ablang2 tokenizer
    # AbLang2 uses aa_to_token attribute
    aa_to_id = {}
    if hasattr(tokenizer, 'aa_to_token'):
        for aa in AMINO_ACIDS:
            token_id = tokenizer.aa_to_token.get(aa)
            if token_id is not None:
                aa_to_id[aa] = token_id
        print(f"  Using aa_to_token mapping: {len(aa_to_id)} AAs found")
    else:
        # Fallback - this shouldn't happen for ablang2
        print("  WARNING: aa_to_token not found, using fallback")
        for i, aa in enumerate(AMINO_ACIDS):
            aa_to_id[aa] = i + 4

    return model, tokenizer, aa_to_id, ablang


def predict_ablang2(ablang, sequence: str, position: int, chain: str,
                    heavy_seq: str, light_seq: str, aa_to_id: Dict[str, int],
                    device: str) -> Optional[Dict]:
    """Get prediction from AbLang2 at masked position.

    AbLang2 expects paired format: "HEAVY|LIGHT"
    """
    # Create masked sequence for the appropriate chain
    if chain == 'heavy':
        seq_list = list(heavy_seq.upper())
        seq_list[position] = '*'  # AbLang2 mask token
        masked_heavy = ''.join(seq_list)
        paired_seq = f"{masked_heavy}|{light_seq}"
        # Position in tokenized: position (0-indexed in heavy)
        token_pos = position
    else:  # light
        seq_list = list(light_seq.upper())
        seq_list[position] = '*'
        masked_light = ''.join(seq_list)
        paired_seq = f"{heavy_seq}|{masked_light}"
        # Position in tokenized: heavy_len + 1 (separator) + position
        token_pos = len(heavy_seq) + 1 + position

    try:
        tokenized = ablang.tokenizer([paired_seq], pad=True, w_extra_tkns=False, device=device)

        with torch.no_grad():
            outputs = ablang.AbLang(tokenized)
            logits = outputs[0, token_pos].cpu()

        aa_logits = {aa: logits[idx].item() for aa, idx in aa_to_id.items() if idx < logits.shape[0]}
        if not aa_logits:
            return None
        predicted_aa = max(aa_logits, key=aa_logits.get)

        return {'predicted_aa': predicted_aa, 'logits': aa_logits}
    except Exception as e:
        return None


# =============================================================================
# AntiBERTy Model
# =============================================================================

def load_antiberty(device: str):
    """Load AntiBERTy using the antiberty package."""
    from antiberty import AntiBERTyRunner

    print("  Loading AntiBERTy...")
    runner = AntiBERTyRunner()
    model = runner.model.to(device)
    tokenizer = runner.tokenizer
    model.eval()

    # Build AA vocab
    aa_to_id = {}
    for aa in AMINO_ACIDS:
        token_id = tokenizer.convert_tokens_to_ids(aa)
        aa_to_id[aa] = token_id

    return model, tokenizer, aa_to_id


def predict_antiberty(model, tokenizer, sequence: str, position: int,
                      aa_to_id: Dict[str, int], device: str) -> Optional[Dict]:
    """Get prediction from AntiBERTy at masked position.

    AntiBERTy expects space-separated amino acids.
    """
    seq_list = list(sequence.upper())
    seq_list[position] = tokenizer.mask_token
    spaced_seq = ' '.join(seq_list)

    tokens = tokenizer(
        spaced_seq,
        return_tensors='pt',
        add_special_tokens=True,
        truncation=True,
        max_length=512
    )
    input_ids = tokens['input_ids'].to(device)
    attention_mask = tokens['attention_mask'].to(device)

    # Token position: +1 for CLS
    token_pos = position + 1

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(outputs, 'logits') and outputs.logits is not None:
            logits = outputs.logits[0, token_pos].cpu()
        elif hasattr(outputs, 'prediction_logits'):
            logits = outputs.prediction_logits[0, token_pos].cpu()
        else:
            return None

    aa_logits = {aa: logits[idx].item() for aa, idx in aa_to_id.items()}
    predicted_aa = max(aa_logits, key=aa_logits.get)

    return {'predicted_aa': predicted_aa, 'logits': aa_logits}


# =============================================================================
# Sapiens Model
# =============================================================================

def load_sapiens(device: str):
    """Load Sapiens models (separate H and L).

    Note: Forced to CPU for stability.
    """
    from transformers import RobertaForMaskedLM, RobertaTokenizer

    device = 'cpu'  # Force CPU for stability

    print("  Loading Sapiens (heavy + light models, CPU for stability)...")
    tokenizer = RobertaTokenizer.from_pretrained("prihodad/biophi-sapiens1-tokenizer")
    heavy_model = RobertaForMaskedLM.from_pretrained("prihodad/biophi-sapiens1-vh")
    light_model = RobertaForMaskedLM.from_pretrained("prihodad/biophi-sapiens1-vl")

    heavy_model.to(device)
    light_model.to(device)
    heavy_model.eval()
    light_model.eval()

    # Build AA vocab
    aa_to_id = {}
    for aa in AMINO_ACIDS:
        token_id = tokenizer.convert_tokens_to_ids(aa)
        aa_to_id[aa] = token_id

    return heavy_model, light_model, tokenizer, aa_to_id, device


def predict_sapiens(heavy_model, light_model, tokenizer, sequence: str,
                    position: int, chain: str, aa_to_id: Dict[str, int],
                    device: str) -> Optional[Dict]:
    """Get prediction from Sapiens at masked position.

    Uses chain-specific model (heavy or light).
    """
    max_seq_len = 143  # Sapiens limit

    if len(sequence) > max_seq_len:
        return None

    model = heavy_model if chain == 'heavy' else light_model

    seq_list = list(sequence.upper())
    seq_list[position] = tokenizer.mask_token
    masked_seq = ''.join(seq_list)

    tokens = tokenizer(
        masked_seq,
        return_tensors='pt',
        padding=False,
        truncation=True,
        max_length=max_seq_len + 2
    )
    input_ids = tokens['input_ids'].to(device)
    attention_mask = tokens['attention_mask'].to(device)

    # Token position: +1 for CLS
    token_pos = position + 1

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[0, token_pos].cpu()

    aa_logits = {aa: logits[idx].item() for aa, idx in aa_to_id.items()}
    predicted_aa = max(aa_logits, key=aa_logits.get)

    return {'predicted_aa': predicted_aa, 'logits': aa_logits}


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Random masking baselines')
    parser.add_argument('--data_path', type=str, default='data/therasabdab_germline.csv')
    parser.add_argument('--exp1_prism', type=str, required=True,
                        help='Path to exp1_random_masking_prism.csv (to get same positions)')
    parser.add_argument('--output_path', type=str,
                        default='data/controllable_generation/exp1_random_masking_baselines.csv')
    parser.add_argument('--models', nargs='+',
                        default=['ESM2_35M', 'ESM2_650M', 'AbLang2', 'AntiBERTy', 'Sapiens'])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()

    print("=" * 80)
    print("EXPERIMENT 1: RANDOM MASKING (BASELINES)")
    print("=" * 80)

    # Load PRISM results to get the same positions
    print(f"\nLoading PRISM results (for position alignment): {args.exp1_prism}")
    df_prism = pd.read_csv(args.exp1_prism)
    print(f"  Loaded {len(df_prism)} positions")
    print(f"  GL positions: {len(df_prism[df_prism['position_type'] == 'GL'])}")
    print(f"  NGL positions: {len(df_prism[df_prism['position_type'] == 'NGL'])}")

    # Load original data for sequences
    print(f"\nLoading sequence data: {args.data_path}")
    df_data = pd.read_csv(args.data_path)
    print(f"  Loaded {len(df_data)} antibodies")

    # Create sequence lookup (both heavy and light for each therapeutic)
    seq_lookup = {}
    for _, row in df_data.iterrows():
        therapeutic = row.get('Therapeutic', '')
        heavy_seq = row.get('HeavySequence', row.get('heavy_sequence', ''))
        light_seq = row.get('LightSequence', row.get('light_sequence', ''))
        if therapeutic:
            if heavy_seq and not pd.isna(heavy_seq):
                seq_lookup[(therapeutic, 'heavy')] = str(heavy_seq)
            if light_seq and not pd.isna(light_seq):
                seq_lookup[(therapeutic, 'light')] = str(light_seq)
            # Store paired info
            if heavy_seq and light_seq and not pd.isna(heavy_seq) and not pd.isna(light_seq):
                seq_lookup[(therapeutic, 'paired')] = (str(heavy_seq), str(light_seq))

    all_results = []

    for model_name in args.models:
        print(f"\n{'='*60}")
        print(f"Processing: {model_name}")
        print("=" * 60)

        try:
            # Load model based on type
            if model_name == 'ESM2_35M':
                model, tokenizer, aa_to_id = load_esm2('facebook/esm2_t12_35M_UR50D', args.device)
                model_type = 'esm2'
            elif model_name == 'ESM2_650M':
                model, tokenizer, aa_to_id = load_esm2('facebook/esm2_t33_650M_UR50D', args.device)
                model_type = 'esm2'
            elif model_name == 'AbLang2':
                model, tokenizer, aa_to_id, ablang = load_ablang2(args.device)
                model_type = 'ablang2'
            elif model_name == 'AntiBERTy':
                model, tokenizer, aa_to_id = load_antiberty(args.device)
                model_type = 'antiberty'
            elif model_name == 'Sapiens':
                heavy_model, light_model, tokenizer, aa_to_id, sapiens_device = load_sapiens(args.device)
                model_type = 'sapiens'
            else:
                print(f"  Unknown model: {model_name}")
                continue

            print(f"  AA tokens found: {len(aa_to_id)}")

        except Exception as e:
            print(f"  ERROR loading {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

        # Process each position from PRISM results
        for idx, row in tqdm(df_prism.iterrows(), total=len(df_prism),
                            desc=f"{model_name}"):
            therapeutic = row['Therapeutic']
            chain = row['chain']
            position = int(row['position'])
            position_type = row['position_type']
            region_id = row['region_id']
            region_type = row['region_type']
            germline_aa = row['germline_aa']
            ground_truth_aa = row['ground_truth_aa']
            is_mutation_site = row['is_mutation_site']

            # Get sequence
            seq_key = (therapeutic, chain)
            if seq_key not in seq_lookup:
                continue
            sequence = seq_lookup[seq_key]

            if position >= len(sequence):
                continue

            # Extract prediction based on model type
            preds = None
            try:
                if model_type == 'esm2':
                    preds = predict_esm2(model, tokenizer, sequence, position, aa_to_id, args.device)
                elif model_type == 'ablang2':
                    # AbLang2 needs paired sequences
                    paired_key = (therapeutic, 'paired')
                    if paired_key in seq_lookup:
                        heavy_seq, light_seq = seq_lookup[paired_key]
                        preds = predict_ablang2(ablang, sequence, position, chain,
                                              heavy_seq, light_seq, aa_to_id, args.device)
                elif model_type == 'antiberty':
                    preds = predict_antiberty(model, tokenizer, sequence, position, aa_to_id, args.device)
                elif model_type == 'sapiens':
                    preds = predict_sapiens(heavy_model, light_model, tokenizer, sequence,
                                          position, chain, aa_to_id, sapiens_device)
            except Exception as e:
                continue

            if preds is None:
                continue

            predicted_aa = preds['predicted_aa']
            is_correct = predicted_aa.upper() == ground_truth_aa.upper()
            is_reversion = predicted_aa.upper() == germline_aa.upper()

            result = {
                'Therapeutic': therapeutic,
                'chain': chain,
                'position': position,
                'position_type': position_type,
                'region_id': region_id,
                'region_type': region_type,
                'germline_aa': germline_aa,
                'ground_truth_aa': ground_truth_aa,
                'is_mutation_site': is_mutation_site,
                'model': model_name,
                'predicted_aa': predicted_aa,
                'is_correct': is_correct,
                'is_reversion': is_reversion,
            }

            # Add logits
            for aa, logit in preds['logits'].items():
                result[f'{aa}_logit'] = logit

            all_results.append(result)

        # Free memory
        if model_type == 'esm2':
            del model
        elif model_type == 'ablang2':
            del model, ablang
        elif model_type == 'antiberty':
            del model
        elif model_type == 'sapiens':
            del heavy_model, light_model
        torch.cuda.empty_cache()

    # Save results
    results_df = pd.DataFrame(all_results)

    output_dir = Path(args.output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(args.output_path, index=False)
    print(f"\n{'='*60}")
    print(f"Results saved to: {args.output_path}")
    print(f"  Total rows: {len(results_df)}")
    print("=" * 60)

    # Print summary
    print("\nSUMMARY BY MODEL AND POSITION TYPE:")
    for model_name in args.models:
        model_df = results_df[results_df['model'] == model_name]
        if len(model_df) == 0:
            continue

        print(f"\n{model_name}:")
        for pos_type in ['GL', 'NGL']:
            subset = model_df[model_df['position_type'] == pos_type]
            if len(subset) > 0:
                acc = subset['is_correct'].mean() * 100
                print(f"  {pos_type} positions: {acc:.1f}% accuracy (n={len(subset)})")


if __name__ == "__main__":
    main()
