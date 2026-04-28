#!/usr/bin/env python
"""
Position-cached masked-marginal LLR for DMS datasets.

All variants in a DMS dataset share the same WT, so `log P(aa | WT with pos_k masked)`
depends only on (pos_k, aa) — not on the variant. We precompute a cache
    cache[(chain, pos)] = log_probs_at_that_masked_position  (shape [vocab])
for each unique mutation position across the dataset, then score each variant
by dict lookup. Mathematically identical to per-variant masked marginal, but
orders of magnitude fewer forward passes.

For cr9114 (65K variants x 8 muts = 524K per-variant forward passes) we run
only 16 forward passes per model. Batches of masked inputs are stacked into
the same forward pass so GPU batch size stays healthy.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from tqdm.auto import tqdm

import warnings
warnings.filterwarnings("ignore")


def find_mutation_indices(wt_seq, mut_seq):
    if len(wt_seq) != len(mut_seq):
        return []
    return [(i, w, m) for i, (w, m) in enumerate(zip(wt_seq, mut_seq)) if w != m]


def collect_unique_positions(df, wt_heavy, wt_light):
    heavy_pos, light_pos = set(), set()
    for _, row in df.iterrows():
        if row['Mutations'] == 'WT':
            continue
        for i, w, m in find_mutation_indices(wt_heavy, row['fv_heavy']):
            heavy_pos.add(i)
        for i, w, m in find_mutation_indices(wt_light, row['fv_light']):
            light_pos.add(i)
    return heavy_pos, light_pos


def _set_inference(m):
    m.train(False)
    return m


# ESM2
def build_cache_esm2(model, tokenizer, wt_seq, positions, device, batch_size=64):
    mask_id = tokenizer.mask_token_id
    all_special_ids = tokenizer.all_special_ids

    inputs = tokenizer(wt_seq, return_tensors="pt", add_special_tokens=True,
                       truncation=True, max_length=1024)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    cache = {}
    positions = sorted(positions)
    for bs in range(0, len(positions), batch_size):
        batch_pos = positions[bs:bs + batch_size]
        B = len(batch_pos)
        batch_input = input_ids.expand(B, -1).clone()
        batch_attn = attention_mask.expand(B, -1)
        for i, pos in enumerate(batch_pos):
            batch_input[i, pos + 1] = mask_id

        with torch.no_grad():
            outputs = model(input_ids=batch_input, attention_mask=batch_attn)
            logits = outputs.logits.clone()
            logits[:, :, all_special_ids] = -float("inf")
            log_probs = F.log_softmax(logits, dim=-1)

        for i, pos in enumerate(batch_pos):
            cache[pos] = log_probs[i, pos + 1, :].detach().cpu()
    return cache


def load_esm2(model_id, device):
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForMaskedLM.from_pretrained(model_id).to(device)
    _set_inference(model)
    aa_to_idx = {aa: tokenizer.convert_tokens_to_ids(aa) for aa in 'ACDEFGHIKLMNPQRSTVWY'}
    return model, tokenizer, aa_to_idx


# AbLang2
def build_cache_ablang2(model, tokenizer, wt_heavy, wt_light,
                        heavy_positions, light_positions, device, batch_size=64):
    mask_id = tokenizer.mask_token
    special_ids = list(tokenizer.all_special_tokens)

    wt_library_str = f"<{wt_heavy}>|<{wt_light}>"
    tokenized_wt = tokenizer([wt_library_str], pad=True, w_extra_tkns=False, device=device)

    heavy_offset = 1
    light_offset = len(wt_heavy) + 4

    def _cache_one_chain(positions, offset):
        cache = {}
        positions = sorted(positions)
        for bs in range(0, len(positions), batch_size):
            batch_pos = positions[bs:bs + batch_size]
            B = len(batch_pos)
            batch_input = tokenized_wt.expand(B, -1).clone()
            for i, pos in enumerate(batch_pos):
                batch_input[i, offset + pos] = mask_id

            with torch.no_grad():
                logits = model(batch_input).clone()
                logits[:, :, special_ids] = -float("inf")
                log_probs = F.log_softmax(logits, dim=-1)

            for i, pos in enumerate(batch_pos):
                cache[pos] = log_probs[i, offset + pos, :].detach().cpu()
        return cache

    return _cache_one_chain(heavy_positions, heavy_offset), _cache_one_chain(light_positions, light_offset)


def load_ablang2(device):
    import ablang2
    ablang = ablang2.pretrained(model_to_use="ablang2-paired", random_init=False, device=device)
    model = ablang.AbLang
    tokenizer = ablang.tokenizer
    _set_inference(model)
    aa_to_idx = {aa: tokenizer.aa_to_token[aa] for aa in 'ACDEFGHIKLMNPQRSTVWY'}
    return model, tokenizer, aa_to_idx


# AntiBERTy (single-chain)
def build_cache_antiberty_chain(model, tokenizer, chain_seq, positions, device, batch_size=64):
    mask_token_id = tokenizer.mask_token_id
    all_special_ids = tokenizer.all_special_ids

    spaced = ' '.join(list(chain_seq))
    tokens = tokenizer(spaced, return_tensors="pt", add_special_tokens=True,
                       truncation=True, max_length=1024)
    input_ids = tokens['input_ids'].to(device)
    attention_mask = tokens['attention_mask'].to(device)

    cache = {}
    positions = sorted(positions)
    for bs in range(0, len(positions), batch_size):
        batch_pos = positions[bs:bs + batch_size]
        B = len(batch_pos)
        batch_input = input_ids.expand(B, -1).clone()
        batch_attn = attention_mask.expand(B, -1)
        for i, pos in enumerate(batch_pos):
            batch_input[i, pos + 1] = mask_token_id

        with torch.no_grad():
            outputs = model(input_ids=batch_input, attention_mask=batch_attn)
            if hasattr(outputs, 'prediction_logits') and outputs.prediction_logits is not None:
                logits = outputs.prediction_logits
            elif hasattr(outputs, 'logits') and outputs.logits is not None:
                logits = outputs.logits
            else:
                return cache
            logits = logits.clone()
            logits[:, :, all_special_ids] = -float("inf")
            log_probs = F.log_softmax(logits, dim=-1)

        for i, pos in enumerate(batch_pos):
            cache[pos] = log_probs[i, pos + 1, :].detach().cpu()
    return cache


def load_antiberty(device):
    from antiberty import AntiBERTyRunner
    runner = AntiBERTyRunner()
    model = runner.model.to(device)
    _set_inference(model)
    tokenizer = runner.tokenizer
    aa_to_idx = {aa: tokenizer.convert_tokens_to_ids(aa) for aa in 'ACDEFGHIKLMNPQRSTVWY'}
    return model, tokenizer, aa_to_idx


# Sapiens (two separate models; forced CPU)
def build_cache_sapiens_chain(model, tokenizer, chain_seq, max_residues,
                              positions, device, batch_size=32):
    if not positions or len(chain_seq) > max_residues:
        return {}
    mask_token_id = tokenizer.mask_token_id
    all_special_ids = tokenizer.all_special_ids

    tokens = tokenizer(chain_seq, return_tensors="pt", padding=False,
                       truncation=True, max_length=max_residues + 2)
    input_ids = tokens['input_ids'].to(device)
    attention_mask = tokens['attention_mask'].to(device)

    cache = {}
    positions = sorted(positions)
    for bs in range(0, len(positions), batch_size):
        batch_pos = positions[bs:bs + batch_size]
        B = len(batch_pos)
        batch_input = input_ids.expand(B, -1).clone()
        batch_attn = attention_mask.expand(B, -1)
        for i, pos in enumerate(batch_pos):
            batch_input[i, pos + 1] = mask_token_id

        with torch.no_grad():
            outputs = model(input_ids=batch_input, attention_mask=batch_attn)
            logits = outputs.logits.clone()
            logits[:, :, all_special_ids] = -float("inf")
            log_probs = F.log_softmax(logits, dim=-1)

        for i, pos in enumerate(batch_pos):
            cache[pos] = log_probs[i, pos + 1, :].detach().cpu()
    return cache


def load_sapiens():
    from transformers import RobertaForMaskedLM, RobertaTokenizer
    device = 'cpu'
    tokenizer = RobertaTokenizer.from_pretrained("prihodad/biophi-sapiens1-tokenizer")
    heavy_model = RobertaForMaskedLM.from_pretrained("prihodad/biophi-sapiens1-vh").to(device)
    light_model = RobertaForMaskedLM.from_pretrained("prihodad/biophi-sapiens1-vl").to(device)
    _set_inference(heavy_model)
    _set_inference(light_model)

    def _chain_max_residues(cfg):
        return cfg.max_position_embeddings - cfg.pad_token_id - 1 - 2

    heavy_max = _chain_max_residues(heavy_model.config)
    light_max = _chain_max_residues(light_model.config)
    aa_to_idx = {aa: tokenizer.convert_tokens_to_ids(aa) for aa in 'ACDEFGHIKLMNPQRSTVWY'}
    return heavy_model, light_model, tokenizer, aa_to_idx, heavy_max, light_max


def score_variant(heavy_cache, light_cache, aa_to_idx, heavy_muts, light_muts):
    s = 0.0
    for pos, wt_aa, mut_aa in heavy_muts:
        wt_id = aa_to_idx.get(wt_aa)
        mut_id = aa_to_idx.get(mut_aa)
        if wt_id is None or mut_id is None or pos not in heavy_cache:
            continue
        s += (heavy_cache[pos][mut_id] - heavy_cache[pos][wt_id]).item()
    for pos, wt_aa, mut_aa in light_muts:
        wt_id = aa_to_idx.get(wt_aa)
        mut_id = aa_to_idx.get(mut_aa)
        if wt_id is None or mut_id is None or pos not in light_cache:
            continue
        s += (light_cache[pos][mut_id] - light_cache[pos][wt_id]).item()
    return s


def evaluate_dataset(df, wt_heavy, wt_light, heavy_positions, light_positions,
                    model_keys, device, batch_size):
    results = {}
    h_pos = sorted(heavy_positions)
    l_pos = sorted(light_positions)

    for model_key in model_keys:
        t0 = time.time()
        print(f"\n{'='*60}\n{model_key}\n{'='*60}")

        try:
            if model_key in ('esm2_35m', 'esm2_650m'):
                model_id = {
                    'esm2_35m': 'facebook/esm2_t12_35M_UR50D',
                    'esm2_650m': 'facebook/esm2_t33_650M_UR50D',
                }[model_key]
                model, tokenizer, aa_to_idx = load_esm2(model_id, device)
                wt_seq = wt_heavy + wt_light
                all_positions = list(h_pos) + [len(wt_heavy) + p for p in l_pos]
                full_cache = build_cache_esm2(model, tokenizer, wt_seq, all_positions, device, batch_size)
                heavy_cache = {p: full_cache[p] for p in h_pos}
                light_cache = {p: full_cache[len(wt_heavy) + p] for p in l_pos}
                del model
                if device == 'cuda':
                    torch.cuda.empty_cache()

            elif model_key == 'ablang2':
                model, tokenizer, aa_to_idx = load_ablang2(device)
                heavy_cache, light_cache = build_cache_ablang2(
                    model, tokenizer, wt_heavy, wt_light, h_pos, l_pos, device, batch_size)
                del model
                if device == 'cuda':
                    torch.cuda.empty_cache()

            elif model_key == 'antiberty':
                model, tokenizer, aa_to_idx = load_antiberty(device)
                heavy_cache = build_cache_antiberty_chain(model, tokenizer, wt_heavy, h_pos, device, batch_size)
                light_cache = build_cache_antiberty_chain(model, tokenizer, wt_light, l_pos, device, batch_size)
                del model
                if device == 'cuda':
                    torch.cuda.empty_cache()

            elif model_key == 'sapiens':
                heavy_model, light_model, tokenizer, aa_to_idx, h_max, l_max = load_sapiens()
                heavy_cache = build_cache_sapiens_chain(heavy_model, tokenizer, wt_heavy, h_max, h_pos, 'cpu', batch_size)
                light_cache = build_cache_sapiens_chain(light_model, tokenizer, wt_light, l_max, l_pos, 'cpu', batch_size)
                del heavy_model, light_model

            else:
                print(f"Unknown model: {model_key}")
                continue

            t_cache = time.time() - t0
            print(f"  Cache built in {t_cache:.1f}s ({len(heavy_cache)} heavy + {len(light_cache)} light positions)")

            t1 = time.time()
            scores = []
            for _, row in df.iterrows():
                if row['Mutations'] == 'WT':
                    scores.append(0.0)
                    continue
                h_muts = find_mutation_indices(wt_heavy, row['fv_heavy'])
                l_muts = find_mutation_indices(wt_light, row['fv_light'])
                if not h_muts and not l_muts:
                    scores.append(0.0)
                    continue
                scores.append(score_variant(heavy_cache, light_cache, aa_to_idx, h_muts, l_muts))

            t_score = time.time() - t1
            print(f"  Scored {len(scores)} variants in {t_score:.1f}s (total: {time.time()-t0:.1f}s)")

            results[model_key] = scores

        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            results[model_key] = [0.0] * len(df)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--models', nargs='+',
                        default=['esm2_35m', 'esm2_650m', 'ablang2', 'antiberty', 'sapiens'],
                        choices=['esm2_35m', 'esm2_650m', 'ablang2', 'antiberty', 'sapiens'])
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    print(f"\nLoading: {args.data_path}")
    df = pd.read_csv(args.data_path)
    print(f"  {len(df)} rows")

    wt_rows = df[df['Mutations'] == 'WT']
    if not len(wt_rows):
        raise ValueError("No WT row found")
    wt_heavy = wt_rows.iloc[0]['fv_heavy']
    wt_light = wt_rows.iloc[0]['fv_light']
    print(f"  WT heavy len={len(wt_heavy)}, light len={len(wt_light)}")

    print("\nCollecting unique mutation positions...")
    heavy_positions, light_positions = collect_unique_positions(df, wt_heavy, wt_light)
    print(f"  heavy: {len(heavy_positions)} unique, light: {len(light_positions)} unique")

    results = evaluate_dataset(df, wt_heavy, wt_light, heavy_positions, light_positions,
                               args.models, device, args.batch_size)

    for model_key, scores in results.items():
        df[f'{model_key}_score'] = scores

    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}")

    if 'fitness' in df.columns:
        print("\n" + "=" * 60)
        print("Spearman / Pearson vs fitness (mutants only, nonzero scores)")
        print("=" * 60)
        dmut = df[df['Mutations'] != 'WT'].copy()
        for model_key in results:
            col = f'{model_key}_score'
            mask = (~dmut[col].isna()) & (~dmut['fitness'].isna()) & (dmut[col] != 0.0)
            if mask.sum() < 10:
                print(f"  {model_key}: n<10, skipping")
                continue
            sr, _ = spearmanr(dmut.loc[mask, col], dmut.loc[mask, 'fitness'])
            pr, _ = pearsonr(dmut.loc[mask, col], dmut.loc[mask, 'fitness'])
            print(f"  {model_key:<12} n={mask.sum():>6}  Spearman={sr:+.4f}  Pearson={pr:+.4f}")


if __name__ == '__main__':
    main()
