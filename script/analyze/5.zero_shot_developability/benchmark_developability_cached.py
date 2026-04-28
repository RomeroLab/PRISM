#!/usr/bin/env python
"""
Per-antibody pseudo-PPL with aggressive batching + CPU multiprocessing for Sapiens.

PPL = exp(-mean log P(aa_i | seq with i masked)).
For each antibody, we build [L x L] batch: L copies of the tokenized sequence,
each masked at a different position. One forward pass -> log_probs for every
position simultaneously. Special-token ids are suppressed in logits.

GPU models (ESM2/AbLang2/AntiBERTy): large batch across antibodies, positions
stacked per antibody. Sapiens: CPU, parallelized across antibodies with
multiprocessing (one worker per CPU, each handles a chunk of antibodies).
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings("ignore")


def _set_inference(m):
    m.train(False)
    return m


# -----------------------------------------------------------------------------
# Per-antibody position batching
# -----------------------------------------------------------------------------

def ppl_one_antibody_hf(model, tokenizer, chain_seq: str, device: str,
                        max_length: int, aa_token_range: Tuple[int, int]) -> float:
    """HuggingFace MaskedLM path. Batches all L position-masks of ONE antibody."""
    tokens = tokenizer(chain_seq, return_tensors="pt", add_special_tokens=True,
                       truncation=True, max_length=max_length)
    input_ids = tokens['input_ids'].to(device)          # [1, L]
    attention_mask = tokens['attention_mask'].to(device)
    L = input_ids.shape[1]
    mask_id = tokenizer.mask_token_id
    all_special_ids = tokenizer.all_special_ids
    lo, hi = aa_token_range

    # AA positions (exclude special tokens)
    tok0 = input_ids[0]
    aa_mask = (tok0 >= lo) & (tok0 <= hi)
    aa_positions = aa_mask.nonzero(as_tuple=True)[0].tolist()
    if not aa_positions:
        return float('inf')

    B = len(aa_positions)
    batch_input = input_ids.expand(B, -1).clone()
    batch_attn = attention_mask.expand(B, -1)
    for i, pos in enumerate(aa_positions):
        batch_input[i, pos] = mask_id

    logs = []
    with torch.no_grad():
        outputs = model(input_ids=batch_input, attention_mask=batch_attn)
        if hasattr(outputs, 'prediction_logits') and outputs.prediction_logits is not None:
            logits = outputs.prediction_logits
        elif hasattr(outputs, 'logits') and outputs.logits is not None:
            logits = outputs.logits
        else:
            return float('inf')
        logits = logits.clone()
        logits[:, :, all_special_ids] = -float('inf')
        log_probs = F.log_softmax(logits, dim=-1)
        for i, pos in enumerate(aa_positions):
            true_id = tok0[pos].item()
            logs.append(log_probs[i, pos, true_id].item())

    return float(np.exp(-np.mean(logs)))


# -----------------------------------------------------------------------------
# ESM2 / AntiBERTy (HuggingFace MaskedLM)
# -----------------------------------------------------------------------------

def evaluate_esm2(model_id, sequences, device):
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForMaskedLM.from_pretrained(model_id).to(device)
    _set_inference(model)
    ppls = []
    for seq in tqdm(sequences, desc=f"{model_id} PPL", leave=False):
        if not seq:
            ppls.append(float('inf')); continue
        ppls.append(ppl_one_antibody_hf(model, tokenizer, seq, device, 1024, (4, 23)))
    del model
    if device == 'cuda':
        torch.cuda.empty_cache()
    return ppls


def evaluate_antiberty(heavy_sequences, light_sequences, device):
    from antiberty import AntiBERTyRunner
    runner = AntiBERTyRunner()
    model = runner.model.to(device)
    _set_inference(model)
    tokenizer = runner.tokenizer
    # AntiBERTy AA ids: A=5 .. Y=23 (21 standard AAs are at 5-24, check specials)
    # Use dynamic range: exclude specials
    all_special = set(tokenizer.all_special_ids)
    # Standard AAs are all non-special tokens
    # For BERT tokenizer we can just identify AAs by convert_tokens_to_ids
    aa_ids = [tokenizer.convert_tokens_to_ids(a) for a in 'ACDEFGHIKLMNPQRSTVWY']
    lo, hi = min(aa_ids), max(aa_ids)

    def _ab_ppl(seq):
        if not seq: return float('inf')
        spaced = ' '.join(list(seq))
        tokens = tokenizer(spaced, return_tensors="pt", add_special_tokens=True,
                           truncation=True, max_length=1024)
        input_ids = tokens['input_ids'].to(device)
        attention_mask = tokens['attention_mask'].to(device)
        L = input_ids.shape[1]
        tok0 = input_ids[0]
        aa_mask = torch.isin(tok0, torch.tensor(aa_ids, device=device))
        aa_positions = aa_mask.nonzero(as_tuple=True)[0].tolist()
        if not aa_positions: return float('inf')

        mask_id = tokenizer.mask_token_id
        B = len(aa_positions)
        batch_input = input_ids.expand(B, -1).clone()
        batch_attn = attention_mask.expand(B, -1)
        for i, pos in enumerate(aa_positions):
            batch_input[i, pos] = mask_id
        with torch.no_grad():
            outputs = model(input_ids=batch_input, attention_mask=batch_attn)
            logits = outputs.prediction_logits if hasattr(outputs, 'prediction_logits') else outputs.logits
            logits = logits.clone()
            logits[:, :, list(all_special)] = -float('inf')
            log_probs = F.log_softmax(logits, dim=-1)
            logs = [log_probs[i, pos, tok0[pos].item()].item() for i, pos in enumerate(aa_positions)]
        return float(np.exp(-np.mean(logs)))

    ppls = []
    for h, l in tqdm(zip(heavy_sequences, light_sequences),
                     total=len(heavy_sequences), desc="AntiBERTy PPL", leave=False):
        combined_logs = []
        for chain_seq in (h, l):
            if not chain_seq: continue
            # get log probs directly
            spaced = ' '.join(list(chain_seq))
            tokens = tokenizer(spaced, return_tensors="pt", add_special_tokens=True,
                               truncation=True, max_length=1024)
            input_ids = tokens['input_ids'].to(device)
            attention_mask = tokens['attention_mask'].to(device)
            tok0 = input_ids[0]
            aa_mask = torch.isin(tok0, torch.tensor(aa_ids, device=device))
            aa_positions = aa_mask.nonzero(as_tuple=True)[0].tolist()
            if not aa_positions: continue
            mask_id = tokenizer.mask_token_id
            B = len(aa_positions)
            batch_input = input_ids.expand(B, -1).clone()
            batch_attn = attention_mask.expand(B, -1)
            for i, pos in enumerate(aa_positions):
                batch_input[i, pos] = mask_id
            with torch.no_grad():
                outputs = model(input_ids=batch_input, attention_mask=batch_attn)
                logits = outputs.prediction_logits if hasattr(outputs, 'prediction_logits') else outputs.logits
                logits = logits.clone()
                logits[:, :, list(all_special)] = -float('inf')
                log_probs = F.log_softmax(logits, dim=-1)
                for i, pos in enumerate(aa_positions):
                    combined_logs.append(log_probs[i, pos, tok0[pos].item()].item())
        ppls.append(float(np.exp(-np.mean(combined_logs))) if combined_logs else float('inf'))
    del model
    if device == 'cuda':
        torch.cuda.empty_cache()
    return ppls


# -----------------------------------------------------------------------------
# AbLang2
# -----------------------------------------------------------------------------

def evaluate_ablang2(heavy_sequences, light_sequences, device):
    import ablang2
    ablang = ablang2.pretrained(model_to_use="ablang2-paired", random_init=False, device=device)
    model = ablang.AbLang
    tokenizer = ablang.tokenizer
    _set_inference(model)
    mask_id = tokenizer.mask_token
    special_ids = list(tokenizer.all_special_tokens)
    aa_ids = [tokenizer.aa_to_token[a] for a in 'ACDEFGHIKLMNPQRSTVWY']

    ppls = []
    for h, l in tqdm(zip(heavy_sequences, light_sequences),
                     total=len(heavy_sequences), desc="AbLang2 PPL", leave=False):
        if not h or not l:
            ppls.append(float('inf')); continue
        paired = f"<{h}>|<{l}>"
        input_ids = tokenizer([paired], pad=True, w_extra_tkns=False, device=device)
        tok0 = input_ids[0]
        aa_mask = torch.isin(tok0, torch.tensor(aa_ids, device=device))
        aa_positions = aa_mask.nonzero(as_tuple=True)[0].tolist()
        if not aa_positions:
            ppls.append(float('inf')); continue
        B = len(aa_positions)
        batch_input = input_ids.expand(B, -1).clone()
        for i, pos in enumerate(aa_positions):
            batch_input[i, pos] = mask_id
        with torch.no_grad():
            logits = model(batch_input).clone()
            logits[:, :, special_ids] = -float('inf')
            log_probs = F.log_softmax(logits, dim=-1)
            logs = [log_probs[i, pos, tok0[pos].item()].item() for i, pos in enumerate(aa_positions)]
        ppls.append(float(np.exp(-np.mean(logs))))
    del model
    if device == 'cuda':
        torch.cuda.empty_cache()
    return ppls


# -----------------------------------------------------------------------------
# Sapiens: CPU multiprocessing
# -----------------------------------------------------------------------------

_SAPIENS_CACHE = {}  # worker-local cache


def _sapiens_worker_init():
    """Load Sapiens models once per worker process."""
    from transformers import RobertaForMaskedLM, RobertaTokenizer
    tokenizer = RobertaTokenizer.from_pretrained("prihodad/biophi-sapiens1-tokenizer")
    heavy_model = RobertaForMaskedLM.from_pretrained("prihodad/biophi-sapiens1-vh")
    light_model = RobertaForMaskedLM.from_pretrained("prihodad/biophi-sapiens1-vl")
    _set_inference(heavy_model)
    _set_inference(light_model)
    h_max = heavy_model.config.max_position_embeddings - heavy_model.config.pad_token_id - 1 - 2
    l_max = light_model.config.max_position_embeddings - light_model.config.pad_token_id - 1 - 2
    aa_ids = [tokenizer.convert_tokens_to_ids(a) for a in 'ACDEFGHIKLMNPQRSTVWY']
    all_special = list(tokenizer.all_special_ids)
    _SAPIENS_CACHE['tokenizer'] = tokenizer
    _SAPIENS_CACHE['heavy_model'] = heavy_model
    _SAPIENS_CACHE['light_model'] = light_model
    _SAPIENS_CACHE['h_max'] = h_max
    _SAPIENS_CACHE['l_max'] = l_max
    _SAPIENS_CACHE['aa_ids'] = aa_ids
    _SAPIENS_CACHE['all_special'] = all_special


def _sapiens_chain_logs(model, tokenizer, seq, max_residues, aa_ids, all_special):
    if not seq or len(seq) > max_residues:
        return []
    tokens = tokenizer(seq, return_tensors="pt", padding=False,
                       truncation=True, max_length=max_residues + 2)
    input_ids = tokens['input_ids']
    attention_mask = tokens['attention_mask']
    tok0 = input_ids[0]
    aa_mask = torch.isin(tok0, torch.tensor(aa_ids))
    aa_positions = aa_mask.nonzero(as_tuple=True)[0].tolist()
    if not aa_positions:
        return []
    mask_id = tokenizer.mask_token_id
    B = len(aa_positions)
    batch_input = input_ids.expand(B, -1).clone()
    batch_attn = attention_mask.expand(B, -1)
    for i, pos in enumerate(aa_positions):
        batch_input[i, pos] = mask_id
    with torch.no_grad():
        outputs = model(input_ids=batch_input, attention_mask=batch_attn)
        logits = outputs.logits.clone()
        logits[:, :, all_special] = -float('inf')
        log_probs = F.log_softmax(logits, dim=-1)
        return [log_probs[i, pos, tok0[pos].item()].item() for i, pos in enumerate(aa_positions)]


def _sapiens_worker_score(args):
    idx, heavy, light = args
    c = _SAPIENS_CACHE
    combined = (
        _sapiens_chain_logs(c['heavy_model'], c['tokenizer'], heavy, c['h_max'], c['aa_ids'], c['all_special'])
        + _sapiens_chain_logs(c['light_model'], c['tokenizer'], light, c['l_max'], c['aa_ids'], c['all_special'])
    )
    ppl = float(np.exp(-np.mean(combined))) if combined else float('inf')
    return idx, ppl


def evaluate_sapiens_parallel(heavy_sequences, light_sequences, workers):
    import multiprocessing as mp
    # Use spawn to avoid issues with forked torch on Linux
    ctx = mp.get_context('spawn')
    args = [(i, h, l) for i, (h, l) in enumerate(zip(heavy_sequences, light_sequences))]

    if workers <= 1:
        _sapiens_worker_init()
        results = [_sapiens_worker_score(a) for a in tqdm(args, desc="Sapiens PPL", leave=False)]
    else:
        with ctx.Pool(workers, initializer=_sapiens_worker_init) as pool:
            results = list(tqdm(pool.imap_unordered(_sapiens_worker_score, args, chunksize=4),
                                total=len(args), desc=f"Sapiens PPL ({workers}w)", leave=False))

    results.sort(key=lambda x: x[0])
    return [r[1] for r in results]


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--models', nargs='+',
                        default=['esm2_35m', 'esm2_650m', 'ablang2', 'antiberty', 'sapiens'],
                        choices=['esm2_35m', 'esm2_650m', 'ablang2', 'antiberty', 'sapiens'])
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--heavy_col', default='vh_protein_sequence')
    parser.add_argument('--light_col', default='vl_protein_sequence')
    parser.add_argument('--sapiens_workers', type=int, default=16)
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    df = pd.read_csv(args.data_path)
    heavy_sequences = df[args.heavy_col].fillna('').tolist()
    light_sequences = df[args.light_col].fillna('').tolist()
    concatenated = [h + l for h, l in zip(heavy_sequences, light_sequences)]
    print(f"  {len(df)} antibodies; heavy mean len {np.mean([len(s) for s in heavy_sequences if s]):.0f}")

    results = {}
    for model_key in args.models:
        t0 = time.time()
        print(f"\n=== {model_key} ===")
        try:
            if model_key in ('esm2_35m', 'esm2_650m'):
                model_id = {'esm2_35m': 'facebook/esm2_t12_35M_UR50D',
                            'esm2_650m': 'facebook/esm2_t33_650M_UR50D'}[model_key]
                ppls = evaluate_esm2(model_id, concatenated, device)
            elif model_key == 'ablang2':
                ppls = evaluate_ablang2(heavy_sequences, light_sequences, device)
            elif model_key == 'antiberty':
                ppls = evaluate_antiberty(heavy_sequences, light_sequences, device)
            elif model_key == 'sapiens':
                ppls = evaluate_sapiens_parallel(heavy_sequences, light_sequences, args.sapiens_workers)
            else:
                continue
            results[model_key] = ppls
            valid = [p for p in ppls if p != float('inf')]
            print(f"  {time.time()-t0:.1f}s | mean={np.mean(valid):.3f}, median={np.median(valid):.3f}, n={len(valid)}")
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback; traceback.print_exc()
            results[model_key] = [float('inf')] * len(df)

    for model_key, ppls in results.items():
        df[f'{model_key}_ppl'] = ppls

    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}")


if __name__ == '__main__':
    main()
