#!/usr/bin/env python
# coding: utf-8

from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
import sys
from pathlib import Path
import pandas as pd

# ! api does not work, requires updates. I got 0's for every sequence variant
def evaluate_prism_model(
    df: pd.DataFrame,
    wt_heavy: str,
    wt_light: str,
    device: str = "cpu",
    heavy_col: str = "fv_heavy",
    light_col: str = "fv_light",
    mutation_col: str = "Mutations",
    hf_repo_id: str = "RomeroLab-Duke/prism-antibody",
) -> Dict[str, List[float]]:
    """
    Evaluate PRISM paired model on the dataset using mutation scoring.
    """
    try:
        import prism
    except ImportError:
        print("WARNING: prism not installed, skipping...")
        return {
            "PRISM_marginalized_score": [0.0] * len(df),
            "PRISM_gl_score": [0.0] * len(df),
            "PRISM_ngl_score": [0.0] * len(df),
            "PRISM_exact_score": [0.0] * len(df),
        }

    print(f"\n{'='*60}")
    print(f"Loading PRISM ({hf_repo_id})...")
    print("=" * 60)

    model = prism.pretrained(hf_repo_id)
    print(f"  Model loaded on {device}")
    print(f"  WT heavy length: {len(wt_heavy)}")
    print(f"  WT light length: {len(wt_light)}")

    out = {
        "PRISM_marginalized_score": [],
        "PRISM_gl_score": [],
        "PRISM_ngl_score": [],
        "PRISM_exact_score": [],
    }

    non_zero_counts = {
        "PRISM_marginalized_score": 0,
        "PRISM_gl_score": 0,
        "PRISM_ngl_score": 0,
        "PRISM_exact_score": 0,
    }

    debug_counts = {
        "non_string": 0,
        "wt_rows": 0,
        "length_mismatch": 0,
        "no_mutations_found": 0,
        "prism_exception": 0,
        "all_zero_with_mutations": 0,
        "scored_rows": 0,
    }

    max_debug_prints = 5

    for idx in tqdm(range(len(df)), desc="Scoring with PRISM"):
        row = df.iloc[idx]

        mutation_str = row.get(mutation_col, None)
        mut_heavy = row.get(heavy_col, None)
        mut_light = row.get(light_col, None)

        # ------------------------------------------------------------
        # Non-string inputs
        # ------------------------------------------------------------
        if not isinstance(mut_heavy, str) or not isinstance(mut_light, str):
            debug_counts["non_string"] += 1
            if debug_counts["non_string"] <= max_debug_prints:
                print(f"\n[DEBUG non_string] idx={idx}")
                print(f"  mutation_str: {mutation_str}")
                print(f"  type(mut_heavy): {type(mut_heavy)}")
                print(f"  type(mut_light): {type(mut_light)}")
                print(f"  mut_heavy: {repr(mut_heavy)}")
                print(f"  mut_light: {repr(mut_light)}")
            for k in out:
                out[k].append(0.0)
            continue

        mut_heavy = mut_heavy.strip()
        mut_light = mut_light.strip()

        # ------------------------------------------------------------
        # WT rows
        # ------------------------------------------------------------
        if mutation_str == "WT":
            debug_counts["wt_rows"] += 1
            if debug_counts["wt_rows"] <= 3:
                print(f"\n[DEBUG WT row] idx={idx} skipped because mutation_str == 'WT'")
            for k in out:
                out[k].append(0.0)
            continue

        # ------------------------------------------------------------
        # Length mismatch
        # ------------------------------------------------------------
        if len(mut_heavy) != len(wt_heavy) or len(mut_light) != len(wt_light):
            debug_counts["length_mismatch"] += 1
            if debug_counts["length_mismatch"] <= max_debug_prints:
                print(f"\n[DEBUG length_mismatch] idx={idx}")
                print(f"  mutation_str: {mutation_str}")
                print(f"  WT heavy len: {len(wt_heavy)}, mut heavy len: {len(mut_heavy)}")
                print(f"  WT light len: {len(wt_light)}, mut light len: {len(mut_light)}")
                print(f"  mut_heavy[:40]: {mut_heavy[:40]}")
                print(f"  mut_light[:40]: {mut_light[:40]}")
            for k in out:
                out[k].append(0.0)
            continue

        heavy_mutations = find_mutation_indices(wt_heavy, mut_heavy)
        light_mutations = find_mutation_indices(wt_light, mut_light)

        # ------------------------------------------------------------
        # No mutations found
        # ------------------------------------------------------------
        if not heavy_mutations and not light_mutations:
            debug_counts["no_mutations_found"] += 1
            if debug_counts["no_mutations_found"] <= max_debug_prints:
                print(f"\n[DEBUG no_mutations_found] idx={idx}")
                print(f"  mutation_str: {mutation_str}")
                print(f"  WT heavy == mut heavy: {wt_heavy == mut_heavy}")
                print(f"  WT light == mut light: {wt_light == mut_light}")
                print(f"  WT heavy[:40]:  {wt_heavy[:40]}")
                print(f"  mut heavy[:40]: {mut_heavy[:40]}")
                print(f"  WT light[:40]:  {wt_light[:40]}")
                print(f"  mut light[:40]: {mut_light[:40]}")
            for k in out:
                out[k].append(0.0)
            continue

        debug_counts["scored_rows"] += 1

        # Print first few mutation summaries
        if debug_counts["scored_rows"] <= max_debug_prints:
            print(f"\n[DEBUG scored_row] idx={idx}")
            print(f"  mutation_str: {mutation_str}")
            print(f"  n_heavy_mutations: {len(heavy_mutations)}")
            print(f"  n_light_mutations: {len(light_mutations)}")
            print(f"  heavy_mutations[:10]: {heavy_mutations[:10]}")
            print(f"  light_mutations[:10]: {light_mutations[:10]}")

        try:
            result = model.score_mutations(
                wt=wt_heavy,
                mutant=mut_heavy,
                wt_light_chains=wt_light,
                mut_light_chains=mut_light,
            )

            if debug_counts["scored_rows"] <= max_debug_prints:
                print(f"  PRISM raw result keys: {list(result.keys())}")
                print(f"  PRISM raw result: {result}")

            marginalized_score = float(result.get("marginalized", {}).get("score", 0.0))
            gl_score = float(result.get("gl", {}).get("score", 0.0))
            ngl_score = float(result.get("ngl", {}).get("score", 0.0))
            exact_score = float(result.get("exact", {}).get("score", 0.0))

            # Diagnose "all zero despite real mutations"
            if (
                marginalized_score == 0.0
                and gl_score == 0.0
                and ngl_score == 0.0
                and exact_score == 0.0
            ):
                debug_counts["all_zero_with_mutations"] += 1
                if debug_counts["all_zero_with_mutations"] <= max_debug_prints:
                    print(f"\n[DEBUG all_zero_with_mutations] idx={idx}")
                    print(f"  mutation_str: {mutation_str}")
                    print(f"  heavy_mutations: {heavy_mutations}")
                    print(f"  light_mutations: {light_mutations}")
                    print(f"  wt_heavy[:60]:  {wt_heavy[:60]}")
                    print(f"  mut_heavy[:60]: {mut_heavy[:60]}")
                    print(f"  wt_light[:60]:  {wt_light[:60]}")
                    print(f"  mut_light[:60]: {mut_light[:60]}")
                    print(f"  raw result: {result}")

                    if "positions" in result:
                        print(f"  result['positions']: {result['positions']}")
                    for key in ["marginalized", "gl", "ngl", "exact"]:
                        if key in result:
                            print(f"  {key}: {result[key]}")

        except Exception as e:
            debug_counts["prism_exception"] += 1
            print(f"\n[DEBUG prism_exception] idx={idx}")
            print(f"  mutation_str: {mutation_str}")
            print(f"  heavy_mutations: {heavy_mutations}")
            print(f"  light_mutations: {light_mutations}")
            print(f"  wt_heavy[:60]:  {wt_heavy[:60]}")
            print(f"  mut_heavy[:60]: {mut_heavy[:60]}")
            print(f"  wt_light[:60]:  {wt_light[:60]}")
            print(f"  mut_light[:60]: {mut_light[:60]}")
            print(f"  ERROR: {e}")
            marginalized_score = 0.0
            gl_score = 0.0
            ngl_score = 0.0
            exact_score = 0.0

        out["PRISM_marginalized_score"].append(marginalized_score)
        out["PRISM_gl_score"].append(gl_score)
        out["PRISM_ngl_score"].append(ngl_score)
        out["PRISM_exact_score"].append(exact_score)

        if marginalized_score != 0.0:
            non_zero_counts["PRISM_marginalized_score"] += 1
        if gl_score != 0.0:
            non_zero_counts["PRISM_gl_score"] += 1
        if ngl_score != 0.0:
            non_zero_counts["PRISM_ngl_score"] += 1
        if exact_score != 0.0:
            non_zero_counts["PRISM_exact_score"] += 1

    print(f"\n  SANITY CHECKS for PRISM:")
    for k, v in out.items():
        print(f"    {k}: total={len(v)}, non_zero={non_zero_counts[k]}, first_5={v[:5]}")

    print(f"\n  DEBUG BREAKDOWN for PRISM:")
    for k, v in debug_counts.items():
        print(f"    {k}: {v}")

    return out

def evaluate_iglm_model(
    df: pd.DataFrame,
    wt_heavy: str,
    wt_light: str,
    device: str,
    heavy_col: str = "fv_heavy",
    light_col: str = "fv_light",
    mutation_col: str = "Mutations",
    heavy_chain_token: str = "[HEAVY]",
    light_chain_token: str = "[LIGHT]",
    species_token: str = "[HUMAN]",
) -> List[float]:
    """
    Evaluate IgLM on the dataset using a mutation-level pseudo-log-likelihood ratio,
    analogous to `evaluate_antiberty_model`.

    For each mutated position, this computes:

        log p(mut_aa | WT context with that position masked)
      - log p(wt_aa  | WT context with that position masked)

    and sums over all mutated positions in heavy and light chains.

    Notes
    -----
    - WT rows get score 0.0
    - Rows with sequence length mismatches are skipped and scored as 0.0
    - This uses single-position masking via `log_likelihood_no_cls`
    """
    from iglm import IgLM

    print(f"\n{'='*60}")
    print("Loading IgLM...")
    print("=" * 60)

    model = IgLM()

    print(f"  Model loaded on {device}")

    scores = []
    non_zero_count = 0

    for idx in tqdm(range(len(df)), desc="Scoring with IgLM"):
        row = df.iloc[idx]

        mutation_str = row.get(mutation_col, None)
        mut_heavy = row.get(heavy_col, None)
        mut_light = row.get(light_col, None)

        # Basic validation
        if not isinstance(mut_heavy, str) or not isinstance(mut_light, str):
            scores.append(0.0)
            continue

        mut_heavy = mut_heavy.strip()
        mut_light = mut_light.strip()

        if mutation_str == "WT":
            scores.append(0.0)
            continue

        if len(mut_heavy) != len(wt_heavy) or len(mut_light) != len(wt_light):
            scores.append(0.0)
            continue

        heavy_mutations = find_mutation_indices(wt_heavy, mut_heavy)
        light_mutations = find_mutation_indices(wt_light, mut_light)

        if not heavy_mutations and not light_mutations:
            scores.append(0.0)
            continue

        score = 0.0

        # ------------------------------------------------------------
        # Heavy-chain mutations
        # ------------------------------------------------------------
        for pos, wt_aa, mut_aa in heavy_mutations:
            # WT amino acid likelihood in WT background
            wt_ll = log_likelihood_no_cls(
                model,
                wt_heavy,
                heavy_chain_token,
                species_token,
                infill_range=(pos, pos + 1),
            )

            # Mutant amino acid likelihood in the same background:
            # create a temporary heavy sequence with only this residue changed
            temp_heavy = wt_heavy[:pos] + mut_aa + wt_heavy[pos + 1 :]
            mut_ll = log_likelihood_no_cls(
                model,
                temp_heavy,
                heavy_chain_token,
                species_token,
                infill_range=(pos, pos + 1),
            )

            score += (mut_ll - wt_ll)

        # ------------------------------------------------------------
        # Light-chain mutations
        # ------------------------------------------------------------
        for pos, wt_aa, mut_aa in light_mutations:
            wt_ll = log_likelihood_no_cls(
                model,
                wt_light,
                light_chain_token,
                species_token,
                infill_range=(pos, pos + 1),
            )

            temp_light = wt_light[:pos] + mut_aa + wt_light[pos + 1 :]
            mut_ll = log_likelihood_no_cls(
                model,
                temp_light,
                light_chain_token,
                species_token,
                infill_range=(pos, pos + 1),
            )

            score += (mut_ll - wt_ll)

        scores.append(float(score))
        if score != 0.0:
            non_zero_count += 1

    print(f"\n  SANITY CHECKS for IgLM:")
    print(f"    Total rows: {len(scores)}")
    print(f"    Non-zero scores: {non_zero_count}")
    print(f"    First 5 scores: {scores[:5]}")

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    return scores

def evaluate_antiberty_model(
    df: pd.DataFrame,
    wt_heavy: str,
    wt_light: str,
    device: str
) -> List[float]:
    """
    Evaluate AntiBERTy model on the dataset.

    AntiBERTy is an unpaired antibody language model, so heavy and light chains
    are scored separately on their own WT backgrounds. The final score is the
    sum of heavy-chain and light-chain mutation log-likelihood ratios.
    """
    try:
        from antiberty import AntiBERTyRunner
        import torch.nn.functional as F
    except ImportError:
        print("WARNING: antiberty not installed, skipping...")
        return [0.0] * len(df)

    print(f"\n{'='*60}")
    print("Loading AntiBERTy...")
    print('='*60)

    runner = AntiBERTyRunner()
    model = runner.model.to(device)
    tokenizer = runner.tokenizer
    model.eval()

    aa_to_idx = {}
    for aa in "ACDEFGHIKLMNPQRSTVWY":
        token_id = tokenizer.convert_tokens_to_ids(aa)
        aa_to_idx[aa] = token_id

    print(f"  Model loaded on {device}")

    # Tokenize WT heavy and WT light separately once
    wt_heavy_spaced = " ".join(list(wt_heavy))
    wt_light_spaced = " ".join(list(wt_light))

    heavy_tokens = tokenizer(
        wt_heavy_spaced,
        return_tensors="pt",
        add_special_tokens=True,
        truncation=True,
        max_length=1024,
    )
    heavy_input_ids = heavy_tokens["input_ids"].to(device)
    heavy_attention_mask = heavy_tokens["attention_mask"].to(device)

    light_tokens = tokenizer(
        wt_light_spaced,
        return_tensors="pt",
        add_special_tokens=True,
        truncation=True,
        max_length=1024,
    )
    light_input_ids = light_tokens["input_ids"].to(device)
    light_attention_mask = light_tokens["attention_mask"].to(device)

    mask_token_id = tokenizer.mask_token_id

    scores = []
    non_zero_count = 0

    for idx in tqdm(range(len(df)), desc="Scoring with AntiBERTy"):
        row = df.iloc[idx]
        mutation_str = row["Mutations"]

        if mutation_str == "WT":
            scores.append(0.0)
            continue

        mut_heavy = str(row["fv_heavy"]).strip()
        mut_light = str(row["fv_light"]).strip()

        heavy_mutations = find_mutation_indices(wt_heavy, mut_heavy)
        light_mutations = find_mutation_indices(wt_light, mut_light)

        if not heavy_mutations and not light_mutations:
            scores.append(0.0)
            continue

        score = 0.0

        # ------------------------------------------------------------
        # Score heavy-chain mutations on WT heavy background
        # ------------------------------------------------------------
        for pos, wt_aa, mut_aa in heavy_mutations:
            token_pos = pos + 1  # +1 for [CLS]
            wt_id = aa_to_idx.get(wt_aa)
            mut_id = aa_to_idx.get(mut_aa)

            if wt_id is None or mut_id is None:
                continue

            masked_input_ids = heavy_input_ids.clone()
            masked_input_ids[0, token_pos] = mask_token_id

            with torch.no_grad():
                outputs = model(
                    input_ids=masked_input_ids,
                    attention_mask=heavy_attention_mask
                )
                if hasattr(outputs, "logits") and outputs.logits is not None:
                    logits = outputs.logits
                elif hasattr(outputs, "prediction_logits"):
                    logits = outputs.prediction_logits
                else:
                    continue

                log_probs = F.log_softmax(logits, dim=-1)

            score += (
                log_probs[0, token_pos, mut_id] - log_probs[0, token_pos, wt_id]
            ).item()

        # ------------------------------------------------------------
        # Score light-chain mutations on WT light background
        # ------------------------------------------------------------
        for pos, wt_aa, mut_aa in light_mutations:
            token_pos = pos + 1  # +1 for [CLS]
            wt_id = aa_to_idx.get(wt_aa)
            mut_id = aa_to_idx.get(mut_aa)

            if wt_id is None or mut_id is None:
                continue

            masked_input_ids = light_input_ids.clone()
            masked_input_ids[0, token_pos] = mask_token_id

            with torch.no_grad():
                outputs = model(
                    input_ids=masked_input_ids,
                    attention_mask=light_attention_mask
                )
                if hasattr(outputs, "logits") and outputs.logits is not None:
                    logits = outputs.logits
                elif hasattr(outputs, "prediction_logits"):
                    logits = outputs.prediction_logits
                else:
                    continue

                log_probs = F.log_softmax(logits, dim=-1)

            score += (
                log_probs[0, token_pos, mut_id] - log_probs[0, token_pos, wt_id]
            ).item()

        scores.append(score)
        if score != 0.0:
            non_zero_count += 1

    print(f"\n  SANITY CHECKS for AntiBERTy:")
    print(f"    Total rows: {len(scores)}")
    print(f"    Non-zero scores: {non_zero_count}")
    print(f"    First 5 scores: {scores[:5]}")

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    return scores

def get_mutation_score_esm2(
    model,
    tokenizer,
    wt_seq: str,
    mut_seq: str,
    device: str,
    aa_to_idx: Dict[str, int]
) -> float:
    """
    Calculate LLR score for mutations using true masked marginal method.

    For each mutation position, replaces that position with <mask> token,
    runs a forward pass, and computes:
        Score = sum over mutations of [log P(mut_aa | masked_context) - log P(wt_aa | masked_context)]

    Args:
        model: ESM-2 model
        tokenizer: ESM-2 tokenizer
        wt_seq: Wild-type sequence
        mut_seq: Mutant sequence
        device: torch device
        aa_to_idx: Amino acid to token ID mapping

    Returns:
        LLR score (float)
    """
    # Find mutations by comparing sequences
    mutations = find_mutation_indices(wt_seq, mut_seq)

    if not mutations:
        return 0.0  # WT or no valid mutations

    # Tokenize WT sequence (used as context)
    inputs = tokenizer(
        wt_seq,
        return_tensors="pt",
        add_special_tokens=True,
        truncation=True,
        max_length=1024
    )
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    mask_token_id = tokenizer.mask_token_id

    score = 0.0
    for pos, wt_aa, mut_aa in mutations:
        # Position in tokenized sequence: +1 for [CLS] token
        token_pos = pos + 1

        # Get token IDs
        wt_id = aa_to_idx.get(wt_aa)
        mut_id = aa_to_idx.get(mut_aa)

        if wt_id is None or mut_id is None:
            continue  # Skip unknown amino acids

        # Create masked input: replace this position with <mask>
        masked_input_ids = input_ids.clone()
        masked_input_ids[0, token_pos] = mask_token_id

        # Forward pass with masked input
        with torch.no_grad():
            outputs = model(input_ids=masked_input_ids, attention_mask=attention_mask)
            log_probs = F.log_softmax(outputs.logits, dim=-1)

        # LLR: log P(Mut | masked) - log P(WT | masked)
        score += (log_probs[0, token_pos, mut_id] - log_probs[0, token_pos, wt_id]).item()

    return score

def evaluate_esm2_model(
    model_name: str,
    model_id: str,
    df: pd.DataFrame,
    wt_heavy: str,
    wt_light: str,
    device: str
) -> List[float]:
    """
    Evaluate an ESM-2 model on the dataset.

    This function is completely independent to prevent variable reuse bugs.

    Args:
        model_name: Human-readable model name
        model_id: HuggingFace model ID
        df: DataFrame with sequences
        wt_heavy: Wild-type heavy chain
        wt_light: Wild-type light chain
        device: torch device

    Returns:
        List of scores for each row
    """
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    print(f"\n{'='*60}")
    print(f"Loading {model_name} ({model_id})...")
    print('='*60)

    # Load model and tokenizer (fresh instances)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForMaskedLM.from_pretrained(model_id)
    model.to(device)
    model.eval()

    # Build AA vocabulary mapping
    aa_to_idx = {}
    for aa in 'ACDEFGHIKLMNPQRSTVWY':
        token_id = tokenizer.convert_tokens_to_ids(aa)
        aa_to_idx[aa] = token_id

    print(f"  Model loaded on {device}")
    print(f"  Vocab size: {len(tokenizer)}")

    # Concatenate WT sequence
    wt_seq = wt_heavy + wt_light

    # Initialize scores list (fresh)
    scores = []
    non_zero_count = 0

    # DEBUG: Track why scores are zero
    debug_stats = {
        'wt_count': 0,
        'length_mismatch': 0,
        'no_mutations_found': 0,
        'scored': 0
    }

    # Process each row
    for idx in tqdm(range(len(df)), desc=f"Scoring with {model_name}"):
        row = df.iloc[idx]
        mutation_str = row['Mutations']

        # Handle WT
        if mutation_str == 'WT':
            scores.append(0.0)
            debug_stats['wt_count'] += 1
            continue

        # Concatenate mutant sequence
        mut_seq = row['fv_heavy'] + row['fv_light']

        # DEBUG: Check for length mismatch
        if len(wt_seq) != len(mut_seq):
            debug_stats['length_mismatch'] += 1
            if debug_stats['length_mismatch'] <= 3:  # Print first 3 examples
                print(f"\n  DEBUG: Length mismatch at idx={idx}, Mutations='{mutation_str}'")
                print(f"    WT length: {len(wt_seq)} (heavy={len(wt_heavy)}, light={len(wt_light)})")
                print(f"    Mut length: {len(mut_seq)} (heavy={len(row['fv_heavy'])}, light={len(row['fv_light'])})")
            scores.append(0.0)
            continue

        # DEBUG: Check if mutations are found
        mutations = find_mutation_indices(wt_seq, mut_seq)
        if not mutations:
            debug_stats['no_mutations_found'] += 1
            if debug_stats['no_mutations_found'] <= 3:  # Print first 3 examples
                print(f"\n  DEBUG: No mutations found at idx={idx}, Mutations='{mutation_str}'")
                print(f"    WT seq[:50]: {wt_seq[:50]}...")
                print(f"    Mut seq[:50]: {mut_seq[:50]}...")
                # Check if sequences are identical
                if wt_seq == mut_seq:
                    print(f"    -> Sequences are IDENTICAL!")
            scores.append(0.0)
            continue

        # Score using dynamic mutation finding
        debug_stats['scored'] += 1
        score = get_mutation_score_esm2(
            model, tokenizer, wt_seq, mut_seq, device, aa_to_idx
        )
        scores.append(score)

        if score != 0.0:
            non_zero_count += 1

    # Sanity checks
    print(f"\n  SANITY CHECKS for {model_name}:")
    print(f"    Total rows: {len(scores)}")
    print(f"    Non-zero scores: {non_zero_count}")
    print(f"    First 5 scores: {scores[:5]}")
    print(f"\n  DEBUG BREAKDOWN:")
    print(f"    WT rows (skipped): {debug_stats['wt_count']}")
    print(f"    Length mismatches: {debug_stats['length_mismatch']}")
    print(f"    No mutations found: {debug_stats['no_mutations_found']}")
    print(f"    Actually scored: {debug_stats['scored']}")

    # Clean up GPU memory
    del model
    if device == 'cuda':
        torch.cuda.empty_cache()

    return scores

def evaluate_ablang2_model(
    df: pd.DataFrame,
    wt_heavy: str,
    wt_light: str,
    device: str
) -> List[float]:
    """
    Evaluate AbLang2 paired model on the dataset.

    AbLang2 paired tokenization uses:
        <HEAVY>|<LIGHT>

    Score = sum over mutations of:
        log P(mut_aa | masked WT context) - log P(wt_aa | masked WT context)
    """
    try:
        import ablang2
    except ImportError:
        print("WARNING: ablang2 not installed, skipping...")
        return [0.0] * len(df)

    print(f"\n{'='*60}")
    print("Loading AbLang2 (paired mode)...")
    print("=" * 60)

    ablang = ablang2.pretrained(
        model_to_use="ablang2-paired",
        random_init=False,
        device=device,
    )
    model = ablang.AbLang
    tokenizer = ablang.tokenizer
    model.eval()

    # Confirmed tokenizer mappings from ABtokenizer
    aa_to_idx = dict(tokenizer.aa_to_token)
    mask_id = tokenizer.mask_token

    print(f"  Model loaded on {device}")
    print(f"  mask_id: {mask_id}")
    print(f"  aa_to_idx: {aa_to_idx}")

    wt_heavy = str(wt_heavy).replace(" ", "").strip().upper()
    wt_light = str(wt_light).replace(" ", "").strip().upper()

    # AbLang2 paired formatting
    paired_seq = f"<{wt_heavy}>|<{wt_light}>"
    tokenized_wt = tokenizer([paired_seq], pad=True, w_extra_tkns=False, device=device)

    # If tokenizer output is dict-like, pull out the token tensor but keep the
    # original container structure for masked forward passes.
    if isinstance(tokenized_wt, torch.Tensor):
        wt_input_ids = tokenized_wt
    elif isinstance(tokenized_wt, dict):
        if "input_ids" in tokenized_wt:
            wt_input_ids = tokenized_wt["input_ids"]
        elif "tokens" in tokenized_wt:
            wt_input_ids = tokenized_wt["tokens"]
        elif "ids" in tokenized_wt:
            wt_input_ids = tokenized_wt["ids"]
        else:
            raise KeyError("Could not find token ids in AbLang2 tokenizer output.")
    else:
        raise TypeError(f"Unsupported tokenizer output type: {type(tokenized_wt)}")

    scores = []
    non_zero_count = 0

    for idx in tqdm(range(len(df)), desc="Scoring with AbLang2"):
        row = df.iloc[idx]
        mutation_str = row["Mutations"]

        if mutation_str == "WT":
            scores.append(0.0)
            continue

        mut_heavy = str(row["fv_heavy"]).replace(" ", "").strip().upper()
        mut_light = str(row["fv_light"]).replace(" ", "").strip().upper()

        heavy_muts = find_mutation_indices(wt_heavy, mut_heavy)
        light_muts = find_mutation_indices(wt_light, mut_light)

        if not heavy_muts and not light_muts:
            scores.append(0.0)
            continue

        score = 0.0

        # ------------------------------------------------------------
        # Heavy-chain mutations
        # Token layout for <HEAVY>|<LIGHT> is:
        #   [start] H... [end] [sep] [start] L... [end]
        # So heavy residue i is at token position i + 1
        # ------------------------------------------------------------
        for pos, wt_aa, mut_aa in heavy_muts:
            wt_id = aa_to_idx.get(wt_aa)
            mut_id = aa_to_idx.get(mut_aa)
            if wt_id is None or mut_id is None:
                continue

            token_pos = pos + 1

            if isinstance(tokenized_wt, torch.Tensor):
                masked_toks = wt_input_ids.clone()
                masked_toks[0, token_pos] = mask_id
            else:
                masked_toks = dict(tokenized_wt)
                masked_ids = wt_input_ids.clone()
                masked_ids[0, token_pos] = mask_id

                if "input_ids" in masked_toks:
                    masked_toks["input_ids"] = masked_ids
                elif "tokens" in masked_toks:
                    masked_toks["tokens"] = masked_ids
                elif "ids" in masked_toks:
                    masked_toks["ids"] = masked_ids
                else:
                    masked_toks = masked_ids

            with torch.no_grad():
                outputs = model(masked_toks)
                log_probs = F.log_softmax(outputs, dim=-1)

            score += (log_probs[0, token_pos, mut_id] - log_probs[0, token_pos, wt_id]).item()

        # ------------------------------------------------------------
        # Light-chain mutations
        # Token layout:
        #   [start] H... [end] [sep] [start] L... [end]
        # So first light residue starts at len(H) + 4
        # ------------------------------------------------------------
        for pos, wt_aa, mut_aa in light_muts:
            wt_id = aa_to_idx.get(wt_aa)
            mut_id = aa_to_idx.get(mut_aa)
            if wt_id is None or mut_id is None:
                continue

            token_pos = len(wt_heavy) + 4 + pos

            if isinstance(tokenized_wt, torch.Tensor):
                masked_toks = wt_input_ids.clone()
                masked_toks[0, token_pos] = mask_id
            else:
                masked_toks = dict(tokenized_wt)
                masked_ids = wt_input_ids.clone()
                masked_ids[0, token_pos] = mask_id

                if "input_ids" in masked_toks:
                    masked_toks["input_ids"] = masked_ids
                elif "tokens" in masked_toks:
                    masked_toks["tokens"] = masked_ids
                elif "ids" in masked_toks:
                    masked_toks["ids"] = masked_ids
                else:
                    masked_toks = masked_ids

            with torch.no_grad():
                outputs = model(masked_toks)
                log_probs = F.log_softmax(outputs, dim=-1)

            score += (log_probs[0, token_pos, mut_id] - log_probs[0, token_pos, wt_id]).item()

        scores.append(score)
        if score != 0.0:
            non_zero_count += 1

    print(f"\n  SANITY CHECKS for AbLang2:")
    print(f"    Total rows: {len(scores)}")
    print(f"    Non-zero scores: {non_zero_count}")
    print(f"    First 5 scores: {scores[:5]}")

    return scores

def evaluate_sapiens_model(
    df: pd.DataFrame,
    wt_heavy: str,
    wt_light: str,
    device: str
) -> List[float]:
    """
    Evaluate Sapiens model on the dataset.

    Sapiens uses separate models for heavy and light chains.
    Note: Forced to CPU due to stability issues.
    """
    from transformers import RobertaForMaskedLM, RobertaTokenizer

    # Force CPU for Sapiens
    device = 'cpu'
    max_seq_len = 143  # Sapiens limit

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

    # Build AA vocabulary mapping
    aa_to_idx = {}
    for aa in 'ACDEFGHIKLMNPQRSTVWY':
        token_id = tokenizer.convert_tokens_to_ids(aa)
        aa_to_idx[aa] = token_id

    print(f"  Models loaded on {device} (forced CPU for stability)")

    scores = []
    non_zero_count = 0
    skipped_length = 0

    for idx in tqdm(range(len(df)), desc="Scoring with Sapiens"):
        row = df.iloc[idx]
        mutation_str = row['Mutations']

        if mutation_str == 'WT':
            scores.append(0.0)
            continue

        mut_heavy = row['fv_heavy']
        mut_light = row['fv_light']

        heavy_muts = find_mutation_indices(wt_heavy, mut_heavy)
        light_muts = find_mutation_indices(wt_light, mut_light)

        if not heavy_muts and not light_muts:
            scores.append(0.0)
            continue

        score = 0.0

        mask_token_id = tokenizer.mask_token_id

        # Score heavy chain mutations (one forward pass per position)
        if heavy_muts and len(wt_heavy) <= max_seq_len:
            tokens = tokenizer(
                wt_heavy,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=max_seq_len + 2
            )
            input_ids = tokens['input_ids'].to(device)
            attention_mask = tokens['attention_mask'].to(device)

            for pos, wt_aa, mut_aa in heavy_muts:
                token_pos = pos + 1
                wt_id = aa_to_idx.get(wt_aa)
                mut_id = aa_to_idx.get(mut_aa)
                if wt_id is None or mut_id is None:
                    continue

                masked_input_ids = input_ids.clone()
                masked_input_ids[0, token_pos] = mask_token_id

                with torch.no_grad():
                    outputs = heavy_model(input_ids=masked_input_ids, attention_mask=attention_mask)
                    log_probs = F.log_softmax(outputs.logits, dim=-1)

                score += (log_probs[0, token_pos, mut_id] - log_probs[0, token_pos, wt_id]).item()
        elif heavy_muts:
            skipped_length += 1

        # Score light chain mutations (one forward pass per position)
        if light_muts and len(wt_light) <= max_seq_len:
            tokens = tokenizer(
                wt_light,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=max_seq_len + 2
            )
            input_ids = tokens['input_ids'].to(device)
            attention_mask = tokens['attention_mask'].to(device)

            for pos, wt_aa, mut_aa in light_muts:
                token_pos = pos + 1
                wt_id = aa_to_idx.get(wt_aa)
                mut_id = aa_to_idx.get(mut_aa)
                if wt_id is None or mut_id is None:
                    continue

                masked_input_ids = input_ids.clone()
                masked_input_ids[0, token_pos] = mask_token_id

                with torch.no_grad():
                    outputs = light_model(input_ids=masked_input_ids, attention_mask=attention_mask)
                    log_probs = F.log_softmax(outputs.logits, dim=-1)

                score += (log_probs[0, token_pos, mut_id] - log_probs[0, token_pos, wt_id]).item()
        elif light_muts:
            skipped_length += 1

        scores.append(score)
        if score != 0.0:
            non_zero_count += 1

    print(f"\n  SANITY CHECKS for Sapiens:")
    print(f"    Total rows: {len(scores)}")
    print(f"    Non-zero scores: {non_zero_count}")
    print(f"    Skipped (length): {skipped_length}")
    print(f"    First 5 scores: {scores[:5]}")

    del heavy_model, light_model

    return scores




def mask_span(seq: Sequence[str], start: int, end: int, append_span: bool = False):
    """
    Replace seq[start:end] with a single [MASK] token.
    Optionally append the removed span to the end, after [SEP].
    """
    masked_seq = list(seq[:start]) + ["[MASK]"] + list(seq[end:]) + ["[SEP]"]
    if append_span:
        masked_seq += list(seq[start:end])
    return masked_seq


def log_likelihood_no_cls(
    model,
    sequence,
    chain_token,
    species_token,
    infill_range=None,
):
    sequence = list(sequence)

    if infill_range is not None:
        sequence = mask_span(
            sequence,
            infill_range[0],
            infill_range[1],
            append_span=True,
        )

    token_list = [chain_token, species_token] + sequence

    if infill_range is not None:
        token_list += [model.tokenizer.cls_token]
    else:
        token_list += [model.tokenizer.sep_token]

    token_ids = model.tokenizer.convert_tokens_to_ids(token_list)

    for tok, tok_id in zip(token_list, token_ids):
        if tok_id == model.tokenizer.unk_token_id:
            print(f"Unknown token detected: {repr(tok)}")

    token_seq = torch.tensor([token_ids], dtype=torch.long, device=model.device)

    assert (token_seq != model.tokenizer.unk_token_id).all(), \
        "Unrecognized token supplied in starting tokens"

    if infill_range is not None:
        eval_start = np.nonzero(
            token_seq[0] == model.tokenizer.sep_token_id
        )[0].item()
    else:
        eval_start = 1

    logits = model.model(token_seq).logits

    if infill_range is not None:
        shift_logits = logits[..., eval_start:-2, :].contiguous()
        shift_labels = token_seq[..., eval_start + 1:-1].contiguous().long()
    else:
        shift_logits = logits[..., eval_start:-1, :].contiguous()
        shift_labels = token_seq[..., eval_start + 1:].contiguous().long()

    nll = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="mean",
    )

    return -nll.item()

def debug_tokenize_sequence(model, sequence, chain_token, species_token):
    token_list = [chain_token, species_token] + list(sequence) + [model.tokenizer.sep_token]
    token_ids = model.tokenizer.convert_tokens_to_ids(token_list)

    bad = [(tok, tok_id) for tok, tok_id in zip(token_list, token_ids)
           if tok_id == model.tokenizer.unk_token_id]

    if bad:
        print("Unknown tokens found:")
        for tok, tok_id in bad:
            print(f"  token={repr(tok)}, id={tok_id}")
    else:
        print("No unknown tokens found.")

    return token_list, token_ids

def compute_iglm_perplexity(
    model,
    heavy: str,
    light: str,
    heavy_chain_token: str = "[HEAVY]",
    light_chain_token: str = "[LIGHT]",
    species_token: str = "[HUMAN]",
) -> float:
    """
    Compute combined heavy+light perplexity from mean per-chain log-likelihoods.
    """
    heavy_ll = model.log_likelihood(heavy, heavy_chain_token, species_token)
    light_ll = model.log_likelihood(light, light_chain_token, species_token)

    total_tokens = (len(heavy) + 1) + (len(light) + 1)
    total_ll = heavy_ll * (len(heavy) + 1) + light_ll * (len(light) + 1)
    total_ll /= total_tokens

    return float(np.exp(-total_ll))


def compute_iglm_pseudo_perplexity(
    model,
    heavy: str,
    light: str,
    heavy_chain_token: str = "[HEAVY]",
    light_chain_token: str = "[LIGHT]",
    species_token: str = "[HUMAN]",
) -> float:
    """
    Compute combined heavy+light pseudo-perplexity by single-position masking.
    """
    heavy_scores = [
        log_likelihood_no_cls(
            model,
            heavy,
            heavy_chain_token,
            species_token,
            infill_range=(j, j + 1),
        )
        for j in range(len(heavy))
    ]

    light_scores = [
        log_likelihood_no_cls(
            model,
            light,
            light_chain_token,
            species_token,
            infill_range=(j, j + 1),
        )
        for j in range(len(light))
    ]

    sum_logs = sum(heavy_scores) + sum(light_scores)
    total_len = len(heavy) + len(light)

    return float(np.exp(-sum_logs / total_len))


def add_iglm_scores_to_dataframe(
    df,
    model,
    heavy_col: str,
    light_col: str,
    perplexity_col: str = "IgLM_Perplexity",
    pseudo_perplexity_col: str = "IgLM_PseudoPerplexity",
    compute_perplexity: bool = True,
    compute_pseudo_perplexity: bool = True,
    show_progress: bool = True,
):
    """
    Add IgLM perplexity and/or pseudo-perplexity columns to a dataframe.
    """
    iterator = df.iterrows()
    if show_progress:
        iterator = tqdm(iterator, total=len(df))

    for i, row in iterator:
        heavy = row[heavy_col]
        light = row[light_col]

        if not isinstance(heavy, str) or not isinstance(light, str):
            raise ValueError(
                f"Row {i} has non-string sequence(s): "
                f"{heavy_col}={type(heavy)}, {light_col}={type(light)}"
            )

        if compute_perplexity:
            df.at[i, perplexity_col] = compute_iglm_perplexity(model, heavy, light)

        if compute_pseudo_perplexity:
            df.at[i, pseudo_perplexity_col] = compute_iglm_pseudo_perplexity(model, heavy, light)

    return df

def find_mutation_indices(wt_seq: str, mut_seq: str) -> List[Tuple[int, str, str]]:
    """
    Dynamically find mutation positions by comparing WT and mutant sequences.

    This is the CRITICAL function that prevents bugs from string parsing.

    Args:
        wt_seq: Wild-type sequence
        mut_seq: Mutant sequence

    Returns:
        List of tuples: (position, wt_aa, mut_aa) for each difference
    """
    if len(wt_seq) != len(mut_seq):
        # Length mismatch - could be insertion/deletion, skip
        return []

    mutations = []
    for i, (wt_aa, mut_aa) in enumerate(zip(wt_seq, mut_seq)):
        if wt_aa != mut_aa:
            mutations.append((i, wt_aa, mut_aa))

    return mutations
