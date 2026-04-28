#!/usr/bin/env python3
"""
FLAb2 Baseline Model Evaluation: Per-Protein Spearman ρ using LLR scoring.

Baselines: ESM2-35M, ESM2-650M, AbLang2, AntiBERTy, Sapiens
Scoring: Masked marginal LLR = Σ over mutations of
    [log P(mut_aa | WT with pos masked) - log P(wt_aa | WT with pos masked)]

Tokenization matches each library's own pseudo_log_likelihood:
- ESM2: <cls>...<eos>, special token ids suppressed in logits.
- AbLang2: "<HEAVY>|<LIGHT>" format with library vocab (M=1, A=14, ...).
- AntiBERTy: single-chain; heavy and light tokenized separately.
- Sapiens: per-chain max_position_embeddings (VH=146, VL=130).

Usage:
    CUDA_VISIBLE_DEVICES=6 python evaluate_flab2_baselines.py
    CUDA_VISIBLE_DEVICES=6 python evaluate_flab2_baselines.py --models esm2_35m esm2_650m
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from tqdm.auto import tqdm

import warnings
warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[3]

# Same exclusions as evaluate_flab2_binding.py
EXCLUDE_SOURCES = {
    "koenig2017mutational_kd_g6.csv",
    "shanehsazzadeh2023unlocking_zerokd_trastuzumab.csv",
    "phillips2021binding_cr9114_h3_kd.csv",
    "phillips2021binding_cr9114_h1_kd.csv",
    "AbRank_dataset.csv.zip",
}

MIN_VARIANTS = 15


def load_flab2_binding():
    """Load and filter FLAb2 binding data (same as evaluate_flab2_binding.py)."""
    df = pd.read_parquet(REPO_ROOT / "data" / "FLAb" / "flab2_binding.parquet")
    df = df[df["light"].notna()].copy()
    df = df[~df["assay_type"].isin(["bind/no bind", "predicted Kd"])]
    df = df[~df["source_file"].isin(EXCLUDE_SOURCES)]
    counts = df["source_file"].value_counts()
    valid_sources = counts[counts >= MIN_VARIANTS].index
    df = df[df["source_file"].isin(valid_sources)].copy()
    return df


def identify_wt(group_df):
    """Identify WT as the most common heavy+light sequence."""
    heavy_counts = group_df["heavy"].value_counts()
    wt_heavy = heavy_counts.index[0]
    wt_rows = group_df[group_df["heavy"] == wt_heavy]
    if len(wt_rows) > 1:
        light_counts = wt_rows["light"].value_counts()
        wt_light = light_counts.index[0]
    else:
        wt_light = wt_rows.iloc[0]["light"]
    return wt_heavy, wt_light


def find_mutations(wt_seq: str, mut_seq: str) -> Optional[List[Tuple[int, str, str]]]:
    """Find mutation positions by comparing WT and mutant sequences."""
    if len(wt_seq) != len(mut_seq):
        return None
    mutations = []
    for i, (w, m) in enumerate(zip(wt_seq, mut_seq)):
        if w != m:
            mutations.append((i, w, m))
    return mutations if mutations else None


# =============================================================================
# Model Scoring Functions
# =============================================================================

def score_esm2(model, tokenizer, aa_to_idx, wt_seq, mutations, device):
    """Masked-marginal LLR with ESM2 on concatenated heavy+light WT context."""
    if not mutations:
        return 0.0
    inputs = tokenizer(wt_seq, return_tensors="pt", add_special_tokens=True,
                       truncation=True, max_length=1024)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    mask_token_id = tokenizer.mask_token_id
    all_special_ids = tokenizer.all_special_ids

    score = 0.0
    for pos, wt_aa, mut_aa in mutations:
        wt_id = aa_to_idx.get(wt_aa)
        mut_id = aa_to_idx.get(mut_aa)
        if wt_id is None or mut_id is None:
            continue
        token_pos = pos + 1  # +1 for <cls>
        masked = input_ids.clone()
        masked[0, token_pos] = mask_token_id
        with torch.no_grad():
            outputs = model(input_ids=masked, attention_mask=attention_mask)
            logits = outputs.logits.clone()
            logits[:, :, all_special_ids] = -float("inf")
            log_probs = F.log_softmax(logits, dim=-1)
        score += (log_probs[0, token_pos, mut_id] - log_probs[0, token_pos, wt_id]).item()
    return score


def score_ablang2(model, tokenizer, aa_to_idx, wt_heavy, wt_light,
                  heavy_muts, light_muts, device):
    """Masked-marginal LLR with AbLang2 paired (<HEAVY>|<LIGHT> format)."""
    if not heavy_muts and not light_muts:
        return 0.0
    wt_library_str = f"<{wt_heavy}>|<{wt_light}>"
    tokenized_wt = tokenizer([wt_library_str], pad=True, w_extra_tkns=False, device=device)
    mask_id = tokenizer.mask_token
    special_ids = list(tokenizer.all_special_tokens)
    heavy_offset = 1
    light_offset = len(wt_heavy) + 4

    score = 0.0
    for muts, offset in ((heavy_muts, heavy_offset), (light_muts, light_offset)):
        for pos, wt_aa, mut_aa in muts:
            wt_id = aa_to_idx.get(wt_aa)
            mut_id = aa_to_idx.get(mut_aa)
            if wt_id is None or mut_id is None:
                continue
            token_pos = offset + pos
            masked = tokenized_wt.clone()
            masked[0, token_pos] = mask_id
            with torch.no_grad():
                logits = model(masked)
                logits[:, :, special_ids] = -float("inf")
                log_probs = F.log_softmax(logits, dim=-1)
            score += (log_probs[0, token_pos, mut_id] - log_probs[0, token_pos, wt_id]).item()
    return score


def score_antiberty(model, tokenizer, aa_to_idx, wt_heavy, wt_light,
                    heavy_muts, light_muts, device):
    """Masked-marginal LLR with AntiBERTy (heavy/light scored separately)."""
    if not heavy_muts and not light_muts:
        return 0.0
    mask_token_id = tokenizer.mask_token_id
    all_special_ids = tokenizer.all_special_ids

    def _score_chain(chain_seq, chain_muts):
        if not chain_muts:
            return 0.0
        spaced = " ".join(list(chain_seq))
        tokens = tokenizer(spaced, return_tensors="pt", add_special_tokens=True,
                           truncation=True, max_length=1024)
        input_ids = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)
        s = 0.0
        for pos, wt_aa, mut_aa in chain_muts:
            wt_id = aa_to_idx.get(wt_aa)
            mut_id = aa_to_idx.get(mut_aa)
            if wt_id is None or mut_id is None:
                continue
            token_pos = pos + 1  # +1 for [CLS]
            masked = input_ids.clone()
            masked[0, token_pos] = mask_token_id
            with torch.no_grad():
                outputs = model(input_ids=masked, attention_mask=attention_mask)
                if hasattr(outputs, "prediction_logits") and outputs.prediction_logits is not None:
                    logits = outputs.prediction_logits
                elif hasattr(outputs, "logits") and outputs.logits is not None:
                    logits = outputs.logits
                else:
                    continue
                logits = logits.clone()
                logits[:, :, all_special_ids] = -float("inf")
                log_probs = F.log_softmax(logits, dim=-1)
            s += (log_probs[0, token_pos, mut_id] - log_probs[0, token_pos, wt_id]).item()
        return s

    return _score_chain(wt_heavy, heavy_muts) + _score_chain(wt_light, light_muts)


def score_sapiens(heavy_model, light_model, tokenizer, aa_to_idx,
                  wt_heavy, wt_light, heavy_muts, light_muts, device,
                  heavy_max_residues=None, light_max_residues=None):
    """Masked-marginal LLR with Sapiens (per-chain max_position_embeddings)."""
    if not heavy_muts and not light_muts:
        return 0.0
    mask_token_id = tokenizer.mask_token_id
    all_special_ids = tokenizer.all_special_ids

    def _score_chain(chain_model, chain_seq, chain_muts, max_residues):
        if not chain_muts or len(chain_seq) > max_residues:
            return 0.0
        tokens = tokenizer(chain_seq, return_tensors="pt", padding=False,
                           truncation=True, max_length=max_residues + 2)
        input_ids = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)
        s = 0.0
        for pos, wt_aa, mut_aa in chain_muts:
            wt_id = aa_to_idx.get(wt_aa)
            mut_id = aa_to_idx.get(mut_aa)
            if wt_id is None or mut_id is None:
                continue
            token_pos = pos + 1  # +1 for <s>
            masked = input_ids.clone()
            masked[0, token_pos] = mask_token_id
            with torch.no_grad():
                outputs = chain_model(input_ids=masked, attention_mask=attention_mask)
                logits = outputs.logits.clone()
                logits[:, :, all_special_ids] = -float("inf")
                log_probs = F.log_softmax(logits, dim=-1)
            s += (log_probs[0, token_pos, mut_id] - log_probs[0, token_pos, wt_id]).item()
        return s

    return (_score_chain(heavy_model, wt_heavy, heavy_muts, heavy_max_residues)
            + _score_chain(light_model, wt_light, light_muts, light_max_residues))


# =============================================================================
# Per-Protein Evaluation
# =============================================================================

def evaluate_protein_baseline(score_fn, group_df, wt_heavy, wt_light):
    """Evaluate a single protein with a baseline model.

    score_fn(wt_heavy, wt_light, heavy_muts, light_muts) -> float
    """
    scores = []
    fitness_values = []

    for _, row in group_df.iterrows():
        var_heavy = row["heavy"]
        var_light = row["light"]

        # Find mutations in heavy and light separately
        heavy_muts = find_mutations(wt_heavy, var_heavy)
        light_muts = find_mutations(wt_light, var_light)

        # Skip length mismatches
        if heavy_muts is None and len(wt_heavy) != len(var_heavy):
            continue
        if light_muts is None and len(wt_light) != len(var_light):
            continue

        h_muts = heavy_muts or []
        l_muts = light_muts or []

        if not h_muts and not l_muts:
            continue  # WT itself

        s = score_fn(wt_heavy, wt_light, h_muts, l_muts)
        scores.append(s)
        fitness_values.append(row["fitness"])

    if len(scores) < 10:
        return None

    scores = np.array(scores)
    fitness = np.array(fitness_values)
    valid = np.isfinite(scores) & np.isfinite(fitness)
    if valid.sum() < 10:
        return None

    rho, _ = spearmanr(scores[valid], fitness[valid])
    return {"llr_score": rho, "n_variants": int(valid.sum())}


# =============================================================================
# Model Loaders
# =============================================================================

def load_esm2(model_id, device):
    """Load ESM2 model and return score function."""
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForMaskedLM.from_pretrained(model_id)
    model.to(device)
    model.eval()

    aa_to_idx = {}
    for aa in "ACDEFGHIKLMNPQRSTVWY":
        aa_to_idx[aa] = tokenizer.convert_tokens_to_ids(aa)

    def score_fn(wt_heavy, wt_light, heavy_muts, light_muts):
        wt_seq = wt_heavy + wt_light
        # Combine mutations, adjusting light chain positions
        all_muts = list(heavy_muts)
        offset = len(wt_heavy)
        for pos, w, m in light_muts:
            all_muts.append((offset + pos, w, m))
        if not all_muts:
            return 0.0
        return score_esm2(model, tokenizer, aa_to_idx, wt_seq, all_muts, device)

    return score_fn, model


def load_ablang2(device):
    """Load AbLang2 paired model and return score function."""
    import ablang2

    ablang = ablang2.pretrained(model_to_use="ablang2-paired", random_init=False, device=device)
    model = ablang.AbLang
    tokenizer = ablang.tokenizer
    model.eval()

    aa_to_idx = {aa: tokenizer.aa_to_token[aa] for aa in "ACDEFGHIKLMNPQRSTVWY"}

    def score_fn(wt_heavy, wt_light, heavy_muts, light_muts):
        return score_ablang2(model, tokenizer, aa_to_idx, wt_heavy, wt_light,
                             heavy_muts, light_muts, device)

    return score_fn, model


def load_antiberty(device):
    """Load AntiBERTy model and return score function."""
    from antiberty import AntiBERTyRunner

    runner = AntiBERTyRunner()
    model = runner.model.to(device)
    tokenizer = runner.tokenizer
    model.eval()

    aa_to_idx = {}
    for aa in "ACDEFGHIKLMNPQRSTVWY":
        aa_to_idx[aa] = tokenizer.convert_tokens_to_ids(aa)

    def score_fn(wt_heavy, wt_light, heavy_muts, light_muts):
        return score_antiberty(model, tokenizer, aa_to_idx, wt_heavy, wt_light,
                               heavy_muts, light_muts, device)

    return score_fn, model


def load_sapiens():
    """Load Sapiens models (forced CPU) and return score function."""
    from transformers import RobertaForMaskedLM, RobertaTokenizer

    device = "cpu"
    tokenizer = RobertaTokenizer.from_pretrained("prihodad/biophi-sapiens1-tokenizer")
    heavy_model = RobertaForMaskedLM.from_pretrained("prihodad/biophi-sapiens1-vh")
    light_model = RobertaForMaskedLM.from_pretrained("prihodad/biophi-sapiens1-vl")
    heavy_model.to(device).eval()
    light_model.to(device).eval()

    def _chain_max_residues(cfg):
        return cfg.max_position_embeddings - cfg.pad_token_id - 1 - 2

    heavy_max = _chain_max_residues(heavy_model.config)
    light_max = _chain_max_residues(light_model.config)

    aa_to_idx = {aa: tokenizer.convert_tokens_to_ids(aa) for aa in "ACDEFGHIKLMNPQRSTVWY"}

    def score_fn(wt_heavy, wt_light, heavy_muts, light_muts):
        return score_sapiens(heavy_model, light_model, tokenizer, aa_to_idx,
                             wt_heavy, wt_light, heavy_muts, light_muts, device,
                             heavy_max_residues=heavy_max, light_max_residues=light_max)

    return score_fn, (heavy_model, light_model)


# =============================================================================
# Main
# =============================================================================

ALL_MODELS = ["esm2_35m", "esm2_650m", "ablang2", "antiberty", "sapiens"]

MODEL_IDS = {
    "esm2_35m": "facebook/esm2_t12_35M_UR50D",
    "esm2_650m": "facebook/esm2_t33_650M_UR50D",
}

MODEL_DISPLAY = {
    "esm2_35m": "ESM2-35M",
    "esm2_650m": "ESM2-650M",
    "ablang2": "AbLang2",
    "antiberty": "AntiBERTy",
    "sapiens": "Sapiens",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--models", nargs="+", default=ALL_MODELS,
                        choices=ALL_MODELS)
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    print("Loading FLAb2 binding data...")
    df = load_flab2_binding()
    sources = sorted(df["source_file"].unique())
    print(f"  {len(df)} variants across {len(sources)} proteins")

    all_results = []

    for model_key in args.models:
        display = MODEL_DISPLAY[model_key]
        print(f"\n{'='*70}")
        print(f"Model: {display}")
        print(f"{'='*70}")

        t_load = time.time()
        try:
            if model_key in MODEL_IDS:
                score_fn, model_obj = load_esm2(MODEL_IDS[model_key], device)
            elif model_key == "ablang2":
                score_fn, model_obj = load_ablang2(device)
            elif model_key == "antiberty":
                score_fn, model_obj = load_antiberty(device)
            elif model_key == "sapiens":
                score_fn, model_obj = load_sapiens()
            else:
                print(f"  Unknown model: {model_key}, skipping")
                continue
        except Exception as e:
            print(f"  Failed to load {display}: {e}")
            import traceback
            traceback.print_exc()
            continue

        print(f"  Loaded in {time.time() - t_load:.1f}s")

        for src in tqdm(sources, desc=display):
            group = df[df["source_file"] == src].copy()
            wt_heavy, wt_light = identify_wt(group)

            t0 = time.time()
            result = evaluate_protein_baseline(score_fn, group, wt_heavy, wt_light)
            elapsed = time.time() - t0

            if result is None:
                continue

            result["model"] = model_key
            result["source_file"] = src
            result["assay_type"] = group["assay_type"].iloc[0]
            result["study"] = group["study"].iloc[0]
            result["elapsed_s"] = elapsed
            all_results.append(result)

        # Cleanup
        del score_fn
        if model_key == "sapiens":
            del model_obj
        else:
            del model_obj
            if device != "cpu":
                torch.cuda.empty_cache()

    # Build results DataFrame
    results_df = pd.DataFrame(all_results)

    out_dir = REPO_ROOT / "data" / "features" / "evaluation_results" / "flab2_binding"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "baseline_per_protein_results.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    # Print summary
    print(f"\n{'='*70}")
    print("PER-PROTEIN RESULTS SUMMARY (LLR Score)")
    print(f"{'='*70}")

    for model_key in args.models:
        mdf = results_df[results_df["model"] == model_key]
        if len(mdf) == 0:
            print(f"\n--- {MODEL_DISPLAY[model_key]}: no results ---")
            continue
        vals = mdf["llr_score"].dropna()
        print(f"\n--- {MODEL_DISPLAY[model_key]} ({len(mdf)} proteins) ---")
        print(f"  Mean ρ:   {vals.mean():.4f}")
        print(f"  Median ρ: {vals.median():.4f}")
        print(f"  Std:      {vals.std():.4f}")
        print(f"  Min:      {vals.min():.4f}")
        print(f"  Max:      {vals.max():.4f}")

    # Cross-model comparison table
    print(f"\n{'='*70}")
    print("COMPARISON TABLE")
    print(f"{'='*70}")
    print(f"{'Model':<15} {'N proteins':>10} {'Mean ρ':>8} {'Median ρ':>10} {'Std':>8}")
    print("-" * 55)
    for model_key in args.models:
        mdf = results_df[results_df["model"] == model_key]
        if len(mdf) == 0:
            continue
        vals = mdf["llr_score"].dropna()
        print(f"{MODEL_DISPLAY[model_key]:<15} {len(mdf):>10} {vals.mean():>8.4f} "
              f"{vals.median():>10.4f} {vals.std():>8.4f}")


if __name__ == "__main__":
    main()
