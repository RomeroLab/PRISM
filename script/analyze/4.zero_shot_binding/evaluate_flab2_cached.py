#!/usr/bin/env python3
"""
FLAb2 baseline per-protein Spearman with position-cached masked marginal.

For each protein group in the cleaned-up FLAb2 binding set, all variants share
the same WT. We precompute log_probs at each unique mutation position (with
that position masked in WT) once per group, then score every variant in the
group by dict lookup. Mathematically identical to per-variant masked marginal.

Cleaned-up set: excludes cr9114 H1/H3, g6.31, trastuzumab (which are already
evaluated as dedicated DMS datasets), and AbRank (mixed, not single-protein DMS).

Usage:
    CUDA_VISIBLE_DEVICES=0 python evaluate_flab2_cached.py \\
        --output_path data/.../flab2_baseline_per_protein_results.csv
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

# Same cleaned-up set as evaluate_flab2_baselines.py
EXCLUDE_SOURCES = {
    "koenig2017mutational_kd_g6.csv",
    "shanehsazzadeh2023unlocking_zerokd_trastuzumab.csv",
    "phillips2021binding_cr9114_h3_kd.csv",
    "phillips2021binding_cr9114_h1_kd.csv",
    "AbRank_dataset.csv.zip",
}

MIN_VARIANTS = 15


# Reuse cached scoring primitives from benchmark_baselines_cached
sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_baselines_cached import (
    find_mutation_indices,
    build_cache_esm2, load_esm2,
    build_cache_ablang2, load_ablang2,
    build_cache_antiberty_chain, load_antiberty,
    build_cache_sapiens_chain, load_sapiens,
    score_variant,
)


def load_flab2_binding():
    df = pd.read_parquet(REPO_ROOT / "data" / "FLAb" / "flab2_binding.parquet")
    df = df[df["light"].notna()].copy()
    df = df[~df["assay_type"].isin(["bind/no bind", "predicted Kd"])]
    df = df[~df["source_file"].isin(EXCLUDE_SOURCES)]
    counts = df["source_file"].value_counts()
    valid_sources = counts[counts >= MIN_VARIANTS].index
    df = df[df["source_file"].isin(valid_sources)].copy()
    return df


def identify_wt(group_df):
    heavy_counts = group_df["heavy"].value_counts()
    wt_heavy = heavy_counts.index[0]
    wt_rows = group_df[group_df["heavy"] == wt_heavy]
    if len(wt_rows) > 1:
        wt_light = wt_rows["light"].value_counts().index[0]
    else:
        wt_light = wt_rows.iloc[0]["light"]
    return wt_heavy, wt_light


def collect_group_positions(group_df, wt_heavy, wt_light):
    heavy_pos, light_pos = set(), set()
    for _, row in group_df.iterrows():
        if len(row["heavy"]) == len(wt_heavy):
            for i, (w, m) in enumerate(zip(wt_heavy, row["heavy"])):
                if w != m:
                    heavy_pos.add(i)
        if len(row["light"]) == len(wt_light):
            for i, (w, m) in enumerate(zip(wt_light, row["light"])):
                if w != m:
                    light_pos.add(i)
    return heavy_pos, light_pos


def score_group(group_df, wt_heavy, wt_light, heavy_cache, light_cache, aa_to_idx):
    scores, fitness_vals = [], []
    for _, row in group_df.iterrows():
        var_h, var_l = row["heavy"], row["light"]
        if len(var_h) != len(wt_heavy) or len(var_l) != len(wt_light):
            continue
        h_muts = find_mutation_indices(wt_heavy, var_h)
        l_muts = find_mutation_indices(wt_light, var_l)
        if not h_muts and not l_muts:
            continue  # WT itself
        s = score_variant(heavy_cache, light_cache, aa_to_idx, h_muts, l_muts)
        scores.append(s)
        fitness_vals.append(row["fitness"])
    if len(scores) < 10:
        return None
    scores = np.array(scores); fitness = np.array(fitness_vals)
    valid = np.isfinite(scores) & np.isfinite(fitness)
    if valid.sum() < 10:
        return None
    rho, _ = spearmanr(scores[valid], fitness[valid])
    return {"llr_score": rho, "n_variants": int(valid.sum())}


def evaluate_model(model_key, df, sources, device, batch_size):
    """Build per-protein cache for a given model, then score every protein."""
    # Load model once
    t0 = time.time()
    if model_key in ("esm2_35m", "esm2_650m"):
        model_id = {"esm2_35m": "facebook/esm2_t12_35M_UR50D",
                    "esm2_650m": "facebook/esm2_t33_650M_UR50D"}[model_key]
        model, tokenizer, aa_to_idx = load_esm2(model_id, device)
        kind = "esm2"
    elif model_key == "ablang2":
        model, tokenizer, aa_to_idx = load_ablang2(device)
        kind = "ablang2"
    elif model_key == "antiberty":
        model, tokenizer, aa_to_idx = load_antiberty(device)
        kind = "antiberty"
    elif model_key == "sapiens":
        heavy_model, light_model, tokenizer, aa_to_idx, h_max, l_max = load_sapiens()
        kind = "sapiens"
    else:
        raise ValueError(f"Unknown model: {model_key}")
    print(f"  Loaded {model_key} in {time.time()-t0:.1f}s")

    rows = []
    for src in tqdm(sources, desc=f"{model_key} per-protein", leave=False):
        group = df[df["source_file"] == src].copy()
        wt_heavy, wt_light = identify_wt(group)
        h_pos, l_pos = collect_group_positions(group, wt_heavy, wt_light)
        if not h_pos and not l_pos:
            continue

        try:
            if kind == "esm2":
                wt_seq = wt_heavy + wt_light
                all_positions = sorted(list(h_pos)) + [len(wt_heavy) + p for p in sorted(l_pos)]
                full_cache = build_cache_esm2(model, tokenizer, wt_seq, all_positions, device, batch_size)
                heavy_cache = {p: full_cache[p] for p in h_pos}
                light_cache = {p: full_cache[len(wt_heavy) + p] for p in l_pos}
            elif kind == "ablang2":
                heavy_cache, light_cache = build_cache_ablang2(
                    model, tokenizer, wt_heavy, wt_light, sorted(h_pos), sorted(l_pos), device, batch_size)
            elif kind == "antiberty":
                heavy_cache = build_cache_antiberty_chain(model, tokenizer, wt_heavy, sorted(h_pos), device, batch_size)
                light_cache = build_cache_antiberty_chain(model, tokenizer, wt_light, sorted(l_pos), device, batch_size)
            elif kind == "sapiens":
                heavy_cache = build_cache_sapiens_chain(heavy_model, tokenizer, wt_heavy, h_max, sorted(h_pos), "cpu", batch_size)
                light_cache = build_cache_sapiens_chain(light_model, tokenizer, wt_light, l_max, sorted(l_pos), "cpu", batch_size)
        except Exception as e:
            print(f"  [{src}] cache build failed: {e}")
            continue

        result = score_group(group, wt_heavy, wt_light, heavy_cache, light_cache, aa_to_idx)
        if result is None:
            continue
        result["model"] = model_key
        result["source_file"] = src
        result["assay_type"] = group["assay_type"].iloc[0]
        result["study"] = group["study"].iloc[0]
        rows.append(result)

    # Cleanup
    if kind == "sapiens":
        del heavy_model, light_model
    else:
        del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--models", nargs="+",
                        default=["esm2_35m", "esm2_650m", "ablang2", "antiberty", "sapiens"],
                        choices=["esm2_35m", "esm2_650m", "ablang2", "antiberty", "sapiens"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--output_path", default=None)
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading FLAb2 binding data (cleaned-up set)...")
    df = load_flab2_binding()
    sources = sorted(df["source_file"].unique())
    print(f"  {len(df)} variants across {len(sources)} proteins")
    print(f"  Excluded: {sorted(EXCLUDE_SOURCES)}")

    all_rows = []
    for model_key in args.models:
        print(f"\n{'='*60}\n{model_key}\n{'='*60}")
        rows = evaluate_model(model_key, df, sources, device, args.batch_size)
        all_rows.extend(rows)

    results_df = pd.DataFrame(all_rows)

    if args.output_path:
        out_path = Path(args.output_path)
    else:
        out_path = REPO_ROOT / "data" / "prism_results" / "3.zero-shot" / "baseline_scores_fixed" / "flab2_baseline_per_protein.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    # Summary
    print(f"\n{'='*60}\nPer-Model Summary (Spearman rho across proteins)\n{'='*60}")
    for model_key in args.models:
        mdf = results_df[results_df["model"] == model_key]
        if not len(mdf):
            print(f"  {model_key}: no results"); continue
        vals = mdf["llr_score"].dropna()
        print(f"  {model_key:<12} n_proteins={len(mdf):>3}  mean_rho={vals.mean():+.4f}  "
              f"median={vals.median():+.4f}  std={vals.std():.4f}")


if __name__ == "__main__":
    main()
