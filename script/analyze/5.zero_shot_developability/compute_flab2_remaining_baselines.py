#!/usr/bin/env python3
"""
Compute per-variant baseline LLR scores for FLAb2 remaining proteins.

Adds ESM2-35M, ESM2-650M, AbLang2, AntiBERTy, Sapiens per-variant scores
to flab2_remaining_per_variant_scores.csv, then recomputes stratified_rho_full.csv.

Uses the exact same data loading and variant filtering as stratified_full_analysis.py
to ensure row-level alignment with existing PRISM scores.

Usage:
    CUDA_VISIBLE_DEVICES=5 python compute_flab2_remaining_baselines.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Use the SAME functions as stratified_full_analysis.py for consistency
from evaluate_flab2_binding import find_mutations, identify_wt

# Use baseline model loaders
from evaluate_flab2_baselines import (
    find_mutations as find_mutations_single,
    load_ablang2,
    load_antiberty,
    load_esm2,
    load_sapiens,
    MODEL_IDS,
)

OUT_DIR = REPO_ROOT / "data" / "features" / "evaluation_results" / "flab2_binding" / "stratified"

EXCLUDE_SOURCES = {
    "koenig2017mutational_kd_g6.csv",
    "shanehsazzadeh2023unlocking_zerokd_trastuzumab.csv",
    "phillips2021binding_cr9114_h3_kd.csv",
    "phillips2021binding_cr9114_h1_kd.csv",
    "AbRank_dataset.csv.zip",
}

MUT_GROUPS = {
    "1": lambda n: n == 1,
    "2-5": lambda n: (n >= 2) & (n <= 5),
    ">5": lambda n: n > 5,
}

BASELINE_COLS = ["esm2_35m_score", "esm2_650m_score", "ablang2_score",
                 "antiberty_score", "sapiens_score"]


def load_flab2_remaining():
    """Load FLAb2 data excluding the 3 DMS datasets. Same as stratified_full_analysis.py."""
    df = pd.read_parquet(REPO_ROOT / "data" / "FLAb" / "flab2_binding.parquet")
    df = df[df["light"].notna()].copy()
    df = df[~df["assay_type"].isin(["bind/no bind", "predicted Kd"])]
    df = df[~df["source_file"].isin(EXCLUDE_SOURCES)]
    counts = df["source_file"].value_counts()
    valid = counts[counts >= 15].index
    df = df[df["source_file"].isin(valid)].copy()
    return df


def build_variant_list(df):
    """Build variant list using the SAME logic as stratified_full_analysis.py.

    Returns list of dicts, one per kept variant, in the exact same order
    as the existing flab2_remaining_per_variant_scores.csv.
    """
    all_variants = []
    sources = sorted(df["source_file"].unique())

    for src in sources:
        group = df[df["source_file"] == src].copy()
        wt_heavy, wt_light = identify_wt(group)

        protein_variants = []
        for _, row in group.iterrows():
            muts = find_mutations(wt_heavy, wt_light, row["heavy"], row["light"])
            if muts is None:
                continue
            protein_variants.append({
                "source_file": src,
                "n_mut": len(muts),
                "fitness": row["fitness"],
                "wt_heavy": wt_heavy,
                "wt_light": wt_light,
                "var_heavy": row["heavy"],
                "var_light": row["light"],
                "heavy_muts": find_mutations_single(wt_heavy, row["heavy"]) or [],
                "light_muts": find_mutations_single(wt_light, row["light"]) or [],
            })

        # Same filter as stratified_full_analysis.py
        if len(protein_variants) < 10:
            continue

        all_variants.extend(protein_variants)

    return all_variants


def score_variants_with_baseline(variants, score_fn, model_name):
    """Score all variants with a baseline model."""
    scores = []
    for v in tqdm(variants, desc=f"  {model_name}"):
        try:
            s = score_fn(v["wt_heavy"], v["wt_light"], v["heavy_muts"], v["light_muts"])
        except Exception:
            s = np.nan
        scores.append(s)
    return scores


def compute_stratified_rho(df, fitness_col, score_cols, group_name):
    """Compute Spearman rho for each score x mutation group."""
    results = []
    for gname, gfilt in MUT_GROUPS.items():
        mask = gfilt(df["n_mut"])
        group = df[mask]

        for col in score_cols:
            if col not in group.columns:
                continue
            scores = group[col].values
            fitness = group[fitness_col].values
            valid = np.isfinite(scores) & np.isfinite(fitness)
            n_valid = int(valid.sum())
            if n_valid < 10:
                rho = np.nan
            else:
                rho, _ = spearmanr(scores[valid], fitness[valid])
            results.append({
                "analysis_group": group_name,
                "mut_group": gname,
                "method": col,
                "rho": rho,
                "n_variants": n_valid,
            })
    return results


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load data
    print("Loading FLAb2 remaining data...")
    flab2_df = load_flab2_remaining()
    print(f"  {len(flab2_df):,} rows, {len(flab2_df['source_file'].unique())} proteins")

    # Build variant list (same order as existing PRISM CSV)
    print("Building variant list (matching stratified_full_analysis.py order)...")
    variants = build_variant_list(flab2_df)
    print(f"  {len(variants):,} variants after filtering")

    # Verify alignment with existing PRISM CSV
    existing = pd.read_csv(OUT_DIR / "flab2_remaining_per_variant_scores.csv")
    print(f"  Existing PRISM CSV: {len(existing):,} rows")

    if len(variants) != len(existing):
        print(f"  WARNING: Row count mismatch ({len(variants)} vs {len(existing)})!")
        print("  Will save baseline scores independently and merge by common keys.")
        use_alignment = False
    else:
        # Verify a sample of rows match
        match = True
        for i in [0, 100, len(variants) - 1]:
            if i < len(variants):
                if (variants[i]["source_file"] != existing.iloc[i]["source_file"] or
                        variants[i]["n_mut"] != existing.iloc[i]["n_mut"]):
                    match = False
                    break
        if match:
            print("  Row alignment verified!")
            use_alignment = True
        else:
            print("  WARNING: Rows don't align, saving independently.")
            use_alignment = False

    # Score with each baseline model
    models_to_run = [
        ("esm2_35m", "esm2_35m_score", lambda d: load_esm2(MODEL_IDS["esm2_35m"], d)),
        ("esm2_650m", "esm2_650m_score", lambda d: load_esm2(MODEL_IDS["esm2_650m"], d)),
        ("ablang2", "ablang2_score", lambda d: load_ablang2(d)),
        ("antiberty", "antiberty_score", lambda d: load_antiberty(d)),
        ("sapiens", "sapiens_score", lambda _: load_sapiens()),
    ]

    baseline_scores = {}

    for model_key, col_name, loader_fn in models_to_run:
        print(f"\n{'='*60}")
        print(f"Model: {model_key}")
        t0 = time.time()

        try:
            score_fn, model_obj = loader_fn(device)
        except Exception as e:
            print(f"  Failed to load: {e}")
            import traceback
            traceback.print_exc()
            continue

        print(f"  Loaded in {time.time() - t0:.1f}s")

        t0 = time.time()
        scores = score_variants_with_baseline(variants, score_fn, model_key)
        elapsed = time.time() - t0
        n_valid = sum(1 for s in scores if np.isfinite(s))
        print(f"  Done in {elapsed:.1f}s ({n_valid:,}/{len(scores):,} valid)")

        baseline_scores[col_name] = scores

        del score_fn, model_obj
        if device != "cpu":
            torch.cuda.empty_cache()

    # Add baseline columns to existing PRISM CSV
    if use_alignment:
        for col_name, scores in baseline_scores.items():
            existing[col_name] = scores
    else:
        # Build a new DataFrame from variants and merge
        baseline_df = pd.DataFrame([{
            "source_file": v["source_file"],
            "n_mut": v["n_mut"],
            "fitness": v["fitness"],
        } for v in variants])
        for col_name, scores in baseline_scores.items():
            baseline_df[col_name] = scores
        # Save baseline-only CSV as backup
        baseline_df.to_csv(OUT_DIR / "flab2_remaining_baseline_scores.csv", index=False)
        print(f"\nSaved baseline-only: {OUT_DIR / 'flab2_remaining_baseline_scores.csv'}")
        # Merge with existing by row index (same order assumption)
        for col_name in baseline_scores:
            if col_name in baseline_df.columns:
                existing[col_name] = baseline_df[col_name].values

    # Save updated CSV
    out_path = OUT_DIR / "flab2_remaining_per_variant_scores.csv"
    existing.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print(f"  Columns: {list(existing.columns)}")

    # ── Recompute stratified rho ──
    print("\n" + "=" * 60)
    print("Recomputing stratified rho with baselines...")

    # Load DMS stratified results (keep as-is)
    old_rho = pd.read_csv(OUT_DIR / "stratified_rho_full.csv")
    dms_rho = old_rho[~old_rho["analysis_group"].isin(["FLAb2 Remaining", "FLAb2 All"])]
    all_results = list(dms_rho.to_dict("records"))

    # FLAb2 Remaining
    prism_cols = [c for c in existing.columns if c.startswith("prism_")]
    available_baseline = [c for c in BASELINE_COLS if c in existing.columns]
    score_cols = prism_cols + available_baseline
    results = compute_stratified_rho(existing, "fitness", score_cols, "FLAb2 Remaining")
    all_results.extend(results)

    # FLAb2 All: combine FLAb2 remaining + DMS datasets
    print("Building FLAb2 All combined dataset...")
    combined_parts = [existing[["n_mut", "fitness"] + score_cols].copy()]

    for ds_name in ["g6.31", "cr9114_h1", "cr9114_h3", "trastuzumab"]:
        dms_path = OUT_DIR / f"{ds_name}_per_variant_scores.csv"
        if dms_path.exists():
            ddf = pd.read_csv(dms_path)
            fitness_col = "h1_mean" if ds_name == "cr9114_h1" else "h3_mean" if ds_name == "cr9114_h3" else "fitness"
            # Select columns present in both datasets
            dms_cols = [c for c in score_cols if c in ddf.columns]
            temp = ddf[["n_mut"] + dms_cols].copy()
            temp["fitness"] = ddf[fitness_col]
            combined_parts.append(temp)
            print(f"  Added {ds_name}: {len(ddf):,} variants")

    combined = pd.concat(combined_parts, ignore_index=True)
    print(f"  Combined: {len(combined):,} variants")

    combined_score_cols = [c for c in score_cols if c in combined.columns]
    results = compute_stratified_rho(combined, "fitness", combined_score_cols, "FLAb2 All")
    all_results.extend(results)

    # Save
    all_df = pd.DataFrame(all_results)
    all_df.to_csv(OUT_DIR / "stratified_rho_full.csv", index=False)
    print(f"\nSaved: {OUT_DIR / 'stratified_rho_full.csv'}")

    # Print summary
    key_methods = ["prism_origin_logit_sum", "prism_alpha_sum",
                   "esm2_35m_score", "esm2_650m_score",
                   "ablang2_score", "antiberty_score", "sapiens_score"]

    for group in ["FLAb2 Remaining", "FLAb2 All"]:
        gdf = all_df[all_df["analysis_group"] == group]
        print(f"\n{'='*60}")
        print(f"  {group}")
        print(f"{'='*60}")
        print(f"  {'Method':<30} {'1 mut':>14} {'2-5 mut':>14} {'>5 mut':>14}")
        print("  " + "-" * 74)
        for method in key_methods:
            mdf = gdf[gdf["method"] == method]
            if len(mdf) == 0:
                continue
            parts = [f"  {method:<30}"]
            for g in ["1", "2-5", ">5"]:
                row = mdf[mdf["mut_group"] == g]
                if len(row) > 0 and row.iloc[0]["n_variants"] >= 10:
                    rho = row.iloc[0]["rho"]
                    n = int(row.iloc[0]["n_variants"])
                    parts.append(f" {rho:>+.4f} ({n:>6,})")
                else:
                    n = int(row.iloc[0]["n_variants"]) if len(row) > 0 else 0
                    parts.append(f"    N/A ({n:>6,})")
            print("".join(parts))


if __name__ == "__main__":
    main()
