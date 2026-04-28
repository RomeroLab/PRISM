#!/usr/bin/env python3
"""
Top-K% Recall for DMS Binding Affinity Benchmarks.

For each model, compute recall@K%: what fraction of the true top-K% binders
(by experimental fitness) are captured in the model's predicted top-K%.

Direction convention: use sign of Spearman rho to determine whether
higher or lower model score predicts better binding.

Usage:
    python compute_topk_recall.py
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from pathlib import Path

REPO_ROOT  = Path(__file__).resolve().parents[3]
# Latest baseline per-variant scores — match report Section 1 (refreshed 2026-04-25)
BASE_DIR   = REPO_ROOT / "data" / "prism_results" / "3.zero-shot" / "baseline_scores_fixed"
# IgLM per-variant scores
IGLM_DIR   = REPO_ROOT / "data" / "baselines" / "iglm" / "DMS"
# v44o PRISM per-variant scores
V44O_DIR   = REPO_ROOT / "data" / "prism_results" / "3.zero-shot" / "dms_prism_v44o_peak_glinv_scores"
OUT_DIR    = BASE_DIR  # save recall results next to baseline data

# Each entry: (baseline_file, iglm_file, v44o_file, fitness_col)
DATASETS = {
    "G6.31":       ("g6.31_baseline_scores.csv",       "g6.31_NB_edits.csv",       "g6.31_prism_hf_scores.csv",       "fitness"),
    "CR9114-H1":   ("cr9114_baseline_scores.csv",      "cr9114_NB_edits.csv",      "cr9114_h1_prism_hf_scores.csv",   "h1_mean"),
    "CR9114-H3":   ("cr9114_baseline_scores.csv",      "cr9114_NB_edits.csv",      "cr9114_h3_prism_hf_scores.csv",   "h3_mean"),
    "Trastuzumab": ("trastuzumab_baseline_scores.csv", "trastuzumab_NB_edits.csv", "trastuzumab_prism_hf_scores.csv", "fitness"),
}

# Baseline model score columns and display names (from baseline_scores_fixed CSVs)
BASELINES = {
    "ESM2-35M":  "esm2_35m_score",
    "ESM2-650M": "esm2_650m_score",
    "AbLang2":   "ablang2_score",
    "AntiBERTy": "antiberty_score",
    "Sapiens":   "sapiens_score",
}
# IgLM signal column from baselines/iglm/DMS/*.csv
IGLM_SCORE_COL = "IgLM_score"
# v44o PRISM canonical signal — NGL-head LLR
V44O_SIGNAL    = "prism_score_ngl"

TOPK_FRACTIONS = [0.01, 0.05, 0.10]


def compute_topk_recall(fitness, scores, k_frac, ascending=False):
    """Compute top-K% recall.

    Args:
        fitness: array of experimental fitness values
        scores: array of model scores
        k_frac: fraction (e.g., 0.05 for top 5%)
        ascending: if True, lower scores = predicted better

    Returns:
        recall: fraction of true top-K% captured by model's top-K%
    """
    n = len(fitness)
    k = max(1, int(np.ceil(n * k_frac)))

    # True top-K%: highest fitness
    true_topk = set(np.argsort(fitness)[-k:])

    # Model's predicted top-K%
    if ascending:
        pred_topk = set(np.argsort(scores)[:k])  # lowest scores
    else:
        pred_topk = set(np.argsort(scores)[-k:])  # highest scores

    recall = len(true_topk & pred_topk) / len(true_topk)
    return recall


def _eval_one(fitness, scores, ds_name, model_name):
    """Compute Spearman rho + recall@K for one (model, dataset). Higher
    score = better binder convention (no per-dataset flipping)."""
    rho, _ = spearmanr(scores, fitness)
    row = {
        "Dataset": ds_name,
        "Model": model_name,
        "N": len(fitness),
        "Spearman ρ": rho,
    }
    for k_frac in TOPK_FRACTIONS:
        recall = compute_topk_recall(fitness, scores, k_frac, ascending=False)
        row[f"Recall@{int(k_frac * 100)}%"] = recall
    return row


def _drop_wt(df):
    """Drop WT rows when present (consistent with report Section 1)."""
    if "Mutations" in df.columns:
        return df[df["Mutations"] != "WT"]
    return df


def main():
    all_rows = []

    for ds_name, (base_f, iglm_f, v44o_f, fitness_col) in DATASETS.items():
        # --- Baselines (5 models): baseline_scores_fixed/*.csv ---
        df_b = _drop_wt(pd.read_csv(BASE_DIR / base_f))
        for model_name, score_col in BASELINES.items():
            valid = df_b[score_col].notna() & df_b[fitness_col].notna()
            dv = df_b[valid]
            if len(dv) < 100:
                continue
            all_rows.append(_eval_one(
                dv[fitness_col].values, dv[score_col].values,
                ds_name, model_name))

        # --- IgLM: baselines/iglm/DMS/*.csv ---
        iglm_path = IGLM_DIR / iglm_f
        if iglm_path.exists():
            df_i = _drop_wt(pd.read_csv(iglm_path))
            valid = (df_i[IGLM_SCORE_COL].notna() & df_i[fitness_col].notna()
                     & (df_i[IGLM_SCORE_COL] != 0))
            dv = df_i[valid]
            if len(dv) >= 100:
                all_rows.append(_eval_one(
                    dv[fitness_col].values, dv[IGLM_SCORE_COL].values,
                    ds_name, "IgLM"))

        # --- PRISM v44o: dms_prism_v44o_peak_glinv_scores/*.csv ---
        v44o_path = V44O_DIR / v44o_f
        if v44o_path.exists():
            df_p = pd.read_csv(v44o_path)
            valid = df_p[V44O_SIGNAL].notna() & df_p["fitness"].notna()
            dv = df_p[valid]
            if len(dv) >= 100:
                all_rows.append(_eval_one(
                    dv["fitness"].values, dv[V44O_SIGNAL].values,
                    ds_name, "PRISM (ours)"))

    results = pd.DataFrame(all_rows)

    # Print per-dataset tables
    for ds_name in DATASETS:
        ds_results = results[results["Dataset"] == ds_name].copy()
        # Sort by raw Spearman ρ (descending) — best directed-rho model first.
        ds_results = ds_results.sort_values("Spearman ρ", ascending=False)

        n_variants = ds_results["N"].iloc[0]
        print(f"\n{'='*85}")
        print(f"  {ds_name} (N={n_variants:,})")
        print(f"{'='*85}")
        print(f"{'Model':<16} {'Spearman ρ':>11} {'Recall@1%':>11} {'Recall@5%':>11} {'Recall@10%':>12}")
        print("-" * 85)

        for _, r in ds_results.iterrows():
            rho_str = f"{r['Spearman ρ']:+.4f}"
            r1 = f"{r['Recall@1%']:.3f}"
            r5 = f"{r['Recall@5%']:.3f}"
            r10 = f"{r['Recall@10%']:.3f}"
            marker = " ◄" if r["Model"] == "PRISM (ours)" else ""
            print(f"{r['Model']:<16} {rho_str:>11} {r1:>11} {r5:>11} {r10:>12}{marker}")

    # Summary table: average across datasets
    print(f"\n\n{'='*85}")
    print("  AVERAGE ACROSS ALL DATASETS")
    print(f"{'='*85}")

    summary_rows = []
    all_models = list(BASELINES.keys()) + ["IgLM", "PRISM (ours)"]
    for model_name in all_models:
        mdf = results[results["Model"] == model_name]
        if mdf.empty:
            continue
        row = {
            "Model": model_name,
            "Mean ρ": mdf["Spearman ρ"].mean(),
            "Mean Recall@1%": mdf["Recall@1%"].mean(),
            "Mean Recall@5%": mdf["Recall@5%"].mean(),
            "Mean Recall@10%": mdf["Recall@10%"].mean(),
        }
        summary_rows.append(row)

    # Sort by mean directed rho (signed) — wrong-direction models drop to bottom
    summary = pd.DataFrame(summary_rows).sort_values("Mean ρ", ascending=False)

    print(f"{'Model':<16} {'Mean ρ':>10} {'Recall@1%':>11} {'Recall@5%':>11} {'Recall@10%':>12}")
    print("-" * 85)
    for _, r in summary.iterrows():
        marker = " ◄" if r["Model"] == "PRISM (ours)" else ""
        print(f"{r['Model']:<16} {r['Mean ρ']:>+10.4f} {r['Mean Recall@1%']:>11.3f} "
              f"{r['Mean Recall@5%']:>11.3f} {r['Mean Recall@10%']:>12.3f}{marker}")

    # Random baseline
    for k_frac in TOPK_FRACTIONS:
        pct = int(k_frac * 100)
    print(f"\n  Random baseline: Recall@K% = K% (e.g., Recall@1% = 0.010, @5% = 0.050, @10% = 0.100)")

    # Save results next to the latest baseline data
    out_path = OUT_DIR / "topk_recall_results.csv"
    results.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
