#!/usr/bin/env python3
"""
Bottom-K% Recall for DMS Binding Affinity Benchmarks.

For each model, compute recall@bottom-K%: what fraction of the true worst-K%
variants (by experimental fitness) are captured in the model's predicted worst-K%.

This measures how well a model can *filter out* the worst candidates,
which is practically important for antibody developability screening.

Direction convention: use sign of Spearman rho to determine whether
higher or lower model score predicts better binding — then invert to
identify predicted worst variants.

Usage:
    python compute_bottomk_recall.py
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
STRAT_DIR = REPO_ROOT / "data" / "features" / "evaluation_results" / "flab2_binding" / "stratified"

DATASETS = {
    "G6.31": ("g6.31_per_variant_scores.csv", "fitness"),
    "CR9114-H1": ("cr9114_h1_per_variant_scores.csv", "h1_mean"),
    "CR9114-H3": ("cr9114_h3_per_variant_scores.csv", "h3_mean"),
    "Trastuzumab": ("trastuzumab_per_variant_scores.csv", "fitness"),
}

MODELS = {
    "PRISM (ours)": "prism_origin_logit_sum",
    "ESM2-35M": "esm2_35m_score",
    "ESM2-650M": "esm2_650m_score",
    "AbLang2": "ablang2_score",
    "AntiBERTy": "antiberty_score",
    "Sapiens": "sapiens_score",
}

BOTTOMK_FRACTIONS = [0.01, 0.05, 0.10]


def compute_bottomk_recall(fitness, scores, k_frac, ascending=False):
    """Compute bottom-K% recall.

    Args:
        fitness: array of experimental fitness values
        scores: array of model scores
        k_frac: fraction (e.g., 0.05 for bottom 5%)
        ascending: if True, lower model scores = predicted better binding
                   (so higher scores = predicted worst)

    Returns:
        recall: fraction of true bottom-K% captured by model's predicted bottom-K%
    """
    n = len(fitness)
    k = max(1, int(np.ceil(n * k_frac)))

    # True bottom-K%: lowest fitness
    true_bottomk = set(np.argsort(fitness)[:k])

    # Model's predicted bottom-K%: opposite end from where "good" is
    if ascending:
        # lower scores = better → higher scores = predicted worst
        pred_bottomk = set(np.argsort(scores)[-k:])
    else:
        # higher scores = better → lower scores = predicted worst
        pred_bottomk = set(np.argsort(scores)[:k])

    recall = len(true_bottomk & pred_bottomk) / len(true_bottomk)
    return recall


def main():
    all_rows = []

    for ds_name, (fname, fitness_col) in DATASETS.items():
        df = pd.read_csv(STRAT_DIR / fname)

        for model_name, score_col in MODELS.items():
            valid = df[score_col].notna() & df[fitness_col].notna()
            dv = df[valid].copy()

            if len(dv) < 100:
                continue

            fitness = dv[fitness_col].values
            scores = dv[score_col].values

            # Determine direction from Spearman rho
            rho, _ = spearmanr(scores, fitness)
            ascending = rho < 0  # if negative rho, lower scores = better fitness

            row = {
                "Dataset": ds_name,
                "Model": model_name,
                "N": len(dv),
                "Spearman ρ": rho,
            }

            for k_frac in BOTTOMK_FRACTIONS:
                recall = compute_bottomk_recall(fitness, scores, k_frac, ascending=ascending)
                pct = int(k_frac * 100)
                row[f"Bottom-Recall@{pct}%"] = recall

            all_rows.append(row)

    results = pd.DataFrame(all_rows)

    # Print per-dataset tables
    for ds_name in DATASETS:
        ds_results = results[results["Dataset"] == ds_name].copy()
        ds_results = ds_results.sort_values("Spearman ρ", key=abs, ascending=False)

        n_variants = ds_results["N"].iloc[0]
        print(f"\n{'='*90}")
        print(f"  {ds_name} — Bottom-K% Recall (worst variant filtering) (N={n_variants:,})")
        print(f"{'='*90}")
        print(f"{'Model':<16} {'Spearman ρ':>11} {'Bot-Recall@1%':>14} "
              f"{'Bot-Recall@5%':>14} {'Bot-Recall@10%':>15}")
        print("-" * 90)

        for _, r in ds_results.iterrows():
            rho_str = f"{r['Spearman ρ']:+.4f}"
            r1 = f"{r['Bottom-Recall@1%']:.3f}"
            r5 = f"{r['Bottom-Recall@5%']:.3f}"
            r10 = f"{r['Bottom-Recall@10%']:.3f}"
            marker = " ◄" if r["Model"] == "PRISM (ours)" else ""
            print(f"{r['Model']:<16} {rho_str:>11} {r1:>14} {r5:>14} {r10:>15}{marker}")

    # Summary table
    print(f"\n\n{'='*90}")
    print("  AVERAGE ACROSS ALL DATASETS — Bottom-K% Recall")
    print(f"{'='*90}")

    summary_rows = []
    for model_name in MODELS:
        mdf = results[results["Model"] == model_name]
        row = {
            "Model": model_name,
            "Mean |ρ|": mdf["Spearman ρ"].abs().mean(),
            "Mean Bot-Recall@1%": mdf["Bottom-Recall@1%"].mean(),
            "Mean Bot-Recall@5%": mdf["Bottom-Recall@5%"].mean(),
            "Mean Bot-Recall@10%": mdf["Bottom-Recall@10%"].mean(),
        }
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows).sort_values("Mean |ρ|", ascending=False)

    print(f"{'Model':<16} {'Mean |ρ|':>10} {'Bot-Recall@1%':>14} "
          f"{'Bot-Recall@5%':>14} {'Bot-Recall@10%':>15}")
    print("-" * 90)
    for _, r in summary.iterrows():
        marker = " ◄" if r["Model"] == "PRISM (ours)" else ""
        print(f"{r['Model']:<16} {r['Mean |ρ|']:>10.4f} {r['Mean Bot-Recall@1%']:>14.3f} "
              f"{r['Mean Bot-Recall@5%']:>14.3f} {r['Mean Bot-Recall@10%']:>15.3f}{marker}")

    print(f"\n  Random baseline: Bottom-Recall@K% = K% "
          f"(e.g., @1% = 0.010, @5% = 0.050, @10% = 0.100)")

    # Save results
    out_path = STRAT_DIR / "bottomk_recall_results.csv"
    results.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
