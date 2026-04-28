#!/usr/bin/env python3
"""
FLAb2 Binding Affinity: Ranking-Based Method Comparison.

For each protein (source_file), rank all methods by |Spearman ρ|,
then compute:
  1. Average rank per method (lower = better)
  2. Median rank per method
  3. Win rate (fraction of proteins where method has rank 1)
  4. Top-3 rate (fraction of proteins where method is in top 3)
  5. Pairwise win matrix (row i beats column j how often)
  6. Sign consistency (how often ρ has the correct direction)
  7. Weighted rank by n_variants (larger DMS datasets count more)

Usage:
    python compare_flab2_methods_ranking.py
    python compare_flab2_methods_ranking.py --metric abs_rho   # rank by |ρ|
    python compare_flab2_methods_ranking.py --metric signed_rho # rank by ρ (positive = better)
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import wilcoxon, friedmanchisquare

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = REPO_ROOT / "data" / "features" / "evaluation_results" / "flab2_binding"

# PRISM signal columns to pick the best one per model
PRISM_SIGNAL_COLS = [
    "origin_logit_sum", "origin_logit_mean",
    "origin_prob_sum", "origin_prob_mean",
    "alpha_sum", "alpha_mean",
    "ngl_logprob_sum", "ngl_logprob_mean",
]

BASELINE_DISPLAY = {
    "esm2_35m": "ESM2-35M",
    "esm2_650m": "ESM2-650M",
    "ablang2": "AbLang2",
    "antiberty": "AntiBERTy",
    "sapiens": "Sapiens",
}


def load_and_unify():
    """Load PRISM and baseline results, unify into a method × protein ρ matrix.

    For PRISM models, each signal is a separate method.
    For baselines, the single LLR score is the method.

    Returns:
        rho_matrix: DataFrame with columns = method names, index = source_file
        n_variants: Series with index = source_file
    """
    # --- PRISM results ---
    prism_df = pd.read_csv(RESULTS_DIR / "per_protein_results.csv")

    prism_methods = {}
    for _, row in prism_df.iterrows():
        model = row["model"]
        source = row["source_file"]
        for sig in PRISM_SIGNAL_COLS:
            method_name = f"PRISM_{model}_{sig}"
            if method_name not in prism_methods:
                prism_methods[method_name] = {}
            prism_methods[method_name][source] = row[sig]

    prism_rho = pd.DataFrame(prism_methods)

    # n_variants from PRISM (pick first model's value)
    nvar_df = prism_df[prism_df["model"] == prism_df["model"].iloc[0]][
        ["source_file", "n_variants"]
    ].set_index("source_file")["n_variants"]

    # --- Baseline results ---
    baseline_df = pd.read_csv(RESULTS_DIR / "baseline_per_protein_results.csv")

    baseline_methods = {}
    for _, row in baseline_df.iterrows():
        model = row["model"]
        display = BASELINE_DISPLAY.get(model, model)
        method_name = f"{display}_LLR"
        source = row["source_file"]
        if method_name not in baseline_methods:
            baseline_methods[method_name] = {}
        baseline_methods[method_name][source] = row["llr_score"]

    baseline_rho = pd.DataFrame(baseline_methods)

    # Merge
    rho_matrix = prism_rho.join(baseline_rho, how="outer")

    # Only keep proteins present in both PRISM and baseline
    all_sources = rho_matrix.index
    rho_matrix = rho_matrix.dropna(how="all")

    return rho_matrix, nvar_df


def compute_rankings(rho_matrix, metric="abs_rho"):
    """Compute per-protein rankings.

    metric: "abs_rho" (rank by |ρ|, higher=better) or "signed_rho" (rank by ρ, higher=better)

    Returns:
        rank_matrix: DataFrame with same shape, values = rank (1 = best)
    """
    if metric == "abs_rho":
        score_matrix = rho_matrix.abs()
    else:
        score_matrix = rho_matrix.copy()

    # Rank per row (protein), higher score = rank 1
    # method="average" for ties
    rank_matrix = score_matrix.rank(axis=1, ascending=False, method="average")
    return rank_matrix


def compute_summary(rank_matrix, rho_matrix, n_variants):
    """Compute summary statistics per method."""
    n_methods = rank_matrix.shape[1]
    n_proteins = rank_matrix.shape[0]

    summary = pd.DataFrame(index=rank_matrix.columns)
    summary["avg_rank"] = rank_matrix.mean()
    summary["median_rank"] = rank_matrix.median()
    summary["std_rank"] = rank_matrix.std()
    summary["win_rate"] = (rank_matrix == 1).sum() / n_proteins
    summary["top3_rate"] = (rank_matrix <= 3).sum() / n_proteins
    summary["top5_rate"] = (rank_matrix <= 5).sum() / n_proteins
    summary["worst_rank"] = rank_matrix.max()
    summary["best_rank"] = rank_matrix.min()

    # Weighted average rank by n_variants
    common = rank_matrix.index.intersection(n_variants.index)
    if len(common) > 0:
        weights = n_variants.loc[common]
        weighted_ranks = rank_matrix.loc[common].multiply(weights, axis=0)
        summary["weighted_avg_rank"] = weighted_ranks.sum() / weights.sum()

    # Mean |ρ| and mean ρ
    summary["mean_abs_rho"] = rho_matrix.abs().mean()
    summary["mean_rho"] = rho_matrix.mean()
    summary["median_abs_rho"] = rho_matrix.abs().median()

    # Sign consistency: fraction of proteins where ρ > 0
    summary["frac_positive"] = (rho_matrix > 0).sum() / rho_matrix.notna().sum()

    summary = summary.sort_values("avg_rank")
    return summary


def pairwise_wins(rho_matrix, metric="abs_rho"):
    """Compute pairwise win matrix.

    wins[i, j] = fraction of proteins where method i beats method j.
    """
    if metric == "abs_rho":
        score_matrix = rho_matrix.abs()
    else:
        score_matrix = rho_matrix.copy()

    methods = score_matrix.columns.tolist()
    n = len(methods)
    wins = pd.DataFrame(np.zeros((n, n)), index=methods, columns=methods)

    for i in range(n):
        for j in range(n):
            if i == j:
                wins.iloc[i, j] = 0.5
                continue
            mi = methods[i]
            mj = methods[j]
            valid = score_matrix[[mi, mj]].dropna()
            if len(valid) == 0:
                wins.iloc[i, j] = np.nan
                continue
            wins.iloc[i, j] = (valid[mi] > valid[mj]).sum() / len(valid)

    return wins


def pairwise_significance(rho_matrix, metric="abs_rho", alpha=0.05):
    """Wilcoxon signed-rank test for pairwise method comparison.

    Returns p-value matrix.
    """
    if metric == "abs_rho":
        score_matrix = rho_matrix.abs()
    else:
        score_matrix = rho_matrix.copy()

    methods = score_matrix.columns.tolist()
    n = len(methods)
    pvals = pd.DataFrame(np.ones((n, n)), index=methods, columns=methods)

    for i in range(n):
        for j in range(i + 1, n):
            mi = methods[i]
            mj = methods[j]
            valid = score_matrix[[mi, mj]].dropna()
            if len(valid) < 10:
                continue
            diff = valid[mi] - valid[mj]
            if (diff == 0).all():
                continue
            try:
                _, p = wilcoxon(diff, alternative="two-sided")
                pvals.iloc[i, j] = p
                pvals.iloc[j, i] = p
            except Exception:
                pass

    return pvals


def pick_best_prism_signal(summary):
    """For each PRISM model, pick the best signal based on avg_rank."""
    best = {}
    for method in summary.index:
        if not method.startswith("PRISM_"):
            continue
        parts = method.split("_", 2)  # PRISM, model, signal
        model = parts[1]
        if model not in best or summary.loc[method, "avg_rank"] < best[model][1]:
            best[model] = (method, summary.loc[method, "avg_rank"])
    return {model: info[0] for model, info in best.items()}


def aggregate_best_per_model(rho_matrix, summary):
    """Create a simplified matrix with best PRISM signal per model + all baselines."""
    best_prism = pick_best_prism_signal(summary)
    keep_methods = []
    rename = {}

    for model, method in best_prism.items():
        keep_methods.append(method)
        signal = method.split("_", 2)[2]
        rename[method] = f"PRISM_{model} ({signal})"

    for col in rho_matrix.columns:
        if not col.startswith("PRISM_"):
            keep_methods.append(col)

    agg = rho_matrix[keep_methods].rename(columns=rename)
    return agg


def plot_rank_distribution(rank_matrix, output_path, title="Rank Distribution per Method"):
    """Box plot of rank distributions."""
    # Sort methods by median rank
    medians = rank_matrix.median().sort_values()
    ordered = medians.index.tolist()

    # Shorten names for display
    short_names = []
    for name in ordered:
        if name.startswith("PRISM_"):
            parts = name.split("_", 2)
            short_names.append(f"PRISM {parts[1]}\n{parts[2]}")
        else:
            short_names.append(name.replace("_LLR", "\n(LLR)"))

    fig, ax = plt.subplots(figsize=(max(12, len(ordered) * 0.6), 7))
    data = [rank_matrix[col].dropna().values for col in ordered]

    bp = ax.boxplot(data, patch_artist=True, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="red", markersize=6))

    colors = plt.cm.Set3(np.linspace(0, 1, len(ordered)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticklabels(short_names, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("Rank (1 = best)")
    ax.set_title(title, fontweight="bold")
    ax.axhline(y=1, color="green", lw=0.5, ls="--", alpha=0.5)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_pairwise_heatmap(wins, output_path, title="Pairwise Win Rate"):
    """Heatmap of pairwise win rates."""
    # Sort by row mean win rate
    order = wins.mean(axis=1).sort_values(ascending=False).index
    wins = wins.loc[order, order]

    # Short names
    short = []
    for name in wins.index:
        if name.startswith("PRISM_"):
            parts = name.split("_", 2)
            short.append(f"PRISM {parts[1]} {parts[2]}")
        else:
            short.append(name)

    wins_plot = wins.copy()
    wins_plot.index = short
    wins_plot.columns = short

    fig, ax = plt.subplots(figsize=(max(10, len(short) * 0.5), max(8, len(short) * 0.4)))
    sns.heatmap(
        wins_plot, annot=True, fmt=".2f", cmap="RdYlGn", center=0.5,
        vmin=0, vmax=1, linewidths=0.5, ax=ax,
        cbar_kws={"label": "Win rate (row beats column)"},
    )
    ax.set_title(title, fontweight="bold")
    plt.xticks(fontsize=6, rotation=45, ha="right")
    plt.yticks(fontsize=6)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_summary_comparison(agg_summary, output_path):
    """Bar chart comparing best methods on key metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    methods = agg_summary.index.tolist()
    short_names = []
    for name in methods:
        if name.startswith("PRISM_"):
            short_names.append(name.replace("PRISM_", "P:").replace(" (", "\n("))
        else:
            short_names.append(name.replace("_LLR", ""))

    colors = []
    for name in methods:
        if "PRISM" in name:
            colors.append("#2196F3")
        else:
            colors.append("#FF9800")

    # Average rank
    ax = axes[0]
    vals = agg_summary["avg_rank"].values
    bars = ax.barh(range(len(methods)), vals, color=colors, alpha=0.8, edgecolor="white")
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(short_names, fontsize=8)
    ax.set_xlabel("Average Rank (lower = better)")
    ax.set_title("Average Rank", fontweight="bold")
    ax.invert_yaxis()
    for bar, val in zip(bars, vals):
        ax.text(val + 0.1, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}", va="center", fontsize=8)

    # Mean |ρ|
    ax = axes[1]
    vals = agg_summary["mean_abs_rho"].values
    bars = ax.barh(range(len(methods)), vals, color=colors, alpha=0.8, edgecolor="white")
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(short_names, fontsize=8)
    ax.set_xlabel("Mean |ρ|")
    ax.set_title("Mean |Spearman ρ|", fontweight="bold")
    ax.invert_yaxis()
    for bar, val in zip(bars, vals):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8)

    # Win rate
    ax = axes[2]
    vals = agg_summary["win_rate"].values
    bars = ax.barh(range(len(methods)), vals, color=colors, alpha=0.8, edgecolor="white")
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(short_names, fontsize=8)
    ax.set_xlabel("Win Rate (fraction)")
    ax.set_title("Win Rate (rank=1)", fontweight="bold")
    ax.invert_yaxis()
    for bar, val in zip(bars, vals):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", fontsize=8)

    plt.suptitle("PRISM vs Baselines: FLAb2 Binding Affinity", fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", default="abs_rho",
                        choices=["abs_rho", "signed_rho"],
                        help="Ranking metric: abs_rho (|ρ|) or signed_rho (ρ)")
    args = parser.parse_args()

    out_dir = RESULTS_DIR / "ranking_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading results...")
    rho_matrix, n_variants = load_and_unify()
    print(f"  {rho_matrix.shape[0]} proteins × {rho_matrix.shape[1]} methods")

    # ================================================================
    # Full comparison (all PRISM signals + baselines)
    # ================================================================
    print(f"\nRanking by: {args.metric}")
    rank_matrix = compute_rankings(rho_matrix, metric=args.metric)
    summary = compute_summary(rank_matrix, rho_matrix, n_variants)

    print(f"\n{'='*90}")
    print("FULL METHOD RANKING (all PRISM signals + baselines)")
    print(f"{'='*90}")
    print(f"{'Rank':<5} {'Method':<40} {'Avg Rank':>9} {'Std':>6} {'Med Rank':>9} "
          f"{'Win%':>6} {'Top3%':>6} {'Mean|ρ|':>8} {'Frac+':>6}")
    print("-" * 96)
    for rank, (method, row) in enumerate(summary.iterrows(), 1):
        print(f"{rank:<5} {method:<40} {row['avg_rank']:>9.2f} {row['std_rank']:>6.2f} {row['median_rank']:>9.1f} "
              f"{row['win_rate']:>6.1%} {row['top3_rate']:>6.1%} "
              f"{row['mean_abs_rho']:>8.4f} {row['frac_positive']:>6.1%}")

    summary.to_csv(out_dir / f"full_ranking_summary_{args.metric}.csv")

    # ================================================================
    # Aggregated: best PRISM signal per model + baselines
    # ================================================================
    agg_rho = aggregate_best_per_model(rho_matrix, summary)
    agg_rank = compute_rankings(agg_rho, metric=args.metric)
    agg_summary = compute_summary(agg_rank, agg_rho, n_variants)

    print(f"\n{'='*90}")
    print("AGGREGATED RANKING (best PRISM signal per model + baselines)")
    print(f"{'='*90}")
    print(f"{'Rank':<5} {'Method':<45} {'Avg Rank':>9} {'Std':>6} {'Med Rank':>9} "
          f"{'Win%':>6} {'Top3%':>6} {'Mean|ρ|':>8}")
    print("-" * 96)
    for rank, (method, row) in enumerate(agg_summary.iterrows(), 1):
        print(f"{rank:<5} {method:<45} {row['avg_rank']:>9.2f} {row['std_rank']:>6.2f} {row['median_rank']:>9.1f} "
              f"{row['win_rate']:>6.1%} {row['top3_rate']:>6.1%} "
              f"{row['mean_abs_rho']:>8.4f}")

    agg_summary.to_csv(out_dir / f"aggregated_ranking_summary_{args.metric}.csv")

    # ================================================================
    # Per-model comparison: each PRISM model vs 5 baselines
    # ================================================================
    best_prism = pick_best_prism_signal(summary)
    baseline_cols = [c for c in rho_matrix.columns if not c.startswith("PRISM_")]

    for model_version, best_method in sorted(best_prism.items()):
        signal_name = best_method.split("_", 2)[2]
        prism_label = f"PRISM_{model_version} ({signal_name})"

        # Build 6-method matrix: 1 PRISM + 5 baselines
        model_rho = rho_matrix[[best_method] + baseline_cols].copy()
        model_rho = model_rho.rename(columns={best_method: prism_label})
        model_rho = model_rho.dropna(how="all")

        model_rank = compute_rankings(model_rho, metric=args.metric)
        model_summary = compute_summary(model_rank, model_rho, n_variants)
        model_wins = pairwise_wins(model_rho, metric=args.metric)
        model_pvals = pairwise_significance(model_rho, metric=args.metric)

        print(f"\n{'='*96}")
        print(f"  {prism_label}  vs  5 Baselines  ({args.metric})")
        print(f"{'='*96}")
        print(f"{'Rank':<5} {'Method':<35} {'Avg Rank':>9} {'Std':>6} {'Med':>5} "
              f"{'Win%':>6} {'Top3%':>6} {'Mean ρ':>8} {'Mean|ρ|':>8} {'Frac+':>6}")
        print("-" * 96)
        for rank, (method, row) in enumerate(model_summary.iterrows(), 1):
            print(f"{rank:<5} {method:<35} {row['avg_rank']:>9.2f} {row['std_rank']:>6.2f} "
                  f"{row['median_rank']:>5.1f} {row['win_rate']:>6.1%} {row['top3_rate']:>6.1%} "
                  f"{row['mean_rho']:>8.4f} {row['mean_abs_rho']:>8.4f} {row['frac_positive']:>6.1%}")

        # Pairwise wins
        print(f"\n  Pairwise win rates (row beats column):")
        short = {m: m[:25] for m in model_wins.index}
        print(model_wins.rename(index=short, columns=short).round(2).to_string())

        # Wilcoxon: PRISM vs each baseline
        print(f"\n  Wilcoxon signed-rank (PRISM vs each baseline):")
        prism_col = prism_label
        for bl in baseline_cols:
            p = model_pvals.loc[prism_col, bl]
            score_p = model_rho[prism_col] if args.metric == "signed_rho" else model_rho[prism_col].abs()
            score_b = model_rho[bl] if args.metric == "signed_rho" else model_rho[bl].abs()
            valid = pd.concat([score_p, score_b], axis=1).dropna()
            mean_diff = (valid.iloc[:, 0] - valid.iloc[:, 1]).mean()
            direction = "PRISM better" if mean_diff > 0 else "baseline better"
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            print(f"    vs {bl:<20s}  p={p:.4f} {sig:>4s}  Δmean={mean_diff:+.4f}  ({direction})")

        # Weighted rank
        if "weighted_avg_rank" in model_summary.columns:
            print(f"\n  Weighted avg rank (by n_variants):")
            weighted = model_summary.sort_values("weighted_avg_rank")
            for r, (m, row) in enumerate(weighted.iterrows(), 1):
                tag = " <--" if m == prism_label else ""
                print(f"    {r}. {m:<35} w_rank={row['weighted_avg_rank']:.2f}  "
                      f"(unwt={row['avg_rank']:.2f}){tag}")

        # Save per-model results
        model_summary.to_csv(out_dir / f"{model_version}_vs_baselines_{args.metric}.csv")
        model_wins.to_csv(out_dir / f"{model_version}_pairwise_wins_{args.metric}.csv")

        # Visualization
        plot_rank_distribution(
            model_rank, out_dir / f"{model_version}_rank_distribution_{args.metric}.png",
            title=f"{prism_label} vs Baselines — Rank Distribution ({args.metric})"
        )
        plot_pairwise_heatmap(
            model_wins, out_dir / f"{model_version}_pairwise_heatmap_{args.metric}.png",
            title=f"{prism_label} vs Baselines — Pairwise Win Rate ({args.metric})"
        )

    # ================================================================
    # Save global outputs
    # ================================================================
    agg_rank.to_csv(out_dir / f"rank_matrix_{args.metric}.csv")
    agg_rho.to_csv(out_dir / f"rho_matrix_aggregated.csv")

    print(f"\nAll outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
