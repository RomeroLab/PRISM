#!/usr/bin/env python
"""
Publication-quality figures for PRISM binding affinity evaluation.
All comparisons use PER-DATASET RANKING (not raw rho).

Generates combined multi-panel figures AND individual panel files in panels/ subdirectory.

Figures:
  1. v34.1b exhaustive scoring (A: heatmap, B: mean rank bar)
  2. v34.1b vs baselines on FLAb2 (A-D: rank, boxplot, violin, table)
  3. FLAb2 dataset statistics (A-F: variants, assays, studies, histogram, groups, table)
  4. v34.1b signal x context heatmap (single panel)
  5. Stratified results (one panel per analysis group)
  5b. Stratified table (single panel)
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import matplotlib.ticker as mticker
from matplotlib.patches import Patch

# -- Style --
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 12,
    "font.weight": "bold",
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "axes.labelsize": 13,
    "axes.labelweight": "bold",
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# Color palette
PRISM_COLOR = "#2563EB"
BASELINE_COLORS = {
    "ESM2-35M": "#6B7280",
    "ESM2-650M": "#374151",
    "AbLang2": "#D97706",
    "AntiBERTy": "#059669",
    "Sapiens": "#7C3AED",
}
DATASET_COLORS = {
    "FLAb2 Remaining": "#8B5CF6",
    "CR9114": "#EF4444",
    "G6.31": "#3B82F6",
    "Trastuzumab": "#10B981",
    "FLAb2 All": "#F59E0B",
}
MUT_GROUP_COLORS = {
    "1": "#3B82F6",
    "2-5": "#10B981",
    ">5": "#EF4444",
}

METHOD_DISPLAY = {
    "prism_origin_logit_sum": "PRISM origin_logit",
    "prism_origin_prob_sum": "PRISM origin_prob",
    "prism_alpha_sum": "PRISM alpha",
    "prism_ngl_logprob_sum": "PRISM ngl_logprob",
    "esm2_35m_score": "ESM2-35M",
    "esm2_650m_score": "ESM2-650M",
    "ablang2_score": "AbLang2",
    "antiberty_score": "AntiBERTy",
    "sapiens_score": "Sapiens",
}
METHOD_COLORS = {
    "prism_origin_logit_sum": "#1D4ED8",
    "prism_origin_prob_sum": "#2563EB",
    "prism_alpha_sum": "#3B82F6",
    "prism_ngl_logprob_sum": "#60A5FA",
    "esm2_35m_score": "#6B7280",
    "esm2_650m_score": "#374151",
    "ablang2_score": "#D97706",
    "antiberty_score": "#059669",
    "sapiens_score": "#7C3AED",
}
PRISM_METHODS = ["prism_origin_logit_sum", "prism_origin_prob_sum",
                 "prism_alpha_sum", "prism_ngl_logprob_sum"]
ALL_METHODS = list(METHOD_DISPLAY.keys())


def _save_panel(fig, panels_dir, name):
    """Save a single-panel figure."""
    path = panels_dir / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"    Panel: {path.name}")


def pretty_method_name(name: str) -> str:
    """Convert method code to readable label."""
    replacements = {
        "A09_origin_prob": "Origin Prob",
        "A10_origin_logit": "Origin Logit",
        "A01_ngl_logprob": "NGL LogProb",
        "A04_ngl_minus_gl": "NGL-GL",
        "A08_ngl_mut_minus_gl_wt": "NGL(mut)-GL(wt)",
        "A11_alpha": "Alpha",
        "A06_llr_gl": "LLR GL",
        "A07_llr_marg": "LLR Marg",
        "A14_aa_head_llr": "AA Head LLR",
        "A13_origin_adaptive_llr": "Origin Adapt. LLR",
        "A05_llr_ngl": "LLR NGL",
        "A12_alpha_weighted_llr": "Alpha-W LLR",
        "A02_gl_logprob": "GL LogProb",
        "A03_marg_logprob": "Marg LogProb",
        "unmasked_mutant": "unmasked",
        "masked_individual": "masked",
        "masked_all_at_once": "masked-all",
        "unmasked_germline": "unmasked-GL",
        "unmasked_wt": "unmasked-WT",
        "N3_gl_center": "GL-ctr",
        "N2_wt_center": "WT-ctr",
        "N1_none": "raw",
        "M1_sum": "sum",
        "M2_mean": "mean",
        "M4_max": "max",
        "M3_min": "min",
    }
    parts = name.split("__")
    labels = []
    for p in parts:
        for k, v in replacements.items():
            p = p.replace(k, v)
        labels.append(p)
    return " | ".join(labels)


def load_per_dataset_results(eval_dir: Path, model: str = "v34.1b"):
    datasets = {
        "CR9114": "cr9114_all_results.csv",
        "G6.31": "g6.31_all_results.csv",
        "Trastuzumab": "trastuzumab_all_results.csv",
    }
    results = {}
    for name, fname in datasets.items():
        path = eval_dir / model / fname
        if path.exists():
            results[name] = pd.read_csv(path)
    return results


def compute_per_dataset_ranks(dataset_results: dict):
    rank_dfs = []
    for ds_name, df in dataset_results.items():
        temp = df[["method", "spearman_rho"]].copy()
        temp[f"rank_{ds_name}"] = temp["spearman_rho"].rank(ascending=False, method="average")
        temp = temp.rename(columns={"spearman_rho": f"rho_{ds_name}"})
        rank_dfs.append(temp.set_index("method")[[f"rank_{ds_name}", f"rho_{ds_name}"]])
    merged = pd.concat(rank_dfs, axis=1, join="inner")
    rank_cols = [c for c in merged.columns if c.startswith("rank_")]
    merged["mean_rank"] = merged[rank_cols].mean(axis=1)
    merged["std_rank"] = merged[rank_cols].std(axis=1)
    merged = merged.sort_values("mean_rank")
    return merged


# =========================================================================
# Figure 1 panels
# =========================================================================
def _draw_fig1a_heatmap(ax, top, mat_rank, ds_names, n_methods_total, n_top):
    """Fig1A: Per-dataset rank heatmap."""
    im = ax.imshow(mat_rank, aspect="auto", cmap="RdBu_r",
                   vmin=1, vmax=n_methods_total * 0.35)
    cmap = plt.cm.RdBu_r
    norm = plt.Normalize(vmin=1, vmax=n_methods_total * 0.35)
    for i in range(mat_rank.shape[0]):
        for j in range(mat_rank.shape[1]):
            rank_val = mat_rank[i, j]
            rgba = cmap(norm(rank_val))
            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            color = "white" if luminance < 0.5 else "#1E293B"
            ax.text(j, i, f"{int(rank_val)}", ha="center", va="center",
                    fontsize=14, color=color, fontweight="bold")
    n_var = {"CR9114": "65,535", "G6.31": "4,274", "Trastuzumab": "36,496"}
    ax.set_xticks(range(3))
    ax.set_xticklabels([f"{d}\n({n_var[d]} var)" for d in ds_names],
                       fontsize=12, fontweight="bold")
    ax.set_yticks(range(n_top))
    ax.set_yticklabels(top["label"], fontsize=11, fontweight="bold")
    ax.set_title(f"Top {n_top} Scoring Methods by Average Rank\n"
                 f"(out of {n_methods_total} methods)", fontsize=13, fontweight="bold", pad=12)
    return im


def _draw_fig1b_meanrank(ax, top, n_top):
    """Fig1B: Mean rank bar chart."""
    mean_ranks = top["mean_rank"].values
    std_ranks = top["std_rank"].values
    bar_colors = [PRISM_COLOR if r <= mean_ranks[2] else "#94A3B8" for r in mean_ranks]
    ax.barh(range(n_top), mean_ranks, xerr=std_ranks, color=bar_colors,
            edgecolor="white", linewidth=0.5,
            capsize=3, error_kw={"lw": 1.2, "capthick": 1.2})
    ax.set_yticks(range(n_top))
    ax.set_yticklabels(top["label"], fontsize=9, fontweight="bold")
    ax.set_xlabel("Mean Rank +/- Std", fontsize=12, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlim(0, max(mean_ranks + std_ranks) * 1.2)
    for i, (mr, sr) in enumerate(zip(mean_ranks, std_ranks)):
        ax.text(mr + sr + 1, i, f"{mr:.1f}", va="center", fontsize=10, fontweight="bold")
    ax.set_title("Mean Rank (+/- Std)", fontsize=13, fontweight="bold")


def plot_exhaustive_scoring(eval_dir, out_dir, panels_dir, n_top=10):
    dataset_results = load_per_dataset_results(eval_dir)
    if len(dataset_results) < 3:
        print("  [SKIP] Not all 3 dataset results found")
        return
    merged = compute_per_dataset_ranks(dataset_results)
    top = merged.head(n_top).copy()
    top["label"] = [f"#{i+1}  {pretty_method_name(m)}" for i, m in enumerate(top.index)]
    n_methods_total = len(merged)
    ds_names = list(dataset_results.keys())
    rank_cols = [f"rank_{d}" for d in ds_names]
    mat_rank = top[rank_cols].values

    # Combined figure
    fig = plt.figure(figsize=(14, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1], wspace=0.05)
    ax_a = fig.add_subplot(gs[0])
    im = _draw_fig1a_heatmap(ax_a, top, mat_rank, ds_names, n_methods_total, n_top)
    fig.colorbar(im, ax=ax_a, shrink=0.6, pad=0.02).set_label(
        "Per-Dataset Rank (lower = better)", fontsize=11, fontweight="bold")
    ax_b = fig.add_subplot(gs[1])
    mean_ranks = top["mean_rank"].values
    std_ranks = top["std_rank"].values
    bar_colors = [PRISM_COLOR if r <= mean_ranks[2] else "#94A3B8" for r in mean_ranks]
    ax_b.barh(range(n_top), mean_ranks, xerr=std_ranks, color=bar_colors,
              edgecolor="white", linewidth=0.5,
              capsize=3, error_kw={"lw": 1.2, "capthick": 1.2})
    ax_b.set_yticks([])
    ax_b.set_xlabel("Mean Rank +/- Std", fontsize=12, fontweight="bold")
    ax_b.invert_yaxis()
    ax_b.set_xlim(0, max(mean_ranks + std_ranks) * 1.2)
    for i, (mr, sr) in enumerate(zip(mean_ranks, std_ranks)):
        ax_b.text(mr + sr + 1, i, f"{mr:.1f}", va="center", fontsize=10, fontweight="bold")
    fig.suptitle(f"PRISM v34.1b -- Top {n_top} Scoring Methods (3 DMS Datasets)",
                 fontsize=15, fontweight="bold", y=1.02)
    out_path = out_dir / "fig1_exhaustive_scoring_top10_rank.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")

    # Individual panels
    fig_a, ax = plt.subplots(figsize=(12, 6))
    im = _draw_fig1a_heatmap(ax, top, mat_rank, ds_names, n_methods_total, n_top)
    fig_a.colorbar(im, ax=ax, shrink=0.6, pad=0.02).set_label(
        "Per-Dataset Rank (lower = better)", fontsize=11, fontweight="bold")
    _save_panel(fig_a, panels_dir, "fig1a_heatmap")

    fig_b, ax = plt.subplots(figsize=(8, 6))
    _draw_fig1b_meanrank(ax, top, n_top)
    _save_panel(fig_b, panels_dir, "fig1b_mean_rank")


# =========================================================================
# Figure 2 panels
# =========================================================================
def _get_fig2_data(ranking_csv, rho_matrix_csv):
    df_rank = pd.read_csv(ranking_csv, index_col=0)
    df_rho_full = pd.read_csv(rho_matrix_csv, index_col=0)
    n_proteins = len(df_rho_full)
    methods = df_rank.index.tolist()
    short_names = []
    for m in methods:
        if "PRISM" in m:
            short_names.append("PRISM v34.1b")
        else:
            short_names.append(m.replace("_LLR", ""))
    colors = []
    for m in methods:
        if "PRISM" in m:
            colors.append(PRISM_COLOR)
        else:
            for bname, bcol in BASELINE_COLORS.items():
                if bname.lower().replace("-", "") in m.lower().replace("-", ""):
                    colors.append(bcol)
                    break
    rho_cols = [c for c in df_rho_full.columns if c in methods]
    df_rho = df_rho_full[rho_cols].dropna()
    rank_mat = df_rho.rank(axis=1, ascending=False, method="average")
    return df_rank, df_rho, rank_mat, methods, short_names, colors, n_proteins


def _draw_fig2a_avg_rank(ax, df_rank, methods, short_names, colors):
    avg_ranks = df_rank["avg_rank"].values
    std_ranks = df_rank["std_rank"].values
    ax.barh(range(len(methods)), avg_ranks, xerr=std_ranks,
            color=colors, edgecolor="white", linewidth=0.8,
            capsize=3, error_kw={"lw": 1.2, "capthick": 1.2})
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(short_names, fontsize=10)
    ax.set_xlabel("Average Rank (lower = better)", fontsize=11)
    ax.set_title("Average Per-Protein Rank", fontsize=12, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlim(0, 6.5)
    ax.axvline(x=3.5, color="gray", ls="--", lw=0.8, alpha=0.5)
    for i, m in enumerate(methods):
        wr = df_rank.loc[m, "win_rate"]
        fp = df_rank.loc[m, "frac_positive"]
        ax.text(6.3, i, f"Win: {wr:.0%}  rho>0: {fp:.0%}",
                va="center", fontsize=8, color="#475569")


def _draw_fig2b_boxplot(ax, rank_mat, methods, short_names, colors, n_proteins):
    rank_data = [rank_mat[m].values if m in rank_mat.columns else np.array([])
                 for m in methods]
    bp = ax.boxplot(rank_data, vert=False, patch_artist=True,
                    widths=0.6, showfliers=True,
                    flierprops=dict(marker="o", markersize=3, alpha=0.4),
                    medianprops=dict(color="black", lw=1.5))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    ax.set_yticks(range(1, len(methods) + 1))
    ax.set_yticklabels(short_names, fontsize=10)
    ax.set_xlabel("Per-Protein Rank (1 = best)", fontsize=11)
    ax.set_title(f"Rank Distribution ({n_proteins} Proteins)", fontsize=12, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlim(0.5, 6.5)


def _draw_fig2c_violin(ax, df_rho, methods, short_names, colors):
    rho_data = [df_rho[m].dropna().values if m in df_rho.columns else np.array([])
                for m in methods]
    vp = ax.violinplot(rho_data, positions=range(len(methods)),
                       vert=False, showmedians=True, showextrema=False)
    for i, body in enumerate(vp["bodies"]):
        body.set_facecolor(colors[i])
        body.set_alpha(0.5)
    vp["cmedians"].set_color("black")
    vp["cmedians"].set_linewidth(1.5)
    for i, vals in enumerate(rho_data):
        if len(vals) == 0:
            continue
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(vals))
        ax.scatter(vals, i + jitter, s=8, alpha=0.4, color=colors[i], edgecolors="none")
    ax.axvline(x=0, color="red", ls="--", lw=1, alpha=0.7)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(short_names, fontsize=10)
    ax.set_xlabel("Signed Spearman rho (per protein)", fontsize=11)
    ax.set_title("Per-Protein Correlation Distribution", fontsize=12, fontweight="bold")
    ax.invert_yaxis()


def _draw_fig2d_table(ax, df_rank, methods, short_names):
    ax.axis("off")
    col_labels = ["Method", "Avg Rank", "Std", "Win%", "Top3%", "Mean rho", "rho>0"]
    table_data = []
    for i, m in enumerate(methods):
        r = df_rank.loc[m]
        table_data.append([
            short_names[i],
            f"{r['avg_rank']:.2f}",
            f"{r['std_rank']:.2f}",
            f"{r['win_rate']:.0%}",
            f"{r['top3_rate']:.0%}",
            f"{r['mean_rho']:+.3f}",
            f"{r['frac_positive']:.0%}",
        ])
    table = ax.table(cellText=table_data, colLabels=col_labels,
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#1E293B")
        table[0, j].set_text_props(color="white", fontweight="bold")
    for j in range(len(col_labels)):
        table[1, j].set_facecolor("#DBEAFE")
    ax.set_title("Summary Statistics (Signed rho Ranking)", fontsize=12, fontweight="bold", pad=20)


def plot_flab2_ranking(ranking_csv, rho_matrix_csv, out_dir, panels_dir):
    data = _get_fig2_data(ranking_csv, rho_matrix_csv)
    df_rank, df_rho, rank_mat, methods, short_names, colors, n_proteins = data

    # Combined figure
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)
    _draw_fig2a_avg_rank(fig.add_subplot(gs[0, 0]), df_rank, methods, short_names, colors)
    _draw_fig2b_boxplot(fig.add_subplot(gs[0, 1]), rank_mat, methods, short_names, colors, n_proteins)
    _draw_fig2c_violin(fig.add_subplot(gs[1, 0]), df_rho, methods, short_names, colors)
    _draw_fig2d_table(fig.add_subplot(gs[1, 1]), df_rank, methods, short_names)
    fig.suptitle(f"PRISM v34.1b vs Baselines -- FLAb2 ({n_proteins} proteins)",
                 fontsize=14, fontweight="bold", y=1.02)
    out_path = out_dir / "fig2_flab2_ranking_comparison.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")

    # Individual panels
    fig, ax = plt.subplots(figsize=(8, 5))
    _draw_fig2a_avg_rank(ax, df_rank, methods, short_names, colors)
    _save_panel(fig, panels_dir, "fig2a_avg_rank")

    fig, ax = plt.subplots(figsize=(8, 5))
    _draw_fig2b_boxplot(ax, rank_mat, methods, short_names, colors, n_proteins)
    _save_panel(fig, panels_dir, "fig2b_rank_boxplot")

    fig, ax = plt.subplots(figsize=(8, 5))
    _draw_fig2c_violin(ax, df_rho, methods, short_names, colors)
    _save_panel(fig, panels_dir, "fig2c_rho_violin")

    fig, ax = plt.subplots(figsize=(8, 5))
    _draw_fig2d_table(ax, df_rank, methods, short_names)
    _save_panel(fig, panels_dir, "fig2d_summary_table")


# =========================================================================
# Figure 3 panels
# =========================================================================
def _prepare_fig3_data(prism_csv, repo_root):
    df_prism = pd.read_csv(prism_csv)
    df_p = df_prism[df_prism["model"] == "v34.1b"].copy()
    benchmark = repo_root / "data" / "prism_results" / "3.zero-shot"
    stratified_dir = repo_root / "data" / "features" / "evaluation_results" / "flab2_binding" / "stratified"

    # Histogram data
    hist_data = {}
    flab2_rem_path = stratified_dir / "flab2_remaining_per_variant_scores.csv"
    if flab2_rem_path.exists():
        hist_data["FLAb2 Remaining"] = pd.read_csv(flab2_rem_path, usecols=["n_mut"])["n_mut"].values
    for label, key in {"G6.31": "g6.31", "CR9114": "cr9114", "Trastuzumab": "trastuzumab"}.items():
        csv_path = benchmark / f"{key}_benchmark_data_clean.csv"
        if csv_path.exists():
            hist_data[label] = pd.read_csv(csv_path, usecols=["n_mut"])["n_mut"].values

    # Mutation stats
    mut_stats_path = stratified_dir / "mutation_stats_all_groups.csv"
    mstats = None
    if mut_stats_path.exists():
        mstats = pd.read_csv(mut_stats_path)
        group_order = ["FLAb2 All", "FLAb2 Remaining", "G6.31", "CR9114", "Trastuzumab"]
        mstats = mstats.set_index("group").loc[
            [g for g in group_order if g in mstats["group"].values]
        ].reset_index()

    return df_p, hist_data, mstats


def _draw_fig3a_variants(ax, df_p):
    n_variants = df_p["n_variants"].values
    proteins = df_p["source_file"].str.replace(".csv", "", regex=False).values
    sort_idx = np.argsort(n_variants)[::-1]
    n_sorted = n_variants[sort_idx]
    p_sorted = proteins[sort_idx]
    assay_types = df_p["assay_type"].values[sort_idx]
    assay_colors_map = {
        "SPR Kd": "#3B82F6", "BLI Kd": "#06B6D4", "Flow Kd": "#8B5CF6",
        "Tite-Seq Kd": "#EC4899", "MAGMA-Seq Kd": "#F97316",
        "IC50": "#EF4444", "EC50": "#10B981",
    }
    bar_colors = [assay_colors_map.get(a, "#94A3B8") for a in assay_types]
    ax.barh(range(len(n_sorted)), n_sorted, color=bar_colors, edgecolor="white", linewidth=0.3)
    ax.set_yticks(range(len(n_sorted)))
    short_names = []
    for p in p_sorted:
        parts = p.split("_")
        short_names.append(f"{parts[0]}_{parts[1][:10]}" if len(parts) >= 2 else p[:20])
    ax.set_yticklabels(short_names, fontsize=5)
    ax.set_xlabel("Number of Variants", fontsize=11)
    ax.set_title(f"Variants per Protein (n={len(n_sorted)})", fontsize=12, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    total_v = n_sorted.sum()
    ax.text(0.95, 0.95, f"Total: {total_v:,} variants\n{len(n_sorted)} proteins",
            transform=ax.transAxes, ha="right", va="top", fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#DBEAFE", alpha=0.9))
    legend_patches = [Patch(facecolor=c, label=a) for a, c in assay_colors_map.items()
                      if a in assay_types]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=7, framealpha=0.9,
              title="Assay Type", title_fontsize=8)


def _draw_fig3b_assay(ax, df_p):
    n_variants = df_p["n_variants"].values
    sort_idx = np.argsort(n_variants)[::-1]
    n_sorted = n_variants[sort_idx]
    assay_types = df_p["assay_type"].values[sort_idx]
    assay_colors_map = {
        "SPR Kd": "#3B82F6", "BLI Kd": "#06B6D4", "Flow Kd": "#8B5CF6",
        "Tite-Seq Kd": "#EC4899", "MAGMA-Seq Kd": "#F97316",
        "IC50": "#EF4444", "EC50": "#10B981",
    }
    assay_counts = pd.Series(assay_types).value_counts()
    assay_variant_sums = {}
    for i, a in enumerate(assay_types):
        assay_variant_sums[a] = assay_variant_sums.get(a, 0) + n_sorted[i]
    pie_colors = [assay_colors_map.get(a, "#94A3B8") for a in assay_counts.index]
    wedges, texts, autotexts = ax.pie(
        assay_counts.values, labels=None,
        autopct=lambda pct: f"{int(round(pct * len(assay_types) / 100))}",
        colors=pie_colors, startangle=90,
        textprops={"fontsize": 9, "fontweight": "bold"},
    )
    for autotext in autotexts:
        autotext.set_color("white")
    ax.legend(
        [f"{a} ({assay_variant_sums[a]:,} var)" for a in assay_counts.index],
        loc="center left", bbox_to_anchor=(0.9, 0.5), fontsize=8,
        title="Assay (total variants)", title_fontsize=9,
    )
    ax.set_title("Assay Type Distribution", fontsize=12, fontweight="bold")


def _draw_fig3c_studies(ax, df_p):
    study_df = pd.DataFrame({
        "study": df_p["study"].values,
        "n_variants": df_p["n_variants"].values,
        "source_file": df_p["source_file"].values,
    })
    study_agg = study_df.groupby("study").agg(
        n_proteins=("source_file", "nunique"),
        total_variants=("n_variants", "sum"),
    ).sort_values("total_variants", ascending=True)
    study_labels = []
    for s in study_agg.index:
        words = s.split()
        label = words[0] if words else s
        for w in words:
            if len(w) == 4 and w.isdigit():
                label = f"{words[0]} {w}"
                break
        study_labels.append(label[:30])
    y_pos = range(len(study_agg))
    ax.barh(y_pos, study_agg["total_variants"], color="#3B82F6", alpha=0.7, edgecolor="white")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(study_labels, fontsize=7)
    ax.set_xlabel("Total Variants", fontsize=11)
    ax.set_title(f"Studies ({len(study_agg)} unique)", fontsize=12, fontweight="bold")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    for i, (_, row) in enumerate(study_agg.iterrows()):
        ax.text(row["total_variants"] * 1.1, i, f"{row['n_proteins']}p",
                va="center", fontsize=7, color="#475569")


def _draw_fig3d_histogram_faceted(axes, hist_data):
    """Draw 4 individual histograms (one per dataset) on a 2x2 axes array."""
    if not hist_data:
        return
    max_mut = max(v.max() for v in hist_data.values())
    bins = np.arange(0.5, min(max_mut + 1.5, 18.5), 1)
    axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    labels = list(hist_data.keys())
    for i, (label, vals) in enumerate(hist_data.items()):
        if i >= len(axes_flat):
            break
        ax = axes_flat[i]
        ax.hist(vals, bins=bins, alpha=0.75,
                color=DATASET_COLORS.get(label, "#94A3B8"),
                edgecolor="white", linewidth=0.5)
        ax.set_yscale("log")
        ax.set_title(f"{label} (n={len(vals):,})", fontsize=10, fontweight="bold")
        ax.tick_params(labelsize=8)
        if i >= 2:  # bottom row
            ax.set_xlabel("Mutations per Variant", fontsize=9)
        if i % 2 == 0:  # left column
            ax.set_ylabel("Count", fontsize=9)
    # Hide unused axes
    for i in range(len(hist_data), len(axes_flat)):
        axes_flat[i].set_visible(False)


def _draw_fig3e_mut_groups(ax, mstats):
    if mstats is None:
        ax.text(0.5, 0.5, "Data not found", transform=ax.transAxes, ha="center", va="center")
        return
    x = np.arange(len(mstats))
    width = 0.25
    bars1 = ax.bar(x - width, mstats["n_1"], width, label="1 mutation",
                   color=MUT_GROUP_COLORS["1"], edgecolor="white")
    bars2 = ax.bar(x, mstats["n_2_5"], width, label="2-5 mutations",
                   color=MUT_GROUP_COLORS["2-5"], edgecolor="white")
    bars3 = ax.bar(x + width, mstats["n_gt5"], width, label=">5 mutations",
                   color=MUT_GROUP_COLORS[">5"], edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(mstats["group"], fontsize=8, rotation=15, ha="right")
    ax.set_ylabel("Count", fontsize=11)
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.set_title("Mutation Groups (All Analysis Groups)", fontsize=12, fontweight="bold")
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h * 1.15,
                        f"{int(h):,}", ha="center", va="bottom", fontsize=6,
                        fontweight="bold", rotation=45)


def _draw_fig3f_table(ax, mstats):
    ax.axis("off")
    if mstats is None:
        return
    table_rows = []
    for _, row in mstats.iterrows():
        table_rows.append([
            row["group"],
            f"{int(row['total']):,}",
            f"{row['mean']:.1f}",
            f"{int(row['median'])}",
            f"{int(row['n_1']):,}",
            f"{int(row['n_2_5']):,}",
            f"{int(row['n_gt5']):,}",
        ])
    col_labels = ["Group", "Total", "Mean", "Med", "1 mut", "2-5", ">5"]
    table = ax.table(cellText=table_rows, colLabels=col_labels,
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.8)
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#1E293B")
        table[0, j].set_text_props(color="white", fontweight="bold")
    for j in range(len(col_labels)):
        table[1, j].set_facecolor("#DBEAFE")
    ax.set_title("Mutation Statistics (All Groups)", fontsize=12, fontweight="bold", pad=20)


def plot_flab2_statistics(prism_csv, baseline_csv, out_dir, panels_dir, repo_root=None):
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[3]
    df_p, hist_data, mstats = _prepare_fig3_data(prism_csv, repo_root)

    # Combined figure
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.35)
    _draw_fig3a_variants(fig.add_subplot(gs[0, 0]), df_p)
    _draw_fig3b_assay(fig.add_subplot(gs[0, 1]), df_p)
    _draw_fig3c_studies(fig.add_subplot(gs[0, 2]), df_p)
    # Panel D: nested 2x2 for faceted histograms
    gs_d = gs[1, 0].subgridspec(2, 2, hspace=0.4, wspace=0.3)
    axes_d = np.array([[fig.add_subplot(gs_d[r, c]) for c in range(2)] for r in range(2)])
    _draw_fig3d_histogram_faceted(axes_d, hist_data)
    _draw_fig3e_mut_groups(fig.add_subplot(gs[1, 1]), mstats)
    _draw_fig3f_table(fig.add_subplot(gs[1, 2]), mstats)
    fig.suptitle("FLAb2 Benchmark Dataset Overview", fontsize=14, fontweight="bold", y=1.02)
    out_path = out_dir / "fig3_flab2_dataset_statistics.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")

    # Individual panels
    fig, ax = plt.subplots(figsize=(8, 10))
    _draw_fig3a_variants(ax, df_p)
    _save_panel(fig, panels_dir, "fig3a_variants_per_protein")

    fig, ax = plt.subplots(figsize=(8, 6))
    _draw_fig3b_assay(ax, df_p)
    _save_panel(fig, panels_dir, "fig3b_assay_types")

    fig, ax = plt.subplots(figsize=(8, 6))
    _draw_fig3c_studies(ax, df_p)
    _save_panel(fig, panels_dir, "fig3c_studies")

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.subplots_adjust(hspace=0.4, wspace=0.3)
    _draw_fig3d_histogram_faceted(axes, hist_data)
    fig.suptitle("Mutation Count Distribution", fontsize=13, fontweight="bold", y=1.02)
    _save_panel(fig, panels_dir, "fig3d_mutation_histogram")

    fig, ax = plt.subplots(figsize=(8, 5))
    _draw_fig3e_mut_groups(ax, mstats)
    _save_panel(fig, panels_dir, "fig3e_mutation_groups")

    fig, ax = plt.subplots(figsize=(8, 4))
    _draw_fig3f_table(ax, mstats)
    _save_panel(fig, panels_dir, "fig3f_mutation_stats_table")


# =========================================================================
# Figure 4 (single panel)
# =========================================================================
def plot_signal_context_heatmap(eval_dir, out_dir, panels_dir):
    dataset_results = load_per_dataset_results(eval_dir)
    if len(dataset_results) < 3:
        print("  [SKIP] Not all 3 dataset results found")
        return

    merged = compute_per_dataset_ranks(dataset_results)
    rows = []
    for method, row in merged.iterrows():
        parts = method.split("__")
        if len(parts) >= 4:
            signal, context, norm_val, agg = parts[0], parts[1], parts[2], parts[3]
            rows.append({"signal": signal, "context": context,
                         "norm": norm_val, "agg": agg, "mean_rank": row["mean_rank"]})
    df = pd.DataFrame(rows)
    pivot = df.groupby(["signal", "context"])["mean_rank"].min().reset_index()
    pivot_table = pivot.pivot(index="signal", columns="context", values="mean_rank")
    signal_order = pivot_table.min(axis=1).sort_values().index
    pivot_table = pivot_table.loc[signal_order]
    signal_labels = [pretty_method_name(s) for s in pivot_table.index]
    n_total = len(merged)

    def _draw(ax, add_colorbar_to_fig=None):
        im = ax.imshow(pivot_table.values, aspect="auto", cmap="RdYlGn_r",
                       vmin=1, vmax=n_total * 0.5)
        for i in range(pivot_table.shape[0]):
            for j in range(pivot_table.shape[1]):
                val = pivot_table.values[i, j]
                if np.isnan(val):
                    continue
                pct = val / n_total * 100
                color = "white" if val < n_total * 0.1 else "black"
                ax.text(j, i, f"{int(val)}\n(top {pct:.0f}%)", ha="center", va="center",
                        fontsize=7.5, color=color,
                        fontweight="bold" if val <= 20 else "normal")
        context_labels = [c.replace("_", " ").title() for c in pivot_table.columns]
        ax.set_xticks(range(len(context_labels)))
        ax.set_xticklabels(context_labels, fontsize=9, rotation=30, ha="right")
        ax.set_yticks(range(len(signal_labels)))
        ax.set_yticklabels(signal_labels, fontsize=9)
        ax.set_title(f"PRISM v34.1b -- Best Average Rank by Signal x Context\n"
                     f"({n_total} methods total)", fontsize=13, fontweight="bold", pad=12)
        if add_colorbar_to_fig is not None:
            cbar = add_colorbar_to_fig.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label("Average Rank (lower = better)", fontsize=10)

    # Combined (same as individual for single-panel figure)
    fig, ax = plt.subplots(figsize=(10, 7))
    _draw(ax, add_colorbar_to_fig=fig)
    out_path = out_dir / "fig4_signal_context_rank_heatmap.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")

    # Individual panel (same content)
    fig, ax = plt.subplots(figsize=(10, 7))
    _draw(ax, add_colorbar_to_fig=fig)
    _save_panel(fig, panels_dir, "fig4_signal_context_heatmap")


# =========================================================================
# Figure 5 panels
# =========================================================================
FIG5_METHODS = ["prism_origin_logit_sum", "esm2_35m_score", "esm2_650m_score",
                "ablang2_score", "antiberty_score", "sapiens_score"]
FIG5_DISPLAY = {
    "prism_origin_logit_sum": "PRISM",
    "esm2_35m_score": "ESM2-35M",
    "esm2_650m_score": "ESM2-650M",
    "ablang2_score": "AbLang2",
    "antiberty_score": "AntiBERTy",
    "sapiens_score": "Sapiens",
}
FIG5_COLORS = {
    "prism_origin_logit_sum": "#2563EB",
    "esm2_35m_score": "#6B7280",
    "esm2_650m_score": "#374151",
    "ablang2_score": "#D97706",
    "antiberty_score": "#059669",
    "sapiens_score": "#7C3AED",
}


def _draw_fig5_flab2_all(ax, ag_df):
    """Draw FLAb2 All: PRISM best vs baselines grouped bar chart."""
    groups_list = ["1", "2-5", ">5"]
    methods = FIG5_METHODS
    n_methods = len(methods)
    n_groups = len(groups_list)
    width = 0.8 / n_methods

    for mi, method in enumerate(methods):
        mdf = ag_df[ag_df["method"] == method]
        rhos = []
        for g in groups_list:
            gdf = mdf[mdf["mut_group"] == g]
            if len(gdf) > 0 and gdf.iloc[0]["n_variants"] >= 10:
                rho_val = gdf.iloc[0]["rho"]
                rhos.append(rho_val if np.isfinite(rho_val) else np.nan)
            else:
                rhos.append(np.nan)
        x = np.arange(n_groups)
        offset = (mi - n_methods / 2 + 0.5) * width
        bars = ax.bar(x + offset, rhos, width * 0.9,
                      label=FIG5_DISPLAY[method],
                      color=FIG5_COLORS[method],
                      edgecolor="white", linewidth=0.5)
        # Add value labels on bars
        for bar, rho in zip(bars, rhos):
            if not np.isnan(rho):
                va = "bottom" if rho >= 0 else "top"
                y_offset = 0.01 if rho >= 0 else -0.01
                ax.text(bar.get_x() + bar.get_width() / 2, rho + y_offset,
                        f"{rho:+.2f}", ha="center", va=va, fontsize=6.5,
                        fontweight="bold", rotation=90)

    # X-axis labels with n
    group_labels = []
    for g in groups_list:
        gdf = ag_df[ag_df["mut_group"] == g]
        if len(gdf) > 0:
            n = int(gdf.iloc[0]["n_variants"])
            group_labels.append(f"{g} mutation{'s' if g != '1' else ''}\n(n={n:,})")
        else:
            group_labels.append(f"{g} mutation{'s' if g != '1' else ''}")

    ax.set_xticks(np.arange(n_groups))
    ax.set_xticklabels(group_labels, fontsize=11)
    ax.set_ylabel("Spearman rho", fontsize=12, fontweight="bold")
    ax.set_title("FLAb2 All (45 proteins, 189K variants)\nPRISM vs Baselines by Mutation Count",
                 fontsize=13, fontweight="bold")
    ax.axhline(y=0, color="black", lw=1, ls="-")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=9, loc="upper left", framealpha=0.9)


def plot_stratified_results(stratified_csv, out_dir, panels_dir):
    df = pd.read_csv(stratified_csv)

    ag_df = df[df["analysis_group"] == "FLAb2 All"]
    if len(ag_df) == 0:
        print("  [SKIP] No FLAb2 All data found")
        return

    # Combined figure (single panel)
    fig, ax = plt.subplots(figsize=(10, 7))
    _draw_fig5_flab2_all(ax, ag_df)
    out_path = out_dir / "fig5_stratified_mutation_count.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")

    # Individual panel (same)
    fig, ax = plt.subplots(figsize=(10, 7))
    _draw_fig5_flab2_all(ax, ag_df)
    _save_panel(fig, panels_dir, "fig5_flab2_all")

    # Stratified table
    _plot_stratified_table(df, out_dir, panels_dir)


def _plot_stratified_table(df, out_dir, panels_dir):
    analysis_groups = ["FLAb2 All", "FLAb2 Remaining", "G6.31",
                       "CR9114 H1", "CR9114 H3", "Trastuzumab"]
    available_groups = [g for g in analysis_groups if g in df["analysis_group"].unique()]
    groups_list = ["1", "2-5", ">5"]

    col_labels = []
    for ag in available_groups:
        for g in groups_list:
            col_labels.append(f"{ag}\n{g} mut")

    table_data = []
    cell_colors = []
    for method in ALL_METHODS:
        row = [METHOD_DISPLAY[method]]
        row_colors = ["#F8FAFC"]
        for ag in available_groups:
            for g in groups_list:
                mdf = df[(df["analysis_group"] == ag) &
                         (df["method"] == method) &
                         (df["mut_group"] == g)]
                if len(mdf) > 0 and mdf.iloc[0]["n_variants"] >= 10:
                    rho = mdf.iloc[0]["rho"]
                    if np.isfinite(rho):
                        row.append(f"{rho:+.3f}")
                        if rho > 0.2:
                            row_colors.append("#BBF7D0")
                        elif rho > 0:
                            row_colors.append("#DCFCE7")
                        elif rho > -0.2:
                            row_colors.append("#FEE2E2")
                        else:
                            row_colors.append("#FECACA")
                    else:
                        row.append("N/A")
                        row_colors.append("#F1F5F9")
                else:
                    row.append("-")
                    row_colors.append("#F1F5F9")
        table_data.append(row)
        cell_colors.append(row_colors)

    all_col_labels = ["Method"] + col_labels
    fig_width = max(20, len(all_col_labels) * 1.2)
    fig, ax = plt.subplots(figsize=(fig_width, 8))
    ax.axis("off")
    table = ax.table(cellText=table_data, colLabels=all_col_labels,
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.5)
    for j in range(len(all_col_labels)):
        table[0, j].set_facecolor("#1E293B")
        table[0, j].set_text_props(color="white", fontweight="bold", fontsize=6)
    for i, row_colors in enumerate(cell_colors):
        for j, color in enumerate(row_colors):
            table[i + 1, j].set_facecolor(color)
        if "PRISM" in table_data[i][0]:
            for j in range(len(all_col_labels)):
                table[i + 1, j].set_text_props(fontweight="bold")
    ax.set_title("Stratified Spearman rho by Mutation Count -- All Analysis Groups\n"
                 "(green = positive, red = negative, '-' = <10 variants)",
                 fontsize=13, fontweight="bold", pad=20)

    out_path = out_dir / "fig5b_stratified_table.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")

    # Also save as individual panel (same content)
    fig2, ax2 = plt.subplots(figsize=(fig_width, 8))
    ax2.axis("off")
    table2 = ax2.table(cellText=table_data, colLabels=all_col_labels,
                       loc="center", cellLoc="center")
    table2.auto_set_font_size(False)
    table2.set_fontsize(7)
    table2.scale(1.0, 1.5)
    for j in range(len(all_col_labels)):
        table2[0, j].set_facecolor("#1E293B")
        table2[0, j].set_text_props(color="white", fontweight="bold", fontsize=6)
    for i, row_colors in enumerate(cell_colors):
        for j, color in enumerate(row_colors):
            table2[i + 1, j].set_facecolor(color)
        if "PRISM" in table_data[i][0]:
            for j in range(len(all_col_labels)):
                table2[i + 1, j].set_text_props(fontweight="bold")
    ax2.set_title("Stratified Spearman rho by Mutation Count -- All Analysis Groups\n"
                  "(green = positive, red = negative, '-' = <10 variants)",
                  fontsize=13, fontweight="bold", pad=20)
    _save_panel(fig2, panels_dir, "fig5b_stratified_table")


# =========================================================================
# Main
# =========================================================================
def main():
    parser = argparse.ArgumentParser(description="Plot binding affinity summary figures")
    parser.add_argument("--eval-dir", type=Path,
                        default=Path("data/features/evaluation_results"))
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    eval_dir = args.eval_dir
    out_dir = args.out_dir or eval_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    panels_dir = out_dir / "panels"
    panels_dir.mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).resolve().parents[3]

    print("Generating binding affinity summary figures...")
    print(f"  Output dir: {out_dir}")
    print(f"  Panels dir: {panels_dir}")

    # Figure 1
    plot_exhaustive_scoring(eval_dir, out_dir, panels_dir)

    # Figure 2
    rank_csv = eval_dir / "flab2_binding" / "ranking_comparison" / "v34.1b_vs_baselines_signed_rho.csv"
    rho_csv = eval_dir / "flab2_binding" / "ranking_comparison" / "rho_matrix_aggregated.csv"
    if rank_csv.exists():
        plot_flab2_ranking(rank_csv, rho_csv, out_dir, panels_dir)
    else:
        print(f"  [SKIP] {rank_csv} not found")

    # Figure 3
    prism_csv = eval_dir / "flab2_binding" / "per_protein_results.csv"
    baseline_csv = eval_dir / "flab2_binding" / "baseline_per_protein_results.csv"
    if prism_csv.exists():
        plot_flab2_statistics(prism_csv, baseline_csv, out_dir, panels_dir, repo_root=repo_root)
    else:
        print(f"  [SKIP] {prism_csv} not found")

    # Figure 4
    plot_signal_context_heatmap(eval_dir, out_dir, panels_dir)

    # Figure 5
    strat_csv = eval_dir / "flab2_binding" / "stratified" / "stratified_rho_full.csv"
    if strat_csv.exists():
        plot_stratified_results(strat_csv, out_dir, panels_dir)
    else:
        print(f"  [SKIP] {strat_csv} not found")

    print("\nDone!")


if __name__ == "__main__":
    main()
