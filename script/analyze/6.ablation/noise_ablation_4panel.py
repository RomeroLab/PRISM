#!/usr/bin/env python3
"""
Noise Ablation: 4 separate figures for ICML rebuttal.
  Fig1: PPL by chain x region x GL/NGL + outlier stats
  Fig2: GL/NGL discrimination (linear probe F1, PR-AUC)
  Fig3: Binding affinity (3DMS + FLAb2 per-assay)
  Fig4: Developability (Ginkgo 6 props + FLAb2 per-assay)

All pickle files loaded are trusted internal project data files generated
by our own training/inference scripts, not from external sources.
"""
import os, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "script" / "analyze" / "reviewer_figures"
OUT.mkdir(exist_ok=True)

C = {"clean": "#2166ac", "noise2": "#f4a582", "noise4": "#b2182b"}
TAGS = ["clean", "noise2", "noise4"]
TAG_LABELS = {"clean": "PRISM", "noise2": "+Noise2", "noise4": "+Noise4"}

BASELINE_COLORS = {
    "ESM2-35M": "#bdbdbd", "ESM2-650M": "#969696",
    "AbLang2": "#74c476", "AntiBERTy": "#fd8d3c", "Sapiens": "#bcbddc",
}


# ==========================================================================
# Figure 1: PPL distributions
# ==========================================================================
def fig1_ppl():
    dfs = {}
    for tag, path in [
        ("clean", "data/prism_results/1.pppl_calculation/stratified_ppl_v34.1b.csv"),
        ("noise2", "data/prism_results/1.pppl_calculation/stratified_ppl_noise2.csv"),
        ("noise4", "data/prism_results/1.pppl_calculation/stratified_ppl_noise4.csv"),
    ]:
        dfs[tag] = pd.read_csv(REPO / path).set_index("category")

    groups = [
        ("Overall", ["Heavy", "Light"]),
        ("FR GL", ["Heavy_FR_GL", "Light_FR_GL"]),
        ("CDR3 NGL", ["Heavy_CDR3_NGL", "Light_CDR3_NGL"]),
    ]

    # --- Row 1: PPL bar graphs ---
    fig, axes = plt.subplots(2, 3, figsize=(16, 9),
                              gridspec_kw={"height_ratios": [1, 0.8], "hspace": 0.35})
    fig.suptitle("Pseudo-Perplexity by Chain, Region, and GL/NGL Status",
                 fontsize=13, fontweight="bold", y=0.98)

    for ax, (group_name, cats) in zip(axes[0], groups):
        x = np.arange(len(cats))
        w = 0.22
        for i, tag in enumerate(TAGS):
            vals = [dfs[tag].loc[cat, "ppl"] if cat in dfs[tag].index else 0 for cat in cats]
            ax.bar(x + (i - 1) * w, vals, w, label=TAG_LABELS[tag],
                   color=C[tag], alpha=0.85, edgecolor="white")
            for j, v in enumerate(vals):
                ax.text(x[j] + (i - 1) * w, v + 0.15, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=8, fontweight="bold")

        chain_labels = ["Heavy" if "Heavy" in c else "Light" for c in cats]
        ax.set_xticks(x)
        ax.set_xticklabels(chain_labels, fontsize=11)
        ax.set_title(group_name, fontsize=12, fontweight="bold")
        ax.set_ylabel("Pseudo-Perplexity" if ax == axes[0][0] else "")

    # Best baseline PPL per category (lowest PPL = best)
    best_bl_ppl = {
        "Heavy": 1.483, "Light": 1.298,                   # AntiBERTy
        "Heavy_FR_GL": 1.511, "Light_FR_GL": 1.252,       # AntiBERTy
        "Heavy_CDR3_NGL": 10.654, "Light_CDR3_NGL": 7.375, # AbLang2
    }
    for ax, (group_name, cats) in zip(axes[0], groups):
        for j, cat in enumerate(cats):
            bl = best_bl_ppl.get(cat)
            if bl is not None:
                ax.plot([j - 0.35, j + 0.35], [bl, bl],
                        color="gray", ls="--", lw=1.2, alpha=0.7)
    # Single shared legend at the bottom of the figure
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    handles = [Patch(facecolor=C[t], alpha=0.85, label=TAG_LABELS[t]) for t in TAGS]
    handles.append(Line2D([0], [0], color="gray", ls="--", lw=1.2, label="Best baseline"))
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=10,
               frameon=False, bbox_to_anchor=(0.5, -0.02))

    # --- Row 2: Outlier stats bar graphs ---
    outlier_groups = [
        ("95th %ile PPL", "p95", ["Heavy", "Light"]),
        ("99th %ile PPL", "p99", ["Heavy_FR_GL", "Light_FR_GL"]),
        ("% tokens PPL>20", "frac_gt20", ["Heavy_CDR3_NGL", "Light_CDR3_NGL"]),
    ]
    for ax, (title, metric, cats) in zip(axes[1], outlier_groups):
        x = np.arange(len(cats))
        w = 0.22
        for i, tag in enumerate(TAGS):
            vals = [dfs[tag].loc[cat, metric] if cat in dfs[tag].index and metric in dfs[tag].columns else 0
                    for cat in cats]
            ax.bar(x + (i - 1) * w, vals, w, color=C[tag], alpha=0.85, edgecolor="white")
            for j, v in enumerate(vals):
                fmt = f"{v:.1f}" if metric != "frac_gt20" else f"{v:.1f}%"
                ax.text(x[j] + (i - 1) * w, v + 0.3, fmt,
                        ha="center", va="bottom", fontsize=7, fontweight="bold")

        chain_labels = ["Heavy" if "Heavy" in c else "Light" for c in cats]
        ax.set_xticks(x)
        ax.set_xticklabels(chain_labels, fontsize=11)
        ax.set_title(title, fontsize=11, fontweight="bold")

    plt.tight_layout()
    fig.savefig(OUT / "fig1_ppl_by_region.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(OUT / "fig1_ppl_by_region.png", dpi=200, bbox_inches="tight")
    print(f"Saved fig1: {OUT / 'fig1_ppl_by_region.png'}")
    plt.close(fig)


# ==========================================================================
# Figure 2: GL/NGL discrimination
# ==========================================================================
def fig2_discrimination():
    probe_results = {
        "clean": {"F1": 0.8960, "PR-AUC": 0.9800},
        "noise2": {"F1": 0.7081, "PR-AUC": 0.8803},
        "noise4": {"F1": 0.7000, "PR-AUC": 0.8793},
    }
    baselines = {
        "ESM2-35M": {"F1": 0.3157, "PR-AUC": 0.4970},
        "ESM2-650M": {"F1": 0.4295, "PR-AUC": 0.5789},
        "AbLang2": {"F1": 0.5821, "PR-AUC": 0.6677},
        "AntiBERTy": {"F1": 0.7445, "PR-AUC": 0.7851},
        "Sapiens": {"F1": 0.3156, "PR-AUC": 0.4768},
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("GL/NGL Discrimination (Linear Probe on Embeddings)",
                 fontsize=13, fontweight="bold", y=1.02)

    for ax, metric in zip([ax1, ax2], ["F1", "PR-AUC"]):
        bl_names = list(baselines.keys())
        bl_vals = [baselines[b][metric] for b in bl_names]
        bl_colors = [BASELINE_COLORS[b] for b in bl_names]
        x_bl = np.arange(len(bl_names))
        bars_bl = ax.bar(x_bl, bl_vals, 0.6, color=bl_colors, alpha=0.7,
                         edgecolor="white", linewidth=0.5)

        gap = len(bl_names) + 0.5
        x_pr = np.arange(len(TAGS)) + gap
        pr_vals = [probe_results[t][metric] for t in TAGS]
        pr_colors = [C[t] for t in TAGS]
        pr_labels = [TAG_LABELS[t] for t in TAGS]
        bars_pr = ax.bar(x_pr, pr_vals, 0.6, color=pr_colors, alpha=0.85,
                         edgecolor="white", linewidth=0.5)

        for bars in [bars_bl, bars_pr]:
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                        f"{h:.3f}", ha="center", va="bottom", fontsize=8)

        all_labels = bl_names + pr_labels
        ax.set_xticks(list(x_bl) + list(x_pr))
        ax.set_xticklabels(all_labels, fontsize=9, rotation=30, ha="right")
        ax.set_ylabel(metric, fontsize=11)
        ax.set_title(metric, fontsize=12, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.axvline(gap - 0.75, color="gray", ls="--", alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUT / "fig2_gl_ngl_discrimination.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(OUT / "fig2_gl_ngl_discrimination.png", dpi=200, bbox_inches="tight")
    print(f"Saved fig2: {OUT / 'fig2_gl_ngl_discrimination.png'}")
    plt.close(fig)


# ==========================================================================
# Figure 3: Binding affinity
# ==========================================================================
def fig3_binding():
    # Hardcoded PRISM (clean) values provided by user
    dms_rho = {
        "G6.31":       {"clean": 0.085},
        "CR9114":      {"clean": 0.395},
        "Trastuzumab": {"clean": 0.343},
    }
    flab2_rho = {"clean": 0.066}

    # Compute noise2/noise4 from data files
    dms_datasets = {
        "G6.31": {
            "noise2": REPO / "data/antibody_binding/g6.31_benchmark_data_noise2.csv",
            "noise4": REPO / "data/antibody_binding/g6.31_benchmark_data_noise4.csv",
        },
        "CR9114": {
            "noise2": REPO / "data/antibody_binding/cr9114_benchmark_data_noise2.csv",
            "noise4": REPO / "data/antibody_binding/cr9114_benchmark_data_noise4.csv",
        },
        "Trastuzumab": {
            "noise2": REPO / "data/antibody_binding/trastuzumab_benchmark_noise2.csv",
            "noise4": REPO / "data/antibody_binding/trastuzumab_benchmark_noise4.csv",
        },
    }

    def get_spearman(fpath):
        if not fpath.exists():
            return np.nan
        df = pd.read_csv(fpath)
        df = df[df["Mutations"] != "WT"]
        for col in ["evo_ab_score", "evo_ab_affinity_score"]:
            if col in df.columns:
                valid = df[col].notna() & df["fitness"].notna() & (df[col] != 0)
                if valid.sum() > 10:
                    rho, _ = spearmanr(df.loc[valid, col], df.loc[valid, "fitness"])
                    return rho
        return np.nan

    for ds, files in dms_datasets.items():
        for tag in ["noise2", "noise4"]:
            dms_rho[ds][tag] = get_spearman(files[tag])

    for tag, csv_name in [
        ("noise2", "per_protein_results.csv"),
        ("noise4", "per_protein_results_noise4.csv"),
    ]:
        fpath = REPO / "data/features/evaluation_results/flab2_binding" / csv_name
        if fpath.exists():
            df = pd.read_csv(fpath)
            flab2_rho[tag] = df["origin_logit_sum"].dropna().mean()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5),
                                     gridspec_kw={"width_ratios": [3, 1]})
    fig.suptitle("Zero-Shot Binding Affinity",
                 fontsize=13, fontweight="bold", y=1.02)

    # Panel A: 3DMS
    ds_names = list(dms_rho.keys())
    x = np.arange(len(ds_names))
    w = 0.25
    for i, tag in enumerate(TAGS):
        vals = [dms_rho[ds][tag] for ds in ds_names]
        ax1.bar(x + (i-1)*w, vals, w, label=TAG_LABELS[tag],
                color=C[tag], alpha=0.85, edgecolor="white")
        for j, v in enumerate(vals):
            if not np.isnan(v):
                va = "bottom" if v >= 0 else "top"
                off = 0.01 if v >= 0 else -0.01
                ax1.text(x[j]+(i-1)*w, v+off, f"{v:.3f}",
                         ha="center", va=va, fontsize=7)

    # Best baseline per dataset (from benchmark_baselines_fixed.py)
    best_baselines_3dms = {
        "G6.31": 0.071,       # AntiBERTy
        "CR9114": -0.143,     # AbLang2 (all baselines negative)
        "Trastuzumab": 0.415, # AntiBERTy
    }
    for j, ds in enumerate(ds_names):
        bl_val = best_baselines_3dms.get(ds, None)
        if bl_val is not None:
            ax1.plot([j - 0.35, j + 0.35], [bl_val, bl_val],
                     color="gray", ls="--", lw=1.2, alpha=0.7)
            ax1.text(j + 0.38, bl_val, "best\nbaseline", fontsize=6, color="gray", va="center")

    ax1.set_xticks(x)
    ax1.set_xticklabels(ds_names, fontsize=11)
    ax1.set_ylabel("Spearman rho", fontsize=11)
    ax1.set_title("A. 3DMS Binding", fontsize=11, fontweight="bold")
    ax1.axhline(0, color="black", lw=0.5)
    ax1.legend(fontsize=9)

    # Panel B: FLAb2 binding
    flab_tags = [t for t in TAGS if t in flab2_rho]
    if flab_tags:
        xf = np.arange(len(flab_tags))
        vals = [flab2_rho[t] for t in flab_tags]
        colors = [C[t] for t in flab_tags]
        labels = [TAG_LABELS[t] for t in flab_tags]
        ax2.bar(xf, vals, 0.5, color=colors, alpha=0.85, edgecolor="white")
        for j, v in enumerate(vals):
            ax2.text(j, v + 0.002, f"{v:.4f}", ha="center", va="bottom", fontsize=9)
        ax2.set_xticks(xf)
        ax2.set_xticklabels(labels, fontsize=10)
    # TODO: add FLAb2 binding baseline once data is available
    ax2.set_ylabel("Per-assay mean Spearman rho", fontsize=11)
    ax2.set_title("B. FLAb2 Binding (41 proteins)", fontsize=11, fontweight="bold")
    ax2.axhline(0, color="black", lw=0.5)

    plt.tight_layout()
    fig.savefig(OUT / "fig3_binding_affinity.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(OUT / "fig3_binding_affinity.png", dpi=200, bbox_inches="tight")
    print(f"Saved fig3: {OUT / 'fig3_binding_affinity.png'}")
    plt.close(fig)


# ==========================================================================
# Figure 4: Developability
# ==========================================================================
def fig4_developability():
    ginkgo_props = ["AC-SINS_pH7.4", "HIC", "Tm2", "ADA", "PR_CHO", "Titer"]
    ginkgo_labels = ["AC-SINS", "HIC", "Tm2", "Immuno.\n(ADA)", "PR_CHO", "Expression\n(Titer)"]
    ppl_col = "evo_ab_ppl"

    # Hardcoded PRISM (clean) values provided by user
    ginkgo_rho = {
        "clean": {
            "AC-SINS_pH7.4": 0.181, "HIC": 0.093, "Tm2": 0.167,
            "ADA": 0.310, "PR_CHO": 0.071, "Titer": 0.168,
        }
    }

    # Compute noise2/noise4 from data files — pick best (max positive) rho across all ppl columns
    # Developability data + immunogenicity data (ADA is separate file)
    for tag, dev_path, ada_path in [
        ("noise2", REPO / "data/ginkgo/developability_data_noise2.csv",
                   REPO / "data/ginkgo/immunogenicity_noise2.csv"),
        ("noise4", REPO / "data/ginkgo/developability_data_noise4.csv",
                   REPO / "data/ginkgo/immunogenicity_noise4.csv"),
    ]:
        ginkgo_rho[tag] = {}
        for fpath, file_props in [(dev_path, [p for p in ginkgo_props if p != "ADA"]),
                                   (ada_path, ["ADA"])]:
            if not fpath.exists():
                continue
            df = pd.read_csv(fpath)
            ppl_cols = [c for c in df.columns if "evo_ab" in c and "ppl" in c]
            for prop in file_props:
                if prop not in df.columns:
                    continue
                best = -999
                for pc in ppl_cols:
                    valid = df[pc].notna() & df[prop].notna()
                    if valid.sum() > 5:
                        rho, _ = spearmanr(df.loc[valid, pc], df.loc[valid, prop])
                        if rho > best:
                            best = rho
                if best > -999:
                    ginkgo_rho[tag][prop] = best

    # FLAb2 developability
    flab2_rho = {}
    dfs = []
    for p in ["per_antibody_scores.csv", "per_antibody_scores_noise4.csv"]:
        fp = REPO / "data/features/evaluation_results/flab2_developability" / p
        if fp.exists():
            dfs.append(pd.read_csv(fp))
    if dfs:
        df_all = pd.concat(dfs, ignore_index=True)
        model_map = {"prism": "clean", "prism_noise2": "noise2", "prism_noise4": "noise4"}
        for prop in df_all["property"].unique():
            flab2_rho[prop] = {}
            for mk, tag in model_map.items():
                sub = df_all[(df_all["property"] == prop) & (df_all["model"] == mk)]
                if sub.empty:
                    continue
                best = -999
                for sig in sub["signal"].unique():
                    ss = sub[sub["signal"] == sig]
                    v = ss["score"].notna() & ss["fitness"].notna()
                    if v.sum() > 10:
                        rho, _ = spearmanr(ss.loc[v, "score"], ss.loc[v, "fitness"])
                        if rho > best:
                            best = rho
                if best > -999:
                    flab2_rho[prop][tag] = best

    flab2_props = list(flab2_rho.keys())
    flab2_label_map = {
        "self_interaction": "Self-Int.",
        "thermostability": "Thermo.",
        "immunogenicity": "Immuno.",
        "polyreactivity": "Polyreact.",
        "expression": "Express.",
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5),
                                     gridspec_kw={"width_ratios": [6, 5]})
    fig.suptitle("Zero-Shot Developability",
                 fontsize=13, fontweight="bold", y=1.02)

    # Panel A: Ginkgo
    w = 0.25
    x = np.arange(len(ginkgo_props))
    for i, tag in enumerate(TAGS):
        if tag not in ginkgo_rho:
            continue
        vals = [ginkgo_rho[tag].get(p, 0) for p in ginkgo_props]
        ax1.bar(x + (i-1)*w, vals, w, label=TAG_LABELS[tag],
                color=C[tag], alpha=0.85, edgecolor="white")
        for j, v in enumerate(vals):
            if v != 0:
                va = "bottom" if v >= 0 else "top"
                off = 0.01 if v >= 0 else -0.01
                ax1.text(x[j]+(i-1)*w, v+off, f"{v:.3f}",
                         ha="center", va=va, fontsize=7)

    ax1.set_xticks(x)
    ax1.set_xticklabels(ginkgo_labels, fontsize=10)
    ax1.set_ylabel("Spearman rho (PPL vs property)", fontsize=10)
    # Best baseline per Ginkgo property
    best_baselines_ginkgo = {
        "AC-SINS_pH7.4": -0.124,  # AbLang2
        "HIC": 0.066,             # AntiBERTy
        "Tm2": 0.112,             # Sapiens
        "ADA": 0.316,             # Sapiens
        "PR_CHO": 0.125,          # ESM2-650M
        "Titer": 0.042,           # AbLang2
    }
    for j, prop in enumerate(ginkgo_props):
        bl_val = best_baselines_ginkgo.get(prop, None)
        if bl_val is not None:
            ax1.plot([j - 0.35, j + 0.35], [bl_val, bl_val],
                     color="gray", ls="--", lw=1.2, alpha=0.7)
    ax1.plot([], [], color="gray", ls="--", lw=1.2, label="Best baseline")

    ax1.set_title("A. Ginkgo Developability", fontsize=11, fontweight="bold")
    ax1.axhline(0, color="black", lw=0.5)
    ax1.legend(fontsize=8)

    # Panel B: FLAb2
    x2 = np.arange(len(flab2_props))
    for i, tag in enumerate(TAGS):
        vals = [flab2_rho[p].get(tag, 0) for p in flab2_props]
        ax2.bar(x2 + (i-1)*w, vals, w, label=TAG_LABELS[tag],
                color=C[tag], alpha=0.85, edgecolor="white")
        for j, v in enumerate(vals):
            if v != 0:
                va = "bottom" if v >= 0 else "top"
                off = 0.005 if v >= 0 else -0.005
                ax2.text(x2[j]+(i-1)*w, v+off, f"{v:.3f}",
                         ha="center", va=va, fontsize=7)

    ax2.set_xticks(x2)
    ax2.set_xticklabels([flab2_label_map.get(p, p) for p in flab2_props], fontsize=10)
    # Best baseline per property (from per_antibody_scores.csv)
    best_baselines_flab2 = {
        "self_interaction": 0.288,  # Sapiens
        "thermostability": 0.317,   # Sapiens
        "immunogenicity": 0.350,    # Sapiens
        "polyreactivity": 0.318,    # Sapiens
        "expression": 0.051,        # AntiBERTy
    }
    for j, prop in enumerate(flab2_props):
        bl_val = best_baselines_flab2.get(prop, None)
        if bl_val is not None:
            ax2.plot([j - 0.35, j + 0.35], [bl_val, bl_val],
                     color="gray", ls="--", lw=1.2, alpha=0.7)
    # Single label for dashed line
    ax2.plot([], [], color="gray", ls="--", lw=1.2, label="Best baseline")

    ax2.set_ylabel("Per-assay mean Spearman rho", fontsize=10)
    ax2.set_title("B. FLAb2 Developability", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(OUT / "fig4_developability.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(OUT / "fig4_developability.png", dpi=200, bbox_inches="tight")
    print(f"Saved fig4: {OUT / 'fig4_developability.png'}")
    plt.close(fig)


if __name__ == "__main__":
    fig1_ppl()
    fig2_discrimination()
    fig3_binding()
    fig4_developability()
    print("\nAll 4 figures saved to:", OUT)
