#!/usr/bin/env python3
"""
Comprehensive baseline-with-IgLM benchmark figures for Section 3 zero-shot.

Outputs to img/3.zero-shot/:
  DMS (3 datasets, 7 models):
    - dms_baseline_iglm_spearman.{png,svg}      (3 panels grid)
    - dms_baseline_iglm_pearson.{png,svg}       (3 panels grid)
  FLAb2 binding (41 assays, 7 models):
    - flab2_binding_spearman_boxplot.{png,svg}  (single box panel)
    - flab2_binding_pearson_boxplot.{png,svg}   (single box panel)
    - flab2_binding_rank1_pct.{png,svg}         (single bar panel)
  GDPa1 developability + ADA immunogenicity (6 properties, 7 models):
    - developability_<prop>.{png,svg}           (Spearman, single panel)
    - developability_<prop>_pearson.{png,svg}   (Pearson, single panel)
  FLAb2 developability (5 properties, 7 models):
    - flab2_dev_<prop>.{png,svg}                (Spearman, single panel)
    - flab2_dev_<prop>_pearson.{png,svg}        (Pearson, single panel)

Conventions:
  - PRISM is leftmost in every plot (n_prism_variants=1, highlighted).
  - IgLM color = #EE7733.
  - Spearman and Pearson always in separate files.
  - Single-property dev plots match style of existing plot_developability_correlations.py
    (bootstrap CI bars, paired bootstrap p-value vs PRISM, significance brackets).

Run from repo root:
    conda run -n iglm python script/analyze/3.zero-shot/plot_baseline_with_iglm.py
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr, spearmanr

REPO = Path(__file__).resolve().parents[3]
OUT_DIR = REPO / "img" / "3.zero-shot" / "with_iglm"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ----- Style (matches plot_developability_correlations.py) -----
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.weight"] = "normal"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"

FONT_CONFIG = {
    "axis_label": {"fontsize": 22, "fontweight": "bold"},
    "tick_label": {"fontsize": 14, "fontweight": "normal"},
    "legend":     {"fontsize": 16},
    "title":      {"fontsize": 22, "fontweight": "bold"},
}

# IgLM = #EE7733 (orange). PRISM first.
MODEL_COLORS = {
    "PRISM":     "#332288",
    "ESM2-35M":  "#DDCC77",
    "ESM2-650M": "#117733",
    "AbLang2":   "#88CCEE",
    "AntiBERTy": "#44AA99",
    "Sapiens":   "#882255",
    "IgLM":      "#EE7733",
}
MODEL_ORDER = ["PRISM", "ESM2-35M", "ESM2-650M", "AbLang2", "AntiBERTy", "Sapiens", "IgLM"]
N_PRISM = 1  # PRISM is the first model


# =============================================================================
# Bootstrap utilities (copied from plot_developability_correlations.py)
# =============================================================================

def bootstrap_correlation(x, y, method="spearman", n_bootstrap=1000, ci=0.95, seed=42):
    rng = np.random.default_rng(seed)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 3:
        return (np.nan, np.nan, np.nan, np.nan)
    if method == "spearman":
        corr, p = stats.spearmanr(x, y)
    else:
        corr, p = stats.pearsonr(x, y)
    boots = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        if method == "spearman":
            r, _ = stats.spearmanr(x[idx], y[idx])
        else:
            r, _ = stats.pearsonr(x[idx], y[idx])
        if not np.isnan(r):
            boots.append(r)
    boots = np.asarray(boots)
    alpha = 1 - ci
    lo = np.percentile(boots, alpha / 2 * 100)
    hi = np.percentile(boots, (1 - alpha / 2) * 100)
    return (corr, lo, hi, p)


def paired_bootstrap_test(x1, x2, y, method="spearman", n_bootstrap=1000, seed=42):
    rng = np.random.default_rng(seed)
    mask = ~(np.isnan(x1) | np.isnan(x2) | np.isnan(y))
    x1, x2, y = x1[mask], x2[mask], y[mask]
    n = len(y)
    if n < 3:
        return 1.0
    if method == "spearman":
        r1, _ = stats.spearmanr(x1, y)
        r2, _ = stats.spearmanr(x2, y)
    else:
        r1, _ = stats.pearsonr(x1, y)
        r2, _ = stats.pearsonr(x2, y)
    obs = r1 - r2
    diffs = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        if method == "spearman":
            d1, _ = stats.spearmanr(x1[idx], y[idx])
            d2, _ = stats.spearmanr(x2[idx], y[idx])
        else:
            d1, _ = stats.pearsonr(x1[idx], y[idx])
            d2, _ = stats.pearsonr(x2[idx], y[idx])
        if not (np.isnan(d1) or np.isnan(d2)):
            diffs.append(d1 - d2)
    diffs = np.asarray(diffs)
    se = np.std(diffs)
    if se > 0:
        z = obs / se
        return 2 * (1 - stats.norm.cdf(abs(z)))
    return 1.0


def significance_stars(p):
    if np.isnan(p): return ""
    if p < 0.0001: return "****"
    if p < 0.001:  return "***"
    if p < 0.01:   return "**"
    if p < 0.05:   return "*"
    return "n.s."


# =============================================================================
# Generic single-panel bar plot (PRISM vs baselines + IgLM)
# =============================================================================

def plot_single_metric_bars(
    correlations: Dict[str, Tuple[float, float, float, float]],
    p_values: Dict[str, float],
    ylabel: str,
    out_stem: str,
    title: Optional[str] = None,
    figsize=(10, 8),
):
    """Single panel bar plot. Models in MODEL_ORDER (PRISM first)."""
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    valid = [m for m in MODEL_ORDER if m in correlations]
    n = len(valid)
    x_pos = np.arange(n)
    corr_values, ci_lo, ci_hi = [], [], []
    for m in valid:
        c, lo, hi, _ = correlations[m]
        corr_values.append(0 if np.isnan(c) else c)
        ci_lo.append(c if np.isnan(lo) else lo)
        ci_hi.append(c if np.isnan(hi) else hi)
    yerr_low = [max(0, c - lo) for c, lo in zip(corr_values, ci_lo)]
    yerr_high = [max(0, hi - c) for c, hi in zip(corr_values, ci_hi)]
    colors = [MODEL_COLORS[m] for m in valid]

    bars = ax.bar(x_pos, corr_values, color=colors, edgecolor="black", linewidth=2,
                  yerr=[yerr_low, yerr_high], capsize=8,
                  error_kw={"linewidth": 2, "capthick": 2})
    # Highlight PRISM
    if "PRISM" in valid:
        idx_prism = valid.index("PRISM")
        bars[idx_prism].set_edgecolor("black")
        bars[idx_prism].set_linewidth(3.5)

    # Significance brackets
    max_ci = max(ci_hi) if ci_hi else 0
    bracket_start = max_ci + 0.10
    bracket_int = 0.12
    if "PRISM" in valid:
        idx_prism = valid.index("PRISM")
        baseline_idxs = [i for i, m in enumerate(valid) if m != "PRISM"]
        baseline_idxs.sort(key=lambda i: abs(i - idx_prism))
        for k, bi in enumerate(baseline_idxs):
            p_val = p_values.get(valid[bi], np.nan)
            if np.isnan(p_val): continue
            by = bracket_start + k * bracket_int
            ax.plot([idx_prism, idx_prism, bi, bi],
                    [by - 0.015, by, by, by - 0.015], color="black", linewidth=1.5)
            ax.text((idx_prism + bi) / 2, by + 0.005, significance_stars(p_val),
                    ha="center", va="bottom", fontsize=18, fontweight="bold")

    # Annotations
    for bar, c, hi in zip(bars, corr_values, ci_hi):
        if c < 0:
            y = max(hi, 0) + 0.02
        else:
            y = hi + 0.02
        ax.text(bar.get_x() + bar.get_width() / 2, y, f"{c:+.3f}",
                ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(valid, fontsize=14, fontweight="bold", rotation=40, ha="right")
    ax.set_ylabel(ylabel, **FONT_CONFIG["axis_label"])
    if title:
        ax.set_title(title, **FONT_CONFIG["title"], pad=12)
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Y-axis: extend to fit annotations + brackets
    n_brackets = sum(1 for m in valid if m != "PRISM" and not np.isnan(p_values.get(m, np.nan)))
    ymax = bracket_start + max(0, n_brackets - 1) * bracket_int + 0.10
    ymin = min(min(ci_lo), -0.05) - 0.05
    ax.set_ylim(ymin, ymax)

    plt.tight_layout()
    fig.savefig(f"{out_stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(f"{out_stem}.svg", format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved: {out_stem}.png/.svg")


# =============================================================================
# DMS data + plot
# =============================================================================

def load_dms_per_variant():
    """
    Returns dict[dataset_label] -> dict with:
      - "fitness": np.ndarray (per-variant target)
      - "<model>": np.ndarray (per-variant score for that model)

    CR9114 panel uses H3 fitness for PRISM (at user's request — PRISM v44o was
    explicitly evaluated against H3 in the markdown); the 6 baselines + IgLM
    keep their H1 evaluation (no H3 baseline scores exist in baseline_scores_fixed).
    The two arrays therefore have different lengths for the CR9114 entry — that
    is fine because plot_dms_grid computes per-model rho independently.
    """
    base_dir = REPO / "data" / "prism_results" / "3.zero-shot" / "baseline_scores_fixed"
    v44o_dir = REPO / "data" / "prism_results" / "3.zero-shot" / "dms_prism_v44o_peak_glinv_scores"
    iglm_dir = REPO / "data" / "baselines" / "iglm" / "DMS"
    cfgs = {
        "CR9114 (H3)": ("cr9114_baseline_scores.csv", "cr9114_h1_prism_hf_scores.csv",
                        "cr9114_h3_prism_hf_scores.csv", "cr9114_NB_edits.csv"),
        "G6.31":       ("g6.31_baseline_scores.csv",  "g6.31_prism_hf_scores.csv",
                        None, "g6.31_NB_edits.csv"),
        "Trastuzumab": ("trastuzumab_baseline_scores.csv", "trastuzumab_prism_hf_scores.csv",
                        None, "trastuzumab_NB_edits.csv"),
    }
    out = {}
    for label, (bf, vf, vf_h3, igf) in cfgs.items():
        base = pd.read_csv(base_dir / bf).reset_index(drop=True)
        v44o = pd.read_csv(v44o_dir / vf)
        iglm = pd.read_csv(iglm_dir / igf, usecols=["fitness", "IgLM_score"]).reset_index(drop=True)
        idx = v44o["row_idx"].values

        entry = {
            # Fitness reference for the 6 baselines + IgLM (H1 for CR9114, native for G6.31/Trast)
            "fitness_baselines": v44o["fitness"].values,
            "ESM2-35M":  base.iloc[idx]["esm2_35m_score"].values,
            "ESM2-650M": base.iloc[idx]["esm2_650m_score"].values,
            "AbLang2":   base.iloc[idx]["ablang2_score"].values,
            "AntiBERTy": base.iloc[idx]["antiberty_score"].values,
            "Sapiens":   base.iloc[idx]["sapiens_score"].values,
            "IgLM":      iglm.iloc[idx]["IgLM_score"].values,
        }

        if vf_h3 is not None:
            # CR9114: PRISM uses H3 (separate row set, independent rho computation)
            v44o_h3 = pd.read_csv(v44o_dir / vf_h3)
            entry["fitness_PRISM"] = v44o_h3["fitness"].values
            entry["PRISM"] = v44o_h3["prism_score_ngl"].values
            print(f"  {label}: baselines+IgLM on H1 (n={len(idx)}), PRISM on H3 (n={len(v44o_h3)})")
        else:
            entry["fitness_PRISM"] = v44o["fitness"].values
            entry["PRISM"] = v44o["prism_score_ngl"].values
            print(f"  {label}: n={len(idx)}")

        out[label] = entry
    return out


# Antigen labels for each DMS dataset (used as panel subtitles)
DMS_SUBTITLE = {
    "CR9114 (H3)":  "CR9114 (Influenza HA)",
    "G6.31":        "G6.31 (VEGF)",
    "Trastuzumab":  "Trastuzumab (HER2)",
}


def plot_dms_single(entry, ds_key, metric, out_stem):
    """Single-panel DMS bar plot for one dataset × one metric.
    For CR9114, PRISM uses H3 fitness while baselines+IgLM use H1 (independent
    rho computations on different row sets).
    """
    fig, ax = plt.subplots(figsize=(8, 6.5))
    x_pos = np.arange(len(MODEL_ORDER))
    values = []
    for m in MODEL_ORDER:
        if m == "PRISM":
            x = entry["PRISM"]; y = entry["fitness_PRISM"]
        else:
            x = entry[m]; y = entry["fitness_baselines"]
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 3:
            values.append(np.nan); continue
        if metric == "spearman":
            r, _ = spearmanr(x[mask], y[mask])
        else:
            r, _ = pearsonr(x[mask], y[mask])
        values.append(r)
    colors = [MODEL_COLORS[m] for m in MODEL_ORDER]
    bars = ax.bar(x_pos, values, color=colors, edgecolor="black", linewidth=1.5)
    bars[0].set_linewidth(3.0)  # PRISM
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(MODEL_ORDER, rotation=30, ha="right", fontsize=14, fontweight="bold")
    ax.set_ylabel("Spearman" if metric == "spearman" else "Pearson",
                  **FONT_CONFIG["axis_label"])
    ax.set_title(DMS_SUBTITLE.get(ds_key, ds_key), **FONT_CONFIG["title"], pad=12)
    for bar, v in zip(bars, values):
        if np.isnan(v): continue
        offset = 0.018 if v >= 0 else -0.028
        va = "bottom" if v >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width() / 2, v + offset,
                f"{v:+.3f}", ha="center", va=va, fontsize=11, fontweight="bold")
    ax.set_ylim(-0.55, 0.65)
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(f"{out_stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(f"{out_stem}.svg", format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved: {out_stem}.png/.svg")


# =============================================================================
# FLAb2 binding box plot data + plots
# =============================================================================

def load_flab2_binding_per_assay():
    base_path = REPO / "data" / "features" / "evaluation_results" / "flab2_binding" / "per_variant_baselines.csv"
    base_long = pd.read_csv(base_path)
    base_model_map = {"esm2_35m": "ESM2-35M", "esm2_650m": "ESM2-650M",
                      "ablang2": "AbLang2", "antiberty": "AntiBERTy", "sapiens": "Sapiens"}
    base_by_src_model = {(src, base_model_map[m]): g.reset_index(drop=True)
                         for (src, m), g in base_long.groupby(["source_file", "model"])}
    v44o_path = REPO / "data" / "features" / "evaluation_results" / "flab2_binding" / "per_variant_v44o_peak.csv"
    v44o_all = pd.read_csv(v44o_path)
    v44o_by_src = {src: g.reset_index(drop=True) for src, g in v44o_all.groupby("source_file")}

    iglm_dir = REPO / "data" / "flab2" / "binding"
    iglm_data = {}
    for fn in os.listdir(iglm_dir):
        if not fn.endswith(".csv") or "_score_correlations" in fn or fn.startswith("all_"): continue
        df = pd.read_csv(iglm_dir / fn)
        if "IgLM_score" not in df.columns: continue
        iglm_data[fn] = df

    canonical = sorted(v44o_all["source_file"].unique())
    rows = []
    for src in canonical:
        # PRISM v44o (directed = +rho per markdown convention for NGL)
        sub_p = v44o_by_src.get(src)
        if sub_p is not None and len(sub_p) >= 10:
            x = sub_p["prism_v44o_ngl_logprob_sum"].values
            y = sub_p["fitness"].values
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() >= 10 and np.std(x[mask]) > 0:
                sr, _ = spearmanr(x[mask], y[mask])
                pr, _ = pearsonr(x[mask], y[mask])
                rows.append({"source_file": src, "model": "PRISM",
                             "spearman_directed": +sr, "pearson_directed": +pr,
                             "spearman_raw": sr, "pearson_raw": pr})

        # 5 baselines (directed = -rho)
        for label in ["ESM2-35M", "ESM2-650M", "AbLang2", "AntiBERTy", "Sapiens"]:
            sub = base_by_src_model.get((src, label))
            if sub is None or len(sub) < 10: continue
            x = sub["score"].values
            y = sub["fitness"].values
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() < 10 or np.std(x[mask]) == 0: continue
            sr, _ = spearmanr(x[mask], y[mask])
            pr, _ = pearsonr(x[mask], y[mask])
            rows.append({"source_file": src, "model": label,
                         "spearman_directed": -sr, "pearson_directed": -pr,
                         "spearman_raw": sr, "pearson_raw": pr})

        # IgLM (directed = -rho)
        match = None
        for fn, df in iglm_data.items():
            if fn == src or fn.replace(".csv", "") == src.replace(".csv", "").replace(".zip", ""):
                match = df; break
            if src.replace(".csv.zip", "") in fn or fn.replace(".csv", "") in src:
                match = df; break
        if match is None: continue
        sub_i = match[match["IgLM_score"] != 0.0].dropna(subset=["IgLM_score", "fitness"])
        if len(sub_i) < 10: continue
        x = sub_i["IgLM_score"].values
        y = sub_i["fitness"].values
        if np.std(x) == 0: continue
        sr, _ = spearmanr(x, y)
        pr, _ = pearsonr(x, y)
        rows.append({"source_file": src, "model": "IgLM",
                     "spearman_directed": -sr, "pearson_directed": -pr,
                     "spearman_raw": sr, "pearson_raw": pr})

    return pd.DataFrame(rows)


def plot_flab2_box(per_assay, metric, out_stem):
    fig, ax = plt.subplots(1, 1, figsize=(11, 7))
    metric_col = f"{metric}_directed"
    ylabel = "Spearman" if metric == "spearman" else "Pearson"
    data = []
    pos = []
    labels = []
    colors = []
    for i, m in enumerate(MODEL_ORDER):
        vals = per_assay[per_assay["model"] == m][metric_col].dropna().values
        data.append(vals)
        pos.append(i)
        labels.append(m)
        colors.append(MODEL_COLORS[m])
    bp = ax.boxplot(data, positions=pos, widths=0.6, patch_artist=True,
                    medianprops=dict(color="black", linewidth=2),
                    boxprops=dict(linewidth=1.5),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2),
                    flierprops=dict(marker="o", markerfacecolor="gray", markersize=4, alpha=0.5))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_edgecolor("black"); patch.set_alpha(0.85)
    bp["boxes"][0].set_linewidth(3.0)  # PRISM
    for i, vals in enumerate(data):
        jitter = np.random.normal(0, 0.05, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, s=10, color="black", alpha=0.35, zorder=3)
    means = [np.mean(v) if len(v) else np.nan for v in data]
    ax.scatter(pos, means, marker="D", s=70, color="white", edgecolor="black",
               linewidths=1.5, zorder=5, label="mean")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_xticks(pos)
    ax.set_xticklabels(labels, fontsize=14, fontweight="bold", rotation=30, ha="right")
    ax.set_ylabel(ylabel, **FONT_CONFIG["axis_label"])
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", fontsize=12)

    # Place mean value text above each box at a uniform y just above global max
    all_vals = np.concatenate([v for v in data if len(v)])
    ymax_data = float(np.nanmax(all_vals)) if len(all_vals) else 1.0
    text_y = ymax_data + 0.05
    for i, mu in enumerate(means):
        ax.text(i, text_y, f"{mu:+.3f}", ha="center", va="bottom",
                fontsize=12, fontweight="bold")
    # Extend ylim so the text is visible
    ymin_data = float(np.nanmin(all_vals)) if len(all_vals) else -1.0
    ax.set_ylim(ymin_data - 0.05, text_y + 0.10)

    plt.tight_layout()
    fig.savefig(f"{out_stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(f"{out_stem}.svg", format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved: {out_stem}.png/.svg")


def _plot_flab2_pct_bar(values_pct, counts, n, ylabel, out_stem):
    """Shared helper: vertical bar of per-method percentage with PRISM first."""
    fig, ax = plt.subplots(figsize=(11, 7))
    x_pos = np.arange(len(MODEL_ORDER))
    colors = [MODEL_COLORS[m] for m in MODEL_ORDER]
    bars = ax.bar(x_pos, values_pct, color=colors, edgecolor="black", linewidth=1.5)
    bars[0].set_linewidth(3.0)
    for bar, p, c1 in zip(bars, values_pct, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, p + 1, f"{p:.1f}%\n({c1}/{n})",
                ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(MODEL_ORDER, rotation=30, ha="right", fontsize=14, fontweight="bold")
    ax.set_ylabel(ylabel, **FONT_CONFIG["axis_label"])
    ax.set_ylim(0, max(max(values_pct), 1) * 1.20)
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(f"{out_stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(f"{out_stem}.svg", format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved: {out_stem}.png/.svg")


def plot_flab2_rank1(per_assay, out_stem):
    pivot = per_assay.pivot(index="source_file", columns="model", values="spearman_directed")
    pivot = pivot[MODEL_ORDER]
    pivot.to_csv(OUT_DIR / "flab2_binding_per_assay_directed_spearman_with_iglm.csv")
    n = len(pivot)
    mat = pivot.values
    ranks = (-mat).argsort(axis=1).argsort(axis=1) + 1
    pct = (ranks == 1).mean(axis=0) * 100
    n1 = (ranks == 1).sum(axis=0)
    _plot_flab2_pct_bar(
        pct, n1, n,
        ylabel="Win rate",
        out_stem=out_stem,
    )
    print(f"  rank-1 distribution:")
    for m, p, c1 in zip(MODEL_ORDER, pct, n1):
        print(f"    {m:<12} {p:>5.1f}% ({int(c1)}/{n})")


def plot_flab2_rank_box(per_assay, out_stem):
    """Per-method rank distribution across 41 FLAb2 binding assays.
    For each assay we rank the 7 methods by directed Spearman ρ (1=best, 7=worst);
    box plot shows the rank distribution per model.
    Y-axis inverted (1 at top, 7 at bottom) so "higher in plot = better"."""
    pivot = per_assay.pivot(index="source_file", columns="model", values="spearman_directed")
    pivot = pivot[MODEL_ORDER]
    n = len(pivot)
    mat = pivot.values  # [n_assays, 7]
    # Rank per assay (1 = highest directed rho)
    ranks_mat = (-mat).argsort(axis=1).argsort(axis=1) + 1  # [n, 7]

    fig, ax = plt.subplots(figsize=(11, 7))
    data = [ranks_mat[:, i] for i in range(len(MODEL_ORDER))]
    pos = list(range(len(MODEL_ORDER)))
    colors = [MODEL_COLORS[m] for m in MODEL_ORDER]

    bp = ax.boxplot(data, positions=pos, widths=0.6, patch_artist=True,
                    medianprops=dict(color="black", linewidth=2),
                    boxprops=dict(linewidth=1.5),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2),
                    flierprops=dict(marker="o", markerfacecolor="gray", markersize=4, alpha=0.5))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_edgecolor("black"); patch.set_alpha(0.85)
    bp["boxes"][0].set_linewidth(3.0)  # PRISM

    # Jittered raw rank points
    for i, vals in enumerate(data):
        jitter = np.random.normal(0, 0.07, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals,
                   s=14, color="black", alpha=0.35, zorder=3)
    # Mean rank marker
    means = [v.mean() for v in data]
    ax.scatter(pos, means, marker="D", s=80, color="white",
               edgecolor="black", linewidths=1.5, zorder=5, label="mean rank")

    # Mean rank text annotation above each box
    for i, m in enumerate(means):
        ax.text(i, 0.55, f"{m:.2f}", ha="center", va="bottom",
                fontsize=12, fontweight="bold")

    ax.set_xticks(pos)
    ax.set_xticklabels(MODEL_ORDER, rotation=30, ha="right", fontsize=14, fontweight="bold")
    ax.set_ylabel("Mean rank", **FONT_CONFIG["axis_label"])
    ax.set_yticks(range(1, 8))
    ax.set_ylim(7.5, 0.3)  # inverted so 1 is at top
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", fontsize=12)

    plt.tight_layout()
    fig.savefig(f"{out_stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(f"{out_stem}.svg", format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved: {out_stem}.png/.svg")
    print(f"  mean ranks:")
    for m, mu in zip(MODEL_ORDER, means):
        print(f"    {m:<12} {mu:.2f}")


# =============================================================================
# GDPa1 dev + ADA dev plots
# =============================================================================

# 6 properties: 5 GDPa1 + ADA
DEV_PROPERTIES = {
    "hydrophobicity":  {"col": "HIC",          "higher_is_better": False, "title": "Hydrophobicity (HIC)"},
    "reactivity":      {"col": "PR_CHO",       "higher_is_better": False, "title": "Polyreactivity (PR_CHO)"},
    "aggregation":     {"col": "AC-SINS_pH7.4","higher_is_better": False, "title": "Self-Interaction (AC-SINS pH7.4)"},
    "thermalstability":{"col": "Tm2",          "higher_is_better": True,  "title": "Thermal Stability (Tm2)"},
    "expression":      {"col": "Titer",        "higher_is_better": True,  "title": "Expression (Titer)"},
    "immunogenicity":  {"col": "ADA",          "higher_is_better": False, "title": "Immunogenicity (ADA)"},
}


def load_gdpa1_with_iglm():
    """GDPa1: baselines + v44o + IgLM merged on antibody name."""
    base = pd.read_csv(REPO / "data" / "prism_results" / "3.zero-shot" / "baseline_scores_fixed" / "developability_ppl_scores.csv")
    v44o = pd.read_csv(REPO / "data" / "prism_results" / "3.zero-shot" / "dev_scores_v44o_peak" / "hf_dev_scores.csv")
    iglm = pd.read_csv(REPO / "data" / "baselines" / "iglm" / "developability" / "gdpa_libraries_gdpa1_w_IgLM_scores.csv")

    df = base.copy()
    # add v44o gl_ppl
    df = df.merge(v44o[["antibody_name", "gl_ppl"]].rename(columns={"gl_ppl": "prism_v44o_ppl"}),
                  on="antibody_name", how="left")
    # add IgLM PsPPL
    df = df.merge(iglm[["antibody_name", "IgLM_PseudoPerplexity"]].rename(
        columns={"IgLM_PseudoPerplexity": "iglm_pspp"}), on="antibody_name", how="left")
    return df


def load_ada_with_iglm():
    """ADA dataset (n=206/217): baselines + v44o + IgLM merged on antibody name."""
    base = pd.read_csv(REPO / "data" / "prism_results" / "3.zero-shot" / "baseline_scores_fixed" / "immunogenicity_ppl_scores.csv")
    v44o = pd.read_csv(REPO / "data" / "prism_results" / "3.zero-shot" / "dev_scores_v44o_peak" / "hf_immuno_scores.csv")
    iglm = pd.read_csv(REPO / "data" / "baselines" / "iglm" / "developability" / "processed_therapeutic_sequences_w_ada_values_w_imgt_w_predictions_w_IgLM_scores.csv")

    # Standardize antibody name column: 'Name' in iglm vs 'antibody_name' in base
    if "Name" in iglm.columns:
        iglm = iglm.rename(columns={"Name": "antibody_name"})

    df = base.copy()
    df = df.merge(v44o[["antibody_name", "gl_ppl"]].rename(columns={"gl_ppl": "prism_v44o_ppl"}),
                  on="antibody_name", how="left")
    df = df.merge(iglm[["antibody_name", "IgLM_PseudoPerplexity"]].rename(
        columns={"IgLM_PseudoPerplexity": "iglm_pspp"}), on="antibody_name", how="left")
    return df


def compute_dev_correlations(df, prop_col, higher_is_better, n_bootstrap=1000):
    """Apply direction correction (multiply PPL by -1 if higher_is_better=True)
    so that positive correlation = correct direction."""
    y = df[prop_col].values
    score_cols = {
        "PRISM":     "prism_v44o_ppl",
        "ESM2-35M":  "esm2_35m_ppl",
        "ESM2-650M": "esm2_650m_ppl",
        "AbLang2":   "ablang2_ppl",
        "AntiBERTy": "antiberty_ppl",
        "Sapiens":   "sapiens_ppl",
        "IgLM":      "iglm_pspp",
    }
    multiplier = -1 if higher_is_better else 1
    sp_corrs, pe_corrs = {}, {}
    raw_data = {}
    for label, col in score_cols.items():
        if col not in df.columns: continue
        x = df[col].values * multiplier
        sp = bootstrap_correlation(x, y, "spearman", n_bootstrap)
        pe = bootstrap_correlation(x, y, "pearson", n_bootstrap)
        sp_corrs[label] = sp
        pe_corrs[label] = pe
        raw_data[label] = x
    # Pairwise p-values vs PRISM
    sp_p, pe_p = {}, {}
    if "PRISM" in raw_data:
        for label in raw_data:
            if label == "PRISM":
                sp_p[label] = 1.0; pe_p[label] = 1.0
            else:
                sp_p[label] = paired_bootstrap_test(raw_data["PRISM"], raw_data[label], y, "spearman", n_bootstrap)
                pe_p[label] = paired_bootstrap_test(raw_data["PRISM"], raw_data[label], y, "pearson", n_bootstrap)
    return sp_corrs, pe_corrs, sp_p, pe_p


def gen_dev_plots(df, prop_keys, prefix, n_bootstrap=1000):
    for key in prop_keys:
        cfg = DEV_PROPERTIES[key]
        if cfg["col"] not in df.columns:
            print(f"  skip {key}: column '{cfg['col']}' not in df")
            continue
        print(f"  {key} (col={cfg['col']}, higher_better={cfg['higher_is_better']})...")
        sp, pe, sp_p, pe_p = compute_dev_correlations(df, cfg["col"], cfg["higher_is_better"], n_bootstrap)
        plot_single_metric_bars(
            sp, sp_p, "Spearman ρ",
            str(OUT_DIR / f"{prefix}_{key}"), title=cfg["title"]
        )
        plot_single_metric_bars(
            pe, pe_p, "Pearson r",
            str(OUT_DIR / f"{prefix}_{key}_pearson"), title=cfg["title"]
        )


# =============================================================================
# FLAb2 dev plots
# =============================================================================

# property name in baseline file -> (file, fitness col, higher_is_better, key for output filename, display title)
FLAB2_DEV_PROPS = {
    "self_interaction": {"file": "flab2_aggregation_ppl_scores.csv", "fitness": "fitness", "higher_is_better": False, "title": "Aggregation (FLAb2 ACSINS)", "v44o_prop": "self_interaction"},
    "polyreactivity":   {"file": "flab2_polyreactivity_ppl_scores.csv", "fitness": "fitness", "higher_is_better": False, "title": "Polyreactivity (FLAb2 PSR)", "v44o_prop": "polyreactivity"},
    "immunogenicity":   {"file": "flab2_immunogenicity_ppl_scores.csv", "fitness": "fitness", "higher_is_better": False, "title": "Immunogenicity (FLAb2 ADA)", "v44o_prop": "immunogenicity"},
    "thermostability":  {"file": "flab2_thermostability_ppl_scores.csv", "fitness": "fitness", "higher_is_better": True,  "title": "Thermal Stability (FLAb2 DSC)", "v44o_prop": "thermostability"},
    "expression":       {"file": "flab2_expression_ppl_scores.csv", "fitness": "fitness", "higher_is_better": True,  "title": "Expression (FLAb2 HEK)", "v44o_prop": "expression"},
}


def load_flab2_dev_for_property(key):
    cfg = FLAB2_DEV_PROPS[key]
    base_dir = REPO / "data" / "prism_results" / "3.zero-shot" / "baseline_scores_fixed"
    base = pd.read_csv(base_dir / cfg["file"])

    # PRISM v44o pll_gl signal from per_antibody_scores.csv (long format)
    pp = pd.read_csv(REPO / "data" / "features" / "evaluation_results" / "flab2_developability" / "per_antibody_scores.csv")
    pp = pp[(pp["model"] == "v44o_peak") & (pp["property"] == cfg["v44o_prop"]) & (pp["signal"] == "ppl_gl")]
    pp = pp[["heavy", "light", "score"]].rename(columns={"score": "prism_v44o_ppl"})
    df = base.merge(pp, on=["heavy", "light"], how="inner")

    # IgLM PsPPL
    iglm = pd.read_csv(REPO / "data" / "flab2" / "developability" / cfg["v44o_prop"] / "scored.csv")
    iglm = iglm[["heavy", "light", "IgLM_PseudoPerplexity"]].rename(columns={"IgLM_PseudoPerplexity": "iglm_pspp"})
    df = df.merge(iglm, on=["heavy", "light"], how="inner")
    return df, cfg


def gen_flab2_dev_plots(n_bootstrap=1000):
    for key in FLAB2_DEV_PROPS:
        df, cfg = load_flab2_dev_for_property(key)
        if len(df) == 0:
            print(f"  skip flab2_{key}: empty merge"); continue
        print(f"  flab2_{key}: n={len(df)}")
        sp, pe, sp_p, pe_p = compute_dev_correlations(df, cfg["fitness"], cfg["higher_is_better"], n_bootstrap)
        plot_single_metric_bars(
            sp, sp_p, "Spearman ρ",
            str(OUT_DIR / f"flab2_dev_{key}"), title=cfg["title"]
        )
        plot_single_metric_bars(
            pe, pe_p, "Pearson r",
            str(OUT_DIR / f"flab2_dev_{key}_pearson"), title=cfg["title"]
        )


# =============================================================================
# GL vs NGL vs Marginalized comparison (PRISM heads vs AbLang2 baseline)
# =============================================================================

def compute_gl_ngl_data(metric="spearman"):
    """Return dict[dataset_label] -> {head: directed_rho} for 9 zero-shot tasks
    using PRISM v44o GL/NGL/Marginalized heads + AbLang2 as baseline.
    'directed' rho is signed so that positive = correct prediction direction.
    """
    fn = spearmanr if metric == "spearman" else pearsonr
    out = {}

    # ---- 1-3: DMS binding (rho positive = matches fitness up direction) ----
    dms_dir = REPO / "data" / "prism_results" / "3.zero-shot" / "dms_prism_v44o_peak_glinv_scores"
    base_dir = REPO / "data" / "prism_results" / "3.zero-shot" / "baseline_scores_fixed"
    dms_cfgs = [
        ("CR9114\n(Influenza HA)", "cr9114_h3_prism_hf_scores.csv", "cr9114_baseline_scores.csv"),
        ("G6.31\n(VEGF)",          "g6.31_prism_hf_scores.csv",      "g6.31_baseline_scores.csv"),
        ("Trastuzumab\n(HER2)",    "trastuzumab_prism_hf_scores.csv","trastuzumab_baseline_scores.csv"),
    ]
    for label, prism_fn, base_fn in dms_cfgs:
        prism = pd.read_csv(dms_dir / prism_fn)
        base = pd.read_csv(base_dir / base_fn).reset_index(drop=True)
        idx = prism["row_idx"].values
        ablang = base.iloc[idx]["ablang2_score"].values
        y = prism["fitness"].values
        out[label] = {
            "GL":      fn(prism["prism_score_gl"], y)[0],
            "NGL":     fn(prism["prism_score_ngl"], y)[0],
            "Marg":    fn(prism["prism_score_marginalized"], y)[0],
            "AbLang2": fn(ablang, y)[0],
        }

    # ---- 4-8: GDPa1 developability (5 properties) ----
    dev_v44o = pd.read_csv(REPO / "data" / "prism_results" / "3.zero-shot" / "dev_scores_v44o_peak" / "hf_dev_scores.csv")
    dev_base = pd.read_csv(REPO / "data" / "prism_results" / "3.zero-shot" / "baseline_scores_fixed" / "developability_ppl_scores.csv")
    dev_merged = dev_v44o.merge(dev_base[["antibody_name", "ablang2_ppl"]], on="antibody_name")

    dev_props = [
        ("Hydrophobicity\n(HIC)",            "HIC",          False),
        ("Polyreactivity\n(PR_CHO)",         "PR_CHO",       False),
        ("Self-Interaction\n(AC-SINS)",      "AC-SINS_pH7.4",False),
        ("Thermal Stability\n(Tm2)",         "Tm2",          True),
        ("Expression\n(Titer)",              "Titer",        True),
    ]
    for label, col, hib in dev_props:
        sub = dev_merged[[col, "gl_ppl", "ngl_ppl", "marg_ppl", "ablang2_ppl"]].dropna()
        # directed: positive = correct. PPL low → property low (raw -rho).
        # If higher property is better, multiply rho by -1 (so low PPL ↔ high property = correct).
        # Equivalently apply sign to PPL values inside fn.
        sign = -1 if hib else +1  # multiply PPL by -1 if higher property is better
        out[label] = {
            "GL":      -fn(sub["gl_ppl"]      * sign, sub[col])[0] if False else fn(sub["gl_ppl"]      * sign, sub[col])[0],
            "NGL":     fn(sub["ngl_ppl"]     * sign, sub[col])[0],
            "Marg":    fn(sub["marg_ppl"]    * sign, sub[col])[0],
            "AbLang2": fn(sub["ablang2_ppl"] * sign, sub[col])[0],
        }

    # ---- 9: ADA immunogenicity (lower = better, no flip) ----
    ada_v44o = pd.read_csv(REPO / "data" / "prism_results" / "3.zero-shot" / "dev_scores_v44o_peak" / "hf_immuno_scores.csv")
    ada_base = pd.read_csv(REPO / "data" / "prism_results" / "3.zero-shot" / "baseline_scores_fixed" / "immunogenicity_ppl_scores.csv")
    ada = ada_v44o.merge(ada_base[["antibody_name", "ablang2_ppl"]], on="antibody_name")
    sub = ada[["ADA", "gl_ppl", "ngl_ppl", "marg_ppl", "ablang2_ppl"]].dropna()
    # ADA: lower = better. PPL low ↔ ADA low = correct → raw rho is already directed
    out["Immunogenicity\n(ADA)"] = {
        "GL":      fn(sub["gl_ppl"],      sub["ADA"])[0],
        "NGL":     fn(sub["ngl_ppl"],     sub["ADA"])[0],
        "Marg":    fn(sub["marg_ppl"],    sub["ADA"])[0],
        "AbLang2": fn(sub["ablang2_ppl"], sub["ADA"])[0],
    }
    return out


def plot_gl_ngl_comparison(metric, out_stem):
    """9-task grouped bar plot of PRISM heads (GL/NGL/Marg) + AbLang2 control.
    Y-axis = directed rho (positive = correct direction). One file per metric.
    """
    data = compute_gl_ngl_data(metric)
    datasets = list(data.keys())
    head_order = ["GL", "NGL", "Marg", "AbLang2"]
    head_colors = {
        "GL":      "#1CC454",  # green
        "NGL":     "#C8327D",  # pink
        "Marg":    "#332288",  # purple
        "AbLang2": "#88CCEE",  # light blue
    }
    n_ds = len(datasets)
    n_heads = len(head_order)
    bar_w = 0.20
    x_centers = np.arange(n_ds)

    fig, ax = plt.subplots(figsize=(18, 6.5))
    for i, head in enumerate(head_order):
        vals = [data[ds][head] for ds in datasets]
        offset = (i - (n_heads - 1) / 2) * bar_w
        bars = ax.bar(x_centers + offset, vals, width=bar_w,
                      color=head_colors[head], edgecolor="black", linewidth=1.2,
                      label=head if head != "Marg" else "Marginalized")
        for bar, v in zip(bars, vals):
            if np.isnan(v): continue
            offset_y = 0.012 if v >= 0 else -0.022
            va = "bottom" if v >= 0 else "top"
            ax.text(bar.get_x() + bar.get_width() / 2, v + offset_y,
                    f"{v:+.2f}", ha="center", va=va, fontsize=8, fontweight="bold")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x_centers)
    ax.set_xticklabels(datasets, fontsize=11, fontweight="bold")
    ax.set_ylabel("Spearman" if metric == "spearman" else "Pearson",
                  **FONT_CONFIG["axis_label"])
    ax.legend(loc="upper left", fontsize=12, ncol=4, frameon=True)
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # Y-limits: leave room for legend at top
    all_vals = [v for ds in datasets for v in data[ds].values() if not np.isnan(v)]
    ymax = max(all_vals); ymin = min(all_vals)
    ax.set_ylim(min(ymin - 0.05, -0.10), ymax + 0.20)

    plt.tight_layout()
    fig.savefig(f"{out_stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(f"{out_stem}.svg", format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved: {out_stem}.png/.svg")


# =============================================================================
# Main
# =============================================================================

def main():
    n_boot = 1000

    print("=" * 60); print("DMS plots"); print("=" * 60)
    dms_data = load_dms_per_variant()
    # 3 datasets × 2 metrics = 6 separate single-panel files
    dataset_slug = {"CR9114 (H3)": "cr9114", "G6.31": "g6.31", "Trastuzumab": "trastuzumab"}
    for ds_key, slug in dataset_slug.items():
        plot_dms_single(dms_data[ds_key], ds_key, "spearman",
                        str(OUT_DIR / f"dms_{slug}_spearman"))
        plot_dms_single(dms_data[ds_key], ds_key, "pearson",
                        str(OUT_DIR / f"dms_{slug}_pearson"))

    print("\n" + "=" * 60); print("FLAb2 binding plots"); print("=" * 60)
    per_assay = load_flab2_binding_per_assay()
    per_assay.to_csv(OUT_DIR / "flab2_binding_per_assay_with_iglm.csv", index=False)
    plot_flab2_box(per_assay, "spearman", str(OUT_DIR / "flab2_binding_spearman_boxplot"))
    plot_flab2_box(per_assay, "pearson",  str(OUT_DIR / "flab2_binding_pearson_boxplot"))
    plot_flab2_rank1(per_assay, str(OUT_DIR / "flab2_binding_rank1_pct"))
    plot_flab2_rank_box(per_assay, str(OUT_DIR / "flab2_binding_rank_boxplot"))

    print("\n" + "=" * 60); print("GDPa1 + ADA dev plots"); print("=" * 60)
    gdpa = load_gdpa1_with_iglm()
    print(f"  GDPa1 merged: n={len(gdpa)}")
    gen_dev_plots(gdpa, ["hydrophobicity", "reactivity", "aggregation", "thermalstability", "expression"],
                  prefix="developability", n_bootstrap=n_boot)
    ada = load_ada_with_iglm()
    print(f"  ADA merged: n={len(ada)}")
    gen_dev_plots(ada, ["immunogenicity"], prefix="developability", n_bootstrap=n_boot)

    print("\n" + "=" * 60); print("FLAb2 dev plots"); print("=" * 60)
    gen_flab2_dev_plots(n_bootstrap=n_boot)


if __name__ == "__main__":
    main()
