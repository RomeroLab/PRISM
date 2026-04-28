#!/usr/bin/env python
"""
Generate 5-bar ablation figures (PRISM Full / Ablation 1 / 2 / 3 / PRISM-less)
for all zero-shot tasks: 3 binding + 5 developability + 1 immunogenicity.

PRISM Full = v44o peak (`v44o_peak_glinv_standalone.ckpt`) — per the
`reports/baseline_benchmark_directed.md` file:
  - DMS                 -> `prism_score_ngl`   in dms_prism_v44o_peak_glinv_scores/*.csv
  - GDPa1 developability -> `gl_ppl`            in dev_scores_v44o_peak/hf_dev_scores.csv
  - ADA immunogenicity   -> `gl_ppl`            in dev_scores_v44o_peak/hf_immuno_scores.csv

PRISM-less is read from per-variant CSVs in `baseline_scores/` so both Spearman
and Pearson are supported.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

PROJECT = Path(__file__).resolve().parents[3]
OUT_DIR = PROJECT / "img" / "3.zero-shot"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ABLATION_COLORS = {
    "PRISM Full": "#332288",
    "Ablation 1": "#CC6677",
    "Ablation 2": "#DDCC77",
    "Ablation 3": "#AA4499",
    "PRISM-less": "#78c679",
}
MODEL_ORDER = ["PRISM Full", "Ablation 1", "Ablation 2", "Ablation 3", "PRISM-less"]


CURRENT_METRIC = "spearman"  # toggled at runtime for pearson pass


def _corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return np.nan
    if CURRENT_METRIC == "pearson":
        return float(pearsonr(x[m], y[m]).statistic)
    return float(spearmanr(x[m], y[m]).correlation)


def rho_for_main(df, ppl_col, target_col, multiply_neg1):
    if ppl_col not in df.columns or target_col not in df.columns:
        return np.nan
    x = df[ppl_col].values
    if multiply_neg1:
        x = -x
    return _corr(x, df[target_col].values)


def rho_for_ablation_csv(csv_path, target_col, multiply_neg1):
    p = Path(csv_path)
    if not p.exists():
        return np.nan
    adf = pd.read_csv(p)
    if target_col not in adf.columns:
        return np.nan
    ppl = None
    if "evo_ab_ppl" in adf.columns and not adf["evo_ab_ppl"].isna().all():
        ppl = adf["evo_ab_ppl"].values
    elif "evo_ab_ppl_aa" in adf.columns and not adf["evo_ab_ppl_aa"].isna().all():
        ppl = adf["evo_ab_ppl_aa"].values
    if ppl is None:
        return np.nan
    if multiply_neg1:
        ppl = -ppl
    return _corr(ppl, adf[target_col].values)


def binding_rho_for_csv(csv_path, score_col, fitness_col="fitness"):
    p = Path(csv_path)
    if not p.exists():
        return np.nan
    adf = pd.read_csv(p)
    if score_col not in adf.columns or fitness_col not in adf.columns:
        return np.nan
    mask = (adf["Mutations"] != "WT") if "Mutations" in adf.columns else np.ones(len(adf), dtype=bool)
    return _corr(adf.loc[mask, score_col].values, adf.loc[mask, fitness_col].values)


def make_bar(ax, values, title, ylabel):
    x = np.arange(len(MODEL_ORDER))
    vals = [values.get(m, np.nan) for m in MODEL_ORDER]
    colors = [ABLATION_COLORS[m] for m in MODEL_ORDER]
    bars = ax.bar(x, vals, color=colors, edgecolor="black", linewidth=1.5, width=0.7)

    # Expand y-range beyond data so labels have breathing room
    finite = [v for v in vals if np.isfinite(v)]
    if finite:
        vmin, vmax = min(finite + [0.0]), max(finite + [0.0])
        span = max(vmax - vmin, 0.2)
        pad = span * 0.18
        ax.set_ylim(vmin - pad, vmax + pad)

    for xi, v in zip(x, vals):
        if np.isfinite(v):
            off_abs = span * 0.03 if finite else 0.02
            off = off_abs if v >= 0 else -off_abs
            va = "bottom" if v >= 0 else "top"
            ax.text(xi, v + off, f"{v:.3f}", ha="center", va=va,
                    fontsize=13, fontweight="bold")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_ORDER, rotation=35, ha="right", fontsize=13, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=17, fontweight="bold")
    ax.set_title(title, fontsize=17, fontweight="bold")
    ax.tick_params(axis="y", labelsize=12)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_linewidth(2)


def save(fig, stem):
    suffix = "_pearson" if CURRENT_METRIC == "pearson" else ""
    png = OUT_DIR / f"{stem}{suffix}.png"
    svg = OUT_DIR / f"{stem}{suffix}.svg"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(svg, format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {png.name}, {svg.name}")


def _metric_label():
    return "Pearson r" if CURRENT_METRIC == "pearson" else "Spearman \u03c1"


# ============================================================================
# HF PRISM scalar lookup helpers (computed on-the-fly from per-variant CSVs
# so both Spearman and Pearson can be reported)
# ============================================================================
def _hf_dms_corr(dataset_tag, mode="ngl"):
    """Compute correlation on-the-fly from v44o peak per-variant score CSV.

    Uses `fitness` column already stored in the score CSV.
    Column naming in v44o peak files: `prism_score_{gl,ngl,marginalized,exact}`.
    """
    p = PROJECT / (
        "data/prism_results/3.zero-shot/dms_prism_v44o_peak_glinv_scores/"
        f"{dataset_tag}_prism_hf_scores.csv"
    )
    if not p.exists():
        return np.nan
    df = pd.read_csv(p)
    col = f"prism_score_{mode}"
    if col not in df.columns or "fitness" not in df.columns:
        return np.nan
    return _corr(df[col].values, df["fitness"].values)


def _hf_dev_corr(target_col, source="dev"):
    """Compute correlation between v44o peak gl_ppl and target column, using per-seq CSV.

    File paths (v44o peak):
      - dev_scores_v44o_peak/hf_dev_scores.csv     (for developability)
      - dev_scores_v44o_peak/hf_immuno_scores.csv  (for immunogenicity)
    """
    p = PROJECT / f"data/prism_results/3.zero-shot/dev_scores_v44o_peak/hf_{source}_scores.csv"
    if not p.exists():
        return np.nan
    df = pd.read_csv(p)
    if "gl_ppl" not in df.columns or target_col not in df.columns:
        return np.nan
    x = df["gl_ppl"].values
    y = df[target_col].values
    # Direction correction for Tm2/Titer (higher target = better -> use -ppl)
    if target_col in ("Tm2", "Titer"):
        x = -x
    return _corr(x, y)


def _baseline_dms_corr(dataset_tag):
    """PRISM-less correlation for a DMS dataset from baseline_scores CSV."""
    p = PROJECT / f"data/prism_results/3.zero-shot/baseline_scores/{dataset_tag}_baseline.csv"
    if not p.exists():
        return np.nan
    df = pd.read_csv(p)
    fit_col = next((c for c in ["fitness", "h1_mean", "h3_mean"] if c in df.columns), None)
    if fit_col is None or "prism_less_score" not in df.columns:
        return np.nan
    return _corr(df["prism_less_score"].values, df[fit_col].values)


def _baseline_dev_corr(target_col, source="dev"):
    """PRISM-less correlation for a dev/immuno target from baseline_scores CSV.

    Uses prism_less_ppl with direction correction for Tm2/Titer.
    """
    p = PROJECT / f"data/prism_results/3.zero-shot/baseline_scores/{source}_baseline.csv"
    if not p.exists():
        return np.nan
    df = pd.read_csv(p)
    if "prism_less_ppl" not in df.columns or target_col not in df.columns:
        return np.nan
    x = df["prism_less_ppl"].values
    if target_col in ("Tm2", "Titer"):
        x = -x
    return _corr(x, df[target_col].values)


# ============================================================================
# Binding (3 datasets)
# ============================================================================
def run_binding():
    # Baseline PRISM-less rho from JSON
    with open(PROJECT / "script/analyze/baseline_comparison/results/dms_binding_baseline.json") as f:
        base = json.load(f)

    datasets = {
        "g6.31": {
            "clean_csv": PROJECT / "data/prism_results/3.zero-shot/g6.31_benchmark_data_clean.csv",
            "ablation_csv_template": PROJECT / "data/antibody_binding/g6.31_benchmark_data_ablation{}.csv",
            "baseline_rho": base["g6.31"]["spearman_rho"],
        },
        "cr9114": {
            "clean_csv": PROJECT / "data/prism_results/3.zero-shot/cr9114_benchmark_data_clean.csv",
            "ablation_csv_template": PROJECT / "data/antibody_binding/cr9114_benchmark_data_ablation{}.csv",
            "baseline_rho": base["cr9114"]["spearman_rho"],
        },
        "trastuzumab": {
            "clean_csv": PROJECT / "data/prism_results/3.zero-shot/trastuzumab_benchmark_data_clean.csv",
            "ablation_csv_template": PROJECT / "data/antibody_binding/trastuzumab_dataset_trimmed_ablation{}.csv",
            "baseline_rho": base["trastuzumab"]["spearman_rho"],
        },
    }

    # HF dataset key mapping for PRISM Full (v44o peak) only.
    # NOTE: for cr9114 we use the H3 fitness column (`cr9114_h3`) for PRISM Full —
    # Ablations 1/2/3 and PRISM-less continue to use H1 (`fitness` col in ablation CSVs,
    # `h1_mean` col in baseline CSVs), since their fitness is tied to those files.
    hf_keys = {"g6.31": "g6.31", "cr9114": "cr9114_h3", "trastuzumab": "trastuzumab"}

    for name, cfg in datasets.items():
        print(f"\n[binding] {name}")
        values = {}
        # PRISM Full: HF summary (NGL mode — matches DPO objective direction)
        hf_key = hf_keys.get(name, name)
        values["PRISM Full"] = _hf_dms_corr(hf_key, mode="ngl")
        print(f"  PRISM Full (HF {hf_key}, ngl) [{CURRENT_METRIC}]: {values['PRISM Full']:.4f}")

        # Ablations - use evo_ab_affinity_score column
        for i in (1, 2, 3):
            p = Path(str(cfg["ablation_csv_template"]).format(i))
            if not p.exists():
                print(f"  Ablation {i}: file missing: {p}")
                continue
            adf = pd.read_csv(p)
            score_col = None
            for candidate in ["evo_ab_affinity_score", "evo_ab_score"]:
                if candidate in adf.columns:
                    score_col = candidate
                    break
            if score_col is None:
                print(f"  Ablation {i}: no score column")
                continue
            values[f"Ablation {i}"] = binding_rho_for_csv(p, score_col)
            print(f"  Ablation {i} ({score_col}): {values[f'Ablation {i}']:.4f}")

        # PRISM-less from per-variant CSV so Spearman/Pearson both supported
        baseline_tag = {"g6.31": "g6.31", "cr9114": "cr9114_h1",
                        "trastuzumab": "trastuzumab"}[name]
        values["PRISM-less"] = _baseline_dms_corr(baseline_tag)
        if not np.isfinite(values["PRISM-less"]):
            # fallback to JSON spearman if csv not yet available
            values["PRISM-less"] = cfg["baseline_rho"]
            print(f"  PRISM-less (JSON fallback): {values['PRISM-less']:.4f}")
        else:
            print(f"  PRISM-less (CSV {baseline_tag}) [{CURRENT_METRIC}]: {values['PRISM-less']:.4f}")

        fig, ax = plt.subplots(figsize=(8, 7))
        make_bar(ax, values, f"{name} — Binding Affinity", _metric_label())
        plt.tight_layout()
        save(fig, f"{name}_ablation_results")


# ============================================================================
# Developability (5 ginkgo properties)
# ============================================================================
def run_developability():
    with open(PROJECT / "script/analyze/baseline_comparison/results/developability_baseline.json") as f:
        base = json.load(f)
    g = base["ginkgo"]

    PROPS = [
        # (key, target_col, multiply_neg1, baseline_key, prism_ppl_col, display, filename)
        ("hydrophobicity", "HIC", False, "HIC", "evo_ab_ppl_final_lower",
         "Hydrophobicity (HIC)", "ablation_hydrophobicity"),
        ("reactivity", "PR_CHO", False, "PR_CHO", "evo_ab_ppl_final_lower",
         "Polyreactivity (PR_CHO)", "ablation_reactivity"),
        ("aggregation", "AC-SINS_pH7.4", False, "AC-SINS_pH7.4", "evo_ab_ppl_final_lower",
         "Self-Interaction (AC-SINS)", "ablation_aggregation"),
        ("thermalstability", "Tm2", True, "Tm2", "evo_ab_ppl_final_lower",
         "Thermal Stability (Tm2)", "ablation_thermalstability"),
        ("expression", "Titer", True, "Titer", "evo_ab_ppl_marginalized",
         "Expression (Titer)", "ablation_expression"),
    ]

    main_csv = PROJECT / "data/ginkgo/developability_v34.1b_ppl.csv"
    main_df = pd.read_csv(main_csv) if main_csv.exists() else None
    if main_df is None:
        print(f"[dev] main CSV missing: {main_csv}")
        return

    # Column in the v34.1b PPL file for PRISM Full
    # (evo_ab_ppl_v34.1b_final_lower is unambiguously PRISM Full)
    prism_full_col = "evo_ab_ppl_v34.1b_final_lower"

    for key, target_col, mul_neg1, base_key, prism_col, display, fname in PROPS:
        print(f"\n[dev] {key}")
        values = {}
        # PRISM Full: HF germline PLL rho (direction already encoded in hf summary)
        values["PRISM Full"] = _hf_dev_corr(target_col, source="dev")
        print(f"  PRISM Full (HF gl_ppl vs {target_col}) [{CURRENT_METRIC}]: {values['PRISM Full']:.4f}")
        for i in (1, 2, 3):
            p = PROJECT / f"data/ginkgo/developability_data_ablation{i}.csv"
            values[f"Ablation {i}"] = rho_for_ablation_csv(p, target_col, mul_neg1)
            print(f"  Ablation {i}: {values[f'Ablation {i}']:.4f}")
        # PRISM-less from per-seq baseline CSV
        pl_val = _baseline_dev_corr(target_col, source="dev")
        if not np.isfinite(pl_val):
            pl_val = g.get(base_key, {}).get("rho", np.nan)
            print(f"  PRISM-less (JSON fallback): {pl_val:.4f}")
        else:
            print(f"  PRISM-less (CSV dev) [{CURRENT_METRIC}]: {pl_val:.4f}")
        values["PRISM-less"] = pl_val

        fig, ax = plt.subplots(figsize=(8, 7))
        make_bar(ax, values, display, _metric_label())
        plt.tight_layout()
        save(fig, fname)


# ============================================================================
# Immunogenicity (ginkgo ADA)
# ============================================================================
def run_immunogenicity():
    with open(PROJECT / "script/analyze/baseline_comparison/results/developability_baseline.json") as f:
        base = json.load(f)

    main_csv = PROJECT / "data/prism_results/benchmarks/v34.1b/developability/immunogenicity_ppl.csv"
    if not main_csv.exists():
        print(f"[immuno] main CSV missing: {main_csv}")
        return
    main_df = pd.read_csv(main_csv)

    target_col = "ADA"
    mul_neg1 = False
    # PRISM Full (v34.1b) — marginalized PPL
    prism_col = "evo_ab_ppl_marginalized"

    print("\n[immuno]")
    values = {}
    # PRISM Full: HF germline PLL rho
    values["PRISM Full"] = _hf_dev_corr(target_col, source="immuno")
    print(f"  PRISM Full (HF gl_ppl vs {target_col}) [{CURRENT_METRIC}]: {values['PRISM Full']:.4f}")
    for i in (1, 2, 3):
        p = PROJECT / f"data/ginkgo/immunogenicity_ablation{i}.csv"
        values[f"Ablation {i}"] = rho_for_ablation_csv(p, target_col, mul_neg1)
        print(f"  Ablation {i}: {values[f'Ablation {i}']:.4f}")
    pl_val = _baseline_dev_corr(target_col, source="immuno")
    if not np.isfinite(pl_val):
        pl_val = base["ginkgo"].get("ADA", {}).get("rho", np.nan)
        print(f"  PRISM-less (JSON fallback): {pl_val:.4f}")
    else:
        print(f"  PRISM-less (CSV immuno) [{CURRENT_METRIC}]: {pl_val:.4f}")
    values["PRISM-less"] = pl_val

    fig, ax = plt.subplots(figsize=(8, 7))
    make_bar(ax, values, "Immunogenicity (ADA)", _metric_label())
    plt.tight_layout()
    save(fig, "immunogenicity_ablation")


if __name__ == "__main__":
    print("=" * 60 + "\n5-bar ablation plots with PRISM-less\n" + "=" * 60)
    for metric in ("spearman", "pearson"):
        globals()["CURRENT_METRIC"] = metric
        print(f"\n{'='*60}\n  METRIC: {metric.upper()}\n{'='*60}")
        run_binding()
        run_developability()
        run_immunogenicity()
    print("\nDone.")
