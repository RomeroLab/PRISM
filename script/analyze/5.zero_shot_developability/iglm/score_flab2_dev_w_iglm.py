#!/usr/bin/env python
# coding: utf-8
"""
Score FLAb2 developability assays with IgLM forward perplexity + pseudo-perplexity.

Mirrors `scoring_gdpa_developability_w_iglm.py` but consumes the 5 FLAb2
developability parquets used by PRISM:
    self_interaction (ACSINS) - flab2_aggregation.parquet
    thermostability (DSC)     - flab2_thermostability.parquet
    immunogenicity (ADA)      - flab2_immunogenicity.parquet
    polyreactivity (PSR)      - flab2_polyreactivity.parquet
    expression (HEK)          - flab2_expression.parquet

Output (one folder per property):
    ./data/flab2/developability/<property>/scored.csv
    ./data/flab2/developability/<property>/correlations.csv
    ./data/flab2/developability/<property>/scatter_perplexity.png
    ./data/flab2/developability/<property>/scatter_pseudoperplexity.png
    ./data/flab2/developability/all_property_correlations.csv
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from iglm import IgLM

for _p in Path(__file__).resolve().parents:
    if (_p / "utils" / "__init__.py").exists():
        sys.path.insert(0, str(_p))
        break
from utils.scoring_utils_batched import add_iglm_scores_to_dataframe_batched

# Batch size for masked-position forwards within a single chain. The two chains
# of a paired antibody (~120 + ~108 positions) each fit in one forward pass at
# batch_size=256 on an L40S, eliminating chunking overhead.
PSPP_BATCH_SIZE = 256


# ============================================================
# Paths / configs (match evaluate_flab2_developability.py in this repo)
# ============================================================
_REPO_ROOT = Path(__file__).resolve().parents[3]
FLAB2_DIR = str(_REPO_ROOT / "data" / "FLAb")
OUT_ROOT = str(_REPO_ROOT / "data" / "flab2" / "developability")
os.makedirs(OUT_ROOT, exist_ok=True)

PROPERTY_CONFIG = {
    "self_interaction": {
        "file": "flab2_aggregation.parquet",
        "assay_types": ["ACSINS"],
    },
    "thermostability": {
        "file": "flab2_thermostability.parquet",
        "assay_types": ["DSC"],
    },
    "immunogenicity": {
        "file": "flab2_immunogenicity.parquet",
        "assay_types": ["ADA"],
    },
    "polyreactivity": {
        "file": "flab2_polyreactivity.parquet",
        "assay_types": ["PSR"],
    },
    "expression": {
        "file": "flab2_expression.parquet",
        "assay_types": ["HEK"],
    },
}

score_cols = ["IgLM_Perplexity", "IgLM_PseudoPerplexity"]


def load_flab2_property(prop_name):
    """Same filtering logic as PRISM's evaluate_flab2_developability.load_flab2_property."""
    cfg = PROPERTY_CONFIG[prop_name]
    df = pd.read_parquet(os.path.join(FLAB2_DIR, cfg["file"]))
    df = df[df["assay_type"].isin(cfg["assay_types"])].copy()
    df = df[df["light"].notna()].copy()
    df = df[df["fitness"].notna()].copy()
    df = df.drop_duplicates(subset=["heavy", "light"], keep="first").reset_index(drop=True)
    return df


# ============================================================
# Initialize IgLM once (shared across all properties)
# ============================================================
print("Loading IgLM...")
iglm = IgLM()


# ============================================================
# Main scoring loop
# ============================================================
all_results = []

for prop_name in PROPERTY_CONFIG:
    print(f"\n{'='*60}\nProperty: {prop_name}\n{'='*60}")

    out_dir = os.path.join(OUT_ROOT, prop_name)
    os.makedirs(out_dir, exist_ok=True)

    df = load_flab2_property(prop_name)
    print(f"Rows: {len(df)} ({PROPERTY_CONFIG[prop_name]['assay_types']})")

    if len(df) == 0:
        print(f"  No rows for {prop_name}, skipping.")
        continue

    # initialize score columns if missing
    for col in score_cols:
        if col not in df.columns:
            df[col] = np.nan

    # ---- score sequences (batched: ~5-7x faster than sequential) ----
    df = add_iglm_scores_to_dataframe_batched(
        df=df,
        model=iglm,
        heavy_col="heavy",
        light_col="light",
        perplexity_col="IgLM_Perplexity",
        pseudo_perplexity_col="IgLM_PseudoPerplexity",
        compute_perplexity=True,
        compute_pseudo_perplexity=True,
        batch_size=PSPP_BATCH_SIZE,
        show_progress=True,
    )

    scored_path = os.path.join(out_dir, "scored.csv")
    df.to_csv(scored_path, index=False)
    print(f"Saved scored dataframe: {scored_path}")

    # ---- correlations ----
    results = []
    for sc in score_cols:
        plot_df = df[[sc, "fitness"]].replace([np.inf, -np.inf], np.nan).dropna()
        n = len(plot_df)
        if n < 2:
            print(f"  Skipping {sc}: not enough non-NaN rows.")
            continue

        x = plot_df[sc].values
        y = plot_df["fitness"].values

        if np.std(x) == 0 or np.std(y) == 0:
            pr, pp, sp, spp = (np.nan,) * 4
        else:
            pr, pp = pearsonr(x, y)
            sp, spp = spearmanr(x, y)

        row = {
            "property": prop_name,
            "score_column": sc,
            "n": n,
            "pearson_r": pr,
            "pearson_p": pp,
            "spearman_rho": sp,
            "spearman_p": spp,
        }
        results.append(row)
        all_results.append(row)

    if not results:
        continue

    results_df = pd.DataFrame(results).sort_values(
        ["score_column", "spearman_rho"], ascending=[True, False]
    ).reset_index(drop=True)

    corr_path = os.path.join(out_dir, "correlations.csv")
    results_df.to_csv(corr_path, index=False)
    print(f"Saved correlations: {corr_path}")
    print(results_df.to_string(index=False))

    # ---- scatter plots ----
    def make_scatter_figure(df, results_df, score_col, out_path):
        sub = results_df[results_df["score_column"] == score_col]
        if len(sub) == 0:
            return
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        plot_df = df[[score_col, "fitness"]].replace([np.inf, -np.inf], np.nan).dropna()
        x, y = plot_df[score_col].values, plot_df["fitness"].values
        ax.scatter(x, y, alpha=0.6, s=12)
        if len(np.unique(x)) > 1:
            m, b = np.polyfit(x, y, 1)
            x_line = np.linspace(x.min(), x.max(), 200)
            ax.plot(x_line, m * x_line + b, linewidth=2)
        row = sub.iloc[0]
        ann = (
            f"Pearson r = {row.pearson_r:.3f} (p = {row.pearson_p:.2e})\n"
            f"Spearman ρ = {row.spearman_rho:.3f} (p = {row.spearman_p:.2e})\n"
            f"N = {int(row.n)}"
        )
        ax.text(0.05, 0.95, ann, transform=ax.transAxes, va="top", ha="left",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))
        ax.set_xlabel(score_col)
        ax.set_ylabel("fitness")
        ax.set_title(f"{prop_name}: {score_col} vs fitness")
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()

    make_scatter_figure(df, results_df, "IgLM_Perplexity",
                        os.path.join(out_dir, "scatter_perplexity.png"))
    make_scatter_figure(df, results_df, "IgLM_PseudoPerplexity",
                        os.path.join(out_dir, "scatter_pseudoperplexity.png"))


# ============================================================
# Combined summary
# ============================================================
if all_results:
    all_df = pd.DataFrame(all_results).sort_values(
        ["property", "score_column"]
    ).reset_index(drop=True)
    all_path = os.path.join(OUT_ROOT, "all_property_correlations.csv")
    all_df.to_csv(all_path, index=False)
    print(f"\nSaved combined correlation summary: {all_path}")
    print(all_df.to_string(index=False))

print("\nDONE")
