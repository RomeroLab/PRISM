#!/usr/bin/env python
# coding: utf-8
"""
Score FLAb2 binding variants with IgLM mutation-LLR.

Mirrors `predicting_binding_affinity_w_IgLM.py` but consumes FLAb2 binding
parquet from the PRISM repo. Within each source_file group we identify WT as
the most common heavy+light pair, then call `evaluate_iglm_model` (per-mutation
masked log-likelihood ratio).

Output (per source_file, mirrors DMS layout):
    ./data/flab2/binding/<source_file>.csv
    ./data/flab2/binding/<source_file>_score_correlations.csv
    ./data/flab2/binding/all_score_correlations.csv
"""

import os
import sys
from pathlib import Path

import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

for _p in Path(__file__).resolve().parents:
    if (_p / "utils" / "__init__.py").exists():
        sys.path.insert(0, str(_p))
        break
from utils.scoring_utils_batched import evaluate_iglm_model_batched

# Batch size for the WT log-prob caching pass at the start of each source_file.
# The whole WT chain fits in one forward pass at batch_size=256 on an L40S.
WT_LOGPROB_BATCH_SIZE = 256


# ============================================================
# Paths / filters (match evaluate_flab2_baselines.py in this repo)
# ============================================================
_REPO_ROOT = Path(__file__).resolve().parents[3]
FLAB2_BINDING_PATH = str(_REPO_ROOT / "data" / "FLAb" / "flab2_binding.parquet")
OUT_DIR = str(_REPO_ROOT / "data" / "flab2" / "binding")

EXCLUDE_SOURCES = {
    "koenig2017mutational_kd_g6.csv",
    "shanehsazzadeh2023unlocking_zerokd_trastuzumab.csv",
    "phillips2021binding_cr9114_h3_kd.csv",
    "phillips2021binding_cr9114_h1_kd.csv",
    "AbRank_dataset.csv.zip",
}
MIN_VARIANTS = 15

os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# Device
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Load IgLM once and reuse across source files (avoids reload per group)
from iglm import IgLM  # noqa: E402
print("Loading IgLM...")
IGLM_MODEL = IgLM()
print(f"  Model loaded on {IGLM_MODEL.device}")


# ============================================================
# Load + filter FLAb2 binding
# ============================================================
df_all = pd.read_parquet(FLAB2_BINDING_PATH)
df_all = df_all[df_all["light"].notna()].copy()
df_all = df_all[~df_all["assay_type"].isin(["bind/no bind", "predicted Kd"])]
df_all = df_all[~df_all["source_file"].isin(EXCLUDE_SOURCES)]

counts = df_all["source_file"].value_counts()
valid_sources = counts[counts >= MIN_VARIANTS].index.tolist()
df_all = df_all[df_all["source_file"].isin(valid_sources)].copy()

source_files = sorted(df_all["source_file"].unique())
print(f"Loaded {len(df_all)} variants across {len(source_files)} source files.")


def identify_wt(group_df):
    """WT = most common heavy+light combo (same logic as evaluate_flab2_baselines.py)."""
    heavy_counts = group_df["heavy"].value_counts()
    wt_heavy = heavy_counts.index[0]
    wt_rows = group_df[group_df["heavy"] == wt_heavy]
    if len(wt_rows) > 1:
        light_counts = wt_rows["light"].value_counts()
        wt_light = light_counts.index[0]
    else:
        wt_light = wt_rows.iloc[0]["light"]
    return wt_heavy, wt_light


# ============================================================
# Score each source file
# ============================================================
failed_sources = []
all_corr_results = []

for source_file in source_files:
    print(f"\nProcessing: {source_file}")

    try:
        df = df_all[df_all["source_file"] == source_file].copy().reset_index(drop=True)
        wt_heavy, wt_light = identify_wt(df)

        print(f"  WT heavy length: {len(wt_heavy)}")
        print(f"  WT light length: {len(wt_light)}")
        print(f"  Total rows: {len(df)}")

        if "IgLM_score" not in df.columns:
            df["IgLM_score"] = evaluate_iglm_model_batched(
                df=df,
                wt_heavy=wt_heavy,
                wt_light=wt_light,
                device=device,
                model=IGLM_MODEL,  # reuse loaded model across all source files
                heavy_col="heavy",
                light_col="light",
                mutation_col="mutations",  # missing column → row.get returns None, treated as non-WT
                batch_size=WT_LOGPROB_BATCH_SIZE,
            )

        out_csv = os.path.join(OUT_DIR, source_file.replace(".csv.zip", ".csv"))
        if not out_csv.endswith(".csv"):
            out_csv += ".csv"
        df.to_csv(out_csv, index=False)
        print(f"  Saved scored CSV: {out_csv}")

        # Correlations vs fitness
        score_cols = [c for c in df.columns if c.endswith("_score")]
        if not score_cols:
            print("  No columns ending with '_score' found. Skipping correlations.")
            continue

        dataset_name = os.path.splitext(os.path.basename(out_csv))[0]
        dataset_corr_rows = []

        for score_col in score_cols:
            valid_df = df[[score_col, "fitness"]].dropna()
            if len(valid_df) < 2:
                print(f"  Skipping {score_col}: fewer than 2 non-NaN rows.")
                continue
            if valid_df[score_col].nunique() < 2 or valid_df["fitness"].nunique() < 2:
                print(f"  Skipping {score_col}: constant column detected.")
                continue

            pearson_val, pearson_p = pearsonr(valid_df[score_col], valid_df["fitness"])
            spearman_val, spearman_p = spearmanr(valid_df[score_col], valid_df["fitness"])

            row = {
                "dataset": dataset_name,
                "score_col": score_col,
                "n": len(valid_df),
                "pearson": pearson_val,
                "pearson_pvalue": pearson_p,
                "spearman": spearman_val,
                "spearman_pvalue": spearman_p,
                "assay_type": df["assay_type"].iloc[0],
                "study": df["study"].iloc[0] if "study" in df.columns else "",
            }
            dataset_corr_rows.append(row)
            all_corr_results.append(row)

        if dataset_corr_rows:
            dataset_corr_df = pd.DataFrame(dataset_corr_rows).sort_values(
                "spearman", ascending=False
            )
            per_dataset_corr_path = os.path.join(
                OUT_DIR, f"{dataset_name}_score_correlations.csv"
            )
            dataset_corr_df.to_csv(per_dataset_corr_path, index=False)
            print(f"  Saved per-dataset correlations: {per_dataset_corr_path}")

            plt.figure(figsize=(10, max(4, 0.5 * len(dataset_corr_df))))
            plt.barh(dataset_corr_df["score_col"], dataset_corr_df["spearman"])
            plt.xlabel("Spearman correlation with fitness")
            plt.ylabel("Score column")
            plt.title(f"{dataset_name}: score vs fitness Spearman correlations")
            plt.gca().invert_yaxis()
            plt.tight_layout()
            figure_path = os.path.join(OUT_DIR, f"{dataset_name}_score_spearman_barplot.png")
            plt.savefig(figure_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"  Saved Spearman bar chart: {figure_path}")

    except Exception as e:
        print(f"ERROR processing {source_file}: {e}")
        failed_sources.append((source_file, str(e)))


# ============================================================
# Combined correlations CSV
# ============================================================
if all_corr_results:
    all_corr_df = pd.DataFrame(all_corr_results).sort_values(
        ["dataset", "spearman"], ascending=[True, False]
    )
    all_corr_outpath = os.path.join(OUT_DIR, "all_score_correlations.csv")
    all_corr_df.to_csv(all_corr_outpath, index=False)
    print(f"\nSaved combined correlation CSV: {all_corr_outpath}")


print(f"\n{'='*60}")
print("DONE")
print("=" * 60)
print(f"Processed: {len(source_files) - len(failed_sources)}")
print(f"Failed: {len(failed_sources)}")

if failed_sources:
    print("\nFailed source files:")
    for sf, err in failed_sources:
        print(f"  - {sf}: {err}")
