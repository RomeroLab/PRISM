#!/usr/bin/env python
# coding: utf-8

import os
import sys
import glob
from pathlib import Path

import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

for _p in Path(__file__).resolve().parents:
    if (_p / "utils" / "__init__.py").exists():
        sys.path.insert(0, str(_p))
        break
from utils.scoring_utils import evaluate_antiberty_model


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

filepaths = sorted(glob.glob("./data/DMS/c*.csv"))
if not filepaths:
    raise FileNotFoundError("No CSV files found in ./data/DMS/")

print(f"Found {len(filepaths)} CSV file(s).")

failed_files = []
all_corr_results = []

for filepath in filepaths:
    print(f"\nProcessing: {filepath}")

    try:
        df = pd.read_csv(filepath)

        required_cols = ["fv_heavy", "fv_light", "Mutations", "fitness"]
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        wt_mask = df["Mutations"].astype(str).str.strip().eq("WT")
        if wt_mask.sum() == 0:
            raise ValueError("No WT row found.")
        if wt_mask.sum() > 1:
            print("WARNING: Multiple WT rows found. Using the first one.")

        wt_row = df.loc[wt_mask].iloc[0]
        wt_heavy = str(wt_row["fv_heavy"]).strip()
        wt_light = str(wt_row["fv_light"]).strip()

        print(f"  WT heavy length: {len(wt_heavy)}")
        print(f"  WT light length: {len(wt_light)}")
        print(f"  Total rows: {len(df)}")

        if "NB_edits_antiberty_score" not in df.columns:
            df["NB_edits_antiberty_score"] = evaluate_antiberty_model(
                df=df,
                wt_heavy=wt_heavy,
                wt_light=wt_light,
                device=device,
            )

        outpath = filepath
        df.to_csv(outpath, index=False)
        print(f"  Saved scored CSV: {outpath}")

        score_cols = [c for c in df.columns if c.endswith("_score")]
        if not score_cols:
            print("  No columns ending with '_score' found. Skipping correlations.")
            continue

        dataset_name = os.path.splitext(os.path.basename(filepath))[0]
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
            }
            dataset_corr_rows.append(row)
            all_corr_results.append(row)

        if dataset_corr_rows:
            dataset_corr_df = pd.DataFrame(dataset_corr_rows).sort_values(
                "spearman", ascending=False
            )

            per_dataset_corr_path = os.path.join(
                "./data/DMS",
                f"{dataset_name}_score_correlations.csv"
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

            figure_path = os.path.join(
                "./data/DMS",
                f"{dataset_name}_score_spearman_barplot.png"
            )
            plt.savefig(figure_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"  Saved Spearman bar chart: {figure_path}")
        else:
            print("  No valid score columns for correlation analysis.")

    except Exception as e:
        print(f"ERROR processing {filepath}: {e}")
        failed_files.append((filepath, str(e)))

if all_corr_results:
    all_corr_df = pd.DataFrame(all_corr_results).sort_values(
        ["dataset", "spearman"], ascending=[True, False]
    )
    all_corr_outpath = "./data/DMS/all_score_correlations.csv"
    all_corr_df.to_csv(all_corr_outpath, index=False)
    print(f"\nSaved combined correlation CSV: {all_corr_outpath}")

print(f"\n{'='*60}")
print("DONE")
print("=" * 60)
print(f"Processed: {len(filepaths) - len(failed_files)}")
print(f"Failed: {len(failed_files)}")

if failed_files:
    print("\nFailed files:")
    for filepath, err in failed_files:
        print(f"  - {filepath}: {err}")
