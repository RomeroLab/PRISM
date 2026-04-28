#!/usr/bin/env python
"""
FLAb2 developability baselines: PPL for each of the 5 FLAb2 properties
(aggregation/ACSINS, thermostability/DSC, immunogenicity/ADA,
 polyreactivity/PSR, expression/HEK).

For each property:
  1. Load FLAb2 parquet, filter by assay_type.
  2. Dedup on (heavy, light) taking mean of fitness per pair.
  3. Compute per-antibody pseudo-PPL with 5 baselines using the cached
     per-antibody position batching from benchmark_developability_cached.py.
  4. Save <property>_flab2_ppl_scores.csv and print Spearman rho vs fitness.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_developability_cached import (
    evaluate_esm2, evaluate_ablang2, evaluate_antiberty, evaluate_sapiens_parallel
)

REPO_ROOT = Path(__file__).resolve().parents[3]

PROPERTY_CONFIG = {
    "aggregation":      {"file": "flab2_aggregation.parquet",      "assay_types": ["ACSINS"]},
    "thermostability":  {"file": "flab2_thermostability.parquet",  "assay_types": ["DSC"]},
    "immunogenicity":   {"file": "flab2_immunogenicity.parquet",   "assay_types": ["ADA"]},
    "polyreactivity":   {"file": "flab2_polyreactivity.parquet",   "assay_types": ["PSR"]},
    "expression":       {"file": "flab2_expression.parquet",       "assay_types": ["HEK"]},
}


def load_and_dedup(property_name: str) -> pd.DataFrame:
    cfg = PROPERTY_CONFIG[property_name]
    df = pd.read_parquet(REPO_ROOT / "data" / "FLAb" / cfg["file"])
    df = df[df["light"].notna() & df["heavy"].notna()].copy()
    df = df[df["assay_type"].isin(cfg["assay_types"])]
    # Collapse replicate measurements of the same antibody: mean fitness
    agg = (
        df.groupby(["heavy", "light"], as_index=False)
          .agg(fitness=("fitness", "mean"), assay_type=("assay_type", "first"),
               study=("study", "first"), n_replicates=("fitness", "size"))
    )
    return agg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--properties", nargs="+", default=list(PROPERTY_CONFIG.keys()),
                        choices=list(PROPERTY_CONFIG.keys()))
    parser.add_argument("--models", nargs="+",
                        default=["esm2_35m", "esm2_650m", "ablang2", "antiberty", "sapiens"],
                        choices=["esm2_35m", "esm2_650m", "ablang2", "antiberty", "sapiens"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sapiens_workers", type=int, default=16)
    parser.add_argument("--out_dir", default=str(REPO_ROOT / "data" / "prism_results" / "3.zero-shot" / "baseline_scores_fixed"))
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for prop in args.properties:
        print(f"\n{'=' * 70}\n[{prop}]\n{'=' * 70}")
        df = load_and_dedup(prop)
        print(f"  {len(df)} unique (heavy, light) pairs")
        heavy_list = df["heavy"].tolist()
        light_list = df["light"].tolist()
        concat_list = [h + l for h, l in zip(heavy_list, light_list)]

        results = {}
        for model_key in args.models:
            t0 = time.time()
            print(f"\n  === {model_key} ===")
            try:
                if model_key in ("esm2_35m", "esm2_650m"):
                    model_id = {"esm2_35m": "facebook/esm2_t12_35M_UR50D",
                                "esm2_650m": "facebook/esm2_t33_650M_UR50D"}[model_key]
                    ppls = evaluate_esm2(model_id, concat_list, device)
                elif model_key == "ablang2":
                    ppls = evaluate_ablang2(heavy_list, light_list, device)
                elif model_key == "antiberty":
                    ppls = evaluate_antiberty(heavy_list, light_list, device)
                elif model_key == "sapiens":
                    ppls = evaluate_sapiens_parallel(heavy_list, light_list, args.sapiens_workers)
                else:
                    continue
                results[model_key] = ppls
                valid = [p for p in ppls if p != float("inf")]
                print(f"    {time.time()-t0:.1f}s | mean={np.mean(valid):.3f}, median={np.median(valid):.3f}")
            except Exception as e:
                print(f"    FAILED: {e}")
                import traceback; traceback.print_exc()
                results[model_key] = [float("inf")] * len(df)

        # Attach PPLs
        for mk, ppls in results.items():
            df[f"{mk}_ppl"] = ppls

        out_path = out_dir / f"flab2_{prop}_ppl_scores.csv"
        df.to_csv(out_path, index=False)
        print(f"\n  Saved: {out_path}")

        # Correlation with fitness
        print(f"\n  Spearman rho ({prop}_ppl vs fitness):")
        for mk in args.models:
            col = f"{mk}_ppl"
            if col not in df.columns:
                continue
            v = df[[col, "fitness"]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(v) < 10:
                continue
            rho, _ = spearmanr(v[col], v["fitness"])
            print(f"    {mk:<14} rho={rho:+.4f}  n={len(v)}")
            summary_rows.append({"property": prop, "model": mk, "spearman_rho": rho, "n": len(v)})

    # Combined summary
    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / "flab2_developability_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n\nSummary saved: {summary_path}")
    print(summary_df.pivot_table(index="property", columns="model", values="spearman_rho").round(4))


if __name__ == "__main__":
    main()
