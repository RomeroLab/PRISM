#!/usr/bin/env python3
"""
Evaluate PRISM + baselines on DMS binding datasets (G6.31, CR9114, Trastuzumab).

PRISM: masked_individual + wt_center scoring (same as FLAb2 pipeline).
Baselines: Spearman ρ from pre-computed LLR scores in benchmark_data_clean.csv.

Outputs per-variant PRISM signals and per-protein Spearman ρ, then merges into
FLAb2 per_protein_results.csv / baseline_per_protein_results.csv.

Usage:
    CUDA_VISIBLE_DEVICES=6 python evaluate_dms_binding.py
    CUDA_VISIBLE_DEVICES=6 python evaluate_dms_binding.py --datasets g6.31 cr9114
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

# Reuse scoring functions from evaluate_flab2_binding
from evaluate_flab2_binding import (
    extract_wt_signals,
    find_mutations,
    format_sequence,
    score_variants_masked_individual,
)
import prism

REPO_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK = REPO_ROOT / "data" / "prism_results" / "3.zero-shot"
FLAB2_DIR = REPO_ROOT / "data" / "features" / "evaluation_results" / "flab2_binding"

# Default to the published HuggingFace Hub checkpoint. Pass a local
# /path/to/file.ckpt via --checkpoint to use a custom-trained model.
DEFAULT_CHECKPOINT = "RomeroLab-Duke/prism-antibody"

BASELINE_COLS = ["esm2_35m_score", "esm2_650m_score", "ablang2_score",
                 "antiberty_score", "sapiens_score"]
BASELINE_NAMES = {
    "esm2_35m_score": "esm2_35m", "esm2_650m_score": "esm2_650m",
    "ablang2_score": "ablang2", "antiberty_score": "antiberty",
    "sapiens_score": "sapiens",
}

# Dataset configs: (csv_name, source_file_for_flab2, fitness_col, assay_type, study)
DATASET_CONFIGS = {
    "g6.31": {
        "csv": "g6.31_benchmark_data_clean.csv",
        "evaluations": [
            {
                "source_file": "koenig2017mutational_kd_g6.csv",
                "fitness_col": "fitness",
                "assay_type": "SPR Kd",
                "study": "Mutational analysis of a therapeutic antibody",
            },
        ],
    },
    "cr9114": {
        "csv": "cr9114_benchmark_data_clean.csv",
        "evaluations": [
            {
                "source_file": "phillips2021binding_cr9114_h1_kd.csv",
                "fitness_col": "h1_mean",
                "assay_type": "SPR Kd",
                "study": "Binding affinity landscapes constrain the evolution of broadly neutralizing anti-influenza antibodies",
            },
            {
                "source_file": "phillips2021binding_cr9114_h3_kd.csv",
                "fitness_col": "h3_mean",
                "assay_type": "SPR Kd",
                "study": "Binding affinity landscapes constrain the evolution of broadly neutralizing anti-influenza antibodies",
            },
        ],
    },
    "trastuzumab": {
        "csv": "trastuzumab_benchmark_data_clean.csv",
        "evaluations": [
            {
                "source_file": "shanehsazzadeh2023unlocking_zerokd_trastuzumab.csv",
                "fitness_col": "fitness",
                "assay_type": "SPR Kd",
                "study": "Unlocking de novo antibody design with generative artificial intelligence",
            },
        ],
    },
}


def evaluate_dms_prism(model_wrapper, df, fitness_col, device, batch_size=128):
    """Run PRISM masked_individual + wt_center scoring on a DMS dataset.

    Args:
        df: DataFrame with fv_heavy, fv_light, n_mut columns + fitness_col
        fitness_col: column name for fitness values

    Returns:
        dict with signal-level Spearman ρ and n_variants
    """
    # Identify WT (n_mut=0)
    wt_rows = df[df["n_mut"] == 0]
    if len(wt_rows) == 0:
        print("    WARNING: No n_mut=0 row found, using most common heavy chain")
        heavy_counts = df["fv_heavy"].value_counts()
        wt_heavy = heavy_counts.index[0]
        wt_row = df[df["fv_heavy"] == wt_heavy].iloc[0]
        wt_light = wt_row["fv_light"]
    else:
        wt_heavy = wt_rows.iloc[0]["fv_heavy"]
        wt_light = wt_rows.iloc[0]["fv_light"]

    wt_formatted = format_sequence(wt_heavy, wt_light)

    # Extract WT reference signals
    print(f"    Extracting WT signals (VH={len(wt_heavy)}, VL={len(wt_light)})...")
    wt_signals = extract_wt_signals(model_wrapper, wt_formatted, device)

    # Prepare variants (exclude WT and length mismatches)
    variants_info = []
    fitness_values = []
    n_skipped_len = 0
    n_skipped_wt = 0

    for _, row in df.iterrows():
        vh, vl = row["fv_heavy"], row["fv_light"]
        muts = find_mutations(wt_heavy, wt_light, vh, vl)
        if muts is None:
            if len(vh) != len(wt_heavy) or len(vl) != len(wt_light):
                n_skipped_len += 1
            else:
                n_skipped_wt += 1
            continue

        fmt = format_sequence(vh, vl)
        fitness_val = row[fitness_col]
        if not np.isfinite(fitness_val):
            continue
        variants_info.append((muts, fmt, fitness_val))
        fitness_values.append(fitness_val)

    print(f"    Variants: {len(variants_info)}, skipped WT: {n_skipped_wt}, "
          f"skipped length: {n_skipped_len}")

    if len(variants_info) < 10:
        return None

    fitness = np.array(fitness_values)

    # Score with progress indication
    total_masks = sum(len(muts) for muts, _, _ in variants_info)
    print(f"    Total masked positions: {total_masks:,} "
          f"(~{total_masks // batch_size:,} batches)")

    scores = score_variants_masked_individual(
        model_wrapper, variants_info, wt_signals, device, batch_size=batch_size
    )
    if scores is None:
        return None

    # Compute Spearman ρ
    results = {"n_variants": len(variants_info)}
    for method_name, method_scores in scores.items():
        valid = np.isfinite(method_scores) & np.isfinite(fitness)
        if valid.sum() < 10:
            results[method_name] = np.nan
            continue
        rho, _ = spearmanr(method_scores[valid], fitness[valid])
        results[method_name] = rho

    return results


def evaluate_dms_baselines(df, fitness_col):
    """Compute baseline Spearman ρ from pre-computed LLR scores."""
    results = []
    for col, model_name in BASELINE_NAMES.items():
        if col not in df.columns:
            continue
        scores = df[col].values
        fitness = df[fitness_col].values
        valid = np.isfinite(scores) & np.isfinite(fitness)
        if valid.sum() < 10:
            rho = np.nan
            n = 0
        else:
            rho, _ = spearmanr(scores[valid], fitness[valid])
            n = int(valid.sum())
        results.append({"llr_score": rho, "n_variants": n, "model": model_name})
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                        help=("HuggingFace Hub model id (default: %(default)s) "
                              "or path to a local .ckpt file"))
    parser.add_argument("--model", default="prism",
                        help="Display name for the loaded model in output CSVs")
    parser.add_argument("--datasets", nargs="+", default=["g6.31", "cr9114", "trastuzumab"],
                        choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--no_merge", action="store_true",
                        help="Don't merge into FLAb2 results")
    args = parser.parse_args()

    device = args.device
    model_name = args.model

    # Load PRISM model (HF Hub id or local checkpoint path)
    ckpt = args.checkpoint
    print(f"Loading PRISM {model_name} from {ckpt}...")
    model_wrapper = prism.pretrained(ckpt, device=device)
    model_wrapper.model.eval()

    all_prism_rows = []
    all_baseline_rows = []

    for ds_name in args.datasets:
        config = DATASET_CONFIGS[ds_name]
        csv_path = BENCHMARK / config["csv"]
        df = pd.read_csv(csv_path)
        print(f"\n{'='*70}")
        print(f"Dataset: {ds_name} ({len(df):,} variants)")
        print(f"{'='*70}")

        for eval_cfg in config["evaluations"]:
            fitness_col = eval_cfg["fitness_col"]
            source_file = eval_cfg["source_file"]
            assay_type = eval_cfg["assay_type"]
            study = eval_cfg["study"]

            print(f"\n  --- {source_file} (fitness={fitness_col}) ---")

            # PRISM scoring
            t0 = time.time()
            prism_result = evaluate_dms_prism(
                model_wrapper, df, fitness_col, device, batch_size=args.batch_size
            )
            elapsed = time.time() - t0

            if prism_result is not None:
                prism_row = {
                    "model": model_name,
                    "source_file": source_file,
                    "assay_type": assay_type,
                    "study": study,
                    "elapsed_s": elapsed,
                    **prism_result,
                }
                all_prism_rows.append(prism_row)

                # Print PRISM results
                print(f"    PRISM ({elapsed:.1f}s):")
                for k, v in prism_result.items():
                    if k != "n_variants" and isinstance(v, float):
                        print(f"      {k}: {v:+.4f}")
                print(f"      n_variants: {prism_result['n_variants']:,}")

            # Baseline scoring
            baseline_results = evaluate_dms_baselines(df, fitness_col)
            for br in baseline_results:
                br["source_file"] = source_file
                br["assay_type"] = assay_type
                br["study"] = study
                br["elapsed_s"] = 0
                all_baseline_rows.append(br)
                print(f"    {br['model']}: ρ={br['llr_score']:+.4f} (n={br['n_variants']:,})")

    # Cleanup
    del model_wrapper
    torch.cuda.empty_cache()

    # Save standalone results
    out_dir = FLAB2_DIR / "dms_additions"
    out_dir.mkdir(parents=True, exist_ok=True)

    prism_df = pd.DataFrame(all_prism_rows)
    baseline_df = pd.DataFrame(all_baseline_rows)

    prism_df.to_csv(out_dir / "dms_prism_results.csv", index=False)
    baseline_df.to_csv(out_dir / "dms_baseline_results.csv", index=False)
    print(f"\nSaved standalone results to {out_dir}/")

    if args.no_merge:
        print("Skipping merge (--no_merge flag set)")
        return

    # Merge into FLAb2 results
    print(f"\n{'='*70}")
    print("MERGING INTO FLAB2 RESULTS")
    print(f"{'='*70}")

    prism_results_path = FLAB2_DIR / "per_protein_results.csv"
    baseline_results_path = FLAB2_DIR / "baseline_per_protein_results.csv"

    existing_prism = pd.read_csv(prism_results_path)
    existing_baseline = pd.read_csv(baseline_results_path)
    print(f"  Existing PRISM: {len(existing_prism)} rows")
    print(f"  Existing baseline: {len(existing_baseline)} rows")

    # Remove any existing entries for these source_files
    new_sources = {r["source_file"] for r in all_prism_rows}
    existing_prism = existing_prism[~existing_prism["source_file"].isin(new_sources)]
    existing_baseline = existing_baseline[~existing_baseline["source_file"].isin(new_sources)]

    updated_prism = pd.concat([existing_prism, prism_df], ignore_index=True)
    updated_baseline = pd.concat([existing_baseline, baseline_df], ignore_index=True)

    updated_prism.to_csv(prism_results_path, index=False)
    updated_baseline.to_csv(baseline_results_path, index=False)

    print(f"  Updated PRISM: {len(updated_prism)} rows (+{len(prism_df)})")
    print(f"  Updated baseline: {len(updated_baseline)} rows (+{len(baseline_df)})")

    # Summary
    v34b = updated_prism[updated_prism["model"] == model_name]
    print(f"\n  {model_name} total: {len(v34b)} proteins, "
          f"{v34b['n_variants'].sum():,} variants")

    # Print signal summary
    signal_cols = [c for c in v34b.columns if c.endswith("_sum") or c.endswith("_mean")]
    if signal_cols:
        print(f"\n  {'Signal':<25} {'Mean ρ':>8} {'Median ρ':>10}")
        print("  " + "-" * 45)
        for col in signal_cols:
            vals = v34b[col].dropna()
            if len(vals) > 0:
                print(f"  {col:<25} {vals.mean():>8.4f} {vals.median():>10.4f}")


if __name__ == "__main__":
    main()
