#!/usr/bin/env python
"""
Score 3 DMS binding datasets (g6.31, cr9114, trastuzumab) with the published
PRISM Full model downloaded from HuggingFace Hub, saving per-variant scores
(all 4 modes: gl, ngl, marginalized, exact) to CSV + reporting Spearman ρ.

Usage:
    CUDA_VISIBLE_DEVICES=4 python score_prism_hf_on_dms.py

Outputs CSVs to: data/prism_results/3.zero-shot/dms_prism_hf_scores/
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
import prism

import os
HF_MODEL_ID = "RomeroLab-Duke/prism-antibody"
MODEL_PATH = os.environ.get("PRISM_MODEL_PATH", HF_MODEL_ID)
OUT_TAG = os.environ.get("PRISM_OUT_TAG", "hf")

REPO = Path(__file__).resolve().parents[3]
BENCH = REPO / "data" / "prism_results" / "3.zero-shot"
OUT_DIR = BENCH / f"dms_prism_{OUT_TAG}_scores"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = [
    ("g6.31", "g6.31_benchmark_data_clean.csv", "fitness"),
    ("cr9114_h1", "cr9114_benchmark_data_clean.csv", "h1_mean"),
    ("cr9114_h3", "cr9114_benchmark_data_clean.csv", "h3_mean"),
    ("trastuzumab", "trastuzumab_benchmark_data_clean.csv", "fitness"),
]


def find_wt_row(df):
    # n_mut == 0 if present, else Mutations == 'WT'
    if "n_mut" in df.columns:
        wt = df[df["n_mut"] == 0]
        if len(wt) > 0:
            return wt.iloc[0]
    if "Mutations" in df.columns:
        wt = df[df["Mutations"].astype(str).str.upper() == "WT"]
        if len(wt) > 0:
            return wt.iloc[0]
    # Fallback: most common sequence
    return df.iloc[df["fv_heavy"].value_counts().index[0] == df["fv_heavy"]].iloc[0]


def score_dataset(model, df, fitness_col, name, batch_size=32):
    wt_row = find_wt_row(df)
    wt_h = wt_row["fv_heavy"]
    wt_l = wt_row["fv_light"]
    print(f"  WT: VH={len(wt_h)}, VL={len(wt_l)}")

    # Filter variants: keep rows whose VH/VL have same length as WT and have finite fitness
    valid_rows = []
    for idx, row in df.iterrows():
        vh = row["fv_heavy"]
        vl = row["fv_light"]
        if len(vh) != len(wt_h) or len(vl) != len(wt_l):
            continue
        if vh == wt_h and vl == wt_l:
            continue  # skip WT itself
        fitness = row.get(fitness_col, None)
        if fitness is None or not np.isfinite(fitness):
            continue
        valid_rows.append((idx, vh, vl, fitness))
    print(f"  Valid variants: {len(valid_rows)}")
    if len(valid_rows) < 10:
        return None

    # Chunk variants to fit memory (score_mutations internally batches per-position;
    # we feed ~200 variants at a time as outer batch)
    outer_bs = 200
    all_scores = {"gl": [], "ngl": [], "marginalized": [], "exact": []}
    all_indices = []
    all_fitness = []

    t0 = time.time()
    for start in tqdm(range(0, len(valid_rows), outer_bs), desc=f"  {name}"):
        chunk = valid_rows[start:start + outer_bs]
        chunk_idx = [x[0] for x in chunk]
        chunk_vh = [x[1] for x in chunk]
        chunk_vl = [x[2] for x in chunk]
        chunk_fit = [x[3] for x in chunk]

        n = len(chunk_vh)
        results = model.score_mutations(
            wt=[wt_h] * n,
            mutant=chunk_vh,
            wt_light_chains=[wt_l] * n,
            mut_light_chains=chunk_vl,
            batch_size=batch_size,
        )
        # score_mutations returns list of dicts when mutant is a list
        if isinstance(results, dict):
            results = [results]

        for res in results:
            all_scores["gl"].append(res["gl"]["score"])
            all_scores["ngl"].append(res["ngl"]["score"])
            all_scores["marginalized"].append(res["marginalized"]["score"])
            all_scores["exact"].append(res["exact"]["score"])
        all_indices.extend(chunk_idx)
        all_fitness.extend(chunk_fit)

    elapsed = time.time() - t0
    print(f"  Elapsed: {elapsed:.1f}s ({len(valid_rows)/elapsed:.2f} variants/s)")

    out_df = pd.DataFrame({
        "row_idx": all_indices,
        "fv_heavy": [valid_rows[i][1] for i in range(len(valid_rows))],
        "fv_light": [valid_rows[i][2] for i in range(len(valid_rows))],
        "fitness": all_fitness,
        "prism_score_gl": all_scores["gl"],
        "prism_score_ngl": all_scores["ngl"],
        "prism_score_marginalized": all_scores["marginalized"],
        "prism_score_exact": all_scores["exact"],
    })

    spearmans = {}
    for mode in ["gl", "ngl", "marginalized", "exact"]:
        s = np.array(all_scores[mode])
        f = np.array(all_fitness)
        ok = np.isfinite(s) & np.isfinite(f)
        if ok.sum() > 10:
            rho, _ = spearmanr(s[ok], f[ok])
            spearmans[mode] = rho
    return out_df, spearmans


def main():
    print(f"Loading PRISM from: {MODEL_PATH}")
    model = prism.pretrained(MODEL_PATH, device="cuda")
    print(f"Model loaded on {model.device}")

    summary = []
    for name, csv_name, fitness_col in DATASETS:
        out_path = OUT_DIR / f"{name}_prism_hf_scores.csv"
        if out_path.exists():
            print(f"\n{'='*70}\n[SKIP] {name}: output already exists at {out_path}\n{'='*70}")
            continue
        csv_path = BENCH / csv_name
        print(f"\n{'='*70}\n{name}: {csv_path.name} ({fitness_col})\n{'='*70}")
        df = pd.read_csv(csv_path)
        print(f"  Loaded {len(df)} rows")

        result = score_dataset(model, df, fitness_col, name)
        if result is None:
            print(f"  SKIPPED")
            continue
        out_df, spearmans = result

        out_path = OUT_DIR / f"{name}_prism_hf_scores.csv"
        out_df.to_csv(out_path, index=False)
        print(f"  Saved: {out_path}")

        print(f"  Spearman ρ by mode:")
        for mode, rho in spearmans.items():
            print(f"    {mode:<15}: {rho:+.4f}")
        summary.append({"dataset": name, "fitness_col": fitness_col,
                        "n_variants": len(out_df), **spearmans})

    sum_df = pd.DataFrame(summary)
    sum_path = OUT_DIR / "summary_spearman.csv"
    sum_df.to_csv(sum_path, index=False)
    print(f"\n{'='*70}\nSummary saved: {sum_path}\n{'='*70}")
    print(sum_df.to_string(index=False))


if __name__ == "__main__":
    main()
