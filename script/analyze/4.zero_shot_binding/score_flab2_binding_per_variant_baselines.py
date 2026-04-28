#!/usr/bin/env python3
"""
Save canonical baseline per-variant LLR scores for FLAb2 binding (41 assays).

Reuses helper functions and score_fns from evaluate_flab2_baselines.py to get
per-variant scores for ESM2-35M, ESM2-650M, AbLang2, AntiBERTy, Sapiens. Saves
a long-format CSV with one row per (source_file, variant, model) so we can
compute both Spearman + Pearson per assay consistently with the canonical
aggregate file (baseline_per_protein.csv).

Output: data/features/evaluation_results/flab2_binding/per_variant_baselines.csv

Run:
    CUDA_VISIBLE_DEVICES=5 python script/analyze/3.zero-shot/score_flab2_binding_per_variant_baselines.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "script" / "analyze" / "3.zero-shot"))

import evaluate_flab2_baselines as bl  # noqa: E402


def score_protein_collect(score_fn, group_df, wt_heavy, wt_light):
    scores = []
    fitness_values = []
    for _, row in group_df.iterrows():
        var_heavy = row["heavy"]
        var_light = row["light"]
        h_muts = bl.find_mutations(wt_heavy, var_heavy)
        l_muts = bl.find_mutations(wt_light, var_light)
        if h_muts is None and len(wt_heavy) != len(var_heavy):
            continue
        if l_muts is None and len(wt_light) != len(var_light):
            continue
        h = h_muts or []
        l = l_muts or []
        if not h and not l:
            continue
        s = score_fn(wt_heavy, wt_light, h, l)
        scores.append(s)
        fitness_values.append(row["fitness"])
    if len(scores) < 10:
        return None
    return pd.DataFrame({"score": scores, "fitness": fitness_values})


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("\nLoading FLAb2 binding data...")
    df = bl.load_flab2_binding()
    sources = sorted(df["source_file"].unique())
    print(f"  {len(df)} variants across {len(sources)} proteins")

    rows = []
    for model_key in bl.ALL_MODELS:
        display = bl.MODEL_DISPLAY[model_key]
        print(f"\n{'='*70}\nModel: {display}\n{'='*70}")
        t0 = time.time()
        if model_key in bl.MODEL_IDS:
            score_fn, model_obj = bl.load_esm2(bl.MODEL_IDS[model_key], device)
        elif model_key == "ablang2":
            score_fn, model_obj = bl.load_ablang2(device)
        elif model_key == "antiberty":
            score_fn, model_obj = bl.load_antiberty(device)
        elif model_key == "sapiens":
            score_fn, model_obj = bl.load_sapiens()
        else:
            continue
        print(f"  Loaded in {time.time() - t0:.1f}s")

        for src in tqdm(sources, desc=display):
            group = df[df["source_file"] == src].copy()
            wt_heavy, wt_light = bl.identify_wt(group)
            res = score_protein_collect(score_fn, group, wt_heavy, wt_light)
            if res is None:
                continue
            res["model"] = model_key
            res["source_file"] = src
            rows.append(res)

        del score_fn
        if model_key == "sapiens":
            del model_obj
        else:
            del model_obj
            if device != "cpu":
                torch.cuda.empty_cache()

    out = pd.concat(rows, ignore_index=True)
    out_path = REPO / "data" / "features" / "evaluation_results" / "flab2_binding" / "per_variant_baselines.csv"
    out.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print(f"  rows: {len(out)} (across {out['source_file'].nunique()} sources, {out['model'].nunique()} models)")


if __name__ == "__main__":
    main()
