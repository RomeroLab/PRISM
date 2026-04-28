#!/usr/bin/env python
"""
Run HF PRISM (RomeroLab-Duke/prism-antibody) germline (GL) PLL on two
target datasets and save Spearman rho vs each target column:
  - ginkgo developability_data.csv  (5 properties: HIC, PR_CHO, AC-SINS_pH7.4, Tm2, Titer)
  - ginkgo immunogenicity.csv        (ADA)

Output:
  data/prism_results/3.zero-shot/hf_dev_scores.csv   — per-seq GL/NGL/Marg PPL
  data/prism_results/3.zero-shot/hf_immuno_scores.csv
  data/prism_results/3.zero-shot/hf_dev_immuno_spearman.csv — summary rho
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
import prism

REPO = Path(__file__).resolve().parents[3]
OUT_DIR = REPO / "data" / "prism_results" / "3.zero-shot"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def score_file(model, csv_path, vh_col, vl_col, tag):
    df = pd.read_csv(csv_path)
    print(f"\n=== {tag}: {csv_path.name} ({len(df)} rows) ===")

    heavy = df[vh_col].astype(str).tolist()
    light = df[vl_col].astype(str).tolist()

    rows = []
    batch_size = 32  # per masked position, not outer
    outer_bs = 16  # seqs per call to PLL
    for i in tqdm(range(0, len(df), outer_bs), desc=f"  PLL {tag}"):
        hv = heavy[i:i + outer_bs]
        lv = light[i:i + outer_bs]
        results = model.pseudo_log_likelihood(
            heavy_chains=hv, light_chains=lv, batch_size=batch_size
        )
        if isinstance(results, dict):
            results = [results]
        for res in results:
            rows.append({
                "gl_pll": res["gl"]["pll"],
                "gl_ppl": res["gl"]["perplexity"],
                "ngl_pll": res["ngl"]["pll"],
                "ngl_ppl": res["ngl"]["perplexity"],
                "marg_pll": res["marginalized"]["pll"],
                "marg_ppl": res["marginalized"]["perplexity"],
            })

    score_df = pd.DataFrame(rows)
    out = pd.concat([df.reset_index(drop=True), score_df], axis=1)
    out_path = OUT_DIR / f"hf_{tag}_scores.csv"
    out.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")
    return out


def rho(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 10:
        return np.nan, 0
    return float(spearmanr(x[m], y[m]).correlation), int(m.sum())


def main():
    import os
    model_path = os.environ.get("PRISM_MODEL_PATH", "RomeroLab-Duke/prism-antibody")
    out_tag = os.environ.get("PRISM_OUT_TAG", "hf")
    # Redirect outputs to tag-specific filenames if not HF
    if out_tag != "hf":
        global OUT_DIR
        # Append tag suffix to output filenames inside score_file via a monkey-patched pattern
        # Simpler: reuse existing filenames but under a tagged subfolder
        OUT_DIR = (REPO / "data" / "prism_results" / "3.zero-shot" / f"dev_scores_{out_tag}")
        OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading PRISM from: {model_path}")
    model = prism.pretrained(model_path, device="cuda")

    # ---- Developability (ginkgo) ----
    dev = score_file(
        model,
        REPO / "data/ginkgo/developability_data.csv",
        "vh_protein_sequence", "vl_protein_sequence",
        "dev",
    )

    # ---- Immunogenicity (ginkgo) ----
    imm = score_file(
        model,
        REPO / "data/ginkgo/immunogenicity.csv",
        "vh_protein_sequence", "vl_protein_sequence",
        "immuno",
    )

    # Spearman vs properties. Note: for each property, the user's ablation
    # plot uses raw PPL (higher PPL = less likely). Direction correction for
    # Tm2 / Titer (higher = better) needs -PPL to match fitness direction.
    PROPS = [
        # (file, target_col, use_col, invert_for_fitness_direction)
        ("dev", "HIC", None, False),
        ("dev", "PR_CHO", None, False),
        ("dev", "AC-SINS_pH7.4", None, False),
        ("dev", "Tm2", None, True),
        ("dev", "Titer", None, True),
        ("immuno", "ADA", None, False),
    ]

    summary = []
    for tag, target_col, _, invert in PROPS:
        score_df = dev if tag == "dev" else imm
        if target_col not in score_df.columns:
            print(f"  [skip] {tag}/{target_col}: not in df")
            continue
        y = score_df[target_col].values
        for mode in ["gl_ppl", "ngl_ppl", "marg_ppl"]:
            x = score_df[mode].values
            if invert:
                x = -x
            r, n = rho(x, y)
            summary.append({
                "dataset": tag, "target": target_col, "mode": mode,
                "inverted_for_direction": invert, "n": n, "spearman": r,
            })
            print(f"  {tag}/{target_col} [{mode}{' (-)'if invert else ''}]: rho={r:+.4f} (n={n})")

    sum_df = pd.DataFrame(summary)
    sum_df.to_csv(OUT_DIR / "hf_dev_immuno_spearman.csv", index=False)
    print(f"\nSummary saved: {OUT_DIR / 'hf_dev_immuno_spearman.csv'}")
    print(sum_df.to_string(index=False))


if __name__ == "__main__":
    main()
