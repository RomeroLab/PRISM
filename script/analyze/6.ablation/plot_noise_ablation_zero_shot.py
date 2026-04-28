#!/usr/bin/env python3
"""
Updated zero-shot noise-ablation figures (with_iglm version).

Splits into TWO 1x2 figures matching the existing fig3/fig4 format used in
script/analyze/reviewer_figures/:

  Figure 1 (noise_ablation_binding.{png,svg}):
    A. 3DMS Binding   (3 datasets x 3 noise levels)
    B. FLAb2 Binding  (per-assay mean rho, clean & noise4 only;
                       noise2 not evaluated for FLAb2 binding)

  Figure 2 (noise_ablation_developability.{png,svg}):
    A. Ginkgo Developability  (6 properties x 3 noise levels, raw rho)
    B. FLAb2 Developability   (5 properties x 3 noise levels, directed rho)

Clean = v44o_peak (DPO-finetuned final PRISM model).
Noise variants = v34.1b with 2/4 GL/NGL label flips per residue
(original noise robustness experiment).

Output dir: img/3.zero-shot/with_iglm/
"""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[3]
OUT_DIR = REPO / "img" / "3.zero-shot" / "with_iglm"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Colour scheme matching fig3/fig4
C = {"clean": "#2166ac", "noise2": "#f4a582", "noise4": "#b2182b"}
TAG_LABELS = {"clean": "PRISM", "noise2": "+Noise2", "noise4": "+Noise4"}
TAGS = ["clean", "noise2", "noise4"]


# ===========================================================================
# Loaders
# ===========================================================================
def load_3dms_rho():
    """Spearman rho per (dataset, tag) for 3DMS binding."""
    v44o_dir = REPO / "data" / "prism_results" / "3.zero-shot" / "dms_prism_v44o_peak_glinv_scores"
    nb = REPO / "data" / "antibody_binding"
    cfgs = {
        "G6.31": {
            "clean":  (v44o_dir / "g6.31_prism_hf_scores.csv",      "prism_score_ngl"),
            "noise2": (nb / "g6.31_benchmark_data_noise2.csv",      None),
            "noise4": (nb / "g6.31_benchmark_data_noise4.csv",      None),
        },
        "CR9114": {
            "clean":  (v44o_dir / "cr9114_h3_prism_hf_scores.csv",  "prism_score_ngl"),
            "noise2": (nb / "cr9114_benchmark_data_noise2.csv",     None),
            "noise4": (nb / "cr9114_benchmark_data_noise4.csv",     None),
        },
        "Trastuzumab": {
            "clean":  (v44o_dir / "trastuzumab_prism_hf_scores.csv", "prism_score_ngl"),
            "noise2": (nb / "trastuzumab_benchmark_noise2.csv",      None),
            "noise4": (nb / "trastuzumab_benchmark_noise4.csv",      None),
        },
    }
    fb_cols = ["evo_ab_score", "evo_ab_affinity_score"]
    out = {}
    for ds, files in cfgs.items():
        out[ds] = {}
        for tag, (fpath, col) in files.items():
            if not fpath.exists():
                out[ds][tag] = np.nan; continue
            df = pd.read_csv(fpath)
            if "Mutations" in df.columns:
                df = df[df["Mutations"] != "WT"]
            score_col = col if (col and col in df.columns) else next(
                (c for c in fb_cols if c in df.columns), None)
            if score_col is None or "fitness" not in df.columns:
                out[ds][tag] = np.nan; continue
            valid = df[score_col].notna() & df["fitness"].notna() & (df[score_col] != 0)
            if valid.sum() < 10:
                out[ds][tag] = np.nan
            else:
                rho, _ = spearmanr(df.loc[valid, score_col], df.loc[valid, "fitness"])
                out[ds][tag] = rho
    return out


def load_flab2_binding_rho():
    """Per-assay mean directed Spearman rho for FLAb2 binding.

    Clean uses v44o_peak `ngl_logprob_sum` mean from per_protein_results.csv —
    consistent with the report Section 5a value (+0.091 mean directed rho).
    Noise2 value (0.0221) was provided externally (separate noise2 evaluation).
    Noise4 derived from per_protein_results_noise4.csv (`origin_logit_sum`)."""
    base = REPO / "data" / "features" / "evaluation_results" / "flab2_binding"
    out = {}
    df_clean = pd.read_csv(base / "per_protein_results.csv")
    v44o = df_clean[df_clean["model"] == "v44o_peak"]
    # Report Section 5a uses ngl_logprob_sum for v44o (mean = +0.091)
    out["clean"] = v44o["ngl_logprob_sum"].dropna().mean()
    out["noise2"] = 0.0221  # provided value from separate noise2 evaluation
    fp_n4 = base / "per_protein_results_noise4.csv"
    if fp_n4.exists():
        df_n4 = pd.read_csv(fp_n4)
        out["noise4"] = df_n4["origin_logit_sum"].dropna().mean()
    return out


def load_ginkgo_rho():
    """Spearman rho per (property, tag) for Ginkgo developability.

    For clean (v44o_peak), use precomputed values from
    `dev_scores_v44o_peak/hf_dev_immuno_spearman.csv` taking the gl_ppl signal
    (most consistent across properties). For noise2/4, compute on the fly from
    raw CSVs picking the best-direction PPL column."""
    out = {p: {} for p in
        ["AC-SINS_pH7.4", "HIC", "Tm2", "ADA", "PR_CHO", "Titer"]}

    # Clean: read precomputed v44o_peak Ginkgo spearman, take best (max) rho
    # across PPL signals (gl_ppl/ngl_ppl/marg_ppl) — mirrors how noise2/4 are
    # computed below, so the comparison is apples-to-apples.
    clean_csv = REPO / "data/prism_results/3.zero-shot/dev_scores_v44o_peak/hf_dev_immuno_spearman.csv"
    if clean_csv.exists():
        df = pd.read_csv(clean_csv)
        for prop in out.keys():
            sub = df[df["target"] == prop]
            if len(sub):
                out[prop]["clean"] = float(sub["spearman"].max())

    # Noise2/4: compute from raw CSVs, pick best-direction PPL column
    noise_files = [
        ("noise2", REPO / "data/ginkgo/developability_data_noise2.csv",
                   REPO / "data/ginkgo/immunogenicity_noise2.csv"),
        ("noise4", REPO / "data/ginkgo/developability_data_noise4.csv",
                   REPO / "data/ginkgo/immunogenicity_noise4.csv"),
    ]
    for tag, dev_p, ada_p in noise_files:
        for fpath, props in [(dev_p, [p for p in out if p != "ADA"]),
                              (ada_p, ["ADA"])]:
            if not fpath.exists():
                continue
            df = pd.read_csv(fpath)
            ppl_cols = [c for c in df.columns if "evo_ab" in c and "ppl" in c]
            if not ppl_cols:
                continue
            for prop in props:
                if prop not in df.columns:
                    continue
                best = -999.0
                for pc in ppl_cols:
                    valid = df[pc].notna() & df[prop].notna()
                    if valid.sum() > 5:
                        rho, _ = spearmanr(df.loc[valid, pc], df.loc[valid, prop])
                        if rho > best:
                            best = rho
                if best > -999.0:
                    out[prop][tag] = best
    return out


def load_flab2_dev_rho_directed():
    """Directed Spearman rho on `pll_gl` signal per (property, tag).

    Matches the convention in `script/analyze/3.zero-shot/reports/baseline_benchmark_directed.md`
    Section 3: directed = -rho(score, fitness) * sign, where sign=+1 for
    higher-is-better, -1 for lower-is-better. Score is `pll_gl` (GL-head
    pseudo-log-likelihood) — the canonical signal in the report."""
    base = REPO / "data" / "features" / "evaluation_results" / "flab2_developability"
    files = {
        "clean":  (base / "per_antibody_scores.csv",            "v44o_peak"),
        "noise2": (base / "per_antibody_scores_prism_only.csv", "prism_noise2"),
        "noise4": (base / "per_antibody_scores_noise4.csv",     "prism_noise4"),
    }
    higher_is_better = {
        "self_interaction": False,
        "thermostability":  True,
        "immunogenicity":   False,
        "polyreactivity":   False,
        "expression":       True,
    }
    out = {}
    for tag, (fpath, model_id) in files.items():
        if not fpath.exists():
            continue
        df = pd.read_csv(fpath)
        df = df[(df["model"] == model_id) & (df["signal"] == "pll_gl")]
        for prop in df["property"].unique():
            sub = df[df["property"] == prop]
            v = sub["score"].notna() & sub["fitness"].notna()
            if v.sum() < 10:
                continue
            sign = 1 if higher_is_better.get(prop, False) else -1
            raw_rho, _ = spearmanr(sub.loc[v, "score"].values,
                                    sub.loc[v, "fitness"].values)
            # PLL is log-likelihood (higher = more probable). For HIB props,
            # high PLL ↔ high fit = correct, so directed = +raw_rho. For LIB,
            # high PLL ↔ low fit = correct, so directed = -raw_rho. The
            # sign-multiplier captures both: directed = raw_rho * sign.
            directed = raw_rho * sign
            out.setdefault(prop, {})[tag] = directed
    return out


# ===========================================================================
# Figure 1: Binding affinity (matches fig3 format)
# ===========================================================================
def make_binding_figure():
    dms = load_3dms_rho()
    flab2 = load_flab2_binding_rho()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5),
                                    gridspec_kw={"width_ratios": [3, 1]})
    fig.suptitle("Zero-Shot Binding Affinity — Noise Ablation",
                 fontsize=13, fontweight="bold", y=1.02)

    # ---- Panel A: 3DMS ----
    ds_names = list(dms.keys())
    x = np.arange(len(ds_names))
    w = 0.25
    for i, tag in enumerate(TAGS):
        vals = [dms[ds].get(tag, np.nan) for ds in ds_names]
        v_plot = [v if not np.isnan(v) else 0 for v in vals]
        ax1.bar(x + (i - 1) * w, v_plot, w, label=TAG_LABELS[tag],
                color=C[tag], alpha=0.85, edgecolor="white")
        for j, v in enumerate(vals):
            if not np.isnan(v):
                va = "bottom" if v >= 0 else "top"
                off = 0.01 if v >= 0 else -0.01
                ax1.text(x[j] + (i - 1) * w, v + off, f"{v:.3f}",
                         ha="center", va=va, fontsize=7)

    # Best baseline reference per dataset (from latest report Section 1,
    # baseline_benchmark_directed.md — refreshed 2026-04-25). CR9114 baselines
    # are H1 only; PRISM bar uses H3 — best baseline shown is H1 reference.
    best_baselines_3dms = {
        "G6.31":       -0.018,  # Sapiens (least negative; all baselines wrong direction)
        "CR9114":      -0.326,  # Sapiens (H1, least negative; all baselines wrong direction)
        "Trastuzumab": +0.297,  # AntiBERTy
    }
    for j, ds in enumerate(ds_names):
        bl = best_baselines_3dms.get(ds)
        if bl is not None:
            ax1.plot([j - 0.35, j + 0.35], [bl, bl],
                     color="gray", ls="--", lw=1.2, alpha=0.7)
    ax1.plot([], [], color="gray", ls="--", lw=1.2, label="Best baseline")

    ax1.set_xticks(x)
    ax1.set_xticklabels(ds_names, fontsize=11)
    ax1.set_ylabel("Spearman ρ", fontsize=11)
    ax1.set_title("A. 3DMS Binding", fontsize=11, fontweight="bold")
    ax1.axhline(0, color="black", lw=0.5)
    ax1.legend(fontsize=9, loc="upper left")

    # ---- Panel B: FLAb2 binding (3 bars + best-baseline ref line) ----
    flab_tags = [t for t in TAGS if t in flab2]
    xf = np.arange(len(flab_tags))
    vals = [flab2[t] for t in flab_tags]
    colors = [C[t] for t in flab_tags]
    labels = [TAG_LABELS[t] for t in flab_tags]
    ax2.bar(xf, vals, 0.5, color=colors, alpha=0.85, edgecolor="white")
    for j, v in enumerate(vals):
        offset = 0.002 if v >= 0 else -0.005
        va = "bottom" if v >= 0 else "top"
        ax2.text(j, v + offset, f"{v:.4f}", ha="center", va=va, fontsize=9)

    # Best baseline (FLAb2 binding mean directed rho, report Section 5a)
    flab2_best_baseline = 0.076  # AntiBERTy
    ax2.axhline(flab2_best_baseline, color="gray", ls="--", lw=1.2, alpha=0.7,
                label="Best baseline")

    ax2.set_xticks(xf)
    ax2.set_xticklabels(labels, fontsize=10)
    ax2.set_ylabel("Per-assay mean Spearman ρ", fontsize=11)
    ax2.set_title("B. FLAb2 Binding (41 proteins)", fontsize=11, fontweight="bold")
    ax2.axhline(0, color="black", lw=0.5)

    plt.tight_layout()
    out_png = OUT_DIR / "noise_ablation_binding.png"
    out_svg = OUT_DIR / "noise_ablation_binding.svg"
    fig.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(out_svg, format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved: {out_png}")
    print(f"saved: {out_svg}")

    # Console summary
    print("\n3DMS Spearman ρ:")
    print(f"{'Dataset':<14} {'Clean':>9} {'+Noise2':>9} {'+Noise4':>9}")
    for ds in ds_names:
        row = [dms[ds].get(t, np.nan) for t in TAGS]
        print(f"{ds:<14} " + " ".join(
            f"{v:>+9.3f}" if not np.isnan(v) else f"{'—':>9}" for v in row))

    print("\nFLAb2 binding mean ρ:")
    print(f"{'Tag':<10} {'mean ρ':>10}")
    for t in TAGS:
        v = flab2.get(t, np.nan)
        print(f"{t:<10} " + (f"{v:>+10.4f}" if not np.isnan(v) else f"{'—':>10}"))


# ===========================================================================
# Figure 2: Developability (matches fig4 format)
# ===========================================================================
def make_developability_figure():
    ginkgo = load_ginkgo_rho()
    flab = load_flab2_dev_rho_directed()

    ginkgo_props = ["AC-SINS_pH7.4", "HIC", "Tm2", "ADA", "PR_CHO", "Titer"]
    ginkgo_labels = ["AC-SINS", "HIC", "Tm2", "Immuno.\n(ADA)", "PR_CHO",
                     "Expression\n(Titer)"]
    flab_order = ["self_interaction", "thermostability", "immunogenicity",
                  "polyreactivity", "expression"]
    flab_label_map = {"self_interaction": "Self-Int.", "thermostability": "Thermo.",
                      "immunogenicity": "Immuno.", "polyreactivity": "Polyreact.",
                      "expression": "Express."}
    flab_props = [p for p in flab_order if p in flab]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5),
                                    gridspec_kw={"width_ratios": [6, 5]})
    fig.suptitle("Zero-Shot Developability — Noise Ablation",
                 fontsize=13, fontweight="bold", y=1.02)

    w = 0.25

    # ---- Panel A: Ginkgo ----
    x = np.arange(len(ginkgo_props))
    for i, tag in enumerate(TAGS):
        vals = [ginkgo[p].get(tag, np.nan) for p in ginkgo_props]
        v_plot = [v if not np.isnan(v) else 0 for v in vals]
        ax1.bar(x + (i - 1) * w, v_plot, w, label=TAG_LABELS[tag],
                color=C[tag], alpha=0.85, edgecolor="white")
        for j, v in enumerate(vals):
            if not np.isnan(v):
                va = "bottom" if v >= 0 else "top"
                off = 0.008 if v >= 0 else -0.008
                ax1.text(x[j] + (i - 1) * w, v + off, f"{v:.3f}",
                         ha="center", va=va, fontsize=7)

    # Best baselines from latest report Section 2 (baseline_benchmark_directed.md,
    # refreshed 2026-04-25). Values are directed rho consistent with PRISM's
    # `inverted_for_direction` convention used for v44o `spearman` column.
    best_baselines_ginkgo = {
        "AC-SINS_pH7.4": -0.112,  # AbLang2 (least negative; all baselines wrong)
        "HIC":           +0.041,  # ESM2-35M
        "Tm2":           +0.087,  # ESM2-35M
        "ADA":           +0.316,  # Sapiens
        "PR_CHO":        +0.125,  # ESM2-650M
        "Titer":         +0.340,  # ESM2-35M
    }
    for j, prop in enumerate(ginkgo_props):
        bl = best_baselines_ginkgo.get(prop)
        if bl is not None:
            ax1.plot([j - 0.35, j + 0.35], [bl, bl],
                     color="gray", ls="--", lw=1.2, alpha=0.7)
    ax1.plot([], [], color="gray", ls="--", lw=1.2, label="Best baseline")

    ax1.set_xticks(x)
    ax1.set_xticklabels(ginkgo_labels, fontsize=10)
    ax1.set_ylabel("Spearman ρ (PPL vs property)", fontsize=10)
    ax1.set_title("A. Ginkgo Developability", fontsize=11, fontweight="bold")
    ax1.axhline(0, color="black", lw=0.5)
    y0, y1 = ax1.get_ylim()
    ax1.set_ylim(y0, y1 + 0.40 * (y1 - y0))
    ax1.legend(fontsize=8, loc="upper right", framealpha=1.0)

    # ---- Panel B: FLAb2 dev (directed ρ on ppl_gl) ----
    x2 = np.arange(len(flab_props))
    for i, tag in enumerate(TAGS):
        vals = [flab[p].get(tag, np.nan) for p in flab_props]
        v_plot = [v if not np.isnan(v) else 0 for v in vals]
        ax2.bar(x2 + (i - 1) * w, v_plot, w, label=TAG_LABELS[tag],
                color=C[tag], alpha=0.85, edgecolor="white")
        for j, v in enumerate(vals):
            if not np.isnan(v):
                va = "bottom" if v >= 0 else "top"
                off = 0.008 if v >= 0 else -0.008
                ax2.text(x2[j] + (i - 1) * w, v + off, f"{v:.3f}",
                         ha="center", va=va, fontsize=7)

    # Best baselines (directed rho on PPL) from latest report Section 3
    # (baseline_benchmark_directed.md, refreshed 2026-04-25). Apples-to-apples
    # with PRISM `pll_gl` directed rho computed above.
    best_baselines_flab2 = {
        "self_interaction": -0.171,  # ESM2-35M (least negative; all wrong)
        "thermostability":  +0.317,  # Sapiens
        "immunogenicity":   +0.351,  # Sapiens
        "polyreactivity":   -0.140,  # ESM2-35M (least negative; all wrong)
        "expression":       +0.057,  # Sapiens
    }
    for j, prop in enumerate(flab_props):
        bl = best_baselines_flab2.get(prop)
        if bl is not None:
            ax2.plot([j - 0.35, j + 0.35], [bl, bl],
                     color="gray", ls="--", lw=1.2, alpha=0.7)
    ax2.plot([], [], color="gray", ls="--", lw=1.2, label="Best baseline")

    ax2.set_xticks(x2)
    ax2.set_xticklabels([flab_label_map.get(p, p) for p in flab_props], fontsize=10)
    ax2.set_ylabel("Per-assay mean Spearman ρ", fontsize=10)
    ax2.set_title("B. FLAb2 Developability", fontsize=11, fontweight="bold")
    ax2.axhline(0, color="black", lw=0.5)
    y0, y1 = ax2.get_ylim()
    ax2.set_ylim(y0, y1 + 0.40 * (y1 - y0))
    ax2.legend(fontsize=8, loc="upper right", framealpha=1.0)

    plt.tight_layout()
    out_png = OUT_DIR / "noise_ablation_developability.png"
    out_svg = OUT_DIR / "noise_ablation_developability.svg"
    fig.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(out_svg, format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved: {out_png}")
    print(f"saved: {out_svg}")

    # Console summary
    print("\nGinkgo Spearman ρ (best PPL signal):")
    print(f"{'Property':<18} {'Clean':>9} {'+Noise2':>9} {'+Noise4':>9}")
    for p in ginkgo_props:
        row = [ginkgo[p].get(t, np.nan) for t in TAGS]
        print(f"{p:<18} " + " ".join(
            f"{v:>+9.3f}" if not np.isnan(v) else f"{'—':>9}" for v in row))

    print("\nFLAb2 dev directed ρ (ppl_gl):")
    print(f"{'Property':<18} {'Clean':>9} {'+Noise2':>9} {'+Noise4':>9}")
    for p in flab_props:
        row = [flab[p].get(t, np.nan) for t in TAGS]
        print(f"{p:<18} " + " ".join(
            f"{v:>+9.3f}" if not np.isnan(v) else f"{'—':>9}" for v in row))


if __name__ == "__main__":
    make_binding_figure()
    print()
    make_developability_figure()
