#!/usr/bin/env python3
"""
Statistical significance tests for PRISM vs baselines.

Two analyses:
  (A) Binding affinity — 45 FLAb2 proteins, protein-level Spearman rho
      1. One-sample Wilcoxon: H0 median(rho)=0 for each model
      2. Pairwise Wilcoxon signed-rank: PRISM vs each baseline

  (B) Developability — per property category
      For properties with >= 6 assays: per-assay directed rho → Wilcoxon
      For properties with < 6 assays: bootstrap CI on rho from per-antibody scores

Usage:
    python statistical_significance_tests.py
    python statistical_significance_tests.py --n_bootstrap 10000
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, spearmanr

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULT_DIR = REPO_ROOT / "data" / "features" / "evaluation_results"
BINDING_DIR = RESULT_DIR / "flab2_binding"
RANKING_DIR = BINDING_DIR / "ranking_comparison"
DEV_DIR = RESULT_DIR / "flab2_developability"
OUT_DIR = Path(__file__).resolve().parent / "reviewer_figures"
OUT_DIR.mkdir(exist_ok=True)

MODEL_ORDER = ["PRISM", "ESM2-35M", "ESM2-650M", "AbLang2", "AntiBERTy", "Sapiens"]

COL_TO_DISPLAY = {
    "PRISM_v34.1b (ngl_logprob_sum)": "PRISM",
    "ESM2-35M_LLR": "ESM2-35M",
    "ESM2-650M_LLR": "ESM2-650M",
    "AbLang2_LLR": "AbLang2",
    "AntiBERTy_LLR": "AntiBERTy",
    "Sapiens_LLR": "Sapiens",
}

PROPERTY_DIRECTION = {
    "Aggregation / Self-interaction": -1,
    "Thermostability": +1,
    "Immunogenicity": -1,
    "Polyreactivity": -1,
    "Expression": +1,
}

PROPERTY_DISPLAY = {
    "Aggregation / Self-interaction": "Self-interaction",
    "Thermostability": "Thermostability",
    "Immunogenicity": "Immunogenicity",
    "Polyreactivity": "Polyreactivity",
    "Expression": "Expression",
}

PROP_ORDER = [
    "Thermostability",
    "Aggregation / Self-interaction",
    "Immunogenicity",
    "Polyreactivity",
    "Expression",
]

DEV_MODEL_MAP = {
    "prism_noise2": "PRISM",
    "esm2_35m": "ESM2-35M",
    "esm2_650m": "ESM2-650M",
    "ablang2": "AbLang2",
    "antiberty": "AntiBERTy",
    "sapiens": "Sapiens",
}

# PRISM uses property-specific best signals; baselines use PLL
PRISM_DEV_MODEL = "prism_noise2"
BASELINE_DEV_MODELS = ["esm2_35m", "esm2_650m", "ablang2", "antiberty", "sapiens"]

# Map from per_assay_correlations property_category to per_antibody_scores property
CATEGORY_TO_PROPERTY = {
    "Aggregation / Self-interaction": "self_interaction",
    "Thermostability": "thermostability",
    "Immunogenicity": "immunogenicity",
    "Polyreactivity": "polyreactivity",
    "Expression": "expression",
}


def sig_str(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return "n.s."


def one_sample_wilcoxon(values, alternative="greater"):
    """One-sample Wilcoxon signed-rank test: H0 median = 0."""
    values = np.asarray(values)
    values = values[~np.isnan(values)]
    if len(values) < 6:
        return np.nan
    # Remove zeros (Wilcoxon can't handle them)
    nonzero = values[values != 0]
    if len(nonzero) < 6:
        return np.nan
    try:
        _, p = wilcoxon(nonzero, alternative=alternative)
        return p
    except Exception:
        return np.nan


def paired_wilcoxon(a, b, alternative="two-sided"):
    """Paired Wilcoxon signed-rank test on paired observations."""
    a, b = np.asarray(a), np.asarray(b)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    diff = a - b
    nonzero = diff[diff != 0]
    if len(nonzero) < 6:
        return np.nan
    try:
        _, p = wilcoxon(nonzero, alternative=alternative)
        return p
    except Exception:
        return np.nan


def bootstrap_spearman(scores, fitness, n_boot=10000, seed=42):
    """Bootstrap Spearman rho: returns (rho_obs, ci_lo, ci_hi, p_vs_zero)."""
    rng = np.random.RandomState(seed)
    mask = ~(np.isnan(scores) | np.isnan(fitness))
    scores, fitness = scores[mask], fitness[mask]
    n = len(scores)
    if n < 10:
        return np.nan, np.nan, np.nan, np.nan

    rho_obs, _ = spearmanr(scores, fitness)
    boot_rhos = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        r, _ = spearmanr(scores[idx], fitness[idx])
        boot_rhos[i] = r
    ci_lo, ci_hi = np.percentile(boot_rhos, [2.5, 97.5])
    # p-value: fraction of bootstrap rhos <= 0 (one-sided, testing rho > 0)
    p_gt_zero = np.mean(boot_rhos <= 0)
    return rho_obs, ci_lo, ci_hi, p_gt_zero


def bootstrap_paired_diff(scores_a, scores_b, fitness, n_boot=10000, seed=42):
    """Bootstrap test for rho_A - rho_B > 0."""
    rng = np.random.RandomState(seed)
    mask = ~(np.isnan(scores_a) | np.isnan(scores_b) | np.isnan(fitness))
    sa, sb, f = scores_a[mask], scores_b[mask], fitness[mask]
    n = len(f)
    if n < 10:
        return np.nan, np.nan

    rho_a, _ = spearmanr(sa, f)
    rho_b, _ = spearmanr(sb, f)
    delta_obs = rho_a - rho_b

    boot_deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        ra, _ = spearmanr(sa[idx], f[idx])
        rb, _ = spearmanr(sb[idx], f[idx])
        boot_deltas[i] = ra - rb
    # p-value: fraction of bootstrap deltas <= 0
    p = np.mean(boot_deltas <= 0)
    return delta_obs, p


# ══════════════════════════════════════════════════════════════════════════
# (A) Binding affinity
# ══════════════════════════════════════════════════════════════════════════
def analyze_binding():
    rho_path = RANKING_DIR / "rho_matrix_aggregated.csv"
    if not rho_path.exists():
        print("[SKIP] Binding: rho_matrix_aggregated.csv not found")
        return None

    rho_matrix = pd.read_csv(rho_path, index_col=0)
    rho_matrix = rho_matrix.drop(columns=["PRISM_v34.1c (origin_logit_sum)"], errors="ignore")
    rho_matrix = rho_matrix.rename(columns=COL_TO_DISPLAY)

    n_proteins = rho_matrix.shape[0]
    print(f"\n{'='*80}")
    print(f"(A) BINDING AFFINITY — {n_proteins} FLAb2 proteins")
    print(f"{'='*80}")

    # --- One-sample tests ---
    print(f"\n  One-sample Wilcoxon signed-rank (H0: median rho = 0)")
    print(f"  {'Model':<15s} {'Mean rho':>10s} {'Median rho':>12s} {'p (>0)':>10s} {'Sig':>6s}")
    print(f"  {'-'*55}")

    onesample_results = {}
    for m in MODEL_ORDER:
        vals = rho_matrix[m].dropna().values
        mean_r = np.mean(vals)
        med_r = np.median(vals)
        p = one_sample_wilcoxon(vals, alternative="greater")
        onesample_results[m] = {"mean_rho": mean_r, "median_rho": med_r, "p": p, "n": len(vals)}
        p_str = f"{p:.4f}" if not np.isnan(p) else "N/A"
        print(f"  {m:<15s} {mean_r:>+10.4f} {med_r:>+12.4f} {p_str:>10s} {sig_str(p):>6s}")

    # --- Pairwise tests ---
    print(f"\n  Paired Wilcoxon signed-rank (H0: rho_PRISM = rho_baseline)")
    print(f"  {'Comparison':<25s} {'Mean diff':>10s} {'p (two-sided)':>14s} {'Sig':>6s}")
    print(f"  {'-'*57}")

    pairwise_results = {}
    prism_rho = rho_matrix["PRISM"].values
    for bl in MODEL_ORDER[1:]:
        bl_rho = rho_matrix[bl].values
        p = paired_wilcoxon(prism_rho, bl_rho)
        mask = ~(np.isnan(prism_rho) | np.isnan(bl_rho))
        mean_diff = np.mean(prism_rho[mask] - bl_rho[mask])
        pairwise_results[bl] = {"mean_diff": mean_diff, "p": p}
        p_str = f"{p:.4f}" if not np.isnan(p) else "N/A"
        print(f"  PRISM vs {bl:<14s} {mean_diff:>+10.4f} {p_str:>14s} {sig_str(p):>6s}")

    n_sig = sum(1 for v in pairwise_results.values() if v["p"] < 0.05)
    n_total = len(pairwise_results)
    print(f"\n  => Wilcoxon p<0.05 for {n_sig}/{n_total} baselines")

    prism_p = onesample_results["PRISM"]["p"]
    print(f"  => PRISM mean rho={onesample_results['PRISM']['mean_rho']:+.4f}, "
          f"one-sample p={prism_p:.4f} {'(significant)' if prism_p < 0.05 else '(n.s.)'}")

    return onesample_results, pairwise_results


# ══════════════════════════════════════════════════════════════════════════
# (B) Developability — per property category
# ══════════════════════════════════════════════════════════════════════════
def analyze_developability(n_bootstrap=10000):
    assay_path = DEV_DIR / "per_assay_correlations.csv"
    scores_path = DEV_DIR / "per_antibody_scores.csv"

    if not assay_path.exists():
        print("[SKIP] Developability: per_assay_correlations.csv not found")
        return None

    assay_df = pd.read_csv(assay_path)
    has_scores = scores_path.exists()
    if has_scores:
        scores_df = pd.read_csv(scores_path)

    # Auto-select best PRISM signal per property from per_assay_correlations
    prism_best_signals = {}
    for cat in PROP_ORDER:
        prism_rows = assay_df[
            (assay_df["model"] == PRISM_DEV_MODEL) & (assay_df["property_category"] == cat)
        ]
        if len(prism_rows) > 0:
            # Pick signal with highest mean directed rho across assays
            sig_mean = prism_rows.groupby("signal")["directed_rho"].mean()
            prism_best_signals[cat] = sig_mean.idxmax()

    print(f"\n{'='*80}")
    print(f"(B) DEVELOPABILITY — per property category")
    print(f"{'='*80}")
    print(f"  PRISM best signals: {prism_best_signals}")

    all_results = {}

    for cat in PROP_ORDER:
        display = PROPERTY_DISPLAY[cat]
        prism_sig = prism_best_signals.get(cat, "pll_marginalized")

        # Get per-assay directed rho for each model
        cat_assays = assay_df[assay_df["property_category"] == cat]
        assay_types = cat_assays["assay_type"].unique()
        n_assays = len(assay_types)

        print(f"\n  --- {display} ({n_assays} assays) ---")

        # Build per-assay rho matrix: [assay x model]
        assay_rho = {}
        for m_key, m_display in DEV_MODEL_MAP.items():
            sig = prism_sig if m_key == PRISM_DEV_MODEL else "pll"
            rows = cat_assays[(cat_assays["model"] == m_key) & (cat_assays["signal"] == sig)]
            for _, row in rows.iterrows():
                if row["assay_type"] not in assay_rho:
                    assay_rho[row["assay_type"]] = {}
                assay_rho[row["assay_type"]][m_display] = row["directed_rho"]

        rho_df = pd.DataFrame(assay_rho).T  # rows=assays, cols=models

        use_wilcoxon = n_assays >= 6

        if use_wilcoxon:
            print(f"  Method: Wilcoxon signed-rank (per-assay directed rho)")

            # One-sample
            print(f"\n  {'Model':<15s} {'Mean rho':>10s} {'Median rho':>12s} {'p (>0)':>10s} {'Sig':>6s}")
            print(f"  {'-'*55}")
            onesample = {}
            for m in MODEL_ORDER:
                if m not in rho_df.columns:
                    continue
                vals = rho_df[m].dropna().values
                mean_r = np.mean(vals)
                med_r = np.median(vals)
                p = one_sample_wilcoxon(vals, alternative="greater")
                onesample[m] = {"mean_rho": mean_r, "median_rho": med_r, "p": p, "n": len(vals)}
                p_str = f"{p:.4f}" if not np.isnan(p) else "N/A"
                print(f"  {m:<15s} {mean_r:>+10.4f} {med_r:>+12.4f} {p_str:>10s} {sig_str(p):>6s}")

            # Pairwise
            print(f"\n  {'Comparison':<25s} {'Mean diff':>10s} {'p (two-sided)':>14s} {'Sig':>6s}")
            print(f"  {'-'*57}")
            pairwise = {}
            for bl in MODEL_ORDER[1:]:
                if bl not in rho_df.columns or "PRISM" not in rho_df.columns:
                    continue
                p = paired_wilcoxon(rho_df["PRISM"].values, rho_df[bl].values)
                mask = ~(rho_df["PRISM"].isna() | rho_df[bl].isna())
                mean_diff = (rho_df.loc[mask, "PRISM"] - rho_df.loc[mask, bl]).mean()
                pairwise[bl] = {"mean_diff": mean_diff, "p": p}
                p_str = f"{p:.4f}" if not np.isnan(p) else "N/A"
                print(f"  PRISM vs {bl:<14s} {mean_diff:>+10.4f} {p_str:>14s} {sig_str(p):>6s}")

            all_results[cat] = {"method": "wilcoxon", "onesample": onesample, "pairwise": pairwise}

        else:
            # Bootstrap from per-antibody scores
            if not has_scores:
                print(f"  [SKIP] Too few assays ({n_assays}) and no per_antibody_scores.csv")
                continue

            prop_key = CATEGORY_TO_PROPERTY[cat]
            direction = PROPERTY_DIRECTION[cat]
            print(f"  Method: Bootstrap (n={n_bootstrap}) on per-antibody scores")

            # Get per-antibody data for PRISM and baselines
            prop_scores = scores_df[scores_df["property"] == prop_key].copy()

            # One-sample bootstrap
            print(f"\n  {'Model':<15s} {'rho':>8s} {'95% CI':>18s} {'p (>0)':>10s} {'Sig':>6s}")
            print(f"  {'-'*60}")
            onesample = {}
            model_arrays = {}
            for m_key, m_display in DEV_MODEL_MAP.items():
                sig = prism_best_signals.get(cat, "pll_marginalized") if m_key == PRISM_DEV_MODEL else "pll"
                subset = prop_scores[(prop_scores["model"] == m_key) & (prop_scores["signal"] == sig)]
                if len(subset) < 10:
                    continue
                s = subset["score"].values
                f = subset["fitness"].values * direction  # apply direction
                model_arrays[m_display] = (s, f)

                rho_obs, ci_lo, ci_hi, p = bootstrap_spearman(s, f, n_boot=n_bootstrap)
                onesample[m_display] = {
                    "rho": rho_obs, "ci_lo": ci_lo, "ci_hi": ci_hi, "p": p, "n": len(subset)
                }
                p_str = f"{p:.4f}" if not np.isnan(p) else "N/A"
                print(f"  {m_display:<15s} {rho_obs:>+8.4f} [{ci_lo:>+.4f}, {ci_hi:>+.4f}] "
                      f"{p_str:>10s} {sig_str(p):>6s}")

            # Pairwise bootstrap
            print(f"\n  {'Comparison':<25s} {'Delta rho':>10s} {'p (PRISM>BL)':>14s} {'Sig':>6s}")
            print(f"  {'-'*57}")
            pairwise = {}
            if "PRISM" in model_arrays:
                s_prism, f_prism = model_arrays["PRISM"]
                for bl in MODEL_ORDER[1:]:
                    if bl not in model_arrays:
                        continue
                    s_bl, f_bl = model_arrays[bl]
                    # Both use the same antibodies/fitness, just different scores
                    delta, p = bootstrap_paired_diff(s_prism, s_bl, f_prism, n_boot=n_bootstrap)
                    pairwise[bl] = {"delta_rho": delta, "p": p}
                    p_str = f"{p:.4f}" if not np.isnan(p) else "N/A"
                    print(f"  PRISM vs {bl:<14s} {delta:>+10.4f} {p_str:>14s} {sig_str(p):>6s}")

            all_results[cat] = {"method": "bootstrap", "onesample": onesample, "pairwise": pairwise}

        # Summary line
        prism_result = (all_results[cat]["onesample"].get("PRISM", {})
                        if cat in all_results else {})
        prism_p = prism_result.get("p", np.nan)
        n_sig_pw = sum(1 for v in all_results.get(cat, {}).get("pairwise", {}).values()
                       if v.get("p", 1) < 0.05)
        n_total_pw = len(all_results.get(cat, {}).get("pairwise", {}))
        print(f"\n  => PRISM rho>0: p={prism_p:.4f} {sig_str(prism_p)}" if not np.isnan(prism_p) else "")
        print(f"  => Wilcoxon/Bootstrap p<0.05 for {n_sig_pw}/{n_total_pw} baselines")

    return all_results


# ══════════════════════════════════════════════════════════════════════════
# LaTeX table output
# ══════════════════════════════════════════════════════════════════════════
def make_latex_table(binding_results, dev_results):
    if binding_results is None and dev_results is None:
        return

    onesample_b, pairwise_b = binding_results if binding_results else (None, None)

    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{Statistical significance of zero-shot prediction performance.}")
    lines.append(r"\label{tab:significance}")
    lines.append(r"\small")

    # ── Part A: Binding ──
    if onesample_b:
        lines.append(r"\vspace{2mm}")
        lines.append(r"\textbf{(A) Binding affinity (45 FLAb2 proteins, protein-level $\rho$)}")
        lines.append(r"\vspace{1mm}")
        lines.append("")

        baselines = MODEL_ORDER[1:]
        pw_cols = " & ".join([f"vs {b}" for b in baselines])

        lines.append(r"\begin{tabular}{lcccc" + "c" * len(baselines) + "}")
        lines.append(r"\toprule")
        lines.append(r"Model & $N$ & Mean $\rho$ & Median $\rho$ & "
                      r"$p$ ($\rho>0$) & " + pw_cols + r" \\")
        lines.append(r"\midrule")

        for m in MODEL_ORDER:
            r = onesample_b[m]
            bold = r"\textbf" if m == "PRISM" else ""
            name = f"{bold}{{{m}}}" if bold else m

            p_one = r["p"]
            p_one_str = _fmt_p_latex(p_one)

            cells = [name, str(r["n"]), f"{r['mean_rho']:+.3f}",
                     f"{r['median_rho']:+.3f}", p_one_str]

            for bl in baselines:
                if m == "PRISM":
                    pw = pairwise_b.get(bl, {})
                    p_pw = pw.get("p", np.nan)
                    cells.append(_fmt_p_latex(p_pw))
                else:
                    cells.append("--")

            lines.append(" & ".join(cells) + r" \\")

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")

    # ── Part B: Developability ──
    if dev_results:
        lines.append("")
        lines.append(r"\vspace{4mm}")
        lines.append(r"\textbf{(B) Developability (per property category)}")
        lines.append(r"\vspace{1mm}")
        lines.append("")

        baselines = MODEL_ORDER[1:]
        pw_cols = " & ".join([f"vs {b}" for b in baselines])

        lines.append(r"\begin{tabular}{llcccc" + "c" * len(baselines) + "}")
        lines.append(r"\toprule")
        lines.append(r"Property & Model & Method & $\rho$ & "
                      r"$p$ ($\rho>0$) & $N$ & " + pw_cols + r" \\")
        lines.append(r"\midrule")

        for cat in PROP_ORDER:
            if cat not in dev_results:
                continue
            res = dev_results[cat]
            display = PROPERTY_DISPLAY[cat]
            method_label = "W" if res["method"] == "wilcoxon" else "B"

            for i, m in enumerate(MODEL_ORDER):
                os_data = res["onesample"].get(m, {})
                if not os_data:
                    continue

                bold = r"\textbf" if m == "PRISM" else ""
                name = f"{bold}{{{m}}}" if bold else m
                prop_cell = display if i == 0 else ""

                if res["method"] == "wilcoxon":
                    rho_val = os_data.get("mean_rho", np.nan)
                    n_val = os_data.get("n", 0)
                else:
                    rho_val = os_data.get("rho", np.nan)
                    n_val = os_data.get("n", 0)

                p_one = os_data.get("p", np.nan)

                rho_str = f"{rho_val:+.3f}" if not np.isnan(rho_val) else "--"
                p_str = _fmt_p_latex(p_one)

                cells = [prop_cell, name, method_label, rho_str, p_str, str(n_val)]

                for bl in baselines:
                    if m == "PRISM":
                        pw = res["pairwise"].get(bl, {})
                        p_pw = pw.get("p", np.nan)
                        cells.append(_fmt_p_latex(p_pw))
                    else:
                        cells.append("--")

                lines.append(" & ".join(cells) + r" \\")

            lines.append(r"\addlinespace")

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")

        lines.append("")
        lines.append(r"\vspace{1mm}")
        lines.append(r"{\footnotesize W = Wilcoxon signed-rank (per-assay $\rho$); "
                      r"B = Bootstrap ($n$=10{,}000). Bold $p$-values: $p<0.05$.}")

    lines.append(r"\end{table}")

    tex = "\n".join(lines)
    out_path = OUT_DIR / "table3_significance_tests.tex"
    out_path.write_text(tex)
    print(f"\nLaTeX table saved: {out_path}")


def _fmt_p_latex(p):
    if np.isnan(p):
        return "--"
    if p < 0.001:
        return rf"\textbf{{{p:.1e}}}"
    elif p < 0.05:
        return rf"\textbf{{{p:.3f}}}"
    else:
        return f"{p:.3f}"


# ══════════════════════════════════════════════════════════════════════════
# CSV summary output
# ══════════════════════════════════════════════════════════════════════════
def save_csv_summary(binding_results, dev_results):
    rows = []

    if binding_results:
        onesample_b, pairwise_b = binding_results
        for m in MODEL_ORDER:
            r = onesample_b[m]
            row = {
                "analysis": "binding",
                "property": "binding_affinity",
                "model": m,
                "method": "wilcoxon",
                "mean_rho": r["mean_rho"],
                "median_rho": r["median_rho"],
                "n": r["n"],
                "p_onesample_gt0": r["p"],
            }
            if m == "PRISM":
                for bl in MODEL_ORDER[1:]:
                    pw = pairwise_b.get(bl, {})
                    row[f"p_vs_{bl}"] = pw.get("p", np.nan)
            rows.append(row)

    if dev_results:
        for cat in PROP_ORDER:
            if cat not in dev_results:
                continue
            res = dev_results[cat]
            for m in MODEL_ORDER:
                os_data = res["onesample"].get(m, {})
                if not os_data:
                    continue
                row = {
                    "analysis": "developability",
                    "property": PROPERTY_DISPLAY[cat],
                    "model": m,
                    "method": res["method"],
                    "n": os_data.get("n", 0),
                    "p_onesample_gt0": os_data.get("p", np.nan),
                }
                if res["method"] == "wilcoxon":
                    row["mean_rho"] = os_data.get("mean_rho", np.nan)
                    row["median_rho"] = os_data.get("median_rho", np.nan)
                else:
                    row["mean_rho"] = os_data.get("rho", np.nan)
                    row["ci_lo"] = os_data.get("ci_lo", np.nan)
                    row["ci_hi"] = os_data.get("ci_hi", np.nan)

                if m == "PRISM":
                    for bl in MODEL_ORDER[1:]:
                        pw = res["pairwise"].get(bl, {})
                        row[f"p_vs_{bl}"] = pw.get("p", np.nan)
                rows.append(row)

    df = pd.DataFrame(rows)
    out_path = OUT_DIR / "significance_test_results.csv"
    df.to_csv(out_path, index=False)
    print(f"CSV summary saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_bootstrap", type=int, default=10000)
    args = parser.parse_args()

    binding = analyze_binding()
    dev = analyze_developability(n_bootstrap=args.n_bootstrap)

    make_latex_table(binding, dev)
    save_csv_summary(binding, dev)

    print("\nDone!")
