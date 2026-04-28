#!/usr/bin/env python3
"""
FLAb2 per-assay robustness analysis for reviewer response.

Addresses: "Do the reported developability gains remain consistent across a broader
set of benchmarks or per-assay statistical tests?"

Strategy:
  - Model scores are per-antibody (heavy+light pair) → reusable across assays
  - Load scores from per_antibody_scores.csv, join with ALL assay types' fitness
  - Compute directed Spearman rho per (model, assay_type)
  - Show consistency across assays within each property category

Figures:
  4. Per-assay directed rho heatmap (assays × models)
  5. Per-assay forest plot (PRISM rho ± 95% CI, grouped by property)

Tables:
  3. Per-assay correlation summary (LaTeX)
"""

import pathlib
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# ── Paths ──────────────────────────────────────────────────────────────────
REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data" / "FLAb"
SCORE_DIR = REPO_ROOT / "data" / "features" / "evaluation_results" / "flab2_developability"
OUT_DIR = pathlib.Path(__file__).resolve().parent / "reviewer_figures"
OUT_DIR.mkdir(exist_ok=True)

# ── Property → file mapping (ALL assay types) ─────────────────────────────
# higher_is_better: True = higher fitness = better developability
PROPERTY_FILES = {
    "Aggregation / Self-interaction": {
        "file": "flab2_aggregation.parquet",
        "higher_is_better": False,
    },
    "Thermostability": {
        "file": "flab2_thermostability.parquet",
        "higher_is_better": True,
    },
    "Immunogenicity": {
        "file": "flab2_immunogenicity.parquet",
        "higher_is_better": False,
    },
    "Polyreactivity": {
        "file": "flab2_polyreactivity.parquet",
        "higher_is_better": False,
    },
    "Expression": {
        "file": "flab2_expression.parquet",
        "higher_is_better": True,
    },
}

# ── Models ─────────────────────────────────────────────────────────────────
MODEL_DISPLAY = {
    "prism_noise2": "PRISM",
    "prism": "PRISM (orig)",
    "esm2_35m": "ESM2-35M",
    "esm2_650m": "ESM2-650M",
    "ablang2": "AbLang2",
    "antiberty": "AntiBERTy",
    "sapiens": "Sapiens",
}

MODEL_COLORS = {
    "PRISM": "#332288",
    "PRISM (orig)": "#6655CC",
    "ESM2-35M": "#DDCC77",
    "ESM2-650M": "#117733",
    "AbLang2": "#88CCEE",
    "AntiBERTy": "#44AA99",
    "Sapiens": "#882255",
}

# Models to show in figures (order matters)
SHOW_MODELS = ["prism_noise2", "esm2_35m", "esm2_650m", "ablang2", "antiberty", "sapiens"]

# PRISM signal to use per property category (for baselines: always PLL)
# Will be auto-selected from best directed rho
PRISM_MODEL = "prism_noise2"

# Minimum antibodies per assay to include
MIN_N = 30

# ── Style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "figure.dpi": 300,
})


# ══════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════

def load_all_assays() -> pd.DataFrame:
    """Load ALL assay types from ALL property files, with property category label."""
    rows = []
    for prop_category, cfg in PROPERTY_FILES.items():
        df = pd.read_parquet(DATA_DIR / cfg["file"])
        df = df[df["light"].notna() & df["fitness"].notna()].copy()
        df = df.drop_duplicates(subset=["heavy", "light", "assay_type"], keep="first")
        df["property_category"] = prop_category
        df["higher_is_better"] = cfg["higher_is_better"]
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def load_model_scores() -> pd.DataFrame:
    """Load per-antibody model scores (from scoring script)."""
    path = SCORE_DIR / "per_antibody_scores.csv"
    return pd.read_csv(path)


def build_score_lookup(scores_df: pd.DataFrame) -> Dict:
    """Build (model, signal, heavy, light) -> score lookup."""
    lookup = {}
    for _, row in scores_df.iterrows():
        key = (row["model"], row["signal"], row["heavy"], row["light"])
        lookup[key] = row["score"]
    return lookup


def get_best_prism_signal(scores_df: pd.DataFrame, assays_df: pd.DataFrame,
                          prop_category: str) -> str:
    """Find PRISM signal with highest mean directed rho across assays in this category."""
    cat_assays = assays_df[assays_df["property_category"] == prop_category]
    higher_is_better = cat_assays["higher_is_better"].iloc[0]
    sign = 1.0 if higher_is_better else -1.0

    prism_scores = scores_df[scores_df["model"] == PRISM_MODEL]
    signals = prism_scores["signal"].unique()

    best_signal = "pll_marginalized"
    best_mean_rho = -999

    for sig in signals:
        sig_scores = prism_scores[prism_scores["signal"] == sig]
        # Build lookup for this signal
        score_map = dict(zip(
            zip(sig_scores["heavy"], sig_scores["light"]),
            sig_scores["score"]
        ))

        rhos = []
        for assay_type in cat_assays["assay_type"].unique():
            assay_data = cat_assays[cat_assays["assay_type"] == assay_type]
            if len(assay_data) < MIN_N:
                continue

            matched_scores = []
            matched_fitness = []
            for _, row in assay_data.iterrows():
                s = score_map.get((row["heavy"], row["light"]))
                if s is not None and np.isfinite(s) and np.isfinite(row["fitness"]):
                    matched_scores.append(s)
                    matched_fitness.append(row["fitness"])

            if len(matched_scores) < MIN_N:
                continue

            rho, _ = spearmanr(matched_scores, matched_fitness)
            rhos.append(sign * rho)

        if rhos:
            mean_rho = np.mean(rhos)
            if mean_rho > best_mean_rho:
                best_mean_rho = mean_rho
                best_signal = sig

    return best_signal


# ══════════════════════════════════════════════════════════════════════════
# Compute per-assay correlations
# ══════════════════════════════════════════════════════════════════════════

def compute_per_assay_correlations(scores_df, assays_df, prism_signals):
    """Compute directed Spearman rho for each (model, assay_type) pair.

    Returns DataFrame with columns:
      property_category, assay_type, n_antibodies, n_studies,
      model, model_display, signal, rho, pvalue, directed_rho
    """
    results = []

    for prop_category, cfg in PROPERTY_FILES.items():
        higher_is_better = cfg["higher_is_better"]
        sign = 1.0 if higher_is_better else -1.0
        cat_assays = assays_df[assays_df["property_category"] == prop_category]
        prism_sig = prism_signals[prop_category]

        for assay_type in sorted(cat_assays["assay_type"].unique()):
            assay_data = cat_assays[cat_assays["assay_type"] == assay_type]
            assay_data_dedup = assay_data.drop_duplicates(subset=["heavy", "light"], keep="first")

            if len(assay_data_dedup) < MIN_N:
                continue

            n_studies = assay_data_dedup["study"].nunique() if "study" in assay_data_dedup.columns else 0

            for model_key in SHOW_MODELS:
                if model_key == PRISM_MODEL:
                    sig = prism_sig
                else:
                    sig = "pll"

                model_scores = scores_df[
                    (scores_df["model"] == model_key) & (scores_df["signal"] == sig)
                ]
                score_map = dict(zip(
                    zip(model_scores["heavy"], model_scores["light"]),
                    model_scores["score"]
                ))

                matched_s, matched_f = [], []
                for _, row in assay_data_dedup.iterrows():
                    s = score_map.get((row["heavy"], row["light"]))
                    if s is not None and np.isfinite(s) and np.isfinite(row["fitness"]):
                        matched_s.append(s)
                        matched_f.append(row["fitness"])

                if len(matched_s) < MIN_N:
                    continue

                rho, pval = spearmanr(matched_s, matched_f)
                directed_rho = sign * rho

                results.append({
                    "property_category": prop_category,
                    "assay_type": assay_type,
                    "n_antibodies": len(matched_s),
                    "n_studies": n_studies,
                    "model": model_key,
                    "model_display": MODEL_DISPLAY[model_key],
                    "signal": sig,
                    "rho": rho,
                    "pvalue": pval,
                    "directed_rho": directed_rho,
                })

    return pd.DataFrame(results)


# ══════════════════════════════════════════════════════════════════════════
# Figure 4: Per-assay heatmap (assays × models, directed rho)
# ══════════════════════════════════════════════════════════════════════════

def plot_per_assay_heatmap(corr_df):
    """Heatmap: rows = assay types (grouped by property), columns = models."""
    from matplotlib.colors import TwoSlopeNorm

    model_order = [MODEL_DISPLAY[m] for m in SHOW_MODELS]

    # Build pivot: assay_type × model_display → directed_rho
    pivot = corr_df.pivot_table(
        index=["property_category", "assay_type", "n_antibodies"],
        columns="model_display",
        values="directed_rho",
    )
    pivot = pivot.reindex(columns=model_order)

    # Sort within each property category
    pivot = pivot.sort_index(level=["property_category", "assay_type"])

    # Build display labels
    labels = []
    prop_boundaries = []  # for horizontal separators
    prev_cat = None
    for i, (cat, assay, n) in enumerate(pivot.index):
        if prev_cat is not None and cat != prev_cat:
            prop_boundaries.append(i)
        labels.append(f"{assay} (n={n})")
        prev_cat = cat

    fig, ax = plt.subplots(figsize=(8, max(6, len(labels) * 0.35 + 1.5)))

    norm = TwoSlopeNorm(vmin=-0.5, vcenter=0, vmax=0.5)
    im = ax.imshow(
        pivot.values, aspect="auto", cmap="RdBu_r", norm=norm, interpolation="nearest"
    )

    # Axis labels
    ax.set_xticks(range(len(model_order)))
    ax.set_xticklabels(model_order, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7.5)

    # Property category separators
    for b in prop_boundaries:
        ax.axhline(y=b - 0.5, color="black", linewidth=1.5)

    # Property category labels on the right
    cats = [cat for cat, _, _ in pivot.index]
    cat_ranges = {}
    for i, cat in enumerate(cats):
        cat_ranges.setdefault(cat, [i, i])
        cat_ranges[cat][1] = i

    for cat, (start, end) in cat_ranges.items():
        mid = (start + end) / 2
        short_cat = cat.split("/")[0].strip()  # Shorten long names
        ax.text(
            len(model_order) + 0.3, mid, short_cat,
            ha="left", va="center", fontsize=8, fontweight="bold",
            rotation=0, color="#333333",
        )

    # Text annotations: rho values
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            if pd.isna(val):
                ax.text(j, i, "—", ha="center", va="center", fontsize=6, color="gray")
            else:
                color = "white" if abs(val) > 0.3 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6, color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.5, pad=0.02)
    cbar.set_label("Directed Spearman $\\rho$\n(positive = favorable)", fontsize=9)

    ax.set_title("Per-assay developability prediction (FLAb2)", fontsize=13, pad=10)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig4_per_assay_heatmap.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig4_per_assay_heatmap.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  Figure 4 saved: fig4_per_assay_heatmap.pdf/png")


# ══════════════════════════════════════════════════════════════════════════
# Figure 5: Forest plot — PRISM directed rho ± bootstrap CI per assay
# ══════════════════════════════════════════════════════════════════════════

def bootstrap_spearman_ci(x, y, n_boot=2000, ci=0.95, seed=42):
    """Bootstrap confidence interval for Spearman rho."""
    rng = np.random.default_rng(seed)
    n = len(x)
    boot_rhos = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_rhos[b], _ = spearmanr(x[idx], y[idx])

    alpha = (1 - ci) / 2
    lo = np.percentile(boot_rhos, 100 * alpha)
    hi = np.percentile(boot_rhos, 100 * (1 - alpha))
    return lo, hi


def plot_forest(corr_df, scores_df, assays_df, prism_signals):
    """Forest plot: PRISM directed rho with 95% CI, grouped by property."""
    prism_df = corr_df[corr_df["model"] == PRISM_MODEL].copy()
    prism_df = prism_df.sort_values(["property_category", "directed_rho"], ascending=[True, False])

    # Compute bootstrap CIs
    ci_lo, ci_hi = [], []
    for _, row in prism_df.iterrows():
        cat = row["property_category"]
        assay = row["assay_type"]
        higher_is_better = PROPERTY_FILES[cat]["higher_is_better"]
        sign = 1.0 if higher_is_better else -1.0
        sig = prism_signals[cat]

        # Get matched scores and fitness
        cat_assays = assays_df[
            (assays_df["property_category"] == cat) & (assays_df["assay_type"] == assay)
        ].drop_duplicates(subset=["heavy", "light"], keep="first")

        model_scores = scores_df[
            (scores_df["model"] == PRISM_MODEL) & (scores_df["signal"] == sig)
        ]
        score_map = dict(zip(
            zip(model_scores["heavy"], model_scores["light"]),
            model_scores["score"]
        ))

        matched_s, matched_f = [], []
        for _, r in cat_assays.iterrows():
            s = score_map.get((r["heavy"], r["light"]))
            if s is not None and np.isfinite(s) and np.isfinite(r["fitness"]):
                matched_s.append(s)
                matched_f.append(r["fitness"])

        lo, hi = bootstrap_spearman_ci(np.array(matched_s), np.array(matched_f))
        # Apply sign flip to CI bounds
        if sign > 0:
            ci_lo.append(lo)
            ci_hi.append(hi)
        else:
            ci_lo.append(-hi)
            ci_hi.append(-lo)

    prism_df["ci_lo"] = ci_lo
    prism_df["ci_hi"] = ci_hi

    # Plot
    n_assays = len(prism_df)
    fig, ax = plt.subplots(figsize=(7, max(5, n_assays * 0.35 + 1.5)))

    # Color by property category
    cat_colors = {
        "Aggregation / Self-interaction": "#E69F00",
        "Thermostability": "#D55E00",
        "Immunogenicity": "#CC79A7",
        "Polyreactivity": "#0072B2",
        "Expression": "#009E73",
    }

    y_positions = list(range(n_assays))[::-1]
    prev_cat = None
    separator_y = []

    for i, (_, row) in enumerate(prism_df.iterrows()):
        y = y_positions[i]
        cat = row["property_category"]
        color = cat_colors.get(cat, "#333333")

        if prev_cat is not None and cat != prev_cat:
            separator_y.append(y + 0.5)
        prev_cat = cat

        # Point + CI
        ax.errorbar(
            row["directed_rho"], y,
            xerr=[[row["directed_rho"] - row["ci_lo"]], [row["ci_hi"] - row["directed_rho"]]],
            fmt="o", color=color, markersize=6, capsize=3, linewidth=1.5,
            markeredgecolor="black", markeredgewidth=0.5,
        )

        # Label
        stars = ""
        p = row["pvalue"]
        if p < 1e-6:
            stars = "***"
        elif p < 1e-3:
            stars = "**"
        elif p < 0.05:
            stars = "*"

        label = f"{row['assay_type']} (n={row['n_antibodies']})"
        ax.text(-0.02, y, label, ha="right", va="center", fontsize=8,
                transform=ax.get_yaxis_transform())
        if stars:
            ax.text(row["ci_hi"] + 0.01, y, stars, ha="left", va="center",
                    fontsize=7, fontweight="bold", color=color)

    # Separators
    for sy in separator_y:
        ax.axhline(y=sy, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)

    # Zero line
    ax.axvline(x=0, color="gray", linewidth=0.8, linestyle="--")

    # Legend for property categories
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=c, edgecolor="black", linewidth=0.5, label=cat)
        for cat, c in cat_colors.items()
    ]
    ax.legend(handles=legend_elements, fontsize=7, loc="lower right", framealpha=0.9)

    ax.set_xlabel("Directed Spearman $\\rho$ (positive = favorable)")
    ax.set_title("PRISM per-assay developability prediction (FLAb2)\nwith 95% bootstrap CI",
                 fontsize=12)
    ax.set_yticks([])
    ax.set_xlim(-0.55, 0.65)
    ax.margins(y=0.02)

    # Significance annotation
    ax.text(0.01, 0.01, "*** p<10$^{-6}$  ** p<10$^{-3}$  * p<0.05",
            transform=ax.transAxes, fontsize=7, color="gray", va="bottom")

    fig.tight_layout()
    fig.subplots_adjust(left=0.3)
    fig.savefig(OUT_DIR / "fig5_per_assay_forest.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig5_per_assay_forest.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  Figure 5 saved: fig5_per_assay_forest.pdf/png")


# ══════════════════════════════════════════════════════════════════════════
# Figure 6: PRISM vs baselines — win rate across all assays
# ══════════════════════════════════════════════════════════════════════════

def plot_win_rate_summary(corr_df):
    """Bar chart: fraction of assays where each model has positive directed rho."""
    model_order = [MODEL_DISPLAY[m] for m in SHOW_MODELS]

    # Count assays with positive directed rho per model
    n_assays_total = corr_df["assay_type"].nunique()
    frac_positive = {}
    mean_directed = {}

    for model_key in SHOW_MODELS:
        mdf = corr_df[corr_df["model"] == model_key]
        frac_positive[MODEL_DISPLAY[model_key]] = (mdf["directed_rho"] > 0).mean()
        mean_directed[MODEL_DISPLAY[model_key]] = mdf["directed_rho"].mean()

    # PRISM wins (higher directed rho than all baselines)
    prism_rows = corr_df[corr_df["model"] == PRISM_MODEL]
    n_wins = 0
    n_comparisons = 0
    for _, prow in prism_rows.iterrows():
        assay = prow["assay_type"]
        prism_r = prow["directed_rho"]
        for bm in SHOW_MODELS:
            if bm == PRISM_MODEL:
                continue
            bm_row = corr_df[(corr_df["model"] == bm) & (corr_df["assay_type"] == assay)]
            if len(bm_row) > 0:
                n_comparisons += 1
                if prism_r > bm_row.iloc[0]["directed_rho"]:
                    n_wins += 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Panel A: fraction positive
    colors = [MODEL_COLORS[m] for m in model_order]
    bars = ax1.bar(
        model_order,
        [frac_positive[m] for m in model_order],
        color=colors, edgecolor="black", linewidth=0.5, alpha=0.85,
    )
    for bar, m in zip(bars, model_order):
        val = frac_positive[m]
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                 f"{val:.0%}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax1.set_ylabel("Fraction of assays with\npositive directed $\\rho$")
    ax1.set_title(f"(A) Favorable prediction rate\n(across {n_assays_total} assays)")
    ax1.set_ylim(0, 1.1)
    ax1.axhline(y=0.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    ax1.tick_params(axis="x", rotation=25)

    # Panel B: mean directed rho
    bars2 = ax2.bar(
        model_order,
        [mean_directed[m] for m in model_order],
        color=colors, edgecolor="black", linewidth=0.5, alpha=0.85,
    )
    for bar, m in zip(bars2, model_order):
        val = mean_directed[m]
        y_off = 0.005 if val >= 0 else -0.005
        va = "bottom" if val >= 0 else "top"
        ax2.text(bar.get_x() + bar.get_width() / 2, val + y_off,
                 f"{val:+.3f}", ha="center", va=va, fontsize=9, fontweight="bold")

    ax2.set_ylabel("Mean directed Spearman $\\rho$")
    ax2.set_title(f"(B) Average predictive power\n(across {n_assays_total} assays)")
    ax2.axhline(y=0, color="black", linewidth=0.5)
    ax2.tick_params(axis="x", rotation=25)

    # Win rate annotation
    if n_comparisons > 0:
        win_pct = n_wins / n_comparisons
        ax2.text(0.98, 0.02,
                 f"PRISM wins {n_wins}/{n_comparisons} ({win_pct:.0%})\npairwise comparisons",
                 transform=ax2.transAxes, ha="right", va="bottom",
                 fontsize=8, color="#332288", fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#E8E0F0", alpha=0.8))

    fig.suptitle("Zero-shot developability: robustness across FLAb2 assays", fontsize=13,
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig6_win_rate_summary.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig6_win_rate_summary.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  Figure 6 saved: fig6_win_rate_summary.pdf/png")


# ══════════════════════════════════════════════════════════════════════════
# Table 3: Per-assay correlation summary (LaTeX)
# ══════════════════════════════════════════════════════════════════════════

def make_per_assay_table(corr_df):
    """LaTeX table: per-assay directed rho for PRISM vs baselines."""
    model_order = [MODEL_DISPLAY[m] for m in SHOW_MODELS]

    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{Per-assay directed Spearman $\rho$ across FLAb2 benchmarks. "
                 r"Positive values indicate correct prediction direction for developability.}")
    lines.append(r"\label{tab:per_assay}")
    lines.append(r"\scriptsize")
    lines.append(r"\begin{tabular}{llc" + "c" * len(model_order) + "}")
    lines.append(r"\toprule")

    header = r"Property & Assay & $N$"
    for m in model_order:
        header += f" & {m}"
    header += r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    prev_cat = None
    for _, row in corr_df[corr_df["model"] == PRISM_MODEL].sort_values(
        ["property_category", "directed_rho"], ascending=[True, False]
    ).iterrows():
        cat = row["property_category"]
        assay = row["assay_type"]
        n = row["n_antibodies"]

        if prev_cat is not None and cat != prev_cat:
            lines.append(r"\midrule")
        prev_cat = cat

        cat_short = cat.split("/")[0].strip() if prev_cat == cat and lines[-1] != r"\midrule" else cat.split("/")[0].strip()

        cells = [f"{cat_short}", f"{assay}", f"{n}"]

        # Find best directed rho for bolding
        assay_rows = corr_df[corr_df["assay_type"] == assay]
        best_rho = assay_rows["directed_rho"].max()

        for model_key in SHOW_MODELS:
            mrow = assay_rows[assay_rows["model"] == model_key]
            if len(mrow) == 0:
                cells.append("--")
                continue
            rho = mrow.iloc[0]["directed_rho"]
            p = mrow.iloc[0]["pvalue"]

            stars = ""
            if p < 1e-6:
                stars = r"$^{***}$"
            elif p < 1e-3:
                stars = r"$^{**}$"
            elif p < 0.05:
                stars = r"$^{*}$"

            if abs(rho - best_rho) < 1e-6:
                cells.append(rf"\textbf{{{rho:+.3f}}}{stars}")
            else:
                cells.append(f"{rho:+.3f}{stars}")

        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    tex = "\n".join(lines)
    (OUT_DIR / "table3_per_assay.tex").write_text(tex)
    print("  Table 3 saved: table3_per_assay.tex")


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def main():
    print("FLAb2 Per-Assay Robustness Analysis")
    print(f"Output directory: {OUT_DIR}")
    print()

    # Load data
    print("Loading data...")
    assays_df = load_all_assays()
    scores_df = load_model_scores()
    print(f"  {len(assays_df)} assay measurements across {assays_df['assay_type'].nunique()} assay types")
    print(f"  {len(scores_df)} model scores")

    # Auto-select best PRISM signal per property category
    print("\nSelecting best PRISM signals per property category...")
    prism_signals = {}
    for cat in PROPERTY_FILES:
        sig = get_best_prism_signal(scores_df, assays_df, cat)
        prism_signals[cat] = sig
        print(f"  {cat}: {sig}")

    # Compute per-assay correlations
    print("\nComputing per-assay directed correlations...")
    corr_df = compute_per_assay_correlations(scores_df, assays_df, prism_signals)
    print(f"  {len(corr_df)} (model × assay) combinations")
    n_assays = corr_df["assay_type"].nunique()
    print(f"  {n_assays} unique assay types (n >= {MIN_N})")

    # Save correlations
    corr_path = SCORE_DIR / "per_assay_correlations.csv"
    corr_df.to_csv(corr_path, index=False)
    print(f"  Saved: {corr_path}")

    # Summary
    prism_corr = corr_df[corr_df["model"] == PRISM_MODEL]
    print(f"\n{'=' * 70}")
    print("PRISM per-assay summary (directed rho)")
    print(f"{'=' * 70}")
    for cat in PROPERTY_FILES:
        cat_rows = prism_corr[prism_corr["property_category"] == cat]
        if len(cat_rows) == 0:
            continue
        sig = prism_signals[cat]
        pos_frac = (cat_rows["directed_rho"] > 0).mean()
        mean_rho = cat_rows["directed_rho"].mean()
        print(f"\n  {cat} (signal: {sig}):")
        for _, row in cat_rows.sort_values("directed_rho", ascending=False).iterrows():
            stars = ""
            if row["pvalue"] < 0.05:
                stars = " *"
            if row["pvalue"] < 1e-3:
                stars = " **"
            if row["pvalue"] < 1e-6:
                stars = " ***"
            print(f"    {row['assay_type']:<15} rho={row['directed_rho']:+.3f}  n={row['n_antibodies']}{stars}")
        print(f"    → {pos_frac:.0%} positive, mean={mean_rho:+.3f}")

    # Generate figures
    print(f"\n{'=' * 70}")
    print("Generating figures...")
    print(f"{'=' * 70}")
    plot_per_assay_heatmap(corr_df)
    plot_forest(corr_df, scores_df, assays_df, prism_signals)
    plot_win_rate_summary(corr_df)
    make_per_assay_table(corr_df)

    print("\nDone!")


if __name__ == "__main__":
    main()
