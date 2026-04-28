#!/usr/bin/env python3
"""
Plot Bottom-K% Recall per antibody.
CR9114-H1 and CR9114-H3 are averaged into a single "CR9114" entry.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

STRAT_DIR = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "features"
    / "evaluation_results"
    / "flab2_binding"
    / "stratified"
)

RECALL_COLS = ["Bottom-Recall@1%", "Bottom-Recall@5%", "Bottom-Recall@10%"]
K_LABELS = ["Bottom 1%", "Bottom 5%", "Bottom 10%"]

MODEL_ORDER = [
    "PRISM (ours)",
    "ESM2-650M",
    "ESM2-35M",
    "AbLang2",
    "AntiBERTy",
    "Sapiens",
]

MODEL_COLORS = {
    "PRISM (ours)": "#2563EB",
    "ESM2-650M": "#6B7280",
    "ESM2-35M": "#9CA3AF",
    "AbLang2": "#10B981",
    "AntiBERTy": "#F59E0B",
    "Sapiens": "#EF4444",
}

ANTIBODIES = ["G6.31", "CR9114", "Trastuzumab"]


def load_and_merge(path):
    df = pd.read_csv(path)

    # Average CR9114-H1 and CR9114-H3 into "CR9114"
    cr_mask = df["Dataset"].isin(["CR9114-H1", "CR9114-H3"])
    cr = (
        df[cr_mask]
        .groupby("Model")[RECALL_COLS]
        .mean()
        .reset_index()
    )
    cr["Dataset"] = "CR9114"

    others = df[~cr_mask][["Dataset", "Model"] + RECALL_COLS].copy()
    return pd.concat([others, cr], ignore_index=True)


def main():
    df = load_and_merge(STRAT_DIR / "bottomk_recall_results.csv")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)

    n_models = len(MODEL_ORDER)
    n_k = len(RECALL_COLS)
    bar_w = 0.13
    group_w = n_models * bar_w

    for ax, ab in zip(axes, ANTIBODIES):
        sub = df[df["Dataset"] == ab]

        for j, (col, klab) in enumerate(zip(RECALL_COLS, K_LABELS)):
            x_base = j * (group_w + 0.18)

            for i, model in enumerate(MODEL_ORDER):
                row = sub[sub["Model"] == model]
                if row.empty:
                    continue
                val = row[col].values[0]
                x = x_base + i * bar_w
                bar = ax.bar(
                    x,
                    val,
                    width=bar_w * 0.88,
                    color=MODEL_COLORS[model],
                    edgecolor="white",
                    linewidth=0.4,
                )
                # Value label
                if val >= 0.005:
                    ax.text(
                        x, val + 0.004, f"{val:.3f}",
                        ha="center", va="bottom", fontsize=5.5, rotation=90,
                    )

            # Random baseline line for this K
            k_frac = [0.01, 0.05, 0.10][j]
            x_left = x_base - bar_w * 0.5
            x_right = x_base + (n_models - 0.5) * bar_w
            ax.hlines(
                k_frac, x_left, x_right,
                colors="#DC2626", linestyles="--", linewidth=0.9, alpha=0.7,
            )

        # X-tick labels (K% groups)
        tick_positions = [
            j * (group_w + 0.18) + (n_models - 1) * bar_w / 2
            for j in range(n_k)
        ]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(K_LABELS, fontsize=9)

        ax.set_title(ab, fontsize=12, fontweight="bold", pad=8)
        ax.set_xlim(-bar_w, n_k * (group_w + 0.18) - 0.18 + bar_w)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", length=0)

    axes[0].set_ylabel("Bottom-K% Recall", fontsize=10)

    # Legend
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=MODEL_COLORS[m]) for m in MODEL_ORDER
    ]
    handles.append(plt.Line2D([0], [0], color="#DC2626", linestyle="--", linewidth=1))
    labels = MODEL_ORDER + ["Random baseline"]
    fig.legend(
        handles, labels,
        loc="upper center",
        ncol=len(labels),
        fontsize=7.5,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )

    fig.suptitle(
        "Bottom-K% Recall: Filtering the Worst Variants",
        fontsize=13, fontweight="bold", y=1.09,
    )

    plt.tight_layout()
    out = STRAT_DIR / "bottomk_recall_per_antibody.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


if __name__ == "__main__":
    main()
