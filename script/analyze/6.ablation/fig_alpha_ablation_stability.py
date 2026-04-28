#!/usr/bin/env python3
"""
Generate figures for reviewer response: Stop-gradient stability ablation.

Figures:
  Fig A: Train loss curves (fixed vs learned α) — overlaid, showing identical volatility
  Fig B: Per-component loss volatility (rolling std) comparison
  Fig C: Alpha trajectory + AA_PPL_NGL divergence (dual axis)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ============================================================
# Config — adjust OUTPUTS_DIR (or set PRISM_OUTPUTS env var) to point at the
# directory holding the two ablation training runs (alpha_fixed / alpha_learned).
# ============================================================
import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUTS_DIR = Path(os.environ.get("PRISM_OUTPUTS", _REPO_ROOT / "outputs"))

RUNS = {
    r"Fixed $\alpha$=1.0": str(OUTPUTS_DIR / "ablation_alpha_fixed_1.0_esm2_t12_35M_UR50D_custom_unfrozen12_lr4e-4_bs256" / "version_0"),
    r"Learned $\alpha$": str(OUTPUTS_DIR / "ablation_alpha_learned_esm2_t12_35M_UR50D_custom_unfrozen12_lr4e-4_bs256" / "version_0"),
}
COLORS = {
    r"Fixed $\alpha$=1.0": "#D6604D",
    r"Learned $\alpha$": "#2166AC",
}
OUT = str(Path(__file__).resolve().parent)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ============================================================
# Load data
# ============================================================
data = {}
for label, log_dir in RUNS.items():
    ea = EventAccumulator(log_dir)
    ea.Reload()
    data[label] = {}
    for tag in ea.Tags().get("scalars", []):
        events = ea.Scalars(tag)
        data[label][tag] = [(e.step, e.value) for e in events]

print("Data loaded.\n")


def rolling_std(values, window=20):
    """Compute rolling standard deviation."""
    stds = []
    steps = []
    for i in range(window, len(values)):
        stds.append(np.std(values[i - window : i]))
        steps.append(i)
    return steps, stds


# ============================================================
# Figure: Combined 3-panel (A, B, C)
# ============================================================
fig = plt.figure(figsize=(16, 12))
gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 1], hspace=0.35, wspace=0.3)

# ------ Panel A1: Train loss curves overlaid ------
ax_a1 = fig.add_subplot(gs[0, 0])

for label in RUNS:
    sv = data[label].get("train/loss_step", [])
    if sv:
        steps, vals = zip(*sv)
        ax_a1.plot(steps, vals, color=COLORS[label], alpha=0.4, linewidth=0.8)
        # Smoothed (rolling mean, window=10)
        w = 10
        if len(vals) > w:
            smooth = np.convolve(vals, np.ones(w) / w, mode="valid")
            ax_a1.plot(steps[w - 1 :], smooth, color=COLORS[label], linewidth=2.0, label=label)

ax_a1.set_xlabel("Optimizer Step")
ax_a1.set_ylabel("Total Train Loss")
ax_a1.set_title("A1. Training Loss Curves", fontweight="bold")
ax_a1.legend(loc="upper right")
ax_a1.set_xlim(0, 500)

# ------ Panel A2: Origin Loss convergence ------
ax_a2 = fig.add_subplot(gs[0, 1])

for label in RUNS:
    sv = data[label].get("train/Origin_Loss_step", [])
    if sv:
        steps, vals = zip(*sv)
        ax_a2.plot(steps, vals, color=COLORS[label], alpha=0.4, linewidth=0.8)
        w = 10
        if len(vals) > w:
            smooth = np.convolve(vals, np.ones(w) / w, mode="valid")
            ax_a2.plot(steps[w - 1 :], smooth, color=COLORS[label], linewidth=2.0, label=label)

# Mark warmup end
ax_a2.axvline(50, color="gray", linewidth=1.0, linestyle="--", alpha=0.6)
ax_a2.text(58, 0.085, "warmup ends", fontsize=8, color="gray", va="top")

ax_a2.set_xlabel("Optimizer Step")
ax_a2.set_ylabel("Origin Loss")
ax_a2.set_title("A2. Origin Head Convergence", fontweight="bold")
ax_a2.legend(loc="upper right")
ax_a2.set_xlim(0, 500)

# ------ Panel B: Rolling std of train loss (volatility) ------
ax_b1 = fig.add_subplot(gs[1, 0])
ax_b2 = fig.add_subplot(gs[1, 1])

for ax_b, tag, title in [
    (ax_b1, "train/loss_step", "B1. Total Loss Volatility"),
    (ax_b2, "train/Final_Loss_step", "B2. Final Loss Volatility"),
]:
    for label in RUNS:
        sv = data[label].get(tag, [])
        if sv:
            _, vals = zip(*sv)
            s_idx, stds = rolling_std(list(vals), window=20)
            ax_b.plot(s_idx, stds, color=COLORS[label], linewidth=1.5, alpha=0.7, label=label)
            # Add mean line
            mean_std = np.mean(stds)
            ax_b.axhline(mean_std, color=COLORS[label], linewidth=1.0, linestyle="--", alpha=0.5)
            ax_b.text(
                len(s_idx) * 0.95, mean_std + 0.02,
                f"mean={mean_std:.3f}",
                color=COLORS[label], fontsize=8, ha="right", va="bottom",
            )

    ax_b.set_xlabel("Step")
    ax_b.set_ylabel("Rolling Std (window=20)")
    ax_b.set_title(title, fontweight="bold")
    ax_b.legend(loc="upper right", fontsize=8)

# ------ Panel C: Alpha trajectory + AA_PPL_NGL comparison ------
ax_c1 = fig.add_subplot(gs[2, 0])
ax_c2 = fig.add_subplot(gs[2, 1])

# C1: Alpha trajectory (learned only)
learned_key = r"Learned $\alpha$"
alpha_train = data[learned_key].get("train/Mean_Alpha_step", [])
alpha_val = data[learned_key].get("val/Mean_Alpha", [])
alpha_cdr = data[learned_key].get("val/Alpha_Mean_CDR", [])
alpha_fr = data[learned_key].get("val/Alpha_Mean_FR", [])

if alpha_train:
    steps, vals = zip(*alpha_train)
    ax_c1.plot(steps, vals, color=COLORS[learned_key], linewidth=1.0, alpha=0.3)
    # Smoothed
    w = 10
    if len(vals) > w:
        smooth = np.convolve(vals, np.ones(w) / w, mode="valid")
        ax_c1.plot(steps[w - 1 :], smooth, color=COLORS[learned_key], linewidth=2.0, label="Mean (train)")

if alpha_cdr:
    s, v = zip(*alpha_cdr)
    ax_c1.plot(s, v, color="#D6604D", linewidth=2.0, linestyle="--", marker="o", markersize=3, label="CDR (val)")
if alpha_fr:
    s, v = zip(*alpha_fr)
    ax_c1.plot(s, v, color="#4393C3", linewidth=2.0, linestyle="--", marker="s", markersize=3, label="FR (val)")

# Fixed alpha reference
ax_c1.axhline(1.0, color=COLORS[r"Fixed $\alpha$=1.0"], linewidth=1.5, linestyle=":", alpha=0.7, label=r"Fixed $\alpha$=1.0")
ax_c1.axhline(0.5, color="gray", linewidth=0.8, linestyle=":", alpha=0.4)
ax_c1.text(10, 0.52, "init=0.5", fontsize=8, color="gray")

ax_c1.set_xlabel("Optimizer Step")
ax_c1.set_ylabel(r"$\alpha$")
ax_c1.set_title(r"C1. Learned $\alpha$ Trajectory", fontweight="bold")
ax_c1.legend(loc="lower right", fontsize=8)
ax_c1.set_ylim(0.35, 1.05)
ax_c1.set_xlim(0, 500)

# C2: val/AA_PPL_NGL comparison
for label in RUNS:
    sv = data[label].get("val/AA_PPL_NGL", [])
    if sv:
        steps, vals = zip(*sv)
        ax_c2.plot(steps, vals, color=COLORS[label], linewidth=2.0, marker="o", markersize=4, label=label)

ax_c2.set_xlabel("Optimizer Step")
ax_c2.set_ylabel("AA Head PPL (NGL positions)")
ax_c2.set_title("C2. AA Head NGL Perplexity", fontweight="bold")
ax_c2.legend(loc="upper right", fontsize=8)

# Annotate final values
for label in RUNS:
    sv = data[label].get("val/AA_PPL_NGL", [])
    if sv:
        last_step, last_val = sv[-1]
        ax_c2.annotate(
            f"{last_val:.2f}",
            xy=(last_step, last_val),
            xytext=(last_step - 80, last_val + 0.3),
            fontsize=9, fontweight="bold", color=COLORS[label],
            arrowprops=dict(arrowstyle="->", color=COLORS[label], lw=1.0),
        )

fig.suptitle(
    r"Stop-Gradient Stability Ablation: Fixed $\alpha$=1.0 vs Learned $\alpha$ (500 steps)",
    fontsize=13, fontweight="bold", y=1.01,
)

plt.savefig(f"{OUT}/fig_stopgrad_stability_ablation.png")
plt.savefig(f"{OUT}/fig_stopgrad_stability_ablation.pdf")
plt.close()
print("Saved: fig_stopgrad_stability_ablation.png/pdf")
