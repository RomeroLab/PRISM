#!/usr/bin/env python3
"""
Plot alpha gating values by IMGT position for PRISM v34.1b model.

Generates per-position violin/box plots showing how alpha varies across
antibody positions, colored by IMGT region (FR1-FR4, CDR1-CDR3).
"""

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
from tqdm.auto import tqdm

# ============================================================
# Config — override via environment variables to use a local checkpoint or
# the published HuggingFace Hub model "RomeroLab-Duke/prism-antibody".
# ============================================================
CHECKPOINT = os.environ.get("PRISM_CHECKPOINT", "RomeroLab-Duke/prism-antibody")
DATA_PATH = os.environ.get(
    "PRISM_DATA_PATH",
    str(_REPO_ROOT / "data" / "train_data" / "paired_anarci_relabeled.parquet"),
)
N_SAMPLES = 500  # number of validation sequences to process
BATCH_SIZE = 32
OUTPUT_DIR = str(Path(__file__).resolve().parent)

# Region mask digit -> region name
REGION_MAP = {
    "0": "FR1",
    "1": "CDR1",
    "2": "FR2",
    "3": "CDR2",
    "4": "FR3",
    "5": "CDR3",
    "6": "FR4",
}

# Colors for each region (publication-friendly)
REGION_COLORS = {
    "FR1": "#4393C3",
    "CDR1": "#D6604D",
    "FR2": "#4393C3",
    "CDR2": "#D6604D",
    "FR3": "#4393C3",
    "CDR3": "#D6604D",
    "FR4": "#4393C3",
}

# ============================================================
# Load model
# ============================================================
print("Loading PRISM v34.1b model...")
import prism

model = prism.pretrained(CHECKPOINT, device="auto")
print(f"Model loaded on {model.device}")

# ============================================================
# Load data
# ============================================================
print("Loading data...")
df = pd.read_parquet(DATA_PATH)
val_df = df[df["split"] == "valid"].sample(n=N_SAMPLES, random_state=42).reset_index(drop=True)
print(f"Using {len(val_df)} validation sequences")

# ============================================================
# Extract alpha values per IMGT position
# ============================================================

def extract_alpha_paired(model, heavy_seqs, light_seqs, rmask_heavy, rmask_light, batch_size=32):
    """Extract alpha values from paired forward pass and map to IMGT positions."""
    records = []
    n = len(heavy_seqs)
    for i in tqdm(range(0, n, batch_size), desc="Processing paired"):
        batch_h = heavy_seqs[i : i + batch_size]
        batch_l = light_seqs[i : i + batch_size]
        batch_rmh = rmask_heavy[i : i + batch_size]
        batch_rml = rmask_light[i : i + batch_size]

        results = model.forward(heavy_chains=batch_h, light_chains=batch_l)
        if not isinstance(results, list):
            results = [results]

        for j, result in enumerate(results):
            # Heavy chain
            alpha_h = result["heavy"]["alpha"]
            seq_h = batch_h[j]
            rm_h = batch_rmh[j]
            seq_len_h = min(len(alpha_h), len(rm_h), len(seq_h))
            for pos in range(seq_len_h):
                records.append({
                    "imgt_pos": pos + 1,
                    "alpha": float(alpha_h[pos]),
                    "region": REGION_MAP.get(rm_h[pos], "Unknown"),
                    "chain": "Heavy",
                    "aa": seq_h[pos],
                })

            # Light chain
            alpha_l = result["light"]["alpha"]
            seq_l = batch_l[j]
            rm_l = batch_rml[j]
            seq_len_l = min(len(alpha_l), len(rm_l), len(seq_l))
            for pos in range(seq_len_l):
                records.append({
                    "imgt_pos": pos + 1,
                    "alpha": float(alpha_l[pos]),
                    "region": REGION_MAP.get(rm_l[pos], "Unknown"),
                    "chain": "Light",
                    "aa": seq_l[pos],
                })
    return records


print("\nExtracting alpha values (paired forward pass)...")
all_records = extract_alpha_paired(
    model,
    val_df["HEAVY_CHAIN_AA_SEQUENCE"].tolist(),
    val_df["LIGHT_CHAIN_AA_SEQUENCE"].tolist(),
    val_df["region_mask_heavy"].tolist(),
    val_df["region_mask_light"].tolist(),
    batch_size=BATCH_SIZE,
)
alpha_df = pd.DataFrame(all_records)
print(f"\nTotal records: {len(alpha_df):,}")

# ============================================================
# Plot: Alpha distribution by IMGT position (violin/box)
# ============================================================

def plot_alpha_by_imgt(alpha_df, chain, output_path):
    """Create a per-position alpha distribution plot for one chain."""
    chain_df = alpha_df[alpha_df["chain"] == chain].copy()
    max_pos = chain_df["imgt_pos"].max()

    fig, ax = plt.subplots(figsize=(20, 5))

    positions = sorted(chain_df["imgt_pos"].unique())

    # Collect data for boxplot
    bp_data = []
    bp_positions = []
    bp_colors = []

    for pos in positions:
        pos_data = chain_df[chain_df["imgt_pos"] == pos]["alpha"].values
        if len(pos_data) > 0:
            bp_data.append(pos_data)
            bp_positions.append(pos)
            region = chain_df[chain_df["imgt_pos"] == pos]["region"].iloc[0]
            bp_colors.append(REGION_COLORS.get(region, "#999999"))

    # Draw boxplot
    bp = ax.boxplot(
        bp_data,
        positions=bp_positions,
        widths=0.7,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", linewidth=1.5),
        whiskerprops=dict(linewidth=0.8),
        capprops=dict(linewidth=0.8),
    )

    for patch, color in zip(bp["boxes"], bp_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Add region background shading
    region_boundaries = {
        "FR1": (1, 26),
        "CDR1": (27, 38),
        "FR2": (39, 55),
        "CDR2": (56, 65),
        "FR3": (66, 104),
        "CDR3": (105, 117),
        "FR4": (118, 128),
    }

    for region_name, (start, end) in region_boundaries.items():
        if start > max_pos:
            continue
        end = min(end, max_pos)
        color = REGION_COLORS.get(region_name, "#999999")
        ax.axvspan(start - 0.5, end + 0.5, alpha=0.08, color=color, zorder=0)
        mid = (start + end) / 2
        ax.text(mid, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1.0,
                region_name, ha="center", va="bottom", fontsize=8, fontweight="bold",
                color=color)

    # Also overlay the median line
    medians = []
    for pos in positions:
        pos_data = chain_df[chain_df["imgt_pos"] == pos]["alpha"].values
        medians.append(np.median(pos_data))
    ax.plot(positions, medians, color="black", linewidth=1.0, alpha=0.5, zorder=5)

    ax.set_xlabel("IMGT Position", fontsize=12)
    ax.set_ylabel("Alpha (gating value)", fontsize=12)
    ax.set_title(f"PRISM v34.1b — Alpha by IMGT Position ({chain} Chain, n={len(chain_df['alpha'].unique()//1)})",
                 fontsize=14)

    # X-axis ticks every 5 positions
    xticks = [p for p in positions if p % 5 == 0 or p == 1]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks, fontsize=8)
    ax.set_xlim(0.5, max_pos + 0.5)

    # Legend
    fr_patch = mpatches.Patch(color="#4393C3", alpha=0.7, label="Framework (FR)")
    cdr_patch = mpatches.Patch(color="#D6604D", alpha=0.7, label="CDR")
    ax.legend(handles=[fr_patch, cdr_patch], loc="upper right", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


# Fix title (remove broken expression)
def plot_alpha_by_imgt_v2(alpha_df, chain, output_path):
    """Create a per-position alpha distribution plot for one chain."""
    chain_df = alpha_df[alpha_df["chain"] == chain].copy()
    max_pos = chain_df["imgt_pos"].max()
    n_seqs = len(alpha_df[alpha_df["chain"] == chain].groupby("imgt_pos").size())

    fig, ax = plt.subplots(figsize=(20, 5))

    positions = sorted(chain_df["imgt_pos"].unique())

    # Collect data for boxplot
    bp_data = []
    bp_positions = []
    bp_colors = []

    for pos in positions:
        pos_data = chain_df[chain_df["imgt_pos"] == pos]["alpha"].values
        if len(pos_data) > 0:
            bp_data.append(pos_data)
            bp_positions.append(pos)
            region = chain_df[chain_df["imgt_pos"] == pos]["region"].iloc[0]
            bp_colors.append(REGION_COLORS.get(region, "#999999"))

    # Draw boxplot
    bp = ax.boxplot(
        bp_data,
        positions=bp_positions,
        widths=0.7,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", linewidth=1.5),
        whiskerprops=dict(linewidth=0.8),
        capprops=dict(linewidth=0.8),
    )

    for patch, color in zip(bp["boxes"], bp_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Add region background shading + labels at top
    region_boundaries = {
        "FR1": (1, 26),
        "CDR1": (27, 38),
        "FR2": (39, 55),
        "CDR2": (56, 65),
        "FR3": (66, 104),
        "CDR3": (105, 117),
        "FR4": (118, 128),
    }

    ymin, ymax = ax.get_ylim()

    for region_name, (start, end) in region_boundaries.items():
        if start > max_pos:
            continue
        end_clipped = min(end, max_pos)
        color = REGION_COLORS.get(region_name, "#999999")
        ax.axvspan(start - 0.5, end_clipped + 0.5, alpha=0.08, color=color, zorder=0)

    # Overlay median line
    medians = []
    for pos in positions:
        pos_data = chain_df[chain_df["imgt_pos"] == pos]["alpha"].values
        medians.append(np.median(pos_data))
    ax.plot(positions, medians, color="black", linewidth=1.0, alpha=0.5, zorder=5)

    ax.set_xlabel("IMGT Position", fontsize=12)
    ax.set_ylabel("Alpha (gating value)", fontsize=12)
    ax.set_title(
        f"PRISM v34.1b — Alpha Gating by IMGT Position ({chain} Chain, n={N_SAMPLES} sequences)",
        fontsize=14,
    )

    # Region labels above plot
    ax2 = ax.secondary_xaxis("top")
    ax2.set_xticks([])
    ax2.set_xticklabels([])
    for region_name, (start, end) in region_boundaries.items():
        if start > max_pos:
            continue
        end_clipped = min(end, max_pos)
        mid = (start + end_clipped) / 2
        color = REGION_COLORS.get(region_name, "#999999")
        ax.annotate(
            region_name, xy=(mid, 1.02), xycoords=("data", "axes fraction"),
            ha="center", va="bottom", fontsize=9, fontweight="bold", color=color,
        )

    # X-axis ticks every 5 positions
    xticks = [p for p in positions if p % 5 == 0 or p == 1]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks, fontsize=8)
    ax.set_xlim(0.5, max_pos + 0.5)

    # Legend
    fr_patch = mpatches.Patch(color="#4393C3", alpha=0.7, label="Framework (FR)")
    cdr_patch = mpatches.Patch(color="#D6604D", alpha=0.7, label="CDR")
    ax.legend(handles=[fr_patch, cdr_patch], loc="upper right", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


# ============================================================
# Also: histogram of alpha grouped by region
# ============================================================

def plot_alpha_histogram_by_region(alpha_df, chain, output_path):
    """Overlaid histograms of alpha split by FR vs CDR regions."""
    chain_df = alpha_df[alpha_df["chain"] == chain].copy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: all 7 regions overlaid
    ax = axes[0]
    region_order = ["FR1", "CDR1", "FR2", "CDR2", "FR3", "CDR3", "FR4"]
    for region in region_order:
        rdata = chain_df[chain_df["region"] == region]["alpha"].values
        if len(rdata) == 0:
            continue
        color = REGION_COLORS[region]
        ls = "-" if "FR" in region else "--"
        ax.hist(rdata, bins=50, density=True, alpha=0.4, color=color, label=region,
                histtype="stepfilled", linewidth=0)
        ax.hist(rdata, bins=50, density=True, alpha=0.9, color=color,
                histtype="step", linewidth=1.5, linestyle=ls)

    ax.set_xlabel("Alpha", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(f"{chain} Chain — Alpha Distribution by Region", fontsize=13)
    ax.legend(fontsize=9, ncol=2)

    # Right: FR (all) vs CDR (all)
    ax = axes[1]
    fr_alpha = chain_df[chain_df["region"].str.startswith("FR")]["alpha"].values
    cdr_alpha = chain_df[chain_df["region"].str.startswith("CDR")]["alpha"].values

    ax.hist(fr_alpha, bins=50, density=True, alpha=0.5, color="#4393C3",
            label=f"FR (n={len(fr_alpha):,})", histtype="stepfilled")
    ax.hist(cdr_alpha, bins=50, density=True, alpha=0.5, color="#D6604D",
            label=f"CDR (n={len(cdr_alpha):,})", histtype="stepfilled")
    ax.hist(fr_alpha, bins=50, density=True, color="#4393C3", histtype="step", linewidth=2)
    ax.hist(cdr_alpha, bins=50, density=True, color="#D6604D", histtype="step", linewidth=2)

    ax.set_xlabel("Alpha", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(f"{chain} Chain — FR vs CDR Alpha", fontsize=13)
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


# ============================================================
# Generate all plots
# ============================================================

print("\n=== Generating plots ===")

# Per-position boxplots
plot_alpha_by_imgt_v2(alpha_df, "Heavy", f"{OUTPUT_DIR}/alpha_by_imgt_heavy.png")
plot_alpha_by_imgt_v2(alpha_df, "Light", f"{OUTPUT_DIR}/alpha_by_imgt_light.png")

# Histograms by region
plot_alpha_histogram_by_region(alpha_df, "Heavy", f"{OUTPUT_DIR}/alpha_histogram_by_region_heavy.png")
plot_alpha_histogram_by_region(alpha_df, "Light", f"{OUTPUT_DIR}/alpha_histogram_by_region_light.png")

# Print summary stats
print("\n=== Summary Statistics ===")
summary = alpha_df.groupby(["chain", "region"])["alpha"].agg(["mean", "std", "median", "count"])
print(summary.to_string())

print("\nDone!")
