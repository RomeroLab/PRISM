# load packages
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from iglm import IgLM

for _p in Path(__file__).resolve().parents:
    if (_p / "utils" / "__init__.py").exists():
        sys.path.insert(0, str(_p))
        break
from utils.scoring_utils import add_iglm_scores_to_dataframe

# ============================================================
# Paths
# ============================================================
data_path = "./data/therasabdab/therasabdab_germline_w_cdr_ranges_w_predictions_and_imgt.csv"
figure_dir = "./data/therasabdab/figures"
os.makedirs(figure_dir, exist_ok=True)

figure_path = os.path.join(
    figure_dir,
    "IgLM_pseudoperplexity_vs_perplexity.png"
)

# ============================================================
# Load data
# ============================================================
df = pd.read_csv(data_path)

# ============================================================
# Keep only rows with string heavy/light sequences for scoring
# ============================================================
valid_mask = (
    df["HeavySequence"].apply(lambda x: isinstance(x, str)) &
    df["LightSequence"].apply(lambda x: isinstance(x, str))
)

n_valid = int(valid_mask.sum())
n_invalid = int((~valid_mask).sum())

print(f"Rows with valid string sequences: {n_valid}")
print(f"Rows skipped (non-string HeavySequence and/or LightSequence): {n_invalid}")

# initialize score columns if they do not already exist
for col in ["IgLM_Perplexity", "IgLM_PseudoPerplexity"]:
    if col not in df.columns:
        df[col] = np.nan

df_valid = df.loc[valid_mask].copy()

# ============================================================
# Initialize model
# ============================================================
iglm = IgLM()

# ============================================================
# Add scores only to valid rows
# ============================================================
df_valid = add_iglm_scores_to_dataframe(
    df=df_valid,
    model=iglm,
    heavy_col="HeavySequence",
    light_col="LightSequence",
    perplexity_col="IgLM_Perplexity",
    pseudo_perplexity_col="IgLM_PseudoPerplexity",
    compute_perplexity=True,
    compute_pseudo_perplexity=True,
    show_progress=True,
)

# write scored values back into full dataframe
df.loc[df_valid.index, "IgLM_Perplexity"] = df_valid["IgLM_Perplexity"]
df.loc[df_valid.index, "IgLM_PseudoPerplexity"] = df_valid["IgLM_PseudoPerplexity"]

# ============================================================
# Save updated dataframe
# ============================================================
df.to_csv(data_path, index=False)

# ============================================================
# Plot IgLM_PseudoPerplexity vs IgLM_Perplexity
# ============================================================
plot_df = df[["IgLM_Perplexity", "IgLM_PseudoPerplexity"]].copy()
plot_df = plot_df.replace([np.inf, -np.inf], np.nan).dropna()

if len(plot_df) < 2:
    raise ValueError("Not enough non-NaN points to compute correlations and plot.")

x = plot_df["IgLM_Perplexity"].values
y = plot_df["IgLM_PseudoPerplexity"].values

pearson_r, pearson_p = pearsonr(x, y)
spearman_rho, spearman_p = spearmanr(x, y)

plt.figure(figsize=(7, 6))
plt.scatter(x, y, alpha=0.7)

# Optional best-fit line
m, b = np.polyfit(x, y, 1)
x_line = np.linspace(x.min(), x.max(), 200)
y_line = m * x_line + b
plt.plot(x_line, y_line, linewidth=2)

annotation_text = (
    f"Pearson r = {pearson_r:.3f} (p = {pearson_p:.2e})\n"
    f"Spearman ρ = {spearman_rho:.3f} (p = {spearman_p:.2e})\n"
    f"N = {len(plot_df)}"
)

plt.text(
    0.05,
    0.95,
    annotation_text,
    transform=plt.gca().transAxes,
    va="top",
    ha="left",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
)

plt.xlabel("IgLM Perplexity")
plt.ylabel("IgLM PseudoPerplexity")
plt.title("IgLM PseudoPerplexity vs. IgLM Perplexity")
plt.tight_layout()
plt.savefig(figure_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved updated dataframe to: {data_path}")
print(f"Saved figure to: {figure_path}")
print(f"Pearson r = {pearson_r:.6f}, p = {pearson_p:.6e}")
print(f"Spearman rho = {spearman_rho:.6f}, p = {spearman_p:.6e}")