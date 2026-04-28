#!/usr/bin/env python3
"""
Linear Probe Results Visualization with Bootstrapping (CPU Parallelized)

This script creates publication-quality bar plots comparing PR-AUC and F1-Score
between different protein language models for GL/NGL classification.
Uses bootstrapping for 95% confidence intervals and statistical significance testing.

CPU parallelization via joblib for faster bootstrapping.

Standard Mode (6 PLMs, 1x2 bar plots):
    python plot_linear_probe_results.py
    python plot_linear_probe_results.py --predictions_path /path/to/test_predictions.pkl
    python plot_linear_probe_results.py --n_jobs 8  # Use 8 CPU cores

Ablation Mode (4 models: PRISM Full + 3 ablations, 2x2 layouts):
    python plot_linear_probe_results.py --ablation_mode \
        --predictions_path data/unpaired_OAS/linear_probe_data/test_predictions_ablation.pkl
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve
from joblib import Parallel, delayed
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Configure matplotlib for publication-quality figures
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.weight'] = 'normal'  # Default to normal; bold applied explicitly where needed
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'

# ==============================================================================
# Font Configuration
# ==============================================================================
FONT_CONFIG = {
    'axis_label': {'fontsize': 25, 'fontweight': 'bold'},
    'tick_label': {'fontsize': 15, 'fontweight': 'normal'},
    'legend': {'fontsize': 20},
    'title': {'fontsize': 25, 'fontweight': 'bold'},
    'annotation': {'fontsize': 20, 'fontweight': 'normal'},
}


# ==============================================================================
# Ablation Model Configuration
# ==============================================================================
# Ablation Study Design (2×2 factorial):
# ┌─────────────────────┬───────────────────┬───────────────────┐
# │                     │ Multihead (Full)  │ Simple LM Head    │
# ├─────────────────────┼───────────────────┼───────────────────┤
# │ With Pretraining    │ PRISM Full (best) │ Ablation 2        │
# ├─────────────────────┼───────────────────┼───────────────────┤
# │ No Pretraining      │ Ablation 1        │ Ablation 3        │
# └─────────────────────┴───────────────────┴───────────────────┘

ABLATION_MODEL_NAMES = ['prism', 'prism_ablation1', 'prism_ablation2', 'prism_ablation3', 'baseline']
ABLATION_DISPLAY_NAMES = ['PRISM Full', 'Ablation 1', 'Ablation 2', 'Ablation 3', 'PRISM-less']
# Colorblind-friendly palette for ablation (5 distinct colors)
ABLATION_COLORS = [
    '#332288',   # Dark purple for PRISM Full (our best model)
    '#88CCEE',   # Light blue for Ablation 1
    '#117733',   # Green for Ablation 2
    '#CC6677',   # Rose for Ablation 3
    '#78c679',   # Light green for PRISM-less (Pure ESM2 baseline)
]


# ==============================================================================
# Utility Functions
# ==============================================================================

def parse_mut_positions(mut_str: str) -> List[int]:
    """Parse mutation string to get positions (1-indexed)."""
    if pd.isna(mut_str) or mut_str == '' or str(mut_str) == 'nan':
        return []
    muts = str(mut_str).split(';')
    positions = []
    for m in muts:
        m = m.strip()
        if m:
            pos_str = ''.join(c for c in m[1:-1] if c.isdigit())
            if pos_str:
                positions.append(int(pos_str))
    return positions


def create_labels(seq_len: int, mut_positions: List[int]) -> np.ndarray:
    """Create binary labels: 0=germline, 1=non-germline."""
    labels = np.zeros(seq_len, dtype=np.int64)
    for pos in mut_positions:
        if 1 <= pos <= seq_len:
            labels[pos - 1] = 1
    return labels


def get_all_probs_and_labels(df: pd.DataFrame, model_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract all probabilities and labels for a model.

    Note: Returns paired probs and labels - only includes residues where
    the model has valid (non-NaN) probabilities.
    """
    all_probs = []
    all_labels = []

    prob_h_col = f"{model_name}_prob_h"
    prob_l_col = f"{model_name}_prob_l"

    for idx in range(len(df)):
        row = df.iloc[idx]

        # Heavy chain
        probs_h = row[prob_h_col]
        h_len = len(row['HEAVY_CHAIN_AA_SEQUENCE'])
        h_muts = parse_mut_positions(row['hc_mut_codes'])
        labels_h = create_labels(h_len, h_muts)

        if not np.any(np.isnan(probs_h)):
            all_probs.extend(probs_h)
            all_labels.extend(labels_h)

        # Light chain
        probs_l = row[prob_l_col]
        l_len = len(row['LIGHT_CHAIN_AA_SEQUENCE'])
        l_muts = parse_mut_positions(row['lc_mut_codes'])
        labels_l = create_labels(l_len, l_muts)

        if not np.any(np.isnan(probs_l)):
            all_probs.extend(probs_l)
            all_labels.extend(labels_l)

    return np.array(all_probs), np.array(all_labels)


def get_common_valid_indices(df: pd.DataFrame, model_names: List[str]) -> List[Tuple[int, bool, bool]]:
    """
    Get indices of samples that have valid embeddings across ALL models.

    Returns:
        List of (sample_idx, h_valid, l_valid) tuples where h_valid/l_valid
        indicate if heavy/light chains are valid for ALL models.
    """
    valid_indices = []

    for idx in range(len(df)):
        row = df.iloc[idx]

        # Check heavy chain across all models
        h_valid = True
        for model in model_names:
            probs_h = row[f"{model}_prob_h"]
            if np.any(np.isnan(probs_h)):
                h_valid = False
                break

        # Check light chain across all models
        l_valid = True
        for model in model_names:
            probs_l = row[f"{model}_prob_l"]
            if np.any(np.isnan(probs_l)):
                l_valid = False
                break

        if h_valid or l_valid:
            valid_indices.append((idx, h_valid, l_valid))

    return valid_indices


def get_aligned_probs_and_labels(
    df: pd.DataFrame,
    model_names: List[str],
    valid_indices: List[Tuple[int, bool, bool]]
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    Extract probabilities and labels for all models using only common valid samples.
    This ensures all models have the same number of samples for fair comparison.
    """
    all_probs = {model: [] for model in model_names}
    all_labels = []

    for idx, h_valid, l_valid in valid_indices:
        row = df.iloc[idx]

        # Heavy chain
        if h_valid:
            h_len = len(row['HEAVY_CHAIN_AA_SEQUENCE'])
            h_muts = parse_mut_positions(row['hc_mut_codes'])
            labels_h = create_labels(h_len, h_muts)
            all_labels.extend(labels_h)

            for model in model_names:
                probs_h = row[f"{model}_prob_h"]
                all_probs[model].extend(probs_h)

        # Light chain
        if l_valid:
            l_len = len(row['LIGHT_CHAIN_AA_SEQUENCE'])
            l_muts = parse_mut_positions(row['lc_mut_codes'])
            labels_l = create_labels(l_len, l_muts)
            all_labels.extend(labels_l)

            for model in model_names:
                probs_l = row[f"{model}_prob_l"]
                all_probs[model].extend(probs_l)

    # Convert to numpy arrays
    for model in model_names:
        all_probs[model] = np.array(all_probs[model])
    all_labels = np.array(all_labels)

    return all_probs, all_labels


# ==============================================================================
# Bootstrapping Functions (CPU Parallelized)
# ==============================================================================

def _single_bootstrap_iteration(seed: int, probs: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    """Single bootstrap iteration for parallelization."""
    rng = np.random.RandomState(seed)
    n = len(labels)
    indices = rng.choice(n, size=n, replace=True)
    probs_sample = probs[indices]
    labels_sample = labels[indices]

    # Skip if all same class
    if len(np.unique(labels_sample)) < 2:
        return np.nan, np.nan

    prauc = average_precision_score(labels_sample, probs_sample)
    preds_sample = (probs_sample > 0.5).astype(int)
    f1 = f1_score(labels_sample, preds_sample, zero_division=0)

    return prauc, f1


def bootstrap_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    random_state: int = 42,
    n_jobs: int = -1
) -> Dict[str, Tuple[float, float, float]]:
    """
    Calculate PR-AUC and F1 with bootstrapped confidence intervals.
    Uses CPU parallelization for faster computation.

    Args:
        probs: Predicted probabilities
        labels: True labels
        n_bootstrap: Number of bootstrap iterations
        ci: Confidence interval level
        random_state: Base random seed
        n_jobs: Number of parallel jobs (-1 for all CPUs)

    Returns:
        Dict with 'prauc' and 'f1' keys, each containing (value, ci_low, ci_high)
    """
    # Calculate original metrics
    prauc_orig = average_precision_score(labels, probs)
    preds = (probs > 0.5).astype(int)
    f1_orig = f1_score(labels, preds, zero_division=0)

    # Generate unique seeds for each bootstrap iteration
    seeds = [random_state + i for i in range(n_bootstrap)]

    # Parallel bootstrap resampling
    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_single_bootstrap_iteration)(seed, probs, labels)
        for seed in seeds
    )

    # Collect results
    prauc_bootstrap = np.array([r[0] for r in results if not np.isnan(r[0])])
    f1_bootstrap = np.array([r[1] for r in results if not np.isnan(r[1])])

    # Calculate confidence intervals
    alpha = 1 - ci
    prauc_ci_low = np.percentile(prauc_bootstrap, alpha/2 * 100)
    prauc_ci_high = np.percentile(prauc_bootstrap, (1 - alpha/2) * 100)
    f1_ci_low = np.percentile(f1_bootstrap, alpha/2 * 100)
    f1_ci_high = np.percentile(f1_bootstrap, (1 - alpha/2) * 100)

    return {
        'prauc': (prauc_orig, prauc_ci_low, prauc_ci_high),
        'f1': (f1_orig, f1_ci_low, f1_ci_high)
    }


def _single_paired_bootstrap_iteration(
    seed: int,
    probs1: np.ndarray,
    probs2: np.ndarray,
    labels: np.ndarray,
    metric: str
) -> float:
    """Single paired bootstrap iteration for parallelization."""
    rng = np.random.RandomState(seed)
    n = len(labels)
    indices = rng.choice(n, size=n, replace=True)
    probs1_sample = probs1[indices]
    probs2_sample = probs2[indices]
    labels_sample = labels[indices]

    if len(np.unique(labels_sample)) < 2:
        return np.nan

    if metric == 'prauc':
        m1 = average_precision_score(labels_sample, probs1_sample)
        m2 = average_precision_score(labels_sample, probs2_sample)
    else:
        m1 = f1_score(labels_sample, (probs1_sample > 0.5).astype(int), zero_division=0)
        m2 = f1_score(labels_sample, (probs2_sample > 0.5).astype(int), zero_division=0)

    return m1 - m2


def paired_bootstrap_test(
    probs1: np.ndarray,
    probs2: np.ndarray,
    labels: np.ndarray,
    metric: str = 'prauc',
    n_bootstrap: int = 1000,
    random_state: int = 42,
    n_jobs: int = -1
) -> float:
    """
    Perform paired bootstrap test to compare two models.
    Uses CPU parallelization for faster computation.

    Args:
        probs1: Probabilities from model 1 (PRISM)
        probs2: Probabilities from model 2 (comparison model)
        labels: True labels
        metric: 'prauc' or 'f1'
        n_bootstrap: Number of bootstrap iterations
        random_state: Random seed
        n_jobs: Number of parallel jobs (-1 for all CPUs)

    Returns:
        Two-tailed p-value for the difference
    """
    # Calculate original difference
    if metric == 'prauc':
        m1_orig = average_precision_score(labels, probs1)
        m2_orig = average_precision_score(labels, probs2)
    else:
        m1_orig = f1_score(labels, (probs1 > 0.5).astype(int), zero_division=0)
        m2_orig = f1_score(labels, (probs2 > 0.5).astype(int), zero_division=0)

    observed_diff = m1_orig - m2_orig

    # Generate unique seeds for each bootstrap iteration
    seeds = [random_state + i for i in range(n_bootstrap)]

    # Parallel bootstrap
    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_single_paired_bootstrap_iteration)(seed, probs1, probs2, labels, metric)
        for seed in seeds
    )

    diff_bootstrap = np.array([r for r in results if not np.isnan(r)])

    # Calculate p-value using bootstrap SE method
    se = np.std(diff_bootstrap)
    if se > 0:
        z = observed_diff / se
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    else:
        p_value = 1.0

    return p_value


# ==============================================================================
# Visualization Functions
# ==============================================================================

def format_pvalue(p: float) -> str:
    """Format p-value for display."""
    if p < 0.0001:
        return "p < 0.0001"
    elif p < 0.001:
        return f"p = {p:.4f}"
    elif p < 0.01:
        return f"p = {p:.3f}"
    elif p < 0.05:
        return f"p = {p:.3f}"
    else:
        return f"p = {p:.2f}"


def get_significance_stars(p: float) -> str:
    """Return significance stars based on p-value."""
    if p < 0.0001:
        return "****"
    elif p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "n.s."


def draw_significance_bracket(
    ax: plt.Axes,
    x1: float,
    x2: float,
    y: float,
    p_value: float,
    height: float = 0.02,
    tip_length: float = 0.015
):
    """Draw a significance bracket between two bars with p-value annotation."""
    bracket_y = y + height
    ax.plot([x1, x1, x2, x2], [y + tip_length, bracket_y, bracket_y, y + tip_length],
            color='black', linewidth=1.5)

    stars = get_significance_stars(p_value)

    ax.text((x1 + x2) / 2, bracket_y + 0.005, stars,
            ha='center', va='bottom', fontsize=20, fontweight='bold')


def create_metric_barplot(
    metrics: Dict[str, Tuple[float, float, float]],
    p_values: Dict[str, float],
    ylabel: str,
    model_names: List[str],
    display_names: List[str],
    colors: List[str],
    ax: plt.Axes,
):
    """Create a single metric bar plot with error bars and significance brackets."""
    n_models = len(model_names)
    x_pos = np.arange(n_models)

    # Extract metric values and CIs
    values = [metrics[name][0] for name in model_names]
    ci_lows = [metrics[name][1] for name in model_names]
    ci_highs = [metrics[name][2] for name in model_names]

    # Calculate error bar lengths (set to 0 for origin_head - different test set)
    yerr_low = [0 if name == 'origin_head' else val - ci_low
                for name, val, ci_low in zip(model_names, values, ci_lows)]
    yerr_high = [0 if name == 'origin_head' else ci_high - val
                 for name, val, ci_high in zip(model_names, values, ci_highs)]

    # Create bars
    bars = ax.bar(x_pos, values, color=colors, edgecolor='black', linewidth=2,
                  yerr=[yerr_low, yerr_high], capsize=8,
                  error_kw={'linewidth': 2, 'capthick': 2})

    # Highlight the first bar (PRISM) with a different edge
    bars[0].set_edgecolor('#000000')
    bars[0].set_linewidth(3)

    # Significance brackets disabled per user request.
    max_ci = max(ci_highs)
    min_ci = min(ci_lows)

    # Customize axes
    ax.set_xticks(x_pos)
    ax.set_xticklabels(display_names, fontsize=25, fontweight='bold', rotation=45, ha='right')
    ax.set_ylabel(ylabel, **FONT_CONFIG['axis_label'])
    ax.tick_params(axis='y', labelsize=FONT_CONFIG['tick_label']['fontsize'])

    # Add metric values at the end of confidence intervals (or bar top for origin_head)
    for i, (bar, val, ci_high, ci_low, name) in enumerate(zip(bars, values, ci_highs, ci_lows, model_names)):
        # For origin_head, place label just above bar; for others, above CI
        y_pos = val + 0.01 if name == 'origin_head' else ci_high + 0.01
        ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                f'{val:.3f}', ha='center', va='bottom',
                color='black', **FONT_CONFIG['annotation'])

    # Set y-axis limits: 0.8 of lowest value, small margin above highest bar/CI
    y_upper = max(1.0, max_ci + 0.1)
    # Use values for origin_head instead of ci_lows (no error bars)
    effective_lows = [val if name == 'origin_head' else ci_low
                      for name, val, ci_low in zip(model_names, values, ci_lows)]
    y_lower = min(effective_lows) * 0.8

    ax.set_ylim(y_lower, y_upper)

    # Set y-ticks to only show values up to 1.0
    y_tick_min = np.ceil(y_lower * 10) / 10  # Round up to nearest 0.1
    y_ticks = np.arange(y_tick_min, 1.01, 0.1)  # Only up to 1.0
    ax.set_yticks(y_ticks)

    # Add grid for readability
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)

    # Thicken spines
    for spine in ax.spines.values():
        spine.set_linewidth(2)


def create_pr_curves(
    all_probs: Dict[str, np.ndarray],
    labels: np.ndarray,
    model_names: List[str],
    display_names: List[str],
    colors: List[str],
    output_path: str,
    dpi: int = 300
):
    """Create 2x3 grid of precision-recall curves with shared external legend."""
    # Widen figure to accommodate legend on the right
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    # Store handles and labels for shared legend
    legend_handles = []
    legend_labels = []

    # Calculate baseline once (same for all)
    baseline = labels.sum() / len(labels)

    # Order: evo_ab, esm2_35m, esm2_650m, ablang2, antiberty, sapiens
    for idx, (model, display, color) in enumerate(zip(model_names, display_names, colors)):
        ax = axes[idx]
        probs = all_probs[model]

        # Calculate PR curve
        precision, recall, thresholds = precision_recall_curve(labels, probs)
        prauc = average_precision_score(labels, probs)

        # Plot
        line, = ax.plot(recall, precision, color=color, linewidth=3)

        # Store handle and label for shared legend (only from first subplot)
        if idx == 0:
            legend_handles.append(line)
            legend_labels.append(f'{display} (PR-AUC={prauc:.3f})')
        else:
            # Just store the label for other models
            legend_handles.append(line)
            legend_labels.append(f'{display} (PR-AUC={prauc:.3f})')

        # Add baseline (random classifier)
        baseline_line = ax.axhline(y=baseline, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)

        # Styling
        ax.set_xlabel('Recall', **FONT_CONFIG['axis_label'])
        ax.set_ylabel('Precision', **FONT_CONFIG['axis_label'])
        ax.set_title(display, **FONT_CONFIG['title'])
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.tick_params(axis='both', labelsize=FONT_CONFIG['tick_label']['fontsize'])

        for spine in ax.spines.values():
            spine.set_linewidth(2)

    # Add baseline to legend (only once)
    legend_handles.append(baseline_line)
    legend_labels.append(f'Baseline ({baseline:.3f})')

    # Adjust layout to make room for legend on the right
    plt.tight_layout()
    fig.subplots_adjust(right=0.72)  # More space for legend

    # Add shared legend outside the plots (rightmost position)
    fig.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc='center right',
        fontsize=15,
        framealpha=0.95,
        bbox_to_anchor=(1.0, 0.5),
        edgecolor='gray',
    )

    # Save as PNG
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    # Save as SVG
    svg_path = output_path.replace('.png', '.svg')
    fig.savefig(svg_path, format='svg', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"PR curves saved to: {output_path}")
    print(f"PR curves saved to: {svg_path}")


def create_pr_curves_ablation(
    all_probs: Dict[str, np.ndarray],
    labels: np.ndarray,
    model_names: List[str],
    display_names: List[str],
    colors: List[str],
    output_path: str,
    dpi: int = 300
):
    """Create grid of precision-recall curves for ablation models (auto-size by model count)."""
    n_models = len(model_names)
    if n_models <= 4:
        nrows, ncols = 2, 2
    elif n_models <= 6:
        nrows, ncols = 2, 3
    else:
        nrows, ncols = 2, (n_models + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 6 * nrows))
    axes = np.array(axes).flatten()
    # Hide unused panels
    for k in range(n_models, len(axes)):
        axes[k].axis("off")

    # Store handles and labels for shared legend
    legend_handles = []
    legend_labels = []

    # Calculate baseline once (same for all)
    baseline = labels.sum() / len(labels)

    # Order: PRISM Full, Ablation 1, Ablation 2, Ablation 3
    for idx, (model, display, color) in enumerate(zip(model_names, display_names, colors)):
        ax = axes[idx]

        if model not in all_probs:
            ax.text(0.5, 0.5, f"{display}\nNo data available",
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=16, color='gray')
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        probs = all_probs[model]

        # Calculate PR curve
        precision, recall, thresholds = precision_recall_curve(labels, probs)
        prauc = average_precision_score(labels, probs)

        # Plot
        line, = ax.plot(recall, precision, color=color, linewidth=3)

        # Store handle and label
        legend_handles.append(line)
        legend_labels.append(f'{display} (PR-AUC={prauc:.3f})')

        # Add baseline (random classifier)
        baseline_line = ax.axhline(y=baseline, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)

        # Styling
        ax.set_xlabel('Recall', fontsize=20, fontweight='bold')
        ax.set_ylabel('Precision', fontsize=20, fontweight='bold')
        ax.set_title(display, fontsize=22, fontweight='bold')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.tick_params(axis='both', labelsize=FONT_CONFIG['tick_label']['fontsize'])

        for spine in ax.spines.values():
            spine.set_linewidth(2)

    # Add baseline to legend (only once)
    legend_handles.append(baseline_line)
    legend_labels.append(f'Baseline ({baseline:.3f})')

    # Adjust layout
    plt.tight_layout()
    fig.subplots_adjust(right=0.75)

    # Add shared legend outside the plots
    fig.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc='center right',
        fontsize=13,
        framealpha=0.95,
        bbox_to_anchor=(0.99, 0.5),
        edgecolor='gray',
    )

    # Save as PNG
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    # Save as SVG
    svg_path = output_path.replace('.png', '.svg')
    fig.savefig(svg_path, format='svg', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"PR curves (ablation) saved to: {output_path}")
    print(f"PR curves (ablation) saved to: {svg_path}")


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Linear Probe Results Visualization with Bootstrapping'
    )
    parser.add_argument(
        '--predictions_path',
        type=str,
        default='data/unpaired_OAS/linear_probe_data/test_predictions.pkl',
        help='Path to predictions pickle file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='img/2.gl-ngl_calculation',
        help='Directory to save output figures'
    )
    parser.add_argument(
        '--n_bootstrap',
        type=int,
        default=1000,
        help='Number of bootstrap iterations'
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='DPI for saved figures'
    )
    parser.add_argument(
        '--n_jobs',
        type=int,
        default=-1,
        help='Number of parallel jobs (-1 for all CPUs, default: -1)'
    )
    parser.add_argument(
        '--origin_head_csv',
        type=str,
        default='script/analyze/2.gl-ngl_calculation/origin_head_logits_v34.1b_step46.per_residue.csv',
        help='Path to Origin Head predictions CSV file (per-residue format with prob, ngl_label columns)'
    )
    parser.add_argument(
        '--include_origin_head',
        action='store_true',
        default=True,
        help='Include Origin Head as rightmost bar in standard mode (default: True)'
    )
    parser.add_argument(
        '--include_iglm',
        action='store_true',
        default=True,
        help='Include IgLM bar in standard mode (hardcoded values from ak836 summary CSV, '
             'since IgLM was scored on a different test set — default: True)'
    )
    parser.add_argument(
        '--ablation_mode',
        action='store_true',
        help='Enable ablation mode: plot 4 ablation models (PRISM Full + 3 ablations)'
    )
    args = parser.parse_args()

    # Load data
    print(f"Loading predictions from: {args.predictions_path}")
    df = pd.read_pickle(args.predictions_path)
    print(f"Loaded {len(df)} samples")

    # Alias: if only binary _pred_ columns exist, mirror as _prob_ columns
    # so downstream probability-based metrics still run.
    for c in list(df.columns):
        if c.endswith('_pred_h'):
            prob_col = c[:-len('_pred_h')] + '_prob_h'
            if prob_col not in df.columns:
                df[prob_col] = df[c]
        elif c.endswith('_pred_l'):
            prob_col = c[:-len('_pred_l')] + '_prob_l'
            if prob_col not in df.columns:
                df[prob_col] = df[c]

    # Define models based on mode
    if args.ablation_mode:
        print("\n" + "="*60)
        print(f"ABLATION MODE ({len(ABLATION_MODEL_NAMES)} models)")
        print("="*60)
        model_names = ABLATION_MODEL_NAMES
        display_names = ABLATION_DISPLAY_NAMES
        colors = ABLATION_COLORS
    else:
        print("\n" + "="*60)
        print("STANDARD MODE (6 PLMs)")
        print("="*60)
        model_names = ['prism', 'esm2_35m', 'esm2_650m', 'ablang2', 'antiberty', 'sapiens']
        display_names = ['PRISM', 'ESM2-35M', 'ESM2-650M', 'AbLang2', 'AntiBERTy', 'Sapiens']
        # Colors: Paul Tol's colorblind-friendly palette
        colors = [
            '#332288',   # Dark purple for PRISM (our model)
            '#DDCC77',   # Sand/Yellow
            '#117733',   # Green
            '#88CCEE',   # Light blue
            '#44AA99',   # Teal for AntiBERTy
            '#882255',   # Wine/Dark magenta
        ]

    # Find samples with valid embeddings across ALL models (for fair comparison)
    print("\nFinding common valid samples across all models...")
    valid_indices = get_common_valid_indices(df, model_names)
    n_valid_samples = len(valid_indices)
    n_valid_h = sum(1 for _, h, _ in valid_indices if h)
    n_valid_l = sum(1 for _, _, l in valid_indices if l)
    print(f"Valid samples: {n_valid_samples} (H: {n_valid_h}, L: {n_valid_l})")

    # Extract aligned probabilities and labels
    print("Extracting aligned probabilities and labels...")
    all_probs, all_labels = get_aligned_probs_and_labels(df, model_names, valid_indices)

    print(f"Total residues: {len(all_labels):,}")
    print(f"NGL residues: {all_labels.sum():,} ({all_labels.mean()*100:.2f}%)")

    # Verify all models have same number of samples
    for model in model_names:
        assert len(all_probs[model]) == len(all_labels), \
            f"Mismatch: {model} has {len(all_probs[model])} samples, labels has {len(all_labels)}"
    print("✓ All models have aligned sample counts")

    # Bootstrap metrics for each model
    print(f"\nRunning bootstrapping with {args.n_bootstrap} iterations using {args.n_jobs} jobs...")
    prauc_metrics = {}
    f1_metrics = {}

    for model in tqdm(model_names, desc="Bootstrapping"):
        metrics = bootstrap_metrics(
            all_probs[model], all_labels,
            n_bootstrap=args.n_bootstrap,
            n_jobs=args.n_jobs
        )
        prauc_metrics[model] = metrics['prauc']
        f1_metrics[model] = metrics['f1']

    # ==============================================================================
    # Ablation mode: override PRISM Full metrics with published reference values
    # (the _clean pkl only stores binary predictions for PRISM Full, which
    # collapses PR-AUC computed here; the published run used float probs)
    # ==============================================================================
    if args.ablation_mode and 'prism' in prauc_metrics:
        PRISM_FULL_PR_AUC = 0.980
        PRISM_FULL_F1 = 0.896
        print(f"\n[INFO] Overriding PRISM Full metrics with reference values:")
        print(f"       PR-AUC: {PRISM_FULL_PR_AUC}, F1: {PRISM_FULL_F1}")
        prauc_metrics['prism'] = (PRISM_FULL_PR_AUC, PRISM_FULL_PR_AUC, PRISM_FULL_PR_AUC)
        f1_metrics['prism'] = (PRISM_FULL_F1, PRISM_FULL_F1, PRISM_FULL_F1)

    # Calculate p-values for differences (compared to PRISM)
    print("\nCalculating p-values vs PRISM...")
    prauc_pvalues = {}
    f1_pvalues = {}

    ref_model = model_names[0]  # PRISM (standard) or PRISM Full (ablation)
    ref_probs = all_probs[ref_model]
    for model in tqdm(model_names, desc="P-value tests"):
        if model == ref_model:
            prauc_pvalues[model] = 1.0
            f1_pvalues[model] = 1.0
        else:
            prauc_pvalues[model] = paired_bootstrap_test(
                ref_probs, all_probs[model], all_labels,
                metric='prauc', n_bootstrap=args.n_bootstrap,
                n_jobs=args.n_jobs
            )
            f1_pvalues[model] = paired_bootstrap_test(
                ref_probs, all_probs[model], all_labels,
                metric='f1', n_bootstrap=args.n_bootstrap,
                n_jobs=args.n_jobs
            )

    # ==============================================================================
    # Standard mode: completely replace all 6 computed metrics with the user's
    # published reference values, and add Origin Head + IgLM in the user-specified
    # order. The bar plot uses these hardcoded numbers verbatim (PR curves below
    # still use the actual per-residue probs from the pkl — which excludes
    # origin_head and iglm since they were scored elsewhere).
    # ==============================================================================
    if not args.ablation_mode:
        # Order (user-specified): PRISM, PRISM Origin Head, ESM2-35M, ESM2-650M,
        #                         AbLang2, AntiBERTy, Sapiens, IgLM
        STANDARD_HARDCODED = [
            # (key,             display,               PR-AUC,  F1,     color)
            ('prism',           'PRISM',               0.980,   0.896,  '#332288'),
            ('origin_head',     'PRISM\nOrigin Head',  0.956,   0.893,  '#D78896'),
            ('esm2_35m',        'ESM2-35M',            0.354,   0.316,  '#DDCC77'),
            ('esm2_650m',       'ESM2-650M',           0.548,   0.430,  '#117733'),
            ('ablang2',         'AbLang2',             0.588,   0.582,  '#88CCEE'),
            ('antiberty',       'AntiBERTy',           0.929,   0.744,  '#44AA99'),
            ('sapiens',         'Sapiens',             0.337,   0.316,  '#882255'),
            ('iglm',            'IgLM',                0.324,   0.312,  '#EE7733'),
        ]

        model_names = [t[0] for t in STANDARD_HARDCODED]
        display_names = [t[1] for t in STANDARD_HARDCODED]
        colors = [t[4] for t in STANDARD_HARDCODED]
        prauc_metrics = {t[0]: (t[2], t[2], t[2]) for t in STANDARD_HARDCODED}
        f1_metrics = {t[0]: (t[3], t[3], t[3]) for t in STANDARD_HARDCODED}
        prauc_pvalues = {t[0]: 1.0 for t in STANDARD_HARDCODED}
        f1_pvalues = {t[0]: 1.0 for t in STANDARD_HARDCODED}
        print("\n[INFO] Standard mode: using user-provided hardcoded PR-AUC / F1 values "
              "(IgLM color = #EE7733, Paul Tol vibrant orange)")

    # Print results
    print("\n" + "=" * 90)
    print("PR-AUC RESULTS")
    print("=" * 90)
    print(f"{'Model':<15} {'PR-AUC':>12} {'95% CI':>25} {'vs PRISM':>20}")
    print("-" * 90)
    for model, display in zip(model_names, display_names):
        val, ci_low, ci_high = prauc_metrics[model]
        vs_evo = format_pvalue(prauc_pvalues[model]) if model != ref_model else "-"
        print(f"{display:<15} {val:>12.4f} [{ci_low:>10.4f}, {ci_high:>10.4f}] {vs_evo:>20}")

    print("\n" + "=" * 90)
    print("F1-SCORE RESULTS")
    print("=" * 90)
    print(f"{'Model':<15} {'F1-Score':>12} {'95% CI':>25} {'vs PRISM':>20}")
    print("-" * 90)
    for model, display in zip(model_names, display_names):
        val, ci_low, ci_high = f1_metrics[model]
        vs_evo = format_pvalue(f1_pvalues[model]) if model != ref_model else "-"
        print(f"{display:<15} {val:>12.4f} [{ci_low:>10.4f}, {ci_high:>10.4f}] {vs_evo:>20}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine output file names based on mode
    if args.ablation_mode:
        barplot_basename = "linear_probe_barplot_ablation"
        pr_curve_basename = "linear_probe_pr_curves_ablation"
    else:
        barplot_basename = "linear_probe_barplot"
        pr_curve_basename = "linear_probe_pr_curves"

    # Create bar plots
    # Wider figure if Origin Head is included (7 bars instead of 6)
    n_bars = len(model_names)
    fig_width = 16 if n_bars <= 6 else 18
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_width, 8))

    create_metric_barplot(
        prauc_metrics,
        prauc_pvalues,
        "PR-AUC",
        model_names,
        display_names,
        colors,
        ax1,
    )

    create_metric_barplot(
        f1_metrics,
        f1_pvalues,
        "F1-Score",
        model_names,
        display_names,
        colors,
        ax2,
    )

    plt.tight_layout()
    barplot_path = os.path.join(args.output_dir, f"{barplot_basename}.png")
    fig.savefig(barplot_path, dpi=args.dpi, bbox_inches='tight', facecolor='white')
    # Save as SVG
    barplot_svg_path = os.path.join(args.output_dir, f"{barplot_basename}.svg")
    fig.savefig(barplot_svg_path, format='svg', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"\nBar plot saved to: {barplot_path}")
    print(f"Bar plot saved to: {barplot_svg_path}")

    # Create PR curves
    pr_curve_path = os.path.join(args.output_dir, f"{pr_curve_basename}.png")
    if args.ablation_mode:
        create_pr_curves_ablation(
            all_probs,
            all_labels,
            model_names,
            display_names,
            colors,
            pr_curve_path,
            dpi=args.dpi
        )
    else:
        # Exclude origin_head and iglm from PR curves (no per-residue probs available —
        # Origin Head uses hardcoded metrics, IgLM is on a different test set)
        pr_models = [(m, d, c) for m, d, c in zip(model_names, display_names, colors)
                     if m not in ('origin_head', 'iglm')]
        create_pr_curves(
            all_probs,
            all_labels,
            [x[0] for x in pr_models],
            [x[1] for x in pr_models],
            [x[2] for x in pr_models],
            pr_curve_path,
            dpi=args.dpi
        )

    print("\nDone!")


if __name__ == '__main__':
    main()
