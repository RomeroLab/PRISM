#!/usr/bin/env python3
"""
Developability Correlation Analysis with Bootstrapping

This script creates publication-quality bar plots comparing Spearman and Pearson
correlations between protein language model PPL scores and experimental developability metrics.

Key Features:
1. Bootstrapping (1000 iterations) for 95% confidence intervals
2. Paired bootstrap test for statistical significance vs PRISM
3. Direction correction: multiply PPL by -1 for metrics where higher = better
4. Generates 5 figures for different developability properties

Properties analyzed:
1. HIC (Hydrophobicity) - no multiplication
2. PR_CHO (Polyreactivity) - no multiplication
3. AC-SINS (Self-Interaction/Aggregation) - no multiplication
4. Tm1 (Thermal Stability) - multiply -1 (lower PPL = higher Tm1)
5. Titer (Expression) - multiply -1 (lower PPL = higher Titer)

Why multiply by -1?
- PPL: Lower = better (more natural sequence)
- Tm1/Titer: Higher = better
- Without -1, we'd get negative correlations (lower PPL → higher Tm1)
- With -1, we get positive correlations that are easier to interpret

Usage:
    python plot_developability_correlations.py \
        --csv_path data/ginkgo/developability_data_with_ppl.csv \
        --output_dir img/developability

Author: DevAnt-LM Team
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Configure matplotlib for publication-quality figures
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'

# =============================================================================
# Font Configuration
# =============================================================================
FONT_CONFIG = {
    'axis_label': {'fontsize': 25, 'fontweight': 'bold'},
    'tick_label': {'fontsize': 15, 'fontweight': 'normal'},
    'legend': {'fontsize': 20},
    'title': {'fontsize': 25, 'fontweight': 'bold'},
    'annotation': {'fontsize': 15, 'fontweight': 'normal'},
}


# =============================================================================
# Bootstrapping Functions (from plot_correlation_comparison.py)
# =============================================================================

def bootstrap_correlation(
    x: np.ndarray,
    y: np.ndarray,
    method: str = 'spearman',
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    random_state: int = 42
) -> Tuple[float, float, float, float]:
    """
    Calculate correlation with bootstrapped confidence intervals.

    Args:
        x: First variable array (PPL values)
        y: Second variable array (target property)
        method: 'spearman' or 'pearson'
        n_bootstrap: Number of bootstrap iterations
        ci: Confidence interval level (default 0.95 for 95% CI)
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (correlation, ci_lower, ci_upper, p_value)
    """
    np.random.seed(random_state)

    # Remove NaN and inf values
    mask = ~(np.isnan(x) | np.isnan(y) | np.isinf(x) | np.isinf(y))
    x_clean = x[mask]
    y_clean = y[mask]
    n = len(x_clean)

    if n < 3:
        return np.nan, np.nan, np.nan, np.nan

    # Calculate original correlation and p-value
    if method == 'spearman':
        corr, p_value = stats.spearmanr(x_clean, y_clean)
    else:
        corr, p_value = stats.pearsonr(x_clean, y_clean)

    # Bootstrap resampling
    bootstrap_corrs = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(n, size=n, replace=True)
        x_sample = x_clean[indices]
        y_sample = y_clean[indices]

        if method == 'spearman':
            r, _ = stats.spearmanr(x_sample, y_sample)
        else:
            r, _ = stats.pearsonr(x_sample, y_sample)

        if not np.isnan(r):
            bootstrap_corrs.append(r)

    bootstrap_corrs = np.array(bootstrap_corrs)

    # Calculate confidence intervals
    alpha = 1 - ci
    ci_lower = np.percentile(bootstrap_corrs, alpha/2 * 100)
    ci_upper = np.percentile(bootstrap_corrs, (1 - alpha/2) * 100)

    return corr, ci_lower, ci_upper, p_value


def paired_bootstrap_test(
    x1: np.ndarray,
    x2: np.ndarray,
    y: np.ndarray,
    method: str = 'spearman',
    n_bootstrap: int = 1000,
    random_state: int = 42
) -> float:
    """
    Perform paired bootstrap test to compare two correlations.

    Tests whether correlation(x1, y) is significantly different from correlation(x2, y).

    Args:
        x1: First predictor (PRISM PPL)
        x2: Second predictor (comparison model PPL)
        y: Target variable (developability metric)
        method: 'spearman' or 'pearson'
        n_bootstrap: Number of bootstrap iterations
        random_state: Random seed

    Returns:
        Two-tailed p-value for the difference
    """
    np.random.seed(random_state)

    # Remove NaN/inf values
    mask = ~(np.isnan(x1) | np.isnan(x2) | np.isnan(y) |
             np.isinf(x1) | np.isinf(x2) | np.isinf(y))
    x1_clean = x1[mask]
    x2_clean = x2[mask]
    y_clean = y[mask]
    n = len(y_clean)

    if n < 3:
        return np.nan

    # Calculate original difference
    if method == 'spearman':
        r1_orig, _ = stats.spearmanr(x1_clean, y_clean)
        r2_orig, _ = stats.spearmanr(x2_clean, y_clean)
    else:
        r1_orig, _ = stats.pearsonr(x1_clean, y_clean)
        r2_orig, _ = stats.pearsonr(x2_clean, y_clean)

    observed_diff = r1_orig - r2_orig

    # Bootstrap
    diff_bootstrap = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(n, size=n, replace=True)
        x1_sample = x1_clean[indices]
        x2_sample = x2_clean[indices]
        y_sample = y_clean[indices]

        if method == 'spearman':
            r1, _ = stats.spearmanr(x1_sample, y_sample)
            r2, _ = stats.spearmanr(x2_sample, y_sample)
        else:
            r1, _ = stats.pearsonr(x1_sample, y_sample)
            r2, _ = stats.pearsonr(x2_sample, y_sample)

        if not (np.isnan(r1) or np.isnan(r2)):
            diff_bootstrap.append(r1 - r2)

    diff_bootstrap = np.array(diff_bootstrap)

    # Calculate p-value using bootstrap SE method
    se = np.std(diff_bootstrap)
    if se > 0:
        z = observed_diff / se
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    else:
        p_value = 1.0

    return p_value


def format_pvalue(p: float) -> str:
    """Format p-value for display."""
    if np.isnan(p):
        return "N/A"
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
    if np.isnan(p):
        return ""
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
    """
    Draw a significance bracket between two bars with p-value annotation.
    """
    if np.isnan(p_value):
        return

    bracket_y = y + height
    ax.plot([x1, x1, x2, x2], [y + tip_length, bracket_y, bracket_y, y + tip_length],
            color='black', linewidth=1.5)

    stars = get_significance_stars(p_value)

    ax.text((x1 + x2) / 2, bracket_y + 0.005, stars,
            ha='center', va='bottom', fontsize=20, fontweight='bold')


def create_correlation_barplot(
    correlations: Dict[str, Tuple[float, float, float, float]],
    p_values: Dict[str, float],
    ylabel: str,
    model_names: List[str],
    display_names: List[str],
    colors: List[str],
    ax: plt.Axes,
    title: Optional[str] = None,
    n_prism_variants: int = 1
):
    """
    Create a single correlation bar plot with error bars and significance brackets.

    Args:
        n_prism_variants: Number of PRISM variants (first N models)
    """
    n_models = len(model_names)
    x_pos = np.arange(n_models)

    # Extract correlation values and CIs
    corr_values = []
    ci_lows = []
    ci_highs = []

    for name in model_names:
        if name in correlations:
            corr, ci_low, ci_high, _ = correlations[name]
            corr_values.append(corr if not np.isnan(corr) else 0)
            ci_lows.append(ci_low if not np.isnan(ci_low) else corr)
            ci_highs.append(ci_high if not np.isnan(ci_high) else corr)
        else:
            corr_values.append(0)
            ci_lows.append(0)
            ci_highs.append(0)

    # Calculate error bar lengths
    yerr_low = [max(0, corr - ci_low) for corr, ci_low in zip(corr_values, ci_lows)]
    yerr_high = [max(0, ci_high - corr) for corr, ci_high in zip(corr_values, ci_highs)]

    # Create bars
    bars = ax.bar(x_pos, corr_values, color=colors, edgecolor='black', linewidth=2,
                  yerr=[yerr_low, yerr_high], capsize=8,
                  error_kw={'linewidth': 2, 'capthick': 2})

    # Find the best PRISM variant (highest correlation value)
    prism_corrs = corr_values[:n_prism_variants]
    best_prism_idx = prism_corrs.index(max(prism_corrs)) if prism_corrs else 0
    best_prism_model = model_names[best_prism_idx] if model_names else None

    # Highlight the best PRISM bar with a thicker edge
    if best_prism_idx < len(bars):
        bars[best_prism_idx].set_edgecolor('#000000')
        bars[best_prism_idx].set_linewidth(3)

    # Add significance brackets between best PRISM and baseline models only
    max_ci = max(ci_highs) if ci_highs else 0
    min_ci = min(ci_lows) if ci_lows else 0
    bracket_start = max_ci + 0.10  # Increased to avoid overlap with values
    bracket_interval = 0.12  # Interval between brackets

    # Only compare best PRISM vs baseline models (not other PRISM variants)
    baseline_models = [(model, model_names.index(model)) for model in model_names[n_prism_variants:]
                       if model in p_values]
    # Sort by span (distance from best_prism_idx), shorter spans first
    baseline_models_sorted = sorted(baseline_models, key=lambda x: abs(x[1] - best_prism_idx))

    for bracket_idx, (model, model_idx) in enumerate(baseline_models_sorted):
        p_val = p_values[model]
        bracket_y = bracket_start + bracket_idx * bracket_interval
        draw_significance_bracket(ax, best_prism_idx, model_idx, bracket_y, p_val)

    n_brackets = len(baseline_models_sorted)
    max_bracket_y = bracket_start + max(0, n_brackets - 1) * bracket_interval

    # Customize axes
    ax.set_xticks(x_pos)
    ax.set_xticklabels(display_names, fontsize=25, fontweight='bold', rotation=45, ha='right')
    ax.set_ylabel(ylabel, **FONT_CONFIG['axis_label'])

    if title:
        ax.set_title(title, **FONT_CONFIG['title'], pad=10)

    # Add correlation values
    # Place annotations to avoid overlap with bars and x-axis labels
    for i, (bar, corr, ci_high, ci_low) in enumerate(zip(bars, corr_values, ci_highs, ci_lows)):
        if corr < 0:
            # Negative value: check if CI extends above 0
            if ci_high > 0:
                # CI extends above 0: place text above CI
                y_pos = ci_high + 0.02
            else:
                # CI is entirely below 0: place text at 0.0
                y_pos = 0.02
        else:
            # Positive value: place text above the upper CI
            y_pos = ci_high + 0.02
        ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                f'{corr:.3f}', ha='center', va='bottom',
                **FONT_CONFIG['annotation'], color='black')

    # Set y-axis limits
    y_upper = max(0.5, max_bracket_y + 0.12)
    # Calculate y_lower based on the lowest value's sign
    if min_ci < 0:
        y_lower = 1.3 * min_ci  # More negative padding for negative values (avoid label overlap)
    else:
        y_lower = 0.9 * min_ci  # Slightly lower padding for positive values
    ax.set_ylim(y_lower, y_upper)

    # Hide y-tick labels above 1.0 (correlations are bounded by [-1, 1])
    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins='auto'))
    yticks = ax.get_yticks()
    ax.set_yticks([t for t in yticks if t <= 1.0])

    # Set y-tick label font
    ax.tick_params(axis='y', labelsize=FONT_CONFIG['tick_label']['fontsize'])
    for label in ax.get_yticklabels():
        label.set_fontweight(FONT_CONFIG['tick_label']['fontweight'])

    # Add grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)

    # Thicken spines
    for spine in ax.spines.values():
        spine.set_linewidth(2)


# =============================================================================
# Property Configuration
# =============================================================================

PROPERTY_CONFIG = {
    'hydrophobicity': {
        'column': 'HIC',
        'multiply_by_neg1': False,
        'display_name': 'Hydrophobicity (HIC)',
        'filename_suffix': '_hydrophobicity.png'
    },
    'reactivity': {
        'column': 'PR_CHO',
        'multiply_by_neg1': False,
        'display_name': 'Polyreactivity (PR_CHO)',
        'filename_suffix': '_reactivity.png'
    },
    'aggregation': {
        'column': 'AC-SINS_pH7.4',
        'multiply_by_neg1': False,
        'display_name': 'Self-Interaction (AC-SINS)',
        'filename_suffix': '_aggregation.png'
    },
    'thermalstability': {
        'column': 'Tm2',
        'multiply_by_neg1': True,
        'display_name': 'Thermal Stability (Tm2)',
        'filename_suffix': '_thermalstability.png'
    },
    'expression': {
        'column': 'Titer',
        'multiply_by_neg1': True,
        'display_name': 'Expression (Titer)',
        'filename_suffix': '_expression.png'
    }
}


def analyze_property(
    df: pd.DataFrame,
    property_config: dict,
    model_columns: List[str],
    display_names: List[str],
    n_bootstrap: int = 1000,
    n_prism_variants: int = 1
) -> Tuple[Dict, Dict, Dict, Dict]:
    """
    Analyze correlations for a single developability property.

    Args:
        df: DataFrame with PPL values and property columns
        property_config: Configuration for the property
        model_columns: List of PPL column names
        display_names: Display names for models
        n_bootstrap: Number of bootstrap iterations
        n_prism_variants: Number of PRISM variants (first N models)

    Returns:
        Tuple of (spearman_correlations, pearson_correlations, spearman_pvalues, pearson_pvalues)
    """
    target_col = property_config['column']
    multiply_neg1 = property_config['multiply_by_neg1']

    if target_col not in df.columns:
        print(f"  WARNING: Column '{target_col}' not found in data")
        return {}, {}, {}, {}

    # Get target values
    y = df[target_col].values

    # If multiply_by_neg1, we multiply PPL values by -1
    # This flips the correlation direction:
    # - Original: Lower PPL correlates with Higher Tm1 → negative correlation
    # - After -1: Higher (-PPL) correlates with Higher Tm1 → positive correlation

    spearman_correlations = {}
    pearson_correlations = {}
    spearman_pvalues = {}
    pearson_pvalues = {}

    for model in model_columns:
        if model not in df.columns:
            print(f"  WARNING: Model column '{model}' not found")
            continue

        x = df[model].values.copy()

        # Apply direction correction
        if multiply_neg1:
            x = -1 * x

        # Spearman correlation
        corr, ci_low, ci_high, p = bootstrap_correlation(
            x, y, method='spearman', n_bootstrap=n_bootstrap
        )
        spearman_correlations[model] = (corr, ci_low, ci_high, p)

        # Pearson correlation
        corr, ci_low, ci_high, p = bootstrap_correlation(
            x, y, method='pearson', n_bootstrap=n_bootstrap
        )
        pearson_correlations[model] = (corr, ci_low, ci_high, p)

    # Find the best PRISM variant (highest Spearman correlation)
    prism_models = model_columns[:n_prism_variants]
    baseline_models = model_columns[n_prism_variants:]

    best_prism_corr = -float('inf')
    best_prism_model = prism_models[0] if prism_models else model_columns[0]
    for model in prism_models:
        if model in spearman_correlations:
            corr = spearman_correlations[model][0]
            if not np.isnan(corr) and corr > best_prism_corr:
                best_prism_corr = corr
                best_prism_model = model

    # Calculate p-values for differences (best PRISM vs baseline models only)
    if best_prism_model in df.columns:
        best_prism_data = df[best_prism_model].values.copy()
        if multiply_neg1:
            best_prism_data = -1 * best_prism_data

        for model in model_columns:
            if model in prism_models:
                # PRISM variants: no p-value comparison needed
                spearman_pvalues[model] = 1.0
                pearson_pvalues[model] = 1.0
            elif model in df.columns:
                # Baseline models: compare with best PRISM
                model_data = df[model].values.copy()
                if multiply_neg1:
                    model_data = -1 * model_data

                spearman_pvalues[model] = paired_bootstrap_test(
                    best_prism_data, model_data, y, method='spearman', n_bootstrap=n_bootstrap
                )
                pearson_pvalues[model] = paired_bootstrap_test(
                    best_prism_data, model_data, y, method='pearson', n_bootstrap=n_bootstrap
                )

    return spearman_correlations, pearson_correlations, spearman_pvalues, pearson_pvalues


def create_property_figure(
    spearman_correlations: Dict,
    pearson_correlations: Dict,
    spearman_pvalues: Dict,
    pearson_pvalues: Dict,
    property_config: dict,
    model_columns: List[str],
    display_names: List[str],
    colors: List[str],
    output_path: str,
    dpi: int = 300,
    n_prism_variants: int = 1,
    vertical_layout: bool = False,
    pearson_only: bool = False
):
    """
    Create a figure with Spearman and Pearson correlation bar plots.
    """
    # Filter to only models that exist in correlations
    valid_models = [m for m in model_columns if m in spearman_correlations]
    valid_display = [display_names[model_columns.index(m)] for m in valid_models]
    valid_colors = [colors[model_columns.index(m)] for m in valid_models]

    # Count valid PRISM variants
    valid_prism_count = sum(1 for m in model_columns[:n_prism_variants] if m in spearman_correlations)

    if pearson_only:
        # Single Pearson plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        fig.suptitle(property_config['display_name'], **FONT_CONFIG['title'], y=0.98)
        create_correlation_barplot(
            pearson_correlations,
            pearson_pvalues,
            "Pearson r",
            valid_models,
            valid_display,
            valid_colors,
            ax,
            n_prism_variants=valid_prism_count
        )
        # Modify output path to include _pearson suffix
        output_path = output_path.replace('.png', '_pearson.png')
    else:
        # Two subplots (Spearman + Pearson)
        if vertical_layout:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 14))
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle(property_config['display_name'], **FONT_CONFIG['title'], y=0.98)
        create_correlation_barplot(
            spearman_correlations,
            spearman_pvalues,
            "Spearman ρ",
            valid_models,
            valid_display,
            valid_colors,
            ax1,
            n_prism_variants=valid_prism_count
        )
        create_correlation_barplot(
            pearson_correlations,
            pearson_pvalues,
            "Pearson r",
            valid_models,
            valid_display,
            valid_colors,
            ax2,
            n_prism_variants=valid_prism_count
        )

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)  # Make room for suptitle

    # Save figure as PNG
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    # Save figure as SVG
    svg_path = output_path.replace('.png', '.svg')
    fig.savefig(svg_path, format='svg', bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f"  ✓ Saved: {output_path}")
    print(f"  ✓ Saved: {svg_path}")


def print_results_table(
    property_name: str,
    spearman_correlations: Dict,
    pearson_correlations: Dict,
    spearman_pvalues: Dict,
    pearson_pvalues: Dict,
    model_columns: List[str],
    display_names: List[str]
):
    """Print formatted results table."""
    print(f"\n{'='*90}")
    print(f"{property_name.upper()}")
    print(f"{'='*90}")

    print(f"\nSPEARMAN CORRELATION:")
    print(f"{'Model':<15} {'Correlation':>12} {'95% CI':>22} {'p-value':>15} {'vs PRISM':>15}")
    print("-" * 80)

    for model, display in zip(model_columns, display_names):
        if model in spearman_correlations:
            corr, ci_low, ci_high, p = spearman_correlations[model]
            vs_evo = format_pvalue(spearman_pvalues.get(model, np.nan)) if model != model_columns[0] else "-"
            print(f"{display:<15} {corr:>12.4f} [{ci_low:>8.4f}, {ci_high:>8.4f}] {format_pvalue(p):>15} {vs_evo:>15}")

    print(f"\nPEARSON CORRELATION:")
    print(f"{'Model':<15} {'Correlation':>12} {'95% CI':>22} {'p-value':>15} {'vs PRISM':>15}")
    print("-" * 80)

    for model, display in zip(model_columns, display_names):
        if model in pearson_correlations:
            corr, ci_low, ci_high, p = pearson_correlations[model]
            vs_evo = format_pvalue(pearson_pvalues.get(model, np.nan)) if model != model_columns[0] else "-"
            print(f"{display:<15} {corr:>12.4f} [{ci_low:>8.4f}, {ci_high:>8.4f}] {format_pvalue(p):>15} {vs_evo:>15}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Create developability correlation plots with bootstrapping',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python plot_developability_correlations.py \\
        --csv_path data/ginkgo/developability_data_with_ppl.csv

    python plot_developability_correlations.py \\
        --csv_path data/ginkgo/developability_data_with_ppl.csv \\
        --output_dir img/developability \\
        --n_bootstrap 1000

Properties analyzed:
    1. HIC (Hydrophobicity) - no direction correction
    2. PR_CHO (Polyreactivity) - no direction correction
    3. AC-SINS (Aggregation/Self-Interaction) - no direction correction
    4. Tm1 (Thermal Stability) - PPL × (-1) for correct direction
    5. Titer (Expression) - PPL × (-1) for correct direction

Note on direction correction:
    For Tm1 and Titer, higher values are better.
    Since lower PPL = better, we multiply PPL by -1 so that:
    - Positive correlation = lower PPL correlates with higher Tm1/Titer
        """
    )

    parser.add_argument('--csv_path', type=str, required=True,
                        help='Path to input CSV file with PPL values')
    parser.add_argument('--output_dir', type=str,
                        default='img/3.zero-shot',
                        help='Directory to save output figures')
    parser.add_argument('--n_bootstrap', type=int, default=1000,
                        help='Number of bootstrap iterations (default: 1000)')
    parser.add_argument('--dpi', type=int, default=300,
                        help='DPI for saved figures (default: 300)')
    parser.add_argument('--prefix', type=str, default='developability',
                        help='Prefix for output filenames (default: developability)')
    parser.add_argument('--properties', type=str, nargs='+',
                        choices=['all', 'hydrophobicity', 'reactivity', 'aggregation',
                                 'thermalstability', 'expression'],
                        default=['all'],
                        help='Properties to analyze: all (default), or specific properties. '
                             'Can specify multiple: --properties hydrophobicity expression')
    parser.add_argument('--evo_ab_variants', type=str, nargs='+',
                        choices=['all', 'gl', 'ngl', 'marg'],
                        default=['all'],
                        help='PRISM variants to include: all (default), gl (Final Head Upper), '
                             'ngl (Final Head Lower), marg (marginalized). '
                             'In standard mode, can specify multiple: --evo_ab_variants gl marg. '
                             'In ablation mode, uses first specified variant as reference.')
    # Ablation mode arguments
    parser.add_argument('--vertical_layout', action='store_true',
                        help='Use vertical layout (2 rows) instead of horizontal (2 columns)')
    parser.add_argument('--ablation_mode', action='store_true',
                        help='Enable ablation study mode: compare PRISM Full vs 3 ablation models')
    parser.add_argument('--ablation1_csv', type=str, default=None,
                        help='Path to Ablation 1 (Multihead+NoPretrain) CSV file')
    parser.add_argument('--ablation2_csv', type=str, default=None,
                        help='Path to Ablation 2 (SimpleHead+Pretrain) CSV file')
    parser.add_argument('--ablation3_csv', type=str, default=None,
                        help='Path to Ablation 3 (SimpleHead+NoPretrain) CSV file')
    parser.add_argument('--pearson_only', action='store_true',
                        help='Generate only Pearson correlation plot (single figure)')

    args = parser.parse_args()

    # Load data
    print("=" * 80)
    print("Developability Correlation Analysis with Bootstrapping")
    print("=" * 80)
    print(f"\nLoading data from: {args.csv_path}")

    df = pd.read_csv(args.csv_path)
    print(f"  Loaded {len(df)} samples")
    print(f"  Columns: {df.columns.tolist()[:15]}...")

    # =================================================================
    # ABLATION MODE: Special handling for ablation study comparison
    # =================================================================
    if args.ablation_mode:
        print("\n" + "=" * 60)
        print("ABLATION STUDY MODE")
        print("=" * 60)

        # PRISM variant mapping for ablation mode
        # Maps variant key -> (column_name, display_name)
        prism_variant_map = {
            'gl': ('evo_ab_ppl', 'PRISM (Upper)'),
            'ngl': ('evo_ab_ppl_final_lower', 'PRISM (Lower)'),
            'marg': ('evo_ab_ppl_marginalized', 'PRISM (Marg)'),
        }

        # Determine which PRISM variant to use
        # Default to 'gl' if 'all' is specified or no valid variant
        if 'all' in args.evo_ab_variants:
            selected_variant = 'gl'  # Default to Upper head for backwards compatibility
        else:
            selected_variant = args.evo_ab_variants[0]  # Use first specified variant

        prism_col, prism_display = prism_variant_map.get(selected_variant, ('evo_ab_ppl', 'PRISM (Upper)'))
        print(f"  PRISM variant selected: {selected_variant} -> {prism_col}")

        # Ablation study colors (Paul Tol's colorblind-friendly palette)
        ABLATION_COLORS = {
            'PRISM Full': '#332288',      # Dark purple (best model)
            'Ablation 1': '#CC6677',      # Rose
            'Ablation 2': '#DDCC77',      # Sand/Yellow
            'Ablation 3': '#AA4499',      # Purple-pink
        }

        # Property-specific PPL column mapping for PRISM Full
        # - expression: marginalized
        # - immunogenicity: marg (marginalized) - handled in immunogenicity script
        # - rest: lowercase (Final Head Lower)
        PRISM_PPL_COLUMNS = {
            'expression': 'evo_ab_ppl_marginalized',
            'hydrophobicity': 'evo_ab_ppl_final_lower',
            'reactivity': 'evo_ab_ppl_final_lower',
            'aggregation': 'evo_ab_ppl_final_lower',
            'thermalstability': 'evo_ab_ppl_final_lower',
        }

        # Load ablation CSVs and merge PPL values
        ablation_files = [
            (args.ablation1_csv, 'ablation1_ppl', 'Ablation 1'),
            (args.ablation2_csv, 'ablation2_ppl', 'Ablation 2'),
            (args.ablation3_csv, 'ablation3_ppl', 'Ablation 3'),
        ]

        # Load ablation model PPL values from their CSV files
        ablation_dfs = {}
        for csv_path, col_name, display_name in ablation_files:
            if csv_path and os.path.exists(csv_path):
                ablation_df = pd.read_csv(csv_path)
                ablation_dfs[display_name] = ablation_df
                print(f"  ✓ {display_name} data loaded from: {csv_path}")
            elif csv_path:
                print(f"  ✗ WARNING: File not found: {csv_path}")

        print(f"\n  Dataset size: {len(df)} samples")
        print(f"  Ablation models loaded: {len(ablation_dfs)}")
        print(f"  Running bootstrapping with {args.n_bootstrap} iterations...\n")

        os.makedirs(args.output_dir, exist_ok=True)

        # Determine which properties to analyze
        if 'all' in args.properties:
            selected_properties = list(PROPERTY_CONFIG.keys())
        else:
            selected_properties = args.properties

        # Analyze each property
        for prop_key in selected_properties:
            if prop_key not in PROPERTY_CONFIG:
                print(f"  WARNING: Unknown property '{prop_key}', skipping...")
                continue
            prop_config = PROPERTY_CONFIG[prop_key]
            print(f"\n{'='*60}")
            print(f"Analyzing: {prop_config['display_name']} (Ablation)")
            print(f"  Column: {prop_config['column']}")
            print(f"  Direction correction (×-1): {prop_config['multiply_by_neg1']}")
            print("=" * 60)

            if prop_config['column'] not in df.columns:
                print(f"  SKIPPED: Column '{prop_config['column']}' not found")
                continue

            # Build model_columns, display_names, colors for this property
            model_columns = []
            display_names = []
            colors = []

            # PRISM Full - use property-specific PPL column
            prism_ppl_col = PRISM_PPL_COLUMNS.get(prop_key, 'evo_ab_ppl_final_lower')
            if prism_ppl_col in df.columns:
                df['prism_full_ppl'] = df[prism_ppl_col]
                model_columns.append('prism_full_ppl')
                display_names.append('PRISM Full')
                colors.append(ABLATION_COLORS['PRISM Full'])
                print(f"  PRISM Full using: {prism_ppl_col}")
            else:
                print(f"  ✗ WARNING: '{prism_ppl_col}' not found for PRISM Full")

            # Ablation models - use evo_ab_ppl or evo_ab_ppl_aa
            for ablation_name in ['Ablation 1', 'Ablation 2', 'Ablation 3']:
                if ablation_name in ablation_dfs:
                    ablation_df = ablation_dfs[ablation_name]
                    col_name = f'{ablation_name.lower().replace(" ", "_")}_ppl'
                    ppl_col = None
                    if 'evo_ab_ppl' in ablation_df.columns and not ablation_df['evo_ab_ppl'].isna().all():
                        ppl_col = ablation_df['evo_ab_ppl'].values
                    elif 'evo_ab_ppl_aa' in ablation_df.columns and not ablation_df['evo_ab_ppl_aa'].isna().all():
                        ppl_col = ablation_df['evo_ab_ppl_aa'].values
                    if ppl_col is not None:
                        df[col_name] = ppl_col
                        model_columns.append(col_name)
                        display_names.append(ablation_name)
                        colors.append(ABLATION_COLORS[ablation_name])

            # Get target values (with optional direction correction)
            target_col = prop_config['column']
            y = df[target_col].values

            # Calculate correlations with bootstrapping
            spearman_correlations = {}
            pearson_correlations = {}
            spearman_pvalues = {}
            pearson_pvalues = {}

            for model in model_columns:
                x = df[model].values
                # Apply direction correction if needed
                if prop_config['multiply_by_neg1']:
                    x = -x

                corr, ci_low, ci_high, p = bootstrap_correlation(
                    x, y, method='spearman', n_bootstrap=args.n_bootstrap
                )
                spearman_correlations[model] = (corr, ci_low, ci_high, p)

                corr, ci_low, ci_high, p = bootstrap_correlation(
                    x, y, method='pearson', n_bootstrap=args.n_bootstrap
                )
                pearson_correlations[model] = (corr, ci_low, ci_high, p)

            # Calculate p-values for differences
            if 'prism_full_ppl' in df.columns:
                best_model_data = df['prism_full_ppl'].values
                if prop_config['multiply_by_neg1']:
                    best_model_data = -best_model_data

                for model in model_columns:
                    if model == 'prism_full_ppl':
                        spearman_pvalues[model] = 1.0
                        pearson_pvalues[model] = 1.0
                    else:
                        model_data = df[model].values
                        if prop_config['multiply_by_neg1']:
                            model_data = -model_data
                        spearman_pvalues[model] = paired_bootstrap_test(
                            best_model_data, model_data, y, method='spearman', n_bootstrap=args.n_bootstrap
                        )
                        pearson_pvalues[model] = paired_bootstrap_test(
                            best_model_data, model_data, y, method='pearson', n_bootstrap=args.n_bootstrap
                        )

            # Print results
            print(f"\nSPEARMAN CORRELATION RESULTS ({prop_config['display_name']})")
            print("-" * 80)
            print(f"{'Model':<25} {'Correlation':>12} {'95% CI':>20} {'p-value':>15}")
            print("-" * 80)
            for model, display in zip(model_columns, display_names):
                corr, ci_low, ci_high, p = spearman_correlations[model]
                print(f"{display:<25} {corr:>12.4f} [{ci_low:>8.4f}, {ci_high:>8.4f}] {format_pvalue(p):>15}")

            # Create figure
            if args.pearson_only:
                # Single Pearson plot
                fig, ax = plt.subplots(1, 1, figsize=(8, 8))
                fig.suptitle(f'{prop_config["display_name"]} (Ablation Study)',
                            **FONT_CONFIG['title'], y=0.98)
                create_correlation_barplot(
                    pearson_correlations, pearson_pvalues, "Pearson r",
                    model_columns, display_names, colors, ax, n_prism_variants=1
                )
                plt.tight_layout()
                plt.subplots_adjust(top=0.93)
                suffix = prop_config['filename_suffix'].replace('.png', '_pearson.png')
                output_path = os.path.join(args.output_dir, f"ablation{suffix}")
            else:
                # Two subplots (Spearman + Pearson)
                if args.vertical_layout:
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 14))
                else:
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
                fig.suptitle(f'{prop_config["display_name"]} (Ablation Study)',
                            **FONT_CONFIG['title'], y=0.98)
                create_correlation_barplot(
                    spearman_correlations, spearman_pvalues, "Spearman ρ",
                    model_columns, display_names, colors, ax1, n_prism_variants=1
                )
                create_correlation_barplot(
                    pearson_correlations, pearson_pvalues, "Pearson r",
                    model_columns, display_names, colors, ax2, n_prism_variants=1
                )
                plt.tight_layout()
                plt.subplots_adjust(top=0.93)
                output_path = os.path.join(args.output_dir, f"ablation{prop_config['filename_suffix']}")

            fig.savefig(output_path, dpi=args.dpi, bbox_inches='tight', facecolor='white')
            svg_path = output_path.replace('.png', '.svg')
            fig.savefig(svg_path, format='svg', bbox_inches='tight', facecolor='white')

            print(f"\n✓ Figure saved to: {output_path}")
            plt.close(fig)

        print(f"\n{'='*80}")
        print("Ablation Analysis Complete!")
        print(f"Figures saved to: {args.output_dir}")
        print("=" * 80)
        return  # Exit early for ablation mode

    # =================================================================
    # STANDARD MODE
    # =================================================================

    # Define PRISM variant mapping
    # GL = Final Head Upper (uppercase/germline tokens)
    # NGL = Final Head Lower (lowercase/non-germline tokens)
    # Marg = Marginalized (sum of upper + lower probabilities)
    # Single variant: all use dark purple #332288
    evo_ab_variant_map = {
        'gl': ('evo_ab_ppl', 'PRISM (Upper)', '#332288'),
        'ngl': ('evo_ab_ppl_final_lower', 'PRISM (Lower)', '#332288'),
        'marg': ('evo_ab_ppl_marginalized', 'PRISM (Marg)', '#332288'),
    }

    # Multiple variants: use gradient colors
    evo_ab_variant_map_multi = {
        'gl': ('evo_ab_ppl', 'PRISM (Upper)', '#332288'),           # Dark purple
        'ngl': ('evo_ab_ppl_final_lower', 'PRISM (Lower)', '#4B32C8'),  # Medium purple
        'marg': ('evo_ab_ppl_marginalized', 'PRISM (Marg)', '#816FDB'),  # Light purple
    }

    # Determine which PRISM variants to include
    # Order: gl, ngl, marg
    if 'all' in args.evo_ab_variants:
        selected_evo_ab = ['gl', 'ngl', 'marg']
    else:
        selected_evo_ab = args.evo_ab_variants

    # Build filtered model columns, display names, and colors
    filtered_model_columns = []
    filtered_display_names = []
    filtered_colors = []

    # If only one PRISM variant selected, use simple "PRISM" label and dark purple
    use_simple_label = (len(selected_evo_ab) == 1)

    # Choose variant map based on number of variants
    variant_map_to_use = evo_ab_variant_map if use_simple_label else evo_ab_variant_map_multi

    # Add selected PRISM variants
    # Order: gl (primary), ngl, marg
    for variant in ['gl', 'ngl', 'marg']:  # Maintain order
        if variant in selected_evo_ab:
            col, name, color = variant_map_to_use[variant]
            filtered_model_columns.append(col)
            # Use "PRISM" if only one variant, otherwise use full name
            filtered_display_names.append('PRISM' if use_simple_label else name)
            filtered_colors.append(color)

    # Track number of PRISM variants
    n_prism_variants = len(filtered_model_columns)

    # Add baseline models (always included)
    baseline_models = [
        ('esm2_35m_ppl', 'ESM2-35M', '#DDCC77'),
        ('esm2_650m_ppl', 'ESM2-650M', '#117733'),
        ('ablang2_ppl', 'AbLang2', '#88CCEE'),
        ('antiberty_ppl', 'AntiBERTy', '#44AA99'),
        ('sapiens_ppl', 'Sapiens', '#882255'),
    ]
    for col, name, color in baseline_models:
        filtered_model_columns.append(col)
        filtered_display_names.append(name)
        filtered_colors.append(color)

    # Use filtered lists
    model_columns = filtered_model_columns
    display_names = filtered_display_names
    colors = filtered_colors

    print(f"\n  PRISM variants selected: {selected_evo_ab}")

    # Check which model columns exist
    existing_models = [m for m in model_columns if m in df.columns]
    print(f"\n  Found PPL columns: {existing_models}")

    if not existing_models:
        print("\nERROR: No PPL columns found in the data!")
        print(f"Expected columns like: {model_columns}")
        print(f"Available columns: {df.columns.tolist()}")
        return

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\nRunning bootstrapping with {args.n_bootstrap} iterations...")
    print(f"Output directory: {args.output_dir}")

    # Determine which properties to analyze
    if 'all' in args.properties:
        selected_properties = list(PROPERTY_CONFIG.keys())
    else:
        selected_properties = args.properties

    print(f"Properties to analyze: {selected_properties}")

    # Analyze each property
    for prop_key in selected_properties:
        if prop_key not in PROPERTY_CONFIG:
            print(f"  WARNING: Unknown property '{prop_key}', skipping...")
            continue
        prop_config = PROPERTY_CONFIG[prop_key]
        print(f"\n{'='*60}")
        print(f"Analyzing: {prop_config['display_name']}")
        print(f"  Column: {prop_config['column']}")
        print(f"  Direction correction (×-1): {prop_config['multiply_by_neg1']}")
        print("=" * 60)

        # Check if column exists
        if prop_config['column'] not in df.columns:
            print(f"  SKIPPED: Column '{prop_config['column']}' not found")
            continue

        # Analyze correlations
        spearman_corrs, pearson_corrs, spearman_pvals, pearson_pvals = analyze_property(
            df=df,
            property_config=prop_config,
            model_columns=model_columns,
            display_names=display_names,
            n_bootstrap=args.n_bootstrap,
            n_prism_variants=n_prism_variants
        )

        if not spearman_corrs:
            print(f"  SKIPPED: No valid correlations calculated")
            continue

        # Print results table
        print_results_table(
            prop_config['display_name'],
            spearman_corrs,
            pearson_corrs,
            spearman_pvals,
            pearson_pvals,
            model_columns,
            display_names
        )

        # Create figure
        output_path = os.path.join(args.output_dir, f"{args.prefix}{prop_config['filename_suffix']}")

        create_property_figure(
            spearman_corrs,
            pearson_corrs,
            spearman_pvals,
            pearson_pvals,
            prop_config,
            model_columns,
            display_names,
            colors,
            output_path,
            dpi=args.dpi,
            n_prism_variants=n_prism_variants,
            vertical_layout=args.vertical_layout,
            pearson_only=args.pearson_only
        )

    print(f"\n{'='*80}")
    print("Analysis Complete!")
    print(f"Figures saved to: {args.output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
