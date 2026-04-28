#!/usr/bin/env python3
"""
Correlation Comparison Plot with Bootstrapping

This script creates publication-quality bar plots comparing Spearman and Pearson
correlations between different protein language models and experimental fitness data.
Uses bootstrapping for 95% confidence intervals and statistical significance testing.

Usage:
    python plot_correlation_comparison.py --csv_path /path/to/data.csv
    python plot_correlation_comparison.py --csv_path /path/to/data.csv --output_dir ./figures
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Configure matplotlib for publication-quality figures
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'

# Font configuration dictionary for consistent styling
FONT_CONFIG = {
    'axis_label': {'fontsize': 25, 'fontweight': 'bold'},
    'tick_label': {'fontsize': 15, 'fontweight': 'normal'},
    'legend': {'fontsize': 20},
    'title': {'fontsize': 25, 'fontweight': 'bold'},
    'annotation': {'fontsize': 15, 'fontweight': 'normal'},
}


def bootstrap_correlation(
    x: np.ndarray,
    y: np.ndarray,
    method: str = 'spearman',
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    random_state: int = 42
) -> Tuple[float, float, float, float]:
    """
    Calculate correlation with bootstrapped confidence intervals.

    Args:
        x: First variable array
        y: Second variable array
        method: 'spearman' or 'pearson'
        n_bootstrap: Number of bootstrap iterations
        ci: Confidence interval level (default 0.95 for 95% CI)
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (correlation, ci_lower, ci_upper, p_value)
    """
    np.random.seed(random_state)

    # Remove NaN values
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]
    n = len(x_clean)

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
    n_bootstrap: int = 10000,
    random_state: int = 42
) -> float:
    """
    Perform paired bootstrap test to compare two correlations.

    Tests whether correlation(x1, y) is significantly different from correlation(x2, y).

    Args:
        x1: First predictor (our model)
        x2: Second predictor (comparison model)
        y: Target variable
        method: 'spearman' or 'pearson'
        n_bootstrap: Number of bootstrap iterations
        random_state: Random seed

    Returns:
        Two-tailed p-value for the difference
    """
    np.random.seed(random_state)

    # Remove NaN values
    mask = ~(np.isnan(x1) | np.isnan(x2) | np.isnan(y))
    x1_clean = x1[mask]
    x2_clean = x2[mask]
    y_clean = y[mask]
    n = len(y_clean)

    # Calculate original difference
    if method == 'spearman':
        r1_orig, _ = stats.spearmanr(x1_clean, y_clean)
        r2_orig, _ = stats.spearmanr(x2_clean, y_clean)
    else:
        r1_orig, _ = stats.pearsonr(x1_clean, y_clean)
        r2_orig, _ = stats.pearsonr(x2_clean, y_clean)

    observed_diff = r1_orig - r2_orig

    # Bootstrap under null hypothesis (permutation-style)
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
    """
    Draw a significance bracket between two bars with p-value annotation.

    Args:
        ax: Matplotlib axes
        x1, x2: x-coordinates of the two bars
        y: y-coordinate for the bracket (top of bars)
        p_value: p-value to display
        height: Height of the bracket above y
        tip_length: Length of the vertical tips
    """
    # Draw the bracket
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
    n_prism_variants: int = 1
):
    """
    Create a single correlation bar plot with error bars and significance brackets.

    Args:
        correlations: Dict mapping model names to (corr, ci_low, ci_high, p_value)
        p_values: Dict mapping model names to p-values (comparison with best PRISM)
        ylabel: Y-axis label
        model_names: List of model column names
        display_names: List of display names for models
        colors: List of colors for bars
        ax: Matplotlib axes
        n_prism_variants: Number of PRISM variants (first N models)
    """
    n_models = len(model_names)
    x_pos = np.arange(n_models)

    # Extract correlation values and CIs
    corr_values = [correlations[name][0] for name in model_names]
    ci_lows = [correlations[name][1] for name in model_names]
    ci_highs = [correlations[name][2] for name in model_names]

    # Calculate error bar lengths
    yerr_low = [corr - ci_low for corr, ci_low in zip(corr_values, ci_lows)]
    yerr_high = [ci_high - corr for corr, ci_high in zip(corr_values, ci_highs)]

    # Create bars
    bars = ax.bar(x_pos, corr_values, color=colors, edgecolor='black', linewidth=2,
                  yerr=[yerr_low, yerr_high], capsize=8,
                  error_kw={'linewidth': 2, 'capthick': 2})

    # Find the best PRISM variant (highest correlation value)
    prism_corrs = corr_values[:n_prism_variants]
    best_prism_idx = prism_corrs.index(max(prism_corrs))
    best_prism_model = model_names[best_prism_idx]

    # Highlight the best PRISM bar with a thicker edge
    bars[best_prism_idx].set_edgecolor('#000000')
    bars[best_prism_idx].set_linewidth(3)

    # Add significance brackets between best PRISM and baseline models only
    # Baseline models are those after the PRISM variants
    max_ci = max(ci_highs)
    min_ci = min(ci_lows)
    bracket_start = max_ci + 0.10  # Start above the highest CI (increased to avoid overlap with values)
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
    ax.tick_params(axis='y', labelsize=FONT_CONFIG['tick_label']['fontsize'])

    # Add correlation values above bars (always above to avoid x-axis label overlap)
    for i, (bar, corr, ci_high, ci_low) in enumerate(zip(bars, corr_values, ci_highs, ci_lows)):
        if corr >= 0:
            # Positive value: place text above the upper CI
            y_pos = ci_high + 0.02
        else:
            # Negative value: place text above the upper CI (near zero), not below bar
            y_pos = ci_high + 0.02
        ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                f'{corr:.3f}', ha='center', va='bottom', color='black',
                **FONT_CONFIG['annotation'])

    # Set y-axis limits: highest bracket + extra padding for stars, minimum 0.5
    # Add more space (0.15) to accommodate the **** annotations above brackets
    y_upper = max(0.5, max_bracket_y + 0.18)
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

    # Add grid for readability
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)

    # Thicken spines
    for spine in ax.spines.values():
        spine.set_linewidth(2)


def main():
    parser = argparse.ArgumentParser(
        description='Create correlation comparison plots with bootstrapping'
    )
    parser.add_argument('--csv_path', type=str, required=True,
                        help='Path to the input CSV file')
    parser.add_argument('--output_dir', type=str,
                        default='img/3.zero-shot',
                        help='Directory to save output figures')
    parser.add_argument('--n_bootstrap', type=int, default=1000,
                        help='Number of bootstrap iterations (default: 1000)')
    parser.add_argument('--target_column', type=str, default='fitness',
                        help='Target column name for correlation (default: fitness)')
    parser.add_argument('--dpi', type=int, default=300,
                        help='DPI for saved figures (default: 300)')
    parser.add_argument('--evo_ab_variants', type=str, nargs='+',
                        choices=['all', 'gl', 'ngl', 'marg'],
                        default=['all'],
                        help='PRISM variants to include: all (default), gl (Final Head Upper), '
                             'ngl (Final Head Lower), marg (marginalized). '
                             'Can specify multiple: --evo_ab_variants gl marg')
    parser.add_argument('--title', type=str, default=None,
                        help='Title to display at the top of the figure (e.g., "CR9114 (influenza)")')
    parser.add_argument('--vertical_layout', action='store_true',
                        help='Use vertical layout (2 rows) instead of horizontal (2 columns)')
    # Ablation mode arguments
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
    print(f"Loading data from: {args.csv_path}")
    df = pd.read_csv(args.csv_path)

    # =================================================================
    # ABLATION MODE: Special handling for ablation study comparison
    # =================================================================
    if args.ablation_mode:
        print("\n" + "=" * 60)
        print("ABLATION STUDY MODE")
        print("=" * 60)

        # Ablation study colors (Paul Tol's colorblind-friendly palette)
        ABLATION_COLORS = {
            'PRISM Full': '#332288',      # Dark purple (best model)
            'Ablation 1': '#CC6677',      # Rose
            'Ablation 2': '#DDCC77',      # Sand/Yellow
            'Ablation 3': '#AA4499',      # Purple-pink
        }

        # Score column mapping for ablation models
        # Best model uses evo_ab_score from output2.csv
        # Ablation models use evo_ab_score from their respective files
        model_columns = []
        display_names = []
        colors = []

        # Best model (PRISM Full) - from the main csv_path (output2.csv)
        if 'evo_ab_score' in df.columns:
            df['prism_full_score'] = df['evo_ab_score']
            model_columns.append('prism_full_score')
            display_names.append('PRISM Full')
            colors.append(ABLATION_COLORS['PRISM Full'])
            print(f"  ✓ PRISM Full score loaded from: {args.csv_path}")
        else:
            print(f"  ✗ WARNING: 'evo_ab_score' not found in {args.csv_path}")

        # Load ablation CSVs and merge scores
        ablation_files = [
            (args.ablation1_csv, 'ablation1_score', 'Ablation 1'),
            (args.ablation2_csv, 'ablation2_score', 'Ablation 2'),
            (args.ablation3_csv, 'ablation3_score', 'Ablation 3'),
        ]

        for csv_path, col_name, display_name in ablation_files:
            if csv_path and os.path.exists(csv_path):
                ablation_df = pd.read_csv(csv_path)
                # Ablation files have evo_ab_affinity_score (new predictions)
                # NOT evo_ab_score (which is copied from best model input)
                if 'evo_ab_affinity_score' in ablation_df.columns:
                    df[col_name] = ablation_df['evo_ab_affinity_score'].values
                    model_columns.append(col_name)
                    display_names.append(display_name)
                    colors.append(ABLATION_COLORS[display_name])
                    print(f"  ✓ {display_name} score loaded from: {csv_path} (column: evo_ab_affinity_score)")
                elif 'evo_ab_score' in ablation_df.columns:
                    # Fallback for older files that might use evo_ab_score
                    df[col_name] = ablation_df['evo_ab_score'].values
                    model_columns.append(col_name)
                    display_names.append(display_name)
                    colors.append(ABLATION_COLORS[display_name])
                    print(f"  ✓ {display_name} score loaded from: {csv_path} (column: evo_ab_score - fallback)")
                else:
                    print(f"  ✗ WARNING: Neither 'evo_ab_affinity_score' nor 'evo_ab_score' found in {csv_path}")
            elif csv_path:
                print(f"  ✗ WARNING: File not found: {csv_path}")

        # Get target values
        y = df[args.target_column].values

        print(f"\n  Dataset size: {len(df)} samples")
        print(f"  Target column: {args.target_column}")
        print(f"  Models to compare: {len(model_columns)}")
        print(f"  Running bootstrapping with {args.n_bootstrap} iterations...\n")

        # Calculate correlations with bootstrapping
        spearman_correlations = {}
        pearson_correlations = {}
        spearman_pvalues = {}
        pearson_pvalues = {}

        for model in model_columns:
            x = df[model].values

            # Spearman correlation
            corr, ci_low, ci_high, p = bootstrap_correlation(
                x, y, method='spearman', n_bootstrap=args.n_bootstrap
            )
            spearman_correlations[model] = (corr, ci_low, ci_high, p)

            # Pearson correlation
            corr, ci_low, ci_high, p = bootstrap_correlation(
                x, y, method='pearson', n_bootstrap=args.n_bootstrap
            )
            pearson_correlations[model] = (corr, ci_low, ci_high, p)

        # Calculate p-values for differences (Best model vs ablations)
        if 'prism_full_score' in df.columns:
            best_model_data = df['prism_full_score'].values

            for model in model_columns:
                if model == 'prism_full_score':
                    spearman_pvalues[model] = 1.0
                    pearson_pvalues[model] = 1.0
                else:
                    model_data = df[model].values
                    spearman_pvalues[model] = paired_bootstrap_test(
                        best_model_data, model_data, y, method='spearman', n_bootstrap=args.n_bootstrap
                    )
                    pearson_pvalues[model] = paired_bootstrap_test(
                        best_model_data, model_data, y, method='pearson', n_bootstrap=args.n_bootstrap
                    )

        # Print results
        print("=" * 80)
        print("SPEARMAN CORRELATION RESULTS (Ablation Study)")
        print("=" * 80)
        print(f"{'Model':<25} {'Correlation':>12} {'95% CI':>20} {'p-value':>15} {'vs PRISM':>15}")
        print("-" * 80)
        for model, display in zip(model_columns, display_names):
            corr, ci_low, ci_high, p = spearman_correlations[model]
            vs_best = format_pvalue(spearman_pvalues[model]) if model != 'prism_full_score' else "-"
            print(f"{display.replace(chr(10), ' '):<25} {corr:>12.4f} [{ci_low:>8.4f}, {ci_high:>8.4f}] {format_pvalue(p):>15} {vs_best:>15}")

        print("\n" + "=" * 80)
        print("PEARSON CORRELATION RESULTS (Ablation Study)")
        print("=" * 80)
        print(f"{'Model':<25} {'Correlation':>12} {'95% CI':>20} {'p-value':>15} {'vs PRISM':>15}")
        print("-" * 80)
        for model, display in zip(model_columns, display_names):
            corr, ci_low, ci_high, p = pearson_correlations[model]
            vs_best = format_pvalue(pearson_pvalues[model]) if model != 'prism_full_score' else "-"
            print(f"{display.replace(chr(10), ' '):<25} {corr:>12.4f} [{ci_low:>8.4f}, {ci_high:>8.4f}] {format_pvalue(p):>15} {vs_best:>15}")

        # Create figure
        if args.pearson_only:
            # Single Pearson plot
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            if args.title:
                fig.suptitle(args.title, fontsize=28, fontweight='bold', y=0.98)
            create_correlation_barplot(
                pearson_correlations,
                pearson_pvalues,
                "Pearson r",
                model_columns,
                display_names,
                colors,
                ax,
                n_prism_variants=1
            )
            plt.tight_layout()
            if args.title:
                plt.subplots_adjust(top=0.93)
            suffix = "_ablation_results_pearson"
        else:
            # Two subplots (Spearman + Pearson)
            if args.vertical_layout:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 12))
            else:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
            if args.title:
                fig.suptitle(args.title, fontsize=28, fontweight='bold', y=0.98)
            create_correlation_barplot(
                spearman_correlations,
                spearman_pvalues,
                "Spearman ρ",
                model_columns,
                display_names,
                colors,
                ax1,
                n_prism_variants=1
            )
            create_correlation_barplot(
                pearson_correlations,
                pearson_pvalues,
                "Pearson r",
                model_columns,
                display_names,
                colors,
                ax2,
                n_prism_variants=1
            )
            plt.tight_layout()
            if args.title:
                plt.subplots_adjust(top=0.93)
            suffix = "_ablation_results"

        # Save ablation figure
        os.makedirs(args.output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(args.csv_path))[0]
        first_part = base_name.split('_')[0]

        output_path = os.path.join(args.output_dir, f"{first_part}{suffix}.png")
        fig.savefig(output_path, dpi=args.dpi, bbox_inches='tight', facecolor='white')
        svg_path = output_path.replace('.png', '.svg')
        fig.savefig(svg_path, format='svg', bbox_inches='tight', facecolor='white')

        print(f"\n✓ Ablation figure saved to: {output_path}")
        print(f"✓ Ablation figure saved to: {svg_path}")

        plt.close(fig)
        return  # Exit early for ablation mode

    # =================================================================
    # STANDARD MODE: Compute variant scores from logP columns if not already present
    # Score = logP(Mut) - lambda * logP(WT)
    # =================================================================
    logp_columns_exist = all(col in df.columns for col in [
        'logP_mut_upper', 'logP_mut_lower', 'logP_mut_marginalized',
        'logP_wt_upper', 'logP_wt_lower', 'logP_wt_marginalized'
    ])

    if logp_columns_exist:
        # Get lambda value (use best_lambda if available, otherwise default to 1.0)
        if 'best_lambda' in df.columns:
            # Use the most common best_lambda value (should be same for all rows)
            lambda_val = df['best_lambda'].mode().iloc[0] if not df['best_lambda'].isna().all() else 1.0
        else:
            lambda_val = 1.0
        print(f"  Using lambda = {lambda_val} for score computation")

        # Compute all three pure variant scores from logP columns
        # Score = logP(Mut) - lambda * logP(WT)

        # Compute evo_ab_score_final_upper (pure GL/upper variant)
        if 'evo_ab_score_final_upper' not in df.columns:
            df['evo_ab_score_final_upper'] = df['logP_mut_upper'] - lambda_val * df['logP_wt_upper']
            print("  Computed: evo_ab_score_final_upper (logP_mut_upper - λ * logP_wt_upper)")

        # Compute evo_ab_score_final_lower (pure NGL/lower variant)
        if 'evo_ab_score_final_lower' not in df.columns:
            df['evo_ab_score_final_lower'] = df['logP_mut_lower'] - lambda_val * df['logP_wt_lower']
            print("  Computed: evo_ab_score_final_lower (logP_mut_lower - λ * logP_wt_lower)")

        # Compute evo_ab_score_marginalized (combined variant)
        if 'evo_ab_score_marginalized' not in df.columns:
            df['evo_ab_score_marginalized'] = df['logP_mut_marginalized'] - lambda_val * df['logP_wt_marginalized']
            print("  Computed: evo_ab_score_marginalized (logP_mut_marg - λ * logP_wt_marg)")

    # Define PRISM variant mapping
    # All three variants use pure head scores (upper/lower/marginalized)
    # Colors: Purple gradient for PRISM variants
    # Note: When using single variant, all use dark purple (#332288) for consistency
    evo_ab_variant_map = {
        'gl': ('evo_ab_score_final_upper', 'PRISM (Upper)', '#332288'),    # Pure GL/upper head
        'ngl': ('evo_ab_score_final_lower', 'PRISM (Lower)', '#332288'),   # Pure NGL/lower head (same color when single)
        'marg': ('evo_ab_score_marginalized', 'PRISM (Marg)', '#332288'),  # Marginalized (both heads)
    }

    # When showing multiple variants, use gradient colors
    evo_ab_variant_map_multi = {
        'gl': ('evo_ab_score_final_upper', 'PRISM (Upper)', '#332288'),    # Dark purple
        'ngl': ('evo_ab_score_final_lower', 'PRISM (Lower)', '#4B32C8'),   # Medium purple
        'marg': ('evo_ab_score_marginalized', 'PRISM (Marg)', '#816FDB'),  # Light purple
    }

    # Determine which PRISM variants to include
    if 'all' in args.evo_ab_variants:
        selected_evo_ab = ['gl', 'ngl', 'marg']
    else:
        selected_evo_ab = args.evo_ab_variants

    # Build filtered model columns, display names, and colors
    model_columns = []
    display_names = []
    colors = []

    # If only one PRISM variant selected, use simple "PRISM" label and dark purple
    use_simple_label = (len(selected_evo_ab) == 1)

    # Choose variant map based on number of variants
    variant_map_to_use = evo_ab_variant_map if use_simple_label else evo_ab_variant_map_multi

    # Add selected PRISM variants (maintain order: gl, ngl, marg)
    for variant in ['gl', 'ngl', 'marg']:
        if variant in selected_evo_ab:
            col, name, color = variant_map_to_use[variant]
            if col in df.columns:
                model_columns.append(col)
                display_names.append('PRISM' if use_simple_label else name)
                colors.append(color)

    print(f"\n  PRISM variants selected: {selected_evo_ab}")
    prism_variants_found = len(model_columns)
    if prism_variants_found == 0:
        print(f"  WARNING: No PRISM score columns found!")
        print(f"  Available columns: {df.columns.tolist()[:15]}...")
        print(f"  Expected columns: evo_ab_score, evo_ab_score_final_lower, evo_ab_score_marginalized")
    else:
        print(f"  Found PRISM variants: {model_columns}")

    # Add baseline models
    baseline_models = [
        ('esm2_35m_score', 'ESM2-35M', '#DDCC77'),
        ('esm2_650m_score', 'ESM2-650M', '#117733'),
        ('ablang2_score', 'AbLang2', '#88CCEE'),
        ('antiberty_score', 'AntiBERTy', '#44AA99'),
        ('sapiens_score', 'Sapiens', '#882255'),
    ]
    for col, name, color in baseline_models:
        if col in df.columns:
            model_columns.append(col)
            display_names.append(name)
            colors.append(color)

    print(f"  Total score columns: {model_columns}")

    if len(model_columns) == 0:
        print("\nERROR: No score columns found in the data!")
        print("Please ensure the CSV contains score columns from the benchmark scripts.")
        return

    # Get target values
    y = df[args.target_column].values

    print(f"\nDataset size: {len(df)} samples")
    print(f"Target column: {args.target_column}")
    print(f"Running bootstrapping with {args.n_bootstrap} iterations...\n")

    # Calculate correlations with bootstrapping
    spearman_correlations = {}
    pearson_correlations = {}
    spearman_pvalues = {}
    pearson_pvalues = {}

    for model in model_columns:
        x = df[model].values

        # Spearman correlation
        corr, ci_low, ci_high, p = bootstrap_correlation(
            x, y, method='spearman', n_bootstrap=args.n_bootstrap
        )
        spearman_correlations[model] = (corr, ci_low, ci_high, p)

        # Pearson correlation
        corr, ci_low, ci_high, p = bootstrap_correlation(
            x, y, method='pearson', n_bootstrap=args.n_bootstrap
        )
        pearson_correlations[model] = (corr, ci_low, ci_high, p)

    # Find the best PRISM variant (highest Spearman correlation)
    n_prism_variants = prism_variants_found
    prism_models = model_columns[:n_prism_variants]
    baseline_models_list = model_columns[n_prism_variants:]

    # Find best PRISM based on Spearman correlation
    best_prism_corr = -float('inf')
    best_prism_model = prism_models[0] if prism_models else model_columns[0]
    for model in prism_models:
        corr = spearman_correlations[model][0]
        if corr > best_prism_corr:
            best_prism_corr = corr
            best_prism_model = model

    print(f"Best PRISM variant: {best_prism_model} (Spearman ρ = {best_prism_corr:.4f})")

    # Calculate p-values for differences (best PRISM vs baseline models only)
    best_prism_data = df[best_prism_model].values

    for model in model_columns:
        if model in prism_models:
            # PRISM variants: no p-value comparison needed
            spearman_pvalues[model] = 1.0
            pearson_pvalues[model] = 1.0
        else:
            # Baseline models: compare with best PRISM
            model_data = df[model].values

            spearman_pvalues[model] = paired_bootstrap_test(
                best_prism_data, model_data, y, method='spearman', n_bootstrap=args.n_bootstrap
            )
            pearson_pvalues[model] = paired_bootstrap_test(
                best_prism_data, model_data, y, method='pearson', n_bootstrap=args.n_bootstrap
            )

    # Print results
    print("=" * 80)
    print("SPEARMAN CORRELATION RESULTS")
    print("=" * 80)
    print(f"{'Model':<15} {'Correlation':>12} {'95% CI':>20} {'p-value':>15} {'vs Best PRISM':>15}")
    print("-" * 80)
    for model, display in zip(model_columns, display_names):
        corr, ci_low, ci_high, p = spearman_correlations[model]
        vs_evo = format_pvalue(spearman_pvalues[model]) if model not in prism_models else "-"
        print(f"{display:<15} {corr:>12.4f} [{ci_low:>8.4f}, {ci_high:>8.4f}] {format_pvalue(p):>15} {vs_evo:>15}")

    print("\n" + "=" * 80)
    print("PEARSON CORRELATION RESULTS")
    print("=" * 80)
    print(f"{'Model':<15} {'Correlation':>12} {'95% CI':>20} {'p-value':>15} {'vs Best PRISM':>15}")
    print("-" * 80)
    for model, display in zip(model_columns, display_names):
        corr, ci_low, ci_high, p = pearson_correlations[model]
        vs_evo = format_pvalue(pearson_pvalues[model]) if model not in prism_models else "-"
        print(f"{display:<15} {corr:>12.4f} [{ci_low:>8.4f}, {ci_high:>8.4f}] {format_pvalue(p):>15} {vs_evo:>15}")

    # Create figure
    if args.pearson_only:
        # Single Pearson plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        if args.title:
            fig.suptitle(args.title, fontsize=28, fontweight='bold', y=0.98)
        create_correlation_barplot(
            pearson_correlations,
            pearson_pvalues,
            "Pearson r",
            model_columns,
            display_names,
            colors,
            ax,
            n_prism_variants=n_prism_variants
        )
        plt.tight_layout()
        if args.title:
            plt.subplots_adjust(top=0.93)
        suffix = "_zero_shot_results_pearson"
    else:
        # Two subplots (Spearman + Pearson)
        if args.vertical_layout:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 14))
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        if args.title:
            fig.suptitle(args.title, fontsize=28, fontweight='bold', y=0.98)
        create_correlation_barplot(
            spearman_correlations,
            spearman_pvalues,
            "Spearman ρ",
            model_columns,
            display_names,
            colors,
            ax1,
            n_prism_variants=n_prism_variants
        )
        create_correlation_barplot(
            pearson_correlations,
            pearson_pvalues,
            "Pearson r",
            model_columns,
            display_names,
            colors,
            ax2,
            n_prism_variants=n_prism_variants
        )
        plt.tight_layout()
        if args.title:
            plt.subplots_adjust(top=0.93)
        suffix = "_zero_shot_results"

    # Determine output path
    os.makedirs(args.output_dir, exist_ok=True)

    # Extract first part of filename before "_" for naming
    # e.g., "g6.31_benchmark_data_output.csv" -> "g6.31"
    base_name = os.path.splitext(os.path.basename(args.csv_path))[0]
    first_part = base_name.split('_')[0]

    output_path = os.path.join(args.output_dir, f"{first_part}{suffix}.png")

    # Save figure as PNG
    fig.savefig(output_path, dpi=args.dpi, bbox_inches='tight', facecolor='white')
    # Save figure as SVG
    svg_path = output_path.replace('.png', '.svg')
    fig.savefig(svg_path, format='svg', bbox_inches='tight', facecolor='white')

    print(f"\n✓ Figure saved to: {output_path}")
    print(f"✓ Figure saved to: {svg_path}")

    plt.close(fig)


if __name__ == '__main__':
    main()
