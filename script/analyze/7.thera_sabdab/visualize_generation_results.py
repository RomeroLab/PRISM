#!/usr/bin/env python
# coding: utf-8

"""
Visualization for Controllable Generation Experiment Results

This script creates comprehensive visualizations comparing PRISM's controllable
generation modes against baseline models.

Figures:
1. Box plots: Metric comparisons across models/modes
2. Heatmaps: Mutation position preferences per CDR
3. Bar charts: CDR mutation ratio comparison
4. Scatter plots: Diversity vs naturalness trade-off
5. Statistical comparison tables

Usage:
    python visualize_generation_results.py \\
        --results_csv results/controllable_generation/evaluation_metrics.csv \\
        --output_dir results/controllable_generation/figures
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from matplotlib.gridspec import GridSpec

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Color scheme for models/modes
MODEL_COLORS = {
    # PRISM modes
    'PRISM_gl': '#1f77b4',        # Blue
    'PRISM_ngl': '#ff7f0e',       # Orange
    'PRISM_final': '#2ca02c',     # Green
    'PRISM_region_specific': '#d62728',  # Red
    # Baselines (with _standard suffix for baseline-only runs)
    'esm2_35m': '#9467bd',        # Purple
    'esm2_650m': '#8c564b',       # Brown
    'ablang2_heavy': '#e377c2',   # Pink
    'ablang2_light': '#bcbd22',   # Yellow-green
    'antiberty': '#7f7f7f',       # Gray
    'sapiens': '#17becf',         # Cyan
    # Standard mode versions (baseline-only experiments)
    'esm2_35m_standard': '#9467bd',
    'esm2_650m_standard': '#8c564b',
    'ablang2_heavy_standard': '#e377c2',
    'ablang2_light_standard': '#bcbd22',
    'antiberty_standard': '#7f7f7f',
    'sapiens_standard': '#17becf',
}

# Hatching patterns for modes
MODE_HATCHES = {
    'gl': '',
    'ngl': '//',
    'final': '\\\\',
    'region_specific': 'xx',
    'standard': '..',
}


def load_results(results_path: str) -> pd.DataFrame:
    """Load evaluation metrics from CSV."""
    df = pd.read_csv(results_path)
    # Create combined model_mode column for easier plotting
    df['model_mode'] = df['model'] + '_' + df['mode']
    return df


def plot_metric_comparison_boxplots(
    df: pd.DataFrame,
    output_dir: str,
    metrics: List[str] = None,
):
    """
    Create box plots comparing metrics across models/modes.

    Args:
        df: Results DataFrame
        output_dir: Output directory
        metrics: Metrics to plot (default: key metrics)
    """
    if metrics is None:
        metrics = [
            ('diversity_mean', 'Sequence Diversity', 'higher is more diverse'),
            ('germline_dist_mean', 'Germline Distance', 'higher is more mutated'),
            ('cdr_mutation_ratio', 'CDR Mutation Ratio', 'higher = more CDR-focused'),
            ('naturalness_mean', 'Naturalness (PPL)', 'lower is more natural'),
        ]

    # Order for x-axis - dynamically build from available models
    # Preferred order: PRISM modes first, then baselines
    preferred_order = ['PRISM_gl', 'PRISM_ngl', 'PRISM_final', 'PRISM_region_specific',
                       'esm2_35m_standard', 'esm2_650m_standard', 'ablang2_heavy_standard',
                       'ablang2_light_standard', 'antiberty_standard', 'sapiens_standard',
                       'esm2_35m', 'esm2_650m', 'ablang2_heavy']
    available = df['model_mode'].unique().tolist()
    order = [o for o in preferred_order if o in available]
    # Add any remaining models not in preferred order
    order.extend([m for m in available if m not in order])

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for idx, (metric, title, subtitle) in enumerate(metrics):
        if idx >= len(axes):
            break

        ax = axes[idx]

        # Filter valid values
        plot_df = df[['model_mode', metric]].dropna()
        if len(plot_df) == 0:
            continue

        # Create box plot
        plot_order = [o for o in order if o in plot_df['model_mode'].unique()]
        sns.boxplot(
            data=plot_df,
            x='model_mode',
            y=metric,
            hue='model_mode',
            order=plot_order,
            hue_order=plot_order,
            palette=MODEL_COLORS,
            ax=ax,
            legend=False,
        )

        ax.set_title(f'{title}\n({subtitle})', fontsize=12, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel(title)

        # Rotate x-axis labels
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/metric_comparison_boxplots.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{output_dir}/metric_comparison_boxplots.svg', bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved metric_comparison_boxplots")


def plot_cdr_mutation_ratio_bar(
    df: pd.DataFrame,
    output_dir: str,
):
    """
    Create bar chart comparing CDR mutation ratios.

    This is the key visualization showing PRISM's ability to focus
    mutations in CDR regions.
    """
    # Aggregate by model_mode
    agg_df = df.groupby('model_mode').agg({
        'cdr_mutation_ratio': ['mean', 'std', 'count'],
    }).reset_index()
    agg_df.columns = ['model_mode', 'mean', 'std', 'count']

    # Sort by CDR ratio
    agg_df = agg_df.sort_values('mean', ascending=True)

    # Order for consistent plotting - dynamically handle available models
    preferred_order = ['PRISM_region_specific', 'PRISM_ngl', 'PRISM_final', 'PRISM_gl',
                       'esm2_35m_standard', 'esm2_650m_standard', 'ablang2_heavy_standard',
                       'ablang2_light_standard', 'antiberty_standard', 'sapiens_standard',
                       'esm2_35m', 'esm2_650m', 'ablang2_heavy']
    available = agg_df['model_mode'].values.tolist()
    order = [o for o in preferred_order if o in available]
    order.extend([m for m in available if m not in order])

    fig, ax = plt.subplots(figsize=(10, 6))

    # Create horizontal bar chart
    y_pos = np.arange(len(order))
    means = [agg_df[agg_df['model_mode'] == o]['mean'].values[0] if o in agg_df['model_mode'].values else 0 for o in order]
    stds = [agg_df[agg_df['model_mode'] == o]['std'].values[0] if o in agg_df['model_mode'].values else 0 for o in order]

    colors = [MODEL_COLORS.get(o, '#333333') for o in order]

    bars = ax.barh(y_pos, means, xerr=stds, capsize=3, color=colors, edgecolor='black', linewidth=1)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(order)
    ax.set_xlabel('CDR Mutation Ratio (higher = more mutations in CDR)', fontsize=11)
    ax.set_title('CDR-Focused Mutation Generation\n(PRISM Region-Specific vs Baselines)',
                fontsize=12, fontweight='bold')

    # Add vertical line at 0.5 (random baseline)
    ax.axvline(x=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.text(0.51, len(order) - 0.5, 'random baseline', fontsize=9, alpha=0.7)

    ax.set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/cdr_mutation_ratio_bar.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{output_dir}/cdr_mutation_ratio_bar.svg', bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved cdr_mutation_ratio_bar")


def plot_diversity_naturalness_scatter(
    df: pd.DataFrame,
    output_dir: str,
):
    """
    Scatter plot showing diversity vs naturalness trade-off.

    Ideal: high diversity (varied sequences) + low naturalness (natural sequences).
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot each model/mode
    for model_mode in df['model_mode'].unique():
        subset = df[df['model_mode'] == model_mode]
        color = MODEL_COLORS.get(model_mode, '#333333')

        ax.scatter(
            subset['diversity_mean'],
            subset['naturalness_mean'],
            c=[color] * len(subset),
            label=model_mode,
            s=100,
            alpha=0.7,
            edgecolors='black',
            linewidth=0.5,
        )

    ax.set_xlabel('Sequence Diversity (higher = more varied)', fontsize=11)
    ax.set_ylabel('Naturalness (PPL, lower = more natural)', fontsize=11)
    ax.set_title('Diversity vs Naturalness Trade-off', fontsize=12, fontweight='bold')

    # Add ideal region annotation
    ax.annotate('Ideal Region\n(high diversity, low PPL)',
               xy=(0.8 * ax.get_xlim()[1], 0.2 * ax.get_ylim()[1]),
               fontsize=10, alpha=0.6,
               ha='center',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/diversity_naturalness_scatter.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{output_dir}/diversity_naturalness_scatter.svg', bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved diversity_naturalness_scatter")


def plot_mutation_distribution_heatmap(
    df: pd.DataFrame,
    output_dir: str,
):
    """
    Heatmap showing mutation counts in different regions.
    """
    # Prepare data for heatmap
    regions = ['fr_mutations_mean', 'cdr1_mutations_mean', 'cdr2_mutations_mean', 'cdr3_mutations_mean']
    region_labels = ['FR', 'CDR1', 'CDR2', 'CDR3']

    # Aggregate by model_mode
    agg_data = df.groupby('model_mode')[regions].mean()

    # Reorder rows - dynamically handle available models
    preferred_order = ['PRISM_gl', 'PRISM_ngl', 'PRISM_final', 'PRISM_region_specific',
                       'esm2_35m_standard', 'esm2_650m_standard', 'ablang2_heavy_standard',
                       'ablang2_light_standard', 'antiberty_standard', 'sapiens_standard',
                       'esm2_35m', 'esm2_650m', 'ablang2_heavy']
    available = agg_data.index.tolist()
    order = [o for o in preferred_order if o in available]
    order.extend([m for m in available if m not in order])
    agg_data = agg_data.reindex(order)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Create heatmap
    sns.heatmap(
        agg_data,
        annot=True,
        fmt='.2f',
        cmap='YlOrRd',
        xticklabels=region_labels,
        ax=ax,
        cbar_kws={'label': 'Mean Mutation Count'},
    )

    ax.set_title('Mutation Distribution by Region', fontsize=12, fontweight='bold')
    ax.set_xlabel('Region')
    ax.set_ylabel('Model / Mode')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/mutation_distribution_heatmap.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{output_dir}/mutation_distribution_heatmap.svg', bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved mutation_distribution_heatmap")


def plot_summary_table(
    df: pd.DataFrame,
    output_dir: str,
):
    """
    Create summary statistics table as a figure.
    """
    # Aggregate metrics
    metrics = ['diversity_mean', 'germline_dist_mean', 'cdr_mutation_ratio',
               'naturalness_mean', 'uniqueness_ratio']
    metric_labels = ['Diversity', 'Germline Dist.', 'CDR Ratio', 'PPL', 'Uniqueness']

    summary = df.groupby('model_mode')[metrics].agg(['mean', 'std'])

    # Order rows - dynamically handle available models
    preferred_order = ['PRISM_gl', 'PRISM_ngl', 'PRISM_final', 'PRISM_region_specific',
                       'esm2_35m_standard', 'esm2_650m_standard', 'ablang2_heavy_standard',
                       'ablang2_light_standard', 'antiberty_standard', 'sapiens_standard',
                       'esm2_35m', 'esm2_650m', 'ablang2_heavy']
    available = summary.index.tolist()
    order = [o for o in preferred_order if o in available]
    order.extend([m for m in available if m not in order])
    summary = summary.reindex(order)

    # Create table figure
    fig, ax = plt.subplots(figsize=(14, len(order) * 0.6 + 2))
    ax.axis('off')

    # Format data for table
    cell_text = []
    for model_mode in order:
        row = []
        for metric in metrics:
            mean = summary.loc[model_mode, (metric, 'mean')]
            std = summary.loc[model_mode, (metric, 'std')]
            row.append(f'{mean:.3f} ± {std:.3f}')
        cell_text.append(row)

    table = ax.table(
        cellText=cell_text,
        rowLabels=order,
        colLabels=metric_labels,
        loc='center',
        cellLoc='center',
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Style header row
    for i in range(len(metric_labels)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    # Style row labels
    for i in range(1, len(order) + 1):
        table[(i, -1)].set_facecolor('#D9E2F3')

    ax.set_title('Summary Statistics (Mean ± Std)', fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/summary_table.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{output_dir}/summary_table.svg', bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved summary_table")


def plot_statistical_comparison(
    df: pd.DataFrame,
    output_dir: str,
):
    """
    Run statistical tests and create comparison figure.
    """
    metrics_to_test = ['diversity_mean', 'germline_dist_mean', 'cdr_mutation_ratio']

    # Compare PRISM region_specific vs baselines
    prism_rs = df[df['model_mode'] == 'PRISM_region_specific']
    baseline_names = ['esm2_35m_standard', 'esm2_650m_standard', 'ablang2_heavy_standard',
                      'ablang2_light_standard', 'antiberty_standard', 'sapiens_standard',
                      'esm2_35m', 'esm2_650m', 'ablang2_heavy']

    results = []
    for baseline in baseline_names:
        baseline_df = df[df['model_mode'] == baseline]
        if len(baseline_df) == 0:
            continue

        for metric in metrics_to_test:
            prism_vals = prism_rs[metric].dropna().values
            baseline_vals = baseline_df[metric].dropna().values

            if len(prism_vals) > 0 and len(baseline_vals) > 0:
                # Mann-Whitney U test
                stat, pval = stats.mannwhitneyu(prism_vals, baseline_vals, alternative='two-sided')
                effect_size = (np.median(prism_vals) - np.median(baseline_vals)) / np.std(baseline_vals + 1e-8)

                results.append({
                    'comparison': f'PRISM_RS vs {baseline}',
                    'metric': metric,
                    'prism_median': np.median(prism_vals),
                    'baseline_median': np.median(baseline_vals),
                    'p_value': pval,
                    'effect_size': effect_size,
                    'significant': pval < 0.05,
                })

    if len(results) == 0:
        print("  ⚠ Not enough data for statistical comparison")
        return

    results_df = pd.DataFrame(results)

    # Save to CSV
    results_df.to_csv(f'{output_dir}/statistical_comparison.csv', index=False)

    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')

    # Format for table
    cell_text = []
    for _, row in results_df.iterrows():
        sig_marker = '***' if row['p_value'] < 0.001 else ('**' if row['p_value'] < 0.01 else ('*' if row['p_value'] < 0.05 else ''))
        cell_text.append([
            row['comparison'],
            row['metric'],
            f"{row['prism_median']:.3f}",
            f"{row['baseline_median']:.3f}",
            f"{row['p_value']:.4f}{sig_marker}",
            f"{row['effect_size']:.2f}",
        ])

    table = ax.table(
        cellText=cell_text,
        colLabels=['Comparison', 'Metric', 'PRISM Median', 'Baseline Median', 'p-value', 'Effect Size'],
        loc='center',
        cellLoc='center',
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Style header
    for i in range(6):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    ax.set_title('Statistical Comparison (Mann-Whitney U Test)\n* p<0.05, ** p<0.01, *** p<0.001',
                fontsize=12, fontweight='bold', y=0.95)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/statistical_comparison.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{output_dir}/statistical_comparison.svg', bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved statistical_comparison")


def plot_combined_figure(
    df: pd.DataFrame,
    output_dir: str,
):
    """
    Create a combined figure for publication.
    """
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    # Panel A: CDR Mutation Ratio Bar
    ax1 = fig.add_subplot(gs[0, 0])
    preferred_order = ['PRISM_region_specific', 'PRISM_ngl', 'PRISM_final', 'PRISM_gl',
                       'esm2_35m_standard', 'esm2_650m_standard', 'ablang2_heavy_standard',
                       'ablang2_light_standard', 'antiberty_standard', 'sapiens_standard',
                       'esm2_35m', 'esm2_650m']
    available = df['model_mode'].unique().tolist()
    order = [o for o in preferred_order if o in available]
    order.extend([m for m in available if m not in order])

    agg_df = df.groupby('model_mode')['cdr_mutation_ratio'].agg(['mean', 'std']).reset_index()
    y_pos = np.arange(len(order))
    means = [agg_df[agg_df['model_mode'] == o]['mean'].values[0] if o in agg_df['model_mode'].values else 0 for o in order]
    stds = [agg_df[agg_df['model_mode'] == o]['std'].values[0] if o in agg_df['model_mode'].values else 0 for o in order]
    colors = [MODEL_COLORS.get(o, '#333333') for o in order]

    ax1.barh(y_pos, means, xerr=stds, capsize=3, color=colors, edgecolor='black', linewidth=1)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(order, fontsize=9)
    ax1.set_xlabel('CDR Mutation Ratio')
    ax1.set_title('A. CDR-Focused Mutation Generation', fontsize=11, fontweight='bold')
    ax1.axvline(x=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax1.set_xlim(0, 1)

    # Panel B: Diversity Comparison
    ax2 = fig.add_subplot(gs[0, 1])
    plot_order = [o for o in order if o in df['model_mode'].unique()]
    sns.boxplot(data=df, x='model_mode', y='diversity_mean', hue='model_mode',
               order=plot_order, hue_order=plot_order, palette=MODEL_COLORS, ax=ax2, legend=False)
    ax2.set_xlabel('')
    ax2.set_ylabel('Diversity')
    ax2.set_title('B. Sequence Diversity', fontsize=11, fontweight='bold')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right', fontsize=9)

    # Panel C: Naturalness (PPL)
    ax3 = fig.add_subplot(gs[1, 0])
    sns.boxplot(data=df, x='model_mode', y='naturalness_mean', hue='model_mode',
               order=plot_order, hue_order=plot_order, palette=MODEL_COLORS, ax=ax3, legend=False)
    ax3.set_xlabel('')
    ax3.set_ylabel('Pseudo-Perplexity')
    ax3.set_title('C. Naturalness Score', fontsize=11, fontweight='bold')
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right', fontsize=9)

    # Panel D: Diversity vs Naturalness scatter
    ax4 = fig.add_subplot(gs[1, 1])
    for model_mode in df['model_mode'].unique():
        subset = df[df['model_mode'] == model_mode]
        color = MODEL_COLORS.get(model_mode, '#333333')
        ax4.scatter(subset['diversity_mean'], subset['naturalness_mean'],
                   c=[color] * len(subset), label=model_mode, s=60, alpha=0.7,
                   edgecolors='black', linewidth=0.5)

    ax4.set_xlabel('Diversity')
    ax4.set_ylabel('Naturalness (PPL)')
    ax4.set_title('D. Diversity-Naturalness Trade-off', fontsize=11, fontweight='bold')
    ax4.legend(fontsize=8, loc='upper right')

    plt.savefig(f'{output_dir}/combined_figure.png', dpi=200, bbox_inches='tight')
    plt.savefig(f'{output_dir}/combined_figure.svg', bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved combined_figure")


def main(results_csv: str, output_dir: str):
    """
    Generate all visualizations.

    Args:
        results_csv: Path to evaluation_metrics.csv
        output_dir: Output directory for figures
    """
    print("=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load results
    print("\nLoading results...")
    df = load_results(results_csv)
    print(f"  Loaded {len(df)} rows, {df['model_mode'].nunique()} model/modes")
    print(f"  Models: {df['model_mode'].unique().tolist()}")

    # Generate visualizations
    print("\nGenerating figures...")

    plot_metric_comparison_boxplots(df, output_dir)
    plot_cdr_mutation_ratio_bar(df, output_dir)
    plot_diversity_naturalness_scatter(df, output_dir)
    plot_mutation_distribution_heatmap(df, output_dir)
    plot_summary_table(df, output_dir)
    plot_statistical_comparison(df, output_dir)
    plot_combined_figure(df, output_dir)

    print("\n" + "=" * 60)
    print(f"All figures saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize controllable generation results')
    parser.add_argument('--results_csv', type=str, required=True,
                       help='Path to evaluation_metrics.csv')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for figures')

    args = parser.parse_args()
    main(args.results_csv, args.output_dir)
