#!/usr/bin/env python
# coding: utf-8
"""
Top-K Accuracy Comparison with Best Strategy per Region

This script generates publication-quality figures comparing PRISM (using optimal
strategy per region) against baseline models.

Best Strategies (from comprehensive sweep):
    - Overall: Final_upper (7.05% top-1, 34.45% top-5)
    - CDR: Final_upper (9.93% top-1, 40.47% top-5)
    - FR: Final_lower (5.56% top-1, 32.64% top-5)

Author: DevAnt-LM Team
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Configure matplotlib for publication-quality figures
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'

# Unified Font Configuration
FONT_CONFIG = {
    'axis_label': {'fontsize': 25, 'fontweight': 'bold'},
    'tick_label': {'fontsize': 15},
    'legend': {'fontsize': 18},
    'title': {'fontsize': 25, 'fontweight': 'bold'},
    'text': {'fontsize': 20},
}

# Standard amino acids
AMINO_ACIDS = list('ACDEFGHIKLMNPQRSTVWY')

# IMGT Region IDs
FR_REGION_IDS = {'0', '2', '4', '6'}
CDR_REGION_IDS = {'1', '3', '5'}

# Paul Tol's colorblind-friendly palette
MODEL_COLORS = {
    'PRISM': '#332288',       # Dark purple
    'ESM2_35M': '#DDCC77',    # Sand/Yellow
    'ESM2_650M': '#117733',   # Green
    'AbLang2': '#88CCEE',     # Light blue
    'AntiBERTy': '#44AA99',   # Teal
    'Sapiens': '#882255',     # Wine/Dark magenta
}

# Model display order
MODEL_ORDER = ['PRISM', 'ESM2_35M', 'ESM2_650M', 'AbLang2', 'AntiBERTy', 'Sapiens']

# Best strategy per region (from comprehensive sweep)
BEST_STRATEGY = {
    'all': 'upper',   # Final_upper
    'CDR': 'upper',   # Final_upper
    'FR': 'lower',    # Final_lower
}


def save_figure_with_svg(fig: plt.Figure, output_path: str, dpi: int = 300):
    """Save figure in both PNG and SVG formats."""
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    svg_path = output_path.rsplit('.png', 1)[0] + '.svg' if output_path.endswith('.png') else output_path + '.svg'
    fig.savefig(svg_path, format='svg', bbox_inches='tight', facecolor='white')
    return svg_path


def get_region_id_at_position(row: pd.Series) -> Optional[str]:
    """Get the region ID for a specific mutation position."""
    chain = row.get('chain', '')
    pos = row.get('position', 0)

    if chain == 'heavy':
        region_mask = row.get('region_mask_heavy', '')
    elif chain == 'light':
        region_mask = row.get('region_mask_light', '')
    else:
        return None

    if pd.isna(region_mask) or not region_mask:
        return None

    idx = int(pos) - 1
    if 0 <= idx < len(region_mask):
        return region_mask[idx]
    return None


def filter_by_region(df: pd.DataFrame, region_type: str) -> pd.DataFrame:
    """Filter dataframe to only include mutations in specified region type."""
    if region_type == 'all':
        return df

    filtered_indices = []
    for idx in range(len(df)):
        row = df.iloc[idx]
        region_id = get_region_id_at_position(row)

        if region_id is None:
            continue

        if region_type == 'CDR' and region_id in CDR_REGION_IDS:
            filtered_indices.append(idx)
        elif region_type == 'FR' and region_id in FR_REGION_IDS:
            filtered_indices.append(idx)

    return df.iloc[filtered_indices].reset_index(drop=True)


def calculate_topk_accuracy(logits: np.ndarray, true_indices: np.ndarray, k: int) -> float:
    """Calculate top-k accuracy."""
    if len(logits) == 0:
        return np.nan
    sorted_indices = np.argsort(-logits, axis=1)[:, :k]
    correct = np.any(sorted_indices == true_indices[:, np.newaxis], axis=1)
    return np.mean(correct)


def calculate_per_antibody_accuracy(
    df: pd.DataFrame,
    logit_cols: List[str],
    k: int
) -> Tuple[np.ndarray, int]:
    """Calculate per-antibody accuracy."""
    antibodies = df['Therapeutic'].unique()
    accuracies = []

    # Get true indices for mutated AA
    true_indices = np.array([
        AMINO_ACIDS.index(aa) if aa in AMINO_ACIDS else -1
        for aa in df['mutated_aa'].values
    ])

    for ab in antibodies:
        ab_mask = df['Therapeutic'] == ab
        ab_true = true_indices[ab_mask]

        # Skip if no valid indices
        valid_mask = ab_true >= 0
        if not np.any(valid_mask):
            continue

        ab_logits = df.loc[ab_mask, logit_cols].values[valid_mask]
        ab_true_valid = ab_true[valid_mask]

        if len(ab_logits) > 0:
            acc = calculate_topk_accuracy(ab_logits, ab_true_valid, k)
            if not np.isnan(acc):
                accuracies.append(acc)

    return np.array(accuracies), len(accuracies)


def get_logit_columns_for_prism(strategy: str) -> List[str]:
    """Get logit column names for PRISM based on strategy."""
    if strategy == 'upper':
        return [f'{aa}_upper' for aa in AMINO_ACIDS]
    else:  # lower
        return [f'{aa}_lower' for aa in AMINO_ACIDS]


def load_and_compute_results(
    baseline_csv: str,
    prism_csv: str,
    region_type: str,
    k_values: List[int] = [1, 3, 5]
) -> Dict[str, Dict[int, Tuple[float, float, int]]]:
    """
    Load data and compute per-antibody accuracy for all models.

    Returns:
        Dict[model_name, Dict[k, (mean_acc, std_acc, n_antibodies)]]
    """
    results = {}

    # Load baseline data
    baseline_df = pd.read_csv(baseline_csv)
    baseline_models = baseline_df['model'].unique()

    # Filter baseline by region
    baseline_filtered = filter_by_region(baseline_df, region_type)
    print(f"  Baseline: {len(baseline_filtered)} mutations after {region_type} filter")

    # Calculate for each baseline model
    aa_cols = list(AMINO_ACIDS)
    for model in baseline_models:
        model_df = baseline_filtered[baseline_filtered['model'] == model]
        if len(model_df) == 0:
            continue

        results[model] = {}
        for k in k_values:
            accs, n_ab = calculate_per_antibody_accuracy(model_df, aa_cols, k)
            if len(accs) > 0:
                results[model][k] = (np.mean(accs), np.std(accs), n_ab)
            else:
                results[model][k] = (np.nan, np.nan, 0)

    # Load PRISM data
    prism_df = pd.read_csv(prism_csv)
    prism_filtered = filter_by_region(prism_df, region_type)
    print(f"  PRISM: {len(prism_filtered)} mutations after {region_type} filter")

    # Get best strategy for this region
    strategy = BEST_STRATEGY.get(region_type, 'upper')
    prism_cols = get_logit_columns_for_prism(strategy)

    print(f"  Using strategy: Final_{strategy}")

    results['PRISM'] = {}
    for k in k_values:
        accs, n_ab = calculate_per_antibody_accuracy(prism_filtered, prism_cols, k)
        if len(accs) > 0:
            results['PRISM'][k] = (np.mean(accs), np.std(accs), n_ab)
        else:
            results['PRISM'][k] = (np.nan, np.nan, 0)

    return results


def create_topk_bar_plot(
    results: Dict[str, Dict[int, Tuple[float, float, int]]],
    k_values: List[int],
    output_path: str,
    region_name: str,
    dpi: int = 300
):
    """Create a grouped bar plot showing top-k accuracy for each model."""
    fig, ax = plt.subplots(figsize=(14, 8))

    # Sort models by MODEL_ORDER
    sorted_models = []
    for model in MODEL_ORDER:
        if model in results:
            sorted_models.append(model)
    for model in results:
        if model not in sorted_models:
            sorted_models.append(model)

    n_models = len(sorted_models)
    n_k = len(k_values)
    bar_width = 0.25
    x = np.arange(n_models)

    # Color shades for different k values
    alpha_values = [1.0, 0.7, 0.5]

    # Plot bars for each k value
    for i, k in enumerate(k_values):
        accuracies = []
        errors = []
        colors = []

        for model in sorted_models:
            if k in results[model]:
                acc, std, _ = results[model][k]
                accuracies.append(acc if not np.isnan(acc) else 0)
                errors.append(std if not np.isnan(std) else 0)
            else:
                accuracies.append(0)
                errors.append(0)
            colors.append(MODEL_COLORS.get(model, '#999999'))

        offset = (i - n_k/2 + 0.5) * bar_width

        bars = ax.bar(
            x + offset,
            accuracies,
            bar_width,
            label=f'Top-{k}',
            color=colors,
            edgecolor='black',
            linewidth=1.5,
            yerr=errors,
            capsize=3,
            alpha=alpha_values[i]
        )

    # Customize axes
    ax.set_xlabel('Model', **FONT_CONFIG['axis_label'])
    ax.set_ylabel('Accuracy', **FONT_CONFIG['axis_label'])

    ax.set_xticks(x)
    ax.set_xticklabels(sorted_models, fontsize=20, fontweight='bold', rotation=45, ha='right')
    ax.set_ylim(0, 0.7)  # Adjusted for visibility
    ax.tick_params(axis='y', labelsize=FONT_CONFIG['tick_label']['fontsize'])

    # Add grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)

    # Add legend for k values
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='gray', edgecolor='black', alpha=alpha_values[i], label=f'Top-{k}')
        for i, k in enumerate(k_values)
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=FONT_CONFIG['legend']['fontsize'])

    # Thicken spines
    for spine in ax.spines.values():
        spine.set_linewidth(2)

    plt.tight_layout()
    svg_path = save_figure_with_svg(fig, output_path, dpi=dpi)
    plt.close(fig)

    print(f"✓ Bar plot saved to: {output_path}")
    print(f"✓ Bar plot (SVG) saved to: {svg_path}")


def create_combined_region_plot(
    all_results: Dict[str, Dict[str, Dict[int, Tuple[float, float, int]]]],
    k_values: List[int],
    output_path: str,
    dpi: int = 300
):
    """Create a combined figure with all three regions as subplots."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    region_names = ['Overall (All)', 'CDR Regions', 'FR Regions']
    region_keys = ['all', 'CDR', 'FR']

    for ax_idx, (ax, region_name, region_key) in enumerate(zip(axes, region_names, region_keys)):
        results = all_results[region_key]

        # Sort models by MODEL_ORDER
        sorted_models = []
        for model in MODEL_ORDER:
            if model in results:
                sorted_models.append(model)
        for model in results:
            if model not in sorted_models:
                sorted_models.append(model)

        n_models = len(sorted_models)
        n_k = len(k_values)
        bar_width = 0.25
        x = np.arange(n_models)

        alpha_values = [1.0, 0.7, 0.5]

        for i, k in enumerate(k_values):
            accuracies = []
            errors = []
            colors = []

            for model in sorted_models:
                if k in results[model]:
                    acc, std, _ = results[model][k]
                    accuracies.append(acc if not np.isnan(acc) else 0)
                    errors.append(std if not np.isnan(std) else 0)
                else:
                    accuracies.append(0)
                    errors.append(0)
                colors.append(MODEL_COLORS.get(model, '#999999'))

            offset = (i - n_k/2 + 0.5) * bar_width

            ax.bar(
                x + offset,
                accuracies,
                bar_width,
                label=f'Top-{k}' if ax_idx == 0 else '',
                color=colors,
                edgecolor='black',
                linewidth=1.0,
                yerr=errors,
                capsize=2,
                alpha=alpha_values[i]
            )

        ax.set_title(region_name, fontsize=22, fontweight='bold')
        ax.set_xlabel('Model', fontsize=18, fontweight='bold')
        if ax_idx == 0:
            ax.set_ylabel('Accuracy', fontsize=18, fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(sorted_models, fontsize=12, fontweight='bold', rotation=45, ha='right')
        ax.set_ylim(0, 0.7)
        ax.tick_params(axis='y', labelsize=12)

        ax.yaxis.grid(True, linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)

        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

    # Add single legend for all subplots
    from matplotlib.patches import Patch
    alpha_values = [1.0, 0.7, 0.5]
    legend_elements = [
        Patch(facecolor='gray', edgecolor='black', alpha=alpha_values[i], label=f'Top-{k}')
        for i, k in enumerate(k_values)
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=3, fontsize=16,
               bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    svg_path = save_figure_with_svg(fig, output_path, dpi=dpi)
    plt.close(fig)

    print(f"✓ Combined plot saved to: {output_path}")
    print(f"✓ Combined plot (SVG) saved to: {svg_path}")


def print_results_table(all_results: Dict[str, Dict[str, Dict[int, Tuple[float, float, int]]]]):
    """Print results in a formatted table."""
    print("\n" + "=" * 100)
    print("TOP-K ACCURACY RESULTS (Per-Antibody Mean ± Std)")
    print("=" * 100)

    for region_key, region_name in [('all', 'Overall'), ('CDR', 'CDR'), ('FR', 'FR')]:
        print(f"\n--- {region_name} Region ---")
        print(f"{'Model':<15} {'Top-1':<20} {'Top-3':<20} {'Top-5':<20} {'N':<6}")
        print("-" * 81)

        results = all_results[region_key]

        # Sort by MODEL_ORDER
        sorted_models = []
        for model in MODEL_ORDER:
            if model in results:
                sorted_models.append(model)

        for model in sorted_models:
            top1 = results[model].get(1, (np.nan, np.nan, 0))
            top3 = results[model].get(3, (np.nan, np.nan, 0))
            top5 = results[model].get(5, (np.nan, np.nan, 0))

            print(f"{model:<15} {top1[0]*100:5.2f}% ± {top1[1]*100:5.2f}%    "
                  f"{top3[0]*100:5.2f}% ± {top3[1]*100:5.2f}%    "
                  f"{top5[0]*100:5.2f}% ± {top5[1]*100:5.2f}%    {top1[2]:<6}")

        print("-" * 81)

    print("\n" + "=" * 100)


def main():
    parser = argparse.ArgumentParser(
        description='Plot top-k accuracy comparison with best strategy per region'
    )
    parser.add_argument('--baseline_csv', type=str,
                        default='data/therasabdab_baseline_logits.csv',
                        help='Path to baseline logits CSV')
    parser.add_argument('--prism_csv', type=str,
                        default='data/therasabdab_evo_ab_logits.csv',
                        help='Path to PRISM logits CSV')
    parser.add_argument('--output_dir', type=str,
                        default='img/4.thera-sabdab',
                        help='Output directory for figures')
    args = parser.parse_args()

    print("=" * 80)
    print("Top-K Accuracy Comparison with Best Strategy per Region")
    print("=" * 80)
    print("\nBest Strategies:")
    print("  - Overall: Final_upper")
    print("  - CDR: Final_upper")
    print("  - FR: Final_lower")

    # Compute results for all regions
    all_results = {}
    k_values = [1, 3, 5]

    for region_key in ['all', 'CDR', 'FR']:
        print(f"\n--- Processing {region_key} region ---")
        all_results[region_key] = load_and_compute_results(
            args.baseline_csv,
            args.prism_csv,
            region_key,
            k_values
        )

    # Print results table
    print_results_table(all_results)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate individual plots for each region
    for region_key, region_name in [('all', 'overall'), ('CDR', 'CDR'), ('FR', 'FR')]:
        output_path = os.path.join(args.output_dir, f'topk_accuracy_{region_name}_best.png')
        create_topk_bar_plot(
            all_results[region_key],
            k_values,
            output_path,
            region_name
        )

    # Create combined plot
    combined_path = os.path.join(args.output_dir, 'topk_accuracy_all_regions_best.png')
    create_combined_region_plot(all_results, k_values, combined_path)

    print("\n✓ All figures generated successfully!")


if __name__ == "__main__":
    main()
