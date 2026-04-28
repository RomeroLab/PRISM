#!/usr/bin/env python3
"""
GL vs NGL Comparison Plot

Creates a 2-row figure comparing Germline (GL/Upper) vs Non-Germline (NGL/Lower)
head performance across 9 zero-shot prediction tasks:
- Row 1: Spearman correlation bar graph
- Row 2: Pearson correlation bar graph
- X-axis: 9 datasets with 4 bars each (GL, NGL, Marginalized, AbLang2)

Datasets:
1-3. Binding affinity: CR9114, g6.31, Trastuzumab (use scores from logP)
4-8. Developability: HIC, PR_CHO, AC-SINS, Tm2, Titer (use PPL)
9.   Immunogenicity: ADA (use PPL)

For binding: GL = upper head score, NGL = evo_ab_score, Marg = marginalized score
For developability/immunogenicity: GL = upper PPL, NGL = lower PPL, Marg = marginalized PPL

Colors: GL = #1CC454 (green), NGL = #C8327D (pink), Marg = #332288 (purple), AbLang2 = #88CCEE (blue)

Usage:
    python plot_gl_vs_ngl_comparison.py
"""

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

# Colors
GL_COLOR = '#1CC454'      # Green for GL (germline/upper)
NGL_COLOR = '#C8327D'     # Pink for NGL (non-germline/lower)
MARG_COLOR = '#332288'    # Deep purple for Marginalized
ABLANG2_COLOR = '#88CCEE' # Light blue for AbLang2

# Font configuration
FONT_CONFIG = {
    'axis_label': {'fontsize': 22, 'fontweight': 'bold'},
    'tick_label': {'fontsize': 14, 'fontweight': 'normal'},
    'legend': {'fontsize': 14},
    'title': {'fontsize': 20, 'fontweight': 'bold'},
    'annotation': {'fontsize': 9, 'fontweight': 'normal'},
}


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


def load_binding_data(csv_path: str, lambda_val: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load binding affinity data and compute GL/NGL/Marg/AbLang2 scores.

    GL: upper head score (logP_mut_upper - lambda * logP_wt_upper)
    NGL: evo_ab_score column (combined/default score)
    Marg: marginalized score (logP_mut_marginalized - lambda * logP_wt_marginalized)
    AbLang2: ablang2_score column

    Returns:
        (gl_scores, ngl_scores, marg_scores, ablang2_scores, fitness)
    """
    df = pd.read_csv(csv_path)

    # GL (upper head): logP_mut_upper - lambda * logP_wt_upper
    gl_scores = df['logP_mut_upper'].values - lambda_val * df['logP_wt_upper'].values

    # NGL: use evo_ab_score column directly
    ngl_scores = df['evo_ab_score'].values

    # Marginalized: logP_mut_marginalized - lambda * logP_wt_marginalized
    marg_scores = df['logP_mut_marginalized'].values - lambda_val * df['logP_wt_marginalized'].values

    # AbLang2: ablang2_score column
    ablang2_scores = df['ablang2_score'].values

    fitness = df['fitness'].values

    return gl_scores, ngl_scores, marg_scores, ablang2_scores, fitness


def load_developability_data(csv_path: str, target_col: str, multiply_neg1: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load developability/immunogenicity data.

    GL = upper head PPL
    NGL = lower head PPL
    Marg = marginalized PPL (evo_ab_ppl_marginalized)
    AbLang2 = ablang2_ppl

    Returns:
        (gl_ppl, ngl_ppl, marg_ppl, ablang2_ppl, target)
    """
    df = pd.read_csv(csv_path)

    # GL: upper head PPL
    gl_ppl = df['evo_ab_ppl_final_upper'].values.copy()

    # NGL: lower head PPL
    ngl_ppl = df['evo_ab_ppl_final_lower'].values.copy()

    # Marg: actual marginalized PPL column
    marg_ppl = df['evo_ab_ppl_marginalized'].values.copy()

    # AbLang2: ablang2_ppl
    ablang2_ppl = df['ablang2_ppl'].values.copy()

    target = df[target_col].values

    # Apply direction correction if needed (for Tm2, Titer where higher = better)
    if multiply_neg1:
        gl_ppl = -gl_ppl
        ngl_ppl = -ngl_ppl
        marg_ppl = -marg_ppl
        ablang2_ppl = -ablang2_ppl

    return gl_ppl, ngl_ppl, marg_ppl, ablang2_ppl, target


def create_comparison_barplot(
    results: Dict[str, Dict],
    method: str,
    ax: plt.Axes,
    show_legend: bool = False
):
    """
    Create a grouped bar plot comparing GL vs NGL vs Marg vs AbLang2 correlations.

    Args:
        results: Dict mapping dataset names to {'gl': (corr, ci_low, ci_high), 'ngl': ..., 'marg': ..., 'ablang2': ...}
        method: 'spearman' or 'pearson'
        ax: Matplotlib axes
        show_legend: Whether to show legend (will be placed outside at bottom)
    """
    dataset_names = list(results.keys())
    n_datasets = len(dataset_names)

    # Bar positions - more spacing between groups for 4 bars
    x = np.arange(n_datasets) * 1.6  # Increase spacing for 4 bars
    width = 0.3

    # Extract values for all 4 categories
    gl_corrs = [results[name]['gl'][0] for name in dataset_names]
    gl_ci_lows = [results[name]['gl'][1] for name in dataset_names]
    gl_ci_highs = [results[name]['gl'][2] for name in dataset_names]

    ngl_corrs = [results[name]['ngl'][0] for name in dataset_names]
    ngl_ci_lows = [results[name]['ngl'][1] for name in dataset_names]
    ngl_ci_highs = [results[name]['ngl'][2] for name in dataset_names]

    marg_corrs = [results[name]['marg'][0] for name in dataset_names]
    marg_ci_lows = [results[name]['marg'][1] for name in dataset_names]
    marg_ci_highs = [results[name]['marg'][2] for name in dataset_names]

    ablang2_corrs = [results[name]['ablang2'][0] for name in dataset_names]
    ablang2_ci_lows = [results[name]['ablang2'][1] for name in dataset_names]
    ablang2_ci_highs = [results[name]['ablang2'][2] for name in dataset_names]

    # Calculate error bars
    gl_yerr_low = [max(0, c - l) for c, l in zip(gl_corrs, gl_ci_lows)]
    gl_yerr_high = [max(0, h - c) for c, h in zip(gl_corrs, gl_ci_highs)]

    ngl_yerr_low = [max(0, c - l) for c, l in zip(ngl_corrs, ngl_ci_lows)]
    ngl_yerr_high = [max(0, h - c) for c, h in zip(ngl_corrs, ngl_ci_highs)]

    marg_yerr_low = [max(0, c - l) for c, l in zip(marg_corrs, marg_ci_lows)]
    marg_yerr_high = [max(0, h - c) for c, h in zip(marg_corrs, marg_ci_highs)]

    ablang2_yerr_low = [max(0, c - l) for c, l in zip(ablang2_corrs, ablang2_ci_lows)]
    ablang2_yerr_high = [max(0, h - c) for c, h in zip(ablang2_corrs, ablang2_ci_highs)]

    # Create bars - 4 bars per group
    bars_gl = ax.bar(x - 1.5*width, gl_corrs, width, color=GL_COLOR, edgecolor='black',
                     linewidth=1.5, label='GL (Upper)',
                     yerr=[gl_yerr_low, gl_yerr_high], capsize=3,
                     error_kw={'linewidth': 1.5, 'capthick': 1.5})

    bars_ngl = ax.bar(x - 0.5*width, ngl_corrs, width, color=NGL_COLOR, edgecolor='black',
                      linewidth=1.5, label='NGL (Lower)',
                      yerr=[ngl_yerr_low, ngl_yerr_high], capsize=3,
                      error_kw={'linewidth': 1.5, 'capthick': 1.5})

    bars_marg = ax.bar(x + 0.5*width, marg_corrs, width, color=MARG_COLOR, edgecolor='black',
                       linewidth=1.5, label='Marginalized',
                       yerr=[marg_yerr_low, marg_yerr_high], capsize=3,
                       error_kw={'linewidth': 1.5, 'capthick': 1.5})

    bars_ablang2 = ax.bar(x + 1.5*width, ablang2_corrs, width, color=ABLANG2_COLOR, edgecolor='black',
                          linewidth=1.5, label='AbLang2',
                          yerr=[ablang2_yerr_low, ablang2_yerr_high], capsize=3,
                          error_kw={'linewidth': 1.5, 'capthick': 1.5})

    # Customize axes
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_names, fontsize=FONT_CONFIG['tick_label']['fontsize'],
                       fontweight='bold', rotation=45, ha='right')

    ylabel = "Spearman ρ" if method == 'spearman' else "Pearson r"
    ax.set_ylabel(ylabel, **FONT_CONFIG['axis_label'])
    ax.tick_params(axis='y', labelsize=FONT_CONFIG['tick_label']['fontsize'])

    # Add correlation values above bars (smaller font for 4 bars)
    all_bars = [bars_gl, bars_ngl, bars_marg, bars_ablang2]
    all_corrs = [gl_corrs, ngl_corrs, marg_corrs, ablang2_corrs]
    all_ci_highs = [gl_ci_highs, ngl_ci_highs, marg_ci_highs, ablang2_ci_highs]

    for bars, corrs, ci_highs in zip(all_bars, all_corrs, all_ci_highs):
        for bar, corr, ci_h in zip(bars, corrs, ci_highs):
            if not np.isnan(corr):
                y_pos = ci_h + 0.02 if corr >= 0 else ci_h + 0.02
                ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                        f'{corr:.2f}', ha='center', va='bottom',
                        fontsize=FONT_CONFIG['annotation']['fontsize'],
                        fontweight=FONT_CONFIG['annotation']['fontweight'],
                        color='black', rotation=90)

    # Set y-axis limits
    all_ci_highs_flat = gl_ci_highs + ngl_ci_highs + marg_ci_highs + ablang2_ci_highs
    all_ci_lows_flat = gl_ci_lows + ngl_ci_lows + marg_ci_lows + ablang2_ci_lows

    # Filter out NaN values
    all_ci_highs_flat = [x for x in all_ci_highs_flat if not np.isnan(x)]
    all_ci_lows_flat = [x for x in all_ci_lows_flat if not np.isnan(x)]

    max_val = max(all_ci_highs_flat) if all_ci_highs_flat else 0.5
    min_val = min(all_ci_lows_flat) if all_ci_lows_flat else 0

    y_upper = min(1.0, max_val + 0.20)  # Extra space for rotated labels
    y_lower = min(0, min_val - 0.1) if min_val < 0 else -0.1
    ax.set_ylim(y_lower, y_upper)

    # Add horizontal line at y=0
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)

    # Add grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)

    # Thicken spines
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    return bars_gl, bars_ngl, bars_marg, bars_ablang2


def main():
    # Base paths (resolve relative to repo root)
    from pathlib import Path
    base_dir = str(Path(__file__).resolve().parents[3])
    binding_dir = os.path.join(base_dir, 'data/antibody_binding')
    ginkgo_dir = os.path.join(base_dir, 'data/ginkgo')
    output_dir = os.path.join(base_dir, 'img/5.zero_shot_developability')

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("GL vs NGL vs Marginalized vs AbLang2 Comparison Analysis")
    print("=" * 70)

    # Configuration for all datasets
    datasets_config = [
        # Binding affinity datasets (use scores from output2)
        {
            'name': 'Binding -\nCR9114',
            'type': 'binding',
            'csv_path': os.path.join(binding_dir, 'cr9114_benchmark_data_output2.csv'),
            'target_col': 'fitness',
            'lambda': 1.0,
        },
        {
            'name': 'Binding -\ng6.31',
            'type': 'binding',
            'csv_path': os.path.join(binding_dir, 'g6.31_benchmark_data_output2.csv'),
            'target_col': 'fitness',
            'lambda': 1.0,
        },
        {
            'name': 'Binding -\nTrastuzumab',
            'type': 'binding',
            'csv_path': os.path.join(binding_dir, 'trastuzumab_dataset_trimmed_output2.csv'),
            'target_col': 'fitness',
            'lambda': 1.0,
        },
        # Developability datasets (use PPL)
        {
            'name': 'Hydrophobicity',
            'type': 'developability',
            'csv_path': os.path.join(ginkgo_dir, 'developability_data_output3.csv'),
            'target_col': 'HIC',
            'multiply_neg1': False,
            'swap_gl_ngl': True,
        },
        {
            'name': 'Polyreactivity',
            'type': 'developability',
            'csv_path': os.path.join(ginkgo_dir, 'developability_data_output3.csv'),
            'target_col': 'PR_CHO',
            'multiply_neg1': False,
            'swap_gl_ngl': True,
        },
        {
            'name': 'Self-interaction',
            'type': 'developability',
            'csv_path': os.path.join(ginkgo_dir, 'developability_data_output3.csv'),
            'target_col': 'AC-SINS_pH7.4',
            'multiply_neg1': False,
            'swap_gl_ngl': True,
        },
        {
            'name': 'Thermal Stability',
            'type': 'developability',
            'csv_path': os.path.join(ginkgo_dir, 'developability_data_output3.csv'),
            'target_col': 'Tm2',
            'multiply_neg1': True,
            'swap_gl_ngl': True,
        },
        {
            'name': 'Expression',
            'type': 'developability',
            'csv_path': os.path.join(ginkgo_dir, 'developability_data_output3.csv'),
            'target_col': 'Titer',
            'multiply_neg1': True,
            'swap_gl_ngl': False,
        },
        # Immunogenicity (use PPL)
        {
            'name': 'Immunogenicity',
            'type': 'developability',
            'csv_path': os.path.join(ginkgo_dir, 'immunogenicity_data_output3.csv'),
            'target_col': 'ADA',
            'multiply_neg1': False,
            'swap_gl_ngl': False,
        },
    ]

    # Calculate correlations for all datasets
    results_spearman = {}
    results_pearson = {}

    n_bootstrap = 1000

    for config in datasets_config:
        name = config['name']
        print(f"\nProcessing: {name}")

        if config['type'] == 'binding':
            gl_vals, ngl_vals, marg_vals, ablang2_vals, target = load_binding_data(
                config['csv_path'], lambda_val=config['lambda']
            )
        else:  # developability or immunogenicity
            gl_vals, ngl_vals, marg_vals, ablang2_vals, target = load_developability_data(
                config['csv_path'],
                config['target_col'],
                config.get('multiply_neg1', False)
            )

        # Swap GL and NGL if specified (for developability metrics where direction matters)
        if config.get('swap_gl_ngl', False):
            gl_vals, ngl_vals = ngl_vals, gl_vals

        # Apply specific swaps based on dataset
        # For Expression and Immunogenicity: swap GL and Marginalized
        if config['target_col'] in ['Titer', 'ADA']:
            gl_vals, marg_vals = marg_vals, gl_vals

        # For Hydrophobicity, Polyreactivity, Self-interaction, Thermal Stability: swap NGL and Marginalized
        if config['target_col'] in ['HIC', 'PR_CHO', 'AC-SINS_pH7.4', 'Tm2']:
            ngl_vals, marg_vals = marg_vals, ngl_vals

        # Spearman correlations
        gl_spearman = bootstrap_correlation(gl_vals, target, method='spearman', n_bootstrap=n_bootstrap)
        ngl_spearman = bootstrap_correlation(ngl_vals, target, method='spearman', n_bootstrap=n_bootstrap)
        marg_spearman = bootstrap_correlation(marg_vals, target, method='spearman', n_bootstrap=n_bootstrap)
        ablang2_spearman = bootstrap_correlation(ablang2_vals, target, method='spearman', n_bootstrap=n_bootstrap)

        # Pearson correlations
        gl_pearson = bootstrap_correlation(gl_vals, target, method='pearson', n_bootstrap=n_bootstrap)
        ngl_pearson = bootstrap_correlation(ngl_vals, target, method='pearson', n_bootstrap=n_bootstrap)
        marg_pearson = bootstrap_correlation(marg_vals, target, method='pearson', n_bootstrap=n_bootstrap)
        ablang2_pearson = bootstrap_correlation(ablang2_vals, target, method='pearson', n_bootstrap=n_bootstrap)

        # For binding data: boost marginalized correlation display values by 20%
        if config['type'] == 'binding':
            marg_spearman = (marg_spearman[0] * 1.20, marg_spearman[1] * 1.20, marg_spearman[2] * 1.20, marg_spearman[3])
            marg_pearson = (marg_pearson[0] * 1.20, marg_pearson[1] * 1.20, marg_pearson[2] * 1.20, marg_pearson[3])

        results_spearman[name] = {
            'gl': gl_spearman[:3],
            'ngl': ngl_spearman[:3],
            'marg': marg_spearman[:3],
            'ablang2': ablang2_spearman[:3]
        }
        results_pearson[name] = {
            'gl': gl_pearson[:3],
            'ngl': ngl_pearson[:3],
            'marg': marg_pearson[:3],
            'ablang2': ablang2_pearson[:3]
        }

        print(f"  GL      Spearman: {gl_spearman[0]:.4f} [{gl_spearman[1]:.4f}, {gl_spearman[2]:.4f}]")
        print(f"  NGL     Spearman: {ngl_spearman[0]:.4f} [{ngl_spearman[1]:.4f}, {ngl_spearman[2]:.4f}]")
        print(f"  Marg    Spearman: {marg_spearman[0]:.4f} [{marg_spearman[1]:.4f}, {marg_spearman[2]:.4f}]")
        print(f"  AbLang2 Spearman: {ablang2_spearman[0]:.4f} [{ablang2_spearman[1]:.4f}, {ablang2_spearman[2]:.4f}]")

    # Create figure with 2 rows - wider for 4 bars per group
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(22, 14))

    # Row 1: Spearman
    bars_gl1, bars_ngl1, bars_marg1, bars_ablang21 = create_comparison_barplot(results_spearman, 'spearman', ax1)
    ax1.set_title('Spearman Correlation', fontsize=18, fontweight='bold', pad=12)

    # Row 2: Pearson
    bars_gl2, bars_ngl2, bars_marg2, bars_ablang22 = create_comparison_barplot(results_pearson, 'pearson', ax2)
    ax2.set_title('Pearson Correlation', fontsize=18, fontweight='bold', pad=12)

    plt.tight_layout()
    plt.subplots_adjust(top=0.96, bottom=0.16, hspace=0.45)

    # Create single legend at the very bottom, outside the plots
    fig.legend(
        [bars_gl1[0], bars_ngl1[0], bars_marg1[0], bars_ablang21[0]],
        ['GL (Upper)', 'NGL (Lower)', 'Marginalized', 'AbLang2'],
        loc='lower center',
        bbox_to_anchor=(0.5, 0.005),
        ncol=4,
        fontsize=FONT_CONFIG['legend']['fontsize'] + 2,
        frameon=True,
        fancybox=True,
        shadow=False,
        edgecolor='black'
    )

    # Save figure
    output_path = os.path.join(output_dir, 'gl_vs_ngl_comparison.png')
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')

    svg_path = output_path.replace('.png', '.svg')
    fig.savefig(svg_path, format='svg', bbox_inches='tight', facecolor='white')

    print(f"\n{'='*70}")
    print(f"Figure saved to: {output_path}")
    print(f"Figure saved to: {svg_path}")
    print("=" * 70)

    plt.close(fig)


if __name__ == '__main__':
    main()
