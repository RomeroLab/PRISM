#!/usr/bin/env python3
"""
PPL Comparison Plot with Wilcoxon Signed-Rank Test

This script creates publication-quality box plots comparing perplexity distributions
between different protein language models. Uses Wilcoxon Signed-Rank test for
statistical significance testing between our model and baselines.

The script automatically detects the latest PRISM model version from the data
by parsing column names with version numbers. Supports three PPL types:
    - Normalized (*_Norm_LL): Merged uppercase/lowercase from AA head
    - Marginalized (*_Marg_LL): Sum P(upper) + P(lower) from final head
    - Original (*_LL): Raw 53-vocab final head output

Supports both integer and decimal versions (e.g., v17, v33, v33.3, v37).
The highest version is used by default.

Outputs (13 plots, both PNG and SVG = 26 files):
    Overall PPL (3 plots):
        - overall_ppl.{png,svg} - Whole (Heavy + Light)
        - overall_ppl_heavy.{png,svg} - Heavy Chain Only
        - overall_ppl_light.{png,svg} - Light Chain Only

    Germline Region PPL (3 plots, each with FR/CDR1,2/CDR3 subplots):
        - germline_region_ppl.{png,svg} - Whole
        - germline_region_ppl_heavy.{png,svg} - Heavy Chain Only
        - germline_region_ppl_light.{png,svg} - Light Chain Only

    Non-Germline Region PPL (3 plots, each with FR/CDR1,2/CDR3 subplots):
        - nongermline_region_ppl.{png,svg} - Whole
        - nongermline_region_ppl_heavy.{png,svg} - Heavy Chain Only
        - nongermline_region_ppl_light.{png,svg} - Light Chain Only

    Non-Germline Focused PPL (3 plots, y-lim 1-20, no p-values):
        - nongermline_region_ppl_focused.{png,svg} - Whole
        - nongermline_region_ppl_focused_heavy.{png,svg} - Heavy Chain Only
        - nongermline_region_ppl_focused_light.{png,svg} - Light Chain Only

    Individual Region PPL (4 plots, same size as Overall):
        - fr_gl_heavy.{png,svg} - Framework GL Heavy Chain
        - fr_gl_light.{png,svg} - Framework GL Light Chain
        - cdr3_ngl_heavy.{png,svg} - CDR3 NGL Heavy Chain
        - cdr3_ngl_light.{png,svg} - CDR3 NGL Light Chain

    PRISM Region Exact PPL (1 plot):
        - evoab_region_ppl_exact.{png,svg} - 53-vocab PPL breakdown

Usage:
    # Auto-detect latest PRISM version with marginalized PPL (v37 default)
    python plot_ppl_comparison.py --data_path /path/to/data.pkl --ppl_type marginalized

    # Use normalized PPL (original behavior)
    python plot_ppl_comparison.py --data_path /path/to/data.pkl --ppl_type normalized

    # Specify output directory
    python plot_ppl_comparison.py --data_path /path/to/data.pkl --output_dir ./figures

    # Override with specific model version (full column prefix)
    python plot_ppl_comparison.py --data_path /path/to/data.pkl --our_model_col ESM2_v37_lambda_mixture_Marg

    # Select specific PRISM version by number (simpler)
    python plot_ppl_comparison.py --data_path /path/to/data.pkl --version 37 --ppl_type marginalized

    # Compare all PRISM versions (table only, no figures)
    python plot_ppl_comparison.py --data_path /path/to/data.pkl --version_summary
"""

import argparse
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Tuple, Optional
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')


def parse_version_string(version_str: str) -> float:
    """
    Parse version string to a sortable float value.

    Handles versions like: 17, 33, 33.3, 34.1a, 34.1b
    Letter suffixes are converted to small decimal offsets (a=0.001, b=0.002, etc.)

    Examples:
        "17" -> 17.0
        "33.3" -> 33.3
        "34.1a" -> 34.101
        "34.1b" -> 34.102
    """
    # Check for letter suffix
    if version_str and version_str[-1].isalpha():
        letter = version_str[-1].lower()
        numeric_part = version_str[:-1]
        # Convert letter to offset: a=0.001, b=0.002, ..., z=0.026
        letter_offset = (ord(letter) - ord('a') + 1) * 0.001
        return float(numeric_part) + letter_offset
    else:
        return float(version_str)


def find_all_evoab_columns(df: pd.DataFrame, ppl_type: str = 'normalized') -> List[Tuple[float, str, str]]:
    """
    Find all PRISM model columns by parsing version numbers.

    Supports three PPL types:
    - normalized: *_Norm_LL columns (merged AA head probabilities)
    - marginalized: *_Marg_LL columns (sum P(upper) + P(lower) from final head)
    - original: *_LL columns (without Norm or Marg suffix)

    Example column patterns:
    - ESM2_multihead_v17_Norm_LL (normalized)
    - ESM2_v37_lambda_mixture_Marg_LL (marginalized)
    - ESM2_v33_Norm_LL, ESM2_v33.3_Norm_LL (decimal versions)
    - ESM2_v34.1b_balanced_Marg_LL (versions with letter suffix)

    Args:
        df: DataFrame with log probability columns
        ppl_type: Type of PPL columns to search for ('normalized', 'marginalized', 'original')

    Returns:
        List of tuples: (version_float, column_prefix, version_str), sorted by version descending.
    """
    # Build pattern based on PPL type
    # Version pattern now supports: 17, 33.3, 34.1a, 34.1b (digit.digit + optional letter)
    version_pattern = r'(\d+(?:\.\d+)?[a-zA-Z]?)'

    if ppl_type == 'marginalized':
        # Pattern for marginalized columns: ESM2_v37_lambda_mixture_Marg_LL
        # Captures: prefix="ESM2_v37_lambda_mixture_Marg", version="37"
        evoab_pattern = re.compile(rf'^(ESM2(?:_multihead)?_v{version_pattern}(?:_\w+)*_Marg)_LL$')
    elif ppl_type == 'original':
        # Pattern for original columns without Norm or Marg suffix
        # Captures: ESM2_multihead_v17_LL -> prefix="ESM2_multihead_v17", version="17"
        # Excludes columns ending with _Norm_LL or _Marg_LL
        evoab_pattern = re.compile(rf'^(ESM2(?:_multihead)?_v{version_pattern}(?:_\w+)*)_LL$')
    else:
        # Default: normalized columns (original behavior)
        # Captures: ESM2_multihead_v33.3_gentle_Norm_LL -> prefix="ESM2_multihead_v33.3_gentle_Norm", version="33.3"
        evoab_pattern = re.compile(rf'^(ESM2(?:_multihead)?_v{version_pattern}(?:_\w+)*_Norm)_LL$')

    version_columns = []
    for col in df.columns:
        match = evoab_pattern.match(col)
        if match:
            prefix = match.group(1)
            version_str = match.group(2)

            # For original type, skip if it's actually a Norm or Marg column
            if ppl_type == 'original':
                if '_Norm_LL' in col or '_Marg_LL' in col:
                    continue

            # Parse version using new function that handles letter suffixes
            version = parse_version_string(version_str)
            version_columns.append((version, prefix, version_str))

    # Sort by version number (descending)
    version_columns.sort(key=lambda x: x[0], reverse=True)
    return version_columns


def find_latest_evoab_column(df: pd.DataFrame, ppl_type: str = 'normalized') -> Optional[str]:
    """
    Find the latest PRISM model column by parsing version numbers.

    Args:
        df: DataFrame with log probability columns
        ppl_type: Type of PPL columns to search for ('normalized', 'marginalized', 'original')

    Returns:
        The column prefix (without _LL suffix) for the highest version, or None if not found.
    """
    version_columns = find_all_evoab_columns(df, ppl_type=ppl_type)

    if not version_columns:
        return None

    latest_prefix = version_columns[0][1]

    # Format version strings for display
    version_display = [f'v{v_str}' for _, _, v_str in version_columns]
    print(f"  Auto-detected PRISM versions ({ppl_type}): {version_display}")
    print(f"  Using latest: {latest_prefix}")

    return latest_prefix

# Configure matplotlib for publication-quality figures
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# Font configuration for consistent styling across all plots
FONT_CONFIG = {
    'label': {'fontsize': 25, 'fontweight': 'bold'},       # x label, y label
    'tick': {'fontsize': 15, 'fontweight': 'normal'},      # x tick, y tick
    'legend': {'fontsize': 20},                             # legend
    'title': {'fontsize': 25, 'fontweight': 'bold'},       # title/subtitle
    'annotation': {'fontsize': 20, 'fontweight': 'normal'}, # annotations, stats boxes
}

# Color palette (Paul Tol's colorblind-friendly)
COLORS = [
    '#332288',   # Dark purple for PRISM (our model)
    '#DDCC77',   # Sand/Yellow for ESM2-35M
    '#117733',   # Green for ESM2-650M
    '#88CCEE',   # Light blue for AbLang2
    '#44AA99',   # Teal for AntiBERTy
    '#882255',   # Wine/Dark magenta for Sapiens
    '#EE7733',   # Orange for IgLM
]


def is_in_cdr_regions(position: int, cdr_regions: List) -> bool:
    """Check if a position falls within any CDR region."""
    if not cdr_regions:
        return False
    for region in cdr_regions:
        if region is None:
            continue
        start, end = region
        if start <= position < end:
            return True
    return False


def calculate_perplexity_from_logprobs(log_probs: np.ndarray, mask: np.ndarray) -> float:
    """Calculate perplexity for positions specified by mask."""
    if not isinstance(log_probs, (list, np.ndarray)) or len(log_probs) == 0:
        return np.nan

    log_probs = np.array(log_probs)
    mask = np.array(mask, dtype=bool)

    if len(log_probs) != len(mask):
        return np.nan

    selected_logprobs = log_probs[mask]

    if len(selected_logprobs) == 0:
        return np.nan

    mean_log_prob = np.mean(selected_logprobs)
    perplexity = np.exp(-mean_log_prob)

    return perplexity


def calculate_detailed_perplexities(row: pd.Series, ll_column: str) -> Dict[str, float]:
    """
    Calculate all detailed perplexity metrics for a single row.

    Returns:
        Dictionary with perplexity metrics for different regions.
        Includes whole, heavy-chain-only, and light-chain-only versions.
    """
    results = {}

    # Define all metric keys for NaN return
    all_metric_keys = [
        'overall', 'overall_heavy', 'overall_light',
        'fr_gl_whole', 'cdr12_gl_whole', 'cdr3_gl_whole',
        'fr_gl_heavy', 'cdr12_gl_heavy', 'cdr3_gl_heavy',
        'fr_gl_light', 'cdr12_gl_light', 'cdr3_gl_light',
        'fr_ngl_whole', 'cdr12_ngl_whole', 'cdr3_ngl_whole',
        'fr_ngl_heavy', 'cdr12_ngl_heavy', 'cdr3_ngl_heavy',
        'fr_ngl_light', 'cdr12_ngl_light', 'cdr3_ngl_light'
    ]

    # Get sequences and log probabilities
    heavy_seq = row['HEAVY_CHAIN_AA_SEQUENCE']
    light_seq = row['LIGHT_CHAIN_AA_SEQUENCE']
    ngl_lowercase_seq = row['NGL_lowercase_seq']
    log_probs = row[ll_column]

    if not isinstance(log_probs, (list, np.ndarray)) or len(log_probs) == 0:
        return {k: np.nan for k in all_metric_keys}

    heavy_len = len(heavy_seq)
    light_len = len(light_seq)
    total_len = heavy_len + light_len

    if len(log_probs) != total_len:
        return {k: np.nan for k in all_metric_keys}

    log_probs = np.array(log_probs)

    # Split log_probs into heavy and light chain portions
    log_probs_heavy = log_probs[:heavy_len]
    log_probs_light = log_probs[heavy_len:]

    # 1. Overall PPL (whole, heavy, light)
    results['overall'] = np.exp(-np.mean(log_probs))
    results['overall_heavy'] = np.exp(-np.mean(log_probs_heavy))
    results['overall_light'] = np.exp(-np.mean(log_probs_light))

    # Get CDR regions
    hc_cdr1 = row.get('HC_CDR1_region', None)
    hc_cdr2 = row.get('HC_CDR2_region', None)
    hc_cdr3 = row.get('HC_CDR3_region', None)
    lc_cdr1 = row.get('LC_CDR1_region', None)
    lc_cdr2 = row.get('LC_CDR2_region', None)
    lc_cdr3 = row.get('LC_CDR3_region', None)

    # Create NGL/GL masks for whole sequence and split by chain
    whole_ngl_mask = np.array([c.islower() for c in ngl_lowercase_seq])
    whole_gl_mask = ~whole_ngl_mask

    # Split NGL/GL masks by chain
    hc_ngl_mask = whole_ngl_mask[:heavy_len]
    hc_gl_mask = whole_gl_mask[:heavy_len]
    lc_ngl_mask = whole_ngl_mask[heavy_len:]
    lc_gl_mask = whole_gl_mask[heavy_len:]

    # CDR masks for heavy chain
    hc_cdr12_mask = np.zeros(heavy_len, dtype=bool)
    hc_cdr3_mask = np.zeros(heavy_len, dtype=bool)

    for i in range(heavy_len):
        if is_in_cdr_regions(i, [hc_cdr1, hc_cdr2]):
            hc_cdr12_mask[i] = True
        if is_in_cdr_regions(i, [hc_cdr3]):
            hc_cdr3_mask[i] = True

    hc_fr_mask = ~(hc_cdr12_mask | hc_cdr3_mask)

    # CDR masks for light chain
    lc_cdr12_mask = np.zeros(light_len, dtype=bool)
    lc_cdr3_mask = np.zeros(light_len, dtype=bool)

    for i in range(light_len):
        if is_in_cdr_regions(i, [lc_cdr1, lc_cdr2]):
            lc_cdr12_mask[i] = True
        if is_in_cdr_regions(i, [lc_cdr3]):
            lc_cdr3_mask[i] = True

    lc_fr_mask = ~(lc_cdr12_mask | lc_cdr3_mask)

    # Whole sequence CDR masks
    whole_cdr12_mask = np.concatenate([hc_cdr12_mask, lc_cdr12_mask])
    whole_cdr3_mask = np.concatenate([hc_cdr3_mask, lc_cdr3_mask])
    whole_fr_mask = np.concatenate([hc_fr_mask, lc_fr_mask])

    # =========================================================================
    # Calculate GL (Germline) metrics - Whole
    # =========================================================================
    results['fr_gl_whole'] = calculate_perplexity_from_logprobs(log_probs, whole_gl_mask & whole_fr_mask)
    results['cdr12_gl_whole'] = calculate_perplexity_from_logprobs(log_probs, whole_gl_mask & whole_cdr12_mask)
    results['cdr3_gl_whole'] = calculate_perplexity_from_logprobs(log_probs, whole_gl_mask & whole_cdr3_mask)

    # =========================================================================
    # Calculate GL (Germline) metrics - Heavy Chain Only
    # =========================================================================
    results['fr_gl_heavy'] = calculate_perplexity_from_logprobs(log_probs_heavy, hc_gl_mask & hc_fr_mask)
    results['cdr12_gl_heavy'] = calculate_perplexity_from_logprobs(log_probs_heavy, hc_gl_mask & hc_cdr12_mask)
    results['cdr3_gl_heavy'] = calculate_perplexity_from_logprobs(log_probs_heavy, hc_gl_mask & hc_cdr3_mask)

    # =========================================================================
    # Calculate GL (Germline) metrics - Light Chain Only
    # =========================================================================
    results['fr_gl_light'] = calculate_perplexity_from_logprobs(log_probs_light, lc_gl_mask & lc_fr_mask)
    results['cdr12_gl_light'] = calculate_perplexity_from_logprobs(log_probs_light, lc_gl_mask & lc_cdr12_mask)
    results['cdr3_gl_light'] = calculate_perplexity_from_logprobs(log_probs_light, lc_gl_mask & lc_cdr3_mask)

    # =========================================================================
    # Calculate NGL (Non-Germline) metrics - Whole
    # =========================================================================
    results['fr_ngl_whole'] = calculate_perplexity_from_logprobs(log_probs, whole_ngl_mask & whole_fr_mask)
    results['cdr12_ngl_whole'] = calculate_perplexity_from_logprobs(log_probs, whole_ngl_mask & whole_cdr12_mask)
    results['cdr3_ngl_whole'] = calculate_perplexity_from_logprobs(log_probs, whole_ngl_mask & whole_cdr3_mask)

    # =========================================================================
    # Calculate NGL (Non-Germline) metrics - Heavy Chain Only
    # =========================================================================
    results['fr_ngl_heavy'] = calculate_perplexity_from_logprobs(log_probs_heavy, hc_ngl_mask & hc_fr_mask)
    results['cdr12_ngl_heavy'] = calculate_perplexity_from_logprobs(log_probs_heavy, hc_ngl_mask & hc_cdr12_mask)
    results['cdr3_ngl_heavy'] = calculate_perplexity_from_logprobs(log_probs_heavy, hc_ngl_mask & hc_cdr3_mask)

    # =========================================================================
    # Calculate NGL (Non-Germline) metrics - Light Chain Only
    # =========================================================================
    results['fr_ngl_light'] = calculate_perplexity_from_logprobs(log_probs_light, lc_ngl_mask & lc_fr_mask)
    results['cdr12_ngl_light'] = calculate_perplexity_from_logprobs(log_probs_light, lc_ngl_mask & lc_cdr12_mask)
    results['cdr3_ngl_light'] = calculate_perplexity_from_logprobs(log_probs_light, lc_ngl_mask & lc_cdr3_mask)

    return results


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


def create_boxplot_with_significance(
    data_dict: Dict[str, np.ndarray],
    model_names: List[str],
    display_names: List[str],
    colors: List[str],
    our_model_idx: int,
    ax: plt.Axes,
    ylabel: str = "Pseudo-Perplexity",
    subtitle: str = None,
    log_scale: bool = False
):
    """
    Create a box plot with Wilcoxon Signed-Rank test significance brackets.

    Args:
        data_dict: Dict mapping model names to arrays of perplexity values
        model_names: List of model column names (in order)
        display_names: List of display names for models
        colors: List of colors for boxes
        our_model_idx: Index of our model (for comparison)
        ax: Matplotlib axes
        ylabel: Y-axis label
        subtitle: Optional subtitle for the subplot
    """
    # Filter to only include models that exist in data_dict
    valid_indices = []
    valid_model_names = []
    valid_display_names = []
    valid_colors = []

    for i, model in enumerate(model_names):
        if model in data_dict and len(data_dict[model]) > 0:
            valid_indices.append(i)
            valid_model_names.append(model)
            valid_display_names.append(display_names[i])
            valid_colors.append(colors[i])

    if len(valid_model_names) == 0:
        ax.text(0.5, 0.5, "No valid data", ha='center', va='center', transform=ax.transAxes,
                **FONT_CONFIG['annotation'])
        return

    # Find new index for our model
    our_model_new_idx = None
    for i, model in enumerate(valid_model_names):
        if model == model_names[our_model_idx]:
            our_model_new_idx = i
            break

    n_models = len(valid_model_names)
    x_pos = np.arange(n_models)

    # Prepare data for boxplot
    plot_data = []
    for model in valid_model_names:
        values = data_dict[model]
        # Clean data: remove NaN and Inf
        clean_values = values[~np.isnan(values) & ~np.isinf(values)]
        plot_data.append(clean_values)

    # Create box plot
    bp = ax.boxplot(plot_data, positions=x_pos, widths=0.6, patch_artist=True,
                    showfliers=True, flierprops={'markersize': 3, 'alpha': 0.5})

    # Color the boxes
    for patch, color in zip(bp['boxes'], valid_colors):
        patch.set_facecolor(color)
        patch.set_edgecolor('black')
        patch.set_linewidth(2)

    # Style medians
    for median in bp['medians']:
        median.set_color('black')
        median.set_linewidth(2)

    # Highlight our model box
    if our_model_new_idx is not None:
        bp['boxes'][our_model_new_idx].set_edgecolor('#000000')
        bp['boxes'][our_model_new_idx].set_linewidth(3)

    # Calculate Wilcoxon p-values and draw brackets
    if our_model_new_idx is not None:
        our_data = plot_data[our_model_new_idx]

        # Get max y value for bracket positioning
        all_values = np.concatenate([d for d in plot_data if len(d) > 0])
        max_y = np.percentile(all_values[~np.isnan(all_values)], 95)  # Use 95th percentile

        if log_scale:
            # For log scale, use multiplicative spacing
            bracket_start = max_y * 1.3
            bracket_multiplier = 1.25  # Each bracket 25% higher
        else:
            bracket_start = max_y * 1.05
            bracket_interval = max_y * 0.12  # Larger interval for bracket spacing

        p_values = {}
        bracket_idx = 0

        for i, model in enumerate(valid_model_names):
            if i == our_model_new_idx:
                continue

            other_data = plot_data[i]

            # Ensure same length for paired test
            min_len = min(len(our_data), len(other_data))
            if min_len > 10:  # Need sufficient samples
                try:
                    _, p_val = stats.wilcoxon(our_data[:min_len], other_data[:min_len])
                except:
                    p_val = 1.0
            else:
                p_val = 1.0

            p_values[model] = p_val

            # Draw bracket (different positioning for log scale)
            if log_scale:
                bracket_y = bracket_start * (bracket_multiplier ** bracket_idx)
                height = bracket_y * 0.08
                tip_length = bracket_y * 0.04
            else:
                bracket_y = bracket_start + bracket_idx * bracket_interval
                height = bracket_interval * 0.3
                tip_length = bracket_interval * 0.15

            draw_significance_bracket(ax, our_model_new_idx, i, bracket_y, p_val,
                                     height=height,
                                     tip_length=tip_length)
            bracket_idx += 1

        # Set y-axis limits (PPL is always >= 1.0)
        n_brackets = len(p_values)
        y_lower = 1.0

        if log_scale:
            ax.set_yscale('log')
            y_upper = bracket_start * (bracket_multiplier ** n_brackets) * 1.3
        else:
            y_upper = bracket_start + n_brackets * bracket_interval + bracket_interval * 0.5

        ax.set_ylim(y_lower, y_upper)

    # Customize axes
    ax.set_xticks(x_pos)
    ax.set_xticklabels(valid_display_names, fontsize=25, fontweight='bold', rotation=45, ha='right')
    ax.set_ylabel(ylabel, **FONT_CONFIG['label'])
    ax.tick_params(axis='y', labelsize=FONT_CONFIG['tick']['fontsize'])

    # Add subtitle if provided
    if subtitle:
        ax.set_title(subtitle, **FONT_CONFIG['title'])

    # Add grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)

    # Thicken spines
    for spine in ax.spines.values():
        spine.set_linewidth(2)


def main():
    parser = argparse.ArgumentParser(
        description='Create PPL comparison plots with Wilcoxon Signed-Rank test'
    )
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the input pickle file')
    parser.add_argument('--output_dir', type=str,
                        default='img/1.pppl_calculation',
                        help='Directory to save output figures')
    parser.add_argument('--our_model_col', type=str, default=None,
                        help='Column prefix for our model (will use _LL suffix). '
                             'If not specified, auto-detects the latest PRISM version.')
    parser.add_argument('--version', type=str, default=None,
                        help='Select PRISM version by number (e.g., 17, 33, 37). '
                             'Simpler than --our_model_col. Default: latest version.')
    parser.add_argument('--ppl_type', type=str, default='marginalized',
                        choices=['normalized', 'marginalized', 'original'],
                        help='Type of PPL to use for PRISM: '
                             'normalized (*_Norm_LL), marginalized (*_Marg_LL), or original (*_LL). '
                             'Default: marginalized (recommended for v37+)')
    parser.add_argument('--version_summary', action='store_true',
                        help='Only print a summary table of all PRISM versions (no figures)')

    # ==========================================================================
    # Ablation Mode Arguments
    # ==========================================================================
    parser.add_argument('--log_scale', action='store_true',
                        help='Use logarithmic scale for y-axis (recommended for large PPL ranges)')
    parser.add_argument('--ablation_mode', action='store_true',
                        help='Enable ablation study mode: compare PRISM Full vs 3 ablation models '
                             'instead of comparing against baseline models (ESM2, AbLang2, etc.)')
    parser.add_argument('--ablation1_col', type=str, default='ESM2_v34.1b_ablation1_no_pretrain',
                        help='Column prefix for Ablation 1 (Multihead + No Pretraining). '
                             'Default: ESM2_v34.1b_ablation1_no_pretrain')
    parser.add_argument('--ablation2_col', type=str, default='ESM2_v34.1b_ablation2_simple_paired',
                        help='Column prefix for Ablation 2 (Simple Head + Pretraining). '
                             'Default: ESM2_v34.1b_ablation2_simple_paired')
    parser.add_argument('--ablation3_col', type=str, default='ESM2_v34.1b_ablation3_simple_no_pretrain',
                        help='Column prefix for Ablation 3 (Simple Head + No Pretraining). '
                             'Default: ESM2_v34.1b_ablation3_simple_no_pretrain')
    parser.add_argument('--prism_less_col', type=str, default='Pretrained_ESM2',
                        help='Column prefix for PRISM-less (pure ESM2 finetune baseline). '
                             'Added as 5th bar in ablation mode when column exists. '
                             'Default: Pretrained_ESM2')

    args = parser.parse_args()

    # Load data
    print(f"Loading data from: {args.data_path}")
    df = pd.read_pickle(args.data_path)
    print(f"Loaded dataframe with shape: {df.shape}")

    # Check required columns
    required_cols = ['NGL_lowercase_seq', 'HEAVY_CHAIN_AA_SEQUENCE', 'LIGHT_CHAIN_AA_SEQUENCE']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}'")

    # =========================================================================
    # Version Summary Mode: Print table of all PRISM versions and exit
    # =========================================================================
    if args.version_summary:
        print("\n" + "=" * 90)
        print(f"EVO-AB VERSION SUMMARY (Overall PPL, type={args.ppl_type})")
        print("=" * 90)

        evoab_versions = find_all_evoab_columns(df, ppl_type=args.ppl_type)

        if not evoab_versions:
            print("No PRISM model columns found!")
            return

        print(f"\nFound {len(evoab_versions)} PRISM version(s)\n")
        print(f"{'Version':<15} {'Column Prefix':<40} {'Median':>10} {'Mean':>10} {'Std':>10} {'N':>8}")
        print("-" * 93)

        for version, prefix, version_str in evoab_versions:
            ll_col = f"{prefix}_LL"
            if ll_col not in df.columns:
                print(f"v{version_str:<14} {prefix:<40} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>8}")
                continue

            # Calculate overall PPL for each sequence
            ppl_values = []
            for idx, row in df.iterrows():
                log_probs = row[ll_col]
                if isinstance(log_probs, (list, np.ndarray)) and len(log_probs) > 0:
                    log_probs = np.array(log_probs)
                    clean_probs = log_probs[~np.isnan(log_probs) & ~np.isinf(log_probs)]
                    if len(clean_probs) > 0:
                        ppl = np.exp(-np.mean(clean_probs))
                        if not np.isnan(ppl) and not np.isinf(ppl):
                            ppl_values.append(ppl)

            if len(ppl_values) > 0:
                ppl_arr = np.array(ppl_values)
                median = np.median(ppl_arr)
                mean = np.mean(ppl_arr)
                std = np.std(ppl_arr)
                n = len(ppl_arr)
                print(f"v{version_str:<14} {prefix:<40} {median:>10.4f} {mean:>10.4f} {std:>10.4f} {n:>8}")
            else:
                print(f"v{version_str:<14} {prefix:<40} {'N/A':>10} {'N/A':>10} {'N/A':>10} {0:>8}")

        print("-" * 93)
        print("\n✓ Version summary complete (no figures generated)")
        return

    # Determine which PRISM model to use
    # Priority: --our_model_col > --version > auto-detect latest
    if args.our_model_col is not None:
        print(f"\nUsing specified PRISM model: {args.our_model_col}")
    elif args.version is not None:
        print(f"\nSearching for PRISM version: v{args.version} (type={args.ppl_type})")
        evoab_versions = find_all_evoab_columns(df, ppl_type=args.ppl_type)

        # Find matching version (use parse_version_string to handle letter suffixes)
        target_version = parse_version_string(args.version)
        matched = None
        for version, prefix, version_str in evoab_versions:
            if version == target_version:
                matched = prefix
                break

        if matched is None:
            available = [f"v{v_str}" for _, _, v_str in evoab_versions]
            raise ValueError(
                f"Version v{args.version} ({args.ppl_type}) not found. "
                f"Available versions: {available}"
            )

        args.our_model_col = matched
        print(f"  Found: {args.our_model_col}")
    else:
        print(f"\nAuto-detecting PRISM model column (type={args.ppl_type})...")
        args.our_model_col = find_latest_evoab_column(df, ppl_type=args.ppl_type)
        if args.our_model_col is None:
            raise ValueError(
                f"Could not auto-detect PRISM model column ({args.ppl_type}). "
                "Please specify --our_model_col or --version explicitly."
            )

    # ==========================================================================
    # Define models to compare (Standard vs Ablation mode)
    # ==========================================================================
    if args.ablation_mode:
        # =====================================================================
        # ABLATION MODE: Compare PRISM Full vs 3 ablation variants
        # =====================================================================
        # Ablation Study Design (2×2 factorial):
        # ┌─────────────────────┬───────────────────┬───────────────────┐
        # │                     │ Multihead (Full)  │ Simple LM Head    │
        # ├─────────────────────┼───────────────────┼───────────────────┤
        # │ With Pretraining    │ PRISM (Full)      │ Ablation 2        │
        # ├─────────────────────┼───────────────────┼───────────────────┤
        # │ No Pretraining      │ Ablation 1        │ Ablation 3        │
        # └─────────────────────┴───────────────────┴───────────────────┘
        print("\n" + "=" * 80)
        print("ABLATION MODE ENABLED")
        print("=" * 80)
        print("  Comparing PRISM (Full) vs 3 ablation models:")
        print(f"    - PRISM (Full): {args.our_model_col}")
        print(f"    - Ablation 1 (Multihead+NoPretrain): {args.ablation1_col}")
        print(f"    - Ablation 2 (SimpleHead+Pretrain): {args.ablation2_col}")
        print(f"    - Ablation 3 (SimpleHead+NoPretrain): {args.ablation3_col}")
        print("=" * 80)

        # Ablation colors (consistent with other ablation scripts)
        ABLATION_COLORS = [
            '#332288',   # Dark purple for PRISM (Full/best)
            '#CC6677',   # Rose for Ablation 1
            '#DDCC77',   # Sand/Yellow for Ablation 2
            '#AA4499',   # Purple-pink for Ablation 3
            '#78c679',   # Light green for PRISM-less (pure ESM2 finetune)
        ]

        model_configs = [
            (args.our_model_col, 'PRISM Full'),
            (args.ablation1_col, 'Ablation 1'),
            (args.ablation2_col, 'Ablation 2'),
            (args.ablation3_col, 'Ablation 3'),
            (args.prism_less_col, 'PRISM-less'),
        ]

        # Check which models exist
        available_models = []
        for prefix, display_name in model_configs:
            ll_col = f"{prefix}_LL"
            if ll_col in df.columns:
                available_models.append((prefix, display_name, ll_col))
                print(f"  ✓ Found: {ll_col}")
            else:
                print(f"  ✗ Missing: {ll_col}")

        if len(available_models) == 0:
            raise ValueError("No valid ablation model columns found!")

        # Our model index
        our_model_idx = 0  # PRISM (Full) is first

        model_prefixes = [m[0] for m in available_models]
        display_names = [m[1] for m in available_models]
        ll_columns = [m[2] for m in available_models]

        # Use ablation colors
        colors_to_use = ABLATION_COLORS[:len(available_models)]

    else:
        # =====================================================================
        # STANDARD MODE: Compare PRISM vs baseline models
        # =====================================================================
        # Format: (column_prefix, display_name)
        model_configs = [
            (args.our_model_col, 'PRISM'),           # Our model (first in display)
            ('Pretrained_ESM2', 'ESM2-35M'),
            ('ESM2_650M', 'ESM2-650M'),
            ('AbLang2', 'AbLang2'),
            ('AntiBERTy', 'AntiBERTy'),
            ('Sapiens', 'Sapiens'),
            ('IgLM', 'IgLM'),
        ]

        # Check which models exist
        available_models = []
        for prefix, display_name in model_configs:
            ll_col = f"{prefix}_LL"
            if ll_col in df.columns:
                available_models.append((prefix, display_name, ll_col))
                print(f"  ✓ Found: {ll_col}")
            else:
                print(f"  ✗ Missing: {ll_col}")

        if len(available_models) == 0:
            raise ValueError("No valid model columns found!")

        # Find our model index
        our_model_idx = 0  # Our model is first in the list

        model_prefixes = [m[0] for m in available_models]
        display_names = [m[1] for m in available_models]
        ll_columns = [m[2] for m in available_models]

        # Use colors in order of available models
        colors_to_use = COLORS[:len(available_models)]

    # Filename prefix for ablation mode (to distinguish output files)
    filename_prefix = "ablation_" if args.ablation_mode else ""

    # Calculate detailed perplexities for all models
    print("\nCalculating detailed perplexities for all models...")

    metrics_to_calc = [
        # Overall metrics (whole, heavy, light)
        'overall', 'overall_heavy', 'overall_light',
        # GL metrics - whole
        'fr_gl_whole', 'cdr12_gl_whole', 'cdr3_gl_whole',
        # GL metrics - heavy chain
        'fr_gl_heavy', 'cdr12_gl_heavy', 'cdr3_gl_heavy',
        # GL metrics - light chain
        'fr_gl_light', 'cdr12_gl_light', 'cdr3_gl_light',
        # NGL metrics - whole
        'fr_ngl_whole', 'cdr12_ngl_whole', 'cdr3_ngl_whole',
        # NGL metrics - heavy chain
        'fr_ngl_heavy', 'cdr12_ngl_heavy', 'cdr3_ngl_heavy',
        # NGL metrics - light chain
        'fr_ngl_light', 'cdr12_ngl_light', 'cdr3_ngl_light'
    ]

    # Store results: {metric: {model: np.array}}
    all_results = {metric: {} for metric in metrics_to_calc}

    for prefix, display_name, ll_col in tqdm(available_models, desc="Processing models"):
        model_results = {metric: [] for metric in metrics_to_calc}

        for idx, row in df.iterrows():
            results = calculate_detailed_perplexities(row, ll_col)
            for metric in metrics_to_calc:
                model_results[metric].append(results.get(metric, np.nan))

        for metric in metrics_to_calc:
            all_results[metric][prefix] = np.array(model_results[metric])

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # =========================================================================
    # Helper function to save both PNG and SVG
    # =========================================================================
    def save_figure(fig, output_dir, filename_base):
        """Save figure in both PNG and SVG formats."""
        for ext in ['png', 'svg']:
            output_path = os.path.join(output_dir, f"{filename_base}.{ext}")
            fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"  ✓ Saved: {output_path}")

    # =========================================================================
    # Plot 1: Overall PPL (Whole, Heavy Chain, Light Chain)
    # =========================================================================
    overall_configs = [
        ('overall', 'Overall PPL (Whole)', 'overall_ppl'),
        ('overall_heavy', 'Overall PPL (Heavy Chain)', 'overall_ppl_heavy'),
        ('overall_light', 'Overall PPL (Light Chain)', 'overall_ppl_light')
    ]

    for idx, (metric, title, filename_base) in enumerate(overall_configs):
        print(f"\nCreating Overall PPL ({idx+1}/3): {title}...")
        fig, ax = plt.subplots(figsize=(10, 8))

        create_boxplot_with_significance(
            data_dict=all_results[metric],
            model_names=model_prefixes,
            display_names=display_names,
            colors=colors_to_use,
            our_model_idx=our_model_idx,
            ax=ax,
            ylabel="Pseudo-Perplexity",
            subtitle=title,
            log_scale=args.log_scale
        )

        plt.tight_layout()
        save_figure(fig, args.output_dir, f"{filename_prefix}{filename_base}")
        plt.close(fig)

    # =========================================================================
    # Plot 2: Germline Region-wise PPL (Whole, Heavy Chain, Light Chain)
    # =========================================================================
    gl_plot_configs = [
        # (suffix, title_suffix, filename_base)
        ('whole', 'Whole', 'germline_region_ppl'),
        ('heavy', 'Heavy Chain', 'germline_region_ppl_heavy'),
        ('light', 'Light Chain', 'germline_region_ppl_light')
    ]

    for idx, (suffix, title_suffix, filename_base) in enumerate(gl_plot_configs):
        print(f"\nCreating GL Region PPL ({idx+1}/3): {title_suffix}...")
        fig, axes = plt.subplots(1, 3, figsize=(18, 7))

        gl_metrics = [
            (f'fr_gl_{suffix}', f'Framework (GL) - {title_suffix}'),
            (f'cdr12_gl_{suffix}', f'CDR1,2 (GL) - {title_suffix}'),
            (f'cdr3_gl_{suffix}', f'CDR3 (GL) - {title_suffix}')
        ]

        for ax, (metric, subtitle) in zip(axes, gl_metrics):
            create_boxplot_with_significance(
                data_dict=all_results[metric],
                model_names=model_prefixes,
                display_names=display_names,
                colors=colors_to_use,
                our_model_idx=our_model_idx,
                ax=ax,
                ylabel="Pseudo-Perplexity",
                subtitle=subtitle,
                log_scale=args.log_scale
            )
        plt.tight_layout()
        save_figure(fig, args.output_dir, f"{filename_prefix}{filename_base}")
        plt.close(fig)

    # =========================================================================
    # Plot 3: Non-Germline Region-wise PPL (Whole, Heavy Chain, Light Chain)
    # =========================================================================
    ngl_plot_configs = [
        # (suffix, title_suffix, filename_base)
        ('whole', 'Whole', 'nongermline_region_ppl'),
        ('heavy', 'Heavy Chain', 'nongermline_region_ppl_heavy'),
        ('light', 'Light Chain', 'nongermline_region_ppl_light')
    ]

    for idx, (suffix, title_suffix, filename_base) in enumerate(ngl_plot_configs):
        print(f"\nCreating NGL Region PPL ({idx+1}/3): {title_suffix}...")
        fig, axes = plt.subplots(1, 3, figsize=(18, 7))

        ngl_metrics = [
            (f'fr_ngl_{suffix}', f'Framework (NGL) - {title_suffix}'),
            (f'cdr12_ngl_{suffix}', f'CDR1,2 (NGL) - {title_suffix}'),
            (f'cdr3_ngl_{suffix}', f'CDR3 (NGL) - {title_suffix}')
        ]

        for ax, (metric, subtitle) in zip(axes, ngl_metrics):
            create_boxplot_with_significance(
                data_dict=all_results[metric],
                model_names=model_prefixes,
                display_names=display_names,
                colors=colors_to_use,
                our_model_idx=our_model_idx,
                ax=ax,
                ylabel="Pseudo-Perplexity",
                subtitle=subtitle,
                log_scale=args.log_scale
            )
        plt.tight_layout()
        save_figure(fig, args.output_dir, f"{filename_prefix}{filename_base}")
        plt.close(fig)

    # =========================================================================
    # Plot 4: Non-Germline Region-wise PPL (Focused View, y-lim 1-20, no p-values)
    # =========================================================================
    ngl_focused_configs = [
        # (suffix, title_suffix, filename_base)
        ('whole', 'Whole', 'nongermline_region_ppl_focused'),
        ('heavy', 'Heavy Chain', 'nongermline_region_ppl_focused_heavy'),
        ('light', 'Light Chain', 'nongermline_region_ppl_focused_light')
    ]

    for idx, (suffix, title_suffix, filename_base) in enumerate(ngl_focused_configs):
        print(f"\nCreating NGL Focused PPL ({idx+1}/3): {title_suffix}...")
        fig, axes = plt.subplots(1, 3, figsize=(18, 7))

        ngl_metrics_focused = [
            (f'fr_ngl_{suffix}', f'Framework (NGL) - {title_suffix}'),
            (f'cdr12_ngl_{suffix}', f'CDR1,2 (NGL) - {title_suffix}'),
            (f'cdr3_ngl_{suffix}', f'CDR3 (NGL) - {title_suffix}')
        ]

        for ax, (metric, subtitle) in zip(axes, ngl_metrics_focused):
            data_dict = all_results[metric]

            # Filter to valid models
            valid_model_names = []
            valid_display_names = []
            valid_colors = []
            for i, model in enumerate(model_prefixes):
                if model in data_dict and len(data_dict[model]) > 0:
                    valid_model_names.append(model)
                    valid_display_names.append(display_names[i])
                    valid_colors.append(colors_to_use[i])

            n_models = len(valid_model_names)
            x_pos = np.arange(n_models)

            # Prepare data for boxplot
            plot_data = []
            for model in valid_model_names:
                values = data_dict[model]
                clean_values = values[~np.isnan(values) & ~np.isinf(values)]
                plot_data.append(clean_values)

            # Create box plot without significance brackets
            bp = ax.boxplot(plot_data, positions=x_pos, widths=0.6, patch_artist=True,
                            showfliers=False)  # Hide outliers for cleaner focused view

            # Color the boxes
            for patch, color in zip(bp['boxes'], valid_colors):
                patch.set_facecolor(color)
                patch.set_edgecolor('black')
                patch.set_linewidth(2)

            # Style medians
            for median in bp['medians']:
                median.set_color('black')
                median.set_linewidth(2)

            # Highlight our model box (index 0)
            bp['boxes'][0].set_edgecolor('#000000')
            bp['boxes'][0].set_linewidth(3)

            # Customize axes
            ax.set_xticks(x_pos)
            ax.set_xticklabels(valid_display_names, fontsize=25, fontweight='bold', rotation=45, ha='right')
            ax.set_ylabel("Pseudo-Perplexity", **FONT_CONFIG['label'])
            ax.tick_params(axis='y', labelsize=FONT_CONFIG['tick']['fontsize'])
            ax.set_title(subtitle, **FONT_CONFIG['title'])

            # Set focused y-axis limits
            ax.set_ylim(1, 20)

            # Add grid
            ax.yaxis.grid(True, linestyle='--', alpha=0.3)
            ax.set_axisbelow(True)

            # Thicken spines
            for spine in ax.spines.values():
                spine.set_linewidth(2)

        plt.tight_layout()
        save_figure(fig, args.output_dir, f"{filename_prefix}{filename_base}")
        plt.close(fig)

    # =========================================================================
    # Plot 5: Individual Region Plots (Same size as Overall, 4 plots)
    # - Framework GL Heavy chain
    # - Framework GL Light chain
    # - CDR3 NGL Heavy chain
    # - CDR3 NGL Light chain
    # =========================================================================
    individual_region_configs = [
        ('fr_gl_heavy', 'Framework (GL) - Heavy Chain', 'fr_gl_heavy'),
        ('fr_gl_light', 'Framework (GL) - Light Chain', 'fr_gl_light'),
        ('cdr3_ngl_heavy', 'CDR3 (NGL) - Heavy Chain', 'cdr3_ngl_heavy'),
        ('cdr3_ngl_light', 'CDR3 (NGL) - Light Chain', 'cdr3_ngl_light'),
    ]

    for idx, (metric, title, filename_base) in enumerate(individual_region_configs):
        print(f"\nCreating Individual Region Plot ({idx+1}/4): {title}...")
        fig, ax = plt.subplots(figsize=(10, 8))

        create_boxplot_with_significance(
            data_dict=all_results[metric],
            model_names=model_prefixes,
            display_names=display_names,
            colors=colors_to_use,
            our_model_idx=our_model_idx,
            ax=ax,
            ylabel="Pseudo-Perplexity",
            subtitle=title,
            log_scale=args.log_scale
        )

        plt.tight_layout()
        save_figure(fig, args.output_dir, f"{filename_prefix}{filename_base}")
        plt.close(fig)

    # =========================================================================
    # Plot 6: PRISM Region-wise PPL Boxplot (7 boxes, exact PPL from 53-vocab)
    # (Only generated for standard mode - not meaningful for ablation comparison)
    # =========================================================================
    if not args.ablation_mode:
        print("\nCreating Plot 5/5: PRISM Region-wise PPL (Exact 53-vocab)...")

        # Get the exact PPL column (original _LL, not _Marg_LL)
        # Convert from marginalized prefix to original prefix
        our_model_exact_col = args.our_model_col.replace('_Marg', '')
        our_model_exact_ll = f"{our_model_exact_col}_LL"

        if our_model_exact_ll not in df.columns:
            print(f"  ⚠ Exact PPL column not found: {our_model_exact_ll}")
            print(f"    Using marginalized column instead: {args.our_model_col}_LL")
            our_model_exact_ll = f"{args.our_model_col}_LL"

        print(f"  Using exact PPL column: {our_model_exact_ll}")

        # Calculate exact PPL for our model only
        exact_ppl_results = {metric: [] for metric in metrics_to_calc}
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="  Calculating exact PPL"):
            results = calculate_detailed_perplexities(row, our_model_exact_ll)
            for metric in metrics_to_calc:
                exact_ppl_results[metric].append(results.get(metric, np.nan))

        for metric in metrics_to_calc:
            exact_ppl_results[metric] = np.array(exact_ppl_results[metric])

        # Prepare data for boxplot
        all_metrics = [
            ('overall', 'Overall'),
            ('fr_gl_whole', 'FR\n(GL)'),
            ('cdr12_gl_whole', 'CDR1,2\n(GL)'),
            ('cdr3_gl_whole', 'CDR3\n(GL)'),
            ('fr_ngl_whole', 'FR\n(NGL)'),
            ('cdr12_ngl_whole', 'CDR1,2\n(NGL)'),
            ('cdr3_ngl_whole', 'CDR3\n(NGL)')
        ]

        # Prepare plot data
        plot_data = []
        for metric_key, _ in all_metrics:
            values = exact_ppl_results[metric_key]
            clean_values = values[~np.isnan(values) & ~np.isinf(values)]
            plot_data.append(clean_values)

        # Create boxplot
        fig, ax = plt.subplots(figsize=(12, 8))

        x_pos = np.arange(len(all_metrics))
        box_labels = [label for _, label in all_metrics]

        # Use different colors for GL vs NGL regions
        box_colors = [
            '#332288',  # Overall - Dark purple
            '#117733',  # FR (GL) - Green
            '#117733',  # CDR1,2 (GL) - Green
            '#117733',  # CDR3 (GL) - Green
            '#882255',  # FR (NGL) - Wine
            '#882255',  # CDR1,2 (NGL) - Wine
            '#882255',  # CDR3 (NGL) - Wine
        ]

        # Create boxplot
        bp = ax.boxplot(plot_data, positions=x_pos, widths=0.6, patch_artist=True,
                        showfliers=True, flierprops={'markersize': 3, 'alpha': 0.5})

        # Color the boxes
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_edgecolor('black')
            patch.set_linewidth(2)

        # Style medians
        for median in bp['medians']:
            median.set_color('black')
            median.set_linewidth(2)

        # Add baseline dashed line at y=53 (random baseline for 53-vocab)
        ax.axhline(y=53, color='red', linestyle='--', linewidth=2, label='Random baseline (53)')

        # Customize axes
        ax.set_xticks(x_pos)
        ax.set_xticklabels(box_labels, **FONT_CONFIG['tick'])
        ax.set_ylabel("Pseudo-Perplexity", **FONT_CONFIG['label'])
        ax.tick_params(axis='y', labelsize=FONT_CONFIG['tick']['fontsize'])

        # No title (as requested)

        # Set y-axis limits (PPL >= 1, extend to show baseline)
        y_max = max(60, np.percentile(np.concatenate(plot_data), 95) * 1.1)
        ax.set_ylim(0, y_max)

        # Add grid
        ax.yaxis.grid(True, linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)

        # Add legend for GL vs NGL and baseline
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        legend_elements = [
            Patch(facecolor='#332288', edgecolor='black', label='Overall'),
            Patch(facecolor='#117733', edgecolor='black', label='Germline (GL)'),
            Patch(facecolor='#882255', edgecolor='black', label='Non-Germline (NGL)'),
            Line2D([0], [0], color='red', linestyle='--', linewidth=2, label='Random baseline (53)')
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=FONT_CONFIG['legend']['fontsize'])

        # Thicken spines
        for spine in ax.spines.values():
            spine.set_linewidth(2)

        plt.tight_layout()
        save_figure(fig, args.output_dir, "evoab_region_ppl_exact")
        plt.close(fig)

    # =========================================================================
    # Print Summary Statistics
    # =========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    for metric in metrics_to_calc:
        print(f"\n{metric}:")
        print(f"{'Model':<20} {'Median':>10} {'Mean':>10} {'Std':>10}")
        print("-" * 50)

        for prefix in model_prefixes:
            if prefix in all_results[metric]:
                values = all_results[metric][prefix]
                clean_values = values[~np.isnan(values) & ~np.isinf(values)]
                if len(clean_values) > 0:
                    median = np.median(clean_values)
                    mean = np.mean(clean_values)
                    std = np.std(clean_values)

                    # Get display name
                    idx = model_prefixes.index(prefix)
                    display = display_names[idx]

                    print(f"{display:<20} {median:>10.4f} {mean:>10.4f} {std:>10.4f}")

    print("\n" + "=" * 80)
    print("✓ All plots saved!")
    print(f"✓ Output directory: {args.output_dir}")
    print(f"✓ PPL type: {args.ppl_type}")
    print(f"✓ PRISM model: {args.our_model_col}")
    print("✓ Files generated:")
    print("    - Overall PPL: 3 (whole, heavy, light)")
    print("    - GL Region PPL: 3 (whole, heavy, light)")
    print("    - NGL Region PPL: 3 (whole, heavy, light)")
    print("    - NGL Focused PPL: 3 (whole, heavy, light)")
    print("    - Individual Region PPL: 4 (FR_GL_H, FR_GL_L, CDR3_NGL_H, CDR3_NGL_L)")
    print("    - PRISM Region Exact: 1")
    print("    Total: 17 plots × 2 formats (PNG + SVG) = 34 files")
    print("=" * 80)


if __name__ == '__main__':
    main()
