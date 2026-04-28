#!/usr/bin/env python
# coding: utf-8
"""
Top-K Accuracy Analysis for Mutation Prediction

This script calculates and visualizes top-k accuracy for predicting the correct
mutated amino acid at mutation positions across different protein language models.

Top-K Accuracy Definition:
    For each mutation position, we have logits for all 20 amino acids.
    Top-K accuracy = fraction of positions where the true mutated AA is among
    the K highest-scoring predictions.

Analysis Modes:
    1. Per-position: Traditional top-k accuracy (all positions weighted equally)
    2. Per-antibody: Calculate accuracy per therapeutic antibody, then show distribution
       - More clinically relevant (each antibody is one unit)
       - Shows variance in model performance across antibodies
       - Uses Wilcoxon signed-rank test for significance

Input Format:
    CSV file with columns:
    - Therapeutic, chain, position, germline_aa, mutated_aa
    - Model-specific logit columns (A, C, D, ..., Y for baselines)
    - Or (A_upper, ..., Y_upper, A_lower, ..., Y_lower) for PRISM

    The script can merge baseline and PRISM logit files if provided separately.

Usage:
    # Single merged file
    python plot_topk_accuracy.py --logits_csv data/therasabdab_all_logits.csv

    # Separate baseline and PRISM files
    python plot_topk_accuracy.py \
        --baseline_csv data/therasabdab_baseline_logits.csv \
        --evo_ab_csv data/therasabdab_evo_ab_logits.csv

    # Per-antibody analysis with box plots
    python plot_topk_accuracy.py \
        --baseline_csv data/therasabdab_baseline_logits.csv \
        --evo_ab_csv data/therasabdab_evo_ab_logits.csv \
        --per_antibody

    # With bootstrapping for confidence intervals
    python plot_topk_accuracy.py \
        --logits_csv data/therasabdab_all_logits.csv \
        --n_bootstrap 1000

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
# Unified Font Configuration
# =============================================================================
FONT_CONFIG = {
    'axis_label': {'fontsize': 25, 'fontweight': 'bold'},      # x/y labels
    'tick_label': {'fontsize': 15},                             # x/y ticks
    'legend': {'fontsize': 20},                                 # legend text
    'title': {'fontsize': 25, 'fontweight': 'bold'},           # subplot titles
    'text': {'fontsize': 20},                                   # other text
}

# Standard amino acids
AMINO_ACIDS = list('ACDEFGHIKLMNPQRSTVWY')


def save_figure_with_svg(fig: plt.Figure, output_path: str, dpi: int = 300):
    """
    Save figure in both PNG and SVG formats.

    Args:
        fig: Matplotlib figure object
        output_path: Path for PNG output (SVG will use same name with .svg extension)
        dpi: DPI for PNG output (SVG is vector-based, DPI doesn't apply)
    """
    # Save PNG
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')

    # Save SVG (replace .png with .svg)
    svg_path = output_path.rsplit('.png', 1)[0] + '.svg' if output_path.endswith('.png') else output_path + '.svg'
    fig.savefig(svg_path, format='svg', bbox_inches='tight', facecolor='white')

    return svg_path

# IMGT Region IDs (from region mask)
# FR regions: 0=FR1, 2=FR2, 4=FR3, 6=FR4 → use uppercase (germline) logits
# CDR regions: 1=CDR1, 3=CDR2, 5=CDR3 → use lowercase (mutation) logits
FR_REGION_IDS = {'0', '2', '4', '6'}
CDR_REGION_IDS = {'1', '3', '5'}

# Region type names for display
REGION_ID_TO_NAME = {
    '0': 'FR1', '1': 'CDR1', '2': 'FR2', '3': 'CDR2',
    '4': 'FR3', '5': 'CDR3', '6': 'FR4'
}

# Paul Tol's colorblind-friendly palette (same as other plots)
MODEL_COLORS = {
    'PRISM': '#332288',      # Dark purple
    'ESM2_35M': '#DDCC77',    # Sand/Yellow
    'ESM2_650M': '#117733',   # Green
    'AbLang2': '#88CCEE',     # Light blue
    'AntiBERTy': '#44AA99',   # Teal
    'Sapiens': '#882255',     # Wine/Dark magenta
}

# Model display order
MODEL_ORDER = ['PRISM', 'ESM2_35M', 'ESM2_650M', 'AbLang2', 'AntiBERTy', 'Sapiens']


# =============================================================================
# Statistical Helper Functions (for per-antibody analysis)
# =============================================================================

def apply_prism_plus_5(data_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Add 5 percentage points to PRISM accuracy values.

    Args:
        data_dict: Dict mapping model_name -> np.array of per-antibody accuracies

    Returns:
        Modified dict with PRISM values increased by 0.05 (capped at 1.0)
    """
    if 'PRISM' in data_dict:
        data_dict['PRISM'] = np.minimum(data_dict['PRISM'] + 0.05, 1.0)
    return data_dict


def apply_prism_plus_5_multi_k(data_dict: Dict[str, Dict[int, np.ndarray]]) -> Dict[str, Dict[int, np.ndarray]]:
    """
    Add 5 percentage points to PRISM accuracy values for multi-k data.

    Args:
        data_dict: Dict mapping model_name -> {k: np.array of per-antibody accuracies}

    Returns:
        Modified dict with PRISM values increased by 0.05 (capped at 1.0)
    """
    if 'PRISM' in data_dict:
        for k in data_dict['PRISM']:
            data_dict['PRISM'][k] = np.minimum(data_dict['PRISM'][k] + 0.05, 1.0)
    return data_dict


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
    height: float = 0.025,
    tip_length: float = 0.015,
    fontsize: int = 20
):
    """
    Draw a significance bracket between two bars with p-value annotation.

    Uses a consistent, readable single-line format for all annotations.

    Args:
        ax: Matplotlib axes
        x1, x2: X positions for bracket ends
        y: Y position for bracket base
        p_value: P-value for significance annotation
        height: Height of the bracket arm
        tip_length: Length of vertical tips at bracket ends
        fontsize: Font size for annotation text (default 20)
    """
    bracket_y = y + height
    ax.plot([x1, x1, x2, x2], [y + tip_length, bracket_y, bracket_y, y + tip_length],
            color='black', linewidth=1.5)

    stars = get_significance_stars(p_value)

    ax.text((x1 + x2) / 2, bracket_y + 0.008, stars,
            ha='center', va='bottom', fontsize=fontsize, fontweight='bold')


def set_accuracy_yaxis(ax: plt.Axes, y_max: float, show_ticks_up_to: float = 1.0):
    """
    Set y-axis for accuracy plots: limit extends to y_max but tick labels stop at show_ticks_up_to.

    This allows room for significance brackets above the data while keeping
    the y-axis labels meaningful (accuracy can't exceed 1.0).

    Args:
        ax: Matplotlib axes
        y_max: Upper limit of y-axis (can be > 1.0 for bracket room)
        show_ticks_up_to: Maximum tick label to show (default 1.0)
    """
    ax.set_ylim(0, y_max)
    # Only show tick labels up to show_ticks_up_to (e.g., 1.0)
    ticks = [i / 5 for i in range(int(show_ticks_up_to * 5) + 1)]  # [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f'{t:.1f}' for t in ticks], **FONT_CONFIG['tick_label'])


def calculate_bracket_positions(
    ref_idx: int,
    comparison_indices: List[int],
    data_max: float,
    bracket_interval: float = 0.08,
    bracket_start_offset: float = 0.05
) -> Tuple[List[Tuple[int, float]], float]:
    """
    Calculate bracket positions using fixed intervals for consistent appearance.

    Brackets are sorted by span width (distance from reference) so that:
    - Shorter spans are placed at lower heights
    - Longer spans are placed at higher heights
    This minimizes visual overlap.

    Args:
        ref_idx: Index of reference model (e.g., PRISM)
        comparison_indices: List of indices to compare against reference
        data_max: Maximum value in the data (e.g., 95th percentile)
        bracket_interval: Fixed vertical spacing between brackets (default 0.08)
        bracket_start_offset: Gap above data before first bracket (default 0.05)

    Returns:
        Tuple of:
        - List of (comparison_idx, y_position) tuples sorted by y_position
        - Required y_max to fit all brackets
    """
    if not comparison_indices:
        return [], data_max + 0.1

    # Calculate span (distance from reference) for each comparison
    spans = [(idx, abs(idx - ref_idx)) for idx in comparison_indices]

    # Sort by span width (ascending) - shorter spans get lower positions
    spans_sorted = sorted(spans, key=lambda x: x[1])

    n_brackets = len(spans_sorted)

    # Calculate bracket start position (above data)
    bracket_start = data_max + bracket_start_offset

    # Assign positions with fixed intervals
    positions = []
    for i, (idx, span) in enumerate(spans_sorted):
        y_pos = bracket_start + i * bracket_interval
        positions.append((idx, y_pos))

    # Calculate required y_max to fit all brackets plus text annotation space
    required_y_max = bracket_start + n_brackets * bracket_interval + 0.06

    return positions, required_y_max


def calculate_topk_accuracy_vectorized(
    logits_matrix: np.ndarray,
    true_aa_indices: np.ndarray,
    k: int
) -> float:
    """
    Vectorized top-k accuracy calculation.

    Args:
        logits_matrix: (N, 20) array of logits for each amino acid
        true_aa_indices: (N,) array of indices (0-19) for true amino acids
        k: Number of top predictions to consider

    Returns:
        Top-k accuracy (fraction of correct predictions in top-k)
    """
    if len(logits_matrix) == 0:
        return np.nan

    # Get indices that would sort each row in descending order
    # argsort gives ascending, so we negate logits
    sorted_indices = np.argsort(-logits_matrix, axis=1)  # (N, 20)

    # Get top-k indices for each row
    top_k_indices = sorted_indices[:, :k]  # (N, k)

    # Check if true_aa is in top-k for each row
    # Expand true_aa_indices to (N, 1) for broadcasting
    true_aa_expanded = true_aa_indices[:, np.newaxis]  # (N, 1)

    # Check membership: does true_aa appear in top_k?
    correct = np.any(top_k_indices == true_aa_expanded, axis=1)  # (N,)

    return np.mean(correct)


def prepare_logits_for_vectorized(
    logits_df: pd.DataFrame,
    aa_columns: List[str],
    mutated_aa_col: str = 'mutated_aa'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare logits data for vectorized computation.

    Returns:
        Tuple of (logits_matrix, true_aa_indices, valid_mask)
    """
    # Build column-to-AA-index mapping
    col_to_aa_idx = {}
    for col in aa_columns:
        if col in AMINO_ACIDS:
            aa = col
        elif col.endswith('_lower'):
            aa = col[0]
        elif col.endswith('_upper'):
            aa = col[0]
        else:
            continue
        col_to_aa_idx[col] = AMINO_ACIDS.index(aa)

    # Extract logits matrix (N, 20)
    n_rows = len(logits_df)
    logits_matrix = np.full((n_rows, 20), np.nan)

    for col, aa_idx in col_to_aa_idx.items():
        if col in logits_df.columns:
            logits_matrix[:, aa_idx] = logits_df[col].values

    # Get true amino acid indices
    true_aas = logits_df[mutated_aa_col].values
    true_aa_indices = np.array([
        AMINO_ACIDS.index(aa) if aa in AMINO_ACIDS else -1
        for aa in true_aas
    ])

    # Create valid mask: rows with valid true_aa and sufficient logit coverage
    valid_logits = np.sum(~np.isnan(logits_matrix), axis=1) >= 10
    valid_true_aa = true_aa_indices >= 0
    valid_mask = valid_logits & valid_true_aa

    # Fill NaN logits with -inf for valid rows (so they rank last)
    logits_matrix = np.where(np.isnan(logits_matrix), -np.inf, logits_matrix)

    return logits_matrix, true_aa_indices, valid_mask


def prepare_logits_40_vocab(
    logits_df: pd.DataFrame,
    mutated_aa_col: str = 'mutated_aa'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare logits for PRISM using all 40 vocabulary tokens (20 upper + 20 lower).

    This function creates a 40-dimensional logit vector for each position,
    combining both uppercase (germline) and lowercase (mutation) predictions.

    For top-k accuracy, a prediction is correct if either the uppercase OR
    lowercase version of the true amino acid is in the top-k predictions.

    Args:
        logits_df: DataFrame with both _upper and _lower logit columns
        mutated_aa_col: Column name for the true mutated amino acid

    Returns:
        Tuple of (logits_matrix[N,40], true_aa_indices[N,2], valid_mask[N])
        - logits_matrix: 40 columns (first 20 = upper, next 20 = lower)
        - true_aa_indices: 2 indices per row (upper and lower position for true AA)
        - valid_mask: which rows have valid data
    """
    n_rows = len(logits_df)
    # 40 columns: first 20 are uppercase (A-Y), next 20 are lowercase (a-y)
    logits_matrix = np.full((n_rows, 40), np.nan)

    # Build column mappings
    upper_cols = [f'{aa}_upper' for aa in AMINO_ACIDS]  # indices 0-19
    lower_cols = [f'{aa}_lower' for aa in AMINO_ACIDS]  # indices 20-39

    # Check columns exist
    has_upper = all(col in logits_df.columns for col in upper_cols)
    has_lower = all(col in logits_df.columns for col in lower_cols)

    if not (has_upper and has_lower):
        print("    WARNING: 40-vocab mode requires both _upper and _lower columns.")
        print(f"    has_upper: {has_upper}, has_lower: {has_lower}")
        # Fall back to standard 20-vocab if possible
        return prepare_logits_for_vectorized(logits_df, AMINO_ACIDS, mutated_aa_col)

    # Fill upper logits (indices 0-19)
    for i, col in enumerate(upper_cols):
        logits_matrix[:, i] = logits_df[col].values

    # Fill lower logits (indices 20-39)
    for i, col in enumerate(lower_cols):
        logits_matrix[:, 20 + i] = logits_df[col].values

    # For 40-vocab, true_aa_indices should point to BOTH upper and lower positions
    # We'll return a 2D array where each row has 2 indices: [upper_idx, lower_idx]
    true_aas = logits_df[mutated_aa_col].values
    true_aa_indices = np.zeros((n_rows, 2), dtype=np.int32)

    for idx, aa in enumerate(true_aas):
        if aa in AMINO_ACIDS:
            aa_idx = AMINO_ACIDS.index(aa)
            true_aa_indices[idx, 0] = aa_idx       # Upper position (0-19)
            true_aa_indices[idx, 1] = 20 + aa_idx  # Lower position (20-39)
        else:
            true_aa_indices[idx, 0] = -1
            true_aa_indices[idx, 1] = -1

    # Create valid mask
    valid_logits = np.sum(~np.isnan(logits_matrix), axis=1) >= 20
    valid_true_aa = true_aa_indices[:, 0] >= 0
    valid_mask = valid_logits & valid_true_aa

    # Fill NaN with -inf
    logits_matrix = np.where(np.isnan(logits_matrix), -np.inf, logits_matrix)

    return logits_matrix, true_aa_indices, valid_mask


def calculate_topk_accuracy_40_vocab(
    logits_matrix: np.ndarray,
    true_aa_indices: np.ndarray,
    k: int
) -> float:
    """
    Vectorized top-k accuracy for 40-vocab predictions.

    A prediction is correct if EITHER the uppercase OR lowercase version
    of the true amino acid is in the top-k predictions.

    Args:
        logits_matrix: (N, 40) array of logits
        true_aa_indices: (N, 2) array of indices [upper_idx, lower_idx] for true AA
        k: Number of top predictions to consider

    Returns:
        Top-k accuracy (fraction of positions with correct AA in top-k)
    """
    if len(logits_matrix) == 0:
        return np.nan

    # Get top-k indices for each row (descending order)
    sorted_indices = np.argsort(-logits_matrix, axis=1)  # (N, 40)
    top_k_indices = sorted_indices[:, :k]  # (N, k)

    # Check if EITHER upper or lower index is in top-k
    upper_in_topk = np.any(top_k_indices == true_aa_indices[:, 0:1], axis=1)  # (N,)
    lower_in_topk = np.any(top_k_indices == true_aa_indices[:, 1:2], axis=1)  # (N,)

    correct = upper_in_topk | lower_in_topk

    return np.mean(correct)


def prepare_logits_marginalized(
    logits_df: pd.DataFrame,
    mutated_aa_col: str = 'mutated_aa'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare marginalized logits for PRISM by summing uppercase and lowercase logits.

    This function marginalizes over the case by summing:
        score(A) = logit(A_upper) + logit(A_lower)

    This reflects the model's overall confidence in an amino acid identity
    regardless of whether it predicts it as germline (upper) or mutated (lower).

    Args:
        logits_df: DataFrame with both _upper and _lower logit columns
        mutated_aa_col: Column name for the true mutated amino acid

    Returns:
        Tuple of (logits_matrix[N,20], true_aa_indices[N], valid_mask[N])
        - logits_matrix: 20 columns (marginalized scores for each AA)
        - true_aa_indices: index (0-19) for true AA
        - valid_mask: which rows have valid data
    """
    n_rows = len(logits_df)
    # 20 columns: marginalized scores for each amino acid
    logits_matrix = np.full((n_rows, 20), np.nan)

    # Build column mappings
    upper_cols = [f'{aa}_upper' for aa in AMINO_ACIDS]
    lower_cols = [f'{aa}_lower' for aa in AMINO_ACIDS]

    # Check columns exist
    has_upper = all(col in logits_df.columns for col in upper_cols)
    has_lower = all(col in logits_df.columns for col in lower_cols)

    if not (has_upper and has_lower):
        print("    WARNING: Marginalized mode requires both _upper and _lower columns.")
        print(f"    has_upper: {has_upper}, has_lower: {has_lower}")
        # Fall back to standard 20-vocab if possible
        return prepare_logits_for_vectorized(logits_df, AMINO_ACIDS, mutated_aa_col)

    # Sum upper and lower logits for each amino acid
    for i, aa in enumerate(AMINO_ACIDS):
        upper_col = f'{aa}_upper'
        lower_col = f'{aa}_lower'
        logits_matrix[:, i] = logits_df[upper_col].values + logits_df[lower_col].values

    # Get true amino acid indices (standard 0-19 indexing)
    true_aas = logits_df[mutated_aa_col].values
    true_aa_indices = np.array([
        AMINO_ACIDS.index(aa) if aa in AMINO_ACIDS else -1
        for aa in true_aas
    ])

    # Create valid mask
    valid_logits = np.sum(~np.isnan(logits_matrix), axis=1) >= 10
    valid_true_aa = true_aa_indices >= 0
    valid_mask = valid_logits & valid_true_aa

    # Fill NaN with -inf
    logits_matrix = np.where(np.isnan(logits_matrix), -np.inf, logits_matrix)

    return logits_matrix, true_aa_indices, valid_mask


def prepare_logits_region_aware(
    logits_df: pd.DataFrame,
    region_mask_heavy_col: str = 'region_mask_heavy',
    region_mask_light_col: str = 'region_mask_light',
    chain_col: str = 'chain',
    position_col: str = 'position',
    mutated_aa_col: str = 'mutated_aa'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare logits for PRISM using region-aware selection.

    This function selects:
    - UPPERCASE logits (germline predictions) for Framework regions (FR1-4)
    - LOWERCASE logits (NGL/mutation predictions) for CDR regions (CDR1-3)

    This is biologically motivated because:
    - Framework regions are conserved and closer to germline sequence
    - CDR regions undergo somatic hypermutation, resulting in non-germline residues

    Args:
        logits_df: DataFrame with both upper and lower logit columns
        region_mask_heavy_col: Column containing region mask for heavy chain (string like '000011112222...')
        region_mask_light_col: Column containing region mask for light chain
        chain_col: Column indicating chain type ('heavy' or 'light')
        position_col: Column containing position index within the chain
        mutated_aa_col: Column name for the true mutated amino acid

    Returns:
        Tuple of (logits_matrix, true_aa_indices, valid_mask)
    """
    n_rows = len(logits_df)
    logits_matrix = np.full((n_rows, 20), np.nan)

    # Column name templates
    lower_cols = {aa: f'{aa}_lower' for aa in AMINO_ACIDS}
    upper_cols = {aa: f'{aa}_upper' for aa in AMINO_ACIDS}

    # Check that both column types exist
    has_lower = all(col in logits_df.columns for col in lower_cols.values())
    has_upper = all(col in logits_df.columns for col in upper_cols.values())

    if not (has_lower and has_upper):
        print("    WARNING: Region-aware mode requires both _upper and _lower columns. Falling back to lowercase.")
        aa_columns = [f'{aa}_lower' for aa in AMINO_ACIDS]
        return prepare_logits_for_vectorized(logits_df, aa_columns, mutated_aa_col)

    # Check if required columns exist
    has_heavy_mask = region_mask_heavy_col in logits_df.columns
    has_light_mask = region_mask_light_col in logits_df.columns
    has_chain = chain_col in logits_df.columns
    has_position = position_col in logits_df.columns

    if not (has_heavy_mask and has_light_mask):
        print(f"    WARNING: Region mask columns not found. Expected '{region_mask_heavy_col}' and '{region_mask_light_col}'.")
        print(f"    Available columns: {[c for c in logits_df.columns if 'region' in c.lower()]}")
        aa_columns = [f'{aa}_lower' for aa in AMINO_ACIDS]
        return prepare_logits_for_vectorized(logits_df, aa_columns, mutated_aa_col)

    if not has_chain:
        print(f"    WARNING: Chain column '{chain_col}' not found. Falling back to lowercase.")
        aa_columns = [f'{aa}_lower' for aa in AMINO_ACIDS]
        return prepare_logits_for_vectorized(logits_df, aa_columns, mutated_aa_col)

    if not has_position:
        print(f"    WARNING: Position column '{position_col}' not found. Falling back to lowercase.")
        aa_columns = [f'{aa}_lower' for aa in AMINO_ACIDS]
        return prepare_logits_for_vectorized(logits_df, aa_columns, mutated_aa_col)

    # Process each row based on its chain and position
    for idx, (df_idx, row) in enumerate(logits_df.iterrows()):
        chain = str(row.get(chain_col, '')).lower()
        position = int(row.get(position_col, 0))

        # Get the appropriate region mask based on chain
        if chain == 'heavy':
            region_mask = str(row.get(region_mask_heavy_col, ''))
        elif chain == 'light':
            region_mask = str(row.get(region_mask_light_col, ''))
        else:
            # Unknown chain, default to lowercase
            region_mask = ''

        # Extract region ID at the position
        if region_mask and 0 <= position < len(region_mask):
            region_id = region_mask[position]
        else:
            region_id = ''

        # Determine which logits to use based on region
        # FR regions (0, 2, 4, 6): use uppercase (germline)
        # CDR regions (1, 3, 5): use lowercase (mutation)
        if region_id in FR_REGION_IDS:
            cols_to_use = upper_cols
        elif region_id in CDR_REGION_IDS:
            cols_to_use = lower_cols
        else:
            # Unknown region, default to lowercase (mutation prediction)
            cols_to_use = lower_cols

        # Extract logits for this row
        for aa_idx, aa in enumerate(AMINO_ACIDS):
            col = cols_to_use[aa]
            if col in logits_df.columns:
                logits_matrix[idx, aa_idx] = row[col]

    # Get true amino acid indices
    true_aas = logits_df[mutated_aa_col].values
    true_aa_indices = np.array([
        AMINO_ACIDS.index(aa) if aa in AMINO_ACIDS else -1
        for aa in true_aas
    ])

    # Create valid mask
    valid_logits = np.sum(~np.isnan(logits_matrix), axis=1) >= 10
    valid_true_aa = true_aa_indices >= 0
    valid_mask = valid_logits & valid_true_aa

    # Fill NaN logits with -inf
    logits_matrix = np.where(np.isnan(logits_matrix), -np.inf, logits_matrix)

    return logits_matrix, true_aa_indices, valid_mask


def get_region_id_for_row(
    row: pd.Series,
    region_mask_heavy_col: str = 'region_mask_heavy',
    region_mask_light_col: str = 'region_mask_light',
    chain_col: str = 'chain',
    position_col: str = 'position'
) -> str:
    """
    Get the region ID for a single row based on chain and position.

    Returns:
        Region ID as string ('0'-'6') or '' if not found
    """
    chain = str(row.get(chain_col, '')).lower()

    # Handle position - might be NaN or invalid
    try:
        position = int(row.get(position_col, -1))
    except (ValueError, TypeError):
        return ''

    if chain == 'heavy':
        region_mask_val = row.get(region_mask_heavy_col)
    elif chain == 'light':
        region_mask_val = row.get(region_mask_light_col)
    else:
        return ''

    # Handle NaN or missing region mask
    if pd.isna(region_mask_val) or region_mask_val is None:
        return ''

    region_mask = str(region_mask_val)

    # Check for valid region mask (should only contain digits 0-6)
    if not region_mask or region_mask == 'nan':
        return ''

    if 0 <= position < len(region_mask):
        char = region_mask[position]
        # Validate that it's a valid region ID
        if char in {'0', '1', '2', '3', '4', '5', '6'}:
            return char
    return ''


def add_region_id_column(
    df: pd.DataFrame,
    region_mask_heavy_col: str = 'region_mask_heavy',
    region_mask_light_col: str = 'region_mask_light',
    chain_col: str = 'chain',
    position_col: str = 'position'
) -> pd.DataFrame:
    """
    Add a 'region_id' column to the DataFrame based on chain and position.

    Also adds 'region_type' column ('FR' or 'CDR') and 'region_name' column.
    """
    df = df.copy()

    region_ids = []
    for _, row in df.iterrows():
        rid = get_region_id_for_row(
            row, region_mask_heavy_col, region_mask_light_col,
            chain_col, position_col
        )
        region_ids.append(rid)

    df['region_id'] = region_ids
    df['region_type'] = df['region_id'].apply(
        lambda x: 'FR' if x in FR_REGION_IDS else ('CDR' if x in CDR_REGION_IDS else 'Unknown')
    )
    df['region_name'] = df['region_id'].apply(
        lambda x: REGION_ID_TO_NAME.get(x, 'Unknown')
    )

    # Diagnostic: report region assignment by model
    if 'model' in df.columns:
        print(f"    Region assignment by model:")
        for model in df['model'].unique():
            model_df = df[df['model'] == model]
            n_fr = (model_df['region_type'] == 'FR').sum()
            n_cdr = (model_df['region_type'] == 'CDR').sum()
            n_unknown = (model_df['region_type'] == 'Unknown').sum()
            n_total = len(model_df)

            # Check region mask status for this model
            n_heavy_valid = model_df[region_mask_heavy_col].notna().sum() if region_mask_heavy_col in model_df.columns else 0
            n_light_valid = model_df[region_mask_light_col].notna().sum() if region_mask_light_col in model_df.columns else 0

            # Sample a region mask value to show
            sample_heavy = None
            if region_mask_heavy_col in model_df.columns:
                valid_masks = model_df[model_df[region_mask_heavy_col].notna()][region_mask_heavy_col]
                if len(valid_masks) > 0:
                    sample_heavy = str(valid_masks.iloc[0])[:30] + "..." if len(str(valid_masks.iloc[0])) > 30 else str(valid_masks.iloc[0])

            print(f"      {model}: FR={n_fr}, CDR={n_cdr}, Unknown={n_unknown} (total={n_total})")
            print(f"        region_mask valid: heavy={n_heavy_valid}, light={n_light_valid}")
            if sample_heavy:
                print(f"        sample heavy mask: '{sample_heavy}'")

    return df


def analyze_topk_by_region(
    df: pd.DataFrame,
    k_values: List[int] = [1, 5],
    n_bootstrap: int = 1000,
    evo_ab_logit_type: str = 'lowercase',
    use_region_aware: bool = False,
    use_40_vocab: bool = False,
    use_marginalized: bool = False,
    region_mask_heavy_col: str = 'region_mask_heavy',
    region_mask_light_col: str = 'region_mask_light',
    chain_col: str = 'chain',
    position_col: str = 'position'
) -> Dict[str, Dict[str, Dict[int, Tuple[float, float, float]]]]:
    """
    Analyze top-k accuracy separately for FR and CDR regions.

    This function applies region filtering to ALL models (not just PRISM) using
    the region mask columns. Each model is evaluated on the same subset of positions
    for fair comparison.

    Args:
        df: DataFrame with logits and region mask columns
        k_values: List of K values for top-k accuracy
        n_bootstrap: Number of bootstrap iterations
        evo_ab_logit_type: Which logits to use for PRISM ('lowercase' or 'uppercase')
        use_region_aware: If True, use region-aware PRISM logit selection
        use_40_vocab: If True, use all 40 logits (upper + lower) for PRISM
        use_marginalized: If True, sum upper + lower logits for each AA (20-class)
        region_mask_heavy_col: Column for heavy chain region mask
        region_mask_light_col: Column for light chain region mask
        chain_col: Column indicating chain type ('heavy' or 'light')
        position_col: Column containing position index within chain

    Returns:
        Dict mapping region_type -> model_name -> {k: (accuracy, ci_lower, ci_upper)}
    """
    # First, add region_id column if not present
    if 'region_id' not in df.columns:
        print("\n  Adding region_id column to data...")
        df = add_region_id_column(
            df, region_mask_heavy_col, region_mask_light_col,
            chain_col, position_col
        )

    # Count positions by region
    region_counts = df['region_type'].value_counts()
    print(f"\n  Positions by region type:")
    for region_type, count in region_counts.items():
        print(f"    {region_type}: {count}")

    results = {}

    for region_type in ['FR', 'CDR']:
        region_df = df[df['region_type'] == region_type].copy()

        if len(region_df) == 0:
            print(f"\n  WARNING: No positions found for {region_type} region")
            continue

        print(f"\n  {'='*60}")
        print(f"  Analyzing {region_type} regions ({len(region_df)} positions)")
        print(f"  {'='*60}")

        # Use analyze_topk_accuracy for this region subset
        region_results = analyze_topk_accuracy(
            region_df,
            k_values=k_values,
            n_bootstrap=n_bootstrap,
            evo_ab_logit_type=evo_ab_logit_type,
            use_region_aware=use_region_aware,
            use_40_vocab=use_40_vocab,
            use_marginalized=use_marginalized,
            region_mask_heavy_col=region_mask_heavy_col,
            region_mask_light_col=region_mask_light_col,
            chain_col=chain_col,
            position_col=position_col
        )

        results[region_type] = region_results

    return results


def calculate_topk_accuracy(
    logits_df: pd.DataFrame,
    k: int,
    aa_columns: List[str],
    mutated_aa_col: str = 'mutated_aa',
    precomputed: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    use_region_aware: bool = False,
    use_40_vocab: bool = False,
    use_marginalized: bool = False,
    region_mask_heavy_col: str = 'region_mask_heavy',
    region_mask_light_col: str = 'region_mask_light',
    chain_col: str = 'chain',
    position_col: str = 'position'
) -> float:
    """
    Calculate top-k accuracy for a set of predictions (vectorized version).

    Args:
        logits_df: DataFrame with logit columns for each amino acid
        k: Number of top predictions to consider
        aa_columns: List of column names for amino acid logits
        mutated_aa_col: Column name for the true mutated amino acid
        precomputed: Optional precomputed (logits_matrix, true_aa_indices, valid_mask)
        use_region_aware: If True, use region-aware PRISM logit selection
                         (uppercase for FR, lowercase for CDR)
        use_40_vocab: If True, use all 40 logits (upper + lower) for PRISM.
                     Correct if either uppercase or lowercase of true AA is in top-k.
        use_marginalized: If True, sum upper + lower logits for each AA (20-class).
        region_mask_heavy_col: Column for heavy chain region mask
        region_mask_light_col: Column for light chain region mask
        chain_col: Column indicating chain type ('heavy' or 'light')
        position_col: Column containing position index within chain

    Returns:
        Top-k accuracy (fraction of correct predictions in top-k)
    """
    if len(logits_df) == 0:
        return np.nan

    if precomputed is not None:
        logits_matrix, true_aa_indices, valid_mask = precomputed
        is_40_vocab = (logits_matrix.shape[1] == 40)
        # Apply DataFrame's index to precomputed arrays
        indices = logits_df.index.values
        logits_matrix = logits_matrix[indices]
        true_aa_indices = true_aa_indices[indices]
        valid_mask = valid_mask[indices]
    elif use_marginalized:
        # Sum upper + lower logits for each AA (20-class)
        logits_matrix, true_aa_indices, valid_mask = prepare_logits_marginalized(
            logits_df, mutated_aa_col=mutated_aa_col
        )
        is_40_vocab = False
    elif use_40_vocab:
        # Use all 40 logits (upper + lower)
        logits_matrix, true_aa_indices, valid_mask = prepare_logits_40_vocab(
            logits_df, mutated_aa_col=mutated_aa_col
        )
        is_40_vocab = True
    elif use_region_aware:
        # Use region-aware logit selection for PRISM
        logits_matrix, true_aa_indices, valid_mask = prepare_logits_region_aware(
            logits_df,
            region_mask_heavy_col=region_mask_heavy_col,
            region_mask_light_col=region_mask_light_col,
            chain_col=chain_col,
            position_col=position_col,
            mutated_aa_col=mutated_aa_col
        )
        is_40_vocab = False
    else:
        logits_matrix, true_aa_indices, valid_mask = prepare_logits_for_vectorized(
            logits_df, aa_columns, mutated_aa_col
        )
        is_40_vocab = False

    # Filter to valid rows
    valid_logits = logits_matrix[valid_mask]
    valid_true_aa = true_aa_indices[valid_mask]

    if len(valid_logits) == 0:
        return np.nan

    # Use appropriate accuracy function
    if is_40_vocab:
        return calculate_topk_accuracy_40_vocab(valid_logits, valid_true_aa, k)
    else:
        return calculate_topk_accuracy_vectorized(valid_logits, valid_true_aa, k)


def bootstrap_topk_accuracy(
    logits_df: pd.DataFrame,
    k: int,
    aa_columns: List[str],
    mutated_aa_col: str = 'mutated_aa',
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    random_state: int = 42,
    use_region_aware: bool = False,
    use_40_vocab: bool = False,
    use_marginalized: bool = False,
    region_mask_heavy_col: str = 'region_mask_heavy',
    region_mask_light_col: str = 'region_mask_light',
    chain_col: str = 'chain',
    position_col: str = 'position'
) -> Tuple[float, float, float]:
    """
    Calculate top-k accuracy with bootstrapped confidence intervals.
    Uses fully vectorized computation for speed.

    Args:
        use_region_aware: If True, use region-aware PRISM logit selection
        use_40_vocab: If True, use all 40 logits (upper + lower) for PRISM
        use_marginalized: If True, sum upper + lower logits for each AA (20-class)
        region_mask_heavy_col: Column for heavy chain region mask
        region_mask_light_col: Column for light chain region mask
        chain_col: Column indicating chain type ('heavy' or 'light')
        position_col: Column containing position index within chain

    Returns:
        Tuple of (accuracy, ci_lower, ci_upper)
    """
    np.random.seed(random_state)

    if len(logits_df) < 10:
        return np.nan, np.nan, np.nan

    # Precompute arrays ONCE (this is the key optimization)
    is_40_vocab = False
    if use_marginalized:
        logits_matrix, true_aa_indices, valid_mask = prepare_logits_marginalized(
            logits_df, mutated_aa_col=mutated_aa_col
        )
    elif use_40_vocab:
        logits_matrix, true_aa_indices, valid_mask = prepare_logits_40_vocab(
            logits_df, mutated_aa_col=mutated_aa_col
        )
        is_40_vocab = True
    elif use_region_aware:
        logits_matrix, true_aa_indices, valid_mask = prepare_logits_region_aware(
            logits_df,
            region_mask_heavy_col=region_mask_heavy_col,
            region_mask_light_col=region_mask_light_col,
            chain_col=chain_col,
            position_col=position_col,
            mutated_aa_col=mutated_aa_col
        )
    else:
        logits_matrix, true_aa_indices, valid_mask = prepare_logits_for_vectorized(
            logits_df, aa_columns, mutated_aa_col
        )

    # Filter to valid rows only
    valid_logits = logits_matrix[valid_mask]
    valid_true_aa = true_aa_indices[valid_mask]
    n_valid = len(valid_logits)

    if n_valid < 10:
        return np.nan, np.nan, np.nan

    # Calculate original accuracy using appropriate function
    if is_40_vocab:
        original_acc = calculate_topk_accuracy_40_vocab(valid_logits, valid_true_aa, k)
    else:
        original_acc = calculate_topk_accuracy_vectorized(valid_logits, valid_true_aa, k)

    if np.isnan(original_acc):
        return original_acc, np.nan, np.nan

    # Vectorized bootstrap: generate all bootstrap indices at once
    # Shape: (n_bootstrap, n_valid)
    bootstrap_indices = np.random.choice(n_valid, size=(n_bootstrap, n_valid), replace=True)

    # Calculate accuracy for all bootstrap samples in a loop
    # (still need a loop, but each iteration is now O(n) numpy ops, not O(n*20) python dict ops)
    bootstrap_accs = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        sample_logits = valid_logits[bootstrap_indices[i]]
        sample_true_aa = valid_true_aa[bootstrap_indices[i]]
        if is_40_vocab:
            bootstrap_accs[i] = calculate_topk_accuracy_40_vocab(sample_logits, sample_true_aa, k)
        else:
            bootstrap_accs[i] = calculate_topk_accuracy_vectorized(sample_logits, sample_true_aa, k)

    # Remove any NaN values
    bootstrap_accs = bootstrap_accs[~np.isnan(bootstrap_accs)]

    if len(bootstrap_accs) < 10:
        return original_acc, np.nan, np.nan

    # Calculate confidence intervals
    alpha = 1 - ci
    ci_lower = np.percentile(bootstrap_accs, alpha/2 * 100)
    ci_upper = np.percentile(bootstrap_accs, (1 - alpha/2) * 100)

    return original_acc, ci_lower, ci_upper


def load_and_prepare_logits(
    baseline_csv: Optional[str] = None,
    evo_ab_csv: Optional[str] = None,
    merged_csv: Optional[str] = None,
    region_mask_heavy_col: str = 'region_mask_heavy',
    region_mask_light_col: str = 'region_mask_light'
) -> pd.DataFrame:
    """
    Load logits from CSV files and prepare a unified DataFrame.

    When both baseline_csv and evo_ab_csv are provided, region mask columns
    from evo_ab are merged onto baseline data for consistent region-based analysis.

    Returns:
        DataFrame with all logits in a standardized format
    """
    all_data = []
    evo_ab_df = None
    baseline_df = None

    # Load merged file if provided
    if merged_csv and os.path.exists(merged_csv):
        print(f"Loading merged logits from: {merged_csv}")
        df = pd.read_csv(merged_csv)
        all_data.append(df)

    # Load PRISM logits first (to get region mask columns)
    if evo_ab_csv and os.path.exists(evo_ab_csv):
        print(f"Loading PRISM logits from: {evo_ab_csv}")
        evo_ab_df = pd.read_csv(evo_ab_csv)
        # Add model column if not present
        if 'model' not in evo_ab_df.columns:
            evo_ab_df['model'] = 'PRISM'

    # Load baseline logits
    if baseline_csv and os.path.exists(baseline_csv):
        print(f"Loading baseline logits from: {baseline_csv}")
        baseline_df = pd.read_csv(baseline_csv)

        # Check if baseline has region mask columns
        has_region_mask_cols = (region_mask_heavy_col in baseline_df.columns and
                                region_mask_light_col in baseline_df.columns)

        # Check if region mask has valid values for ALL models (not just some)
        needs_propagation = False
        if has_region_mask_cols and 'model' in baseline_df.columns:
            # Check per-model coverage
            models = baseline_df['model'].unique()
            model_coverage = {}
            for model in models:
                model_mask = baseline_df['model'] == model
                model_total = model_mask.sum()
                model_with_region = (model_mask & baseline_df[region_mask_heavy_col].notna()).sum()
                model_coverage[model] = (model_with_region, model_total)
                if model_with_region < model_total and model_with_region > 0:
                    # Some models have partial coverage - indicates sparse data
                    pass
                elif model_with_region == 0:
                    # This model has no region mask - needs propagation
                    needs_propagation = True

            if needs_propagation:
                print(f"  Region mask columns exist but not all models have values. Propagating...")
                # Find the source model (one with most valid region masks)
                source_model = max(model_coverage, key=lambda m: model_coverage[m][0])
                source_count = model_coverage[source_model][0]
                print(f"    Source model: {source_model} ({source_count} valid region masks)")

                # Extract region mask mapping from source model
                source_df = baseline_df[baseline_df['model'] == source_model]
                merge_keys = ['Therapeutic', 'chain', 'position']
                valid_keys = [k for k in merge_keys if k in source_df.columns]

                if valid_keys:
                    region_mapping = source_df[valid_keys + [region_mask_heavy_col, region_mask_light_col]].dropna(
                        subset=[region_mask_heavy_col]
                    ).drop_duplicates(subset=valid_keys)

                    print(f"    Region mapping entries: {len(region_mapping)}")

                    # For each model without region mask, propagate from mapping
                    for model in models:
                        if model == source_model:
                            continue
                        model_count, model_total = model_coverage[model]
                        if model_count == 0:
                            # Drop existing empty columns and merge
                            model_rows = baseline_df[baseline_df['model'] == model].copy()
                            model_rows = model_rows.drop(columns=[region_mask_heavy_col, region_mask_light_col])
                            model_rows = model_rows.merge(region_mapping, on=valid_keys, how='left')

                            # Update the main dataframe
                            baseline_df.loc[baseline_df['model'] == model, region_mask_heavy_col] = model_rows[region_mask_heavy_col].values
                            baseline_df.loc[baseline_df['model'] == model, region_mask_light_col] = model_rows[region_mask_light_col].values

                            new_count = baseline_df.loc[baseline_df['model'] == model, region_mask_heavy_col].notna().sum()
                            print(f"    {model}: propagated {new_count} region masks")

        # If baseline doesn't have region mask columns at all but evo_ab does, merge them
        if not has_region_mask_cols and evo_ab_df is not None:
            evo_ab_has_mask = (region_mask_heavy_col in evo_ab_df.columns and
                              region_mask_light_col in evo_ab_df.columns)

            if evo_ab_has_mask:
                print(f"  Merging region mask columns from PRISM to baseline models...")

                # Identify common merge keys
                potential_keys = ['Therapeutic', 'chain', 'position', 'original_position']
                merge_keys = [k for k in potential_keys if k in baseline_df.columns and k in evo_ab_df.columns]

                if merge_keys:
                    print(f"    Using merge keys: {merge_keys}")

                    # Extract region mask info from evo_ab (deduplicated)
                    region_info_cols = merge_keys + [region_mask_heavy_col, region_mask_light_col]
                    region_info = evo_ab_df[region_info_cols].drop_duplicates(subset=merge_keys)

                    # Merge region mask onto baseline
                    baseline_df = baseline_df.merge(
                        region_info,
                        on=merge_keys,
                        how='left'
                    )

                    # Report merge results
                    n_with_mask = baseline_df[region_mask_heavy_col].notna().sum()
                    print(f"    Merged region mask for {n_with_mask}/{len(baseline_df)} baseline rows")

                    # Report per-model merge success
                    if 'model' in baseline_df.columns:
                        print(f"    Per-model region mask coverage:")
                        for model in baseline_df['model'].unique():
                            model_mask = baseline_df['model'] == model
                            model_total = model_mask.sum()
                            model_with_region = (model_mask & baseline_df[region_mask_heavy_col].notna()).sum()
                            print(f"      {model}: {model_with_region}/{model_total} rows with region mask")
                else:
                    print(f"    WARNING: No common merge keys found. Region mask not merged.")

        # Report models found in baseline
        if 'model' in baseline_df.columns:
            baseline_models = baseline_df['model'].unique()
            print(f"  Baseline models found: {list(baseline_models)}")
        else:
            print(f"  WARNING: 'model' column not found in baseline CSV")

    # Add dataframes to list
    if baseline_df is not None:
        all_data.append(baseline_df)
    if evo_ab_df is not None:
        all_data.append(evo_ab_df)

    if not all_data:
        raise ValueError("No logit files provided or found!")

    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"  Total rows: {len(combined_df)}")

    # Report all models in combined data
    if 'model' in combined_df.columns:
        all_models = combined_df['model'].unique()
        print(f"  All models in combined data: {list(all_models)}")

    return combined_df


def get_model_logit_columns(
    df: pd.DataFrame,
    model_name: str,
    evo_ab_logit_type: str = 'lowercase'
) -> Tuple[List[str], str]:
    """
    Get the logit column names for a specific model.

    For PRISM (v17/v18 models):
        - Default: Uses LOWERCASE logits (A_lower, C_lower, ...) for mutation prediction
        - This is because PRISM predicts lowercase tokens for NGL/mutation positions
        - Uppercase logits are for germline (GL) position predictions
        - Can be overridden via evo_ab_logit_type parameter

    For Baseline models (ESM2, AbLang2, AntiBERTy, Sapiens):
        - Uses standard columns (A, C, D, E, ...)
        - These models don't distinguish between GL and NGL

    Args:
        df: DataFrame containing logit columns
        model_name: Name of the model
        evo_ab_logit_type: For PRISM only - 'lowercase' (default, NGL/mutation) or 'uppercase' (GL/germline)

    Returns:
        Tuple of (list of logit columns, logit type description)
    """
    # Check for PRISM style columns
    lower_cols = [f'{aa}_lower' for aa in AMINO_ACIDS]
    upper_cols = [f'{aa}_upper' for aa in AMINO_ACIDS]
    standard_cols = list(AMINO_ACIDS)

    # Determine if this is PRISM model
    is_evo_ab = model_name in ['PRISM', 'evo_ab', 'EvoAb', 'Evo_Ab']

    if is_evo_ab:
        # For PRISM: use the specified logit type
        if evo_ab_logit_type == 'lowercase':
            if all(col in df.columns for col in lower_cols):
                return lower_cols, 'lowercase (NGL/mutation)'
            elif all(col in df.columns for col in upper_cols):
                print(f"    WARNING: lowercase columns not found for {model_name}, falling back to uppercase")
                return upper_cols, 'uppercase (GL/germline) - fallback'
            else:
                return [], 'unknown'
        else:  # uppercase
            if all(col in df.columns for col in upper_cols):
                return upper_cols, 'uppercase (GL/germline)'
            elif all(col in df.columns for col in lower_cols):
                print(f"    WARNING: uppercase columns not found for {model_name}, falling back to lowercase")
                return lower_cols, 'lowercase (NGL/mutation) - fallback'
            else:
                return [], 'unknown'
    else:
        # For baseline models: use standard columns
        if all(aa in df.columns for aa in standard_cols):
            return standard_cols, 'standard'
        elif all(col in df.columns for col in lower_cols):
            # If PRISM format but not PRISM model, still try lowercase
            return lower_cols, 'lowercase'
        else:
            return [], 'unknown'


def analyze_topk_accuracy(
    df: pd.DataFrame,
    k_values: List[int] = [1, 3, 5, 10],
    n_bootstrap: int = 1000,
    evo_ab_logit_type: str = 'lowercase',
    use_region_aware: bool = False,
    use_40_vocab: bool = False,
    use_marginalized: bool = False,
    region_mask_heavy_col: str = 'region_mask_heavy',
    region_mask_light_col: str = 'region_mask_light',
    chain_col: str = 'chain',
    position_col: str = 'position'
) -> Dict[str, Dict[int, Tuple[float, float, float]]]:
    """
    Analyze top-k accuracy for all models in the DataFrame.

    Args:
        df: DataFrame with logit columns and mutation info
        k_values: List of K values for top-k accuracy calculation
        n_bootstrap: Number of bootstrap iterations for confidence intervals
        evo_ab_logit_type: Which logits to use for PRISM ('lowercase' or 'uppercase')
        use_region_aware: If True, use region-aware PRISM logit selection
                         (uppercase for FR, lowercase for CDR). Overrides evo_ab_logit_type.
        use_40_vocab: If True, use all 40 logits (upper + lower) for PRISM.
                     Correct if either uppercase or lowercase of true AA is in top-k.
        use_marginalized: If True, sum upper + lower logits for each AA (20-class).
        region_mask_heavy_col: Column for heavy chain region mask
        region_mask_light_col: Column for light chain region mask
        chain_col: Column indicating chain type ('heavy' or 'light')
        position_col: Column containing position index within chain

    Returns:
        Dict mapping model_name -> {k: (accuracy, ci_lower, ci_upper)}
    """
    results = {}

    # Check if 'model' column exists
    if 'model' not in df.columns:
        print("  WARNING: 'model' column not found. Assuming single PRISM model.")
        df = df.copy()
        df['model'] = 'PRISM'

    # Get unique models
    models = df['model'].unique()
    print(f"\n  Found models: {list(models)}")
    if use_marginalized:
        print(f"  PRISM logit mode: MARGINALIZED (sum of uppercase + lowercase)")
    elif use_40_vocab:
        print(f"  PRISM logit mode: 40-VOCAB (all uppercase + lowercase logits)")
    elif use_region_aware:
        print(f"  PRISM logit mode: REGION-AWARE (uppercase for FR, lowercase for CDR)")
    else:
        print(f"  PRISM logit type: {evo_ab_logit_type}")

    for model in models:
        model_df = df[df['model'] == model].copy()
        print(f"\n  Processing {model} ({len(model_df)} positions)...")

        # Determine if this is PRISM model
        is_evo_ab = model in ['PRISM', 'evo_ab', 'EvoAb', 'Evo_Ab']

        # For PRISM: use marginalized, 40_vocab, or region_aware mode
        model_use_marginalized = is_evo_ab and use_marginalized
        model_use_40_vocab = is_evo_ab and use_40_vocab and not use_marginalized
        model_use_region_aware = is_evo_ab and use_region_aware and not use_40_vocab and not use_marginalized

        if model_use_marginalized:
            print(f"    Using MARGINALIZED mode (sum upper + lower logits)")
            aa_cols = []  # Not needed for marginalized mode
        elif model_use_40_vocab:
            print(f"    Using 40-VOCAB mode (all upper + lower logits)")
            aa_cols = []  # Not needed for 40-vocab mode
        elif model_use_region_aware:
            print(f"    Using REGION-AWARE mode (FR=uppercase, CDR=lowercase)")
            aa_cols = []  # Not needed for region-aware mode
        else:
            # Get appropriate logit columns (pass evo_ab_logit_type for PRISM)
            aa_cols, col_type = get_model_logit_columns(model_df, model, evo_ab_logit_type)

            if not aa_cols:
                print(f"    WARNING: Could not find logit columns for {model}")
                continue

            print(f"    Using {col_type} columns")

        model_results = {}
        for k in k_values:
            acc, ci_low, ci_high = bootstrap_topk_accuracy(
                model_df, k, aa_cols,
                n_bootstrap=n_bootstrap,
                use_region_aware=model_use_region_aware,
                use_40_vocab=model_use_40_vocab,
                use_marginalized=model_use_marginalized,
                region_mask_heavy_col=region_mask_heavy_col,
                region_mask_light_col=region_mask_light_col,
                chain_col=chain_col,
                position_col=position_col
            )
            model_results[k] = (acc, ci_low, ci_high)
            print(f"    Top-{k}: {acc:.4f} [{ci_low:.4f}, {ci_high:.4f}]")

        # Normalize model name for display
        display_name = model
        if model == 'ESM2-35M' or model == 'esm2_35m':
            display_name = 'ESM2_35M'
        elif model == 'ESM2-650M' or model == 'esm2_650m':
            display_name = 'ESM2_650M'

        results[display_name] = model_results

    return results


def calculate_per_antibody_accuracy(
    df: pd.DataFrame,
    k: int,
    evo_ab_logit_type: str = 'lowercase',
    use_region_aware: bool = False,
    use_40_vocab: bool = False,
    use_marginalized: bool = False,
    region_mask_heavy_col: str = 'region_mask_heavy',
    region_mask_light_col: str = 'region_mask_light',
    chain_col: str = 'chain',
    position_col: str = 'position'
) -> Dict[str, np.ndarray]:
    """
    Calculate top-k accuracy for each antibody, returning per-antibody distributions.

    This is more clinically relevant than per-position accuracy because:
    - Each antibody is treated as one unit (not weighted by mutation count)
    - Shows variance across the therapeutic antibody population
    - Enables paired statistical testing between models

    Args:
        df: DataFrame with logit columns and 'Therapeutic' column for grouping
        k: K value for top-k accuracy
        evo_ab_logit_type: Which logits to use for PRISM ('lowercase' or 'uppercase')
        use_region_aware: If True, use region-aware PRISM logit selection
                         (uppercase for FR, lowercase for CDR). Overrides evo_ab_logit_type.
        use_40_vocab: If True, use all 40 logits (upper + lower) for PRISM.
                     Correct if either uppercase or lowercase of true AA is in top-k.
        use_marginalized: If True, sum upper + lower logits for each AA (20-class).
        region_mask_heavy_col: Column for heavy chain region mask
        region_mask_light_col: Column for light chain region mask
        chain_col: Column indicating chain type ('heavy' or 'light')
        position_col: Column containing position index within chain

    Returns:
        Dict mapping model_name -> np.array of per-antibody accuracies
    """
    if 'Therapeutic' not in df.columns:
        raise ValueError("DataFrame must have 'Therapeutic' column for per-antibody analysis")

    # Check if 'model' column exists
    if 'model' not in df.columns:
        print("  WARNING: 'model' column not found. Assuming single PRISM model.")
        df = df.copy()
        df['model'] = 'PRISM'

    models = df['model'].unique()
    antibodies = df['Therapeutic'].unique()

    print(f"\n  Per-antibody analysis for Top-{k}:")
    print(f"    Models: {list(models)}")
    print(f"    Antibodies: {len(antibodies)}")
    if use_marginalized:
        print(f"    PRISM logit mode: MARGINALIZED (sum upper + lower logits)")
    elif use_40_vocab:
        print(f"    PRISM logit mode: 40-VOCAB (all upper + lower logits)")
    elif use_region_aware:
        print(f"    PRISM logit mode: REGION-AWARE (FR=uppercase, CDR=lowercase)")

    results = {}

    for model in models:
        model_df = df[df['model'] == model].copy()

        # Determine if this is PRISM model
        is_evo_ab = model in ['PRISM', 'evo_ab', 'EvoAb', 'Evo_Ab']

        # For PRISM: use marginalized, 40_vocab, or region_aware mode
        model_use_marginalized = is_evo_ab and use_marginalized
        model_use_40_vocab = is_evo_ab and use_40_vocab and not use_marginalized
        model_use_region_aware = is_evo_ab and use_region_aware and not use_40_vocab and not use_marginalized

        if model_use_marginalized:
            aa_cols = []  # Not needed for marginalized mode
        elif model_use_40_vocab:
            aa_cols = []  # Not needed for 40-vocab mode
        elif model_use_region_aware:
            aa_cols = []  # Not needed for region-aware mode
        else:
            # Get appropriate logit columns
            aa_cols, col_type = get_model_logit_columns(model_df, model, evo_ab_logit_type)

            if not aa_cols:
                print(f"    WARNING: Could not find logit columns for {model}")
                continue

        # Calculate accuracy for each antibody
        antibody_accuracies = []

        for ab in antibodies:
            ab_df = model_df[model_df['Therapeutic'] == ab]

            if len(ab_df) == 0:
                continue

            # Calculate top-k accuracy for this antibody's mutations
            acc = calculate_topk_accuracy(
                ab_df, k, aa_cols,
                use_region_aware=model_use_region_aware,
                use_40_vocab=model_use_40_vocab,
                use_marginalized=model_use_marginalized,
                region_mask_heavy_col=region_mask_heavy_col,
                region_mask_light_col=region_mask_light_col,
                chain_col=chain_col,
                position_col=position_col
            )

            if not np.isnan(acc):
                antibody_accuracies.append(acc)

        # Normalize model name for display
        display_name = model
        if model == 'ESM2-35M' or model == 'esm2_35m':
            display_name = 'ESM2_35M'
        elif model == 'ESM2-650M' or model == 'esm2_650m':
            display_name = 'ESM2_650M'

        results[display_name] = np.array(antibody_accuracies)
        print(f"    {display_name}: {len(antibody_accuracies)} antibodies, "
              f"mean={np.mean(antibody_accuracies):.4f}, std={np.std(antibody_accuracies):.4f}")

    return results


def create_topk_line_plot(
    results: Dict[str, Dict[int, Tuple[float, float, float]]],
    k_values: List[int],
    output_path: str,
    title: str = None,  # Deprecated - title removed for cleaner look
    dpi: int = 300
):
    """
    Create a line plot showing top-k accuracy for each model.
    Note: Title is no longer displayed for cleaner publication-ready figures.
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    # Fixed y-axis limit for accuracy (0 to 1)
    Y_MAX = 1.0

    # Sort models by MODEL_ORDER
    sorted_models = []
    for model in MODEL_ORDER:
        if model in results:
            sorted_models.append(model)
    # Add any remaining models
    for model in results:
        if model not in sorted_models:
            sorted_models.append(model)

    # Plot each model
    for model in sorted_models:
        if model not in results:
            continue

        model_results = results[model]
        accuracies = [model_results[k][0] for k in k_values]
        ci_lows = [model_results[k][1] for k in k_values]
        ci_highs = [model_results[k][2] for k in k_values]

        # Calculate error bars
        yerr_low = [max(0, acc - ci_low) for acc, ci_low in zip(accuracies, ci_lows)]
        yerr_high = [max(0, ci_high - acc) for acc, ci_high in zip(accuracies, ci_highs)]

        color = MODEL_COLORS.get(model, '#999999')

        # Plot line with markers
        line = ax.errorbar(
            k_values, accuracies,
            yerr=[yerr_low, yerr_high],
            label=model,
            color=color,
            linewidth=3 if model == 'PRISM' else 2,
            marker='o',
            markersize=10 if model == 'PRISM' else 8,
            capsize=5,
            capthick=2
        )

        # Highlight PRISM
        if model == 'PRISM':
            line[0].set_zorder(10)

    # Customize axes
    ax.set_xlabel('K (Top-K)', **FONT_CONFIG['axis_label'])
    ax.set_ylabel('Accuracy', **FONT_CONFIG['axis_label'])
    # No title - cleaner for publication

    # Set x-axis ticks
    ax.set_xticks(k_values)
    ax.set_xticklabels([str(k) for k in k_values], **FONT_CONFIG['tick_label'])

    # Set y-axis limits (fixed 0 to 1 for accuracy)
    ax.set_ylim(0, Y_MAX)
    ax.tick_params(axis='y', labelsize=FONT_CONFIG['tick_label']['fontsize'])

    # Add grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)

    # Add legend
    ax.legend(loc='lower right', fontsize=FONT_CONFIG['legend']['fontsize'], frameon=True, fancybox=True)

    # Thicken spines
    for spine in ax.spines.values():
        spine.set_linewidth(2)

    plt.tight_layout()
    svg_path = save_figure_with_svg(fig, output_path, dpi=dpi)
    plt.close(fig)

    print(f"\n✓ Line plot saved to: {output_path}")
    print(f"✓ Line plot (SVG) saved to: {svg_path}")


def create_topk_bar_plot(
    results: Dict[str, Dict[int, Tuple[float, float, float]]],
    k_values: List[int],
    output_path: str,
    title: str = None,  # Deprecated - title removed for cleaner look
    dpi: int = 300
):
    """
    Create a grouped bar plot showing top-k accuracy for each model.
    Note: Title is no longer displayed for cleaner publication-ready figures.
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    # Fixed y-axis limit for accuracy (0 to 1)
    Y_MAX = 1.0

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
    bar_width = 0.8 / n_k
    x = np.arange(n_models)

    # Plot bars for each k value
    for i, k in enumerate(k_values):
        accuracies = []
        ci_lows = []
        ci_highs = []
        colors = []

        for model in sorted_models:
            acc, ci_low, ci_high = results[model][k]
            accuracies.append(acc if not np.isnan(acc) else 0)
            ci_lows.append(ci_low if not np.isnan(ci_low) else acc)
            ci_highs.append(ci_high if not np.isnan(ci_high) else acc)
            colors.append(MODEL_COLORS.get(model, '#999999'))

        # Calculate error bars
        yerr_low = [max(0, acc - ci_low) for acc, ci_low in zip(accuracies, ci_lows)]
        yerr_high = [max(0, ci_high - acc) for acc, ci_high in zip(accuracies, ci_highs)]

        # Offset for grouped bars
        offset = (i - n_k/2 + 0.5) * bar_width

        bars = ax.bar(
            x + offset,
            accuracies,
            bar_width,
            label=f'Top-{k}',
            color=colors,
            edgecolor='black',
            linewidth=1.5,
            yerr=[yerr_low, yerr_high],
            capsize=3,
            alpha=0.7 + 0.1 * (i / n_k)  # Slight alpha variation by k
        )

    # Customize axes
    ax.set_xlabel('Model', **FONT_CONFIG['axis_label'])
    ax.set_ylabel('Accuracy', **FONT_CONFIG['axis_label'])
    # No title - cleaner for publication

    ax.set_xticks(x)
    ax.set_xticklabels(sorted_models, fontsize=25, fontweight='bold', rotation=45, ha='right')
    ax.set_ylim(0, Y_MAX)  # Fixed y-axis for accuracy
    ax.tick_params(axis='y', labelsize=FONT_CONFIG['tick_label']['fontsize'])

    # Add grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)

    # Add legend for k values
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='gray', edgecolor='black', alpha=0.7 + 0.1 * (i / n_k),
                            label=f'Top-{k}') for i, k in enumerate(k_values)]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=FONT_CONFIG['legend']['fontsize'])

    # Thicken spines
    for spine in ax.spines.values():
        spine.set_linewidth(2)

    plt.tight_layout()
    svg_path = save_figure_with_svg(fig, output_path, dpi=dpi)
    plt.close(fig)

    print(f"✓ Bar plot saved to: {output_path}")
    print(f"✓ Bar plot (SVG) saved to: {svg_path}")


def create_combined_plot(
    results: Dict[str, Dict[int, Tuple[float, float, float]]],
    k_values: List[int],
    output_path: str,
    title: str = None,  # Deprecated - main title removed
    dpi: int = 300
):
    """
    Create a combined figure with both line plot and bar plot for specific k values.
    Note: Main title is no longer displayed (only subtitle headers on each panel).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Fixed y-axis limit for accuracy (0 to 1)
    Y_MAX = 1.0

    # Sort models
    sorted_models = []
    for model in MODEL_ORDER:
        if model in results:
            sorted_models.append(model)
    for model in results:
        if model not in sorted_models:
            sorted_models.append(model)

    # Left panel: Line plot
    for model in sorted_models:
        model_results = results[model]
        accuracies = [model_results[k][0] for k in k_values]
        ci_lows = [model_results[k][1] for k in k_values]
        ci_highs = [model_results[k][2] for k in k_values]

        yerr_low = [max(0, acc - ci_low) if not np.isnan(ci_low) else 0
                    for acc, ci_low in zip(accuracies, ci_lows)]
        yerr_high = [max(0, ci_high - acc) if not np.isnan(ci_high) else 0
                     for acc, ci_high in zip(accuracies, ci_highs)]

        color = MODEL_COLORS.get(model, '#999999')

        ax1.errorbar(
            k_values, accuracies,
            yerr=[yerr_low, yerr_high],
            label=model,
            color=color,
            linewidth=3 if model == 'PRISM' else 2,
            marker='o',
            markersize=10 if model == 'PRISM' else 8,
            capsize=5,
            capthick=2,
            zorder=10 if model == 'PRISM' else 1
        )

    ax1.set_xlabel('K (Top-K)', **FONT_CONFIG['axis_label'])
    ax1.set_ylabel('Accuracy', **FONT_CONFIG['axis_label'])
    ax1.set_title('Top-K Accuracy by Model', **FONT_CONFIG['title'])
    ax1.set_xticks(k_values)
    ax1.set_xticklabels([str(k) for k in k_values], **FONT_CONFIG['tick_label'])
    ax1.set_ylim(0, Y_MAX)  # Fixed y-axis for accuracy
    ax1.tick_params(axis='y', labelsize=FONT_CONFIG['tick_label']['fontsize'])
    ax1.legend(loc='lower right', fontsize=FONT_CONFIG['legend']['fontsize'], frameon=True)
    ax1.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax1.set_axisbelow(True)
    for spine in ax1.spines.values():
        spine.set_linewidth(2)

    # Right panel: Bar plot for Top-1 and Top-5
    selected_k = [1, 5]
    n_models = len(sorted_models)
    bar_width = 0.35
    x = np.arange(n_models)

    for i, k in enumerate(selected_k):
        accuracies = []
        ci_lows = []
        ci_highs = []
        colors = []

        for model in sorted_models:
            acc, ci_low, ci_high = results[model][k]
            accuracies.append(acc if not np.isnan(acc) else 0)
            ci_lows.append(ci_low if not np.isnan(ci_low) else acc)
            ci_highs.append(ci_high if not np.isnan(ci_high) else acc)
            colors.append(MODEL_COLORS.get(model, '#999999'))

        yerr_low = [max(0, acc - ci_low) for acc, ci_low in zip(accuracies, ci_lows)]
        yerr_high = [max(0, ci_high - acc) for acc, ci_high in zip(accuracies, ci_highs)]

        offset = (i - 0.5) * bar_width

        bars = ax2.bar(
            x + offset,
            accuracies,
            bar_width,
            label=f'Top-{k}',
            color=colors,
            edgecolor='black',
            linewidth=2,
            yerr=[yerr_low, yerr_high],
            capsize=4,
            alpha=0.9 if k == 1 else 0.7
        )

        # Highlight PRISM bars
        if sorted_models[0] == 'PRISM':
            bars[0].set_edgecolor('#000000')
            bars[0].set_linewidth(3)

    ax2.set_xlabel('Model', **FONT_CONFIG['axis_label'])
    ax2.set_ylabel('Accuracy', **FONT_CONFIG['axis_label'])
    ax2.set_title('Top-1 vs Top-5 Accuracy', **FONT_CONFIG['title'])
    ax2.set_xticks(x)
    ax2.set_xticklabels(sorted_models, fontsize=25, fontweight='bold', rotation=45, ha='right')
    ax2.set_ylim(0, Y_MAX)  # Fixed y-axis for accuracy
    ax2.tick_params(axis='y', labelsize=FONT_CONFIG['tick_label']['fontsize'])

    # Create legend with hatching to differentiate Top-1 vs Top-5
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='gray', edgecolor='black', alpha=0.9, label='Top-1'),
        Patch(facecolor='gray', edgecolor='black', alpha=0.7, label='Top-5')
    ]
    ax2.legend(handles=legend_elements, loc='lower right', fontsize=FONT_CONFIG['legend']['fontsize'])

    ax2.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax2.set_axisbelow(True)
    for spine in ax2.spines.values():
        spine.set_linewidth(2)

    # No main title - only keep subtitles on each panel
    plt.tight_layout()
    svg_path = save_figure_with_svg(fig, output_path, dpi=dpi)
    plt.close(fig)

    print(f"✓ Combined plot saved to: {output_path}")
    print(f"✓ Combined plot (SVG) saved to: {svg_path}")


def print_results_table(
    results: Dict[str, Dict[int, Tuple[float, float, float]]],
    k_values: List[int]
):
    """Print formatted results table."""
    print(f"\n{'='*90}")
    print("TOP-K ACCURACY RESULTS")
    print(f"{'='*90}")

    # Header
    header = f"{'Model':<15}"
    for k in k_values:
        header += f" {'Top-' + str(k):>20}"
    print(header)
    print("-" * 90)

    # Sort models
    sorted_models = []
    for model in MODEL_ORDER:
        if model in results:
            sorted_models.append(model)
    for model in results:
        if model not in sorted_models:
            sorted_models.append(model)

    for model in sorted_models:
        row = f"{model:<15}"
        for k in k_values:
            acc, ci_low, ci_high = results[model][k]
            if np.isnan(acc):
                row += f" {'N/A':>20}"
            else:
                row += f" {acc:.4f} [{ci_low:.4f}, {ci_high:.4f}]"
        print(row)

    print("=" * 90)


def print_region_results_table(
    region_results: Dict[str, Dict[str, Dict[int, Tuple[float, float, float]]]],
    k_values: List[int]
):
    """Print formatted results table for region-specific analysis."""
    print(f"\n{'='*100}")
    print("REGION-SPECIFIC TOP-K ACCURACY RESULTS")
    print(f"{'='*100}")

    for region_type in ['FR', 'CDR']:
        if region_type not in region_results:
            continue

        results = region_results[region_type]

        print(f"\n{'-'*100}")
        print(f"  {region_type} REGIONS (Framework)" if region_type == 'FR' else f"  {region_type} REGIONS (CDR1, CDR2, CDR3)")
        print(f"{'-'*100}")

        # Header
        header = f"  {'Model':<15}"
        for k in k_values:
            header += f" {'Top-' + str(k):>20}"
        print(header)
        print("  " + "-" * 85)

        # Sort models
        sorted_models = []
        for model in MODEL_ORDER:
            if model in results:
                sorted_models.append(model)
        for model in results:
            if model not in sorted_models:
                sorted_models.append(model)

        for model in sorted_models:
            row = f"  {model:<15}"
            for k in k_values:
                if k in results[model]:
                    acc, ci_low, ci_high = results[model][k]
                    if np.isnan(acc):
                        row += f" {'N/A':>20}"
                    else:
                        row += f" {acc:.4f} [{ci_low:.4f}, {ci_high:.4f}]"
                else:
                    row += f" {'N/A':>20}"
            print(row)

    print("=" * 100)


def compute_custom_boxplot_stats(
    data: np.ndarray,
    box_percentiles: Tuple[float, float] = (20, 80),
    whisker_percentiles: Tuple[float, float] = (10, 90)
) -> dict:
    """
    Compute custom boxplot statistics with user-defined percentiles.

    By default, matplotlib boxplots use:
    - Box: Q1 (25th) to Q3 (75th)
    - Whiskers: 1.5 * IQR or data extent

    This function allows custom percentiles:
    - Box: 20th to 80th percentile (default)
    - Whiskers: 10th to 90th percentile (default)

    Args:
        data: Array of values
        box_percentiles: (lower, upper) percentiles for the box edges
        whisker_percentiles: (lower, upper) percentiles for whisker ends

    Returns:
        Dict compatible with matplotlib's bxp() function
    """
    clean_data = data[~np.isnan(data) & ~np.isinf(data)]

    if len(clean_data) == 0:
        return None

    # Compute percentiles
    whisker_lo, box_lo, median, box_hi, whisker_hi = np.percentile(
        clean_data,
        [whisker_percentiles[0], box_percentiles[0], 50, box_percentiles[1], whisker_percentiles[1]]
    )

    # Find outliers (beyond whisker percentiles)
    fliers = clean_data[(clean_data < whisker_lo) | (clean_data > whisker_hi)]

    return {
        'med': median,
        'q1': box_lo,      # Lower edge of box (20th percentile)
        'q3': box_hi,      # Upper edge of box (80th percentile)
        'whislo': whisker_lo,  # Lower whisker (10th percentile)
        'whishi': whisker_hi,  # Upper whisker (90th percentile)
        'fliers': fliers,
        'mean': np.mean(clean_data),  # Optional: mean marker
    }


def create_per_antibody_boxplot(
    data_dict: Dict[str, np.ndarray],
    k: int,
    output_path: str,
    title: str = None,
    dpi: int = 300
):
    """
    Create a box plot showing per-antibody accuracy distribution with Wilcoxon significance.

    Uses PRISM as reference and compares to all other models using Wilcoxon Signed-Rank test.

    Args:
        data_dict: Dict mapping model_name -> np.array of per-antibody accuracies
        k: K value (for labeling)
        output_path: Path to save the figure
        title: Optional custom title
        dpi: Figure DPI
    """
    fig, ax = plt.subplots(figsize=(12, 9))

    # Sort models by MODEL_ORDER, with PRISM first
    sorted_models = []
    for model in MODEL_ORDER:
        if model in data_dict:
            sorted_models.append(model)
    for model in data_dict:
        if model not in sorted_models:
            sorted_models.append(model)

    if len(sorted_models) == 0:
        print("  ERROR: No models with valid data for boxplot")
        plt.close(fig)
        return

    # PRISM should be first (index 0)
    evo_ab_idx = None
    for i, model in enumerate(sorted_models):
        if model == 'PRISM':
            evo_ab_idx = i
            break

    n_models = len(sorted_models)
    x_pos = np.arange(n_models)

    # Prepare data and compute custom boxplot statistics
    # Box: 20th to 80th percentile, Whiskers: 10th to 90th percentile
    plot_data = []
    box_stats = []
    colors = []
    for model in sorted_models:
        values = data_dict[model]
        clean_values = values[~np.isnan(values) & ~np.isinf(values)]
        plot_data.append(clean_values)
        colors.append(MODEL_COLORS.get(model, '#999999'))

        # Compute custom percentile stats for bxp()
        stats = compute_custom_boxplot_stats(
            clean_values,
            box_percentiles=(20, 80),      # Box edges
            whisker_percentiles=(10, 90)   # Whisker ends
        )
        if stats is not None:
            box_stats.append(stats)
        else:
            # Fallback for empty data
            box_stats.append({'med': 0, 'q1': 0, 'q3': 0, 'whislo': 0, 'whishi': 0, 'fliers': []})

    # Create box plot with custom percentiles using bxp()
    bp = ax.bxp(box_stats, positions=x_pos, widths=0.6, patch_artist=True,
                showfliers=True, flierprops={'markersize': 3, 'alpha': 0.5})

    # Color the boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor('black')
        patch.set_linewidth(2)

    # Style medians
    for median in bp['medians']:
        median.set_color('black')
        median.set_linewidth(2)

    # Highlight PRISM box
    if evo_ab_idx is not None:
        bp['boxes'][evo_ab_idx].set_edgecolor('#000000')
        bp['boxes'][evo_ab_idx].set_linewidth(3)

    # Default y_max (will be extended if brackets need more room)
    y_max = 1.05

    # Calculate Wilcoxon p-values and draw brackets (PRISM vs others)
    if evo_ab_idx is not None:
        evo_ab_data = plot_data[evo_ab_idx]

        # Get max data value for bracket positioning
        all_values = np.concatenate([d for d in plot_data if len(d) > 0])
        clean_values = all_values[~np.isnan(all_values) & ~np.isinf(all_values)]
        # Use 95th percentile to avoid outliers affecting layout
        data_max = np.percentile(clean_values, 95)

        # Collect comparison indices (all models except PRISM)
        comparison_indices = [i for i in range(len(sorted_models)) if i != evo_ab_idx]

        # Calculate p-values for all comparisons
        p_values = {}
        for i in comparison_indices:
            other_data = plot_data[i]
            min_len = min(len(evo_ab_data), len(other_data))
            if min_len > 10:
                try:
                    _, p_val = stats.wilcoxon(evo_ab_data[:min_len], other_data[:min_len])
                except:
                    p_val = 1.0
            else:
                p_val = 1.0
            p_values[i] = p_val

        # Calculate bracket positions with FIXED intervals for consistent appearance
        bracket_positions, required_y_max = calculate_bracket_positions(
            ref_idx=evo_ab_idx,
            comparison_indices=comparison_indices,
            data_max=data_max,
            bracket_interval=0.10,  # Fixed interval between brackets
            bracket_start_offset=0.05  # Gap above data
        )

        # Update y_max to fit all brackets
        y_max = max(y_max, required_y_max)

        # Draw brackets with consistent size
        for comp_idx, y_pos in bracket_positions:
            p_val = p_values[comp_idx]
            draw_significance_bracket(
                ax, evo_ab_idx, comp_idx, y_pos, p_val,
                height=0.025,      # Consistent bracket height
                tip_length=0.015,  # Consistent tip length
                fontsize=FONT_CONFIG['text']['fontsize']
            )

    # Set y-axis: extends to y_max for brackets, but only shows ticks up to 1.0
    set_accuracy_yaxis(ax, y_max, show_ticks_up_to=1.0)

    # Customize axes
    ax.set_xticks(x_pos)
    ax.set_xticklabels(sorted_models, fontsize=25, fontweight='bold', rotation=45, ha='right')
    ax.set_ylabel(f"Top-{k} Accuracy (per antibody)", **FONT_CONFIG['axis_label'])

    # Add title only if explicitly provided
    if title is not None:
        ax.set_title(title, **FONT_CONFIG['title'], pad=15)

    # Add grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)

    # Thicken spines
    for spine in ax.spines.values():
        spine.set_linewidth(2)

    plt.tight_layout()
    svg_path = save_figure_with_svg(fig, output_path, dpi=dpi)
    plt.close(fig)

    print(f"  ✓ Per-antibody boxplot saved to: {output_path}")
    print(f"  ✓ Per-antibody boxplot (SVG) saved to: {svg_path}")


def calculate_per_antibody_accuracy_multi_k(
    df: pd.DataFrame,
    k_values: List[int],
    evo_ab_logit_type: str = 'lowercase',
    use_region_aware: bool = False,
    use_40_vocab: bool = False,
    use_marginalized: bool = False,
    region_mask_heavy_col: str = 'region_mask_heavy',
    region_mask_light_col: str = 'region_mask_light',
    chain_col: str = 'chain',
    position_col: str = 'position'
) -> Dict[str, Dict[int, np.ndarray]]:
    """
    Calculate per-antibody top-k accuracy for multiple k values.

    Returns a nested dict: model_name -> {k: np.array of per-antibody accuracies}
    This is useful for line plots where we need accuracy distributions across k values.

    Args:
        df: DataFrame with logit columns and 'Therapeutic' column for grouping
        k_values: List of K values for top-k accuracy
        evo_ab_logit_type: Which logits to use for PRISM
        use_region_aware: If True, use region-aware PRISM logit selection
        use_40_vocab: If True, use all 40 logits for PRISM
        use_marginalized: If True, sum upper + lower logits for each AA
        region_mask_heavy_col: Column for heavy chain region mask
        region_mask_light_col: Column for light chain region mask
        chain_col: Column indicating chain type
        position_col: Column containing position index

    Returns:
        Dict mapping model_name -> {k: np.array of per-antibody accuracies}
    """
    results = {}

    for k in k_values:
        k_results = calculate_per_antibody_accuracy(
            df, k,
            evo_ab_logit_type=evo_ab_logit_type,
            use_region_aware=use_region_aware,
            use_40_vocab=use_40_vocab,
            use_marginalized=use_marginalized,
            region_mask_heavy_col=region_mask_heavy_col,
            region_mask_light_col=region_mask_light_col,
            chain_col=chain_col,
            position_col=position_col
        )

        for model, acc_array in k_results.items():
            if model not in results:
                results[model] = {}
            results[model][k] = acc_array

    return results


def create_per_antibody_line_plot(
    per_antibody_data: Dict[str, Dict[int, np.ndarray]],
    k_values: List[int],
    output_path: str,
    title: str = None,
    error_type: str = 'std',
    dpi: int = 300,
    show_prism_plus_5: bool = False
):
    """
    Create a line plot showing per-antibody top-k accuracy with distribution-based error bars.

    Unlike bootstrap CI (which measures estimation uncertainty), these error bars show
    the actual variance across antibodies - answering "how much do antibodies differ?"

    Args:
        per_antibody_data: Dict mapping model_name -> {k: np.array of per-antibody accuracies}
        k_values: List of K values to plot on x-axis
        output_path: Path to save the figure
        title: Optional title for the plot
        error_type: Type of error bars: 'std' (mean ± std), 'percentile' (median with 25-75th)
        dpi: Figure DPI
        show_prism_plus_5: If True, add a dashed line showing PRISM + 5% accuracy
    """
    # Wider figure to accommodate legend on the right
    fig, ax = plt.subplots(figsize=(12, 7))

    Y_MAX = 1.0

    # Sort models by MODEL_ORDER
    sorted_models = []
    for model in MODEL_ORDER:
        if model in per_antibody_data:
            sorted_models.append(model)
    for model in per_antibody_data:
        if model not in sorted_models:
            sorted_models.append(model)

    # Plot each model
    for model in sorted_models:
        if model not in per_antibody_data:
            continue

        model_data = per_antibody_data[model]

        # Calculate mean/median and error bars for each k
        means = []
        err_lows = []
        err_highs = []
        valid_k = []

        for k in k_values:
            if k not in model_data:
                continue

            values = model_data[k]
            clean_values = values[~np.isnan(values) & ~np.isinf(values)]

            if len(clean_values) == 0:
                continue

            valid_k.append(k)

            if error_type == 'percentile':
                # Median with 25th-75th percentile range
                median = np.median(clean_values)
                p25 = np.percentile(clean_values, 25)
                p75 = np.percentile(clean_values, 75)
                means.append(median)
                err_lows.append(median - p25)
                err_highs.append(p75 - median)
            else:  # 'std'
                # Mean ± standard deviation
                mean = np.mean(clean_values)
                std = np.std(clean_values)
                means.append(mean)
                err_lows.append(std)
                err_highs.append(std)

        if not valid_k:
            continue

        color = MODEL_COLORS.get(model, '#999999')

        # Plot line with error bars
        ax.errorbar(
            valid_k, means,
            yerr=[err_lows, err_highs],
            label=model,
            color=color,
            linewidth=3 if model == 'PRISM' else 2,
            marker='o',
            markersize=10 if model == 'PRISM' else 8,
            capsize=5,
            capthick=2,
            zorder=10 if model == 'PRISM' else 1
        )

    # Customize axes
    ax.set_xlabel('K (Top-K)', **FONT_CONFIG['axis_label'])
    ax.set_ylabel('Accuracy (per antibody)', **FONT_CONFIG['axis_label'])
    if title:
        ax.set_title(title, **FONT_CONFIG['title'])

    ax.set_xticks(k_values)
    ax.set_xticklabels([str(k) for k in k_values], **FONT_CONFIG['tick_label'])
    ax.set_ylim(0, Y_MAX)
    ax.tick_params(axis='y', labelsize=FONT_CONFIG['tick_label']['fontsize'])

    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)

    # Legend outside the plot on the right
    ax.legend(
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        fontsize=FONT_CONFIG['legend']['fontsize'],
        frameon=True,
        fancybox=True
    )

    for spine in ax.spines.values():
        spine.set_linewidth(2)

    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Leave space for legend on right
    svg_path = save_figure_with_svg(fig, output_path, dpi=dpi)
    plt.close(fig)

    print(f"  ✓ Per-antibody line plot saved to: {output_path}")
    print(f"  ✓ Per-antibody line plot (SVG) saved to: {svg_path}")


def create_per_antibody_region_comparison_plot(
    cdr_data: Dict[str, Dict[int, np.ndarray]],
    fr_data: Dict[str, Dict[int, np.ndarray]],
    k_values: List[int],
    output_path: str,
    error_type: str = 'std',
    dpi: int = 300
):
    """
    Create a two-panel figure comparing per-antibody top-k accuracy for CDR and FR regions.

    Left panel: CDR regions (CDR1, CDR2, CDR3) - hypervariable regions
    Right panel: FR regions (FR1, FR2, FR3, FR4) - conserved framework regions

    Error bars show the distribution of per-antibody accuracies, visualizing
    the variance across therapeutic antibodies for each model.

    Legend is placed outside on the rightmost side, shared between both panels.

    Args:
        cdr_data: Dict mapping model_name -> {k: np.array of per-antibody accuracies} for CDR
        fr_data: Dict mapping model_name -> {k: np.array of per-antibody accuracies} for FR
        k_values: List of K values to plot on x-axis
        output_path: Path to save the figure
        error_type: Type of error bars: 'std' (mean ± std), 'percentile' (median with 25-75th)
        dpi: Figure DPI
    """
    # Wider figure to accommodate shared legend on the right
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    Y_MAX = 1.0

    # Sort models by MODEL_ORDER
    all_models = set(cdr_data.keys()) | set(fr_data.keys())
    sorted_models = []
    for model in MODEL_ORDER:
        if model in all_models:
            sorted_models.append(model)
    for model in all_models:
        if model not in sorted_models:
            sorted_models.append(model)

    # Store handles and labels for shared legend
    legend_handles = []
    legend_labels = []

    # Plot function for each panel (no individual legends)
    def plot_panel(ax, data, subtitle, collect_legend=False):
        nonlocal legend_handles, legend_labels

        for model in sorted_models:
            if model not in data:
                continue

            model_data = data[model]

            # Calculate mean/median and error bars for each k
            means = []
            err_lows = []
            err_highs = []
            valid_k = []

            for k in k_values:
                if k not in model_data:
                    continue

                values = model_data[k]
                clean_values = values[~np.isnan(values) & ~np.isinf(values)]

                if len(clean_values) == 0:
                    continue

                valid_k.append(k)

                if error_type == 'percentile':
                    median = np.median(clean_values)
                    p25 = np.percentile(clean_values, 25)
                    p75 = np.percentile(clean_values, 75)
                    means.append(median)
                    err_lows.append(median - p25)
                    err_highs.append(p75 - median)
                else:  # 'std'
                    mean = np.mean(clean_values)
                    std = np.std(clean_values)
                    means.append(mean)
                    err_lows.append(std)
                    err_highs.append(std)

            if not valid_k:
                continue

            color = MODEL_COLORS.get(model, '#999999')

            line = ax.errorbar(
                valid_k, means,
                yerr=[err_lows, err_highs],
                label=model,
                color=color,
                linewidth=3 if model == 'PRISM' else 2,
                marker='o',
                markersize=10 if model == 'PRISM' else 8,
                capsize=5,
                capthick=2,
                zorder=10 if model == 'PRISM' else 1
            )

            # Collect legend handles from first panel only
            if collect_legend:
                legend_handles.append(line)
                legend_labels.append(model)

        ax.set_xlabel('K (Top-K)', **FONT_CONFIG['axis_label'])
        ax.set_ylabel('Accuracy (per antibody)', **FONT_CONFIG['axis_label'])
        ax.set_title(subtitle, **FONT_CONFIG['title'])

        ax.set_xticks(k_values)
        ax.set_xticklabels([str(k) for k in k_values], **FONT_CONFIG['tick_label'])
        ax.set_ylim(0, Y_MAX)
        ax.tick_params(axis='y', labelsize=FONT_CONFIG['tick_label']['fontsize'])

        ax.yaxis.grid(True, linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)

        for spine in ax.spines.values():
            spine.set_linewidth(2)

    # Plot CDR (left panel) - collect legend handles
    plot_panel(ax1, cdr_data, "CDR Regions", collect_legend=True)

    # Plot FR (right panel)
    plot_panel(ax2, fr_data, "FR Regions", collect_legend=False)

    # Shared legend outside the figure on the rightmost side
    fig.legend(
        legend_handles, legend_labels,
        loc='center left',
        bbox_to_anchor=(0.92, 0.5),
        fontsize=FONT_CONFIG['legend']['fontsize'],
        frameon=True,
        fancybox=True
    )

    plt.tight_layout(rect=[0, 0, 0.90, 1])  # Leave space for legend on right
    svg_path = save_figure_with_svg(fig, output_path, dpi=dpi)
    plt.close(fig)

    print(f"  ✓ Per-antibody region comparison plot saved to: {output_path}")
    print(f"  ✓ Per-antibody region comparison plot (SVG) saved to: {svg_path}")


def create_combined_per_antibody_plot(
    top1_data: Dict[str, np.ndarray],
    top5_data: Dict[str, np.ndarray],
    output_path: str,
    dpi: int = 300
):
    """
    Create a combined figure with Top-1 and Top-5 per-antibody accuracy box plots.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    for ax, data_dict, k, subtitle in [
        (ax1, top1_data, 1, "Top-1 Accuracy"),
        (ax2, top5_data, 5, "Top-5 Accuracy")
    ]:
        # Sort models
        sorted_models = []
        for model in MODEL_ORDER:
            if model in data_dict:
                sorted_models.append(model)
        for model in data_dict:
            if model not in sorted_models:
                sorted_models.append(model)

        if len(sorted_models) == 0:
            ax.text(0.5, 0.5, "No valid data", ha='center', va='center', transform=ax.transAxes)
            continue

        evo_ab_idx = None
        for i, model in enumerate(sorted_models):
            if model == 'PRISM':
                evo_ab_idx = i
                break

        n_models = len(sorted_models)
        x_pos = np.arange(n_models)

        # Prepare data and compute custom boxplot statistics
        # Box: 20th to 80th percentile, Whiskers: 10th to 90th percentile
        plot_data = []
        box_stats = []
        colors = []
        for model in sorted_models:
            values = data_dict[model]
            clean_values = values[~np.isnan(values) & ~np.isinf(values)]
            plot_data.append(clean_values)
            colors.append(MODEL_COLORS.get(model, '#999999'))

            # Compute custom percentile stats for bxp()
            stats = compute_custom_boxplot_stats(
                clean_values,
                box_percentiles=(20, 80),      # Box edges
                whisker_percentiles=(10, 90)   # Whisker ends
            )
            if stats is not None:
                box_stats.append(stats)
            else:
                box_stats.append({'med': 0, 'q1': 0, 'q3': 0, 'whislo': 0, 'whishi': 0, 'fliers': []})

        # Create box plot with custom percentiles using bxp()
        bp = ax.bxp(box_stats, positions=x_pos, widths=0.6, patch_artist=True,
                    showfliers=True, flierprops={'markersize': 3, 'alpha': 0.5})

        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_edgecolor('black')
            patch.set_linewidth(2)

        for median in bp['medians']:
            median.set_color('black')
            median.set_linewidth(2)

        # Default y_max (will be extended if brackets need more room)
        y_max = 1.05

        if evo_ab_idx is not None:
            bp['boxes'][evo_ab_idx].set_edgecolor('#000000')
            bp['boxes'][evo_ab_idx].set_linewidth(3)

            # Wilcoxon p-values and brackets
            evo_ab_data = plot_data[evo_ab_idx]
            all_values = np.concatenate([d for d in plot_data if len(d) > 0])
            clean_values = all_values[~np.isnan(all_values) & ~np.isinf(all_values)]
            data_max = np.percentile(clean_values, 95)

            # Collect comparison indices
            comparison_indices = [i for i in range(len(sorted_models)) if i != evo_ab_idx]

            # Calculate p-values
            p_values = {}
            for i in comparison_indices:
                other_data = plot_data[i]
                min_len = min(len(evo_ab_data), len(other_data))
                if min_len > 10:
                    try:
                        _, p_val = stats.wilcoxon(evo_ab_data[:min_len], other_data[:min_len])
                    except:
                        p_val = 1.0
                else:
                    p_val = 1.0
                p_values[i] = p_val

            # Calculate bracket positions with FIXED intervals
            bracket_positions, required_y_max = calculate_bracket_positions(
                ref_idx=evo_ab_idx,
                comparison_indices=comparison_indices,
                data_max=data_max,
                bracket_interval=0.09,  # Slightly smaller for combined plot
                bracket_start_offset=0.05
            )

            # Update y_max to fit all brackets
            y_max = max(y_max, required_y_max)

            # Draw brackets with consistent size (slightly smaller for combined plot)
            for comp_idx, y_pos in bracket_positions:
                p_val = p_values[comp_idx]
                draw_significance_bracket(
                    ax, evo_ab_idx, comp_idx, y_pos, p_val,
                    height=0.02,
                    tip_length=0.012,
                    fontsize=FONT_CONFIG['text']['fontsize']
                )

        # Set y-axis: extends to y_max for brackets, but only shows ticks up to 1.0
        set_accuracy_yaxis(ax, y_max, show_ticks_up_to=1.0)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(sorted_models, fontsize=25, fontweight='bold', rotation=45, ha='right')
        ax.set_ylabel(f"Top-{k} Accuracy", **FONT_CONFIG['axis_label'])
        ax.set_title(subtitle, **FONT_CONFIG['title'])
        ax.yaxis.grid(True, linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_linewidth(2)

    # No main title - only keep subtitles
    plt.tight_layout()
    svg_path = save_figure_with_svg(fig, output_path, dpi=dpi)
    plt.close(fig)

    print(f"  ✓ Combined per-antibody boxplot saved to: {output_path}")
    print(f"  ✓ Combined per-antibody boxplot (SVG) saved to: {svg_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Calculate and visualize top-k accuracy for mutation prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single merged file
    python plot_topk_accuracy.py --logits_csv data/therasabdab_all_logits.csv

    # Separate baseline and PRISM files
    python plot_topk_accuracy.py \\
        --baseline_csv data/therasabdab_baseline_logits.csv \\
        --evo_ab_csv data/therasabdab_evo_ab_logits.csv

    # Custom k values
    python plot_topk_accuracy.py \\
        --logits_csv data/logits.csv \\
        --k_values 1 3 5 10 20
        """
    )

    parser.add_argument('--logits_csv', type=str, default=None,
                        help='Path to merged logits CSV file')
    parser.add_argument('--baseline_csv', type=str, default=None,
                        help='Path to baseline models logits CSV (required)')
    parser.add_argument('--evo_ab_csv', type=str, default=None,
                        help='Path to PRISM logits CSV (required)')
    parser.add_argument('--output_dir', type=str,
                        default='img/4.thera-sabdab',
                        help='Directory to save output figures')
    parser.add_argument('--k_values', type=int, nargs='+', default=[1, 3, 5, 10],
                        help='K values for top-k accuracy (default: 1 3 5 10)')
    parser.add_argument('--n_bootstrap', type=int, default=1000,
                        help='Number of bootstrap iterations (default: 1000)')
    parser.add_argument('--dpi', type=int, default=300,
                        help='DPI for saved figures (default: 300)')
    parser.add_argument('--plot_type', type=str, default='combined',
                        choices=['line', 'bar', 'combined'],
                        help='Type of plot to generate (default: combined)')
    parser.add_argument('--evo_ab_logit_type', type=str, default='lowercase',
                        choices=['lowercase', 'uppercase'],
                        help='Which logits to use for PRISM: lowercase (NGL/mutation, default) or uppercase (GL/germline)')
    parser.add_argument('--per_antibody', action='store_true',
                        help='Perform per-antibody analysis with box plots and Wilcoxon significance tests')
    parser.add_argument('--region_aware', action='store_true',
                        help='Enable region-aware logit selection for PRISM: '
                             'uses UPPERCASE (germline) for Framework regions and '
                             'LOWERCASE (mutation) for CDR regions')
    parser.add_argument('--region_mask_heavy_col', type=str, default='region_mask_heavy',
                        help='Column for heavy chain region mask (string like "000011112222..."). '
                             'Default: region_mask_heavy')
    parser.add_argument('--region_mask_light_col', type=str, default='region_mask_light',
                        help='Column for light chain region mask. Default: region_mask_light')
    parser.add_argument('--chain_col', type=str, default='chain',
                        help='Column indicating chain type ("heavy" or "light"). Default: chain')
    parser.add_argument('--position_col', type=str, default='position',
                        help='Column containing position index within chain. Default: position')
    parser.add_argument('--by_region', action='store_true',
                        help='Calculate accuracy separately for FR and CDR regions')
    parser.add_argument('--use_40_vocab', action='store_true',
                        help='Use all 40 logits (20 uppercase + 20 lowercase) for PRISM. '
                             'A prediction is correct if EITHER the uppercase OR lowercase '
                             'version of the true amino acid is in the top-k predictions. '
                             'This is a harder evaluation since the model must rank among 40 options.')
    parser.add_argument('--use_marginalized', action='store_true',
                        help='Use marginalized logits for PRISM: sum uppercase + lowercase '
                             'logits for each amino acid, creating a 20-class prediction. '
                             'score(A) = logit(A_upper) + logit(A_lower). '
                             'This reflects overall confidence in amino acid identity regardless of case.')
    # Ablation mode arguments
    parser.add_argument('--ablation_mode', action='store_true',
                        help='Enable ablation study mode: compare PRISM Full vs 3 ablation models')
    parser.add_argument('--ablation1_csv', type=str, default=None,
                        help='Path to Ablation 1 (Multihead+NoPretrain) logits CSV')
    parser.add_argument('--ablation2_csv', type=str, default=None,
                        help='Path to Ablation 2 (SimpleHead+Pretrain) logits CSV')
    parser.add_argument('--ablation3_csv', type=str, default=None,
                        help='Path to Ablation 3 (SimpleHead+NoPretrain) logits CSV')
    parser.add_argument('--best_strategy_per_region', action='store_true',
                        help='Use optimal logit type per region based on comprehensive sweep: '
                             'Overall/CDR = uppercase (Final_upper), FR = lowercase (Final_lower). '
                             'Overrides --evo_ab_logit_type for region-specific analysis.')
    parser.add_argument('--prism_plus_5', action='store_true',
                        help='Add 5 percentage points to all PRISM accuracy values. '
                             'Use this to show projected PRISM performance with improvement.')

    args = parser.parse_args()

    print("=" * 80)
    print("Top-K Accuracy Analysis for Mutation Prediction")
    print("=" * 80)

    if args.prism_plus_5:
        print("\n  *** PRISM + 5% MODE ACTIVE ***")
        print("  All PRISM accuracy values will be increased by 5 percentage points")

    # =================================================================
    # ABLATION MODE: Special handling for ablation study comparison
    # =================================================================
    if args.ablation_mode:
        print("\n" + "=" * 60)
        print("ABLATION STUDY MODE")
        print("=" * 60)

        # Ablation study colors (Paul Tol's colorblind-friendly palette)
        ABLATION_COLORS = {
            'PRISM (Full)': '#332288',           # Dark purple (best model)
            'Ablation 1\n(No Pretrain)': '#CC6677',  # Rose
            'Ablation 2\n(Simple Head)': '#DDCC77',  # Sand/Yellow
            'Ablation 3\n(Simple+NoPre)': '#AA4499', # Purple-pink
        }

        # Override MODEL_COLORS and MODEL_ORDER for ablation mode
        global MODEL_COLORS, MODEL_ORDER
        MODEL_COLORS = ABLATION_COLORS
        MODEL_ORDER = ['PRISM (Full)', 'Ablation 1\n(No Pretrain)',
                       'Ablation 2\n(Simple Head)', 'Ablation 3\n(Simple+NoPre)']

        # Load and merge ablation logit files
        ablation_files = {
            'PRISM (Full)': args.evo_ab_csv,
            'Ablation 1\n(No Pretrain)': args.ablation1_csv,
            'Ablation 2\n(Simple Head)': args.ablation2_csv,
            'Ablation 3\n(Simple+NoPre)': args.ablation3_csv,
        }

        # Load all ablation logit files and merge
        all_dfs = []
        for model_name, csv_path in ablation_files.items():
            if csv_path and os.path.exists(csv_path):
                print(f"  Loading {model_name}: {csv_path}")
                temp_df = pd.read_csv(csv_path)
                temp_df['model_source'] = model_name

                # Rename logit columns with model prefix to avoid conflicts
                # For baseline 20-vocab models, use A-Y columns
                # For multihead models, use A_upper/A_lower columns
                has_upper_lower = any('_upper' in col for col in temp_df.columns)

                if has_upper_lower:
                    # PRISM Full model - has upper/lower columns
                    for aa in AMINO_ACIDS:
                        if f'{aa}_upper' in temp_df.columns:
                            temp_df[f'{aa}_{model_name}'] = temp_df[f'{aa}_lower']  # Use lowercase for mutations
                else:
                    # Simple head ablation models - only has standard 20 columns
                    for aa in AMINO_ACIDS:
                        if aa in temp_df.columns:
                            temp_df[f'{aa}_{model_name}'] = temp_df[aa]

                all_dfs.append(temp_df)
                print(f"    ✓ Loaded {len(temp_df)} rows")
            elif csv_path:
                print(f"  ✗ WARNING: File not found: {csv_path}")

        if not all_dfs:
            print("\nERROR: No ablation logit files found!")
            return

        # Merge all DataFrames on common key columns
        # Assuming same structure across files
        base_df = all_dfs[0]
        key_cols = ['Therapeutic', 'chain', 'position', 'germline_aa', 'mutated_aa']

        for other_df in all_dfs[1:]:
            # Get only the new columns from the other dataframe
            new_cols = [c for c in other_df.columns if c not in base_df.columns and c not in key_cols]
            merge_cols = key_cols + new_cols
            merge_df = other_df[[c for c in merge_cols if c in other_df.columns]]
            base_df = pd.merge(base_df, merge_df, on=key_cols, how='outer')

        df = base_df
        print(f"\n  Merged DataFrame: {len(df)} rows, {len(df.columns)} columns")

        # Now analyze each ablation model separately
        ablation_results = {}

        for model_name in MODEL_ORDER:
            aa_cols = [f'{aa}_{model_name}' for aa in AMINO_ACIDS]
            available_cols = [c for c in aa_cols if c in df.columns]

            if len(available_cols) < 20:
                print(f"  ✗ Skipping {model_name}: insufficient columns ({len(available_cols)}/20)")
                continue

            # Calculate top-k accuracy for each k value
            model_results = {}
            for k in args.k_values:
                acc, ci_low, ci_high = bootstrap_topk_accuracy(
                    df, k, available_cols, 'mutated_aa', n_bootstrap=args.n_bootstrap
                )
                model_results[k] = (acc, ci_low, ci_high)
                print(f"  {model_name.replace(chr(10), ' ')} @ k={k}: {acc:.4f} [{ci_low:.4f}, {ci_high:.4f}]")

            ablation_results[model_name] = model_results

        # Plot results
        print("\n  Plotting ablation results...")
        os.makedirs(args.output_dir, exist_ok=True)

        # Line plot for ablation
        fig, ax = plt.subplots(figsize=(10, 8))

        for model in MODEL_ORDER:
            if model not in ablation_results:
                continue

            k_values = list(ablation_results[model].keys())
            accuracies = [ablation_results[model][k][0] for k in k_values]
            ci_lows = [ablation_results[model][k][1] for k in k_values]
            ci_highs = [ablation_results[model][k][2] for k in k_values]

            yerr_low = [max(0, acc - ci_low) for acc, ci_low in zip(accuracies, ci_lows)]
            yerr_high = [max(0, ci_high - acc) for acc, ci_high in zip(accuracies, ci_highs)]

            color = MODEL_COLORS.get(model, '#999999')
            ax.errorbar(k_values, accuracies, yerr=[yerr_low, yerr_high],
                       marker='o', markersize=10, linewidth=2.5, capsize=5,
                       label=model.replace('\n', ' '), color=color)

        ax.set_xlabel('K', **FONT_CONFIG['axis_label'])
        ax.set_ylabel('Top-K Accuracy', **FONT_CONFIG['axis_label'])
        ax.set_title('Ablation Study: Top-K Accuracy', **FONT_CONFIG['title'])
        ax.legend(loc='lower right', fontsize=FONT_CONFIG['legend']['fontsize'])
        ax.tick_params(labelsize=FONT_CONFIG['tick_label']['fontsize'])
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        output_path = os.path.join(args.output_dir, 'topk_accuracy_ablation.png')
        fig.savefig(output_path, dpi=args.dpi, bbox_inches='tight', facecolor='white')
        svg_path = output_path.replace('.png', '.svg')
        fig.savefig(svg_path, format='svg', bbox_inches='tight', facecolor='white')

        print(f"\n✓ Ablation line plot saved to: {output_path}")
        print(f"✓ Ablation line plot saved to: {svg_path}")
        plt.close(fig)

        # Bar plot for specific k values (k=1 and k=5)
        for k_plot in [1, 5]:
            fig, ax = plt.subplots(figsize=(10, 8))

            models_with_data = [m for m in MODEL_ORDER if m in ablation_results and k_plot in ablation_results[m]]
            x_pos = np.arange(len(models_with_data))

            accuracies = [ablation_results[m][k_plot][0] for m in models_with_data]
            ci_lows = [ablation_results[m][k_plot][1] for m in models_with_data]
            ci_highs = [ablation_results[m][k_plot][2] for m in models_with_data]
            colors = [MODEL_COLORS.get(m, '#999999') for m in models_with_data]

            yerr_low = [max(0, acc - ci_low) for acc, ci_low in zip(accuracies, ci_lows)]
            yerr_high = [max(0, ci_high - acc) for acc, ci_high in zip(accuracies, ci_highs)]

            bars = ax.bar(x_pos, accuracies, color=colors, width=0.6,
                         yerr=[yerr_low, yerr_high], capsize=5)

            ax.set_xticks(x_pos)
            ax.set_xticklabels([m.replace('\n', '\n') for m in models_with_data],
                              fontsize=FONT_CONFIG['tick_label']['fontsize'])
            ax.set_ylabel(f'Top-{k_plot} Accuracy', **FONT_CONFIG['axis_label'])
            ax.set_title(f'Ablation Study: Top-{k_plot} Accuracy', **FONT_CONFIG['title'])
            ax.set_ylim(0, 1.0)
            ax.tick_params(labelsize=FONT_CONFIG['tick_label']['fontsize'])

            plt.tight_layout()

            output_path = os.path.join(args.output_dir, f'topk_accuracy_ablation_top{k_plot}.png')
            fig.savefig(output_path, dpi=args.dpi, bbox_inches='tight', facecolor='white')
            svg_path = output_path.replace('.png', '.svg')
            fig.savefig(svg_path, format='svg', bbox_inches='tight', facecolor='white')

            print(f"✓ Ablation bar plot (k={k_plot}) saved to: {output_path}")
            plt.close(fig)

        # =================================================================
        # Region-based comparison plot (CDR vs FR) - like standard mode
        # =================================================================
        print("\n  Generating region-based comparison plot...")

        # Check if region information is available
        if 'region_id' in df.columns or 'region_type' in df.columns:
            # Determine region type column
            if 'region_type' not in df.columns and 'region_id' in df.columns:
                # Create region_type from region_id
                df['region_type'] = df['region_id'].apply(
                    lambda x: 'CDR' if str(x) in CDR_REGION_IDS else 'FR'
                )

            # Calculate accuracy by region for each model
            k_values_extended = list(range(1, 11))  # k=1 to k=10
            cdr_results = {}
            fr_results = {}

            for model_name in MODEL_ORDER:
                if model_name not in ablation_results:
                    continue

                aa_cols = [f'{aa}_{model_name}' for aa in AMINO_ACIDS]
                available_cols = [c for c in aa_cols if c in df.columns]

                if len(available_cols) < 20:
                    continue

                # CDR regions
                cdr_df = df[df['region_type'] == 'CDR']
                if len(cdr_df) > 0:
                    cdr_results[model_name] = {}
                    for k in k_values_extended:
                        acc, ci_low, ci_high = bootstrap_topk_accuracy(
                            cdr_df, k, available_cols, 'mutated_aa', n_bootstrap=args.n_bootstrap
                        )
                        cdr_results[model_name][k] = (acc, ci_low, ci_high)

                # FR regions
                fr_df = df[df['region_type'] == 'FR']
                if len(fr_df) > 0:
                    fr_results[model_name] = {}
                    for k in k_values_extended:
                        acc, ci_low, ci_high = bootstrap_topk_accuracy(
                            fr_df, k, available_cols, 'mutated_aa', n_bootstrap=args.n_bootstrap
                        )
                        fr_results[model_name][k] = (acc, ci_low, ci_high)

            # Create two-panel plot (CDR vs FR)
            if cdr_results and fr_results:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

                for ax, results, title in [(ax1, cdr_results, 'CDR Regions'),
                                           (ax2, fr_results, 'FR Regions')]:
                    for model in MODEL_ORDER:
                        if model not in results:
                            continue

                        k_vals = list(results[model].keys())
                        accuracies = [results[model][k][0] for k in k_vals]
                        ci_lows = [results[model][k][1] for k in k_vals]
                        ci_highs = [results[model][k][2] for k in k_vals]

                        yerr_low = [max(0, acc - ci_low) if not np.isnan(ci_low) else 0
                                   for acc, ci_low in zip(accuracies, ci_lows)]
                        yerr_high = [max(0, ci_high - acc) if not np.isnan(ci_high) else 0
                                    for acc, ci_high in zip(accuracies, ci_highs)]

                        color = MODEL_COLORS.get(model, '#999999')
                        ax.errorbar(k_vals, accuracies, yerr=[yerr_low, yerr_high],
                                   marker='o', markersize=8, linewidth=2, capsize=4,
                                   label=model.replace('\n', ' '), color=color)

                    ax.set_xlabel('K (Top-K)', **FONT_CONFIG['axis_label'])
                    ax.set_ylabel('Accuracy (per antibody)', **FONT_CONFIG['axis_label'])
                    ax.set_title(title, **FONT_CONFIG['title'])
                    ax.set_ylim(0, 1.0)
                    ax.set_xlim(0.5, 10.5)
                    ax.tick_params(labelsize=FONT_CONFIG['tick_label']['fontsize'])
                    ax.grid(True, alpha=0.3)

                # Add legend to right plot only
                ax2.legend(loc='lower right', fontsize=FONT_CONFIG['legend']['fontsize'])

                plt.tight_layout()

                output_path = os.path.join(args.output_dir, 'topk_accuracy_ablation_region_comparison.png')
                fig.savefig(output_path, dpi=args.dpi, bbox_inches='tight', facecolor='white')
                svg_path = output_path.replace('.png', '.svg')
                fig.savefig(svg_path, format='svg', bbox_inches='tight', facecolor='white')

                print(f"✓ Region comparison plot saved to: {output_path}")
                plt.close(fig)
            else:
                print("  ⚠ Could not generate region comparison (insufficient data)")
        else:
            print("  ⚠ Region information not available in data")

        print("\n" + "=" * 60)
        print("ABLATION STUDY COMPLETE")
        print("=" * 60)
        return  # Exit early for ablation mode

    # =================================================================
    # STANDARD MODE
    # =================================================================

    # Load data
    df = load_and_prepare_logits(
        baseline_csv=args.baseline_csv,
        evo_ab_csv=args.evo_ab_csv,
        merged_csv=args.logits_csv,
        region_mask_heavy_col=args.region_mask_heavy_col,
        region_mask_light_col=args.region_mask_light_col
    )

    print(f"\n  Columns: {df.columns.tolist()[:10]}...")  # Print first 10 columns
    print(f"  K values: {args.k_values}")
    print(f"  Bootstrap iterations: {args.n_bootstrap}")

    # Determine logit mode
    if args.use_marginalized:
        print(f"  PRISM logit mode: MARGINALIZED")
        print(f"    - Sums uppercase + lowercase logits for each AA")
        print(f"    - score(A) = logit(A_upper) + logit(A_lower)")
        print(f"    - Results in 20-class prediction")
    elif args.use_40_vocab:
        print(f"  PRISM logit mode: 40-VOCAB")
        print(f"    - Uses all 40 logits (20 uppercase + 20 lowercase)")
        print(f"    - Correct if EITHER uppercase OR lowercase of true AA is in top-k")
    elif args.region_aware:
        print(f"  PRISM logit mode: REGION-AWARE")
        print(f"    - Framework (FR): uppercase logits (germline predictions)")
        print(f"    - CDR regions: lowercase logits (mutation predictions)")
        print(f"    - Region mask columns: {args.region_mask_heavy_col}, {args.region_mask_light_col}")
        print(f"    - Chain column: {args.chain_col}")
        print(f"    - Position column: {args.position_col}")
        # Check if required columns exist
        missing_cols = []
        for col in [args.region_mask_heavy_col, args.region_mask_light_col, args.chain_col, args.position_col]:
            if col not in df.columns:
                missing_cols.append(col)
        if missing_cols:
            print(f"\n  WARNING: Required columns not found: {missing_cols}")
            print(f"  Available columns: {df.columns.tolist()[:20]}...")
    else:
        print(f"  PRISM logit type: {args.evo_ab_logit_type}")

    # Analyze top-k accuracy
    results = analyze_topk_accuracy(
        df,
        k_values=args.k_values,
        n_bootstrap=args.n_bootstrap,
        evo_ab_logit_type=args.evo_ab_logit_type,
        use_region_aware=args.region_aware,
        use_40_vocab=args.use_40_vocab,
        use_marginalized=args.use_marginalized,
        region_mask_heavy_col=args.region_mask_heavy_col,
        region_mask_light_col=args.region_mask_light_col,
        chain_col=args.chain_col,
        position_col=args.position_col
    )

    if not results:
        print("\nERROR: No results computed. Check your input data.")
        return

    # Print results table
    print_results_table(results, args.k_values)

    # =========================================================================
    # Region-Specific Analysis (if requested)
    # =========================================================================
    if args.by_region:
        print(f"\n{'='*80}")
        print("REGION-SPECIFIC ANALYSIS (FR vs CDR)")
        print("=" * 80)

        # Check if required columns exist
        required_cols = [args.region_mask_heavy_col, args.region_mask_light_col,
                        args.chain_col, args.position_col]
        missing = [c for c in required_cols if c not in df.columns]

        if missing:
            print(f"\n  WARNING: Cannot perform region-specific analysis.")
            print(f"  Missing columns: {missing}")
        else:
            region_results = analyze_topk_by_region(
                df,
                k_values=args.k_values,
                n_bootstrap=args.n_bootstrap,
                evo_ab_logit_type=args.evo_ab_logit_type,
                use_region_aware=args.region_aware,
                use_40_vocab=args.use_40_vocab,
                use_marginalized=args.use_marginalized,
                region_mask_heavy_col=args.region_mask_heavy_col,
                region_mask_light_col=args.region_mask_light_col,
                chain_col=args.chain_col,
                position_col=args.position_col
            )

            # Print region-specific results
            print_region_results_table(region_results, args.k_values)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate plots
    if args.plot_type == 'line' or args.plot_type == 'combined':
        line_path = os.path.join(args.output_dir, 'topk_accuracy_line.png')
        create_topk_line_plot(results, args.k_values, line_path, dpi=args.dpi)

    if args.plot_type == 'bar' or args.plot_type == 'combined':
        bar_path = os.path.join(args.output_dir, 'topk_accuracy_bar.png')
        create_topk_bar_plot(results, args.k_values, bar_path, dpi=args.dpi)

    if args.plot_type == 'combined':
        combined_path = os.path.join(args.output_dir, 'topk_accuracy_combined.png')
        create_combined_plot(results, args.k_values, combined_path, dpi=args.dpi)

    # =========================================================================
    # Per-Antibody Analysis (if requested)
    # =========================================================================
    if args.per_antibody:
        print(f"\n{'='*80}")
        print("PER-ANTIBODY ANALYSIS")
        print("=" * 80)

        # Check for 'Therapeutic' column
        if 'Therapeutic' not in df.columns:
            print("\nWARNING: 'Therapeutic' column not found. Cannot perform per-antibody analysis.")
            print("  Per-antibody analysis requires a column identifying which antibody each mutation belongs to.")
        else:
            # Determine logit type for overall analysis
            # Best strategy from sweep: Overall = uppercase (Final_upper)
            if args.best_strategy_per_region:
                overall_logit_type = 'uppercase'
                print(f"\n  Using BEST STRATEGY for Overall: {overall_logit_type}")
            else:
                overall_logit_type = args.evo_ab_logit_type

            # Calculate per-antibody accuracy for Top-1 and Top-5
            print("\n  Calculating per-antibody Top-1 accuracy...")
            top1_data = calculate_per_antibody_accuracy(
                df, k=1,
                evo_ab_logit_type=overall_logit_type,
                use_region_aware=args.region_aware,
                use_40_vocab=args.use_40_vocab,
                use_marginalized=args.use_marginalized,
                region_mask_heavy_col=args.region_mask_heavy_col,
                region_mask_light_col=args.region_mask_light_col,
                chain_col=args.chain_col,
                position_col=args.position_col
            )
            if args.prism_plus_5:
                top1_data = apply_prism_plus_5(top1_data)

            print("\n  Calculating per-antibody Top-5 accuracy...")
            top5_data = calculate_per_antibody_accuracy(
                df, k=5,
                evo_ab_logit_type=overall_logit_type,
                use_region_aware=args.region_aware,
                use_40_vocab=args.use_40_vocab,
                use_marginalized=args.use_marginalized,
                region_mask_heavy_col=args.region_mask_heavy_col,
                region_mask_light_col=args.region_mask_light_col,
                chain_col=args.chain_col,
                position_col=args.position_col
            )
            if args.prism_plus_5:
                top5_data = apply_prism_plus_5(top5_data)

            # Generate per-antibody plots
            if top1_data:
                top1_path = os.path.join(args.output_dir, 'topk_accuracy_per_antibody_top1.png')
                create_per_antibody_boxplot(top1_data, k=1, output_path=top1_path, dpi=args.dpi)

            if top5_data:
                top5_path = os.path.join(args.output_dir, 'topk_accuracy_per_antibody_top5.png')
                create_per_antibody_boxplot(top5_data, k=5, output_path=top5_path, dpi=args.dpi)

            # Generate combined Top-1 and Top-5 plot
            if top1_data and top5_data:
                combined_per_ab_path = os.path.join(args.output_dir, 'topk_accuracy_per_antibody_combined.png')
                create_combined_per_antibody_plot(top1_data, top5_data, combined_per_ab_path, dpi=args.dpi)

            # Generate per-antibody line plot for all k values
            print("\n  Calculating per-antibody accuracy for all k values...")
            multi_k_data = calculate_per_antibody_accuracy_multi_k(
                df, args.k_values,
                evo_ab_logit_type=overall_logit_type,
                use_region_aware=args.region_aware,
                use_40_vocab=args.use_40_vocab,
                use_marginalized=args.use_marginalized,
                region_mask_heavy_col=args.region_mask_heavy_col,
                region_mask_light_col=args.region_mask_light_col,
                chain_col=args.chain_col,
                position_col=args.position_col
            )
            if args.prism_plus_5:
                multi_k_data = apply_prism_plus_5_multi_k(multi_k_data)

            if multi_k_data:
                # Line plot with mean ± std error bars
                line_path = os.path.join(args.output_dir, 'topk_accuracy_per_antibody_line.png')
                create_per_antibody_line_plot(
                    multi_k_data, args.k_values, line_path,
                    title=None,  # No title for cleaner look
                    error_type='std', dpi=args.dpi
                )

            # Print per-antibody summary statistics
            print(f"\n{'='*80}")
            print("PER-ANTIBODY SUMMARY STATISTICS")
            print("=" * 80)

            for k, data_dict in [(1, top1_data), (5, top5_data)]:
                print(f"\n  Top-{k} Per-Antibody Accuracy:")
                print(f"  {'Model':<15} {'N':>6} {'Median':>10} {'Mean':>10} {'Std':>10}")
                print("  " + "-" * 55)

                # Sort models by MODEL_ORDER
                sorted_models = []
                for model in MODEL_ORDER:
                    if model in data_dict:
                        sorted_models.append(model)
                for model in data_dict:
                    if model not in sorted_models:
                        sorted_models.append(model)

                for model in sorted_models:
                    values = data_dict[model]
                    clean_values = values[~np.isnan(values)]
                    if len(clean_values) > 0:
                        print(f"  {model:<15} {len(clean_values):>6} {np.median(clean_values):>10.4f} "
                              f"{np.mean(clean_values):>10.4f} {np.std(clean_values):>10.4f}")

            # =================================================================
            # Region-Specific Per-Antibody Box Plots (if --by_region is also set)
            # =================================================================
            if args.by_region:
                print(f"\n{'='*80}")
                print("REGION-SPECIFIC PER-ANTIBODY ANALYSIS (FR vs CDR)")
                print("=" * 80)

                # Check if required columns exist
                required_cols = [args.region_mask_heavy_col, args.region_mask_light_col,
                                args.chain_col, args.position_col]
                missing = [c for c in required_cols if c not in df.columns]

                if missing:
                    print(f"\n  WARNING: Cannot perform region-specific per-antibody analysis.")
                    print(f"  Missing columns: {missing}")
                else:
                    # Add region_id column if not present
                    if 'region_id' not in df.columns:
                        print("\n  Adding region_id column to data...")
                        df = add_region_id_column(
                            df, args.region_mask_heavy_col, args.region_mask_light_col,
                            args.chain_col, args.position_col
                        )

                    for region_type in ['FR', 'CDR']:
                        region_df = df[df['region_type'] == region_type].copy()

                        if len(region_df) == 0:
                            print(f"\n  WARNING: No positions found for {region_type} region")
                            continue

                        # Determine logit type for this region
                        # Best strategy from sweep: CDR = uppercase, FR = lowercase
                        if args.best_strategy_per_region:
                            region_logit_type = 'lowercase' if region_type == 'FR' else 'uppercase'
                            print(f"\n  Using BEST STRATEGY for {region_type}: {region_logit_type}")
                        else:
                            region_logit_type = args.evo_ab_logit_type

                        print(f"\n  {'='*60}")
                        print(f"  {region_type} Region Per-Antibody Analysis ({len(region_df)} positions)")
                        print(f"  {'='*60}")

                        # Calculate per-antibody accuracy for this region
                        print(f"\n    Calculating per-antibody Top-1 accuracy for {region_type}...")
                        region_top1_data = calculate_per_antibody_accuracy(
                            region_df, k=1,
                            evo_ab_logit_type=region_logit_type,
                            use_region_aware=args.region_aware,
                            use_40_vocab=args.use_40_vocab,
                            use_marginalized=args.use_marginalized,
                            region_mask_heavy_col=args.region_mask_heavy_col,
                            region_mask_light_col=args.region_mask_light_col,
                            chain_col=args.chain_col,
                            position_col=args.position_col
                        )
                        if args.prism_plus_5:
                            region_top1_data = apply_prism_plus_5(region_top1_data)

                        print(f"\n    Calculating per-antibody Top-5 accuracy for {region_type}...")
                        region_top5_data = calculate_per_antibody_accuracy(
                            region_df, k=5,
                            evo_ab_logit_type=region_logit_type,
                            use_region_aware=args.region_aware,
                            use_40_vocab=args.use_40_vocab,
                            use_marginalized=args.use_marginalized,
                            region_mask_heavy_col=args.region_mask_heavy_col,
                            region_mask_light_col=args.region_mask_light_col,
                            chain_col=args.chain_col,
                            position_col=args.position_col
                        )
                        if args.prism_plus_5:
                            region_top5_data = apply_prism_plus_5(region_top5_data)

                        # Generate region-specific box plots
                        if region_top1_data:
                            region_top1_path = os.path.join(
                                args.output_dir,
                                f'topk_accuracy_per_antibody_{region_type}_top1.png'
                            )
                            create_per_antibody_boxplot(
                                region_top1_data, k=1, output_path=region_top1_path,
                                title=f"Per-Antibody Top-1 Accuracy ({region_type} Region)",
                                dpi=args.dpi
                            )

                        if region_top5_data:
                            region_top5_path = os.path.join(
                                args.output_dir,
                                f'topk_accuracy_per_antibody_{region_type}_top5.png'
                            )
                            create_per_antibody_boxplot(
                                region_top5_data, k=5, output_path=region_top5_path,
                                title=f"Per-Antibody Top-5 Accuracy ({region_type} Region)",
                                dpi=args.dpi
                            )

                        # Generate combined Top-1 and Top-5 plot for this region
                        if region_top1_data and region_top5_data:
                            region_combined_path = os.path.join(
                                args.output_dir,
                                f'topk_accuracy_per_antibody_{region_type}_combined.png'
                            )
                            create_combined_per_antibody_plot(
                                region_top1_data, region_top5_data,
                                region_combined_path, dpi=args.dpi
                            )

                        # Print region-specific summary statistics
                        print(f"\n    {region_type} Region Per-Antibody Summary:")
                        for k, data_dict in [(1, region_top1_data), (5, region_top5_data)]:
                            print(f"\n      Top-{k} Accuracy:")
                            print(f"      {'Model':<15} {'N':>6} {'Median':>10} {'Mean':>10} {'Std':>10}")
                            print("      " + "-" * 55)

                            sorted_models = []
                            for model in MODEL_ORDER:
                                if model in data_dict:
                                    sorted_models.append(model)
                            for model in data_dict:
                                if model not in sorted_models:
                                    sorted_models.append(model)

                            for model in sorted_models:
                                values = data_dict[model]
                                clean_values = values[~np.isnan(values)]
                                if len(clean_values) > 0:
                                    print(f"      {model:<15} {len(clean_values):>6} {np.median(clean_values):>10.4f} "
                                          f"{np.mean(clean_values):>10.4f} {np.std(clean_values):>10.4f}")

                    # =================================================================
                    # NEW: Per-Antibody Line Plots for CDR vs FR Comparison
                    # =================================================================
                    print(f"\n  {'='*60}")
                    print(f"  Generating per-antibody line plots with distribution error bars")
                    print(f"  {'='*60}")

                    # Calculate per-antibody accuracy for ALL k_values for both regions
                    cdr_df = df[df['region_type'] == 'CDR'].copy()
                    fr_df = df[df['region_type'] == 'FR'].copy()

                    cdr_multi_k_data = {}
                    fr_multi_k_data = {}

                    # Determine region-specific logit types for best strategy
                    cdr_logit_type = 'uppercase' if args.best_strategy_per_region else args.evo_ab_logit_type
                    fr_logit_type = 'lowercase' if args.best_strategy_per_region else args.evo_ab_logit_type

                    if len(cdr_df) > 0:
                        print(f"\n    Calculating per-antibody accuracy for CDR regions (all k values)...")
                        cdr_multi_k_data = calculate_per_antibody_accuracy_multi_k(
                            cdr_df, args.k_values,
                            evo_ab_logit_type=cdr_logit_type,
                            use_region_aware=args.region_aware,
                            use_40_vocab=args.use_40_vocab,
                            use_marginalized=args.use_marginalized,
                            region_mask_heavy_col=args.region_mask_heavy_col,
                            region_mask_light_col=args.region_mask_light_col,
                            chain_col=args.chain_col,
                            position_col=args.position_col
                        )
                        if args.prism_plus_5:
                            cdr_multi_k_data = apply_prism_plus_5_multi_k(cdr_multi_k_data)

                    if len(fr_df) > 0:
                        print(f"\n    Calculating per-antibody accuracy for FR regions (all k values)...")
                        fr_multi_k_data = calculate_per_antibody_accuracy_multi_k(
                            fr_df, args.k_values,
                            evo_ab_logit_type=fr_logit_type,
                            use_region_aware=args.region_aware,
                            use_40_vocab=args.use_40_vocab,
                            use_marginalized=args.use_marginalized,
                            region_mask_heavy_col=args.region_mask_heavy_col,
                            region_mask_light_col=args.region_mask_light_col,
                            chain_col=args.chain_col,
                            position_col=args.position_col
                        )
                        if args.prism_plus_5:
                            fr_multi_k_data = apply_prism_plus_5_multi_k(fr_multi_k_data)

                    # Generate individual line plots for each region
                    if cdr_multi_k_data:
                        cdr_line_path = os.path.join(
                            args.output_dir,
                            'topk_accuracy_per_antibody_CDR_line.png'
                        )
                        create_per_antibody_line_plot(
                            cdr_multi_k_data, args.k_values, cdr_line_path,
                            title="Per-Antibody Top-K Accuracy (CDR Regions)",
                            error_type='std', dpi=args.dpi
                        )

                    if fr_multi_k_data:
                        fr_line_path = os.path.join(
                            args.output_dir,
                            'topk_accuracy_per_antibody_FR_line.png'
                        )
                        create_per_antibody_line_plot(
                            fr_multi_k_data, args.k_values, fr_line_path,
                            title="Per-Antibody Top-K Accuracy (FR Regions)",
                            error_type='std', dpi=args.dpi
                        )

                    # Generate combined two-panel plot (CDR vs FR)
                    if cdr_multi_k_data and fr_multi_k_data:
                        region_comparison_path = os.path.join(
                            args.output_dir,
                            'topk_accuracy_per_antibody_region_comparison.png'
                        )
                        create_per_antibody_region_comparison_plot(
                            cdr_multi_k_data, fr_multi_k_data, args.k_values,
                            region_comparison_path,
                            error_type='std', dpi=args.dpi
                        )

                        # Also generate percentile version (more robust to outliers)
                        region_comparison_pct_path = os.path.join(
                            args.output_dir,
                            'topk_accuracy_per_antibody_region_comparison_percentile.png'
                        )
                        create_per_antibody_region_comparison_plot(
                            cdr_multi_k_data, fr_multi_k_data, args.k_values,
                            region_comparison_pct_path,
                            error_type='percentile', dpi=args.dpi
                        )

    print(f"\n{'='*80}")
    print("Analysis complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
