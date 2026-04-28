#!/usr/bin/env python
# coding: utf-8

"""
Visualize Unpaired OAS Statistics
==================================

This script generates visualizations for unpaired antibody sequences:
- NGL mutation distribution histograms
- Summary statistics
- P90 threshold calculations

Memory-efficient: processes data in chunks to avoid loading everything into RAM.
"""

import os
import sys
import numpy as np
import pandas as pd
import dask.dataframe as dd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

STAGE_2_HEAVY_DIR = "./stage_2_heavy_parquet"
STAGE_2_LIGHT_DIR = "./stage_2_light_parquet"
OUTPUT_DIR = "./img"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════
# FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

class TeeOutput:
    """Capture stdout to both console and file"""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def calculate_statistics(ddf, chain_name):
    """
    Calculate comprehensive statistics for a chain without loading all data.

    Args:
        ddf: Dask DataFrame
        chain_name: "Heavy" or "Light"

    Returns:
        dict with statistics
    """
    print(f"\n{'='*70}")
    print(f"{chain_name} Chain Statistics")
    print('='*70)

    stats = {}

    # Total count
    n_total = len(ddf)
    stats['count'] = n_total
    print(f"  Total sequences: {n_total:,}")

    # B-type distribution
    if 'BType' in ddf.columns:
        print(f"\n  B-Type Distribution:")
        btype_counts = ddf['BType'].value_counts().compute()
        stats['btype_counts'] = btype_counts
        for btype, count in btype_counts.items():
            pct = 100 * count / n_total
            print(f"    {btype}: {count:,} ({pct:.2f}%)")

        # Percent of sequences with > 1 NGL mutation by B-type
        if 'num_ngl_muts' in ddf.columns:
            print(f"\n  Percent with > 1 NGL mutation by B-Type:")
            for btype in btype_counts.index:
                ddf_btype = ddf[ddf['BType'] == btype]
                n_btype = len(ddf_btype)
                n_gt1 = (ddf_btype['num_ngl_muts'] > 1).sum().compute()
                pct_gt1 = 100 * n_gt1 / n_btype if n_btype > 0 else 0
                print(f"    {btype}: {pct_gt1:.2f}% ({n_gt1:,} / {n_btype:,})")
                if 'btype_gt1_pct' not in stats:
                    stats['btype_gt1_pct'] = {}
                stats['btype_gt1_pct'][btype] = pct_gt1

            # Average NGL mutations by B-type
            print(f"\n  Average NGL mutations by B-Type:")
            btype_means = {}
            for btype in btype_counts.index:
                ddf_btype = ddf[ddf['BType'] == btype]
                mean_ngl = ddf_btype['num_ngl_muts'].astype('float64').mean().compute()
                btype_means[btype] = mean_ngl
                print(f"    {btype}: {mean_ngl:.2f}")
            stats['btype_means'] = btype_means

    # NGL mutation statistics
    if 'num_ngl_muts' in ddf.columns:
        print(f"\n  NGL Mutation Statistics (All Sequences):")

        # Convert to standard int64 to avoid nullable Int64 issues with percentiles
        ngl_series = ddf['num_ngl_muts'].astype('float64')

        # Basic statistics (from all sequences)
        stats['ngl_mean'] = ngl_series.mean().compute()
        stats['ngl_std'] = ngl_series.std().compute()
        stats['ngl_min'] = ngl_series.min().compute()
        stats['ngl_max'] = ngl_series.max().compute()

        # Percentiles (from all sequences)
        stats['ngl_25'] = ngl_series.quantile(0.25).compute()
        stats['ngl_50'] = ngl_series.quantile(0.50).compute()
        stats['ngl_75'] = ngl_series.quantile(0.75).compute()

        print(f"    Mean:     {stats['ngl_mean']:.2f}")
        print(f"    Median:   {stats['ngl_50']:.2f}")
        print(f"    Std Dev:  {stats['ngl_std']:.2f}")
        print(f"    Min:      {stats['ngl_min']:.0f}")
        print(f"    Q1 (25%): {stats['ngl_25']:.2f}")
        print(f"    Q3 (75%): {stats['ngl_75']:.2f}")
        print(f"    Max:      {stats['ngl_max']:.0f}")

        # Calculate P90/P95/P99 from Naive B-cells only (matching paired antibody approach)
        if 'BType' in ddf.columns:
            print(f"\n  Naive B-Cell Thresholds (for filtering):")
            ddf_naive = ddf[ddf['BType'] == 'Naive-B-Cells']
            n_naive = len(ddf_naive)

            if n_naive > 0:
                ngl_naive = ddf_naive['num_ngl_muts'].astype('float64')
                stats['naive_count'] = n_naive
                stats['naive_pct'] = 100 * n_naive / n_total
                stats['ngl_p90'] = ngl_naive.quantile(0.90).compute()
                stats['ngl_p95'] = ngl_naive.quantile(0.95).compute()
                stats['ngl_p99'] = ngl_naive.quantile(0.99).compute()

                print(f"    Naive B-cells: {n_naive:,} ({stats['naive_pct']:.2f}%)")
                print(f"    P90 (Naive):   {stats['ngl_p90']:.2f}")
                print(f"    P95 (Naive):   {stats['ngl_p95']:.2f}")
                print(f"    P99 (Naive):   {stats['ngl_p99']:.2f}")
            else:
                print(f"    WARNING: No Naive B-cells found!")
                # Fallback to all sequences
                stats['ngl_p90'] = ngl_series.quantile(0.90).compute()
                stats['ngl_p95'] = ngl_series.quantile(0.95).compute()
                stats['ngl_p99'] = ngl_series.quantile(0.99).compute()
                print(f"    P90 (All):     {stats['ngl_p90']:.2f}")
                print(f"    P95 (All):     {stats['ngl_p95']:.2f}")
                print(f"    P99 (All):     {stats['ngl_p99']:.2f}")

            # Also calculate thresholds for Unsorted B-cells
            print(f"\n  Unsorted B-Cell Thresholds (additional):")
            ddf_unsorted = ddf[ddf['BType'] == 'Unsorted-B-Cells']
            n_unsorted = len(ddf_unsorted)

            if n_unsorted > 0:
                ngl_unsorted = ddf_unsorted['num_ngl_muts'].astype('float64')
                stats['unsorted_count'] = n_unsorted
                stats['unsorted_pct'] = 100 * n_unsorted / n_total
                stats['unsorted_p90'] = ngl_unsorted.quantile(0.90).compute()
                stats['unsorted_p95'] = ngl_unsorted.quantile(0.95).compute()
                stats['unsorted_p99'] = ngl_unsorted.quantile(0.99).compute()

                print(f"    Unsorted B-cells: {n_unsorted:,} ({stats['unsorted_pct']:.2f}%)")
                print(f"    P90 (Unsorted):   {stats['unsorted_p90']:.2f}")
                print(f"    P95 (Unsorted):   {stats['unsorted_p95']:.2f}")
                print(f"    P99 (Unsorted):   {stats['unsorted_p99']:.2f}")
            else:
                print(f"    No Unsorted B-cells found in dataset")
        else:
            print(f"\n  WARNING: BType column not found, using all sequences for thresholds")
            stats['ngl_p90'] = ngl_series.quantile(0.90).compute()
            stats['ngl_p95'] = ngl_series.quantile(0.95).compute()
            stats['ngl_p99'] = ngl_series.quantile(0.99).compute()
            print(f"    P90:      {stats['ngl_p90']:.2f}")
            print(f"    P95:      {stats['ngl_p95']:.2f}")
            print(f"    P99:      {stats['ngl_p99']:.2f}")

        # Count sequences with 0 NGL mutations
        zero_ngl = (ddf['num_ngl_muts'] == 0).sum().compute()
        stats['zero_ngl_count'] = zero_ngl
        stats['zero_ngl_pct'] = 100 * zero_ngl / n_total
        print(f"\n  Sequences with 0 NGL mutations: {zero_ngl:,} ({stats['zero_ngl_pct']:.2f}%)")

        # Sequences above P90
        above_p90 = (ddf['num_ngl_muts'] > stats['ngl_p90']).sum().compute()
        stats['above_p90_count'] = above_p90
        stats['above_p90_pct'] = 100 * above_p90 / n_total
        print(f"  Sequences above P90 ({stats['ngl_p90']:.2f}): {above_p90:,} ({stats['above_p90_pct']:.2f}%)")

    return stats


def sample_for_histogram(ddf, sample_size=1000000):
    """
    Sample data for histogram plotting.

    Args:
        ddf: Dask DataFrame
        sample_size: Number of rows to sample

    Returns:
        pandas Series with sampled NGL values
    """
    n_total = len(ddf)

    if n_total <= sample_size:
        # If dataset is small enough, use all data
        print(f"    Using all {n_total:,} sequences for histogram")
        return ddf['num_ngl_muts'].compute()
    else:
        # Sample uniformly
        frac = sample_size / n_total
        print(f"    Sampling {sample_size:,} / {n_total:,} sequences ({frac*100:.2f}%) for histogram")
        return ddf['num_ngl_muts'].sample(frac=frac, random_state=42).compute()


def sample_naive_for_histogram(ddf, sample_size=1000000):
    """
    Sample naive B-cell data for histogram plotting.

    Args:
        ddf: Dask DataFrame
        sample_size: Number of rows to sample

    Returns:
        pandas Series with sampled NGL values from naive B-cells
    """
    if 'BType' not in ddf.columns:
        print(f"    WARNING: No BType column, returning empty series")
        return pd.Series([], dtype='float64')

    ddf_naive = ddf[ddf['BType'] == 'Naive-B-Cells']
    n_total = len(ddf_naive)

    if n_total == 0:
        print(f"    WARNING: No Naive B-cells found")
        return pd.Series([], dtype='float64')

    if n_total <= sample_size:
        print(f"    Using all {n_total:,} naive B-cell sequences for histogram")
        return ddf_naive['num_ngl_muts'].compute()
    else:
        frac = sample_size / n_total
        print(f"    Sampling {sample_size:,} / {n_total:,} naive B-cell sequences ({frac*100:.2f}%) for histogram")
        return ddf_naive['num_ngl_muts'].sample(frac=frac, random_state=42).compute()


def plot_ngl_distribution(heavy_data, light_data, heavy_stats, light_stats, output_dir):
    """
    Create comprehensive NGL mutation distribution plots.

    Args:
        heavy_data: pandas Series with heavy chain NGL values
        light_data: pandas Series with light chain NGL values
        heavy_stats: dict with heavy chain statistics
        light_stats: dict with light chain statistics
        output_dir: directory to save plots
    """
    print(f"\n{'='*70}")
    print("Creating Visualizations")
    print('='*70)

    # Figure 1: Side-by-side histograms
    print("  [1/8] Creating side-by-side histograms...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Heavy chain
    axes[0].hist(heavy_data, bins=100, alpha=0.7, color='blue', edgecolor='black')
    axes[0].axvline(heavy_stats['ngl_p90'], color='red', linestyle='--', linewidth=2,
                    label=f"P90 = {heavy_stats['ngl_p90']:.2f}")
    axes[0].axvline(heavy_stats['ngl_50'], color='green', linestyle='--', linewidth=2,
                    label=f"Median = {heavy_stats['ngl_50']:.2f}")
    axes[0].set_xlabel('Number of NGL Mutations', fontsize=14)
    axes[0].set_ylabel('Frequency', fontsize=14)
    axes[0].set_title(f'Heavy Chain NGL Distribution\n(n={heavy_stats["count"]:,})', fontsize=16)
    axes[0].legend(fontsize=12)
    axes[0].grid(alpha=0.3)

    # Light chain
    axes[1].hist(light_data, bins=100, alpha=0.7, color='orange', edgecolor='black')
    axes[1].axvline(light_stats['ngl_p90'], color='red', linestyle='--', linewidth=2,
                    label=f"P90 = {light_stats['ngl_p90']:.2f}")
    axes[1].axvline(light_stats['ngl_50'], color='green', linestyle='--', linewidth=2,
                    label=f"Median = {light_stats['ngl_50']:.2f}")
    axes[1].set_xlabel('Number of NGL Mutations', fontsize=14)
    axes[1].set_ylabel('Frequency', fontsize=14)
    axes[1].set_title(f'Light Chain NGL Distribution\n(n={light_stats["count"]:,})', fontsize=16)
    axes[1].legend(fontsize=12)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, "ngl_distribution_sidebyside.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"    Saved: {fig_path}")
    plt.close()

    # Figure 2: Overlayed distributions
    print("  [2/8] Creating overlayed distribution...")
    fig, ax = plt.subplots(figsize=(12, 8))

    ax.hist(heavy_data, bins=100, alpha=0.5, color='blue', label='Heavy Chain', edgecolor='black')
    ax.hist(light_data, bins=100, alpha=0.5, color='orange', label='Light Chain', edgecolor='black')

    ax.axvline(heavy_stats['ngl_p90'], color='darkblue', linestyle='--', linewidth=2,
               label=f"Heavy P90 = {heavy_stats['ngl_p90']:.2f}")
    ax.axvline(light_stats['ngl_p90'], color='darkorange', linestyle='--', linewidth=2,
               label=f"Light P90 = {light_stats['ngl_p90']:.2f}")

    ax.set_xlabel('Number of NGL Mutations', fontsize=14)
    ax.set_ylabel('Frequency', fontsize=14)
    ax.set_title('NGL Mutation Distribution: Heavy vs Light Chains', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, "ngl_distribution_overlay.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"    Saved: {fig_path}")
    plt.close()

    # Figure 3: Box plots
    print("  [3/8] Creating box plots...")
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create dataframe for seaborn
    plot_data = pd.DataFrame({
        'NGL Mutations': list(heavy_data) + list(light_data),
        'Chain': ['Heavy']*len(heavy_data) + ['Light']*len(light_data)
    })

    sns.boxplot(data=plot_data, x='Chain', y='NGL Mutations', ax=ax, palette=['blue', 'orange'])
    ax.set_title('NGL Mutation Distribution: Box Plot Comparison', fontsize=16)
    ax.set_ylabel('Number of NGL Mutations', fontsize=14)
    ax.set_xlabel('Chain Type', fontsize=14)
    ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    fig_path = os.path.join(output_dir, "ngl_boxplot.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"    Saved: {fig_path}")
    plt.close()

    # Figure 4: Summary statistics table
    print("  [4/8] Creating summary statistics figure...")
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')

    # Create table data
    table_data = [
        ['Metric', 'Heavy Chain', 'Light Chain'],
        ['─'*30, '─'*20, '─'*20],
        ['Total Sequences', f"{heavy_stats['count']:,}", f"{light_stats['count']:,}"],
        ['', '', ''],
        ['NGL Mutations:', '', ''],
        ['  Mean', f"{heavy_stats['ngl_mean']:.2f}", f"{light_stats['ngl_mean']:.2f}"],
        ['  Median', f"{heavy_stats['ngl_50']:.2f}", f"{light_stats['ngl_50']:.2f}"],
        ['  Std Dev', f"{heavy_stats['ngl_std']:.2f}", f"{light_stats['ngl_std']:.2f}"],
        ['  Min', f"{heavy_stats['ngl_min']:.0f}", f"{light_stats['ngl_min']:.0f}"],
        ['  Max', f"{heavy_stats['ngl_max']:.0f}", f"{light_stats['ngl_max']:.0f}"],
        ['  Q1 (25%)', f"{heavy_stats['ngl_25']:.2f}", f"{light_stats['ngl_25']:.2f}"],
        ['  Q3 (75%)', f"{heavy_stats['ngl_75']:.2f}", f"{light_stats['ngl_75']:.2f}"],
        ['  P90', f"{heavy_stats['ngl_p90']:.2f}", f"{light_stats['ngl_p90']:.2f}"],
        ['  P95', f"{heavy_stats['ngl_p95']:.2f}", f"{light_stats['ngl_p95']:.2f}"],
        ['  P99', f"{heavy_stats['ngl_p99']:.2f}", f"{light_stats['ngl_p99']:.2f}"],
        ['', '', ''],
        ['Zero NGL Mutations', f"{heavy_stats['zero_ngl_count']:,} ({heavy_stats['zero_ngl_pct']:.2f}%)",
         f"{light_stats['zero_ngl_count']:,} ({light_stats['zero_ngl_pct']:.2f}%)"],
        ['Above P90', f"{heavy_stats['above_p90_count']:,} ({heavy_stats['above_p90_pct']:.2f}%)",
         f"{light_stats['above_p90_count']:,} ({light_stats['above_p90_pct']:.2f}%)"],
    ]

    table = ax.table(cellText=table_data, cellLoc='left', loc='center',
                     colWidths=[0.4, 0.3, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)

    # Style header row
    for i in range(3):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Style data rows
    for i in range(2, len(table_data)):
        for j in range(3):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#E7E6E6')

    ax.set_title('Unpaired OAS: Summary Statistics', fontsize=18, weight='bold', pad=20)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, "summary_statistics.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"    Saved: {fig_path}")
    plt.close()


def plot_btype_piechart(heavy_stats, light_stats, output_dir):
    """
    Create pie charts showing B-type distribution with external legends.
    Refined style: No text on pie, detailed legend on the side.
    """
    print("  [5/8] Creating B-type pie charts (Refined Style)...")
    
    # Increase figure size to accommodate the external legend (width, height)
    fig, axes = plt.subplots(1, 2, figsize=(22, 10))

    chains = [
        ('Heavy Chain', heavy_stats, axes[0]), 
        ('Light Chain', light_stats, axes[1])
    ]

    # Set color palette (using tab20 as there might be many categories)
    colors = sns.color_palette('tab20')

    for title, stats, ax in chains:
        if 'btype_counts' in stats and len(stats['btype_counts']) > 0:
            counts = stats['btype_counts']
            total = counts.sum()
            
            # Create formatted labels for the legend: "Name (Count | Percentage%)"
            # Example: Naive-B-Cells (660,712 | 35.0%)
            labels = [
                f"{name} ({val:,} | {val/total*100:.1f}%)" 
                for name, val in zip(counts.index, counts.values)
            ]
            
            # Draw the pie chart
            # Note: autopct and labels are removed to keep the chart clean (no text inside)
            wedges, _ = ax.pie(
                counts.values, 
                startangle=90, 
                counterclock=False, # Clockwise
                colors=colors[:len(counts)]
            )
            
            ax.set_title(f'{title} Distribution', fontsize=18, pad=20)
            
            # Add the legend to the right side
            # bbox_to_anchor places it outside the plot area (center right)
            ax.legend(
                wedges, 
                labels,
                title="BTypes",
                loc="center left",
                bbox_to_anchor=(1, 0, 0.5, 1),
                fontsize=11
            )
        else:
            # Handle empty data case
            ax.text(0.5, 0.5, 'No B-Type data', ha='center', va='center')
            ax.set_title(f'{title} Distribution', fontsize=18)
            ax.axis('off')

    # Adjust layout to prevent clipping of the legend
    plt.tight_layout()
    
    # Increase spacing between subplots to ensure legends don't overlap with the other chart
    plt.subplots_adjust(wspace=0.8)

    fig_path = os.path.join(output_dir, "btype_piechart.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"    Saved: {fig_path}")
    plt.close()

import matplotlib.patches as patches

def plot_mutational_regimes(ddf_heavy, ddf_light, target_btype, output_dir):
    """
    Create pie charts for mutational regimes with specific P90 thresholds.
    Style: Viridis reverse colormap, broken line at threshold, separate legend.
    
    Args:
        ddf_heavy: Heavy chain Dask DataFrame
        ddf_light: Light chain Dask DataFrame
        target_btype: The BType to filter and plot (e.g., 'Memory-B-Cells', 'Unsorted-B-Cells')
        output_dir: Output directory
    """
    print(f"  [>] Creating Mutational Regimes Pie Chart for: {target_btype}...")

    # Hardcoded Naive P90 thresholds as requested
    THRESHOLDS = {'Heavy': 3, 'Light': 2}
    
    # Setup plot: 2 rows (Heavy, Light), 1 column
    fig, axes = plt.subplots(2, 1, figsize=(12, 14))
    
    chains = [('Heavy', ddf_heavy, axes[0]), ('Light', ddf_light, axes[1])]
    
    # 21 colors for 0 to 20+ mutations (Viridis Reversed)
    cmap = plt.get_cmap('viridis_r')
    colors = [cmap(i/20) for i in range(21)]
    
    # Legend labels
    legend_labels = [f"{i} muts" for i in range(20)] + ["≥20 muts"]

    for chain_name, ddf, ax in chains:
        # 1. Filter data by BType
        # We use .compute() on value_counts which is memory efficient
        if 'BType' not in ddf.columns:
            print(f"    WARNING: 'BType' column missing in {chain_name} chain.")
            continue
            
        # Get counts for specific BType
        # Filter first, then count to avoid loading unnecessary data
        ddf_filtered = ddf[ddf['BType'] == target_btype]
        mutation_counts = ddf_filtered['num_ngl_muts'].value_counts().compute()
        
        if len(mutation_counts) == 0:
            ax.text(0.5, 0.5, f"No data for {target_btype}", ha='center')
            continue

        # 2. Binning (0 to 19, and 20+)
        binned_counts = np.zeros(21)
        for muts, count in mutation_counts.items():
            if muts >= 20:
                binned_counts[20] += count
            else:
                binned_counts[int(muts)] += count
                
        total_seqs = np.sum(binned_counts)
        if total_seqs == 0:
            continue

        # 3. Prepare Pie Data
        # We need to sort indices 0..20 to ensure colors map correctly
        pie_data = binned_counts
        
        # 4. Draw Pie Chart
        wedges, _ = ax.pie(
            pie_data, 
            startangle=90, 
            counterclock=False, # Clockwise
            colors=colors,
            wedgeprops={'linewidth': 0.5, 'edgecolor': 'white'} # Thin white lines between slices
        )
        
        # 5. Add Threshold Line (Dashed)
        # Calculate angle for the threshold
        threshold = THRESHOLDS[chain_name]
        
        # Calculate fraction of data UP TO and INCLUDING the threshold
        # Indices 0, 1, ..., threshold correspond to counts <= threshold
        upto_thresh_count = np.sum(binned_counts[:threshold + 1])
        fraction_upto = upto_thresh_count / total_seqs
        
        # Calculate percentage ABOVE threshold
        pct_above = (1 - fraction_upto) * 100
        
        # Angle in degrees (Standard position)
        # 90 degrees is start. Clockwise means subtracting.
        theta = 90 - (fraction_upto * 360)
        
        # Convert to radians for plotting line
        theta_rad = np.deg2rad(theta)
        
        # Draw dashed line from center to edge
        r = 1.05 # Slightly longer than radius 1
        x = r * np.cos(theta_rad)
        y = r * np.sin(theta_rad)
        
        ax.plot([0, x], [0, y], color='black', linestyle='--', linewidth=2, zorder=10)
        
        # Add annotation for threshold line
        ax.text(
            x * 1.15, y * 1.15, 
            f"Naive p90\n≈{threshold}", 
            ha='center', va='center', 
            fontsize=12, fontweight='bold'
        )
        
        # 6. Add Percentage Box (>muts p90)
        # Position at bottom right relative to pie
        bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9)
        ax.text(
            0.5, -1.1, 
            f">muts p90: {pct_above:.2f}%", 
            ha='center', va='center', 
            size=14, 
            bbox=bbox_props
        )

        # 7. Titles and Legend
        ax.set_title(f"num_ngl_muts_{'hc' if chain_name=='Heavy' else 'lc'} ({target_btype})", fontsize=16, loc='left')
        
        # Legend styling (3 columns, right side)
        ax.legend(
            wedges, legend_labels,
            title="Mutational Regimes",
            title_fontsize=14,
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),
            ncol=3,
            fontsize=10,
            frameon=False
        )
        
    plt.tight_layout()
    plt.subplots_adjust(right=0.75) # Make room for legend
    
    filename = f"mutational_regimes_{target_btype.replace(' ', '_')}.png"
    fig_path = os.path.join(output_dir, filename)
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"    Saved: {fig_path}")
    plt.close()


def plot_naive_ngl_distribution(heavy_naive, light_naive, heavy_stats, light_stats, output_dir):
    """
    Create histograms showing NGL distribution for naive B-cells only.

    Args:
        heavy_naive: pandas Series with heavy chain naive B-cell NGL values
        light_naive: pandas Series with light chain naive B-cell NGL values
        heavy_stats: dict with heavy chain statistics
        light_stats: dict with light chain statistics
        output_dir: directory to save plots
    """
    print("  [6/8] Creating naive B-cell NGL histograms...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Heavy chain
    if len(heavy_naive) > 0:
        axes[0].hist(heavy_naive, bins=100, alpha=0.7, color='blue', edgecolor='black')
        axes[0].axvline(heavy_stats['ngl_p90'], color='red', linestyle='--', linewidth=2,
                        label=f"P90 = {heavy_stats['ngl_p90']:.2f}")
        axes[0].axvline(heavy_stats['ngl_p95'], color='orange', linestyle='--', linewidth=2,
                        label=f"P95 = {heavy_stats['ngl_p95']:.2f}")
        axes[0].axvline(heavy_stats['ngl_p99'], color='purple', linestyle='--', linewidth=2,
                        label=f"P99 = {heavy_stats['ngl_p99']:.2f}")
        axes[0].set_xlabel('Number of NGL Mutations', fontsize=14)
        axes[0].set_ylabel('Frequency', fontsize=14)
        axes[0].set_title(f'Heavy Chain: Naive B-Cell NGL Distribution\\n(n={len(heavy_naive):,})', fontsize=16)
        axes[0].legend(fontsize=12)
        axes[0].grid(alpha=0.3)
    else:
        axes[0].text(0.5, 0.5, 'No Naive B-cell data', ha='center', va='center', transform=axes[0].transAxes)
        axes[0].set_title('Heavy Chain: Naive B-Cell NGL Distribution', fontsize=16)

    # Light chain
    if len(light_naive) > 0:
        axes[1].hist(light_naive, bins=100, alpha=0.7, color='orange', edgecolor='black')
        axes[1].axvline(light_stats['ngl_p90'], color='red', linestyle='--', linewidth=2,
                        label=f"P90 = {light_stats['ngl_p90']:.2f}")
        axes[1].axvline(light_stats['ngl_p95'], color='orange', linestyle='--', linewidth=2,
                        label=f"P95 = {light_stats['ngl_p95']:.2f}")
        axes[1].axvline(light_stats['ngl_p99'], color='purple', linestyle='--', linewidth=2,
                        label=f"P99 = {light_stats['ngl_p99']:.2f}")
        axes[1].set_xlabel('Number of NGL Mutations', fontsize=14)
        axes[1].set_ylabel('Frequency', fontsize=14)
        axes[1].set_title(f'Light Chain: Naive B-Cell NGL Distribution\\n(n={len(light_naive):,})', fontsize=16)
        axes[1].legend(fontsize=12)
        axes[1].grid(alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, 'No Naive B-cell data', ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title('Light Chain: Naive B-Cell NGL Distribution', fontsize=16)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, "naive_ngl_distribution.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"    Saved: {fig_path}")
    plt.close()


def plot_btype_average_ngl(heavy_stats, light_stats, output_dir):
    """
    Create bar graph showing average NGL mutations by B-type.

    Args:
        heavy_stats: dict with heavy chain statistics
        light_stats: dict with light chain statistics
        output_dir: directory to save plots
    """
    print("  [7/8] Creating B-type average NGL bar graphs...")

    if 'btype_means' not in heavy_stats or 'btype_means' not in light_stats:
        print("    WARNING: No B-type means data available, skipping")
        return

    # Get all B-types
    btypes = sorted(set(list(heavy_stats['btype_means'].keys()) + list(light_stats['btype_means'].keys())))

    if len(btypes) == 0:
        print("    WARNING: No B-types found, skipping")
        return

    heavy_means = [heavy_stats['btype_means'].get(bt, 0) for bt in btypes]
    light_means = [light_stats['btype_means'].get(bt, 0) for bt in btypes]

    fig, ax = plt.subplots(figsize=(12, 8))

    x = np.arange(len(btypes))
    width = 0.35

    bars1 = ax.bar(x - width/2, heavy_means, width, label='Heavy Chain', color='blue', alpha=0.7)
    bars2 = ax.bar(x + width/2, light_means, width, label='Light Chain', color='orange', alpha=0.7)

    ax.set_xlabel('B-Cell Type', fontsize=14)
    ax.set_ylabel('Average NGL Mutations', fontsize=14)
    ax.set_title('Average NGL Mutations by B-Cell Type', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(btypes, rotation=45, ha='right')
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    fig_path = os.path.join(output_dir, "btype_average_ngl.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"    Saved: {fig_path}")
    plt.close()


def save_thresholds(heavy_stats, light_stats, output_dir):
    """
    Save P90 thresholds to file for downstream use.

    Args:
        heavy_stats: dict with heavy chain statistics
        light_stats: dict with light chain statistics
        output_dir: directory to save file
    """
    print(f"\n{'='*70}")
    print("Saving Thresholds")
    print('='*70)

    threshold_file = os.path.join(output_dir, "p90_thresholds.npz")

    # Prepare data to save (including unsorted if available)
    save_data = {
        'hc_p90': heavy_stats['ngl_p90'],
        'lc_p90': light_stats['ngl_p90'],
        'hc_p95': heavy_stats['ngl_p95'],
        'lc_p95': light_stats['ngl_p95'],
        'hc_p99': heavy_stats['ngl_p99'],
        'lc_p99': light_stats['ngl_p99'],
        'hc_mean': heavy_stats['ngl_mean'],
        'lc_mean': light_stats['ngl_mean'],
        'hc_median': heavy_stats['ngl_50'],
        'lc_median': light_stats['ngl_50']
    }

    # Add unsorted thresholds if available
    if 'unsorted_p90' in heavy_stats:
        save_data['hc_unsorted_p90'] = heavy_stats['unsorted_p90']
        save_data['hc_unsorted_p95'] = heavy_stats['unsorted_p95']
        save_data['hc_unsorted_p99'] = heavy_stats['unsorted_p99']

    if 'unsorted_p90' in light_stats:
        save_data['lc_unsorted_p90'] = light_stats['unsorted_p90']
        save_data['lc_unsorted_p95'] = light_stats['unsorted_p95']
        save_data['lc_unsorted_p99'] = light_stats['unsorted_p99']

    np.savez(threshold_file, **save_data)
    print(f"  Saved: {threshold_file}")

    # Also save as text file for easy reading
    txt_file = os.path.join(output_dir, "p90_thresholds.txt")
    with open(txt_file, 'w') as f:
        f.write("Unpaired OAS Thresholds\n")
        f.write("="*50 + "\n\n")

        f.write("NAIVE B-CELL THRESHOLDS (Primary)\n")
        f.write("-"*50 + "\n")
        f.write(f"Heavy Chain P90: {heavy_stats['ngl_p90']:.2f}\n")
        f.write(f"Light Chain P90: {light_stats['ngl_p90']:.2f}\n")
        f.write(f"\nHeavy Chain P95: {heavy_stats['ngl_p95']:.2f}\n")
        f.write(f"Light Chain P95: {light_stats['ngl_p95']:.2f}\n")
        f.write(f"\nHeavy Chain P99: {heavy_stats['ngl_p99']:.2f}\n")
        f.write(f"Light Chain P99: {light_stats['ngl_p99']:.2f}\n")

        # Add unsorted thresholds if available
        if 'unsorted_p90' in heavy_stats or 'unsorted_p90' in light_stats:
            f.write("\n\nUNSORTED B-CELL THRESHOLDS (Additional)\n")
            f.write("-"*50 + "\n")

            if 'unsorted_p90' in heavy_stats:
                f.write(f"Heavy Chain P90: {heavy_stats['unsorted_p90']:.2f}\n")
                f.write(f"Heavy Chain P95: {heavy_stats['unsorted_p95']:.2f}\n")
                f.write(f"Heavy Chain P99: {heavy_stats['unsorted_p99']:.2f}\n")
            else:
                f.write("Heavy Chain: No Unsorted B-cells found\n")

            f.write("\n")

            if 'unsorted_p90' in light_stats:
                f.write(f"Light Chain P90: {light_stats['unsorted_p90']:.2f}\n")
                f.write(f"Light Chain P95: {light_stats['unsorted_p95']:.2f}\n")
                f.write(f"Light Chain P99: {light_stats['unsorted_p99']:.2f}\n")
            else:
                f.write("Light Chain: No Unsorted B-cells found\n")

        f.write("\n" + "="*50 + "\n")
        f.write("Note: Naive B-cell thresholds are used for filtering.\n")
        f.write("      Unsorted thresholds are provided for reference.\n")
    print(f"  Saved: {txt_file}")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    # Set up stdout logging
    log_file = os.path.join(OUTPUT_DIR, "statistics_log.txt")
    tee = TeeOutput(log_file)
    original_stdout = sys.stdout
    sys.stdout = tee

    try:
        print("="*70)
        print("UNPAIRED OAS VISUALIZATION")
        print("="*70)
        print(f"\nLogging output to: {log_file}")

        # Check if data exists
        if not os.path.exists(STAGE_2_HEAVY_DIR):
            print(f"\nERROR: Heavy chain data not found: {STAGE_2_HEAVY_DIR}")
            print("Run the main processing script first!")
            return

        if not os.path.exists(STAGE_2_LIGHT_DIR):
            print(f"\nERROR: Light chain data not found: {STAGE_2_LIGHT_DIR}")
            print("Run the main processing script first!")
            return

        # Load heavy chain data
        print(f"\nLoading heavy chain data from {STAGE_2_HEAVY_DIR}...")
        ddf_heavy = dd.read_parquet(STAGE_2_HEAVY_DIR)
        heavy_stats = calculate_statistics(ddf_heavy, "Heavy")

        # Load light chain data
        print(f"\nLoading light chain data from {STAGE_2_LIGHT_DIR}...")
        ddf_light = dd.read_parquet(STAGE_2_LIGHT_DIR)
        light_stats = calculate_statistics(ddf_light, "Light")

        # Sample data for histograms
        print(f"\n{'='*70}")
        print("Sampling Data for Histograms")
        print('='*70)
        print("  Heavy chain:")
        heavy_sample = sample_for_histogram(ddf_heavy, sample_size=1000000)
        print("  Light chain:")
        light_sample = sample_for_histogram(ddf_light, sample_size=1000000)
        print("  Heavy chain (naive B-cells):")
        heavy_naive = sample_naive_for_histogram(ddf_heavy, sample_size=1000000)
        print("  Light chain (naive B-cells):")
        light_naive = sample_naive_for_histogram(ddf_light, sample_size=1000000)

        # Create visualizations
        plot_ngl_distribution(heavy_sample, light_sample, heavy_stats, light_stats, OUTPUT_DIR)
        plot_btype_piechart(heavy_stats, light_stats, OUTPUT_DIR)
        plot_naive_ngl_distribution(heavy_naive, light_naive, heavy_stats, light_stats, OUTPUT_DIR)
        plot_btype_average_ngl(heavy_stats, light_stats, OUTPUT_DIR)

        print(f"\n{'='*70}")
        print("Generating Mutational Regime Pie Charts")
        print('='*70)
        
        # 1. Unsorted B-Cells
        plot_mutational_regimes(ddf_heavy, ddf_light, 'Unsorted-B-Cells', OUTPUT_DIR)
        
        # 2. Memory B-Cells
        plot_mutational_regimes(ddf_heavy, ddf_light, 'Memory-B-Cells', OUTPUT_DIR)

        # Save thresholds
        save_thresholds(heavy_stats, light_stats, OUTPUT_DIR)

        print(f"\n{'='*70}")
        print("VISUALIZATION COMPLETE")
        print('='*70)
        print(f"  All figures saved to: {OUTPUT_DIR}")
        print(f"  Thresholds saved to: {os.path.join(OUTPUT_DIR, 'p90_thresholds.npz')}")
        print(f"  Statistics log: {log_file}")

    finally:
        # Restore stdout and close log file
        sys.stdout = original_stdout
        tee.close()
        print(f"\nLog saved to: {log_file}")


if __name__ == "__main__":
    main()
