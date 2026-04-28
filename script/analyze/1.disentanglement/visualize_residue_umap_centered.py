#!/usr/bin/env python
"""
Visualize GL vs NGL residue embeddings using UMAP with Mean Centering.

This script removes amino acid identity bias by subtracting the per-amino-acid
mean vector from each embedding, allowing the UMAP visualization to focus on
the GL vs NGL distinction rather than amino acid clustering.

Standard Mode (2x3 grid):
- Row 1: evo-ab, esm2-35m, esm2-650m
- Row 2: ablang2, antiberty, sapiens

Ablation Mode (2x2 grid):
- Row 1: PRISM Full, Ablation 1 (Multihead + No Pretrain)
- Row 2: Ablation 2 (Simple + Pretrain), Ablation 3 (Simple + No Pretrain)

Usage:
    # Standard mode (6 models, 2x3 grid)
    python visualize_residue_umap_centered.py

    # Ablation mode (4 models, 2x2 grid)
    python visualize_residue_umap_centered.py --ablation_mode \
        --input_file data/unpaired_OAS/linear_probe_data/test_linear_ablations.pkl
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from tqdm import tqdm


# =============================================================================
# FONT CONFIGURATION - Unified font sizes for publication-quality figures
# =============================================================================
FONT_CONFIG = {
    # Axis labels (x label, y label)
    "axis_label": {"fontsize": 25, "fontweight": "bold"},
    # Tick labels (x tick, y tick)
    "tick_label": {"fontsize": 15, "fontweight": "normal"},
    # Legend
    "legend": {"fontsize": 20},
    # Title/subtitle
    "title": {"fontsize": 25, "fontweight": "bold"},
    # Annotations, stats boxes, other text
    "annotation": {"fontsize": 20, "fontweight": "normal"},
}


# Model configurations: (display_name, file_model_name, embed_col_base)
MODEL_CONFIGS = [
    # Row 1
    ("PRISM", "evo_ab", "embed_evo_ab"),
    ("ESM2-35M", "esm2_35m", "embed_esm2_35m"),
    ("ESM2-650M", "esm2_650m", "embed_esm2_650m"),
    # Row 2
    ("AbLang2", "ablang2", "embed_ablang2"),
    ("AntiBERTy", "antiberty", "embed_antiberty"),
    ("Sapiens", "sapiens", "embed_sapiens"),
]

# Ablation model configurations for 2x2 grid: (display_name, embed_col_base)
# Ablation Study Design (2×2 factorial):
# ┌─────────────────────┬───────────────────┬───────────────────┐
# │                     │ Multihead (Full)  │ Simple LM Head    │
# ├─────────────────────┼───────────────────┼───────────────────┤
# │ With Pretraining    │ PRISM Full (best) │ Ablation 2        │
# ├─────────────────────┼───────────────────┼───────────────────┤
# │ No Pretraining      │ Ablation 1        │ Ablation 3        │
# └─────────────────────┴───────────────────┴───────────────────┘
ABLATION_MODEL_CONFIGS = [
    # Row 1
    ("PRISM Full\n(Multihead + Pretrain)", "embed_evo_ab"),
    ("Ablation 1\n(Multihead + No Pretrain)", "embed_evo_ab_ablation1"),
    # Row 2
    ("Ablation 2\n(Simple + Pretrain)", "embed_evo_ab_ablation2"),
    ("Ablation 3\n(Simple + No Pretrain)", "embed_evo_ab_ablation3"),
]

# Base path for embedding files
DATA_BASE_PATH = Path(__file__).parent.parent.parent / "data" / "unpaired_OAS" / "linear_probe_data"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize GL vs NGL embeddings using UMAP with mean centering"
    )

    # Ablation mode arguments
    parser.add_argument(
        "--ablation_mode",
        action="store_true",
        help="Enable ablation mode: 2x2 grid comparing PRISM Full vs 3 ablation models",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="Path to single pickle file with all ablation embeddings (required for ablation mode). "
             "Should contain columns like embed_evo_ab_h, embed_evo_ab_ablation1_h, etc.",
    )
    parser.add_argument(
        "--original_embed_file",
        type=str,
        default=None,
        help="Path to pickle file with original PRISM Full embeddings (embed_evo_ab_h/l). "
             "If not provided, will try to find test_linear_evo_ab.pkl in the same directory.",
    )

    # Sampling and UMAP parameters
    parser.add_argument(
        "--n_samples",
        type=int,
        default=200,
        help="Number of sequences to sample (default: 200)",
    )
    parser.add_argument(
        "--n_neighbors",
        type=int,
        default=100,
        help="UMAP n_neighbors parameter (default: 100)",
    )
    parser.add_argument(
        "--min_dist",
        type=float,
        default=0.1,
        help="UMAP min_dist parameter (default: 0.1)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(Path(__file__).parent.parent.parent / "img" / "2.gl-ngl_calculation"),
        help="Output directory for the figure",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random state for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--chain",
        type=str,
        default="both",
        choices=["heavy", "light", "both"],
        help="Which chain(s) to analyze: heavy, light, or both (default: both)",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=32,
        help="Number of parallel jobs for UMAP (default: 32)",
    )
    parser.add_argument(
        "--knn_k",
        type=int,
        default=5,
        help="k for k-NN classifier evaluation (default: 5)",
    )
    parser.add_argument(
        "--skip_knn",
        action="store_true",
        help="Skip k-NN classifier accuracy calculation (only use ARI)",
    )
    return parser.parse_args()


def get_ngl_positions_from_mut_codes(mut_codes) -> set:
    """
    Extract NGL mutation positions from mutation codes (e.g., 'A10N', 'G25L').
    Returns a set of positions where NGL mutations occur (0-indexed).
    """
    if pd.isna(mut_codes) or mut_codes is None:
        return set()

    ngl_positions = set()

    if isinstance(mut_codes, str):
        mutations = mut_codes.replace(';', ',').split(',')
    elif isinstance(mut_codes, (list, tuple)):
        mutations = mut_codes
    else:
        return set()

    for mut in mutations:
        mut = str(mut).strip()
        if not mut:
            continue

        if len(mut) >= 3:
            try:
                position_str = mut[1:-1]
                position = int(position_str) - 1  # Convert to 0-indexed
                ngl_positions.add(position)
            except (ValueError, IndexError):
                continue

    return ngl_positions


def collect_residue_embeddings_with_aa(
    df: pd.DataFrame,
    embed_col: str,
    n_samples: int,
    chain: str = "both",
    random_state: int = 42,
) -> Tuple[List[np.ndarray], List[int], List[str], int, int]:
    """
    Collect embeddings for residues with amino acid identity.

    Returns:
        Tuple of (embedding vectors, binary labels (0=GL, 1=NGL),
                  amino acid characters, gl_count, ngl_count)
    """
    # Determine chain configurations
    chain_configs = []
    if chain in ["heavy", "both"]:
        chain_configs.append(("heavy", "HEAVY_CHAIN_AA_SEQUENCE", f"{embed_col}_h", "hc_mut_codes"))
    if chain in ["light", "both"]:
        chain_configs.append(("light", "LIGHT_CHAIN_AA_SEQUENCE", f"{embed_col}_l", "lc_mut_codes"))

    # Check if required columns exist
    for chain_name, seq_col, embed_col_name, mut_codes_col in chain_configs:
        if embed_col_name not in df.columns:
            return [], [], [], 0, 0

    gl_vectors = []
    ngl_vectors = []
    gl_residues = []
    ngl_residues = []

    # Sample sequences
    if len(df) > n_samples:
        sampled_df = df.sample(n=n_samples, random_state=random_state)
    else:
        sampled_df = df

    for idx, row in sampled_df.iterrows():
        for chain_name, seq_col_name, embed_col_name, mut_codes_col in chain_configs:
            seq = row.get(seq_col_name)
            emb = row.get(embed_col_name)
            mut_codes = row.get(mut_codes_col, None)

            if pd.isna(seq) or emb is None or (hasattr(emb, '__len__') and len(emb) == 0):
                continue

            # Convert embedding to numpy
            if hasattr(emb, "numpy"):
                emb = emb.numpy()
            elif not isinstance(emb, np.ndarray):
                emb = np.array(emb)

            if len(seq) != len(emb):
                continue

            ngl_positions = get_ngl_positions_from_mut_codes(mut_codes)

            for pos, (char, vector) in enumerate(zip(seq, emb)):
                if not char.isalpha():
                    continue

                aa = char.upper()  # Normalize to uppercase

                if pos in ngl_positions:
                    ngl_vectors.append(vector)
                    ngl_residues.append(aa)
                else:
                    gl_vectors.append(vector)
                    gl_residues.append(aa)

    # Downsample GL to match NGL count (1:1 ratio)
    if len(gl_vectors) > len(ngl_vectors) and len(ngl_vectors) > 0:
        np.random.seed(random_state)
        indices = np.random.choice(len(gl_vectors), size=len(ngl_vectors), replace=False)
        gl_vectors = [gl_vectors[i] for i in indices]
        gl_residues = [gl_residues[i] for i in indices]

    # Combine
    all_vectors = gl_vectors + ngl_vectors
    all_labels = [0] * len(gl_vectors) + [1] * len(ngl_vectors)
    all_residues = gl_residues + ngl_residues

    return all_vectors, all_labels, all_residues, len(gl_vectors), len(ngl_vectors)


def mean_center_embeddings(
    vectors: List[np.ndarray],
    residues: List[str],
) -> np.ndarray:
    """
    Perform mean centering: subtract per-amino-acid mean from each embedding.

    This removes the amino acid identity signal, allowing UMAP to focus on
    other variations (like GL vs NGL).

    Args:
        vectors: List of embedding vectors
        residues: List of amino acid characters (same length as vectors)

    Returns:
        Centered embedding matrix (N x D)
    """
    X = np.stack(vectors, axis=0)  # (N, D)
    residues_arr = np.array(residues)

    # Get unique amino acids
    unique_aas = np.unique(residues_arr)

    # Calculate mean for each amino acid and subtract
    X_centered = X.copy()

    for aa in unique_aas:
        mask = residues_arr == aa
        if mask.sum() > 0:
            aa_mean = X[mask].mean(axis=0)
            X_centered[mask] -= aa_mean

    return X_centered


def run_umap(
    X: np.ndarray,
    n_neighbors: int,
    min_dist: float,
    n_jobs: int = -1,
) -> np.ndarray:
    """
    Run UMAP dimensionality reduction.
    """
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=None,  # None for parallelization
        metric='cosine',
        n_jobs=n_jobs,
        verbose=False,
    )
    X_umap = reducer.fit_transform(X)
    return X_umap


def calculate_ari_from_umap(X_umap: np.ndarray, true_labels: List[int]) -> float:
    """
    Calculate Adjusted Rand Index by clustering UMAP output with K-means.
    """
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    predicted_labels = kmeans.fit_predict(X_umap)
    ari = adjusted_rand_score(true_labels, predicted_labels)
    return ari


def calculate_knn_accuracy(X_umap: np.ndarray, labels: List[int], k: int = 5) -> float:
    """
    Calculate k-NN classifier accuracy using cross-validation on UMAP coordinates.

    This measures local separability: can we predict GL/NGL from nearby points?
    """
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_umap, labels, cv=5, scoring='accuracy')
    return scores.mean()


def create_2x3_visualization(
    results: Dict[str, Tuple[np.ndarray, List[int], float, Optional[float], int, int]],
    output_path: Path,
    knn_k: int,
    skip_knn: bool = False,
) -> None:
    """
    Create a 2x3 grid visualization of UMAP results for all models.

    Args:
        results: Dict mapping model name to (X_umap, labels, ari, knn_acc, gl_count, ngl_count)
        output_path: Path to save the figure
        knn_k: k value used for k-NN
        skip_knn: If True, don't display k-NN accuracy in titles
    """
    print("Creating 2x3 visualization...")

    plt.style.use("seaborn-v0_8-whitegrid")
    # Widen figure to accommodate legend on the right
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=150)
    axes = axes.flatten()

    # Define colors (GL/NGL palette)
    gl_color = "#1CC454"   # Green for Germline
    ngl_color = "#C8327D"  # Magenta for Somatic

    # Model order for plotting
    model_order = [config[0] for config in MODEL_CONFIGS]

    # Track counts for the shared annotation
    first_gl_count = None
    first_ngl_count = None
    legend_handles = None

    for idx, model_name in enumerate(model_order):
        ax = axes[idx]

        if model_name not in results:
            ax.text(0.5, 0.5, f"{model_name}\nNo data available",
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=FONT_CONFIG["annotation"]["fontsize"],
                   fontweight=FONT_CONFIG["annotation"]["fontweight"],
                   color='gray')
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        X_umap, labels, ari, knn_acc, gl_count, ngl_count = results[model_name]

        # Store first valid counts for shared annotation
        if first_gl_count is None:
            first_gl_count = gl_count
            first_ngl_count = ngl_count

        # Create DataFrame for plotting
        plot_df = pd.DataFrame({
            "UMAP 1": X_umap[:, 0],
            "UMAP 2": X_umap[:, 1],
            "Type": ["Germline (GL)" if l == 0 else "Somatic (NGL)" for l in labels],
        })

        # Plot GL points first (background), then NGL on top
        gl_mask = plot_df["Type"] == "Germline (GL)"
        scatter_gl = ax.scatter(
            plot_df.loc[gl_mask, "UMAP 1"],
            plot_df.loc[gl_mask, "UMAP 2"],
            c=gl_color, alpha=0.6, s=10, label="Germline (GL)",
            edgecolor="none", marker="o"
        )
        scatter_ngl = ax.scatter(
            plot_df.loc[~gl_mask, "UMAP 1"],
            plot_df.loc[~gl_mask, "UMAP 2"],
            c=ngl_color, alpha=0.6, s=10, label="Somatic (NGL)",
            edgecolor="none", marker="o"
        )

        # Store handles for the shared legend (from first subplot)
        if legend_handles is None:
            legend_handles = [scatter_gl, scatter_ngl]

        # Title with model name and metrics
        if skip_knn or knn_acc is None:
            ax.set_title(f"{model_name}\nARI: {ari:.3f}",
                        fontsize=FONT_CONFIG["title"]["fontsize"],
                        fontweight=FONT_CONFIG["title"]["fontweight"], pad=10)
        else:
            ax.set_title(f"{model_name}\nARI: {ari:.3f} | {knn_k}-NN Acc: {knn_acc:.3f}",
                        fontsize=FONT_CONFIG["title"]["fontsize"],
                        fontweight=FONT_CONFIG["title"]["fontweight"], pad=10)

        # Reduced axis label size (fontsize 15, non-bold)
        ax.set_xlabel("UMAP 1", fontsize=15)
        ax.set_ylabel("UMAP 2", fontsize=15)

        # Set tick label font sizes
        ax.tick_params(axis='both', labelsize=FONT_CONFIG["tick_label"]["fontsize"])

    # Adjust layout to make room for legend on the right
    plt.tight_layout()
    fig.subplots_adjust(right=0.82)

    # Create a combined legend with color markers and sample counts in one box
    if legend_handles is not None and first_gl_count is not None:
        # Add legend with sample counts included in labels
        combined_labels = [
            f"Germline (GL)\nn = {first_gl_count}",
            f"Somatic (NGL)\nn = {first_ngl_count}"
        ]
        leg = fig.legend(
            handles=legend_handles,
            labels=combined_labels,
            loc='center right',
            fontsize=16,
            framealpha=0.95,
            bbox_to_anchor=(0.99, 0.5),
            labelspacing=1.2,  # Space between legend entries
            handletextpad=0.8,  # Space between marker and text
            borderpad=1.0,  # Padding inside legend box
            edgecolor='gray',
        )
        # Left-align the text in the legend
        for text in leg.get_texts():
            text.set_ha('left')
    # Save as PNG
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    # Save as SVG
    svg_path = str(output_path).replace('.png', '.svg')
    plt.savefig(svg_path, format='svg', bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"Figure saved to: {output_path}")
    print(f"Figure saved to: {svg_path}")


def create_2x2_visualization(
    results: Dict[str, Tuple[np.ndarray, List[int], float, Optional[float], int, int]],
    output_path: Path,
    knn_k: int,
    skip_knn: bool = False,
) -> None:
    """
    Create a 2x2 grid visualization of UMAP results for ablation models.

    Args:
        results: Dict mapping model name to (X_umap, labels, ari, knn_acc, gl_count, ngl_count)
        output_path: Path to save the figure
        knn_k: k value used for k-NN
        skip_knn: If True, don't display k-NN accuracy in titles
    """
    print("Creating 2x2 ablation visualization...")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=150)
    axes = axes.flatten()

    # Define colors (GL/NGL palette)
    gl_color = "#1CC454"   # Green for Germline
    ngl_color = "#C8327D"  # Magenta for Somatic

    # Model order for plotting (from ABLATION_MODEL_CONFIGS)
    model_order = [config[0] for config in ABLATION_MODEL_CONFIGS]

    # Track counts for the shared annotation
    first_gl_count = None
    first_ngl_count = None
    legend_handles = None

    for idx, model_name in enumerate(model_order):
        ax = axes[idx]

        if model_name not in results:
            ax.text(0.5, 0.5, f"{model_name}\nNo data available",
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=FONT_CONFIG["annotation"]["fontsize"],
                   fontweight=FONT_CONFIG["annotation"]["fontweight"],
                   color='gray')
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        X_umap, labels, ari, knn_acc, gl_count, ngl_count = results[model_name]

        # Store first valid counts for shared annotation
        if first_gl_count is None:
            first_gl_count = gl_count
            first_ngl_count = ngl_count

        # Create DataFrame for plotting
        plot_df = pd.DataFrame({
            "UMAP 1": X_umap[:, 0],
            "UMAP 2": X_umap[:, 1],
            "Type": ["Germline (GL)" if l == 0 else "Somatic (NGL)" for l in labels],
        })

        # Plot GL points first (background), then NGL on top
        gl_mask = plot_df["Type"] == "Germline (GL)"
        scatter_gl = ax.scatter(
            plot_df.loc[gl_mask, "UMAP 1"],
            plot_df.loc[gl_mask, "UMAP 2"],
            c=gl_color, alpha=0.6, s=15, label="Germline (GL)",
            edgecolor="none", marker="o"
        )
        scatter_ngl = ax.scatter(
            plot_df.loc[~gl_mask, "UMAP 1"],
            plot_df.loc[~gl_mask, "UMAP 2"],
            c=ngl_color, alpha=0.6, s=15, label="Somatic (NGL)",
            edgecolor="none", marker="o"
        )

        # Store handles for the shared legend (from first subplot)
        if legend_handles is None:
            legend_handles = [scatter_gl, scatter_ngl]

        # Title with model name and metrics (smaller font for multiline model names)
        if skip_knn or knn_acc is None:
            ax.set_title(f"{model_name}\nARI: {ari:.3f}",
                        fontsize=20,
                        fontweight=FONT_CONFIG["title"]["fontweight"], pad=10)
        else:
            ax.set_title(f"{model_name}\nARI: {ari:.3f} | {knn_k}-NN: {knn_acc:.3f}",
                        fontsize=20,
                        fontweight=FONT_CONFIG["title"]["fontweight"], pad=10)

        # Axis labels
        ax.set_xlabel("UMAP 1", fontsize=16)
        ax.set_ylabel("UMAP 2", fontsize=16)

        # Set tick label font sizes
        ax.tick_params(axis='both', labelsize=FONT_CONFIG["tick_label"]["fontsize"])

    # Adjust layout to make room for legend on the right
    plt.tight_layout()
    fig.subplots_adjust(right=0.85)

    # Create a combined legend with color markers and sample counts in one box
    if legend_handles is not None and first_gl_count is not None:
        combined_labels = [
            f"Germline (GL)\nn = {first_gl_count}",
            f"Somatic (NGL)\nn = {first_ngl_count}"
        ]
        leg = fig.legend(
            handles=legend_handles,
            labels=combined_labels,
            loc='center right',
            fontsize=16,
            framealpha=0.95,
            bbox_to_anchor=(0.98, 0.5),
            labelspacing=1.2,
            handletextpad=0.8,
            borderpad=1.0,
            edgecolor='gray',
        )
        for text in leg.get_texts():
            text.set_ha('left')

    # Save as PNG
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    # Save as SVG
    svg_path = str(output_path).replace('.png', '.svg')
    plt.savefig(svg_path, format='svg', bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"Figure saved to: {output_path}")
    print(f"Figure saved to: {svg_path}")


def process_single_model(
    df: pd.DataFrame,
    model_name: str,
    embed_col_base: str,
    args,
) -> Optional[Tuple[np.ndarray, List[int], float, Optional[float], int, int]]:
    """
    Process a single model's embeddings and compute UMAP + metrics.

    Args:
        df: DataFrame with embeddings
        model_name: Display name for the model
        embed_col_base: Base name for embedding columns (e.g., 'embed_evo_ab')
        args: Command line arguments

    Returns:
        Tuple of (X_umap, labels, ari, knn_acc, gl_count, ngl_count) or None if failed
    """
    print(f"\n{'='*60}")
    print(f"Processing {model_name}")
    print(f"{'='*60}")

    # Collect embeddings with amino acid identity
    vectors, labels, residues, gl_count, ngl_count = collect_residue_embeddings_with_aa(
        df=df,
        embed_col=embed_col_base,
        n_samples=args.n_samples,
        chain=args.chain,
        random_state=args.random_state,
    )

    if len(vectors) == 0:
        print(f"  No embeddings found for {model_name}. Skipping.")
        return None

    if len(set(labels)) < 2:
        print(f"  Only one class found for {model_name}. Skipping.")
        return None

    print(f"  Collected {gl_count} GL and {ngl_count} NGL residues (balanced 1:1)")

    # Count amino acid distribution
    unique_aas, aa_counts = np.unique(residues, return_counts=True)
    print(f"  Amino acids present: {len(unique_aas)} types")

    # Step 1: Mean centering (remove amino acid identity bias)
    print(f"  Applying mean centering to remove amino acid identity bias...")
    X_centered = mean_center_embeddings(vectors, residues)
    print(f"  Centered embedding shape: {X_centered.shape}")

    # Step 2: Run UMAP on centered embeddings
    print(f"  Running UMAP (n_neighbors={args.n_neighbors}, min_dist={args.min_dist}, n_jobs={args.n_jobs})...")
    X_umap = run_umap(
        X=X_centered,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        n_jobs=args.n_jobs,
    )

    # Step 3: Calculate metrics
    ari = calculate_ari_from_umap(X_umap, labels)
    print(f"  Adjusted Rand Index: {ari:.4f}")

    if args.skip_knn:
        knn_acc = None
    else:
        knn_acc = calculate_knn_accuracy(X_umap, labels, k=args.knn_k)
        print(f"  {args.knn_k}-NN Classifier Accuracy: {knn_acc:.4f}")

    return (X_umap, labels, ari, knn_acc, gl_count, ngl_count)


def main():
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # =========================================================================
    # ABLATION MODE: Single file with multiple embedding types, 2x2 grid
    # =========================================================================
    if args.ablation_mode:
        print("Running in ABLATION MODE (2x2 grid)")

        # Validate input file
        if args.input_file is None:
            # Try default path
            default_input = DATA_BASE_PATH / "test_linear_ablations.pkl"
            if default_input.exists():
                args.input_file = str(default_input)
                print(f"Using default input file: {args.input_file}")
            else:
                raise ValueError(
                    "Ablation mode requires --input_file with all ablation embeddings. "
                    f"Expected file at: {default_input}"
                )

        input_path = Path(args.input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        print(f"Loading: {input_path}")
        df = pd.read_pickle(input_path)
        print(f"Loaded DataFrame with {len(df)} rows")

        # Check available embedding columns
        embed_cols = [c for c in df.columns if c.startswith("embed_") and c.endswith("_h")]
        print(f"Available embedding columns: {embed_cols}")

        # Check if PRISM Full embeddings are missing and try to load from separate file
        if "embed_evo_ab_h" not in df.columns:
            print("\n  PRISM Full embeddings (embed_evo_ab_h) not found in ablation file.")
            print("  Attempting to load from original embeddings file...")

            # Determine original embeddings file path
            if args.original_embed_file:
                original_path = Path(args.original_embed_file)
            else:
                # Try default location: same directory, test_linear_evo_ab.pkl
                original_path = input_path.parent / "test_linear_evo_ab.pkl"

            if original_path.exists():
                print(f"  Loading original embeddings from: {original_path}")
                df_original = pd.read_pickle(original_path)

                # Check if the original file has the required columns
                if "embed_evo_ab_h" in df_original.columns:
                    # Merge embedding columns from original file
                    # Use index alignment to ensure correct row matching
                    cols_to_merge = ["embed_evo_ab_h", "embed_evo_ab_l"]
                    cols_available = [c for c in cols_to_merge if c in df_original.columns]

                    if len(cols_available) > 0:
                        # Reset index on both to ensure alignment
                        df = df.reset_index(drop=True)
                        df_original = df_original.reset_index(drop=True)

                        # Verify row counts match
                        if len(df) == len(df_original):
                            for col in cols_available:
                                df[col] = df_original[col]
                            print(f"  ✓ Merged columns: {cols_available}")
                        else:
                            print(f"  ⚠ Row count mismatch: ablation={len(df)}, original={len(df_original)}")
                            print("    Skipping PRISM Full model.")
                else:
                    print(f"  ⚠ Original file doesn't have embed_evo_ab_h column.")
            else:
                print(f"  ⚠ Original embeddings file not found: {original_path}")
                print("    Use --original_embed_file to specify the path.")
                print("    Skipping PRISM Full model.")

        # Process each ablation model
        for model_name, embed_col_base in tqdm(ABLATION_MODEL_CONFIGS, desc="Processing ablation models"):
            # Check if embedding columns exist
            h_col = f"{embed_col_base}_h"
            if h_col not in df.columns:
                print(f"  Warning: Column {h_col} not found. Skipping {model_name}.")
                continue

            result = process_single_model(df, model_name, embed_col_base, args)
            if result is not None:
                results[model_name] = result

        if len(results) == 0:
            raise ValueError("No valid embeddings found for any ablation model!")

        # Create 2x2 visualization
        output_filename = "umap_gl_ngl_ablation.png"
        output_path = output_dir / output_filename

        create_2x2_visualization(
            results=results,
            output_path=output_path,
            knn_k=args.knn_k,
            skip_knn=args.skip_knn,
        )

        # Print summary
        print("\n" + "="*60)
        print("SUMMARY - Ablation UMAP Results (Mean-Centered)")
        print("="*60)
        model_configs = ABLATION_MODEL_CONFIGS

    # =========================================================================
    # STANDARD MODE: Multiple files, 2x3 grid
    # =========================================================================
    else:
        print("Running in STANDARD MODE (2x3 grid)")

        for model_name, file_model_name, embed_col_base in tqdm(MODEL_CONFIGS, desc="Processing models"):
            # Load model-specific embedding file
            input_file = DATA_BASE_PATH / f"test_linear_{file_model_name}.pkl"

            if not input_file.exists():
                print(f"  File not found: {input_file}. Skipping.")
                continue

            print(f"  Loading: {input_file}")
            df = pd.read_pickle(input_file)
            print(f"  Loaded DataFrame with {len(df)} rows")

            result = process_single_model(df, model_name, embed_col_base, args)
            if result is not None:
                results[model_name] = result

        if len(results) == 0:
            raise ValueError("No valid embeddings found for any model!")

        # Create 2x3 visualization
        output_filename = "umap_gl_ngl_mean_centered.png"
        output_path = output_dir / output_filename

        create_2x3_visualization(
            results=results,
            output_path=output_path,
            knn_k=args.knn_k,
            skip_knn=args.skip_knn,
        )

        # Print summary
        print("\n" + "="*60)
        print("SUMMARY - Mean-Centered UMAP Results")
        print("="*60)
        model_configs = [(name, None) for name, _, _ in MODEL_CONFIGS]

    # Print results table
    if args.skip_knn:
        print(f"{'Model':<35} {'ARI':>8} {'GL':>8} {'NGL':>8}")
        print("-" * 65)
        for model_name, _ in (ABLATION_MODEL_CONFIGS if args.ablation_mode else [(n, None) for n, _, _ in MODEL_CONFIGS]):
            if model_name in results:
                _, _, ari, _, gl_count, ngl_count = results[model_name]
                # Clean up multiline model names for table
                clean_name = model_name.replace('\n', ' ')
                print(f"{clean_name:<35} {ari:>8.4f} {gl_count:>8} {ngl_count:>8}")
            else:
                clean_name = model_name.replace('\n', ' ')
                print(f"{clean_name:<35} {'N/A':>8} {'N/A':>8} {'N/A':>8}")
    else:
        print(f"{'Model':<35} {'ARI':>8} {f'{args.knn_k}-NN Acc':>10} {'GL':>8} {'NGL':>8}")
        print("-" * 75)
        for model_name, _ in (ABLATION_MODEL_CONFIGS if args.ablation_mode else [(n, None) for n, _, _ in MODEL_CONFIGS]):
            if model_name in results:
                _, _, ari, knn_acc, gl_count, ngl_count = results[model_name]
                clean_name = model_name.replace('\n', ' ')
                print(f"{clean_name:<35} {ari:>8.4f} {knn_acc:>10.4f} {gl_count:>8} {ngl_count:>8}")
            else:
                clean_name = model_name.replace('\n', ' ')
                print(f"{clean_name:<35} {'N/A':>8} {'N/A':>10} {'N/A':>8} {'N/A':>8}")

    print("\nDone!")


if __name__ == "__main__":
    main()
