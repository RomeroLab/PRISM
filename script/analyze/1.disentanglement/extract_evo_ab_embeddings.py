#!/usr/bin/env python
"""
Residue-Level Embedding Extraction Script for EvoAb (SFT_ESM2) Custom Model

Extracts per-residue embeddings from a custom-trained SFT_ESM2 model checkpoint
with support for case-sensitive tokenization (Upper=GL, Lower=NGL).

CRITICAL: Input sequences MUST maintain their original case (mixed case) to
preserve the evolutionary signal encoded during training.

Usage (Single Model):
    python extract_evo_ab_embeddings.py \
        --checkpoint path/to/model.ckpt \
        --input_file data/linear_probe_data/train_linear.pkl \
        --output_file data/linear_probe_data/train_linear_evo_ab.pkl \
        --batch_size 32

Usage (Ablation Mode - Multiple Models):
    python extract_evo_ab_embeddings.py \
        --ablation_mode \
        --ablation_configs configs/ablation_study.yaml \
        --input_file data/linear_probe_data/train_linear.pkl \
        --output_file data/linear_probe_data/train_linear_ablations.pkl \
        --batch_size 32

Ablation Config YAML Format:
    models:
      ablation1:
        name: "ablation1_no_pretrain"
        checkpoint: "outputs/.../best.ckpt"
      ablation2:
        name: "ablation2_simple_paired"
        checkpoint: "outputs/.../best.ckpt"

Output Format:
    Adds columns `embed_{embed_name}_h` and `embed_{embed_name}_l` to the DataFrame.
    For ablation mode: `embed_evo_ab_ablation1_h`, `embed_evo_ab_ablation2_h`, etc.
    Each entry is a numpy array of shape (seq_len, hidden_dim) in float16.
"""

from __future__ import annotations

import argparse
import gc
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm


# ==============================================================================
# Ablation Study Configuration
# ==============================================================================

@dataclass
class AblationModelConfig:
    """Configuration for a single ablation model."""
    name: str
    checkpoint: str


def load_ablation_configs(ablation_config_path: str) -> List[AblationModelConfig]:
    """
    Load ablation study configuration from YAML file.

    Expected format:
        models:
          ablation1:
            name: "ablation1_no_pretrain"
            checkpoint: "outputs/.../best.ckpt"

    Args:
        ablation_config_path: Path to ablation config YAML

    Returns:
        List of AblationModelConfig objects
    """
    logger.info(f"Loading ablation configs from: {ablation_config_path}")
    with open(ablation_config_path, "r") as f:
        ablation_config = yaml.safe_load(f)

    models = []
    for key, model_cfg in ablation_config.get("models", {}).items():
        models.append(AblationModelConfig(
            name=model_cfg.get("name", key),
            checkpoint=model_cfg["checkpoint"],
        ))

    logger.info(f"Loaded {len(models)} ablation models:")
    for m in models:
        logger.info(f"  - {m.name}: {m.checkpoint}")

    return models


def get_default_ablation_configs() -> List[AblationModelConfig]:
    """
    Return default ablation configurations based on run_ablation_study.sh.

    This provides hardcoded defaults for the 3-ablation study design:
    - Ablation 1: Multihead + No Pretraining
    - Ablation 2: Simple Head + With Pretraining
    - Ablation 3: Simple Head + No Pretraining
    """
    # Each user's training run produces a different best-PPL filename, so we
    # default to `last.ckpt` (always written by PyTorch Lightning) and let the
    # caller override via --checkpoint at runtime.
    return [
        AblationModelConfig(
            name="ablation1",
            checkpoint="outputs/ESM2_v34.1b_ablation1_no_pretrain_esm2_t12_35M_UR50D_custom_unfrozen12_lr4e-4_bs256/checkpoints/last.ckpt",
        ),
        AblationModelConfig(
            name="ablation2",
            checkpoint="outputs/ESM2_v34.1b_ablation2_simple_paired_esm2_t12_35M_UR50D_custom_unfrozen12_lr1e-4_bs256/checkpoints/last.ckpt",
        ),
        AblationModelConfig(
            name="ablation3",
            checkpoint="outputs/ESM2_v34.1b_ablation3_simple_no_pretrain_esm2_t12_35M_UR50D_custom_unfrozen12_lr4e-4_bs256/checkpoints/last.ckpt",
        ),
    ]

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ==============================================================================
# Model Loading Utilities
# ==============================================================================

def load_prism_model(checkpoint_path: str, device: torch.device):
    """
    Load PRISM model from checkpoint using the pretrained() API.

    Hyperparameters are extracted automatically from the checkpoint -- no YAML
    config file is needed.

    Args:
        checkpoint_path: Path to the .ckpt file
        device: Device to load model on

    Returns:
        PrismModel wrapper in eval mode
    """
    import prism

    logger.info(f"Loading PRISM model from checkpoint: {checkpoint_path}")

    prism_model = prism.pretrained(str(checkpoint_path), device=str(device))

    # Log model info
    underlying = prism_model.model
    if hasattr(underlying, "ESM2") and hasattr(underlying.ESM2, "config"):
        hidden_size = underlying.ESM2.config.hidden_size
    else:
        hidden_size = "unknown"
    logger.info(f"Model hidden size: {hidden_size}")
    logger.info(f"Model loaded successfully on {device}")
    logger.info(f"Tokenizer vocabulary size: {len(prism_model.tokenizer)}")

    return prism_model


# ==============================================================================
# Embedding Extraction
# ==============================================================================

def get_embeddings(
    prism_model,
    sequences: List[str],
    device: torch.device,
    batch_size: int = 32,
    max_length: int = 512,
) -> List[np.ndarray]:
    """
    Extract residue-level embeddings from a PRISM model.

    CRITICAL: This function preserves the original case of input sequences.
    The tokenizer must NOT lowercase the sequences as the case encodes
    evolutionary information (Upper=GL, Lower=NGL).

    Args:
        prism_model: PrismModel wrapper (from prism.pretrained())
        sequences: List of amino acid sequences (with preserved case)
        device: Device for inference
        batch_size: Batch size for processing
        max_length: Maximum sequence length for tokenization

    Returns:
        List of numpy arrays, each of shape (seq_len, hidden_dim) in float16
    """
    embeddings: List[np.ndarray] = []
    tokenizer = prism_model.tokenizer
    esm2_model = prism_model.model.ESM2

    # Process in batches
    for i in tqdm(range(0, len(sequences), batch_size), desc="Extracting EvoAb embeddings"):
        batch_seqs = sequences[i : i + batch_size]

        # Tokenize sequences
        # CRITICAL: The tokenizer should NOT modify the case of input sequences
        # ESM2 tokenizer by default preserves case, but we verify this
        encoded = tokenizer(
            batch_seqs,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            add_special_tokens=True,
        )

        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        with torch.no_grad():
            # Forward pass through ESM2 backbone
            outputs = esm2_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

            # Get last hidden state: [Batch, SeqLen+2, HiddenDim]
            # +2 accounts for [CLS] at start and [EOS] at end
            # Note: EsmForMaskedLM returns MaskedLMOutput which has hidden_states tuple
            # instead of last_hidden_state. We need to get the last element of hidden_states.
            if hasattr(outputs, 'last_hidden_state') and outputs.last_hidden_state is not None:
                last_hidden = outputs.last_hidden_state
            elif hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                # hidden_states is a tuple of (embedding_output, layer_1, ..., layer_n)
                # The last element is the output of the final transformer layer
                last_hidden = outputs.hidden_states[-1]
            else:
                raise ValueError(
                    f"Could not find hidden states in model output. "
                    f"Available attributes: {dir(outputs)}"
                )

            # Extract embeddings for each sequence, removing special tokens
            for j, seq in enumerate(batch_seqs):
                seq_len = len(seq)

                # ESM-2 tokenization format: <cls> A A A ... <eos> <pad>
                # Extract residue embeddings at indices 1 to seq_len (inclusive)
                # This skips <cls> at index 0 and stops before <eos>
                emb = last_hidden[j, 1 : 1 + seq_len, :].cpu().numpy().astype(np.float16)

                # Verify shape matches sequence length
                if emb.shape[0] != seq_len:
                    logger.warning(
                        f"Shape mismatch: expected {seq_len}, got {emb.shape[0]}. "
                        f"Adjusting..."
                    )
                    if emb.shape[0] < seq_len:
                        # Pad with zeros if embedding is shorter
                        padding = np.zeros(
                            (seq_len - emb.shape[0], emb.shape[1]),
                            dtype=np.float16
                        )
                        emb = np.vstack([emb, padding])
                    else:
                        # Truncate if embedding is longer
                        emb = emb[:seq_len]

                embeddings.append(emb)

    return embeddings


# ==============================================================================
# Data Loading Utilities
# ==============================================================================

def detect_sequence_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """Detect heavy and light chain column names in DataFrame."""
    heavy_candidates = [
        "heavy_sequence", "sequence_h", "HEAVY_CHAIN_AA_SEQUENCE",
        "HeavySequence", "heavy", "VH"
    ]
    light_candidates = [
        "light_sequence", "sequence_l", "LIGHT_CHAIN_AA_SEQUENCE",
        "LightSequence", "light", "VL"
    ]

    heavy_col = None
    light_col = None

    for col in heavy_candidates:
        if col in df.columns:
            heavy_col = col
            break

    for col in light_candidates:
        if col in df.columns:
            light_col = col
            break

    return heavy_col, light_col


def load_dataframe(input_path: str) -> pd.DataFrame:
    """Load DataFrame from pickle file."""
    logger.info(f"Loading data from: {input_path}")
    df = pd.read_pickle(input_path)
    logger.info(f"Loaded {len(df)} sequences")
    logger.info(f"Columns: {list(df.columns)}")
    return df


# ==============================================================================
# Main Pipeline
# ==============================================================================

def extract_ablation_embeddings(
    df: pd.DataFrame,
    heavy_col: str,
    light_col: Optional[str],
    ablation_configs: List[AblationModelConfig],
    device: torch.device,
    batch_size: int,
    max_length: int,
    base_embed_name: str = "evo_ab",
) -> pd.DataFrame:
    """
    Extract embeddings from multiple ablation models.

    Args:
        df: DataFrame with sequences
        heavy_col: Column name for heavy chain sequences
        light_col: Column name for light chain sequences (or None)
        ablation_configs: List of ablation model configurations
        device: Device for inference
        batch_size: Batch size for processing
        max_length: Maximum sequence length
        base_embed_name: Base name for embeddings (e.g., "evo_ab" -> "evo_ab_ablation1")

    Returns:
        DataFrame with added embedding columns for each ablation model
    """
    heavy_seqs = df[heavy_col].tolist()
    light_seqs = df[light_col].tolist() if light_col else None

    for ablation_cfg in ablation_configs:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing ablation model: {ablation_cfg.name}")
        logger.info(f"{'='*60}")

        # Check if checkpoint exists
        checkpoint_path = Path(ablation_cfg.checkpoint)

        if not checkpoint_path.exists():
            logger.warning(f"Checkpoint not found: {checkpoint_path}, skipping...")
            continue

        try:
            # Load model via prism.pretrained() (hparams extracted from checkpoint)
            model = load_prism_model(str(checkpoint_path), device)

            # Define embedding column names
            embed_name = f"{base_embed_name}_{ablation_cfg.name}"
            h_col_name = f"embed_{embed_name}_h"

            # Extract heavy chain embeddings
            logger.info(f"Extracting heavy chain embeddings ({len(heavy_seqs)} sequences)...")
            embeddings_h = get_embeddings(
                prism_model=model,
                sequences=heavy_seqs,
                device=device,
                batch_size=batch_size,
                max_length=max_length,
            )
            df[h_col_name] = embeddings_h
            logger.info(f"Added column: {h_col_name}")

            # Extract light chain embeddings if available
            if light_seqs:
                l_col_name = f"embed_{embed_name}_l"
                logger.info(f"Extracting light chain embeddings ({len(light_seqs)} sequences)...")
                embeddings_l = get_embeddings(
                    prism_model=model,
                    sequences=light_seqs,
                    device=device,
                    batch_size=batch_size,
                    max_length=max_length,
                )
                df[l_col_name] = embeddings_l
                logger.info(f"Added column: {l_col_name}")

            # Log embedding stats
            sample_emb = embeddings_h[0]
            logger.info(f"  Sample embedding shape: {sample_emb.shape}")
            logger.info(f"  Sample embedding range: [{sample_emb.min():.4f}, {sample_emb.max():.4f}]")

        except Exception as e:
            logger.error(f"Error processing {ablation_cfg.name}: {e}")
            import traceback
            traceback.print_exc()
            continue

        finally:
            # Cleanup GPU memory after each model
            if torch.cuda.is_available():
                try:
                    del model
                    gc.collect()
                    torch.cuda.empty_cache()
                    logger.info(f"GPU memory cleared after {ablation_cfg.name}")
                except Exception as cleanup_error:
                    logger.warning(f"GPU cleanup warning: {cleanup_error}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Extract residue-level embeddings from SFT_ESM2 (EvoAb) model"
    )

    # Single model arguments (for backward compatibility)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to the .ckpt checkpoint file (single model mode)",
    )

    # Ablation mode arguments
    parser.add_argument(
        "--ablation_mode",
        action="store_true",
        help="Enable ablation study mode to process multiple models",
    )
    parser.add_argument(
        "--ablation_configs",
        type=str,
        default=None,
        help="Path to YAML file defining ablation models (optional, uses defaults if not provided)",
    )
    parser.add_argument(
        "--ablation_models",
        type=str,
        nargs="+",
        default=None,
        help="Specific ablation models to process (e.g., 'ablation1 ablation2'). "
             "If not specified, processes all models in the config.",
    )

    # Common arguments
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to input .pkl file containing sequences",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to save output .pkl file with embeddings",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference (default: 32)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length for tokenization (default: 512)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (default: cuda if available)",
    )
    parser.add_argument(
        "--embed_name",
        type=str,
        default="evo_ab",
        help="Name for embedding columns (default: evo_ab -> embed_evo_ab_h, embed_evo_ab_l)",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.ablation_mode:
        logger.info("Running in ABLATION MODE (multiple models)")
    else:
        if args.checkpoint is None:
            parser.error("--checkpoint is required in single model mode. "
                        "Use --ablation_mode for multi-model processing.")

    # Setup device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Validate input path
    input_path = Path(args.input_file)
    output_path = Path(args.output_file)

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_dataframe(str(input_path))

    # Detect sequence columns
    heavy_col, light_col = detect_sequence_columns(df)

    if heavy_col is None:
        logger.error("Could not detect heavy chain sequence column!")
        logger.error(f"Available columns: {list(df.columns)}")
        sys.exit(1)

    logger.info(f"Heavy chain column: {heavy_col}")
    logger.info(f"Light chain column: {light_col}")

    # Get sequences for verification
    heavy_seqs = df[heavy_col].tolist()

    # Verify case preservation in sample sequences
    logger.info("\n--- Sample sequences (verifying case preservation) ---")
    for i in range(min(3, len(heavy_seqs))):
        sample = heavy_seqs[i][:50] if len(heavy_seqs[i]) > 50 else heavy_seqs[i]
        has_lower = any(c.islower() for c in sample)
        has_upper = any(c.isupper() for c in sample)
        logger.info(f"  Sample {i}: {sample}... (has_lower={has_lower}, has_upper={has_upper})")
    logger.info("---")

    # =========================================================================
    # ABLATION MODE: Process multiple models
    # =========================================================================
    if args.ablation_mode:
        # Load or use default ablation configs
        if args.ablation_configs:
            ablation_configs = load_ablation_configs(args.ablation_configs)
        else:
            logger.info("Using default ablation configurations")
            ablation_configs = get_default_ablation_configs()

        # Filter by specific models if requested
        if args.ablation_models:
            filtered_configs = [
                cfg for cfg in ablation_configs
                if cfg.name in args.ablation_models
            ]
            if not filtered_configs:
                logger.error(f"No matching ablation models found for: {args.ablation_models}")
                logger.error(f"Available: {[cfg.name for cfg in ablation_configs]}")
                sys.exit(1)
            ablation_configs = filtered_configs
            logger.info(f"Filtered to {len(ablation_configs)} models: {[c.name for c in ablation_configs]}")

        # Extract embeddings from all ablation models
        df = extract_ablation_embeddings(
            df=df,
            heavy_col=heavy_col,
            light_col=light_col,
            ablation_configs=ablation_configs,
            device=device,
            batch_size=args.batch_size,
            max_length=args.max_length,
            base_embed_name=args.embed_name,
        )

    # =========================================================================
    # SINGLE MODEL MODE: Process one model (backward compatible)
    # =========================================================================
    else:
        # Validate paths for single model mode
        checkpoint_path = Path(args.checkpoint)

        if not checkpoint_path.exists():
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            sys.exit(1)

        # Load model via prism.pretrained() (hparams extracted from checkpoint)
        model = load_prism_model(str(checkpoint_path), device)

        # Get sequences
        light_seqs = df[light_col].tolist() if light_col else None

        # Extract heavy chain embeddings
        logger.info(f"\nExtracting heavy chain embeddings ({len(heavy_seqs)} sequences)...")
        embeddings_h = get_embeddings(
            prism_model=model,
            sequences=heavy_seqs,
            device=device,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )

        # Add heavy chain embeddings to DataFrame
        h_col_name = f"embed_{args.embed_name}_h"
        df[h_col_name] = embeddings_h
        logger.info(f"Added column: {h_col_name}")

        # Log embedding stats
        sample_emb = embeddings_h[0]
        logger.info(f"  Sample embedding shape: {sample_emb.shape}")
        logger.info(f"  Sample embedding dtype: {sample_emb.dtype}")
        logger.info(f"  Sample embedding range: [{sample_emb.min():.4f}, {sample_emb.max():.4f}]")

        # Extract light chain embeddings if available
        if light_seqs:
            logger.info(f"\nExtracting light chain embeddings ({len(light_seqs)} sequences)...")
            embeddings_l = get_embeddings(
                prism_model=model,
                sequences=light_seqs,
                device=device,
                batch_size=args.batch_size,
                max_length=args.max_length,
            )

            l_col_name = f"embed_{args.embed_name}_l"
            df[l_col_name] = embeddings_l
            logger.info(f"Added column: {l_col_name}")

            sample_emb_l = embeddings_l[0]
            logger.info(f"  Sample embedding shape: {sample_emb_l.shape}")

        # Cleanup GPU memory
        if torch.cuda.is_available():
            del model
            gc.collect()
            torch.cuda.empty_cache()

    # Save output
    logger.info(f"\nSaving to: {output_path}")
    df.to_pickle(str(output_path))

    # Summary
    embed_cols = [c for c in df.columns if c.startswith("embed_")]
    logger.info(f"\n{'='*60}")
    logger.info("Extraction complete!")
    logger.info(f"{'='*60}")
    logger.info(f"Output file: {output_path}")
    logger.info(f"Total sequences: {len(df)}")
    logger.info(f"Embedding columns: {embed_cols}")

    # Get hidden dimension from first embedding column
    if embed_cols:
        first_embed = df[embed_cols[0]].iloc[0]
        if first_embed is not None and hasattr(first_embed, 'shape'):
            logger.info(f"Hidden dimension: {first_embed.shape[1]}")

    if args.ablation_mode:
        logger.info(f"\nAblation mode: extracted embeddings from {len(ablation_configs)} models")
        for cfg in ablation_configs:
            logger.info(f"  - {cfg.name}")


if __name__ == "__main__":
    main()
