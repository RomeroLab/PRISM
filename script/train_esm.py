#!/usr/bin/env python
# coding: utf-8

"""
Training script for ESM2 supervised fine-tuning using PRISM library.

Usage:
    python train_esm.py --config config.yaml
"""

import argparse
import os
import pathlib
import yaml
import pandas as pd
import numpy as np
import torch
import torch.serialization
import pytorch_lightning as pl

# PyTorch 2.6+ defaults weights_only=True in torch.load, which rejects
# pickled pathlib.PosixPath objects found in older checkpoints.
torch.serialization.add_safe_globals([pathlib.PosixPath])
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback
from pytorch_lightning.loggers import TensorBoardLogger
from pathlib import Path

from prism import SFT_ESM2, SFTDataModule, LazyShardedDataModule, GeneVocabulary
from transformers import AutoTokenizer


class MaskRatioSchedulerCallback(Callback):
    """Callback to update the mask ratio schedule step counter during training."""

    def __init__(self, step_counter):
        super().__init__()
        self.step_counter = step_counter

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        """Update the step counter before each training batch."""
        self.step_counter['n'] = trainer.global_step


# [CHANGE v17] NGL Masking Schedule Callback
class NGLMaskScheduleCallback(Callback):
    """
    Callback to update NGL masking probability during training.

    This enables curriculum learning where NGL tokens are masked more aggressively
    early in training (e.g., 80%) and less aggressively later (e.g., 50%).
    """

    def __init__(self, data_module):
        super().__init__()
        self.data_module = data_module

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Update NGL mask probability after each training batch."""
        # Call update_step which will recalculate NGL mask prob based on schedule
        self.data_module.update_step(trainer.global_step)


# [NEW v34] Developability Tracking Callback
class DevelopabilityCallback(Callback):
    """
    Callback to track developability Pearson correlations during training.

    [v34] Logs Pearson r for all 5 developability properties every N
    validation steps, enabling real-time tracking of model quality.

    [v34.1 FIX] Always logs metrics on every validation epoch (using last
    computed values) so ModelCheckpoint can monitor them consistently.

    Properties tracked:
    - HIC (hydrophobicity): lower is better → expect positive r with PPL
    - PR_CHO (reactivity): lower is better → expect positive r with PPL
    - AC-SINS (aggregation): lower is better → expect positive r with PPL
    - Tm2 (thermal stability): higher is better → expect negative r with PPL
    - Titer (expression): higher is better → expect negative r with PPL
    """

    # Property configuration: column name and whether to invert for correlation
    PROPERTIES = {
        'HIC': {'column': 'HIC', 'invert_ppl': False},  # lower is better
        'PR_CHO': {'column': 'PR_CHO', 'invert_ppl': False},  # lower is better
        'AC_SINS': {'column': 'AC-SINS_pH7.4', 'invert_ppl': False},  # lower is better
        'Tm2': {'column': 'Tm2', 'invert_ppl': True},  # higher is better
        'Titer': {'column': 'Titer', 'invert_ppl': True},  # higher is better
    }

    def __init__(
        self,
        data_path: str,
        eval_every_n: int = 10,
        sample_size: int = 100,
        num_masks: int = 3,
        batch_size: int = 8,
    ):
        """
        Args:
            data_path: Path to developability data CSV
            eval_every_n: Run developability eval every N validation epochs
            sample_size: Number of sequences to sample for evaluation
            num_masks: Number of masking rounds for pseudo-PPL estimation
            batch_size: Batch size for inference
        """
        super().__init__()
        self.data_path = data_path
        self.eval_every_n = eval_every_n
        self.sample_size = sample_size
        self.num_masks = num_masks
        self.batch_size = batch_size
        self.epoch_count = 0
        self.data = None
        self.property_values = {}
        # [FIX v34.1] Store last computed correlations for consistent logging
        # Initialize with -inf so mode='max' checkpoints don't save until real values
        self.last_correlations = {name: float('-inf') for name in self.PROPERTIES}

    def setup(self, trainer, pl_module, stage=None):
        """Load developability data on setup."""
        if self.data is None:
            try:
                full_data = pd.read_csv(self.data_path)
                # Sample subset for speed
                if len(full_data) > self.sample_size:
                    self.data = full_data.sample(n=self.sample_size, random_state=42)
                else:
                    self.data = full_data

                # Prepare sequences (concatenate VH + VL)
                self.sequences = [
                    f"{row['vh_protein_sequence']}{row['vl_protein_sequence']}"
                    for _, row in self.data.iterrows()
                ]

                # Extract all 5 property values
                for name, cfg in self.PROPERTIES.items():
                    col = cfg['column']
                    if col in self.data.columns:
                        self.property_values[name] = self.data[col].values
                    else:
                        self.property_values[name] = None

                print(f"  [DevelopabilityCallback] Loaded {len(self.sequences)} sequences")
                available = [k for k, v in self.property_values.items() if v is not None]
                print(f"    Properties available: {', '.join(available)}")

            except Exception as e:
                print(f"  [DevelopabilityCallback] Warning: Could not load data: {e}")
                self.data = None

    def on_validation_epoch_end(self, trainer, pl_module):
        """Compute Pearson r for all 5 properties every N validation epochs."""
        self.epoch_count += 1

        # Determine if we should compute new correlations this epoch
        should_compute = (self.epoch_count % self.eval_every_n == 0)

        # Compute new correlations if at eval interval and data is loaded
        if should_compute and self.data is not None and len(self.sequences) > 0:
            try:
                from scipy.stats import pearsonr

                # Compute Final Head Upper PPL
                ppls = pl_module.compute_developability_ppl(
                    sequences=self.sequences,
                    v_genes=None,
                    j_genes=None,
                    mask_prob=0.15,
                    num_masks=self.num_masks,
                    batch_size=self.batch_size,
                )
                ppls = np.array(ppls)

                # Log Pearson r for each property
                print(f"  [Developability] Epoch {self.epoch_count}:")
                for name, cfg in self.PROPERTIES.items():
                    values = self.property_values.get(name)
                    if values is None:
                        continue

                    # Filter valid (non-NaN) entries
                    valid_mask = ~np.isnan(ppls) & ~np.isnan(values)
                    if valid_mask.sum() < 10:
                        continue

                    ppl_valid = ppls[valid_mask]
                    val_valid = values[valid_mask]

                    # Invert PPL for "higher is better" properties
                    # This makes positive r = good model
                    if cfg['invert_ppl']:
                        ppl_for_corr = -ppl_valid
                    else:
                        ppl_for_corr = ppl_valid

                    # Compute Pearson r
                    r, pval = pearsonr(ppl_for_corr, val_valid)

                    # Store the computed correlation
                    self.last_correlations[name] = r

                    sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
                    print(f"    {name}: r={r:.3f}{sig}")

            except Exception as e:
                print(f"  [DevelopabilityCallback] Error: {e}")

        # [FIX v34.1] ALWAYS log metrics using pl_module.log() so checkpoints can see them
        # This ensures ModelCheckpoint finds the metrics on every validation epoch
        for name in self.PROPERTIES:
            r_value = self.last_correlations[name]
            # Use pl_module.log() instead of trainer.logger.log_metrics()
            # so that ModelCheckpoint can monitor these metrics
            pl_module.log(
                f'val/dev_r_{name}',
                r_value,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,  # Important for DDP
            )


def load_config(config_path):
    """
    Load configuration from YAML file.
    Raises error if file doesn't exist or required fields are missing.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"Config file is empty: {config_path}")

    return config


def validate_config(config):
    """
    Validate that all required fields are present in config.
    Raises KeyError if any required field is missing.
    """
    required_fields = {
        'data': ['data_path', 'batch_size', 'mask_prob', 'num_workers'],
        'model': ['model_identifier', 'num_unfrozen_transformer_blocks', 'random_weights', 'add_custom_tokens',
                  'activation_function'],
        'training': [
            'seed',
            'peak_learning_rate',
            'adam_beta1',
            'adam_beta2',
            'adam_epsilon',
            'weight_decay',
            'warmup_steps',
            'max_steps',
            'logging_steps',
            'eval_steps',
            'loss_type',
            'gradient_accumulation_steps',
            'gradient_clip_val'
        ],
        'trainer': ['devices', 'accelerator', 'precision'],
        'logging': ['output_dir', 'experiment_name'],
        'checkpointing': ['top_k'],
        'early_stopping': ['enabled', 'patience', 'monitor']
    }

    # Optional field with default value
    optional_fields = {
        'model': {
            'custom_token_strategy': 'mask_tokens',  # Default to original strategy
            'tie_word_embeddings': True,  # Default to True for backward compatibility
            'ngl_loss_alpha': 1.0  # [v7.0] Default to 1.0 (no reweighting) for backward compatibility
        },
        'trainer': {
            'strategy': 'auto'  # [DDP] Default to 'auto' for backward compatibility
        }
    }

    # Check each section and its required fields
    for section, fields in required_fields.items():
        if section not in config:
            raise KeyError(f"Missing required section in config: '{section}'")

        for field in fields:
            if field not in config[section]:
                raise KeyError(f"Missing required field in config['{section}']: '{field}'")

    # Set default values for optional fields
    for section, fields in optional_fields.items():
        if section in config:
            for field, default_value in fields.items():
                if field not in config[section]:
                    config[section][field] = default_value
                    print(f"ℹ Setting default value for {section}.{field} = {default_value}")

    # Validate custom_token_strategy value
    if config['model']['add_custom_tokens']:
        strategy = config['model']['custom_token_strategy']
        valid_strategies = ['mask_tokens', 'lowercase_ngl', 'hybrid_lowercase']
        if strategy not in valid_strategies:
            raise ValueError(f"Invalid custom_token_strategy: '{strategy}'. Must be one of {valid_strategies}")

    # Validate mask_ratio_schedule if provided
    if 'mask_ratio_schedule' in config['model'] and config['model']['mask_ratio_schedule'].get('enabled', False):
        schedule = config['model']['mask_ratio_schedule']
        required_schedule_fields = ['start_ratio', 'end_ratio', 'start_step', 'end_step']
        for field in required_schedule_fields:
            if field not in schedule:
                raise KeyError(f"Missing required field in mask_ratio_schedule: '{field}'")

        # Validate ratios are in [0, 1]
        if not (0 <= schedule['start_ratio'] <= 1):
            raise ValueError(f"start_ratio must be between 0 and 1, got {schedule['start_ratio']}")
        if not (0 <= schedule['end_ratio'] <= 1):
            raise ValueError(f"end_ratio must be between 0 and 1, got {schedule['end_ratio']}")

        # Validate steps are sensible
        if schedule['start_step'] < 0:
            raise ValueError(f"start_step must be >= 0, got {schedule['start_step']}")
        if schedule['end_step'] <= schedule['start_step']:
            raise ValueError(f"end_step ({schedule['end_step']}) must be > start_step ({schedule['start_step']})")

        # Validate strategy supports scheduling (only mask_tokens and hybrid_lowercase)
        if config['model']['add_custom_tokens'] and config['model']['custom_token_strategy'] not in ['mask_tokens', 'hybrid_lowercase']:
            raise ValueError(f"mask_ratio_schedule is only supported for 'mask_tokens' and 'hybrid_lowercase' strategies, "
                           f"not '{config['model']['custom_token_strategy']}'")

    print("✓ Configuration validated successfully")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train ESM2 model for antibody sequences')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    args = parser.parse_args()

    print("="*80)
    print("ESM2 Supervised Fine-Tuning with PRISM")
    print("="*80)

    # =========================================================================
    # [v31.2] NCCL and CUDA Optimizations for faster DDP training
    # =========================================================================
    import os

    # NCCL optimizations for 2-GPU DDP
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    os.environ.setdefault("NCCL_NSOCKS_PERTHREAD", "4")
    os.environ.setdefault("NCCL_SOCKET_NTHREADS", "2")
    os.environ.setdefault("NCCL_P2P_DISABLE", "0")
    os.environ.setdefault("NCCL_SHM_DISABLE", "0")
    os.environ.setdefault("NCCL_BUFFSIZE", "8388608")  # 8MB buffer
    os.environ.setdefault("NCCL_TREE_THRESHOLD", "0")
    os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")

    # CUDA memory management
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:512")

    # Load and validate configuration
    print(f"\nLoading configuration from: {args.config}")
    config = load_config(args.config)
    validate_config(config)

    # Set global random seed for reproducibility
    print("\n" + "="*80)
    print("Setting random seeds for reproducibility...")
    print("="*80)
    seed = config['training']['seed']
    pl.seed_everything(seed, workers=True)
    print(f"  Global seed set to: {seed}")

    # =========================================================================
    # [v31.2] Enable TF32 and Flash Attention for Ampere+ GPUs
    # =========================================================================
    if torch.cuda.is_available():
        # TF32 for faster matmul (10-20% speedup on Ampere+)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

        # Enable Flash/Memory-Efficient Attention (PyTorch 2.0+)
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)  # Disable slow fallback

        print(f"  TF32 enabled: {torch.backends.cuda.matmul.allow_tf32}")
        print(f"  Flash Attention: {torch.backends.cuda.flash_sdp_enabled()}")
        print(f"  Memory-Efficient Attention: {torch.backends.cuda.mem_efficient_sdp_enabled()}")

    # =========================================================================
    # [FIX] Initialize Tokenizer EARLY and Add Custom Tokens
    # =========================================================================
    # CRITICAL: The tokenizer must be fully configured BEFORE being passed to
    # SFTDataModule. This fixes the initialization order bug where:
    # - AntibodyMLMCollator computes ngl_token_ids_set during __init__
    # - If custom tokens (lowercase AAs) aren't added yet, ngl_token_ids_set is EMPTY
    # - This causes NGL-targeted masking to fail silently
    #
    # By initializing the tokenizer here and adding all custom tokens, we ensure
    # both SFTDataModule and SFT_ESM2 receive a consistent, fully-configured tokenizer.
    # =========================================================================
    print("\n" + "="*80)
    print("Initializing Tokenizer with Custom Tokens...")
    print("="*80)

    model_identifier = config['model']['model_identifier']
    tokenizer = AutoTokenizer.from_pretrained(f"facebook/{model_identifier}")
    original_vocab_size = len(tokenizer)
    print(f"  Base tokenizer loaded: {model_identifier}")
    print(f"  Original vocabulary size: {original_vocab_size}")

    # Add custom tokens based on strategy
    add_custom_tokens = config['model']['add_custom_tokens']
    custom_token_strategy = config['model']['custom_token_strategy']

    if add_custom_tokens:
        if custom_token_strategy == "mask_tokens":
            # Add GERM_MASK and NONGERM_MASK special tokens
            new_tokens = ["<GERM_MASK>", "<NONGERM_MASK>"]
            num_added = tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
            print(f"  Added {num_added} mask tokens: {new_tokens}")

        elif custom_token_strategy == "lowercase_ngl":
            # Add lowercase amino acid tokens for NGL positions
            lowercase_aa = ['a', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l',
                           'm', 'n', 'p', 'q', 'r', 's', 't', 'v', 'w', 'y']
            num_added = tokenizer.add_tokens(lowercase_aa)
            print(f"  Added {num_added} lowercase amino acid tokens")

            # Verify token IDs for debugging
            sample_ids = [tokenizer.convert_tokens_to_ids(aa) for aa in lowercase_aa[:5]]
            print(f"  Sample lowercase token IDs (a,c,d,e,f): {sample_ids}")

        elif custom_token_strategy == "hybrid_lowercase":
            # Add both mask tokens and lowercase amino acids
            mask_tokens = ["<GERM_MASK>", "<NONGERM_MASK>"]
            num_mask = tokenizer.add_special_tokens({"additional_special_tokens": mask_tokens})

            lowercase_aa = ['a', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l',
                           'm', 'n', 'p', 'q', 'r', 's', 't', 'v', 'w', 'y']
            num_aa = tokenizer.add_tokens(lowercase_aa)
            print(f"  Added {num_mask} mask tokens and {num_aa} lowercase amino acid tokens")

        print(f"  Updated vocabulary size: {len(tokenizer)}")
    else:
        print(f"  Using standard ESM2 tokenizer (no custom tokens)")

    # Print configuration
    print("\nConfiguration:")
    print(yaml.dump(config, default_flow_style=False, indent=2))

    # Load data
    print("="*80)
    print("Loading data...")
    print("="*80)
    data_path = Path(config['data']['data_path'])
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    # =========================================================================
    # [v31] Check if lazy loading is enabled BEFORE loading data
    # =========================================================================
    use_lazy_loading = config['data'].get('lazy_loading', False)

    # [OPTIMIZATION] Support both pickle and parquet formats
    # Parquet is preferred for large datasets (2-5x smaller, faster loading)
    if use_lazy_loading:
        # For lazy loading, we only need to read metadata and gene columns initially
        # The full data will be loaded in each DDP process during setup()
        print(f"\n[v31] LAZY LOADING: Reading only metadata (gene columns) initially...")

        # =========================================================================
        # [v31.4] Handle pre-split shard directory
        # =========================================================================
        if data_path.is_dir():
            # Pre-split shard directory - read metadata from first shard or valid file
            print(f"  [v31.4] PRE-SPLIT SHARD DIRECTORY detected: {data_path}")

            # Use valid.parquet for metadata (it has all columns)
            valid_file = data_path / "valid.parquet"
            if not valid_file.exists():
                # Fall back to first shard
                valid_file = data_path / "train_shard_0.parquet"

            if not valid_file.exists():
                raise FileNotFoundError(f"No valid.parquet or train_shard_0.parquet found in {data_path}")

            import pyarrow.parquet as pq
            pf = pq.ParquetFile(valid_file)
            all_columns = pf.schema_arrow.names

            # Count total rows across all shards
            total_train_rows = 0
            shard_idx = 0
            while True:
                shard_file = data_path / f"train_shard_{shard_idx}.parquet"
                if not shard_file.exists():
                    break
                pf_shard = pq.ParquetFile(shard_file)
                total_train_rows += pf_shard.metadata.num_rows
                shard_idx += 1

            print(f"  Found {shard_idx} training shards with {total_train_rows:,} total samples")

            # Only read gene columns if needed (for gene vocabulary)
            gene_cols_to_read = []
            v_gene_cols = ['v_gene', 'v_gene_heavy', 'v_gene_light']
            j_gene_cols = ['j_gene', 'j_gene_heavy', 'j_gene_light']
            for col in v_gene_cols + j_gene_cols:
                if col in all_columns:
                    gene_cols_to_read.append(col)

            # [FIX v31.4] Read gene columns from ALL files to get complete gene vocabulary
            # Reading only from valid.parquet caused size mismatch when resuming training
            if gene_cols_to_read:
                print(f"  Reading gene columns from ALL shard files for complete vocabulary...")
                dfs_for_genes = []

                # Read from all training shards
                for i in range(shard_idx):
                    shard_file = data_path / f"train_shard_{i}.parquet"
                    df_shard_genes = pd.read_parquet(shard_file, columns=gene_cols_to_read)
                    dfs_for_genes.append(df_shard_genes)
                    print(f"    Shard {i}: {len(df_shard_genes):,} rows")

                # Also read from valid and test
                if valid_file.exists():
                    df_valid_genes = pd.read_parquet(valid_file, columns=gene_cols_to_read)
                    dfs_for_genes.append(df_valid_genes)
                    print(f"    Valid: {len(df_valid_genes):,} rows")

                test_file = data_path / "test.parquet"
                if test_file.exists():
                    df_test_genes = pd.read_parquet(test_file, columns=gene_cols_to_read)
                    dfs_for_genes.append(df_test_genes)
                    print(f"    Test: {len(df_test_genes):,} rows")

                # Concatenate all gene data
                df = pd.concat(dfs_for_genes, ignore_index=True)
                print(f"  Total rows for gene vocabulary: {len(df):,}")
                print(f"  Columns read: {gene_cols_to_read}")

                # Free memory
                del dfs_for_genes
                import gc
                gc.collect()
            else:
                # No gene columns needed, create minimal df
                df = pd.DataFrame({'split': ['valid']})  # Dummy df
                print(f"  No gene columns needed")

            # Add split column to pass validation check later
            if 'split' not in df.columns:
                df['split'] = 'valid'

        elif data_path.suffix == '.parquet':
            import pyarrow.parquet as pq
            pf = pq.ParquetFile(data_path)
            total_rows = pf.metadata.num_rows
            # Get column names from parquet schema
            all_columns = pf.schema_arrow.names

            # Only read gene columns if needed (for gene vocabulary)
            gene_cols_to_read = []
            v_gene_cols = ['v_gene', 'v_gene_heavy', 'v_gene_light']
            j_gene_cols = ['j_gene', 'j_gene_heavy', 'j_gene_light']
            for col in v_gene_cols + j_gene_cols:
                if col in all_columns:
                    gene_cols_to_read.append(col)

            # Also need to verify cluster_id or split column exists
            if 'cluster_id' in all_columns:
                cols_to_read = ['cluster_id'] + gene_cols_to_read
            elif 'split' in all_columns:
                cols_to_read = ['split'] + gene_cols_to_read
            else:
                raise ValueError("Parquet must contain either 'cluster_id' or 'split' column")

            df_metadata = pd.read_parquet(data_path, columns=cols_to_read)
            print(f"  Total rows in file: {total_rows:,}")
            print(f"  Columns read for metadata: {cols_to_read}")

            # Create a minimal df for gene vocabulary building
            df = df_metadata
        else:
            # For pickle, we unfortunately need to load the full file
            # but we'll delete it after extracting gene vocabulary
            print(f"  Warning: Pickle format requires full load for metadata extraction")
            df = pd.read_pickle(data_path)

        print(f"  Columns available: {list(df.columns)}")
    else:
        # Standard loading: load full dataframe
        if data_path.is_dir():
            # [v31.4] Shard directory - not recommended without lazy_loading
            print(f"WARNING: Using shard directory without lazy_loading is not recommended!")
            print(f"Loading all shards and concatenating (will use lots of memory)...")
            dfs = []
            shard_idx = 0
            while True:
                shard_file = data_path / f"train_shard_{shard_idx}.parquet"
                if not shard_file.exists():
                    break
                dfs.append(pd.read_parquet(shard_file))
                shard_idx += 1
            valid_file = data_path / "valid.parquet"
            test_file = data_path / "test.parquet"
            if valid_file.exists():
                dfs.append(pd.read_parquet(valid_file))
            if test_file.exists():
                dfs.append(pd.read_parquet(test_file))
            df = pd.concat(dfs, ignore_index=True)
        elif data_path.suffix == '.parquet':
            print(f"Loading parquet file (optimized format)...")
            df = pd.read_parquet(data_path)
        else:
            print(f"Loading pickle file...")
            df = pd.read_pickle(data_path)
        print(f"Loaded dataframe with shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")

    # Verify either 'cluster_id' or 'split' column exists for data splitting
    if 'cluster_id' not in df.columns and 'split' not in df.columns:
        raise ValueError("Dataframe must contain either 'cluster_id' or 'split' column for data splitting")

    if 'split' in df.columns:
        print(f"\nDataset will be split using pre-defined 'split' column")
    else:
        print(f"\nDataset will be split using cluster-based strategy in SFTDataModule")

    # =========================================================================
    # [NEW] Build Gene Vocabulary for V/J Gene Conditioning
    # =========================================================================
    # [v33.2] Support loading pre-saved gene vocabulary from JSON file
    # This is required when fine-tuning to match the pretrained model's vocabulary
    gene_vocab = None
    use_germline_genes = config['model'].get('use_germline_genes', False)

    if use_germline_genes:
        print("\n" + "="*80)
        print("Building Gene Vocabulary for V/J Gene Conditioning...")
        print("="*80)

        # [v33.2] Check if a pre-saved gene vocabulary path is specified
        gene_vocab_path = config['model'].get('gene_vocab_path', None)

        if gene_vocab_path:
            # Load gene vocabulary from JSON file
            import json
            gene_vocab_path = Path(gene_vocab_path)

            if not gene_vocab_path.exists():
                raise FileNotFoundError(f"Gene vocabulary file not found: {gene_vocab_path}")

            with open(gene_vocab_path, 'r') as f:
                vocab_data = json.load(f)

            genes = vocab_data['genes']
            gene_vocab = GeneVocabulary(genes=genes)

            print(f"  [v33.2] Loaded pre-saved gene vocabulary from: {gene_vocab_path}")
            print(f"  Source: {vocab_data.get('source', 'unknown')}")
            print(f"  Total genes: {len(genes)}")
            print(f"  Gene vocabulary size: {len(gene_vocab)} (including [PAD] and [UNK])")
            print(f"  [UNK] ID: {gene_vocab.unk_id}")
            print(f"  [PAD] ID: {gene_vocab.pad_id}")
        else:
            # Build gene vocabulary from current dataset
            # Check for gene columns (support both heavy/light specific and general columns)
            v_gene_cols = ['v_gene', 'v_gene_heavy', 'v_gene_light']
            j_gene_cols = ['j_gene', 'j_gene_heavy', 'j_gene_light']

            available_v_cols = [col for col in v_gene_cols if col in df.columns]
            available_j_cols = [col for col in j_gene_cols if col in df.columns]

            if not available_v_cols and not available_j_cols:
                raise ValueError(
                    f"use_germline_genes=True but no gene columns found. "
                    f"Expected one of: {v_gene_cols + j_gene_cols}"
                )

            # Collect all unique genes from all available columns
            all_genes = set()
            for col in available_v_cols + available_j_cols:
                unique_genes = df[col].dropna().unique()
                all_genes.update(unique_genes)
                print(f"  Found {len(unique_genes)} unique genes in column '{col}'")

            # Build gene vocabulary
            gene_vocab = GeneVocabulary(genes=sorted(list(all_genes)))

            print(f"\n  Total unique genes: {len(all_genes)}")
            print(f"  Gene vocabulary size: {len(gene_vocab)} (including [PAD] and [UNK])")
            print(f"  [UNK] ID: {gene_vocab.unk_id}")
            print(f"  [PAD] ID: {gene_vocab.pad_id}")
    else:
        print(f"\nGene Conditioning: Disabled (use_germline_genes=False)")
    print(f"  Seed: {config['training']['seed']}")
    print(f"  Strategy: First 20k clusters → validation, Next 20k → test, Remaining → train")

    # Create data module
    print("\n" + "="*80)
    print("Creating data module...")
    print("="*80)

    # Prepare mask ratio schedule if enabled
    mask_ratio_schedule = None
    if config['model'].get('mask_ratio_schedule', {}).get('enabled', False):
        schedule_config = config['model']['mask_ratio_schedule']
        # Create mutable step counter that will be updated during training
        step_counter = {'n': 0}
        mask_ratio_schedule = {
            'enabled': True,
            'start_ratio': schedule_config['start_ratio'],
            'end_ratio': schedule_config['end_ratio'],
            'start_step': schedule_config['start_step'],
            'end_step': schedule_config['end_step'],
            'current_step': step_counter
        }
        print(f"\nMask Ratio Scheduling:")
        print(f"  Enabled: Yes")
        print(f"  Start ratio: {schedule_config['start_ratio']:.1%} (custom masks)")
        print(f"  End ratio: {schedule_config['end_ratio']:.1%} (custom masks)")
        print(f"  Schedule: steps {schedule_config['start_step']} → {schedule_config['end_step']}")
    else:
        print(f"\nMask Ratio Scheduling: Disabled (using default 50% split)")

    # [NEW] Get NGL-targeted masking settings from config
    ngl_targeted_masking = config['model'].get('ngl_targeted_masking', False)
    ngl_mask_prob = config['model'].get('ngl_mask_prob', 0.8)

    if ngl_targeted_masking:
        print(f"\nNGL-Targeted Masking:")
        print(f"  Enabled: Yes")
        print(f"  GL masking probability: {config['data']['mask_prob']}")
        print(f"  NGL masking probability: {ngl_mask_prob}")
        print(f"  Effect: NGL tokens masked at {ngl_mask_prob*100:.0f}% rate to force context learning")
    else:
        print(f"\nNGL-Targeted Masking: Disabled")

    # [FIX] Get region embedding setting from config
    use_region_embedding = config['model'].get('use_region_embedding', False)

    if use_region_embedding:
        print(f"\nRegion Embedding: ENABLED")
    else:
        print(f"\nRegion Embedding: Disabled")

    # [NEW] Get multihead architecture settings from config
    use_multihead_architecture = config['model'].get('use_multihead_architecture', False)

    if use_multihead_architecture:
        print(f"\n" + "="*60)
        print("MULTIHEAD ARCHITECTURE MODE")
        print("="*60)
        print(f"  AA Loss Weight: {config['model'].get('aa_loss_weight', 1.0)}")
        print(f"  Mutation Loss Weight: {config['model'].get('mut_loss_weight', 5.0)}")
        print(f"  Mutation Focal Gamma: {config['model'].get('mut_focal_gamma', 2.0)}")
        print(f"  Input Processing: All tokens forced to UPPERCASE")
        print(f"  Output Heads: AA Identity + Binary Mutation")
        print("="*60)

    # =========================================================================
    # [NEW v31] Choose DataModule based on lazy_loading setting
    # LazyShardedDataModule is required for large datasets with DDP to avoid
    # memory duplication when processes fork (solves 2x memory issue)
    # (use_lazy_loading was already set at the top of main())
    # =========================================================================
    enable_sharding = config['data'].get('enable_sharding', True)

    if use_lazy_loading:
        print("\n" + "="*80)
        print("[v31] LAZY SHARDED DATA LOADING MODE")
        print("="*80)
        print(f"  Lazy loading: ENABLED (data loaded in setup() after DDP spawn)")
        print(f"  DDP sharding: {'ENABLED' if enable_sharding else 'DISABLED'}")
        print(f"  Benefit: Each GPU loads only its shard → ~50% memory per process")
        print("="*80 + "\n")

        # Free the pre-loaded DataFrame - it will be loaded lazily in each process
        del df
        import gc
        gc.collect()

        data_module = LazyShardedDataModule(
            data_path=str(data_path),  # Pass path, not DataFrame!
            batch_size=config['data']['batch_size'],
            mask_prob=config['data']['mask_prob'],
            tokenizer=tokenizer,
            seed=config['training']['seed'],
            num_workers=config['data']['num_workers'],
            gene_vocab=gene_vocab,
            use_germline_genes=use_germline_genes,
            ngl_targeted_masking=ngl_targeted_masking,
            ngl_mask_prob=ngl_mask_prob,
            use_region_embedding=use_region_embedding,
            ngl_mask_schedule=config['model'].get('ngl_mask_schedule', None),
            # Region-wise masking
            use_region_masking=config['model'].get('use_region_masking', False),
            cdr_mask_prob=config['model'].get('cdr_mask_prob', 0.4),
            fr_mask_prob=config['model'].get('fr_mask_prob', 0.15),
            # FR Span masking
            use_fr_span_masking=config['model'].get('use_fr_span_masking', False),
            fr_span_min_length=config['model'].get('fr_span_min_length', 3),
            fr_span_max_length=config['model'].get('fr_span_max_length', 6),
            # DDP sharding control
            enable_sharding=enable_sharding,
            # Validation sampling to reduce 1hr+ validation to ~6 minutes
            val_sample_ratio=config['data'].get('val_sample_ratio', 0.1),
            # [v35.1] Combined 4-rate masking
            cdr_ngl_mask_prob=config['model'].get('cdr_ngl_mask_prob', None),
            fr_ngl_mask_prob=config['model'].get('fr_ngl_mask_prob', None),
            # [v37] Coherence masking
            use_coherence_masking=config['model'].get('use_coherence_masking', False),
            coherence_prob=config['model'].get('coherence_prob', 0.3),
            coherence_ngl_mask_prob=config['model'].get('coherence_ngl_mask_prob', 0.5),
            # [v38] 3-class origin labels
            use_3class_origin=config['model'].get('use_3class_origin', False),
            synthetic_ngl_data_path=config['data'].get('synthetic_ngl_data_path', None),
            # [v40] SynNGL auxiliary signals
            use_synth_masking=config['model'].get('use_synth_masking', False),
            cdr_synth_mask_prob=config['model'].get('cdr_synth_mask_prob', 0.45),
            fr_synth_mask_prob=config['model'].get('fr_synth_mask_prob', 0.30),
            use_mpnn_origin_smoothing=config['model'].get('use_mpnn_origin_smoothing', False),
            origin_label_smooth_factor=config['model'].get('origin_label_smooth_factor', 0.2),
            # [rebuttal] NGL label noise injection
            ngl_label_noise_flips=config['data'].get('ngl_label_noise_flips', 0),
        )
    else:
        # [FIX] Pass the pre-configured tokenizer to SFTDataModule
        # This ensures the collate functions see the correct token IDs for NGL-targeted masking
        data_module = SFTDataModule(
            data_frame=df,
            batch_size=config['data']['batch_size'],
            mask_prob=config['data']['mask_prob'],
            tokenizer=tokenizer,  # Pass pre-configured tokenizer
            seed=config['training']['seed'],
            num_workers=config['data']['num_workers'],
            # Gene vocabulary for V/J gene conditioning
            gene_vocab=gene_vocab,
            use_germline_genes=use_germline_genes,
            # NGL-targeted masking
            ngl_targeted_masking=ngl_targeted_masking,
            ngl_mask_prob=ngl_mask_prob,
            # Region embedding
            use_region_embedding=use_region_embedding,
            # NGL masking schedule
            ngl_mask_schedule=config['model'].get('ngl_mask_schedule', None),
            # Region-wise masking
            use_region_masking=config['model'].get('use_region_masking', False),
            cdr_mask_prob=config['model'].get('cdr_mask_prob', 0.4),
            fr_mask_prob=config['model'].get('fr_mask_prob', 0.15),
            # FR Span masking
            use_fr_span_masking=config['model'].get('use_fr_span_masking', False),
            fr_span_min_length=config['model'].get('fr_span_min_length', 3),
            fr_span_max_length=config['model'].get('fr_span_max_length', 6),
            # [v35.1] Combined 4-rate masking
            cdr_ngl_mask_prob=config['model'].get('cdr_ngl_mask_prob', None),
            fr_ngl_mask_prob=config['model'].get('fr_ngl_mask_prob', None),
            # [v38] 3-class origin labels
            use_3class_origin=config['model'].get('use_3class_origin', False),
        )

    # Calculate gradient accumulation steps if set to -1
    # [DDP] Account for number of GPUs in effective batch size calculation
    gradient_accumulation_steps = config['training']['gradient_accumulation_steps']
    batch_size = config['data']['batch_size']

    # Get number of devices for DDP
    # [DDP] Use config value for batch size calculation, let PyTorch Lightning handle actual GPU detection
    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')

    requested_devices = config['trainer']['devices']
    if isinstance(requested_devices, list):
        num_devices = len(requested_devices)
    elif requested_devices == -1 or requested_devices == "auto":
        # Auto-detect available GPUs
        num_devices = max(1, available_gpus)
        # Update config so PyTorch Lightning uses correct value
        config['trainer']['devices'] = num_devices
    else:
        num_devices = requested_devices

    strategy = config['trainer'].get('strategy', 'auto')

    # [DDP] Auto-select ddp strategy when multiple devices requested
    if num_devices > 1 and strategy == 'auto':
        strategy = 'ddp'
        print(f"\n  [INFO] Multiple GPUs requested ({num_devices}), auto-selecting strategy='ddp'")

    print("\n" + "="*80)
    print("Distributed Training Configuration")
    print("="*80)
    print(f"  CUDA_VISIBLE_DEVICES: {cuda_visible}")
    print(f"  Available GPUs (torch.cuda.device_count()): {available_gpus}")
    print(f"  Requested devices: {num_devices}")
    print(f"  Strategy: {strategy}")

    if gradient_accumulation_steps == -1:
        # Auto-calculate: 8192 / (batch_size * num_devices) to maintain effective batch size of 8192
        # Formula: effective_batch_size = batch_size * grad_accum * num_gpus
        gradient_accumulation_steps = max(1, 8192 // (batch_size * num_devices))
        effective_batch_size = batch_size * gradient_accumulation_steps * num_devices
        print(f"\n  Gradient Accumulation Auto-Configuration:")
        print(f"    Target effective batch size: 8192")
        print(f"    Per-GPU batch size: {batch_size}")
        print(f"    Number of GPUs: {num_devices}")
        print(f"    Auto-calculated gradient_accumulation_steps: {gradient_accumulation_steps}")
        print(f"    Effective batch size: {batch_size} × {gradient_accumulation_steps} × {num_devices} = {effective_batch_size}")
    else:
        effective_batch_size = batch_size * gradient_accumulation_steps * num_devices
        print(f"\n  Gradient Accumulation Configuration:")
        print(f"    Per-GPU batch size: {batch_size}")
        print(f"    Number of GPUs: {num_devices}")
        print(f"    Gradient accumulation steps: {gradient_accumulation_steps}")
        print(f"    Effective batch size: {batch_size} × {gradient_accumulation_steps} × {num_devices} = {effective_batch_size}")

    # Convert eval_steps from global steps to batch steps for PyTorch Lightning
    # PyTorch Lightning's val_check_interval expects batch steps, not global steps
    eval_steps_global = config['training']['eval_steps']
    eval_steps_batches = eval_steps_global * gradient_accumulation_steps
    print(f"\n  Validation interval: every {eval_steps_global} global steps ({eval_steps_batches} batches)")

    # Create model
    # [FIX] Pass the same pre-configured tokenizer to ensure consistency
    print("\n" + "="*80)
    print("Creating model...")
    print("="*80)
    model = SFT_ESM2(
        seed=config['training']['seed'],
        model_identifier=config['model']['model_identifier'],
        tokenizer=tokenizer,  # [FIX] Pass pre-configured tokenizer
        peak_learning_rate=config['training']['peak_learning_rate'],
        adam_beta1=config['training']['adam_beta1'],
        adam_beta2=config['training']['adam_beta2'],
        adam_epsilon=config['training']['adam_epsilon'],
        WD=config['training']['weight_decay'],
        warmup_steps=config['training']['warmup_steps'],
        max_steps=config['training']['max_steps'],
        logging_steps=config['training']['logging_steps'],
        eval_steps=config['training']['eval_steps'],
        batch_size=config['data']['batch_size'],
        mask_prob=config['data']['mask_prob'],
        num_unfrozen_transformer_blocks=config['model']['num_unfrozen_transformer_blocks'],
        loss_type=config['training']['loss_type'],
        random_weights=config['model']['random_weights'],
        add_custom_tokens=config['model']['add_custom_tokens'],
        custom_token_strategy=config['model']['custom_token_strategy'],
        activation_function=config['model']['activation_function'],
        fix_swiglu_double_activation=config['model'].get('fix_swiglu_double_activation', True),
        # V/J Gene Conditioning
        use_germline_genes=use_germline_genes,
        num_genes=len(gene_vocab) if gene_vocab is not None else 0,
        gene_embedding_dim=config['model'].get('gene_embedding_dim', 64),
        gene_embedding_dropout=config['model'].get('gene_embedding_dropout', 0.1),
        # Region Embedding
        use_region_embedding=use_region_embedding,
        num_regions=config['model'].get('num_regions', 8),
        region_embedding_dim=config['model'].get('region_embedding_dim', 32),
        # Asymmetric Input/Output Strategy - Weight Tying Control
        tie_word_embeddings=config['model'].get('tie_word_embeddings', True),
        # Multihead Architecture (AA + Mutation)
        use_multihead_architecture=use_multihead_architecture,
        aa_loss_weight=config['model'].get('aa_loss_weight', 1.0),
        mut_loss_weight=config['model'].get('mut_loss_weight', 5.0),
        mut_focal_gamma=config['model'].get('mut_focal_gamma', 2.0),
        # Region-aware Alpha Gating
        use_alpha_gating=config['model'].get('use_alpha_gating', False),
        fixed_alpha_value=config['model'].get('fixed_alpha_value', None),
        final_loss_weight=config['model'].get('final_loss_weight', 1.0),
        aa_focal_gamma=config['model'].get('aa_focal_gamma', 2.0),
        origin_focal_gamma=config['model'].get('origin_focal_gamma', 2.0),
        # NGL Loss Reweighting
        ngl_loss_alpha=config['model'].get('ngl_loss_alpha', 10.0),
        # Multiplicative Gating
        use_multiplicative_gating=config['model'].get('use_multiplicative_gating', False),
        gating_temperature=config['model'].get('gating_temperature', 1.0),
        gating_temperature_warmup_steps=config['model'].get('gating_temperature_warmup_steps', 0),
        # Sequential Detach Architecture
        detach_origin_gradient=config['model'].get('detach_origin_gradient', True),
        # Origin head dropout for regularization
        origin_head_dropout=config['model'].get('origin_head_dropout', 0.1),
        # Soft AA Learning - Allow NGL positions in AA loss with reduced weight
        aa_loss_ngl_weight=config['model'].get('aa_loss_ngl_weight', 0.0),
        # Region-Balanced Loss - Equalize FR and CDR contribution to loss
        use_region_balanced_loss=config['model'].get('use_region_balanced_loss', False),
        # CDR-Targeted Loss Boosting
        use_cdr_loss_boosting=config['model'].get('use_cdr_loss_boosting', False),
        cdr_loss_multiplier=config['model'].get('cdr_loss_multiplier', 3.0),
        # Dual AA Heads (v35/v36)
        use_dual_aa_heads=config['model'].get('use_dual_aa_heads', False),
        dual_aa_heads_conditioned=config['model'].get('dual_aa_heads_conditioned', True),
        # [v35.1] Asymmetric NGL AA head loss weight
        ngl_aa_loss_weight=config['model'].get('ngl_aa_loss_weight', None),
        # [v35.1] Detach head outputs from final loss to eliminate gradient competition
        detach_heads_from_final_loss=config['model'].get('detach_heads_from_final_loss', False),
        # [v35.1b] Label smoothing for NGL AA head to prevent overconfidence
        ngl_label_smoothing=config['model'].get('ngl_label_smoothing', 0.0),
        # [v37] GL-NGL Divergence Loss
        divergence_loss_weight=config['model'].get('divergence_loss_weight', 0.0),
        divergence_warmup_steps=config['model'].get('divergence_warmup_steps', 2000),
        max_kl_divergence=config['model'].get('max_kl_divergence', 10.0),
        divergence_type=config['model'].get('divergence_type', 'kl'),
        # [v37] SHM-Based Sample Weighting
        use_shm_weighting=config['model'].get('use_shm_weighting', False),
        shm_beta=config['model'].get('shm_beta', 1.0),
        shm_mean_ngl=config['model'].get('shm_mean_ngl', 15.0),
        # [v38] 3-Class Origin Head (GL / SynNGL / NGL)
        num_origin_classes=config['model'].get('num_origin_classes', 2),
        synth_weight=config['model'].get('synth_weight', 0.5),
        origin_class_weights=config['model'].get('origin_class_weights', None),
        # [v40] SynNGL Auxiliary Signals
        use_synth_divergence=config['model'].get('use_synth_divergence', False),
        synth_div_weight=config['model'].get('synth_div_weight', 0.3),
        use_mpnn_gl_weighting=config['model'].get('use_mpnn_gl_weighting', False),
        mpnn_min_weight=config['model'].get('mpnn_min_weight', 0.3),
    )

    # =========================================================================
    # [v32.1] Load pretrained weights for fine-tuning (if specified)
    # =========================================================================
    # This loads ONLY model weights, not optimizer state, allowing fine-tuning
    # with a fresh optimizer (e.g., lower learning rate for continued training)
    pretrained_ckpt_path = config.get('pretrained_checkpoint_path', None)
    if pretrained_ckpt_path:
        pretrained_ckpt_path = Path(pretrained_ckpt_path)
        print("\n" + "="*80)
        print("Loading pretrained weights for fine-tuning...")
        print("="*80)

        if not pretrained_ckpt_path.exists():
            raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrained_ckpt_path}")

        print(f"  Checkpoint: {pretrained_ckpt_path}")

        # Load checkpoint
        checkpoint = torch.load(pretrained_ckpt_path, map_location='cpu')
        state_dict = checkpoint['state_dict']

        # Remove non-persistent buffers if present
        if 'aa_indices' in state_dict:
            del state_dict['aa_indices']
            print("  Removed 'aa_indices' buffer (non-persistent)")

        # Verify vocabulary sizes match
        model_vocab_size = len(model.tokenizer)
        checkpoint_vocab_size = state_dict['ESM2.lm_head.decoder.weight'].shape[0]

        print(f"  Model vocab size: {model_vocab_size}")
        print(f"  Checkpoint vocab size: {checkpoint_vocab_size}")

        if model_vocab_size != checkpoint_vocab_size:
            raise ValueError(
                f"Vocabulary size mismatch! Model: {model_vocab_size}, Checkpoint: {checkpoint_vocab_size}\n"
                f"Ensure config settings (add_custom_tokens, custom_token_strategy) match the pretrained model."
            )

        # Load state dict (model weights only, no optimizer state)
        # Use strict=False to allow loading older checkpoints that may have
        # missing or extra parameters (they'll use initialized defaults)
        # [v38] Filter out shape-mismatched keys (e.g. 2-class→3-class mut_head)
        model_state = model.state_dict()
        keys_to_skip = []
        for k, v in state_dict.items():
            if k in model_state and v.shape != model_state[k].shape:
                keys_to_skip.append(k)
                print(f"  ℹ Skipping shape-mismatched key: {k} "
                      f"(ckpt: {tuple(v.shape)} → model: {tuple(model_state[k].shape)})")
        for k in keys_to_skip:
            del state_dict[k]
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print(f"  ✓ Pretrained weights loaded successfully")
        if missing_keys:
            print(f"  ℹ Missing keys (using initialized defaults): {missing_keys}")
        if unexpected_keys:
            print(f"  ℹ Unexpected keys (ignored): {unexpected_keys}")
        print(f"  Note: Optimizer will start fresh with LR={config['training']['peak_learning_rate']}")

        # [v35.1] Copy AA head weights to NGL head for warm-start dual heads
        copy_aa_to_ngl = config.get('model', {}).get('copy_aa_head_to_ngl', False)
        if copy_aa_to_ngl and model.use_dual_aa_heads:
            print(f"\n  [v35.1] Copying AA head weights → NGL AA head (warm start)...")
            with torch.no_grad():
                model.ngl_aa_head_dense.weight.copy_(model.aa_head_dense.weight)
                model.ngl_aa_head_dense.bias.copy_(model.aa_head_dense.bias)
                model.ngl_aa_head_layer_norm.weight.copy_(model.aa_head_layer_norm.weight)
                model.ngl_aa_head_layer_norm.bias.copy_(model.aa_head_layer_norm.bias)
                model.ngl_aa_head_decoder.weight.copy_(model.aa_head_decoder.weight)
                model.ngl_aa_head_bias.copy_(model.aa_head_bias)
            print(f"  ✓ NGL AA head initialized from AA head weights")

        # Clean up
        del checkpoint
        del state_dict
        torch.cuda.empty_cache()

    # =========================================================================
    # [v31.2] torch.compile for 10-30% training speedup (PyTorch 2.0+)
    # =========================================================================
    use_compile = config['trainer'].get('compile', False)
    if use_compile:
        compile_mode = config['trainer'].get('compile_mode', 'reduce-overhead')
        print(f"\n" + "="*80)
        print(f"Applying torch.compile (mode='{compile_mode}')...")
        print("="*80)
        print(f"  This may take a few minutes for initial compilation...")
        print(f"  Expected speedup: 10-30% faster training")

        # Configure torch.compile for better performance
        import torch._dynamo as dynamo
        dynamo.config.suppress_errors = True  # Continue on graph breaks
        dynamo.config.cache_size_limit = 256  # Increase cache for large models

        # Compile the underlying ESM2 model (not the Lightning module)
        # This is more compatible with DDP
        model.ESM2 = torch.compile(model.ESM2, mode=compile_mode, dynamic=False)
        print(f"  ✓ ESM2 backbone compiled successfully")

    # Setup logging and checkpointing
    print("\n" + "="*80)
    print("Setting up logging and checkpointing...")
    print("="*80)
    output_dir = Path(config['logging']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create descriptive experiment name
    # Format: {experiment_name}_{model_identifier}_{custom_tokens}_{num_unfrozen}_{lr}_{batch_size}
    custom_tokens_str = "custom" if config['model']['add_custom_tokens'] else "standard"
    lr_str = f"{config['training']['peak_learning_rate']:.0e}".replace('e-0', 'e-').replace('e+0', 'e+')

    experiment_folder_name = (
        f"{config['logging']['experiment_name']}_"
        f"{config['model']['model_identifier']}_"
        f"{custom_tokens_str}_"
        f"unfrozen{config['model']['num_unfrozen_transformer_blocks']}_"
        f"lr{lr_str}_"
        f"bs{config['data']['batch_size']}"
    )

    print(f"\n  Experiment folder name: {experiment_folder_name}")

    # TensorBoard logger
    logger = TensorBoardLogger(
        save_dir=output_dir,
        name=experiment_folder_name
    )
    print(f"  TensorBoard logs will be saved to: {logger.log_dir}")

    # Checkpoint callbacks - configurable via checkpointing.monitor
    top_k = config['checkpointing']['top_k']
    ckpt_dir = output_dir / experiment_folder_name / 'checkpoints'

    # Get monitor metric from config (default to val/ppl_ngl_upper)
    # NOTE: val/ppl_ngl_mixed is deprecated - not always available in multihead architectures
    ckpt_monitor = config['checkpointing'].get('monitor', 'val/ppl_ngl_upper')

    # Determine checkpointing strategy based on monitor metric
    if ckpt_monitor == 'val/Final_PPL_NGL':
        # Single checkpoint strategy for NGL-focused training
        # Monitor val/Final_PPL_NGL - the priority metric for NGL prediction performance
        checkpoint_callback = ModelCheckpoint(
            dirpath=ckpt_dir,
            filename='val_Final_PPL_NGL-{epoch:02d}-{val/Final_PPL_NGL:.4f}',
            monitor='val/Final_PPL_NGL',
            mode='min',
            save_top_k=top_k,
            save_last=True,
            verbose=True,
            auto_insert_metric_name=False
        )
        callbacks = [checkpoint_callback]
        print(f"\n  SINGLE CHECKPOINTING STRATEGY:")
        print(f"  Saving top-{top_k} based on val/Final_PPL_NGL (NGL Priority)")

    elif ckpt_monitor == 'val/Final_PPL_All':
        # Single checkpoint strategy for balanced training (v33.4+)
        # Monitor val/Final_PPL_All - overall perplexity for GL+NGL balanced learning
        checkpoint_callback = ModelCheckpoint(
            dirpath=ckpt_dir,
            filename='val_Final_PPL_All-{epoch:02d}-{val/Final_PPL_All:.4f}',
            monitor='val/Final_PPL_All',
            mode='min',
            save_top_k=top_k,
            save_last=True,
            verbose=True,
            auto_insert_metric_name=False
        )
        callbacks = [checkpoint_callback]
        print(f"\n  SINGLE CHECKPOINTING STRATEGY (Balanced):")
        print(f"  Saving top-{top_k} based on val/Final_PPL_All (Overall PPL)")

    elif ckpt_monitor == 'dual_balanced':
        # [v34] Dual checkpoint strategy for balanced multihead training
        # Saves best checkpoints for BOTH PPL_All and PPL_NGL
        # - PPL_All best: Use for developability prediction tasks
        # - PPL_NGL best: Use if NGL prediction quality matters
        checkpoint_ppl_all = ModelCheckpoint(
            dirpath=ckpt_dir,
            filename='val_Final_PPL_All-{epoch:02d}-{val/Final_PPL_All:.4f}',
            monitor='val/Final_PPL_All',
            mode='min',
            save_top_k=top_k,
            save_last=True,
            verbose=True,
            auto_insert_metric_name=False
        )
        checkpoint_ppl_ngl = ModelCheckpoint(
            dirpath=ckpt_dir,
            filename='val_Final_PPL_NGL-{epoch:02d}-{val/Final_PPL_NGL:.4f}',
            monitor='val/Final_PPL_NGL',
            mode='min',
            save_top_k=top_k,
            save_last=False,  # Only one callback needs save_last
            verbose=True,
            auto_insert_metric_name=False
        )
        callbacks = [checkpoint_ppl_all, checkpoint_ppl_ngl]
        print(f"\n  [v34] DUAL BALANCED CHECKPOINTING STRATEGY:")
        print(f"  Checkpoint A: Saving top-{top_k} based on val/Final_PPL_All (Developability)")
        print(f"  Checkpoint B: Saving top-{top_k} based on val/Final_PPL_NGL (NGL Quality)")

    elif ckpt_monitor == 'val/ppl_ngl_upper':
        # Single checkpoint strategy for inference utility
        checkpoint_callback = ModelCheckpoint(
            dirpath=ckpt_dir,
            filename='val_ppl_upper-{epoch:02d}-{val/ppl_ngl_upper:.4f}',
            monitor='val/ppl_ngl_upper',
            mode='min',
            save_top_k=top_k,
            save_last=True,
            verbose=True,
            auto_insert_metric_name=False
        )
        callbacks = [checkpoint_callback]
        print(f"\n  SINGLE CHECKPOINTING STRATEGY:")
        print(f"  Saving top-{top_k} based on val/ppl_ngl_upper (Inference Utility)")

    elif ckpt_monitor == 'val/ppl_all':
        # [ABLATION] Single checkpoint strategy for simple head (non-multihead) models
        # Used in ablation studies where multihead architecture is disabled
        checkpoint_callback = ModelCheckpoint(
            dirpath=ckpt_dir,
            filename='val_ppl_all-{epoch:02d}-{val/ppl_all:.4f}',
            monitor='val/ppl_all',
            mode='min',
            save_top_k=top_k,
            save_last=True,
            verbose=True,
            auto_insert_metric_name=False
        )
        callbacks = [checkpoint_callback]
        print(f"\n  SINGLE CHECKPOINTING STRATEGY (Simple Head):")
        print(f"  Saving top-{top_k} based on val/ppl_all (Standard PPL)")

    elif ckpt_monitor == 'val/NGL_PPL_All':
        # [v38] PRISM pretrain stage - monitor NGL head perplexity
        checkpoint_callback = ModelCheckpoint(
            dirpath=ckpt_dir,
            filename='val_NGL_PPL_All-{epoch:02d}-{val/NGL_PPL_All:.4f}',
            monitor='val/NGL_PPL_All',
            mode='min',
            save_top_k=top_k,
            save_last=True,
            verbose=True,
            auto_insert_metric_name=False
        )
        callbacks = [checkpoint_callback]
        print(f"\n  [v38] PRISM PRETRAIN CHECKPOINTING:")
        print(f"  Saving top-{top_k} based on val/NGL_PPL_All (NGL Head Performance)")

    elif ckpt_monitor == 'val/Origin_F1':
        # [v38.1a] PRISM finetune with Origin F1 monitoring (no final_loss experiment)
        checkpoint_callback = ModelCheckpoint(
            dirpath=ckpt_dir,
            filename='val_Origin_F1-{epoch:02d}-{val/Origin_F1:.4f}',
            monitor='val/Origin_F1',
            mode='max',  # Higher F1 is better
            save_top_k=top_k,
            save_last=True,
            verbose=True,
            auto_insert_metric_name=False
        )
        callbacks = [checkpoint_callback]
        print(f"\n  [v38.1a] PRISM FINETUNE CHECKPOINTING (Origin F1):")
        print(f"  Saving top-{top_k} based on val/Origin_F1 (Origin Head Classification)")

    elif ckpt_monitor == 'val/Final_PPL_All':
        # [v38.1] PRISM finetune stage - monitor Final head perplexity
        checkpoint_callback = ModelCheckpoint(
            dirpath=ckpt_dir,
            filename='val_Final_PPL_All-{epoch:02d}-{val/Final_PPL_All:.4f}',
            monitor='val/Final_PPL_All',
            mode='min',
            save_top_k=top_k,
            save_last=True,
            verbose=True,
            auto_insert_metric_name=False
        )
        callbacks = [checkpoint_callback]
        print(f"\n  [v38.1] PRISM FINETUNE CHECKPOINTING:")
        print(f"  Saving top-{top_k} based on val/Final_PPL_All (Final Head Performance)")

    else:
        # Dual checkpointing strategy (default)
        # Checkpoint A: Monitor val/Final_PPL_All (Overall Performance)
        checkpoint_val_ppl_all = ModelCheckpoint(
            dirpath=ckpt_dir,
            filename='val_Final_PPL_All-{epoch:02d}-{val/Final_PPL_All:.4f}',
            monitor='val/Final_PPL_All',
            mode='min',
            save_top_k=top_k,
            save_last=True,
            verbose=True,
            auto_insert_metric_name=False
        )
        # Checkpoint B: Monitor val/ppl_ngl_upper (Inference Utility - PRIORITY)
        checkpoint_val_ppl_upper = ModelCheckpoint(
            dirpath=ckpt_dir,
            filename='val_ppl_upper-{epoch:02d}-{val/ppl_ngl_upper:.4f}',
            monitor='val/ppl_ngl_upper',
            mode='min',
            save_top_k=top_k,
            save_last=False,
            verbose=True,
            auto_insert_metric_name=False
        )
        callbacks = [checkpoint_val_ppl_all, checkpoint_val_ppl_upper]
        print(f"\n  DUAL CHECKPOINTING STRATEGY:")
        print(f"  Checkpoint A: Saving top-{top_k} based on val/Final_PPL_All (Overall Performance)")
        print(f"  Checkpoint B: Saving top-{top_k} based on val/ppl_ngl_upper (Inference Utility)")

    # Add mask ratio scheduler callback if enabled
    if mask_ratio_schedule is not None and mask_ratio_schedule['enabled']:
        scheduler_callback = MaskRatioSchedulerCallback(step_counter)
        callbacks.append(scheduler_callback)
        print(f"  Mask ratio scheduler callback added")

    # [CHANGE v17] Add NGL masking schedule callback if enabled
    ngl_mask_schedule = config['model'].get('ngl_mask_schedule', None)
    if ngl_mask_schedule is not None and ngl_mask_schedule.get('enabled', False):
        ngl_schedule_callback = NGLMaskScheduleCallback(data_module)
        callbacks.append(ngl_schedule_callback)
        print(f"  [v17] NGL masking schedule callback added")
        print(f"        Schedule: {ngl_mask_schedule.get('start_prob', 0.8)} -> {ngl_mask_schedule.get('end_prob', 0.5)}")
        print(f"        Steps: {ngl_mask_schedule.get('start_step', 0)} -> {ngl_mask_schedule.get('end_step', 3000)}")

    # Early stopping (optional, controlled by config)
    # [CHANGE v4.0] Default to monitoring val/ppl_ngl_upper (Inference Utility - PRIORITY)
    if config['early_stopping']['enabled']:
        # Determine the monitor metric - default to val/ppl_ngl_upper for real-world utility
        early_stop_monitor = config['early_stopping'].get('monitor', 'val/ppl_ngl_upper')

        # Override to val/ppl_ngl_upper if the config has old metric name
        if early_stop_monitor == 'val/ppl_ngl':
            early_stop_monitor = 'val/ppl_ngl_upper'
            print(f"  [NOTE] Overriding early_stopping.monitor from 'val/ppl_ngl' to 'val/ppl_ngl_upper'")

        # Determine mode: use config value if present, otherwise auto-detect from metric name
        # [FIX v38] Use case-insensitive matching for auto-detection
        early_stop_mode = config['early_stopping'].get('mode', None)
        if early_stop_mode is None:
            monitor_lower = early_stop_monitor.lower()
            if 'ppl' in monitor_lower or 'loss' in monitor_lower:
                early_stop_mode = 'min'  # Lower is better for perplexity and loss
            else:
                early_stop_mode = 'max'  # Higher is better for accuracy, F1, etc.

        early_stop_callback = EarlyStopping(
            monitor=early_stop_monitor,
            patience=config['early_stopping']['patience'],
            mode=early_stop_mode,
            verbose=True
        )
        callbacks.append(early_stop_callback)
        print(f"  Early stopping enabled: monitor={early_stop_monitor}, patience={config['early_stopping']['patience']}, mode={early_stop_mode}")
    else:
        print("  Early stopping disabled")

    # [NEW v34] Add developability tracking callback if enabled
    dev_config = config.get('developability_tracking', None)
    if dev_config is not None and dev_config.get('enabled', False):
        dev_data_path = dev_config.get('data_path', 'data/ginkgo/developability_data.csv')
        dev_eval_every_n = dev_config.get('eval_every_n', 10)
        dev_sample_size = dev_config.get('sample_size', 100)

        developability_callback = DevelopabilityCallback(
            data_path=dev_data_path,
            eval_every_n=dev_eval_every_n,
            sample_size=dev_sample_size,
            num_masks=3,
            batch_size=8,
        )
        callbacks.append(developability_callback)
        print(f"  [v34] Developability tracking enabled:")
        print(f"        Data: {dev_data_path}")
        print(f"        Eval every {dev_eval_every_n} validation epochs")

        # [v34.1] Add checkpoints for each developability property (best by correlation)
        # Higher r = better model (after PPL inversion in callback), so mode='max'
        dev_properties = ['HIC', 'PR_CHO', 'AC_SINS', 'Tm2', 'Titer']
        print(f"  [v34.1] Developability checkpointing enabled:")
        for prop in dev_properties:
            dev_checkpoint = ModelCheckpoint(
                dirpath=ckpt_dir,
                filename=f'best_dev_{prop}-{{val/dev_r_{prop}:.4f}}',
                monitor=f'val/dev_r_{prop}',
                mode='max',  # Higher correlation = better
                save_top_k=1,
                save_last=False,
                verbose=True,
                auto_insert_metric_name=False
            )
            callbacks.append(dev_checkpoint)
            print(f"          Checkpoint: best_dev_{prop} (monitor: val/dev_r_{prop})")

    # Create trainer
    # [DDP] Configure strategy for distributed training
    print("\n" + "="*80)
    print("Creating trainer...")
    print("="*80)

    # Build trainer kwargs
    trainer_kwargs = {
        'max_steps': config['training']['max_steps'],
        'accelerator': config['trainer']['accelerator'],
        'devices': config['trainer']['devices'],
        'precision': config['trainer']['precision'],
        'accumulate_grad_batches': gradient_accumulation_steps,
        'logger': logger,
        'callbacks': callbacks,
        'log_every_n_steps': config['training']['logging_steps'],
        'val_check_interval': eval_steps_batches,  # Use converted batch steps
        'gradient_clip_val': config['training']['gradient_clip_val'],
        'gradient_clip_algorithm': "norm",
    }

    # Add strategy if specified (for DDP/FSDP)
    # [v31.4] DDPStrategy without static_graph (causes issues with multihead architecture)
    if strategy != 'auto':
        if strategy in ['ddp', 'ddp_find_unused_parameters_true']:
            from pytorch_lightning.strategies import DDPStrategy
            use_find_unused = (strategy == 'ddp_find_unused_parameters_true')
            trainer_kwargs['strategy'] = DDPStrategy(
                find_unused_parameters=use_find_unused,
                # static_graph=False: Multihead architecture has dynamic graph due to
                # conditional logic (alpha gating, origin head, etc.) that changes gradients
                static_graph=False,
                gradient_as_bucket_view=True,  # Reduce memory copies
                bucket_cap_mb=25,  # Gradient bucket size tuning
            )
            print(f"  [DDP Optimization] static_graph=False, gradient_as_bucket_view=True")
        else:
            trainer_kwargs['strategy'] = strategy

    trainer = pl.Trainer(**trainer_kwargs)

    print(f"\nTrainer configuration:")
    print(f"  Max steps: {config['training']['max_steps']} (training controlled by steps, not epochs)")
    print(f"  Strategy: {strategy}")
    print(f"  Number of devices: {num_devices}")
    print(f"  Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"  Effective batch size: {effective_batch_size}")
    print(f"  Accelerator: {config['trainer']['accelerator']}")
    print(f"  Precision: {config['trainer']['precision']}")
    print(f"  Output directory: {output_dir / experiment_folder_name}")

    # Start training
    print("\n" + "="*80)
    print("Starting training...")
    print("="*80)

    # =========================================================================
    # [v31.4] Auto-resume from last checkpoint if training was interrupted
    # [FIX v38] Check if checkpoint training was already completed/stopped
    # =========================================================================
    last_ckpt_path = ckpt_dir / "last.ckpt"
    if last_ckpt_path.exists():
        # Peek at checkpoint to check if training is actually resumable
        _ckpt_meta = torch.load(str(last_ckpt_path), map_location='cpu')
        _ckpt_step = _ckpt_meta.get('global_step', 0)
        _max_steps = config['training']['max_steps']

        # Check if EarlyStopping had already stopped the previous run
        _es_stopped = False
        _ckpt_callbacks = _ckpt_meta.get('callbacks', {})
        for _cb_key, _cb_state in _ckpt_callbacks.items():
            if 'EarlyStopping' in str(_cb_key) and isinstance(_cb_state, dict):
                if _cb_state.get('stopped_epoch', -1) >= 0 and _cb_state.get('wait_count', 0) >= _cb_state.get('patience', 999):
                    _es_stopped = True
                    break
        del _ckpt_meta  # Free memory

        if _ckpt_step >= _max_steps:
            print(f"\n  [AUTO-RESUME] Found last checkpoint at step {_ckpt_step}, but max_steps={_max_steps} already reached.")
            print(f"  Starting fresh training (ignoring stale checkpoint).")
            trainer.fit(model, data_module)
        elif _es_stopped:
            print(f"\n  [AUTO-RESUME] Found last checkpoint at step {_ckpt_step}, but EarlyStopping had already triggered.")
            print(f"  Starting fresh training (ignoring stale checkpoint).")
            print(f"  To force resume, delete {last_ckpt_path} and re-run.")
            trainer.fit(model, data_module)
        else:
            print(f"\n  [AUTO-RESUME] Found last checkpoint: {last_ckpt_path}")
            print(f"  Resuming training from step {_ckpt_step} (max_steps={_max_steps})...")
            trainer.fit(model, data_module, ckpt_path=str(last_ckpt_path))
    else:
        print(f"\n  Starting fresh training (no checkpoint found)")
        trainer.fit(model, data_module)

    # Test evaluation on BEST checkpoint
    print("\n" + "="*80)
    print(f"Running test evaluation on BEST checkpoint (top-1 by {ckpt_monitor})...")
    print("="*80)

    # Get the best checkpoint path based on strategy
    if ckpt_monitor == 'val/Final_PPL_NGL':
        # [NEW v6.0] Single checkpoint strategy for alpha gating
        best_ckpt_path = checkpoint_callback.best_model_path if checkpoint_callback.best_model_path else None
        if best_ckpt_path:
            print(f"\nBest checkpoint by val/Final_PPL_NGL (Alpha Gating Priority Metric):")
            print(f"  {Path(best_ckpt_path).name}")
    elif ckpt_monitor == 'val/Final_PPL_All':
        # Single checkpoint strategy for balanced training
        best_ckpt_path = checkpoint_callback.best_model_path if checkpoint_callback.best_model_path else None
        if best_ckpt_path:
            print(f"\nBest checkpoint by val/Final_PPL_All (Balanced Training):")
            print(f"  {Path(best_ckpt_path).name}")
    elif ckpt_monitor == 'val/ppl_ngl_upper':
        # Single checkpoint strategy
        best_ckpt_path = checkpoint_callback.best_model_path if checkpoint_callback.best_model_path else None
        if best_ckpt_path:
            print(f"\nBest checkpoint by {ckpt_monitor}:")
            print(f"  {Path(best_ckpt_path).name}")
    elif ckpt_monitor == 'val/ppl_all':
        # [ABLATION] Simple head checkpoint strategy
        best_ckpt_path = checkpoint_callback.best_model_path if checkpoint_callback.best_model_path else None
        if best_ckpt_path:
            print(f"\nBest checkpoint by val/ppl_all (Simple Head):")
            print(f"  {Path(best_ckpt_path).name}")
    elif ckpt_monitor == 'val/NGL_PPL_All':
        # [v38] PRISM pretrain checkpoint strategy
        best_ckpt_path = checkpoint_callback.best_model_path if checkpoint_callback.best_model_path else None
        if best_ckpt_path:
            print(f"\nBest checkpoint by val/NGL_PPL_All (PRISM Pretrain):")
            print(f"  {Path(best_ckpt_path).name}")
    else:
        # Dual checkpoint strategy
        best_ckpt_path = checkpoint_val_ppl_upper.best_model_path if checkpoint_val_ppl_upper.best_model_path else None
        if best_ckpt_path:
            print(f"\nBest checkpoint by val/ppl_ngl_upper (Inference Utility):")
            print(f"  {Path(best_ckpt_path).name}")
            # Also report the best by val/Final_PPL_All for comparison
            if checkpoint_val_ppl_all.best_model_path:
                print(f"\nBest checkpoint by val/Final_PPL_All (Overall Performance):")
                print(f"  {Path(checkpoint_val_ppl_all.best_model_path).name}")

    if best_ckpt_path:
        # Test the best checkpoint
        print(f"\n{'='*80}")
        print(f"Testing best checkpoint: {Path(best_ckpt_path).name}")
        print(f"{'='*80}")

        test_result = trainer.test(model, data_module, ckpt_path=best_ckpt_path)

        # Print test results
        print("\n" + "="*80)
        print(f"Test Results (Best by {ckpt_monitor})")
        print("="*80)
        for key, value in test_result[0].items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
    else:
        print(f"\nNo best checkpoint found by {ckpt_monitor}. Skipping test evaluation.")

    print("\n" + "="*80)
    print("Training completed!")
    print("="*80)

    print(f"\nAll checkpoints saved to: {output_dir / experiment_folder_name / 'checkpoints'}")
    print(f"TensorBoard logs saved to: {logger.log_dir}")
    print(f"\nTo view TensorBoard logs, run:")
    print(f"  tensorboard --logdir={logger.log_dir}")


if __name__ == "__main__":
    main()
