#!/usr/bin/env python
# coding: utf-8

"""
Inference script for ESM2 supervised fine-tuning using PRISM library.

Usage:
    python inference_esm.py --config config.yaml --checkpoint path/to/checkpoint.ckpt
"""

import argparse
import yaml
import pandas as pd
import pytorch_lightning as pl
from pathlib import Path

import prism
from prism import SFTDataModule


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
        'model': ['model_identifier'],
        'training': ['seed'],
        'trainer': ['devices', 'accelerator', 'precision']
    }

    # Check each section and its required fields
    for section, fields in required_fields.items():
        if section not in config:
            raise KeyError(f"Missing required section in config: '{section}'")

        for field in fields:
            if field not in config[section]:
                raise KeyError(f"Missing required field in config['{section}']: '{field}'")

    print(" Configuration validated successfully")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run inference on ESM2 model for antibody sequences')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Optional: Override data path from config (must be a pickle file)')
    args = parser.parse_args()

    print("="*80)
    print("ESM2 Inference with PRISM")
    print("="*80)

    # Load and validate configuration
    print(f"\nLoading configuration from: {args.config}")
    config = load_config(args.config)
    validate_config(config)

    # Print configuration
    print("\nConfiguration:")
    print(yaml.dump(config, default_flow_style=False, indent=2))

    # Verify checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    print(f"\nCheckpoint: {checkpoint_path}")

    # Load data
    print("="*80)
    print("Loading data...")
    print("="*80)

    # Use override data path if provided, otherwise use config data path
    data_path = Path(args.data_path) if args.data_path else Path(config['data']['data_path'])

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_pickle(data_path)
    print(f"Loaded dataframe with shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    # Handle split column
    has_split_column = 'split' in df.columns
    if has_split_column:
        print(f"\nSplit distribution:")
        print(df['split'].value_counts())

        # Filter for test split
        test_df = df[df['split'] == 'test'].copy()
        print(f"\nUsing test split only: {len(test_df)} samples")

        if len(test_df) == 0:
            raise ValueError("No test samples found in dataset")
    else:
        print("\nNo 'split' column found - using entire dataset for inference")
        test_df = df.copy()
        # Add a dummy split column for compatibility with SFTDataModule
        #test_df['split'] = 'test'

    # Load model using prism.pretrained() API
    print("\n" + "="*80)
    print("Loading model...")
    print("="*80)

    # Determine gene vocab path if applicable
    gene_vocab_path = config['model'].get('gene_vocab_path', None)
    prism_model = prism.pretrained(
        str(checkpoint_path),
        device="cpu",  # Load on CPU first; trainer will handle device placement
        gene_vocab_path=gene_vocab_path,
    )
    model = prism_model.model
    tokenizer = prism_model.tokenizer

    # Create data module
    print("\n" + "="*80)
    print("Creating data module...")
    print("="*80)
    data_module = SFTDataModule(
        data_frame=test_df,
        batch_size=1024,
        mask_prob=config['data']['mask_prob'],
        tokenizer=tokenizer,
        seed=config['training']['seed'],
        num_workers=config['data']['num_workers'],
    )

    # Create trainer for inference
    print("\n" + "="*80)
    print("Creating trainer for inference...")
    print("="*80)
    trainer = pl.Trainer(
        accelerator=config['trainer']['accelerator'],
        devices=config['trainer']['devices'],
        precision=config['trainer']['precision'],
        logger=False  # No logging for inference
    )

    print(f"\nTrainer configuration:")
    print(f"  Devices: {config['trainer']['devices']}")
    print(f"  Accelerator: {config['trainer']['accelerator']}")
    print(f"  Precision: {config['trainer']['precision']}")

    # Run inference/testing
    print("\n" + "="*80)
    print("Running inference on test set...")
    print("="*80)
    print(f"Using checkpoint: {checkpoint_path.name}")

    test_result = trainer.test(model, data_module)

    # Print results
    print("\n" + "="*80)
    print("Inference Results")
    print("="*80)

    result = test_result[0]
    print(f"\nTest Loss: {result['test_loss']:.4f}")
    print(f"\nPerplexity Metrics:")
    print(f"  Heavy chain (germline): {result['test_perplexity_heavy_gl']:.4f}")
    print(f"  Heavy chain (non-germline): {result['test_perplexity_heavy_ngl']:.4f}")
    print(f"  Light chain (germline): {result['test_perplexity_light_gl']:.4f}")
    print(f"  Light chain (non-germline): {result['test_perplexity_light_ngl']:.4f}")

    # Print custom token metrics if available
    if 'test_perplexity_heavy_gl_custom' in result:
        print(f"\nCustom Token Perplexity Metrics:")
        print(f"  Heavy chain (germline, custom): {result['test_perplexity_heavy_gl_custom']:.4f}")
        print(f"  Heavy chain (non-germline, custom): {result['test_perplexity_heavy_ngl_custom']:.4f}")
        print(f"  Light chain (germline, custom): {result['test_perplexity_light_gl_custom']:.4f}")
        print(f"  Light chain (non-germline, custom): {result['test_perplexity_light_ngl_custom']:.4f}")

    print("\n" + "="*80)
    print("Inference completed!")
    print("="*80)


if __name__ == "__main__":
    main()
