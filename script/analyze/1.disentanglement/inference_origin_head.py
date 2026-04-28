#!/usr/bin/env python
# coding: utf-8
"""
inference_origin_head.py - Origin Head Inference on Test Set

This script loads the DevAnt-LM model (SFT_ESM2 with multihead architecture),
runs inference on the test set, and extracts the Origin Head logits for each residue.

Key Operations:
1. Load trained DevAnt-LM checkpoint
2. Load test set from pickle file
3. Tokenize sequences (UPPERCASE for consistent context)
4. Run forward pass through the model
5. Extract Origin Head logits (raw, before sigmoid)
6. Save per-residue logits with ground truth NGL labels

Output Format:
    - Per-residue CSV with columns: seq_idx, residue_idx, amino_acid, logit, prob, ngl_label
    - Or pickled DataFrame with full results

Usage:
    # Basic usage with default test set
    python inference_origin_head.py --checkpoint path/to/checkpoint.ckpt

    # With custom test set and output path
    python inference_origin_head.py --checkpoint path/to/checkpoint.ckpt \
        --test_data path/to/test.pkl --output path/to/output.pkl

    # With mask-based inference (recommended for unbiased predictions)
    python inference_origin_head.py --checkpoint path/to/checkpoint.ckpt --mask_based --batch_size 64

    # Validation-style evaluation (requires --config for masking settings)
    python inference_origin_head.py --checkpoint path/to/checkpoint.ckpt --config path/to/config.yaml \
        --validation_style

Author: DevAnt-LM GL/NGL Analysis Pipeline
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

# Add prism to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    import prism
    from prism.multimodal_io import GeneVocabulary
    from prism.io_utils import make_collate_fn_multihead
except ImportError as e:
    print(f"[ERROR] Could not import prism: {e}")
    print("Please ensure the package is installed or add to PYTHONPATH:")
    print("  pip install -e . (from prism root)")
    sys.exit(1)

import yaml


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_CONFIG = "configs/config_esm2_v17.yaml"
DEFAULT_TEST_DATA = "data/unpaired_OAS/linear_probe_data/test_linear.pkl"
AMINO_ACIDS = list('ACDEFGHIKLMNPQRSTVWY')


# =============================================================================
# Data Processing Helpers
# =============================================================================

def parse_mutation_codes(mut_codes: str) -> List[int]:
    """
    Parse mutation codes like 'A40D;A61T;M83I' into 0-indexed positions.

    Args:
        mut_codes: Semicolon-separated mutation codes (e.g., 'A40D;A61T')

    Returns:
        List of 0-indexed mutation positions
    """
    if pd.isna(mut_codes) or mut_codes == '' or mut_codes is None:
        return []

    positions = []
    for mut in str(mut_codes).split(';'):
        mut = mut.strip()
        if not mut:
            continue
        # Extract position from mutation code (e.g., 'A40D' -> 40)
        # Position is 1-indexed in the code, convert to 0-indexed
        try:
            pos_str = ''.join(c for c in mut[1:-1] if c.isdigit())
            if pos_str:
                pos = int(pos_str) - 1  # Convert to 0-indexed
                positions.append(pos)
        except (ValueError, IndexError):
            continue

    return positions


def compute_ngl_mask_from_alignment(seq: str, germline_seq: str) -> List[int]:
    """
    Compute NGL mask by comparing sequence to germline alignment.

    Args:
        seq: Observed sequence
        germline_seq: Germline alignment (X = unknown/masked in germline)

    Returns:
        Binary mask where 1 = NGL (mutation), 0 = GL (germline match)
    """
    mask = []
    for i, (s, g) in enumerate(zip(seq, germline_seq)):
        if g == 'X' or g == '-':
            # Unknown germline position, treat as GL
            mask.append(0)
        elif s != g:
            # Mismatch = NGL (mutation)
            mask.append(1)
        else:
            # Match = GL (germline)
            mask.append(0)
    return mask


def apply_case_encoding(seq: str, ngl_mask: List[int]) -> str:
    """
    Apply case encoding to sequence: uppercase for GL, lowercase for NGL.

    This is critical for the Origin Head to work properly, as the model
    was trained with case-encoded input where lowercase = NGL positions.

    Args:
        seq: Amino acid sequence (any case)
        ngl_mask: Binary mask (0=GL, 1=NGL)

    Returns:
        Case-encoded sequence (uppercase=GL, lowercase=NGL)
    """
    encoded = []
    for aa, is_ngl in zip(seq.upper(), ngl_mask):
        if is_ngl:
            encoded.append(aa.lower())
        else:
            encoded.append(aa.upper())
    return ''.join(encoded)


def process_linear_probe_row(row: pd.Series) -> Tuple[str, List[int], str]:
    """
    Process a row from linear probe dataset to get combined sequence and NGL mask.

    Args:
        row: DataFrame row with heavy/light chain data

    Returns:
        Tuple of (combined_sequence, ngl_mask, case_encoded_sequence)
    """
    hc_seq = row['HEAVY_CHAIN_AA_SEQUENCE']
    lc_seq = row['LIGHT_CHAIN_AA_SEQUENCE']

    # Combine sequences (heavy + light)
    combined_seq = hc_seq + lc_seq

    # Method 1: Use germline alignment if available
    if 'HEAVY_CHAIN_AA_GERMLINE_ALIGNMENT' in row.index and 'LIGHT_CHAIN_AA_GERMLINE_ALIGNMENT' in row.index:
        hc_gl = row['HEAVY_CHAIN_AA_GERMLINE_ALIGNMENT']
        lc_gl = row['LIGHT_CHAIN_AA_GERMLINE_ALIGNMENT']

        if pd.notna(hc_gl) and pd.notna(lc_gl):
            hc_mask = compute_ngl_mask_from_alignment(hc_seq, hc_gl)
            lc_mask = compute_ngl_mask_from_alignment(lc_seq, lc_gl)
            ngl_mask = hc_mask + lc_mask
            case_encoded = apply_case_encoding(combined_seq, ngl_mask)
            return combined_seq, ngl_mask, case_encoded

    # Method 2: Use mutation codes
    hc_mut_codes = row.get('hc_mut_codes', '')
    lc_mut_codes = row.get('lc_mut_codes', '')

    hc_mut_positions = parse_mutation_codes(hc_mut_codes)
    lc_mut_positions = parse_mutation_codes(lc_mut_codes)

    # Create NGL mask
    hc_mask = [1 if i in hc_mut_positions else 0 for i in range(len(hc_seq))]
    lc_mask = [1 if i in lc_mut_positions else 0 for i in range(len(lc_seq))]
    ngl_mask = hc_mask + lc_mask

    case_encoded = apply_case_encoding(combined_seq, ngl_mask)
    return combined_seq, ngl_mask, case_encoded


# =============================================================================
# Model Loading (adapted from 02_run_inference.py)
# =============================================================================

def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def build_gene_vocab(data_path: str) -> GeneVocabulary:
    """Build Gene Vocabulary from the training data or a pre-built JSON file."""
    print(f"  Building gene vocabulary from: {data_path}")

    if not Path(data_path).exists():
        raise FileNotFoundError(f"Training data not found: {data_path}")

    if data_path.endswith('.json'):
        with open(data_path, 'r') as f:
            vocab_data = json.load(f)
        if 'genes' not in vocab_data:
            raise ValueError(f"JSON file must contain 'genes' key")
        all_genes = vocab_data['genes']
    else:
        df = pd.read_pickle(data_path)
        gene_cols = []
        possible_cols = ['v_gene_heavy', 'v_gene_light', 'j_gene_heavy', 'j_gene_light',
                         'v_gene', 'j_gene']
        for col in possible_cols:
            if col in df.columns:
                gene_cols.append(col)
        if not gene_cols:
            raise ValueError(f"No gene columns found in {list(df.columns)}")
        all_genes = set()
        for col in gene_cols:
            all_genes.update(df[col].dropna().unique())
        all_genes = sorted(list(all_genes))

    vocab = GeneVocabulary(genes=all_genes)
    print(f"  Built vocabulary with {len(vocab)} genes")
    return vocab


# =============================================================================
# Inference Class
# =============================================================================

def evaluate_validation_style(
    model,
    data: List[Dict],
    config: dict,
    device: str = 'cuda',
    num_eval_passes: int = 10,
    mask_prob: float = 0.35,
    gene_vocab: Optional[GeneVocabulary] = None
) -> Dict[str, float]:
    """
    Evaluate Origin Head exactly like training validation does.

    This uses the same collator and masking strategy as training, then
    computes metrics only on MASKED positions (where labels_mut != -1.0).

    Args:
        model: The loaded SFT_ESM2 model
        data: List of dicts with keys:
            - 'sequence': amino acid sequence
            - 'ngl_mask': list of 0/1 for GL/NGL
            - 'v_gene_heavy', 'j_gene_heavy', 'v_gene_light', 'j_gene_light': gene names (optional)
            - 'region_ids': region mask tensor aligned with sequence (optional)
        config: Model config dict
        device: Device for inference
        num_eval_passes: Number of random masking passes (for stable estimate)
        mask_prob: Masking probability
        gene_vocab: GeneVocabulary for encoding gene names (optional)

    Returns:
        Dict with F1, PR-AUC, and accuracy metrics
    """
    from sklearn.metrics import precision_recall_curve, auc, f1_score, precision_score, recall_score

    # Check if gene/region conditioning is enabled in config
    model_conf = config.get('model', {})
    use_germline_genes = model_conf.get('use_germline_genes', False)
    use_region_embedding = model_conf.get('use_region_embedding', False)

    print(f"\n[Validation-Style Evaluation]")
    print(f"  Using same masking logic as training validation")
    print(f"  Mask probability: {mask_prob:.1%}")
    print(f"  Number of random passes: {num_eval_passes}")
    print(f"  Gene conditioning: {use_germline_genes} (vocab: {gene_vocab is not None})")
    print(f"  Region embedding: {use_region_embedding}")

    # Build collator (same as training)
    tokenizer = model.tokenizer

    # Get config values
    model_conf = config.get('model', {})
    use_region_masking = model_conf.get('use_region_masking', False)
    cdr_mask_prob = model_conf.get('cdr_mask_prob', 0.35)
    fr_mask_prob = model_conf.get('fr_mask_prob', 0.35)

    collate_fn = make_collate_fn_multihead(
        tokenizer=tokenizer,
        mask_prob=mask_prob,
        gene_vocab=None,
        use_germline_genes=False,
        ngl_targeted_masking=False,
        ngl_mask_prob=mask_prob,
        use_region_embedding=False,
        use_region_masking=use_region_masking,
        cdr_mask_prob=cdr_mask_prob,
        fr_mask_prob=fr_mask_prob,
        silent=True
    )

    # Collect all predictions and labels across multiple passes
    all_probs = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for pass_idx in range(num_eval_passes):
            # Process data in batches
            batch_size = 32
            for i in range(0, len(data), batch_size):
                batch_data = data[i:i+batch_size]

                # Create batch items with gene/region info
                batch_items = []
                for item in batch_data:
                    seq = item['sequence']
                    ngl_mask = item['ngl_mask']

                    # Create case-encoded sequence (lowercase for NGL)
                    case_seq = apply_case_encoding(seq, ngl_mask)

                    # Tokenize
                    tokens = tokenizer(
                        case_seq,
                        add_special_tokens=True,
                        truncation=True,
                        max_length=512,
                        return_tensors=None
                    )

                    # Create NGL mask tensor aligned with tokens
                    token_len = len(tokens['input_ids'])
                    ngl_tensor = torch.zeros(token_len)
                    seq_len = min(len(ngl_mask), token_len - 2)
                    for j in range(seq_len):
                        ngl_tensor[j + 1] = float(ngl_mask[j])

                    batch_item = {
                        'input_ids': torch.tensor(tokens['input_ids']),
                        'attention_mask': torch.tensor(tokens['attention_mask']),
                        'ngl_mask': ngl_tensor
                    }

                    # Add gene IDs if available
                    if use_germline_genes and gene_vocab is not None:
                        v_gene_h = item.get('v_gene_heavy', 'UNK')
                        j_gene_h = item.get('j_gene_heavy', 'UNK')
                        v_gene_l = item.get('v_gene_light', 'UNK')
                        j_gene_l = item.get('j_gene_light', 'UNK')

                        # Encode genes - combine heavy and light (like training)
                        # The model expects single v_gene_id and j_gene_id per sequence
                        # During training, paired data encodes them together
                        # Use heavy chain genes as the primary (common approach)
                        batch_item['v_gene_id'] = gene_vocab.encode(v_gene_h)
                        batch_item['j_gene_id'] = gene_vocab.encode(j_gene_h)

                    # Add region IDs if available
                    if use_region_embedding and 'region_ids' in item:
                        region_ids = item['region_ids']
                        # Align with token length (add 0 for [CLS] at start)
                        region_tensor = torch.zeros(token_len, dtype=torch.long)
                        reg_len = min(len(region_ids), token_len - 2)
                        for j in range(reg_len):
                            region_tensor[j + 1] = int(region_ids[j])
                        batch_item['region_ids'] = region_tensor

                    batch_items.append(batch_item)

                # Pad batch manually
                max_len = max(len(item['input_ids']) for item in batch_items)
                curr_batch_size = len(batch_items)

                input_ids_batch = torch.zeros(curr_batch_size, max_len, dtype=torch.long)
                attention_mask_batch = torch.zeros(curr_batch_size, max_len, dtype=torch.long)
                ngl_mask_batch = torch.zeros(curr_batch_size, max_len)

                # Initialize gene/region tensors if needed
                v_gene_ids = None
                j_gene_ids = None
                region_ids_batch = None

                if use_germline_genes and gene_vocab is not None:
                    v_gene_ids = torch.zeros(curr_batch_size, dtype=torch.long)
                    j_gene_ids = torch.zeros(curr_batch_size, dtype=torch.long)

                if use_region_embedding:
                    region_ids_batch = torch.zeros(curr_batch_size, max_len, dtype=torch.long)

                for j, item in enumerate(batch_items):
                    seq_len = len(item['input_ids'])
                    input_ids_batch[j, :seq_len] = item['input_ids']
                    attention_mask_batch[j, :seq_len] = item['attention_mask']
                    ngl_mask_batch[j, :seq_len] = item['ngl_mask']

                    if v_gene_ids is not None and 'v_gene_id' in item:
                        v_gene_ids[j] = item['v_gene_id']
                        j_gene_ids[j] = item['j_gene_id']

                    if region_ids_batch is not None and 'region_ids' in item:
                        reg_len = len(item['region_ids'])
                        region_ids_batch[j, :reg_len] = item['region_ids']

                # Force uppercase (like training)
                if hasattr(model, 'lowercase_aa_token_ids') and model.lowercase_aa_token_ids:
                    for lower_id, upper_id in model.lowercase_aa_token_ids.items():
                        input_ids_batch[input_ids_batch == lower_id] = upper_id

                # Create labels_mut from ngl_mask (1.0=NGL, 0.0=GL)
                labels_mut = ngl_mask_batch.clone()

                # Apply region-aware masking (like training)
                # CDR regions (2, 4, 6) get cdr_mask_prob, FR regions (1, 3, 5, 7) get fr_mask_prob
                rand_probs = torch.rand_like(labels_mut)

                if region_ids_batch is not None and use_region_masking:
                    # Region-aware masking
                    is_cdr = (region_ids_batch == 2) | (region_ids_batch == 4) | (region_ids_batch == 6)
                    is_fr = (region_ids_batch == 1) | (region_ids_batch == 3) | (region_ids_batch == 5) | (region_ids_batch == 7)
                    to_mask = torch.zeros_like(labels_mut, dtype=torch.bool)
                    to_mask[is_cdr] = rand_probs[is_cdr] < cdr_mask_prob
                    to_mask[is_fr] = rand_probs[is_fr] < fr_mask_prob
                else:
                    # Uniform masking
                    to_mask = rand_probs < mask_prob

                to_mask[:, 0] = False  # Don't mask [CLS]
                to_mask = to_mask & (attention_mask_batch.bool())

                # Set non-masked positions to -1.0 (like training)
                labels_mut[~to_mask] = -1.0

                # Apply masking to input_ids
                mask_token_id = tokenizer.mask_token_id
                if mask_token_id is None:
                    mask_token_id = tokenizer.convert_tokens_to_ids('<mask>')

                # 80% -> [MASK], 10% -> random, 10% -> same
                rand_mask = torch.rand_like(input_ids_batch.float())
                mask_positions = to_mask & (rand_mask < 0.8)
                input_ids_batch[mask_positions] = mask_token_id

                # Move to device
                input_ids_batch = input_ids_batch.to(device)
                attention_mask_batch = attention_mask_batch.to(device)
                labels_mut = labels_mut.to(device)

                if v_gene_ids is not None:
                    v_gene_ids = v_gene_ids.to(device)
                    j_gene_ids = j_gene_ids.to(device)

                if region_ids_batch is not None:
                    region_ids_batch = region_ids_batch.to(device)

                # Forward pass with gene/region conditioning (always multihead)
                _, _, logits_mut, _, _, _ = model._forward_multihead(
                    input_ids=input_ids_batch,
                    attention_mask=attention_mask_batch,
                    v_gene_ids=v_gene_ids,
                    j_gene_ids=j_gene_ids,
                    region_ids=region_ids_batch
                )

                # Collect predictions only for masked positions
                valid_mask = (labels_mut != -1.0)
                if valid_mask.any():
                    probs = torch.sigmoid(logits_mut[valid_mask])
                    targets = labels_mut[valid_mask]

                    all_probs.extend(probs.cpu().numpy().tolist())
                    all_labels.extend(targets.cpu().numpy().tolist())

    # Compute metrics
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # Binary labels
    all_labels_binary = (all_labels > 0.5).astype(int)
    all_preds = (all_probs > 0.5).astype(int)

    # F1 Score
    tp = ((all_preds == 1) & (all_labels_binary == 1)).sum()
    fp = ((all_preds == 1) & (all_labels_binary == 0)).sum()
    fn = ((all_preds == 0) & (all_labels_binary == 1)).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

    # PR-AUC
    precision_curve, recall_curve, _ = precision_recall_curve(all_labels_binary, all_probs)
    pr_auc = auc(recall_curve, precision_curve)

    # ROC-AUC
    from sklearn.metrics import roc_auc_score
    roc_auc = roc_auc_score(all_labels_binary, all_probs)

    # Accuracy
    accuracy = (all_preds == all_labels_binary).mean()

    metrics = {
        'F1': f1,
        'Precision': precision,
        'Recall': recall,
        'PR_AUC': pr_auc,
        'ROC_AUC': roc_auc,
        'Accuracy': accuracy,
        'N_samples': len(all_probs),
        'N_NGL': int(all_labels_binary.sum()),
        'N_GL': int((all_labels_binary == 0).sum()),
    }

    print(f"\n  Results (validation-style):")
    print(f"    F1:       {metrics['F1']:.4f}")
    print(f"    PR-AUC:   {metrics['PR_AUC']:.4f}")
    print(f"    ROC-AUC:  {metrics['ROC_AUC']:.4f}")
    print(f"    Accuracy: {metrics['Accuracy']:.4f}")
    print(f"    Samples:  {metrics['N_samples']} ({metrics['N_NGL']} NGL, {metrics['N_GL']} GL)")

    return metrics


class OriginHeadInference:
    """Inference wrapper for Origin Head predictions."""

    def __init__(
        self,
        checkpoint_path: str,
        config_path: Optional[str] = None,
        gene_vocab_data_path: Optional[str] = None,
        device: str = 'cuda'
    ):
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self.gene_vocab_data_path = gene_vocab_data_path
        self.device = device
        self.model = None
        self.tokenizer = None
        self.gene_vocab = None
        self.config = None
        self._prism_model = None

    def load_model(self):
        """Load model from checkpoint using prism.pretrained() API."""
        print(f"\n{'='*60}")
        print("Loading DevAnt-LM Model for Origin Head Inference")
        print('='*60)
        print(f"  Checkpoint: {self.checkpoint_path}")

        # Load config if provided (used for validation-style evaluation settings)
        if self.config_path:
            print(f"  Config:     {self.config_path}")
            self.config = load_config(self.config_path)

        # Load model via prism.pretrained() (hparams extracted from checkpoint)
        print(f"\n  Loading checkpoint with prism.pretrained()...")
        self._prism_model = prism.pretrained(
            self.checkpoint_path,
            device=self.device,
            gene_vocab_path=self.gene_vocab_data_path,
        )
        self.model = self._prism_model.model
        self.tokenizer = self._prism_model.tokenizer
        self.gene_vocab = self._prism_model.gene_vocab

        # Build gene vocab from data path if prism.pretrained() didn't resolve it
        if self.gene_vocab is None and self.gene_vocab_data_path:
            self.gene_vocab = build_gene_vocab(self.gene_vocab_data_path)

        print(f"\n  Model loaded successfully!")
        print(f"  Device: {self.device}")
        print(f"  Vocab size: {len(self.tokenizer)}")
        print(f"  Multihead: {getattr(self.model, 'use_multihead_architecture', False)}")
        print(f"  Activation: {getattr(self.model, 'activation_function', 'gelu')}")

    def get_origin_logits(self, sequence: str, use_case_encoding: bool = True) -> Tuple[List[float], List[float], List[str]]:
        """
        Get Origin Head raw logits and probabilities for each residue.

        Args:
            sequence: Amino acid sequence (case-encoded if use_case_encoding=True)
            use_case_encoding: If True, preserve case (lowercase=NGL). If False, convert to uppercase.

        Returns:
            Tuple of (logits_list, probabilities_list, amino_acids_list)
        """
        if not use_case_encoding:
            sequence = sequence.upper()

        tokens = self.tokenizer(
            sequence,
            return_tensors="pt",
            add_special_tokens=True,
            truncation=True,
            max_length=512
        )
        input_ids = tokens['input_ids'].to(self.device)
        attention_mask = tokens['attention_mask'].to(self.device)

        with torch.no_grad():
            # Forward pass through multihead architecture
            _, _, logits_mut, _, _, _ = self.model._forward_multihead(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            # Get both raw logits and probabilities
            raw_logits = logits_mut[0]
            probs = torch.sigmoid(raw_logits)

        # Exclude special tokens [CLS] at position 0 and [EOS] at end
        seq_len = len(sequence)
        logits_list = raw_logits[1:seq_len+1].cpu().tolist()
        probs_list = probs[1:seq_len+1].cpu().tolist()
        amino_acids = list(sequence.upper())  # Return uppercase for consistency

        return logits_list, probs_list, amino_acids

    def get_origin_logits_masked_batched(
        self, sequence: str, batch_size: int = 32
    ) -> Tuple[List[float], List[float], List[str]]:
        """
        Batched mask-out approach for Origin Head logits.

        For each position, mask it and get the Origin Head prediction at that position.
        This avoids case-leakage through input token embeddings.

        Args:
            sequence: Amino acid sequence
            batch_size: Number of positions to process in parallel

        Returns:
            Tuple of (logits_list, probabilities_list, amino_acids_list)
        """
        sequence = sequence.upper()
        seq_len = len(sequence)

        mask_token_id = self.tokenizer.mask_token_id
        if mask_token_id is None:
            mask_token_id = self.tokenizer.convert_tokens_to_ids('<mask>')

        tokens = self.tokenizer(
            sequence,
            return_tensors="pt",
            add_special_tokens=True,
            truncation=True,
            max_length=512
        )
        base_input_ids = tokens['input_ids']
        base_attention_mask = tokens['attention_mask']

        logits_list = [0.0] * seq_len
        probs_list = [0.0] * seq_len

        with torch.no_grad():
            for batch_start in range(0, seq_len, batch_size):
                batch_end = min(batch_start + batch_size, seq_len)
                batch_positions = list(range(batch_start, batch_end))
                current_batch_size = len(batch_positions)

                batch_input_ids = base_input_ids.repeat(current_batch_size, 1).to(self.device)
                batch_attention_mask = base_attention_mask.repeat(current_batch_size, 1).to(self.device)

                for i, pos in enumerate(batch_positions):
                    token_pos = pos + 1  # +1 for [CLS]
                    batch_input_ids[i, token_pos] = mask_token_id

                # Forward pass through multihead architecture
                _, _, logits_mut, _, _, _ = self.model._forward_multihead(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask,
                )

                for i, pos in enumerate(batch_positions):
                    token_pos = pos + 1
                    logit = logits_mut[i, token_pos].item()
                    prob = torch.sigmoid(logits_mut[i, token_pos]).item()
                    logits_list[pos] = logit
                    probs_list[pos] = prob

        amino_acids = list(sequence)
        return logits_list, probs_list, amino_acids


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Run Origin Head inference on test set and save logits'
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.ckpt)')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to model config YAML (optional, only needed for validation-style eval settings)')
    parser.add_argument('--test_data', type=str, default=None,
                        help='Path to test data pickle file')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to output file (.pkl or .csv)')
    parser.add_argument('--gene_vocab_data_path', type=str, default=None,
                        help='Path to training data for gene vocabulary')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device for inference')
    parser.add_argument('--mask_based', action='store_true',
                        help='Use mask-out approach for context-free predictions')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for mask-based inference (default: 32)')
    parser.add_argument('--max_sequences', type=int, default=None,
                        help='Maximum number of sequences to process (for testing)')
    parser.add_argument('--save_per_residue', action='store_true',
                        help='Save per-residue CSV in addition to sequence-level pickle')
    parser.add_argument('--validation_style', action='store_true',
                        help='Use validation-style evaluation (matches training metrics exactly)')
    parser.add_argument('--num_eval_passes', type=int, default=10,
                        help='Number of random masking passes for validation-style (default: 10)')
    parser.add_argument('--mask_prob', type=float, default=0.35,
                        help='Masking probability for validation-style (default: 0.35)')

    args = parser.parse_args()

    print("=" * 70)
    print("DevAnt-LM Origin Head Inference Pipeline")
    print("=" * 70)

    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent

    # Set paths
    checkpoint_path = Path(args.checkpoint)
    config_path = Path(args.config) if args.config else None

    if args.test_data:
        test_data_path = Path(args.test_data)
    else:
        test_data_path = project_root / DEFAULT_TEST_DATA

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = script_dir / "origin_head_predictions.pkl"

    # Validate paths
    if not checkpoint_path.exists():
        print(f"\n[ERROR] Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    if config_path is not None and not config_path.exists():
        print(f"\n[ERROR] Config file not found: {config_path}")
        sys.exit(1)
    if not test_data_path.exists():
        print(f"\n[ERROR] Test data not found: {test_data_path}")
        sys.exit(1)

    print(f"\nConfiguration:")
    print(f"  Checkpoint: {checkpoint_path}")
    if config_path:
        print(f"  Config:     {config_path}")
    print(f"  Test data:  {test_data_path}")
    print(f"  Output:     {output_path}")

    # Device setup
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("\n[WARNING] CUDA not available, using CPU")
        device = 'cpu'
    print(f"  Device:     {device}")

    # Gene vocab path
    gene_vocab_path = args.gene_vocab_data_path
    if gene_vocab_path is None:
        # Try to find gene vocab from config or default path
        if config_path:
            config = load_config(str(config_path))
            if config.get('model', {}).get('use_germline_genes', False):
                default_gene_path = project_root / "data" / "unpaired_OAS" / "annotated_data_final" / "paired_with_clusters_filtered_ngl.pkl"
                if default_gene_path.exists():
                    gene_vocab_path = str(default_gene_path)

    # Load model via prism.pretrained()
    inference = OriginHeadInference(
        checkpoint_path=str(checkpoint_path),
        config_path=str(config_path) if config_path else None,
        gene_vocab_data_path=gene_vocab_path,
        device=device
    )
    inference.load_model()

    # Load test data
    print(f"\nLoading test data from: {test_data_path}")
    test_df = pd.read_pickle(test_data_path)
    print(f"  Total sequences: {len(test_df)}")

    # Detect data format
    is_linear_probe_format = 'HEAVY_CHAIN_AA_SEQUENCE' in test_df.columns
    is_combined_format = 'sequence' in test_df.columns and 'NGL_mask' in test_df.columns

    if is_linear_probe_format:
        print(f"  Format: Linear probe (separate heavy/light chains)")
        required_cols = ['HEAVY_CHAIN_AA_SEQUENCE', 'LIGHT_CHAIN_AA_SEQUENCE']
    elif is_combined_format:
        print(f"  Format: Combined sequence with NGL_mask")
        required_cols = ['sequence', 'NGL_mask']
    else:
        print(f"\n[ERROR] Unrecognized data format")
        print(f"  Expected either:")
        print(f"    - Linear probe format: HEAVY_CHAIN_AA_SEQUENCE, LIGHT_CHAIN_AA_SEQUENCE")
        print(f"    - Combined format: sequence, NGL_mask")
        print(f"  Available columns: {list(test_df.columns)}")
        sys.exit(1)

    for col in required_cols:
        if col not in test_df.columns:
            print(f"\n[ERROR] Required column '{col}' not found in test data")
            print(f"  Available columns: {list(test_df.columns)}")
            sys.exit(1)

    # Optionally limit sequences for testing
    if args.max_sequences:
        test_df = test_df.head(args.max_sequences)
        print(f"  Limited to {len(test_df)} sequences for testing")

    # ==========================================================================
    # VALIDATION-STYLE EVALUATION (replicates training validation exactly)
    # ==========================================================================
    if args.validation_style:
        if inference.config is None:
            print("\n[ERROR] --config is required for --validation_style evaluation")
            sys.exit(1)

        print("\n" + "=" * 70)
        print("VALIDATION-STYLE EVALUATION")
        print("=" * 70)
        print("This evaluation replicates training validation exactly:")
        print("  1. Apply random masking (same probability as training)")
        print("  2. Compute Origin Head metrics only on MASKED positions")
        print("  3. Use multiple passes for stable estimates")

        # Prepare data for validation-style evaluation (with gene/region info)
        eval_data = []
        for idx, row in test_df.iterrows():
            if is_linear_probe_format:
                sequence, ngl_mask, _ = process_linear_probe_row(row)
            else:
                sequence = row['sequence']
                ngl_mask = row['NGL_mask']

            # Ensure NGL mask is a list
            if isinstance(ngl_mask, str):
                ngl_mask = [int(x) for x in ngl_mask]
            elif isinstance(ngl_mask, np.ndarray):
                ngl_mask = ngl_mask.tolist()

            item = {
                'sequence': sequence.upper(),
                'ngl_mask': ngl_mask
            }

            # Add gene information if available
            if 'v_gene_heavy' in row.index:
                item['v_gene_heavy'] = row['v_gene_heavy']
                item['j_gene_heavy'] = row['j_gene_heavy']
            if 'v_gene_light' in row.index:
                item['v_gene_light'] = row['v_gene_light']
                item['j_gene_light'] = row['j_gene_light']

            # Add region information if available
            if 'region_mask_heavy' in row.index and 'region_mask_light' in row.index:
                # Combine heavy and light region masks like the sequence
                region_h = row['region_mask_heavy']
                region_l = row['region_mask_light']
                if isinstance(region_h, str):
                    region_h = [int(x) for x in region_h]
                if isinstance(region_l, str):
                    region_l = [int(x) for x in region_l]
                # Combined format: heavy + light (matching sequence format)
                combined_region = list(region_h) + list(region_l)
                item['region_ids'] = combined_region

            eval_data.append(item)

        # Run validation-style evaluation with gene vocab
        val_metrics = evaluate_validation_style(
            model=inference.model,
            data=eval_data,
            config=inference.config,
            device=device,
            num_eval_passes=args.num_eval_passes,
            mask_prob=args.mask_prob,
            gene_vocab=inference.gene_vocab
        )

        # Save validation metrics
        val_output_path = output_path.with_suffix('.validation_metrics.json')
        with open(val_output_path, 'w') as f:
            json.dump(val_metrics, f, indent=2)
        print(f"\n  Validation metrics saved to: {val_output_path}")

        print("\n" + "=" * 70)
        print("Validation-style evaluation complete!")
        print("=" * 70)

        # If only validation-style, exit here
        if not args.save_per_residue and not args.mask_based:
            print("\n[INFO] Use --save_per_residue or --mask_based for additional analysis")
            return

    # Select inference method
    if args.mask_based:
        print(f"\nRunning inference (MASK-BASED Origin Head approach)...")
        print(f"  Batch size: {args.batch_size}")
        inference_method = lambda seq: inference.get_origin_logits_masked_batched(
            seq, batch_size=args.batch_size
        )
    else:
        print(f"\nRunning inference (Direct Origin Head approach)...")
        print(f"  NOTE: Input converted to uppercase - use --mask_based for unbiased predictions")
        inference_method = inference.get_origin_logits

    # Run inference
    all_results = []
    per_residue_results = []

    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Processing"):
        # Get sequence and NGL mask based on data format
        if is_linear_probe_format:
            sequence, ngl_mask, case_encoded_seq = process_linear_probe_row(row)
        else:
            sequence = row['sequence']
            ngl_mask = row['NGL_mask']
            # For combined format, assume sequence is already case-encoded or create encoding
            if 'case_encoded' in row.index:
                case_encoded_seq = row['case_encoded']
            else:
                # Apply case encoding based on NGL mask
                ngl_mask_list = ngl_mask if isinstance(ngl_mask, list) else list(ngl_mask)
                case_encoded_seq = apply_case_encoding(sequence, ngl_mask_list)

        # Ensure NGL mask is a list
        if isinstance(ngl_mask, str):
            ngl_mask = [int(x) for x in ngl_mask]
        elif isinstance(ngl_mask, np.ndarray):
            ngl_mask = ngl_mask.tolist()

        # Run inference with case-encoded sequence (lowercase for NGL positions)
        logits_list, probs_list, amino_acids = inference_method(case_encoded_seq)

        # Align lengths (truncate if necessary due to max_length)
        min_len = min(len(logits_list), len(ngl_mask))
        logits_list = logits_list[:min_len]
        probs_list = probs_list[:min_len]
        ngl_mask = ngl_mask[:min_len]
        amino_acids = amino_acids[:min_len]

        # Store sequence-level results
        result = {
            'seq_idx': idx,
            'sequence': sequence[:min_len],
            'logits': logits_list,
            'probs': probs_list,
            'ngl_labels': ngl_mask,
            'mean_logit': np.mean(logits_list),
            'mean_prob': np.mean(probs_list),
            'mean_logit_gl': np.mean([l for l, m in zip(logits_list, ngl_mask) if m == 0]) if any(m == 0 for m in ngl_mask) else np.nan,
            'mean_logit_ngl': np.mean([l for l, m in zip(logits_list, ngl_mask) if m == 1]) if any(m == 1 for m in ngl_mask) else np.nan,
            'n_gl': sum(1 for m in ngl_mask if m == 0),
            'n_ngl': sum(1 for m in ngl_mask if m == 1),
        }
        all_results.append(result)

        # Per-residue results (for detailed CSV)
        if args.save_per_residue:
            for res_idx, (aa, logit, prob, label) in enumerate(zip(amino_acids, logits_list, probs_list, ngl_mask)):
                per_residue_results.append({
                    'seq_idx': idx,
                    'residue_idx': res_idx,
                    'amino_acid': aa,
                    'logit': logit,
                    'prob': prob,
                    'ngl_label': label
                })

    # Save results
    results_df = pd.DataFrame(all_results)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if str(output_path).endswith('.csv'):
        # Save simplified CSV (without list columns)
        csv_df = results_df.drop(columns=['logits', 'probs', 'ngl_labels', 'sequence'])
        csv_df.to_csv(output_path, index=False)
    else:
        results_df.to_pickle(output_path)

    print(f"\n{'='*70}")
    print("SUMMARY")
    print("="*70)
    print(f"Processed: {len(results_df)} sequences")
    print(f"Output saved to: {output_path}")

    # Save per-residue CSV if requested
    if args.save_per_residue and per_residue_results:
        per_res_path = output_path.with_suffix('.per_residue.csv')
        per_res_df = pd.DataFrame(per_residue_results)
        per_res_df.to_csv(per_res_path, index=False)
        print(f"Per-residue CSV: {per_res_path}")
        print(f"  Total residues: {len(per_res_df)}")

    # Statistics
    print(f"\nOrigin Head Logit Statistics:")
    print(f"  Mean logit (all):  {results_df['mean_logit'].mean():.4f}")
    print(f"  Mean logit (GL):   {results_df['mean_logit_gl'].mean():.4f}")
    print(f"  Mean logit (NGL):  {results_df['mean_logit_ngl'].mean():.4f}")
    print(f"\nOrigin Head Probability Statistics:")
    print(f"  Mean prob (all):   {results_df['mean_prob'].mean():.4f}")

    # Classification metrics (using prob > 0.5 as threshold)
    if args.save_per_residue and per_residue_results:
        per_res_df = pd.DataFrame(per_residue_results)
        predictions = (per_res_df['prob'] > 0.5).astype(int)
        labels = per_res_df['ngl_label']
        accuracy = (predictions == labels).mean()

        # Per-class accuracy
        gl_mask = labels == 0
        ngl_mask = labels == 1
        gl_acc = (predictions[gl_mask] == labels[gl_mask]).mean() if gl_mask.any() else np.nan
        ngl_acc = (predictions[ngl_mask] == labels[ngl_mask]).mean() if ngl_mask.any() else np.nan

        print(f"\nClassification Metrics (threshold=0.5):")
        print(f"  Overall Accuracy:  {accuracy:.4f}")
        print(f"  GL Accuracy:       {gl_acc:.4f}")
        print(f"  NGL Accuracy:      {ngl_acc:.4f}")

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
