#!/usr/bin/env python
# coding: utf-8
"""
Extract amino acid logits at mutation positions using Evo-Ab (SFT_ESM2) model.

This script extracts the full 20-AA logit distribution at each mutation position
identified in therasabdab_germline.csv using masked prediction with the Evo-Ab model.

Key Features:
- Uses multihead architecture with alpha gating (v17/v18/v33 compatible)
- Outputs logits for BOTH uppercase (germline) and lowercase (mutation) tokens
- Supports gene vocabulary loading for gene-conditioned models (JSON preferred)
- Supports region mask embeddings for region-aware models
- Supports separate heavy/light chain gene columns
- **NEW**: Supports saving RAW logits (before softmax) for post-hoc temperature scaling

Output format (CSV):
    Therapeutic, chain, position, germline_aa, mutated_aa,
    A_upper, C_upper, ..., Y_upper,  # Uppercase token logits (GL prediction)
    A_lower, C_lower, ..., Y_lower   # Lowercase token logits (NGL prediction)

Logit Types:
    --output_type log_prob (default): Outputs log-probabilities after softmax
    --output_type raw_logits: Outputs raw logits BEFORE softmax (enables post-hoc temperature)
    --output_type both: Outputs both raw logits and log-probs in separate column sets

Usage:
    # Basic usage with gene vocabulary JSON (preferred)
    conda run -n devant python extract_mutation_logits_evo_ab.py \
        --data_path ../../data/therasabdab_germline.csv \
        --config ../../configs/config_esm2_v33.yaml \
        --checkpoint path/to/checkpoint.ckpt \
        --gene_vocab_json ../../data/unpaired_OAS/annotated_data_final/gene_vocabulary.json \
        --output_path ../../data/therasabdab_evo_ab_logits.csv

    # With region masks (if model supports region embeddings)
    conda run -n devant python extract_mutation_logits_evo_ab.py \
        --data_path ../../data/therasabdab_germline.csv \
        --config ../../configs/config_esm2_v33.yaml \
        --checkpoint path/to/checkpoint.ckpt \
        --gene_vocab_json ../../data/unpaired_OAS/annotated_data_final/gene_vocabulary.json \
        --use_region_mask \
        --output_path ../../data/therasabdab_evo_ab_logits.csv

Reference: benchmark_evo_ab_fixed.py and inference_esm_with_logprobs.py for model loading logic.
"""

import argparse
import re
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

# Standard amino acids
AMINO_ACIDS = list('ACDEFGHIKLMNPQRSTVWY')

# Import prism modules
try:
    import prism
    from prism import SFT_ESM2
    from prism.multimodal_io import GeneVocabulary
except ImportError:
    print("[Error] Could not import prism. Please ensure the package is in your PYTHONPATH.")
    print("  Try: pip install -e . (from prism root)")
    sys.exit(1)




def parse_mutations(mutation_str: str) -> List[Tuple[int, str, str]]:
    """
    Parse mutation string into list of (position, germline_aa, mutated_aa).

    Args:
        mutation_str: Comma-separated mutations like "Q3K,S36N,E51D"

    Returns:
        List of (position, germline_aa, mutated_aa) tuples
    """
    if pd.isna(mutation_str) or not mutation_str or mutation_str == '':
        return []

    mutations = []
    for mut in mutation_str.split(','):
        mut = mut.strip()
        if not mut:
            continue

        # Parse format: {germline_aa}{position}{mutated_aa}
        # Handle insertion codes like "111A" -> position "111A"
        match = re.match(r'^([A-Z])(\d+[A-Za-z]?)([A-Z])$', mut)
        if match:
            germline_aa = match.group(1)
            position_str = match.group(2)
            mutated_aa = match.group(3)

            # Convert position to integer (ignore insertion code for now)
            try:
                position = int(re.match(r'(\d+)', position_str).group(1))
                mutations.append((position, germline_aa, mutated_aa))
            except (ValueError, AttributeError):
                continue

    return mutations


class EvoAbLogitExtractor:
    """Extract logits from Evo-Ab model at mutation positions."""

    def __init__(
        self,
        checkpoint_path: str,
        gene_vocab_json: Optional[str] = None,
        use_region_mask: bool = False,
        device: str = 'cuda',
        batch_size: int = 32,
        output_type: str = 'raw_logits'  # 'log_prob', 'raw_logits', or 'both'
    ):
        self.checkpoint_path = checkpoint_path
        self.gene_vocab_json = gene_vocab_json
        self.use_region_mask = use_region_mask
        self.device = device
        self.batch_size = batch_size
        self.output_type = output_type  # Controls whether to save raw logits or log-probs

        self.model = None
        self.tokenizer = None
        self.gene_vocab = None

        # Token ID mappings
        self.uppercase_aa_to_idx = {}
        self.lowercase_aa_to_idx = {}

        # Model capability flags
        self.use_germline_genes = False
        self.use_region_embedding = False

    def load_model(self):
        """Load model from checkpoint using prism.pretrained() API."""
        print(f"\n{'='*60}")
        print(f"Loading Evo-Ab model...")
        print(f"  Checkpoint: {self.checkpoint_path}")
        print('='*60)

        # Use prism.pretrained() API - loads checkpoint with hparams automatically
        prism_model = prism.pretrained(
            self.checkpoint_path,
            device=str(self.device),
            gene_vocab_path=self.gene_vocab_json,
        )
        self.model = prism_model.model
        self.tokenizer = prism_model.tokenizer
        self.gene_vocab = prism_model.gene_vocab

        # Check model capabilities from loaded model
        self.use_germline_genes = getattr(self.model, 'use_germline_genes', False)
        self.use_region_embedding = getattr(self.model, 'use_region_embedding', False)

        # Build vocabulary mappings
        for aa in AMINO_ACIDS:
            self.uppercase_aa_to_idx[aa] = self.tokenizer.convert_tokens_to_ids(aa)
            # Try lowercase (v17/v18/v33 models have lowercase tokens)
            lowercase_id = self.tokenizer.convert_tokens_to_ids(aa.lower())
            if lowercase_id != self.tokenizer.unk_token_id:
                self.lowercase_aa_to_idx[aa] = lowercase_id
            else:
                # Fallback: try using lowercase_aa_token_ids mapping from model
                if hasattr(self.model, 'lowercase_aa_token_ids'):
                    upper_id = self.uppercase_aa_to_idx[aa]
                    if upper_id in self.model.lowercase_aa_token_ids:
                        self.lowercase_aa_to_idx[aa] = self.model.lowercase_aa_token_ids[upper_id]

        print(f"\n  Model loaded on {self.device}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Uppercase AA tokens: {len(self.uppercase_aa_to_idx)}")
        print(f"  Lowercase AA tokens: {len(self.lowercase_aa_to_idx)}")

        # Print model architecture info
        use_multihead = getattr(self.model, 'use_multihead_architecture', False)
        use_alpha_gating = getattr(self.model, 'use_alpha_gating', False)
        has_prism_v2 = hasattr(self.model, '_forward_multihead_prism')
        has_gl_head = hasattr(self.model, 'gl_head_dense') and self.model.gl_head_dense is not None
        has_trust_head = hasattr(self.model, 'trust_head') and self.model.trust_head is not None

        print(f"  Multihead: {use_multihead}, Alpha gating: {use_alpha_gating}")
        print(f"  PRISM v2 (separate GL/NGL heads): {has_prism_v2 and has_gl_head}")
        print(f"  Trust head (v38.4): {has_trust_head}")
        print(f"  Gene conditioning: {self.use_germline_genes}")
        print(f"  Region embedding: {self.use_region_embedding}")

        # Store architecture info
        self.has_prism_v2 = has_prism_v2 and has_gl_head
        self.has_trust_head = has_trust_head

        if self.use_region_mask and not self.use_region_embedding:
            print(f"  WARNING: --use_region_mask specified but model doesn't use region embeddings")

        # [SIMPLE HEAD DETECTION] Track model type for output handling
        self.is_simple_head = not use_multihead
        if self.is_simple_head:
            print(f"\n  {'='*50}")
            print(f"  [SIMPLE HEAD MODEL DETECTED]")
            print(f"  {'='*50}")
            print(f"  This model does NOT use multihead architecture.")
            print(f"  Lowercase token logits ({len(self.lowercase_aa_to_idx)} tokens) will still")
            print(f"  be extracted but have NO semantic GL/NGL distinction.")
            print(f"  For analysis, use ONLY the *_upper columns.")
            print(f"  The *_lower columns are NOT meaningful for simple head models.")
            print(f"  {'='*50}")

    def get_logits_at_positions(
        self,
        sequence: str,
        positions: List[int],
        v_gene_id: Optional[int] = None,
        j_gene_id: Optional[int] = None,
        region_mask: Optional[str] = None
    ) -> Dict[int, Dict[str, float]]:
        """
        Get logits for all 20 AAs at specified positions using masked prediction.

        Args:
            sequence: Full amino acid sequence
            positions: List of 0-indexed positions to extract logits for
            v_gene_id: V-gene ID for gene conditioning (optional)
            j_gene_id: J-gene ID for gene conditioning (optional)
            region_mask: Region mask string (e.g., "000011112222...") for region embedding (optional)

        Returns:
            Dict mapping position -> {
                'A_upper': logit, 'C_upper': logit, ...,  # Final combined logits
                'A_lower': logit, 'C_lower': logit, ...,
                'A_gl': logit, ...,  # Raw GL head (if PRISM v2)
                'A_ngl': logit, ...,  # Raw NGL head (if PRISM v2)
                'alpha': float,  # P(GL) from origin head
                'trust': float,  # Trust value (if v38.4)
                'logits_mut': float  # Raw origin head logit
            }
        """
        if not positions:
            return {}

        sequence = sequence.upper()
        results = {}

        use_multihead = getattr(self.model, 'use_multihead_architecture', False)
        use_alpha_gating = getattr(self.model, 'use_alpha_gating', False)
        use_multiplicative_gating = getattr(self.model, 'use_multiplicative_gating', False)

        # Tokenize sequence
        tokens = self.tokenizer(
            sequence,
            return_tensors="pt",
            add_special_tokens=True,
            truncation=True,
            max_length=1024
        )
        input_ids = tokens['input_ids'].to(self.device)
        attention_mask = tokens['attention_mask'].to(self.device)

        # Prepare gene IDs if using gene conditioning
        v_gene_ids_tensor = None
        j_gene_ids_tensor = None
        if self.use_germline_genes and v_gene_id is not None and j_gene_id is not None:
            v_gene_ids_tensor = torch.tensor([v_gene_id], dtype=torch.long, device=self.device)
            j_gene_ids_tensor = torch.tensor([j_gene_id], dtype=torch.long, device=self.device)

        # Prepare region mask if using region embedding
        region_ids_tensor = None
        if self.use_region_embedding and self.use_region_mask and region_mask:
            # Convert region mask string to tensor
            # Region mask format: "0001112222333..." where each digit is region ID
            # Need to add special token positions (CLS at start, EOS at end)
            seq_len = input_ids.shape[1]
            region_ids = torch.zeros(1, seq_len, dtype=torch.long, device=self.device)

            # Fill in region IDs (skip CLS token at position 0)
            for i, char in enumerate(region_mask):
                if i + 1 < seq_len - 1:  # Skip CLS and EOS
                    region_ids[0, i + 1] = int(char)

            region_ids_tensor = region_ids

        for pos in positions:
            token_pos = pos + 1  # +1 for [CLS] token

            if token_pos >= input_ids.shape[1] - 1:  # -1 for [EOS]
                continue

            # Create masked input
            masked_input = input_ids.clone()
            masked_input[0, token_pos] = self.tokenizer.mask_token_id

            # Initialize component storage
            alpha_val = None
            trust_val = None
            logits_mut_val = None
            gl_logits = None
            ngl_logits = None

            with torch.no_grad():
                # [FIX] Check use_prism_architecture flag, not just method existence
                # v34.1b has use_multihead=True but use_prism_architecture=False
                # It should use _forward_multihead (with origin conditioning), not _forward_multihead_prism
                use_prism = getattr(self.model, 'use_prism_architecture', False)

                if use_multihead and use_prism and hasattr(self.model, '_forward_multihead_prism'):
                    # PRISM v2 architecture (v38+) with separate GL and NGL heads
                    forward_kwargs = {
                        'input_ids': masked_input,
                        'attention_mask': attention_mask,
                    }

                    if v_gene_ids_tensor is not None and j_gene_ids_tensor is not None:
                        forward_kwargs['v_gene_ids'] = v_gene_ids_tensor
                        forward_kwargs['j_gene_ids'] = j_gene_ids_tensor

                    if region_ids_tensor is not None:
                        forward_kwargs['region_ids'] = region_ids_tensor

                    # Get all component outputs
                    logits_gl, logits_ngl, logits_mut, alpha, trust, logits_final, _ = \
                        self.model._forward_multihead_prism(**forward_kwargs)

                    # Extract component values at this position
                    if alpha is not None:
                        alpha_val = alpha[0, token_pos, 0].item()
                    if trust is not None:
                        trust_val = trust[0, token_pos, 0].item()
                    if logits_mut is not None:
                        logits_mut_val = logits_mut[0, token_pos].item()
                    if logits_gl is not None:
                        gl_logits = logits_gl[0, token_pos]  # [V]
                    if logits_ngl is not None:
                        ngl_logits = logits_ngl[0, token_pos]  # [V]

                    if logits_final is not None:
                        raw_logits = logits_final[0, token_pos]
                        # Final logits are already combined, apply softmax
                        log_probs = F.log_softmax(raw_logits, dim=-1)
                    else:
                        # Fallback to NGL head
                        raw_logits = ngl_logits if ngl_logits is not None else torch.zeros(33)
                        log_probs = F.log_softmax(raw_logits, dim=-1)

                elif use_multihead and hasattr(self.model, '_forward_multihead'):
                    # Original multihead architecture
                    forward_kwargs = {
                        'input_ids': masked_input,
                        'attention_mask': attention_mask,
                    }

                    if v_gene_ids_tensor is not None and j_gene_ids_tensor is not None:
                        forward_kwargs['v_gene_ids'] = v_gene_ids_tensor
                        forward_kwargs['j_gene_ids'] = j_gene_ids_tensor

                    if region_ids_tensor is not None:
                        forward_kwargs['region_ids'] = region_ids_tensor

                    logits_aa, _, logits_mut, alpha, logits_final, _ = self.model._forward_multihead(**forward_kwargs)

                    # Extract component values
                    if alpha is not None:
                        alpha_val = alpha[0, token_pos, 0].item()
                    if logits_mut is not None:
                        logits_mut_val = logits_mut[0, token_pos].item()
                    # In original multihead, AA head serves as both GL and NGL
                    ngl_logits = logits_aa[0, token_pos] if logits_aa is not None else None

                    if use_alpha_gating and logits_final is not None:
                        raw_logits = logits_final[0, token_pos]
                        if use_multiplicative_gating:
                            log_probs = raw_logits
                        else:
                            log_probs = F.log_softmax(raw_logits, dim=-1)
                    else:
                        outputs = self.model.ESM2(input_ids=masked_input, attention_mask=attention_mask)
                        raw_logits = outputs.logits[0, token_pos]
                        log_probs = F.log_softmax(raw_logits, dim=-1)

                elif self.use_germline_genes and v_gene_ids_tensor is not None:
                    outputs = self.model._forward_with_gene_conditioning(
                        input_ids=masked_input,
                        attention_mask=attention_mask,
                        v_gene_ids=v_gene_ids_tensor,
                        j_gene_ids=j_gene_ids_tensor
                    )
                    raw_logits = outputs.logits[0, token_pos]
                    log_probs = F.log_softmax(raw_logits, dim=-1)
                else:
                    outputs = self.model.ESM2(input_ids=masked_input, attention_mask=attention_mask)
                    raw_logits = outputs.logits[0, token_pos]
                    log_probs = F.log_softmax(raw_logits, dim=-1)

            # Extract logits for each amino acid (both uppercase and lowercase)
            aa_logits = {}

            # Determine which values to save based on output_type
            save_raw = self.output_type in ['raw_logits', 'both']
            save_logprob = self.output_type in ['log_prob', 'both']

            # Uppercase logits (germline predictions from Final head)
            for aa in AMINO_ACIDS:
                idx = self.uppercase_aa_to_idx.get(aa)
                if idx is not None:
                    if save_raw:
                        aa_logits[f'{aa}_upper'] = raw_logits[idx].item()
                    if save_logprob:
                        suffix = '_logprob' if save_raw else ''
                        aa_logits[f'{aa}_upper{suffix}'] = log_probs[idx].item()
                else:
                    if save_raw:
                        aa_logits[f'{aa}_upper'] = float('nan')
                    if save_logprob:
                        suffix = '_logprob' if save_raw else ''
                        aa_logits[f'{aa}_upper{suffix}'] = float('nan')

            # Lowercase logits (mutation predictions from Final head)
            for aa in AMINO_ACIDS:
                idx = self.lowercase_aa_to_idx.get(aa)
                if idx is not None:
                    if save_raw:
                        aa_logits[f'{aa}_lower'] = raw_logits[idx].item()
                    if save_logprob:
                        suffix = '_logprob' if save_raw else ''
                        aa_logits[f'{aa}_lower{suffix}'] = log_probs[idx].item()
                else:
                    if save_raw:
                        aa_logits[f'{aa}_lower'] = float('nan')
                    if save_logprob:
                        suffix = '_logprob' if save_raw else ''
                        aa_logits[f'{aa}_lower{suffix}'] = float('nan')

            # =====================================================================
            # [NEW] Extract component logits for post-hoc mixing experiments
            # =====================================================================
            # GL Head raw logits (if PRISM v2)
            if gl_logits is not None:
                for i, aa in enumerate(AMINO_ACIDS):
                    aa_idx = 4 + i  # AA tokens start at index 4 in ESM2 vocabulary
                    if aa_idx < gl_logits.shape[0]:
                        aa_logits[f'{aa}_gl'] = gl_logits[aa_idx].item()

            # NGL Head raw logits (if available)
            if ngl_logits is not None:
                for i, aa in enumerate(AMINO_ACIDS):
                    aa_idx = 4 + i  # AA tokens start at index 4 in ESM2 vocabulary
                    if aa_idx < ngl_logits.shape[0]:
                        aa_logits[f'{aa}_ngl'] = ngl_logits[aa_idx].item()

            # Origin head values
            if alpha_val is not None:
                aa_logits['alpha'] = alpha_val
            if trust_val is not None:
                aa_logits['trust'] = trust_val
            if logits_mut_val is not None:
                aa_logits['logits_mut'] = logits_mut_val

            results[pos] = aa_logits

        return results

    def extract_logits_for_dataframe(
        self,
        df: pd.DataFrame,
        v_gene_heavy_col: str = 'v_gene_heavy',
        j_gene_heavy_col: str = 'j_gene_heavy',
        v_gene_light_col: str = 'v_gene_light',
        j_gene_light_col: str = 'j_gene_light',
        region_mask_heavy_col: str = 'region_mask_heavy',
        region_mask_light_col: str = 'region_mask_light'
    ) -> pd.DataFrame:
        """
        Extract logits at all mutation positions for the dataframe.

        Args:
            df: DataFrame with columns: Therapeutic, HeavySequence, LightSequence,
                mutations_heavy, mutations_light, and optionally gene/region columns
            v_gene_heavy_col: Column name for heavy chain V gene
            j_gene_heavy_col: Column name for heavy chain J gene
            v_gene_light_col: Column name for light chain V gene
            j_gene_light_col: Column name for light chain J gene
            region_mask_heavy_col: Column name for heavy chain region mask
            region_mask_light_col: Column name for light chain region mask

        Returns:
            DataFrame with logits for each mutation position
        """
        all_results = []

        # Check which columns are available
        has_gene_cols = (
            self.use_germline_genes and
            self.gene_vocab is not None and
            v_gene_heavy_col in df.columns and
            j_gene_heavy_col in df.columns
        )
        has_region_cols = (
            self.use_region_mask and
            self.use_region_embedding and
            region_mask_heavy_col in df.columns
        )

        if has_gene_cols:
            print(f"  Using gene conditioning from columns: {v_gene_heavy_col}, {j_gene_heavy_col}, etc.")
        if has_region_cols:
            print(f"  Using region masks from columns: {region_mask_heavy_col}, {region_mask_light_col}")

        for idx in tqdm(range(len(df)), desc="Extracting Evo-Ab logits"):
            row = df.iloc[idx]
            therapeutic = row['Therapeutic']

            # Get gene IDs for heavy chain
            v_gene_id_heavy = None
            j_gene_id_heavy = None
            if has_gene_cols:
                v_gene_heavy = row.get(v_gene_heavy_col)
                j_gene_heavy = row.get(j_gene_heavy_col)
                if pd.notna(v_gene_heavy) and pd.notna(j_gene_heavy):
                    v_gene_id_heavy = self.gene_vocab.encode(v_gene_heavy)
                    j_gene_id_heavy = self.gene_vocab.encode(j_gene_heavy)

            # Get region mask for heavy chain
            region_mask_heavy = None
            if has_region_cols:
                region_mask_heavy = row.get(region_mask_heavy_col)
                if pd.isna(region_mask_heavy):
                    region_mask_heavy = None

            # Process heavy chain mutations
            heavy_seq = row.get('HeavySequence', '')
            mutations_heavy = parse_mutations(row.get('mutations_heavy', ''))

            if heavy_seq and heavy_seq != 'na' and mutations_heavy:
                positions = [m[0] - 1 for m in mutations_heavy]  # Convert to 0-indexed
                valid_positions = [p for p in positions if 0 <= p < len(heavy_seq)]

                logits = self.get_logits_at_positions(
                    heavy_seq,
                    valid_positions,
                    v_gene_id=v_gene_id_heavy,
                    j_gene_id=j_gene_id_heavy,
                    region_mask=region_mask_heavy
                )

                for (imgt_pos, germ_aa, mut_aa), pos in zip(mutations_heavy, positions):
                    if pos in logits:
                        result = {
                            'Therapeutic': therapeutic,
                            'chain': 'heavy',
                            'position': imgt_pos,
                            'germline_aa': germ_aa,
                            'mutated_aa': mut_aa,
                        }
                        result.update(logits[pos])
                        all_results.append(result)

            # Get gene IDs for light chain
            v_gene_id_light = None
            j_gene_id_light = None
            if has_gene_cols and v_gene_light_col in df.columns and j_gene_light_col in df.columns:
                v_gene_light = row.get(v_gene_light_col)
                j_gene_light = row.get(j_gene_light_col)
                if pd.notna(v_gene_light) and pd.notna(j_gene_light):
                    v_gene_id_light = self.gene_vocab.encode(v_gene_light)
                    j_gene_id_light = self.gene_vocab.encode(j_gene_light)

            # Get region mask for light chain
            region_mask_light = None
            if has_region_cols and region_mask_light_col in df.columns:
                region_mask_light = row.get(region_mask_light_col)
                if pd.isna(region_mask_light):
                    region_mask_light = None

            # Process light chain mutations
            light_seq = row.get('LightSequence', '')
            mutations_light = parse_mutations(row.get('mutations_light', ''))

            if light_seq and light_seq != 'na' and mutations_light:
                positions = [m[0] - 1 for m in mutations_light]  # Convert to 0-indexed
                valid_positions = [p for p in positions if 0 <= p < len(light_seq)]

                logits = self.get_logits_at_positions(
                    light_seq,
                    valid_positions,
                    v_gene_id=v_gene_id_light,
                    j_gene_id=j_gene_id_light,
                    region_mask=region_mask_light
                )

                for (imgt_pos, germ_aa, mut_aa), pos in zip(mutations_light, positions):
                    if pos in logits:
                        result = {
                            'Therapeutic': therapeutic,
                            'chain': 'light',
                            'position': imgt_pos,
                            'germline_aa': germ_aa,
                            'mutated_aa': mut_aa,
                        }
                        result.update(logits[pos])
                        all_results.append(result)

        return pd.DataFrame(all_results)


def main():
    parser = argparse.ArgumentParser(
        description='Extract amino acid logits at mutation positions using Evo-Ab (SFT_ESM2)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with gene vocabulary JSON (preferred)
  python extract_mutation_logits_evo_ab.py \\
      --data_path ../../data/therasabdab_germline.csv \\
      --config ../../configs/config_esm2_v33.yaml \\
      --checkpoint path/to/checkpoint.ckpt \\
      --gene_vocab_json ../../data/unpaired_OAS/annotated_data_final/gene_vocabulary.json

  # With region masks (if model supports region embeddings)
  python extract_mutation_logits_evo_ab.py \\
      --data_path ../../data/therasabdab_germline.csv \\
      --config ../../configs/config_esm2_v33.yaml \\
      --checkpoint path/to/checkpoint.ckpt \\
      --gene_vocab_json ../../data/unpaired_OAS/annotated_data_final/gene_vocabulary.json \\
      --use_region_mask

  # Without gene conditioning (for models without gene embedding)
  python extract_mutation_logits_evo_ab.py \\
      --data_path ../../data/therasabdab_germline.csv \\
      --config ../../configs/config_esm2_v17.yaml \\
      --checkpoint path/to/checkpoint.ckpt
        """
    )
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to therasabdab_germline.csv')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Output path for logits CSV')

    # Gene vocabulary arguments
    parser.add_argument('--gene_vocab_json', type=str,
                        default='../../data/unpaired_OAS/annotated_data_final/gene_vocabulary.json',
                        help='Path to gene vocabulary JSON file (preferred method)')

    # Gene column arguments
    parser.add_argument('--v_gene_heavy_col', type=str, default='v_gene_heavy',
                        help='Column name for heavy chain V gene (default: v_gene_heavy)')
    parser.add_argument('--j_gene_heavy_col', type=str, default='j_gene_heavy',
                        help='Column name for heavy chain J gene (default: j_gene_heavy)')
    parser.add_argument('--v_gene_light_col', type=str, default='v_gene_light',
                        help='Column name for light chain V gene (default: v_gene_light)')
    parser.add_argument('--j_gene_light_col', type=str, default='j_gene_light',
                        help='Column name for light chain J gene (default: j_gene_light)')

    # Region mask arguments
    parser.add_argument('--use_region_mask', action='store_true',
                        help='Use region masks for region-aware models')
    parser.add_argument('--region_mask_heavy_col', type=str, default='region_mask_heavy',
                        help='Column name for heavy chain region mask (default: region_mask_heavy)')
    parser.add_argument('--region_mask_light_col', type=str, default='region_mask_light',
                        help='Column name for light chain region mask (default: region_mask_light)')

    # Other arguments
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (default: cuda)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for processing multiple positions (default: 32)')

    # Output type arguments (NEW)
    parser.add_argument('--output_type', type=str, default='raw_logits',
                        choices=['log_prob', 'raw_logits', 'both'],
                        help='Type of values to output: '
                             'raw_logits (default, enables post-hoc temperature scaling), '
                             'log_prob (log-probabilities after softmax), '
                             'both (save both raw logits and log-probs)')

    args = parser.parse_args()

    print("=" * 80)
    print("Extract Mutation Position Logits - Evo-Ab (SFT_ESM2)")
    print("=" * 80)

    # Load data
    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"ERROR: Input file not found: {data_path}")
        sys.exit(1)

    print(f"\nLoading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"  Loaded {len(df)} rows")
    print(f"  Columns: {df.columns.tolist()}")

    # Count total mutations
    total_mut_heavy = df['mutations_heavy'].dropna().apply(
        lambda x: len(x.split(',')) if x else 0
    ).sum()
    total_mut_light = df['mutations_light'].dropna().apply(
        lambda x: len(x.split(',')) if x else 0
    ).sum()
    print(f"  Total mutations: {total_mut_heavy} heavy + {total_mut_light} light = {total_mut_heavy + total_mut_light}")

    # Check for gene and region columns
    print(f"\n  Gene columns available:")
    for col in [args.v_gene_heavy_col, args.j_gene_heavy_col, args.v_gene_light_col, args.j_gene_light_col]:
        status = "✓" if col in df.columns else "✗"
        print(f"    {status} {col}")

    print(f"\n  Region mask columns available:")
    for col in [args.region_mask_heavy_col, args.region_mask_light_col]:
        status = "✓" if col in df.columns else "✗"
        print(f"    {status} {col}")

    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    print(f"Output type: {args.output_type}")
    if args.output_type == 'raw_logits':
        print("  → Saving RAW LOGITS (before softmax) - enables post-hoc temperature scaling")
    elif args.output_type == 'log_prob':
        print("  → Saving LOG-PROBABILITIES (after softmax)")
    else:
        print("  → Saving BOTH raw logits and log-probabilities")

    # Initialize extractor
    extractor = EvoAbLogitExtractor(
        checkpoint_path=args.checkpoint,
        gene_vocab_json=args.gene_vocab_json,
        use_region_mask=args.use_region_mask,
        device=device,
        batch_size=args.batch_size,
        output_type=args.output_type
    )

    # Load model
    extractor.load_model()

    # Extract logits
    result_df = extractor.extract_logits_for_dataframe(
        df,
        v_gene_heavy_col=args.v_gene_heavy_col,
        j_gene_heavy_col=args.j_gene_heavy_col,
        v_gene_light_col=args.v_gene_light_col,
        j_gene_light_col=args.j_gene_light_col,
        region_mask_heavy_col=args.region_mask_heavy_col,
        region_mask_light_col=args.region_mask_light_col
    )

    # Reorder columns based on output type
    base_cols = ['Therapeutic', 'chain', 'position', 'germline_aa', 'mutated_aa']

    # Component columns (GL head, NGL head, alpha, trust, logits_mut)
    gl_cols = [f'{aa}_gl' for aa in AMINO_ACIDS]
    ngl_cols = [f'{aa}_ngl' for aa in AMINO_ACIDS]
    component_cols = ['alpha', 'trust', 'logits_mut']

    if args.output_type == 'raw_logits':
        # Raw logits only
        upper_cols = [f'{aa}_upper' for aa in AMINO_ACIDS]
        lower_cols = [f'{aa}_lower' for aa in AMINO_ACIDS]
        all_cols = base_cols + upper_cols + lower_cols + gl_cols + ngl_cols + component_cols
    elif args.output_type == 'log_prob':
        # Log-probs only (original behavior)
        upper_cols = [f'{aa}_upper' for aa in AMINO_ACIDS]
        lower_cols = [f'{aa}_lower' for aa in AMINO_ACIDS]
        all_cols = base_cols + upper_cols + lower_cols + gl_cols + ngl_cols + component_cols
    else:  # 'both'
        # Both raw logits and log-probs
        upper_raw = [f'{aa}_upper' for aa in AMINO_ACIDS]
        lower_raw = [f'{aa}_lower' for aa in AMINO_ACIDS]
        upper_logprob = [f'{aa}_upper_logprob' for aa in AMINO_ACIDS]
        lower_logprob = [f'{aa}_lower_logprob' for aa in AMINO_ACIDS]
        all_cols = base_cols + upper_raw + lower_raw + upper_logprob + lower_logprob + gl_cols + ngl_cols + component_cols

    # Only keep columns that exist
    existing_cols = [c for c in all_cols if c in result_df.columns]
    result_df = result_df[existing_cols]

    # Save
    if args.output_path:
        output_path = Path(args.output_path)
    else:
        output_path = data_path.parent / f"{data_path.stem}_evo_ab_logits.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False)

    print(f"\n{'='*80}")
    print("SUMMARY")
    print("="*80)
    print(f"Results saved to: {output_path}")
    print(f"Total rows: {len(result_df)}")
    print(f"Chains: {result_df['chain'].value_counts().to_dict()}")
    print(f"Output type: {args.output_type}")
    print(f"\nColumns ({len(result_df.columns)} total): {list(result_df.columns)[:10]}...")

    # Add post-hoc temperature scaling guide if raw_logits
    if args.output_type in ['raw_logits', 'both']:
        print(f"\n{'─'*60}")
        print("POST-HOC TEMPERATURE SCALING GUIDE")
        print("─"*60)
        print("To apply temperature scaling to raw logits:")
        print("  1. Load the CSV: df = pd.read_csv('output.csv')")
        print("  2. Scale logits: scaled = df['{AA}_upper'] / temperature")
        print("  3. Apply softmax: log_probs = log_softmax(scaled)")
        print("")
        print("Example temperature values to try:")
        print("  T=0.5  → Sharper predictions (more confident)")
        print("  T=1.0  → Original (no scaling)")
        print("  T=2.0  → Softer predictions (more uniform)")
        print("─"*60)

    # [SIMPLE HEAD DETECTION] Add warning in summary
    if extractor.is_simple_head:
        print(f"\n{'!'*60}")
        print("[IMPORTANT] SIMPLE HEAD MODEL - INTERPRETATION GUIDE")
        print("!"*60)
        print("  Model type: SIMPLE HEAD (use_multihead_architecture=False)")
        print("  ")
        print("  VALID columns for analysis:")
        print("    - {AA}_upper (e.g., A_upper, C_upper, ..., Y_upper)")
        print("    These represent standard amino acid identity predictions.")
        print("  ")
        print("  INVALID columns (ignore for simple head models):")
        print("    - {AA}_lower (e.g., A_lower, C_lower, ..., Y_lower)")
        print("    These are NOT semantically meaningful for simple head.")
        print("!"*60)

    # Sample output
    if len(result_df) > 0:
        print(f"\nSample (first 3 rows):")
        print(result_df.head(3).to_string())

    print("=" * 80)


if __name__ == "__main__":
    main()
