#!/usr/bin/env python
# coding: utf-8

"""
Data loading utilities including Dataset, DataModule, and collate functions.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import random


class SeqSeqDataset(torch.utils.data.Dataset):
    """A custom PyTorch dataset for protein sequence-sequence data.

    Supports both paired (heavy+light) and unpaired (heavy-only or light-only) sequences.
    For unpaired data, the missing chain column should be None/NaN.

    """

    def __init__(self, data_frame, tokenizer, mask_prob, use_germline_genes=False, use_region_embedding=False,
                 use_3class_origin=False, use_synth_masking=False, ngl_label_noise_flips=0):
        self.df = data_frame
        self.tokenizer = tokenizer
        self.mask_prob = mask_prob
        self.use_germline_genes = use_germline_genes
        self.use_region_embedding = use_region_embedding
        self.use_3class_origin = use_3class_origin
        self.use_synth_masking = use_synth_masking
        self.ngl_label_noise_flips = ngl_label_noise_flips  # Number of GL/NGL positions to flip per chain

        # Detect available gene columns
        self.v_gene_col_heavy = 'v_gene_heavy' if 'v_gene_heavy' in self.df.columns else None
        self.v_gene_col_light = 'v_gene_light' if 'v_gene_light' in self.df.columns else None
        self.j_gene_col_heavy = 'j_gene_heavy' if 'j_gene_heavy' in self.df.columns else None
        self.j_gene_col_light = 'j_gene_light' if 'j_gene_light' in self.df.columns else None

        # Fallback to single column names
        if self.v_gene_col_heavy is None and 'v_gene' in self.df.columns:
            self.v_gene_col_heavy = 'v_gene'
        if self.j_gene_col_heavy is None and 'j_gene' in self.df.columns:
            self.j_gene_col_heavy = 'j_gene'

        # Detect region mask columns
        self.region_mask_col_heavy = 'region_mask_heavy' if 'region_mask_heavy' in self.df.columns else None
        self.region_mask_col_light = 'region_mask_light' if 'region_mask_light' in self.df.columns else None

        # Fallback to single column name
        if self.region_mask_col_heavy is None and 'region_mask' in self.df.columns:
            self.region_mask_col_heavy = 'region_mask'

        # Detect if this is unpaired data (has 'chain_type' column)
        self.is_unpaired = 'chain_type' in self.df.columns
        if self.is_unpaired:
            heavy_count = (self.df['chain_type'] == 'heavy').sum()
            light_count = (self.df['chain_type'] == 'light').sum()
            print(f"[SeqSeqDataset] Unpaired mode detected: {heavy_count} heavy, {light_count} light chains")

    def __len__(self):
        return len(self.df)

    def _is_valid_sequence(self, seq):
        """Check if a sequence is valid (not None/NaN/pd.NA and not empty)."""
        if seq is None:
            return False
        # Handle pandas NA type (from parquet files)
        if pd.isna(seq):
            return False
        if isinstance(seq, str) and len(seq) == 0:
            return False
        return True

    def _apply_label_noise(self, ngl_mask, sequence, seq_len):
        """Flip N NGL/GL labels deterministically per chain.

        Uses hash(sequence) as seed so the same positions are flipped every epoch.
        Flips are uniform random over all valid AA positions:
          - GL(0) → NGL(1): simulates false positive mutation call
          - NGL(1) → GL(0): simulates missed mutation (back mutation)
        """
        n_flips = min(self.ngl_label_noise_flips, seq_len)
        if n_flips <= 0:
            return ngl_mask

        rng = np.random.RandomState(hash(sequence) % (2**31))
        flip_positions = rng.choice(seq_len, size=n_flips, replace=False)

        ngl_mask = ngl_mask.copy()
        for pos in flip_positions:
            ngl_mask[pos] = 1 - ngl_mask[pos]  # 0→1 or 1→0

        return ngl_mask

    # concatenate heavy and light chains (or single chain for unpaired)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Get sequences (may be None for unpaired data)
        heavy = row.get('HEAVY_CHAIN_AA_SEQUENCE', None)
        light = row.get('LIGHT_CHAIN_AA_SEQUENCE', None)

        # Check which chains are valid
        has_heavy = self._is_valid_sequence(heavy)
        has_light = self._is_valid_sequence(light)

        # Build combined sequence based on available chains
        if has_heavy and has_light:
            # Paired: heavy + <CLS><CLS> + light
            combined = f"{heavy}{self.tokenizer.cls_token}{self.tokenizer.cls_token}{light}"
            heavy_len = len(heavy)
            light_len = len(light)
        elif has_heavy:
            # Unpaired heavy only
            combined = heavy
            heavy_len = len(heavy)
            light_len = 0
        elif has_light:
            # Unpaired light only
            combined = light
            heavy_len = 0
            light_len = len(light)
        else:
            raise ValueError(f"Row {idx} has no valid sequence (both heavy and light are None/empty)")

        encoding = self.tokenizer(
            combined,
            padding="max_length",
            truncation=True,
            max_length=320,
            add_special_tokens=True,
        )
        seq_token_ids = encoding["input_ids"]

        # Create NGL mask from mutation codes
        # Initialize masks as zeros (GL positions)
        ngl_mask_heavy = np.zeros(heavy_len, dtype=np.int32) if heavy_len > 0 else np.array([], dtype=np.int32)
        ngl_mask_light = np.zeros(light_len, dtype=np.int32) if light_len > 0 else np.array([], dtype=np.int32)

        # [v38] 3-class origin: first apply synthetic NGL (class 1), then SHM mutations override to class 2
        if self.use_3class_origin and heavy_len > 0:
            synth_col = 'synthetic_ngl_heavy'
            if synth_col in row.index and pd.notna(row[synth_col]) and str(row[synth_col]):
                synth_str = str(row[synth_col])
                for pos in range(min(len(synth_str), heavy_len)):
                    if synth_str[pos] == '1':
                        ngl_mask_heavy[pos] = 1  # SynNGL = class 1

        if self.use_3class_origin and light_len > 0:
            synth_col = 'synthetic_ngl_light'
            if synth_col in row.index and pd.notna(row[synth_col]) and str(row[synth_col]):
                synth_str = str(row[synth_col])
                for pos in range(min(len(synth_str), light_len)):
                    if synth_str[pos] == '1':
                        ngl_mask_light[pos] = 1  # SynNGL = class 1

        # Check if mutation code columns exist — SHM mutations set class 2 (3-class) or 1 (2-class)
        mut_class = 2 if self.use_3class_origin else 1

        if 'hc_mut_codes' in row.index and pd.notna(row['hc_mut_codes']) and row['hc_mut_codes']:
            hc_mut_str = str(row['hc_mut_codes'])
            hc_muts = hc_mut_str.replace(',', ';').split(';')
            for mut in hc_muts:
                if mut.strip():
                    pos_str = ''.join(filter(str.isdigit, mut))
                    if pos_str:
                        pos = int(pos_str) - 1  # Convert to 0-indexed
                        if 0 <= pos < heavy_len:
                            ngl_mask_heavy[pos] = mut_class

        if 'lc_mut_codes' in row.index and pd.notna(row['lc_mut_codes']) and row['lc_mut_codes']:
            lc_mut_str = str(row['lc_mut_codes'])
            lc_muts = lc_mut_str.replace(',', ';').split(';')
            for mut in lc_muts:
                if mut.strip():
                    pos_str = ''.join(filter(str.isdigit, mut))
                    if pos_str:
                        pos = int(pos_str) - 1  # Convert to 0-indexed
                        if 0 <= pos < light_len:
                            ngl_mask_light[pos] = mut_class

        # [Rebuttal] Apply deterministic NGL label noise: flip N positions per chain
        # Uses hash of sequence as seed → same positions flipped every epoch
        if self.ngl_label_noise_flips > 0:
            if heavy_len > 0:
                ngl_mask_heavy = self._apply_label_noise(ngl_mask_heavy, heavy, heavy_len)
            if light_len > 0:
                ngl_mask_light = self._apply_label_noise(ngl_mask_light, light, light_len)

        # Concatenate masks based on available chains
        if has_heavy and has_light:
            # Paired: heavy + [0, 0] for CLS tokens + light
            ngl_mask = np.concatenate((ngl_mask_heavy, [0, 0], ngl_mask_light))
        elif has_heavy:
            # Unpaired heavy only
            ngl_mask = ngl_mask_heavy
        else:
            # Unpaired light only
            ngl_mask = ngl_mask_light

        # Pad to match tokenized sequence length (account for leading CLS and trailing padding)
        ngl_mask = np.pad(ngl_mask, (0, len(seq_token_ids) - len(ngl_mask)), 'constant', constant_values=0)

        # [NEW] Extract region IDs if enabled
        # Region IDs: 0=special tokens, 1=FR1, 2=CDR1, 3=FR2, 4=CDR2, 5=FR3, 6=CDR3, 7=FR4
        region_ids = None
        if self.use_region_embedding:
            # Initialize with 0 (special token region ID)
            region_ids = np.zeros(len(seq_token_ids), dtype=np.int32)

            # Get region masks for heavy and light chains
            region_mask_heavy_str = None
            region_mask_light_str = None

            if self.region_mask_col_heavy and self.region_mask_col_heavy in row.index:
                region_mask_heavy_str = row[self.region_mask_col_heavy]
                if pd.isna(region_mask_heavy_str):
                    region_mask_heavy_str = None

            if self.region_mask_col_light and self.region_mask_col_light in row.index:
                region_mask_light_str = row[self.region_mask_col_light]
                if pd.isna(region_mask_light_str):
                    region_mask_light_str = None

            # Parse region masks (strings of digits like "1111122223333...")
            # Handle different cases: paired, unpaired heavy, unpaired light
            if has_heavy and region_mask_heavy_str is not None:
                try:
                    region_vals_heavy = [int(c) for c in str(region_mask_heavy_str)]
                    # Position 0 is [CLS], so heavy chain starts at position 1
                    for i, val in enumerate(region_vals_heavy[:heavy_len]):
                        region_ids[1 + i] = val
                except ValueError:
                    pass  # Invalid region mask, keep as zeros

            if has_light and region_mask_light_str is not None:
                try:
                    region_vals_light = [int(c) for c in str(region_mask_light_str)]
                    if has_heavy:
                        # Paired: Light chain starts after [CLS] + heavy + [CLS][CLS]
                        light_start = 1 + heavy_len + 2
                    else:
                        # Unpaired light only: Light chain starts after [CLS]
                        light_start = 1
                    for i, val in enumerate(region_vals_light[:light_len]):
                        if light_start + i < len(region_ids):
                            region_ids[light_start + i] = val
                except ValueError:
                    pass  # Invalid region mask, keep as zeros

        # [NEW] Extract V/J gene labels if enabled
        v_gene_heavy = None
        v_gene_light = None
        j_gene_heavy = None
        j_gene_light = None

        if self.use_germline_genes:
            if self.v_gene_col_heavy and self.v_gene_col_heavy in row.index:
                v_gene_heavy = row[self.v_gene_col_heavy]
                if pd.isna(v_gene_heavy):
                    v_gene_heavy = None

            if self.v_gene_col_light and self.v_gene_col_light in row.index:
                v_gene_light = row[self.v_gene_col_light]
                if pd.isna(v_gene_light):
                    v_gene_light = None

            if self.j_gene_col_heavy and self.j_gene_col_heavy in row.index:
                j_gene_heavy = row[self.j_gene_col_heavy]
                if pd.isna(j_gene_heavy):
                    j_gene_heavy = None

            if self.j_gene_col_light and self.j_gene_col_light in row.index:
                j_gene_light = row[self.j_gene_col_light]
                if pd.isna(j_gene_light):
                    j_gene_light = None

        # [v40] Extract SynNGL mask and MPNN GL probabilities if synth masking enabled
        synth_mask = None
        mpnn_gl_prob = None
        if self.use_synth_masking:
            # Build synth_mask from synthetic_ngl columns (same logic as 3-class but binary)
            synth_mask_heavy = np.zeros(heavy_len, dtype=np.int32) if heavy_len > 0 else np.array([], dtype=np.int32)
            synth_mask_light = np.zeros(light_len, dtype=np.int32) if light_len > 0 else np.array([], dtype=np.int32)

            if heavy_len > 0:
                synth_col = 'synthetic_ngl_heavy'
                if synth_col in row.index and pd.notna(row[synth_col]) and str(row[synth_col]):
                    synth_str = str(row[synth_col])
                    for pos in range(min(len(synth_str), heavy_len)):
                        if synth_str[pos] == '1':
                            synth_mask_heavy[pos] = 1

            if light_len > 0:
                synth_col = 'synthetic_ngl_light'
                if synth_col in row.index and pd.notna(row[synth_col]) and str(row[synth_col]):
                    synth_str = str(row[synth_col])
                    for pos in range(min(len(synth_str), light_len)):
                        if synth_str[pos] == '1':
                            synth_mask_light[pos] = 1

            # Concatenate synth masks
            if has_heavy and has_light:
                synth_mask = np.concatenate((synth_mask_heavy, [0, 0], synth_mask_light))
            elif has_heavy:
                synth_mask = synth_mask_heavy
            else:
                synth_mask = synth_mask_light

            synth_mask = np.pad(synth_mask, (0, len(seq_token_ids) - len(synth_mask)), 'constant', constant_values=0)

            # Extract MPNN germline probabilities
            mpnn_gl_prob_heavy = np.ones(heavy_len, dtype=np.float32) if heavy_len > 0 else np.array([], dtype=np.float32)
            mpnn_gl_prob_light = np.ones(light_len, dtype=np.float32) if light_len > 0 else np.array([], dtype=np.float32)

            if heavy_len > 0 and 'mpnn_gl_prob_heavy' in row.index and pd.notna(row.get('mpnn_gl_prob_heavy')):
                import pickle
                try:
                    probs = pickle.loads(row['mpnn_gl_prob_heavy'])
                    mpnn_gl_prob_heavy[:min(len(probs), heavy_len)] = probs[:heavy_len]
                except Exception:
                    pass

            if light_len > 0 and 'mpnn_gl_prob_light' in row.index and pd.notna(row.get('mpnn_gl_prob_light')):
                import pickle
                try:
                    probs = pickle.loads(row['mpnn_gl_prob_light'])
                    mpnn_gl_prob_light[:min(len(probs), light_len)] = probs[:light_len]
                except Exception:
                    pass

            # Concatenate MPNN probs
            if has_heavy and has_light:
                mpnn_gl_prob = np.concatenate((mpnn_gl_prob_heavy, [1.0, 1.0], mpnn_gl_prob_light))
            elif has_heavy:
                mpnn_gl_prob = mpnn_gl_prob_heavy
            else:
                mpnn_gl_prob = mpnn_gl_prob_light

            mpnn_gl_prob = np.pad(mpnn_gl_prob, (0, len(seq_token_ids) - len(mpnn_gl_prob)), 'constant', constant_values=1.0)

        # Return tuple based on enabled features
        # Order: (token_ids, ngl_mask, [v_gene_heavy, v_gene_light, j_gene_heavy, j_gene_light], [region_ids], [synth_mask, mpnn_gl_prob])
        base = [torch.tensor(seq_token_ids, dtype=torch.long), ngl_mask]

        if self.use_germline_genes:
            base.extend([v_gene_heavy, v_gene_light, j_gene_heavy, j_gene_light])

        if self.use_region_embedding:
            base.append(region_ids)

        if self.use_synth_masking:
            base.append(synth_mask)
            base.append(mpnn_gl_prob)

        return tuple(base)


def make_collate_fn_multihead(tokenizer, mask_prob,
                               gene_vocab=None, use_germline_genes=False,
                               ngl_targeted_masking=False, ngl_mask_prob=0.8,
                               use_region_embedding=False, silent=False,
                               use_region_masking=False, cdr_mask_prob=0.4, fr_mask_prob=0.15,
                               use_fr_span_masking=False, fr_span_min_length=3, fr_span_max_length=6,
                               cdr_ngl_mask_prob=None, fr_ngl_mask_prob=None,
                               use_coherence_masking=False, coherence_prob=0.3,
                               coherence_ngl_mask_prob=0.5,
                               use_3class_origin=False,
                               use_synth_masking=False,
                               cdr_synth_mask_prob=0.45, fr_synth_mask_prob=0.30,
                               use_mpnn_origin_smoothing=False,
                               origin_label_smooth_factor=0.2):
    """
    Collate function for Decoupled Multi-Task Head Architecture.

    This collate function prepares data for the AA + Mutation dual-head model:
    - Input: ALL tokens forced to UPPERCASE (solves untrained embedding issues)
    - labels_aa: UPPERCASE versions of target tokens (for AA identity head)
    - labels_mut: Binary tensor (1.0=NGL, 0.0=GL) for mutation detection head

    Args:
        tokenizer: The tokenizer (must have lowercase AA tokens added)
        mask_prob: Probability of masking GL tokens (default 0.15)
        gene_vocab: GeneVocabulary instance for encoding gene names to IDs
        use_germline_genes: Whether to include gene IDs in batch output
        ngl_targeted_masking: If True, apply higher masking probability to NGL positions
        ngl_mask_prob: Probability of masking NGL tokens (default 0.8)
        use_region_embedding: Whether to include region IDs in batch output
        silent: If True, suppress initialization log messages (for repeated calls)
        use_region_masking: If True, apply region-wise masking probabilities (v19)
        cdr_mask_prob: Masking probability for CDR regions (default 0.4)
        fr_mask_prob: Masking probability for FR regions (default 0.15)
        use_fr_span_masking: [NEW v23] If True, use span masking for FR regions
        fr_span_min_length: [NEW v23] Minimum span length for FR masking (default 3)
        fr_span_max_length: [NEW v23] Maximum span length for FR masking (default 6)

    Returns:
        Collate function that returns:
        - input_ids: [B, L] - All uppercase token IDs
        - labels_aa: [B, L] - Uppercase AA labels (-100 for non-masked positions)
        - labels_mut: [B, L] - Binary mutation labels (-1.0 for non-masked positions)
        - attention_mask: [B, L] - Attention mask
        - ngl_masks_tensor: [B, L] - Boolean mask for NGL positions
        - (optional) v_gene_ids, j_gene_ids, region_ids_tensor

    Region-wise Masking (v19):
        When use_region_masking=True and region_ids are available:
        - CDR regions (region_id ∈ {2, 4, 6}): masked with cdr_mask_prob (default 0.4)
        - FR regions (region_id ∈ {1, 3, 5, 7}): masked with fr_mask_prob (default 0.15)
        - Special tokens (region_id == 0): never masked
        This overrides ngl_targeted_masking when region_ids are available.

    FR Span Masking (v23):
        When use_fr_span_masking=True (requires use_region_masking=True):
        - CDR regions: Random token masking (unchanged from v19)
        - FR regions: Span masking - randomly select starting positions and mask
          continuous spans of fr_span_min_length to fr_span_max_length tokens
        - This forces the model to learn long-range structural dependencies in FRs
        - Span masking is harder than random masking as it requires contextual
          understanding to reconstruct contiguous missing regions
    """

    PAD = tokenizer.pad_token_id
    CLS = tokenizer.cls_token_id
    MASK = tokenizer.mask_token_id
    V = tokenizer.vocab_size

    # Create uppercase/lowercase mappings
    lowercase_aa = ['a', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l',
                   'm', 'n', 'p', 'q', 'r', 's', 't', 'v', 'w', 'y']

    # lower -> upper mapping (for forcing inputs to uppercase)
    uppercase_aa_mapping = {
        tokenizer.convert_tokens_to_ids(aa.lower()): tokenizer.convert_tokens_to_ids(aa.upper())
        for aa in lowercase_aa
    }

    # upper -> lower mapping (for creating lowercase labels when needed)
    lowercase_aa_mapping = {v: k for k, v in uppercase_aa_mapping.items()}

    if not silent:
        print(f"[make_collate_fn_multihead] Multihead collate function initialized")
        if use_synth_masking and use_region_masking and ngl_targeted_masking and cdr_ngl_mask_prob is not None:
            print(f"[make_collate_fn_multihead] Combined 6-rate masking ENABLED:")
            print(f"  CDR-NGL={cdr_ngl_mask_prob}, FR-NGL={fr_ngl_mask_prob}")
            print(f"  CDR-SynNGL={cdr_synth_mask_prob}, FR-SynNGL={fr_synth_mask_prob}")
            print(f"  CDR-GL={cdr_mask_prob}, FR-GL={fr_mask_prob}")
        elif use_region_masking and ngl_targeted_masking and cdr_ngl_mask_prob is not None:
            print(f"[make_collate_fn_multihead] Combined 4-rate masking ENABLED:")
            print(f"  CDR-NGL={cdr_ngl_mask_prob}, FR-NGL={fr_ngl_mask_prob}, CDR-GL={cdr_mask_prob}, FR-GL={fr_mask_prob}")
        elif use_region_masking:
            print(f"[make_collate_fn_multihead] Region-wise masking ENABLED: CDR={cdr_mask_prob}, FR={fr_mask_prob}")
            if use_fr_span_masking:
                print(f"[make_collate_fn_multihead] FR Span Masking ENABLED: span_length={fr_span_min_length}-{fr_span_max_length}")
        elif ngl_targeted_masking:
            print(f"[make_collate_fn_multihead] NGL-targeted masking: GL={mask_prob}, NGL={ngl_mask_prob}")
        else:
            print(f"[make_collate_fn_multihead] Uniform masking: {mask_prob}")
        if use_coherence_masking:
            print(f"[make_collate_fn_multihead] Coherence Masking ENABLED: prob={coherence_prob}, NGL_mask={coherence_ngl_mask_prob}")

    def collate_fn(batch):
        # Unpack batch based on enabled features
        # [v40] Flexible unpacking: each item is a tuple of varying length
        # Order: (token_ids, ngl_mask, [genes...], [region_ids], [synth_mask, mpnn_gl_prob])
        synth_masks_list = None
        mpnn_gl_probs_list = None
        region_ids_list = None
        v_genes_heavy = v_genes_light = j_genes_heavy = j_genes_light = None

        if use_synth_masking:
            if use_germline_genes and use_region_embedding:
                input_ids_list, ngl_masks, v_genes_heavy, v_genes_light, j_genes_heavy, j_genes_light, region_ids_list, synth_masks_list, mpnn_gl_probs_list = zip(*batch)
            elif use_germline_genes:
                input_ids_list, ngl_masks, v_genes_heavy, v_genes_light, j_genes_heavy, j_genes_light, synth_masks_list, mpnn_gl_probs_list = zip(*batch)
            elif use_region_embedding:
                input_ids_list, ngl_masks, region_ids_list, synth_masks_list, mpnn_gl_probs_list = zip(*batch)
            else:
                input_ids_list, ngl_masks, synth_masks_list, mpnn_gl_probs_list = zip(*batch)
        elif use_germline_genes and use_region_embedding:
            input_ids_list, ngl_masks, v_genes_heavy, v_genes_light, j_genes_heavy, j_genes_light, region_ids_list = zip(*batch)
        elif use_germline_genes:
            input_ids_list, ngl_masks, v_genes_heavy, v_genes_light, j_genes_heavy, j_genes_light = zip(*batch)
        elif use_region_embedding:
            input_ids_list, ngl_masks, region_ids_list = zip(*batch)
        else:
            input_ids_list, ngl_masks = zip(*batch)

        B = len(input_ids_list)
        input_ids = torch.stack(input_ids_list, dim=0)  # [B, L]

        # Stack NGL masks
        ngl_masks_raw = torch.from_numpy(np.stack(ngl_masks, axis=0))  # [B, L] int32

        # [v38] For masking purposes, any non-zero value is "mutated" (NGL or SynNGL)
        if use_3class_origin:
            ngl_masks_tensor = (ngl_masks_raw > 0)  # bool: True for SynNGL(1) + NGL(2)
        else:
            ngl_masks_tensor = ngl_masks_raw.bool()  # bool: True for NGL(1)

        # [v40] Stack synth masks and MPNN probs if available
        synth_masks_tensor = None
        mpnn_gl_probs_tensor = None
        if use_synth_masking and synth_masks_list is not None:
            synth_masks_tensor = torch.from_numpy(np.stack(synth_masks_list, axis=0)).bool()
            mpnn_gl_probs_tensor = torch.from_numpy(np.stack(mpnn_gl_probs_list, axis=0)).float()

        # =====================================================================
        # Step 1: Create labels_aa as UPPERCASE versions of original tokens
        # =====================================================================
        original_tokens = input_ids.clone()

        labels_aa = input_ids.clone()
        for lower_id, upper_id in uppercase_aa_mapping.items():
            labels_aa[labels_aa == lower_id] = upper_id

        # =====================================================================
        # Step 2: Create labels_mut
        # [v38] 3-class: long tensor with classes 0/1/2, ignore=-1
        #       2-class: float tensor with 0.0/1.0, ignore=-1.0
        # =====================================================================
        if use_3class_origin:
            labels_mut = ngl_masks_raw.long()  # [B, L] - 0=GL, 1=SynNGL, 2=NGL
        else:
            labels_mut = ngl_masks_tensor.float()  # [B, L] - 1.0 for NGL, 0.0 for GL

            # [v40 Approach B] MPNN-based origin label smoothing
            # SynNGL positions get soft labels: (1 - mpnn_gl_prob) * smooth_factor
            # NGL positions stay 1.0, pure GL stays 0.0
            if use_mpnn_origin_smoothing and synth_masks_tensor is not None and mpnn_gl_probs_tensor is not None:
                synth_only = synth_masks_tensor & ~ngl_masks_tensor  # SynNGL but not NGL
                smooth_values = (1.0 - mpnn_gl_probs_tensor) * origin_label_smooth_factor
                labels_mut[synth_only] = smooth_values[synth_only]

        # =====================================================================
        # Step 3: Force ALL input_ids to UPPERCASE
        # =====================================================================
        for lower_id, upper_id in uppercase_aa_mapping.items():
            input_ids[input_ids == lower_id] = upper_id

        # =====================================================================
        # Step 4: Apply masking (Region-wise, NGL-targeted, or uniform)
        # =====================================================================
        # Identify special tokens (don't mask these)
        special = (
            (input_ids == tokenizer.cls_token_id)
            | (input_ids == tokenizer.pad_token_id)
        )

        # =====================================================================
        # [NEW v37] Coherence Masking Mode
        # With probability coherence_prob, switch to coherence mode:
        # - GL positions are NEVER masked (all visible)
        # - Only NGL positions are masked at coherence_ngl_mask_prob
        # This forces the model to predict NGL from full GL context
        # =====================================================================
        coherence_active = False
        if use_coherence_masking and random.random() < coherence_prob:
            coherence_active = True
            probability_matrix = torch.zeros_like(input_ids, dtype=torch.float)
            probability_matrix[ngl_masks_tensor] = coherence_ngl_mask_prob
            probability_matrix[special] = 0.0
            rand = torch.rand(input_ids.shape)
            to_mask = (rand < probability_matrix)

        # =====================================================================
        # [NEW v35.1] Combined 4-Rate Masking Strategy
        # When both region_masking and ngl_targeted_masking are enabled with
        # cdr_ngl_mask_prob set, apply 4 different masking rates:
        #   CDR-NGL → cdr_ngl_mask_prob, FR-NGL → fr_ngl_mask_prob
        #   CDR-GL → cdr_mask_prob, FR-GL → fr_mask_prob
        # =====================================================================
        elif not coherence_active and (use_region_masking and ngl_targeted_masking and cdr_ngl_mask_prob is not None
                and use_region_embedding and region_ids_list is not None):
            region_ids_tensor_local = torch.from_numpy(np.stack(region_ids_list, axis=0)).long()
            B, L = region_ids_tensor_local.shape

            # Region masks
            cdr_mask = (region_ids_tensor_local == 1) | (region_ids_tensor_local == 3) | (region_ids_tensor_local == 5)
            fr_mask = (region_ids_tensor_local == 0) | (region_ids_tensor_local == 2) | (region_ids_tensor_local == 4) | (region_ids_tensor_local == 6)

            # Build probability matrix (4-rate or 6-rate with synth masking)
            probability_matrix = torch.full(labels_aa.shape, mask_prob, dtype=torch.float)

            # [v40] 6-rate masking: NGL > SynNGL > GL priority
            if use_synth_masking and synth_masks_tensor is not None:
                synth_only = synth_masks_tensor & ~ngl_masks_tensor  # SynNGL without NGL
                pure_gl = ~ngl_masks_tensor & ~synth_masks_tensor    # Pure GL

                # NGL positions (highest priority)
                _fr_ngl = fr_ngl_mask_prob if fr_ngl_mask_prob is not None else fr_mask_prob
                probability_matrix.masked_fill_(cdr_mask & ngl_masks_tensor, cdr_ngl_mask_prob)
                probability_matrix.masked_fill_(fr_mask & ngl_masks_tensor, _fr_ngl)

                # SynNGL positions (middle priority)
                probability_matrix.masked_fill_(cdr_mask & synth_only, cdr_synth_mask_prob)
                probability_matrix.masked_fill_(fr_mask & synth_only, fr_synth_mask_prob)

                # Pure GL positions (lowest priority)
                probability_matrix.masked_fill_(cdr_mask & pure_gl, cdr_mask_prob)
                probability_matrix.masked_fill_(fr_mask & pure_gl, fr_mask_prob)
            else:
                # Standard 4-rate masking
                gl_mask = ~ngl_masks_tensor
                probability_matrix.masked_fill_(cdr_mask & gl_mask, cdr_mask_prob)
                probability_matrix.masked_fill_(fr_mask & gl_mask, fr_mask_prob)

                _fr_ngl = fr_ngl_mask_prob if fr_ngl_mask_prob is not None else fr_mask_prob
                probability_matrix.masked_fill_(cdr_mask & ngl_masks_tensor, cdr_ngl_mask_prob)
                probability_matrix.masked_fill_(fr_mask & ngl_masks_tensor, _fr_ngl)

            # Special tokens: never mask
            probability_matrix.masked_fill_(special, 0.0)

            rand = torch.rand(input_ids.shape)
            to_mask = (rand < probability_matrix)

        # =====================================================================
        # [NEW v19] Region-wise Masking Strategy
        # CDR regions (2, 4, 6) get higher mask prob, FR regions (1, 3, 5, 7) get lower
        # This forces balanced learning between conserved FRs and variable CDRs
        #
        # [NEW v23] FR Span Masking Extension
        # When use_fr_span_masking=True, FR regions use span masking instead of random
        # This forces the model to learn long-range structural dependencies
        # =====================================================================
        elif use_region_masking and use_region_embedding and region_ids_list is not None:
            # Stack region IDs early for masking (will be returned later too)
            region_ids_tensor_local = torch.from_numpy(np.stack(region_ids_list, axis=0)).long()  # [B, L]
            B, L = region_ids_tensor_local.shape

            # [FIX v26] Corrected Region ID mapping based on actual data:
            # 0=FR1, 1=CDR1, 2=FR2, 3=CDR2, 4=FR3, 5=CDR3, 6=FR4
            # CDR regions: {1, 3, 5} (CDR1, CDR2, CDR3)
            cdr_mask = (region_ids_tensor_local == 1) | (region_ids_tensor_local == 3) | (region_ids_tensor_local == 5)
            # FR regions: {0, 2, 4, 6} (FR1, FR2, FR3, FR4)
            fr_mask = (region_ids_tensor_local == 0) | (region_ids_tensor_local == 2) | (region_ids_tensor_local == 4) | (region_ids_tensor_local == 6)

            # =====================================================================
            # [NEW v23] FR Span Masking: mask continuous spans in FR regions
            # =====================================================================
            if use_fr_span_masking:
                # Initialize to_mask tensor
                to_mask = torch.zeros(B, L, dtype=torch.bool)

                for b in range(B):
                    # ---------------------------------------------------------
                    # CDR regions: Random masking (unchanged from v19)
                    # ---------------------------------------------------------
                    cdr_positions = cdr_mask[b].nonzero(as_tuple=False).flatten()
                    if len(cdr_positions) > 0:
                        # Random masking with cdr_mask_prob
                        cdr_rand = torch.rand(len(cdr_positions))
                        cdr_to_mask = cdr_positions[cdr_rand < cdr_mask_prob]
                        to_mask[b, cdr_to_mask] = True

                    # ---------------------------------------------------------
                    # FR regions: Span masking (NEW v23)
                    # [FIX v26] Corrected FR region IDs: {0, 2, 4, 6}
                    # ---------------------------------------------------------
                    for fr_region_id in [0, 2, 4, 6]:
                        fr_region_positions = (region_ids_tensor_local[b] == fr_region_id).nonzero(as_tuple=False).flatten()

                        if len(fr_region_positions) < fr_span_min_length:
                            # Region too small for span masking, use random masking
                            if len(fr_region_positions) > 0:
                                fr_rand = torch.rand(len(fr_region_positions))
                                fr_to_mask = fr_region_positions[fr_rand < fr_mask_prob]
                                to_mask[b, fr_to_mask] = True
                            continue

                        # Calculate how many tokens to mask based on fr_mask_prob
                        n_tokens_in_region = len(fr_region_positions)
                        target_n_masked = int(n_tokens_in_region * fr_mask_prob)

                        if target_n_masked < fr_span_min_length:
                            # Not enough tokens to mask, skip or mask minimum
                            if target_n_masked > 0:
                                # Mask a few random tokens instead
                                fr_rand = torch.rand(len(fr_region_positions))
                                fr_to_mask = fr_region_positions[fr_rand < fr_mask_prob]
                                to_mask[b, fr_to_mask] = True
                            continue

                        # Find contiguous segments in this FR region
                        # (positions should be contiguous, but verify)
                        positions_sorted = fr_region_positions.sort()[0]

                        # Mask spans until we reach target number of masked tokens
                        n_masked = 0
                        max_attempts = 20  # Prevent infinite loops
                        attempts = 0

                        while n_masked < target_n_masked and attempts < max_attempts:
                            attempts += 1
                            # Random span length
                            span_len = random.randint(fr_span_min_length, fr_span_max_length)
                            span_len = min(span_len, target_n_masked - n_masked)  # Don't exceed target

                            if span_len < fr_span_min_length:
                                break

                            # Random starting position within the region
                            max_start_idx = len(positions_sorted) - span_len
                            if max_start_idx < 0:
                                break

                            start_idx = random.randint(0, max_start_idx)
                            span_positions = positions_sorted[start_idx:start_idx + span_len]

                            # Check if any position already masked
                            already_masked = to_mask[b, span_positions].any().item()
                            if already_masked:
                                continue  # Try another span

                            # Mask this span
                            to_mask[b, span_positions] = True
                            n_masked += span_len

                # Don't mask special tokens
                to_mask = to_mask & ~special

            else:
                # ---------------------------------------------------------
                # Standard v19 behavior: Random masking for both CDR and FR
                # ---------------------------------------------------------
                # Build probability matrix based on region IDs
                probability_matrix = torch.full(labels_aa.shape, mask_prob, dtype=torch.float)

                # CDR regions: mask with cdr_mask_prob (higher difficulty)
                probability_matrix.masked_fill_(cdr_mask, cdr_mask_prob)

                # FR regions: mask with fr_mask_prob (lower difficulty)
                probability_matrix.masked_fill_(fr_mask, fr_mask_prob)

                # Special tokens: never mask
                probability_matrix.masked_fill_(special, 0.0)

                rand = torch.rand(input_ids.shape)
                to_mask = (rand < probability_matrix)

        elif ngl_targeted_masking:
            # Different masking probabilities for GL vs NGL positions
            probability_matrix = torch.full(labels_aa.shape, mask_prob, dtype=torch.float)
            probability_matrix.masked_fill_(ngl_masks_tensor, ngl_mask_prob)
            probability_matrix.masked_fill_(special, 0.0)

            rand = torch.rand(input_ids.shape)
            to_mask = (rand < probability_matrix)
        else:
            # Standard uniform masking
            rand = torch.rand(input_ids.shape)
            to_mask = (rand < mask_prob) & ~special

        # Mark non-masked positions as -100 in labels_aa (ignore in loss)
        labels_aa[~to_mask] = -100

        # Mark non-masked positions as ignore sentinel in labels_mut
        # [v38] 3-class uses -1 (int), 2-class uses -1.0 (float)
        if use_3class_origin:
            labels_mut[~to_mask] = -1
        else:
            labels_mut[~to_mask] = -1.0

        # =====================================================================
        # Step 5: Apply 80/10/10 masking strategy to input_ids
        # =====================================================================
        rand2 = torch.rand(input_ids.shape)

        # 80% -> [MASK]
        mask_mask = to_mask & (rand2 < 0.8)
        input_ids[mask_mask] = MASK

        # 10% -> random token
        rand_mask = to_mask & (rand2 >= 0.8) & (rand2 < 0.9)
        num_rand = rand_mask.sum().item()
        if num_rand > 0:
            input_ids[rand_mask] = torch.randint(0, V, (num_rand,))

        # 10% -> keep original (no action needed, already uppercase)

        # =====================================================================
        # Step 6: Create attention mask
        # =====================================================================
        attention_mask = (input_ids != PAD).long()

        # =====================================================================
        # Step 7: Encode gene IDs if enabled
        # =====================================================================
        v_gene_ids = None
        j_gene_ids = None
        if use_germline_genes and gene_vocab is not None:
            v_gene_ids = torch.tensor([
                gene_vocab.encode(v_h if v_h is not None else v_l)
                for v_h, v_l in zip(v_genes_heavy, v_genes_light)
            ], dtype=torch.long)

            j_gene_ids = torch.tensor([
                gene_vocab.encode(j_h if j_h is not None else j_l)
                for j_h, j_l in zip(j_genes_heavy, j_genes_light)
            ], dtype=torch.long)

        # =====================================================================
        # Step 8: Stack region IDs if enabled
        # =====================================================================
        region_ids_tensor = None
        if use_region_embedding and region_ids_list is not None:
            region_ids_tensor = torch.from_numpy(np.stack(region_ids_list, axis=0)).long()

        # =====================================================================
        # Return based on enabled features
        # Format: (input_ids, labels_aa, labels_mut, attention_mask, ngl_masks, ...)
        # [v37] When coherence masking is enabled, append coherence_flags as last element
        # =====================================================================
        # [v37] Coherence flag tensor (scalar: 1.0 if coherence mode, 0.0 otherwise)
        coherence_flags = torch.tensor(1.0 if coherence_active else 0.0)

        if use_germline_genes and use_region_embedding:
            base = (input_ids, labels_aa, labels_mut, attention_mask, ngl_masks_tensor, v_gene_ids, j_gene_ids, region_ids_tensor)
        elif use_germline_genes:
            base = (input_ids, labels_aa, labels_mut, attention_mask, ngl_masks_tensor, v_gene_ids, j_gene_ids)
        elif use_region_embedding:
            base = (input_ids, labels_aa, labels_mut, attention_mask, ngl_masks_tensor, region_ids_tensor)
        else:
            base = (input_ids, labels_aa, labels_mut, attention_mask, ngl_masks_tensor)

        # [v40] Append synth_masks and mpnn_gl_probs if synth masking enabled
        if use_synth_masking and synth_masks_tensor is not None:
            base = base + (synth_masks_tensor, mpnn_gl_probs_tensor)

        if use_coherence_masking:
            return base + (coherence_flags,)
        else:
            return base

    return collate_fn


class LazyShardedDataModule(pl.LightningDataModule):
    """
    Memory-efficient DataModule for large datasets with DDP training.

    CRITICAL FIX FOR DDP MEMORY DUPLICATION:
    ========================================
    In standard DDP, data is loaded in the main process BEFORE spawning workers.
    When processes fork(), each child gets a copy-on-write reference that eventually
    becomes a full copy (e.g., 2 GPUs × 120GB = 240GB, exceeding 200GB RAM limit).

    SOLUTION:
    =========
    1. Don't load data in __init__ - only store the path
    2. Load data in setup() which runs AFTER DDP spawns processes
    3. Each GPU loads only its shard of TRAINING data (N/world_size samples)
    4. Val/test data is small enough to load fully on each process

    MEMORY CALCULATION:
    ==================
    For 60M sample dataset with 2 GPUs:
    - Training: 60M / 2 = 30M samples per GPU ≈ 45-50GB each
    - Val/Test: ~200k samples × 2 processes ≈ 1GB
    - Total per process: ~50GB (well under 100GB limit)
    - Total memory: ~100GB (well under 200GB limit)

    NOTE: This class has the same interface as SFTDataModule but accepts data_path
    instead of data_frame. Use enable_sharding=True for DDP training.
    """

    def __init__(self, data_path, batch_size, mask_prob, tokenizer, seed, num_workers=8,
                 gene_vocab=None, use_germline_genes=False,
                 ngl_targeted_masking=False, ngl_mask_prob=0.8,
                 use_region_embedding=False,
                 ngl_mask_schedule=None,
                 use_region_masking=False, cdr_mask_prob=0.4, fr_mask_prob=0.15,
                 use_fr_span_masking=False, fr_span_min_length=3, fr_span_max_length=6,
                 enable_sharding=True,
                 val_sample_ratio=0.1,
                 cdr_ngl_mask_prob=None, fr_ngl_mask_prob=None,
                 use_coherence_masking=False, coherence_prob=0.3,
                 coherence_ngl_mask_prob=0.5,
                 use_3class_origin=False,
                 synthetic_ngl_data_path=None,
                 use_synth_masking=False,
                 cdr_synth_mask_prob=0.45, fr_synth_mask_prob=0.30,
                 use_mpnn_origin_smoothing=False,
                 origin_label_smooth_factor=0.2,
                 ngl_label_noise_flips=0):
        """
        Initialize the LazyShardedDataModule.

        IMPORTANT: Data is NOT loaded here - only in setup() after DDP spawns.

        Args:
            data_path: Path to parquet or pickle file (NOT a DataFrame!)
            enable_sharding: If True, each GPU loads only its shard of training data
            cdr_ngl_mask_prob: [v35.1] Masking probability for CDR-NGL positions (None=disabled)
            fr_ngl_mask_prob: [v35.1] Masking probability for FR-NGL positions (None=disabled)
            use_synth_masking: [v40] If True, return synth_mask and mpnn_gl_prob in batch
            cdr_synth_mask_prob: [v40] Masking prob for CDR-SynNGL positions (default 0.45)
            fr_synth_mask_prob: [v40] Masking prob for FR-SynNGL positions (default 0.30)
            ngl_label_noise_flips: [rebuttal] Flip N GL/NGL labels per chain in train set (0=disabled)
            use_mpnn_origin_smoothing: [v40] If True, apply MPNN-based origin label smoothing
            origin_label_smooth_factor: [v40] Smoothing factor for SynNGL origin labels (default 0.2)
            (All other args are same as SFTDataModule)
        """
        super().__init__()

        # Store path, NOT data
        self.data_path = str(data_path)
        self.enable_sharding = enable_sharding

        # Store all other parameters
        self.batch_size = batch_size
        self.mask_prob = mask_prob
        self.tokenizer = tokenizer
        self.seed = seed
        self.num_workers = num_workers
        self.gene_vocab = gene_vocab
        self.use_germline_genes = use_germline_genes
        self.ngl_targeted_masking = ngl_targeted_masking
        self.ngl_mask_prob = ngl_mask_prob
        self.use_region_embedding = use_region_embedding
        self.ngl_mask_schedule = ngl_mask_schedule
        self.use_region_masking = use_region_masking
        self.cdr_mask_prob = cdr_mask_prob
        self.fr_mask_prob = fr_mask_prob
        self.use_fr_span_masking = use_fr_span_masking
        self.fr_span_min_length = fr_span_min_length
        self.fr_span_max_length = fr_span_max_length
        self.val_sample_ratio = val_sample_ratio
        self.cdr_ngl_mask_prob = cdr_ngl_mask_prob
        self.fr_ngl_mask_prob = fr_ngl_mask_prob
        self.use_coherence_masking = use_coherence_masking
        self.coherence_prob = coherence_prob
        self.coherence_ngl_mask_prob = coherence_ngl_mask_prob
        self.use_3class_origin = use_3class_origin
        self.synthetic_ngl_data_path = synthetic_ngl_data_path
        self.use_synth_masking = use_synth_masking
        self.cdr_synth_mask_prob = cdr_synth_mask_prob
        self.fr_synth_mask_prob = fr_synth_mask_prob
        self.use_mpnn_origin_smoothing = use_mpnn_origin_smoothing
        self.origin_label_smooth_factor = origin_label_smooth_factor
        self.ngl_label_noise_flips = ngl_label_noise_flips

        # Placeholders - datasets will be created in setup()
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self.collate_fn = None
        self.collate_fn_eval = None

        # Current step for scheduled masking
        self.current_step = 0

        print(f"[LazyShardedDataModule] Initialized with data_path: {self.data_path}")
        print(f"[LazyShardedDataModule] DDP Sharding: {'ENABLED' if enable_sharding else 'DISABLED'}")
        print(f"[LazyShardedDataModule] Validation sampling: {val_sample_ratio*100:.0f}% (random subset each time)")
        print(f"[LazyShardedDataModule] Data will be loaded in setup() AFTER DDP spawns processes")

    def _create_collate_functions(self):
        """Create multihead collate functions for training and evaluation."""

        # Verify lowercase custom tokens are present
        test_token_id = self.tokenizer.convert_tokens_to_ids('a')
        if test_token_id == self.tokenizer.unk_token_id:
            raise ValueError(
                "[LazyShardedDataModule] ERROR: Lowercase amino acid tokens not found in tokenizer!"
            )

        # Training collate with masking
        self.collate_fn = make_collate_fn_multihead(
            self.tokenizer, self.mask_prob,
            gene_vocab=self.gene_vocab,
            use_germline_genes=self.use_germline_genes,
            ngl_targeted_masking=self.ngl_targeted_masking,
            ngl_mask_prob=self.ngl_mask_prob,
            use_region_embedding=self.use_region_embedding,
            use_region_masking=self.use_region_masking,
            cdr_mask_prob=self.cdr_mask_prob,
            fr_mask_prob=self.fr_mask_prob,
            use_fr_span_masking=self.use_fr_span_masking,
            fr_span_min_length=self.fr_span_min_length,
            fr_span_max_length=self.fr_span_max_length,
            cdr_ngl_mask_prob=self.cdr_ngl_mask_prob,
            fr_ngl_mask_prob=self.fr_ngl_mask_prob,
            use_coherence_masking=self.use_coherence_masking,
            coherence_prob=self.coherence_prob,
            coherence_ngl_mask_prob=self.coherence_ngl_mask_prob,
            use_3class_origin=self.use_3class_origin,
            use_synth_masking=self.use_synth_masking,
            cdr_synth_mask_prob=self.cdr_synth_mask_prob,
            fr_synth_mask_prob=self.fr_synth_mask_prob,
            use_mpnn_origin_smoothing=self.use_mpnn_origin_smoothing,
            origin_label_smooth_factor=self.origin_label_smooth_factor,
        )

        # Eval collate with standard masking (no 4-rate, no NGL-targeted, no coherence)
        self.collate_fn_eval = make_collate_fn_multihead(
            self.tokenizer, self.mask_prob,
            gene_vocab=self.gene_vocab,
            use_germline_genes=self.use_germline_genes,
            ngl_targeted_masking=False,
            ngl_mask_prob=self.ngl_mask_prob,
            use_region_embedding=self.use_region_embedding,
            use_region_masking=False,
            cdr_mask_prob=self.cdr_mask_prob,
            fr_mask_prob=self.fr_mask_prob,
            use_fr_span_masking=False,
            use_3class_origin=self.use_3class_origin,
            use_synth_masking=self.use_synth_masking,
        )

    def setup(self, stage=None):
        """
        Load data AFTER DDP spawns processes.

        This is the key fix: By loading data here instead of __init__,
        each DDP process loads its own data independently without forking
        a parent process that already has the data in memory.

        MEMORY OPTIMIZATION (v31.1):
        ============================
        For parquet files with 'split' column, we use PyArrow filters to load
        ONLY the rows we need, avoiding the 140GB peak memory spike from loading
        the entire DataFrame first.
        """
        # Get DDP rank info from trainer
        if self.trainer is not None and hasattr(self.trainer, 'world_size'):
            world_size = self.trainer.world_size
            rank = self.trainer.global_rank
        else:
            world_size = 1
            rank = 0

        print(f"[LazyShardedDataModule] setup() called on rank {rank}/{world_size}")

        # Create collate functions (only once per process)
        if self.collate_fn is None:
            self._create_collate_functions()

        # =========================================================================
        # [v31.4] PRE-SPLIT SHARD FILES: Most efficient - each GPU loads its own file
        # Created by: python scripts/split_parquet_for_ddp.py
        # =========================================================================
        from pathlib import Path
        data_path = Path(self.data_path)

        # Check if data_path is a directory with pre-split shard files
        if data_path.is_dir():
            shard_file = data_path / f"train_shard_{rank}.parquet"
            valid_file = data_path / "valid.parquet"
            test_file = data_path / "test.parquet"

            if shard_file.exists() and valid_file.exists():
                print(f"[Rank {rank}] PRE-SPLIT SHARD MODE: Loading from {data_path}")
                print(f"[Rank {rank}] >>> Loading shard file: {shard_file.name}")

                # Each GPU loads ONLY its own shard file - no 140GB peak!
                df_train = pd.read_parquet(shard_file)
                print(f"[Rank {rank}] Loaded training shard: {len(df_train):,} samples")

                # [v38/v40] Load sidecar synthetic NGL parquet if 3-class origin or synth masking is enabled
                if (self.use_3class_origin or self.use_synth_masking) and self.synthetic_ngl_data_path:
                    synth_path = Path(self.synthetic_ngl_data_path)
                    synth_cols = ['synthetic_ngl_heavy', 'synthetic_ngl_light', 'mpnn_gl_prob_heavy', 'mpnn_gl_prob_light']
                    if synth_path.is_dir():
                        # Sidecar shards: synth_ngl_shard_{rank}.parquet
                        synth_shard = synth_path / f"synth_ngl_shard_{rank}.parquet"
                        if synth_shard.exists():
                            df_synth = pd.read_parquet(synth_shard)
                            # Merge by positional index (sidecar has same row ordering)
                            for col in synth_cols:
                                if col in df_synth.columns:
                                    df_train[col] = df_synth[col].values[:len(df_train)]
                            print(f"[Rank {rank}] Merged sidecar synth NGL labels from {synth_shard.name}")
                    elif synth_path.exists():
                        # Single parquet file (for paired data)
                        df_synth = pd.read_parquet(synth_path)
                        for col in synth_cols:
                            if col in df_synth.columns:
                                df_train[col] = df_synth[col].values[:len(df_train)]
                        print(f"[Rank {rank}] Merged synth NGL labels from {synth_path}")

                df_val = pd.read_parquet(valid_file)
                print(f"[Rank {rank}] Loaded validation: {len(df_val):,} samples")

                if test_file.exists():
                    df_test = pd.read_parquet(test_file)
                    print(f"[Rank {rank}] Loaded test: {len(df_test):,} samples")
                else:
                    df_test = df_val.head(100)  # Minimal test set if not present
                    print(f"[Rank {rank}] No test file, using minimal test set")

                # Skip to dataset creation (jump over the old loading logic)
                print(f"[Rank {rank}] Split sizes - Train: {len(df_train):,}, Val: {len(df_val):,}, Test: {len(df_test):,}")

                # Create datasets (same parameters as used in the rest of setup())
                self.train_ds = SeqSeqDataset(
                    df_train, self.tokenizer, self.mask_prob,
                    use_germline_genes=self.use_germline_genes,
                    use_region_embedding=self.use_region_embedding,
                    use_3class_origin=self.use_3class_origin,
                    use_synth_masking=self.use_synth_masking,
                    ngl_label_noise_flips=self.ngl_label_noise_flips,
                )
                self.val_ds = SeqSeqDataset(
                    df_val, self.tokenizer, self.mask_prob,
                    use_germline_genes=self.use_germline_genes,
                    use_region_embedding=self.use_region_embedding,
                    use_3class_origin=self.use_3class_origin,
                    use_synth_masking=self.use_synth_masking,
                )
                self.test_ds = SeqSeqDataset(
                    df_test, self.tokenizer, self.mask_prob,
                    use_germline_genes=self.use_germline_genes,
                    use_region_embedding=self.use_region_embedding,
                    use_3class_origin=self.use_3class_origin,
                    use_synth_masking=self.use_synth_masking,
                )

                print(f"[Rank {rank}] Datasets created successfully")
                return  # Done! Skip the rest of setup()

            else:
                raise FileNotFoundError(
                    f"Pre-split shard files not found in {data_path}. "
                    f"Expected: train_shard_{rank}.parquet, valid.parquet. "
                    f"Run: python scripts/split_parquet_for_ddp.py --input <parquet> --output-dir {data_path} --num-shards {world_size}"
                )

        # =========================================================================
        # [v31.2] TRUE SHARDED LOADING: Each GPU loads ONLY its shard from disk
        # This avoids the 140GB peak memory that was causing OOM with 2 GPUs
        # =========================================================================
        if self.data_path.endswith('.parquet'):
            import pyarrow.parquet as pq

            print(f"[Rank {rank}] Loading parquet with TRUE sharded loading...")

            # Check if 'split' column exists using PyArrow metadata (no data load)
            pf = pq.ParquetFile(self.data_path)
            columns = pf.schema_arrow.names
            has_split_column = 'split' in columns

            if has_split_column:
                print(f"[Rank {rank}] Using split-column filtering (memory-efficient)")

                # Load val and test first (small, ~1-2% of data each)
                df_val = pd.read_parquet(self.data_path, filters=[('split', '==', 'valid')])
                print(f"[Rank {rank}] Loaded validation: {len(df_val):,} samples")

                df_test = pd.read_parquet(self.data_path, filters=[('split', '==', 'test')])
                print(f"[Rank {rank}] Loaded test: {len(df_test):,} samples")

                # =====================================================================
                # [v31.2] SEQUENTIAL SHARDED LOADING for training data
                # Problem: Loading 60M rows uses 140GB, and 2 GPUs loading
                # simultaneously = 280GB peak (exceeds 200GB limit)
                #
                # Solution: Load one GPU at a time using distributed barrier
                # - GPU 0 loads, extracts shard, frees memory
                # - Barrier synchronization
                # - GPU 1 loads, extracts shard, frees memory
                # Peak memory = 140GB (one GPU loading at a time)
                # =====================================================================
                if self.enable_sharding and world_size > 1:
                    import gc
                    import torch.distributed as dist

                    # Step 1: Count train rows efficiently (just split column)
                    split_col = pd.read_parquet(self.data_path, columns=['split'])
                    n_train_total = (split_col['split'] == 'train').sum()
                    del split_col
                    gc.collect()

                    # Step 2: Calculate this GPU's shard boundaries
                    shard_size = n_train_total // world_size
                    start_idx = rank * shard_size
                    end_idx = start_idx + shard_size if rank < world_size - 1 else n_train_total

                    print(f"[Rank {rank}] SEQUENTIAL SHARDED LOADING: rows {start_idx:,} to {end_idx:,} of {n_train_total:,}")

                    # Step 3: Sequential loading - one GPU at a time
                    # This prevents simultaneous 140GB loads that exceed RAM
                    for loading_rank in range(world_size):
                        if rank == loading_rank:
                            print(f"[Rank {rank}] >>> Loading training data (other GPUs waiting)...")
                            df_train_full = pd.read_parquet(self.data_path, filters=[('split', '==', 'train')])
                            print(f"[Rank {rank}] Loaded {len(df_train_full):,} samples, extracting shard...")

                            # Extract shard and immediately free full data
                            df_train = df_train_full.iloc[start_idx:end_idx].copy()
                            del df_train_full
                            gc.collect()

                            print(f"[Rank {rank}] Shard ready: {len(df_train):,} samples ({100*len(df_train)/n_train_total:.1f}%)")

                        # Synchronize: wait for this rank to finish before next starts
                        if dist.is_initialized():
                            dist.barrier()
                        print(f"[Rank {rank}] Barrier passed for loading_rank={loading_rank}")

                    print(f"[Rank {rank}] All GPUs have loaded their shards")
                else:
                    # No sharding - load all training data
                    df_train = pd.read_parquet(self.data_path, filters=[('split', '==', 'train')])
                    print(f"[Rank {rank}] Loaded training: {len(df_train):,} samples (no sharding)")
            else:
                # Fall back to loading everything (cluster-based splitting)
                print(f"[Rank {rank}] No 'split' column, loading full parquet...")
                df = pd.read_parquet(self.data_path)
                print(f"[Rank {rank}] Loaded {len(df)} total samples")

                # Cluster-based splitting
                if 'cluster_id' not in df.columns:
                    raise ValueError("DataFrame must contain either 'split' or 'cluster_id' column")

                clusters = df['cluster_id'].unique()
                rng = np.random.default_rng(self.seed)
                rng.shuffle(clusters)

                val_ids = clusters[:20000]
                test_ids = clusters[20000:40000]
                train_ids = clusters[40000:]

                df_train = df[df['cluster_id'].isin(train_ids)].reset_index(drop=True)
                df_val = df[df['cluster_id'].isin(val_ids)].reset_index(drop=True)
                df_test = df[df['cluster_id'].isin(test_ids)].reset_index(drop=True)

                del df
                import gc
                gc.collect()
        else:
            # Pickle: must load everything
            print(f"[Rank {rank}] Loading pickle file: {self.data_path}")
            df = pd.read_pickle(self.data_path)
            print(f"[Rank {rank}] Loaded {len(df)} total samples")

            if 'split' in df.columns:
                df_train = df[df['split'] == 'train'].reset_index(drop=True)
                df_val = df[df['split'] == 'valid'].reset_index(drop=True)
                df_test = df[df['split'] == 'test'].reset_index(drop=True)
            else:
                if 'cluster_id' not in df.columns:
                    raise ValueError("DataFrame must contain either 'split' or 'cluster_id' column")

                clusters = df['cluster_id'].unique()
                rng = np.random.default_rng(self.seed)
                rng.shuffle(clusters)

                val_ids = clusters[:20000]
                test_ids = clusters[20000:40000]
                train_ids = clusters[40000:]

                df_train = df[df['cluster_id'].isin(train_ids)].reset_index(drop=True)
                df_val = df[df['cluster_id'].isin(val_ids)].reset_index(drop=True)
                df_test = df[df['cluster_id'].isin(test_ids)].reset_index(drop=True)

            del df
            import gc
            gc.collect()

        print(f"[Rank {rank}] Split sizes - Train: {len(df_train):,}, Val: {len(df_val):,}, Test: {len(df_test):,}")

        # =========================================================================
        # SHARD TRAINING DATA FOR DDP (only for pickle files - parquet already sharded above)
        # =========================================================================
        if self.enable_sharding and world_size > 1 and not self.data_path.endswith('.parquet'):
            n_train = len(df_train)
            shard_size = n_train // world_size
            start_idx = rank * shard_size
            end_idx = start_idx + shard_size if rank < world_size - 1 else n_train

            # Slice and immediately free the rest
            df_train_shard = df_train.iloc[start_idx:end_idx].copy()  # .copy() to free original
            del df_train
            import gc
            gc.collect()

            df_train = df_train_shard.reset_index(drop=True)

            print(f"[Rank {rank}] SHARDING (pickle): Using samples {start_idx:,} to {end_idx:,} ({len(df_train):,} samples)")
            print(f"[Rank {rank}] Memory savings: {n_train:,} → {len(df_train):,} samples ({100*len(df_train)/n_train:.1f}%)")

        # =========================================================================
        # [v40] Load sidecar synthetic NGL data for non-shard path
        # =========================================================================
        if (self.use_3class_origin or self.use_synth_masking) and self.synthetic_ngl_data_path:
            from pathlib import Path
            synth_path = Path(self.synthetic_ngl_data_path)
            if synth_path.exists():
                df_synth = pd.read_parquet(synth_path)
                for df_target, label in [(df_train, 'train'), (df_val, 'val'), (df_test, 'test')]:
                    for col in ['synthetic_ngl_heavy', 'synthetic_ngl_light', 'mpnn_gl_prob_heavy', 'mpnn_gl_prob_light']:
                        if col in df_synth.columns and col not in df_target.columns:
                            df_target[col] = df_synth[col].values[:len(df_target)]
                print(f"[Rank {rank}] Merged sidecar synth NGL labels from {synth_path}")
                del df_synth
            else:
                print(f"[Rank {rank}] WARNING: synthetic_ngl_data_path not found: {synth_path}")

        # =========================================================================
        # Create Datasets
        # =========================================================================
        self.train_ds = SeqSeqDataset(
            df_train, self.tokenizer, self.mask_prob,
            use_germline_genes=self.use_germline_genes,
            use_region_embedding=self.use_region_embedding,
            use_3class_origin=self.use_3class_origin,
            use_synth_masking=self.use_synth_masking,
            ngl_label_noise_flips=self.ngl_label_noise_flips,
        )
        self.val_ds = SeqSeqDataset(
            df_val, self.tokenizer, self.mask_prob,
            use_germline_genes=self.use_germline_genes,
            use_region_embedding=self.use_region_embedding,
            use_3class_origin=self.use_3class_origin,
            use_synth_masking=self.use_synth_masking,
        )
        self.test_ds = SeqSeqDataset(
            df_test, self.tokenizer, self.mask_prob,
            use_germline_genes=self.use_germline_genes,
            use_region_embedding=self.use_region_embedding,
            use_3class_origin=self.use_3class_origin,
            use_synth_masking=self.use_synth_masking,
        )

        print(f"[Rank {rank}] Datasets created - Train: {len(self.train_ds)}, Val: {len(self.val_ds)}, Test: {len(self.test_ds)}")

    def update_step(self, step, update_interval=100):
        """Update current step for scheduled masking (same as SFTDataModule)."""
        self.current_step = step

        if step % update_interval != 0:
            return

        # Update collate function if using scheduled NGL masking
        if self.ngl_targeted_masking and self.ngl_mask_schedule is not None and self.ngl_mask_schedule.get('enabled', False):
            new_ngl_prob = self._get_scheduled_ngl_mask_prob()

            self.collate_fn = make_collate_fn_multihead(
                self.tokenizer, self.mask_prob,
                gene_vocab=self.gene_vocab,
                use_germline_genes=self.use_germline_genes,
                ngl_targeted_masking=self.ngl_targeted_masking,
                ngl_mask_prob=new_ngl_prob,
                use_region_embedding=self.use_region_embedding,
                silent=True,
                use_region_masking=self.use_region_masking,
                cdr_mask_prob=self.cdr_mask_prob,
                fr_mask_prob=self.fr_mask_prob,
                use_fr_span_masking=self.use_fr_span_masking,
                fr_span_min_length=self.fr_span_min_length,
                fr_span_max_length=self.fr_span_max_length,
                cdr_ngl_mask_prob=self.cdr_ngl_mask_prob,
                fr_ngl_mask_prob=self.fr_ngl_mask_prob,
            )

    def _get_scheduled_ngl_mask_prob(self):
        """Get current NGL mask probability based on schedule."""
        if self.ngl_mask_schedule is None or not self.ngl_mask_schedule.get('enabled', False):
            return self.ngl_mask_prob

        start_prob = self.ngl_mask_schedule.get('start_prob', 0.8)
        end_prob = self.ngl_mask_schedule.get('end_prob', 0.5)
        start_step = self.ngl_mask_schedule.get('start_step', 0)
        end_step = self.ngl_mask_schedule.get('end_step', 3000)

        if self.current_step <= start_step:
            return start_prob
        elif self.current_step >= end_step:
            return end_prob
        else:
            progress = (self.current_step - start_step) / (end_step - start_step)
            return start_prob + (end_prob - start_prob) * progress

    def seed_worker(self, worker_id):
        """Initialize random seeds for each worker."""
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def train_dataloader(self):
        gen = torch.Generator()
        gen.manual_seed(self.seed)

        # IMPORTANT: With sharding, we DON'T use DistributedSampler
        # Each process has its own shard, so we just shuffle locally
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,  # Local shuffle within shard
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
            worker_init_fn=self.seed_worker,
            generator=gen,
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=4 if self.num_workers > 0 else None  # [v31.2] Increased for better GPU overlap
        )

    def val_dataloader(self):
        # [v31.3] Random 10% sampling for validation - different subset each time
        # With 6.6M validation samples, using all takes 1+ hour per validation
        # Sampling 10% (~664k) reduces to ~6 minutes while still being statistically representative
        from torch.utils.data import Subset
        import random

        val_sample_ratio = getattr(self, 'val_sample_ratio', 0.1)  # Default 10%
        n_val = len(self.val_ds)
        n_sample = int(n_val * val_sample_ratio)

        # Random indices WITHOUT fixed seed - different each validation
        indices = random.sample(range(n_val), n_sample)
        val_subset = Subset(self.val_ds, indices)

        print(f"[Validation] Sampling {n_sample:,} / {n_val:,} ({val_sample_ratio*100:.0f}%) samples (random subset)")

        gen = torch.Generator()
        gen.manual_seed(self.seed)
        return DataLoader(
            val_subset,  # Use subset instead of full val_ds
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn_eval,
            num_workers=self.num_workers,
            pin_memory=True,
            worker_init_fn=self.seed_worker,
            generator=gen,
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=4 if self.num_workers > 0 else None  # [v31.2] Increased for better GPU overlap
        )

    def test_dataloader(self):
        gen = torch.Generator()
        gen.manual_seed(self.seed)
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn_eval,
            num_workers=self.num_workers,
            pin_memory=True,
            worker_init_fn=self.seed_worker,
            generator=gen,
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=4 if self.num_workers > 0 else None  # [v31.2] Increased for better GPU overlap
        )


class SFTDataModule(pl.LightningDataModule):
    """A PyTorch Lightning Data Module to handle data loading using a 'split' column"""

    def __init__(self, data_frame, batch_size, mask_prob, tokenizer, seed, num_workers=8,
                 gene_vocab=None, use_germline_genes=False,
                 ngl_targeted_masking=False, ngl_mask_prob=0.8,
                 use_region_embedding=False,
                 ngl_mask_schedule=None,
                 use_region_masking=False, cdr_mask_prob=0.4, fr_mask_prob=0.15,
                 use_fr_span_masking=False, fr_span_min_length=3, fr_span_max_length=6,
                 cdr_ngl_mask_prob=None, fr_ngl_mask_prob=None,
                 use_3class_origin=False):
        """
        Initialize the SFTDataModule.

        Args:
            data_frame: DataFrame containing antibody sequences
            batch_size: Batch size for training
            mask_prob: Probability of masking GL tokens (default 0.15)
            tokenizer: Pre-configured HuggingFace tokenizer with lowercase NGL tokens ALREADY ADDED.
            seed: Random seed for reproducibility
            num_workers: Number of data loading workers
            gene_vocab: GeneVocabulary for V/J gene conditioning
            use_germline_genes: Whether to use gene conditioning
            ngl_targeted_masking: Whether to apply higher masking to NGL tokens
            ngl_mask_prob: Masking probability for NGL tokens (default 0.8)
            use_region_embedding: Whether to include region IDs in batch output
            ngl_mask_schedule: NGL masking schedule configuration
            use_region_masking: Whether to use region-wise masking (CDR/FR)
            cdr_mask_prob: Masking probability for CDR regions (default 0.4)
            fr_mask_prob: Masking probability for FR regions (default 0.15)
            use_fr_span_masking: Whether to use span masking for FR regions
            fr_span_min_length: Minimum span length for FR masking (default 3)
            fr_span_max_length: Maximum span length for FR masking (default 6)
            cdr_ngl_mask_prob: [v35.1] Masking probability for CDR-NGL positions (None=disabled)
            fr_ngl_mask_prob: [v35.1] Masking probability for FR-NGL positions (None=disabled)
            use_3class_origin: [v38] If True, use 3-class origin labels (GL/SynNGL/NGL)
        """
        super().__init__()
        self.df = data_frame
        self.batch_size = batch_size
        self.mask_prob = mask_prob
        self.gene_vocab = gene_vocab
        self.use_germline_genes = use_germline_genes
        self.ngl_targeted_masking = ngl_targeted_masking
        self.ngl_mask_prob = ngl_mask_prob
        self.use_region_embedding = use_region_embedding
        self.use_region_masking = use_region_masking
        self.cdr_mask_prob = cdr_mask_prob
        self.fr_mask_prob = fr_mask_prob
        self.use_fr_span_masking = use_fr_span_masking
        self.fr_span_min_length = fr_span_min_length
        self.fr_span_max_length = fr_span_max_length
        self.cdr_ngl_mask_prob = cdr_ngl_mask_prob
        self.fr_ngl_mask_prob = fr_ngl_mask_prob
        self.use_3class_origin = use_3class_origin
        self.tokenizer = tokenizer
        self.seed = seed
        self.num_workers = num_workers

        # Log tokenizer configuration
        print(f"[SFTDataModule] Using pre-configured tokenizer")
        print(f"[SFTDataModule] Vocabulary size: {len(self.tokenizer)}")

        # Verify lowercase custom tokens are present
        test_token_id = self.tokenizer.convert_tokens_to_ids('a')
        if test_token_id == self.tokenizer.unk_token_id:
            raise ValueError(
                "[SFTDataModule] ERROR: Lowercase amino acid tokens not found in tokenizer! "
                "Ensure tokenizer.add_tokens() was called in train_esm.py before passing tokenizer."
            )
        print(f"[SFTDataModule] Verified: lowercase token 'a' has ID {test_token_id}")

        # =====================================================================
        # Multihead Architecture collate functions
        # =====================================================================
        print(f"\n[SFTDataModule] === MULTIHEAD ARCHITECTURE MODE ===")
        print(f"[SFTDataModule] Using make_collate_fn_multihead for AA+Mutation dual-head model")

        # Training collate with NGL-targeted or region-wise masking
        self.collate_fn = make_collate_fn_multihead(
            self.tokenizer, self.mask_prob,
            gene_vocab=self.gene_vocab,
            use_germline_genes=self.use_germline_genes,
            ngl_targeted_masking=self.ngl_targeted_masking,
            ngl_mask_prob=self.ngl_mask_prob,
            use_region_embedding=self.use_region_embedding,
            use_region_masking=self.use_region_masking,
            cdr_mask_prob=self.cdr_mask_prob,
            fr_mask_prob=self.fr_mask_prob,
            use_fr_span_masking=self.use_fr_span_masking,
            fr_span_min_length=self.fr_span_min_length,
            fr_span_max_length=self.fr_span_max_length,
            cdr_ngl_mask_prob=self.cdr_ngl_mask_prob,
            fr_ngl_mask_prob=self.fr_ngl_mask_prob,
            use_3class_origin=self.use_3class_origin,
        )

        # Eval collate with standard masking (no NGL-targeted, no region-wise, no span masking)
        self.collate_fn_eval = make_collate_fn_multihead(
            self.tokenizer, self.mask_prob,
            gene_vocab=self.gene_vocab,
            use_germline_genes=self.use_germline_genes,
            ngl_targeted_masking=False,
            ngl_mask_prob=self.ngl_mask_prob,
            use_region_embedding=self.use_region_embedding,
            use_region_masking=False,
            cdr_mask_prob=self.cdr_mask_prob,
            fr_mask_prob=self.fr_mask_prob,
            use_fr_span_masking=False,
            use_3class_origin=self.use_3class_origin,
        )

        # Print masking strategy
        if self.use_region_masking:
            if self.use_fr_span_masking:
                print(f"[SFTDataModule] Training: Region-wise masking with FR Span Masking (CDR={self.cdr_mask_prob}, FR={self.fr_mask_prob}, span={self.fr_span_min_length}-{self.fr_span_max_length})")
            else:
                print(f"[SFTDataModule] Training: Region-wise masking (CDR={self.cdr_mask_prob}, FR={self.fr_mask_prob})")
        else:
            print(f"[SFTDataModule] Training: NGL-targeted masking (GL={self.mask_prob}, NGL={self.ngl_mask_prob})")
        print(f"[SFTDataModule] Validation/Test: Standard {self.mask_prob:.0%} uniform masking")
        print(f"[SFTDataModule] Collate returns: (input_ids, labels_aa, labels_mut, attention_mask, ngl_masks, ...)")
        print(f"[SFTDataModule] ========================================\n")

        # Log gene conditioning status
        if self.use_germline_genes:
            print(f"[SFTDataModule] V/J Gene Conditioning: ENABLED (vocab size: {len(self.gene_vocab)})")
        else:
            print(f"[SFTDataModule] V/J Gene Conditioning: Disabled")

        # Log region embedding status
        if self.use_region_embedding:
            print(f"[SFTDataModule] Region Embedding: ENABLED")
        else:
            print(f"[SFTDataModule] Region Embedding: Disabled")

        # Log region-wise masking status
        if self.use_region_masking:
            print(f"[SFTDataModule] Region-wise Masking: ENABLED")
            print(f"  CDR mask prob: {self.cdr_mask_prob}")
            print(f"  FR mask prob: {self.fr_mask_prob}")
            if self.use_fr_span_masking:
                print(f"  FR Span Masking: ENABLED (span length: {self.fr_span_min_length}-{self.fr_span_max_length})")

        # NGL masking schedule configuration
        self.ngl_mask_schedule = ngl_mask_schedule
        if ngl_mask_schedule is not None and ngl_mask_schedule.get('enabled', False):
            print(f"  [v17] NGL Masking Schedule Enabled:")
            print(f"        Start prob: {ngl_mask_schedule.get('start_prob', 0.8)}")
            print(f"        End prob: {ngl_mask_schedule.get('end_prob', 0.5)}")
            print(f"        Schedule: step {ngl_mask_schedule.get('start_step', 0)} -> {ngl_mask_schedule.get('end_step', 3000)}")

        # Current step counter for scheduled masking
        self.current_step = 0

    def _get_scheduled_ngl_mask_prob(self):
        """
        [CHANGE v17] Get current NGL mask probability based on schedule.
        Uses linear interpolation between start_prob and end_prob.

        Returns:
            float: Current NGL mask probability
        """
        if self.ngl_mask_schedule is None or not self.ngl_mask_schedule.get('enabled', False):
            # No schedule, return fixed ngl_mask_prob
            return self.ngl_mask_prob

        start_prob = self.ngl_mask_schedule.get('start_prob', 0.8)
        end_prob = self.ngl_mask_schedule.get('end_prob', 0.5)
        start_step = self.ngl_mask_schedule.get('start_step', 0)
        end_step = self.ngl_mask_schedule.get('end_step', 3000)

        if self.current_step <= start_step:
            return start_prob
        elif self.current_step >= end_step:
            return end_prob
        else:
            # Linear interpolation
            progress = (self.current_step - start_step) / (end_step - start_step)
            return start_prob + (end_prob - start_prob) * progress

    def update_step(self, step, update_interval=100):
        """
        [CHANGE v17] Update current step for scheduled masking.
        Should be called from training loop after each step.
        Only updates collate function every `update_interval` steps to reduce overhead.

        Args:
            step: Current global step
            update_interval: How often to recreate collate function (default: 100 steps)
        """
        self.current_step = step

        # Only update collate function every update_interval steps to reduce overhead
        if step % update_interval != 0:
            return

        # Update collate function if using scheduled NGL masking
        if self.ngl_targeted_masking and self.ngl_mask_schedule is not None and self.ngl_mask_schedule.get('enabled', False):
            new_ngl_prob = self._get_scheduled_ngl_mask_prob()

            # Recreate collate function with updated NGL probability (silently)
            self.collate_fn = make_collate_fn_multihead(
                self.tokenizer, self.mask_prob,
                gene_vocab=self.gene_vocab,
                use_germline_genes=self.use_germline_genes,
                ngl_targeted_masking=self.ngl_targeted_masking,
                ngl_mask_prob=new_ngl_prob,
                use_region_embedding=self.use_region_embedding,
                silent=True,
                use_region_masking=self.use_region_masking,
                cdr_mask_prob=self.cdr_mask_prob,
                fr_mask_prob=self.fr_mask_prob,
                use_fr_span_masking=self.use_fr_span_masking,
                fr_span_min_length=self.fr_span_min_length,
                fr_span_max_length=self.fr_span_max_length
            )

    def setup(self, stage=None):
        """
        Split data into train/valid/test splits.

        Two strategies:
        1. If 'split' column exists: Use predefined splits from the column (values: 'train', 'valid', 'test')
        2. Otherwise: Use cluster-based splitting (requires 'cluster_id' column with at least 40k clusters)
        """

        # Strategy 1: Check if 'split' column exists
        if 'split' in self.df.columns:
            print(f'[SFTDataModule] Found "split" column - using predefined splits')

            # Get unique split values
            split_values = self.df['split'].unique()
            print(f'[SFTDataModule] Available splits: {sorted(split_values)}')

            # Split dataframes based on split column
            df_train = self.df[self.df['split'] == 'train'].reset_index(drop=True)
            df_val = self.df[self.df['split'] == 'valid'].reset_index(drop=True)
            df_test = self.df[self.df['split'] == 'test'].reset_index(drop=True)

            print(f'[SFTDataModule] Train samples: {len(df_train)}')
            print(f'[SFTDataModule] Validation samples: {len(df_val)}')
            print(f'[SFTDataModule] Test samples: {len(df_test)}')

            # Check if any split is empty
            if len(df_train) == 0:
                print(f'[SFTDataModule] WARNING: No training samples found!')
            if len(df_val) == 0:
                print(f'[SFTDataModule] WARNING: No validation samples found!')
            if len(df_test) == 0:
                print(f'[SFTDataModule] WARNING: No test samples found!')

        # Strategy 2: Cluster-based splitting
        else:
            print(f'[SFTDataModule] No "split" column found - using cluster-based splitting')

            if 'cluster_id' not in self.df.columns:
                raise ValueError(
                    "DataFrame must contain either a 'split' column or a 'cluster_id' column for splitting"
                )

            # Shuffle clusters reproducibly
            clusters = self.df['cluster_id'].unique()
            n_clusters = len(clusters)
            print(f'[SFTDataModule] Total unique clusters: {n_clusters}')

            if n_clusters < 40000:
                raise ValueError(
                    f"Dataset has only {n_clusters} clusters, but need at least 40,000 "
                    f"for the specified split (20k val + 20k test)"
                )

            rng = np.random.default_rng(self.seed)
            rng.shuffle(clusters)

            # Partition clusters (non-overlapping)
            val_ids = clusters[:20000]
            test_ids = clusters[20000:40000]
            train_ids = clusters[40000:]

            print(f'[SFTDataModule] Validation clusters: {len(val_ids)} (first 20,000)')
            print(f'[SFTDataModule] Test clusters: {len(test_ids)} (20,000-40,000)')
            print(f'[SFTDataModule] Train clusters: {len(train_ids)} (40,000+)')

            # Split dataframes by cluster IDs
            df_train = self.df[self.df['cluster_id'].isin(train_ids)].reset_index(drop=True)
            df_val = self.df[self.df['cluster_id'].isin(val_ids)].reset_index(drop=True)
            df_test = self.df[self.df['cluster_id'].isin(test_ids)].reset_index(drop=True)

            print(f'[SFTDataModule] Train samples: {len(df_train)}')
            print(f'[SFTDataModule] Validation samples: {len(df_val)}')
            print(f'[SFTDataModule] Test samples: {len(df_test)}')

        # Build Datasets (same for both strategies)
        self.train_ds = SeqSeqDataset(df_train, self.tokenizer, self.mask_prob,
                                      use_germline_genes=self.use_germline_genes,
                                      use_region_embedding=self.use_region_embedding,
                                      use_3class_origin=self.use_3class_origin)
        self.val_ds   = SeqSeqDataset(df_val, self.tokenizer, self.mask_prob,
                                      use_germline_genes=self.use_germline_genes,
                                      use_region_embedding=self.use_region_embedding,
                                      use_3class_origin=self.use_3class_origin)
        self.test_ds  = SeqSeqDataset(df_test, self.tokenizer, self.mask_prob,
                                      use_germline_genes=self.use_germline_genes,
                                      use_region_embedding=self.use_region_embedding,
                                      use_3class_origin=self.use_3class_origin)

    def seed_worker(self, worker_id):
        # Function to initialize random seeds for each worker process
        worker_seed = torch.initial_seed() % 2**32  # Compute a seed for the worker based on the initial seed of the torch Generator
        np.random.seed(worker_seed)  # Set NumPy's random seed based on the worker seed
        random.seed(worker_seed)  # Set Python's built-in random module's seed

    def train_dataloader(self):
        gen = torch.Generator()
        gen.manual_seed(self.seed)
        # [OPTIMIZATION] Added persistent_workers and prefetch_factor for large dataset training
        # prefetch_factor=2 (default) to limit memory usage with large datasets
        return DataLoader(
            self.train_ds, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn,
            num_workers=self.num_workers, pin_memory=True, worker_init_fn=self.seed_worker, generator=gen,
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=2 if self.num_workers > 0 else None
        )

    def val_dataloader(self):
        gen = torch.Generator()
        gen.manual_seed(self.seed)
        return DataLoader(
            self.val_ds, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_fn_eval,
            num_workers=self.num_workers, pin_memory=True, worker_init_fn=self.seed_worker, generator=gen,
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=2 if self.num_workers > 0 else None
        )

    def test_dataloader(self):
        gen = torch.Generator()
        gen.manual_seed(self.seed)
        return DataLoader(
            self.test_ds, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_fn_eval,
            num_workers=self.num_workers, pin_memory=True, worker_init_fn=self.seed_worker, generator=gen,
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=2 if self.num_workers > 0 else None
        )
