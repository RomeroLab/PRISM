#!/usr/bin/env python
# coding: utf-8

"""
Inference script for ESM2 that calculates per-position log probabilities and per-sequence perplexity.

This script:
1. Loads a dataframe (pickle file)
2. For each sequence, masks each position and calculates the log probability
3. Stores per-position log probabilities as a list
4. Calculates per-sequence perplexity
5. Saves results back to pickle with columns: {experiment_name}_LL and {experiment_name}_PP
6. Supports processing multiple models sequentially, adding all results to the same dataframe

Usage Examples:

    # Single model (backward compatible)
    python inference_esm_with_logprobs.py --config config.yaml --checkpoint model.ckpt --data_path data.pkl

    # Multiple models using JSON string
    python inference_esm_with_logprobs.py --models_json '{"config1.yaml": "checkpoint1.ckpt", "config2.yaml": "checkpoint2.ckpt"}' --data_path data.pkl

    # Multiple models using JSON file
    python inference_esm_with_logprobs.py --models_json_file models.json --data_path data.pkl

    # With lowercase NGL strategy and normalized vocabulary calculation
    # (calculates both 53-vocab and 33-vocab normalized results)
    python inference_esm_with_logprobs.py --models_json_file models.json --data_path data.pkl --lowercase_seq_column NGL_lowercase_seq --use_restricted_vocab

    # With gene conditioning (using JSON vocabulary - preferred)
    python inference_esm_with_logprobs.py --config config.yaml --checkpoint model.ckpt --data_path data.pkl \\
        --gene_vocab_json data/unpaired_OAS/annotated_data_final/gene_vocabulary.json

JSON file format (models.json):
    {
        "configs/config1.yaml": "checkpoints/model1.ckpt",
        "configs/config2.yaml": "checkpoints/model2.ckpt"
    }

Gene vocabulary JSON format (gene_vocabulary.json):
    {
        "genes": ["IGHV1-2", "IGHV1-3", ...],
        "source": "...",
        "total_genes": 365,
        "vocab_size_with_special": 367
    }
"""

import argparse
import yaml
import json
import pandas as pd
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pathlib import Path
from tqdm.auto import tqdm
import numpy as np

import prism
from prism.multimodal_io import GeneVocabulary


def load_config(config_path):
    """Load configuration from YAML file."""
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"Config file is empty: {config_path}")

    return config


def validate_config(config):
    """Validate that all required fields are present in config."""
    required_fields = {
        'model': ['model_identifier'],
        'logging': ['experiment_name']
    }

    for section, fields in required_fields.items():
        if section not in config:
            raise KeyError(f"Missing required section in config: '{section}'")

        for field in fields:
            if field not in config[section]:
                raise KeyError(f"Missing required field in config['{section}']: '{field}'")

    # custom_token_strategy is optional, defaults to "mask_tokens" for backward compatibility
    if 'custom_token_strategy' not in config['model']:
        config['model']['custom_token_strategy'] = "mask_tokens"
        print("Note: custom_token_strategy not specified in config, using default: 'mask_tokens'")

    # Validate custom_token_strategy
    if config['model']['add_custom_tokens']:
        valid_strategies = ['mask_tokens', 'lowercase_ngl', 'hybrid_lowercase']
        if config['model']['custom_token_strategy'] not in valid_strategies:
            raise ValueError(f"Invalid custom_token_strategy: '{config['model']['custom_token_strategy']}'. "
                           f"Must be one of {valid_strategies}")

    # Set defaults for multimodal parameters (optional)
    if 'use_germline_genes' not in config['model']:
        config['model']['use_germline_genes'] = False
    if 'gene_embedding_dim' not in config['model']:
        config['model']['gene_embedding_dim'] = 64
    if 'gene_embedding_dropout' not in config['model']:
        config['model']['gene_embedding_dropout'] = 0.1

    print("✓ Configuration validated successfully")


def load_model(checkpoint_path, device, gene_vocab_json=None):
    """
    Load a PRISM model from a checkpoint using the prism.pretrained() API.

    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on (e.g., "cuda", "cpu")
        gene_vocab_json: Optional path to gene vocabulary JSON file

    Returns:
        tuple: (model, tokenizer, prism_model) - the SFT_ESM2 model, tokenizer, and PrismModel wrapper
    """
    prism_model = prism.pretrained(
        str(checkpoint_path),
        device=str(device),
        gene_vocab_path=gene_vocab_json,
    )
    return prism_model.model, prism_model.tokenizer, prism_model


def calculate_per_position_logprobs_and_perplexity(model, tokenizer, sequences, device, batch_size=None,
                                                    use_lowercase_strategy=False, lowercase_sequences=None,
                                                    use_restricted_vocab=False,
                                                    compute_marginalized=False,
                                                    compute_region_conditioned=False,
                                                    region_masks=None,
                                                    use_germline_genes=False, v_gene_ids=None, j_gene_ids=None,
                                                    use_multihead_architecture=False,
                                                    use_alpha_gating=False,
                                                    use_multiplicative_gating=False):
    """
    Calculate per-position log probabilities and per-sequence perplexity for a list of sequences.

    Uses efficient batched processing: processes the same position across multiple sequences at once,
    similar to the _run_ppl_calculation_on_full_dataset method in the model.

    Supports two gating modes when use_alpha_gating=True:
    - v6 Additive (use_multiplicative_gating=False): Returns raw logits, needs log_softmax
    - v13 Multiplicative (use_multiplicative_gating=True): Returns log-probs, no log_softmax needed

    Args:
        model: The ESM2 model
        tokenizer: The tokenizer
        sequences: List of sequences (strings) - UPPERCASE sequences used for model input
        device: Device to run inference on
        batch_size: Batch size for processing sequences. If None, auto-selects based on architecture:
                   - Multihead: 512 (requires output_hidden_states=True, more memory)
                   - Standard: 4096
        use_lowercase_strategy: If True, use lowercase_sequences to determine which token IDs to extract
        lowercase_sequences: List of sequences with lowercase letters marking NGL positions.
                           Only used to determine which token ID to extract from logits (lowercase for NGL, uppercase for GL)
        use_restricted_vocab: If True, returns both original (53 vocab) and normalized (33 vocab) results:
                            - Original (53 vocab): log probs from final head combining AA identity × GL/NGL status
                            - Normalized (33 vocab): log probs from AA head directly (pure amino acid identity)
                              NOTE: For multihead architecture, we use logits_aa directly instead of merging
                              the 53-vocab final logits. This is more robust because:
                              1. Avoids temperature scaling issues (training uses T=0.5, inference T=1.0)
                              2. Avoids origin head interference (for multiplicative gating, alpha is NOT trained)
                              3. Gives cleaner amino acid identity measurement
                            Returns tuple of 4 lists instead of 2.
        compute_marginalized: If True, computes marginalized log probs from final head (53-vocab) by summing
                            P(upper) + P(lower) using logsumexp. This cancels out origin head contribution
                            when alpha=1. Returns additional 2 lists (marginalized_log_probs, marginalized_ppl).
        use_germline_genes: If True, use V/J gene conditioning during inference
        v_gene_ids: Tensor of V-gene IDs [N] - required if use_germline_genes=True
        j_gene_ids: Tensor of J-gene IDs [N] - required if use_germline_genes=True
        use_multihead_architecture: If True, use multihead forward pass (aa_head + mut_head) and
                                   reconstruct full logits using _reconstruct_full_logits()
        use_alpha_gating: If True, use alpha gating to combine AA and Origin logits for 53-vocab output
        use_multiplicative_gating: If True, use v13 multiplicative gating (log-prob summation).
                                  If False, use v6 additive mixing (raw logits).

    Returns:
        If use_restricted_vocab=False and compute_marginalized=False:
            log_probs_list: List of lists, where each inner list contains log probabilities for each position
            perplexity_list: List of perplexity values (one per sequence)
        If use_restricted_vocab=True:
            Tuple of (log_probs_list, perplexity_list, normalized_log_probs_list, normalized_perplexity_list)
        If compute_marginalized=True:
            Adds marginalized_log_probs_list, marginalized_perplexity_list to the return tuple
    """
    # Set default batch size based on architecture
    # Multihead requires output_hidden_states=True which uses significantly more GPU memory
    if batch_size is None:
        if use_multihead_architecture:
            batch_size = 512  # Smaller batch for multihead (stores all hidden states)
            print(f"Auto-selected batch_size={batch_size} for multihead architecture (memory-intensive)")
        else:
            batch_size = 4096  # Larger batch for standard forward pass
            print(f"Auto-selected batch_size={batch_size} for standard architecture")

    model.eval()
    model = model.to(device)

    mask_token_id = tokenizer.mask_token_id

    # Tokenize all sequences with padding to same length (using UPPERCASE sequences)
    print("Tokenizing sequences...")
    tokens = tokenizer(sequences, return_tensors="pt", add_special_tokens=True, padding=True, truncation=True, max_length=512)
    all_input_ids = tokens['input_ids'].to(device)  # [N, L]
    all_attention_mask = tokens['attention_mask'].to(device)  # [N, L]

    N, L = all_input_ids.shape
    print(f"Tokenized {N} sequences with max length {L}")

    # Initialize tensor to store log probabilities (on CPU to save GPU memory)
    log_probs_tensor = torch.zeros(N, L, dtype=torch.float32, device="cpu")

    # If use_restricted_vocab, also store normalized log probs
    normalized_log_probs_tensor = None
    if use_restricted_vocab:
        normalized_log_probs_tensor = torch.zeros(N, L, dtype=torch.float32, device="cpu")

    # If compute_marginalized, store marginalized log probs (sum of P_upper + P_lower from final head)
    marginalized_log_probs_tensor = None
    if compute_marginalized:
        marginalized_log_probs_tensor = torch.zeros(N, L, dtype=torch.float32, device="cpu")

    # Region-conditioned PPL tensors (allocated after aa_mask validation below)
    region_conditioned_log_probs_tensor = None
    upper_to_lower_map = None
    region_is_cdr = None

    # Create mask for amino acid positions (IDs 4-23 for standard uppercase AAs)
    aa_mask = (all_input_ids >= 4) & (all_input_ids <= 23)  # [N, L]

    # Build region-conditioned CDR mask aligned to token positions
    if compute_region_conditioned and region_masks is not None:
        _CDR_REGIONS = {'1', '3', '5'}
        region_is_cdr = torch.zeros(N, L, dtype=torch.bool)
        aa_mask_cpu = aa_mask.cpu()
        for seq_idx in range(N):
            rmask = region_masks[seq_idx]
            if not rmask:
                continue
            aa_positions = aa_mask_cpu[seq_idx].nonzero(as_tuple=True)[0]
            n_chars = min(len(rmask), len(aa_positions))
            for i in range(n_chars):
                if rmask[i] in _CDR_REGIONS:
                    region_is_cdr[seq_idx, aa_positions[i]] = True
        region_is_cdr = region_is_cdr.to(device)

        # Pre-build uppercase→lowercase token ID mapping for vectorized extraction
        if model.lowercase_aa_token_ids is not None:
            vocab_size = len(tokenizer)
            upper_to_lower_map = torch.arange(vocab_size, dtype=torch.long, device=device)
            for uid, lid in model.lowercase_aa_token_ids.items():
                upper_to_lower_map[uid] = lid
            cdr_count = region_is_cdr.sum().item()
            fr_count = int(aa_mask.sum().item()) - cdr_count
            print(f"Region-conditioned PPL enabled: {cdr_count} CDR positions (→NGL), {fr_count} FR positions (→GL)")
            region_conditioned_log_probs_tensor = torch.zeros(N, L, dtype=torch.float32, device="cpu")
        else:
            print("WARNING: Model has no lowercase tokens — region-conditioned PPL unavailable")
            compute_region_conditioned = False

    # Build vocabulary masks for merging uppercase/lowercase (if enabled)
    is_ngl_position = None  # [N, L] boolean mask indicating NGL positions

    if use_restricted_vocab and use_lowercase_strategy and model.lowercase_aa_token_ids is not None:
        print("\nPreparing for normalized vocab calculation (merging uppercase/lowercase)...")

        # Create is_ngl_position mask [N, L] to track which positions are NGL
        is_ngl_position = torch.zeros(N, L, dtype=torch.bool, device=device)

        # Create mapping from lowercase token ID to uppercase token ID
        lowercase_to_uppercase = {}
        for uppercase_id, lowercase_id in model.lowercase_aa_token_ids.items():
            lowercase_to_uppercase[lowercase_id] = uppercase_id

        print(f"  Will merge {len(lowercase_to_uppercase)} lowercase/uppercase token pairs")

    # If using lowercase strategy, build target token ID mapping from lowercase_sequences
    if use_lowercase_strategy and model.lowercase_aa_token_ids is not None:
        if lowercase_sequences is None:
            raise ValueError("lowercase_sequences must be provided when use_lowercase_strategy=True")

        print("Using lowercase NGL strategy for log probability extraction")
        print(f"Lowercase token mapping has {len(model.lowercase_aa_token_ids)} entries")

        # Create mapping: for each sequence position, store the token ID we should extract log prob for
        print("Building target token ID mapping from NGL_lowercase_seq...")
        target_token_ids = all_input_ids.clone()  # Start with uppercase token IDs [N, L]

        for seq_idx in tqdm(range(N), desc="Mapping target tokens", leave=False):
            lowercase_seq = lowercase_sequences[seq_idx]

            # Track position in the original sequence string
            char_idx = 0

            # Iterate through tokenized positions (skip <cls> at position 0)
            for tok_idx in range(1, L):
                token_id = all_input_ids[seq_idx, tok_idx].item()

                # Skip special tokens
                if token_id in [tokenizer.cls_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id]:
                    continue

                # Check if this is an amino acid token
                if token_id < 4 or token_id > 23:
                    continue

                # Get corresponding character from lowercase_seq
                if char_idx < len(lowercase_seq):
                    char = lowercase_seq[char_idx]

                    if char.islower():
                        # NGL position: use lowercase token ID for extraction
                        uppercase_token_id = token_id  # Current token is uppercase

                        if uppercase_token_id in model.lowercase_aa_token_ids:
                            lowercase_token_id = model.lowercase_aa_token_ids[uppercase_token_id]
                            target_token_ids[seq_idx, tok_idx] = lowercase_token_id

                        # Mark this position as NGL
                        if use_restricted_vocab and is_ngl_position is not None:
                            is_ngl_position[seq_idx, tok_idx] = True
                    # else: GL position, keep uppercase token ID (already in target_token_ids)

                    char_idx += 1

        target_token_ids = target_token_ids.to(device)
        print(f"Built target token mapping: {target_token_ids.shape}")

    else:
        # Standard strategy: extract log probs for uppercase tokens
        target_token_ids = all_input_ids
        print("Using standard strategy (extracting uppercase token log probs)")

    print(f"Total AA positions to process: {aa_mask.sum().item()}")

    # Iterate through each position in the sequence
    pbar = tqdm(range(L), desc="Processing positions")

    # For debugging: track first position we process
    debug_printed = False

    for pos in pbar:
        # Check if this position has any amino acids across all sequences
        if not aa_mask[:, pos].any():
            continue

        # Process in batches to avoid OOM
        for batch_start in range(0, N, batch_size):
            batch_end = min(batch_start + batch_size, N)

            # Get batch
            batch_input_ids = all_input_ids[batch_start:batch_end]  # [B, L]
            batch_attention_mask = all_attention_mask[batch_start:batch_end]  # [B, L]

            # Get target tokens at position pos (what we want to extract log prob for)
            # This could be lowercase or uppercase depending on the strategy and whether position is NGL
            target_tokens_at_pos = target_token_ids[batch_start:batch_end, pos].unsqueeze(1)  # [B, 1]

            # Create masked input (mask position pos with [MASK] token)
            # INPUT ALWAYS USES UPPERCASE SEQUENCE
            masked_input = batch_input_ids.clone()
            masked_input[:, pos] = mask_token_id

            # Forward pass - with or without gene conditioning, with or without multihead architecture
            with torch.no_grad():
                if use_multihead_architecture:
                    # =====================================================================
                    # MULTIHEAD ARCHITECTURE: Use aa_head + mut_head (+ alpha_head if enabled)
                    # =====================================================================
                    # Get batch gene IDs if using gene conditioning
                    batch_v_gene_ids = None
                    batch_j_gene_ids = None
                    if use_germline_genes and v_gene_ids is not None and j_gene_ids is not None:
                        batch_v_gene_ids = v_gene_ids[batch_start:batch_end].to(device)
                        batch_j_gene_ids = j_gene_ids[batch_start:batch_end].to(device)

                    # Use _forward_multihead to get aa_head, mut_head, alpha, and final logits
                    # [v6.0] Now returns 6 values: logits_aa, logits_aa_ngl, logits_mut, alpha, logits_final, hidden_states
                    logits_aa, _, logits_mut, alpha, logits_final, _ = model._forward_multihead(
                        input_ids=masked_input,
                        attention_mask=batch_attention_mask,
                        v_gene_ids=batch_v_gene_ids,
                        j_gene_ids=batch_j_gene_ids
                    )

                    # =====================================================================
                    # Full vocab (53): Use alpha-gated logits if available, else reconstruct
                    # =====================================================================
                    if use_alpha_gating and logits_final is not None:
                        logits_at_pos = logits_final[:, pos, :]  # [B, 53]

                        if use_multiplicative_gating:
                            # [v13] Multiplicative Gating: _construct_53_vocab_logits returns LOG PROBABILITIES
                            # due to log-space probability combination:
                            #   log P(uppercase) = log_softmax(logits_aa) + log_sigmoid(-logits_mut)
                            #   log P(lowercase) = log_softmax(logits_aa) + log_sigmoid(logits_mut)
                            # DO NOT apply log_softmax again - it would destroy the probability distribution!
                            log_probs_at_pos = logits_at_pos  # Already log probs
                        else:
                            # [v6] Additive Mixing: _construct_53_vocab_logits returns RAW LOGITS
                            # Formula: GL = AA - alpha*Mut, NGL = AA + alpha*Mut
                            # Need to apply log_softmax to convert to log probabilities
                            log_probs_at_pos = F.log_softmax(logits_at_pos, dim=-1)  # [B, 53]
                    else:
                        # Legacy: Reconstruct full 53-vocab logits using _reconstruct_full_logits
                        # This combines aa_head (33 effective AAs) with mut_head (GL/NGL probability)
                        # Result: log P(uppercase_AA) = log P(AA) + log P(GL)
                        #         log P(lowercase_AA) = log P(AA) + log P(NGL)
                        full_logits = model._reconstruct_full_logits(logits_aa, logits_mut)  # [B, L, V]

                        # Get logits at the masked position
                        logits_at_pos = full_logits[:, pos, :]  # [B, V] - already log probabilities!

                        # For multihead without alpha gating, _reconstruct_full_logits returns LOG PROBABILITIES
                        # So we DON'T apply log_softmax again
                        log_probs_at_pos = logits_at_pos  # [B, V] - these are already log probs

                    # =====================================================================
                    # Normalized vocab (33): Use AA head directly (CORRECT approach)
                    # =====================================================================
                    # IMPORTANT: For 33-vocab calculation, we use logits_aa directly.
                    #
                    # Why NOT merge 53-vocab final logits:
                    # 1. The 53-vocab includes origin head probabilities (P(GL/NGL))
                    # 2. Temperature scaling may be applied to 53-vocab (training/inference mismatch)
                    # 3. For multiplicative gating: alpha head is NOT trained (no gradient path)
                    # 4. Merging via logaddexp is mathematically equivalent ONLY when T=1.0
                    #
                    # Using AA head directly:
                    # - Gives pure amino acid identity prediction
                    # - No origin head or temperature interference
                    # - Cleaner and more robust
                    normalized_log_probs_at_pos = None
                    if use_restricted_vocab and normalized_log_probs_tensor is not None:
                        # Use AA head logits directly - bypasses origin head completely
                        # AA head outputs 53 tokens, but we want 33 effective tokens (merge upper/lower)
                        aa_logits_at_pos = logits_aa[:, pos, :]  # [B, 53]

                        # Merge uppercase/lowercase pairs in logit space using logaddexp
                        # This gives us the "total probability" for each AA identity
                        merged_logits = aa_logits_at_pos.clone()  # [B, 53]
                        if model.lowercase_aa_token_ids is not None:
                            for upper_id, lower_id in model.lowercase_aa_token_ids.items():
                                upper_logit = aa_logits_at_pos[:, upper_id]
                                lower_logit = aa_logits_at_pos[:, lower_id]
                                # Merge: log(exp(upper) + exp(lower))
                                merged = torch.logaddexp(upper_logit, lower_logit)
                                merged_logits[:, upper_id] = merged
                                merged_logits[:, lower_id] = float('-inf')  # Zero out lowercase

                        # Apply log_softmax to get proper probabilities over 33 effective tokens
                        log_probs_aa_merged = F.log_softmax(merged_logits, dim=-1)  # [B, 53] but 33 effective

                        # Create output tensor - copy merged probs to both upper and lower positions
                        # This allows correct extraction regardless of whether target is upper or lower
                        normalized_log_probs_at_pos = log_probs_aa_merged.clone()

                        # For lowercase positions, copy from corresponding uppercase
                        if model.lowercase_aa_token_ids is not None:
                            for upper_id, lower_id in model.lowercase_aa_token_ids.items():
                                normalized_log_probs_at_pos[:, lower_id] = log_probs_aa_merged[:, upper_id]

                    # =====================================================================
                    # Marginalized: Sum P(upper) + P(lower) from Final Head using logsumexp
                    # =====================================================================
                    # This is different from normalized:
                    # - Normalized: Merges AA head logits BEFORE softmax
                    # - Marginalized: Sums probabilities AFTER softmax from final head
                    # When alpha=1, marginalized cancels origin head and equals AA head
                    marginalized_log_probs_at_pos = None
                    if compute_marginalized and marginalized_log_probs_tensor is not None and model.lowercase_aa_token_ids is not None:
                        # Start with final head log probs (already computed above)
                        marginalized_log_probs_at_pos = log_probs_at_pos.clone()  # [B, 53]

                        # For each AA, compute log(P_upper + P_lower) = logsumexp(log_P_upper, log_P_lower)
                        for upper_id, lower_id in model.lowercase_aa_token_ids.items():
                            log_prob_upper = log_probs_at_pos[:, upper_id]
                            log_prob_lower = log_probs_at_pos[:, lower_id]
                            # Marginalize: log(exp(log_P_upper) + exp(log_P_lower))
                            marginalized = torch.logaddexp(log_prob_upper, log_prob_lower)
                            # Store in both positions for easy extraction
                            marginalized_log_probs_at_pos[:, upper_id] = marginalized
                            marginalized_log_probs_at_pos[:, lower_id] = marginalized

                elif use_germline_genes and v_gene_ids is not None and j_gene_ids is not None:
                    # Use gene-conditioned forward pass (non-multihead)
                    batch_v_gene_ids = v_gene_ids[batch_start:batch_end].to(device)
                    batch_j_gene_ids = j_gene_ids[batch_start:batch_end].to(device)
                    outputs = model._forward_with_gene_conditioning(
                        input_ids=masked_input,
                        attention_mask=batch_attention_mask,
                        v_gene_ids=batch_v_gene_ids,
                        j_gene_ids=batch_j_gene_ids
                    )
                    logits = outputs.logits  # [B, L, V]

                    # Get logits at the masked position
                    logits_at_pos = logits[:, pos, :]  # [B, V]

                    # Calculate log probabilities over full vocabulary
                    log_probs_at_pos = F.log_softmax(logits_at_pos, dim=-1)  # [B, V]

                    # For non-multihead, normalized uses log-sum-exp merging
                    normalized_log_probs_at_pos = None
                    if use_restricted_vocab and normalized_log_probs_tensor is not None:
                        normalized_logits_at_pos = logits_at_pos.clone()
                        for lowercase_id, uppercase_id in lowercase_to_uppercase.items():
                            upper_logits = normalized_logits_at_pos[:, uppercase_id]
                            lower_logits = normalized_logits_at_pos[:, lowercase_id]
                            merged_logits = torch.logaddexp(upper_logits, lower_logits)
                            normalized_logits_at_pos[:, uppercase_id] = merged_logits
                            normalized_logits_at_pos[:, lowercase_id] = float('-inf')
                        normalized_log_probs_at_pos = F.log_softmax(normalized_logits_at_pos, dim=-1)

                    # Marginalized: sum P(upper) + P(lower) after softmax
                    marginalized_log_probs_at_pos = None
                    if compute_marginalized and marginalized_log_probs_tensor is not None:
                        marginalized_log_probs_at_pos = log_probs_at_pos.clone()
                        for lowercase_id, uppercase_id in lowercase_to_uppercase.items():
                            log_prob_upper = log_probs_at_pos[:, uppercase_id]
                            log_prob_lower = log_probs_at_pos[:, lowercase_id]
                            marginalized = torch.logaddexp(log_prob_upper, log_prob_lower)
                            marginalized_log_probs_at_pos[:, uppercase_id] = marginalized
                            marginalized_log_probs_at_pos[:, lowercase_id] = marginalized
                else:
                    # Standard forward pass without gene conditioning (non-multihead)
                    outputs = model.ESM2(
                        input_ids=masked_input,
                        attention_mask=batch_attention_mask
                    )
                    logits = outputs.logits  # [B, L, V]

                    # Get logits at the masked position
                    logits_at_pos = logits[:, pos, :]  # [B, V]

                    # Calculate log probabilities over full vocabulary
                    log_probs_at_pos = F.log_softmax(logits_at_pos, dim=-1)  # [B, V]

                    # For non-multihead, normalized uses log-sum-exp merging
                    normalized_log_probs_at_pos = None
                    if use_restricted_vocab and normalized_log_probs_tensor is not None:
                        normalized_logits_at_pos = logits_at_pos.clone()
                        for lowercase_id, uppercase_id in lowercase_to_uppercase.items():
                            upper_logits = normalized_logits_at_pos[:, uppercase_id]
                            lower_logits = normalized_logits_at_pos[:, lowercase_id]
                            merged_logits = torch.logaddexp(upper_logits, lower_logits)
                            normalized_logits_at_pos[:, uppercase_id] = merged_logits
                            normalized_logits_at_pos[:, lowercase_id] = float('-inf')
                        normalized_log_probs_at_pos = F.log_softmax(normalized_logits_at_pos, dim=-1)

                    # Marginalized: sum P(upper) + P(lower) after softmax
                    marginalized_log_probs_at_pos = None
                    if compute_marginalized and marginalized_log_probs_tensor is not None:
                        marginalized_log_probs_at_pos = log_probs_at_pos.clone()
                        for lowercase_id, uppercase_id in lowercase_to_uppercase.items():
                            log_prob_upper = log_probs_at_pos[:, uppercase_id]
                            log_prob_lower = log_probs_at_pos[:, lowercase_id]
                            marginalized = torch.logaddexp(log_prob_upper, log_prob_lower)
                            marginalized_log_probs_at_pos[:, uppercase_id] = marginalized
                            marginalized_log_probs_at_pos[:, lowercase_id] = marginalized

            # DEBUG OUTPUT: Print details for first 3 sequences at each position (only first batch)
            if batch_start == 0 and not debug_printed:
                print("\n" + "="*80)
                print(f"DEBUG OUTPUT FOR POSITION {pos}")
                print("="*80)

                # Print token-idx relationship (full vocabulary)
                print("\n1) TOKEN-IDX RELATIONSHIP (showing first 60 tokens):")
                print("-" * 80)
                vocab_size = len(tokenizer)
                for idx in range(min(60, vocab_size)):
                    token = tokenizer.convert_ids_to_tokens([idx])[0]
                    print(f"  idx {idx:3d}: '{token}'")

                # Print details for first 3 sequences
                num_debug_seqs = min(3, batch_end - batch_start)
                for seq_idx in range(num_debug_seqs):
                    # Only print if this position is an AA position for this sequence
                    if not aa_mask[seq_idx, pos]:
                        continue

                    print(f"\n{'-'*80}")
                    print(f"SEQUENCE {seq_idx}, POSITION {pos}")
                    print(f"{'-'*80}")

                    # Print original and target token info
                    original_token_id = all_input_ids[seq_idx, pos].item()
                    target_token_id = target_tokens_at_pos[seq_idx, 0].item()
                    original_token = tokenizer.convert_ids_to_tokens([original_token_id])[0]
                    target_token = tokenizer.convert_ids_to_tokens([target_token_id])[0]

                    print(f"Original token ID: {original_token_id} ('{original_token}')")
                    print(f"Target token ID: {target_token_id} ('{target_token}')")

                    if use_restricted_vocab and is_ngl_position is not None:
                        is_ngl = is_ngl_position[seq_idx, pos].item()
                        print(f"Position type: {'NGL (lowercase)' if is_ngl else 'GL (uppercase)'}")

                    # 2) Full logits (top 20 values)
                    print(f"\n2) FULL LOGITS (top 20 values):")
                    logits_seq = logits_at_pos[seq_idx]  # [V]
                    top_logit_values, top_logit_indices = torch.topk(logits_seq, k=min(20, vocab_size))
                    for rank, (val, idx) in enumerate(zip(top_logit_values, top_logit_indices)):
                        token = tokenizer.convert_ids_to_tokens([idx.item()])[0]
                        print(f"  Rank {rank+1:2d}: idx {idx.item():3d} ('{token}') = {val.item():8.4f}")

                    # 3) Full log probs (top 20 values)
                    print(f"\n3) FULL LOG PROBS (top 20 values):")
                    log_probs_seq = log_probs_at_pos[seq_idx]  # [V]
                    top_logprob_values, top_logprob_indices = torch.topk(log_probs_seq, k=min(20, vocab_size))
                    for rank, (val, idx) in enumerate(zip(top_logprob_values, top_logprob_indices)):
                        token = tokenizer.convert_ids_to_tokens([idx.item()])[0]
                        prob = torch.exp(val).item()
                        print(f"  Rank {rank+1:2d}: idx {idx.item():3d} ('{token}') = {val.item():8.4f} (prob={prob:.6f})")

                    # 4) Which index we extract and its value
                    print(f"\n4) EXTRACTED TARGET TOKEN:")
                    target_log_prob = log_probs_seq[target_token_id].item()
                    target_prob = np.exp(target_log_prob)
                    print(f"  Target token ID: {target_token_id} ('{target_token}')")
                    print(f"  Log probability: {target_log_prob:.6f}")
                    print(f"  Probability: {target_prob:.6f}")

                    # Find rank of target token
                    sorted_indices = torch.argsort(log_probs_seq, descending=True)
                    target_rank = (sorted_indices == target_token_id).nonzero(as_tuple=True)[0].item() + 1
                    print(f"  Rank in distribution: {target_rank}/{vocab_size}")

                debug_printed = True
                print("\n" + "="*80 + "\n")

            # Extract log probability for target tokens
            # If NGL position: extracts lowercase token log prob
            # If GL position: extracts uppercase token log prob
            log_prob_target_tokens = log_probs_at_pos.gather(1, target_tokens_at_pos).squeeze(1)  # [B]

            # Mask out non-AA positions (set to 0)
            current_aa_mask = aa_mask[batch_start:batch_end, pos]  # [B]
            log_prob_target_tokens = log_prob_target_tokens * current_aa_mask.float()

            # Store results (GPU -> CPU)
            log_probs_tensor[batch_start:batch_end, pos] = log_prob_target_tokens.cpu()

            # If use_restricted_vocab, also extract and store normalized log probs
            if use_restricted_vocab and normalized_log_probs_at_pos is not None:
                # For normalized, we need to extract from the UPPERCASE token position
                # (since lowercase logits were merged into uppercase)
                # Map target_tokens to uppercase equivalents
                normalized_target_tokens = target_tokens_at_pos.clone()
                for b_idx in range(len(normalized_target_tokens)):
                    token_id = normalized_target_tokens[b_idx, 0].item()
                    # If this is a lowercase token, map it to uppercase
                    if token_id in lowercase_to_uppercase:
                        normalized_target_tokens[b_idx, 0] = lowercase_to_uppercase[token_id]

                # Extract normalized log prob
                normalized_log_prob_target_tokens = normalized_log_probs_at_pos.gather(1, normalized_target_tokens).squeeze(1)  # [B]
                normalized_log_prob_target_tokens = normalized_log_prob_target_tokens * current_aa_mask.float()

                # Store normalized results (GPU -> CPU)
                normalized_log_probs_tensor[batch_start:batch_end, pos] = normalized_log_prob_target_tokens.cpu()

            # If compute_marginalized, also extract and store marginalized log probs
            if compute_marginalized and marginalized_log_probs_at_pos is not None:
                # For marginalized, we can extract from either uppercase or lowercase position
                # (they have the same value after marginalization)
                # Use target_tokens_at_pos directly
                marginalized_log_prob_target_tokens = marginalized_log_probs_at_pos.gather(1, target_tokens_at_pos).squeeze(1)  # [B]
                marginalized_log_prob_target_tokens = marginalized_log_prob_target_tokens * current_aa_mask.float()

                # Store marginalized results (GPU -> CPU)
                marginalized_log_probs_tensor[batch_start:batch_end, pos] = marginalized_log_prob_target_tokens.cpu()

            # If compute_region_conditioned, extract GL and NGL log probs and select by region
            if compute_region_conditioned and region_is_cdr is not None and upper_to_lower_map is not None:
                upper_ids = all_input_ids[batch_start:batch_end, pos]  # [B]
                lower_ids = upper_to_lower_map[upper_ids]  # [B]

                gl_lp = log_probs_at_pos.gather(1, upper_ids.unsqueeze(1)).squeeze(1)  # [B]
                ngl_lp = log_probs_at_pos.gather(1, lower_ids.unsqueeze(1)).squeeze(1)  # [B]

                batch_is_cdr = region_is_cdr[batch_start:batch_end, pos]  # [B]
                rc_lp = torch.where(batch_is_cdr, ngl_lp, gl_lp)  # [B]
                rc_lp = rc_lp * current_aa_mask.float()

                region_conditioned_log_probs_tensor[batch_start:batch_end, pos] = rc_lp.cpu()

    # Convert tensor to list of lists (only AA positions)
    print("\nConverting results to per-sequence lists...")
    log_probs_list = []
    perplexity_list = []
    normalized_log_probs_list = []
    normalized_perplexity_list = []
    marginalized_log_probs_list = []
    marginalized_perplexity_list = []
    rc_log_probs_list = []
    rc_perplexity_list = []

    for i in tqdm(range(N), desc="Computing per-sequence perplexity"):
        # Get AA positions for this sequence
        seq_aa_mask = aa_mask[i].cpu()  # [L]
        seq_log_probs = log_probs_tensor[i][seq_aa_mask].numpy().tolist()  # List of log probs at AA positions

        # Calculate perplexity
        if len(seq_log_probs) > 0:
            mean_log_prob = np.mean(seq_log_probs)
            perplexity = np.exp(-mean_log_prob)
        else:
            perplexity = 0.0

        log_probs_list.append(seq_log_probs)
        perplexity_list.append(perplexity)

        # If use_restricted_vocab, also compute normalized results
        if use_restricted_vocab and normalized_log_probs_tensor is not None:
            seq_normalized_log_probs = normalized_log_probs_tensor[i][seq_aa_mask].numpy().tolist()

            # Calculate normalized perplexity
            if len(seq_normalized_log_probs) > 0:
                mean_normalized_log_prob = np.mean(seq_normalized_log_probs)
                normalized_perplexity = np.exp(-mean_normalized_log_prob)
            else:
                normalized_perplexity = 0.0

            normalized_log_probs_list.append(seq_normalized_log_probs)
            normalized_perplexity_list.append(normalized_perplexity)

        # If compute_marginalized, also compute marginalized results
        if compute_marginalized and marginalized_log_probs_tensor is not None:
            seq_marginalized_log_probs = marginalized_log_probs_tensor[i][seq_aa_mask].numpy().tolist()

            # Calculate marginalized perplexity
            if len(seq_marginalized_log_probs) > 0:
                mean_marginalized_log_prob = np.mean(seq_marginalized_log_probs)
                marginalized_perplexity = np.exp(-mean_marginalized_log_prob)
            else:
                marginalized_perplexity = 0.0

            marginalized_log_probs_list.append(seq_marginalized_log_probs)
            marginalized_perplexity_list.append(marginalized_perplexity)

        # If compute_region_conditioned, also compute region-conditioned results
        if compute_region_conditioned and region_conditioned_log_probs_tensor is not None:
            seq_rc_log_probs = region_conditioned_log_probs_tensor[i][seq_aa_mask].numpy().tolist()

            if len(seq_rc_log_probs) > 0:
                mean_rc_log_prob = np.mean(seq_rc_log_probs)
                rc_perplexity = np.exp(-mean_rc_log_prob)
            else:
                rc_perplexity = 0.0

            rc_log_probs_list.append(seq_rc_log_probs)
            rc_perplexity_list.append(rc_perplexity)

    # Return based on enabled options
    result = [log_probs_list, perplexity_list]
    if use_restricted_vocab:
        result.extend([normalized_log_probs_list, normalized_perplexity_list])
    if compute_marginalized:
        result.extend([marginalized_log_probs_list, marginalized_perplexity_list])
    if compute_region_conditioned and region_conditioned_log_probs_tensor is not None:
        result.extend([rc_log_probs_list, rc_perplexity_list])

    return tuple(result) if len(result) > 2 else tuple(result)


def validate_model_can_load(config_path, checkpoint_path, device, gene_vocab=None, gene_vocab_json=None):
    """
    Validate that a model can be created and checkpoint loaded without errors.

    Args:
        config_path: Path to config YAML file
        checkpoint_path: Path to model checkpoint
        device: Device to use for loading
        gene_vocab: GeneVocabulary instance (not used with new API, kept for interface compat)
        gene_vocab_json: Path to gene vocabulary JSON file

    Returns:
        tuple: (success: bool, error_message: str or None, config: dict or None)
    """
    try:
        # Load and validate configuration
        config = load_config(config_path)
        validate_config(config)

        # Verify checkpoint exists
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            return False, f"Checkpoint file not found: {checkpoint_path}", None

        # Try to load model using prism.pretrained() API
        model, tokenizer, prism_model = load_model(
            checkpoint_path, device="cpu", gene_vocab_json=gene_vocab_json
        )

        # Clean up
        del model
        del tokenizer
        del prism_model
        torch.cuda.empty_cache()

        return True, None, config

    except Exception as e:
        import traceback
        return False, f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}", None


def process_single_model(config_path, checkpoint_path, df, sequence_column='sequence',
                        lowercase_seq_column=None, use_restricted_vocab=False,
                        compute_marginalized=False,
                        compute_region_conditioned=False,
                        gene_vocab=None, gene_vocab_json=None,
                        v_gene_column=None, j_gene_column=None,
                        use_separate_chains=False,
                        v_gene_heavy_column='v_gene_heavy', v_gene_light_column='v_gene_light',
                        j_gene_heavy_column='j_gene_heavy', j_gene_light_column='j_gene_light',
                        region_mask_heavy_column='region_mask_heavy', region_mask_light_column='region_mask_light',
                        batch_size=None, use_compile=False):
    """
    Process a single model and add its results to the dataframe.

    Args:
        config_path: Path to config YAML file
        checkpoint_path: Path to model checkpoint
        df: DataFrame containing sequences
        sequence_column: Name of the column containing sequences
        lowercase_seq_column: Name of column with lowercase sequences for NGL strategy
        use_restricted_vocab: Whether to use restricted vocabulary for softmax
        compute_marginalized: Whether to compute marginalized PPL (sum P_upper + P_lower from final head)
        gene_vocab: GeneVocabulary instance (used for gene encoding in inference)
        gene_vocab_json: Path to gene vocabulary JSON file (used by prism.pretrained() for model loading)
        v_gene_column: Name of column with V-gene labels (single column mode)
        j_gene_column: Name of column with J-gene labels (single column mode)
        use_separate_chains: If True, use separate heavy/light chain columns
        v_gene_heavy_column: Name of column with heavy chain V-gene labels
        v_gene_light_column: Name of column with light chain V-gene labels
        j_gene_heavy_column: Name of column with heavy chain J-gene labels
        j_gene_light_column: Name of column with light chain J-gene labels
        region_mask_heavy_column: Name of column with heavy chain region mask
        region_mask_light_column: Name of column with light chain region mask
        batch_size: Batch size for inference. If None, auto-selects based on architecture
        use_compile: If True, use torch.compile for faster inference (PyTorch 2.0+)

    Returns:
        df: DataFrame with added columns for this model's results
    """
    print("\n" + "="*80)
    print(f"Processing model:")
    print(f"  Config: {config_path}")
    print(f"  Checkpoint: {checkpoint_path}")
    print("="*80)

    # Load and validate configuration
    print(f"\nLoading configuration from: {config_path}")
    config = load_config(config_path)
    validate_config(config)

    experiment_name = config['logging']['experiment_name']
    print(f"\nExperiment name: {experiment_name}")

    # Verify checkpoint exists
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    print(f"Checkpoint: {checkpoint_path}")

    # Check if sequence column exists
    if sequence_column not in df.columns:
        raise ValueError(f"Sequence column '{sequence_column}' not found in dataframe. "
                        f"Available columns: {df.columns.tolist()}")

    print(f"\nUsing sequence column: '{sequence_column}'")

    # Check if lowercase sequence column is provided and exists
    use_lowercase_strategy = False
    lowercase_sequences = None

    if lowercase_seq_column:
        if lowercase_seq_column not in df.columns:
            raise ValueError(f"Lowercase sequence column '{lowercase_seq_column}' not found in dataframe. "
                           f"Available columns: {df.columns.tolist()}")

        print(f"Using lowercase sequence column for NGL strategy: '{lowercase_seq_column}'")
        use_lowercase_strategy = True
        lowercase_sequences = df[lowercase_seq_column].tolist()
    else:
        print("No lowercase sequence column provided - using standard strategy")

    # Validate restricted vocabulary flag
    if use_restricted_vocab:
        if not use_lowercase_strategy:
            raise ValueError("use_restricted_vocab requires lowercase_seq_column to be specified!")
        print("\n" + "="*80)
        print("NORMALIZED VOCABULARY MODE ENABLED")
        print("="*80)
        print("  Will calculate BOTH:")
        print("  1. Original (53 vocab): log probs from final head (53-token output)")
        print("  2. Normalized (33 vocab): log probs from final head, merged via log-sum-exp")
        print("     - Uses FINAL HEAD logits (not aa_head)")
        print("     - Merges uppercase/lowercase pairs: log(exp(upper) + exp(lower))")
        print("     - Results in 33 effective tokens (like original ESM2)")
        print("  Saves 4 columns: {exp}_LL, {exp}_PP, {exp}_Norm_LL, {exp}_Norm_PP")
        print("="*80)
    else:
        print("\nUsing standard calculation (53 vocab only, saves 2 columns)")

    # Auto-detect if lowercase strategy should be used based on custom_token_strategy
    if config['model']['custom_token_strategy'] in ['lowercase_ngl', 'hybrid_lowercase'] and not use_lowercase_strategy:
        print(f"\nWARNING: Model was trained with '{config['model']['custom_token_strategy']}' strategy,")
        print(f"         but --lowercase_seq_column was not provided for inference!")
        print(f"         This will cause incorrect perplexity calculation for NGL positions.")
        print(f"         Please re-run with: --lowercase_seq_column NGL_lowercase_seq")
    elif config['model']['custom_token_strategy'] == 'mask_tokens' and use_lowercase_strategy:
        print(f"\nWARNING: Model was trained with 'mask_tokens' strategy,")
        print(f"         but --lowercase_seq_column was provided for inference!")
        print(f"         This will cause incorrect perplexity calculation.")
        print(f"         Please re-run WITHOUT --lowercase_seq_column flag.")

    print(f"Number of sequences: {len(df)}")

    # Check if multimodal features are required
    use_germline_genes = config['model'].get('use_germline_genes', False)
    num_genes = 0
    v_gene_ids = None
    j_gene_ids = None

    if use_germline_genes:
        print("\n" + "="*80)
        print("MULTIMODAL: Gene Conditioning Enabled")
        print("="*80)

        if gene_vocab is None:
            raise ValueError("Config requires use_germline_genes=True but no gene_vocab provided!")

        num_genes = len(gene_vocab)
        print(f"  Gene vocabulary size: {num_genes}")

        if use_separate_chains:
            # Heavy/light chain mode: encode genes separately then concatenate
            print(f"  Mode: Separate heavy/light chain columns")
            print(f"  V-gene columns: '{v_gene_heavy_column}', '{v_gene_light_column}'")
            print(f"  J-gene columns: '{j_gene_heavy_column}', '{j_gene_light_column}'")

            # Encode heavy chain V-gene IDs
            v_gene_heavy_ids = torch.tensor(
                [gene_vocab.encode(g) for g in df[v_gene_heavy_column].tolist()],
                dtype=torch.long
            )
            # Encode light chain V-gene IDs
            v_gene_light_ids = torch.tensor(
                [gene_vocab.encode(g) for g in df[v_gene_light_column].tolist()],
                dtype=torch.long
            )
            # Encode heavy chain J-gene IDs
            j_gene_heavy_ids = torch.tensor(
                [gene_vocab.encode(g) for g in df[j_gene_heavy_column].tolist()],
                dtype=torch.long
            )
            # Encode light chain J-gene IDs
            j_gene_light_ids = torch.tensor(
                [gene_vocab.encode(g) for g in df[j_gene_light_column].tolist()],
                dtype=torch.long
            )

            # For now, we use heavy chain genes as the primary conditioning
            # (This matches the model's expectation of single V/J gene per sequence)
            # TODO: Future enhancement - support multi-chain gene conditioning
            v_gene_ids = v_gene_heavy_ids
            j_gene_ids = j_gene_heavy_ids

            print(f"  Using heavy chain genes for conditioning (v_gene_heavy, j_gene_heavy)")
            print(f"  Encoded {len(v_gene_ids)} V-gene IDs and {len(j_gene_ids)} J-gene IDs")

        else:
            # Single column mode
            print(f"  Mode: Single v_gene/j_gene columns")

            # Validate gene columns exist
            if v_gene_column not in df.columns:
                raise ValueError(f"V-gene column '{v_gene_column}' not found in dataframe. "
                               f"Available columns: {df.columns.tolist()}")
            if j_gene_column not in df.columns:
                raise ValueError(f"J-gene column '{j_gene_column}' not found in dataframe. "
                               f"Available columns: {df.columns.tolist()}")

            # Encode gene IDs
            print(f"  Encoding gene IDs from columns: '{v_gene_column}', '{j_gene_column}'")
            v_gene_ids = torch.tensor(
                [gene_vocab.encode(g) for g in df[v_gene_column].tolist()],
                dtype=torch.long
            )
            j_gene_ids = torch.tensor(
                [gene_vocab.encode(g) for g in df[j_gene_column].tolist()],
                dtype=torch.long
            )
            print(f"  Encoded {len(v_gene_ids)} V-gene IDs and {len(j_gene_ids)} J-gene IDs")

    # Load model using prism.pretrained() API
    print("\n" + "="*80)
    print("Loading model...")
    print("="*80)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if use_germline_genes:
        print(f"  Gene conditioning: enabled ({num_genes} genes)")
    else:
        print(f"  Gene conditioning: disabled")

    # Determine gene vocab JSON path for prism.pretrained()
    # Prefer the explicit argument, fall back to config
    gene_vocab_json_path = gene_vocab_json or config['model'].get('gene_vocab_path', None)

    # Load model via prism.pretrained() API
    model, tokenizer, prism_model = load_model(
        checkpoint_path, device=str(device), gene_vocab_json=gene_vocab_json_path
    )

    # Read architecture flags from the loaded model's hyperparameters for logging
    use_multihead_architecture = getattr(model, 'use_multihead_architecture', True)
    use_alpha_gating = getattr(model, 'use_alpha_gating', True)
    use_multiplicative_gating_flag = getattr(model, 'use_multiplicative_gating', True)
    gating_temperature = getattr(model, 'gating_temperature', 1.0)

    if use_multihead_architecture:
        print(f"\n" + "="*60)
        print("MULTIHEAD ARCHITECTURE MODE")
        print("="*60)
        print(f"  AA Head: Predicts amino acid identity (33 effective tokens)")
        print(f"  Mutation Head: Predicts GL/NGL probability")

        if use_alpha_gating:
            if use_multiplicative_gating_flag:
                print(f"  [v13/v33.6] Alpha-Weighted Multiplicative Gating: ENABLED")
                print(f"    - Formula: P(token) = P(AA identity) x P(GL/NGL status)^alpha")
                print(f"    - Alpha: Learned per-position gating [0=pure AA, 1=full gating]")
                print(f"    - Gating Temperature (training): {gating_temperature}")
            else:
                print(f"  [v6] Additive Mixing: ENABLED")
        else:
            print(f"  Alpha Gating: disabled (using legacy _reconstruct_full_logits)")

        if use_restricted_vocab:
            print(f"  Normalized (33 vocab): Uses AA head directly (NOT merged from 53-vocab)")
        print("="*60)

    print("Model loaded successfully")

    # Optional: Use torch.compile for faster inference (PyTorch 2.0+)
    if use_compile:
        print("\nCompiling model with torch.compile...")
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("Model compiled successfully (first inference will be slower due to compilation)")
        except Exception as e:
            print(f"torch.compile failed: {e}")
            print("  Continuing without compilation...")

    # Calculate per-position log probabilities and perplexity
    print("\n" + "="*80)
    print("Calculating per-position log probabilities and perplexity...")
    print("="*80)

    sequences = df[sequence_column].tolist()
    use_multiplicative_gating = use_multiplicative_gating_flag

    # Build region masks for region-conditioned PPL (CDR→NGL, FR→GL)
    region_masks = None
    if compute_region_conditioned:
        if region_mask_heavy_column in df.columns:
            rmh = df[region_mask_heavy_column].fillna("").astype(str)
            if region_mask_light_column in df.columns:
                rml = df[region_mask_light_column].fillna("").astype(str)
                region_masks = (rmh + rml).tolist()
                print(f"Region-conditioned PPL: using '{region_mask_heavy_column}' + '{region_mask_light_column}'")
            else:
                region_masks = rmh.tolist()
                print(f"Region-conditioned PPL: using '{region_mask_heavy_column}' only (unpaired)")
        else:
            print(f"WARNING: --compute_region_conditioned specified but '{region_mask_heavy_column}' "
                  f"column not found in dataframe. Skipping region-conditioned PPL.")
            compute_region_conditioned = False

    # Log batch size being used
    if batch_size is not None:
        print(f"\nUsing custom batch_size={batch_size}")

    results = calculate_per_position_logprobs_and_perplexity(
        model, tokenizer, sequences, device,
        batch_size=batch_size,  # Pass custom batch size
        use_lowercase_strategy=use_lowercase_strategy,
        lowercase_sequences=lowercase_sequences,
        use_restricted_vocab=use_restricted_vocab,
        compute_marginalized=compute_marginalized,
        compute_region_conditioned=compute_region_conditioned,
        region_masks=region_masks,
        use_germline_genes=use_germline_genes,
        v_gene_ids=v_gene_ids,
        j_gene_ids=j_gene_ids,
        use_multihead_architecture=use_multihead_architecture,
        use_alpha_gating=use_alpha_gating,
        use_multiplicative_gating=use_multiplicative_gating
    )

    # Unpack results based on enabled options
    # Base results are always first: log_probs_list, perplexity_list
    # Then normalized if use_restricted_vocab
    # Then marginalized if compute_marginalized
    result_idx = 0
    log_probs_list = results[result_idx]
    perplexity_list = results[result_idx + 1]
    result_idx += 2

    normalized_log_probs_list = None
    normalized_perplexity_list = None
    if use_restricted_vocab:
        normalized_log_probs_list = results[result_idx]
        normalized_perplexity_list = results[result_idx + 1]
        result_idx += 2

    marginalized_log_probs_list = None
    marginalized_perplexity_list = None
    if compute_marginalized:
        marginalized_log_probs_list = results[result_idx]
        marginalized_perplexity_list = results[result_idx + 1]
        result_idx += 2

    rc_log_probs_list = None
    rc_perplexity_list = None
    if compute_region_conditioned:
        rc_log_probs_list = results[result_idx]
        rc_perplexity_list = results[result_idx + 1]

    # Add results to dataframe
    print("\n" + "="*80)
    print("Adding results to dataframe...")
    print("="*80)

    # Always save base columns
    ll_column = f"{experiment_name}_LL"
    pp_column = f"{experiment_name}_PP"
    df[ll_column] = log_probs_list
    df[pp_column] = perplexity_list
    added_columns = [ll_column, pp_column]

    # Save normalized columns if enabled
    if use_restricted_vocab and normalized_log_probs_list is not None:
        norm_ll_column = f"{experiment_name}_Norm_LL"
        norm_pp_column = f"{experiment_name}_Norm_PP"
        df[norm_ll_column] = normalized_log_probs_list
        df[norm_pp_column] = normalized_perplexity_list
        added_columns.extend([norm_ll_column, norm_pp_column])

    # Save marginalized columns if enabled
    if compute_marginalized and marginalized_log_probs_list is not None:
        marg_ll_column = f"{experiment_name}_Marg_LL"
        marg_pp_column = f"{experiment_name}_Marg_PP"
        df[marg_ll_column] = marginalized_log_probs_list
        df[marg_pp_column] = marginalized_perplexity_list
        added_columns.extend([marg_ll_column, marg_pp_column])

    # Save region-conditioned columns if enabled
    if compute_region_conditioned and rc_log_probs_list is not None:
        rc_ll_column = f"{experiment_name}_RC_LL"
        rc_pp_column = f"{experiment_name}_RC_PP"
        df[rc_ll_column] = rc_log_probs_list
        df[rc_pp_column] = rc_perplexity_list
        added_columns.extend([rc_ll_column, rc_pp_column])

    print(f"Added columns: {', '.join(added_columns)}")

    # Print statistics
    print("\n" + "="*80)
    print(f"Statistics for {experiment_name}")
    print("="*80)

    print(f"\n[Original - 53 vocab] Perplexity statistics:")
    print(f"  Mean: {np.mean(perplexity_list):.4f}")
    print(f"  Median: {np.median(perplexity_list):.4f}")
    print(f"  Min: {np.min(perplexity_list):.4f}")
    print(f"  Max: {np.max(perplexity_list):.4f}")

    print(f"\n[Original - 53 vocab] Log probability statistics (across all positions):")
    all_log_probs = [lp for seq_lps in log_probs_list for lp in seq_lps]
    print(f"  Mean: {np.mean(all_log_probs):.4f}")
    print(f"  Median: {np.median(all_log_probs):.4f}")
    print(f"  Min: {np.min(all_log_probs):.4f}")
    print(f"  Max: {np.max(all_log_probs):.4f}")

    if use_restricted_vocab:
        print(f"\n[Normalized - 33 vocab] Perplexity statistics:")
        print(f"  Mean: {np.mean(normalized_perplexity_list):.4f}")
        print(f"  Median: {np.median(normalized_perplexity_list):.4f}")
        print(f"  Min: {np.min(normalized_perplexity_list):.4f}")
        print(f"  Max: {np.max(normalized_perplexity_list):.4f}")

        print(f"\n[Normalized - 33 vocab] Log probability statistics (across all positions):")
        all_normalized_log_probs = [lp for seq_lps in normalized_log_probs_list for lp in seq_lps]
        print(f"  Mean: {np.mean(all_normalized_log_probs):.4f}")
        print(f"  Median: {np.median(all_normalized_log_probs):.4f}")
        print(f"  Min: {np.min(all_normalized_log_probs):.4f}")
        print(f"  Max: {np.max(all_normalized_log_probs):.4f}")

    if compute_marginalized and marginalized_perplexity_list is not None:
        print(f"\n[Marginalized] Perplexity statistics:")
        print(f"  Mean: {np.mean(marginalized_perplexity_list):.4f}")
        print(f"  Median: {np.median(marginalized_perplexity_list):.4f}")
        print(f"  Min: {np.min(marginalized_perplexity_list):.4f}")
        print(f"  Max: {np.max(marginalized_perplexity_list):.4f}")

        print(f"\n[Marginalized] Log probability statistics (across all positions):")
        all_marginalized_log_probs = [lp for seq_lps in marginalized_log_probs_list for lp in seq_lps]
        print(f"  Mean: {np.mean(all_marginalized_log_probs):.4f}")
        print(f"  Median: {np.median(all_marginalized_log_probs):.4f}")
        print(f"  Min: {np.min(all_marginalized_log_probs):.4f}")
        print(f"  Max: {np.max(all_marginalized_log_probs):.4f}")

    if compute_region_conditioned and rc_perplexity_list is not None:
        print(f"\n[Region-Conditioned (CDR→NGL, FR→GL)] Perplexity statistics:")
        print(f"  Mean: {np.mean(rc_perplexity_list):.4f}")
        print(f"  Median: {np.median(rc_perplexity_list):.4f}")
        print(f"  Min: {np.min(rc_perplexity_list):.4f}")
        print(f"  Max: {np.max(rc_perplexity_list):.4f}")

        print(f"\n[Region-Conditioned] Log probability statistics (across all positions):")
        all_rc_log_probs = [lp for seq_lps in rc_log_probs_list for lp in seq_lps]
        print(f"  Mean: {np.mean(all_rc_log_probs):.4f}")
        print(f"  Median: {np.median(all_rc_log_probs):.4f}")
        print(f"  Min: {np.min(all_rc_log_probs):.4f}")
        print(f"  Max: {np.max(all_rc_log_probs):.4f}")

    print("\n" + "="*80)
    print(f"✓ Completed processing for {experiment_name}")
    print("="*80)

    return df


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Calculate per-position log probabilities and perplexity for one or more models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single model (backward compatible)
  python inference_esm_with_logprobs.py --config config.yaml --checkpoint model.ckpt --data_path data.pkl

  # Multiple models using JSON string
  python inference_esm_with_logprobs.py --models_json '{"config1.yaml": "checkpoint1.ckpt", "config2.yaml": "checkpoint2.ckpt"}' --data_path data.pkl

  # Multiple models using JSON file
  python inference_esm_with_logprobs.py --models_json_file models.json --data_path data.pkl

  # With lowercase NGL strategy
  python inference_esm_with_logprobs.py --models_json_file models.json --data_path data.pkl --lowercase_seq_column NGL_lowercase_seq

  # With restricted vocabulary
  python inference_esm_with_logprobs.py --models_json_file models.json --data_path data.pkl --lowercase_seq_column NGL_lowercase_seq --use_restricted_vocab
        """
    )

    # Single model arguments (backward compatible)
    parser.add_argument('--config', type=str, help='Path to config YAML file (single model mode)')
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint (single model mode)')

    # Multiple models arguments
    parser.add_argument('--models_json', type=str,
                       help='JSON string of config-checkpoint pairs: {"config1.yaml": "checkpoint1.ckpt", ...}')
    parser.add_argument('--models_json_file', type=str,
                       help='Path to JSON file containing config-checkpoint pairs')

    # Data arguments
    parser.add_argument('--data_path', type=str, required=True, help='Path to input pickle file (DataFrame)')
    parser.add_argument('--output_path', type=str, default=None, help='Optional: Override output path')
    parser.add_argument('--sequence_column', type=str, default='sequence',
                       help='Name of the column containing sequences (default: sequence)')
    parser.add_argument('--lowercase_seq_column', type=str, default=None,
                       help='Optional: Name of column with lowercase sequences for NGL strategy (e.g., NGL_lowercase_seq)')
    parser.add_argument('--use_restricted_vocab', action='store_true',
                       help='Calculate both original (53 vocab) and normalized (33 vocab) results. '
                            'Original: log probs directly from 53-token logits. '
                            'Normalized: log probs after merging uppercase/lowercase AA logits into 33 tokens (like original ESM2). '
                            'Saves 4 columns per model: {exp}_LL, {exp}_PP, {exp}_Norm_LL, {exp}_Norm_PP. '
                            'Only works with --lowercase_seq_column.')
    parser.add_argument('--compute_marginalized', action='store_true',
                       help='Compute marginalized PPL by summing P(upper) + P(lower) from final head using logsumexp. '
                            'This cancels out origin head contribution when alpha=1. '
                            'Adds {exp}_Marg_LL and {exp}_Marg_PP columns. '
                            'Only works with --lowercase_seq_column.')
    parser.add_argument('--compute_region_conditioned', action='store_true',
                       help='Compute region-conditioned PPL: uses GL (uppercase) log-prob for FR positions '
                            'and NGL (lowercase) log-prob for CDR positions. '
                            'Requires region_mask_heavy (and optionally region_mask_light) columns in dataframe. '
                            'Region encoding: 0=FR1, 1=CDR1, 2=FR2, 3=CDR2, 4=FR3, 5=CDR3, 6=FR4. '
                            'Adds {exp}_RC_LL and {exp}_RC_PP columns.')

    # Multimodal (gene conditioning) arguments
    # Supports two formats:
    # 1. Single v_gene/j_gene columns (for pre-concatenated sequences)
    # 2. Separate heavy/light chain columns (v_gene_heavy, v_gene_light, etc.)
    parser.add_argument('--v_gene_column', type=str, default=None,
                       help='Name of column with V-gene labels. If not provided, will look for '
                            'v_gene_heavy and v_gene_light columns.')
    parser.add_argument('--j_gene_column', type=str, default=None,
                       help='Name of column with J-gene labels. If not provided, will look for '
                            'j_gene_heavy and j_gene_light columns.')
    parser.add_argument('--v_gene_heavy_column', type=str, default='v_gene_heavy',
                       help='Name of column with heavy chain V-gene labels (default: v_gene_heavy)')
    parser.add_argument('--v_gene_light_column', type=str, default='v_gene_light',
                       help='Name of column with light chain V-gene labels (default: v_gene_light)')
    parser.add_argument('--j_gene_heavy_column', type=str, default='j_gene_heavy',
                       help='Name of column with heavy chain J-gene labels (default: j_gene_heavy)')
    parser.add_argument('--j_gene_light_column', type=str, default='j_gene_light',
                       help='Name of column with light chain J-gene labels (default: j_gene_light)')
    parser.add_argument('--region_mask_heavy_column', type=str, default='region_mask_heavy',
                       help='Name of column with heavy chain region mask (default: region_mask_heavy)')
    parser.add_argument('--region_mask_light_column', type=str, default='region_mask_light',
                       help='Name of column with light chain region mask (default: region_mask_light)')
    parser.add_argument('--gene_vocab_json', type=str,
                       default='data/unpaired_OAS/annotated_data_final/gene_vocabulary.json',
                       help='Path to gene vocabulary JSON file (preferred method). '
                            'The JSON should contain a "genes" list. '
                            '(default: gene_vocabulary.json)')
    parser.add_argument('--gene_vocab_data_path', type=str, default=None,
                       help='[DEPRECATED] Path to training data pickle file for building gene vocabulary. '
                            'Use --gene_vocab_json instead for faster loading. '
                            'Only used if --gene_vocab_json is not found.')

    # Performance arguments
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size for inference. If not specified, auto-selects based on architecture: '
                            'multihead=512, standard=4096. For L40S GPUs (46GB), try 2048-4096 for multihead, '
                            '8192-16384 for standard.')
    parser.add_argument('--compile', action='store_true',
                       help='Use torch.compile for faster inference (PyTorch 2.0+). '
                            'Can provide 1.5-2x speedup after initial compilation.')
    args = parser.parse_args()

    print("="*80)
    print("ESM2 Per-Position Log Probability and Perplexity Calculation")
    print("="*80)

    # Determine which mode: single model or multiple models
    models_dict = {}

    if args.models_json or args.models_json_file:
        # Multiple models mode
        if args.models_json:
            print("\nLoading models from JSON string...")
            models_dict = json.loads(args.models_json)
        elif args.models_json_file:
            print(f"\nLoading models from JSON file: {args.models_json_file}")
            with open(args.models_json_file, 'r') as f:
                models_dict = json.load(f)

        if args.config or args.checkpoint:
            print("\nWARNING: --config and --checkpoint are ignored when using --models_json or --models_json_file")

    elif args.config and args.checkpoint:
        # Single model mode (backward compatible)
        print("\nSingle model mode")
        models_dict = {args.config: args.checkpoint}

    else:
        raise ValueError(
            "You must provide either:\n"
            "  1. --config and --checkpoint (single model), OR\n"
            "  2. --models_json (JSON string), OR\n"
            "  3. --models_json_file (path to JSON file)"
        )

    print(f"\nProcessing {len(models_dict)} model(s):")
    for config_path, checkpoint_path in models_dict.items():
        print(f"  - Config: {config_path}")
        print(f"    Checkpoint: {checkpoint_path}")

    # Load data FIRST (needed for gene vocabulary building)
    print("\n" + "=" * 80)
    print("Loading data...")
    print("=" * 80)

    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_pickle(data_path)
    print(f"Loaded dataframe with shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    # Check if any model requires gene conditioning
    print("\n" + "=" * 80)
    print("Checking multimodal requirements...")
    print("=" * 80)

    any_model_needs_genes = False
    for config_path in models_dict.keys():
        config = load_config(config_path)
        if config['model'].get('use_germline_genes', False):
            any_model_needs_genes = True
            print(f"  {config_path}: requires gene conditioning")
        else:
            print(f"  {config_path}: no gene conditioning")

    # Build gene vocabulary if any model needs it
    gene_vocab = None
    use_separate_chains = False  # Flag for heavy/light chain mode
    if any_model_needs_genes:
        print("\n" + "=" * 80)
        print("Loading gene vocabulary...")
        print("=" * 80)

        # Priority 1: Try to load from JSON file (preferred method - fast and consistent)
        gene_vocab_json_path = Path(args.gene_vocab_json) if args.gene_vocab_json else None

        if gene_vocab_json_path and gene_vocab_json_path.exists():
            print(f"  Loading gene vocabulary from JSON: {gene_vocab_json_path}")
            gene_vocab = GeneVocabulary.from_json(gene_vocab_json_path)

            # Determine use_separate_chains based on inference data columns
            if args.v_gene_heavy_column in df.columns and args.v_gene_light_column in df.columns:
                use_separate_chains = True
                print(f"  Using separate heavy/light chain gene columns for inference")
            elif args.v_gene_column is not None and args.v_gene_column in df.columns:
                use_separate_chains = False
                print(f"  Using single v_gene/j_gene columns for inference")

        # Priority 2: Fall back to building from pickle file (deprecated)
        elif args.gene_vocab_data_path:
            print(f"  [DEPRECATED] Building gene vocabulary from pickle: {args.gene_vocab_data_path}")
            print(f"  Consider using --gene_vocab_json for faster loading")
            vocab_df = pd.read_pickle(args.gene_vocab_data_path)
            print(f"  Training data shape: {vocab_df.shape}")

            # Determine which column format to use
            if args.v_gene_column is not None and args.v_gene_column in df.columns:
                # Single v_gene/j_gene column mode
                if args.j_gene_column not in df.columns:
                    raise ValueError(f"J-gene column '{args.j_gene_column}' not found in dataframe. "
                                   f"Available columns: {df.columns.tolist()}")

                # Build vocab from vocab_df (training data or inference data)
                v_col = args.v_gene_column if args.v_gene_column in vocab_df.columns else args.v_gene_column
                j_col = args.j_gene_column if args.j_gene_column in vocab_df.columns else args.j_gene_column
                gene_vocab = GeneVocabulary.from_dataframe(
                    vocab_df,
                    v_gene_col=v_col,
                    j_gene_col=j_col
                )
                print(f"  Mode: Single v_gene/j_gene columns")
                print(f"  Built gene vocabulary with {len(gene_vocab)} genes")
                print(f"  V-gene column: '{args.v_gene_column}'")
                print(f"  J-gene column: '{args.j_gene_column}'")

            elif args.v_gene_heavy_column in df.columns and args.v_gene_light_column in df.columns:
                # Heavy/light chain column mode
                use_separate_chains = True

                # Validate all heavy/light columns exist in inference data
                required_cols = [
                    args.v_gene_heavy_column, args.v_gene_light_column,
                    args.j_gene_heavy_column, args.j_gene_light_column
                ]
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    raise ValueError(f"Missing gene columns in inference data: {missing_cols}. "
                                   f"Available columns: {df.columns.tolist()}")

                # Determine which columns to use for building vocab (may differ between training/inference data)
                vocab_cols = []
                for col in required_cols:
                    if col in vocab_df.columns:
                        vocab_cols.append(col)

                # If training data has different column names, try common alternatives
                if not vocab_cols:
                    # Try v_gene/j_gene as single columns (training data format)
                    if 'v_gene' in vocab_df.columns and 'j_gene' in vocab_df.columns:
                        vocab_cols = ['v_gene', 'j_gene']
                        print(f"  Using training data columns: {vocab_cols}")
                    else:
                        raise ValueError(f"Could not find gene columns in vocabulary data. "
                                       f"Available: {vocab_df.columns.tolist()}")

                # Build vocabulary from vocab_df
                all_genes = set()
                for col in vocab_cols:
                    all_genes.update(vocab_df[col].dropna().unique())
                gene_vocab = GeneVocabulary(genes=sorted(list(all_genes)))

                print(f"  Mode: Separate heavy/light chain columns")
                print(f"  Built gene vocabulary with {len(gene_vocab)} genes")
                print(f"  V-gene columns: '{args.v_gene_heavy_column}', '{args.v_gene_light_column}'")
                print(f"  J-gene columns: '{args.j_gene_heavy_column}', '{args.j_gene_light_column}'")

                # Check for region mask columns (optional)
                if args.region_mask_heavy_column in df.columns and args.region_mask_light_column in df.columns:
                    print(f"  Region mask columns: '{args.region_mask_heavy_column}', '{args.region_mask_light_column}'")
                else:
                    print(f"  Region mask columns: not found (will not use region embeddings)")

            else:
                raise ValueError(
                    f"Could not find gene columns in dataframe.\n"
                    f"Expected either:\n"
                    f"  1. --v_gene_column and --j_gene_column, OR\n"
                    f"  2. Heavy/light columns: {args.v_gene_heavy_column}, {args.v_gene_light_column}, "
                    f"{args.j_gene_heavy_column}, {args.j_gene_light_column}\n"
                    f"Available columns: {df.columns.tolist()}"
                )
        else:
            raise ValueError(
                f"Gene conditioning required but no gene vocabulary source found.\n"
                f"Please provide either:\n"
                f"  1. --gene_vocab_json (preferred): Path to gene_vocabulary.json\n"
                f"  2. --gene_vocab_data_path (deprecated): Path to training data pickle"
            )
    else:
        print("  No models require gene conditioning")

    # Validate all models can be loaded BEFORE starting long inference runs
    print("\n" + "="*80)
    print("VALIDATION PHASE: Checking all models can be loaded...")
    print("="*80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    validation_results = []
    all_valid = True

    # Determine gene_vocab_json path for validation
    gene_vocab_json_for_validation = None
    if gene_vocab is not None and hasattr(args, 'gene_vocab_json') and args.gene_vocab_json:
        gene_vocab_json_path = Path(args.gene_vocab_json)
        if gene_vocab_json_path.exists():
            gene_vocab_json_for_validation = str(gene_vocab_json_path)

    for idx, (config_path, checkpoint_path) in enumerate(models_dict.items(), 1):
        print(f"[{idx}/{len(models_dict)}] Validating: {config_path}")
        success, error_msg, config = validate_model_can_load(
            config_path, checkpoint_path, device,
            gene_vocab=gene_vocab,
            gene_vocab_json=gene_vocab_json_for_validation,
        )

        if success:
            experiment_name = config['logging']['experiment_name']
            use_genes = config['model'].get('use_germline_genes', False)
            use_multihead = config['model'].get('use_multihead_architecture', False)
            print(f"  SUCCESS: Model loads correctly")
            print(f"    - Experiment name: {experiment_name}")
            print(f"    - Gene conditioning: {'enabled' if use_genes else 'disabled'}")
            print(f"    - Multihead architecture: {'enabled' if use_multihead else 'disabled'}")
            print(f"    - Custom token strategy: {config['model'].get('custom_token_strategy', 'lowercase_ngl')}")
            validation_results.append((config_path, True, None))
        else:
            print(f"  ✗ FAILED: {error_msg}")
            validation_results.append((config_path, False, error_msg))
            all_valid = False
        print()

    # Summary
    print("=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    successful = sum(1 for _, success, _ in validation_results if success)
    failed = len(validation_results) - successful

    print(f"Total models: {len(validation_results)}")
    print(f"✓ Successful: {successful}")
    print(f"✗ Failed: {failed}")

    if not all_valid:
        print("\n" + "=" * 80)
        print("VALIDATION FAILED - The following models have errors:")
        print("=" * 80)
        for config_path, success, error_msg in validation_results:
            if not success:
                print(f"\n✗ {config_path}")
                print(f"  Error: {error_msg}")
        print("\n" + "=" * 80)
        print("Please fix the errors above before running inference.")
        print("=" * 80)
        return

    print("\n✓ All models validated successfully! Proceeding with inference...\n")

    # Process each model sequentially
    for config_path, checkpoint_path in models_dict.items():
        df = process_single_model(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            df=df,
            sequence_column=args.sequence_column,
            lowercase_seq_column=args.lowercase_seq_column,
            use_restricted_vocab=args.use_restricted_vocab,
            compute_marginalized=args.compute_marginalized,
            compute_region_conditioned=args.compute_region_conditioned,
            gene_vocab=gene_vocab,
            gene_vocab_json=args.gene_vocab_json if any_model_needs_genes else None,
            v_gene_column=args.v_gene_column,
            j_gene_column=args.j_gene_column,
            use_separate_chains=use_separate_chains,
            v_gene_heavy_column=args.v_gene_heavy_column,
            v_gene_light_column=args.v_gene_light_column,
            j_gene_heavy_column=args.j_gene_heavy_column,
            j_gene_light_column=args.j_gene_light_column,
            region_mask_heavy_column=args.region_mask_heavy_column,
            region_mask_light_column=args.region_mask_light_column,
            batch_size=args.batch_size,
            use_compile=args.compile,
        )

    # Save final results
    print("\n" + "="*80)
    print("Saving final results...")
    print("="*80)

    if args.output_path:
        output_path = Path(args.output_path)
    else:
        # Create output filename with all experiment names
        output_path = data_path.parent / f"{data_path.stem}_with_perplexities.pkl"

    df.to_pickle(output_path)
    print(f"\n✓ All results saved to: {output_path}")
    print(f"✓ Final dataframe shape: {df.shape}")
    print(f"✓ Final columns: {df.columns.tolist()}")

    print("\n" + "="*80)
    print("All models processed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
