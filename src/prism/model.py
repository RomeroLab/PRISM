#!/usr/bin/env python
# coding: utf-8

"""
PyTorch Lightning module for ESM2 supervised fine-tuning.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import random
from transformers import AutoModelForMaskedLM, AutoTokenizer, get_cosine_schedule_with_warmup, AutoConfig
from tqdm.auto import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SwiGLU(nn.Module):
    """
    SwiGLU activation function from "GLU Variants Improve Transformer" (https://arxiv.org/abs/2002.05202)

    SwiGLU(x) = Swish(xW) ⊗ xV = (x * sigmoid(x * W)) ⊗ (x * V)
    where ⊗ is element-wise multiplication

    This replaces the standard FFN activation in transformers:
    FFN(x) = GELU(xW1)W2  →  FFN(x) = SwiGLU(xW1)W2
    """
    def __init__(self, dim_in, dim_out, bias=True):
        super().__init__()
        # SwiGLU requires two linear projections (gate and value)
        self.w = nn.Linear(dim_in, dim_out, bias=bias)
        self.v = nn.Linear(dim_in, dim_out, bias=bias)

    def forward(self, x):
        # Swish activation: x * sigmoid(x)
        swish_gate = F.silu(self.w(x))  # F.silu is Swish/SiLU activation
        return swish_gate * self.v(x)


class SFT_ESM2(pl.LightningModule):
    """PyTorch Lightning module for supervised fine-tuning of ESM2 models"""

    def __init__(self,
        seed,
        model_identifier,
        peak_learning_rate,
        adam_beta1,
        adam_beta2,
        adam_epsilon,
        WD,
        warmup_steps,
        max_steps,
        logging_steps,
        eval_steps,
        batch_size,
        mask_prob,
        num_unfrozen_transformer_blocks,
        loss_type,
        random_weights = False,
        add_custom_tokens = False,
        custom_token_strategy = "mask_tokens",  # "mask_tokens", "lowercase_ngl", or "hybrid_lowercase"
        activation_function = "gelu",
        fix_swiglu_double_activation = True,  # If True, monkey-patch to skip GELU after SwiGLU (pure SwiGLU)
        # [NEW] V/J Gene Conditioning
        use_germline_genes = False,
        num_genes = 0,
        gene_embedding_dim = 64,
        gene_embedding_dropout = 0.1,
        # [NEW] Region Embedding
        use_region_embedding = False,
        num_regions = 8,  # [NEW v24] Number of regions (FR1, CDR1, FR2, CDR2, FR3, CDR3, FR4, padding)
        region_embedding_dim = 32,  # [NEW v24] Region embedding dimension
        # [FIX] Accept pre-configured tokenizer
        tokenizer = None,
        # [NEW] Asymmetric Input/Output Strategy - Weight Tying Control
        tie_word_embeddings = True,
        # [NEW] Multihead Architecture (AA + Mutation)
        use_multihead_architecture = False,
        aa_loss_weight = 1.0,
        mut_loss_weight = 5.0,
        mut_focal_gamma = 2.0,
        # [NEW v6.0] Region-aware Alpha Gating
        use_alpha_gating = False,
        fixed_alpha_value = None,  # [NEW] If set (e.g., 1.0), bypass alpha_head and use constant alpha
        final_loss_weight = 1.0,
        aa_focal_gamma = 2.0,
        origin_focal_gamma = 2.0,
        # [NEW v7.0] NGL Loss Reweighting
        # [CHANGE v17] Default reduced from 50.0 to 10.0 for stability
        ngl_loss_alpha = 10.0,  # Weight multiplier for NGL tokens in loss (default 10.0)
        # [NEW v13.0] Multiplicative Gating
        use_multiplicative_gating = False,  # If True, use log-prob summation; if False, use additive mixing (v6)
        gating_temperature = 1.0,  # Temperature for sharpening (only used when use_multiplicative_gating=True)
        # [CHANGE v17] Gating temperature warmup steps
        gating_temperature_warmup_steps = 0,  # Steps before applying temperature (0 = no warmup)
        # [NEW v16.0] Sequential Detach Architecture
        # [CHANGE v17] Default set to True for gradient isolation
        detach_origin_gradient = True,  # If True, detach origin_logits to prevent AA loss from corrupting Origin Head
        # [CHANGE v17] Origin head dropout for regularization
        origin_head_dropout = 0.1,  # Dropout rate for Origin Head to prevent extreme logits
        # [NEW v18] Soft AA Learning - Allow NGL positions in AA loss with reduced weight
        aa_loss_ngl_weight = 0.0,  # Weight for NGL positions in AA loss (0.0 = exclude NGL, >0 = soft learning)
        # [NEW v25] Region-Balanced Loss - Equalize FR and CDR contribution to loss
        use_region_balanced_loss = False,  # If True, balance loss contribution from FR and CDR regions
        # [NEW v28] CDR-Targeted Loss Boosting
        use_cdr_loss_boosting = False,  # If True, apply additional loss multiplier to CDR positions
        cdr_loss_multiplier = 3.0,  # Multiplier for CDR positions (default 3.0)
        # [NEW v35/v36] Dual AA Heads - separate GL and NGL amino acid distributions
        use_dual_aa_heads = False,      # If True, create separate GL and NGL AA heads
        dual_aa_heads_conditioned = True,  # v35=True (both heads see origin conditioning), v36=False (raw hidden states)
        # [NEW v35.1] Separate weight for NGL AA head loss (default: same as aa_loss_weight)
        ngl_aa_loss_weight = None,
        # [NEW v35.1] Detach head outputs from final loss to eliminate gradient competition
        detach_heads_from_final_loss = False,
        # [NEW v35.1b] Label smoothing for NGL AA head to prevent overconfidence
        ngl_label_smoothing = 0.0,
        # [NEW v37] GL-NGL Divergence Loss
        divergence_loss_weight = 0.0,      # Weight for divergence loss (0.0 = disabled)
        divergence_warmup_steps = 2000,    # Steps before enabling divergence loss
        max_kl_divergence = 10.0,          # Clamp per-position KL to this max
        divergence_type = "kl",            # "kl" or "js"
        # [NEW v37] SHM-Based Sample Weighting
        use_shm_weighting = False,         # If True, upweight loss for high-SHM batches
        shm_beta = 1.0,                    # Strength of SHM upweighting
        shm_mean_ngl = 15.0,              # Expected mean NGL count per sequence (normalization)
        # [NEW v38] 3-Class Origin Head (GL / SynNGL / NGL)
        num_origin_classes = 2,            # 2=binary (backward compat), 3=3-class with SynNGL
        synth_weight = 0.5,               # SynNGL mixing weight: P_eff_NGL = P(NGL) + sw*P(SynNGL)
        origin_class_weights = None,       # Per-class focal loss weights [GL, SynNGL, NGL] (None=uniform)
        # [NEW v40] SynNGL Auxiliary Signals
        use_synth_divergence = False,      # If True, add divergence loss at SynNGL positions
        synth_div_weight = 0.3,            # Weight for SynNGL divergence relative to NGL divergence
        use_mpnn_gl_weighting = False,     # If True, weight GL head loss by MPNN P(germline)
        mpnn_min_weight = 0.3,             # Min weight for MPNN GL weighting (clamp floor)
        ):
        super().__init__()

        # log hyperparameters to file
        self.save_hyperparameters(ignore=['tokenizer'])  # Don't save tokenizer object

        # [NEW] Store tie_word_embeddings setting
        self.tie_word_embeddings = tie_word_embeddings

        # [NEW v38] Store 3-class origin head settings
        self.num_origin_classes = num_origin_classes
        self.synth_weight = synth_weight
        if origin_class_weights is not None:
            self.register_buffer(
                'origin_class_weights',
                torch.tensor(origin_class_weights, dtype=torch.float32),
                persistent=False
            )
        else:
            self.origin_class_weights = None

        # fix seeds for reproducibility
        self.seed = seed
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        # model and tokenizer
        self.model_identifier = model_identifier
        self.add_custom_tokens = add_custom_tokens
        self.custom_token_strategy = custom_token_strategy

        # [FIX] Use pre-configured tokenizer if provided, otherwise load fresh
        # When tokenizer is provided from train_esm.py, custom tokens are already added.
        # This ensures consistency between DataModule and Model tokenizers.
        if tokenizer is not None:
            self.tokenizer = tokenizer
            print(f"[SFT_ESM2] Using pre-configured tokenizer")
            print(f"[SFT_ESM2] Vocabulary size: {len(self.tokenizer)}")

            # Still need to set up internal mappings based on strategy
            if self.add_custom_tokens and self.custom_token_strategy == "lowercase_ngl":
                    lowercase_aa = ['a', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l',
                                   'm', 'n', 'p', 'q', 'r', 's', 't', 'v', 'w', 'y']
                    self.lowercase_aa_token_ids = {
                        self.tokenizer.convert_tokens_to_ids(aa.upper()): self.tokenizer.convert_tokens_to_ids(aa.lower())
                        for aa in lowercase_aa
                    }
                    print(f"[SFT_ESM2] Created mapping for {len(self.lowercase_aa_token_ids)} amino acid pairs")
                    self.germ_mask_token_id = None
                    self.nongerm_mask_token_id = None

            else:
                self.germ_mask_token_id = None
                self.nongerm_mask_token_id = None
                self.lowercase_aa_token_ids = None
        else:
            # Legacy path: Load tokenizer and add tokens internally
            # This is kept for backward compatibility but not recommended
            print(f"[SFT_ESM2] WARNING: Loading tokenizer internally (legacy mode)")
            self.tokenizer = AutoTokenizer.from_pretrained(f"facebook/{self.model_identifier}")

            # Conditionally add custom tokens based on strategy
            if self.add_custom_tokens and self.custom_token_strategy == "lowercase_ngl":
                    # New strategy: Add lowercase amino acids for NGL positions
                    lowercase_aa = ['a', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l',
                                   'm', 'n', 'p', 'q', 'r', 's', 't', 'v', 'w', 'y']
                    num_added_tokens = self.tokenizer.add_tokens(lowercase_aa)
                    print(f"Added {num_added_tokens} lowercase amino acid tokens to tokenizer")
                    print(f"New vocabulary size: {len(self.tokenizer)}")

                    # Store mapping of uppercase to lowercase token IDs
                    self.lowercase_aa_token_ids = {
                        self.tokenizer.convert_tokens_to_ids(aa.upper()): self.tokenizer.convert_tokens_to_ids(aa.lower())
                        for aa in lowercase_aa
                    }
                    print(f"Created mapping for {len(self.lowercase_aa_token_ids)} amino acid pairs")
                    self.germ_mask_token_id = None
                    self.nongerm_mask_token_id = None

            else:
                print(f"Using standard ESM2 tokenizer without custom tokens")
                print(f"Vocabulary size: {len(self.tokenizer)}")
                self.germ_mask_token_id = None
                self.nongerm_mask_token_id = None
                self.lowercase_aa_token_ids = None
        
        # [NEW] Define 20 standard AA indices (assuming standard ESM vocab)
        # ID 4 (L) to 23 (C)
        # Note: Don't specify device here - will be moved to correct device automatically
        # persistent=False means it won't be saved in checkpoints, and we'll initialize it on load
        self.register_buffer('aa_indices', torch.arange(4, 24), persistent=False)


        if random_weights == True:
            print("Initializing model with RANDOM weights (training from scratch)")
            cfg   = AutoConfig.from_pretrained(f"facebook/{model_identifier}")
            cfg.vocab_size = len(self.tokenizer)  # Update config vocab size
            # [NEW] Set tie_word_embeddings in config before model creation
            cfg.tie_word_embeddings = self.tie_word_embeddings
            base  = AutoModelForMaskedLM.from_config(cfg)   # random weights
            num_layers = base.config.num_hidden_layers

            # [NEW] Smart Initialization for decoupled LM Head (Asymmetric Input/Output Strategy)
            if not self.tie_word_embeddings and self.add_custom_tokens and self.custom_token_strategy == "lowercase_ngl":
                print(f"[SFT_ESM2] Asymmetric I/O: tie_word_embeddings=False, performing Smart Initialization...")
                self._smart_initialize_lm_head(base)

            # Replace GELU with SwiGLU if requested (before LoRA)
            if activation_function.lower() == "swiglu":
                print(f"Replacing GELU activation with SwiGLU in transformer blocks...")
                self._replace_gelu_with_swiglu(base, num_layers, fix_double_activation=fix_swiglu_double_activation)
                print(f"✓ Successfully replaced activation functions")
            elif activation_function.lower() == "gelu":
                print(f"Using default GELU activation")
            else:
                raise ValueError(f"Unknown activation function: {activation_function}. Must be 'gelu' or 'swiglu'")

            # With random weights, freeze layers according to num_unfrozen_transformer_blocks
            # First, freeze everything
            for p in base.parameters():
                p.requires_grad = False

            # Then unfreeze only the last num_unfrozen_transformer_blocks layers + LM head + embeddings if needed
            for name, p in base.named_parameters():
                if name.startswith("lm_head") \
                or name == "esm.encoder.emb_layer_norm_after" \
                or (add_custom_tokens and name.startswith("esm.embeddings")) \
                or any(name.startswith(f"esm.encoder.layer.{i}")
                       for i in range(num_layers - num_unfrozen_transformer_blocks, num_layers)):
                    p.requires_grad = True

            print(f"Frozen first {num_layers - num_unfrozen_transformer_blocks} layers")
            print(f"Trainable: last {num_unfrozen_transformer_blocks} transformer blocks + LM head + layer norm")
            if add_custom_tokens:
                print(f"           + embeddings (for custom tokens)")

            self.ESM2 = base

        else:
            base = AutoModelForMaskedLM.from_pretrained(f"facebook/{model_identifier}")
            num_layers = base.config.num_hidden_layers

            # [NEW] Update tie_word_embeddings in loaded model config
            if not self.tie_word_embeddings:
                base.config.tie_word_embeddings = False
                # Untie the weights if they were tied
                if hasattr(base, 'tie_weights'):
                    # For ESM2, we need to manually untie the weights by creating a separate LM head
                    # The decoder weight needs to be a separate copy
                    if hasattr(base, 'lm_head') and hasattr(base.lm_head, 'decoder'):
                        # Check if decoder weight is tied to embeddings
                        if base.lm_head.decoder.weight is base.esm.embeddings.word_embeddings.weight:
                            # Create a separate copy of the weights
                            old_weight = base.lm_head.decoder.weight.data.clone()
                            base.lm_head.decoder.weight = nn.Parameter(old_weight)
                            print(f"[SFT_ESM2] Untied LM head decoder weights from embeddings")

            # Resize token embeddings if custom tokens are added
            if self.add_custom_tokens:
                old_vocab_size = base.config.vocab_size
                num_added_tokens = len(self.tokenizer) - old_vocab_size
                base.resize_token_embeddings(len(self.tokenizer))

                # Also need to manually resize the LM head bias which isn't handled by resize_token_embeddings
                # ESM2 has a separate bias parameter in the lm_head
                if hasattr(base, 'lm_head') and hasattr(base.lm_head, 'bias'):
                    old_bias = base.lm_head.bias.data
                    new_bias = torch.zeros(len(self.tokenizer), dtype=old_bias.dtype, device=old_bias.device)
                    new_bias[:old_vocab_size] = old_bias
                    base.lm_head.bias = nn.Parameter(new_bias)

                print(f"Resized model embeddings from {old_vocab_size} to {len(self.tokenizer)}")
                print(f"Note: New token embeddings ({num_added_tokens} tokens) are randomly initialized")
                print(f"      Existing token embeddings ({old_vocab_size} tokens) are preserved from pretrained weights")

                # [NEW] Smart Initialization for decoupled LM Head (Asymmetric Input/Output Strategy)
                if not self.tie_word_embeddings and self.custom_token_strategy == "lowercase_ngl":
                    print(f"[SFT_ESM2] Asymmetric I/O: tie_word_embeddings=False, performing Smart Initialization...")
                    self._smart_initialize_lm_head(base)

            # Replace GELU with SwiGLU if requested (before LoRA)
            if activation_function.lower() == "swiglu":
                print(f"Replacing GELU activation with SwiGLU in transformer blocks...")
                self._replace_gelu_with_swiglu(base, num_layers, fix_double_activation=fix_swiglu_double_activation)
                print(f"✓ Successfully replaced activation functions")
            elif activation_function.lower() == "gelu":
                print(f"Using default GELU activation")
            else:
                raise ValueError(f"Unknown activation function: {activation_function}. Must be 'gelu' or 'swiglu'")

            # Freeze all base parameters first
            for p in base.parameters():
                p.requires_grad = False

            self.ESM2 = base

            # Unfreeze LM head, emb_layer_norm_after, and num_unfrozen_transformer_blocks transformer blocks
            # If custom tokens are added, also unfreeze embeddings so they can be trained
            for name, p in self.ESM2.named_parameters():
                if name.startswith("lm_head") \
                or name == "esm.encoder.emb_layer_norm_after" \
                or (add_custom_tokens and name.startswith("esm.embeddings")) \
                or any(name.startswith(f"esm.encoder.layer.{i}")
                       for i in range(num_layers - num_unfrozen_transformer_blocks, num_layers)):
                    p.requires_grad = True

            print(f"Frozen first {num_layers - num_unfrozen_transformer_blocks} layers")
            print(f"Trainable: last {num_unfrozen_transformer_blocks} transformer blocks + LM head + layer norm")
            if add_custom_tokens:
                print(f"           + embeddings (for custom tokens)")

        # hyperparameters
        self.peak_learning_rate              = peak_learning_rate
        self.adam_beta1                      = adam_beta1
        self.adam_beta2                      = adam_beta2
        self.adam_epsilon                    = adam_epsilon
        self.WD                              = WD
        self.warmup_steps                    = warmup_steps
        self.max_steps                       = max_steps
        self.logging_steps                   = logging_steps
        self.eval_steps                      = eval_steps
        self.batch_size                      = batch_size
        self.mask_prob                       = mask_prob
        self.num_unfrozen_transformer_blocks = num_unfrozen_transformer_blocks
        self.loss_type = loss_type

        # GL/NGL classification auxiliary task
        # Create uppercase and lowercase amino acid token ID tensors for logit-based GL/NGL discrimination
        # These will be used to extract and sum logits from the LM head
        # NOTE: We use different names (gl_token_ids, ngl_token_ids) to avoid conflict with lowercase_aa_token_ids dict
        if self.add_custom_tokens and self.custom_token_strategy == "lowercase_ngl":
            # Map each uppercase AA to its lowercase counterpart
            uppercase_aa = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                           'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
            lowercase_aa = [aa.lower() for aa in uppercase_aa]

            # Get token IDs for uppercase (GL) and lowercase (NGL) amino acids
            uppercase_token_ids = [self.tokenizer.convert_tokens_to_ids(aa) for aa in uppercase_aa]
            lowercase_token_ids = [self.tokenizer.convert_tokens_to_ids(aa) for aa in lowercase_aa]

            # Register as buffers (will be moved to device automatically)
            # Use gl_token_ids and ngl_token_ids to avoid name conflict with lowercase_aa_token_ids dict
            self.register_buffer('gl_token_ids', torch.tensor(uppercase_token_ids, dtype=torch.long), persistent=False)
            self.register_buffer('ngl_token_ids', torch.tensor(lowercase_token_ids, dtype=torch.long), persistent=False)

            print(f"✓ Registered uppercase/lowercase AA token ID buffers for logit-based GL/NGL discrimination")
            print(f"  Uppercase (GL) token IDs: {uppercase_token_ids}")
            print(f"  Lowercase (NGL) token IDs: {lowercase_token_ids}")
        else:
            self.gl_token_ids = None
            self.ngl_token_ids = None

        # =========================================================================
        # [NEW] V/J Gene Conditioning Embeddings
        # =========================================================================
        self.use_germline_genes = use_germline_genes
        self.num_genes = num_genes
        self.gene_embedding_dim = gene_embedding_dim

        if self.use_germline_genes and num_genes > 0:
            # Get hidden size from model config
            if hasattr(self.ESM2, 'config'):
                hidden_size = self.ESM2.config.hidden_size
            elif hasattr(self.ESM2, 'base_model') and hasattr(self.ESM2.base_model, 'config'):
                hidden_size = self.ESM2.base_model.config.hidden_size
            else:
                raise ValueError("Could not determine hidden size from model")

            # V-gene and J-gene embeddings
            self.v_gene_embedding = nn.Embedding(num_genes, gene_embedding_dim)
            self.j_gene_embedding = nn.Embedding(num_genes, gene_embedding_dim)

            # Project concatenated [V; J] embeddings to hidden_size
            self.gene_projection = nn.Linear(gene_embedding_dim * 2, hidden_size)

            # Dropout for gene embeddings
            self.gene_dropout = nn.Dropout(gene_embedding_dropout)

            # Initialize embeddings with small random values
            nn.init.normal_(self.v_gene_embedding.weight, mean=0.0, std=0.02)
            nn.init.normal_(self.j_gene_embedding.weight, mean=0.0, std=0.02)
            nn.init.xavier_uniform_(self.gene_projection.weight, gain=0.1)
            nn.init.zeros_(self.gene_projection.bias)

            print(f"  ✓ Gene Conditioning: ENABLED")
            print(f"    Number of genes: {num_genes}")
            print(f"    Gene embedding dim: {gene_embedding_dim}")
            print(f"    Projection: {gene_embedding_dim * 2} -> {hidden_size}")
            print(f"    Dropout: {gene_embedding_dropout}")
        else:
            self.v_gene_embedding = None
            self.j_gene_embedding = None
            self.gene_projection = None
            self.gene_dropout = None

        # =========================================================================
        # [NEW] Region Embedding
        # =========================================================================
        self.use_region_embedding = use_region_embedding
        self.num_regions = num_regions
        self.region_embedding_dim = region_embedding_dim

        if self.use_region_embedding and num_regions > 0:
            # Get hidden size from model config
            if hasattr(self.ESM2, 'config'):
                hidden_size = self.ESM2.config.hidden_size
            elif hasattr(self.ESM2, 'base_model') and hasattr(self.ESM2.base_model, 'config'):
                hidden_size = self.ESM2.base_model.config.hidden_size
            else:
                raise ValueError("Could not determine hidden size from model")

            # Region embedding: maps region_id (0=pad, 1=FR1, 2=CDR1, ..., 7=FR4) to embedding
            self.region_embedding = nn.Embedding(num_regions, region_embedding_dim)
            self.region_projection = nn.Linear(region_embedding_dim, hidden_size)
            self.region_dropout = nn.Dropout(0.1)

            # Initialize with small values for stable training
            nn.init.normal_(self.region_embedding.weight, mean=0.0, std=0.02)
            nn.init.xavier_uniform_(self.region_projection.weight, gain=0.1)
            nn.init.zeros_(self.region_projection.bias)

            print(f"  ✓ Region Embedding: ENABLED")
            print(f"    Number of regions: {num_regions}")
            print(f"    Region embedding dim: {region_embedding_dim}")
            print(f"    Projection: {region_embedding_dim} -> {hidden_size}")
        else:
            self.region_embedding = None
            self.region_projection = None
            self.region_dropout = None

        # =========================================================================

        # PPL-specific batch size
        self.ppl_batch_size = 1024

        # [NEW for v2.0] Lists to store step outputs
        self.validation_step_outputs = []
        self.test_step_outputs = []

        # parameters for custom training
        self.automatic_optimization = True

        # =========================================================================
        # [NEW] Multihead Architecture: AA Head + Mutation Head + Alpha Gating
        # =========================================================================
        self.use_multihead_architecture = use_multihead_architecture
        self.aa_loss_weight = aa_loss_weight
        self.mut_loss_weight = mut_loss_weight
        self.mut_focal_gamma = mut_focal_gamma
        # [NEW v6.0] Alpha gating parameters
        self.use_alpha_gating = use_alpha_gating
        self.fixed_alpha_value = fixed_alpha_value
        self.final_loss_weight = final_loss_weight
        self.aa_focal_gamma = aa_focal_gamma
        self.origin_focal_gamma = origin_focal_gamma
        # [NEW v7.0] NGL Loss Reweighting
        self.ngl_loss_alpha = ngl_loss_alpha
        # [NEW v13.0] Multiplicative Gating
        self.use_multiplicative_gating = use_multiplicative_gating
        self.gating_temperature = gating_temperature
        # [CHANGE v17] Gating temperature warmup
        self.gating_temperature_warmup_steps = gating_temperature_warmup_steps
        # [NEW v16.0] Sequential Detach Architecture
        self.detach_origin_gradient = detach_origin_gradient
        # [CHANGE v17] Origin head dropout for regularization
        self.origin_head_dropout_rate = origin_head_dropout
        # [NEW v18] Soft AA Learning - Allow NGL positions in AA loss with reduced weight
        self.aa_loss_ngl_weight = aa_loss_ngl_weight
        # [NEW v25] Region-Balanced Loss - Equalize FR and CDR contribution to loss
        self.use_region_balanced_loss = use_region_balanced_loss
        # [NEW v28] CDR-Targeted Loss Boosting
        self.use_cdr_loss_boosting = use_cdr_loss_boosting
        self.cdr_loss_multiplier = cdr_loss_multiplier
        # [NEW v35/v36] Dual AA Heads
        self.use_dual_aa_heads = use_dual_aa_heads
        self.dual_aa_heads_conditioned = dual_aa_heads_conditioned
        # [NEW v35.1] NGL AA head loss weight (defaults to aa_loss_weight if not set)
        self.ngl_aa_loss_weight = ngl_aa_loss_weight if ngl_aa_loss_weight is not None else aa_loss_weight
        # [NEW v35.1] Detach head outputs from final loss
        self.detach_heads_from_final_loss = detach_heads_from_final_loss
        # [NEW v35.1b] Label smoothing for NGL AA head
        self.ngl_label_smoothing = ngl_label_smoothing

        # [NEW v37] GL-NGL Divergence Loss
        self.divergence_loss_weight = divergence_loss_weight
        self.divergence_warmup_steps = divergence_warmup_steps
        self.max_kl_divergence = max_kl_divergence
        self.divergence_type = divergence_type

        # [NEW v37] SHM-Based Sample Weighting
        self.use_shm_weighting = use_shm_weighting
        self.shm_beta = shm_beta
        self.shm_mean_ngl = shm_mean_ngl

        # [NEW v40] SynNGL Auxiliary Signals
        self.use_synth_divergence = use_synth_divergence
        self.synth_div_weight = synth_div_weight
        self.use_mpnn_gl_weighting = use_mpnn_gl_weighting
        self.mpnn_min_weight = mpnn_min_weight

        # [NEW v38.4] Trust Head for PRISM

        if self.use_multihead_architecture:
            print(f"\n{'='*60}")
            print(f"MULTIHEAD ARCHITECTURE: AA + Mutation Dual-Head Model")
            print(f"{'='*60}")

            # Get hidden size and layer norm eps from model config
            if hasattr(self.ESM2, 'config'):
                hidden_size = self.ESM2.config.hidden_size
                vocab_size = self.ESM2.config.vocab_size
                layer_norm_eps = getattr(self.ESM2.config, 'layer_norm_eps', 1e-5)
            elif hasattr(self.ESM2, 'base_model') and hasattr(self.ESM2.base_model, 'config'):
                hidden_size = self.ESM2.base_model.config.hidden_size
                vocab_size = self.ESM2.base_model.config.vocab_size
                layer_norm_eps = getattr(self.ESM2.base_model.config, 'layer_norm_eps', 1e-5)
            else:
                raise ValueError("Could not determine hidden size from model")

            # -----------------------------------------------------------------
            # Head 1: Amino Acid Identity Head (ESM2 LMHead architecture)
            # Structure: Dense → GELU → LayerNorm → Decoder
            # This matches the original ESM2 lm_head for proper expressiveness
            # -----------------------------------------------------------------
            self.aa_head_dense = nn.Linear(hidden_size, hidden_size)
            self.aa_head_layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
            self.aa_head_decoder = nn.Linear(hidden_size, vocab_size, bias=False)
            self.aa_head_bias = nn.Parameter(torch.zeros(vocab_size))

            # Smart Initialization: Copy weights from original ESM2 lm_head
            print(f"  Initializing AA Head from ESM2 lm_head...")
            self._smart_initialize_aa_head()

            print(f"  ✓ AA Head (ESM2 LMHead structure):")
            print(f"      Dense: Linear({hidden_size} -> {hidden_size})")
            print(f"      Activation: GELU")
            print(f"      LayerNorm: ({hidden_size}, eps={layer_norm_eps})")
            print(f"      Decoder: Linear({hidden_size} -> {vocab_size})")
            print(f"    Loss weight: {aa_loss_weight}")

            # -----------------------------------------------------------------
            # Head 2: Mutation State Head (Origin Head)
            # [v38] Supports 2-class (binary GL/NGL) or 3-class (GL/SynNGL/NGL)
            # [CHANGE v17] Added dropout for regularization to prevent extreme logits
            # -----------------------------------------------------------------
            # [v38] Binary (1 output + sigmoid) or 3-class (3 outputs + softmax)
            mut_head_out = self.num_origin_classes if self.num_origin_classes >= 3 else 1
            self.mut_head = nn.Linear(hidden_size, mut_head_out, bias=True)

            # [CHANGE v17] Origin head dropout to prevent extreme logit values
            self.mut_dropout = nn.Dropout(self.origin_head_dropout_rate)

            # Initialize with small values for stable training
            nn.init.xavier_uniform_(self.mut_head.weight, gain=0.1)
            nn.init.zeros_(self.mut_head.bias)

            print(f"  ✓ Mutation Head (Origin Head): Linear({hidden_size} -> {mut_head_out})")
            print(f"    [v17] Origin dropout: {self.origin_head_dropout_rate}")
            if self.num_origin_classes == 3:
                print(f"    [v38] 3-class mode: GL(0) / SynNGL(1) / NGL(2)")
                print(f"    [v38] synth_weight: {self.synth_weight}")
                if self.origin_class_weights is not None:
                    print(f"    [v38] class_weights: {self.origin_class_weights.tolist()}")
            print(f"    Loss weight: {mut_loss_weight}")
            print(f"    Focal gamma: {mut_focal_gamma}")

            # -----------------------------------------------------------------
            # [NEW v14/v16] Origin Projection for Sequential Conditional Architecture
            # Projects origin_logits back to hidden_size to condition AA Head
            # This allows the AA Head to "see" the mutation prediction
            # [v16] With detach_origin_gradient=True, origin_logits are detached
            #       before projection, preventing AA loss from affecting Origin Head
            # -----------------------------------------------------------------
            origin_proj_in = self.num_origin_classes if self.num_origin_classes >= 3 else 1
            self.origin_projection = nn.Linear(origin_proj_in, hidden_size, bias=True)

            # Initialize with small values so conditioning starts subtle
            nn.init.xavier_uniform_(self.origin_projection.weight, gain=0.1)
            nn.init.zeros_(self.origin_projection.bias)

            print(f"  ✓ Origin Projection (Sequential Conditioning): Linear({origin_proj_in} -> {hidden_size})")
            print(f"    Enables AA Head to condition on Origin Head predictions")
            print(f"    [v16] Gradient Detach: {detach_origin_gradient}")

            # -----------------------------------------------------------------
            # [NEW v6.0] Head 3: Alpha Head (Region-aware Gating)
            # Outputs per-position alpha values [0, 1] for combining AA and Origin logits
            # alpha_head: Hidden → Linear → Sigmoid → [B, L, 1]
            # -----------------------------------------------------------------
            if self.use_alpha_gating:
                self.alpha_head = nn.Sequential(
                    nn.Linear(hidden_size, 1, bias=True),
                    nn.Sigmoid()
                )
                # Initialize with small values so alpha starts near 0.5
                nn.init.xavier_uniform_(self.alpha_head[0].weight, gain=0.1)
                nn.init.zeros_(self.alpha_head[0].bias)

                print(f"  ✓ Alpha Head (Region-aware Gating): Linear({hidden_size} -> 1) + Sigmoid")
                print(f"    Output range: [0, 1] per position")
                print(f"    Final (53 vocab) Loss weight: {final_loss_weight}")
                print(f"    AA Focal gamma: {aa_focal_gamma}")
                print(f"    Origin Focal gamma: {origin_focal_gamma}")
            else:
                self.alpha_head = None

            # [NEW v28] Log CDR loss boosting settings
            if self.use_cdr_loss_boosting:
                print(f"  ✓ CDR Loss Boosting: ENABLED (multiplier={self.cdr_loss_multiplier}x)")
                print(f"    Combined effect: CDR-NGL = {self.ngl_loss_alpha * self.cdr_loss_multiplier}x, CDR-GL = {self.cdr_loss_multiplier}x")
                print(f"                     FR-NGL = {self.ngl_loss_alpha}x, FR-GL = 1.0x")

            # -----------------------------------------------------------------
            # [NEW v35/v36] NGL AA Head - Separate amino acid distribution for NGL positions
            # v35 (conditioned=True): Both heads receive origin-conditioned hidden states
            # v36 (conditioned=False): Both heads receive raw hidden states (no origin info)
            # -----------------------------------------------------------------
            if self.use_dual_aa_heads:
                self.ngl_aa_head_dense = nn.Linear(hidden_size, hidden_size)
                self.ngl_aa_head_layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
                self.ngl_aa_head_decoder = nn.Linear(hidden_size, vocab_size, bias=False)
                self.ngl_aa_head_bias = nn.Parameter(torch.zeros(vocab_size))

                # Smart Initialization: Copy weights from original ESM2 lm_head
                print(f"  Initializing NGL AA Head from ESM2 lm_head...")
                self._smart_initialize_ngl_aa_head()

                mode_str = "conditioned (v35)" if self.dual_aa_heads_conditioned else "raw (v36)"
                print(f"  ✓ NGL AA Head (ESM2 LMHead structure, {mode_str}):")
                print(f"      Dense: Linear({hidden_size} -> {hidden_size})")
                print(f"      Activation: GELU")
                print(f"      LayerNorm: ({hidden_size}, eps={layer_norm_eps})")
                print(f"      Decoder: Linear({hidden_size} -> {vocab_size})")
            else:
                self.ngl_aa_head_dense = None
                self.ngl_aa_head_layer_norm = None
                self.ngl_aa_head_decoder = None
                self.ngl_aa_head_bias = None

            print(f"{'='*60}\n")
        else:
            self.aa_head_dense = None
            self.aa_head_layer_norm = None
            self.aa_head_decoder = None
            self.aa_head_bias = None
            self.mut_head = None
            self.origin_projection = None
            self.alpha_head = None
            # [NEW v38] PRISM heads
            self.gl_head_dense = None
            self.gl_head_layer_norm = None
            self.gl_head_decoder = None
            self.gl_head_bias = None
            # [NEW v38.4] Trust head
            self.trust_head = None
            # [NEW v35/v36] NGL AA head
            self.ngl_aa_head_dense = None
            self.ngl_aa_head_layer_norm = None
            self.ngl_aa_head_decoder = None
            self.ngl_aa_head_bias = None

    def on_load_checkpoint(self, checkpoint):
        """
        Hook called when loading a checkpoint.
        Ensures aa_indices buffer is always initialized, even for old checkpoints.
        Also removes aa_indices from state_dict if present (for backward compatibility).
        [v38] Auto-reshapes mut_head and origin_projection when loading 2-class into 3-class.
        """
        # Remove aa_indices from state_dict if present (it's now non-persistent)
        if 'state_dict' in checkpoint and 'aa_indices' in checkpoint['state_dict']:
            del checkpoint['state_dict']['aa_indices']

        # Initialize aa_indices if it doesn't exist (backward compatibility)
        if not hasattr(self, 'aa_indices') or self.aa_indices is None:
            self.register_buffer('aa_indices', torch.arange(4, 24), persistent=False)

        # [v38] Auto-reshape mut_head from 2-class (1,H) to 3-class (3,H)
        if 'state_dict' in checkpoint and self.num_origin_classes == 3:
            sd = checkpoint['state_dict']

            # mut_head.weight: (1, H) → (3, H) by repeating
            if 'mut_head.weight' in sd and sd['mut_head.weight'].shape[0] == 1:
                old_w = sd['mut_head.weight']  # (1, H)
                H = old_w.shape[1]
                new_w = torch.zeros(3, H, dtype=old_w.dtype, device=old_w.device)
                new_w[0] = -old_w[0]  # GL ≈ -logit (sigmoid(-x) maps to GL)
                new_w[1] = torch.zeros(H, dtype=old_w.dtype)  # SynNGL: zero-init
                new_w[2] = old_w[0]   # NGL ≈ +logit
                sd['mut_head.weight'] = new_w

            # mut_head.bias: (1,) → (3,) similarly
            if 'mut_head.bias' in sd and sd['mut_head.bias'].shape[0] == 1:
                old_b = sd['mut_head.bias']
                new_b = torch.zeros(3, dtype=old_b.dtype, device=old_b.device)
                new_b[0] = -old_b[0]
                new_b[2] = old_b[0]
                sd['mut_head.bias'] = new_b

            # origin_projection.weight: (H, 1) → (H, 3) by repeating
            if 'origin_projection.weight' in sd and sd['origin_projection.weight'].shape[1] == 1:
                old_w = sd['origin_projection.weight']  # (H, 1)
                H = old_w.shape[0]
                new_w = torch.zeros(H, 3, dtype=old_w.dtype, device=old_w.device)
                new_w[:, 0] = old_w[:, 0]  # GL channel
                new_w[:, 2] = old_w[:, 0]  # NGL channel (same init as GL)
                sd['origin_projection.weight'] = new_w

            # origin_projection.bias stays (H,) — no change needed

    def _get_gene_conditioned_inputs_embeds(self, input_ids, v_gene_ids=None, j_gene_ids=None, region_ids=None):
        """
        Get input embeddings with V/J gene conditioning and region embeddings.

        [v24 UPDATE] This method now includes:
        1. Gets base token embeddings from ESM2
        2. If gene conditioning is enabled, embeds V and J gene IDs
        3. [v24] If region embedding is enabled, adds per-position region embeddings
        4. Projects combined embeddings back to hidden_size
        5. [v24] If position-aware genes are enabled, uses cross-attention instead of broadcast

        Args:
            input_ids: [B, L] tensor of token IDs
            v_gene_ids: [B] tensor of V-gene IDs (optional)
            j_gene_ids: [B] tensor of J-gene IDs (optional)
            region_ids: [B, L] tensor of region IDs (optional) [v24]

        Returns:
            inputs_embeds: [B, L, H] tensor of conditioned embeddings
        """
        # Get base token embeddings from EsmForMaskedLM
        inputs_embeds = self.ESM2.esm.embeddings.word_embeddings(input_ids)

        # =====================================================================
        # Add region embeddings (per-position FR/CDR information)
        # =====================================================================
        if self.use_region_embedding and self.region_embedding is not None and region_ids is not None:
            # Get region embeddings: [B, L] -> [B, L, region_dim]
            region_emb = self.region_embedding(region_ids)

            # Project to hidden size: [B, L, region_dim] -> [B, L, H]
            region_emb = self.region_projection(region_emb)
            region_emb = self.region_dropout(region_emb)

            # Add to inputs_embeds
            inputs_embeds = inputs_embeds + region_emb

        # =====================================================================
        # Add gene conditioning if enabled
        # =====================================================================
        if (self.use_germline_genes and self.v_gene_embedding is not None
            and v_gene_ids is not None and j_gene_ids is not None):

            # Get V and J gene embeddings: [B] -> [B, gene_embedding_dim]
            v_emb = self.v_gene_embedding(v_gene_ids)  # [B, gene_dim]
            j_emb = self.j_gene_embedding(j_gene_ids)  # [B, gene_dim]

            # Concatenate V and J embeddings: [B, gene_dim * 2]
            gene_emb = torch.cat([v_emb, j_emb], dim=-1)  # [B, gene_dim * 2]

            # Project to hidden size: [B, hidden_size]
            gene_emb = self.gene_projection(gene_emb)  # [B, hidden_size]

            # Apply dropout
            gene_emb = self.gene_dropout(gene_emb)  # [B, hidden_size]

            # Broadcast across sequence length: [B, hidden_size] -> [B, L, hidden_size]
            gene_emb = gene_emb.unsqueeze(1)  # [B, 1, hidden_size]

            # Add to inputs_embeds (broadcasted addition)
            inputs_embeds = inputs_embeds + gene_emb  # [B, L, hidden_size]

        return inputs_embeds

    def _forward_with_gene_conditioning(self, input_ids, attention_mask, labels=None,
                                         v_gene_ids=None, j_gene_ids=None, region_ids=None,
                                         output_hidden_states=False):
        """
        Forward pass with optional V/J gene conditioning and region embeddings.

        [v24 UPDATE] Now also supports:
        - Region embeddings (per-position FR/CDR information)
        - Biophysical features (per-position chemical properties)
        - Position-aware gene conditioning (cross-attention)

        Args:
            input_ids: [B, L] token IDs
            attention_mask: [B, L] attention mask
            labels: [B, L] optional labels for MLM loss
            v_gene_ids: [B] optional V-gene IDs
            j_gene_ids: [B] optional J-gene IDs
            region_ids: [B, L] optional region IDs [v24]
            output_hidden_states: Whether to return hidden states

        Returns:
            Model outputs (with logits, loss if labels provided, hidden_states if requested)
        """
        # [v24] Check if we need custom embeddings (gene or region)
        use_custom_embeds = (
            (self.use_germline_genes and v_gene_ids is not None and j_gene_ids is not None) or
            (self.use_region_embedding and region_ids is not None)
        )

        if use_custom_embeds:
            # Get conditioned embeddings (gene + region)
            inputs_embeds = self._get_gene_conditioned_inputs_embeds(input_ids, v_gene_ids, j_gene_ids, region_ids)

            # Forward with inputs_embeds (NOT input_ids).  Older transformers
            # versions (< 4.57) apply the ESM ``token_dropout`` step
            # unconditionally on ``input_ids``, which crashes when we pass
            # ``inputs_embeds`` alone.  Disable it for the duration of the
            # call; we are bypassing the MLM mask-dropout training trick
            # along the conditioned path anyway.
            emb_layer = self.ESM2.esm.embeddings
            prev_token_dropout = emb_layer.token_dropout
            emb_layer.token_dropout = False
            try:
                outputs = self.ESM2(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    labels=labels,
                    output_hidden_states=output_hidden_states
                )
            finally:
                emb_layer.token_dropout = prev_token_dropout
        else:
            # Standard forward with input_ids
            outputs = self.ESM2(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                output_hidden_states=output_hidden_states
            )

        return outputs

    def training_step(self, batch, batch_idx):
        return self._training_step_multihead(batch, batch_idx)

    # [NEW for v2.0] Add epoch_start hooks to clear output lists
    def on_validation_epoch_start(self):
        self.validation_step_outputs.clear()

    def on_test_epoch_start(self):
        self.test_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        return self._validation_step_multihead(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        """
        [MODIFIED for v2.0, v27 - multihead support]
        Test step computes loss on ALL VOCABULARY TOKENS.
        It also saves reconstructed original data for PPL calculation (20 AA constraint)
        in on_test_epoch_end.
        """
        # Unpack batch with optional gene IDs and region IDs
        # Multihead architecture uses different batch format with labels_aa and labels_mut
        v_gene_ids = None
        j_gene_ids = None
        region_ids = None

        # [v40] Extract synth_masks and mpnn_gl_probs if present (not used in test, just discard)
        batch = list(batch)
        if len(batch) > 5 and batch[-1].dim() == 0:
            batch.pop()  # coherence_flag
        if len(batch) >= 7 and batch[-1].dim() == 2 and batch[-2].dim() == 2:
            if batch[-1].dtype == torch.float32 and (batch[-2].dtype == torch.bool or batch[-2].dtype in (torch.int32, torch.int64)):
                batch.pop()  # mpnn_gl_probs
                batch.pop()  # synth_masks

        # Multihead batch format: (input_ids, labels_aa, labels_mut, attn_mask, ngl_masks, ...)
        if self.use_germline_genes and self.use_region_embedding and len(batch) == 8:
            input_ids, labels, _, attn_mask, ngl_masks, v_gene_ids, j_gene_ids, region_ids = batch
            v_gene_ids = v_gene_ids.to(self.device)
            j_gene_ids = j_gene_ids.to(self.device)
            region_ids = region_ids.to(self.device)
        elif self.use_germline_genes and len(batch) == 7:
            input_ids, labels, _, attn_mask, ngl_masks, v_gene_ids, j_gene_ids = batch
            v_gene_ids = v_gene_ids.to(self.device)
            j_gene_ids = j_gene_ids.to(self.device)
        elif self.use_region_embedding and len(batch) == 6:
            input_ids, labels, _, attn_mask, ngl_masks, region_ids = batch
            region_ids = region_ids.to(self.device)
        else:
            input_ids, labels, _, attn_mask, ngl_masks = batch

        input_ids = input_ids.to(self.device)
        labels = labels.to(self.device)
        ngl_masks = ngl_masks.to(self.device)

        # --- 1. 손실(Loss) 계산 (기존과 동일) ---
        if self.loss_type == 'focal_loss':
            outputs = self._forward_with_gene_conditioning(
                input_ids=input_ids,
                attention_mask=attn_mask,
                v_gene_ids=v_gene_ids,
                j_gene_ids=j_gene_ids,
                region_ids=region_ids  # [v24]
            )
            logits = outputs.logits
            test_loss = self._focal_loss(logits, labels, gamma=2.0, ignore_index=-100)
        elif self.loss_type == 'cross_entropy':
            outputs = self._forward_with_gene_conditioning(
                input_ids=input_ids,
                attention_mask=attn_mask,
                labels=labels,
                v_gene_ids=v_gene_ids,
                j_gene_ids=j_gene_ids,
                region_ids=region_ids  # [v24]
            )
            test_loss = outputs.loss
        
        self.log('test_loss', test_loss, prog_bar=False, on_step=False, on_epoch=True, batch_size=input_ids.size(0), sync_dist=True)

        # --- 2. PPL 계산을 위한 원본 데이터 복원 ---
        original_input_ids = input_ids.clone()
        masked_positions = (labels != -100)
        original_input_ids[masked_positions] = labels[masked_positions]

        if self.add_custom_tokens:
            if self.custom_token_strategy == "mask_tokens":
                germ_mask_pos = (input_ids == self.germ_mask_token_id)
                nongerm_mask_pos = (input_ids == self.nongerm_mask_token_id)
                original_input_ids[germ_mask_pos] = labels[germ_mask_pos]
                original_input_ids[nongerm_mask_pos] = labels[nongerm_mask_pos]
            elif self.custom_token_strategy == "lowercase_ngl":
                # For lowercase strategy, replace lowercase tokens with their uppercase equivalents
                # This is needed because lowercase tokens are NGL-specific and we need standard AAs for PPL
                for upper_id, lower_id in self.lowercase_aa_token_ids.items():
                    lowercase_positions = (input_ids == lower_id)
                    original_input_ids[lowercase_positions] = upper_id
        
        # --- 3. 데이터 반환 (CPU로 보내 GPU 메모리 절약) ---
        output_data = {
            'original_input_ids': original_input_ids.cpu(),
            'attn_mask': attn_mask.cpu(),
            'ngl_masks': ngl_masks.cpu(),
            'v_gene_ids': v_gene_ids.cpu() if v_gene_ids is not None else None,
            'j_gene_ids': j_gene_ids.cpu() if j_gene_ids is not None else None,
            'region_ids': region_ids.cpu() if region_ids is not None else None,
        }
        # [NEW for v2.0] Manually append outputs
        self.test_step_outputs.append(output_data)
        return output_data # Return is still needed

    def on_validation_epoch_end(self):
        """
        [MODIFIED]
        No longer calculates perplexity - validation losses are logged directly in validation_step.
        """
        pass

    def on_test_epoch_end(self):
        """
        [MODIFIED for v2.0]
        모든 test_step의 출력을 모아서 PPL을 일괄 계산합니다.
        """
        # [MODIFIED for v2.0] Access internal list
        if not self.test_step_outputs:
            print("No test outputs to process.")
            return

        print(f"\nCollating test data for PPL calculation...")
        # 1. 모든 배치의 출력을 하나로 합침
        all_original_ids = torch.cat([out['original_input_ids'] for out in self.test_step_outputs], dim=0)
        all_attn_masks = torch.cat([out['attn_mask'] for out in self.test_step_outputs], dim=0)
        all_ngl_masks = torch.cat([out['ngl_masks'] for out in self.test_step_outputs], dim=0)

        # [FIX] Collect gene/region IDs for conditioned PPL calculation
        all_v_gene_ids = None
        all_j_gene_ids = None
        all_region_ids = None
        if self.test_step_outputs[0].get('v_gene_ids') is not None:
            all_v_gene_ids = torch.cat([out['v_gene_ids'] for out in self.test_step_outputs], dim=0)
        if self.test_step_outputs[0].get('j_gene_ids') is not None:
            all_j_gene_ids = torch.cat([out['j_gene_ids'] for out in self.test_step_outputs], dim=0)
        if self.test_step_outputs[0].get('region_ids') is not None:
            all_region_ids = torch.cat([out['region_ids'] for out in self.test_step_outputs], dim=0)

        print(f"Total test samples: {all_original_ids.size(0)}. Starting PPL calculation...")

        # 2. PPL 계산 수행 (20-AA Constrained)
        self._run_ppl_calculation_on_full_dataset(
            all_original_ids,
            all_attn_masks,
            all_ngl_masks,
            prefix="test",
            all_v_gene_ids=all_v_gene_ids,
            all_j_gene_ids=all_j_gene_ids,
            all_region_ids=all_region_ids,
        )
        print("Test PPL calculation complete.")
        
    def _run_ppl_calculation_on_full_dataset(self, all_original_ids, all_attn_masks, all_ngl_masks, prefix,
                                                all_v_gene_ids=None, all_j_gene_ids=None, all_region_ids=None):
        """
        Calculate pseudo-perplexity using FULL VOCABULARY.

        Calculation method (Ablang2-style "Average of PPLs"):
        1. Collect log-probs over FULL vocabulary to [N, L] tensor.
        2. For each sequence, calculate average log-prob for each category (heavy_gl, etc.).
        3. For each sequence, calculate PPL (PPL_seq = exp(-avg_log_prob_seq)).
        4. Take final average of PPL values across sequences that have tokens in the category.

        Only amino acid positions (IDs 4-23) are included in the calculation,
        but log probabilities are computed over the FULL vocabulary.
        """

        N, L = all_original_ids.shape
        mask_token_id = self.tokenizer.mask_token_id

        # --- 1. Initialize tensor to store log probabilities ---
        # (Store on CPU to save GPU memory)
        log_probs_tensor = torch.zeros(N, L, dtype=torch.float32, device="cpu")

        # --- 2. Move tensors to GPU for mask creation ---
        all_original_ids_device = all_original_ids.to(self.device)
        all_attn_masks_device = all_attn_masks.to(self.device)
        all_ngl_masks_device = all_ngl_masks.to(self.device)

        # --- 3. Create masks ---
        # 3a. Heavy/Light chain masks (excluding special tokens)
        heavy_chain_mask, light_chain_mask = self._create_chain_masks(
            all_original_ids_device,
            all_attn_masks_device
        ) # (N, L)

        # 3b. Pure 20 AA mask (IDs 4-23)
        # PPL calculation is performed *only* at these positions
        pure_aa_mask = self._create_pure_aa_mask(all_original_ids_device) # (N, L)

        # 3c. Master mask for PPL calculation (pure AA positions only)
        master_ppl_mask = pure_aa_mask

        # [DEBUG] Log mask statistics
        if (not hasattr(self, 'trainer') or self.trainer.is_global_zero):
            print(f"\n[DEBUG] Mask Creation Statistics:")
            print(f"  Total sequences: {N}")
            print(f"  Sequence length: {L}")
            print(f"  Pure AA positions (total): {pure_aa_mask.sum().item()}")
            print(f"  Pure AA positions per seq (first 5): {pure_aa_mask.sum(dim=1)[:5].cpu().numpy().tolist()}")
            print(f"  Heavy chain positions (total): {heavy_chain_mask.sum().item()}")
            print(f"  Light chain positions (total): {light_chain_mask.sum().item()}")
            print(f"  NGL positions (total): {all_ngl_masks_device.sum().item()}")

            # Check first sequence in detail
            print(f"\n  First sequence token IDs (pos 0-50): {all_original_ids_device[0, :50].cpu().numpy().tolist()}")
            print(f"  First sequence pure_aa_mask (pos 0-50): {pure_aa_mask[0, :50].cpu().numpy().astype(int).tolist()}")
            print(f"  First sequence heavy_mask (pos 0-50): {heavy_chain_mask[0, :50].cpu().numpy().astype(int).tolist()}")
            print(f"  First sequence light_mask (pos 0-50): {light_chain_mask[0, :50].cpu().numpy().astype(int).tolist()}")
            print(f"  First sequence ngl_mask (pos 0-50): {all_ngl_masks_device[0, :50].cpu().numpy().astype(int).tolist()}\n")

        # --- 4. Initialize tqdm progress bar ---
        pbar = tqdm(
            range(L), # Iterate from 0 to L-1
            desc=f"PPL (Pos 1...{L}) [Full Vocab]",
            leave=False,
            disable=(hasattr(self, 'trainer') and not self.trainer.is_global_zero)
        )

        # --- 5. Iterate through sequence length L (Collect Log Probabilities) ---
        positions_processed = 0

        # [NEW] Create debug log file
        debug_log_path = None
        if (not hasattr(self, 'trainer') or self.trainer.is_global_zero):
            import os
            debug_log_path = f"debug_ppl_{prefix}.txt"
            with open(debug_log_path, 'w') as f:
                f.write(f"=== PPL Debug Log for {prefix} ===\n")
                f.write(f"Total sequences: {N}, Sequence length: {L}\n")
                f.write(f"Pure AA positions total: {pure_aa_mask.sum().item()}\n\n")

        for i in pbar:

            # 5a. Check if position 'i' has any AA tokens
            if not master_ppl_mask[:, i].any():
                continue

            positions_processed += 1

            # 5b. [KEY STEP] Iterate in chunks of ppl_batch_size (1024)
            for batch_start in range(0, N, self.ppl_batch_size):
                batch_end = min(batch_start + self.ppl_batch_size, N)

                # Current chunk data (CPU -> GPU)
                chunk_ids = all_original_ids[batch_start:batch_end].to(self.device)
                chunk_attn = all_attn_masks[batch_start:batch_end].to(self.device)

                # Ground truth tokens at position i
                true_token_ids_at_i = chunk_ids[:, i].unsqueeze(1) # [B, 1]

                # Create masked input (mask position i)
                masked_chunk_input = chunk_ids.clone()
                masked_chunk_input[:, i] = mask_token_id

                # 5c. Forward pass through model (with gene/region conditioning)
                chunk_v = all_v_gene_ids[batch_start:batch_end].to(self.device) if all_v_gene_ids is not None else None
                chunk_j = all_j_gene_ids[batch_start:batch_end].to(self.device) if all_j_gene_ids is not None else None
                chunk_region = all_region_ids[batch_start:batch_end].to(self.device) if all_region_ids is not None else None

                with torch.no_grad():
                    outputs = self._forward_with_gene_conditioning(
                        input_ids=masked_chunk_input,
                        attention_mask=chunk_attn,
                        v_gene_ids=chunk_v,
                        j_gene_ids=chunk_j,
                        region_ids=chunk_region,
                    )
                    logits = outputs.logits # [B, L, V] where V = vocab size (33+)

                # 5d. Extract logits at masked position 'i'
                logits_at_i = logits[:, i, :] # [B, V] where V is full vocabulary

                # 5e. Calculate log probability over FULL vocabulary
                log_probs_at_i = F.log_softmax(logits_at_i, dim=-1) # [B, V]

                # 5f. Get log prob for the true token (using original token IDs)
                log_prob_true_tokens = log_probs_at_i.gather(1, true_token_ids_at_i).squeeze(1) # [B]

                # 5g. Check if current position is an AA position
                current_pos_aa_mask = pure_aa_mask[batch_start:batch_end, i] # [B]

                # 5h. Mask out non-AA positions (set to 0)
                # This ensures we only accumulate log probs for actual amino acid positions
                log_prob_true_tokens = log_prob_true_tokens * current_pos_aa_mask.float()

                # [NEW] Debug logging for first position and first few samples
                if debug_log_path and positions_processed <= 3 and batch_start == 0:
                    with open(debug_log_path, 'a') as f:
                        f.write(f"\n{'='*80}\n")
                        f.write(f"Position i={i} (processed position #{positions_processed})\n")
                        f.write(f"{'='*80}\n")
                        f.write(f"Samples analyzed: first 5 sequences\n\n")

                        for seq_idx in range(min(5, batch_end - batch_start)):
                            f.write(f"\n--- Sequence {seq_idx} ---\n")
                            f.write(f"True token ID at position {i}: {true_token_ids_at_i[seq_idx].item()}\n")
                            f.write(f"Is AA position: {current_pos_aa_mask[seq_idx].item()}\n")
                            f.write(f"Full logits at pos {i} (first 10): {logits_at_i[seq_idx, :10].cpu().numpy().round(3).tolist()}\n")
                            f.write(f"Full log probs (first 10): {log_probs_at_i[seq_idx, :10].cpu().numpy().round(3).tolist()}\n")
                            f.write(f"Log prob for true token: {log_prob_true_tokens[seq_idx].item():.6f}\n")

                # 5i. Store results in tensor (GPU -> CPU)
                log_probs_tensor[batch_start:batch_end, i] = log_prob_true_tokens.cpu()

        # [DEBUG] Log position processing statistics
        if (not hasattr(self, 'trainer') or self.trainer.is_global_zero):
            print(f"\n[DEBUG] Position Processing:")
            print(f"  Total positions in sequence: {L}")
            print(f"  Positions with AA tokens: {positions_processed}")
            print(f"  Log probs tensor non-zero entries: {(log_probs_tensor != 0).sum().item()}")
            print(f"  Log probs tensor sample (seq 0, pos 0-50): {log_probs_tensor[0, :50].numpy().round(3).tolist()}\n")

        # --- 6. Calculate PPL (Ablang2 method: "Average of PPLs") ---

        # Move tensors to GPU for computation
        log_probs_tensor = log_probs_tensor.to(self.device)
        # heavy_chain_mask, light_chain_mask, pure_aa_mask, all_ngl_masks_device are already on GPU

        # [MODIFIED] All masks are based on pure_aa_mask
        masks_info = {
            'total': pure_aa_mask, # Total now includes *only* pure AA positions
            'heavy_gl': heavy_chain_mask & ~all_ngl_masks_device & pure_aa_mask,
            'heavy_ngl': heavy_chain_mask & all_ngl_masks_device & pure_aa_mask,
            'light_gl': light_chain_mask & ~all_ngl_masks_device & pure_aa_mask,
            'light_ngl': light_chain_mask & all_ngl_masks_device & pure_aa_mask,
        }

        # [NEW] Dictionary to store per-sequence PPL values for all categories
        per_seq_ppl_results = {}

        for key, mask in masks_info.items():
            # mask shape is [N, L] (on GPU)

            # 1. Sum log-probs per sequence
            log_probs_masked = log_probs_tensor * mask
            sum_log_probs_per_seq = log_probs_masked.sum(dim=1) # [N]

            # 2. Count tokens per sequence
            count_tokens_per_seq = mask.sum(dim=1) # [N]

            # 3. Calculate average log-prob (avoid division by zero)
            avg_log_prob_per_seq = torch.zeros(N, device=self.device)
            # Only sequences with at least 1 token in this category
            valid_seq_mask = (count_tokens_per_seq > 0)

            if valid_seq_mask.sum() == 0:
                ppl = torch.tensor(0.0, device=self.device)

                # [NEW] Store per-sequence PPL (all zeros for this category)
                per_seq_ppl_results[key] = {
                    'ppl_per_seq': torch.zeros(N).cpu().numpy(),
                    'valid_mask': valid_seq_mask.cpu().numpy(),
                    'token_count_per_seq': count_tokens_per_seq.cpu().numpy(),
                    'avg_log_prob_per_seq': avg_log_prob_per_seq.cpu().numpy(),
                    'mean_ppl': 0.0
                }

                # [DEBUG]
                if (not hasattr(self, 'trainer') or self.trainer.is_global_zero):
                    print(f"\n[DEBUG] PPL Category '{key}': SKIPPED (No valid tokens found)")

            else:
                # 4. Calculate average log-prob per sequence
                avg_log_prob_per_seq[valid_seq_mask] = sum_log_probs_per_seq[valid_seq_mask] / count_tokens_per_seq[valid_seq_mask]

                # 5. Calculate PPL per sequence
                ppl_per_seq = torch.exp(-avg_log_prob_per_seq)

                # 6. Final average PPL (only over valid sequences)
                ppl = ppl_per_seq[valid_seq_mask].mean()

                # [NEW] Store per-sequence PPL results
                per_seq_ppl_results[key] = {
                    'ppl_per_seq': ppl_per_seq.cpu().numpy(),
                    'valid_mask': valid_seq_mask.cpu().numpy(),
                    'token_count_per_seq': count_tokens_per_seq.cpu().numpy(),
                    'avg_log_prob_per_seq': avg_log_prob_per_seq.cpu().numpy(),
                    'mean_ppl': ppl.item()
                }

                # --- [DEBUG] Debug logging ---
                if (not hasattr(self, 'trainer') or self.trainer.is_global_zero):
                    print("\n" + "="*40 + f" [DEBUG] PPL Category: '{key}' (FULL VOCAB) " + "="*23)
                    print(f"  > mask.shape: {mask.shape}, Total Tokens in Category: {mask.sum().item()}")
                    print(f"    Sample (seq 0, pos 0-30): {mask[0, :30].cpu().numpy().astype(int).tolist()}")

                    print(f"  > sum_log_probs_per_seq.shape: {sum_log_probs_per_seq.shape}")
                    print(f"    Sample (first 5 seqs): {sum_log_probs_per_seq[:5].cpu().numpy().round(3).tolist()}")

                    print(f"  > count_tokens_per_seq.shape: {count_tokens_per_seq.shape}")
                    print(f"    Sample (first 5 seqs): {count_tokens_per_seq[:5].cpu().numpy().tolist()}")

                    print(f"  > avg_log_prob_per_seq.shape: {avg_log_prob_per_seq.shape}")
                    print(f"    Sample (first 5 seqs): {avg_log_prob_per_seq[:5].cpu().numpy().round(3).tolist()}")

                    print(f"  > ppl_per_seq (VALID ONLY).count: {valid_seq_mask.sum().item()}")
                    print(f"    Sample (first 5 VALID seqs): {ppl_per_seq[valid_seq_mask][:5].cpu().numpy().round(3).tolist()}")

                    print(f"  > FINAL MEAN PPL: {ppl.item():.4f}")
                    print("="*100 + "\n")

                    # [NEW] Write detailed PPL calculation to debug log
                    if debug_log_path:
                        with open(debug_log_path, 'a') as f:
                            f.write(f"\n{'='*80}\n")
                            f.write(f"PPL CALCULATION FOR CATEGORY: {key}\n")
                            f.write(f"{'='*80}\n\n")
                            f.write(f"Total tokens in category: {mask.sum().item()}\n")
                            f.write(f"Valid sequences (with at least 1 token): {valid_seq_mask.sum().item()}\n")
                            f.write(f"Final mean PPL: {ppl.item():.4f}\n\n")

                            f.write(f"First 10 sequences breakdown:\n")
                            for seq_idx in range(min(10, N)):
                                if valid_seq_mask[seq_idx]:
                                    f.write(f"\nSeq {seq_idx}: ")
                                    f.write(f"tokens={count_tokens_per_seq[seq_idx].item()}, ")
                                    f.write(f"sum_log_prob={sum_log_probs_per_seq[seq_idx].item():.4f}, ")
                                    f.write(f"avg_log_prob={avg_log_prob_per_seq[seq_idx].item():.4f}, ")
                                    f.write(f"PPL={ppl_per_seq[seq_idx].item():.4f}")

                                    # Show which positions contributed
                                    contributing_positions = mask[seq_idx].nonzero(as_tuple=True)[0]
                                    if len(contributing_positions) > 0:
                                        sample_positions = contributing_positions[:10].cpu().numpy().tolist()
                                        f.write(f"\n  Contributing positions (first 10): {sample_positions}")

                                        # Show log probs at those positions
                                        sample_log_probs = log_probs_tensor[seq_idx, contributing_positions[:10]].cpu().numpy().round(4).tolist()
                                        f.write(f"\n  Log probs at those positions: {sample_log_probs}\n")

                            f.write(f"\n")
                # --- End debug logging ---

            log_name = f'{prefix}_perplexity_{key}'
            self.log(log_name, ppl, prog_bar=True, on_step=False, on_epoch=True)

        # [NEW] Save per-sequence PPL results to pickle file
        if (not hasattr(self, 'trainer') or self.trainer.is_global_zero):
            import pickle
            import os

            pkl_path = f"per_seq_ppl_{prefix}.pkl"

            # Add metadata to the results
            per_seq_ppl_results['metadata'] = {
                'num_sequences': N,
                'sequence_length': L,
                'total_positions_processed': positions_processed,
                'prefix': prefix,
                'categories': list(masks_info.keys())
            }

            with open(pkl_path, 'wb') as f:
                pickle.dump(per_seq_ppl_results, f)

            print(f"\n[SAVE] Per-sequence PPL results saved to: {pkl_path}")
            print(f"       Keys: {list(per_seq_ppl_results.keys())}")
            print(f"       Each category contains: ppl_per_seq, valid_mask, token_count_per_seq, avg_log_prob_per_seq, mean_ppl")

        # [NEW] Write final summary to debug log
        if debug_log_path:
            with open(debug_log_path, 'a') as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"FINAL SUMMARY\n")
                f.write(f"{'='*80}\n")
                f.write(f"Total positions processed: {positions_processed}\n")
                f.write(f"Log probs tensor shape: {log_probs_tensor.shape}\n")
                f.write(f"Non-zero log prob entries: {(log_probs_tensor != 0).sum().item()}\n")
                f.write(f"\nDebug log saved to: {debug_log_path}\n")
                f.write(f"Per-sequence PPL saved to: per_seq_ppl_{prefix}.pkl\n")
            print(f"\n[DEBUG] Detailed debug log saved to: {debug_log_path}")

        # Clean up GPU memory
        del log_probs_tensor, heavy_chain_mask, light_chain_mask, pure_aa_mask, all_ngl_masks_device, all_original_ids_device, all_attn_masks_device
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return # self.log handles logging
        
    def _create_chain_masks(self, input_ids, attn_mask):
        """
        Create boolean masks to separate HEAVY and LIGHT chain AMINO ACIDS.
        <cls> and <eos> tokens are EXCLUDED.
        """
        B, L = input_ids.shape
        heavy_chain_mask = torch.zeros_like(input_ids, dtype=torch.bool, device=input_ids.device)
        light_chain_mask = torch.zeros_like(input_ids, dtype=torch.bool, device=input_ids.device)

        cls_token_id = self.tokenizer.cls_token_id

        for b in range(B):
            cls_positions = (input_ids[b] == cls_token_id).nonzero(as_tuple=True)[0]

            if len(cls_positions) >= 3:
                first_cls = cls_positions[0].item()
                second_cls = cls_positions[1].item()
                third_cls = cls_positions[2].item()

                # Heavy chain: from after first CLS to before second CLS
                heavy_chain_mask[b, first_cls+1:second_cls] = True

                # Light chain: from after third CLS to end of valid sequence
                valid_length = attn_mask[b].sum().item()
                # [MODIFIED] <eos> 토큰 (valid_length-1)을 *제외*하기 위해 valid_length-1 까지
                light_chain_mask[b, third_cls+1:valid_length-1] = True

        return heavy_chain_mask, light_chain_mask

    # [NEW] Helper function to get 20-AA mask
    def _create_pure_aa_mask(self, input_ids):
        """
        Creates a boolean mask for positions that are standard amino acids (IDs 4-23).
        Assumes standard ESM2 vocabulary.
        """
        # IDs 4 (L) to 23 (C) are the 20 standard amino acids
        is_aa_mask = (input_ids >= 4) & (input_ids <= 23)
        return is_aa_mask

    def _calculate_masked_loss(self, logits, labels, position_mask=None):
        """
        Calculate cross-entropy loss for specific positions.
        Uses ALL VOCABULARY TOKENS (not restricted to 20 AA).

        Args:
            logits: Model output logits [B, L, V] where V is full vocabulary size
            labels: Target labels [B, L] (with -100 for positions to ignore)
            position_mask: Optional boolean mask [B, L] indicating which positions to include.
                          If None, uses all positions where labels != -100.

        Returns:
            Scalar loss value (computed over all vocab tokens)
        """
        B, L, V = logits.shape

        # Flatten logits and labels
        logits_flat = logits.view(-1, V)  # [B*L, V]
        labels_flat = labels.view(-1)      # [B*L]

        # Create mask for valid positions (not -100)
        valid_mask = (labels_flat != -100)

        # If position_mask is provided, combine it with valid_mask
        if position_mask is not None:
            position_mask_flat = position_mask.view(-1)  # [B*L]
            combined_mask = valid_mask & position_mask_flat
        else:
            combined_mask = valid_mask

        # Check if there are any valid positions
        if combined_mask.sum() == 0:
            # Return zero loss if no valid positions
            return torch.tensor(0.0, device=logits.device)

        # Select only the valid positions
        logits_masked = logits_flat[combined_mask]    # [N, V] where N = number of valid positions
        labels_masked = labels_flat[combined_mask]    # [N]

        # Calculate cross-entropy loss
        if self.loss_type == 'focal_loss':
            loss = self._focal_loss_from_logits(logits_masked, labels_masked, gamma=2.0)
        else:  # cross_entropy
            loss = F.cross_entropy(logits_masked, labels_masked)

        return loss

    def _focal_loss_from_logits(self, logits, targets, gamma):
        """
        Focal loss for already-filtered logits and targets (no ignore_index needed).
        """
        log_probs = F.log_softmax(logits, dim=-1)           # (N, V)
        log_pt = log_probs.gather(1, targets.unsqueeze(1))  # (N, 1)
        pt = log_pt.exp().squeeze(1)                        # (N,)

        focal_weight = (1.0 - pt).pow(gamma)                # (N,)
        loss = -focal_weight * log_pt.squeeze(1)            # (N,)

        return loss.mean()

    def _focal_loss(self, logits, targets, gamma, ignore_index = -100, ngl_mask = None, ngl_alpha = None, exclude_ngl = False, exclude_gl = False, label_smoothing = 0.0, position_weights = None):
        """
        Token-wise focal loss for masked-language modelling (γ-focusing).
        FL = (1 - p_t)^γ · CE
        where CE = −log p_t and p_t is the prob assigned to the true class.

        Only positions whose label ≠ ignore_index contribute to the loss.

        [NEW v7.0] NGL Loss Reweighting:
        If ngl_mask is provided, NGL tokens (ngl_mask==1) are weighted by ngl_alpha.
        This helps the model focus more on learning mutation positions.

        [NEW v8.0] Exclude NGL from AA Loss:
        If exclude_ngl=True and ngl_mask is provided, NGL positions are completely
        excluded from loss computation. This prevents gradient conflict between
        AA Head (predicts uppercase) and Final Head (predicts lowercase for NGL).

        [NEW v35/v36] Exclude GL from NGL AA Loss:
        If exclude_gl=True and ngl_mask is provided, GL positions (ngl_mask==0) are
        excluded. This is used for the NGL AA head which only trains on NGL positions.

        Args:
            logits: [B, L, V] or [N, V] - Model predictions
            targets: [B, L] or [N] - Ground truth labels
            gamma: Focal loss focusing parameter
            ignore_index: Label value to ignore
            ngl_mask: [B, L] or [N] - Binary mask (1=NGL, 0=GL), optional
            ngl_alpha: Weight multiplier for NGL tokens, optional (uses self.ngl_loss_alpha if None)
            exclude_ngl: If True, completely exclude NGL positions from loss (for GL AA head)
            exclude_gl: If True, completely exclude GL positions from loss (for NGL AA head)

        Returns:
            Weighted mean loss
        """
        vocab = logits.size(-1)
        logits  = logits.view(-1, vocab)      # [B*L, V]
        targets = targets.view(-1)            # [B*L]

        valid = targets != ignore_index

        # [NEW v8.0] Exclude NGL positions if requested (for AA Loss)
        # This prevents AA Head from learning on NGL positions, avoiding gradient conflict
        if exclude_ngl and ngl_mask is not None:
            ngl_mask_flat = ngl_mask.view(-1)
            gl_only_mask = (ngl_mask_flat == 0)  # Only GL positions
            valid = valid & gl_only_mask

        # [NEW v35/v36] Exclude GL positions if requested (for NGL AA Head Loss)
        # [FIX v39] Use > 0 instead of == 1 for 3-class compatibility
        # (0=GL, 1=SynNGL, 2=NGL) — NGL head should train on both SynNGL and NGL
        if exclude_gl and ngl_mask is not None:
            ngl_mask_flat = ngl_mask.view(-1)
            ngl_only_mask = (ngl_mask_flat > 0)  # SynNGL (1) + NGL (2) positions
            valid = valid & ngl_only_mask

        # Handle edge case: no valid positions after masking
        if not valid.any():
            return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

        logits  = logits[valid]
        targets = targets[valid]

        # [NEW v7.0] Handle NGL mask for reweighting (only if not excluding NGL)
        ngl_weights = None
        if ngl_mask is not None and not exclude_ngl:
            ngl_mask_flat = ngl_mask.view(-1)
            ngl_mask_valid = ngl_mask_flat[valid]

            # Determine alpha value
            alpha = ngl_alpha if ngl_alpha is not None else self.ngl_loss_alpha

            # Create weight tensor: NGL positions get alpha weight, GL positions get 1.0
            # [FIX v39] Use > 0 for 3-class compatibility (SynNGL + NGL both get upweighted)
            ngl_weights = torch.where(
                ngl_mask_valid > 0,
                torch.tensor(alpha, device=logits.device, dtype=logits.dtype),
                torch.tensor(1.0, device=logits.device, dtype=logits.dtype)
            )

        log_probs = torch.log_softmax(logits, dim=-1)          # (N, V)
        log_pt = log_probs.gather(1, targets.unsqueeze(1))     # (N, 1)
        pt = log_pt.exp().squeeze(1)                           # (N,)

        focal_weight = (1.0 - pt).pow(gamma)                   # (N,)

        # [NEW v35.1b] Label smoothing: soft cross-entropy with focal weighting
        # loss = focal_weight * ((1 - ε) * (-log_pt) + ε * (-mean(log_probs)))
        if label_smoothing > 0.0:
            smooth_loss = -log_probs.mean(dim=-1)              # (N,) uniform CE component
            ce = (1.0 - label_smoothing) * (-log_pt.squeeze(1)) + label_smoothing * smooth_loss
            loss = focal_weight * ce                           # (N,)
        else:
            loss = -focal_weight * log_pt.squeeze(1)           # (N,)

        # [CHANGE v17] Per-token loss clamp to prevent numerical instability
        # This matches the PPL calculation clamp (max=20.0) for consistency
        # Note: exp(20) ≈ 485 million, which is a reasonable upper bound for PPL
        loss = torch.clamp(loss, max=20.0)

        # [NEW v40] Apply position-level weights (e.g., MPNN GL weighting)
        if position_weights is not None:
            position_weights_flat = position_weights.view(-1)[valid]
            loss = loss * position_weights_flat

        # [NEW v7.0] Apply NGL reweighting if provided
        # [CHANGE v17] Use weighted mean to prevent batch-size dependent loss scaling
        # This ensures NGL reweighting doesn't cause loss explosion with larger batches
        if ngl_weights is not None:
            # Weighted sum normalized by total weight for stable batch-size invariant loss
            total_weight = ngl_weights.sum()
            return (loss * ngl_weights).sum() / (total_weight + 1e-8)
        elif position_weights is not None:
            # Use mean weighted by position weights
            total_weight = position_weights.view(-1)[valid].sum()
            return loss.sum() / (total_weight + 1e-8)
        else:
            return loss.mean()

    def _focal_loss_region_balanced(self, logits, targets, gamma, ignore_index=-100,
                                     ngl_mask=None, region_ids=None, ngl_alpha=None):
        """
        [NEW v25] Region-balanced focal loss for equalizing FR and CDR contribution.

        This addresses the FR PPL >> CDR PPL imbalance by ensuring equal gradient
        contribution from Framework (FR) and CDR regions regardless of their sizes.

        Loss = 0.5 * Loss_FR + 0.5 * Loss_CDR

        where each region loss is computed as the weighted mean of per-token losses
        within that region, with optional NGL reweighting applied.

        Region IDs: 0=special, 1=FR1, 2=CDR1, 3=FR2, 4=CDR2, 5=FR3, 6=CDR3, 7=FR4
        CDR = {2, 4, 6}, FR = {1, 3, 5, 7}

        Args:
            logits: [B, L, V] - Model predictions
            targets: [B, L] - Ground truth labels
            gamma: Focal loss focusing parameter
            ignore_index: Label value to ignore (-100)
            ngl_mask: [B, L] - Binary NGL mask (1=NGL, 0=GL), optional
            region_ids: [B, L] - Region IDs for each position
            ngl_alpha: Weight multiplier for NGL tokens

        Returns:
            Region-balanced weighted mean loss
        """
        vocab = logits.size(-1)
        batch_size, seq_len = targets.shape

        # Flatten tensors
        logits_flat = logits.view(-1, vocab)  # [B*L, V]
        targets_flat = targets.view(-1)       # [B*L]

        # Valid mask (not ignore_index)
        valid = targets_flat != ignore_index

        # If no region_ids provided, fall back to regular focal loss
        if region_ids is None:
            return self._focal_loss(logits, targets, gamma, ignore_index, ngl_mask, ngl_alpha)

        region_ids_flat = region_ids.view(-1)  # [B*L]

        # [FIX v26] Corrected Region ID mapping based on actual data:
        # 0=FR1, 1=CDR1, 2=FR2, 3=CDR2, 4=FR3, 5=CDR3, 6=FR4
        # CDR = {1, 3, 5} (CDR1, CDR2, CDR3)
        cdr_mask = (region_ids_flat == 1) | (region_ids_flat == 3) | (region_ids_flat == 5)
        # FR = {0, 2, 4, 6} (FR1, FR2, FR3, FR4)
        fr_mask = (region_ids_flat == 0) | (region_ids_flat == 2) | (region_ids_flat == 4) | (region_ids_flat == 6)

        # Combined masks with valid positions
        cdr_valid = valid & cdr_mask
        fr_valid = valid & fr_mask

        # Handle edge case: no valid positions in either region
        if not cdr_valid.any() and not fr_valid.any():
            return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

        # Compute per-token focal loss for all valid positions
        log_probs = torch.log_softmax(logits_flat, dim=-1)  # [B*L, V]
        log_pt = log_probs.gather(1, targets_flat.unsqueeze(1).clamp(min=0))  # [B*L, 1]
        pt = log_pt.exp().squeeze(1)  # [B*L]
        focal_weight = (1.0 - pt).pow(gamma)  # [B*L]
        per_token_loss = -focal_weight * log_pt.squeeze(1)  # [B*L]
        per_token_loss = torch.clamp(per_token_loss, max=20.0)  # Numerical stability

        # Prepare NGL weights if provided
        # [FIX v39] Use > 0 for 3-class compatibility
        ngl_weights_flat = None
        if ngl_mask is not None:
            ngl_mask_flat = ngl_mask.view(-1)
            alpha_val = ngl_alpha if ngl_alpha is not None else self.ngl_loss_alpha
            ngl_weights_flat = torch.where(
                ngl_mask_flat > 0,
                torch.tensor(alpha_val, device=logits.device, dtype=logits.dtype),
                torch.tensor(1.0, device=logits.device, dtype=logits.dtype)
            )

        # Compute CDR loss
        loss_cdr = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        if cdr_valid.any():
            cdr_losses = per_token_loss[cdr_valid]
            if ngl_weights_flat is not None:
                cdr_weights = ngl_weights_flat[cdr_valid]
                loss_cdr = (cdr_losses * cdr_weights).sum() / (cdr_weights.sum() + 1e-8)
            else:
                loss_cdr = cdr_losses.mean()

        # Compute FR loss
        loss_fr = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        if fr_valid.any():
            fr_losses = per_token_loss[fr_valid]
            if ngl_weights_flat is not None:
                fr_weights = ngl_weights_flat[fr_valid]
                loss_fr = (fr_losses * fr_weights).sum() / (fr_weights.sum() + 1e-8)
            else:
                loss_fr = fr_losses.mean()

        # [NEW v28] Apply CDR loss boosting if enabled
        # This increases the weight of CDR loss to prioritize CDR accuracy
        if self.use_cdr_loss_boosting:
            loss_cdr = loss_cdr * self.cdr_loss_multiplier

        # Balanced combination: equal weight to FR and CDR
        # Handle cases where one region might be empty
        # [CHANGE v28] With CDR boosting, CDR gets cdr_loss_multiplier × 0.5 effective weight
        if cdr_valid.any() and fr_valid.any():
            balanced_loss = 0.5 * loss_cdr + 0.5 * loss_fr
        elif cdr_valid.any():
            balanced_loss = loss_cdr
        else:
            balanced_loss = loss_fr

        return balanced_loss

    # =========================================================================
    # [NEW v37] GL-NGL Divergence Loss
    # =========================================================================
    @staticmethod
    def _compute_divergence_at_positions(logits_aa_gl, logits_aa_ngl, valid_mask,
                                          max_kl_divergence, divergence_type):
        """Helper to compute divergence at specific positions."""
        if not valid_mask.any():
            return torch.tensor(0.0, device=logits_aa_gl.device), torch.tensor(0, device=logits_aa_gl.device)

        gl_logits_20 = logits_aa_gl[:, :, 4:24][valid_mask]
        ngl_logits_20 = logits_aa_ngl[:, :, 4:24][valid_mask]

        p_ngl = F.softmax(ngl_logits_20, dim=-1)
        p_gl = F.softmax(gl_logits_20, dim=-1)

        if divergence_type == "kl":
            log_p_ngl = F.log_softmax(ngl_logits_20, dim=-1)
            log_p_gl = F.log_softmax(gl_logits_20, dim=-1)
            div_per_pos = (p_ngl * (log_p_ngl - log_p_gl)).sum(dim=-1)
        elif divergence_type == "js":
            m = 0.5 * (p_ngl + p_gl)
            log_m = torch.log(m + 1e-8)
            log_p_ngl = torch.log(p_ngl + 1e-8)
            log_p_gl = torch.log(p_gl + 1e-8)
            kl_ngl_m = (p_ngl * (log_p_ngl - log_m)).sum(dim=-1)
            kl_gl_m = (p_gl * (log_p_gl - log_m)).sum(dim=-1)
            div_per_pos = 0.5 * kl_ngl_m + 0.5 * kl_gl_m
        else:
            raise ValueError(f"Unknown divergence_type: {divergence_type}. Must be 'kl' or 'js'")

        div_clamped = torch.clamp(div_per_pos, max=max_kl_divergence)
        return div_clamped.sum(), valid_mask.sum()

    @staticmethod
    def _compute_divergence_loss(logits_aa_gl, logits_aa_ngl, ngl_masks, labels_aa,
                                  max_kl_divergence=10.0, divergence_type="kl",
                                  synth_masks=None, synth_div_weight=0.3):
        """
        Compute divergence between GL and NGL AA head predictions at masked positions.

        At NGL positions, we want the GL and NGL heads to produce DIFFERENT amino acid
        distributions (specialization). We maximize divergence by returning negative mean.

        [v40] Optionally also compute divergence at SynNGL positions with lower weight.

        Args:
            logits_aa_gl: [B, L, V] - GL AA head logits (full vocab)
            logits_aa_ngl: [B, L, V] - NGL AA head logits (full vocab)
            ngl_masks: [B, L] - 1 for NGL positions, 0 for GL
            labels_aa: [B, L] - AA labels (-100 for non-masked)
            max_kl_divergence: float - clamp per-position KL
            divergence_type: str - "kl" or "js"
            synth_masks: [B, L] - 1 for SynNGL positions, 0 otherwise (optional)
            synth_div_weight: float - weight for SynNGL divergence relative to NGL (default 0.3)

        Returns:
            loss: scalar - negative mean divergence (minimize this to maximize divergence)
        """
        # NGL positions: masked AND NGL
        valid_ngl = (ngl_masks > 0) & (labels_aa != -100)

        div_ngl_sum, count_ngl = SFT_ESM2._compute_divergence_at_positions(
            logits_aa_gl, logits_aa_ngl, valid_ngl, max_kl_divergence, divergence_type
        )

        # [v40] SynNGL positions: masked AND SynNGL AND NOT NGL
        if synth_masks is not None and synth_div_weight > 0.0:
            valid_synth = synth_masks.bool() & (ngl_masks == 0) & (labels_aa != -100)
            div_synth_sum, count_synth = SFT_ESM2._compute_divergence_at_positions(
                logits_aa_gl, logits_aa_ngl, valid_synth, max_kl_divergence, divergence_type
            )

            total = div_ngl_sum + synth_div_weight * div_synth_sum
            count = count_ngl.float() + synth_div_weight * count_synth.float()
        else:
            total = div_ngl_sum
            count = count_ngl.float()

        if count < 1e-8:
            return torch.tensor(0.0, device=logits_aa_gl.device)

        return -(total / count)

    def _binary_focal_loss(self, logits, targets, gamma, alpha, ignore_index=-100):
        """
        Binary focal loss for GL/NGL classification with class weighting.

        Args:
            logits: [B, L, 1] or [B, L] - raw logits from classification head
            targets: [B, L] - binary targets (0=GL, 1=NGL)
            gamma: Focusing parameter (typically 2.0)
            alpha: Weight for positive class (NGL). Should be higher for minority class.
                   If alpha=0.75, then NGL (class 1) has weight 0.75, GL (class 0) has weight 0.25
            ignore_index: Positions to ignore (e.g., padding, special tokens)

        Returns:
            Scalar loss
        """
        # Reshape inputs
        if logits.dim() == 3:
            logits = logits.squeeze(-1)  # [B, L, 1] -> [B, L]

        logits = logits.view(-1)   # [B*L]
        targets = targets.view(-1).float()  # [B*L]

        # Filter out ignored positions
        if ignore_index is not None:
            valid = targets != ignore_index
            logits = logits[valid]
            targets = targets[valid]

        # Calculate probabilities using sigmoid (binary classification)
        probs = torch.sigmoid(logits)  # [N]

        # Calculate p_t (probability of the true class)
        # If target=1 (NGL), p_t = prob; if target=0 (GL), p_t = 1-prob
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # Calculate alpha_t (class weight)
        # If target=1 (NGL), alpha_t = alpha; if target=0 (GL), alpha_t = 1-alpha
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)

        # Focal loss formula: FL = -alpha_t * (1 - p_t)^gamma * log(p_t)
        focal_weight = (1.0 - p_t).pow(gamma)

        # Binary cross entropy
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        # Combine
        loss = alpha_t * focal_weight * bce

        return loss.mean()

    def _multiclass_focal_loss(self, logits, targets, gamma, class_weights=None, ignore_index=-1):
        """
        [v38] Multiclass focal loss for 3-class GL/SynNGL/NGL classification.

        Focal loss: FL = -class_weight * (1-pt)^gamma * log(pt)

        Args:
            logits: [B, L, C] or [N, C] - raw logits for C classes
            targets: [B, L] or [N] - integer class labels
            gamma: Focusing parameter (typically 2.0)
            class_weights: [C] tensor for imbalanced classes (None=uniform)
            ignore_index: Target value to ignore (default -1)

        Returns:
            Scalar loss (0.0 if all targets are ignored)
        """
        # Flatten
        if logits.dim() == 3:
            B, L, C = logits.shape
            logits = logits.view(-1, C)  # [B*L, C]
            targets = targets.view(-1)    # [B*L]
        else:
            C = logits.shape[-1]

        # Filter ignored positions
        if ignore_index is not None:
            valid = targets != ignore_index
            logits = logits[valid]
            targets = targets[valid]

        if logits.numel() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        # Compute p_t via log_softmax for numerical stability
        log_p = F.log_softmax(logits, dim=-1)  # [N, C]
        p = torch.exp(log_p)

        # Gather p_t for true class
        log_p_t = log_p.gather(1, targets.unsqueeze(1)).squeeze(1)  # [N]
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)          # [N]

        # Focal weight: (1-pt)^gamma
        focal_weight = (1.0 - p_t).pow(gamma)  # [N]

        # Class weights
        if class_weights is not None:
            if not isinstance(class_weights, torch.Tensor):
                class_weights = torch.tensor(class_weights, device=logits.device, dtype=logits.dtype)
            w_t = class_weights[targets]  # [N]
        else:
            w_t = 1.0

        # Focal loss: -w_t * (1-pt)^gamma * log(pt)
        loss = -w_t * focal_weight * log_p_t

        return loss.mean()

    def configure_optimizers(self):
        # Collect trainable parameters from ESM2 model
        trainable_params = list(filter(lambda p: p.requires_grad, self.ESM2.parameters()))

        # Add multihead architecture parameters if enabled
        if self.use_multihead_architecture:
            # AA head components (Dense → LayerNorm → Decoder + Bias)
            if self.aa_head_dense is not None:
                trainable_params.extend(self.aa_head_dense.parameters())
            if self.aa_head_layer_norm is not None:
                trainable_params.extend(self.aa_head_layer_norm.parameters())
            if self.aa_head_decoder is not None:
                trainable_params.extend(self.aa_head_decoder.parameters())
            if self.aa_head_bias is not None:
                trainable_params.append(self.aa_head_bias)
            # Mutation head + dropout
            if self.mut_head is not None:
                trainable_params.extend(self.mut_head.parameters())
            if self.mut_dropout is not None:
                trainable_params.extend(self.mut_dropout.parameters())
            # [NEW v6.0] Alpha head for region-aware gating
            if self.alpha_head is not None:
                trainable_params.extend(self.alpha_head.parameters())

            # Origin projection for sequential conditioning
            if self.origin_projection is not None:
                trainable_params.extend(self.origin_projection.parameters())

        # [FIX] Gene conditioning modules (defined on self, NOT under self.ESM2)
        if self.use_germline_genes:
            if self.v_gene_embedding is not None:
                trainable_params.extend(self.v_gene_embedding.parameters())
            if self.j_gene_embedding is not None:
                trainable_params.extend(self.j_gene_embedding.parameters())
            if self.gene_projection is not None:
                trainable_params.extend(self.gene_projection.parameters())

        # [FIX] Region embedding modules
        if self.use_region_embedding:
            if self.region_embedding is not None:
                trainable_params.extend(self.region_embedding.parameters())
            if self.region_projection is not None:
                trainable_params.extend(self.region_projection.parameters())

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.peak_learning_rate,
            betas=(self.adam_beta1, self.adam_beta2),
            eps=self.adam_epsilon,
            weight_decay=self.WD
        )

        # Cosine decay scheduler with linear warmup
        # Warms up linearly from 0 to peak_lr over warmup_steps
        # Then decays with cosine annealing from peak_lr to 0 until max_steps
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.max_steps
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }

    def _smart_initialize_lm_head(self, model):
        """
        Smart Initialization for decoupled LM Head (Asymmetric Input/Output Strategy).

        When tie_word_embeddings=False, the LM Head will have randomly initialized weights.
        This method copies weights from the Input Embeddings to the LM Head to prevent
        convergence issues.

        CRITICAL: For the lowercase_ngl strategy, we also copy the weights of uppercase tokens
        (GL) from the Input Embeddings to the corresponding lowercase token slots (NGL)
        in the LM Head. This allows the LM Head to predict both GL and NGL tokens while
        the input embeddings only need to represent uppercase tokens.

        Args:
            model: The ESM2 model with untied embeddings
        """
        with torch.no_grad():
            # Get references to the embedding and LM head weights
            input_embeddings = model.esm.embeddings.word_embeddings.weight  # [V, H]
            lm_head_weight = model.lm_head.decoder.weight  # [V, H]

            # Step 1: Copy all Input Embedding weights to the LM Head
            lm_head_weight.copy_(input_embeddings)
            print(f"[Smart Init] Copied all {input_embeddings.size(0)} embeddings to LM Head")

            # Step 2: Copy uppercase AA weights to lowercase token slots in LM Head
            # self.lowercase_aa_token_ids is a dict: {upper_id: lower_id}
            if self.lowercase_aa_token_ids is not None:
                num_copied = 0
                for upper_id, lower_id in self.lowercase_aa_token_ids.items():
                    # Copy weight from uppercase position to lowercase position in LM Head
                    lm_head_weight[lower_id] = input_embeddings[upper_id].clone()
                    num_copied += 1

                print(f"[Smart Init] Copied {num_copied} uppercase AA embeddings to lowercase slots in LM Head")
                print(f"[Smart Init] This allows LM Head to predict both GL (uppercase) and NGL (lowercase) tokens")

            # Also copy bias if it exists
            if hasattr(model.lm_head, 'bias') and model.lm_head.bias is not None:
                # For bias, we just initialize the lowercase positions with the uppercase bias values
                lm_head_bias = model.lm_head.bias  # [V]
                if self.lowercase_aa_token_ids is not None:
                    for upper_id, lower_id in self.lowercase_aa_token_ids.items():
                        lm_head_bias[lower_id] = lm_head_bias[upper_id].clone()
                    print(f"[Smart Init] Copied uppercase AA biases to lowercase slots in LM Head bias")

    def _smart_initialize_aa_head(self):
        """
        Smart Initialization for AA Head in Multihead Architecture.

        Copies weights from the original ESM2 lm_head to our AA head components:
        - aa_head_dense ← lm_head.dense
        - aa_head_layer_norm ← lm_head.layer_norm
        - aa_head_decoder ← lm_head.decoder (input embeddings)
        - aa_head_bias ← lm_head.bias

        This ensures the AA head starts with pretrained/initialized weights.
        """
        with torch.no_grad():
            # Get references to the original ESM2 lm_head
            lm_head = self.ESM2.lm_head
            input_embeddings = self.ESM2.esm.embeddings.word_embeddings.weight

            # Copy dense layer weights
            if hasattr(lm_head, 'dense'):
                self.aa_head_dense.weight.copy_(lm_head.dense.weight)
                self.aa_head_dense.bias.copy_(lm_head.dense.bias)
                print(f"    [Smart Init] Copied lm_head.dense weights")

            # Copy layer norm weights
            if hasattr(lm_head, 'layer_norm'):
                self.aa_head_layer_norm.weight.copy_(lm_head.layer_norm.weight)
                self.aa_head_layer_norm.bias.copy_(lm_head.layer_norm.bias)
                print(f"    [Smart Init] Copied lm_head.layer_norm weights")

            # Copy decoder weights from input embeddings (or lm_head.decoder if untied)
            if hasattr(lm_head, 'decoder') and hasattr(lm_head.decoder, 'weight'):
                self.aa_head_decoder.weight.copy_(lm_head.decoder.weight)
                print(f"    [Smart Init] Copied lm_head.decoder weights")
            else:
                # Fallback: use input embeddings
                self.aa_head_decoder.weight.copy_(input_embeddings)
                print(f"    [Smart Init] Copied input embeddings to decoder")

            # Copy bias
            if hasattr(lm_head, 'bias') and lm_head.bias is not None:
                self.aa_head_bias.copy_(lm_head.bias)
                print(f"    [Smart Init] Copied lm_head.bias")

    def _smart_initialize_ngl_aa_head(self):
        """
        Smart Initialization for NGL AA Head (v35/v36).
        Copies weights from the original ESM2 lm_head to the NGL AA head components,
        mirroring _smart_initialize_aa_head.
        """
        with torch.no_grad():
            lm_head = self.ESM2.lm_head
            input_embeddings = self.ESM2.esm.embeddings.word_embeddings.weight

            if hasattr(lm_head, 'dense'):
                self.ngl_aa_head_dense.weight.copy_(lm_head.dense.weight)
                self.ngl_aa_head_dense.bias.copy_(lm_head.dense.bias)
                print(f"    [Smart Init NGL] Copied lm_head.dense weights")

            if hasattr(lm_head, 'layer_norm'):
                self.ngl_aa_head_layer_norm.weight.copy_(lm_head.layer_norm.weight)
                self.ngl_aa_head_layer_norm.bias.copy_(lm_head.layer_norm.bias)
                print(f"    [Smart Init NGL] Copied lm_head.layer_norm weights")

            if hasattr(lm_head, 'decoder') and hasattr(lm_head.decoder, 'weight'):
                self.ngl_aa_head_decoder.weight.copy_(lm_head.decoder.weight)
                print(f"    [Smart Init NGL] Copied lm_head.decoder weights")
            else:
                self.ngl_aa_head_decoder.weight.copy_(input_embeddings)
                print(f"    [Smart Init NGL] Copied input embeddings to decoder")

            if hasattr(lm_head, 'bias') and lm_head.bias is not None:
                self.ngl_aa_head_bias.copy_(lm_head.bias)
                print(f"    [Smart Init NGL] Copied lm_head.bias")

    def _forward_multihead(self, input_ids, attention_mask, v_gene_ids=None, j_gene_ids=None, region_ids=None):
        """
        Forward pass for multihead architecture with Sequential Conditional Architecture.

        [MODIFIED v14.0] Implements Sequential Conditional Architecture:
        1. Run Origin Head first to get mutation/origin logits
        2. Project origin_logits back to hidden_size using origin_projection
        3. Add projected features to original hidden_states (conditioning)
        4. Pass conditioned hidden states to AA Head

        [MODIFIED v16.0] Implements Gradient Detach (detach_origin_gradient=True):
        - When enabled, origin_logits are detached before projection
        - This ensures the Origin Head is trained ONLY by mut_loss
        - Prevents AA loss and final_loss from corrupting Origin Head gradients
        - Solves the "task interference" problem in multi-task learning

        [MODIFIED v24] Added region_ids support for FR/CDR differentiation

        This allows the AA Head to "see" the mutation prediction, addressing the
        information flow problem where the AA Head needs to know if a mutation is
        occurring to predict the correct amino acid (e.g., 'a' instead of 'A').

        Args:
            input_ids: [B, L] - Input token IDs (all uppercase)
            attention_mask: [B, L] - Attention mask
            v_gene_ids: [B] - Optional V-gene IDs
            j_gene_ids: [B] - Optional J-gene IDs
            region_ids: [B, L] - Optional region IDs [v24]

        Returns:
            logits_aa: [B, L, V] - GL AA identity logits (33 vocab)
            logits_aa_ngl: [B, L, V] or None - NGL AA identity logits (33 vocab, v35/v36 only)
            logits_mut: [B, L] - Mutation state logits (binary, origin head)
            alpha: [B, L, 1] or None - Region-aware gating values (if use_alpha_gating)
            logits_final: [B, L, 53] or None - Final combined 53-vocab logits (if use_alpha_gating)
            hidden_states: [B, L, H] - Hidden states from backbone (original, unconditioned)
        """
        # =====================================================================
        # Step 1: Get hidden states from ESM2 backbone
        # [v24] Now includes region embeddings
        # =====================================================================
        use_custom_embeds = (
            (self.use_germline_genes and v_gene_ids is not None and j_gene_ids is not None) or
            (self.use_region_embedding and region_ids is not None)
        )

        if use_custom_embeds:
            inputs_embeds = self._get_gene_conditioned_inputs_embeds(input_ids, v_gene_ids, j_gene_ids, region_ids)
            # See _forward_with_gene_conditioning for rationale: older HF
            # versions crash on ``token_dropout`` when input_ids is None.
            emb_layer = self.ESM2.esm.embeddings
            prev_token_dropout = emb_layer.token_dropout
            emb_layer.token_dropout = False
            try:
                outputs = self.ESM2(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )
            finally:
                emb_layer.token_dropout = prev_token_dropout
        else:
            outputs = self.ESM2(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )

        # Get last hidden state (original, unconditioned)
        hidden_states = outputs.hidden_states[-1]  # [B, L, H]

        # =====================================================================
        # Step 2: Run Origin Head FIRST (Sequential Conditional Architecture)
        # [CHANGE v17] Apply dropout after Origin Head to regularize logits
        # [v38] Supports 2-class [B,L,1] or 3-class [B,L,3] origin logits
        # =====================================================================
        logits_mut_raw = self.mut_head(hidden_states)  # [B, L, num_origin_classes]
        logits_mut_raw = self.mut_dropout(logits_mut_raw)

        if self.num_origin_classes == 2:
            # Binary: squeeze to [B, L] for backward compat
            logits_mut = logits_mut_raw.squeeze(-1)  # [B, L]
        else:
            # 3-class: keep [B, L, 3] shape
            logits_mut = logits_mut_raw  # [B, L, 3]

        # =====================================================================
        # Step 3: Project origin logits back to hidden_size and condition
        # [v16.0] THE GUARDRAIL: Detach origin logits if detach_origin_gradient=True
        # =====================================================================
        if self.detach_origin_gradient:
            logits_mut_for_projection = logits_mut_raw.detach()
        else:
            logits_mut_for_projection = logits_mut_raw

        # Project origin logits to hidden_size: [B, L, C] -> [B, L, H]
        origin_features = self.origin_projection(logits_mut_for_projection)  # [B, L, H]

        # Create conditioned hidden states by adding origin features
        # This allows AA Head to "see" the mutation prediction
        conditioned_hidden_states = hidden_states + origin_features  # [B, L, H]

        # =====================================================================
        # Step 4: Run AA Head(s)
        # v34: Single AA head on conditioned_hidden_states
        # v35: GL AA head + NGL AA head, both on conditioned_hidden_states
        # v36: GL AA head + NGL AA head, both on raw hidden_states; alpha on raw too
        # =====================================================================
        if self.use_dual_aa_heads and self.dual_aa_heads_conditioned:
            # v35: Both heads use conditioned hidden states (origin info injected)
            aa_input = conditioned_hidden_states
            ngl_aa_input = conditioned_hidden_states
        elif self.use_dual_aa_heads and not self.dual_aa_heads_conditioned:
            # v36: Both heads use raw hidden states (no origin info)
            aa_input = hidden_states
            ngl_aa_input = hidden_states
        else:
            # v34: Single AA head on conditioned hidden states
            aa_input = conditioned_hidden_states

        # GL AA Head (Dense → GELU → LayerNorm → Decoder + bias)
        aa_hidden = self.aa_head_dense(aa_input)
        aa_hidden = F.gelu(aa_hidden)
        aa_hidden = self.aa_head_layer_norm(aa_hidden)
        logits_aa = self.aa_head_decoder(aa_hidden) + self.aa_head_bias  # [B, L, V]

        # NGL AA Head (v35/v36 only)
        logits_aa_ngl = None
        if self.use_dual_aa_heads and self.ngl_aa_head_dense is not None:
            ngl_hidden = self.ngl_aa_head_dense(ngl_aa_input)
            ngl_hidden = F.gelu(ngl_hidden)
            ngl_hidden = self.ngl_aa_head_layer_norm(ngl_hidden)
            logits_aa_ngl = self.ngl_aa_head_decoder(ngl_hidden) + self.ngl_aa_head_bias  # [B, L, V]

        # =====================================================================
        # Step 5: Compute alpha and final 53-vocab logits if alpha gating is enabled
        # =====================================================================
        alpha = None
        logits_final = None

        if self.use_alpha_gating and self.alpha_head is not None:
            if self.fixed_alpha_value is not None:
                # [ABLATION] Fixed alpha — bypass alpha_head entirely
                alpha = torch.full(
                    (hidden_states.size(0), hidden_states.size(1), 1),
                    self.fixed_alpha_value,
                    device=hidden_states.device, dtype=hidden_states.dtype,
                )
            elif self.use_dual_aa_heads and not self.dual_aa_heads_conditioned:
                # v36: Alpha head uses raw hidden states (no origin conditioning)
                alpha = self.alpha_head(hidden_states)  # [B, L, 1]
            else:
                alpha = self.alpha_head(conditioned_hidden_states)  # [B, L, 1]

            # Construct 53-vocab logits
            # [FIX v38.1b] Detach origin logits to prevent loss_final from
            # backpropagating into origin head — origin head should only receive
            # gradient from origin_loss (same principle as detach_origin_gradient)
            logits_mut_detached = logits_mut.detach() if self.detach_origin_gradient else logits_mut
            if self.use_dual_aa_heads and logits_aa_ngl is not None:
                # v35/v36: GL head for uppercase, NGL head for lowercase
                logits_final = self._construct_53_vocab_logits_dual_heads(logits_aa, logits_aa_ngl, logits_mut_detached, alpha)
            else:
                # v34: Single AA head for both
                logits_final = self._construct_53_vocab_logits_multiplicative(logits_aa, logits_mut_detached, alpha)

        return logits_aa, logits_aa_ngl, logits_mut, alpha, logits_final, hidden_states

    def _construct_53_vocab_logits_multiplicative(self, logits_aa, logits_mut, alpha):
        """
        [v13/v33.6/v38] Construct 53-vocab logits using ALPHA-WEIGHTED MULTIPLICATIVE GATING.

        Supports both 2-class (binary) and 3-class (GL/SynNGL/NGL) origin heads.

        For 2-class (binary):
            log P(GL)  = log_sigmoid(-logits_mut)
            log P(NGL) = log_sigmoid(logits_mut)

        For 3-class (v38):
            log_p = log_softmax(logits_mut, dim=-1)  # [B, L, 3]
            P_eff_NGL = P(NGL) + sw * P(SynNGL)
            P_eff_GL  = P(GL) + (1-sw) * P(SynNGL)

        Args:
            logits_aa: [B, L, V] - AA identity logits (33 vocab)
            logits_mut: [B, L] (2-class) or [B, L, 3] (3-class) - Origin head logits
            alpha: [B, L, 1] - Per-position gating weights in [0, 1]

        Returns:
            logits_final: [B, L, 53] - LOG PROBABILITIES (do NOT apply log_softmax again!)
        """
        B, L, V_orig = logits_aa.shape  # V_orig = 33

        # Step 1: Compute log-probabilities for AA identity
        log_p_aa = F.log_softmax(logits_aa, dim=-1)  # [B, L, 33]

        # Step 2: Compute effective log-probabilities for GL/NGL status
        if self.num_origin_classes == 3:
            # [v38] 3-class gating with SynNGL mixing
            log_p = F.log_softmax(logits_mut, dim=-1)  # [B, L, 3]
            log_p_gl_raw = log_p[:, :, 0]    # [B, L]
            log_p_syn = log_p[:, :, 1]        # [B, L]
            log_p_ngl_raw = log_p[:, :, 2]   # [B, L]

            sw = self.synth_weight
            log_sw = torch.log(torch.tensor(sw, device=logits_mut.device, dtype=logits_mut.dtype))
            log_1_minus_sw = torch.log(torch.tensor(1.0 - sw, device=logits_mut.device, dtype=logits_mut.dtype))

            # Effective probs: P_eff_NGL = P(NGL) + sw * P(SynNGL)
            #                  P_eff_GL  = P(GL) + (1-sw) * P(SynNGL)
            log_p_ngl = torch.logaddexp(log_p_ngl_raw, log_sw + log_p_syn)  # [B, L]
            log_p_gl = torch.logaddexp(log_p_gl_raw, log_1_minus_sw + log_p_syn)  # [B, L]
        else:
            # 2-class binary gating (original behavior)
            log_p_ngl = F.logsigmoid(logits_mut)         # [B, L] - log P(NGL)
            log_p_gl = F.logsigmoid(-logits_mut)         # [B, L] - log P(GL)

        # Expand for broadcasting: [B, L] -> [B, L, 1]
        log_p_ngl = log_p_ngl.unsqueeze(-1)
        log_p_gl = log_p_gl.unsqueeze(-1)

        # Step 3: Initialize final logits tensor
        V_final = 53
        logits_final = torch.full(
            (B, L, V_final),
            fill_value=-1e9,  # Very negative = effectively zero probability
            device=logits_aa.device,
            dtype=logits_aa.dtype
        )

        # Step 4: [v33.6] Alpha-weighted combination in log-space
        # When alpha = 0: log P(upper) = log P(AA) + 0 = log P(AA) (pure identity)
        # When alpha = 1: log P(upper) = log P(AA) + log P(GL) (full gating)
        # alpha shape: [B, L, 1], log_p_gl shape: [B, L, 1]
        alpha_weighted_log_p_gl = alpha * log_p_gl      # [B, L, 1]
        alpha_weighted_log_p_ngl = alpha * log_p_ngl    # [B, L, 1]

        # GL (Uppercase) logits: log P(uppercase AA_i) = log P(AA_i) + alpha * log P(GL)
        logits_final[:, :, :V_orig] = log_p_aa + alpha_weighted_log_p_gl

        # NGL (Lowercase) logits: log P(lowercase aa_i) = log P(AA_i) + alpha * log P(NGL)
        if self.lowercase_aa_token_ids is not None:
            for upper_id, lower_id in self.lowercase_aa_token_ids.items():
                log_p_this_aa = log_p_aa[:, :, upper_id]  # [B, L]
                logits_final[:, :, lower_id] = log_p_this_aa + alpha_weighted_log_p_ngl.squeeze(-1)

        # Step 5: Temperature Scaling (Sharpening)
        # [CHANGE v17] Temperature warmup: start with T=1.0, then apply configured temperature
        # This helps stabilize early training before applying sharpening
        T = 1.0
        if hasattr(self, 'global_step') and self.global_step >= self.gating_temperature_warmup_steps:
            T = self.gating_temperature
        elif self.gating_temperature_warmup_steps == 0:
            # No warmup configured, use temperature directly
            T = self.gating_temperature

        if T != 1.0:
            logits_final = logits_final / T

        return logits_final

    def _construct_53_vocab_logits_dual_heads(self, logits_aa_gl, logits_aa_ngl, logits_mut, alpha):
        """
        [v35/v36/v38] Construct 53-vocab logits using SEPARATE GL and NGL AA heads.

        Supports both 2-class (binary) and 3-class (GL/SynNGL/NGL) origin heads.

        Args:
            logits_aa_gl: [B, L, V] - GL AA head logits (33 vocab)
            logits_aa_ngl: [B, L, V] - NGL AA head logits (33 vocab)
            logits_mut: [B, L] (2-class) or [B, L, 3] (3-class) - Origin head logits
            alpha: [B, L, 1] - Per-position gating weights in [0, 1]

        Returns:
            logits_final: [B, L, 53] - LOG PROBABILITIES (do NOT apply log_softmax again!)
        """
        B, L, V_orig = logits_aa_gl.shape  # V_orig = 33

        # Step 1: Compute log-probabilities for AA identity from SEPARATE heads
        if self.detach_heads_from_final_loss:
            log_p_aa_gl = F.log_softmax(logits_aa_gl.detach(), dim=-1)
            log_p_aa_ngl = F.log_softmax(logits_aa_ngl.detach(), dim=-1)
        else:
            log_p_aa_gl = F.log_softmax(logits_aa_gl, dim=-1)
            log_p_aa_ngl = F.log_softmax(logits_aa_ngl, dim=-1)

        # Step 2: Compute effective log-probabilities for GL/NGL status
        if self.num_origin_classes == 3:
            log_p = F.log_softmax(logits_mut, dim=-1)  # [B, L, 3]
            log_p_gl_raw = log_p[:, :, 0]
            log_p_syn = log_p[:, :, 1]
            log_p_ngl_raw = log_p[:, :, 2]

            sw = self.synth_weight
            log_sw = torch.log(torch.tensor(sw, device=logits_mut.device, dtype=logits_mut.dtype))
            log_1_minus_sw = torch.log(torch.tensor(1.0 - sw, device=logits_mut.device, dtype=logits_mut.dtype))

            log_p_ngl = torch.logaddexp(log_p_ngl_raw, log_sw + log_p_syn)
            log_p_gl = torch.logaddexp(log_p_gl_raw, log_1_minus_sw + log_p_syn)
        else:
            log_p_ngl = F.logsigmoid(logits_mut)
            log_p_gl = F.logsigmoid(-logits_mut)

        # Expand for broadcasting: [B, L] -> [B, L, 1]
        log_p_ngl = log_p_ngl.unsqueeze(-1)
        log_p_gl = log_p_gl.unsqueeze(-1)

        # Step 3: Initialize final logits tensor
        V_final = 53
        logits_final = torch.full(
            (B, L, V_final),
            fill_value=-1e9,
            device=logits_aa_gl.device,
            dtype=logits_aa_gl.dtype
        )

        # Step 4: Alpha-weighted combination in log-space with SEPARATE heads
        alpha_weighted_log_p_gl = alpha * log_p_gl      # [B, L, 1]
        alpha_weighted_log_p_ngl = alpha * log_p_ngl    # [B, L, 1]

        # GL (Uppercase) logits: use GL AA head
        logits_final[:, :, :V_orig] = log_p_aa_gl + alpha_weighted_log_p_gl

        # NGL (Lowercase) logits: use NGL AA head
        if self.lowercase_aa_token_ids is not None:
            for upper_id, lower_id in self.lowercase_aa_token_ids.items():
                log_p_this_aa_ngl = log_p_aa_ngl[:, :, upper_id]  # [B, L]
                logits_final[:, :, lower_id] = log_p_this_aa_ngl + alpha_weighted_log_p_ngl.squeeze(-1)

        # Step 5: Temperature Scaling (same as v34)
        T = 1.0
        if hasattr(self, 'global_step') and self.global_step >= self.gating_temperature_warmup_steps:
            T = self.gating_temperature
        elif self.gating_temperature_warmup_steps == 0:
            T = self.gating_temperature

        if T != 1.0:
            logits_final = logits_final / T

        return logits_final

    def _training_step_multihead(self, batch, batch_idx):
        """
        Training step for multihead architecture (AA + Mutation heads).

        [MODIFIED v6.0] Now supports alpha gating with 53-vocab final loss.

        Batch format from make_collate_fn_multihead:
        - input_ids: [B, L] - All uppercase token IDs
        - labels_aa: [B, L] - Uppercase AA labels (-100 for non-masked)
        - labels_mut: [B, L] - Binary mutation labels (-1.0 for non-masked)
        - attention_mask: [B, L]
        - ngl_masks: [B, L]
        - (optional) v_gene_ids, j_gene_ids, region_ids

        Loss Structure (when use_alpha_gating=True):
            Total Loss = final_loss_weight * L_53 + aa_loss_weight * L_33 + mut_loss_weight * L_origin
        """
        # [v37] Extract coherence_flags if present (last element is scalar when coherence masking is on)
        coherence_flag = None
        batch = list(batch)
        if len(batch) > 5 and batch[-1].dim() == 0:
            coherence_flag = batch.pop()

        # [v40] Extract synth_masks and mpnn_gl_probs if present
        # These are the last 2 elements when use_synth_masking is active (before coherence flag)
        synth_masks = None
        mpnn_gl_probs = None
        if len(batch) >= 7 and batch[-1].dim() == 2 and batch[-2].dim() == 2:
            # Check if the last two elements are synth_masks (bool/int) and mpnn_gl_probs (float)
            if batch[-1].dtype == torch.float32 and (batch[-2].dtype == torch.bool or batch[-2].dtype in (torch.int32, torch.int64)):
                mpnn_gl_probs = batch.pop()
                synth_masks = batch.pop()

        # Unpack batch based on enabled features
        v_gene_ids = None
        j_gene_ids = None
        region_ids = None

        if self.use_germline_genes and self.use_region_embedding and len(batch) == 8:
            input_ids, labels_aa, labels_mut, attn_mask, ngl_masks, v_gene_ids, j_gene_ids, region_ids = batch
            v_gene_ids = v_gene_ids.to(self.device)
            j_gene_ids = j_gene_ids.to(self.device)
            region_ids = region_ids.to(self.device)
        elif self.use_germline_genes and len(batch) == 7:
            input_ids, labels_aa, labels_mut, attn_mask, ngl_masks, v_gene_ids, j_gene_ids = batch
            v_gene_ids = v_gene_ids.to(self.device)
            j_gene_ids = j_gene_ids.to(self.device)
        elif self.use_region_embedding and len(batch) == 6:
            input_ids, labels_aa, labels_mut, attn_mask, ngl_masks, region_ids = batch
            region_ids = region_ids.to(self.device)
        else:
            input_ids, labels_aa, labels_mut, attn_mask, ngl_masks = batch

        input_ids = input_ids.to(self.device)
        labels_aa = labels_aa.to(self.device)
        labels_mut = labels_mut.to(self.device)
        attn_mask = attn_mask.to(self.device)
        ngl_masks = ngl_masks.to(self.device)
        if synth_masks is not None:
            synth_masks = synth_masks.to(self.device)
        if mpnn_gl_probs is not None:
            mpnn_gl_probs = mpnn_gl_probs.to(self.device)

        # Forward pass through multihead architecture
        logits_aa, logits_aa_ngl, logits_mut, alpha, logits_final, _ = self._forward_multihead(
            input_ids=input_ids,
            attention_mask=attn_mask,
            v_gene_ids=v_gene_ids,
            j_gene_ids=j_gene_ids,
            region_ids=region_ids  # [v24]
        )

        # =====================================================================
        # Task 1: AA Identity Loss (33 vocab) - Focal Loss
        # [NEW v8.0] Exclude NGL positions from AA loss to prevent gradient conflict
        # AA Head learns only on GL positions (uppercase targets)
        # NGL prediction is handled exclusively by Final Head (53 vocab)
        # [CHANGE v17] Set exclude_ngl=True to completely remove NGL from AA loss
        # [NEW v18] Soft AA Learning: If aa_loss_ngl_weight > 0, include NGL with reduced weight
        # [NEW v35/v36] When dual heads: GL AA head trains on GL-only, NGL AA head on NGL-only
        # =====================================================================
        # [v40 Approach D] MPNN-weighted GL head loss
        gl_position_weights = None
        if self.use_mpnn_gl_weighting and mpnn_gl_probs is not None:
            gl_position_weights = torch.clamp(mpnn_gl_probs, min=self.mpnn_min_weight)
            gl_position_weights[ngl_masks > 0] = 1.0  # Don't weight NGL positions

        if self.use_dual_aa_heads:
            # v35/v36: GL AA head trains only on GL positions
            loss_aa = self._focal_loss(
                logits_aa, labels_aa,
                gamma=self.aa_focal_gamma,
                ignore_index=-100,
                ngl_mask=ngl_masks,
                exclude_ngl=True,  # GL head: GL positions only
                position_weights=gl_position_weights,
            )
        elif self.aa_loss_ngl_weight > 0.0:
            # [NEW v18] Soft AA Learning: Include NGL positions with reduced weight
            # This allows the AA Head to learn NGL tokens as "soft targets" for affinity optimization
            loss_aa = self._focal_loss(
                logits_aa, labels_aa,
                gamma=self.aa_focal_gamma,
                ignore_index=-100,
                ngl_mask=ngl_masks,
                ngl_alpha=self.aa_loss_ngl_weight,  # Use reduced weight for NGL positions
                exclude_ngl=False  # Include NGL positions with soft weighting
            )
        else:
            # [CHANGE v17] Default behavior: Exclude NGL positions - AA Head is GL-only
            loss_aa = self._focal_loss(
                logits_aa, labels_aa,
                gamma=self.aa_focal_gamma,
                ignore_index=-100,
                ngl_mask=ngl_masks,
                exclude_ngl=True  # Exclude NGL positions completely
            )

        # [NEW v35/v36] NGL AA Head Loss - trains only on NGL positions
        # [NEW v35.1b] Apply label smoothing to prevent NGL head overconfidence
        loss_aa_ngl = torch.tensor(0.0, device=self.device)
        if self.use_dual_aa_heads and logits_aa_ngl is not None:
            loss_aa_ngl = self._focal_loss(
                logits_aa_ngl, labels_aa,
                gamma=self.aa_focal_gamma,
                ignore_index=-100,
                ngl_mask=ngl_masks,
                exclude_gl=True,  # NGL head: NGL positions only
                label_smoothing=self.ngl_label_smoothing,
            )

        # =====================================================================
        # Task 2: Origin/Mutation Detection Loss
        # [v38] 3-class uses multiclass focal loss; 2-class uses binary focal loss
        # =====================================================================
        if self.num_origin_classes == 3:
            # [v38] 3-class: labels_mut is long tensor, ignore_index=-1
            loss_origin = self._multiclass_focal_loss(
                logits_mut, labels_mut,
                gamma=self.origin_focal_gamma,
                class_weights=self.origin_class_weights,
                ignore_index=-1,
            )
        else:
            # 2-class binary: labels_mut is float tensor, ignore_index=-1.0
            valid_mask = (labels_mut != -1.0)
            if valid_mask.any():
                logits_mut_valid = logits_mut[valid_mask]
                labels_mut_valid = labels_mut[valid_mask]
                loss_origin = self._binary_focal_loss(
                    logits_mut_valid,
                    labels_mut_valid,
                    gamma=self.origin_focal_gamma,
                    alpha=0.5,
                    ignore_index=None
                )
            else:
                loss_origin = torch.tensor(0.0, device=self.device)

        # =====================================================================
        # [NEW v6.0] Task 3: Final 53-vocab Loss (if alpha gating enabled)
        # [NEW v7.0] Also apply NGL reweighting to 53-vocab loss
        # [NEW v25] Option to use region-balanced loss for FR/CDR equilibrium
        # =====================================================================
        loss_final = torch.tensor(0.0, device=self.device)

        if self.use_alpha_gating and logits_final is not None:
            # Construct 53-vocab labels from labels_aa and ngl_masks
            # For GL positions: use uppercase label
            # For NGL positions: map to lowercase label
            labels_53 = self._construct_53_vocab_labels(labels_aa, ngl_masks)

            # [NEW v25] Use region-balanced loss if enabled and region_ids available
            if self.use_region_balanced_loss and region_ids is not None:
                loss_final = self._focal_loss_region_balanced(
                    logits_final, labels_53,
                    gamma=2.0,
                    ignore_index=-100,
                    ngl_mask=ngl_masks,
                    region_ids=region_ids
                )
            else:
                # Compute final loss using focal loss with NGL reweighting
                loss_final = self._focal_loss(
                    logits_final, labels_53,
                    gamma=2.0,
                    ignore_index=-100,
                    ngl_mask=ngl_masks  # [NEW v7.0] NGL reweighting
                )

        # =====================================================================
        # Combined Loss
        # [v35/v36] Add NGL AA head loss when dual heads are enabled
        # =====================================================================
        if self.use_alpha_gating:
            total_loss = (self.final_loss_weight * loss_final +
                         self.aa_loss_weight * loss_aa +
                         self.mut_loss_weight * loss_origin)
        else:
            total_loss = (self.aa_loss_weight * loss_aa) + (self.mut_loss_weight * loss_origin)

        # [v35/v36] Add NGL AA head loss
        # [v35.1] Use ngl_aa_loss_weight for asymmetric NGL head training
        if self.use_dual_aa_heads:
            total_loss = total_loss + self.ngl_aa_loss_weight * loss_aa_ngl

        # =====================================================================
        # [NEW v37] SHM-Based Sample Weighting
        # Upweight NGL loss for sequences with more NGL positions (higher SHM)
        # [v38] In 3-class mode, count both SynNGL(1) + NGL(2) as mutation positions
        # =====================================================================
        shm_weight = torch.tensor(1.0, device=self.device)
        if self.num_origin_classes == 3:
            mean_ngl_count = (ngl_masks > 0).sum(dim=1).float().mean()
        else:
            mean_ngl_count = ngl_masks.sum(dim=1).float().mean()

        if self.use_shm_weighting and self.use_dual_aa_heads:
            shm_weight = 1.0 + self.shm_beta * (mean_ngl_count / self.shm_mean_ngl)
            # Apply SHM weight only to NGL-related losses (not GL losses)
            total_loss = total_loss + (shm_weight - 1.0) * self.ngl_aa_loss_weight * loss_aa_ngl

        # =====================================================================
        # [NEW v37] GL-NGL Divergence Loss
        # Maximize KL divergence between GL and NGL heads at NGL positions
        # =====================================================================
        loss_divergence = torch.tensor(0.0, device=self.device)

        if (self.divergence_loss_weight > 0.0 and self.use_dual_aa_heads
                and logits_aa_ngl is not None
                and self.global_step >= self.divergence_warmup_steps):
            loss_divergence = self._compute_divergence_loss(
                logits_aa, logits_aa_ngl, ngl_masks, labels_aa,
                max_kl_divergence=self.max_kl_divergence,
                divergence_type=self.divergence_type,
                synth_masks=synth_masks if self.use_synth_divergence else None,
                synth_div_weight=self.synth_div_weight,
            )
            total_loss = total_loss + self.divergence_loss_weight * loss_divergence

        # Logging
        batch_size = input_ids.size(0)
        self.log('train/loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        self.log('train/AA_Loss', loss_aa, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=batch_size)
        self.log('train/Origin_Loss', loss_origin, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=batch_size)

        # [v35/v36] Log NGL AA head loss
        if self.use_dual_aa_heads:
            self.log('train/NGL_AA_Loss', loss_aa_ngl, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=batch_size)

        # [v37] Log divergence loss
        if self.divergence_loss_weight > 0.0 and self.use_dual_aa_heads:
            self.log('train/Divergence_Loss', loss_divergence, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=batch_size)

        # [v37] Log SHM weighting metrics
        if self.use_shm_weighting:
            self.log('train/SHM_Weight', shm_weight, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=batch_size)
            self.log('train/Mean_NGL_Count', mean_ngl_count, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=batch_size)

        if self.use_alpha_gating:
            self.log('train/Final_Loss', loss_final, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=batch_size)
            # Log mean alpha for monitoring
            if alpha is not None:
                mean_alpha = alpha.mean()
                self.log('train/Mean_Alpha', mean_alpha, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=batch_size)

            # [NEW v25] Log region-balanced loss components for monitoring
            if self.use_region_balanced_loss and region_ids is not None:
                self.log('train/Region_Balanced', torch.tensor(1.0), on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=batch_size)

        return total_loss

    def _construct_53_vocab_labels(self, labels_aa, ngl_masks):
        """
        Construct 53-vocab labels from 33-vocab AA labels and NGL masks.

        [NEW v6.0] Maps uppercase AA labels to lowercase for NGL positions.

        Args:
            labels_aa: [B, L] - Uppercase AA labels (-100 for non-masked)
            ngl_masks: [B, L] - Binary NGL mask (0=GL, 1=NGL)

        Returns:
            labels_53: [B, L] - 53-vocab labels with lowercase for NGL positions
        """
        labels_53 = labels_aa.clone()

        if self.lowercase_aa_token_ids is not None:
            # Map NGL positions to lowercase labels
            for upper_id, lower_id in self.lowercase_aa_token_ids.items():
                # Where label == upper_id AND position is NGL, change to lower_id
                # [FIX v39] Use > 0 for 3-class compatibility (SynNGL + NGL → lowercase)
                ngl_and_upper = (labels_53 == upper_id) & (ngl_masks > 0)
                labels_53[ngl_and_upper] = lower_id

        return labels_53

    def _validation_step_multihead(self, batch, batch_idx):
        """
        Validation step for multihead architecture.

        [MODIFIED v6.0] Comprehensive metrics for all three heads (Origin, AA, Final).

        =====================================================================
        METRIC GROUPS (as per requirements):
        =====================================================================

        Group A: Mutation/Origin Head (GL/NGL Classification)
        - val/Origin_F1: F1 Score for GL/NGL classification
        - val/Origin_PR_AUC: Precision-Recall AUC
        - val/Origin_Loss: Origin head Focal Loss

        Group B: AA Head (33 Vocab)
        - val/AA_Loss: AA head Focal Loss
        - val/AA_PPL_All: Perplexity over all masked tokens
        - val/AA_PPL_NGL: Perplexity for NGL positions only

        Group C: Final Head (53 Vocab - Alpha-combined)
        - val/Final_Loss: Final 53-vocab Focal Loss
        - val/Final_PPL_All: Perplexity over all masked tokens
        - val/Final_PPL_NGL: Perplexity for NGL positions (PRIORITY METRIC)

        Additional:
        - val/Mean_Alpha: Average alpha value for monitoring gating behavior

        =====================================================================
        IMPORTANT: PPL is calculated using CrossEntropy (base e), NOT Focal Loss
        =====================================================================
        """
        # [v37] Extract coherence_flags if present
        batch = list(batch)
        if len(batch) > 5 and batch[-1].dim() == 0:
            batch.pop()  # Remove coherence_flag (not used in validation metrics directly)

        # [v40] Extract synth_masks and mpnn_gl_probs if present (not used in val, just discard)
        if len(batch) >= 7 and batch[-1].dim() == 2 and batch[-2].dim() == 2:
            if batch[-1].dtype == torch.float32 and (batch[-2].dtype == torch.bool or batch[-2].dtype in (torch.int32, torch.int64)):
                batch.pop()  # mpnn_gl_probs
                batch.pop()  # synth_masks

        # Unpack batch
        v_gene_ids = None
        j_gene_ids = None
        region_ids = None

        if self.use_germline_genes and self.use_region_embedding and len(batch) == 8:
            input_ids, labels_aa, labels_mut, attn_mask, ngl_masks, v_gene_ids, j_gene_ids, region_ids = batch
            v_gene_ids = v_gene_ids.to(self.device)
            j_gene_ids = j_gene_ids.to(self.device)
            region_ids = region_ids.to(self.device)
        elif self.use_germline_genes and len(batch) == 7:
            input_ids, labels_aa, labels_mut, attn_mask, ngl_masks, v_gene_ids, j_gene_ids = batch
            v_gene_ids = v_gene_ids.to(self.device)
            j_gene_ids = j_gene_ids.to(self.device)
        elif self.use_region_embedding and len(batch) == 6:
            input_ids, labels_aa, labels_mut, attn_mask, ngl_masks, region_ids = batch
            region_ids = region_ids.to(self.device)
        else:
            input_ids, labels_aa, labels_mut, attn_mask, ngl_masks = batch

        input_ids = input_ids.to(self.device)
        labels_aa = labels_aa.to(self.device)
        labels_mut = labels_mut.to(self.device)
        attn_mask = attn_mask.to(self.device)
        ngl_masks = ngl_masks.to(self.device)

        batch_size = input_ids.size(0)

        with torch.no_grad():
            # Forward pass with alpha gating
            logits_aa, logits_aa_ngl, logits_mut, alpha, logits_final, _ = self._forward_multihead(
                input_ids=input_ids,
                attention_mask=attn_mask,
                v_gene_ids=v_gene_ids,
                j_gene_ids=j_gene_ids,
                region_ids=region_ids  # [v24]
            )

            # Common masks
            # [v38] Valid mask depends on dtype: int(-1) for 3-class, float(-1.0) for 2-class
            if self.num_origin_classes == 3:
                valid_mask = (labels_mut != -1)
            else:
                valid_mask = (labels_mut != -1.0)
            masked_pos = (labels_aa.view(-1) != -100)  # Valid AA label positions

            # Flatten tensors for metric computation
            labels_aa_flat = labels_aa.view(-1)  # [B*L]
            ngl_masks_flat = ngl_masks.view(-1)  # [B*L]
            # [v38] In 3-class mode, NGL positions have value > 0 (SynNGL=1, NGL=2)
            if self.num_origin_classes == 3:
                ngl_subset_mask = masked_pos & (ngl_masks_flat > 0)
            else:
                ngl_subset_mask = masked_pos & (ngl_masks_flat == 1)

            # =================================================================
            # GROUP A: Origin Head Metrics (GL/NGL Classification)
            # =================================================================
            origin_loss = torch.tensor(0.0, device=self.device)
            origin_f1 = torch.tensor(0.0, device=self.device)
            origin_pr_auc = torch.tensor(0.0, device=self.device)

            if valid_mask.any():
                if self.num_origin_classes == 3:
                    # [v38] 3-class origin loss
                    origin_loss = self._multiclass_focal_loss(
                        logits_mut, labels_mut,
                        gamma=self.origin_focal_gamma,
                        class_weights=self.origin_class_weights,
                        ignore_index=-1,
                    )
                    # For F1: binarize as "any mutation" (class > 0) vs GL (class 0)
                    valid_labels = labels_mut[valid_mask]
                    valid_logits = logits_mut[valid_mask]  # [N, 3]
                    probs_3class = F.softmax(valid_logits, dim=-1)
                    # P(any mutation) = P(SynNGL) + P(NGL)
                    probs = probs_3class[:, 1] + probs_3class[:, 2]
                    preds = (probs > 0.5).long()
                    targets = (valid_labels > 0).long()
                else:
                    # 2-class binary origin loss
                    origin_loss = self._binary_focal_loss(
                        logits_mut[valid_mask],
                        labels_mut[valid_mask],
                        gamma=self.origin_focal_gamma,
                        alpha=0.5,
                        ignore_index=None
                    )
                    probs = torch.sigmoid(logits_mut[valid_mask])
                    preds = (probs > 0.5).long()
                    targets = (labels_mut[valid_mask] > 0.5).long()

                # F1 Score
                tp = ((preds == 1) & (targets == 1)).sum().float()
                fp = ((preds == 1) & (targets == 0)).sum().float()
                fn = ((preds == 0) & (targets == 1)).sum().float()

                precision = tp / (tp + fp + 1e-8)
                recall = tp / (tp + fn + 1e-8)
                origin_f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

                # PR-AUC (Precision-Recall AUC approximation)
                # Using a simplified trapezoidal approximation
                origin_pr_auc = self._compute_pr_auc(probs, targets)

            # =================================================================
            # GROUP B: AA Head Metrics (33 Vocab)
            # [NEW v8.0] Exclude NGL positions from AA loss to match training
            # =================================================================
            # AA Loss (Focal) - GL positions only
            aa_loss = self._focal_loss(
                logits_aa, labels_aa,
                gamma=self.aa_focal_gamma,
                ignore_index=-100,
                ngl_mask=ngl_masks,
                exclude_ngl=True  # [NEW v8.0] Match training behavior
            )

            # AA PPL using CrossEntropy (NOT Focal Loss)
            logits_aa_flat = logits_aa.view(-1, logits_aa.size(-1))  # [B*L, V]

            # [NEW v8.0] GL-only mask for AA metrics (matching training behavior)
            gl_subset_mask = masked_pos & (ngl_masks_flat == 0)

            # AA_PPL_All: Perplexity over all masked tokens (legacy, includes NGL)
            aa_ppl_all = torch.tensor(0.0, device=self.device)
            if masked_pos.any():
                ce_aa_all = F.cross_entropy(logits_aa_flat[masked_pos], labels_aa_flat[masked_pos])
                aa_ppl_all = torch.exp(ce_aa_all)

            # [NEW v8.0] AA_PPL_GL: Perplexity for GL positions only (matches training)
            aa_ppl_gl = torch.tensor(0.0, device=self.device)
            if gl_subset_mask.any():
                ce_aa_gl = F.cross_entropy(logits_aa_flat[gl_subset_mask], labels_aa_flat[gl_subset_mask])
                aa_ppl_gl = torch.exp(ce_aa_gl)

            # AA_PPL_NGL: Perplexity for NGL positions (for reference only - not trained on)
            aa_ppl_ngl = torch.tensor(0.0, device=self.device)
            if ngl_subset_mask.any():
                ce_aa_ngl = F.cross_entropy(logits_aa_flat[ngl_subset_mask], labels_aa_flat[ngl_subset_mask])
                aa_ppl_ngl = torch.exp(ce_aa_ngl)

            # [NEW v18] AA_PPL_NGL_Weighted: Weighted average PPL when using soft AA learning
            # This monitors how well the AA Head accepts mutations when aa_loss_ngl_weight > 0
            aa_ppl_ngl_weighted = torch.tensor(0.0, device=self.device)
            if self.aa_loss_ngl_weight > 0.0 and gl_subset_mask.any() and ngl_subset_mask.any():
                # Compute weighted average: GL weight = 1.0, NGL weight = aa_loss_ngl_weight
                gl_count = gl_subset_mask.sum().float()
                ngl_count = ngl_subset_mask.sum().float()

                # Cross-entropy for GL and NGL separately
                ce_gl = F.cross_entropy(logits_aa_flat[gl_subset_mask], labels_aa_flat[gl_subset_mask], reduction='sum')
                ce_ngl = F.cross_entropy(logits_aa_flat[ngl_subset_mask], labels_aa_flat[ngl_subset_mask], reduction='sum')

                # Weighted average loss
                total_weight = gl_count + self.aa_loss_ngl_weight * ngl_count
                weighted_ce = (ce_gl + self.aa_loss_ngl_weight * ce_ngl) / (total_weight + 1e-8)
                aa_ppl_ngl_weighted = torch.exp(weighted_ce)

            # =================================================================
            # [NEW v35/v36] GROUP B2: NGL AA Head Metrics (33 Vocab)
            # =================================================================
            ngl_aa_ppl_all = torch.tensor(0.0, device=self.device)
            ngl_aa_ppl_ngl = torch.tensor(0.0, device=self.device)
            ngl_aa_loss = torch.tensor(0.0, device=self.device)

            if self.use_dual_aa_heads and logits_aa_ngl is not None:
                # NGL AA Loss (Focal) - NGL positions only
                ngl_aa_loss = self._focal_loss(
                    logits_aa_ngl, labels_aa,
                    gamma=self.aa_focal_gamma,
                    ignore_index=-100,
                    ngl_mask=ngl_masks,
                    exclude_gl=True
                )

                logits_aa_ngl_flat = logits_aa_ngl.view(-1, logits_aa_ngl.size(-1))  # [B*L, V]

                # NGL_AA_PPL_All: NGL head perplexity over all masked tokens
                if masked_pos.any():
                    ce_ngl_aa_all = F.cross_entropy(logits_aa_ngl_flat[masked_pos], labels_aa_flat[masked_pos])
                    ngl_aa_ppl_all = torch.exp(ce_ngl_aa_all)

                # NGL_AA_PPL_NGL: NGL head perplexity on NGL positions only
                if ngl_subset_mask.any():
                    ce_ngl_aa_ngl = F.cross_entropy(logits_aa_ngl_flat[ngl_subset_mask], labels_aa_flat[ngl_subset_mask])
                    ngl_aa_ppl_ngl = torch.exp(ce_ngl_aa_ngl)

            # =================================================================
            # [NEW v37] GROUP B3: GL-NGL Divergence Metrics
            # =================================================================
            gl_ngl_kl = torch.tensor(0.0, device=self.device)
            divergence_at_cdr = torch.tensor(0.0, device=self.device)
            divergence_at_fr = torch.tensor(0.0, device=self.device)
            val_mean_ngl_count = torch.tensor(0.0, device=self.device)

            # Mean NGL count per sample
            val_mean_ngl_count = ngl_masks.sum(dim=1).float().mean()

            if self.use_dual_aa_heads and logits_aa_ngl is not None:
                # Compute overall GL-NGL KL at NGL positions
                kl_loss = self._compute_divergence_loss(
                    logits_aa, logits_aa_ngl, ngl_masks, labels_aa,
                    max_kl_divergence=self.max_kl_divergence,
                    divergence_type="kl",
                )
                gl_ngl_kl = -kl_loss  # Negate to get positive KL value

                # Region-specific divergence (if region_ids available)
                if region_ids is not None:
                    region_ids_flat_v37 = region_ids.view(-1)
                    # CDR: {1, 3, 5}
                    cdr_mask_v37 = (region_ids == 1) | (region_ids == 3) | (region_ids == 5)
                    cdr_ngl_mask = ngl_masks * cdr_mask_v37.float()
                    if cdr_ngl_mask.any():
                        kl_cdr = self._compute_divergence_loss(
                            logits_aa, logits_aa_ngl, cdr_ngl_mask, labels_aa,
                            max_kl_divergence=self.max_kl_divergence,
                            divergence_type="kl",
                        )
                        divergence_at_cdr = -kl_cdr

                    # FR: {0, 2, 4, 6}
                    fr_mask_v37 = (region_ids == 0) | (region_ids == 2) | (region_ids == 4) | (region_ids == 6)
                    fr_ngl_mask = ngl_masks * fr_mask_v37.float()
                    if fr_ngl_mask.any():
                        kl_fr = self._compute_divergence_loss(
                            logits_aa, logits_aa_ngl, fr_ngl_mask, labels_aa,
                            max_kl_divergence=self.max_kl_divergence,
                            divergence_type="kl",
                        )
                        divergence_at_fr = -kl_fr

            # =================================================================
            # GROUP C: Final Head Metrics (53 Vocab - Alpha-combined)
            # =================================================================
            final_loss = torch.tensor(0.0, device=self.device)
            final_ppl_all = torch.tensor(0.0, device=self.device)
            final_ppl_ngl = torch.tensor(0.0, device=self.device)
            mean_alpha = torch.tensor(0.5, device=self.device)

            # [NEW v7.0] Initialize new NGL PPL metrics
            max_ppl_ngl = torch.tensor(0.0, device=self.device)
            worst_10_pct_ppl_ngl = torch.tensor(0.0, device=self.device)

            if self.use_alpha_gating and logits_final is not None:
                # Construct 53-vocab labels
                labels_53 = self._construct_53_vocab_labels(labels_aa, ngl_masks)
                labels_53_flat = labels_53.view(-1)

                # Final Loss (Focal)
                final_loss = self._focal_loss(logits_final, labels_53, gamma=2.0, ignore_index=-100)

                # Final PPL using CrossEntropy
                logits_final_flat = logits_final.view(-1, logits_final.size(-1))  # [B*L, 53]

                # Final_PPL_All: Perplexity over all masked tokens
                if masked_pos.any():
                    ce_final_all = F.cross_entropy(logits_final_flat[masked_pos], labels_53_flat[masked_pos])
                    final_ppl_all = torch.exp(ce_final_all)

                # Final_PPL_NGL: Perplexity for NGL positions (PRIORITY METRIC)
                if ngl_subset_mask.any():
                    ce_final_ngl = F.cross_entropy(logits_final_flat[ngl_subset_mask], labels_53_flat[ngl_subset_mask])
                    final_ppl_ngl = torch.exp(ce_final_ngl)

                    # =================================================================
                    # [NEW v7.0] Compute Max_PPL_NGL and Worst_10_Percent_PPL_NGL
                    # These metrics help identify and monitor worst-case NGL predictions
                    # =================================================================
                    # Compute per-token cross-entropy loss (reduction='none')
                    ce_per_token = F.cross_entropy(
                        logits_final_flat[ngl_subset_mask],
                        labels_53_flat[ngl_subset_mask],
                        reduction='none'
                    )  # [num_ngl_tokens]

                    # Clamp loss for numerical stability (max PPL ~ exp(20) ~ 485 million)
                    ce_per_token_clamped = torch.clamp(ce_per_token, max=20.0)

                    # Per-token PPL
                    ppl_per_token = torch.exp(ce_per_token_clamped)  # [num_ngl_tokens]

                    # Max PPL NGL: Worst single token prediction
                    max_ppl_ngl = ppl_per_token.max()

                    # Worst 10% PPL NGL: Average of top 10% worst predictions
                    num_tokens = ppl_per_token.size(0)
                    top_k = max(1, int(num_tokens * 0.1))  # At least 1 token
                    top_k_values, _ = torch.topk(ppl_per_token, k=top_k)
                    worst_10_pct_ppl_ngl = top_k_values.mean()

                # Mean Alpha for monitoring
                if alpha is not None:
                    mean_alpha = alpha.mean()

            # =================================================================
            # [NEW v17/v19] Region-wise PPL and Alpha Metrics (CDR vs FR)
            # Region IDs: 0=special, 1=FR1, 2=CDR1, 3=FR2, 4=CDR2, 5=FR3, 6=CDR3, 7=FR4
            # CDR = {2, 4, 6}, FR = {1, 3, 5, 7}
            # =================================================================
            final_ppl_cdr = torch.tensor(0.0, device=self.device)
            final_ppl_fr = torch.tensor(0.0, device=self.device)
            # [NEW v19] Region-wise Alpha statistics
            alpha_mean_cdr = torch.tensor(0.5, device=self.device)
            alpha_std_cdr = torch.tensor(0.0, device=self.device)
            alpha_mean_fr = torch.tensor(0.5, device=self.device)
            alpha_std_fr = torch.tensor(0.0, device=self.device)

            if region_ids is not None:
                region_ids_flat = region_ids.view(-1)  # [B*L]

                # [FIX v26] Corrected Region ID mapping: 0=FR1, 1=CDR1, 2=FR2, 3=CDR2, 4=FR3, 5=CDR3, 6=FR4
                # CDR mask: region_ids in {1, 3, 5} (CDR1, CDR2, CDR3)
                cdr_mask = (region_ids_flat == 1) | (region_ids_flat == 3) | (region_ids_flat == 5)
                cdr_subset_mask = masked_pos & cdr_mask

                # FR mask: region_ids in {0, 2, 4, 6} (FR1, FR2, FR3, FR4)
                fr_mask = (region_ids_flat == 0) | (region_ids_flat == 2) | (region_ids_flat == 4) | (region_ids_flat == 6)
                fr_subset_mask = masked_pos & fr_mask

                if self.use_alpha_gating and logits_final is not None:
                    logits_final_flat = logits_final.view(-1, logits_final.size(-1))
                    labels_53_flat = labels_53.view(-1)

                    # Final_PPL_CDR: Perplexity on CDR positions
                    if cdr_subset_mask.any():
                        ce_final_cdr = F.cross_entropy(logits_final_flat[cdr_subset_mask], labels_53_flat[cdr_subset_mask])
                        final_ppl_cdr = torch.exp(ce_final_cdr)

                    # Final_PPL_FR: Perplexity on FR positions
                    if fr_subset_mask.any():
                        ce_final_fr = F.cross_entropy(logits_final_flat[fr_subset_mask], labels_53_flat[fr_subset_mask])
                        final_ppl_fr = torch.exp(ce_final_fr)
                # =============================================================
                # [NEW v19] Region-wise Alpha Statistics
                # Compute Mean and Std of Alpha separately for CDR and FR regions
                # Alpha indicates model's confidence in mutation vs germline
                # Alpha ~0.5 is "indecisive", ~0 is confident GL, ~1 is confident NGL
                # =============================================================
                if self.use_alpha_gating and alpha is not None:
                    alpha_flat = alpha.view(-1)  # [B*L]

                    # Alpha statistics for CDR regions
                    if cdr_subset_mask.any():
                        alpha_cdr = alpha_flat[cdr_subset_mask]
                        alpha_mean_cdr = alpha_cdr.mean()
                        alpha_std_cdr = alpha_cdr.std() if alpha_cdr.numel() > 1 else torch.tensor(0.0, device=self.device)

                    # Alpha statistics for FR regions
                    if fr_subset_mask.any():
                        alpha_fr = alpha_flat[fr_subset_mask]
                        alpha_mean_fr = alpha_fr.mean()
                        alpha_std_fr = alpha_fr.std() if alpha_fr.numel() > 1 else torch.tensor(0.0, device=self.device)

        # =====================================================================
        # LOGGING - Organized by Group
        # [CHANGE v17] Metric organization and documentation
        # =====================================================================

        # -------------------------------------------------------------------------
        # Group A: Origin Head Metrics (GL/NGL Classification)
        # - Origin_F1: F1 score for mutation detection (KEY METRIC)
        # - Origin_PR_AUC: Precision-Recall AUC for imbalanced classification
        # - Origin_Loss: Focal loss for Origin Head training
        # [DDP] sync_dist=True ensures metrics are aggregated across all GPUs
        # -------------------------------------------------------------------------
        self.log('val/Origin_F1', origin_f1, on_step=False, on_epoch=True, prog_bar=False, batch_size=batch_size, sync_dist=True)
        self.log('val/Origin_PR_AUC', origin_pr_auc, on_step=False, on_epoch=True, prog_bar=False, batch_size=batch_size, sync_dist=True)
        self.log('val/Origin_Loss', origin_loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=batch_size, sync_dist=True)

        # -------------------------------------------------------------------------
        # Group B: AA Head Metrics (33 Vocab - Uppercase Only)
        # [CHANGE v17] AA_PPL_GL is the PRIMARY metric for AA Head since it only trains on GL
        # - AA_PPL_GL: Perplexity on GL (germline) positions - MAIN AA HEAD METRIC
        # - AA_PPL_All/AA_PPL_NGL: Auxiliary/reference metrics only
        # -------------------------------------------------------------------------
        self.log('val/AA_Loss', aa_loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=batch_size, sync_dist=True)
        # [CHANGE v17] AA_PPL_GL is the main AA head metric (exclude_ngl=True in training)
        self.log('val/AA_PPL_GL', aa_ppl_gl, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True)  # PRIMARY AA METRIC
        self.log('val/AA_PPL_All', aa_ppl_all, on_step=False, on_epoch=True, prog_bar=False, batch_size=batch_size, sync_dist=True)  # Auxiliary
        self.log('val/AA_PPL_NGL', aa_ppl_ngl, on_step=False, on_epoch=True, prog_bar=False, batch_size=batch_size, sync_dist=True)  # Reference only
        # [NEW v18] AA_PPL_NGL_Weighted: Weighted PPL for monitoring soft AA learning
        if self.aa_loss_ngl_weight > 0.0:
            self.log('val/AA_PPL_NGL_Weighted', aa_ppl_ngl_weighted, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True)

        # -------------------------------------------------------------------------
        # [v35/v36] Group B2: NGL AA Head Metrics (33 Vocab)
        # -------------------------------------------------------------------------
        if self.use_dual_aa_heads:
            self.log('val/NGL_AA_Loss', ngl_aa_loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=batch_size, sync_dist=True)
            self.log('val/NGL_AA_PPL_All', ngl_aa_ppl_all, on_step=False, on_epoch=True, prog_bar=False, batch_size=batch_size, sync_dist=True)
            self.log('val/NGL_AA_PPL_NGL', ngl_aa_ppl_ngl, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True)

        # -------------------------------------------------------------------------
        # [v37] Group B3: GL-NGL Divergence Metrics
        # -------------------------------------------------------------------------
        if self.use_dual_aa_heads:
            self.log('val/GL_NGL_KL', gl_ngl_kl, on_step=False, on_epoch=True, prog_bar=False, batch_size=batch_size, sync_dist=True)
            if region_ids is not None:
                self.log('val/Divergence_At_CDR', divergence_at_cdr, on_step=False, on_epoch=True, prog_bar=False, batch_size=batch_size, sync_dist=True)
                self.log('val/Divergence_At_FR', divergence_at_fr, on_step=False, on_epoch=True, prog_bar=False, batch_size=batch_size, sync_dist=True)
        self.log('val/Mean_NGL_Count', val_mean_ngl_count, on_step=False, on_epoch=True, prog_bar=False, batch_size=batch_size, sync_dist=True)

        # [v37] NGL Coherence PPL (placeholder - computed when coherence masking is active)
        self.log('val/NGL_Coherence_PPL', torch.tensor(0.0, device=self.device), on_step=False, on_epoch=True, prog_bar=False, batch_size=batch_size, sync_dist=True)

        # -------------------------------------------------------------------------
        # Group C: Final Head Metrics (53 Vocab - Alpha-combined)
        # - Final_PPL_NGL: Perplexity on NGL positions - PRIORITY CHECKPOINT METRIC
        # - Final_PPL_All: Overall perplexity
        # -------------------------------------------------------------------------
        self.log('val/Final_Loss', final_loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=batch_size, sync_dist=True)
        self.log('val/Final_PPL_All', final_ppl_all, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True)
        self.log('val/Final_PPL_NGL', final_ppl_ngl, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True)

        # -------------------------------------------------------------------------
        # Group D: NGL Outlier Detection Metrics (53 Vocab)
        # [NEW v7.0] These metrics help identify worst-case predictions
        # -------------------------------------------------------------------------
        self.log('val/Max_PPL_NGL', max_ppl_ngl, on_step=False, on_epoch=True, prog_bar=False, batch_size=batch_size, sync_dist=True)
        self.log('val/Worst_10_Percent_PPL_NGL', worst_10_pct_ppl_ngl, on_step=False, on_epoch=True, prog_bar=False, batch_size=batch_size, sync_dist=True)

        # -------------------------------------------------------------------------
        # Group E: Region-wise PPL Metrics (CDR vs FR)
        # [NEW v17] Requires use_region_embedding=True in config
        # - Final_PPL_CDR: Perplexity on CDR regions (CDR1+CDR2+CDR3)
        # - Final_PPL_FR: Perplexity on FR regions (FR1+FR2+FR3+FR4)
        # -------------------------------------------------------------------------
        if region_ids is not None:
            self.log('val/Final_PPL_CDR', final_ppl_cdr, on_step=False, on_epoch=True, prog_bar=False, batch_size=batch_size, sync_dist=True)
            self.log('val/Final_PPL_FR', final_ppl_fr, on_step=False, on_epoch=True, prog_bar=False, batch_size=batch_size, sync_dist=True)

            # -------------------------------------------------------------------------
            # Group F: Region-wise Alpha Metrics (CDR vs FR) [NEW v19]
            # - Alpha_Mean_CDR/FR: Average alpha value per region
            # - Alpha_Std_CDR/FR: Alpha variance per region (should be high if model is decisive)
            # Alpha ~0.5 is "indecisive", ~0 is confident GL, ~1 is confident NGL
            # -------------------------------------------------------------------------
            if self.use_alpha_gating:
                self.log('val/Alpha_Mean_CDR', alpha_mean_cdr, on_step=False, on_epoch=True, prog_bar=False, batch_size=batch_size, sync_dist=True)
                self.log('val/Alpha_Std_CDR', alpha_std_cdr, on_step=False, on_epoch=True, prog_bar=False, batch_size=batch_size, sync_dist=True)
                self.log('val/Alpha_Mean_FR', alpha_mean_fr, on_step=False, on_epoch=True, prog_bar=False, batch_size=batch_size, sync_dist=True)
                self.log('val/Alpha_Std_FR', alpha_std_fr, on_step=False, on_epoch=True, prog_bar=False, batch_size=batch_size, sync_dist=True)

        # Alpha monitoring
        if self.use_alpha_gating:
            self.log('val/Mean_Alpha', mean_alpha, on_step=False, on_epoch=True, prog_bar=False, batch_size=batch_size, sync_dist=True)

        # Legacy compatibility metrics
        self.log('val/ppl_all', final_ppl_all, on_step=False, on_epoch=True, prog_bar=False, batch_size=batch_size, sync_dist=True)
        self.log('val/ppl_ngl_upper', final_ppl_ngl, on_step=False, on_epoch=True, prog_bar=False, batch_size=batch_size, sync_dist=True)
        self.log('val/f1_case', origin_f1, on_step=False, on_epoch=True, prog_bar=False, batch_size=batch_size, sync_dist=True)

        # Combined loss for return
        if self.use_alpha_gating:
            total_loss = (self.final_loss_weight * final_loss +
                         self.aa_loss_weight * aa_loss +
                         self.mut_loss_weight * origin_loss)
        else:
            total_loss = aa_loss + origin_loss

        return total_loss

    def _compute_pr_auc(self, probs, targets):
        """
        Compute Precision-Recall AUC using a simplified approximation.

        [NEW v6.0] Approximates PR-AUC by computing precision and recall at multiple thresholds.

        Args:
            probs: [N] - Predicted probabilities
            targets: [N] - Binary targets (0 or 1)

        Returns:
            pr_auc: Scalar PR-AUC value
        """
        # Sort by probability descending
        sorted_indices = torch.argsort(probs, descending=True)
        sorted_targets = targets[sorted_indices].float()

        # Compute cumulative TP and FP
        cumsum_tp = torch.cumsum(sorted_targets, dim=0)
        cumsum_fp = torch.cumsum(1 - sorted_targets, dim=0)

        total_positives = targets.sum().float()
        total_negatives = (1 - targets.float()).sum()

        if total_positives == 0:
            return torch.tensor(0.0, device=probs.device)

        # Precision and Recall at each threshold
        n = len(sorted_targets)
        precision = cumsum_tp / torch.arange(1, n + 1, device=probs.device).float()
        recall = cumsum_tp / total_positives

        # Compute AUC using trapezoidal rule
        # Prepend (0, 1) and append (1, 0) for proper integration
        recall_diff = torch.cat([recall[:1], recall[1:] - recall[:-1]])
        pr_auc = (precision * recall_diff).sum()

        return pr_auc

    @staticmethod
    def _swiglu_intermediate_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Patched forward for EsmIntermediate that skips the hardcoded gelu().

        EsmIntermediate.forward() hardcodes gelu() as a direct function call
        after self.dense(). When self.dense is replaced with SwiGLU (which
        already contains SiLU activation internally), the gelu() must be
        skipped to avoid double activation: GELU(SiLU(xW) * xV).

        This patched forward simply returns self.dense(x) without gelu().
        """
        hidden_states = self.dense(hidden_states)
        return hidden_states

    def _replace_gelu_with_swiglu(self, model, num_layers, fix_double_activation=False):
        """
        Replace GELU activation with SwiGLU in ESM2 transformer blocks.

        ESM2 uses a standard FFN structure:
        FFN(x) = Linear2(GELU(Linear1(x)))

        We replace this with:
        FFN(x) = Linear2(SwiGLU(x))

        where SwiGLU(x) = Swish(xW) ⊗ xV

        NOTE on double activation:
        EsmIntermediate.forward() hardcodes gelu() as a direct function call.
        Without the monkey-patch fix, the effective computation is GELU(SwiGLU(x)).
        All checkpoints trained before this fix use GELU(SwiGLU(x)).

        Args:
            fix_double_activation: If True, monkey-patch forward to skip GELU.
                Set False for checkpoint compatibility with pre-fix models.
                Set True only for training from scratch.
        """
        import types

        for layer_idx in range(num_layers):
            # Access the intermediate dense layer in ESM2
            # Path: esm.encoder.layer.{i}.intermediate.dense
            layer = model.esm.encoder.layer[layer_idx]

            if hasattr(layer, 'intermediate') and hasattr(layer.intermediate, 'dense'):
                # Get original dimensions
                original_dense = layer.intermediate.dense
                dim_in = original_dense.in_features
                dim_out = original_dense.out_features
                has_bias = original_dense.bias is not None

                # Create SwiGLU module with same dimensions
                swiglu = SwiGLU(dim_in, dim_out, bias=has_bias)

                # Initialize SwiGLU weights from the original dense layer
                # Copy weights to the gate (w) projection
                with torch.no_grad():
                    swiglu.w.weight.copy_(original_dense.weight)
                    if has_bias:
                        swiglu.w.bias.copy_(original_dense.bias)

                    # Initialize value (v) projection with small random values
                    # This helps with training stability
                    nn.init.xavier_uniform_(swiglu.v.weight, gain=0.1)
                    if has_bias:
                        nn.init.zeros_(swiglu.v.bias)

                # Replace the dense layer with SwiGLU
                layer.intermediate.dense = swiglu

                # Optionally monkey-patch forward to skip the hardcoded gelu() call
                if fix_double_activation:
                    layer.intermediate.forward = types.MethodType(
                        SFT_ESM2._swiglu_intermediate_forward, layer.intermediate
                    )

        activation_mode = "pure SwiGLU" if fix_double_activation else "GELU(SwiGLU) [checkpoint-compatible]"
        print(f"  Replaced dense with SwiGLU in {num_layers} layers, mode: {activation_mode}")

    @torch.no_grad()
    def compute_developability_ppl(
        self,
        sequences: list,
        v_genes: list = None,
        j_genes: list = None,
        mask_prob: float = 0.15,
        num_masks: int = 5,
        batch_size: int = 8,
    ) -> list:
        """
        Compute Final Head Upper PPL for developability assessment.

        Uses masked pseudo-perplexity with the Final Head (53-vocab) but only
        considers uppercase (germline) logits for scoring. This provides a
        "developability PPL" that correlates with expression, stability, etc.

        [v34] This method was added to enable real-time developability tracking
        during training, addressing the observation that AA Head PPL (which
        excludes NGL during training) doesn't correlate with developability.

        Args:
            sequences: List of antibody sequences (concatenated VH+VL)
            v_genes: List of V-gene IDs (optional, for conditioning)
            j_genes: List of J-gene IDs (optional, for conditioning)
            mask_prob: Probability of masking each position (default: 0.15)
            num_masks: Number of random masking rounds for Monte Carlo estimate
            batch_size: Batch size for inference

        Returns:
            List of PPL values, one per sequence
        """
        self.eval()
        device = next(self.parameters()).device

        # Get tokenizer
        tokenizer = self.tokenizer if hasattr(self, 'tokenizer') else \
                    AutoTokenizer.from_pretrained(self.model_identifier)

        all_ppls = []

        # Process in batches
        for batch_start in range(0, len(sequences), batch_size):
            batch_end = min(batch_start + batch_size, len(sequences))
            batch_seqs = sequences[batch_start:batch_end]

            # Tokenize sequences
            encoded = tokenizer(
                batch_seqs,
                padding=True,
                truncation=True,
                max_length=320,
                return_tensors='pt'
            )
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)

            # Handle gene conditioning
            v_gene_ids = None
            j_gene_ids = None
            if self.use_germline_genes and v_genes is not None and j_genes is not None:
                batch_v = v_genes[batch_start:batch_end]
                batch_j = j_genes[batch_start:batch_end]
                # Convert gene names to IDs if they're strings
                if hasattr(self, 'gene_vocab') and self.gene_vocab is not None:
                    v_gene_ids = torch.tensor([
                        self.gene_vocab.get_id(g) for g in batch_v
                    ], device=device)
                    j_gene_ids = torch.tensor([
                        self.gene_vocab.get_id(g) for g in batch_j
                    ], device=device)

            # Identify valid positions (amino acids, not special tokens)
            # ESM2 standard AA tokens are 4-23
            valid_pos_mask = (input_ids >= 4) & (input_ids <= 23) & (attention_mask == 1)

            # Monte Carlo estimation of masked pseudo-perplexity
            batch_log_probs = torch.zeros(input_ids.shape[0], device=device)
            batch_counts = torch.zeros(input_ids.shape[0], device=device)

            for _ in range(num_masks):
                # Create random mask
                mask_rand = torch.rand_like(input_ids.float())
                mask_positions = (mask_rand < mask_prob) & valid_pos_mask

                # Skip if no positions to mask
                if not mask_positions.any():
                    continue

                # Create masked input
                masked_input = input_ids.clone()
                masked_input[mask_positions] = tokenizer.mask_token_id

                # Forward pass through multihead architecture
                logits_aa, _, logits_mut, alpha, logits_final, _ = self._forward_multihead(
                    input_ids=masked_input,
                    attention_mask=attention_mask,
                    v_gene_ids=v_gene_ids,
                    j_gene_ids=j_gene_ids,
                    region_ids=None  # Not needed for developability PPL
                )

                # Use Final Head logits if available, else use AA Head
                if logits_final is not None:
                    # Extract uppercase (GL) logits from 53-vocab
                    # Standard AA tokens are at indices 4-23 in ESM2
                    upper_logits = logits_final[:, :, 4:24]  # [B, L, 20]

                    # Apply log_softmax for probability
                    log_probs = F.log_softmax(upper_logits, dim=-1)
                else:
                    # Fallback to AA Head (33-vocab)
                    log_probs = F.log_softmax(logits_aa[:, :, 4:24], dim=-1)

                # Get log probability of true tokens at masked positions
                # Map original token IDs (4-23) to indices (0-19)
                true_tokens = input_ids - 4  # Shift to 0-19 range
                true_tokens = true_tokens.clamp(0, 19)  # Safety clamp

                # Gather log probs for true tokens
                for b in range(input_ids.shape[0]):
                    masked_pos = mask_positions[b].nonzero().squeeze(-1)
                    if masked_pos.numel() > 0:
                        for pos in masked_pos:
                            tok_idx = true_tokens[b, pos].item()
                            batch_log_probs[b] += log_probs[b, pos, tok_idx].item()
                            batch_counts[b] += 1

            # Compute PPL for each sequence in batch
            for b in range(input_ids.shape[0]):
                if batch_counts[b] > 0:
                    avg_nll = -batch_log_probs[b] / batch_counts[b]
                    ppl = torch.exp(torch.tensor(avg_nll)).item()
                else:
                    ppl = float('nan')
                all_ppls.append(ppl)

        return all_ppls