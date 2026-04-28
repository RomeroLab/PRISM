#!/usr/bin/env python
"""
Residue-Level Embedding Extraction Script for Paired Antibody Sequences

Extracts per-residue embeddings from multiple PLMs:
- ESM-2 35M (facebook/esm2_t12_35M_UR50D)
- ESM-2 650M (facebook/esm2_t33_650M_UR50D)
- AbLang2 (if available)
- AntiBERTy (if available)
- Sapiens (if available)

Usage:
    python extract_residue_embeddings.py --input_file data/train_linear.pkl --batch_size 16
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ==============================================================================
# Configuration
# ==============================================================================

@dataclass
class ModelConfig:
    """Configuration for a model wrapper."""
    name: str
    model_id: str
    embedding_dim: int
    use_fp16: bool = False
    max_length: int = 512


MODEL_CONFIGS: Dict[str, ModelConfig] = {
    "esm2_35m": ModelConfig(
        name="esm2_35m",
        model_id="facebook/esm2_t12_35M_UR50D",
        embedding_dim=480,
        use_fp16=False,
        max_length=1024,
    ),
    "esm2_650m": ModelConfig(
        name="esm2_650m",
        model_id="facebook/esm2_t33_650M_UR50D",
        embedding_dim=1280,
        use_fp16=True,  # Use fp16 for larger model
        max_length=1024,
    ),
    "ablang2": ModelConfig(
        name="ablang2",
        model_id="ablang2-paired",
        embedding_dim=768,
        use_fp16=False,
        max_length=512,
    ),
    "antiberty": ModelConfig(
        name="antiberty",
        model_id="antiberty",
        embedding_dim=512,
        use_fp16=False,
        max_length=512,
    ),
    "sapiens_h": ModelConfig(
        name="sapiens_h",
        model_id="prihodad/biophi-sapiens1-vh",
        embedding_dim=768,
        use_fp16=False,
        max_length=148,  # Model's max_position_embeddings (146 AA + 2 special tokens)
    ),
    "sapiens_l": ModelConfig(
        name="sapiens_l",
        model_id="prihodad/biophi-sapiens1-vl",
        embedding_dim=768,
        use_fp16=False,
        max_length=148,  # Model's max_position_embeddings (146 AA + 2 special tokens)
    ),
}


# ==============================================================================
# Abstract Base Class for Model Wrappers
# ==============================================================================

class BaseModelWrapper(ABC):
    """Abstract base class for PLM wrappers."""

    def __init__(self, config: ModelConfig, device: torch.device):
        self.config = config
        self.device = device
        self.model = None
        self.tokenizer = None

    @abstractmethod
    def load_model(self) -> None:
        """Load the model and tokenizer."""
        pass

    @abstractmethod
    def extract_embeddings(
        self,
        sequences: List[str],
        batch_size: int,
    ) -> List[np.ndarray]:
        """
        Extract residue-level embeddings for a list of sequences.

        Args:
            sequences: List of amino acid sequences
            batch_size: Batch size for inference

        Returns:
            List of numpy arrays, each of shape (seq_len, embedding_dim)
        """
        pass

    def to_device(self) -> None:
        """Move model to device."""
        if self.model is not None:
            self.model = self.model.to(self.device)
            if self.config.use_fp16 and self.device.type == "cuda":
                self.model = self.model.half()

    def eval_mode(self) -> None:
        """Set model to eval mode."""
        if self.model is not None:
            self.model.eval()


# ==============================================================================
# ESM-2 Wrapper (HuggingFace Transformers)
# ==============================================================================

class ESM2Wrapper(BaseModelWrapper):
    """Wrapper for ESM-2 models via HuggingFace Transformers."""

    def load_model(self) -> None:
        from transformers import AutoModel, AutoTokenizer

        logger.info(f"Loading ESM-2 model: {self.config.model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_id)
        self.model = AutoModel.from_pretrained(self.config.model_id)
        self.to_device()
        self.eval_mode()
        logger.info(f"ESM-2 model loaded successfully on {self.device}")

    def extract_embeddings(
        self,
        sequences: List[str],
        batch_size: int,
    ) -> List[np.ndarray]:
        embeddings_list: List[np.ndarray] = []

        for i in tqdm(
            range(0, len(sequences), batch_size),
            desc=f"Extracting {self.config.name}",
        ):
            batch_seqs = sequences[i : i + batch_size]

            # Tokenize batch
            inputs = self.tokenizer(
                batch_seqs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
            )

            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)

            # Cast to half precision if needed
            if self.config.use_fp16 and self.device.type == "cuda":
                pass  # Model is already in half, inputs don't need conversion

            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )

                # Get last hidden state: (batch, seq_len, hidden_dim)
                hidden_states = outputs.last_hidden_state

                # Process each sequence in batch
                for j, seq in enumerate(batch_seqs):
                    seq_len = len(seq)
                    # ESM-2 adds <cls> at start and <eos> at end
                    # Extract embeddings for actual residues (skip special tokens)
                    # Indices 1 to seq_len+1 correspond to residues
                    seq_embeddings = hidden_states[j, 1 : seq_len + 1, :].cpu()

                    # Convert to float32 numpy for storage
                    seq_embeddings_np = seq_embeddings.float().numpy().astype(np.float16)

                    # Verify shape
                    if seq_embeddings_np.shape[0] != seq_len:
                        logger.warning(
                            f"Shape mismatch for {self.config.name}: "
                            f"expected {seq_len}, got {seq_embeddings_np.shape[0]}"
                        )
                        # Pad or truncate to match
                        if seq_embeddings_np.shape[0] < seq_len:
                            padding = np.zeros(
                                (seq_len - seq_embeddings_np.shape[0], self.config.embedding_dim),
                                dtype=np.float16,
                            )
                            seq_embeddings_np = np.vstack([seq_embeddings_np, padding])
                        else:
                            seq_embeddings_np = seq_embeddings_np[:seq_len]

                    embeddings_list.append(seq_embeddings_np)

        return embeddings_list


# ==============================================================================
# AbLang2 Wrapper
# ==============================================================================

class AbLang2Wrapper(BaseModelWrapper):
    """Wrapper for AbLang2 antibody language model.

    AbLang2 uses a different API:
    - For paired mode: expects list of tuples [(heavy, light), ...]
    - Uses mode='rescoding' for residue-level embeddings
    - Returns list of numpy arrays with shape (seq_len + special_tokens, 480)
    """

    def load_model(self) -> None:
        try:
            import ablang2
        except ImportError:
            raise ImportError("ablang2 package not found")

        logger.info("Loading AbLang2 model (paired mode)...")
        self.ablang = ablang2.pretrained(
            model_to_use="ablang2-paired",
            random_init=False,
            device=str(self.device),
        )
        # Keep reference for eval mode
        self.model = self.ablang.AbLang
        self.eval_mode()
        logger.info(f"AbLang2 model loaded successfully on {self.device}")

    def extract_embeddings_paired(
        self,
        heavy_sequences: List[str],
        light_sequences: List[str],
        batch_size: int,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Extract embeddings from AbLang2 for paired sequences.

        Args:
            heavy_sequences: List of heavy chain sequences
            light_sequences: List of light chain sequences
            batch_size: Batch size for inference

        Returns:
            Tuple of (heavy_embeddings, light_embeddings)
        """
        embeddings_h: List[np.ndarray] = []
        embeddings_l: List[np.ndarray] = []

        for i in tqdm(
            range(0, len(heavy_sequences), batch_size),
            desc=f"Extracting {self.config.name}",
        ):
            batch_h = heavy_sequences[i : i + batch_size]
            batch_l = light_sequences[i : i + batch_size]

            # AbLang2 expects list of tuples for paired mode
            paired_seqs = list(zip(batch_h, batch_l))

            # Use rescoding mode for residue-level embeddings
            # Returns list of arrays, each (seq_len + special_tokens, hidden_dim)
            batch_embeddings = self.ablang(paired_seqs, mode='rescoding')

            for j, (h_seq, l_seq) in enumerate(zip(batch_h, batch_l)):
                emb = batch_embeddings[j]  # numpy array
                h_len = len(h_seq)
                l_len = len(l_seq)

                # AbLang2 paired format: [CLS] H [SEP] [CLS] L [SEP] [PAD...]
                # Heavy: positions 1 to h_len (skip CLS at 0)
                # Light: positions h_len+3 to h_len+3+l_len (skip SEP, CLS after heavy)

                # Extract heavy embeddings (skip first CLS token)
                h_emb = emb[1:h_len + 1, :]

                # Extract light embeddings (skip separator tokens between H and L)
                # Format: CLS H SEP CLS L SEP, so L starts at h_len + 3
                l_start = h_len + 3
                l_emb = emb[l_start:l_start + l_len, :]

                # Validate shapes
                if h_emb.shape[0] != h_len:
                    logger.warning(f"AbLang2 heavy shape mismatch: expected {h_len}, got {h_emb.shape[0]}")
                    # Adjust if needed
                    if h_emb.shape[0] < h_len:
                        padding = np.zeros((h_len - h_emb.shape[0], self.config.embedding_dim), dtype=np.float32)
                        h_emb = np.vstack([h_emb, padding])
                    else:
                        h_emb = h_emb[:h_len]

                if l_emb.shape[0] != l_len:
                    logger.warning(f"AbLang2 light shape mismatch: expected {l_len}, got {l_emb.shape[0]}")
                    if l_emb.shape[0] < l_len:
                        padding = np.zeros((l_len - l_emb.shape[0], self.config.embedding_dim), dtype=np.float32)
                        l_emb = np.vstack([l_emb, padding])
                    else:
                        l_emb = l_emb[:l_len]

                embeddings_h.append(h_emb.astype(np.float16))
                embeddings_l.append(l_emb.astype(np.float16))

        return embeddings_h, embeddings_l

    def extract_embeddings(
        self,
        sequences: List[str],
        batch_size: int,
    ) -> List[np.ndarray]:
        """
        Extract embeddings for single chain (fallback, not typical for paired model).
        For paired usage, use extract_embeddings_paired instead.
        """
        embeddings_list: List[np.ndarray] = []

        for i in tqdm(
            range(0, len(sequences), batch_size),
            desc=f"Extracting {self.config.name}",
        ):
            batch_seqs = sequences[i : i + batch_size]

            # For single sequences, wrap as tuples with empty partner
            # This may not work well - ablang2-paired expects paired input
            try:
                batch_embeddings = self.ablang(batch_seqs, mode='rescoding')
                for j, seq in enumerate(batch_seqs):
                    emb = batch_embeddings[j]
                    seq_len = len(seq)
                    # Skip CLS token at position 0
                    seq_emb = emb[1:seq_len + 1, :]
                    embeddings_list.append(seq_emb.astype(np.float16))
            except Exception as e:
                logger.warning(f"AbLang2 single-chain extraction failed: {e}")
                # Return zero embeddings as fallback
                for seq in batch_seqs:
                    embeddings_list.append(
                        np.zeros((len(seq), self.config.embedding_dim), dtype=np.float16)
                    )

        return embeddings_list


# ==============================================================================
# AntiBERTy Wrapper
# ==============================================================================

class AntiBERTyWrapper(BaseModelWrapper):
    """Wrapper for AntiBERTy antibody language model."""

    def load_model(self) -> None:
        try:
            from antiberty import AntiBERTyRunner
        except ImportError:
            raise ImportError("antiberty package not found")

        logger.info("Loading AntiBERTy model...")
        self.runner = AntiBERTyRunner()
        self.model = self.runner.model
        self.tokenizer = self.runner.tokenizer

        # Move to device
        self.model = self.model.to(self.device)
        self.eval_mode()
        logger.info(f"AntiBERTy model loaded successfully on {self.device}")

    def extract_embeddings(
        self,
        sequences: List[str],
        batch_size: int,
    ) -> List[np.ndarray]:
        """
        Extract embeddings from AntiBERTy.
        AntiBERTy expects space-separated amino acids.

        Note: AntiBERTy has a max sequence length of ~510 tokens (512 - 2 for CLS/SEP).
        Each amino acid is one token, so max sequence length is ~510 AA.
        """
        embeddings_list: List[np.ndarray] = [None] * len(sequences)  # Pre-allocate to maintain order
        max_seq_len = self.config.max_length - 2  # Account for CLS and SEP tokens

        for i in tqdm(
            range(0, len(sequences), batch_size),
            desc=f"Extracting {self.config.name}",
        ):
            batch_seqs = sequences[i : i + batch_size]
            batch_start_idx = i

            # Separate valid and invalid sequences, tracking original indices
            valid_seqs = []
            valid_local_indices = []  # Index within batch
            for local_idx, seq in enumerate(batch_seqs):
                global_idx = batch_start_idx + local_idx
                if len(seq) <= max_seq_len:
                    valid_seqs.append(seq)
                    valid_local_indices.append(local_idx)
                else:
                    logger.warning(
                        f"AntiBERTy: Sequence too long ({len(seq)} > {max_seq_len}), using zero embeddings"
                    )
                    embeddings_list[global_idx] = np.zeros(
                        (len(seq), self.config.embedding_dim), dtype=np.float16
                    )

            if not valid_seqs:
                continue

            # AntiBERTy expects space-separated residues
            spaced_seqs = [" ".join(list(seq)) for seq in valid_seqs]

            tokenizer_out = self.tokenizer(
                spaced_seqs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
            )

            input_ids = tokenizer_out["input_ids"].to(self.device)
            attention_mask = tokenizer_out["attention_mask"].to(self.device)

            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )

                # Get last hidden state
                if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
                    hidden_states = outputs.hidden_states[-1]
                else:
                    # Fallback for older versions
                    hidden_states = outputs.last_hidden_state

                for j, (seq, local_idx) in enumerate(zip(valid_seqs, valid_local_indices)):
                    global_idx = batch_start_idx + local_idx
                    seq_len = len(seq)
                    # Skip CLS token at position 0, take seq_len residues
                    seq_emb = hidden_states[j, 1 : seq_len + 1, :].cpu().float().numpy()

                    # Verify and adjust shape if needed
                    if seq_emb.shape[0] != seq_len:
                        if seq_emb.shape[0] < seq_len:
                            padding = np.zeros(
                                (seq_len - seq_emb.shape[0], self.config.embedding_dim),
                                dtype=np.float32,
                            )
                            seq_emb = np.vstack([seq_emb, padding])
                        else:
                            seq_emb = seq_emb[:seq_len]

                    embeddings_list[global_idx] = seq_emb.astype(np.float16)

        return embeddings_list


# ==============================================================================
# Sapiens Wrapper
# ==============================================================================

class SapiensWrapper(BaseModelWrapper):
    """Wrapper for Sapiens humanization model (separate H/L models).

    Note: Sapiens uses RoBERTa with max_position_embeddings=146. Due to CUDA kernel
    instability with this model, it runs on CPU by default to ensure stability.
    """

    def __init__(self, config: ModelConfig, device: torch.device, chain_type: str):
        super().__init__(config, device)
        self.chain_type = chain_type  # "H" or "L"
        # Force CPU for Sapiens due to CUDA instability
        self.device = torch.device("cpu")

    def load_model(self) -> None:
        from transformers import RobertaForMaskedLM, RobertaTokenizer

        logger.info(f"Loading Sapiens model: {self.config.model_id}")
        self.model = RobertaForMaskedLM.from_pretrained(self.config.model_id)
        self.tokenizer = RobertaTokenizer.from_pretrained(
            "prihodad/biophi-sapiens1-tokenizer"
        )
        # Keep on CPU for stability
        self.model = self.model.to(self.device)
        self.eval_mode()
        logger.info(f"Sapiens ({self.chain_type}) loaded successfully on {self.device} (forced CPU for stability)")

    def extract_embeddings(
        self,
        sequences: List[str],
        batch_size: int,  # Ignored for Sapiens - always processes one at a time
    ) -> List[np.ndarray]:
        """Extract embeddings from Sapiens (RoBERTa-based).

        IMPORTANT: Sapiens is processed ONE SEQUENCE AT A TIME to avoid CUDA
        kernel errors that occur with batched inference on padded sequences.
        The batch_size parameter is ignored for this model.

        Note: Sapiens model has max_position_embeddings=146. RoBERTa uses position
        IDs starting at 2 (padding_idx + 1), so for N tokens, positions are 2 to N+1.
        Max safe tokens = 146 - 2 = 144 tokens = 142 AA + 2 special tokens.
        Using 143 AA as the limit to be safe.
        """
        embeddings_list: List[np.ndarray] = []
        max_seq_len = 143  # Safe limit: 143 AA + 2 special = 145 tokens, positions 2-146

        for idx, seq in enumerate(tqdm(sequences, desc=f"Extracting {self.config.name}")):
            # Check sequence length
            if len(seq) > max_seq_len:
                logger.warning(
                    f"Sapiens: Sequence {idx} too long ({len(seq)} > {max_seq_len}), using zero embeddings"
                )
                embeddings_list.append(
                    np.zeros((len(seq), self.config.embedding_dim), dtype=np.float16)
                )
                continue

            try:
                # Tokenize single sequence (no padding needed)
                inputs = self.tokenizer(
                    seq,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=max_seq_len + 2,  # +2 for CLS and SEP
                )

                input_ids = inputs["input_ids"].to(self.device)
                attention_mask = inputs["attention_mask"].to(self.device)

                with torch.no_grad():
                    # RoBERTa base model
                    base_model = self.model.roberta
                    outputs = base_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                    )

                    hidden_states = outputs.last_hidden_state

                    seq_len = len(seq)
                    # Skip CLS token (position 0), take seq_len residues
                    seq_emb = hidden_states[0, 1 : seq_len + 1, :].cpu().float().numpy()

                    if seq_emb.shape[0] != seq_len:
                        if seq_emb.shape[0] < seq_len:
                            padding = np.zeros(
                                (seq_len - seq_emb.shape[0], self.config.embedding_dim),
                                dtype=np.float32,
                            )
                            seq_emb = np.vstack([seq_emb, padding])
                        else:
                            seq_emb = seq_emb[:seq_len]

                    embeddings_list.append(seq_emb.astype(np.float16))

            except Exception as e:
                logger.warning(f"Sapiens: Error at sequence {idx} (len={len(seq)}): {e}")
                embeddings_list.append(
                    np.zeros((len(seq), self.config.embedding_dim), dtype=np.float16)
                )
                # Try to recover CUDA state
                if torch.cuda.is_available():
                    try:
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                    except:
                        pass

        return embeddings_list


# ==============================================================================
# Model Registry
# ==============================================================================

class ModelRegistry:
    """Registry for available models with graceful fallback."""

    def __init__(self, device: torch.device):
        self.device = device
        self.available_models: Dict[str, BaseModelWrapper] = {}
        self._check_and_register_models()

    def _check_and_register_models(self) -> None:
        """Check availability and register models."""

        # ESM-2 models (always available via transformers)
        logger.info("Checking ESM-2 models...")
        try:
            from transformers import AutoModel, AutoTokenizer

            for model_key in ["esm2_35m", "esm2_650m"]:
                config = MODEL_CONFIGS[model_key]
                wrapper = ESM2Wrapper(config, self.device)
                self.available_models[model_key] = wrapper
                logger.info(f"  [OK] {model_key} registered")
        except ImportError as e:
            logger.warning(f"[WARNING] transformers not available: {e}")

        # AbLang2
        logger.info("Checking AbLang2...")
        try:
            import ablang2

            config = MODEL_CONFIGS["ablang2"]
            wrapper = AbLang2Wrapper(config, self.device)
            self.available_models["ablang2"] = wrapper
            logger.info("  [OK] ablang2 registered")
        except ImportError:
            logger.warning("[WARNING] AbLang2 library not found. Skipping...")

        # AntiBERTy
        logger.info("Checking AntiBERTy...")
        try:
            from antiberty import AntiBERTyRunner

            config = MODEL_CONFIGS["antiberty"]
            wrapper = AntiBERTyWrapper(config, self.device)
            self.available_models["antiberty"] = wrapper
            logger.info("  [OK] antiberty registered")
        except ImportError:
            logger.warning("[WARNING] AntiBERTy library not found. Skipping...")

        # Sapiens (separate H and L models)
        logger.info("Checking Sapiens...")
        try:
            from transformers import RobertaForMaskedLM, RobertaTokenizer

            # Test if Sapiens models are accessible
            # We'll register both H and L variants
            config_h = MODEL_CONFIGS["sapiens_h"]
            config_l = MODEL_CONFIGS["sapiens_l"]
            wrapper_h = SapiensWrapper(config_h, self.device, chain_type="H")
            wrapper_l = SapiensWrapper(config_l, self.device, chain_type="L")
            self.available_models["sapiens_h"] = wrapper_h
            self.available_models["sapiens_l"] = wrapper_l
            logger.info("  [OK] sapiens_h and sapiens_l registered")
        except ImportError:
            logger.warning("[WARNING] Sapiens dependencies not found. Skipping...")

        logger.info(f"\nTotal models registered: {len(self.available_models)}")
        logger.info(f"Available: {list(self.available_models.keys())}")

    def load_model(self, model_name: str) -> Optional[BaseModelWrapper]:
        """Load a specific model."""
        if model_name not in self.available_models:
            logger.warning(f"Model {model_name} not available")
            return None

        wrapper = self.available_models[model_name]
        try:
            wrapper.load_model()
            return wrapper
        except Exception as e:
            logger.error(f"Failed to load {model_name}: {e}")
            return None


# ==============================================================================
# Data Loading Utilities
# ==============================================================================

def detect_sequence_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """Detect heavy and light chain column names."""
    heavy_candidates = ["heavy_sequence", "sequence_h", "HEAVY_CHAIN_AA_SEQUENCE", "HeavySequence"]
    light_candidates = ["light_sequence", "sequence_l", "LIGHT_CHAIN_AA_SEQUENCE", "LightSequence"]

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

    # Check for single sequence column
    if heavy_col is None and light_col is None:
        if "sequence" in df.columns:
            logger.info("Found single 'sequence' column. Treating as single-chain data.")
            return "sequence", None

    return heavy_col, light_col


def load_data(input_path: str) -> pd.DataFrame:
    """Load data from pickle file."""
    logger.info(f"Loading data from: {input_path}")
    df = pd.read_pickle(input_path)
    logger.info(f"Loaded {len(df)} sequences")
    return df


# ==============================================================================
# Main Extraction Pipeline
# ==============================================================================

def extract_and_save_embeddings_per_model(
    df: pd.DataFrame,
    heavy_col: str,
    light_col: Optional[str],
    registry: ModelRegistry,
    batch_size: int,
    output_dir: Path,
    input_stem: str,
    models_to_use: Optional[List[str]] = None,
) -> List[str]:
    """
    Extract embeddings from each model and save to separate files.

    Args:
        df: DataFrame with sequences
        heavy_col: Column name for heavy chain sequences
        light_col: Column name for light chain sequences (or None for single-chain)
        registry: ModelRegistry with available models
        batch_size: Batch size for inference
        output_dir: Directory to save output files
        input_stem: Base name for output files (e.g., "train_linear")
        models_to_use: Optional list of specific models to use

    Returns:
        List of saved file paths
    """
    saved_files: List[str] = []

    # Determine which models to use
    if models_to_use:
        model_names = [m for m in models_to_use if m in registry.available_models]
    else:
        model_names = list(registry.available_models.keys())

    # Get sequences
    heavy_seqs = df[heavy_col].tolist()
    light_seqs = df[light_col].tolist() if light_col else None

    for model_name in model_names:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing model: {model_name}")
        logger.info(f"{'='*60}")

        wrapper = registry.load_model(model_name)
        if wrapper is None:
            continue

        try:
            # Create a fresh copy for this model's embeddings
            result_df = df.copy()

            # Handle Sapiens separately (chain-specific models)
            if model_name == "sapiens_h":
                embeddings_h = wrapper.extract_embeddings(heavy_seqs, batch_size)
                result_df[f"embed_{model_name}"] = embeddings_h
                logger.info(f"Added column: embed_{model_name}")

            elif model_name == "sapiens_l":
                if light_seqs:
                    embeddings_l = wrapper.extract_embeddings(light_seqs, batch_size)
                    result_df[f"embed_{model_name}"] = embeddings_l
                    logger.info(f"Added column: embed_{model_name}")
                else:
                    logger.info("No light chain sequences, skipping sapiens_l")
                    continue

            elif model_name == "ablang2":
                # AbLang2 paired mode uses extract_embeddings_paired
                if light_seqs:
                    embeddings_h, embeddings_l = wrapper.extract_embeddings_paired(
                        heavy_seqs, light_seqs, batch_size
                    )
                    result_df[f"embed_{model_name}_h"] = embeddings_h
                    result_df[f"embed_{model_name}_l"] = embeddings_l
                    logger.info(f"Added columns: embed_{model_name}_h, embed_{model_name}_l")
                else:
                    embeddings = wrapper.extract_embeddings(heavy_seqs, batch_size)
                    result_df[f"embed_{model_name}"] = embeddings
                    logger.info(f"Added column: embed_{model_name}")

            else:
                # Standard models (ESM-2, AntiBERTy)
                # Process heavy and light chains separately
                embeddings_h = wrapper.extract_embeddings(heavy_seqs, batch_size)
                result_df[f"embed_{model_name}_h"] = embeddings_h
                logger.info(f"Added column: embed_{model_name}_h")

                if light_seqs:
                    embeddings_l = wrapper.extract_embeddings(light_seqs, batch_size)
                    result_df[f"embed_{model_name}_l"] = embeddings_l
                    logger.info(f"Added column: embed_{model_name}_l")

            # Save this model's embeddings to a separate file
            output_path = output_dir / f"{input_stem}_{model_name}.pkl"
            logger.info(f"Saving to: {output_path}")
            result_df.to_pickle(output_path)
            saved_files.append(str(output_path))

            # Summary for this model
            embed_cols = [c for c in result_df.columns if c.startswith("embed_")]
            logger.info(f"Saved {len(result_df)} sequences with columns: {embed_cols}")

        except Exception as e:
            logger.error(f"Error processing {model_name}: {e}")
            import traceback
            traceback.print_exc()

            # Try to recover CUDA state after error
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                except:
                    pass
            continue

        finally:
            # Clear GPU memory after each model
            if torch.cuda.is_available():
                try:
                    if wrapper is not None and hasattr(wrapper, 'model') and wrapper.model is not None:
                        del wrapper.model
                        wrapper.model = None
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                except Exception as cleanup_error:
                    logger.warning(f"GPU cleanup warning: {cleanup_error}")

    return saved_files


# ==============================================================================
# Main Entry Point
# ==============================================================================

def process_single_file(
    input_path: Path,
    output_dir: Path,
    registry: ModelRegistry,
    batch_size: int,
    models_to_use: Optional[List[str]] = None,
) -> List[str]:
    """Process a single pickle file and save embeddings per model.

    Returns:
        List of saved file paths
    """
    logger.info(f"\n{'#'*70}")
    logger.info(f"Processing: {input_path.name}")
    logger.info(f"{'#'*70}")

    # Load data
    df = load_data(str(input_path))

    # Detect sequence columns
    heavy_col, light_col = detect_sequence_columns(df)
    if heavy_col is None:
        logger.error(f"Could not detect sequence columns in {input_path.name}. Skipping...")
        return []

    logger.info(f"Heavy chain column: {heavy_col}")
    logger.info(f"Light chain column: {light_col}")

    # Extract and save embeddings per model
    saved_files = extract_and_save_embeddings_per_model(
        df=df,
        heavy_col=heavy_col,
        light_col=light_col,
        registry=registry,
        batch_size=batch_size,
        output_dir=output_dir,
        input_stem=input_path.stem,
        models_to_use=models_to_use,
    )

    return saved_files


def main():
    parser = argparse.ArgumentParser(
        description="Extract residue-level embeddings from PLMs for paired antibody sequences"
    )
    parser.add_argument(
        "--input_folder",
        type=str,
        required=True,
        help="Path to folder containing pickle files (train*.pkl, val*.pkl, test*.pkl)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*_linear.pkl",
        help="Glob pattern to match pickle files (default: *.pkl)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for inference (default: 16)",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="Specific models to use (default: all available)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (default: cuda if available)",
    )

    args = parser.parse_args()

    # Setup device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Find all pickle files in the input folder
    input_folder = Path(args.input_folder)
    if not input_folder.is_dir():
        logger.error(f"Input folder does not exist: {input_folder}")
        sys.exit(1)

    pkl_files = sorted(input_folder.glob(args.pattern))
    # Exclude files that already have _embeddings suffix
    pkl_files = [f for f in pkl_files if "_embeddings" not in f.stem]

    if not pkl_files:
        logger.error(f"No pickle files found in {input_folder} matching pattern '{args.pattern}'")
        sys.exit(1)

    logger.info(f"Found {len(pkl_files)} pickle files to process:")
    for f in pkl_files:
        logger.info(f"  - {f.name}")

    # Initialize model registry once (shared across all files)
    registry = ModelRegistry(device)

    # Track all saved files
    all_saved_files: List[str] = []

    # Process each file
    for pkl_file in pkl_files:
        saved_files = process_single_file(
            input_path=pkl_file,
            output_dir=input_folder,  # Save in same folder
            registry=registry,
            batch_size=args.batch_size,
            models_to_use=args.models,
        )
        all_saved_files.extend(saved_files)

    logger.info(f"\n{'='*70}")
    logger.info("All files processed successfully!")
    logger.info(f"{'='*70}")
    logger.info(f"\nTotal files saved: {len(all_saved_files)}")
    for f in all_saved_files:
        logger.info(f"  - {Path(f).name}")


if __name__ == "__main__":
    main()
