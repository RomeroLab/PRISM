#!/usr/bin/env python
# coding: utf-8

"""
High-level inference API for PRISM models.

Quick start::

    import prism

    model  = prism.pretrained("checkpoint.ckpt")
    result = model.forward("EVQLVESGGGLVQ...")  # single forward pass
    # result["final_logits"]  -> [L, 53] combined logits
    # result["aa_logits"]     -> [L, 33] AA head logits
    # result["origin_logits"] -> [L]     GL/NGL logits
    # result["alpha"]         -> [L]     gating values
    # result["embedding"]     -> [L, H]  per-residue embeddings

All methods accept a single string or a list of strings.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    hf_hub_download = None


# =============================================================================
# Module-level helpers
# =============================================================================

def _resolve_device(device: str) -> str:
    """Resolve ``"auto"`` to ``"cuda"`` or ``"cpu"``; pass others through."""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _resolve_checkpoint(checkpoint_path: str) -> str:
    """If *checkpoint_path* is a HF Hub model ID, download and return local path.

    Detection logic:
    - If the path exists on disk, return it as-is.
    - If it contains ``/`` but does **not** start with ``/``, ``.``, or ``~``,
      treat it as a HF Hub ``repo_id`` and download ``checkpoint.ckpt``.
    - Otherwise raise :class:`FileNotFoundError`.
    """
    path = Path(checkpoint_path)
    if path.exists():
        return str(path)

    # Looks like a HF Hub ID: "org/model-name"
    if "/" in checkpoint_path and not checkpoint_path.startswith(("/", ".", "~")):
        if hf_hub_download is None:
            raise ImportError(
                "Install huggingface-hub to download models: "
                "pip install huggingface-hub"
            )
        return hf_hub_download(
            repo_id=checkpoint_path,
            filename="checkpoint.ckpt",
        )

    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")


def _get_bundled_gene_vocab() -> str:
    """Return path to the ``gene_vocabulary.json`` bundled with the package."""
    return str(Path(__file__).parent / "data" / "gene_vocabulary.json")


def _ensure_list(x):
    """Wrap a single string in a list; return ``(list, was_single)``."""
    if isinstance(x, str):
        return [x], True
    return x, False


def _unwrap_single(result, was_single):
    """If the caller passed a single string, unwrap the batch dimension."""
    if not was_single:
        return result
    if isinstance(result, list) and len(result) == 1:
        return result[0]
    if isinstance(result, np.ndarray) and result.ndim >= 1 and result.shape[0] == 1:
        return result[0]
    return result


# =============================================================================
# Factory
# =============================================================================

def pretrained(
    checkpoint_path: str,
    device: str = "auto",
    gene_vocab_path: Optional[str] = None,
) -> "PrismModel":
    """Load a pretrained PRISM model from checkpoint.

    Extracts hyperparameters from the checkpoint automatically --- no YAML config needed.

    *checkpoint_path* can be:

    - A local ``.ckpt`` file path (``"/path/to/model.ckpt"``)
    - A Hugging Face Hub model ID (``"RomeroLab-Duke/prism-antibody"``).
      The checkpoint will be downloaded and cached automatically.

    Args:
        checkpoint_path: Path to a ``.ckpt`` file **or** a HF Hub ``repo_id``.
        device: Device to load model on (``"auto"``, ``"cuda"``, ``"cpu"``, ``"cuda:0"``, etc.).
            Defaults to ``"auto"`` which picks CUDA when available, else CPU.
        gene_vocab_path: Optional path to gene_vocabulary.json. If ``None``,
            looks in checkpoint hparams, then falls back to the bundled copy.

    Returns:
        PrismModel wrapper ready for inference.
    """
    from .model import SFT_ESM2
    from .multimodal_io import GeneVocabulary

    device = _resolve_device(device)

    ckpt_path = _resolve_checkpoint(checkpoint_path)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hparams = checkpoint.get("hyper_parameters", {})

    # Resolve gene vocab: explicit arg > checkpoint hparam > bundled fallback
    gene_vocab = None
    num_genes = 0
    if hparams.get("use_germline_genes", False):
        vocab_path = gene_vocab_path or hparams.get("gene_vocab_path")
        if not vocab_path or not Path(vocab_path).exists():
            vocab_path = _get_bundled_gene_vocab()
        if vocab_path and Path(vocab_path).exists():
            gene_vocab = GeneVocabulary.from_json(vocab_path)
            num_genes = len(gene_vocab)
        else:
            # Last resort: use num_genes from hparams
            num_genes = hparams.get("num_genes", 0)

    # Build model kwargs from checkpoint hparams
    model_kwargs = dict(
        seed=hparams.get("seed", 42),
        model_identifier=hparams.get("model_identifier", "esm2_t12_35M_UR50D"),
        num_unfrozen_transformer_blocks=hparams.get("num_unfrozen_transformer_blocks", 12),
        random_weights=True,  # Will be overwritten by checkpoint weights
        add_custom_tokens=hparams.get("add_custom_tokens", True),
        custom_token_strategy=hparams.get("custom_token_strategy", "lowercase_ngl"),
        activation_function=hparams.get("activation_function", "gelu"),
        fix_swiglu_double_activation=hparams.get("fix_swiglu_double_activation", False),
        # Gene conditioning
        use_germline_genes=hparams.get("use_germline_genes", False),
        num_genes=num_genes,
        gene_embedding_dim=hparams.get("gene_embedding_dim", 64),
        gene_embedding_dropout=hparams.get("gene_embedding_dropout", 0.1),
        # Region embedding
        use_region_embedding=hparams.get("use_region_embedding", False),
        num_regions=hparams.get("num_regions", 8),
        region_embedding_dim=hparams.get("region_embedding_dim", 32),
        # Multihead architecture
        tie_word_embeddings=hparams.get("tie_word_embeddings", False),
        use_multihead_architecture=hparams.get("use_multihead_architecture", True),
        aa_loss_weight=hparams.get("aa_loss_weight", 1.0),
        mut_loss_weight=hparams.get("mut_loss_weight", 1.5),
        mut_focal_gamma=hparams.get("mut_focal_gamma", 2.0),
        use_alpha_gating=hparams.get("use_alpha_gating", True),
        final_loss_weight=hparams.get("final_loss_weight", 2.0),
        aa_focal_gamma=hparams.get("aa_focal_gamma", 2.0),
        origin_focal_gamma=hparams.get("origin_focal_gamma", 2.0),
        ngl_loss_alpha=hparams.get("ngl_loss_alpha", 3.0),
        use_multiplicative_gating=hparams.get("use_multiplicative_gating", True),
        gating_temperature=hparams.get("gating_temperature", 0.5),
        gating_temperature_warmup_steps=hparams.get("gating_temperature_warmup_steps", 500),
        detach_origin_gradient=hparams.get("detach_origin_gradient", True),
        origin_head_dropout=hparams.get("origin_head_dropout", 0.1),
        aa_loss_ngl_weight=hparams.get("aa_loss_ngl_weight", 0.5),
        use_region_balanced_loss=hparams.get("use_region_balanced_loss", True),
        use_cdr_loss_boosting=hparams.get("use_cdr_loss_boosting", False),
        cdr_loss_multiplier=hparams.get("cdr_loss_multiplier", 2.0),
        # Dual AA Heads (v35/v36)
        use_dual_aa_heads=hparams.get("use_dual_aa_heads", False),
        dual_aa_heads_conditioned=hparams.get("dual_aa_heads_conditioned", True),
        # Dummy training args (required by __init__ but unused at inference)
        peak_learning_rate=1e-4,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        WD=0.01,
        warmup_steps=100,
        max_steps=1000,
        logging_steps=10,
        eval_steps=50,
        loss_type="focal_loss",
        batch_size=1,
        mask_prob=0.15,
    )

    model = SFT_ESM2(**model_kwargs)

    # Load state dict
    state_dict = checkpoint["state_dict"]
    # Remove non-persistent buffers that may be in old checkpoints
    keys_to_remove = [k for k in state_dict if "aa_indices" in k]
    for k in keys_to_remove:
        del state_dict[k]
    # Filter out keys with shape mismatches (e.g. old 1-output mut_head vs new 2/3-output)
    model_state = model.state_dict()
    keys_to_skip = []
    for k, v in state_dict.items():
        if k in model_state and v.shape != model_state[k].shape:
            keys_to_skip.append(k)
    for k in keys_to_skip:
        del state_dict[k]
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    return PrismModel(model, gene_vocab=gene_vocab, device=device)


# =============================================================================
# Main inference wrapper
# =============================================================================

class PrismModel:
    """High-level wrapper for PRISM inference.

    Provides clean methods for common tasks: pseudo-log-likelihood,
    embedding extraction, origin prediction, mutation scoring, and raw logits.

    All public methods accept either a single sequence string or a list of
    strings.  When a single string is passed the batch dimension is removed
    from the return value for convenience.

    Attributes:
        AA_ORDER: The 20 standard amino acids in alphabetical order.
            ``GL_INDICES[i]`` and ``NGL_INDICES[i]`` correspond to
            ``AA_ORDER[i]`` in the 53-vocab logits.
    """

    AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"

    def __init__(self, model, gene_vocab=None, device="cpu"):
        self.model = model
        self.tokenizer = model.tokenizer
        self.gene_vocab = gene_vocab
        self.device = device

        # Build AA token mappings
        self.uppercase_aa_to_idx: Dict[str, int] = {}
        self.lowercase_aa_to_idx: Dict[str, int] = {}
        for aa in self.AA_ORDER:
            self.uppercase_aa_to_idx[aa] = self.tokenizer.convert_tokens_to_ids(aa)
            lower_id = self.tokenizer.convert_tokens_to_ids(aa.lower())
            if lower_id != self.tokenizer.unk_token_id:
                self.lowercase_aa_to_idx[aa.lower()] = lower_id

        # Collect all standard AA token IDs (for marginalization)
        self._aa_upper_ids = torch.tensor(
            [self.uppercase_aa_to_idx[aa] for aa in self.AA_ORDER],
            device=device,
        )
        self._aa_lower_ids = torch.tensor(
            [self.lowercase_aa_to_idx.get(aa.lower(), -1) for aa in self.AA_ORDER],
            device=device,
        )

        # Precompute GL/NGL index arrays for slicing 53-vocab logits -> [*, 20]
        self._gl_indices = np.array(
            [self.uppercase_aa_to_idx[aa] for aa in self.AA_ORDER]
        )
        self._ngl_indices = np.array(
            [self.lowercase_aa_to_idx.get(aa.lower(), -1) for aa in self.AA_ORDER]
        )

        # Public constants for user-side slicing of 53-vocab logits
        self.GL_INDICES: np.ndarray = self._gl_indices
        self.NGL_INDICES: np.ndarray = self._ngl_indices

        # Cache for ANARCI annotations: sequence -> (v_gene, j_gene, region_mask)
        self._anarci_cache: Dict[str, Tuple[Optional[str], Optional[str], Optional[str]]] = {}

        # Whether this model uses gene/region conditioning
        self._uses_genes = getattr(model, "use_germline_genes", False)
        self._uses_regions = getattr(model, "use_region_embedding", False)

    # =========================================================================
    # Auto ANARCI Annotation
    # =========================================================================

    _IMGT_REGIONS = {
        "fr1": (1, 26, "0"), "cdr1": (27, 38, "1"), "fr2": (39, 55, "2"),
        "cdr2": (56, 65, "3"), "fr3": (66, 104, "4"), "cdr3": (105, 117, "5"),
        "fr4": (118, 128, "6"),
    }

    def _anarci_annotate(self, sequence: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Run ANARCI on a single sequence and return (v_gene, j_gene, region_mask).

        Results are cached by sequence string.
        """
        if sequence in self._anarci_cache:
            return self._anarci_cache[sequence]

        try:
            from anarci import anarci as run_anarci
        except ImportError:
            logger.warning("anarci not installed; skipping auto gene/region annotation")
            self._anarci_cache[sequence] = (None, None, None)
            return (None, None, None)

        try:
            numbering, details, _ = run_anarci(
                [("q", sequence)], scheme="imgt", assign_germline=True,
            )
        except Exception:
            self._anarci_cache[sequence] = (None, None, None)
            return (None, None, None)

        if not numbering[0]:
            self._anarci_cache[sequence] = (None, None, None)
            return (None, None, None)

        chain_numbering = numbering[0][0][0]
        detail = details[0][0]

        # Gene names (strip allele)
        v_gene = detail["germlines"]["v_gene"][0][1].split("*")[0]
        j_gene = detail["germlines"]["j_gene"][0][1].split("*")[0]

        # Region mask
        mask_chars = []
        for (pos_tuple, aa) in chain_numbering:
            if aa != "-":
                pos_num = pos_tuple[0]
                for _, (s, e, code) in self._IMGT_REGIONS.items():
                    if s <= pos_num <= e:
                        mask_chars.append(code)
                        break
        region_mask = "".join(mask_chars)

        result = (v_gene, j_gene, region_mask)
        self._anarci_cache[sequence] = result
        return result

    def _auto_conditioning(
        self,
        heavy_seqs: List[str],
        light_seqs: Optional[List[str]],
        v_gene_ids: Optional[torch.Tensor],
        j_gene_ids: Optional[torch.Tensor],
        region_ids: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Auto-fill gene/region conditioning via ANARCI when not provided.

        Only runs when the model was trained with gene/region conditioning
        and the user did not supply tensors explicitly.

        Returns the (possibly filled) v_gene_ids, j_gene_ids, region_ids.
        """
        needs_genes = self._uses_genes and v_gene_ids is None and self.gene_vocab is not None
        needs_regions = self._uses_regions and region_ids is None

        if not needs_genes and not needs_regions:
            return v_gene_ids, j_gene_ids, region_ids

        n = len(heavy_seqs)
        v_list = []
        j_list = []
        rmh_list = []
        rml_list = []

        for i in range(n):
            vh, jh, rm_h = self._anarci_annotate(heavy_seqs[i])
            v_list.append(vh)
            j_list.append(jh)
            rmh_list.append(rm_h)

            if light_seqs is not None:
                _, _, rm_l = self._anarci_annotate(light_seqs[i])
                rml_list.append(rm_l)
            else:
                rml_list.append(None)

        # Build gene ID tensors
        if needs_genes:
            v_gene_ids = torch.tensor(
                [self.gene_vocab.encode(v) for v in v_list], dtype=torch.long,
            )
            j_gene_ids = torch.tensor(
                [self.gene_vocab.encode(j) for j in j_list], dtype=torch.long,
            )

        # Build region ID tensor
        if needs_regions:
            max_len = 512
            all_rids = []
            for i in range(n):
                rids = [0]  # CLS
                if rmh_list[i]:
                    for ch in rmh_list[i]:
                        rids.append(int(ch) + 1)
                if light_seqs is not None:
                    rids.append(0)  # CLS sep
                    rids.append(0)  # CLS sep
                    if rml_list[i]:
                        for ch in rml_list[i]:
                            rids.append(int(ch) + 1)
                rids.append(0)  # EOS
                if len(rids) > max_len:
                    rids = rids[:max_len]
                else:
                    rids.extend([0] * (max_len - len(rids)))
                all_rids.append(rids)
            region_ids = torch.tensor(all_rids, dtype=torch.long)

        return v_gene_ids, j_gene_ids, region_ids

    # =========================================================================
    # get_tokenizer
    # =========================================================================

    def get_tokenizer(self):
        """Return a PrismTokenizer wrapping this model's internal tokenizer.

        Returns:
            PrismTokenizer with the same vocabulary as this model.
        """
        from .tokenizer import PrismTokenizer

        tok = PrismTokenizer.__new__(PrismTokenizer)
        tok._model_identifier = getattr(self.model, 'esm_model_name', 'esm2_t12_35M_UR50D')
        tok._tokenizer = self.tokenizer

        # Build GL/NGL mappings from model's internal tokenizer
        tok.gl_token_ids = {}
        tok.ngl_token_ids = {}
        tok.gl_to_ngl = {}
        for aa in PrismTokenizer.AA_ORDER:
            gl_id = self.tokenizer.convert_tokens_to_ids(aa)
            ngl_id = self.tokenizer.convert_tokens_to_ids(aa.lower())
            tok.gl_token_ids[aa] = gl_id
            if ngl_id != self.tokenizer.unk_token_id:
                tok.ngl_token_ids[aa.lower()] = ngl_id
                tok.gl_to_ngl[gl_id] = ngl_id

        return tok

    # =========================================================================
    # Paired Chain Support
    # =========================================================================

    def _format_sequences(
        self,
        sequences: List[str],
        light_chains: Optional[List[str]],
    ) -> List[str]:
        """Combine heavy and light chains into paired format.

        When ``light_chains`` is provided, each heavy chain is concatenated
        with its corresponding light chain using the ``<cls><cls>`` separator
        to match the training format: ``VH<cls><cls>VL``.

        Args:
            sequences: Heavy chain sequence(s) (already converted to list).
            light_chains: Optional light chain sequence(s), one per heavy chain.

        Returns:
            List of formatted sequences (paired or heavy-only).

        Raises:
            ValueError: If lengths of sequences and light_chains don't match.
        """
        if light_chains is None:
            return sequences
        if len(sequences) != len(light_chains):
            raise ValueError(
                f"light_chains length mismatch: got {len(sequences)} heavy chain(s) "
                f"but {len(light_chains)} light chain(s)"
            )
        sep = self.tokenizer.cls_token
        return [f"{vh}{sep}{sep}{vl}" for vh, vl in zip(sequences, light_chains)]

    # =========================================================================
    # Core Inference Methods
    # =========================================================================

    @torch.no_grad()
    def pseudo_log_likelihood(
        self,
        sequences: Optional[Union[str, List[str]]] = None,
        *,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        heavy_chains: Optional[Union[str, List[str]]] = None,
        light_chains: Optional[Union[str, List[str]]] = None,
        region_masks: Optional[Union[str, List[str]]] = None,
        preserve_case: bool = False,
        v_gene_ids: Optional[torch.Tensor] = None,
        j_gene_ids: Optional[torch.Tensor] = None,
        region_ids: Optional[torch.Tensor] = None,
        batch_size: int = 32,
    ) -> Union[Dict[str, Dict[str, object]], List[Dict[str, Dict[str, object]]]]:
        """Compute pseudo-log-likelihood (PLL) and perplexity.

        Masks each AA position one at a time, predicts log P(true token),
        and returns PLL, perplexity, and per-position log-probs for all
        three scoring modes (marginalized, GL, NGL) in a single pass.

        Args:
            sequences: Amino acid sequence(s) for unpaired input.
            input_ids: Pre-tokenized input IDs ``[B, L]`` from
                :class:`PrismTokenizer`.  Mutually exclusive with
                ``sequences`` / ``heavy_chains`` / ``light_chains``.
            attention_mask: Attention mask ``[B, L]`` matching
                ``input_ids``.
            heavy_chains: Heavy chain sequence(s).  Use with
                ``light_chains`` for paired input.
            light_chains: Light chain sequence(s).
            region_masks: Optional region mask string(s).  Digit string
                where each character encodes the region of the
                corresponding AA position: 0=FR1, 1=CDR1, 2=FR2,
                3=CDR2, 4=FR3, 5=CDR3, 6=FR4.  When provided, an
                additional ``"region_conditioned"`` key is included in
                the output that uses GL log-probs for FR positions and
                NGL log-probs for CDR positions.
            preserve_case: If True, do **not** force input to uppercase.
                Lowercase characters are tokenized as NGL tokens.
                This affects the ``"exact"`` mode (uses the actual
                token -- uppercase for GL, lowercase for NGL) **and**
                the model context (unmasked tokens preserve their
                original casing, matching the training format).
                Default False (all uppercase, backward-compatible).
            batch_size: Number of masked positions per forward pass.

        Returns:
            Dict (single input) or list of dicts, each with keys
            ``"marginalized"``, ``"gl"``, ``"ngl"``.  Each sub-dict
            contains:

            - ``"pll"``: ``float`` -- sum of per-position log-probs
            - ``"perplexity"``: ``float`` -- ``exp(-pll / L)``
            - ``"per_position"``: ``[L]`` numpy array

            When ``region_masks`` is provided, an additional
            ``"region_conditioned"`` key is included with the same
            sub-dict structure.
        """
        # --- Input resolution ---
        _use_input_ids = input_ids is not None

        if _use_input_ids:
            # Pre-tokenized path: input_ids + attention_mask
            str_args = (sequences, heavy_chains, light_chains)
            if any(a is not None for a in str_args):
                raise ValueError(
                    "Cannot specify both `input_ids` and "
                    "`sequences`/`heavy_chains`/`light_chains`."
                )
            if attention_mask is None:
                attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
                attention_mask = attention_mask.unsqueeze(0)
            was_single = input_ids.shape[0] == 1
            # Compute AA lengths from attention_mask (total tokens - special tokens)
            formatted = None  # not used in input_ids path
            seq_aa_lengths = []
            special_ids_set = {
                self.tokenizer.cls_token_id,
                self.tokenizer.eos_token_id,
                self.tokenizer.pad_token_id,
            }
            special_ids_set.discard(None)
            for b in range(input_ids.shape[0]):
                n_special = sum(
                    1 for pos in range(attention_mask[b].sum().item())
                    if input_ids[b, pos].item() in special_ids_set
                )
                seq_aa_lengths.append(attention_mask[b].sum().item() - n_special)
        else:
            if sequences is not None and (heavy_chains is not None or light_chains is not None):
                raise ValueError(
                    "Cannot specify both `sequences` and `heavy_chains`/`light_chains`."
                )
            if sequences is None and heavy_chains is None and light_chains is None:
                raise ValueError(
                    "Must provide one of: `sequences`, `heavy_chains`, "
                    "`light_chains`, or `input_ids`."
                )

            is_paired = heavy_chains is not None and light_chains is not None
            if sequences is not None:
                seqs, was_single = _ensure_list(sequences)
                formatted = seqs
                raw_heavy = seqs
                raw_light = None
                seq_aa_lengths = [len(s) for s in seqs]
            elif is_paired:
                heavy_chains, was_single = _ensure_list(heavy_chains)
                light_chains, _ = _ensure_list(light_chains)
                formatted = self._format_sequences(heavy_chains, light_chains)
                raw_heavy = heavy_chains
                raw_light = light_chains
                seq_aa_lengths = [len(h) + len(l) for h, l in zip(heavy_chains, light_chains)]
            else:
                single = heavy_chains if heavy_chains is not None else light_chains
                seqs, was_single = _ensure_list(single)
                formatted = seqs
                raw_heavy = seqs
                raw_light = None
                seq_aa_lengths = [len(s) for s in seqs]

            v_gene_ids, j_gene_ids, region_ids = self._auto_conditioning(
                raw_heavy, raw_light, v_gene_ids, j_gene_ids, region_ids,
            )

        # --- Region mask resolution ---
        _CDR_REGIONS = {1, 3, 5}  # CDR1, CDR2, CDR3
        region_masks_list: Optional[List[str]] = None
        if region_masks is not None:
            if isinstance(region_masks, str):
                region_masks_list = [region_masks] if was_single else [region_masks]
            else:
                region_masks_list = list(region_masks)
            if len(region_masks_list) != len(seq_aa_lengths):
                raise ValueError(
                    f"Number of region_masks ({len(region_masks_list)}) "
                    f"must match number of sequences ({len(seq_aa_lengths)})."
                )
            for idx, (rmask, aa_len) in enumerate(zip(region_masks_list, seq_aa_lengths)):
                if len(rmask) != aa_len:
                    raise ValueError(
                        f"region_mask length ({len(rmask)}) does not match "
                        f"sequence AA length ({aa_len}) for sequence {idx}."
                    )

        mask_token_id = self.tokenizer.mask_token_id
        special_ids = {
            self.tokenizer.cls_token_id,
            self.tokenizer.eos_token_id,
            self.tokenizer.pad_token_id,
        }
        special_ids.discard(None)

        # Build reverse mapping: NGL token ID -> GL token ID
        _ngl_to_gl: Dict[int, int] = {}
        if self.model.lowercase_aa_token_ids:
            for gl_id, ngl_id in self.model.lowercase_aa_token_ids.items():
                _ngl_to_gl[ngl_id] = gl_id

        all_results = []
        n_seqs = input_ids.shape[0] if _use_input_ids else len(formatted)

        for seq_idx in tqdm(range(n_seqs), desc="PLL sequences"):
            if _use_input_ids:
                seq_input_ids = input_ids[seq_idx:seq_idx + 1]
                seq_attn_mask = attention_mask[seq_idx:seq_idx + 1]
            else:
                seq_input_ids, seq_attn_mask = self._tokenize(
                    [formatted[seq_idx]], uppercase=not preserve_case,
                )
            seq_len = seq_attn_mask[0].sum().item()

            maskable = []
            for pos in range(seq_len):
                if seq_input_ids[0, pos].item() not in special_ids:
                    maskable.append(pos)

            n_aa = len(maskable)
            original_ids = [seq_input_ids[0, pos].item() for pos in maskable]

            # Pre-allocate per-position arrays for all 4 modes
            gl_logps = np.zeros(n_aa, dtype=np.float64)
            ngl_logps = np.zeros(n_aa, dtype=np.float64)
            marg_logps = np.zeros(n_aa, dtype=np.float64)
            exact_logps = np.zeros(n_aa, dtype=np.float64)

            for chunk_start in tqdm(
                range(0, n_aa, batch_size),
                desc=f"  seq {seq_idx + 1} positions",
                total=(n_aa + batch_size - 1) // batch_size,
                leave=False,
            ):
                chunk_positions = maskable[chunk_start:chunk_start + batch_size]
                chunk_orig_ids = original_ids[chunk_start:chunk_start + batch_size]
                bs = len(chunk_positions)

                batch_input = seq_input_ids.expand(bs, -1).clone()
                batch_attn = seq_attn_mask.expand(bs, -1)
                for i, pos in enumerate(chunk_positions):
                    batch_input[i, pos] = mask_token_id

                # Expand per-sequence conditioning to batch size
                _v = v_gene_ids[seq_idx:seq_idx + 1].expand(bs).to(self.device) if v_gene_ids is not None else None
                _j = j_gene_ids[seq_idx:seq_idx + 1].expand(bs).to(self.device) if j_gene_ids is not None else None
                _r = region_ids[seq_idx:seq_idx + 1, :batch_input.shape[1]].expand(bs, -1).to(self.device) if region_ids is not None else None

                _, _, _, _, logits_final, _ = self.model._forward_multihead(
                    input_ids=batch_input,
                    attention_mask=batch_attn,
                    v_gene_ids=_v,
                    j_gene_ids=_j,
                    region_ids=_r,
                )

                for i, (pos, orig_id) in enumerate(zip(chunk_positions, chunk_orig_ids)):
                    log_probs = F.log_softmax(logits_final[i, pos], dim=-1)
                    idx = chunk_start + i

                    # Resolve GL (uppercase) and NGL (lowercase) token IDs
                    # orig_id may be uppercase or lowercase when preserve_case=True
                    if orig_id in _ngl_to_gl:
                        # orig_id is a lowercase (NGL) token
                        upper_id = _ngl_to_gl[orig_id]
                        lower_id = orig_id
                    else:
                        # orig_id is an uppercase (GL) token
                        upper_id = orig_id
                        lower_id = self.model.lowercase_aa_token_ids.get(orig_id)

                    # GL: uppercase token log-prob
                    gl_logps[idx] = log_probs[upper_id].item()

                    # NGL: lowercase token log-prob
                    ngl_logps[idx] = log_probs[lower_id if lower_id is not None else upper_id].item()

                    # Marginalized: logsumexp(GL, NGL)
                    marg_logps[idx] = self._marginalized_logp(log_probs, upper_id)

                    # Exact: log-prob of the actual input token
                    exact_logps[idx] = log_probs[orig_id].item()

            aa_len = seq_aa_lengths[seq_idx]

            def _build_mode_dict(pos_logps):
                pll = float(pos_logps.sum())
                return {
                    "pll": pll,
                    "perplexity": float(np.exp(-pll / aa_len)),
                    "per_position": pos_logps,
                }

            result_dict = {
                "marginalized": _build_mode_dict(marg_logps),
                "gl": _build_mode_dict(gl_logps),
                "ngl": _build_mode_dict(ngl_logps),
                "exact": _build_mode_dict(exact_logps),
            }

            # Region-conditioned: CDR -> NGL, FR -> GL
            if region_masks_list is not None:
                rmask = region_masks_list[seq_idx]
                rc_logps = np.zeros(n_aa, dtype=np.float64)
                for i in range(n_aa):
                    region_val = int(rmask[i])
                    if region_val in _CDR_REGIONS:
                        rc_logps[i] = ngl_logps[i]
                    else:
                        rc_logps[i] = gl_logps[i]
                result_dict["region_conditioned"] = _build_mode_dict(rc_logps)

            all_results.append(result_dict)

        return _unwrap_single(all_results, was_single)

    @torch.no_grad()
    def score_mutations(
        self,
        wt: Optional[Union[str, List[str]]] = None,
        mutant: Optional[Union[str, List[str]]] = None,
        *,
        wt_input_ids: Optional[torch.Tensor] = None,
        mut_input_ids: Optional[torch.Tensor] = None,
        wt_attention_mask: Optional[torch.Tensor] = None,
        mut_attention_mask: Optional[torch.Tensor] = None,
        wt_light_chains: Optional[Union[str, List[str]]] = None,
        mut_light_chains: Optional[Union[str, List[str]]] = None,
        v_gene_ids: Optional[torch.Tensor] = None,
        j_gene_ids: Optional[torch.Tensor] = None,
        region_ids: Optional[torch.Tensor] = None,
        batch_size: int = 32,
    ) -> Union[Dict, List[Dict]]:
        """Score mutations using masked marginal log-likelihood ratios.

        For each mutation position, masks that position in both the WT and
        mutant sequences, runs a forward pass, and computes a delta score.

        The WT side always uses GL (uppercase) log-probabilities, since WT
        residues are germline.  The mutant side uses the mode-specific
        log-probabilities::

            marginalized: logP_marg(mut_AA) - logP_gl(wt_AA)
            gl:           logP_gl(mut_AA)   - logP_gl(wt_AA)
            ngl:          logP_ngl(mut_AA)  - logP_gl(wt_AA)
            exact:        logP(orig_token)  - logP(orig_token)  [symmetric]

        All four scoring modes are computed in a single pass.  Total cost:
        ``2 * ceil(M / batch_size)`` forward passes per WT/mutant pair,
        where M = number of mutation positions.

        Args:
            wt: Wild-type sequence(s).
            mutant: Mutant sequence(s), same length as WT.
            wt_input_ids: Pre-tokenized WT input IDs ``[B, L]``.
                Mutually exclusive with ``wt``/``mutant`` string args.
            mut_input_ids: Pre-tokenized mutant input IDs ``[B, L]``.
            wt_attention_mask: Attention mask for ``wt_input_ids``.
            mut_attention_mask: Attention mask for ``mut_input_ids``.
            wt_light_chains: Optional WT light chain(s) for paired input.
            mut_light_chains: Optional mutant light chain(s) for paired input.
            batch_size: Number of masked positions per forward pass.

        Returns:
            Dict (single pair) or list of dicts, each with:

            - ``"positions"``: list of 0-indexed mutation positions
            - ``"marginalized"``: ``{"score": float, "per_position": [M]}``
            - ``"gl"``: ``{"score": float, "per_position": [M]}``
            - ``"ngl"``: ``{"score": float, "per_position": [M]}``
            - ``"exact"``: ``{"score": float, "per_position": [M]}``
        """
        _use_input_ids = wt_input_ids is not None or mut_input_ids is not None

        # Build reverse mapping for NGL-aware exact mode
        _ngl_to_gl: Dict[int, int] = {}
        if self.model.lowercase_aa_token_ids:
            for gl_id, ngl_id in self.model.lowercase_aa_token_ids.items():
                _ngl_to_gl[ngl_id] = gl_id

        if _use_input_ids:
            # --- Pre-tokenized path ---
            if wt is not None or mutant is not None:
                raise ValueError(
                    "Cannot specify both `wt_input_ids`/`mut_input_ids` and "
                    "`wt`/`mutant` string arguments."
                )
            if wt_input_ids is None or mut_input_ids is None:
                raise ValueError(
                    "Must provide both `wt_input_ids` and `mut_input_ids`."
                )
            wt_input_ids = wt_input_ids.to(self.device)
            mut_input_ids = mut_input_ids.to(self.device)
            if wt_input_ids.dim() == 1:
                wt_input_ids = wt_input_ids.unsqueeze(0)
                mut_input_ids = mut_input_ids.unsqueeze(0)
            if wt_attention_mask is None:
                wt_attention_mask = (wt_input_ids != self.tokenizer.pad_token_id).long()
            if mut_attention_mask is None:
                mut_attention_mask = (mut_input_ids != self.tokenizer.pad_token_id).long()
            wt_attention_mask = wt_attention_mask.to(self.device)
            mut_attention_mask = mut_attention_mask.to(self.device)

            was_single = wt_input_ids.shape[0] == 1
            n_pairs = wt_input_ids.shape[0]

            special_ids = {
                self.tokenizer.cls_token_id,
                self.tokenizer.eos_token_id,
                self.tokenizer.pad_token_id,
            }
            special_ids.discard(None)
            mask_token_id = self.tokenizer.mask_token_id
            all_results = []

            for idx in range(n_pairs):
                cur_wt_ids = wt_input_ids[idx:idx + 1]
                cur_mut_ids = mut_input_ids[idx:idx + 1]
                cur_wt_attn = wt_attention_mask[idx:idx + 1]
                cur_mut_attn = mut_attention_mask[idx:idx + 1]
                wt_len = cur_wt_attn[0].sum().item()
                mut_len = cur_mut_attn[0].sum().item()

                # Find mutation positions by token ID diff (skip special tokens)
                mut_positions = []
                for pos in range(1, min(wt_len, mut_len) - 1):  # skip CLS and EOS
                    wt_tok = cur_wt_ids[0, pos].item()
                    mut_tok = cur_mut_ids[0, pos].item()
                    if wt_tok != mut_tok and wt_tok not in special_ids:
                        mut_positions.append(pos - 1)  # 0-indexed AA position

                if not mut_positions:
                    empty = {"score": 0.0, "per_position": np.array([])}
                    all_results.append({
                        "positions": [],
                        "marginalized": empty.copy(),
                        "gl": empty.copy(),
                        "ngl": empty.copy(),
                        "exact": empty.copy(),
                    })
                    continue

                n_mut = len(mut_positions)
                wt_gl = np.zeros(n_mut, dtype=np.float64)
                wt_ngl = np.zeros(n_mut, dtype=np.float64)
                wt_marg = np.zeros(n_mut, dtype=np.float64)
                wt_exact = np.zeros(n_mut, dtype=np.float64)
                mut_gl_arr = np.zeros(n_mut, dtype=np.float64)
                mut_ngl = np.zeros(n_mut, dtype=np.float64)
                mut_marg = np.zeros(n_mut, dtype=np.float64)
                mut_exact = np.zeros(n_mut, dtype=np.float64)

                def _extract_from_ids(seq_ids, seq_attn, positions, arrays):
                    gl_arr, ngl_arr, marg_arr, exact_arr = arrays
                    for chunk_start in range(0, len(positions), batch_size):
                        chunk_pos = positions[chunk_start:chunk_start + batch_size]
                        bs = len(chunk_pos)

                        batch_input = seq_ids.expand(bs, -1).clone()
                        batch_attn = seq_attn.expand(bs, -1)
                        tok_positions = [p + 1 for p in chunk_pos]  # +1 for CLS
                        for i, tok_pos in enumerate(tok_positions):
                            batch_input[i, tok_pos] = mask_token_id

                        _v = v_gene_ids[idx:idx + 1].expand(bs).to(self.device) if v_gene_ids is not None else None
                        _j = j_gene_ids[idx:idx + 1].expand(bs).to(self.device) if j_gene_ids is not None else None
                        _r = region_ids[idx:idx + 1, :batch_input.shape[1]].expand(bs, -1).to(self.device) if region_ids is not None else None

                        _, _, _, _, logits_final, _ = self.model._forward_multihead(
                            input_ids=batch_input,
                            attention_mask=batch_attn,
                            v_gene_ids=_v,
                            j_gene_ids=_j,
                            region_ids=_r,
                        )

                        for i, (aa_pos, tok_pos) in enumerate(zip(chunk_pos, tok_positions)):
                            log_probs = F.log_softmax(logits_final[i, tok_pos], dim=-1)
                            ci = chunk_start + i
                            orig_id = seq_ids[0, tok_pos].item()

                            # Resolve GL/NGL IDs
                            if orig_id in _ngl_to_gl:
                                upper_id = _ngl_to_gl[orig_id]
                                lower_id = orig_id
                            else:
                                upper_id = orig_id
                                lower_id = self.model.lowercase_aa_token_ids.get(orig_id)

                            gl_arr[ci] = log_probs[upper_id].item()
                            ngl_arr[ci] = log_probs[lower_id if lower_id is not None else upper_id].item()
                            marg_arr[ci] = self._marginalized_logp(log_probs, upper_id)
                            exact_arr[ci] = log_probs[orig_id].item()

                _extract_from_ids(
                    cur_wt_ids, cur_wt_attn, mut_positions,
                    (wt_gl, wt_ngl, wt_marg, wt_exact),
                )
                _extract_from_ids(
                    cur_mut_ids, cur_mut_attn, mut_positions,
                    (mut_gl_arr, mut_ngl, mut_marg, mut_exact),
                )

                def _build_mode(wt_arr, mut_arr):
                    per_pos = mut_arr - wt_arr
                    return {"score": float(per_pos.sum()), "per_position": per_pos}

                all_results.append({
                    "positions": mut_positions,
                    "marginalized": _build_mode(wt_gl, mut_marg),
                    "gl": _build_mode(wt_gl, mut_gl_arr),
                    "ngl": _build_mode(wt_gl, mut_ngl),
                    "exact": _build_mode(wt_exact, mut_exact),
                })

            return _unwrap_single(all_results, was_single)

        else:
            # --- String path ---
            if wt is None or mutant is None:
                raise ValueError(
                    "Must provide both `wt` and `mutant` string arguments, "
                    "or both `wt_input_ids` and `mut_input_ids`."
                )

            wt_list, was_single = _ensure_list(wt)
            mut_list, _ = _ensure_list(mutant)
            if wt_light_chains is not None:
                wt_light_chains, _ = _ensure_list(wt_light_chains)
            if mut_light_chains is not None:
                mut_light_chains, _ = _ensure_list(mut_light_chains)

            # Keep raw heavy chains for position detection
            raw_wt = wt_list
            raw_mut = mut_list

            wt_formatted = self._format_sequences(wt_list, wt_light_chains)
            mut_formatted = self._format_sequences(mut_list, mut_light_chains)
            if len(wt_formatted) != len(mut_formatted):
                raise ValueError(
                    f"Number of WT sequences ({len(wt_formatted)}) must match "
                    f"number of mutant sequences ({len(mut_formatted)})."
                )

            # Auto-fill gene/region conditioning (uses WT sequences)
            v_gene_ids, j_gene_ids, region_ids = self._auto_conditioning(
                raw_wt, wt_light_chains, v_gene_ids, j_gene_ids, region_ids,
            )

            mask_token_id = self.tokenizer.mask_token_id
            all_results = []

            for idx in range(len(wt_formatted)):
                # Validate sequence lengths match within each pair
                if len(raw_wt[idx]) != len(raw_mut[idx]):
                    raise ValueError(
                        f"WT and mutant sequence lengths differ for pair {idx}: "
                        f"WT={len(raw_wt[idx])}, mutant={len(raw_mut[idx])}. "
                        f"score_mutations requires equal-length sequences "
                        f"(substitutions only, no insertions/deletions)."
                    )

                # Find mutation positions on raw heavy chains
                mut_positions = [
                    i for i, (w, m) in enumerate(zip(raw_wt[idx], raw_mut[idx]))
                    if w != m
                ]

                if not mut_positions:
                    empty = {"score": 0.0, "per_position": np.array([])}
                    all_results.append({
                        "positions": [],
                        "marginalized": empty.copy(),
                        "gl": empty.copy(),
                        "ngl": empty.copy(),
                        "exact": empty.copy(),
                    })
                    continue

                n_mut = len(mut_positions)
                # Pre-allocate: 4 modes x (wt, mut) per-position log-probs
                wt_gl = np.zeros(n_mut, dtype=np.float64)
                wt_ngl = np.zeros(n_mut, dtype=np.float64)
                wt_marg = np.zeros(n_mut, dtype=np.float64)
                wt_exact = np.zeros(n_mut, dtype=np.float64)
                mut_gl = np.zeros(n_mut, dtype=np.float64)
                mut_ngl = np.zeros(n_mut, dtype=np.float64)
                mut_marg = np.zeros(n_mut, dtype=np.float64)
                mut_exact = np.zeros(n_mut, dtype=np.float64)

                # Tokenize both
                wt_ids, wt_attn = self._tokenize([wt_formatted[idx]])
                mut_ids, mut_attn = self._tokenize([mut_formatted[idx]])

                def _extract_logprobs(input_ids_local, attn_mask, positions, aa_token_ids, arrays):
                    """Masked forward at each position, extract 4-mode log-probs."""
                    gl_arr, ngl_arr, marg_arr, exact_arr = arrays
                    for chunk_start in range(0, len(positions), batch_size):
                        chunk_pos = positions[chunk_start:chunk_start + batch_size]
                        chunk_aa_ids = aa_token_ids[chunk_start:chunk_start + batch_size]
                        bs = len(chunk_pos)

                        batch_input = input_ids_local.expand(bs, -1).clone()
                        batch_attn = attn_mask.expand(bs, -1)
                        for i, pos in enumerate(chunk_pos):
                            tok_pos = pos + 1  # +1 for CLS
                            batch_input[i, tok_pos] = mask_token_id

                        _v = v_gene_ids[idx:idx + 1].expand(bs).to(self.device) if v_gene_ids is not None else None
                        _j = j_gene_ids[idx:idx + 1].expand(bs).to(self.device) if j_gene_ids is not None else None
                        _r = region_ids[idx:idx + 1, :batch_input.shape[1]].expand(bs, -1).to(self.device) if region_ids is not None else None

                        _, _, _, _, logits_final, _ = self.model._forward_multihead(
                            input_ids=batch_input,
                            attention_mask=batch_attn,
                            v_gene_ids=_v,
                            j_gene_ids=_j,
                            region_ids=_r,
                        )

                        for i, (pos, aa_id) in enumerate(zip(chunk_pos, chunk_aa_ids)):
                            tok_pos = pos + 1
                            log_probs = F.log_softmax(logits_final[i, tok_pos], dim=-1)
                            ci = chunk_start + i

                            gl_arr[ci] = log_probs[aa_id].item()
                            lower_id = self.model.lowercase_aa_token_ids.get(aa_id, aa_id)
                            ngl_arr[ci] = log_probs[lower_id].item()
                            marg_arr[ci] = self._marginalized_logp(log_probs, aa_id)
                            exact_arr[ci] = log_probs[aa_id].item()

                # Get AA token IDs at mutation positions
                wt_aa_ids = [
                    self.uppercase_aa_to_idx.get(raw_wt[idx][p], 0)
                    for p in mut_positions
                ]
                mut_aa_ids = [
                    self.uppercase_aa_to_idx.get(raw_mut[idx][p], 0)
                    for p in mut_positions
                ]

                # Masked forward for WT and mutant
                _extract_logprobs(
                    wt_ids, wt_attn, mut_positions, wt_aa_ids,
                    (wt_gl, wt_ngl, wt_marg, wt_exact),
                )
                _extract_logprobs(
                    mut_ids, mut_attn, mut_positions, mut_aa_ids,
                    (mut_gl, mut_ngl, mut_marg, mut_exact),
                )

                def _build_score(mut_arr, wt_arr):
                    delta = mut_arr - wt_arr
                    return {"score": float(delta.sum()), "per_position": delta}

                all_results.append({
                    "positions": mut_positions,
                    "marginalized": _build_score(mut_marg, wt_gl),
                    "gl": _build_score(mut_gl, wt_gl),
                    "ngl": _build_score(mut_ngl, wt_gl),
                    "exact": _build_score(mut_exact, wt_exact),
                })

            return _unwrap_single(all_results, was_single)

    # =========================================================================
    # Internal Forward Pass
    # =========================================================================

    @torch.no_grad()
    def _forward_batch(
        self,
        sequences: List[str],
        batch_size: int,
        mask_positions: Optional[List[int]] = None,
        v_gene_ids: Optional[torch.Tensor] = None,
        j_gene_ids: Optional[torch.Tensor] = None,
        region_ids: Optional[torch.Tensor] = None,
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor, int]]:
        """Run forward pass and collect per-sequence unpadded outputs.

        Args:
            sequences: Formatted sequence strings.
            batch_size: Batch size for inference.
            mask_positions: Optional 0-indexed AA positions to mask before
                the forward pass.  The same positions are masked for every
                sequence in the batch.
            v_gene_ids: Optional V-gene IDs ``[N]``.
            j_gene_ids: Optional J-gene IDs ``[N]``.
            region_ids: Optional region IDs ``[N, L]``.

        Returns:
            List of ``(logits_final, logits_aa, logits_mut, alpha, hidden, seq_len)``
            tuples, one per sequence.  All tensors are on CPU.
        """
        results = []
        n = len(sequences)
        mask_token_id = self.tokenizer.mask_token_id

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_seqs = sequences[start:end]

            input_ids, attn_mask = self._tokenize(batch_seqs)

            # Apply masking: mask_positions are 0-indexed AA positions
            # Token position = AA position + 1 (for leading CLS)
            if mask_positions:
                for pos in mask_positions:
                    tok_pos = pos + 1  # +1 for CLS token
                    if tok_pos < input_ids.shape[1] - 1:  # don't mask EOS
                        input_ids[:, tok_pos] = mask_token_id

            # Slice conditioning tensors for this mini-batch
            batch_v = v_gene_ids[start:end].to(self.device) if v_gene_ids is not None else None
            batch_j = j_gene_ids[start:end].to(self.device) if j_gene_ids is not None else None
            batch_r = region_ids[start:end, :input_ids.shape[1]].to(self.device) if region_ids is not None else None

            logits_aa, _, logits_mut, alpha, logits_final, hidden = self.model._forward_multihead(
                input_ids=input_ids,
                attention_mask=attn_mask,
                v_gene_ids=batch_v,
                j_gene_ids=batch_j,
                region_ids=batch_r,
            )

            for i in range(len(batch_seqs)):
                seq_len = attn_mask[i].sum().item()
                results.append((
                    logits_final[i, :seq_len].cpu(),
                    logits_aa[i, :seq_len].cpu(),
                    logits_mut[i, :seq_len].cpu(),
                    alpha[i, :seq_len].cpu() if alpha is not None else None,
                    hidden[i, :seq_len].cpu(),
                    seq_len,
                ))

        return results

    @torch.no_grad()
    def _forward_batch_from_ids(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        batch_size: int,
        mask_positions: Optional[List[int]] = None,
        v_gene_ids: Optional[torch.Tensor] = None,
        j_gene_ids: Optional[torch.Tensor] = None,
        region_ids: Optional[torch.Tensor] = None,
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor, int]]:
        """Run forward pass from pre-tokenized input IDs.

        Same as ``_forward_batch`` but skips tokenization.

        Args:
            input_ids: ``[B, L]`` token IDs.
            attention_mask: ``[B, L]`` attention mask.
            batch_size: Batch size for inference.
            mask_positions: Optional 0-indexed AA positions to mask.
            v_gene_ids: Optional V-gene IDs ``[B]``.
            j_gene_ids: Optional J-gene IDs ``[B]``.
            region_ids: Optional region IDs ``[B, L]``.

        Returns:
            List of ``(logits_final, logits_aa, logits_mut, alpha, hidden, seq_len)``
            tuples, one per sequence.
        """
        results = []
        n = input_ids.shape[0]
        mask_token_id = self.tokenizer.mask_token_id

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_ids = input_ids[start:end].clone()
            batch_attn = attention_mask[start:end]

            if mask_positions:
                for pos in mask_positions:
                    tok_pos = pos + 1
                    if tok_pos < batch_ids.shape[1] - 1:
                        batch_ids[:, tok_pos] = mask_token_id

            batch_v = v_gene_ids[start:end].to(self.device) if v_gene_ids is not None else None
            batch_j = j_gene_ids[start:end].to(self.device) if j_gene_ids is not None else None
            batch_r = region_ids[start:end, :batch_ids.shape[1]].to(self.device) if region_ids is not None else None

            logits_aa, _, logits_mut, alpha, logits_final, hidden = self.model._forward_multihead(
                input_ids=batch_ids,
                attention_mask=batch_attn,
                v_gene_ids=batch_v,
                j_gene_ids=batch_j,
                region_ids=batch_r,
            )

            for i in range(end - start):
                seq_len = batch_attn[i].sum().item()
                results.append((
                    logits_final[i, :seq_len].cpu(),
                    logits_aa[i, :seq_len].cpu(),
                    logits_mut[i, :seq_len].cpu(),
                    alpha[i, :seq_len].cpu() if alpha is not None else None,
                    hidden[i, :seq_len].cpu(),
                    seq_len,
                ))

        return results

    @torch.no_grad()
    def forward(
        self,
        sequences: Optional[Union[str, List[str]]] = None,
        *,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        heavy_chains: Optional[Union[str, List[str]]] = None,
        light_chains: Optional[Union[str, List[str]]] = None,
        mask_positions: Optional[List[int]] = None,
        v_gene_ids: Optional[torch.Tensor] = None,
        j_gene_ids: Optional[torch.Tensor] = None,
        region_ids: Optional[torch.Tensor] = None,
        batch_size: int = 32,
    ) -> Union[Dict[str, np.ndarray], List[Dict[str, np.ndarray]]]:
        """Run a single forward pass and return all model outputs.

        Returns raw logits (pre-softmax), per-residue embeddings, alpha
        gating, and origin logits --- everything from one forward pass.

        Four input modes:

        1. **Unpaired** --- pass ``sequences`` (positional or keyword)::

            result = model.forward("EVQLV...")

        2. **Single chain with explicit type**::

            result = model.forward(heavy_chains="EVQLV...")
            result = model.forward(light_chains="DIQMT...")

        3. **Paired** --- pass both ``heavy_chains`` and ``light_chains``::

            result = model.forward(heavy_chains="EVQLV...", light_chains="DIQMT...")
            # Returns {"heavy": {...}, "light": {...}}

        4. **Pre-tokenized** --- pass ``input_ids`` and ``attention_mask``::

            result = model.forward(input_ids=ids, attention_mask=mask)

        For GL/NGL-specific logits, slice using :attr:`GL_INDICES` /
        :attr:`NGL_INDICES`::

            gl  = result["final_logits"][:, model.GL_INDICES]   # [L, 20]

        Args:
            sequences: Amino acid sequence(s) for unpaired input.  Cannot be
                used together with ``heavy_chains`` or ``light_chains``.
            input_ids: Pre-tokenized input IDs ``[B, L]`` from
                :class:`PrismTokenizer`.  Mutually exclusive with string args.
            attention_mask: Attention mask ``[B, L]`` matching ``input_ids``.
            heavy_chains: Heavy chain sequence(s).  When used alone, behaves
                like ``sequences``.  When used with ``light_chains``, returns
                paired results with ``"heavy"`` and ``"light"`` sub-dicts.
            light_chains: Light chain sequence(s).  When used alone, behaves
                like ``sequences``.  When used with ``heavy_chains``, returns
                paired results.
            mask_positions: Optional 0-indexed AA positions to mask before
                the forward pass.  Same positions masked for all sequences.
            batch_size: Batch size for inference.

        Returns:
            **Unpaired** (single sequence) --- dict with keys:
                ``"final_logits"`` ``[L, 53]``,
                ``"aa_logits"`` ``[L, 33]``,
                ``"origin_logits"`` ``[L]``,
                ``"alpha"`` ``[L]``,
                ``"embedding"`` ``[L, H]``.

            **Paired** (single) --- ``{"heavy": {dict}, "light": {dict}}``.

            **List input** --- list of the above dicts.

        Raises:
            ValueError: If ``sequences`` is used with ``heavy_chains`` or
                ``light_chains``, or if no input is provided.
        """
        _use_input_ids = input_ids is not None

        if _use_input_ids:
            # --- Pre-tokenized path ---
            str_args = (sequences, heavy_chains, light_chains)
            if any(a is not None for a in str_args):
                raise ValueError(
                    "Cannot specify both `input_ids` and "
                    "`sequences`/`heavy_chains`/`light_chains`."
                )
            if attention_mask is None:
                attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
                attention_mask = attention_mask.unsqueeze(0)
            was_single = input_ids.shape[0] == 1
            is_paired = False  # Can't auto-detect paired from input_ids

            outputs = self._forward_batch_from_ids(
                input_ids, attention_mask, batch_size, mask_positions=mask_positions,
                v_gene_ids=v_gene_ids, j_gene_ids=j_gene_ids, region_ids=region_ids,
            )

            all_results = []
            for idx, (logits_final, logits_aa, logits_mut, alpha, hidden, seq_len) in enumerate(outputs):
                def _build_dict(final_t, aa_t, mut_t, alpha_t, hidden_t):
                    d = {
                        "final_logits": final_t.numpy(),
                        "aa_logits": aa_t.numpy(),
                        "origin_logits": mut_t.numpy(),
                        "alpha": alpha_t.numpy() if alpha_t is not None else np.zeros(final_t.shape[0]),
                        "embedding": hidden_t.numpy(),
                    }
                    return d

                # Strip leading CLS and trailing EOS to get AA-only outputs
                inner_final = logits_final[1:-1]
                inner_aa = logits_aa[1:-1]
                inner_mut = logits_mut[1:-1]
                inner_alpha = alpha[1:-1] if alpha is not None else None
                inner_hidden = hidden[1:-1]

                all_results.append(_build_dict(
                    inner_final, inner_aa, inner_mut, inner_alpha, inner_hidden,
                ))

            return _unwrap_single(all_results, was_single)

        # --- String path ---
        if sequences is not None and (heavy_chains is not None or light_chains is not None):
            raise ValueError(
                "Cannot specify both `sequences` and `heavy_chains`/`light_chains`. "
                "Use `sequences` for unpaired input, or `heavy_chains`/`light_chains` for explicit chain types."
            )
        if sequences is None and heavy_chains is None and light_chains is None:
            raise ValueError(
                "Must provide one of: `sequences`, `heavy_chains`, or `light_chains`."
            )

        is_paired = heavy_chains is not None and light_chains is not None

        if sequences is not None:
            # Unpaired mode
            seqs, was_single = _ensure_list(sequences)
            formatted = seqs
            raw_heavy = seqs
            raw_light = None
        elif is_paired:
            # Paired mode
            heavy_chains, was_single = _ensure_list(heavy_chains)
            light_chains, _ = _ensure_list(light_chains)
            raw_heavy = heavy_chains
            raw_light = light_chains
            formatted = self._format_sequences(heavy_chains, light_chains)
        else:
            # Single chain (heavy_chains= only or light_chains= only)
            single = heavy_chains if heavy_chains is not None else light_chains
            seqs, was_single = _ensure_list(single)
            formatted = seqs
            raw_heavy = seqs
            raw_light = None

        v_gene_ids, j_gene_ids, region_ids = self._auto_conditioning(
            raw_heavy, raw_light, v_gene_ids, j_gene_ids, region_ids,
        )

        outputs = self._forward_batch(
            formatted, batch_size, mask_positions=mask_positions,
            v_gene_ids=v_gene_ids, j_gene_ids=j_gene_ids, region_ids=region_ids,
        )

        all_results = []
        for idx, (logits_final, logits_aa, logits_mut, alpha, hidden, seq_len) in enumerate(outputs):

            def _build_dict(final_t, aa_t, mut_t, alpha_t, hidden_t):
                d = {
                    "final_logits": final_t.numpy(),
                    "aa_logits": aa_t.numpy(),
                    "origin_logits": mut_t.numpy(),
                    "alpha": alpha_t.numpy() if alpha_t is not None else np.zeros(final_t.shape[0]),
                    "embedding": hidden_t.numpy(),
                }
                return d

            # Strip leading CLS and trailing EOS to get AA-only outputs
            inner_final = logits_final[1:-1]
            inner_aa = logits_aa[1:-1]
            inner_mut = logits_mut[1:-1]
            inner_alpha = alpha[1:-1] if alpha is not None else None
            inner_hidden = hidden[1:-1]

            if is_paired:
                len_h = len(raw_heavy[idx])
                len_l = len(raw_light[idx])
                heavy_end = len_h
                light_start = len_h + 2  # skip 2 CLS separators

                def _slice(t, start, end):
                    return t[start:end] if t is not None else None

                all_results.append({
                    "heavy": _build_dict(
                        inner_final[:heavy_end], inner_aa[:heavy_end],
                        inner_mut[:heavy_end], _slice(inner_alpha, 0, heavy_end),
                        inner_hidden[:heavy_end],
                    ),
                    "light": _build_dict(
                        inner_final[light_start:light_start + len_l],
                        inner_aa[light_start:light_start + len_l],
                        inner_mut[light_start:light_start + len_l],
                        _slice(inner_alpha, light_start, light_start + len_l),
                        inner_hidden[light_start:light_start + len_l],
                    ),
                })
            else:
                all_results.append(_build_dict(
                    inner_final, inner_aa, inner_mut, inner_alpha, inner_hidden,
                ))

        return _unwrap_single(all_results, was_single)

    # =========================================================================
    # PLL-Guided Generation
    # =========================================================================

    @torch.no_grad()
    def generate(
        self,
        *,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        heavy_chains: Optional[Union[str, List[str]]] = None,
        light_chains: Optional[Union[str, List[str]]] = None,
        n_samples: int = 100,
        n_mutations: int = 5,
        mode: str = "full",
        seed: Optional[int] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        pool_size: int = 30,
        position_temperature: float = 1.0,
        exclude_positions: Optional[np.ndarray] = None,
        region_labels: Optional[np.ndarray] = None,
        heavy_chain_length: Optional[int] = None,
        return_masked_data: bool = False,
        masked_data: Optional[Dict[str, np.ndarray]] = None,
        randomize_n_mutations: bool = False,
        v_gene_ids: Optional[torch.Tensor] = None,
        j_gene_ids: Optional[torch.Tensor] = None,
        region_ids: Optional[torch.Tensor] = None,
        batch_size: int = 32,
    ) -> Union[List[Dict], Tuple[List[Dict], Dict[str, np.ndarray]]]:
        """Generate antibody variants using PLL-guided sampling.

        Algorithm:

        1. **Collect** --- mask each position one at a time, collect
           pre-gating logits (L forward passes, cached and reusable).
        2. **Select positions** --- rank by WT log-probability, sample
           via Gumbel-Top-k with controllable temperature.
        3. **Sample amino acids** --- draw from GL, NGL, marginalized,
           or region-specific logits with temperature, top-k, and
           nucleus sampling.

        Args:
            input_ids: Pre-tokenized input IDs ``[1, L]`` from
                :class:`PrismTokenizer`.  Mutually exclusive with
                string arguments.
            attention_mask: Attention mask ``[1, L]``.
            heavy_chains: Heavy chain sequence(s) for string convenience.
            light_chains: Light chain sequence(s) for paired input.
            n_samples: Number of variants to generate.
            n_mutations: Max mutations per variant.
            mode: Sampling mode: ``"full"``, ``"gl"``, ``"ngl"``, or
                ``"region_specific"``.
            seed: Random seed for reproducibility.
            temperature: AA sampling temperature (lower = more
                conservative).
            top_k: Only consider top-k AAs per position.
            top_p: Nucleus sampling threshold.
            pool_size: Candidate pool size for position selection
                (top-N worst positions).
            position_temperature: Temperature for Gumbel-Top-k position
                selection (lower = more deterministic).
            exclude_positions: 0-indexed AA positions to never mutate.
            region_labels: ``[L]`` array of 0 (FR) or 1 (CDR) labels
                for ``"region_specific"`` mode.
            heavy_chain_length: Length of heavy chain (for auto region
                detection via ANARCI in ``"region_specific"`` mode).
            return_masked_data: If True, return
                ``(variants, masked_data)`` so the cache can be reused.
            masked_data: Pre-computed masked logits from a previous call
                with ``return_masked_data=True``.  Skips L forward passes.
            randomize_n_mutations: If True, each sample draws
                ``n_mut ~ Beta(2,1)`` in ``[1, n_mutations]``.
            batch_size: Forward pass batch size for collecting logits.

        Returns:
            List of variant dicts, each with:

            - ``"sequence"``: str --- mutated amino acid sequence
            - ``"mutations"``: str --- e.g. ``"S7A,G10D"`` (1-indexed)
            - ``"positions"``: list of int --- 0-indexed mutation positions
            - ``"mode"``: str --- the sampling mode used
            - ``"n_mut"``: int --- actual number of mutations

            If ``return_masked_data=True``, returns
            ``(variants, masked_data_dict)`` instead.

        Raises:
            ValueError: If input arguments are inconsistent.
        """
        from scipy.special import logsumexp as scipy_logsumexp

        _VALID_MODES = {"full", "gl", "ngl", "region_specific"}
        if mode not in _VALID_MODES:
            raise ValueError(
                f"Unknown mode: '{mode}'. Must be one of {sorted(_VALID_MODES)}."
            )

        _use_input_ids = input_ids is not None
        if _use_input_ids:
            str_args = (heavy_chains, light_chains)
            if any(a is not None for a in str_args):
                raise ValueError(
                    "Cannot specify both `input_ids` and "
                    "`heavy_chains`/`light_chains`."
                )
            input_ids = input_ids.to(self.device)
            if attention_mask is None:
                attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
            attention_mask = attention_mask.to(self.device)
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
                attention_mask = attention_mask.unsqueeze(0)
            # Use first sequence only (generate is single-sequence)
            seq_ids = input_ids[0:1]
            seq_attn = attention_mask[0:1]
        else:
            if heavy_chains is None:
                raise ValueError(
                    "Must provide `input_ids` or `heavy_chains` for generation."
                )
            if isinstance(heavy_chains, list):
                heavy_chains = heavy_chains[0]
            if light_chains is not None and isinstance(light_chains, list):
                light_chains = light_chains[0]
            if light_chains is not None:
                formatted = self._format_sequences([heavy_chains], [light_chains])
            else:
                formatted = [heavy_chains]
            seq_ids, seq_attn = self._tokenize(formatted)

            # Auto-fill gene/region conditioning
            v_gene_ids, j_gene_ids, region_ids = self._auto_conditioning(
                [heavy_chains],
                [light_chains] if light_chains is not None else None,
                v_gene_ids, j_gene_ids, region_ids,
            )

        # --- Resolve WT combined sequence ---
        special_ids_set = {
            self.tokenizer.cls_token_id,
            self.tokenizer.eos_token_id,
            self.tokenizer.pad_token_id,
        }
        special_ids_set.discard(None)
        mask_token_id = self.tokenizer.mask_token_id
        seq_len = seq_attn[0].sum().item()

        maskable = []
        for pos in range(seq_len):
            if seq_ids[0, pos].item() not in special_ids_set:
                maskable.append(pos)

        n_positions = len(maskable)

        # Build the WT combined sequence from token IDs
        id_to_aa = {}
        for aa in self.AA_ORDER:
            upper_id = self.tokenizer.convert_tokens_to_ids(aa)
            id_to_aa[upper_id] = aa
            lower_id = self.tokenizer.convert_tokens_to_ids(aa.lower())
            if lower_id != self.tokenizer.unk_token_id:
                id_to_aa[lower_id] = aa  # NGL tokens also decode to uppercase AA

        combined_seq_chars = []
        for tok_pos in maskable:
            tok_id = seq_ids[0, tok_pos].item()
            combined_seq_chars.append(id_to_aa.get(tok_id, "X"))
        combined_seq = "".join(combined_seq_chars)

        # --- Step 1: Collect masked logits (or reuse cache) ---
        if masked_data is None:
            logits_53_all = np.zeros((n_positions, 53), dtype=np.float32)

            for chunk_start in tqdm(
                range(0, n_positions, batch_size),
                desc="Collecting masked logits",
                total=(n_positions + batch_size - 1) // batch_size,
            ):
                chunk_positions = maskable[chunk_start:chunk_start + batch_size]
                bs = len(chunk_positions)

                batch_input = seq_ids.expand(bs, -1).clone()
                batch_attn = seq_attn.expand(bs, -1)
                for i, pos in enumerate(chunk_positions):
                    batch_input[i, pos] = mask_token_id

                _v = v_gene_ids[0:1].expand(bs).to(self.device) if v_gene_ids is not None else None
                _j = j_gene_ids[0:1].expand(bs).to(self.device) if j_gene_ids is not None else None
                _r = region_ids[0:1, :batch_input.shape[1]].expand(bs, -1).to(self.device) if region_ids is not None else None

                logits_aa, _, _, _, logits_final, _ = self.model._forward_multihead(
                    input_ids=batch_input,
                    attention_mask=batch_attn,
                    v_gene_ids=_v,
                    j_gene_ids=_j,
                    region_ids=_r,
                )

                # Use logits_aa (pre-gating) for generation
                source_logits = logits_aa
                for i, pos in enumerate(chunk_positions):
                    idx = chunk_start + i
                    vocab_size = min(source_logits.shape[-1], 53)
                    logits_53_all[idx, :vocab_size] = source_logits[i, pos].cpu().numpy()[:vocab_size]

            # Compute WT log-probs
            gl_indices = self.GL_INDICES
            ngl_indices = self.NGL_INDICES
            logits_t = torch.from_numpy(logits_53_all).float()
            log_probs_all = F.log_softmax(logits_t, dim=-1).numpy()

            aa_to_idx = {aa: i for i, aa in enumerate(self.AA_ORDER)}
            wt_gl = np.zeros(n_positions, dtype=np.float64)
            wt_ngl = np.zeros(n_positions, dtype=np.float64)
            wt_marg = np.zeros(n_positions, dtype=np.float64)

            for pos_i in range(n_positions):
                aa = combined_seq[pos_i].upper()
                if aa not in aa_to_idx:
                    continue
                aa_i = aa_to_idx[aa]
                gl_lp = log_probs_all[pos_i, gl_indices[aa_i]]
                ngl_lp = log_probs_all[pos_i, ngl_indices[aa_i]]
                wt_gl[pos_i] = gl_lp
                wt_ngl[pos_i] = ngl_lp
                wt_marg[pos_i] = scipy_logsumexp([gl_lp, ngl_lp])

            masked_data = {
                "logits_53": logits_53_all,
                "wt_gl": wt_gl,
                "wt_ngl": wt_ngl,
                "wt_marg": wt_marg,
                "combined_seq": combined_seq,
            }

        logits_53 = masked_data["logits_53"]
        wt_gl = masked_data["wt_gl"]
        wt_ngl = masked_data["wt_ngl"]
        wt_marg = masked_data["wt_marg"]
        combined_seq = masked_data.get("combined_seq", combined_seq)

        gl_indices = self.GL_INDICES
        ngl_indices = self.NGL_INDICES

        # --- Resolve region labels for region_specific mode ---
        if mode == "region_specific" and region_labels is None:
            try:
                region_labels = self._detect_regions(
                    combined_seq, heavy_chain_length=heavy_chain_length,
                )
            except Exception:
                raise ValueError(
                    "region_specific mode requires `region_labels` or ANARCI. "
                    "Install ANARCI (`pip install anarci`) or provide labels manually."
                )

        # --- Select position-scoring array based on mode ---
        if mode == "gl":
            wt_scores = wt_gl
        elif mode == "ngl":
            wt_scores = wt_ngl
        elif mode == "region_specific" and region_labels is not None:
            wt_scores = np.where(region_labels == 0, wt_gl, wt_ngl)
        else:  # "full" or fallback
            wt_scores = wt_marg

        # --- Step 2 & 3: Generate variants ---
        rng = np.random.default_rng(seed)
        records = []

        for i in range(n_samples):
            sample_seed = rng.integers(0, 2**31)

            actual_n = n_mutations
            if randomize_n_mutations:
                actual_n = int(np.round(1 + (n_mutations - 1) * rng.beta(2, 1)))
                actual_n = min(actual_n, n_mutations)

            positions = self._select_positions(
                wt_scores, actual_n, pool_size,
                position_temperature=position_temperature,
                seed=sample_seed,
                exclude_positions=exclude_positions,
            )

            selected_logits = logits_53[positions]

            selected_regions = None
            if region_labels is not None and mode == "region_specific":
                selected_regions = region_labels[positions]

            new_aas = self._sample_amino_acids(
                selected_logits, mode=mode,
                gl_indices=gl_indices, ngl_indices=ngl_indices,
                temperature=temperature,
                top_k=top_k, top_p=top_p,
                seed=sample_seed + 1,
                region_labels=selected_regions,
            )

            variant_seq = self._apply_mutations(combined_seq, positions, new_aas)
            mutation_str = self._format_mutations(combined_seq, positions, new_aas)

            records.append({
                "sequence": variant_seq,
                "mutations": mutation_str,
                "positions": positions.tolist(),
                "mode": mode,
                "n_mut": len(positions),
            })

        if return_masked_data:
            return records, masked_data
        return records

    # =========================================================================
    # Generate helpers (private)
    # =========================================================================

    @staticmethod
    def _select_positions(
        wt_log_probs: np.ndarray,
        n_mutations: int,
        pool_size: int,
        position_temperature: float = 1.0,
        seed: Optional[int] = None,
        exclude_positions: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Select mutation positions using Gumbel-Top-k trick."""
        wt_log_probs = wt_log_probs.copy()
        if exclude_positions is not None and len(exclude_positions) > 0:
            wt_log_probs[exclude_positions] = np.inf

        L = int((wt_log_probs < np.inf).sum())
        n_mutations = min(n_mutations, L)
        pool_size = min(pool_size, L)
        pool_size = max(pool_size, n_mutations)

        rng = np.random.default_rng(seed)
        sorted_indices = np.argsort(wt_log_probs)
        pool = sorted_indices[:pool_size]

        if pool_size == n_mutations:
            return np.sort(pool)

        pool_scores = wt_log_probs[pool]
        uniform = rng.uniform(1e-10, 1.0 - 1e-10, size=pool_size)
        gumbel_noise = -np.log(-np.log(uniform))
        perturbed = pool_scores + gumbel_noise * position_temperature
        selected_in_pool = np.argsort(perturbed)[:n_mutations]
        selected = pool[selected_in_pool]
        return np.sort(selected)

    @staticmethod
    def _sample_amino_acids(
        logits_53: np.ndarray,
        mode: str,
        gl_indices: np.ndarray,
        ngl_indices: np.ndarray,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        seed: Optional[int] = None,
        region_labels: Optional[np.ndarray] = None,
        aa_order: str = "ACDEFGHIKLMNPQRSTVWY",
    ) -> List[str]:
        """Sample amino acids from head-specific logits."""
        K = logits_53.shape[0]
        rng = np.random.default_rng(seed)

        if mode == "gl":
            logits_20 = logits_53[:, gl_indices]
        elif mode == "ngl":
            logits_20 = logits_53[:, ngl_indices]
        elif mode == "full":
            gl_logits = logits_53[:, gl_indices]
            ngl_logits = logits_53[:, ngl_indices]
            logits_20 = np.logaddexp(gl_logits, ngl_logits)
        elif mode == "region_specific":
            if region_labels is None:
                raise ValueError("region_labels required for 'region_specific' mode")
            gl_logits = logits_53[:, gl_indices]
            ngl_logits = logits_53[:, ngl_indices]
            logits_20 = np.where(
                region_labels[:, None] == 0,
                gl_logits,
                ngl_logits,
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

        if temperature <= 0:
            temperature = 1e-10

        logits_20 = logits_20 / temperature

        sampled_aas = []
        for i in range(K):
            logits_i = logits_20[i].copy()

            if top_k is not None and top_k < 20:
                threshold = np.partition(logits_i, -top_k)[-top_k]
                logits_i[logits_i < threshold] = -np.inf

            if top_p is not None:
                sorted_idx = np.argsort(logits_i)[::-1]
                sorted_logits = logits_i[sorted_idx]
                probs = np.exp(sorted_logits - sorted_logits.max())
                probs /= probs.sum()
                cum_probs = np.cumsum(probs)
                cutoff = np.searchsorted(cum_probs, top_p) + 1
                mask = np.ones(20, dtype=bool)
                mask[sorted_idx[:cutoff]] = False
                logits_i[mask] = -np.inf

            logits_i = logits_i - logits_i.max()
            probs = np.exp(logits_i)
            probs_sum = probs.sum()
            if probs_sum <= 0:
                probs = np.ones(20) / 20
            else:
                probs /= probs_sum

            aa_idx = rng.choice(20, p=probs)
            sampled_aas.append(aa_order[aa_idx])

        return sampled_aas

    @staticmethod
    def _apply_mutations(
        wt_sequence: str,
        positions: np.ndarray,
        new_aas: List[str],
    ) -> str:
        """Apply mutations at specified positions."""
        seq_list = list(wt_sequence)
        for pos, aa in zip(positions, new_aas):
            seq_list[pos] = aa
        return "".join(seq_list)

    @staticmethod
    def _format_mutations(
        wt_sequence: str,
        positions: np.ndarray,
        new_aas: List[str],
    ) -> str:
        """Format mutations as 'A1W,D4Y' (1-indexed)."""
        parts = []
        for pos, new_aa in zip(positions, new_aas):
            wt_aa = wt_sequence[pos]
            parts.append(f"{wt_aa}{pos + 1}{new_aa}")
        return ",".join(parts)

    @staticmethod
    def _detect_regions(
        combined_seq: str,
        heavy_chain_length: Optional[int] = None,
    ) -> np.ndarray:
        """Detect FR/CDR regions using ANARCI.

        Args:
            combined_seq: Combined VH+VL amino acid sequence.
            heavy_chain_length: Length of heavy chain for splitting.

        Returns:
            [L] array of 0 (FR) or 1 (CDR) labels.
        """
        try:
            from anarci import anarci as run_anarci
        except ImportError:
            raise ImportError(
                "ANARCI is required for automatic region detection. "
                "Install it with: pip install anarci"
            )

        L = len(combined_seq)
        labels = np.zeros(L, dtype=np.int32)

        # IMGT CDR definitions (numbering ranges)
        cdr_ranges = {
            "H": [(27, 38), (56, 65), (105, 117)],   # CDR1, CDR2, CDR3
            "L": [(27, 38), (56, 65), (105, 117)],
        }

        if heavy_chain_length is not None:
            vh_seq = combined_seq[:heavy_chain_length]
            vl_seq = combined_seq[heavy_chain_length:]
            chains = [("H", vh_seq, 0), ("L", vl_seq, heavy_chain_length)]
        else:
            chains = [("H", combined_seq, 0)]

        for chain_type, seq, offset in chains:
            try:
                results = run_anarci([("seq", seq)], scheme="imgt")
                if results and results[0] and results[0][0]:
                    numbering = results[0][0][0]
                    for seq_idx_in_chain, (imgt_pos, _insertion) in enumerate(numbering):
                        imgt_num = imgt_pos[0]
                        for cdr_start, cdr_end in cdr_ranges[chain_type]:
                            if cdr_start <= imgt_num <= cdr_end:
                                global_pos = offset + seq_idx_in_chain
                                if global_pos < L:
                                    labels[global_pos] = 1
                                break
            except Exception:
                pass

        return labels

    # =========================================================================
    # Germline Prediction
    # =========================================================================

    def _detect_paired_split(self, ids: torch.Tensor) -> Optional[int]:
        """Find consecutive ``<cls><cls>`` separator in token IDs.

        In PRISM paired format the tokenized sequence is::

            <cls> VH_tokens <cls> <cls> VL_tokens <eos>

        The leading ``<cls>`` at position 0 is always single. The
        ``<cls><cls>`` pair only appears as the heavy/light separator.

        Args:
            ids: 1-D token ID tensor for a single sequence.

        Returns:
            Heavy chain AA length (number of AA tokens before the
            separator), or ``None`` if no ``<cls><cls>`` is found
            (i.e. the sequence is unpaired).
        """
        cls_id = self.tokenizer.cls_token_id
        for i in range(1, len(ids) - 1):
            if ids[i].item() == cls_id and ids[i + 1].item() == cls_id:
                # Position i is the first separator <cls>.
                # AA tokens are at positions 1 .. i-1 → length = i - 1
                return i - 1
        return None

    @torch.no_grad()
    def predict_germline(
        self,
        heavy_chains: Optional[Union[str, List[str]]] = None,
        light_chains: Optional[Union[str, List[str]]] = None,
        *,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        ngl_threshold: float = 0.5,
        v_gene_ids: Optional[torch.Tensor] = None,
        j_gene_ids: Optional[torch.Tensor] = None,
        region_ids: Optional[torch.Tensor] = None,
        batch_size: int = 32,
    ) -> Union[Dict, List[Dict]]:
        """Predict germline (unmutated) sequences from antibody sequences.

        Identifies somatically hypermutated (NGL) positions via the origin
        head and reverts them to the top-scoring germline amino acid
        predicted by the model.

        Algorithm:

        1. Run a single forward pass to obtain ``origin_logits`` (for NGL
           detection) and ``final_logits`` (for GL amino acid prediction).
        2. Positions where ``P(NGL) > ngl_threshold`` are classified as
           non-germline.
        3. At NGL positions the observed residue is replaced with the
           highest-scoring GL amino acid from ``final_logits``.

        Four input modes (same as :meth:`forward`):

        1. **Paired** — ``heavy_chains`` + ``light_chains``
        2. **Unpaired** — ``heavy_chains`` only
        3. **Pre-tokenized paired** — ``input_ids`` containing a
           ``<cls><cls>`` separator (auto-detected)
        4. **Pre-tokenized unpaired** — ``input_ids`` without separator

        Args:
            heavy_chains: Heavy chain sequence(s).
            light_chains: Light chain sequence(s) for paired input.
            input_ids: Pre-tokenized input IDs ``[B, L]``.  Mutually
                exclusive with ``heavy_chains`` / ``light_chains``.
            attention_mask: Attention mask ``[B, L]``.
            ngl_threshold: Positions with ``P(NGL) > threshold`` are
                reverted to the predicted germline residue.
                Default ``0.5``.
            batch_size: Batch size for inference.

        Returns:
            **Paired** (single) — dict with ``"heavy"`` and ``"light"``
            sub-dicts, each containing:

            - ``"sequence"``: ``str`` — original amino acid sequence
            - ``"predicted_germline"``: ``str`` — germline-reverted sequence
            - ``"ngl_positions"``: ``list[int]`` — 0-indexed NGL positions
            - ``"ngl_count"``: ``int``
            - ``"ngl_probs"``: ``[L]`` numpy array of P(NGL)

            **Unpaired** (single) — flat dict with the same keys.

            **List input** — list of the above dicts.
        """
        # --- Input resolution ---
        _use_input_ids = input_ids is not None

        if _use_input_ids:
            str_args = (heavy_chains, light_chains)
            if any(a is not None for a in str_args):
                raise ValueError(
                    "Cannot specify both `input_ids` and "
                    "`heavy_chains`/`light_chains`."
                )
            if attention_mask is None:
                attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
                attention_mask = attention_mask.unsqueeze(0)
            was_single = input_ids.shape[0] == 1

            outputs = self._forward_batch_from_ids(
                input_ids, attention_mask, batch_size,
                v_gene_ids=v_gene_ids, j_gene_ids=j_gene_ids,
                region_ids=region_ids,
            )

            # Detect paired per sequence from raw input_ids
            paired_splits = []
            for b in range(input_ids.shape[0]):
                seq_len = attention_mask[b].sum().item()
                paired_splits.append(
                    self._detect_paired_split(input_ids[b, :seq_len])
                )

            # Build id_to_aa mapping for sequence reconstruction
            id_to_aa = {}
            for aa in self.AA_ORDER:
                id_to_aa[self.tokenizer.convert_tokens_to_ids(aa)] = aa
                lower_id = self.tokenizer.convert_tokens_to_ids(aa.lower())
                if lower_id != self.tokenizer.unk_token_id:
                    id_to_aa[lower_id] = aa

            # Reconstruct original AA sequences from token IDs
            raw_seqs = []
            for b in range(input_ids.shape[0]):
                seq_len = attention_mask[b].sum().item()
                chars = []
                for pos in range(seq_len):
                    tid = input_ids[b, pos].item()
                    if tid in id_to_aa:
                        chars.append(id_to_aa[tid])
                raw_seqs.append("".join(chars))

        else:
            if heavy_chains is None and light_chains is None:
                raise ValueError(
                    "Must provide one of: `heavy_chains`, or "
                    "`heavy_chains`+`light_chains`, or `input_ids`."
                )

            is_paired = heavy_chains is not None and light_chains is not None

            if is_paired:
                heavy_chains, was_single = _ensure_list(heavy_chains)
                light_chains, _ = _ensure_list(light_chains)
                raw_heavy = heavy_chains
                raw_light = light_chains
                formatted = self._format_sequences(heavy_chains, light_chains)
            else:
                single = heavy_chains if heavy_chains is not None else light_chains
                seqs, was_single = _ensure_list(single)
                raw_heavy = seqs
                raw_light = None
                formatted = seqs

            v_gene_ids, j_gene_ids, region_ids = self._auto_conditioning(
                raw_heavy, raw_light, v_gene_ids, j_gene_ids, region_ids,
            )

            outputs = self._forward_batch(
                formatted, batch_size,
                v_gene_ids=v_gene_ids, j_gene_ids=j_gene_ids,
                region_ids=region_ids,
            )

            # For string path: build raw_seqs (concatenated) and paired_splits
            raw_seqs = []
            paired_splits = []
            for idx in range(len(formatted)):
                if is_paired:
                    raw_seqs.append(raw_heavy[idx] + raw_light[idx])
                    paired_splits.append(len(raw_heavy[idx]))
                else:
                    raw_seqs.append(raw_heavy[idx])
                    paired_splits.append(None)

        # --- Process each sequence ---
        gl_indices = self._gl_indices  # [20] array of GL token IDs
        aa_order = np.array(list(self.AA_ORDER))

        all_results = []
        for idx, (logits_final, logits_aa, logits_mut, alpha, hidden, seq_len) in enumerate(outputs):
            # Strip leading CLS and trailing EOS
            inner_final = logits_final[1:-1]   # [L_total, 53]
            inner_mut = logits_mut[1:-1]       # [L_total]

            # P(NGL) via sigmoid (2-class binary origin head)
            ngl_probs = torch.sigmoid(inner_mut).numpy()

            # GL argmax: slice final logits to GL token columns, take argmax
            gl_logits = inner_final[:, gl_indices].numpy()  # [L_total, 20]
            gl_argmax_idx = np.argmax(gl_logits, axis=1)    # [L_total]
            gl_argmax_aa = aa_order[gl_argmax_idx]           # [L_total] array of chars

            # Build predicted germline
            original_seq = raw_seqs[idx]
            ngl_mask = ngl_probs > ngl_threshold
            pred_chars = list(original_seq)
            for pos in range(len(pred_chars)):
                if pos < len(ngl_mask) and ngl_mask[pos]:
                    pred_chars[pos] = gl_argmax_aa[pos]
            predicted_germline = "".join(pred_chars)

            # Split into heavy/light if paired
            split_len = paired_splits[idx]
            if split_len is not None:
                len_h = split_len
                len_l = len(original_seq) - len_h

                def _build_chain_dict(seq, pred_gl, probs, offset, length):
                    chain_probs = probs[offset:offset + length]
                    chain_ngl = np.where(chain_probs > ngl_threshold)[0].tolist()
                    return {
                        "sequence": seq,
                        "predicted_germline": pred_gl,
                        "ngl_positions": chain_ngl,
                        "ngl_count": len(chain_ngl),
                        "ngl_probs": chain_probs,
                    }

                # In paired format inner tokens are: VH (len_h) + <cls><cls> (2) + VL (len_l)
                # ngl_probs covers all inner tokens; skip the 2 separator positions
                all_results.append({
                    "heavy": _build_chain_dict(
                        original_seq[:len_h],
                        predicted_germline[:len_h],
                        ngl_probs, 0, len_h,
                    ),
                    "light": _build_chain_dict(
                        original_seq[len_h:],
                        predicted_germline[len_h:],
                        ngl_probs, len_h + 2, len_l,
                    ),
                })
            else:
                seq_len_aa = len(original_seq)
                seq_probs = ngl_probs[:seq_len_aa]
                ngl_positions = np.where(seq_probs > ngl_threshold)[0].tolist()
                all_results.append({
                    "sequence": original_seq,
                    "predicted_germline": predicted_germline,
                    "ngl_positions": ngl_positions,
                    "ngl_count": len(ngl_positions),
                    "ngl_probs": seq_probs,
                })

        return _unwrap_single(all_results, was_single)

    # =========================================================================
    # Public Utilities
    # =========================================================================

    @staticmethod
    def marginalize_logits(logits_53: torch.Tensor, model) -> torch.Tensor:
        """Marginalize 53-vocab logits to 20 effective AA log-probabilities.

        For each of the 20 amino acids, computes:
            log P(AA) = logsumexp(log P(AA_upper), log P(AA_lower))

        Args:
            logits_53: [..., 53] raw logits from the final head.
            model: The SFT_ESM2 model (for token ID mappings).

        Returns:
            [..., 20] marginalized log-probabilities.
        """
        log_probs = F.log_softmax(logits_53, dim=-1)
        shape = log_probs.shape[:-1]
        result = torch.zeros(*shape, 20, device=logits_53.device)

        for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY"):
            upper_id = model.tokenizer.convert_tokens_to_ids(aa)
            lower_id = model.lowercase_aa_token_ids.get(upper_id)
            if lower_id is not None:
                result[..., i] = torch.logsumexp(
                    torch.stack([log_probs[..., upper_id], log_probs[..., lower_id]], dim=-1),
                    dim=-1,
                )
            else:
                result[..., i] = log_probs[..., upper_id]

        return result

    # =========================================================================
    # Internal Helpers
    # =========================================================================

    def _tokenize(
        self,
        sequences: List[str],
        uppercase: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize sequences with optional uppercase forcing, padding, and device placement.

        Handles paired sequences containing ``<cls>`` separators by
        uppercasing only the amino acid portions, not the special tokens.

        Args:
            sequences: List of amino acid sequence strings.
            uppercase: If True (default), force all AA characters to
                uppercase before tokenization.  Set to False to preserve
                case (lowercase chars become NGL tokens).
        """
        cls_token = self.tokenizer.cls_token or "<cls>"
        if uppercase:
            upper_seqs = []
            for s in sequences:
                if cls_token in s:
                    # Paired sequence: uppercase each chain segment separately
                    parts = s.split(cls_token)
                    upper_seqs.append(cls_token.join(p.upper() for p in parts))
                else:
                    upper_seqs.append(s.upper())
            to_tokenize = upper_seqs
        else:
            to_tokenize = sequences

        encoded = self.tokenizer(
            to_tokenize,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attn_mask = encoded["attention_mask"].to(self.device)
        return input_ids, attn_mask

    def _encode_genes(
        self,
        genes: Optional[List[str]],
        start: int,
        end: Optional[int],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Encode V/J gene strings to tensor IDs."""
        if genes is None or self.gene_vocab is None:
            return None, None
        if end is None:
            end = start + 1

        batch_genes = genes[start:end]
        gene_ids = torch.tensor(
            [self.gene_vocab.encode(g) for g in batch_genes],
            device=self.device,
        )
        # Return same IDs for both v and j (caller should pass separate lists)
        return gene_ids, gene_ids

    def _encode_regions(
        self,
        region_masks: Optional[List[str]],
        sequences: List[str],
        start: int,
        end: Optional[int],
    ) -> Optional[torch.Tensor]:
        """Encode region mask strings to tensor.

        Region mask format: digit string like "0011122233344456666"
        Values: 0=FR1, 1=CDR1, 2=FR2, 3=CDR2, 4=FR3, 5=CDR3, 6=FR4
        Model expects +1 offset (0=padding).
        """
        if region_masks is None:
            return None
        if end is None:
            end = start + 1

        batch_masks = region_masks[start:end]
        max_len = max(len(s) for s in sequences) + 2  # +2 for special tokens

        region_ids = torch.zeros(len(batch_masks), max_len, dtype=torch.long, device=self.device)
        for i, mask_str in enumerate(batch_masks):
            if mask_str:
                for j, ch in enumerate(mask_str):
                    if j + 1 < max_len:  # +1 for CLS token
                        region_ids[i, j + 1] = int(ch) + 1  # +1 offset

        return region_ids

    def _marginalized_logp(self, log_probs: torch.Tensor, upper_token_id: int) -> float:
        """Compute marginalized log probability: logsumexp(upper, lower)."""
        lower_id = self.model.lowercase_aa_token_ids.get(upper_token_id)
        if lower_id is not None:
            return torch.logsumexp(
                torch.tensor([log_probs[upper_token_id], log_probs[lower_id]]),
                dim=0,
            ).item()
        return log_probs[upper_token_id].item()

    # =========================================================================
    # Data Loading
    # =========================================================================

    @staticmethod
    def _load_data(data_path: str) -> pd.DataFrame:
        """Load antibody data from parquet, pickle, or CSV.

        If the DataFrame lacks a ``split`` column, a random 90/5/5
        train/valid/test split is added (deterministic, seed=42).

        Args:
            data_path: Path to ``.parquet``, ``.pkl``/``.pickle``, or ``.csv`` file.

        Returns:
            DataFrame with at least a ``split`` column.

        Raises:
            ValueError: For unsupported file extensions.
        """
        path = Path(data_path)
        if path.suffix == ".parquet":
            df = pd.read_parquet(data_path)
        elif path.suffix in (".pkl", ".pickle"):
            df = pd.read_pickle(data_path)  # noqa: S301 - trusted data files only
        elif path.suffix == ".csv":
            df = pd.read_csv(data_path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        # Auto-split if no 'split' column
        if "split" not in df.columns:
            n = len(df)
            indices = np.random.RandomState(42).permutation(n)
            n_val = max(1, int(n * 0.05))
            n_test = max(1, int(n * 0.05))
            split = np.array(["train"] * n)
            split[indices[:n_val]] = "valid"
            split[indices[n_val:n_val + n_test]] = "test"
            df["split"] = split

        return df

    # =========================================================================
    # Finetuning
    # =========================================================================

    def finetune(
        self,
        data_path: str,
        output_dir: str = "prism_finetune_output",
        # Training hyperparameters
        max_steps: int = 5000,
        learning_rate: float = 1e-4,
        batch_size: int = 32,
        warmup_steps: int = 500,
        weight_decay: float = 0.01,
        mask_prob: float = 0.15,
        gradient_accumulation_steps: int = 1,
        # Trainer settings
        devices: int = 1,
        precision: str = "bf16-mixed",
        val_check_interval: Optional[Union[int, float]] = 500,
        # Data settings
        num_workers: int = 4,
        seed: int = 42,
    ) -> str:
        """Finetune the model on additional antibody data.

        All transformer layers are unfrozen.  A fresh optimizer (AdamW +
        cosine schedule) is created.  After training the best checkpoint
        is loaded back so the model is ready for inference.

        Args:
            data_path: Path to ``.parquet``, ``.pkl``, or ``.csv`` file with
                at least ``HEAVY_CHAIN_AA_SEQUENCE`` and/or
                ``LIGHT_CHAIN_AA_SEQUENCE`` columns.
            output_dir: Directory for checkpoints and logs.
            max_steps: Total training steps.
            learning_rate: Peak learning rate for AdamW.
            batch_size: Per-device batch size.
            warmup_steps: Linear warmup steps.
            weight_decay: AdamW weight decay.
            mask_prob: Masking probability for MLM training.
            gradient_accumulation_steps: Gradient accumulation steps.
            devices: Number of GPUs (or ``1`` for CPU/single-GPU).
            precision: Training precision (``"bf16-mixed"``, ``"32"``, etc.).
            val_check_interval: Validate every N steps (int) or fraction of epoch (float).
            num_workers: DataLoader workers.
            seed: Random seed.

        Returns:
            Path to the best checkpoint file.
        """
        from .io_utils import SFTDataModule

        logger = logging.getLogger(__name__)

        # 1. Load data
        df = self._load_data(data_path)
        logger.info(
            "Loaded %d sequences (%d train, %d valid, %d test)",
            len(df),
            (df["split"] == "train").sum(),
            (df["split"] == "valid").sum(),
            (df["split"] == "test").sum(),
        )

        # 2. Update model training hparams (used by configure_optimizers)
        self.model.peak_learning_rate = learning_rate
        self.model.warmup_steps = warmup_steps
        self.model.max_steps = max_steps
        self.model.WD = weight_decay
        self.model.mask_prob = mask_prob
        self.model.batch_size = batch_size

        # 3. Unfreeze all parameters
        for p in self.model.parameters():
            p.requires_grad = True

        # 4. Build DataModule
        dm = SFTDataModule(
            data_frame=df,
            batch_size=batch_size,
            mask_prob=mask_prob,
            tokenizer=self.tokenizer,
            seed=seed,
            num_workers=num_workers,
            gene_vocab=self.gene_vocab,
            use_germline_genes=self.model.use_germline_genes,
            use_region_embedding=self.model.use_region_embedding,
        )

        # 5. Checkpoint callback
        ckpt_callback = ModelCheckpoint(
            dirpath=str(Path(output_dir) / "checkpoints"),
            filename="best-{step}-{val/Final_PPL_All:.4f}",
            monitor="val/Final_PPL_All",
            mode="min",
            save_top_k=1,
            every_n_train_steps=val_check_interval if isinstance(val_check_interval, int) else None,
        )

        # 6. Build Trainer
        trainer = pl.Trainer(
            max_steps=max_steps,
            devices=devices,
            precision=precision,
            accumulate_grad_batches=gradient_accumulation_steps,
            val_check_interval=val_check_interval,
            default_root_dir=output_dir,
            callbacks=[ckpt_callback],
            enable_progress_bar=True,
            logger=pl.loggers.TensorBoardLogger(output_dir, name="finetune"),
        )

        # 7. Train
        self.model.train()
        trainer.fit(self.model, datamodule=dm)

        # 8. Load best checkpoint back
        best_path = trainer.checkpoint_callback.best_model_path
        if best_path:
            self._reload_best_checkpoint(best_path)

        # 9. Inference mode
        self.model.eval()

        return best_path

    def _reload_best_checkpoint(self, checkpoint_path: str) -> None:
        """Load weights from a checkpoint back into ``self.model``."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        # Remove non-persistent buffers
        keys_to_remove = [k for k in state_dict if "aa_indices" in k]
        for k in keys_to_remove:
            del state_dict[k]
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
