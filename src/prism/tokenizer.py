"""PrismTokenizer — HuggingFace-style tokenizer for PRISM antibody models.

Wraps the ESM2 tokenizer with PRISM-specific logic:
- 53-vocab: 33 ESM2 base tokens + 20 lowercase AA tokens (NGL)
- Automatic uppercasing of input sequences
- Paired sequence support (VH<cls><cls>VL)
- GL/NGL token ID mappings

Usage::

    from prism import PrismTokenizer

    tokenizer = PrismTokenizer()

    # Single sequence
    result = tokenizer("EVQLVESGGGLVQ", return_tensors="pt")
    # result["input_ids"], result["attention_mask"]

    # Paired heavy + light
    result = tokenizer("EVQLV...", light_chain="DIQMT...", return_tensors="pt")

    # Encode / decode
    ids = tokenizer.encode("EVQLV")
    seq = tokenizer.decode(ids, skip_special_tokens=True)

    # Paired encode / decode
    ids = tokenizer.encode_paired("EVQLV", "DIQMT")
    heavy, light = tokenizer.decode_paired(ids)
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from transformers import AutoTokenizer


AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
_LOWERCASE_AA = [aa.lower() for aa in AA_ORDER]


class PrismTokenizer:
    """HuggingFace-compatible tokenizer for PRISM antibody models.

    Attributes:
        AA_ORDER: The 20 standard amino acids in alphabetical order.
        gl_token_ids: ``{uppercase_AA: token_id}`` for germline tokens.
        ngl_token_ids: ``{lowercase_AA: token_id}`` for non-germline tokens.
        gl_to_ngl: ``{gl_token_id: ngl_token_id}`` mapping.
        vocab_size: Total vocabulary size (53).
    """

    AA_ORDER: str = AA_ORDER

    def __init__(self, model_identifier: str = "esm2_t12_35M_UR50D"):
        self._model_identifier = model_identifier
        self._tokenizer = AutoTokenizer.from_pretrained(f"facebook/{model_identifier}")

        # Add 20 lowercase AA tokens for NGL
        num_added = self._tokenizer.add_tokens(_LOWERCASE_AA)
        # If tokens were already added (re-init), num_added may be 0

        # Build GL (uppercase) token ID mapping
        self.gl_token_ids: Dict[str, int] = {}
        for aa in AA_ORDER:
            self.gl_token_ids[aa] = self._tokenizer.convert_tokens_to_ids(aa)

        # Build NGL (lowercase) token ID mapping
        self.ngl_token_ids: Dict[str, int] = {}
        for aa in _LOWERCASE_AA:
            self.ngl_token_ids[aa] = self._tokenizer.convert_tokens_to_ids(aa)

        # Build GL → NGL token ID mapping
        self.gl_to_ngl: Dict[int, int] = {}
        for aa in AA_ORDER:
            gl_id = self.gl_token_ids[aa]
            ngl_id = self.ngl_token_ids[aa.lower()]
            self.gl_to_ngl[gl_id] = ngl_id

    # =========================================================================
    # Class methods
    # =========================================================================

    @classmethod
    def from_pretrained(cls, model_identifier: str) -> "PrismTokenizer":
        """Create a PrismTokenizer from an ESM2 model identifier.

        Args:
            model_identifier: ESM2 model name, e.g. ``"esm2_t12_35M_UR50D"``.
                The ``facebook/`` prefix is added automatically.

        Returns:
            Configured PrismTokenizer instance.
        """
        return cls(model_identifier=model_identifier)

    # =========================================================================
    # Properties (delegate to internal tokenizer)
    # =========================================================================

    @property
    def vocab_size(self) -> int:
        return len(self._tokenizer)

    @property
    def cls_token_id(self) -> int:
        return self._tokenizer.cls_token_id

    @property
    def eos_token_id(self) -> int:
        return self._tokenizer.eos_token_id

    @property
    def pad_token_id(self) -> int:
        return self._tokenizer.pad_token_id

    @property
    def mask_token_id(self) -> int:
        return self._tokenizer.mask_token_id

    @property
    def unk_token_id(self) -> int:
        return self._tokenizer.unk_token_id

    @property
    def cls_token(self) -> str:
        return self._tokenizer.cls_token

    @property
    def eos_token(self) -> str:
        return self._tokenizer.eos_token

    @property
    def pad_token(self) -> str:
        return self._tokenizer.pad_token

    @property
    def mask_token(self) -> str:
        return self._tokenizer.mask_token

    # =========================================================================
    # Core: __call__
    # =========================================================================

    def __call__(
        self,
        sequences: Union[str, List[str]],
        *,
        light_chain: Optional[Union[str, List[str]]] = None,
        preserve_case: bool = False,
        return_tensors: Optional[str] = "pt",
        padding: bool = True,
        truncation: bool = True,
        max_length: int = 512,
    ) -> Dict[str, Union[torch.Tensor, np.ndarray, List]]:
        """Tokenize amino acid sequence(s).

        Args:
            sequences: Single sequence string or list of sequences.
                For paired input, these are the heavy chains.
            light_chain: Optional light chain(s). When provided,
                sequences are formatted as ``VH<cls><cls>VL``.
            preserve_case: If True, do **not** force uppercase.
                Lowercase characters are tokenized as NGL tokens.
                Default False (uppercase, backward-compatible).
            return_tensors: ``"pt"`` for PyTorch, ``"np"`` for NumPy,
                ``None`` for plain lists.
            padding: Pad shorter sequences to match the longest.
            truncation: Truncate sequences longer than ``max_length``.
            max_length: Maximum token length (including special tokens).

        Returns:
            Dict with ``"input_ids"`` and ``"attention_mask"``.

        Raises:
            ValueError: If sequences is empty or light_chain count mismatches.
        """
        seqs, was_single = self._ensure_list(sequences)

        # Validate
        for s in seqs:
            if len(s.strip()) == 0:
                raise ValueError("Empty sequence is not allowed.")

        # Strip whitespace
        seqs = [s.strip() for s in seqs]

        # Format paired sequences
        if light_chain is not None:
            lcs, _ = self._ensure_list(light_chain)
            lcs = [lc.strip() for lc in lcs]
            if len(seqs) != len(lcs):
                raise ValueError(
                    f"light_chain length mismatch: got {len(seqs)} heavy chain(s) "
                    f"but {len(lcs)} light chain(s)"
                )
            sep = self._tokenizer.cls_token
            seqs = [f"{vh}{sep}{sep}{vl}" for vh, vl in zip(seqs, lcs)]

        # Uppercase (only AA portions, not special tokens)
        if not preserve_case:
            seqs = self._uppercase_sequences(seqs)

        # Tokenize
        if padding and return_tensors is not None:
            encoded = self._tokenizer(
                seqs,
                return_tensors=return_tensors,
                padding=True,
                truncation=truncation,
                max_length=max_length,
            )
            return dict(encoded)
        elif padding and return_tensors is None:
            encoded = self._tokenizer(
                seqs,
                padding=True,
                truncation=truncation,
                max_length=max_length,
            )
            return {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
            }
        else:
            # No padding — return list of lists (variable length)
            all_ids = []
            all_masks = []
            for s in seqs:
                enc = self._tokenizer(
                    s,
                    truncation=truncation,
                    max_length=max_length,
                )
                all_ids.append(enc["input_ids"])
                all_masks.append(enc["attention_mask"])
            return {"input_ids": all_ids, "attention_mask": all_masks}

    # =========================================================================
    # encode / decode
    # =========================================================================

    def encode(self, sequence: str, preserve_case: bool = False) -> List[int]:
        """Encode a single sequence to a flat list of token IDs.

        Includes CLS and EOS tokens.

        Args:
            sequence: Amino acid sequence string.
            preserve_case: If True, do not force uppercase.

        Returns:
            List of integer token IDs.
        """
        seq = sequence.strip()
        if not preserve_case:
            seq = seq.upper()
        return self._tokenizer.encode(seq)

    def encode_paired(self, heavy_chain: str, light_chain: str) -> List[int]:
        """Encode a paired heavy + light chain sequence.

        Formats as ``VH<cls><cls>VL`` before encoding.

        Args:
            heavy_chain: Heavy chain sequence.
            light_chain: Light chain sequence.

        Returns:
            List of integer token IDs.
        """
        sep = self._tokenizer.cls_token
        formatted = f"{heavy_chain.strip().upper()}{sep}{sep}{light_chain.strip().upper()}"
        return self._tokenizer.encode(formatted)

    def decode(
        self,
        token_ids: Union[List[int], torch.Tensor, np.ndarray],
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode token IDs back to a string.

        Args:
            token_ids: Token IDs as list, tensor, or array.
            skip_special_tokens: If True, omit CLS/EOS/PAD tokens.

        Returns:
            Decoded amino acid string.
        """
        if isinstance(token_ids, (torch.Tensor, np.ndarray)):
            token_ids = token_ids.tolist()
        # Filter out padding
        if skip_special_tokens:
            token_ids = [
                t for t in token_ids
                if t != self.pad_token_id
            ]
        decoded = self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
        # Remove spaces that HF tokenizer may insert between tokens
        return decoded.replace(" ", "")

    def batch_decode(
        self,
        token_ids_batch: Union[List[List[int]], torch.Tensor, np.ndarray],
        skip_special_tokens: bool = True,
    ) -> List[str]:
        """Decode a batch of token ID sequences.

        Args:
            token_ids_batch: 2D token IDs (batch_size x seq_len).
            skip_special_tokens: If True, omit special tokens.

        Returns:
            List of decoded strings.
        """
        if isinstance(token_ids_batch, (torch.Tensor, np.ndarray)):
            token_ids_batch = token_ids_batch.tolist()
        return [
            self.decode(ids, skip_special_tokens=skip_special_tokens)
            for ids in token_ids_batch
        ]

    def decode_paired(
        self,
        token_ids: Union[List[int], torch.Tensor, np.ndarray],
        skip_special_tokens: bool = True,
    ) -> Tuple[str, str]:
        """Decode paired token IDs back to (heavy_chain, light_chain).

        Splits on the ``<cls><cls>`` separator.

        Args:
            token_ids: Token IDs from ``encode_paired()``.
            skip_special_tokens: If True, omit CLS/EOS/PAD.

        Returns:
            Tuple of ``(heavy_chain, light_chain)`` strings.

        Raises:
            ValueError: If no ``<cls><cls>`` separator is found.
        """
        if isinstance(token_ids, (torch.Tensor, np.ndarray)):
            token_ids = token_ids.tolist()

        cls_id = self.cls_token_id
        eos_id = self.eos_token_id

        # Strip leading CLS and trailing EOS
        inner = token_ids
        if inner and inner[0] == cls_id:
            inner = inner[1:]
        if inner and inner[-1] == eos_id:
            inner = inner[:-1]

        # Find <cls><cls> separator
        sep_idx = None
        for i in range(len(inner) - 1):
            if inner[i] == cls_id and inner[i + 1] == cls_id:
                sep_idx = i
                break

        if sep_idx is None:
            raise ValueError(
                "No <cls><cls> separator found. "
                "This does not appear to be a paired sequence."
            )

        heavy_ids = inner[:sep_idx]
        light_ids = inner[sep_idx + 2:]  # skip the 2 CLS tokens

        heavy = self._tokenizer.decode(heavy_ids, skip_special_tokens=skip_special_tokens).replace(" ", "")
        light = self._tokenizer.decode(light_ids, skip_special_tokens=skip_special_tokens).replace(" ", "")

        return heavy, light

    # =========================================================================
    # tokenize (returns token strings)
    # =========================================================================

    def tokenize(self, sequence: str) -> List[str]:
        """Tokenize a sequence into token strings (no special tokens).

        Args:
            sequence: Amino acid sequence.

        Returns:
            List of token strings, one per residue.
        """
        seq = sequence.strip().upper()
        return self._tokenizer.tokenize(seq)

    # =========================================================================
    # Internal helpers
    # =========================================================================

    @staticmethod
    def _ensure_list(x: Union[str, List[str]]) -> Tuple[List[str], bool]:
        if isinstance(x, str):
            return [x], True
        return list(x), False

    def _uppercase_sequences(self, sequences: List[str]) -> List[str]:
        """Uppercase only the AA portions, preserving special tokens like <cls>."""
        cls_token = self._tokenizer.cls_token or "<cls>"
        result = []
        for s in sequences:
            if cls_token in s:
                parts = s.split(cls_token)
                result.append(cls_token.join(p.upper() for p in parts))
            else:
                result.append(s.upper())
        return result

    def __repr__(self) -> str:
        return (
            f"PrismTokenizer(model='{self._model_identifier}', "
            f"vocab_size={self.vocab_size})"
        )

    def __len__(self) -> int:
        return self.vocab_size
