#!/usr/bin/env python
# coding: utf-8

"""
Multi-modal data loading utilities for Antibody Language Models.

This module provides:
- GeneVocabulary: Maps V-gene and J-gene strings to unique integer IDs
- AntibodyDataset: Dataset for loading antibody sequences with gene and region annotations
- AntibodyDataCollator: Configurable collator for multi-modal inputs (sequence, gene context, region)

Usage:
    from prism import GeneVocabulary, AntibodyDataset, AntibodyDataCollator

    # Build vocabulary from data
    gene_vocab = GeneVocabulary.from_dataframe(df, v_gene_col='v_gene', j_gene_col='j_gene')

    # Create dataset
    dataset = AntibodyDataset(file_paths=['data.pkl'], tokenizer=tokenizer)

    # Create collator with feature toggles
    collator = AntibodyDataCollator(
        tokenizer=tokenizer,
        gene_vocab=gene_vocab,
        use_germline_genes=True,
        use_region_embedding=True,
        max_length=320
    )
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Optional, Union, Any
from pathlib import Path
from dataclasses import dataclass, field


class GeneVocabulary:
    """
    A helper class to map V-gene and J-gene strings to unique integer IDs.

    Supports:
    - [UNK] for unknown genes
    - [PAD] for padding
    - Building vocabulary from dataset

    Attributes:
        gene_to_id: Dict mapping gene strings to integer IDs
        id_to_gene: Dict mapping integer IDs back to gene strings
        unk_id: ID for unknown genes
        pad_id: ID for padding
    """

    # Special tokens
    PAD_TOKEN = "[PAD]"
    UNK_TOKEN = "[UNK]"

    def __init__(self, genes: Optional[List[str]] = None):
        """
        Initialize the vocabulary.

        Args:
            genes: Optional list of gene strings to build vocabulary from.
                   If None, creates an empty vocabulary with only special tokens.
        """
        self.gene_to_id: Dict[str, int] = {}
        self.id_to_gene: Dict[int, str] = {}

        # Add special tokens first
        self._add_gene(self.PAD_TOKEN)
        self._add_gene(self.UNK_TOKEN)

        # Store special token IDs
        self.pad_id = self.gene_to_id[self.PAD_TOKEN]
        self.unk_id = self.gene_to_id[self.UNK_TOKEN]

        # Add genes if provided
        if genes is not None:
            for gene in genes:
                if gene is not None and gene not in self.gene_to_id:
                    self._add_gene(gene)

    def _add_gene(self, gene: str) -> int:
        """Add a gene to the vocabulary and return its ID."""
        if gene not in self.gene_to_id:
            gene_id = len(self.gene_to_id)
            self.gene_to_id[gene] = gene_id
            self.id_to_gene[gene_id] = gene
        return self.gene_to_id[gene]

    def encode(self, gene: Optional[str]) -> int:
        """
        Encode a gene string to its integer ID.

        Args:
            gene: Gene string or None

        Returns:
            Integer ID (UNK ID if gene is unknown or None)
        """
        if gene is None or pd.isna(gene):
            return self.unk_id
        return self.gene_to_id.get(gene, self.unk_id)

    def decode(self, gene_id: int) -> str:
        """
        Decode an integer ID back to gene string.

        Args:
            gene_id: Integer ID

        Returns:
            Gene string (UNK token if ID is unknown)
        """
        return self.id_to_gene.get(gene_id, self.UNK_TOKEN)

    def __len__(self) -> int:
        """Return the vocabulary size."""
        return len(self.gene_to_id)

    def __contains__(self, gene: str) -> bool:
        """Check if a gene is in the vocabulary."""
        return gene in self.gene_to_id

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        v_gene_col: str = 'v_gene',
        j_gene_col: str = 'j_gene'
    ) -> 'GeneVocabulary':
        """
        Build vocabulary from a DataFrame.

        Args:
            df: DataFrame containing gene columns
            v_gene_col: Name of V-gene column
            j_gene_col: Name of J-gene column

        Returns:
            GeneVocabulary instance
        """
        genes = set()

        if v_gene_col in df.columns:
            genes.update(df[v_gene_col].dropna().unique())
        if j_gene_col in df.columns:
            genes.update(df[j_gene_col].dropna().unique())

        return cls(genes=sorted(list(genes)))

    @classmethod
    def from_files(
        cls,
        file_paths: List[Union[str, Path]],
        v_gene_col: str = 'v_gene',
        j_gene_col: str = 'j_gene'
    ) -> 'GeneVocabulary':
        """
        Build vocabulary from multiple pickle files.

        Args:
            file_paths: List of paths to pickle files
            v_gene_col: Name of V-gene column
            j_gene_col: Name of J-gene column

        Returns:
            GeneVocabulary instance
        """
        genes = set()

        for path in file_paths:
            df = pd.read_pickle(path)
            if v_gene_col in df.columns:
                genes.update(df[v_gene_col].dropna().unique())
            if j_gene_col in df.columns:
                genes.update(df[j_gene_col].dropna().unique())

        return cls(genes=sorted(list(genes)))

    def save(self, path: Union[str, Path]) -> None:
        """Save vocabulary to a file."""
        import json
        with open(path, 'w') as f:
            json.dump({
                'gene_to_id': self.gene_to_id,
                'pad_id': self.pad_id,
                'unk_id': self.unk_id
            }, f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'GeneVocabulary':
        """Load vocabulary from a file."""
        import json
        with open(path, 'r') as f:
            data = json.load(f)

        vocab = cls()
        vocab.gene_to_id = data['gene_to_id']
        vocab.id_to_gene = {int(v): k for k, v in vocab.gene_to_id.items()}
        vocab.pad_id = data['pad_id']
        vocab.unk_id = data['unk_id']
        return vocab

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> 'GeneVocabulary':
        """
        Load vocabulary from a JSON file with 'genes' list format.

        This method handles the simplified JSON format used by train_esm.py:
        {
            "genes": ["IGHV1-2", "IGHV1-3", ...],
            "source": "...",
            "total_genes": 365,
            "vocab_size_with_special": 367
        }

        Args:
            path: Path to JSON file containing 'genes' list

        Returns:
            GeneVocabulary instance with genes in the same order as the JSON

        Raises:
            FileNotFoundError: If the JSON file doesn't exist
            KeyError: If 'genes' key is missing from JSON
        """
        import json
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Gene vocabulary JSON not found: {path}")

        with open(path, 'r') as f:
            data = json.load(f)

        if 'genes' not in data:
            raise KeyError(f"JSON file must contain 'genes' key: {path}")

        # Use the genes list directly (order is preserved)
        genes = data['genes']
        vocab = cls(genes=genes)

        print(f"  Loaded gene vocabulary from JSON: {len(vocab)} genes (including special tokens)")
        return vocab


class AntibodyDataset(Dataset):
    """
    PyTorch Dataset for loading antibody sequences with multi-modal annotations.

    Loads data from pickle files and returns raw dictionaries containing:
    - input_text: Amino acid sequence string
    - v_gene: V-gene label string (or None)
    - j_gene: J-gene label string (or None)
    - region_mask: String of digits representing regions (or None)

    Attributes:
        df: Combined DataFrame from all input files
        tokenizer: HuggingFace tokenizer (stored but not used for tokenization in __getitem__)
        sequence_col: Name of the sequence column
        v_gene_col: Name of the V-gene column
        j_gene_col: Name of the J-gene column
        region_mask_col: Name of the region mask column
    """

    def __init__(
        self,
        file_paths: Union[str, Path, List[Union[str, Path]]],
        tokenizer: Any,
        sequence_col: str = 'sequence_alignment_aa',
        v_gene_col: str = 'v_gene',
        j_gene_col: str = 'j_gene',
        region_mask_col: str = 'region_mask'
    ):
        """
        Initialize the dataset.

        Args:
            file_paths: Single path or list of paths to pickle files
            tokenizer: HuggingFace tokenizer instance
            sequence_col: Name of the amino acid sequence column
            v_gene_col: Name of the V-gene column
            j_gene_col: Name of the J-gene column
            region_mask_col: Name of the region mask column
        """
        self.tokenizer = tokenizer
        self.sequence_col = sequence_col
        self.v_gene_col = v_gene_col
        self.j_gene_col = j_gene_col
        self.region_mask_col = region_mask_col

        # Ensure file_paths is a list
        if isinstance(file_paths, (str, Path)):
            file_paths = [file_paths]

        # Load and concatenate all DataFrames
        dfs = []
        for path in file_paths:
            path = Path(path)
            if not path.exists():
                raise FileNotFoundError(f"Data file not found: {path}")
            dfs.append(pd.read_pickle(path))

        self.df = pd.concat(dfs, ignore_index=True)

        # Validate required column
        if self.sequence_col not in self.df.columns:
            raise ValueError(f"Required column '{self.sequence_col}' not found in DataFrame")

        # Log available columns
        available_cols = []
        if self.v_gene_col in self.df.columns:
            available_cols.append(self.v_gene_col)
        if self.j_gene_col in self.df.columns:
            available_cols.append(self.j_gene_col)
        if self.region_mask_col in self.df.columns:
            available_cols.append(self.region_mask_col)

        print(f"[AntibodyDataset] Loaded {len(self.df)} samples from {len(file_paths)} file(s)")
        print(f"[AntibodyDataset] Available annotation columns: {available_cols}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample.

        Args:
            idx: Sample index

        Returns:
            Dictionary containing:
                - input_text: str, amino acid sequence
                - v_gene: str or None, V-gene label
                - j_gene: str or None, J-gene label
                - region_mask: str or None, region mask string
        """
        row = self.df.iloc[idx]

        # Get sequence (required)
        input_text = str(row[self.sequence_col])

        # Get optional annotations
        v_gene = row.get(self.v_gene_col, None)
        if pd.isna(v_gene):
            v_gene = None

        j_gene = row.get(self.j_gene_col, None)
        if pd.isna(j_gene):
            j_gene = None

        region_mask = row.get(self.region_mask_col, None)
        if pd.isna(region_mask):
            region_mask = None
        elif region_mask is not None:
            region_mask = str(region_mask)

        return {
            'input_text': input_text,
            'v_gene': v_gene,
            'j_gene': j_gene,
            'region_mask': region_mask
        }


@dataclass
class AntibodyDataCollatorConfig:
    """Configuration for AntibodyDataCollator."""
    use_germline_genes: bool = False
    use_region_embedding: bool = False
    max_length: int = 320
    padding: str = 'longest'  # 'longest' for batch-level, 'max_length' for fixed
    truncation: bool = True
    add_special_tokens: bool = True
    # Region ID for special tokens ([CLS], [SEP], [PAD])
    special_token_region_id: int = 0
    # NGL-targeted masking: apply higher masking probability to NGL (lowercase) tokens
    # This forces the model to predict NGL mutations from context rather than copying
    ngl_targeted_masking: bool = False
    ngl_mask_prob: float = 0.8  # Masking probability for NGL tokens (default: 80%)
    mask_prob: float = 0.15  # Masking probability for GL tokens (default: 15%)


class AntibodyDataCollator:
    """
    Data collator for multi-modal antibody inputs.

    Handles:
    - Tokenization of sequences with proper padding/truncation
    - Gene ID encoding (optional, controlled by use_germline_genes)
    - Region mask processing (optional, controlled by use_region_embedding)

    The region mask is aligned exactly with input_ids, including proper handling
    of special tokens which receive a designated region ID.

    Output dictionary keys:
    - input_ids: LongTensor [B, L]
    - attention_mask: LongTensor [B, L]
    - v_gene_ids: LongTensor [B] (if use_germline_genes=True, else None)
    - j_gene_ids: LongTensor [B] (if use_germline_genes=True, else None)
    - region_ids: LongTensor [B, L] (if use_region_embedding=True, else None)
    """

    def __init__(
        self,
        tokenizer: Any,
        gene_vocab: Optional[GeneVocabulary] = None,
        use_germline_genes: bool = False,
        use_region_embedding: bool = False,
        max_length: int = 320,
        padding: str = 'longest',
        truncation: bool = True,
        add_special_tokens: bool = True,
        special_token_region_id: int = 0
    ):
        """
        Initialize the collator.

        Args:
            tokenizer: HuggingFace tokenizer
            gene_vocab: GeneVocabulary instance (required if use_germline_genes=True)
            use_germline_genes: Whether to include gene IDs in output
            use_region_embedding: Whether to include region IDs in output
            max_length: Maximum sequence length
            padding: Padding strategy ('longest' or 'max_length')
            truncation: Whether to truncate sequences
            add_special_tokens: Whether to add special tokens during tokenization
            special_token_region_id: Region ID to assign to special tokens
        """
        self.tokenizer = tokenizer
        self.gene_vocab = gene_vocab
        self.use_germline_genes = use_germline_genes
        self.use_region_embedding = use_region_embedding
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        self.add_special_tokens = add_special_tokens
        self.special_token_region_id = special_token_region_id

        # Validate configuration
        if self.use_germline_genes and self.gene_vocab is None:
            raise ValueError("gene_vocab is required when use_germline_genes=True")

        # Get special token IDs for region mask alignment
        self.cls_token_id = tokenizer.cls_token_id
        self.sep_token_id = tokenizer.sep_token_id if hasattr(tokenizer, 'sep_token_id') else None
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id if hasattr(tokenizer, 'eos_token_id') else None

        print(f"[AntibodyDataCollator] Initialized with:")
        print(f"  use_germline_genes: {use_germline_genes}")
        print(f"  use_region_embedding: {use_region_embedding}")
        print(f"  max_length: {max_length}")
        print(f"  padding: {padding}")

    def _process_region_mask(
        self,
        region_mask_str: Optional[str],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Process region mask string to match tokenized input_ids length.

        Handles:
        - Converting string digits to integers
        - Aligning with special tokens (assigning special_token_region_id)
        - Padding/truncation to match input_ids

        Args:
            region_mask_str: String of digits (e.g., "00001111222233334444")
            input_ids: Tokenized input IDs [L]
            attention_mask: Attention mask [L]

        Returns:
            LongTensor of region IDs [L]
        """
        seq_len = input_ids.shape[0]

        # Initialize with special token region ID
        region_ids = torch.full(
            (seq_len,),
            self.special_token_region_id,
            dtype=torch.long
        )

        if region_mask_str is None:
            return region_ids

        # Convert string to list of integers
        try:
            region_values = [int(c) for c in region_mask_str]
        except ValueError:
            # If conversion fails, return default
            return region_ids

        # Find positions that are NOT special tokens
        # Special tokens: CLS, SEP, PAD, EOS
        special_token_ids = {self.cls_token_id, self.pad_token_id}
        if self.sep_token_id is not None:
            special_token_ids.add(self.sep_token_id)
        if self.eos_token_id is not None:
            special_token_ids.add(self.eos_token_id)
        special_token_ids.discard(None)

        # Find non-special token positions
        non_special_positions = []
        for i, token_id in enumerate(input_ids.tolist()):
            if token_id not in special_token_ids:
                non_special_positions.append(i)

        # Assign region values to non-special positions
        n_positions = len(non_special_positions)
        n_regions = len(region_values)

        # Handle truncation: use first n_positions region values
        # Handle padding: region values beyond sequence get special_token_region_id (already set)
        for i, pos in enumerate(non_special_positions):
            if i < n_regions:
                region_ids[pos] = region_values[i]

        return region_ids

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate a batch of samples.

        Args:
            batch: List of dictionaries from AntibodyDataset

        Returns:
            Dictionary with:
                - input_ids: LongTensor [B, L]
                - attention_mask: LongTensor [B, L]
                - v_gene_ids: LongTensor [B] or None
                - j_gene_ids: LongTensor [B] or None
                - region_ids: LongTensor [B, L] or None
        """
        # Extract texts for tokenization
        texts = [sample['input_text'] for sample in batch]

        # Tokenize all sequences
        encoding = self.tokenizer(
            texts,
            padding=self.padding,
            truncation=self.truncation,
            max_length=self.max_length,
            add_special_tokens=self.add_special_tokens,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids']  # [B, L]
        attention_mask = encoding['attention_mask']  # [B, L]
        B, L = input_ids.shape

        # Process gene IDs if enabled
        v_gene_ids = None
        j_gene_ids = None
        if self.use_germline_genes:
            v_genes = [sample['v_gene'] for sample in batch]
            j_genes = [sample['j_gene'] for sample in batch]

            v_gene_ids = torch.tensor(
                [self.gene_vocab.encode(g) for g in v_genes],
                dtype=torch.long
            )
            j_gene_ids = torch.tensor(
                [self.gene_vocab.encode(g) for g in j_genes],
                dtype=torch.long
            )

        # Process region masks if enabled
        region_ids = None
        if self.use_region_embedding:
            region_ids_list = []
            for i, sample in enumerate(batch):
                region_mask = self._process_region_mask(
                    sample['region_mask'],
                    input_ids[i],
                    attention_mask[i]
                )
                region_ids_list.append(region_mask)
            region_ids = torch.stack(region_ids_list, dim=0)  # [B, L]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'v_gene_ids': v_gene_ids,
            'j_gene_ids': j_gene_ids,
            'region_ids': region_ids
        }


class AntibodyMLMCollator(AntibodyDataCollator):
    """
    Extended collator that also applies MLM (Masked Language Modeling) masking.

    Inherits all multi-modal processing from AntibodyDataCollator and adds:
    - Dynamic MLM masking with configurable probability
    - NGL-targeted masking: optionally apply higher masking probability for NGL (lowercase) tokens
    - Labels tensor for MLM loss computation

    Additional output:
    - labels: LongTensor [B, L] with -100 for non-masked positions
    """

    def __init__(
        self,
        tokenizer: Any,
        mask_prob: float = 0.15,
        gene_vocab: Optional[GeneVocabulary] = None,
        use_germline_genes: bool = False,
        use_region_embedding: bool = False,
        max_length: int = 320,
        padding: str = 'longest',
        truncation: bool = True,
        add_special_tokens: bool = True,
        special_token_region_id: int = 0,
        ngl_targeted_masking: bool = False,
        ngl_mask_prob: float = 0.8
    ):
        """
        Initialize the MLM collator.

        Args:
            tokenizer: HuggingFace tokenizer
            mask_prob: Probability of masking each GL (germline) token (default: 0.15)
            gene_vocab: GeneVocabulary instance
            use_germline_genes: Whether to include gene IDs
            use_region_embedding: Whether to include region IDs
            max_length: Maximum sequence length
            padding: Padding strategy
            truncation: Whether to truncate
            add_special_tokens: Whether to add special tokens
            special_token_region_id: Region ID for special tokens
            ngl_targeted_masking: If True, apply higher masking probability to NGL (lowercase) tokens
            ngl_mask_prob: Probability of masking NGL tokens when ngl_targeted_masking=True (default: 0.8)
        """
        super().__init__(
            tokenizer=tokenizer,
            gene_vocab=gene_vocab,
            use_germline_genes=use_germline_genes,
            use_region_embedding=use_region_embedding,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            add_special_tokens=add_special_tokens,
            special_token_region_id=special_token_region_id
        )
        self.mask_prob = mask_prob
        self.ngl_targeted_masking = ngl_targeted_masking
        self.ngl_mask_prob = ngl_mask_prob
        self.mask_token_id = tokenizer.mask_token_id
        self.vocab_size = tokenizer.vocab_size

        # Build set of NGL (lowercase) token IDs for targeted masking
        self.ngl_token_ids_set = set()
        if self.ngl_targeted_masking:
            lowercase_aa = ['a', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l',
                            'm', 'n', 'p', 'q', 'r', 's', 't', 'v', 'w', 'y']
            for aa in lowercase_aa:
                token_id = tokenizer.convert_tokens_to_ids(aa)
                # Only add if it's a valid token (not UNK)
                if token_id != tokenizer.unk_token_id:
                    self.ngl_token_ids_set.add(token_id)

        print(f"[AntibodyMLMCollator] GL mask probability: {mask_prob}")
        if self.ngl_targeted_masking:
            print(f"[AntibodyMLMCollator] NGL-targeted masking: ENABLED (prob={ngl_mask_prob})")
            print(f"[AntibodyMLMCollator] NGL token IDs: {sorted(self.ngl_token_ids_set)}")
        else:
            print(f"[AntibodyMLMCollator] NGL-targeted masking: disabled")

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate batch and apply MLM masking.

        If ngl_targeted_masking is enabled:
        - NGL (lowercase) tokens are masked with probability ngl_mask_prob (default 0.8)
        - GL (uppercase) tokens are masked with probability mask_prob (default 0.15)

        Returns parent output plus:
            - labels: LongTensor [B, L] for MLM loss
        """
        # Get base collated output
        output = super().__call__(batch)

        input_ids = output['input_ids'].clone()
        labels = output['input_ids'].clone()
        B, L = input_ids.shape

        # Don't mask special tokens
        special_token_ids = {self.cls_token_id, self.pad_token_id}
        if self.sep_token_id is not None:
            special_token_ids.add(self.sep_token_id)
        if self.eos_token_id is not None:
            special_token_ids.add(self.eos_token_id)
        special_token_ids.discard(None)

        special_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for token_id in special_token_ids:
            special_mask |= (input_ids == token_id)

        # Build probability matrix with different rates for NGL vs GL tokens
        probability_matrix = torch.full(labels.shape, self.mask_prob)

        if self.ngl_targeted_masking and len(self.ngl_token_ids_set) > 0:
            # Identify NGL positions (lowercase tokens)
            ngl_mask = torch.zeros_like(labels, dtype=torch.bool)
            for ngl_id in self.ngl_token_ids_set:
                ngl_mask |= (labels == ngl_id)

            # Set higher masking probability for NGL positions
            probability_matrix.masked_fill_(ngl_mask, self.ngl_mask_prob)

        # Don't mask special tokens
        probability_matrix.masked_fill_(special_mask, 0.0)

        # Perform Bernoulli sampling to select positions to mask
        rand = torch.rand(input_ids.shape)
        to_mask = (rand < probability_matrix)

        # Set labels for non-masked positions to -100 (ignored in loss)
        labels[~to_mask] = -100

        # Apply 80-10-10 masking strategy using fresh random values
        rand2 = torch.rand(input_ids.shape)

        # 80% of masked positions -> [MASK]
        mask_mask = to_mask & (rand2 < 0.8)
        input_ids[mask_mask] = self.mask_token_id

        # 10% of masked positions -> random token
        rand_mask = to_mask & (rand2 >= 0.8) & (rand2 < 0.9)
        if rand_mask.any():
            input_ids[rand_mask] = torch.randint(
                low=0,
                high=self.vocab_size,
                size=(rand_mask.sum(),)
            )

        # Remaining 10% keep original tokens (no action needed)

        # Update output
        output['input_ids'] = input_ids
        output['labels'] = labels

        return output
