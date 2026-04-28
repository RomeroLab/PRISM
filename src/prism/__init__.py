"""
PRISM - Partitioning Residue Identity in Somatic Maturation

A PyTorch Lightning framework for supervised fine-tuning of ESM2 protein
language models on antibody sequences with germline/non-germline awareness.

Quick start::

    import prism

    model  = prism.pretrained("checkpoint.ckpt")

    # 4 core methods:
    result = model.forward("EVQLV...")                 # logits + embedding
    pll    = model.pseudo_log_likelihood("EVQLV...")    # PLL + perplexity (4 modes)
    score  = model.score_mutations(wt="EVQLV...", mutant="EVQLG...")  # mutation scoring
    gl     = model.predict_germline(heavy_chains="EVQLV...", light_chains="DIQMT...")  # germline prediction

All methods accept a single string or a list of strings.
"""

from .api import pretrained, PrismModel
from .model import SFT_ESM2
from .tokenizer import PrismTokenizer
from .io_utils import (
    SeqSeqDataset,
    SFTDataModule,
    LazyShardedDataModule,
    make_collate_fn_multihead,
)
from .multimodal_io import (
    GeneVocabulary,
    AntibodyDataset,
    AntibodyDataCollator,
    AntibodyDataCollatorConfig,
    AntibodyMLMCollator,
)
from .utils import get_device

__version__ = "1.1.1"

__all__ = [
    "pretrained",
    "PrismModel",
    "PrismTokenizer",
    "SFT_ESM2",
    "SeqSeqDataset",
    "SFTDataModule",
    "LazyShardedDataModule",
    "make_collate_fn_multihead",
    "GeneVocabulary",
    "AntibodyDataset",
    "AntibodyDataCollator",
    "AntibodyDataCollatorConfig",
    "AntibodyMLMCollator",
    "get_device",
]
