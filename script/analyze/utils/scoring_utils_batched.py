#!/usr/bin/env python
# coding: utf-8
"""
Batched IgLM scoring utilities.

Numerically equivalent to ``utils.scoring_utils`` but exploits two redundancies
in the original sequential implementation:

1. **Within-sequence batching** — For pseudo-perplexity we mask each of the L
   positions in turn. Every masked variant has the same total token length, so
   we can stack all L variants into a single ``[L, T]`` batch and run one
   forward pass instead of L sequential ones.

2. **WT-once + logit lookup (mutation LLR)** — IgLM is autoregressive, so the
   logits at the ``[SEP]`` position depend ONLY on tokens up to and including
   ``[SEP]`` (causal attention). The post-``[SEP]`` "target span" — which is
   the only thing that differs between the WT-LL and mutant-LL forward passes
   in ``evaluate_iglm_model`` — does not affect the prediction. Therefore we
   can run **one** forward pass per masked WT position and read off both
   ``log P(wt_aa)`` and ``log P(mut_aa)`` from the same logit distribution.
   Per source-file the WT log-probs are cached and reused across all variants.

Numerical equivalence verified by ``smoke_test_batched_equivalence`` below.
"""

from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm


__all__ = [
    "compute_per_position_logprobs",
    "compute_iglm_pseudo_perplexity_batched",
    "add_iglm_scores_to_dataframe_batched",
    "evaluate_iglm_model_batched",
    "find_mutation_indices",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def find_mutation_indices(wt_seq: str, mut_seq: str) -> List[Tuple[int, str, str]]:
    """Same definition as ``utils.scoring_utils.find_mutation_indices``."""
    if len(wt_seq) != len(mut_seq):
        return []
    return [(i, w, m) for i, (w, m) in enumerate(zip(wt_seq, mut_seq)) if w != m]


# ---------------------------------------------------------------------------
# Core batched primitive
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_per_position_logprobs(
    model,
    sequence: str,
    chain_token: str,
    species_token: str,
    batch_size: int = 128,
) -> torch.Tensor:
    """
    For each position ``i`` in ``sequence``, compute the log-probability
    distribution over the vocabulary at the masked-position prediction site.

    Constructs all L masked variants of the form
    ``[CHAIN][SPECIES] AA1 ... [MASK]_i ... AAN [SEP]``
    and runs them in batches. Reads logits at the ``[SEP]`` position (which is
    where IgLM's autoregressive head predicts the masked AA).

    Returns
    -------
    log_probs : torch.Tensor, shape ``[L, vocab_size]``, dtype float32, on CPU
    """
    tokenizer = model.tokenizer
    device = model.device

    seq_chars = list(sequence)
    L = len(seq_chars)

    chain_id = tokenizer.convert_tokens_to_ids(chain_token)
    species_id = tokenizer.convert_tokens_to_ids(species_token)
    sep_id = tokenizer.sep_token_id
    mask_id = tokenizer.convert_tokens_to_ids("[MASK]")
    aa_ids = tokenizer.convert_tokens_to_ids(seq_chars)

    base_ids = [chain_id, species_id] + aa_ids + [sep_id]
    base = torch.tensor(base_ids, dtype=torch.long, device=device)
    T = base.shape[0]
    sep_pos = T - 1
    aa_offset = 2  # [CHAIN][SPECIES] -> AAs start at index 2

    vocab_size = model.model.config.vocab_size
    out = torch.empty(L, vocab_size, dtype=torch.float32)

    for start in range(0, L, batch_size):
        end = min(start + batch_size, L)
        bs = end - start

        batch = base.unsqueeze(0).expand(bs, -1).clone()
        for j, i in enumerate(range(start, end)):
            batch[j, aa_offset + i] = mask_id

        logits = model.model(batch).logits  # [bs, T, vocab]
        sep_logits = logits[:, sep_pos, :]  # [bs, vocab]
        out[start:end] = F.log_softmax(sep_logits, dim=-1).float().cpu()

    return out


# ---------------------------------------------------------------------------
# Pseudo-perplexity (developability scripts)
# ---------------------------------------------------------------------------


def _gather_target_logprobs(log_probs: torch.Tensor, sequence: str, tokenizer) -> torch.Tensor:
    """log_probs[i, original_aa_token_id_at_i] for each position i."""
    target_ids = torch.tensor(
        tokenizer.convert_tokens_to_ids(list(sequence)), dtype=torch.long
    )
    return log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1)  # [L]


def compute_iglm_pseudo_perplexity_batched(
    model,
    heavy: str,
    light: str,
    heavy_chain_token: str = "[HEAVY]",
    light_chain_token: str = "[LIGHT]",
    species_token: str = "[HUMAN]",
    batch_size: int = 128,
) -> float:
    """Batched drop-in replacement for ``compute_iglm_pseudo_perplexity``."""
    tokenizer = model.tokenizer

    log_probs_h = compute_per_position_logprobs(
        model, heavy, heavy_chain_token, species_token, batch_size
    )
    log_probs_l = compute_per_position_logprobs(
        model, light, light_chain_token, species_token, batch_size
    )

    sum_h = _gather_target_logprobs(log_probs_h, heavy, tokenizer).sum().item()
    sum_l = _gather_target_logprobs(log_probs_l, light, tokenizer).sum().item()

    total_len = len(heavy) + len(light)
    return float(np.exp(-(sum_h + sum_l) / total_len))


# Forward perplexity (single forward per chain) is already a single forward
# call in IgLM's ``log_likelihood``, so no batching needed beyond chain-level.
def compute_iglm_perplexity(
    model,
    heavy: str,
    light: str,
    heavy_chain_token: str = "[HEAVY]",
    light_chain_token: str = "[LIGHT]",
    species_token: str = "[HUMAN]",
) -> float:
    """Same as ``utils.scoring_utils.compute_iglm_perplexity``; reproduced here
    so the batched module is self-contained."""
    heavy_ll = model.log_likelihood(heavy, heavy_chain_token, species_token)
    light_ll = model.log_likelihood(light, light_chain_token, species_token)
    total_tokens = (len(heavy) + 1) + (len(light) + 1)
    total_ll = heavy_ll * (len(heavy) + 1) + light_ll * (len(light) + 1)
    total_ll /= total_tokens
    return float(np.exp(-total_ll))


def add_iglm_scores_to_dataframe_batched(
    df: pd.DataFrame,
    model,
    heavy_col: str,
    light_col: str,
    perplexity_col: str = "IgLM_Perplexity",
    pseudo_perplexity_col: str = "IgLM_PseudoPerplexity",
    compute_perplexity: bool = True,
    compute_pseudo_perplexity: bool = True,
    batch_size: int = 128,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Batched drop-in replacement for ``add_iglm_scores_to_dataframe``."""
    iterator = df.iterrows()
    if show_progress:
        iterator = tqdm(iterator, total=len(df), desc="IgLM scoring (batched)")

    for i, row in iterator:
        heavy = row[heavy_col]
        light = row[light_col]
        if not isinstance(heavy, str) or not isinstance(light, str):
            raise ValueError(
                f"Row {i} has non-string sequence(s): "
                f"{heavy_col}={type(heavy)}, {light_col}={type(light)}"
            )

        if compute_perplexity:
            df.at[i, perplexity_col] = compute_iglm_perplexity(model, heavy, light)
        if compute_pseudo_perplexity:
            df.at[i, pseudo_perplexity_col] = compute_iglm_pseudo_perplexity_batched(
                model, heavy, light, batch_size=batch_size,
            )

    return df


# ---------------------------------------------------------------------------
# Mutation LLR (binding script)
# ---------------------------------------------------------------------------


def evaluate_iglm_model_batched(
    df: pd.DataFrame,
    wt_heavy: str,
    wt_light: str,
    device: str,
    model=None,
    heavy_col: str = "fv_heavy",
    light_col: str = "fv_light",
    mutation_col: str = "Mutations",
    heavy_chain_token: str = "[HEAVY]",
    light_chain_token: str = "[LIGHT]",
    species_token: str = "[HUMAN]",
    batch_size: int = 128,
) -> List[float]:
    """Batched drop-in replacement for ``evaluate_iglm_model``.

    Strategy: compute WT masked log-probs at every position once (heavy + light
    chain), then for each variant convert mutations to a pure tensor lookup.
    Numerically identical to the sequential version.
    """
    from iglm import IgLM

    own_model = model is None
    if own_model:
        print(f"\n{'='*60}\nLoading IgLM...\n{'='*60}")
        model = IgLM()
        print(f"  Model loaded on {model.device}")

    tokenizer = model.tokenizer

    # WT log-probs cached once (this is the entire compute budget for the
    # source_file in practice; per-variant work is just lookups).
    print(f"  Caching WT logprobs (heavy len={len(wt_heavy)}, light len={len(wt_light)})...")
    log_probs_heavy = compute_per_position_logprobs(
        model, wt_heavy, heavy_chain_token, species_token, batch_size
    )  # [L_h, vocab]
    log_probs_light = compute_per_position_logprobs(
        model, wt_light, light_chain_token, species_token, batch_size
    )  # [L_l, vocab]

    aa_to_id = {aa: tokenizer.convert_tokens_to_ids(aa) for aa in "ACDEFGHIKLMNPQRSTVWY"}

    scores: List[float] = []
    non_zero_count = 0

    for idx in tqdm(range(len(df)), desc="Scoring with IgLM (batched)"):
        row = df.iloc[idx]
        mutation_str = row.get(mutation_col, None)
        mut_heavy = row.get(heavy_col, None)
        mut_light = row.get(light_col, None)

        if not isinstance(mut_heavy, str) or not isinstance(mut_light, str):
            scores.append(0.0)
            continue
        mut_heavy = mut_heavy.strip()
        mut_light = mut_light.strip()

        if mutation_str == "WT":
            scores.append(0.0)
            continue
        if len(mut_heavy) != len(wt_heavy) or len(mut_light) != len(wt_light):
            scores.append(0.0)
            continue

        h_muts = find_mutation_indices(wt_heavy, mut_heavy)
        l_muts = find_mutation_indices(wt_light, mut_light)
        if not h_muts and not l_muts:
            scores.append(0.0)
            continue

        score = 0.0
        for pos, wt_aa, mut_aa in h_muts:
            wt_id = aa_to_id.get(wt_aa)
            mut_id = aa_to_id.get(mut_aa)
            if wt_id is None or mut_id is None:
                continue
            score += (
                log_probs_heavy[pos, mut_id].item()
                - log_probs_heavy[pos, wt_id].item()
            )
        for pos, wt_aa, mut_aa in l_muts:
            wt_id = aa_to_id.get(wt_aa)
            mut_id = aa_to_id.get(mut_aa)
            if wt_id is None or mut_id is None:
                continue
            score += (
                log_probs_light[pos, mut_id].item()
                - log_probs_light[pos, wt_id].item()
            )

        scores.append(float(score))
        if score != 0.0:
            non_zero_count += 1

    print(f"\n  SANITY CHECKS for IgLM (batched):")
    print(f"    Total rows: {len(scores)}")
    print(f"    Non-zero scores: {non_zero_count}")
    print(f"    First 5 scores: {scores[:5]}")

    if own_model:
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    return scores


# ---------------------------------------------------------------------------
# Smoke test (numerical equivalence vs sequential)
# ---------------------------------------------------------------------------


def smoke_test_batched_equivalence(model=None, atol: float = 1e-4) -> None:
    """Compare batched vs sequential outputs on a small Fv. Asserts equivalence."""
    from iglm import IgLM

    from utils.scoring_utils import (
        compute_iglm_pseudo_perplexity as ref_pspp,
        evaluate_iglm_model as ref_evaluate,
    )

    if model is None:
        model = IgLM()

    heavy = "QVQLVQSGAEVKKPGSSVMVSCQASGGPLRNYVINWVRQAPGQGPEWMGGIIPVLGTVHYAPKFQGRVTITADESTNTAYMELSSLRSEDTAMYYCATEAGYGNYGAFDIWGQGTMVTVSS"
    light = "EIVLTQSPGTLSLSPGERATLSCRASQSVSSSYLAWYQQKPGQAPRLLIYGASSRATGIPDRFSGSGSGTDFTLTISRLEPEDFAVYYCQQYGSSPLTFGQGTKVEIK"

    # 1) PsPPL equivalence
    pspp_ref = ref_pspp(model, heavy, light)
    pspp_new = compute_iglm_pseudo_perplexity_batched(model, heavy, light, batch_size=64)
    print(f"PsPPL  ref={pspp_ref:.6f}  batched={pspp_new:.6f}  diff={abs(pspp_ref - pspp_new):.2e}")
    assert abs(pspp_ref - pspp_new) < atol * max(1.0, abs(pspp_ref)), (
        f"PsPPL mismatch: ref={pspp_ref}, batched={pspp_new}"
    )

    # 2) Mutation-LLR equivalence on a tiny variant set
    variants = pd.DataFrame([
        {"fv_heavy": heavy[:5] + "A" + heavy[6:], "fv_light": light, "Mutations": "H6A"},
        {"fv_heavy": heavy, "fv_light": light[:10] + "G" + light[11:], "Mutations": "L11G"},
        {"fv_heavy": heavy[:50] + "K" + heavy[51:],
         "fv_light": light[:30] + "T" + light[31:], "Mutations": "H51K_L31T"},
        {"fv_heavy": heavy, "fv_light": light, "Mutations": "WT"},
    ])

    ref_scores = ref_evaluate(variants, heavy, light, device=str(model.device))
    new_scores = evaluate_iglm_model_batched(
        variants, heavy, light, device=str(model.device), model=model, batch_size=64,
    )
    for r, n, m in zip(ref_scores, new_scores, variants["Mutations"]):
        print(f"  {m:<10} ref={r:.6f}  batched={n:.6f}  diff={abs(r-n):.2e}")
        assert abs(r - n) < atol * max(1.0, abs(r)), f"LLR mismatch on {m}: {r} vs {n}"

    print("\nSMOKE TEST PASSED — batched is numerically equivalent to sequential.")


if __name__ == "__main__":
    smoke_test_batched_equivalence()
