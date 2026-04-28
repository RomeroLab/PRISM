#!/usr/bin/env python
"""
PLL-Guided Sequence Generation for Baseline Models (AbLang2, ESM2).

Same pipeline as pll_guided_sampler.py but for single-head MLMs:
1. One-out masking -> collect logits (L forward passes)
2. Gumbel-Top-k position selection
3. Categorical AA sampling with temperature
4. FASTA output

Usage:
    python pll_guided_sampler_baselines.py \
        --model ablang2 \
        --sequence "EVQLV..." --light_chain "DIQMT..." \
        --n_samples 100 --n_mutations 5 --seed 42 \
        --output variants.fasta

    python pll_guided_sampler_baselines.py \
        --model esm2 \
        --sequence "EVQLV..." --light_chain "DIQMT..." \
        --n_samples 100 --n_mutations 5 --seed 42 \
        --output variants.fasta
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"


# =============================================================================
# Model wrappers
# =============================================================================

class AbLang2Wrapper:
    """Wrapper for AbLang2 paired model."""

    def __init__(self, device: str = "cuda"):
        import ablang2
        self.model_obj = ablang2.pretrained(
            model_to_use="ablang2-paired", random_init=False, ncpu=1, device=device,
        )
        self.tokenizer = self.model_obj.tokenizer
        self.device = device
        self.mask_token_id = self.tokenizer.mask_token  # 23

        # Build AA index mapping: AA char -> token ID
        self.aa_to_id = {
            aa: self.tokenizer.aa_to_token[aa] for aa in AA_ORDER
            if aa in self.tokenizer.aa_to_token
        }
        self.aa_token_ids = np.array([self.aa_to_id[aa] for aa in AA_ORDER])

        # Special token IDs
        self.start_id = self.tokenizer.start_token    # 0
        self.end_id = self.tokenizer.end_token         # 22
        self.sep_id = self.tokenizer.sep_token         # 25
        self.pad_id = self.tokenizer.pad_token         # 21

    def model_name(self) -> str:
        return "ablang2"

    def tokenize(self, vh: str, vl: Optional[str]) -> Tuple[torch.Tensor, List[int], str]:
        """Tokenize and return (token_ids [1, L], maskable_positions, combined_seq).

        AbLang2 paired format: <VH>|<VL>
        All AA positions (both chains) are returned as maskable.
        """
        if vl is not None:
            tokens = self.tokenizer([(vh, vl)], pad=True)
            combined_seq = vh + vl
        else:
            # AbLang2 requires pairs; duplicate VH as placeholder
            tokens = self.tokenizer([(vh, vh)], pad=True)
            combined_seq = vh

        tokens = tokens.to(self.device)
        special = {self.start_id, self.end_id, self.sep_id, self.pad_id}

        maskable = [i for i in range(tokens.shape[1]) if tokens[0, i].item() not in special]

        return tokens, maskable, combined_seq

    @torch.no_grad()
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass -> [B, L, vocab_size] logits."""
        return self.model_obj.AbLang(token_ids)


class ESM2Wrapper:
    """Wrapper for HuggingFace ESM2-35M."""

    def __init__(self, device: str = "cuda", esm2_model: str = "facebook/esm2_t12_35M_UR50D"):
        from transformers import EsmForMaskedLM, EsmTokenizer
        self.tokenizer = EsmTokenizer.from_pretrained(esm2_model)
        self.model = EsmForMaskedLM.from_pretrained(esm2_model).to(device)
        self.model.eval()
        self.device = device
        self.mask_token_id = self.tokenizer.mask_token_id

        # Build AA index mapping
        self.aa_to_id = {aa: self.tokenizer.convert_tokens_to_ids(aa) for aa in AA_ORDER}
        self.aa_token_ids = np.array([self.aa_to_id[aa] for aa in AA_ORDER])

        # Special tokens
        self.cls_id = self.tokenizer.cls_token_id
        self.eos_id = self.tokenizer.eos_token_id
        self.pad_id = self.tokenizer.pad_token_id

    def model_name(self) -> str:
        return "esm2_35m"

    def tokenize(self, vh: str, vl: Optional[str]) -> Tuple[torch.Tensor, List[int], str]:
        """Tokenize VH (+ optional VL as simple concatenation).

        Vanilla ESM2 was never trained with <cls> separators between chains,
        so VH and VL are concatenated directly to avoid confusing the model.
        All AA positions (both chains) are returned as maskable.
        """
        if vl is not None:
            seq = vh + vl
            combined_seq = vh + vl
        else:
            seq = vh
            combined_seq = vh

        encoded = self.tokenizer(
            seq, return_tensors="pt", add_special_tokens=True,
            padding=False, truncation=True, max_length=512,
        )
        tokens = encoded["input_ids"].to(self.device)

        special = {self.cls_id, self.eos_id, self.pad_id}
        special.discard(None)

        maskable = [i for i in range(tokens.shape[1]) if tokens[0, i].item() not in special]

        return tokens, maskable, combined_seq

    @torch.no_grad()
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass -> [B, L, vocab_size] logits."""
        return self.model(input_ids=token_ids).logits


# =============================================================================
# Core pipeline
# =============================================================================

@torch.no_grad()
def collect_masked_logits(
    wrapper,
    vh: str,
    vl: Optional[str] = None,
    batch_size: int = 32,
) -> Dict[str, np.ndarray]:
    """One-out masking: mask each position (VH+VL), collect per-AA log-probs.

    Returns:
        logits_20: [L, 20] log-probs for 20 standard AAs
        wt_log_probs: [L] log P(wt_aa) at each position
        combined_seq: str — the full sequence used (VH+VL or just VH)
    """
    tokens, maskable, combined_seq = wrapper.tokenize(vh, vl)
    n_pos = len(maskable)
    aa_ids = wrapper.aa_token_ids

    logits_20_all = np.zeros((n_pos, 20), dtype=np.float32)
    wt_log_probs = np.zeros(n_pos, dtype=np.float64)

    aa_to_idx = {aa: i for i, aa in enumerate(AA_ORDER)}

    for chunk_start in tqdm(
        range(0, n_pos, batch_size),
        desc=f"Masked logits ({wrapper.model_name()})",
        total=(n_pos + batch_size - 1) // batch_size,
    ):
        chunk = maskable[chunk_start: chunk_start + batch_size]
        bs = len(chunk)

        batch_input = tokens.expand(bs, -1).clone()
        for i, pos in enumerate(chunk):
            batch_input[i, pos] = wrapper.mask_token_id

        logits = wrapper.forward(batch_input)  # [bs, L, vocab]

        for i, pos in enumerate(chunk):
            idx = chunk_start + i
            log_probs = F.log_softmax(logits[i, pos].float(), dim=-1).cpu().numpy()
            logits_20_all[idx] = log_probs[aa_ids]

            wt_aa = combined_seq[idx]
            if wt_aa in aa_to_idx:
                wt_log_probs[idx] = logits_20_all[idx, aa_to_idx[wt_aa]]

    return {"logits_20": logits_20_all, "wt_log_probs": wt_log_probs, "combined_seq": combined_seq}


def select_positions(
    wt_log_probs: np.ndarray,
    n_mutations: int,
    pool_size: int,
    position_temperature: float = 1.0,
    seed: Optional[int] = None,
    exclude_positions: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Gumbel-Top-k position selection."""
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

    return np.sort(pool[selected_in_pool])


def sample_amino_acids(
    logits_20: np.ndarray,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    seed: Optional[int] = None,
) -> List[str]:
    """Sample AAs from 20-dim logits with temperature."""
    K = logits_20.shape[0]
    rng = np.random.default_rng(seed)

    if temperature <= 0:
        temperature = 1e-10

    logits = logits_20 / temperature

    sampled = []
    for i in range(K):
        li = logits[i].copy()

        if top_k is not None and top_k < 20:
            threshold = np.partition(li, -top_k)[-top_k]
            li[li < threshold] = -np.inf

        if top_p is not None:
            sorted_idx = np.argsort(li)[::-1]
            sorted_logits = li[sorted_idx]
            probs = np.exp(sorted_logits - sorted_logits.max())
            probs /= probs.sum()
            cum = np.cumsum(probs)
            cutoff = np.searchsorted(cum, top_p) + 1
            mask = np.ones(20, dtype=bool)
            mask[sorted_idx[:cutoff]] = False
            li[mask] = -np.inf

        li = li - li.max()
        probs = np.exp(li)
        s = probs.sum()
        probs = probs / s if s > 0 else np.ones(20) / 20

        sampled.append(AA_ORDER[rng.choice(20, p=probs)])
    return sampled


def apply_mutations(wt: str, positions: np.ndarray, new_aas: List[str]) -> str:
    seq = list(wt)
    for pos, aa in zip(positions, new_aas):
        seq[pos] = aa
    return "".join(seq)


def format_mutations(wt: str, positions: np.ndarray, new_aas: List[str]) -> str:
    return ",".join(f"{wt[p]}{p+1}{aa}" for p, aa in zip(positions, new_aas))


def write_fasta(records: List[Dict], output_path: str, line_width: int = 80):
    with open(output_path, "w") as f:
        for rec in records:
            header = (
                f">{rec['id']}|model={rec['model']}|n_mut={rec['n_mut']}"
                f"|T={rec['temperature']}|mutations={rec['mutations']}"
            )
            f.write(header + "\n")
            seq = rec["sequence"]
            if line_width > 0:
                for i in range(0, len(seq), line_width):
                    f.write(seq[i: i + line_width] + "\n")
            else:
                f.write(seq + "\n")


def generate_variants(
    wrapper,
    vh: str,
    vl: Optional[str],
    n_samples: int,
    n_mutations: int,
    pool_size: int,
    temperature: float = 1.0,
    position_temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    seed: Optional[int] = None,
    exclude_positions: Optional[np.ndarray] = None,
    batch_size: int = 32,
    randomize_n_mutations: bool = False,
) -> List[Dict]:
    """Full PLL-guided generation pipeline for baseline models.

    Mutations are applied to the full combined sequence (VH+VL).
    """
    rng = np.random.default_rng(seed)

    combined_seq = vh + vl if vl else vh
    print(f"Collecting masked logits for {len(combined_seq)} positions ({wrapper.model_name()})...")
    masked_data = collect_masked_logits(wrapper, vh, vl, batch_size=batch_size)
    logits_20 = masked_data["logits_20"]
    wt_lp = masked_data["wt_log_probs"]
    combined_seq = masked_data["combined_seq"]

    records = []
    for i in tqdm(range(n_samples), desc=f"Generating variants ({wrapper.model_name()})"):
        s = rng.integers(0, 2**31)

        actual_n = n_mutations
        if randomize_n_mutations:
            actual_n = int(np.round(1 + (n_mutations - 1) * rng.beta(2, 1)))
            actual_n = min(actual_n, n_mutations)

        positions = select_positions(
            wt_lp, actual_n, pool_size,
            position_temperature=position_temperature,
            seed=s, exclude_positions=exclude_positions,
        )

        new_aas = sample_amino_acids(
            logits_20[positions],
            temperature=temperature, top_k=top_k, top_p=top_p, seed=s + 1,
        )

        variant = apply_mutations(combined_seq, positions, new_aas)
        records.append({
            "id": f"var_{i+1:04d}",
            "model": wrapper.model_name(),
            "n_mut": len(positions),
            "mutations": format_mutations(combined_seq, positions, new_aas),
            "temperature": temperature,
            "sequence": variant,
        })

    return records


# =============================================================================
# CLI
# =============================================================================

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="PLL-Guided Generation for Baseline Models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", required=True, choices=["ablang2", "esm2"],
                   help="Baseline model")
    p.add_argument("--heavy_chain", required=True, help="Heavy chain (VH) WT sequence")
    p.add_argument("--light_chain", required=True, help="Light chain (VL) WT sequence")

    p.add_argument("--n_samples", type=int, default=100)
    p.add_argument("--n_mutations", type=int, default=5)
    p.add_argument("--pool_size", type=int, default=20)

    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--position_temperature", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=None)
    p.add_argument("--top_p", type=float, default=None)

    p.add_argument("--randomize_n_mutations", action="store_true",
                   help="Randomize mutation count per variant: n_mut ~ Beta(2,1) in [1, n_mutations]")
    p.add_argument("--exclude_positions", default="1",
                   help="1-indexed positions to exclude (comma-sep). 'none' to skip.")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output", default=None, help="Output FASTA path")

    return p.parse_args(argv)


def main():
    args = parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading {args.model} on {device}...")
    if args.model == "ablang2":
        wrapper = AbLang2Wrapper(device=device)
    else:
        wrapper = ESM2Wrapper(device=device)
    print("Model loaded.")

    exclude_positions = None
    if args.exclude_positions.lower() != "none":
        exclude_positions = np.array(
            [int(x) - 1 for x in args.exclude_positions.split(",")], dtype=np.int64
        )
        print(f"Excluding positions (1-indexed): {args.exclude_positions}")

    records = generate_variants(
        wrapper=wrapper,
        vh=args.heavy_chain,
        vl=args.light_chain,
        n_samples=args.n_samples,
        n_mutations=args.n_mutations,
        pool_size=args.pool_size,
        temperature=args.temperature,
        position_temperature=args.position_temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
        exclude_positions=exclude_positions,
        batch_size=args.batch_size,
        randomize_n_mutations=args.randomize_n_mutations,
    )

    combined_seq = args.heavy_chain + args.light_chain
    wt_record = {
        "id": "WT", "model": wrapper.model_name(), "n_mut": 0,
        "mutations": "", "temperature": 0.0, "sequence": combined_seq,
    }
    records = [wt_record] + records

    # Insert VH|VL separator for FASTA output
    vh_len = len(args.heavy_chain)
    for rec in records:
        seq = rec["sequence"]
        rec["sequence"] = seq[:vh_len] + "|" + seq[vh_len:]

    if args.output:
        out_path = args.output
    else:
        out_path = f"pll_guided_{wrapper.model_name()}_n{args.n_samples}_m{args.n_mutations}.fasta"

    write_fasta(records, out_path)
    print(f"Wrote {len(records)} sequences to {out_path}")


if __name__ == "__main__":
    main()
