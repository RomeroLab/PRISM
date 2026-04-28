#!/usr/bin/env python
"""
PLL-Guided Antibody Sequence Generation

Generates antibody variants by:
1. One-out masking: mask each position, collect logits (L forward passes)
2. Position selection: rank by WT probability, sample via Gumbel-Top-k
3. AA sampling: sample from GL/NGL/Full/Region-specific logits with temperature

References:
- Salazar et al., ACL 2020 (Masked LM scoring / PLL)
- Hie et al., Nature Biotech 2023 (Efficient evolution of antibodies)
- Kool et al., ICML 2019 (Gumbel-Top-k trick)
- Tagliabue et al., 2025 (Stochastic beam search for protein engineering)
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import logsumexp as scipy_logsumexp
from tqdm.auto import tqdm

AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_ORDER)}


def compute_wt_log_probs(
    logits_53: np.ndarray,
    sequence: str,
    gl_indices: np.ndarray,
    ngl_indices: np.ndarray,
    aa_order: str = AA_ORDER,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute WT log-probabilities from 53-vocab logits for each position.

    Args:
        logits_53: [L, 53] raw logits at each masked position.
        sequence: WT amino acid sequence (length L).
        gl_indices: [20] indices of GL (uppercase) tokens in 53-vocab.
        ngl_indices: [20] indices of NGL (lowercase) tokens in 53-vocab.
        aa_order: 20-character string mapping position i to amino acid.

    Returns:
        wt_gl:   [L] log P_GL(wt_aa) at each position
        wt_ngl:  [L] log P_NGL(wt_aa) at each position
        wt_marg: [L] logsumexp(GL, NGL) at each position
    """
    L = len(sequence)
    assert logits_53.shape[0] == L

    logits_t = torch.from_numpy(logits_53).float()
    log_probs = F.log_softmax(logits_t, dim=-1).numpy()

    aa_to_idx = {aa: i for i, aa in enumerate(aa_order)}

    wt_gl = np.zeros(L, dtype=np.float64)
    wt_ngl = np.zeros(L, dtype=np.float64)
    wt_marg = np.zeros(L, dtype=np.float64)

    for pos in range(L):
        aa = sequence[pos].upper()
        if aa not in aa_to_idx:
            continue
        aa_i = aa_to_idx[aa]
        gl_lp = log_probs[pos, gl_indices[aa_i]]
        ngl_lp = log_probs[pos, ngl_indices[aa_i]]
        wt_gl[pos] = gl_lp
        wt_ngl[pos] = ngl_lp
        wt_marg[pos] = scipy_logsumexp([gl_lp, ngl_lp])

    return wt_gl, wt_ngl, wt_marg


@torch.no_grad()
def collect_masked_logits(
    model,
    sequence: str,
    batch_size: int = 32,
    light_chain: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """One-out masking: mask each AA position, collect 53-vocab logits.

    When light_chain is provided, the combined sequence (VH+VL) is used
    and ALL positions (both chains) are masked and scored.

    Args:
        model: PrismModel instance (from prism.pretrained()).
        sequence: Combined AA sequence to mutate. If light_chain is given,
            this is the heavy chain and the combined VH+VL is built internally.
        batch_size: Number of masked positions per forward pass.
        light_chain: Optional light chain for paired input.

    Returns:
        {
            "logits_53":      [L, 53] full logits at each masked position,
            "wt_gl":          [L] GL log P(wt),
            "wt_ngl":         [L] NGL log P(wt),
            "wt_marg":        [L] marginalized log P(wt),
            "alpha":          [L] alpha gating values (unmasked forward),
            "combined_seq":   str — the full sequence used (VH+VL or just VH),
        }
    """
    if light_chain is not None:
        combined_seq = sequence + light_chain
        formatted = [model._format_sequences([sequence], [light_chain])[0]]
    else:
        combined_seq = sequence
        formatted = [sequence]

    input_ids, attn_mask = model._tokenize(formatted)
    seq_len = attn_mask[0].sum().item()
    mask_token_id = model.tokenizer.mask_token_id

    special_ids = {
        model.tokenizer.cls_token_id,
        model.tokenizer.eos_token_id,
        model.tokenizer.pad_token_id,
    }
    special_ids.discard(None)

    maskable = []
    for pos in range(seq_len):
        if input_ids[0, pos].item() not in special_ids:
            maskable.append(pos)

    n_positions = len(maskable)
    logits_53_all = np.zeros((n_positions, 53), dtype=np.float32)

    for chunk_start in tqdm(
        range(0, n_positions, batch_size),
        desc="Collecting masked logits",
        total=(n_positions + batch_size - 1) // batch_size,
    ):
        chunk_positions = maskable[chunk_start : chunk_start + batch_size]
        bs = len(chunk_positions)

        batch_input = input_ids.expand(bs, -1).clone()
        batch_attn = attn_mask.expand(bs, -1)
        for i, pos in enumerate(chunk_positions):
            batch_input[i, pos] = mask_token_id

        logits_aa, _, _, _, logits_final, _ = model.model._forward_multihead(
            input_ids=batch_input,
            attention_mask=batch_attn,
        )

        # Use logits_aa (pre-gating) NOT logits_final (post-gating).
        # In single-AA-head models (v34), alpha gating adds a position-wise
        # constant offset to GL vs NGL tokens, making their distributions
        # identical after softmax. logits_aa preserves genuine per-AA
        # differences between GL and NGL channels.
        source_logits = logits_aa
        for i, pos in enumerate(chunk_positions):
            idx = chunk_start + i
            vocab_size = min(source_logits.shape[-1], 53)
            logits_53_all[idx, :vocab_size] = source_logits[i, pos].cpu().numpy()[:vocab_size]

    # Get alpha from unmasked forward pass
    _, _, _, alpha_raw, _, _ = model.model._forward_multihead(
        input_ids=input_ids,
        attention_mask=attn_mask,
    )
    total_len = len(combined_seq)
    alpha = np.zeros(total_len, dtype=np.float32)
    if alpha_raw is not None:
        for i in range(min(total_len, len(maskable))):
            tok_pos = maskable[i]
            alpha[i] = alpha_raw[0, tok_pos].cpu().item()

    gl_indices = model.GL_INDICES
    ngl_indices = model.NGL_INDICES
    wt_gl, wt_ngl, wt_marg = compute_wt_log_probs(
        logits_53_all, combined_seq, gl_indices, ngl_indices
    )

    return {
        "logits_53": logits_53_all,
        "wt_gl": wt_gl,
        "wt_ngl": wt_ngl,
        "wt_marg": wt_marg,
        "alpha": alpha,
        "combined_seq": combined_seq,
    }


def select_positions(
    wt_log_probs: np.ndarray,
    n_mutations: int,
    pool_size: int,
    position_temperature: float = 1.0,
    seed: Optional[int] = None,
    exclude_positions: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Select mutation positions using Gumbel-Top-k trick.

    1. Rank all positions by WT log-probability (ascending = worst first)
    2. Take the bottom pool_size positions as candidates
    3. Within pool, add Gumbel noise scaled by position_temperature
       and select the top-k (lowest perturbed score)

    Args:
        wt_log_probs: [L] log P(wt_aa) at each position.
        n_mutations: Number of positions to select (K).
        pool_size: Candidate pool size (M).
        position_temperature: Controls randomness in selection.
        seed: Random seed.
        exclude_positions: 0-indexed positions to exclude from selection
            (e.g., N/C-terminals). These are set to +inf before ranking.

    Returns:
        [K] array of 0-indexed selected positions (sorted ascending).
    """
    wt_log_probs = wt_log_probs.copy()
    if exclude_positions is not None and len(exclude_positions) > 0:
        wt_log_probs[exclude_positions] = np.inf  # Never selected as "worst"

    L = int((wt_log_probs < np.inf).sum())  # Eligible positions
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


def sample_amino_acids(
    logits_53: np.ndarray,
    mode: str,
    gl_indices: np.ndarray,
    ngl_indices: np.ndarray,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    seed: Optional[int] = None,
    region_labels: Optional[np.ndarray] = None,
    aa_order: str = AA_ORDER,
) -> List[str]:
    """Sample amino acids from head-specific logits.

    Modes:
        "gl":              GL (uppercase) token logits only.
        "ngl":             NGL (lowercase) token logits only.
        "full":            Marginalized logits: logsumexp(GL, NGL).
        "region_specific": CDR->NGL, FR->GL. Requires region_labels.

    Args:
        logits_53: [K, 53] raw logits at K selected positions.
        mode: Sampling mode.
        gl_indices: [20] GL token indices in 53-vocab.
        ngl_indices: [20] NGL token indices in 53-vocab.
        temperature: Sampling temperature.
        top_k: Top-k filtering.
        top_p: Nucleus sampling threshold.
        seed: Random seed.
        region_labels: [K] 0=FR, 1=CDR. Required for "region_specific".
        aa_order: 20 amino acids in order.

    Returns:
        List of K sampled amino acid characters.
    """
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

    # Guard against temperature=0 (use argmax)
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


def apply_mutations(
    wt_sequence: str,
    positions: np.ndarray,
    new_aas: List[str],
) -> str:
    """Apply mutations at specified positions.

    Args:
        wt_sequence: Wild-type amino acid sequence.
        positions: [K] array of 0-indexed positions to mutate.
        new_aas: [K] list of replacement amino acids.

    Returns:
        Mutated sequence string.
    """
    seq_list = list(wt_sequence)
    for pos, aa in zip(positions, new_aas):
        seq_list[pos] = aa
    return "".join(seq_list)


def format_mutations(
    wt_sequence: str,
    positions: np.ndarray,
    new_aas: List[str],
) -> str:
    """Format mutations as 'A1W,D4Y,I8V' (1-indexed).

    Args:
        wt_sequence: Wild-type amino acid sequence.
        positions: [K] array of 0-indexed positions.
        new_aas: [K] list of replacement amino acids.

    Returns:
        Comma-separated mutation string.
    """
    parts = []
    for pos, new_aa in zip(positions, new_aas):
        wt_aa = wt_sequence[pos]
        parts.append(f"{wt_aa}{pos + 1}{new_aa}")
    return ",".join(parts)


def write_fasta(
    records: List[Dict],
    output_path: str,
    line_width: int = 80,
) -> None:
    """Write variant records to FASTA file.

    Args:
        records: List of dicts with keys: id, mode, n_mut, mutations,
                 temperature, sequence.
        output_path: Path to output FASTA file.
        line_width: Characters per line for sequence wrapping.
    """
    with open(output_path, "w") as f:
        for rec in records:
            header = (
                f">{rec['id']}|mode={rec['mode']}|n_mut={rec['n_mut']}"
                f"|T={rec['temperature']}|mutations={rec['mutations']}"
            )
            f.write(header + "\n")
            seq = rec["sequence"]
            if line_width > 0:
                for i in range(0, len(seq), line_width):
                    f.write(seq[i : i + line_width] + "\n")
            else:
                f.write(seq + "\n")


def generate_variants(
    model,
    sequence: str,
    n_samples: int,
    n_mutations: int,
    mode: str,
    pool_size: int,
    temperature: float = 1.0,
    position_temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    light_chain: Optional[str] = None,
    region_labels: Optional[np.ndarray] = None,
    batch_size: int = 32,
    seed: Optional[int] = None,
    masked_data: Optional[Dict[str, np.ndarray]] = None,
    exclude_positions: Optional[np.ndarray] = None,
    randomize_n_mutations: bool = False,
) -> List[Dict]:
    """Full PLL-guided generation pipeline.

    Args:
        model: PrismModel instance (from prism.pretrained()).
        sequence: Heavy chain WT sequence (or full sequence if unpaired).
        n_samples: Number of variants to generate.
        n_mutations: Number of mutations per variant (or max if randomized).
        mode: Sampling mode (gl, ngl, full, region_specific).
        pool_size: Candidate pool size for position selection.
        temperature: AA sampling temperature.
        position_temperature: Position selection temperature.
        top_k: Top-k AA sampling filter.
        top_p: Nucleus sampling threshold.
        light_chain: Optional light chain for paired input. When provided,
            mutations are applied to both VH and VL.
        region_labels: [L] 0=FR, 1=CDR labels for region_specific mode.
            Must cover the full combined sequence (VH+VL) if light_chain is given.
        batch_size: Forward pass batch size.
        seed: Random seed for reproducibility.
        masked_data: Pre-computed masked logits (from collect_masked_logits).
            If None, logits are collected automatically. Pass this to avoid
            redundant forward passes when running multiple modes.
        exclude_positions: 0-indexed positions to never mutate (relative to
            the combined VH+VL sequence).
        randomize_n_mutations: If True, each sample draws n_mut ~ Beta(2,1) in [1, n_mutations].

    Returns:
        List of variant record dicts.
    """
    rng = np.random.default_rng(seed)

    if masked_data is None:
        combined_len = len(sequence) + (len(light_chain) if light_chain else 0)
        print(f"Step 1: Collecting masked logits for {combined_len} positions...")
        masked_data = collect_masked_logits(
            model, sequence, batch_size=batch_size, light_chain=light_chain,
        )
    logits_53 = masked_data["logits_53"]
    wt_gl = masked_data["wt_gl"]
    wt_ngl = masked_data["wt_ngl"]
    wt_marg = masked_data["wt_marg"]

    # Use the combined sequence (VH+VL) for mutation
    combined_seq = masked_data.get("combined_seq", sequence)

    gl_indices = model.GL_INDICES
    ngl_indices = model.NGL_INDICES

    # Select position-scoring array based on mode:
    #   GL:              GL log-probs → targets positions where germline model disagrees
    #   NGL:             NGL log-probs → targets positions where mutation model disagrees
    #   Full:            marginalized log-probs → combined view
    #   Region-specific: FR uses GL, CDR uses NGL
    if mode == "gl":
        wt_scores = wt_gl
    elif mode == "ngl":
        wt_scores = wt_ngl
    elif mode == "region_specific" and region_labels is not None:
        wt_scores = np.where(region_labels == 0, wt_gl, wt_ngl)
    else:  # "full" or fallback
        wt_scores = wt_marg

    records = []
    for i in tqdm(range(n_samples), desc=f"Generating variants ({mode})"):
        sample_seed = rng.integers(0, 2**31)

        actual_n = n_mutations
        if randomize_n_mutations:
            actual_n = int(np.round(1 + (n_mutations - 1) * rng.beta(2, 1)))
            actual_n = min(actual_n, n_mutations)

        positions = select_positions(
            wt_scores, actual_n, pool_size,
            position_temperature=position_temperature,
            seed=sample_seed,
            exclude_positions=exclude_positions,
        )

        selected_logits = logits_53[positions]

        selected_regions = None
        if region_labels is not None and mode == "region_specific":
            selected_regions = region_labels[positions]

        new_aas = sample_amino_acids(
            selected_logits, mode=mode,
            gl_indices=gl_indices, ngl_indices=ngl_indices,
            temperature=temperature,
            top_k=top_k, top_p=top_p,
            seed=sample_seed + 1,
            region_labels=selected_regions,
        )

        variant_seq = apply_mutations(combined_seq, positions, new_aas)
        mutation_str = format_mutations(combined_seq, positions, new_aas)

        records.append({
            "id": f"var_{i + 1:04d}",
            "mode": mode,
            "n_mut": len(positions),
            "mutations": mutation_str,
            "temperature": temperature,
            "sequence": variant_seq,
        })

    return records


def parse_region_string(region_str: str) -> np.ndarray:
    """Parse region string to FR(0)/CDR(1) labels.

    Region IDs: 1=FR1, 2=CDR1, 3=FR2, 4=CDR2, 5=FR3, 6=CDR3, 7=FR4
    Even IDs (2,4,6) are CDR=1, odd IDs (1,3,5,7) are FR=0.

    Args:
        region_str: String of digits 1-7 (one per residue).

    Returns:
        [L] array of 0 (FR) or 1 (CDR) labels.
    """
    labels = np.zeros(len(region_str), dtype=np.int32)
    for i, ch in enumerate(region_str):
        region_id = int(ch)
        if region_id in (2, 4, 6):
            labels[i] = 1
    return labels


def load_config(config_path: str, preset: Optional[str] = None) -> Dict:
    """Load YAML config and optionally apply a preset.

    Priority: preset values override base config values.

    Args:
        config_path: Path to YAML config file.
        preset: Name of a preset defined in the config's 'presets' section.

    Returns:
        Flat dict of config values (presets section removed).
    """
    import yaml

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    presets = cfg.pop("presets", {})

    if preset is not None:
        if preset not in presets:
            available = ", ".join(presets.keys()) if presets else "(none)"
            raise ValueError(f"Unknown preset '{preset}'. Available: {available}")
        for k, v in presets[preset].items():
            cfg[k] = v

    return cfg


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments with optional YAML config + preset.

    Resolution order: defaults < config file < preset < CLI args.

    Args:
        argv: Argument list (defaults to sys.argv if None).

    Returns:
        Parsed argument namespace.
    """
    # First pass: grab --config and --preset before full parse
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    pre.add_argument("--preset", default=None)
    pre_args, _ = pre.parse_known_args(argv)

    # Load config defaults if provided
    config_defaults = {}
    if pre_args.config is not None:
        config_defaults = load_config(pre_args.config, pre_args.preset)

    p = argparse.ArgumentParser(
        description="PLL-Guided Antibody Sequence Generation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--config", default=None, help="Path to YAML config file")
    p.add_argument("--preset", default=None, help="Preset name from config (e.g., conservative, diverse)")

    p.add_argument("--checkpoint", default=None, help="Path to PRISM checkpoint or HF Hub ID")
    p.add_argument("--heavy_chain", default=None, help="Heavy chain (VH) WT sequence")
    p.add_argument("--light_chain", default=None, help="Light chain (VL) WT sequence")
    p.add_argument("--region_string", default=None,
                   help="Region annotation (digits 1-7) covering VH+VL combined")

    p.add_argument("--mode", default=None, choices=["gl", "ngl", "full", "region_specific"],
                   help="Sampling mode")
    p.add_argument("--n_samples", type=int, default=None, help="Number of variants")
    p.add_argument("--n_mutations", type=int, default=None, help="Mutations per variant (max if randomized)")
    p.add_argument("--pool_size", type=int, default=None, help="Candidate pool size")

    p.add_argument("--temperature", type=float, default=None, help="AA sampling temperature")
    p.add_argument("--position_temperature", type=float, default=None,
                   help="Position selection temperature")
    p.add_argument("--top_k", type=int, default=None, help="Top-k AA sampling")
    p.add_argument("--top_p", type=float, default=None, help="Nucleus sampling threshold")

    p.add_argument("--exclude_positions", default=None, help=(
        "Comma-separated 1-indexed positions to exclude from mutation "
        "(e.g., '1' to skip N-terminal, '1,120,121' for terminals). "
        "Use 'none' to exclude nothing."))
    p.add_argument("--randomize_n_mutations", action="store_true", default=None,
                   help="Randomize mutation count per variant: n_mut ~ Beta(2,1) in [1, n_mutations]")
    p.add_argument("--batch_size", type=int, default=None, help="Forward pass batch size")
    p.add_argument("--device", default=None, help="Device: auto, cuda, cpu")
    p.add_argument("--seed", type=int, default=None, help="Random seed")

    p.add_argument("--output", default=None, help="Output FASTA path")
    p.add_argument("--run_all_modes", action="store_true", default=None, help="Run all 4 modes")

    args = p.parse_args(argv)

    # Merge: config_defaults < CLI args (CLI wins when not None)
    # Hardcoded fallbacks for when neither config nor CLI provides a value
    FALLBACKS = {
        "checkpoint": None, "heavy_chain": None, "light_chain": None,
        "region_string": None, "mode": "full", "n_samples": 100,
        "n_mutations": 5, "pool_size": 20, "temperature": 1.0,
        "position_temperature": 1.0, "top_k": None, "top_p": None,
        "exclude_positions": "1", "randomize_n_mutations": False,
        "batch_size": 32, "device": "auto", "seed": None,
        "output": None, "run_all_modes": False,
    }

    for key, fallback in FALLBACKS.items():
        cli_val = getattr(args, key, None)
        if cli_val is not None:
            continue  # CLI takes priority
        if key in config_defaults and config_defaults[key] is not None:
            setattr(args, key, config_defaults[key])
        else:
            setattr(args, key, fallback)

    # Validate required fields
    if args.checkpoint is None:
        p.error("--checkpoint is required (or set 'checkpoint' in config)")
    if args.heavy_chain is None:
        p.error("--heavy_chain is required (or set 'heavy_chain' in config)")
    if args.light_chain is None:
        p.error("--light_chain is required (or set 'light_chain' in config)")

    return args


def main():
    """Main entry point for PLL-guided generation."""
    args = parse_args()

    import prism
    print(f"Loading model from {args.checkpoint}...")
    model = prism.pretrained(args.checkpoint, device=args.device)
    print(f"Model loaded on {model.device}")

    sequence = args.heavy_chain
    light_chain = args.light_chain
    combined_seq = sequence + light_chain

    region_labels = None
    if args.region_string:
        region_labels = parse_region_string(args.region_string)
        assert len(region_labels) == len(combined_seq), (
            f"Region string length ({len(region_labels)}) != "
            f"combined sequence length ({len(combined_seq)})"
        )

    # Parse exclude_positions (1-indexed CLI → 0-indexed internal)
    exclude_positions = None
    if args.exclude_positions.lower() != "none":
        exclude_positions = np.array(
            [int(x) - 1 for x in args.exclude_positions.split(",")], dtype=np.int64
        )
        print(f"Excluding positions (1-indexed): {args.exclude_positions}")

    modes = ["gl", "ngl", "full", "region_specific"] if args.run_all_modes else [args.mode]

    # Pre-compute masked logits once (L forward passes) — reused across modes
    print(f"Collecting masked logits for {len(combined_seq)} positions (VH={len(sequence)}, VL={len(light_chain)})")
    masked_data = collect_masked_logits(
        model, sequence, batch_size=args.batch_size, light_chain=light_chain,
    )

    for mode in modes:
        if mode == "region_specific" and region_labels is None:
            print(f"Skipping region_specific mode: --region_string not provided")
            continue

        print(f"\n{'='*60}")
        print(f"Generating {args.n_samples} variants | mode={mode} | n_mut={args.n_mutations}")
        print(f"  T={args.temperature} | pos_T={args.position_temperature} | pool={args.pool_size}")
        print(f"{'='*60}")

        records = generate_variants(
            model=model,
            sequence=sequence,
            n_samples=args.n_samples,
            n_mutations=args.n_mutations,
            mode=mode,
            pool_size=args.pool_size,
            temperature=args.temperature,
            position_temperature=args.position_temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            light_chain=args.light_chain,
            region_labels=region_labels,
            batch_size=args.batch_size,
            seed=args.seed,
            masked_data=masked_data,
            exclude_positions=exclude_positions,
            randomize_n_mutations=args.randomize_n_mutations,
        )

        wt_record = {
            "id": "WT",
            "mode": "none",
            "n_mut": 0,
            "mutations": "",
            "temperature": 0.0,
            "sequence": combined_seq,
        }
        records = [wt_record] + records

        # Insert VH|VL separator for FASTA output
        vh_len = len(sequence)
        for rec in records:
            seq = rec["sequence"]
            rec["sequence"] = seq[:vh_len] + "|" + seq[vh_len:]

        if args.output and not args.run_all_modes:
            out_path = args.output
        else:
            stem = Path(args.checkpoint).stem if Path(args.checkpoint).exists() else "prism"
            out_path = (
                f"pll_guided_{stem}_{mode}_n{args.n_samples}"
                f"_m{args.n_mutations}_T{args.temperature}.fasta"
            )

        write_fasta(records, out_path)
        print(f"Wrote {len(records)} sequences to {out_path}")


if __name__ == "__main__":
    main()
