#!/usr/bin/env python3
"""
FLAb2 Developability Scoring: compute per-antibody scores for all models.

This script ONLY computes scores and saves them. Analysis (correlations,
direction conventions, best signal selection) is done separately in
plot_flab2_reviewer_figures.py.

Models: PRISM v34.1b, ESM2-35M, ESM2-650M, AbLang2, AntiBERTy, Sapiens
Scoring: Masked pseudo-log-likelihood (PLL) — mask each position, sum log P(true_aa | context)
  - Baselines: standard PLL + perplexity
  - PRISM: PLL in 4 modes (marginalized, GL, NGL, exact) + derived signals (alpha stats, etc.)

Output: per_antibody_scores.csv (long format: property, heavy, light, fitness, model, signal, score)

Usage:
    CUDA_VISIBLE_DEVICES=3 python evaluate_flab2_developability.py
    CUDA_VISIBLE_DEVICES=3 python evaluate_flab2_developability.py --properties self_interaction
    CUDA_VISIBLE_DEVICES=3 python evaluate_flab2_developability.py --models prism esm2_650m
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
import prism
from tqdm.auto import tqdm

import warnings

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

# =============================================================================
# Property -> FLAb2 dataset configuration
# =============================================================================

PROPERTY_CONFIG = {
    "self_interaction": {
        "file": "flab2_aggregation.parquet",
        "assay_types": ["ACSINS"],
    },
    "thermostability": {
        "file": "flab2_thermostability.parquet",
        "assay_types": ["DSC"],
    },
    "immunogenicity": {
        "file": "flab2_immunogenicity.parquet",
        "assay_types": ["ADA"],
    },
    "polyreactivity": {
        "file": "flab2_polyreactivity.parquet",
        "assay_types": ["PSR"],
    },
    "expression": {
        "file": "flab2_expression.parquet",
        "assay_types": ["HEK"],
    },
}

ALL_PROPERTIES = list(PROPERTY_CONFIG.keys())

# =============================================================================
# Model configurations
# =============================================================================

# PRISM checkpoint registry. Default points at the published HuggingFace Hub
# model; add local entries via --checkpoint_overrides on the command line.
PRISM_CHECKPOINTS = {
    "prism": "RomeroLab-Duke/prism-antibody",
}

BASELINE_MODEL_IDS = {
    "esm2_35m": "facebook/esm2_t12_35M_UR50D",
    "esm2_650m": "facebook/esm2_t33_650M_UR50D",
}

MODEL_DISPLAY = {
    "prism": "PRISM",
    "esm2_35m": "ESM2-35M",
    "esm2_650m": "ESM2-650M",
    "ablang2": "AbLang2",
    "antiberty": "AntiBERTy",
    "sapiens": "Sapiens",
}

ALL_MODELS = list(MODEL_DISPLAY.keys())


# =============================================================================
# Data loading
# =============================================================================


def load_flab2_property(prop_name: str) -> pd.DataFrame:
    """Load and filter FLAb2 data for a given developability property."""
    cfg = PROPERTY_CONFIG[prop_name]
    path = REPO_ROOT / "data" / "FLAb" / cfg["file"]
    df = pd.read_parquet(path)

    # Filter to selected assay types
    df = df[df["assay_type"].isin(cfg["assay_types"])].copy()

    # Paired only (need both heavy + light for fair comparison)
    df = df[df["light"].notna()].copy()

    # Drop NaN fitness
    df = df[df["fitness"].notna()].copy()

    # Deduplicate: keep first occurrence per unique (heavy, light) pair
    df = df.drop_duplicates(subset=["heavy", "light"], keep="first").reset_index(drop=True)

    return df


# =============================================================================
# PRISM PLL scoring
# =============================================================================


def score_prism_pll(
    model_wrapper, heavy_chains: List[str], light_chains: List[str], batch_size: int = 32
) -> Dict[str, np.ndarray]:
    """Score antibodies with PRISM pseudo_log_likelihood.

    Returns dict of signal_name -> [n_antibodies] arrays.
    Signals: pll_marginalized, pll_gl, pll_ngl, ppl_marginalized, ppl_gl, ppl_ngl,
             plus derived signals (alpha_min, alpha_std, gl_logp_mean, gl_logp_std,
             ngl_logp_mean, ngl_frac_below_threshold).
    """
    n = len(heavy_chains)

    # PLL from API
    results = model_wrapper.pseudo_log_likelihood(
        heavy_chains=heavy_chains, light_chains=light_chains, batch_size=batch_size
    )
    if not isinstance(results, list):
        results = [results]

    pll_marg = np.array([r["marginalized"]["pll"] for r in results])
    pll_gl = np.array([r["gl"]["pll"] for r in results])
    pll_ngl = np.array([r["ngl"]["pll"] for r in results])
    ppl_marg = np.array([r["marginalized"]["perplexity"] for r in results])
    ppl_gl = np.array([r["gl"]["perplexity"] for r in results])
    ppl_ngl = np.array([r["ngl"]["perplexity"] for r in results])

    # Derived per-position signals from PLL outputs
    gl_logp_mean = np.array([r["gl"]["per_position"].mean() for r in results])
    gl_logp_std = np.array([r["gl"]["per_position"].std() for r in results])
    ngl_logp_mean = np.array([r["ngl"]["per_position"].mean() for r in results])

    # Fraction of positions with NGL logp < log(0.01)
    threshold = np.log(0.01)
    ngl_frac_below = np.array(
        [(r["ngl"]["per_position"] < threshold).mean() for r in results]
    )

    # Alpha and origin signals via forward pass
    alpha_min = np.full(n, np.nan)
    alpha_std = np.full(n, np.nan)
    alpha_mean = np.full(n, np.nan)

    print("  Computing alpha signals via forward pass...")
    for i in tqdm(range(n), desc="  PRISM forward", leave=False):
        try:
            fwd = model_wrapper.forward(
                heavy_chains=heavy_chains[i], light_chains=light_chains[i]
            )
            # Paired output: {"heavy": {...}, "light": {...}}
            if "heavy" in fwd and "light" in fwd:
                alpha_h = fwd["heavy"]["alpha"]
                alpha_l = fwd["light"]["alpha"]
                alpha_all = np.concatenate([alpha_h, alpha_l])
            else:
                alpha_all = fwd["alpha"]

            alpha_min[i] = alpha_all.min()
            alpha_std[i] = alpha_all.std()
            alpha_mean[i] = alpha_all.mean()
        except Exception as e:
            print(f"  Warning: forward failed for seq {i}: {e}")
            continue

    return {
        "pll_marginalized": pll_marg,
        "pll_gl": pll_gl,
        "pll_ngl": pll_ngl,
        "ppl_marginalized": ppl_marg,
        "ppl_gl": ppl_gl,
        "ppl_ngl": ppl_ngl,
        "gl_logp_mean": gl_logp_mean,
        "gl_logp_std": gl_logp_std,
        "ngl_logp_mean": ngl_logp_mean,
        "ngl_frac_below": ngl_frac_below,
        "alpha_min": alpha_min,
        "alpha_std": alpha_std,
        "alpha_mean": alpha_mean,
    }


# =============================================================================
# Baseline PLL scoring (ESM2, AbLang2, AntiBERTy, Sapiens)
# =============================================================================


def compute_pll_esm2(model, tokenizer, sequence, device, batch_size=32):
    """Masked PLL for ESM2. Returns (pll, perplexity)."""
    inputs = tokenizer(
        sequence,
        return_tensors="pt",
        add_special_tokens=True,
        truncation=True,
        max_length=1024,
    )
    input_ids = inputs["input_ids"].to(device)
    attn_mask = inputs["attention_mask"].to(device)

    mask_id = tokenizer.mask_token_id
    special_ids = {tokenizer.cls_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id}
    special_ids.discard(None)

    seq_len = attn_mask[0].sum().item()
    maskable = [i for i in range(seq_len) if input_ids[0, i].item() not in special_ids]
    original_ids = [input_ids[0, pos].item() for pos in maskable]

    total_logp = 0.0
    for start in range(0, len(maskable), batch_size):
        chunk_pos = maskable[start : start + batch_size]
        chunk_orig = original_ids[start : start + batch_size]
        bs = len(chunk_pos)

        batch = input_ids.expand(bs, -1).clone()
        batch_attn = attn_mask.expand(bs, -1)
        for i, pos in enumerate(chunk_pos):
            batch[i, pos] = mask_id

        with torch.no_grad():
            outputs = model(input_ids=batch, attention_mask=batch_attn)
            logits = outputs.logits

        for i, (pos, orig_id) in enumerate(zip(chunk_pos, chunk_orig)):
            lp = F.log_softmax(logits[i, pos], dim=-1)
            total_logp += lp[orig_id].item()

    n_aa = len(maskable)
    ppl = np.exp(-total_logp / n_aa) if n_aa > 0 else float("inf")
    return total_logp, ppl


def compute_pll_ablang2(model, tokenizer, heavy, light, device, batch_size=32):
    """Masked PLL for AbLang2 paired model."""
    paired_seq = f"{heavy}|{light}"
    tokenized_full = tokenizer([paired_seq], pad=True, w_extra_tkns=False, device=device)
    input_ids = tokenized_full  # AbLang2 tokenizer returns tensor directly

    seq_len = input_ids.shape[1]
    # AbLang2: no CLS/EOS, separator is '|' which we skip
    # Positions: 0..len(heavy)-1 = heavy, len(heavy) = separator, len(heavy)+1.. = light
    sep_pos = len(heavy)
    maskable = [i for i in range(seq_len) if i != sep_pos]
    original_ids = [input_ids[0, pos].item() for pos in maskable]

    # AbLang2 mask token
    if hasattr(tokenizer, "mask_token"):
        mask_id = tokenizer.aa_to_token.get("*", 23)  # '*' is mask in ablang2
    else:
        mask_id = 23  # default mask token ID for ablang2

    total_logp = 0.0
    for start in range(0, len(maskable), batch_size):
        chunk_pos = maskable[start : start + batch_size]
        chunk_orig = original_ids[start : start + batch_size]
        bs = len(chunk_pos)

        batch = input_ids.expand(bs, -1).clone()
        for i, pos in enumerate(chunk_pos):
            batch[i, pos] = mask_id

        with torch.no_grad():
            outputs = model(batch)
            logits = outputs

        for i, (pos, orig_id) in enumerate(zip(chunk_pos, chunk_orig)):
            lp = F.log_softmax(logits[i, pos], dim=-1)
            total_logp += lp[orig_id].item()

    n_aa = len(maskable)
    ppl = np.exp(-total_logp / n_aa) if n_aa > 0 else float("inf")
    return total_logp, ppl


def compute_pll_antiberty(model, tokenizer, sequence, device, batch_size=32):
    """Masked PLL for AntiBERTy."""
    spaced_seq = " ".join(list(sequence))
    tokens = tokenizer(
        spaced_seq,
        return_tensors="pt",
        add_special_tokens=True,
        truncation=True,
        max_length=1024,
    )
    input_ids = tokens["input_ids"].to(device)
    attn_mask = tokens["attention_mask"].to(device)

    mask_id = tokenizer.mask_token_id
    special_ids = {tokenizer.cls_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id}
    if tokenizer.sep_token_id is not None:
        special_ids.add(tokenizer.sep_token_id)
    special_ids.discard(None)

    seq_len = attn_mask[0].sum().item()
    maskable = [i for i in range(seq_len) if input_ids[0, i].item() not in special_ids]
    original_ids = [input_ids[0, pos].item() for pos in maskable]

    total_logp = 0.0
    for start in range(0, len(maskable), batch_size):
        chunk_pos = maskable[start : start + batch_size]
        chunk_orig = original_ids[start : start + batch_size]
        bs = len(chunk_pos)

        batch = input_ids.expand(bs, -1).clone()
        batch_attn = attn_mask.expand(bs, -1)
        for i, pos in enumerate(chunk_pos):
            batch[i, pos] = mask_id

        with torch.no_grad():
            outputs = model(input_ids=batch, attention_mask=batch_attn)
            if hasattr(outputs, "logits") and outputs.logits is not None:
                logits = outputs.logits
            elif hasattr(outputs, "prediction_logits"):
                logits = outputs.prediction_logits
            else:
                return 0.0, float("inf")

        for i, (pos, orig_id) in enumerate(zip(chunk_pos, chunk_orig)):
            lp = F.log_softmax(logits[i, pos], dim=-1)
            total_logp += lp[orig_id].item()

    n_aa = len(maskable)
    ppl = np.exp(-total_logp / n_aa) if n_aa > 0 else float("inf")
    return total_logp, ppl


def compute_pll_sapiens(
    heavy_model, light_model, tokenizer, heavy, light, device, batch_size=32, max_seq_len=143
):
    """Masked PLL for Sapiens (separate VH/VL models)."""
    special_ids = {tokenizer.cls_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id}
    special_ids.discard(None)
    mask_id = tokenizer.mask_token_id

    total_logp = 0.0
    total_aa = 0

    for chain_model, chain_seq in [(heavy_model, heavy), (light_model, light)]:
        if len(chain_seq) > max_seq_len:
            continue

        tokens = tokenizer(
            chain_seq,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=max_seq_len + 2,
        )
        input_ids = tokens["input_ids"].to(device)
        attn_mask = tokens["attention_mask"].to(device)

        seq_len = attn_mask[0].sum().item()
        maskable = [i for i in range(seq_len) if input_ids[0, i].item() not in special_ids]
        original_ids = [input_ids[0, pos].item() for pos in maskable]

        for start in range(0, len(maskable), batch_size):
            chunk_pos = maskable[start : start + batch_size]
            chunk_orig = original_ids[start : start + batch_size]
            bs = len(chunk_pos)

            batch = input_ids.expand(bs, -1).clone()
            batch_attn = attn_mask.expand(bs, -1)
            for i, pos in enumerate(chunk_pos):
                batch[i, pos] = mask_id

            with torch.no_grad():
                outputs = chain_model(input_ids=batch, attention_mask=batch_attn)
                logits = outputs.logits

            for i, (pos, orig_id) in enumerate(zip(chunk_pos, chunk_orig)):
                lp = F.log_softmax(logits[i, pos], dim=-1)
                total_logp += lp[orig_id].item()

        total_aa += len(maskable)

    ppl = np.exp(-total_logp / total_aa) if total_aa > 0 else float("inf")
    return total_logp, ppl


# =============================================================================
# Model loaders
# =============================================================================


def load_prism(device):
    """Load PRISM model."""
    import prism

    ckpt = str(PRISM_CHECKPOINTS["prism"])
    model_wrapper = prism.pretrained(ckpt, device=device)
    model_wrapper.model.eval()
    return model_wrapper


def load_esm2(model_id, device):
    """Load ESM2 model."""
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForMaskedLM.from_pretrained(model_id)
    model.to(device).eval()
    return model, tokenizer


def load_ablang2_model(device):
    """Load AbLang2 paired model."""
    import ablang2

    ablang = ablang2.pretrained(model_to_use="ablang2-paired", random_init=False, device=device)
    model = ablang.AbLang
    tokenizer = ablang.tokenizer
    model.eval()
    return model, tokenizer


def load_antiberty_model(device):
    """Load AntiBERTy model."""
    from antiberty import AntiBERTyRunner

    runner = AntiBERTyRunner()
    model = runner.model.to(device)
    tokenizer = runner.tokenizer
    model.eval()
    return model, tokenizer


def load_sapiens_model():
    """Load Sapiens (CPU-only)."""
    from transformers import RobertaForMaskedLM, RobertaTokenizer

    device = "cpu"
    tokenizer = RobertaTokenizer.from_pretrained("prihodad/biophi-sapiens1-tokenizer")
    heavy_model = RobertaForMaskedLM.from_pretrained("prihodad/biophi-sapiens1-vh")
    light_model = RobertaForMaskedLM.from_pretrained("prihodad/biophi-sapiens1-vl")
    heavy_model.to(device).eval()
    light_model.to(device).eval()
    return heavy_model, light_model, tokenizer


# =============================================================================
# Scoring dispatch
# =============================================================================


def score_antibodies(
    model_key: str,
    heavy_chains: List[str],
    light_chains: List[str],
    device: str,
    batch_size: int = 32,
) -> Dict[str, np.ndarray]:
    """Score a list of antibodies with a given model.

    Returns dict of score_name -> [n_antibodies] arrays.
    """
    n = len(heavy_chains)

    if model_key in PRISM_CHECKPOINTS:
        ckpt = str(PRISM_CHECKPOINTS[model_key])
        model_wrapper = prism.pretrained(ckpt, device=device)
        scores = score_prism_pll(model_wrapper, heavy_chains, light_chains, batch_size)
        del model_wrapper
        torch.cuda.empty_cache()
        return scores

    # Baseline models: compute PLL per antibody
    pll_scores = np.zeros(n)
    ppl_scores = np.zeros(n)

    if model_key in BASELINE_MODEL_IDS:
        model, tokenizer = load_esm2(BASELINE_MODEL_IDS[model_key], device)
        for i in tqdm(range(n), desc=f"  {MODEL_DISPLAY[model_key]} PLL"):
            seq = heavy_chains[i] + light_chains[i]
            pll_scores[i], ppl_scores[i] = compute_pll_esm2(
                model, tokenizer, seq, device, batch_size
            )
        del model, tokenizer

    elif model_key == "ablang2":
        model, tokenizer = load_ablang2_model(device)
        for i in tqdm(range(n), desc=f"  {MODEL_DISPLAY[model_key]} PLL"):
            pll_scores[i], ppl_scores[i] = compute_pll_ablang2(
                model, tokenizer, heavy_chains[i], light_chains[i], device, batch_size
            )
        del model, tokenizer

    elif model_key == "antiberty":
        model, tokenizer = load_antiberty_model(device)
        for i in tqdm(range(n), desc=f"  {MODEL_DISPLAY[model_key]} PLL"):
            seq = heavy_chains[i] + light_chains[i]
            pll_scores[i], ppl_scores[i] = compute_pll_antiberty(
                model, tokenizer, seq, device, batch_size
            )
        del model, tokenizer

    elif model_key == "sapiens":
        heavy_model, light_model, tokenizer = load_sapiens_model()
        for i in tqdm(range(n), desc=f"  {MODEL_DISPLAY[model_key]} PLL"):
            pll_scores[i], ppl_scores[i] = compute_pll_sapiens(
                heavy_model, light_model, tokenizer,
                heavy_chains[i], light_chains[i], "cpu", batch_size
            )
        del heavy_model, light_model, tokenizer

    if device != "cpu":
        torch.cuda.empty_cache()

    return {"pll": pll_scores, "ppl": ppl_scores}


# =============================================================================
# Main — pure scoring, no analysis
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="FLAb2 Developability Scoring")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--properties",
        nargs="+",
        default=ALL_PROPERTIES,
        choices=ALL_PROPERTIES,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=ALL_MODELS,
        choices=ALL_MODELS,
    )
    parser.add_argument(
        "--checkpoint_overrides", nargs="*", default=[],
        metavar="LABEL=PATH_OR_HF_ID",
        help="Add or override entries in PRISM_CHECKPOINTS at runtime, "
             "e.g. --checkpoint_overrides prism_local=/path/to/last.ckpt",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    for entry in args.checkpoint_overrides:
        if "=" not in entry:
            raise SystemExit(f"--checkpoint_overrides expects LABEL=SPEC, got {entry!r}")
        label, spec = entry.split("=", 1)
        PRISM_CHECKPOINTS[label.strip()] = spec.strip()

    device = args.device if torch.cuda.is_available() else "cpu"

    out_dir = REPO_ROOT / "data" / "features" / "evaluation_results" / "flab2_developability"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load all property datasets
    print("Loading FLAb2 developability datasets...")
    datasets = {}
    for prop in args.properties:
        df = load_flab2_property(prop)
        cfg = PROPERTY_CONFIG[prop]
        print(f"  {prop}: {len(df)} antibodies ({', '.join(cfg['assay_types'])})")
        datasets[prop] = df

    all_score_results = []

    for model_key in args.models:
        display = MODEL_DISPLAY[model_key]
        print(f"\n{'=' * 70}")
        print(f"Model: {display}")
        print(f"{'=' * 70}")
        t_model_start = time.time()

        for prop in args.properties:
            df = datasets[prop]
            heavy_chains = df["heavy"].tolist()
            light_chains = df["light"].tolist()

            print(f"\n  Property: {prop} (n={len(df)})")
            t0 = time.time()

            try:
                scores = score_antibodies(
                    model_key, heavy_chains, light_chains, device, args.batch_size
                )
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
                continue

            elapsed = time.time() - t0
            print(f"  Scored {len(scores)} signals in {elapsed:.1f}s")

            # Save per-antibody scores (long format)
            for sig_name, vals in scores.items():
                score_df = df[["heavy", "light", "fitness", "assay_type"]].copy()
                if "study" in df.columns:
                    score_df["study"] = df["study"]
                score_df["model"] = model_key
                score_df["property"] = prop
                score_df["signal"] = sig_name
                score_df["score"] = vals
                all_score_results.append(score_df)

        print(f"\n  Total time for {display}: {time.time() - t_model_start:.1f}s")

    # Save per-antibody scores
    if all_score_results:
        scores_df = pd.concat(all_score_results, ignore_index=True)
        scores_path = out_dir / "per_antibody_scores.csv"
        scores_df.to_csv(scores_path, index=False)
        print(f"\nSaved per-antibody scores: {scores_path}")
        print(f"  {len(scores_df)} rows")
        print(f"  Models: {scores_df['model'].unique().tolist()}")
        print(f"  Properties: {scores_df['property'].unique().tolist()}")
        print(f"  Signals per model:")
        for m in scores_df["model"].unique():
            sigs = scores_df[scores_df["model"] == m]["signal"].unique().tolist()
            print(f"    {m}: {sigs}")
    else:
        print("\nNo scores computed.")

    print("\nDone. Run plot_flab2_reviewer_figures.py for correlation analysis and figures.")


if __name__ == "__main__":
    main()
