#!/usr/bin/env python3
"""
FLAb2 Binding Affinity Evaluation: Per-Protein Spearman ρ.

Scoring: masked_individual + wt_center (WT as germline proxy)
Signals: origin_logit, origin_prob, ngl_logprob, alpha
Aggregations: sum, mean

Usage:
    CUDA_VISIBLE_DEVICES=6 python evaluate_flab2_binding.py
"""

import os
import sys
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
import prism

REPO_ROOT = Path(__file__).resolve().parents[3]

# Default checkpoint registry. Each entry maps a short label to a checkpoint
# spec that `prism.pretrained()` accepts (HuggingFace Hub id or local path).
# Override on the command line with `--checkpoints "label1=spec1" ...`.
CHECKPOINTS = {
    "prism": "RomeroLab-Duke/prism-antibody",
}

# Exclude already-evaluated or unsuitable datasets
EXCLUDE_SOURCES = {
    "koenig2017mutational_kd_g6.csv",                   # = g6.31
    "shanehsazzadeh2023unlocking_zerokd_trastuzumab.csv",  # = trastuzumab
    "phillips2021binding_cr9114_h3_kd.csv",              # = cr9114 h3
    "phillips2021binding_cr9114_h1_kd.csv",              # = cr9114 h1
    "AbRank_dataset.csv.zip",                            # mixed, not single-protein DMS
}

MIN_VARIANTS = 15


def load_flab2_binding():
    """Load and filter FLAb2 binding data."""
    df = pd.read_parquet(REPO_ROOT / "data" / "FLAb" / "flab2_binding.parquet")

    # Paired only
    df = df[df["light"].notna()].copy()

    # Exclude binary and predicted
    df = df[~df["assay_type"].isin(["bind/no bind", "predicted Kd"])]

    # Exclude known datasets
    df = df[~df["source_file"].isin(EXCLUDE_SOURCES)]

    # Filter by group size
    counts = df["source_file"].value_counts()
    valid_sources = counts[counts >= MIN_VARIANTS].index
    df = df[df["source_file"].isin(valid_sources)].copy()

    return df


def identify_wt(group_df):
    """Identify WT as the sequence with max fitness or most common sequence."""
    # For -log(Kd) datasets, higher = tighter binding = likely WT
    # Use most common heavy chain as WT
    heavy_counts = group_df["heavy"].value_counts()
    wt_heavy = heavy_counts.index[0]
    wt_rows = group_df[group_df["heavy"] == wt_heavy]

    if len(wt_rows) > 1:
        # If multiple rows share the same heavy, pick by most common light
        light_counts = wt_rows["light"].value_counts()
        wt_light = light_counts.index[0]
        wt_row = wt_rows[wt_rows["light"] == wt_light].iloc[0]
    else:
        wt_row = wt_rows.iloc[0]

    return wt_row["heavy"], wt_row["light"]


def find_mutations(wt_heavy, wt_light, var_heavy, var_light):
    """Find mutation positions relative to WT in the concatenated VH+VL sequence.

    Returns list of (position_in_concat, wt_aa, mut_aa).
    Only works when sequences are same length.
    """
    mutations = []
    if len(wt_heavy) != len(var_heavy) or len(wt_light) != len(var_light):
        return None  # length mismatch, skip

    for i in range(len(wt_heavy)):
        if wt_heavy[i] != var_heavy[i]:
            mutations.append((i, wt_heavy[i], var_heavy[i]))

    offset = len(wt_heavy)
    for i in range(len(wt_light)):
        if wt_light[i] != var_light[i]:
            mutations.append((offset + i, wt_light[i], var_light[i]))

    return mutations if mutations else None  # None for WT itself


def format_sequence(heavy, light, cls_token="<cls>"):
    """Format as VH<cls><cls>VL for ESM2 tokenizer."""
    return f"{heavy}{cls_token}{cls_token}{light}"


def seq_positions_to_token_positions(input_ids, special_ids):
    """Map AA sequence positions (0-indexed) to token positions.

    Returns list of token positions for each AA in the sequence.
    """
    aa_token_positions = []
    for pos in range(len(input_ids)):
        tid = input_ids[pos].item() if hasattr(input_ids[pos], "item") else input_ids[pos]
        if tid not in special_ids:
            aa_token_positions.append(pos)
    return aa_token_positions


def extract_wt_signals(model_wrapper, wt_formatted, device):
    """Run unmasked forward pass on WT to get reference signals at all positions."""
    model = model_wrapper.model
    tokenizer = model_wrapper.tokenizer

    cls_id = tokenizer.cls_token_id
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    special_ids = {cls_id, eos_id, pad_id}
    special_ids.discard(None)

    encoded = tokenizer(wt_formatted, return_tensors="pt", padding=False,
                        truncation=True, max_length=512, add_special_tokens=True)
    input_ids = encoded["input_ids"].to(device)
    attn_mask = encoded["attention_mask"].to(device)

    unk_gene_id = 1
    v_ids = torch.full((1,), unk_gene_id, dtype=torch.long, device=device)
    j_ids = torch.full((1,), unk_gene_id, dtype=torch.long, device=device)

    with torch.no_grad():
        logits_aa, logits_aa_ngl, logits_mut, alpha, logits_final, _ = \
            model._forward_multihead(
                input_ids=input_ids, attention_mask=attn_mask,
                v_gene_ids=v_ids if model.use_germline_genes else None,
                j_gene_ids=j_ids if model.use_germline_genes else None,
                region_ids=None,
            )

    sl = attn_mask[0].sum().item()
    aa_positions = seq_positions_to_token_positions(input_ids[0, :sl], special_ids)

    # Per-position signals (AA positions only)
    aa_idx = torch.tensor(aa_positions, dtype=torch.long, device=device)
    origin_logits = logits_mut[0, aa_idx].cpu().numpy()
    origin_probs = torch.sigmoid(logits_mut[0, aa_idx]).cpu().numpy()
    alphas = alpha[0, aa_idx].cpu().numpy() if alpha is not None else np.zeros(len(aa_positions))

    # NGL logprob: for WT, get logP(wt_token) from final head's NGL slots
    ngl_logprobs = np.zeros(len(aa_positions), dtype=np.float32)
    if logits_final is not None:
        lp_final = F.log_softmax(logits_final[0], dim=-1)
        lowercase_map = model.lowercase_aa_token_ids
        for i, tok_pos in enumerate(aa_positions):
            orig_id = input_ids[0, tok_pos].item()
            lower_id = lowercase_map.get(orig_id)
            if lower_id is not None:
                ngl_logprobs[i] = lp_final[tok_pos, lower_id].item()
            else:
                ngl_logprobs[i] = lp_final[tok_pos, orig_id].item()

    return {
        "origin_logit": origin_logits,
        "origin_prob": origin_probs,
        "alpha": alphas,
        "ngl_logprob": ngl_logprobs,
        "aa_positions": aa_positions,
        "input_ids": input_ids[0, :sl].cpu(),
    }


def score_variants_masked_individual(model_wrapper, variants_info, wt_signals, device,
                                     batch_size=128):
    """Score variants using masked_individual approach.

    variants_info: list of (mutation_positions_in_seq, formatted_seq, fitness)
    wt_signals: dict from extract_wt_signals

    Returns dict of signal_name -> [n_variants, ] aggregated scores
    """
    model = model_wrapper.model
    tokenizer = model_wrapper.tokenizer

    mask_token_id = tokenizer.mask_token_id
    cls_id = tokenizer.cls_token_id
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    special_ids = {cls_id, eos_id, pad_id}
    special_ids.discard(None)
    lowercase_map = model.lowercase_aa_token_ids
    unk_gene_id = 1

    n_variants = len(variants_info)

    # Collect all (variant_idx, mutation_seq_pos, formatted_seq) to batch
    items = []
    for vi, (mutations, fmt_seq, _) in enumerate(variants_info):
        for seq_pos, wt_aa, mut_aa in mutations:
            items.append((vi, seq_pos, fmt_seq))

    if not items:
        return None

    # Pre-tokenize all unique sequences
    unique_seqs = list(set(it[2] for it in items))
    seq_to_encoded = {}
    for seq in unique_seqs:
        enc = tokenizer(seq, return_tensors="pt", padding=False, truncation=True,
                        max_length=512, add_special_tokens=True)
        seq_to_encoded[seq] = enc["input_ids"][0]

    # Initialize per-variant accumulators
    signals = {
        "origin_logit": [[] for _ in range(n_variants)],
        "origin_prob": [[] for _ in range(n_variants)],
        "alpha": [[] for _ in range(n_variants)],
        "ngl_logprob": [[] for _ in range(n_variants)],
    }

    for chunk_start in range(0, len(items), batch_size):
        chunk = items[chunk_start: chunk_start + batch_size]
        bs = len(chunk)

        # Build masked input batch
        max_len = max(len(seq_to_encoded[it[2]]) for it in chunk)
        batch_input = torch.full((bs, max_len), pad_id, dtype=torch.long, device=device)
        batch_attn = torch.zeros((bs, max_len), dtype=torch.long, device=device)
        token_positions = []  # which token position to read from

        for i, (vi, seq_pos, fmt_seq) in enumerate(chunk):
            ids = seq_to_encoded[fmt_seq]
            sl = len(ids)
            batch_input[i, :sl] = ids.to(device)
            batch_attn[i, :sl] = 1

            # Map sequence position to token position
            aa_tok_positions = seq_positions_to_token_positions(ids, special_ids)
            if seq_pos < len(aa_tok_positions):
                tok_pos = aa_tok_positions[seq_pos]
                batch_input[i, tok_pos] = mask_token_id
                token_positions.append(tok_pos)
            else:
                token_positions.append(0)  # fallback, will be filtered

        v_ids = None
        j_ids = None
        if model.use_germline_genes:
            v_ids = torch.full((bs,), unk_gene_id, dtype=torch.long, device=device)
            j_ids = torch.full((bs,), unk_gene_id, dtype=torch.long, device=device)

        with torch.no_grad():
            logits_aa, logits_aa_ngl, logits_mut, alpha_out, logits_final, _ = \
                model._forward_multihead(
                    input_ids=batch_input, attention_mask=batch_attn,
                    v_gene_ids=v_ids, j_gene_ids=j_ids, region_ids=None,
                )

        # Extract signals at masked positions
        for i, (vi, seq_pos, fmt_seq) in enumerate(chunk):
            tp = token_positions[i]

            # Origin signals at masked position
            ol = logits_mut[i, tp].item()
            op = torch.sigmoid(logits_mut[i, tp]).item()
            al = alpha_out[i, tp].item() if alpha_out is not None else 0.0

            # NGL logprob at masked position
            nl = 0.0
            if logits_final is not None:
                lp = F.log_softmax(logits_final[i, tp], dim=-1)
                # Get the original (unmasked) token to find its NGL logprob
                orig_ids = seq_to_encoded[fmt_seq]
                orig_id = orig_ids[tp].item()
                lower_id = lowercase_map.get(orig_id)
                if lower_id is not None:
                    nl = lp[lower_id].item()
                else:
                    nl = lp[orig_id].item()

            # WT-center: subtract WT signal at same seq position
            wt_ol = wt_signals["origin_logit"][seq_pos] if seq_pos < len(wt_signals["origin_logit"]) else 0.0
            wt_op = wt_signals["origin_prob"][seq_pos] if seq_pos < len(wt_signals["origin_prob"]) else 0.0
            wt_al = wt_signals["alpha"][seq_pos] if seq_pos < len(wt_signals["alpha"]) else 0.0
            wt_nl = wt_signals["ngl_logprob"][seq_pos] if seq_pos < len(wt_signals["ngl_logprob"]) else 0.0

            signals["origin_logit"][vi].append(ol - wt_ol)
            signals["origin_prob"][vi].append(op - wt_op)
            signals["alpha"][vi].append(al - wt_al)
            signals["ngl_logprob"][vi].append(nl - wt_nl)

    # Aggregate per variant
    results = {}
    for sig_name in signals:
        sums = np.full(n_variants, np.nan)
        means = np.full(n_variants, np.nan)
        for vi in range(n_variants):
            vals = signals[sig_name][vi]
            if vals:
                arr = np.array(vals)
                sums[vi] = np.sum(arr)
                means[vi] = np.mean(arr)
        results[f"{sig_name}_sum"] = sums
        results[f"{sig_name}_mean"] = means

    return results


def evaluate_protein(model_wrapper, group_df, device):
    """Evaluate a single protein group. Returns per-signal Spearman ρ."""
    wt_heavy, wt_light = identify_wt(group_df)
    wt_formatted = format_sequence(wt_heavy, wt_light)

    # Extract WT reference signals
    wt_signals = extract_wt_signals(model_wrapper, wt_formatted, device)

    # Prepare variants
    variants_info = []
    fitness_values = []
    for _, row in group_df.iterrows():
        muts = find_mutations(wt_heavy, wt_light, row["heavy"], row["light"])
        if muts is None:
            continue  # length mismatch or WT itself
        fmt = format_sequence(row["heavy"], row["light"])
        variants_info.append((muts, fmt, row["fitness"]))
        fitness_values.append(row["fitness"])

    if len(variants_info) < 10:
        return None

    fitness = np.array(fitness_values)

    # Score
    scores = score_variants_masked_individual(model_wrapper, variants_info, wt_signals, device)
    if scores is None:
        return None

    # Compute Spearman ρ for each signal×aggregation
    results = {"n_variants": len(variants_info)}
    for method_name, method_scores in scores.items():
        valid = np.isfinite(method_scores) & np.isfinite(fitness)
        if valid.sum() < 10:
            results[method_name] = np.nan
            continue
        rho, pval = spearmanr(method_scores[valid], fitness[valid])
        results[method_name] = rho

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--models", nargs="+", default=["prism"],
                        help=f"Which entries from CHECKPOINTS to evaluate. "
                             f"Available: {list(CHECKPOINTS.keys())}. "
                             f"Override CHECKPOINTS at the top of this file or via "
                             f"--checkpoint_overrides to add custom local paths.")
    parser.add_argument("--checkpoint_overrides", nargs="*", default=[],
                        metavar="LABEL=PATH_OR_HF_ID",
                        help="Add or override checkpoint specs at runtime, "
                             "e.g. --checkpoint_overrides v34.1b_local=/path/to/last.ckpt")
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    device = args.device

    for entry in args.checkpoint_overrides:
        if "=" not in entry:
            raise SystemExit(f"--checkpoint_overrides expects LABEL=SPEC, got {entry!r}")
        label, spec = entry.split("=", 1)
        CHECKPOINTS[label.strip()] = spec.strip()

    print("Loading FLAb2 binding data...")
    df = load_flab2_binding()
    sources = df["source_file"].unique()
    print(f"  {len(df)} variants across {len(sources)} proteins")

    all_results = []

    for model_name in args.models:
        ckpt = str(CHECKPOINTS[model_name])
        print(f"\n{'='*70}")
        print(f"Model: {model_name}")
        print(f"{'='*70}")

        model_wrapper = prism.pretrained(ckpt, device=device)
        model_wrapper.model.eval()

        for src in tqdm(sources, desc=f"{model_name}"):
            group = df[df["source_file"] == src].copy()
            t0 = time.time()

            result = evaluate_protein(model_wrapper, group, device)
            elapsed = time.time() - t0

            if result is None:
                continue

            result["model"] = model_name
            result["source_file"] = src
            result["assay_type"] = group["assay_type"].iloc[0]
            result["study"] = group["study"].iloc[0]
            result["elapsed_s"] = elapsed
            all_results.append(result)

        del model_wrapper
        torch.cuda.empty_cache()

    # Build results DataFrame
    results_df = pd.DataFrame(all_results)

    out_dir = REPO_ROOT / "data" / "features" / "evaluation_results" / "flab2_binding"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_dir / "per_protein_results.csv", index=False)
    print(f"\nSaved: {out_dir / 'per_protein_results.csv'}")

    # Print summary
    signal_cols = [c for c in results_df.columns
                   if c.endswith("_sum") or c.endswith("_mean")]

    print(f"\n{'='*70}")
    print("PER-PROTEIN RESULTS SUMMARY")
    print(f"{'='*70}")

    for model_name in args.models:
        mdf = results_df[results_df["model"] == model_name]
        print(f"\n--- {model_name} ({len(mdf)} proteins) ---")
        print(f"{'Signal':<25} {'Mean ρ':>8} {'Median ρ':>10} {'Std':>8} {'Min':>8} {'Max':>8}")
        print("-" * 75)
        for col in signal_cols:
            vals = mdf[col].dropna()
            if len(vals) == 0:
                continue
            print(f"{col:<25} {vals.mean():>8.4f} {vals.median():>10.4f} "
                  f"{vals.std():>8.4f} {vals.min():>8.4f} {vals.max():>8.4f}")

    # Cross-model comparison
    if len(args.models) == 2:
        m1, m2 = args.models
        df1 = results_df[results_df["model"] == m1].set_index("source_file")
        df2 = results_df[results_df["model"] == m2].set_index("source_file")
        common = df1.index.intersection(df2.index)

        print(f"\n{'='*70}")
        print(f"HEAD-TO-HEAD: {m1} vs {m2} ({len(common)} shared proteins)")
        print(f"{'='*70}")
        print(f"{'Signal':<25} {m1+' mean':>12} {m2+' mean':>12} {'Δ':>8} {m1+' wins':>10}")
        print("-" * 75)
        for col in signal_cols:
            v1 = df1.loc[common, col].dropna()
            v2 = df2.loc[common, col].dropna()
            both = v1.index.intersection(v2.index)
            if len(both) < 5:
                continue
            mean1 = v1[both].mean()
            mean2 = v2[both].mean()
            wins1 = (v1[both] > v2[both]).sum()
            print(f"{col:<25} {mean1:>12.4f} {mean2:>12.4f} {mean1 - mean2:>8.4f} "
                  f"{wins1}/{len(both)}")


if __name__ == "__main__":
    main()
