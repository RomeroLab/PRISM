#!/usr/bin/env python
# coding: utf-8

"""
Inference script for PureESM2 baseline model — masked marginal log-likelihood scoring.

For each sequence, masks each amino acid position one at a time, runs a forward pass,
and records the log-probability of the true token. Outputs per-position LL lists
(for downstream stratification by region/chain/GL-NGL) and per-sequence perplexity.

Output columns added to the CSV:
    {experiment_name}_LL   — list of per-position log-probs (AA positions only)
    {experiment_name}_PP   — scalar pseudo-perplexity

Usage:
    python script/inference_pure_esm_with_logprobs.py \
        --checkpoint outputs/ESM2_baseline_finetune_.../checkpoints/val_ppl-53-1.5111.ckpt \
        --data_path data/ginkgo/developability_data.csv \
        --output_path data/ginkgo/developability_baseline_ppl.csv \
        --experiment_name ESM2_baseline \
        --batch_size 4096
"""

import argparse
import json
import pathlib

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.serialization
from pathlib import Path
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer

torch.serialization.add_safe_globals([pathlib.PosixPath])


def load_model(checkpoint_path, model_identifier="esm2_t12_35M_UR50D", device="cuda"):
    """Load PureESM2 weights from a Lightning checkpoint into a HuggingFace model."""
    config = AutoConfig.from_pretrained(f"facebook/{model_identifier}")
    model = AutoModelForMaskedLM.from_config(config)
    tokenizer = AutoTokenizer.from_pretrained(f"facebook/{model_identifier}")

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt["state_dict"]

    # Lightning wraps keys as "model.esm.*" -> strip "model." prefix
    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            cleaned[k[len("model."):]] = v
        else:
            cleaned[k] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"  Warning — missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  Warning — unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    model.eval()
    model.to(device)
    del ckpt, state_dict, cleaned
    torch.cuda.empty_cache()
    print(f"Model loaded: {model_identifier} on {device}")
    return model, tokenizer


def build_sequences(df, cls_token):
    """Build VH<cls><cls>VL concatenated sequences from dataframe."""
    vh_col = next(
        (c for c in ["vh_protein_sequence", "HEAVY_CHAIN_AA_SEQUENCE"] if c in df.columns),
        None,
    )
    vl_col = next(
        (c for c in ["vl_protein_sequence", "LIGHT_CHAIN_AA_SEQUENCE"] if c in df.columns),
        None,
    )

    if vh_col is None and vl_col is None:
        raise ValueError("No VH/VL sequence columns found")

    sequences = []
    for _, row in df.iterrows():
        vh = str(row[vh_col]) if vh_col and pd.notna(row.get(vh_col)) else ""
        vl = str(row[vl_col]) if vl_col and pd.notna(row.get(vl_col)) else ""
        if vh and vl:
            sequences.append(f"{vh}{cls_token}{cls_token}{vl}")
        elif vh:
            sequences.append(vh)
        elif vl:
            sequences.append(vl)
    return sequences


@torch.no_grad()
def masked_marginal_logprobs(model, tokenizer, sequences, device="cuda", batch_size=4096):
    """Compute per-position masked marginal log-probabilities.

    For each AA position, mask it, forward pass, extract log P(true token).

    Returns:
        log_probs_list: list of list[float], per-position log-probs (AA only)
        perplexity_list: list[float], per-sequence pseudo-perplexity
    """
    mask_token_id = tokenizer.mask_token_id

    tokens = tokenizer(
        sequences, return_tensors="pt", add_special_tokens=True,
        padding=True, truncation=True, max_length=512,
    )
    all_input_ids = tokens["input_ids"].to(device)
    all_attention_mask = tokens["attention_mask"].to(device)
    N, L = all_input_ids.shape
    print(f"Tokenized {N} sequences, max length {L}")

    # AA positions: ESM2 standard token IDs 4-23
    aa_mask = (all_input_ids >= 4) & (all_input_ids <= 23)

    log_probs_tensor = torch.zeros(N, L, dtype=torch.float32, device="cpu")

    for pos in tqdm(range(L), desc="Masking positions"):
        if not aa_mask[:, pos].any():
            continue

        for batch_start in range(0, N, batch_size):
            batch_end = min(batch_start + batch_size, N)

            masked_input = all_input_ids[batch_start:batch_end].clone()
            masked_input[:, pos] = mask_token_id

            outputs = model(
                input_ids=masked_input,
                attention_mask=all_attention_mask[batch_start:batch_end],
            )
            logits_at_pos = outputs.logits[:, pos, :]
            log_probs_at_pos = F.log_softmax(logits_at_pos, dim=-1)

            true_tokens = all_input_ids[batch_start:batch_end, pos].unsqueeze(1)
            lp = log_probs_at_pos.gather(1, true_tokens).squeeze(1)
            lp = lp * aa_mask[batch_start:batch_end, pos].float()

            log_probs_tensor[batch_start:batch_end, pos] = lp.cpu()

    log_probs_list = []
    perplexity_list = []
    for i in range(N):
        seq_mask = aa_mask[i].cpu()
        seq_lp = log_probs_tensor[i][seq_mask].numpy().tolist()
        ppl = float(np.exp(-np.mean(seq_lp))) if seq_lp else 0.0
        log_probs_list.append(seq_lp)
        perplexity_list.append(ppl)

    return log_probs_list, perplexity_list


def main():
    parser = argparse.ArgumentParser(description="PureESM2 baseline masked marginal inference")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default=None,
                        help="Output CSV path (default: overwrite data_path)")
    parser.add_argument("--experiment_name", type=str, default="ESM2_baseline")
    parser.add_argument("--model_identifier", type=str, default="esm2_t12_35M_UR50D")
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--sequence_column", type=str, default=None,
                        help="Column with pre-built sequences. If set, skip VH/VL concatenation.")
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    print("=" * 70)
    print("PureESM2 Baseline — Masked Marginal Inference")
    print("=" * 70)

    model, tokenizer = load_model(args.checkpoint, args.model_identifier, args.device)

    data_path = Path(args.data_path)
    if data_path.suffix == ".csv":
        df = pd.read_csv(data_path)
    elif data_path.suffix == ".parquet":
        df = pd.read_parquet(data_path)
    elif data_path.suffix == ".pkl":
        df = pd.read_pickle(data_path)
    else:
        raise ValueError(f"Unsupported file format: {data_path.suffix}")
    print(f"Loaded {len(df)} rows from {data_path}")

    if args.sequence_column and args.sequence_column in df.columns:
        sequences = df[args.sequence_column].tolist()
        print(f"Using pre-built sequences from column '{args.sequence_column}'")
    else:
        sequences = build_sequences(df, tokenizer.cls_token)
    assert len(sequences) == len(df), f"Sequence count mismatch: {len(sequences)} vs {len(df)}"
    print(f"{len(sequences)} sequences ready")

    log_probs_list, perplexity_list = masked_marginal_logprobs(
        model, tokenizer, sequences,
        device=args.device, batch_size=args.batch_size,
    )

    ll_col = f"{args.experiment_name}_LL"
    pp_col = f"{args.experiment_name}_PP"

    output_path = Path(args.output_path) if args.output_path else data_path

    if output_path.suffix == ".csv":
        df[ll_col] = [json.dumps(lp) for lp in log_probs_list]
    else:
        df[ll_col] = log_probs_list
    df[pp_col] = perplexity_list
    if output_path.suffix == ".pkl":
        df.to_pickle(output_path)
    elif output_path.suffix == ".parquet":
        df.to_parquet(output_path, index=False)
    else:
        df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")
    print(f"  Columns added: {ll_col}, {pp_col}")
    print(f"  Mean PPL: {np.mean(perplexity_list):.4f}")
    print(f"  Median PPL: {np.median(perplexity_list):.4f}")


if __name__ == "__main__":
    main()
