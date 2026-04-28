#!/usr/bin/env python
"""
One-shot: extract Pure ESM2 (PRISM-less) residue embeddings for sequences in a
target data file, run the trained linear probe, and write baseline_prob_h/l,
baseline_pred_h/l columns back into the same file.
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.serialization
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer

torch.serialization.add_safe_globals([pathlib.PosixPath])

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class LinearProbe(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 2):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)


def load_baseline_backbone(checkpoint_path, model_identifier, device):
    logger.info(f"Loading ESM2 backbone from: {checkpoint_path}")
    config = AutoConfig.from_pretrained(f"facebook/{model_identifier}")
    model = AutoModel.from_config(config)
    tokenizer = AutoTokenizer.from_pretrained(f"facebook/{model_identifier}")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]
    cleaned = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith("model."):
            nk = nk[len("model."):]
        if nk.startswith("esm."):
            nk = nk[len("esm."):]
        if nk.startswith("lm_head.") or nk.startswith("contact_head."):
            continue
        cleaned[nk] = v
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        logger.warning(f"Missing (expected pooler/contact): {missing[:3]}...")
    if unexpected:
        logger.warning(f"Unexpected: {unexpected[:3]}...")

    model.train(False)
    model.to(device)
    del ckpt, state_dict, cleaned
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return model, tokenizer


@torch.no_grad()
def score_sequences(
    sequences,
    backbone,
    tokenizer,
    probe,
    device,
    batch_size=128,
    max_length=1024,
    desc="Scoring",
):
    probs_all, preds_all = [], []
    for i in tqdm(range(0, len(sequences), batch_size), desc=desc):
        batch = sequences[i : i + batch_size]
        enc = tokenizer(
            batch, return_tensors="pt", padding=True,
            truncation=True, max_length=max_length,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        hidden = backbone(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state

        logits = probe(hidden)
        probs = torch.softmax(logits, dim=-1)[..., 1]

        for j, seq in enumerate(batch):
            L = len(seq)
            p = probs[j, 1 : L + 1].cpu().numpy().astype(np.float32)
            if p.shape[0] != L:
                if p.shape[0] < L:
                    p = np.concatenate([p, np.zeros(L - p.shape[0], dtype=np.float32)])
                else:
                    p = p[:L]
            probs_all.append(p)
            preds_all.append((p > 0.5).astype(np.int64))

    return probs_all, preds_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, nargs="+",
                        help="Target data files to add baseline_* columns to")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--probe_ckpt", required=True)
    parser.add_argument("--model_identifier", default="esm2_t12_35M_UR50D")
    parser.add_argument("--model_name", default="baseline")
    parser.add_argument("--input_dim", type=int, default=480)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    backbone, tokenizer = load_baseline_backbone(
        args.checkpoint, args.model_identifier, device
    )

    probe = LinearProbe(input_dim=args.input_dim)
    probe_ckpt = torch.load(args.probe_ckpt, map_location=device, weights_only=False)
    probe.load_state_dict(probe_ckpt["model_state_dict"])
    probe.to(device)
    probe.train(False)
    logger.info(f"Probe loaded (val PR-AUC: {probe_ckpt.get('val_prauc', 'N/A')})")

    mn = args.model_name
    ph_col, pl_col = f"{mn}_prob_h", f"{mn}_prob_l"
    dh_col, dl_col = f"{mn}_pred_h", f"{mn}_pred_l"

    for target_path in args.target:
        p = Path(target_path)
        if not p.exists():
            logger.warning(f"Skipping missing: {p}")
            continue

        logger.info(f"\n=== Target: {p} ===")
        df = pd.read_pickle(p)
        logger.info(f"Rows: {len(df)}")

        heavy = df["HEAVY_CHAIN_AA_SEQUENCE"].tolist()
        light = df["LIGHT_CHAIN_AA_SEQUENCE"].tolist()

        probs_h, preds_h = score_sequences(
            heavy, backbone, tokenizer, probe, device,
            batch_size=args.batch_size, desc=f"{p.stem}-H",
        )
        probs_l, preds_l = score_sequences(
            light, backbone, tokenizer, probe, device,
            batch_size=args.batch_size, desc=f"{p.stem}-L",
        )

        df[ph_col] = probs_h
        df[pl_col] = probs_l
        df[dh_col] = preds_h
        df[dl_col] = preds_l

        df.to_pickle(p)
        logger.info(f"Saved with {ph_col}, {pl_col}, {dh_col}, {dl_col}")

    logger.info("\nAll targets done.")


if __name__ == "__main__":
    main()
