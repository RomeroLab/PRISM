#!/usr/bin/env python
"""
Run the trained PRISM-less linear probe on test set and merge per-residue
probability/prediction columns into the existing test_predictions data files.

Produces columns `baseline_prob_h`, `baseline_prob_l`, `baseline_pred_h`,
`baseline_pred_l` in the target files.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

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


def is_zero_embedding(e: np.ndarray) -> bool:
    return np.allclose(e, 0, atol=1e-8)


@torch.no_grad()
def run_inference(df_embed, probe, embed_prefix, device):
    probs_h, probs_l, preds_h, preds_l = [], [], [], []
    eh = f"{embed_prefix}_h"
    el = f"{embed_prefix}_l"

    for idx in tqdm(range(len(df_embed)), desc="Inference"):
        row = df_embed.iloc[idx]
        embed_h = row[eh]
        embed_l = row[el]

        if is_zero_embedding(embed_h):
            h_len = len(row["HEAVY_CHAIN_AA_SEQUENCE"])
            probs_h.append(np.full(h_len, np.nan))
            preds_h.append(np.full(h_len, -1, dtype=np.int64))
        else:
            t = torch.from_numpy(embed_h.astype(np.float32)).to(device)
            p = torch.softmax(probe(t), dim=-1)[:, 1].cpu().numpy()
            probs_h.append(p)
            preds_h.append((p > 0.5).astype(np.int64))

        if is_zero_embedding(embed_l):
            l_len = len(row["LIGHT_CHAIN_AA_SEQUENCE"])
            probs_l.append(np.full(l_len, np.nan))
            preds_l.append(np.full(l_len, -1, dtype=np.int64))
        else:
            t = torch.from_numpy(embed_l.astype(np.float32)).to(device)
            p = torch.softmax(probe(t), dim=-1)[:, 1].cpu().numpy()
            probs_l.append(p)
            preds_l.append((p > 0.5).astype(np.int64))

    return probs_h, probs_l, preds_h, preds_l


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed_pkl", required=True)
    parser.add_argument("--probe_ckpt", required=True)
    parser.add_argument("--embed_prefix", default="embed_baseline")
    parser.add_argument("--model_name", default="baseline")
    parser.add_argument("--input_dim", type=int, default=480)
    parser.add_argument("--merge_into", nargs="+", default=[])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    logger.info(f"Loading embeddings: {args.embed_pkl}")
    df_embed = pd.read_pickle(args.embed_pkl)
    logger.info(f"Loaded {len(df_embed)} rows")

    logger.info(f"Loading probe checkpoint: {args.probe_ckpt}")
    probe = LinearProbe(input_dim=args.input_dim)
    ckpt = torch.load(args.probe_ckpt, map_location=device, weights_only=False)
    probe.load_state_dict(ckpt["model_state_dict"])
    probe.to(device)
    probe.train(False)
    logger.info(f"Loaded probe (val PR-AUC at save: {ckpt.get('val_prauc', 'N/A')})")

    probs_h, probs_l, preds_h, preds_l = run_inference(
        df_embed, probe, args.embed_prefix, device
    )

    mn = args.model_name
    prob_h_col = f"{mn}_prob_h"
    prob_l_col = f"{mn}_prob_l"
    pred_h_col = f"{mn}_pred_h"
    pred_l_col = f"{mn}_pred_l"

    df_embed[prob_h_col] = probs_h
    df_embed[prob_l_col] = probs_l
    df_embed[pred_h_col] = preds_h
    df_embed[pred_l_col] = preds_l
    df_embed.to_pickle(args.embed_pkl)
    logger.info(f"Updated embed file with {mn} columns: {args.embed_pkl}")

    for target_path in args.merge_into:
        p = Path(target_path)
        if not p.exists():
            logger.warning(f"Target not found, skipping: {p}")
            continue

        logger.info(f"\nMerging into: {p}")
        df_target = pd.read_pickle(p)

        if len(df_target) != len(df_embed):
            logger.warning(
                f"  Row count mismatch: target={len(df_target)}, embed={len(df_embed)}. "
                f"Assuming positional alignment on min(len)."
            )

        n = min(len(df_target), len(df_embed))
        df_target[prob_h_col] = pd.Series(
            [probs_h[i] for i in range(n)] + [None] * (len(df_target) - n)
        )
        df_target[prob_l_col] = pd.Series(
            [probs_l[i] for i in range(n)] + [None] * (len(df_target) - n)
        )
        df_target[pred_h_col] = pd.Series(
            [preds_h[i] for i in range(n)] + [None] * (len(df_target) - n)
        )
        df_target[pred_l_col] = pd.Series(
            [preds_l[i] for i in range(n)] + [None] * (len(df_target) - n)
        )

        df_target.to_pickle(p)
        logger.info(f"  Saved with {mn}_prob_h/l, {mn}_pred_h/l")

    logger.info("\nDone.")


if __name__ == "__main__":
    main()
