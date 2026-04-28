#!/usr/bin/env python
"""
Single-panel UMAP of PRISM-less residue embeddings colored by GL/NGL.
Mean-centers per AA identity to remove amino acid bias.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import KNeighborsClassifier

logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)


def parse_mut_positions(s):
    if pd.isna(s) or s == "" or str(s) == "nan":
        return []
    out = []
    for m in str(s).split(";"):
        m = m.strip()
        if not m:
            continue
        pos = "".join(c for c in m[1:-1] if c.isdigit())
        if pos:
            out.append(int(pos))
    return out


def collect(df, embed_prefix, n_samples, rng):
    vectors, labels, residues = [], [], []
    h_col, l_col = f"{embed_prefix}_h", f"{embed_prefix}_l"
    for idx in range(len(df)):
        row = df.iloc[idx]
        for seq_col, embed_col, mut_col in [
            ("HEAVY_CHAIN_AA_SEQUENCE", h_col, "hc_mut_codes"),
            ("LIGHT_CHAIN_AA_SEQUENCE", l_col, "lc_mut_codes"),
        ]:
            seq = row[seq_col]
            emb = np.asarray(row[embed_col])
            muts = parse_mut_positions(row[mut_col])
            ngl_set = set(p - 1 for p in muts if 1 <= p <= len(seq))
            for i, aa in enumerate(seq):
                if i >= emb.shape[0]:
                    break
                vectors.append(emb[i].astype(np.float32))
                labels.append(1 if i in ngl_set else 0)
                residues.append(aa)

    vectors = np.asarray(vectors, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    residues = np.asarray(residues)

    gl_idx = np.where(labels == 0)[0]
    ngl_idx = np.where(labels == 1)[0]
    n = min(len(gl_idx), len(ngl_idx), n_samples // 2)
    gl_pick = rng.choice(gl_idx, size=n, replace=False)
    ngl_pick = rng.choice(ngl_idx, size=n, replace=False)
    keep = np.concatenate([gl_pick, ngl_pick])
    rng.shuffle(keep)
    return vectors[keep], labels[keep], residues[keep], n, n


def mean_center(vectors, residues):
    out = vectors.copy()
    for aa in np.unique(residues):
        mask = residues == aa
        out[mask] = out[mask] - out[mask].mean(axis=0, keepdims=True)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default="data/unpaired_OAS/linear_probe_data/test_linear_baseline.pkl")
    parser.add_argument("--embed_prefix", default="embed_baseline")
    parser.add_argument("--title", default="PRISM-less\n(Pure ESM2 finetune)")
    parser.add_argument("--output_dir", default="img/2.gl-ngl_calculation")
    parser.add_argument("--output_name", default="umap_gl_ngl_baseline")
    parser.add_argument("--n_samples", type=int, default=5000)
    parser.add_argument("--n_neighbors", type=int, default=30)
    parser.add_argument("--min_dist", type=float, default=0.3)
    parser.add_argument("--knn_k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_jobs", type=int, default=-1)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading: {args.input_file}")
    df = pd.read_pickle(args.input_file)
    logger.info(f"Rows: {len(df)}")

    logger.info("Collecting residues...")
    vectors, labels, residues, gl, ngl = collect(df, args.embed_prefix, args.n_samples, rng)
    logger.info(f"GL={gl}, NGL={ngl}, dim={vectors.shape[1]}")

    logger.info("Mean-centering by AA identity...")
    Xc = mean_center(vectors, residues)

    logger.info(f"Running UMAP (n_neighbors={args.n_neighbors}, min_dist={args.min_dist})...")
    reducer = umap.UMAP(
        n_neighbors=args.n_neighbors, min_dist=args.min_dist,
        metric="cosine", n_components=2, random_state=args.seed, n_jobs=args.n_jobs,
    )
    Xu = reducer.fit_transform(Xc)

    km = KMeans(n_clusters=2, n_init=10, random_state=args.seed).fit(Xu)
    ari = adjusted_rand_score(labels, km.labels_)
    knn = KNeighborsClassifier(n_neighbors=args.knn_k).fit(Xu, labels)
    knn_acc = knn.score(Xu, labels)
    logger.info(f"ARI={ari:.4f}, {args.knn_k}-NN Acc={knn_acc:.4f}")

    gl_color, ngl_color = "#1CC454", "#C8327D"
    fig, ax = plt.subplots(figsize=(7, 7), dpi=150)
    gl_mask = labels == 0
    ax.scatter(Xu[gl_mask, 0], Xu[gl_mask, 1], c=gl_color, s=15, alpha=0.6,
               edgecolor="none")
    ax.scatter(Xu[~gl_mask, 0], Xu[~gl_mask, 1], c=ngl_color, s=15, alpha=0.6,
               edgecolor="none")
    ax.set_title(f"{args.title}\nARI: {ari:.3f} | {args.knn_k}-NN: {knn_acc:.3f}",
                 fontsize=18, fontweight="bold")
    ax.set_xlabel("UMAP 1", fontsize=16, fontweight="bold")
    ax.set_ylabel("UMAP 2", fontsize=16, fontweight="bold")
    ax.tick_params(axis="both", labelsize=13)
    for s in ax.spines.values():
        s.set_linewidth(2)
    plt.tight_layout()

    png = out_dir / f"{args.output_name}.png"
    svg = out_dir / f"{args.output_name}.svg"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(svg, format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved: {png}")
    logger.info(f"Saved: {svg}")


if __name__ == "__main__":
    main()
