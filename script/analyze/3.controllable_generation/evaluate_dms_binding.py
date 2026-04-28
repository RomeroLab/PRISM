#!/usr/bin/env python
"""
Evaluate PLL-guided sampler outputs with DMS binding affinity prediction.

Pipeline: FASTA → AntiBERTy embeddings → Ridge model → binding affinity scores

Uses the in-repo seq_utils (script/analyze/utils/) and pre-trained Ridge models
(script/analyze/3.controllable_generation/dms_surrogates/).
"""
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Add script/analyze/ to path so `from utils import seq_utils` resolves
for _p in Path(__file__).resolve().parents:
    if (_p / "utils" / "__init__.py").exists():
        sys.path.insert(0, str(_p))
        break
from utils import seq_utils

# ============================================================
# Config
# ============================================================
PLL_RESULTS_DIR = Path(__file__).parent / "pll_results"
EMB_DIR = PLL_RESULTS_DIR / "embeddings"
EMB_DIR.mkdir(parents=True, exist_ok=True)

# Antibody name mapping: our dir name → model dir name
AB_MAP = {
    "trast": "trastuzumab",
    "cr9114": "cr9114",
    "g631": "g6.31",
}

MODEL_DIR = Path(__file__).resolve().parent / "dms_surrogates"

# Map antibody name → in-repo Ridge model filename (cr9114_ridge.joblib, etc.)
RIDGE_FILENAME = {
    "cr9114": "cr9114_ridge.joblib",
    "g6.31": "g631_ridge.joblib",
    "trastuzumab": "trastuzumab_ridge.joblib",
}

SEQ_ID_COL = "sequence_id"
SEQ_COL = "VH|VL_Formatted"
HEAVY_COL = "fv_heavy"
LIGHT_COL = "fv_light"


# ============================================================
# Helpers (same as scoring_w_IgLM)
# ============================================================
def read_fasta(filepath):
    records = []
    header = None
    seq_chunks = []
    with open(filepath, "r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(seq_chunks)))
                header = line[1:].strip()
                seq_chunks = []
            else:
                seq_chunks.append(line)
    if header is not None:
        records.append((header, "".join(seq_chunks)))
    return records


def split_paired_sequence(seq):
    seq = str(seq).strip()
    if "|" not in seq:
        raise ValueError(f"Expected 'heavy|light' format, got: {seq[:80]}")
    parts = seq.split("|")
    if len(parts) != 2:
        raise ValueError(f"Expected exactly one '|', got {len(parts)-1}")
    return parts[0].strip(), parts[1].strip()


def load_mean_pooled_embedding(path, cache):
    if path in cache:
        return cache[path]
    arr = np.load(path, allow_pickle=False)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D at {path}, got {arr.shape}")
    mean_vec = np.array(arr.mean(axis=0), dtype=np.float32, copy=True)
    cache[path] = mean_vec
    return mean_vec


def build_feature_matrix(df, heavy_emb_col, light_emb_col):
    X = []
    cache = {}
    for _, row in df.iterrows():
        h = load_mean_pooled_embedding(row[heavy_emb_col], cache)
        l = load_mean_pooled_embedding(row[light_emb_col], cache)
        X.append(np.concatenate([h, l]))
    return np.stack(X).astype(np.float32)


# ============================================================
# Process one FASTA file
# ============================================================
def process_fasta(fasta_path: Path, ab_key: str):
    ab_name = AB_MAP[ab_key]
    model_path = MODEL_DIR / RIDGE_FILENAME[ab_name]
    fasta_stem = fasta_path.stem

    print(f"\n{'='*60}")
    print(f"Processing: {fasta_path.name} ({ab_name})")
    print(f"{'='*60}")

    # --- Read FASTA ---
    records = read_fasta(str(fasta_path))
    if not records:
        print(f"  [WARN] Empty FASTA, skipping.")
        return None
    print(f"  Loaded {len(records)} sequences")

    # --- Build DataFrame ---
    rows = []
    wt_seq = None
    for i, (header, paired_seq) in enumerate(records):
        heavy, light = split_paired_sequence(paired_seq)
        if i == 0:
            wt_seq = paired_seq.strip()
        rows.append({
            SEQ_ID_COL: header or f"seq_{i:04d}",
            SEQ_COL: paired_seq.strip(),
            HEAVY_COL: heavy,
            LIGHT_COL: light,
            "is_wt": i == 0,
            "source": fasta_stem,
            "antibody": ab_name,
        })
    df = pd.DataFrame(rows)

    # --- AntiBERTy embeddings ---
    emb_prefix = fasta_stem + "_antiberty"
    heavy_emb_col = f"{emb_prefix}_H_raw_emb_path"
    light_emb_col = f"{emb_prefix}_L_raw_emb_path"

    print(f"  Running AntiBERTy embeddings...")
    t0 = time.time()
    df = seq_utils.get_antiberty_embeddings(
        df=df,
        seq_col=SEQ_COL,
        WT=wt_seq,
        batch_size=8,
        compute_raw=True,
        save_cls_embeddings=False,
        save_mean_embeddings=False,
        save_raw_embeddings=True,
        save_delta_embeddings=False,
        compute_cosine_to_wt=False,
        compute_rmsd_to_wt=False,
        embeddings_out_dir=str(EMB_DIR),
        embeddings_prefix=emb_prefix,
    )
    print(f"  Embeddings done in {time.time()-t0:.1f}s")

    # --- Build features & predict ---
    df = df.dropna(subset=[heavy_emb_col, light_emb_col]).reset_index(drop=True)
    X = build_feature_matrix(df, heavy_emb_col, light_emb_col)
    print(f"  Feature matrix: {X.shape}")

    # Load pre-trained Ridge model (trained on DMS data, saved via joblib)
    model = joblib.load(model_path)
    pred = model.predict(X)
    df["pred_mean"] = pred
    df["rank_desc_pred_mean"] = df["pred_mean"].rank(method="first", ascending=False).astype(int)

    wt_rows = df["is_wt"].fillna(False).astype(bool)
    if wt_rows.sum() == 1:
        wt_pred = float(df.loc[wt_rows, "pred_mean"].iloc[0])
        df["delta_pred_vs_wt"] = df["pred_mean"] - wt_pred
        print(f"  WT predicted score = {wt_pred:.6f}")

    # --- Save CSV ---
    out_csv = fasta_path.with_suffix(".csv")
    df.to_csv(out_csv, index=False)
    print(f"  Saved: {out_csv.name} ({len(df)} rows)")
    return df


# ============================================================
# Main
# ============================================================
def main():
    all_results = []

    for ab_key in ["trast", "cr9114", "g631"]:
        ab_dir = PLL_RESULTS_DIR / ab_key
        fasta_files = sorted(ab_dir.glob("*.fasta"))
        print(f"\n{'#'*60}")
        print(f"# Antibody: {AB_MAP[ab_key]} ({len(fasta_files)} FASTA files)")
        print(f"{'#'*60}")

        for fasta_path in fasta_files:
            df = process_fasta(fasta_path, ab_key)
            if df is not None:
                all_results.append(df)

    # --- Summary ---
    if all_results:
        summary = pd.concat(all_results, ignore_index=True)
        summary_path = PLL_RESULTS_DIR / "all_binding_predictions.csv"
        summary.to_csv(summary_path, index=False)
        print(f"\n{'='*60}")
        print(f"ALL DONE. Combined CSV: {summary_path}")
        print(f"Total sequences scored: {len(summary)}")
        print(f"{'='*60}")

        # Quick stats per source
        stats = summary.groupby(["antibody", "source"]).agg(
            n=("pred_mean", "count"),
            mean_pred=("pred_mean", "mean"),
            mean_delta=("delta_pred_vs_wt", "mean"),
        ).round(4)
        print(f"\n{stats.to_string()}")


if __name__ == "__main__":
    main()
