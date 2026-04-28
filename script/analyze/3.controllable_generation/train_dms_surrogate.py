#!/usr/bin/env python
# coding: utf-8

import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

for _p in Path(__file__).resolve().parents:
    if (_p / "utils" / "__init__.py").exists():
        sys.path.insert(0, str(_p))
        break
from utils import seq_utils as seq_utils

# ============================================================
# Paths / config
# ============================================================
PARENT_SEQ_NAMES = ['g6.31', 'cr9114', 'trastuzumab']
MODEL_NAMES = ["ablang2", "esm2_35m", "esm2_650m", "iglm"]

SEQ_ID_COL = "sequence_id"
SEQ_COL = "VH|VL_Formatted"
HEAVY_COL = "fv_heavy"
LIGHT_COL = "fv_light"

EMB_DIR = Path("./generation/embeddings")
EMB_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# FASTA parsing
# ============================================================
def read_fasta(filepath: str):
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


def split_paired_sequence(seq: str):
    seq = str(seq).strip()
    if "|" not in seq:
        raise ValueError(
            f"Expected paired sequence in format 'heavy|light', but got: {seq[:100]}"
        )

    parts = seq.split("|")
    if len(parts) != 2:
        raise ValueError(
            f"Expected exactly one '|' separator, but got {len(parts) - 1} in: {seq[:100]}"
        )

    heavy, light = parts[0].strip(), parts[1].strip()

    if not heavy:
        raise ValueError("Heavy chain sequence is empty.")
    if not light:
        raise ValueError("Light chain sequence is empty.")

    return heavy, light


# ============================================================
# Embedding helpers
# ============================================================
def run_embedding_model(df, name, fn, common_kwargs, extra_kwargs):
    batch_size = 8
    return fn(
        df=df,
        batch_size=batch_size,
        **common_kwargs,
        **extra_kwargs,
    )


# ============================================================
# Scoring helpers
# ============================================================
def load_mean_pooled_embedding(path, cache):
    if path in cache:
        return cache[path]

    arr = np.load(path, allow_pickle=False)

    if arr.ndim != 2:
        raise ValueError(f"Expected 2D embedding at {path}, got shape {arr.shape}")

    if arr.shape[0] < 1:
        raise ValueError(f"Embedding at {path} has no tokens: shape {arr.shape}")

    mean_vec = np.array(arr.mean(axis=0), dtype=np.float32, copy=True)

    if mean_vec.ndim != 1:
        raise ValueError(
            f"Expected mean-pooled embedding to be 1D, got shape {mean_vec.shape}"
        )

    cache[path] = mean_vec
    return mean_vec


def build_feature_matrix_from_mean_pooling(df, heavy_emb_col, light_emb_col):
    X = []
    emb_cache = {}

    expected_h_dim = None
    expected_l_dim = None

    heavy_paths = df[heavy_emb_col].tolist()
    light_paths = df[light_emb_col].tolist()

    for i, (h_path, l_path) in enumerate(zip(heavy_paths, light_paths)):
        h_mean = load_mean_pooled_embedding(h_path, emb_cache)
        l_mean = load_mean_pooled_embedding(l_path, emb_cache)

        if expected_h_dim is None:
            expected_h_dim = h_mean.shape[0]
            expected_l_dim = l_mean.shape[0]
        else:
            if h_mean.shape[0] != expected_h_dim:
                raise ValueError(
                    f"Row {i}: inconsistent heavy mean embedding dim {h_mean.shape[0]}, "
                    f"expected {expected_h_dim}"
                )
            if l_mean.shape[0] != expected_l_dim:
                raise ValueError(
                    f"Row {i}: inconsistent light mean embedding dim {l_mean.shape[0]}, "
                    f"expected {expected_l_dim}"
                )

        x = np.concatenate([h_mean, l_mean], axis=0)
        X.append(x)

    X = np.stack(X, axis=0).astype(np.float32, copy=False)
    return X



# ============================================================
# Main
# ============================================================
def main(parent_seq_name: str, model_name: str):
    fasta_path = Path(f"./generation/pll_guided_{model_name}_n100_m3_{parent_seq_name}.fasta")
    model_path = f"./binding_affinity_predictors/{parent_seq_name}/final_model_all_splits/ridge_model.joblib"

    if not fasta_path.exists():
        print(f"[WARN] Missing FASTA, skipping: {fasta_path}")
        return

    fasta_stem = fasta_path.stem
    emb_prefix = fasta_stem

    out_csv_with_emb = fasta_path.with_name(f"{fasta_stem}.csv")
    out_csv_scored = fasta_path.with_name(f"{fasta_stem}.csv")

    heavy_emb_col = f"{emb_prefix}_antiberty_H_raw_emb_path"
    light_emb_col = f"{emb_prefix}_antiberty_L_raw_emb_path"

    print(f"\n[INFO] === parent={parent_seq_name} | model={model_name} ===")

    records = read_fasta(str(fasta_path))
    if len(records) == 0:
        raise ValueError(f"No FASTA records found in {fasta_path}")

    print(f"[INFO] Loaded {len(records)} FASTA entries from {fasta_path}")

    rows = []
    wt_seq = None

    for i, (header, paired_seq) in enumerate(records):
        heavy, light = split_paired_sequence(paired_seq)

        if i == 0:
            wt_seq = paired_seq.strip()

        rows.append(
            {
                SEQ_ID_COL: header if header else f"seq_{i:04d}",
                SEQ_COL: paired_seq.strip(),
                HEAVY_COL: heavy,
                LIGHT_COL: light,
                "Mutations": "WT" if i == 0 else f"generated_{i:04d}",
                "is_wt": i == 0,
                "source_fasta": str(fasta_path),
            }
        )

    df = pd.DataFrame(rows)

    if wt_seq is None:
        raise ValueError("Could not determine WT sequence from first FASTA entry.")

    print(f"[INFO] WT sequence identified from first FASTA entry.")
    print(f"[INFO] Dataframe shape: {df.shape}")

    embed_plan = [
        (
            "antiberty",
            "get_antiberty_embeddings",
            dict(embeddings_prefix=f"{emb_prefix}_antiberty"),
        ),
    ]

    save_delta = wt_seq is not None

    common_embedding_kwargs = dict(
        seq_col=SEQ_COL,
        WT=wt_seq,
        compute_raw=True,
        save_cls_embeddings=False,
        save_mean_embeddings=False,
        save_raw_embeddings=True,
        save_delta_embeddings=save_delta,
        compute_cosine_to_wt=False,
        compute_rmsd_to_wt=False,
        embeddings_out_dir=str(EMB_DIR),
    )

    failures = []

    for name, fn_name, extra_kwargs in embed_plan:
        fn = getattr(seq_utils, fn_name, None)
        if fn is None:
            failures.append((name, fn_name, "missing function in seq_utils"))
            print(f"[WARN] Skipping {name}: seq_utils missing '{fn_name}'")
            continue

        print(f"\n[INFO] === {name}: raw embeddings (saving to {EMB_DIR}) ===")
        try:
            df = run_embedding_model(
                df=df,
                name=name,
                fn=fn,
                common_kwargs=common_embedding_kwargs,
                extra_kwargs=extra_kwargs,
            )
            print(f"[OK] {name} completed.")
        except Exception as e:
            failures.append((name, fn_name, repr(e)))
            print(f"[WARN] {name} failed; continuing. Error: {e}")

    df.to_csv(out_csv_with_emb, index=False)
    print(f"\n[INFO] Saved dataframe with embeddings -> {out_csv_with_emb}")

    if failures:
        print("\n[WARN] Some embedding functions failed:")
        for name, fn_name, err in failures:
            print(f"  - {name} ({fn_name}): {err}")
        raise RuntimeError("One or more embedding functions failed; cannot continue to scoring.")

    required_cols = [heavy_emb_col, light_emb_col, HEAVY_COL, LIGHT_COL, SEQ_COL]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Input dataframe is missing required columns for scoring: {missing_cols}\n"
            f"Expected heavy emb col: {heavy_emb_col}\n"
            f"Expected light emb col: {light_emb_col}"
        )

    n_before = len(df)
    df = df.dropna(subset=[heavy_emb_col, light_emb_col]).reset_index(drop=True)
    n_after = len(df)
    print(f"[INFO] Dropped {n_before - n_after} rows with missing embedding paths")
    print(f"[INFO] Remaining rows for scoring: {len(df)}")

    print(f"[INFO] Building mean-pooled feature matrix...")
    t0 = time.time()
    X = build_feature_matrix_from_mean_pooling(df, heavy_emb_col, light_emb_col)
    print(
        f"[INFO] Feature matrix shape: {X.shape} | "
        f"size = {X.nbytes / 1e6:.2f} MB | "
        f"built in {time.time() - t0:.2f} s"
    )

    print(f"[INFO] Loading pretrained model: {model_path}")
    model = joblib.load(model_path)

    print(f"[INFO] Predicting scores...")
    t0 = time.time()
    pred = model.predict(X)
    print(f"[INFO] Prediction completed in {time.time() - t0:.2f} s")

    out_df = df.copy()
    out_df["pred_mean"] = pred
    out_df["rank_desc_pred_mean"] = (
        out_df["pred_mean"].rank(method="first", ascending=False).astype(int)
    )

    if "is_wt" in out_df.columns:
        wt_rows = out_df["is_wt"].fillna(False).astype(bool)
        if wt_rows.sum() == 1:
            wt_pred = float(out_df.loc[wt_rows, "pred_mean"].iloc[0])
            out_df["delta_pred_vs_wt"] = out_df["pred_mean"] - wt_pred
            print(f"[INFO] WT predicted score = {wt_pred:.6f}")

    out_df.to_csv(out_csv_scored, index=False)
    print(f"[INFO] Saved scored CSV -> {out_csv_scored}")
    print("[INFO] Done.")


if __name__ == "__main__":
    for parent_seq_name in PARENT_SEQ_NAMES:
        for model_name in MODEL_NAMES:
            main(parent_seq_name, model_name)