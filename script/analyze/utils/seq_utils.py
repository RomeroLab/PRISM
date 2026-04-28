# Importing Packages
import os
import torch
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import transformers
import importlib.util
from pathlib import Path
import re
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch, Rectangle

def get_antiberty_embeddings(
    df: pd.DataFrame,
    seq_col: str = "VH|VL_Sequence",
    WT: str = None,
    batch_size: int = 16,
    compute_raw: bool = False,
    *,
    save_cls_embeddings: bool = False,
    save_mean_embeddings: bool = False,
    save_raw_embeddings: bool = False,
    save_delta_embeddings: bool = False,   # if True, saved embeddings are (emb - WT_emb) (requires WT)
    compute_cosine_to_wt: bool = True,
    compute_rmsd_to_wt: bool = True,
    embeddings_out_dir: str = None,
    embeddings_prefix: str = "antiberty",
    row_id_col: str = "row_id",
):
    """
    AntiBERTy embedding extraction.

    - Cosine similarity to WT computed using ORIGINAL embeddings (requires WT).
    - RMSD to WT computed using ORIGINAL embeddings (requires WT).
    - Optionally save original embeddings.
    - Optionally save delta embeddings (emb - WT_emb) (requires WT).

    WT=None behavior:
      - Metrics to WT are skipped (even if compute_*_to_wt=True).
      - Delta saving is disabled (even if save_delta_embeddings=True).
      - Original embeddings can still be saved.
    """
    import os
    import numpy as np
    import pandas as pd
    import torch
    from tqdm import tqdm
    from antiberty import AntiBERTyRunner

    # -----------------------------
    # helpers
    # -----------------------------
    def _cosine(a, b):
        a = np.asarray(a, np.float32).ravel()
        b = np.asarray(b, np.float32).ravel()
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0.0 or nb == 0.0:
            return np.nan
        return float(np.dot(a, b) / (na * nb))

    def _rmsd(a, b):
        a = np.asarray(a, np.float32).ravel()
        b = np.asarray(b, np.float32).ravel()
        if a.shape != b.shape:
            raise ValueError(f"RMSD shape mismatch {a.shape} vs {b.shape}")
        return float(np.sqrt(np.mean((a - b) ** 2)))

    def _avg2(a, b):
        if np.isnan(a) and np.isnan(b):
            return np.nan
        if np.isnan(a):
            return float(b)
        if np.isnan(b):
            return float(a)
        return float(0.5 * (a + b))

    def _clean(s):
        return s.replace(" ", "").strip().upper() if isinstance(s, str) else ""

    def _split(seq):
        if "|" not in seq:
            return None
        h, l = seq.split("|", 1)
        h, l = h.strip(), l.strip()
        if not h or not l:
            return None
        return h, l

    def _ensure_dir(p):
        if p is None:
            return None
        os.makedirs(p, exist_ok=True)
        return p

    def _save(arr, path):
        np.save(path, np.asarray(arr, np.float32))
        return path

    def _rid(i):
        if row_id_col in df.columns:
            try:
                v = df.at[i, row_id_col]
                if pd.notna(v):
                    return str(int(v))
            except Exception:
                pass
        return str(int(i))

    def _ensure_path_col(df, col):
        if col not in df.columns:
            df[col] = pd.Series([None] * len(df), dtype="object")
        elif not pd.api.types.is_object_dtype(df[col].dtype):
            df[col] = df[col].astype("object")

    # -----------------------------
    # WT handling / validation
    # -----------------------------
    WT_available = WT is not None and isinstance(WT, str) and WT.strip() != ""
    WT_clean = _clean(WT) if WT_available else ""
    if WT_available and "|" not in WT_clean:
        raise ValueError("WT must be provided as 'VH|VL' when not None.")

    if not WT_available:
        compute_cosine_to_wt = False
        compute_rmsd_to_wt = False
        save_delta_embeddings = False  # cannot compute deltas without WT

    if save_raw_embeddings and not compute_raw:
        raise ValueError("save_raw_embeddings=True requires compute_raw=True.")

    save_any = save_cls_embeddings or save_mean_embeddings or save_raw_embeddings
    if save_any and embeddings_out_dir is None:
        raise ValueError("embeddings_out_dir required when saving embeddings.")
    embeddings_out_dir = _ensure_dir(embeddings_out_dir) if save_any else None

    df = df.copy()

    # -----------------------------
    # metric columns (only if WT available and enabled)
    # -----------------------------
    metric_cols = []
    if compute_cosine_to_wt:
        metric_cols += [
            f"{embeddings_prefix}_cls_cosine_to_WT",
            f"{embeddings_prefix}_mean_cosine_to_WT",
        ]
        if compute_raw:
            metric_cols.append(f"{embeddings_prefix}_raw_cosine_to_WT")

    if compute_rmsd_to_wt:
        metric_cols += [
            f"{embeddings_prefix}_cls_rmsd_to_WT",
            f"{embeddings_prefix}_mean_rmsd_to_WT",
        ]
        if compute_raw:
            metric_cols.append(f"{embeddings_prefix}_raw_rmsd_to_WT")

    for c in metric_cols:
        if c not in df.columns:
            df[c] = np.nan

    # path columns
    if embeddings_out_dir is not None:
        if save_cls_embeddings:
            for c in (f"{embeddings_prefix}_H_cls_emb_path", f"{embeddings_prefix}_L_cls_emb_path"):
                _ensure_path_col(df, c)
        if save_mean_embeddings:
            for c in (f"{embeddings_prefix}_H_mean_emb_path", f"{embeddings_prefix}_L_mean_emb_path"):
                _ensure_path_col(df, c)
        if save_raw_embeddings and compute_raw:
            for c in (f"{embeddings_prefix}_H_raw_emb_path", f"{embeddings_prefix}_L_raw_emb_path"):
                _ensure_path_col(df, c)

    # -----------------------------
    # model
    # -----------------------------
    antiberty = AntiBERTyRunner()
    antiberty.model.eval()

    # -----------------------------
    # collect sequences
    # -----------------------------
    valid_idxs, heavy, light = [], [], []
    for i, seq in enumerate(df[seq_col].astype(str).tolist()):
        pair = _split(_clean(seq))
        if pair is None:
            continue
        h, l = pair
        valid_idxs.append(i)
        heavy.append(h)
        light.append(l)

    if not valid_idxs:
        return df

    # -----------------------------
    # embedding helper (batched)
    # -----------------------------
    def _embed(seqs, desc="Embedding"):
        raw, cls, mean = [], [], []
        for start in tqdm(range(0, len(seqs), batch_size), desc=desc):
            batch = seqs[start : start + batch_size]
            with torch.no_grad():
                reps = antiberty.embed(batch)
            for r in reps:
                r_np = r.detach().cpu().numpy().astype(np.float32, copy=False)
                raw.append(r_np)
                cls.append(r_np[0])
                mean.append(r_np.mean(axis=0))
        return raw, cls, mean

    # Variant embeddings (always)
    h_raw, h_cls, h_mean = _embed(heavy, desc="AntiBERTy heavy")
    l_raw, l_cls, l_mean = _embed(light, desc="AntiBERTy light")

    # WT embeddings (only if available)
    if WT_available:
        wt_h, wt_l = WT_clean.split("|", 1)
        wt_h_raw, wt_h_cls, wt_h_mean = _embed([wt_h.strip()], desc="WT heavy")
        wt_l_raw, wt_l_cls, wt_l_mean = _embed([wt_l.strip()], desc="WT light")
        wt_h_raw, wt_h_cls, wt_h_mean = wt_h_raw[0], wt_h_cls[0], wt_h_mean[0]
        wt_l_raw, wt_l_cls, wt_l_mean = wt_l_raw[0], wt_l_cls[0], wt_l_mean[0]
    else:
        wt_h_raw = wt_h_cls = wt_h_mean = None
        wt_l_raw = wt_l_cls = wt_l_mean = None

    # -----------------------------
    # compute metrics + saving
    # -----------------------------
    for k, row in enumerate(valid_idxs):

        # ----- Metrics (only if WT available)
        if WT_available:
            # CLS
            if compute_cosine_to_wt:
                df.at[row, f"{embeddings_prefix}_cls_cosine_to_WT"] = _avg2(
                    _cosine(h_cls[k], wt_h_cls),
                    _cosine(l_cls[k], wt_l_cls),
                )
            if compute_rmsd_to_wt:
                df.at[row, f"{embeddings_prefix}_cls_rmsd_to_WT"] = _avg2(
                    _rmsd(h_cls[k], wt_h_cls),
                    _rmsd(l_cls[k], wt_l_cls),
                )

            # MEAN
            if compute_cosine_to_wt:
                df.at[row, f"{embeddings_prefix}_mean_cosine_to_WT"] = _avg2(
                    _cosine(h_mean[k], wt_h_mean),
                    _cosine(l_mean[k], wt_l_mean),
                )
            if compute_rmsd_to_wt:
                df.at[row, f"{embeddings_prefix}_mean_rmsd_to_WT"] = _avg2(
                    _rmsd(h_mean[k], wt_h_mean),
                    _rmsd(l_mean[k], wt_l_mean),
                )

            # RAW
            if compute_raw:
                Lh = min(h_raw[k].shape[0], wt_h_raw.shape[0])
                Ll = min(l_raw[k].shape[0], wt_l_raw.shape[0])

                if compute_cosine_to_wt:
                    df.at[row, f"{embeddings_prefix}_raw_cosine_to_WT"] = _avg2(
                        _cosine(h_raw[k][:Lh].ravel(), wt_h_raw[:Lh].ravel()),
                        _cosine(l_raw[k][:Ll].ravel(), wt_l_raw[:Ll].ravel()),
                    )
                if compute_rmsd_to_wt:
                    df.at[row, f"{embeddings_prefix}_raw_rmsd_to_WT"] = _avg2(
                        _rmsd(h_raw[k][:Lh].ravel(), wt_h_raw[:Lh].ravel()),
                        _rmsd(l_raw[k][:Ll].ravel(), wt_l_raw[:Ll].ravel()),
                    )

        # ----- Saving embeddings (always possible for ORIGINAL; delta only if WT available)
        if embeddings_out_dir is not None:
            rid = _rid(row)
            suffix = "_deltaWT" if (save_delta_embeddings and WT_available) else ""

            def _maybe_delta(v, wt):
                if save_delta_embeddings and WT_available:
                    return v - wt
                return v

            if save_cls_embeddings:
                df.at[row, f"{embeddings_prefix}_H_cls_emb_path"] = _save(
                    _maybe_delta(h_cls[k], wt_h_cls),
                    os.path.join(embeddings_out_dir, f"{embeddings_prefix}_H_cls{suffix}_row{rid}.npy"),
                )
                df.at[row, f"{embeddings_prefix}_L_cls_emb_path"] = _save(
                    _maybe_delta(l_cls[k], wt_l_cls),
                    os.path.join(embeddings_out_dir, f"{embeddings_prefix}_L_cls{suffix}_row{rid}.npy"),
                )

            if save_mean_embeddings:
                df.at[row, f"{embeddings_prefix}_H_mean_emb_path"] = _save(
                    _maybe_delta(h_mean[k], wt_h_mean),
                    os.path.join(embeddings_out_dir, f"{embeddings_prefix}_H_mean{suffix}_row{rid}.npy"),
                )
                df.at[row, f"{embeddings_prefix}_L_mean_emb_path"] = _save(
                    _maybe_delta(l_mean[k], wt_l_mean),
                    os.path.join(embeddings_out_dir, f"{embeddings_prefix}_L_mean{suffix}_row{rid}.npy"),
                )

            if save_raw_embeddings and compute_raw:
                # if WT not available, save full raw; if WT available and delta requested, save aligned delta
                if WT_available and save_delta_embeddings:
                    Lh = min(h_raw[k].shape[0], wt_h_raw.shape[0])
                    Ll = min(l_raw[k].shape[0], wt_l_raw.shape[0])
                    h_save = h_raw[k][:Lh] - wt_h_raw[:Lh]
                    l_save = l_raw[k][:Ll] - wt_l_raw[:Ll]
                else:
                    h_save = h_raw[k]
                    l_save = l_raw[k]

                df.at[row, f"{embeddings_prefix}_H_raw_emb_path"] = _save(
                    h_save,
                    os.path.join(embeddings_out_dir, f"{embeddings_prefix}_H_raw{suffix}_row{rid}.npy"),
                )
                df.at[row, f"{embeddings_prefix}_L_raw_emb_path"] = _save(
                    l_save,
                    os.path.join(embeddings_out_dir, f"{embeddings_prefix}_L_raw{suffix}_row{rid}.npy"),
                )

    return df




