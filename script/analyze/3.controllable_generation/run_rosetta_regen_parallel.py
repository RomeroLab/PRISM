#!/usr/bin/env python
"""Parallel Rosetta interface ΔΔG for v44o regen results (3 ab × 3 config × 7 model = 63 fasta).

Input dir: results/controllable_generation/v44o_regen_<TS>/<ab>/n_mut_<N>/*.fasta
Output: per-fasta CSV next to fasta + combined v44o_regen_interface_ddg_results.csv
"""
import csv
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import run_rosetta_interface_ddg as ridg

N_WORKERS = 32

if len(sys.argv) > 1:
    REGEN_DIR = Path(sys.argv[1])
else:
    raise SystemExit(
        "Usage: python run_rosetta_regen_parallel.py <regen_dir>\n"
        "  <regen_dir> is the output directory from pll_guided_sampler.py "
        "containing per-antibody fastas (e.g. results/controllable_generation/<run>)"
    )
OUT_DIR = REGEN_DIR / "interface_ddg"
OUT_DIR.mkdir(exist_ok=True)

_W = {}


def init_worker():
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    import pyrosetta
    pyrosetta.init("-ignore_unrecognized_res -ex1 -ex2 -mute all")
    from pyrosetta.rosetta.core.scoring import ScoreFunctionFactory
    _W["scorefxn"] = ScoreFunctionFactory.create_score_function("ref2015")


def _ensure_pose(ab_key):
    if ab_key in _W:
        return
    from pyrosetta import pose_from_pdb
    pdb_file, h_chain, l_chain, _, vh_len, interface_str = ridg.AB_CONFIG[ab_key]
    wt_pose = pose_from_pdb(str(ridg.PDB_DIR / pdb_file))
    _W[ab_key] = {
        "wt_pose": wt_pose,
        "h_chain": h_chain,
        "l_chain": l_chain,
        "vh_len": vh_len,
        "interface_str": interface_str,
        "chain_map": {
            "H": ridg.build_chain_residue_map(wt_pose, h_chain),
            "L": ridg.build_chain_residue_map(wt_pose, l_chain),
        },
    }


def worker_task(task):
    ab_key, fasta_stem, idx, header = task
    _ensure_pose(ab_key)
    info = _W[ab_key]
    wt_pose = info["wt_pose"]
    chain_map = info["chain_map"]
    vh_len = info["vh_len"]
    interface_str = info["interface_str"]
    scorefxn = _W["scorefxn"]

    rec_id = header.split("|")[0]
    mutations = ridg.parse_mutations_from_header(header, vh_len)
    if not mutations:
        return None

    mut_resnums, mut_aas, applied_strs = [], [], []
    for chain, pos, wt_aa, mut_aa in mutations:
        resnum = chain_map.get(chain, {}).get(pos)
        if resnum is None:
            continue
        pdb_aa = ridg.get_residue_aa(wt_pose, resnum)
        if pdb_aa != wt_aa:
            continue
        mut_resnums.append(resnum)
        mut_aas.append(mut_aa)
        applied_strs.append(f"{chain}:{wt_aa}{pos}{mut_aa}")

    if not mut_resnums:
        return {"status": "skipped", "fasta_stem": fasta_stem, "ab_key": ab_key, "id": rec_id}

    mutation_str = ",".join(applied_strs)
    try:
        result = ridg.compute_interface_ddg(wt_pose, scorefxn, mut_resnums, mut_aas, interface_str)
        record = {
            "id": rec_id, "source": fasta_stem, "antibody": ab_key,
            "n_mut": len(mut_resnums), "mutations": mutation_str,
            "interface_ddg": round(result["interface_ddg"], 3),
            "wt_dg_bind": round(result["wt_dg_bind"], 3),
            "mut_dg_bind": round(result["mut_dg_bind"], 3),
        }
        return {"status": "ok", "fasta_stem": fasta_stem, "ab_key": ab_key, "record": record}
    except Exception as e:
        record = {
            "id": rec_id, "source": fasta_stem, "antibody": ab_key,
            "n_mut": len(mut_resnums), "mutations": mutation_str,
            "interface_ddg": float("nan"), "wt_dg_bind": float("nan"), "mut_dg_bind": float("nan"),
        }
        return {"status": "error", "fasta_stem": fasta_stem, "ab_key": ab_key, "error": str(e), "record": record}


def make_tasks():
    tasks = []
    fasta_to_outdir = {}  # for per-fasta CSV save
    for ab_key in ridg.AB_CONFIG:
        ab_dir = REGEN_DIR / ab_key
        if not ab_dir.exists():
            continue
        for n_mut_dir in sorted(ab_dir.glob("n_mut_*")):
            for fasta_path in sorted(n_mut_dir.glob("*.fasta")):
                fasta_stem = fasta_path.stem
                fasta_to_outdir[fasta_stem] = n_mut_dir
                idx = -1
                with open(fasta_path) as f:
                    for line in f:
                        if line.startswith(">"):
                            idx += 1
                            if idx == 0:
                                continue
                            tasks.append((ab_key, fasta_stem, idx, line.strip()[1:]))
    return tasks, fasta_to_outdir


def main():
    tasks, fasta_to_outdir = make_tasks()
    print(f"Total tasks: {len(tasks)}  |  workers: {N_WORKERS}", flush=True)

    by_fasta = defaultdict(list)
    skipped = defaultdict(int)
    errors = defaultdict(int)
    fasta_ab = {}

    t0 = time.time()
    n_done = 0
    with ProcessPoolExecutor(max_workers=N_WORKERS, initializer=init_worker) as ex:
        for r in ex.map(worker_task, tasks, chunksize=4):
            n_done += 1
            if r is None:
                continue
            fasta_ab[r["fasta_stem"]] = r["ab_key"]
            if r["status"] == "skipped":
                skipped[r["fasta_stem"]] += 1
            elif r["status"] == "ok":
                by_fasta[r["fasta_stem"]].append(r["record"])
            elif r["status"] == "error":
                by_fasta[r["fasta_stem"]].append(r["record"])
                errors[r["fasta_stem"]] += 1
            if n_done % 100 == 0:
                el = time.time() - t0
                rate = n_done / el
                eta = (len(tasks) - n_done) / rate / 60
                print(f"  [{n_done}/{len(tasks)}] elapsed={el/60:.1f}min  rate={rate:.2f}/s  ETA={eta:.1f}min", flush=True)

    # Save per-fasta CSV next to fasta
    all_recs = []
    for fasta_stem, records in by_fasta.items():
        out_csv = fasta_to_outdir[fasta_stem] / f"{fasta_stem}_interface_ddg.csv"
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            w.writeheader()
            w.writerows(records)
        all_recs.extend(records)

    # Combined
    if all_recs:
        import pandas as pd
        df = pd.DataFrame(all_recs)
        out = REGEN_DIR / "v44o_regen_interface_ddg_results.csv"
        df.to_csv(out, index=False)
        print(f"\nCombined: {out}  rows={len(df)}")

        valid = df.dropna(subset=["interface_ddg"])
        for ab in sorted(valid["antibody"].unique()):
            ab_df = valid[valid["antibody"] == ab]
            print(f"\n=== {ab} ===")
            stat = ab_df.groupby("source").agg(
                n=("interface_ddg", "count"),
                mean=("interface_ddg", "mean"),
                median=("interface_ddg", "median"),
                pct_improved=("interface_ddg", lambda x: (x < 0).mean() * 100),
                best=("interface_ddg", "min"),
            ).round(3).sort_values("median")
            print(stat.to_string())

    total = time.time() - t0
    print(f"\nTotal wall time: {total/3600:.2f}h ({total/60:.1f}min)", flush=True)


if __name__ == "__main__":
    main()
