#!/usr/bin/env python
"""Parallel Rosetta stability ΔΔG (single-protein) for v44o regen results.

Reuses run_rosetta_ddg's compute_ddg_repack via mutation-level parallelism.
"""
import csv
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import run_rosetta_ddg as rdg

N_WORKERS = 32

if len(sys.argv) > 1:
    REGEN_DIR = Path(sys.argv[1])
else:
    raise SystemExit(
        "Usage: python run_rosetta_stability_regen_parallel.py <regen_dir>"
    )
OUT_DIR = REGEN_DIR / "stability_ddg"
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
    pdb_file, h_chain, l_chain, vh_len = rdg.AB_CONFIG[ab_key]
    # Use cleaned PDBs (ligand-stripped) from interface_ddg dir; raw PDBs in
    # data/structural_pdb/ contain NAG/EDO ligands that lack CA atoms and crash repack.
    cleaned_dir = Path(__file__).resolve().parent / "pll_results" / "ddg_results"
    cleaned_pdb = cleaned_dir / pdb_file.replace(".pdb", "_clean.pdb")
    pdb_path = cleaned_pdb if cleaned_pdb.exists() else (rdg.PDB_DIR / pdb_file)
    wt_pose = pose_from_pdb(str(pdb_path))
    _W[ab_key] = {
        "wt_pose": wt_pose,
        "h_chain": h_chain,
        "l_chain": l_chain,
        "vh_len": vh_len,
        "chain_map": {
            "H": rdg.build_chain_residue_map(wt_pose, h_chain),
            "L": rdg.build_chain_residue_map(wt_pose, l_chain),
        },
    }


def worker_task(task):
    ab_key, fasta_stem, idx, header = task
    _ensure_pose(ab_key)
    info = _W[ab_key]
    wt_pose = info["wt_pose"]
    chain_map = info["chain_map"]
    vh_len = info["vh_len"]
    scorefxn = _W["scorefxn"]

    rec_id = header.split("|")[0]
    mutations = rdg.parse_mutations_from_header(header, vh_len)
    if not mutations:
        return None

    mut_resnums, mut_aas, applied_strs = [], [], []
    skip_this = False
    for chain, pos, wt_aa, mut_aa in mutations:
        resnum = chain_map.get(chain, {}).get(pos)
        if resnum is None:
            skip_this = True
            break
        pdb_aa = rdg.get_residue_aa(wt_pose, resnum)
        if pdb_aa != wt_aa:
            continue
        mut_resnums.append(resnum)
        mut_aas.append(mut_aa)
        applied_strs.append(f"{chain}:{wt_aa}{pos}{mut_aa}")

    if skip_this or not mut_resnums:
        return {"status": "skipped", "fasta_stem": fasta_stem, "ab_key": ab_key, "id": rec_id}

    mutation_str = ",".join(applied_strs)
    try:
        result = rdg.compute_ddg_repack(wt_pose, scorefxn, mut_resnums, mut_aas)
        record = {
            "id": rec_id, "source": fasta_stem, "antibody": ab_key,
            "n_mut": len(mut_resnums), "mutations": mutation_str,
            "ddg": round(result["ddg"], 3),
            "wt_energy": round(result["wt_energy"], 3),
            "mut_energy": round(result["mut_energy"], 3),
        }
        return {"status": "ok", "fasta_stem": fasta_stem, "ab_key": ab_key, "record": record}
    except Exception as e:
        record = {
            "id": rec_id, "source": fasta_stem, "antibody": ab_key,
            "n_mut": len(mut_resnums), "mutations": mutation_str,
            "ddg": float("nan"), "wt_energy": float("nan"), "mut_energy": float("nan"),
        }
        return {"status": "error", "fasta_stem": fasta_stem, "ab_key": ab_key, "error": str(e), "record": record}


def make_tasks():
    tasks = []
    fasta_to_outdir = {}
    for ab_key in rdg.AB_CONFIG:
        ab_dir = REGEN_DIR / ab_key
        if not ab_dir.exists():
            continue
        for n_mut_dir in sorted(ab_dir.glob("n_mut_*")):
            for fasta_path in sorted(n_mut_dir.glob("*.fasta")):
                # Skip eval CSVs (only fastas)
                if not fasta_path.name.endswith(".fasta"):
                    continue
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

    t0 = time.time()
    n_done = 0
    with ProcessPoolExecutor(max_workers=N_WORKERS, initializer=init_worker) as ex:
        for r in ex.map(worker_task, tasks, chunksize=4):
            n_done += 1
            if r is None:
                continue
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

    all_recs = []
    for fasta_stem, records in by_fasta.items():
        out_csv = fasta_to_outdir[fasta_stem] / f"{fasta_stem}_stability_ddg.csv"
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            w.writeheader()
            w.writerows(records)
        all_recs.extend(records)

    if all_recs:
        import pandas as pd
        df = pd.DataFrame(all_recs)
        out = REGEN_DIR / "v44o_regen_stability_ddg_results.csv"
        df.to_csv(out, index=False)
        print(f"\nCombined: {out}  rows={len(df)}")

    total = time.time() - t0
    print(f"\nTotal wall time: {total/3600:.2f}h ({total/60:.1f}min)", flush=True)


if __name__ == "__main__":
    main()
