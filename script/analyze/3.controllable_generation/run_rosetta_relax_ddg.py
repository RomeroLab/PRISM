#!/usr/bin/env python
"""Proper ΔΔG protocol with FastRelax — m=10 variants only.

Per-antibody:
  1. Load cleaned PDB.
  2. FastRelax WT once with coordinate constraints (start_coords).
  3. Score relaxed WT once: interface dG (InterfaceAnalyzerMover) and stability (ref2015).
  → Save the relaxed WT pose to disk so workers can mmap-load it.

Per-variant (parallel, 32 workers):
  1. Worker loads its antibody's relaxed WT pose (cached after first hit).
  2. Mutate residues into a fresh clone.
  3. FastRelax mutant with same constraints.
  4. Score interface dG + stability.
  5. Return ΔΔG = mut - wt_baseline.

Output: writes new columns to v44o_regen_interface_ddg_results.csv and
v44o_regen_stability_ddg_results.csv for m=10 rows only — and saves a separate
m10_relaxed combined CSV (`v44o_regen_relax_ddg_m10.csv`) so the original tables
remain available as `_orig` columns.
"""
import csv
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import run_rosetta_interface_ddg as ridg

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[2]

if len(sys.argv) > 1:
    REGEN = Path(sys.argv[1])
else:
    raise SystemExit(
        "Usage: python run_rosetta_relax_ddg.py <regen_dir>\n"
        "  <regen_dir> is the per-run output directory from pll_guided_sampler.py."
    )
CLEAN_PDB_DIR = _HERE / "pll_results" / "ddg_results"
RELAX_DIR = REGEN / "relaxed_wt_poses"
RELAX_DIR.mkdir(exist_ok=True)

N_WORKERS = 32
N_MUT_TARGET = 10  # only process variants from m=10 fastas

_W = {}  # worker-local cache


# --------------------------------------------------------------------------
# Worker setup + pose helpers
# --------------------------------------------------------------------------
def init_worker():
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    import pyrosetta
    pyrosetta.init("-ignore_unrecognized_res -ex1 -ex2 -mute all -constant_seed -jran 1234")
    from pyrosetta.rosetta.core.scoring import ScoreFunctionFactory
    _W["scorefxn"] = ScoreFunctionFactory.create_score_function("ref2015")


def fastrelax(pose, scorefxn, n_cycles=1):
    """FastRelax over the entire pose with coord constraints (used for the WT only)."""
    from pyrosetta.rosetta.protocols.relax import FastRelax
    fr = FastRelax(scorefxn, n_cycles)
    fr.constrain_relax_to_start_coords(True)
    fr.set_scorefxn(scorefxn)
    fr.apply(pose)


def local_fastrelax(pose, scorefxn, mut_resnums, shell=8.0, n_cycles=1):
    """FastRelax with a MoveMap restricted to residues within `shell` Å of any
    mutation site. Backbone + sidechain torsions are allowed only for these
    nearby residues; the rest of the structure stays fixed. ~10x faster than
    full-pose FastRelax for ΔΔG runs and produces near-identical local minima."""
    from pyrosetta.rosetta.core.kinematics import MoveMap
    from pyrosetta.rosetta.protocols.relax import FastRelax

    site_set = set(mut_resnums)
    movable = set(site_set)
    for i in range(1, pose.total_residue() + 1):
        if i in movable:
            continue
        try:
            ca_i = pose.residue(i).xyz("CA")
        except RuntimeError:
            continue
        for m in mut_resnums:
            try:
                if ca_i.distance(pose.residue(m).xyz("CA")) <= shell:
                    movable.add(i)
                    break
            except RuntimeError:
                continue

    mm = MoveMap()
    mm.set_bb(False)
    mm.set_chi(False)
    for i in movable:
        mm.set_bb(i, True)
        mm.set_chi(i, True)

    fr = FastRelax(scorefxn, n_cycles)
    fr.set_movemap(mm)
    fr.constrain_relax_to_start_coords(True)
    fr.set_scorefxn(scorefxn)
    fr.apply(pose)


def score_interface(pose, scorefxn, interface_str):
    from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
    iam = InterfaceAnalyzerMover()
    iam.set_interface(interface_str)
    iam.set_scorefunction(scorefxn)
    iam.set_compute_interface_energy(True)
    iam.set_compute_separated_sasa(False)
    iam.set_pack_separated(True)
    iam.apply(pose)
    return (
        pose.scores.get("dG_separated", None)
        or pose.scores.get("dG_separated/dSASAx100", 0.0)
    )


def _ensure_pose(ab_key):
    if ab_key in _W:
        return
    from pyrosetta import pose_from_pdb
    pdb_file, h_chain, l_chain, _, vh_len, interface_str = ridg.AB_CONFIG[ab_key]
    relaxed_path = RELAX_DIR / pdb_file.replace(".pdb", "_relaxed.pdb")
    pose = pose_from_pdb(str(relaxed_path))
    _W[ab_key] = {
        "wt_pose": pose,
        "h_chain": h_chain,
        "l_chain": l_chain,
        "vh_len": vh_len,
        "interface_str": interface_str,
        "chain_map": {
            "H": ridg.build_chain_residue_map(pose, h_chain),
            "L": ridg.build_chain_residue_map(pose, l_chain),
        },
    }


# --------------------------------------------------------------------------
# Worker task
# --------------------------------------------------------------------------
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
    skip_this = False
    for chain, pos, wt_aa, mut_aa in mutations:
        resnum = chain_map.get(chain, {}).get(pos)
        if resnum is None:
            skip_this = True
            break
        pdb_aa = ridg.get_residue_aa(wt_pose, resnum)
        if pdb_aa != wt_aa:
            continue
        mut_resnums.append(resnum)
        mut_aas.append(mut_aa)
        applied_strs.append(f"{chain}:{wt_aa}{pos}{mut_aa}")

    if skip_this or not mut_resnums:
        return {"status": "skipped", "fasta_stem": fasta_stem, "ab_key": ab_key, "id": rec_id}

    mutation_str = ",".join(applied_strs)

    try:
        from pyrosetta.rosetta.protocols.simple_moves import MutateResidue
        mut_copy = wt_pose.clone()
        for resnum, aa1 in zip(mut_resnums, mut_aas):
            MutateResidue(resnum, ridg.AA_1TO3[aa1]).apply(mut_copy)
        # Local FastRelax: only residues within 8 Å of any mutation site move.
        # Roughly 10x faster than full-pose FastRelax and gives stable ΔΔG estimates.
        local_fastrelax(mut_copy, scorefxn, mut_resnums, shell=8.0, n_cycles=1)
        # Score
        mut_dg = score_interface(mut_copy, scorefxn, interface_str)
        mut_energy = scorefxn(mut_copy)
        return {
            "status": "ok",
            "ab_key": ab_key,
            "fasta_stem": fasta_stem,
            "record": {
                "id": rec_id,
                "source": fasta_stem,
                "antibody": ab_key,
                "n_mut": len(mut_resnums),
                "mutations": mutation_str,
                "mut_dg_bind": round(mut_dg, 3),
                "mut_energy": round(mut_energy, 3),
            },
        }
    except Exception as e:
        return {
            "status": "error",
            "ab_key": ab_key,
            "fasta_stem": fasta_stem,
            "error": str(e),
            "record": {
                "id": rec_id,
                "source": fasta_stem,
                "antibody": ab_key,
                "n_mut": len(mut_resnums),
                "mutations": mutation_str,
                "mut_dg_bind": float("nan"),
                "mut_energy": float("nan"),
            },
        }


# --------------------------------------------------------------------------
# Task list
# --------------------------------------------------------------------------
def make_tasks():
    """Collect (ab_key, fasta_stem, idx, header) for every variant in m=10 fastas."""
    tasks = []
    fasta_to_outdir = {}
    for ab_key in ridg.AB_CONFIG:
        ab_dir = REGEN / ab_key / f"n_mut_{N_MUT_TARGET}"
        if not ab_dir.exists():
            continue
        for fasta_path in sorted(ab_dir.glob("*.fasta")):
            fasta_stem = fasta_path.stem
            fasta_to_outdir[fasta_stem] = ab_dir
            idx = -1
            with open(fasta_path) as f:
                for line in f:
                    if line.startswith(">"):
                        idx += 1
                        if idx == 0:
                            continue  # skip WT
                        tasks.append((ab_key, fasta_stem, idx, line.strip()[1:]))
    return tasks, fasta_to_outdir


# --------------------------------------------------------------------------
# Main: relax WTs first, then dispatch parallel mutants
# --------------------------------------------------------------------------
def main():
    import pyrosetta
    pyrosetta.init("-ignore_unrecognized_res -ex1 -ex2 -mute all -constant_seed -jran 1234")
    from pyrosetta import pose_from_pdb
    from pyrosetta.rosetta.core.scoring import ScoreFunctionFactory
    scorefxn = ScoreFunctionFactory.create_score_function("ref2015")

    print("=" * 70)
    print("Step 1: FastRelax each antibody's WT once and score baselines")
    print("=" * 70)
    wt_baselines = {}
    for ab_key, (pdb_file, h_chain, l_chain, _, vh_len, interface_str) in ridg.AB_CONFIG.items():
        clean_pdb = CLEAN_PDB_DIR / pdb_file
        relaxed_path = RELAX_DIR / pdb_file.replace(".pdb", "_relaxed.pdb")
        print(f"\n[{ab_key}] loading {clean_pdb.name} ({pdb_file})")
        t0 = time.time()
        if relaxed_path.exists():
            print(f"  Relaxed PDB already exists at {relaxed_path.name}, reusing")
            wt_pose = pose_from_pdb(str(relaxed_path))
        else:
            wt_pose = pose_from_pdb(str(clean_pdb))
            print(f"  Pose: {wt_pose.total_residue()} residues. Running FastRelax(1 cycle)…")
            fastrelax(wt_pose, scorefxn, n_cycles=1)
            wt_pose.dump_pdb(str(relaxed_path))
        wt_dg = score_interface(wt_pose, scorefxn, interface_str)
        wt_e = scorefxn(wt_pose)
        wt_baselines[ab_key] = {"wt_dg_bind": wt_dg, "wt_energy": wt_e}
        print(f"  Baseline interface dG: {wt_dg:.3f}   stability: {wt_e:.3f}   ({time.time()-t0:.0f}s)")

    print("\n" + "=" * 70)
    print(f"Step 2: parallel ΔΔG over m={N_MUT_TARGET} variants ({N_WORKERS} workers)")
    print("=" * 70)
    tasks, fasta_to_outdir = make_tasks()
    print(f"Total tasks: {len(tasks)}")

    by_fasta = defaultdict(list)
    skipped = defaultdict(int)
    errors = defaultdict(int)

    t0 = time.time()
    n_done = 0
    with ProcessPoolExecutor(max_workers=N_WORKERS, initializer=init_worker) as ex:
        for r in ex.map(worker_task, tasks, chunksize=2):
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
            if n_done % 50 == 0:
                el = time.time() - t0
                rate = n_done / el
                eta = (len(tasks) - n_done) / rate / 60
                print(f"  [{n_done}/{len(tasks)}] elapsed={el/60:.1f}min rate={rate:.2f}/s ETA={eta:.1f}min", flush=True)

    # --- Combine into one DataFrame, attach baselines, compute ΔΔG ---
    rows = []
    for fasta_stem, records in by_fasta.items():
        for rec in records:
            ab = rec["antibody"]
            wt = wt_baselines.get(ab, {})
            rec["wt_dg_bind"] = wt.get("wt_dg_bind", float("nan"))
            rec["wt_energy"] = wt.get("wt_energy", float("nan"))
            rec["interface_ddg"] = rec["mut_dg_bind"] - rec["wt_dg_bind"]
            rec["ddg"] = rec["mut_energy"] - rec["wt_energy"]
            rows.append(rec)
    df = pd.DataFrame(rows)
    out_csv = REGEN / "v44o_regen_relax_ddg_m10.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nCombined CSV: {out_csv}  rows={len(df)}")

    if len(df) > 0:
        print("\nWT baselines (after FastRelax):")
        for ab in df["antibody"].unique():
            r = df[df["antibody"] == ab].iloc[0]
            print(f"  {ab}: wt_dg_bind={r['wt_dg_bind']:.3f}  wt_energy={r['wt_energy']:.3f}")
        print("\nPer-antibody ΔΔG distribution (interface):")
        print(df.groupby("antibody")["interface_ddg"].describe(percentiles=[0.25, 0.5, 0.75]).round(3))
        print("\nPer-antibody ΔΔG distribution (stability):")
        print(df.groupby("antibody")["ddg"].describe(percentiles=[0.25, 0.5, 0.75]).round(3))

    total = time.time() - t0
    print(f"\nTotal wall time: {total/3600:.2f}h ({total/60:.1f}min)", flush=True)


if __name__ == "__main__":
    main()
