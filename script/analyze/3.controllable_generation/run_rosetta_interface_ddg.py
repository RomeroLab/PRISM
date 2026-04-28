#!/usr/bin/env python
"""
Rosetta Interface ΔΔG: Antibody-Antigen binding energy change upon mutation.

Uses InterfaceAnalyzerMover to compute:
  ΔG_bind = E_complex - E_separated
  ΔΔG_bind = ΔG_bind(mut) - ΔG_bind(wt)

Usage:
    conda run -n base python run_rosetta_interface_ddg.py
"""
import csv
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple

# ============================================================
# Config
# ============================================================
PDB_DIR = Path(__file__).parent / "pll_results" / "ddg_results"  # cleaned PDBs
PLL_DIR = Path(__file__).parent / "pll_results"
OUT_DIR = PLL_DIR / "interface_ddg_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# antibody key → (clean_pdb, heavy, light, antigen_chains, vh_length)
# interface string format: "Ab_Ag" where Ab=heavy+light chain letters, Ag=antigen chain letters
AB_CONFIG = {
    "trast": ("1N8Z_clean.pdb", "B", "A", ["C"], 120, "BA_C"),
    "cr9114": ("4FQI_clean.pdb", "H", "L", ["A", "B"], 121, "HL_AB"),
    "g631": ("2FJH_clean.pdb", "H", "L", ["V", "W"], 120, "HL_VW"),
}

AA_3TO1 = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
}
AA_1TO3 = {v: k for k, v in AA_3TO1.items()}


# ============================================================
# Helpers
# ============================================================
def parse_mutations_from_header(header, vh_len):
    m = re.search(r"mutations=([^|]*)", header)
    if not m or not m.group(1).strip():
        return []
    mutations = []
    for part in m.group(1).strip().split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            chain_prefix, rest = part.split(":", 1)
            chain = chain_prefix.strip().upper()
            mutations.append((chain, int(rest[1:-1]), rest[0], rest[-1]))
        else:
            pos = int(part[1:-1])
            if pos <= vh_len:
                mutations.append(("H", pos, part[0], part[-1]))
            else:
                mutations.append(("L", pos - vh_len, part[0], part[-1]))
    return mutations


def build_chain_residue_map(pose, chain_id):
    pdb_info = pose.pdb_info()
    mapping = {}
    seq_pos = 0
    for i in range(1, pose.total_residue() + 1):
        if pdb_info.chain(i) == chain_id:
            seq_pos += 1
            mapping[seq_pos] = i
    return mapping


def get_residue_aa(pose, resnum):
    aa3 = pose.residue(resnum).name3().strip()
    return AA_3TO1.get(aa3, "X")


# ============================================================
# Interface ΔΔG
# ============================================================
def compute_interface_ddg(wt_pose, scorefxn, mut_resnums, mut_aas, interface_str, shell=8.0):
    """Compute binding ΔΔG using InterfaceAnalyzerMover.

    ΔΔG = dG_bind(mut) - dG_bind(wt)
    where dG_bind = E_complex - E_separated (computed by InterfaceAnalyzerMover)
    """
    from pyrosetta.rosetta.protocols.simple_moves import MutateResidue
    from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
    from pyrosetta.rosetta.core.pack.task import TaskFactory
    from pyrosetta.rosetta.core.pack.task.operation import (
        InitializeFromCommandline, RestrictToRepacking,
    )
    from pyrosetta.rosetta.protocols.minimization_packing import PackRotamersMover

    def repack_shell(pose, sites):
        tf = TaskFactory()
        tf.push_back(InitializeFromCommandline())
        tf.push_back(RestrictToRepacking())
        task = tf.create_task_and_apply_taskoperations(pose)
        site_set = set(sites)
        for i in range(1, pose.total_residue() + 1):
            if i in site_set:
                continue
            try:
                close = any(
                    pose.residue(i).xyz("CA").distance(pose.residue(m).xyz("CA")) <= shell
                    for m in sites
                )
            except RuntimeError:
                close = False
            if not close:
                task.nonconst_residue_task(i).prevent_repacking()
        PackRotamersMover(scorefxn, task).apply(pose)

    def get_dg_bind(pose):
        iam = InterfaceAnalyzerMover()
        iam.set_interface(interface_str)
        iam.set_scorefunction(scorefxn)
        iam.set_compute_interface_energy(True)
        iam.set_compute_separated_sasa(False)
        iam.set_pack_separated(True)
        iam.apply(pose)
        return pose.scores.get("dG_separated", None) or pose.scores.get("dG_separated/dSASAx100", 0.0)

    # WT: repack around mutation sites, then score interface
    wt_copy = wt_pose.clone()
    repack_shell(wt_copy, mut_resnums)
    wt_dg = get_dg_bind(wt_copy)

    # Mutant: mutate + repack, then score interface
    mut_copy = wt_pose.clone()
    for resnum, aa1 in zip(mut_resnums, mut_aas):
        MutateResidue(resnum, AA_1TO3[aa1]).apply(mut_copy)
    repack_shell(mut_copy, mut_resnums)
    mut_dg = get_dg_bind(mut_copy)

    return {
        "interface_ddg": mut_dg - wt_dg,
        "wt_dg_bind": wt_dg,
        "mut_dg_bind": mut_dg,
    }


# ============================================================
# Process one FASTA
# ============================================================
def process_fasta(fasta_path, ab_key, wt_pose, scorefxn, h_chain, l_chain, vh_len, interface_str):
    chain_map = {
        "H": build_chain_residue_map(wt_pose, h_chain),
        "L": build_chain_residue_map(wt_pose, l_chain),
    }

    headers = []
    with open(fasta_path) as f:
        for line in f:
            if line.startswith(">"):
                headers.append(line.strip()[1:])

    results = []
    skipped = 0

    for idx, header in enumerate(headers):
        if idx == 0:
            continue

        rec_id = header.split("|")[0]
        mutations = parse_mutations_from_header(header, vh_len)
        if not mutations:
            continue

        mut_resnums = []
        mut_aas = []
        applied_strs = []

        for chain, pos, wt_aa, mut_aa in mutations:
            resnum = chain_map.get(chain, {}).get(pos)
            if resnum is None:
                continue
            pdb_aa = get_residue_aa(wt_pose, resnum)
            if pdb_aa != wt_aa:
                continue
            mut_resnums.append(resnum)
            mut_aas.append(mut_aa)
            applied_strs.append(f"{chain}:{wt_aa}{pos}{mut_aa}")

        if not mut_resnums:
            skipped += 1
            continue

        mutation_str = ",".join(applied_strs)
        t0 = time.time()

        try:
            result = compute_interface_ddg(wt_pose, scorefxn, mut_resnums, mut_aas, interface_str)
            elapsed = time.time() - t0
            if idx % 25 == 0:
                print(f"    [{idx}/{len(headers)-1}] {mutation_str} → iΔΔG={result['interface_ddg']:+.2f} ({elapsed:.1f}s)")

            results.append({
                "id": rec_id,
                "source": fasta_path.stem,
                "antibody": ab_key,
                "n_mut": len(mut_resnums),
                "mutations": mutation_str,
                "interface_ddg": round(result["interface_ddg"], 3),
                "wt_dg_bind": round(result["wt_dg_bind"], 3),
                "mut_dg_bind": round(result["mut_dg_bind"], 3),
            })
        except Exception as e:
            print(f"    [{idx}] {mutation_str} FAILED: {e}")
            results.append({
                "id": rec_id,
                "source": fasta_path.stem,
                "antibody": ab_key,
                "n_mut": len(mut_resnums),
                "mutations": mutation_str,
                "interface_ddg": float("nan"),
                "wt_dg_bind": float("nan"),
                "mut_dg_bind": float("nan"),
            })

    return results, skipped


# ============================================================
# Main
# ============================================================
def main():
    import pyrosetta
    from pyrosetta import pose_from_pdb
    from pyrosetta.rosetta.core.scoring import ScoreFunctionFactory

    pyrosetta.init("-ignore_unrecognized_res -ex1 -ex2 -mute all")
    scorefxn = ScoreFunctionFactory.create_score_function("ref2015")

    all_results = []

    for ab_key, (pdb_file, h_chain, l_chain, ag_chains, vh_len, interface_str) in AB_CONFIG.items():
        pdb_path = PDB_DIR / pdb_file
        ab_dir = PLL_DIR / ab_key
        fasta_files = sorted(ab_dir.glob("*.fasta"))

        print(f"\n{'#'*60}")
        print(f"# {ab_key} | PDB={pdb_file} | interface={interface_str}")
        print(f"# {len(fasta_files)} FASTA files")
        print(f"{'#'*60}")

        wt_pose = pose_from_pdb(str(pdb_path))
        print(f"  PDB: {wt_pose.total_residue()} residues")

        for fasta_path in fasta_files:
            print(f"\n  === {fasta_path.name} ===")
            results, skipped = process_fasta(
                fasta_path, ab_key, wt_pose, scorefxn, h_chain, l_chain, vh_len, interface_str,
            )
            all_results.extend(results)
            print(f"  Scored: {len(results)}, Skipped: {skipped}")

            if results:
                out_csv = OUT_DIR / f"{fasta_path.stem}_interface_ddg.csv"
                with open(out_csv, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
                    w.writeheader()
                    w.writerows(results)

    # Combined + summary
    if all_results:
        import pandas as pd
        df = pd.DataFrame(all_results)
        df.to_csv(OUT_DIR / "all_interface_ddg_results.csv", index=False)

        def get_model(s):
            if "ablang2" in s: return "AbLang2"
            if "esm2" in s: return "ESM2"
            if "_gl_" in s: return "PRISM-GL"
            if "_ngl_" in s: return "PRISM-NGL"
            if "_full_" in s: return "PRISM-Full"
            if "_region_" in s: return "PRISM-Region"
            return s

        df["model"] = df["source"].apply(get_model)
        valid = df.dropna(subset=["interface_ddg"])

        print(f"\n{'='*60}")
        print(f"ALL DONE. Total scored: {len(valid)}")
        print(f"{'='*60}")

        for ab in valid["antibody"].unique():
            ab_df = valid[valid["antibody"] == ab]
            print(f"\n=== {ab} ===")
            stats = ab_df.groupby("model").agg(
                n=("interface_ddg", "count"),
                mean=("interface_ddg", "mean"),
                median=("interface_ddg", "median"),
                pct_improved=("interface_ddg", lambda x: (x < 0).mean() * 100),
                best=("interface_ddg", "min"),
            ).round(3)
            stats = stats.sort_values("median")
            print(stats.to_string())


if __name__ == "__main__":
    main()
