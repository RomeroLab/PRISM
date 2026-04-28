#!/usr/bin/env python
"""
Rosetta ΔΔG Scoring for PRISM-Generated Antibody Variants

Reads a wildtype PDB and a FASTA file of mutant sequences (from pll_guided_sampler.py),
identifies mutations by sequence alignment, applies them with PyRosetta,
and computes Cartesian ΔΔG via FastRelax.

Output: CSV with columns [id, mode, n_mut, mutations, ddg, wt_energy, mut_energy, ...]

Prerequisites:
    pip install pyrosetta  (requires license — academic free at https://www.pyrosetta.org)

Usage:
    python rosetta_ddg.py \
        --pdb wt.pdb \
        --fasta variants.fasta \
        --chain H \
        --output ddg_results.csv \
        --n_relax 3

    # Faster single-point (no relax, just repack):
    python rosetta_ddg.py \
        --pdb wt.pdb \
        --fasta variants.fasta \
        --chain H \
        --output ddg_results.csv \
        --mode repack

    # With pre-relaxed WT (skip redundant WT relax):
    python rosetta_ddg.py \
        --pdb wt_relaxed.pdb \
        --fasta variants.fasta \
        --chain H \
        --output ddg_results.csv \
        --wt_prerelaxed
"""

import argparse
import csv
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

AA_3TO1 = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
}
AA_1TO3 = {v: k for k, v in AA_3TO1.items()}


# ═══════════════════════════════════════════════════════════════════════════════
# FASTA Parsing
# ═══════════════════════════════════════════════════════════════════════════════


def parse_fasta(fasta_path: str) -> List[Dict]:
    """Parse FASTA file from pll_guided_sampler.py.

    Expected header format:
        >var_0001|mode=full|n_mut=3|T=1.0|mutations=A52W,D98Y
    or generic:
        >seq_id

    Returns:
        List of dicts with keys: id, mode, n_mut, temperature, mutations, sequence.
    """
    records = []
    current_header = None
    current_seq_lines = []

    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_header is not None:
                    records.append(_parse_record(current_header, "".join(current_seq_lines)))
                current_header = line[1:]
                current_seq_lines = []
            else:
                current_seq_lines.append(line)

    if current_header is not None:
        records.append(_parse_record(current_header, "".join(current_seq_lines)))

    return records


def _parse_record(header: str, sequence: str) -> Dict:
    """Parse a single FASTA record header into structured dict."""
    fields = header.split("|")
    rec = {
        "id": fields[0],
        "mode": "",
        "n_mut": 0,
        "temperature": 0.0,
        "mutations_header": "",
        "sequence": sequence,
    }

    for field in fields[1:]:
        if "=" in field:
            key, val = field.split("=", 1)
            key = key.strip()
            val = val.strip()
            if key == "mode":
                rec["mode"] = val
            elif key == "n_mut":
                rec["n_mut"] = int(val)
            elif key == "T":
                rec["temperature"] = float(val)
            elif key == "mutations":
                rec["mutations_header"] = val

    return rec


# ═══════════════════════════════════════════════════════════════════════════════
# Mutation Detection
# ═══════════════════════════════════════════════════════════════════════════════


def detect_mutations(wt_seq: str, mut_seq: str) -> List[Tuple[int, str, str]]:
    """Detect mutations by pairwise comparison.

    Args:
        wt_seq: Wild-type amino acid sequence.
        mut_seq: Mutant amino acid sequence (same length).

    Returns:
        List of (0-indexed position, wt_aa, mut_aa) tuples.
    """
    if len(wt_seq) != len(mut_seq):
        raise ValueError(
            f"Sequence length mismatch: WT={len(wt_seq)}, mut={len(mut_seq)}. "
            "Only substitution mutations are supported (no indels)."
        )
    mutations = []
    for i, (wt_aa, mut_aa) in enumerate(zip(wt_seq, mut_seq)):
        if wt_aa != mut_aa:
            mutations.append((i, wt_aa, mut_aa))
    return mutations


def build_seq_to_pdb_mapping(pose, chain_id: str) -> List[int]:
    """Build mapping from 0-indexed sequence position to Rosetta pose residue number.

    Args:
        pose: PyRosetta Pose object.
        chain_id: PDB chain letter (e.g., "H", "L").

    Returns:
        List of pose residue numbers, one per sequence position.
        mapping[seq_idx] = pose_resnum.
    """
    pdb_info = pose.pdb_info()
    mapping = []
    for i in range(1, pose.total_residue() + 1):
        if pdb_info.chain(i) == chain_id:
            mapping.append(i)
    return mapping


def extract_chain_sequence(pose, chain_id: str) -> str:
    """Extract amino acid sequence for a specific chain from a Pose.

    Args:
        pose: PyRosetta Pose object.
        chain_id: PDB chain letter.

    Returns:
        Amino acid sequence string.
    """
    pdb_info = pose.pdb_info()
    seq = []
    for i in range(1, pose.total_residue() + 1):
        if pdb_info.chain(i) == chain_id:
            aa3 = pose.residue(i).name3().strip()
            aa1 = AA_3TO1.get(aa3, "X")
            seq.append(aa1)
    return "".join(seq)


# ═══════════════════════════════════════════════════════════════════════════════
# PyRosetta Energy Computation
# ═══════════════════════════════════════════════════════════════════════════════


def relax_pose(pose, scorefxn, cartesian: bool = True):
    """Apply FastRelax to a pose (in-place).

    Args:
        pose: PyRosetta Pose to relax.
        scorefxn: Score function.
        cartesian: Use Cartesian-space relax (recommended for ddG).
    """
    from pyrosetta.rosetta.protocols.relax import FastRelax

    relax = FastRelax()
    relax.set_scorefxn(scorefxn)
    if cartesian:
        relax.cartesian(True)
    relax.apply(pose)


def repack_around_mutations(pose, scorefxn, mut_pose_resnums: List[int], shell: float = 8.0):
    """Repack sidechains around mutation sites.

    Args:
        pose: PyRosetta Pose.
        scorefxn: Score function.
        mut_pose_resnums: Pose residue numbers of mutated positions.
        shell: Repack shell radius in Angstroms.
    """
    from pyrosetta.rosetta.core.pack.task import TaskFactory
    from pyrosetta.rosetta.core.pack.task.operation import (
        InitializeFromCommandline,
        RestrictToRepacking,
    )
    from pyrosetta.rosetta.protocols.minimization_packing import PackRotamersMover

    tf = TaskFactory()
    tf.push_back(InitializeFromCommandline())
    tf.push_back(RestrictToRepacking())

    packer_task = tf.create_task_and_apply_taskoperations(pose)

    # Restrict to neighborhood of mutations
    mut_set = set(mut_pose_resnums)
    for i in range(1, pose.total_residue() + 1):
        if i in mut_set:
            continue
        # Check distance to any mutation site
        close = False
        for m in mut_pose_resnums:
            if pose.residue(i).xyz("CA").distance(pose.residue(m).xyz("CA")) <= shell:
                close = True
                break
        if not close:
            packer_task.nonconst_residue_task(i).prevent_repacking()

    packer = PackRotamersMover(scorefxn, packer_task)
    packer.apply(pose)


def compute_ddg_relax(
    wt_pose,
    scorefxn,
    mutations: List[Tuple[int, str, str]],
    seq_to_pose: List[int],
    n_relax: int = 3,
    wt_prerelaxed: bool = False,
) -> Dict[str, float]:
    """Compute ΔΔG using Cartesian FastRelax protocol.

    For each of n_relax trajectories, relax WT and mutant independently,
    then take the best (lowest energy) from each.

    Args:
        wt_pose: WT Pose (will be cloned, not modified).
        scorefxn: Cartesian score function.
        mutations: List of (seq_idx, wt_aa, mut_aa).
        seq_to_pose: Mapping from seq idx to pose residue number.
        n_relax: Number of independent relax trajectories.
        wt_prerelaxed: If True, skip WT relax and score as-is.

    Returns:
        Dict with ddg, wt_energy, mut_energy, wt_energies, mut_energies.
    """
    from pyrosetta.rosetta.protocols.simple_moves import MutateResidue

    # WT energy
    if wt_prerelaxed:
        wt_energies = [scorefxn(wt_pose.clone())]
    else:
        wt_energies = []
        for _ in range(n_relax):
            wt_copy = wt_pose.clone()
            relax_pose(wt_copy, scorefxn, cartesian=True)
            wt_energies.append(scorefxn(wt_copy))

    # Mutant energy
    mut_energies = []
    for _ in range(n_relax):
        mut_copy = wt_pose.clone()
        for seq_idx, _, mut_aa in mutations:
            pose_resnum = seq_to_pose[seq_idx]
            MutateResidue(pose_resnum, AA_1TO3[mut_aa]).apply(mut_copy)
        relax_pose(mut_copy, scorefxn, cartesian=True)
        mut_energies.append(scorefxn(mut_copy))

    wt_best = min(wt_energies)
    mut_best = min(mut_energies)

    return {
        "ddg": mut_best - wt_best,
        "wt_energy": wt_best,
        "mut_energy": mut_best,
    }


def compute_ddg_repack(
    wt_pose,
    scorefxn,
    mutations: List[Tuple[int, str, str]],
    seq_to_pose: List[int],
    shell: float = 8.0,
) -> Dict[str, float]:
    """Compute ΔΔG using fast repack-only protocol (no backbone relax).

    Much faster than full relax (~seconds vs ~minutes per variant).
    Less accurate but good for ranking large libraries.

    Args:
        wt_pose: WT Pose.
        scorefxn: Score function (ref2015 recommended).
        mutations: List of (seq_idx, wt_aa, mut_aa).
        seq_to_pose: Mapping from seq idx to pose residue number.
        shell: Repack shell radius in Angstroms.

    Returns:
        Dict with ddg, wt_energy, mut_energy.
    """
    from pyrosetta.rosetta.protocols.simple_moves import MutateResidue

    mut_pose_resnums = [seq_to_pose[si] for si, _, _ in mutations]

    # WT: repack around mutation sites for fair comparison
    wt_copy = wt_pose.clone()
    repack_around_mutations(wt_copy, scorefxn, mut_pose_resnums, shell)
    wt_energy = scorefxn(wt_copy)

    # Mutant: mutate + repack
    mut_copy = wt_pose.clone()
    for seq_idx, _, mut_aa in mutations:
        pose_resnum = seq_to_pose[seq_idx]
        MutateResidue(pose_resnum, AA_1TO3[mut_aa]).apply(mut_copy)
    repack_around_mutations(mut_copy, scorefxn, mut_pose_resnums, shell)
    mut_energy = scorefxn(mut_copy)

    return {
        "ddg": mut_energy - wt_energy,
        "wt_energy": wt_energy,
        "mut_energy": mut_energy,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════════════════════


def run_ddg_pipeline(
    pdb_path: str,
    fasta_path: str,
    chain_id: str,
    output_csv: str,
    mode: str = "cartesian",
    n_relax: int = 3,
    wt_prerelaxed: bool = False,
    shell: float = 8.0,
) -> None:
    """Run full ΔΔG pipeline: load PDB, parse FASTA, score all variants.

    Args:
        pdb_path: Path to wildtype PDB.
        fasta_path: Path to FASTA with mutant sequences.
        chain_id: Chain letter to mutate (e.g., "H").
        output_csv: Output CSV path.
        mode: "cartesian" (FastRelax, accurate) or "repack" (fast, approximate).
        n_relax: Number of relax trajectories per variant (cartesian mode only).
        wt_prerelaxed: Skip WT relaxation if PDB is already relaxed.
        shell: Repack shell radius in Angstroms (repack mode only).
    """
    import pyrosetta
    from pyrosetta import pose_from_pdb
    from pyrosetta.rosetta.core.scoring import ScoreFunctionFactory

    # Initialize PyRosetta
    flags = "-ignore_unrecognized_res -ex1 -ex2 -mute all"
    if mode == "cartesian":
        flags += " -relax:cartesian"
    pyrosetta.init(flags)

    # Load WT structure
    print(f"Loading WT PDB: {pdb_path}")
    wt_pose = pose_from_pdb(pdb_path)

    # Extract WT sequence and build mapping
    wt_seq = extract_chain_sequence(wt_pose, chain_id)
    seq_to_pose = build_seq_to_pdb_mapping(wt_pose, chain_id)
    print(f"Chain {chain_id}: {len(wt_seq)} residues")
    print(f"WT sequence: {wt_seq[:30]}...{wt_seq[-10:]}")

    # Score function
    if mode == "cartesian":
        scorefxn = ScoreFunctionFactory.create_score_function("ref2015_cart")
    else:
        scorefxn = ScoreFunctionFactory.create_score_function("ref2015")

    # Parse FASTA
    print(f"\nParsing FASTA: {fasta_path}")
    records = parse_fasta(fasta_path)
    print(f"Found {len(records)} sequences")

    # Pre-relax WT once if using cartesian mode (saves repeated work)
    wt_energy_ref = None
    if mode == "cartesian" and wt_prerelaxed:
        wt_energy_ref = scorefxn(wt_pose)
        print(f"WT energy (pre-relaxed): {wt_energy_ref:.2f} REU")

    # Process each variant
    results = []
    skipped = 0
    t_start = time.time()

    for i, rec in enumerate(records):
        seq = rec["sequence"]
        rec_id = rec["id"]

        # Skip WT record
        if rec_id == "WT" or seq == wt_seq:
            print(f"  [{i+1}/{len(records)}] {rec_id}: WT (skipped)")
            continue

        # Detect mutations
        try:
            mutations = detect_mutations(wt_seq, seq)
        except ValueError as e:
            print(f"  [{i+1}/{len(records)}] {rec_id}: SKIP — {e}")
            skipped += 1
            continue

        if not mutations:
            print(f"  [{i+1}/{len(records)}] {rec_id}: no mutations (skipped)")
            continue

        mutation_str = ",".join(f"{wt}{pos+1}{mt}" for pos, wt, mt in mutations)
        print(f"  [{i+1}/{len(records)}] {rec_id}: {mutation_str} ...", end=" ", flush=True)

        t0 = time.time()
        try:
            if mode == "cartesian":
                result = compute_ddg_relax(
                    wt_pose, scorefxn, mutations, seq_to_pose,
                    n_relax=n_relax, wt_prerelaxed=wt_prerelaxed,
                )
            else:
                result = compute_ddg_repack(
                    wt_pose, scorefxn, mutations, seq_to_pose, shell=shell,
                )

            elapsed = time.time() - t0
            print(f"ΔΔG={result['ddg']:+.2f} REU ({elapsed:.1f}s)")

            results.append({
                "id": rec_id,
                "mode": rec["mode"],
                "n_mut": len(mutations),
                "mutations": mutation_str,
                "ddg": round(result["ddg"], 3),
                "wt_energy": round(result["wt_energy"], 3),
                "mut_energy": round(result["mut_energy"], 3),
                "temperature": rec["temperature"],
                "sequence": seq,
            })

        except Exception as e:
            elapsed = time.time() - t0
            print(f"FAILED ({elapsed:.1f}s): {e}")
            results.append({
                "id": rec_id,
                "mode": rec["mode"],
                "n_mut": len(mutations),
                "mutations": mutation_str,
                "ddg": float("nan"),
                "wt_energy": float("nan"),
                "mut_energy": float("nan"),
                "temperature": rec["temperature"],
                "sequence": seq,
            })

    total_time = time.time() - t_start

    # Write CSV
    print(f"\nWriting results to: {output_csv}")
    fieldnames = [
        "id", "mode", "n_mut", "mutations",
        "ddg", "wt_energy", "mut_energy",
        "temperature", "sequence",
    ]
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  Scored:  {len(results)} variants")
    print(f"  Skipped: {skipped}")
    print(f"  Time:    {total_time:.1f}s ({total_time/max(len(results),1):.1f}s per variant)")
    if results:
        ddgs = [r["ddg"] for r in results if r["ddg"] == r["ddg"]]  # exclude NaN
        if ddgs:
            print(f"  ΔΔG range: [{min(ddgs):+.2f}, {max(ddgs):+.2f}] REU")
            stabilizing = sum(1 for d in ddgs if d < -1.0)
            neutral = sum(1 for d in ddgs if -1.0 <= d <= 1.0)
            destabilizing = sum(1 for d in ddgs if d > 1.0)
            print(f"  Stabilizing (< -1 REU):   {stabilizing}")
            print(f"  Neutral (-1 to +1 REU):   {neutral}")
            print(f"  Destabilizing (> +1 REU): {destabilizing}")
    print(f"\nOutput: {output_csv}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Compute Rosetta ΔΔG for PRISM-generated antibody variants",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--pdb", required=True, help="Wildtype PDB file")
    p.add_argument("--fasta", required=True, help="FASTA file with mutant sequences")
    p.add_argument("--chain", default="H", help="Chain ID to mutate")
    p.add_argument("--output", default="ddg_results.csv", help="Output CSV path")
    p.add_argument(
        "--mode", default="cartesian", choices=["cartesian", "repack"],
        help="cartesian: FastRelax (accurate, ~2-5 min/variant). "
             "repack: sidechain-only (fast, ~5-10 sec/variant)",
    )
    p.add_argument("--n_relax", type=int, default=3, help="Relax trajectories per variant (cartesian only)")
    p.add_argument("--wt_prerelaxed", action="store_true", help="Skip WT relax (PDB already relaxed)")
    p.add_argument("--shell", type=float, default=8.0, help="Repack shell radius in Angstroms")
    return p.parse_args(argv)


def main():
    args = parse_args()
    run_ddg_pipeline(
        pdb_path=args.pdb,
        fasta_path=args.fasta,
        chain_id=args.chain,
        output_csv=args.output,
        mode=args.mode,
        n_relax=args.n_relax,
        wt_prerelaxed=args.wt_prerelaxed,
        shell=args.shell,
    )


if __name__ == "__main__":
    main()
