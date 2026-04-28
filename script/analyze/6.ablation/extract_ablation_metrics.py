#!/usr/bin/env python3
"""
Extract and compare metrics from alpha ablation TensorBoard logs.

Usage:
    python script/analyze/extract_ablation_metrics.py

Reads TensorBoard event files from the two ablation runs and produces
a side-by-side comparison in both txt and csv formats.
"""

import os
import glob
import numpy as np
from collections import defaultdict

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    print("ERROR: tensorboard not installed. Run: pip install tensorboard")
    raise

# ============================================================
# Config — update these paths after training
# ============================================================
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_BASE = os.environ.get("PRISM_OUTPUTS", str(_REPO_ROOT / "outputs"))
RUNS = {
    "alpha_fixed_1.0": "ablation_alpha_fixed_1.0_esm2_t12_35M_UR50D_custom_unfrozen12_lr4e-4_bs256",
    "alpha_learned": "ablation_alpha_learned_esm2_t12_35M_UR50D_custom_unfrozen12_lr4e-4_bs256",
}

# Key metrics to extract
TRAIN_METRICS = [
    "train/loss",
    "train/AA_Loss",
    "train/Origin_Loss",
    "train/Final_Loss",
    "train/NGL_AA_Loss",
    "train/Mean_Alpha",
]
VAL_METRICS = [
    "val/Final_PPL_All",
    "val/Final_PPL_NGL",
    "val/AA_PPL_GL",
    "val/AA_PPL_NGL",
    "val/Origin_Accuracy",
    "val/Mean_Alpha",
    "val/Alpha_Mean_CDR",
    "val/Alpha_Mean_FR",
]

OUT_DIR = str(Path(__file__).resolve().parent)


def find_event_file(run_name):
    """Find the TensorBoard event file for a run."""
    # Try common patterns
    patterns = [
        os.path.join(OUTPUT_BASE, run_name, "version_*", "events.out.tfevents.*"),
        os.path.join(OUTPUT_BASE, run_name, "checkpoints", "..", "version_*", "events.out.tfevents.*"),
        os.path.join(OUTPUT_BASE, run_name, "**", "events.out.tfevents.*"),
    ]
    for pattern in patterns:
        files = glob.glob(pattern, recursive=True)
        if files:
            return os.path.dirname(files[0])
    return None


def extract_scalars(log_dir, tags):
    """Extract scalar values from TensorBoard logs."""
    ea = EventAccumulator(log_dir)
    ea.Reload()
    available = ea.Tags().get("scalars", [])

    data = {}
    for tag in tags:
        if tag in available:
            events = ea.Scalars(tag)
            data[tag] = [(e.step, e.value) for e in events]
        else:
            data[tag] = []
    return data


def compute_loss_volatility(steps_values, window=20):
    """Compute rolling std of loss as a measure of training stability."""
    if len(steps_values) < window:
        return float("nan"), float("nan")
    values = np.array([v for _, v in steps_values])
    # Rolling std
    stds = []
    for i in range(window, len(values)):
        stds.append(np.std(values[i - window : i]))
    return np.mean(stds), np.max(stds)


def main():
    results = {}

    for label, run_name in RUNS.items():
        log_dir = find_event_file(run_name)
        if log_dir is None:
            print(f"WARNING: No event file found for '{run_name}'")
            print(f"  Searched in: {OUTPUT_BASE}/{run_name}/")
            print(f"  Skipping {label}...")
            results[label] = None
            continue

        print(f"Loading {label}: {log_dir}")
        all_tags = TRAIN_METRICS + VAL_METRICS
        data = extract_scalars(log_dir, all_tags)
        results[label] = data

    # ============================================================
    # Generate comparison report
    # ============================================================
    lines = []
    lines.append("=" * 90)
    lines.append("ALPHA ABLATION: Early Training Stability Comparison")
    lines.append("=" * 90)
    lines.append("")

    for label in RUNS:
        if results.get(label) is None:
            lines.append(f"[{label}] — NO DATA (run not yet completed)")
            lines.append("")
            continue

        data = results[label]
        lines.append(f"--- {label} ---")

        # Train loss volatility
        if data.get("train/loss"):
            mean_std, max_std = compute_loss_volatility(data["train/loss"])
            lines.append(f"  Train Loss Volatility (rolling-20 std):  mean={mean_std:.4f}  max={max_std:.4f}")

            # Final train loss value
            final_step, final_val = data["train/loss"][-1]
            lines.append(f"  Train Loss (final step {final_step}): {final_val:.4f}")

        # Per-component losses (last 50 steps average)
        for tag in ["train/AA_Loss", "train/Origin_Loss", "train/Final_Loss", "train/NGL_AA_Loss"]:
            if data.get(tag) and len(data[tag]) >= 10:
                last_vals = [v for _, v in data[tag][-50:]]
                lines.append(f"  {tag:<30s} last-50 mean={np.mean(last_vals):.4f}  std={np.std(last_vals):.4f}")

        # Alpha trajectory
        if data.get("train/Mean_Alpha"):
            alpha_vals = [v for _, v in data["train/Mean_Alpha"]]
            lines.append(f"  Mean Alpha: start={alpha_vals[0]:.4f}  end={alpha_vals[-1]:.4f}  "
                         f"min={min(alpha_vals):.4f}  max={max(alpha_vals):.4f}")

        # Val metrics
        lines.append("")
        lines.append("  Validation metrics (all epochs):")
        for tag in VAL_METRICS:
            if data.get(tag):
                for step, val in data[tag]:
                    lines.append(f"    {tag:<30s} step={step:>5d}  value={val:.4f}")

        lines.append("")

    # ============================================================
    # Side-by-side comparison table
    # ============================================================
    lines.append("=" * 90)
    lines.append("SIDE-BY-SIDE: Train Loss Volatility")
    lines.append("=" * 90)

    header = f"{'Metric':<35s}"
    for label in RUNS:
        header += f"  {label:>20s}"
    lines.append(header)
    lines.append("-" * 90)

    compare_tags = [
        ("Train Loss Volatility (mean std)", lambda d: compute_loss_volatility(d.get("train/loss", []))[0]),
        ("Train Loss Volatility (max std)", lambda d: compute_loss_volatility(d.get("train/loss", []))[1]),
        ("Final Train Loss", lambda d: d["train/loss"][-1][1] if d.get("train/loss") else float("nan")),
        ("Final Mean Alpha", lambda d: d["train/Mean_Alpha"][-1][1] if d.get("train/Mean_Alpha") else float("nan")),
        ("AA Loss (last-50 std)", lambda d: np.std([v for _, v in d["train/AA_Loss"][-50:]]) if d.get("train/AA_Loss") and len(d["train/AA_Loss"]) >= 10 else float("nan")),
        ("Final Loss (last-50 std)", lambda d: np.std([v for _, v in d["train/Final_Loss"][-50:]]) if d.get("train/Final_Loss") and len(d["train/Final_Loss"]) >= 10 else float("nan")),
    ]

    for metric_name, extractor in compare_tags:
        row = f"{metric_name:<35s}"
        for label in RUNS:
            if results.get(label) is not None:
                val = extractor(results[label])
                row += f"  {val:>20.4f}"
            else:
                row += f"  {'N/A':>20s}"
        lines.append(row)

    lines.append("=" * 90)
    lines.append("")
    lines.append("INTERPRETATION GUIDE:")
    lines.append("  - Higher loss volatility (std) = more unstable training")
    lines.append("  - If alpha_fixed has higher volatility → covariate shift is real")
    lines.append("  - If similar → learnable alpha doesn't meaningfully stabilize early training")
    lines.append("")

    # Write output
    report = "\n".join(lines)
    print("\n" + report)

    out_path = os.path.join(OUT_DIR, "ablation_alpha_stability_report.txt")
    with open(out_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to: {out_path}")

    # Also save raw step-by-step data as CSV for plotting
    for label in RUNS:
        if results.get(label) is None:
            continue
        data = results[label]
        csv_path = os.path.join(OUT_DIR, f"ablation_{label}_train_loss.csv")
        if data.get("train/loss"):
            with open(csv_path, "w") as f:
                f.write("step,loss\n")
                for step, val in data["train/loss"]:
                    f.write(f"{step},{val:.6f}\n")
            print(f"CSV saved: {csv_path}")


if __name__ == "__main__":
    main()
