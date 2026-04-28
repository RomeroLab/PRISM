#!/usr/bin/env python3
"""
Comprehensive TensorBoard Event File Parser for Training Analysis.

Reads all scalar metrics from a TensorBoard event file and produces
a detailed summary grouped by category, with trend analysis for key metrics.
"""

import sys
from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Pass the TensorBoard event file path as the first CLI argument, e.g.:
#   python parse_tb_events.py outputs/<run>/version_0/events.out.tfevents.*
if len(sys.argv) > 1:
    EVENT_FILE = sys.argv[1]
else:
    raise SystemExit(
        "Usage: python parse_tb_events.py <path-to-tfevents-file>"
    )


def load_event_file(path):
    """Load all scalars from a TensorBoard event file."""
    ea = EventAccumulator(path, size_guidance={"scalars": 0})  # 0 = load all
    ea.Reload()
    return ea


def categorize_tag(tag):
    """Assign a tag to a display category."""
    tag_lower = tag.lower()
    if tag_lower.startswith("train/") or tag_lower.startswith("train_"):
        return "TRAIN"
    elif tag_lower.startswith("val/") or tag_lower.startswith("val_"):
        return "VALIDATION"
    elif tag_lower.startswith("dev/") or tag_lower.startswith("dev_"):
        return "DEVELOPABILITY"
    elif "lr" in tag_lower or "learning_rate" in tag_lower:
        return "LEARNING RATE"
    elif tag_lower.startswith("hp_metric") or tag_lower.startswith("hparam"):
        return "HPARAMS"
    elif tag_lower.startswith("epoch"):
        return "EPOCH"
    else:
        return "OTHER"


def is_key_metric(tag):
    """Check if a tag is a key metric we want trend data for."""
    tag_lower = tag.lower()
    keywords = ["ppl", "loss", "dev_", "perplexity", "accuracy", "acc", "pearson", "corr"]
    return any(kw in tag_lower for kw in keywords)


def format_value(v):
    """Format a numeric value for display."""
    if abs(v) < 0.001 or abs(v) >= 100000:
        return f"{v:.6e}"
    elif abs(v) < 1:
        return f"{v:.6f}"
    else:
        return f"{v:.4f}"


def main():
    print("=" * 100)
    print("TENSORBOARD EVENT FILE ANALYSIS")
    print("=" * 100)
    print(f"\nEvent file: {EVENT_FILE}")
    print()

    # Load events
    ea = load_event_file(EVENT_FILE)
    tags = ea.Tags().get("scalars", [])

    if not tags:
        print("ERROR: No scalar tags found in event file!")
        sys.exit(1)

    print(f"Total scalar tags found: {len(tags)}")
    print()

    # Collect all data
    all_data = {}
    for tag in sorted(tags):
        events = ea.Scalars(tag)
        all_data[tag] = events

    # Find global step range and epoch info
    all_steps = set()
    max_epoch = 0
    for tag, events in all_data.items():
        for e in events:
            all_steps.add(e.step)
        if "epoch" in tag.lower():
            for e in events:
                max_epoch = max(max_epoch, e.value)

    if all_steps:
        print(f"Step range: {min(all_steps)} to {max(all_steps)}")
        print(f"Total unique steps logged: {len(all_steps)}")
    if max_epoch > 0:
        print(f"Max epoch reached: {max_epoch:.1f}")
    print()

    # Group by category
    categories = defaultdict(list)
    for tag in sorted(tags):
        cat = categorize_tag(tag)
        categories[cat].append(tag)

    # Print category overview
    print("-" * 100)
    print("CATEGORY OVERVIEW")
    print("-" * 100)
    for cat in sorted(categories.keys()):
        print(f"  {cat}: {len(categories[cat])} metrics")
    print()

    # Detailed summary per category
    for cat in ["LEARNING RATE", "EPOCH", "TRAIN", "VALIDATION", "DEVELOPABILITY", "HPARAMS", "OTHER"]:
        if cat not in categories:
            continue

        tag_list = categories[cat]
        print("=" * 100)
        print(f"  {cat} METRICS ({len(tag_list)} tags)")
        print("=" * 100)
        print()

        # Header
        print(f"{'Tag':<55} {'First':>12} {'Last':>12} {'Min':>12} {'Max':>12} {'MinStep':>8} {'MaxStep':>8} {'#Pts':>6}")
        print("-" * 125)

        for tag in sorted(tag_list):
            events = all_data[tag]
            if not events:
                continue

            values = [e.value for e in events]
            steps = [e.step for e in events]

            first_val = values[0]
            last_val = values[-1]
            min_val = min(values)
            max_val = max(values)
            min_idx = values.index(min_val)
            max_idx = values.index(max_val)
            min_step = steps[min_idx]
            max_step = steps[max_idx]

            # Truncate tag name for display
            tag_display = tag if len(tag) <= 54 else tag[:51] + "..."

            print(
                f"{tag_display:<55} "
                f"{format_value(first_val):>12} "
                f"{format_value(last_val):>12} "
                f"{format_value(min_val):>12} "
                f"{format_value(max_val):>12} "
                f"{min_step:>8} "
                f"{max_step:>8} "
                f"{len(values):>6}"
            )

        print()

    # Trend analysis for key metrics
    print("=" * 100)
    print("  TREND ANALYSIS: KEY METRICS (last 10 data points)")
    print("=" * 100)
    print()

    key_tags = [tag for tag in sorted(tags) if is_key_metric(tag)]

    if not key_tags:
        print("No key metrics found matching PPL/loss/dev_ patterns.")
    else:
        for tag in key_tags:
            events = all_data[tag]
            if not events:
                continue

            last_n = events[-10:]
            values = [e.value for e in last_n]

            # Compute trend direction
            if len(values) >= 2:
                trend_val = values[-1] - values[0]
                trend_pct = (trend_val / abs(values[0]) * 100) if values[0] != 0 else 0
                if trend_val < 0:
                    trend_str = f"DECREASING ({trend_pct:+.2f}%)"
                elif trend_val > 0:
                    trend_str = f"INCREASING ({trend_pct:+.2f}%)"
                else:
                    trend_str = "FLAT"
            else:
                trend_str = "N/A (single point)"

            print(f"  {tag}")
            print(f"  Trend over last {len(last_n)} points: {trend_str}")
            print(f"  {'Step':>10}  {'Value':>14}")
            print(f"  {'-'*10}  {'-'*14}")
            for e in last_n:
                print(f"  {e.step:>10}  {format_value(e.value):>14}")
            print()

    # Summary statistics
    print("=" * 100)
    print("  TRAINING SUMMARY")
    print("=" * 100)
    print()

    # Find the best validation PPL (lowest)
    ppl_tags = [t for t in tags if "ppl" in t.lower() and "val" in t.lower()]
    for tag in sorted(ppl_tags):
        events = all_data[tag]
        if events:
            values = [e.value for e in events]
            steps = [e.step for e in events]
            best_val = min(values)
            best_step = steps[values.index(best_val)]
            last_val = values[-1]
            print(f"  {tag}")
            print(f"    Best: {format_value(best_val)} at step {best_step}")
            print(f"    Last: {format_value(last_val)} at step {steps[-1]}")
            print()

    # Find best validation loss (lowest)
    loss_tags = [t for t in tags if "loss" in t.lower() and "val" in t.lower()]
    for tag in sorted(loss_tags):
        events = all_data[tag]
        if events:
            values = [e.value for e in events]
            steps = [e.step for e in events]
            best_val = min(values)
            best_step = steps[values.index(best_val)]
            last_val = values[-1]
            print(f"  {tag}")
            print(f"    Best: {format_value(best_val)} at step {best_step}")
            print(f"    Last: {format_value(last_val)} at step {steps[-1]}")
            print()

    # Developability correlations
    dev_tags = [t for t in tags if "dev_" in t.lower() or "developability" in t.lower()]
    if dev_tags:
        print("  --- Developability Correlations ---")
        for tag in sorted(dev_tags):
            events = all_data[tag]
            if events:
                values = [e.value for e in events]
                steps = [e.step for e in events]
                best_val = max(values)  # higher correlation is better
                best_step = steps[values.index(best_val)]
                last_val = values[-1]
                print(f"  {tag}")
                print(f"    Best (max): {format_value(best_val)} at step {best_step}")
                print(f"    Last:       {format_value(last_val)} at step {steps[-1]}")
                print()

    # Training config summary from hparams
    print("  --- Hyperparameters (from hparams.yaml) ---")
    hparams = {
        "max_steps": 37000,
        "peak_lr": 1e-4,
        "batch_size": 256,
        "warmup_steps": 500,
        "mask_prob": 0.15,
        "model": "esm2_t12_35M_UR50D",
        "loss_type": "focal_loss",
        "dual_aa_heads": True,
        "use_alpha_gating": True,
        "use_cdr_loss_boosting": True,
        "use_region_balanced_loss": True,
    }
    for k, v in hparams.items():
        print(f"    {k}: {v}")

    print()
    print("=" * 100)
    print("  ANALYSIS COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()
