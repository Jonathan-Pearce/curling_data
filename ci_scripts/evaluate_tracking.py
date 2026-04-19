"""CI evaluation script — runs stone tracking quality metrics on the full
output parquet and writes a combined metrics JSON file.

Usage (from repo root):

    python ci_scripts/evaluate_tracking.py

Options
-------
--input  PATH   Path to the tracked shot-locations parquet
                (default: output/shot_locations.parquet)
--out    PATH   Output path for the metrics JSON
                (default: ci_scripts/evaluation_results/tracking_metrics.json)

Exit codes
----------
0   Metrics written successfully (or input file absent — handled gracefully).
1   Input file absent (with --require-input flag).
2   Internal error during metric computation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

import pandas as pd  # noqa: E402

from tracking.track_stones import (  # noqa: E402
    cap_pressure_rate,
    delivery_anomaly_rate,
    displacement_distribution,
    displacement_symmetry,
    evaluate_tracking,
    id_continuity_rate,
    slot_swap_rate,
    stone_count_consistency_rate,
)


def _run_metrics(df: pd.DataFrame) -> dict:
    """Compute the full metric suite and return a flat dict."""
    metrics: dict = {}

    metrics.update(evaluate_tracking(df))

    disp = displacement_distribution(df)
    # Exclude the raw numpy array — not JSON-serialisable
    metrics.update({k: v for k, v in disp.items() if k != "displacements"})

    metrics.update(id_continuity_rate(df))
    metrics.update(slot_swap_rate(df))
    metrics.update(cap_pressure_rate(df))
    metrics.update(displacement_symmetry(df))
    metrics.update(stone_count_consistency_rate(df))
    metrics.update(delivery_anomaly_rate(df))

    return metrics


def _print_summary(metrics: dict) -> None:
    """Print a human-readable summary table to stdout."""
    # ── Headline metrics ──────────────────────────────────────────────────
    thresholds = {
        "spurious_id_rate":            ("≤ 0.02", lambda v: v <= 0.02),
        "id_continuity_rate":          ("≥ 0.98", lambda v: v >= 0.98),
        "slot_swap_rate":              ("= 0.0",  lambda v: v == 0.0),
        "stone_count_consistency_rate":("≥ 0.99", lambda v: v >= 0.99),
        "delivery_anomaly_rate":       ("≤ 0.10", lambda v: v <= 0.10),
        "cap_pressure_rate":           ("≤ 0.10", lambda v: v <= 0.10),
        "threshold_zone_fraction":     ("≤ 0.05", lambda v: v <= 0.05),
        "near_zero_fraction":          ("≥ 0.90", lambda v: v >= 0.90),
        "still_stone_p95_displacement":("< cap",  lambda v: v < 0.13),
        "separation_ratio":            ("≥ 5",    lambda v: v >= 5),
    }

    print(f"\n{'Metric':<40} {'Value':>10}  {'Threshold':>10}  Status")
    print("-" * 75)

    for key, (threshold_str, check) in thresholds.items():
        val = metrics.get(key)
        if val is None or val != val:  # None or NaN
            status = "—"
            val_str = "n/a"
        else:
            status = "OK" if check(val) else "WARN"
            val_str = f"{val:.4f}"
        print(f"  {key:<38} {val_str:>10}  {threshold_str:>10}  {status}")

    print()
    # ── Size info ─────────────────────────────────────────────────────────
    for key in ("total_ends", "total_stone_appearances", "total_links",
                "total_shot_pairs", "total_non_first_shots"):
        if key in metrics:
            print(f"  {key}: {metrics[key]:,}")

    # ── Anomaly examples (delivery_anomaly_rate) ──────────────────────────
    examples = metrics.get("anomaly_examples", [])
    if examples:
        print(f"\n  Sample delivery anomalies (first {len(examples)}):")
        for ex in examples:
            print(
                f"    event={ex['event_id']} match={ex['match_id']} "
                f"end={ex['end_number']} shot={ex['shot_number']} "
                f"new_ids={ex['new_ids_created']}"
            )


def _print_markdown(metrics: dict) -> str:
    """Return a GitHub Step Summary compatible Markdown string."""
    lines = ["## Stone Tracking Quality\n"]

    # Headline table
    thresholds = [
        ("spurious_id_rate",             "≤ 0.02", lambda v: v <= 0.02),
        ("id_continuity_rate",           "≥ 0.98", lambda v: v >= 0.98),
        ("slot_swap_rate",               "= 0.0",  lambda v: v == 0.0),
        ("stone_count_consistency_rate", "≥ 0.99", lambda v: v >= 0.99),
        ("delivery_anomaly_rate",        "≤ 0.10", lambda v: v <= 0.10),
        ("cap_pressure_rate",            "≤ 0.10", lambda v: v <= 0.10),
        ("threshold_zone_fraction",      "≤ 0.05", lambda v: v <= 0.05),
        ("near_zero_fraction",           "≥ 0.90", lambda v: v >= 0.90),
        ("still_stone_p95_displacement", "< 0.13", lambda v: v < 0.13),
        ("separation_ratio",             "≥ 5",    lambda v: v >= 5),
    ]

    lines.append("| Metric | Value | Threshold | Status |")
    lines.append("| --- | ---: | ---: | :---: |")

    for key, threshold_str, check in thresholds:
        val = metrics.get(key)
        if val is None or val != val:  # None or NaN
            status = "—"
            val_str = "n/a"
        else:
            status = "✅" if check(val) else "⚠️"
            val_str = f"{val:.4f}"
        lines.append(f"| `{key}` | {val_str} | {threshold_str} | {status} |")

    # Volume row
    ends = metrics.get("total_ends", "n/a")
    links = metrics.get("total_links", "n/a")
    lines.append(f"\n_Ends: {ends:,}  ·  Matched links: {links:,}_"
                 if isinstance(ends, int) else
                 f"\n_Ends: {ends}  ·  Matched links: {links}_")

    # Displacement detail
    lines.append("\n### Displacement Distribution\n")
    lines.append("| Metric | Value |")
    lines.append("| --- | ---: |")
    for key in ("median_displacement", "p95_displacement",
                "near_zero_fraction", "threshold_zone_fraction"):
        val = metrics.get(key)
        val_str = f"{val:.4f}" if val is not None and val == val else "n/a"
        lines.append(f"| `{key}` | {val_str} |")

    # Delivery anomaly examples
    examples = metrics.get("anomaly_examples", [])
    if examples:
        lines.append(f"\n### Sample Delivery Anomalies (first {len(examples)})\n")
        lines.append("| event_id | match_id | end | shot | new_ids |")
        lines.append("| --- | --- | --- | --- | --- |")
        for ex in examples:
            lines.append(
                f"| {ex['event_id']} | {ex['match_id']} | "
                f"{ex['end_number']} | {ex['shot_number']} | "
                f"{ex['new_ids_created']} |"
            )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate stone tracking quality metrics for CI."
    )
    parser.add_argument(
        "--input",
        default=os.path.join(_REPO_ROOT, "output", "shot_locations.parquet"),
        metavar="PATH",
        help="Path to tracked shot_locations.parquet (default: output/shot_locations.parquet)",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(
            _REPO_ROOT, "ci_scripts", "evaluation_results", "tracking_metrics.json"
        ),
        metavar="PATH",
        help="Output path for metrics JSON",
    )
    parser.add_argument(
        "--require-input",
        action="store_true",
        help="Exit with code 1 if the input file is absent (default: skip gracefully).",
    )
    parser.add_argument(
        "--markdown",
        metavar="FILE",
        default=None,
        help="Also write a Markdown summary to FILE (use - for stdout).",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        msg = f"Input file not found: {args.input}"
        print(msg)
        if args.require_input:
            sys.exit(1)
        print("Skipping tracking evaluation (shot_locations.parquet not committed).")
        sys.exit(0)

    print(f"Reading {args.input} …")
    df = pd.read_parquet(args.input)
    print(f"  {len(df):,} rows loaded.")

    print("\nComputing tracking metrics …")
    try:
        metrics = _run_metrics(df)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR computing metrics: {exc}")
        sys.exit(2)

    _print_summary(metrics)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=lambda o: None)
    print(f"\nMetrics written to {args.out}")

    if args.markdown:
        md = _print_markdown(metrics)
        if args.markdown == "-":
            print(md)
        else:
            with open(args.markdown, "w", encoding="utf-8") as f:
                f.write(md)
            print(f"Markdown summary written to {args.markdown}")


if __name__ == "__main__":
    main()
