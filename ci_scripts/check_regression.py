"""Compare current stone-detection metrics against a stored baseline and report
regressions.

Usage
-----
    # Check for regressions (report only — exit 0 always):
    python ci_scripts/check_regression.py \\
        --current  ci_scripts/evaluation_results/metrics.json \\
        --baseline ci_scripts/evaluation_results/baseline_scraping.json

    # Check and write a Markdown report:
    python ci_scripts/check_regression.py \\
        --current  ci_scripts/evaluation_results/metrics.json \\
        --baseline ci_scripts/evaluation_results/baseline_scraping.json \\
        --markdown ci_scripts/evaluation_results/regression_report.md

    # Update the baseline with the current results (after manual verification):
    python ci_scripts/check_regression.py \\
        --current  ci_scripts/evaluation_results/metrics.json \\
        --baseline ci_scripts/evaluation_results/baseline_scraping.json \\
        --update-baseline

Regression rules
----------------
Per-PDF (when a baseline entry exists for that filename):
  - F1 dropped by more than ``max_f1_regression`` percentage points → REGRESSION
  - ghost_recall dropped by more than ``max_ghost_recall_regression`` points → REGRESSION

Absolute thresholds (applied to every evaluated PDF regardless of baseline):
  - F1 < ``min_f1`` → WARN
  - ghost_recall < ``min_ghost_recall`` → WARN
  - pos_err_p95_norm > ``max_pos_err_p95_norm`` → WARN

No regressions or warnings → exit 0.
Regressions or warnings found → prints report and exits 0 (report-only; never
blocks the build).
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def _fmt(val, decimals=1):
    if val is None or val != val:
        return "n/a"
    return f"{val:.{decimals}f}"


def check(current_path: str, baseline_path: str) -> tuple[list[str], list[str], list[str]]:
    """Compare current metrics against baseline.

    Returns
    -------
    regressions : list of str  — per-PDF regressions
    warnings    : list of str  — absolute threshold violations
    info        : list of str  — informational notes (new PDFs, baseline updates)
    """
    with open(current_path, encoding="utf-8") as f:
        current = json.load(f)
    with open(baseline_path, encoding="utf-8") as f:
        baseline = json.load(f)

    thresholds = baseline.get("thresholds", {})
    min_f1 = thresholds.get("min_f1", 97.0)
    max_pos_err = thresholds.get("max_pos_err_p95_norm", 0.05)
    min_ghost_recall = thresholds.get("min_ghost_recall", 85.0)
    max_f1_reg = thresholds.get("max_f1_regression", 2.0)
    max_ghost_reg = thresholds.get("max_ghost_recall_regression", 5.0)

    baseline_by_file = {r["filename"]: r for r in baseline.get("results", [])}
    current_results = current.get("results", [])

    regressions: list[str] = []
    warnings: list[str] = []
    info: list[str] = []

    for r in current_results:
        fname = r["filename"]
        f1 = r.get("f1")
        ghost_rec = r.get("ghost_recall")
        pos_err = r.get("pos_err_p95_norm")

        # Absolute threshold checks
        if f1 is not None and f1 < min_f1:
            warnings.append(
                f"  {fname}: F1 = {_fmt(f1)}% is below threshold {min_f1}%"
            )
        if ghost_rec is not None and ghost_rec < min_ghost_recall:
            warnings.append(
                f"  {fname}: ghost_recall = {_fmt(ghost_rec)}% "
                f"is below threshold {min_ghost_recall}%"
            )
        if pos_err is not None and pos_err > max_pos_err:
            warnings.append(
                f"  {fname}: pos_err_p95 = {_fmt(pos_err, 4)} "
                f"exceeds threshold {max_pos_err}"
            )

        # Per-PDF regression checks against baseline
        if fname not in baseline_by_file:
            info.append(f"  {fname}: no baseline entry — recording as new.")
            continue

        base = baseline_by_file[fname]
        base_f1 = base.get("f1")
        base_ghost = base.get("ghost_recall")

        if f1 is not None and base_f1 is not None:
            drop = base_f1 - f1
            if drop > max_f1_reg:
                regressions.append(
                    f"  {fname}: F1 dropped {drop:.1f} pp "
                    f"({_fmt(base_f1)}% → {_fmt(f1)}%)  [threshold: {max_f1_reg} pp]"
                )

        if ghost_rec is not None and base_ghost is not None:
            drop = base_ghost - ghost_rec
            if drop > max_ghost_reg:
                regressions.append(
                    f"  {fname}: ghost_recall dropped {drop:.1f} pp "
                    f"({_fmt(base_ghost)}% → {_fmt(ghost_rec)}%)  "
                    f"[threshold: {max_ghost_reg} pp]"
                )

    return regressions, warnings, info


def _markdown_report(
    regressions: list[str],
    warnings: list[str],
    info: list[str],
    current_path: str,
    baseline_path: str,
) -> str:
    lines = ["## Detection Regression Report\n"]

    if regressions:
        lines.append("### ⚠️ Regressions\n")
        for r in regressions:
            lines.append(f"- {r.strip()}")
        lines.append("")
    else:
        lines.append("### ✅ No Regressions\n")

    if warnings:
        lines.append("### ⚠️ Threshold Warnings\n")
        for w in warnings:
            lines.append(f"- {w.strip()}")
        lines.append("")

    if info:
        lines.append("### ℹ️ Info\n")
        for i in info:
            lines.append(f"- {i.strip()}")
        lines.append("")

    lines.append(
        f"_Current: `{os.path.basename(current_path)}`  ·  "
        f"Baseline: `{os.path.basename(baseline_path)}`_"
    )
    return "\n".join(lines) + "\n"


def update_baseline(current_path: str, baseline_path: str) -> None:
    """Merge current results into the baseline file."""
    with open(current_path, encoding="utf-8") as f:
        current = json.load(f)
    with open(baseline_path, encoding="utf-8") as f:
        baseline = json.load(f)

    baseline_by_file = {r["filename"]: r for r in baseline.get("results", [])}

    updated = 0
    for r in current.get("results", []):
        fname = r["filename"]
        baseline_by_file[fname] = {
            "filename": fname,
            "f1": r.get("f1"),
            "ghost_recall": r.get("ghost_recall"),
            "pos_err_p95_norm": r.get("pos_err_p95_norm"),
            "ends_evaluated": r.get("ends_evaluated"),
        }
        updated += 1

    baseline["results"] = list(baseline_by_file.values())
    with open(baseline_path, "w", encoding="utf-8") as f:
        json.dump(baseline, f, indent=2)
    print(f"Baseline updated: {updated} PDF entries written to {baseline_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare current detection metrics against a stored baseline."
    )
    parser.add_argument("--current", required=True, metavar="PATH",
                        help="Path to current metrics.json")
    parser.add_argument("--baseline", required=True, metavar="PATH",
                        help="Path to baseline_scraping.json")
    parser.add_argument("--markdown", metavar="FILE", default=None,
                        help="Write Markdown report to FILE (use - for stdout).")
    parser.add_argument("--update-baseline", action="store_true",
                        help="Merge current results into the baseline file.")
    args = parser.parse_args()

    if not os.path.isfile(args.current):
        print(f"Current metrics file not found: {args.current}")
        sys.exit(0)

    if args.update_baseline:
        update_baseline(args.current, args.baseline)
        return

    regressions, warnings, info = check(args.current, args.baseline)

    if regressions:
        print("\n⚠️  REGRESSIONS DETECTED:")
        for r in regressions:
            print(r)
    else:
        print("\n✅  No regressions detected.")

    if warnings:
        print("\n⚠️  THRESHOLD WARNINGS:")
        for w in warnings:
            print(w)

    if info:
        print("\nℹ️  Info:")
        for i in info:
            print(i)

    if args.markdown:
        md = _markdown_report(regressions, warnings, info, args.current, args.baseline)
        if args.markdown == "-":
            print(md)
        else:
            with open(args.markdown, "w", encoding="utf-8") as f:
                f.write(md)
            print(f"\nMarkdown report written to {args.markdown}")


if __name__ == "__main__":
    main()
