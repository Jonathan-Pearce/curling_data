"""Print a human-readable metrics table from the CI evaluation results JSON.

Usage:
    python ci_scripts/print_metrics.py ci_scripts/evaluation_results/metrics.json
    python ci_scripts/print_metrics.py ci_scripts/evaluation_results/metrics.json --markdown
"""

import json
import sys


def _fmt(val, fmt=".1f", fallback="n/a"):
    if val is None or val != val:  # None or NaN
        return fallback
    return format(val, fmt)


def print_plain(results, skipped):
    header = (
        f"{'PDF':<45} {'P%':>6} {'R%':>6} {'F1%':>6} {'ends':>5} "
        f"{'p95err':>7} {'GhostP%':>8} {'GhostR%':>8}"
    )
    print(f"\n{header}")
    print("-" * 95)
    for r in results:
        print(
            f"{r['filename']:<45} "
            f"{_fmt(r.get('precision')):>6} "
            f"{_fmt(r.get('recall')):>6} "
            f"{_fmt(r.get('f1')):>6} "
            f"{r.get('ends_evaluated', 0):>5} "
            f"{_fmt(r.get('pos_err_p95_norm'), '.4f'):>7} "
            f"{_fmt(r.get('ghost_precision')):>8} "
            f"{_fmt(r.get('ghost_recall')):>8}"
        )
    if skipped:
        print(f"\nSkipped ({len(skipped)} not found): {', '.join(skipped)}")


def print_markdown(results, skipped):
    lines = ["## Stone Detection Accuracy\n"]
    lines.append(
        "| PDF | P% | R% | F1% | Ends | Pos Err p95 | Ghost P% | Ghost R% |"
    )
    lines.append("| --- | --: | --: | --: | ---: | --: | --: | --: |")
    for r in results:
        lines.append(
            f"| {r['filename']} "
            f"| {_fmt(r.get('precision'))} "
            f"| {_fmt(r.get('recall'))} "
            f"| {_fmt(r.get('f1'))} "
            f"| {r.get('ends_evaluated', 0)} "
            f"| {_fmt(r.get('pos_err_p95_norm'), '.4f')} "
            f"| {_fmt(r.get('ghost_precision'))} "
            f"| {_fmt(r.get('ghost_recall'))} |"
        )
    if skipped:
        lines.append(
            f"\n_Skipped ({len(skipped)} PDFs not found in `example raw data/`)_"
        )
    print("\n".join(lines))


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Print detection metrics table from CI evaluation JSON."
    )
    parser.add_argument("metrics_json", help="Path to metrics.json")
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Output GitHub-flavoured Markdown (for step summary).",
    )
    args = parser.parse_args()

    with open(args.metrics_json, encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", [])
    skipped = data.get("skipped", [])

    if not results:
        msg = (
            "No PDFs were evaluated (all skipped or no PDFs committed).\n"
            "Add PDFs listed in tests/eval_manifest.json to 'example raw data/' "
            "to enable evaluation."
        )
        print(msg)
        return

    if args.markdown:
        print_markdown(results, skipped)
    else:
        print_plain(results, skipped)


if __name__ == "__main__":
    main()
