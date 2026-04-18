"""Print a human-readable metrics table from the CI evaluation results JSON.

Usage:
    python ci_scripts/print_metrics.py ci_scripts/evaluation_results/metrics.json
"""

import json
import sys


def main():
    if len(sys.argv) < 2:
        print("Usage: print_metrics.py <metrics.json>")
        sys.exit(1)

    path = sys.argv[1]
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", [])
    skipped = data.get("skipped", [])

    if not results:
        print("No PDFs were evaluated (all skipped or no PDFs committed).")
        print("Add PDFs listed in tests/eval_manifest.json to 'example raw data/' to enable evaluation.")
        return

    print(f"\n{'PDF':<45} {'P%':>6} {'R%':>6} {'F1%':>6} {'ends':>5}")
    print("-" * 70)
    for r in results:
        print(
            f"{r['filename']:<45} "
            f"{r['precision']:>6.1f} "
            f"{r['recall']:>6.1f} "
            f"{r['f1']:>6.1f} "
            f"{r['ends_evaluated']:>5}"
        )

    if skipped:
        print(f"\nSkipped ({len(skipped)} not found): {', '.join(skipped)}")


if __name__ == "__main__":
    main()
