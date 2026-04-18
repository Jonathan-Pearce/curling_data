"""CI evaluation script — runs stone detection accuracy checks across all
manifested PDFs and writes a combined metrics JSON file.

Usage (from repo root, with PYTHONPATH=src or after ``pip install -e .``):

    PYTHONPATH=src python ci_scripts/evaluate_all.py

Options
-------
--manifest PATH   Path to the PDF manifest JSON (default: tests/eval_manifest.json)
--pdf-dir  DIR    Directory containing the PDF files
                  (default: "example raw data")
--out      PATH   Output path for the combined metrics JSON
                  (default: ci_scripts/evaluation_results/metrics.json)
--max-ends N      Override the per-PDF max_ends for all PDFs (useful for
                  quick local runs: --max-ends 2)
"""

import argparse
import json
import os
import sys

# Allow running as ``python ci_scripts/evaluate_all.py`` from the repo root.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from scraping.evaluate_detection import evaluate_pdf, print_summary  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate stone detection accuracy for all PDFs in the manifest."
    )
    parser.add_argument(
        "--manifest",
        default=os.path.join(_REPO_ROOT, "tests", "eval_manifest.json"),
        metavar="PATH",
        help="Path to eval_manifest.json (default: tests/eval_manifest.json)",
    )
    parser.add_argument(
        "--pdf-dir",
        default=os.path.join(_REPO_ROOT, "example raw data"),
        metavar="DIR",
        help='Directory containing PDF files (default: "example raw data")',
    )
    parser.add_argument(
        "--out",
        default=os.path.join(_REPO_ROOT, "ci_scripts", "evaluation_results", "metrics.json"),
        metavar="PATH",
        help="Output path for combined metrics JSON",
    )
    parser.add_argument(
        "--max-ends",
        type=int,
        default=None,
        metavar="N",
        help="Override max_ends for all PDFs (per-manifest setting used when absent)",
    )
    args = parser.parse_args()

    with open(args.manifest, encoding="utf-8") as f:
        manifest = json.load(f)

    pdf_entries = manifest["pdfs"]
    all_results = []
    skipped = []

    for entry in pdf_entries:
        pdf_path = os.path.join(args.pdf_dir, entry["filename"])

        if not os.path.isfile(pdf_path):
            skipped.append(entry["filename"])
            print(f"  [SKIP] {entry['filename']} — not found in {args.pdf_dir!r}")
            continue

        max_ends = args.max_ends if args.max_ends is not None else entry.get("max_ends", 5)
        print(f"\n{'=' * 56}")
        print(f"  {entry['filename']}")
        print(f"  Event : {entry['event']}")
        print(f"  Max ends: {max_ends}")
        print(f"{'=' * 56}")

        summary, _ = evaluate_pdf(pdf_path, max_ends=max_ends, verbose=True)
        print_summary(summary)

        all_results.append({
            "filename": entry["filename"],
            "event": entry["event"],
            "year": entry["year"],
            "gender": entry["gender"],
            "type": entry["type"],
            **summary,
        })

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"results": all_results, "skipped": skipped}, f, indent=2)

    print(f"\n{'=' * 56}")
    print(f"  Evaluated : {len(all_results)} PDF(s)")
    print(f"  Skipped   : {len(skipped)} PDF(s) (not found)")
    if skipped:
        for s in skipped:
            print(f"              {s}")
    print(f"  Results written to: {args.out}")
    print(f"{'=' * 56}")

    if all_results:
        avg_f1 = sum(r["f1"] for r in all_results) / len(all_results)
        avg_prec = sum(r["precision"] for r in all_results) / len(all_results)
        avg_rec = sum(r["recall"] for r in all_results) / len(all_results)
        print(f"\n  Summary across {len(all_results)} PDF(s):")
        print(f"    Avg Precision : {avg_prec:.2f} %")
        print(f"    Avg Recall    : {avg_rec:.2f} %")
        print(f"    Avg F1        : {avg_f1:.2f} %")
        print()


if __name__ == "__main__":
    main()
