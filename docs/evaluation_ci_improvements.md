# Evaluation & CI Improvement Plan

**Date:** 2026-04-18

This document captures the review of evaluation methods and CI infrastructure for the
two upstream pipeline stages — **data scraping** (stone detection from PDFs) and **stone
tracking** — and the plan to improve them before GNN training begins.

---

## Current State

### Data Scraping

| Component | Status |
|---|---|
| Stone detection evaluation (`evaluate_detection.py`) | ✅ Implemented |
| Metrics: precision, recall, F1, position error (median/p95), miss rate | ✅ Implemented |
| Separate active-stone and ghost-ring evaluation | ✅ Implemented |
| CI evaluation manifest (10+ PDFs across years, genders, event types) | ✅ Implemented |
| `evaluate_all.py` + `print_metrics.py` producing a metrics JSON | ✅ Implemented |
| `evaluate-scraping` job in `ci.yml` uploading results as an artifact | ✅ Implemented |

**Key limitation:** PDFs are not committed to the repo, so the CI job always reports all
PDFs as `skipped`.  The infrastructure exists but produces no live signal on every push.

---

### Stone Tracking

| Component | Status |
|---|---|
| `evaluate_tracking` — aggregate headline metrics (spurious ID rate etc.) | ✅ Implemented |
| `displacement_distribution` — matched-link displacement statistics | ✅ Implemented |
| `id_continuity_rate` — stationary stone re-ID rate | ✅ Implemented |
| `slot_swap_rate` — physically impossible ID swaps | ✅ Implemented |
| `cap_pressure_rate` — near-threshold miss rate | ✅ Implemented |
| `displacement_symmetry` — shot-stone vs. still-stone separation | ✅ Implemented |
| `stone_count_consistency_rate` — detector anomaly signal | ✅ Implemented |
| `verify_stone_tracking.py` — 8 internal consistency checks | ✅ Implemented |
| Algorithm comparison: greedy vs. Hungarian (`compare_tracking`) | ✅ Implemented |
| **CI integration (any of the above run in CI)** | ❌ Missing |

All tracking metric functions exist and are unit-tested but are **never run against
real output data in CI**.

---

## Identified Gaps

### Gap 1 — Scraping CI metrics are computed but not enforced

The `evaluate-scraping` job is `continue-on-error: true` with no threshold comparison.
Even if metrics are excellent locally, a code regression that drops F1 by 10 points
would not be flagged.

### Gap 2 — Stone tracking has no CI integration

The `evaluate_tracking` family of functions exists, tested, and works — but no script
runs them on `output/shot_locations.parquet` in CI and no job reports them.

### Gap 3 — No historical metric tracking / trend visibility

Each CI run uploads an artifact but there is no way to see whether quality is improving
or degrading over time.  Metrics disappear in the artifact expiry window.

### Gap 4 — No ground-truth for tracking *correctness*

Displacement-based metrics confirm internal consistency but cannot verify the tracker
assigns the *correct* identity.  A tracker that re-IDs to the wrong previous position
can score perfectly on all self-consistent metrics.

### Gap 5 — `delivery_anomaly_rate` not yet implemented

`docs/tracking_evaluation.md` describes Option 2 (physics-based sanity check: each
non-first shot should create exactly one new stone ID) but the metric is not yet in
`track_stones.py`.

### Gap 6 — `print_metrics.py` shows only P/R/F1/ends

The `pos_err_p95_norm`, `ghost_precision`, and `ghost_recall` fields are already in the
metrics JSON but are not shown in the job log table.

---

## Implementation Plan

### Item 1 — `ci_scripts/evaluate_tracking.py` ✅ Done

CI script that reads `output/shot_locations.parquet` and runs the full metric suite:
`evaluate_tracking`, `displacement_distribution`, `id_continuity_rate`, `slot_swap_rate`,
`cap_pressure_rate`, `displacement_symmetry`, `stone_count_consistency_rate`,
`delivery_anomaly_rate`.  Writes `ci_scripts/evaluation_results/tracking_metrics.json`.
Exits gracefully (non-zero but continue-on-error) if the parquet is absent.

### Item 2 — `delivery_anomaly_rate` metric ✅ Done

Added to `track_stones.py`.  Counts the fraction of non-first shots where new stone IDs
created ≠ 1.  Accompanies `evaluate_tracking` in every evaluation report.

### Item 3 — Tracking evaluation CI job ✅ Done

New `evaluate-tracking` job in `ci.yml`.  Triggers on pushes/PRs that change
`src/tracking/**` or `output/shot_locations.parquet`.  Uploads
`tracking_metrics.json` as an artifact and writes a GitHub Step Summary table.

### Item 4 — GitHub Step Summary for both evaluation jobs ✅ Done

Both `evaluate-scraping` and `evaluate-tracking` now write Markdown tables directly
to `$GITHUB_STEP_SUMMARY` so metrics are visible on the PR/push check page without
downloading artifacts.

### Item 5 — Extended `print_metrics.py` table ✅ Done

`print_metrics.py` now shows `pos_err_p95`, `ghost_precision%`, `ghost_recall%`
columns in both plain-text and `--markdown` modes.

### Item 6 — Baseline JSON + regression check for scraping ✅ Done

`ci_scripts/evaluation_results/baseline_scraping.json` stores per-PDF baseline F1
and position-error values.  `ci_scripts/check_regression.py` compares the current
metrics run against the baseline and writes a regression report.  The evaluate-scraping
CI job calls this and writes the result to the step summary.

The baseline starts empty (no PDFs committed).  When PDFs are evaluated locally, run:

```bash
python ci_scripts/evaluate_all.py --out ci_scripts/evaluation_results/metrics.json
python ci_scripts/check_regression.py \
  --current  ci_scripts/evaluation_results/metrics.json \
  --baseline ci_scripts/evaluation_results/baseline_scraping.json \
  --update-baseline
```

---

## Future Work (not yet implemented)

### Ground-truth annotation sample

Annotate 5–10 ends by hand (covering at least one double take-out and one draw-heavy
end).  Store in `eval_gt/tracking_ground_truth.json`.  Add a ground-truth comparison
mode to `verify_stone_tracking.py` that measures assignment accuracy against the labels.
Even a small sample gives enough signal to detect systematic cap miscalibration or
swap errors near the threshold boundary.

### Data readiness report

A single `ci_scripts/data_readiness_report.py` that aggregates all evaluation signals
into a pass/fail table suitable as a final gate before `build_graph_dataset.py`:

| Section | Key metric | Threshold |
|---|---|---|
| Stone detection | Aggregate F1 | ≥ 97 % |
| Stone detection | Position error p95 | < 0.05 units |
| Ghost detection | F1 | ≥ 85 % |
| Stone tracking | `spurious_id_rate` | < 0.02 |
| Stone tracking | `id_continuity_rate` | ≥ 0.98 |
| Stone tracking | `slot_swap_rate` | = 0 |
| Stone tracking | `stone_count_consistency_rate` | ≥ 0.99 |
| Data completeness | Rows in parquet | > 500 k |
| Data completeness | NULL `stone_id` rate mid-end | < 5 % |
