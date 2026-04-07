# Project Action Plan

**Date:** 2026-04-07

This document answers the four project-state questions from the clean-up review
and sets out the immediate next steps.

---

## Q1 — Do we need to implement CI/CD?

**Yes — and it is now in place.**

A `tests.yml` GitHub Actions workflow has been added that runs the full pytest
suite on every push and pull request to `main`.  The existing `extract_shot_data.yml`
workflow handles on-demand bulk extraction.

Together these give:

| Workflow | Trigger | Purpose |
|---|---|---|
| `tests.yml` | Push / PR to `main` | Automated unit-test gate |
| `extract_shot_data.yml` | Manual `workflow_dispatch` | Bulk-scrape new events |

No further CI/CD work is needed at this stage.  If/when a GNN training pipeline is
added, a separate `train.yml` triggered manually (or on schedule) would be appropriate.

---

## Q2 — Is Phase 1 (data scraping) sufficient?

**Largely yes, with two known gaps.**

### Current state

- 260 WCF events processed (2013–2026); men's, women's, mixed, and mixed doubles.
- Stone detection quality improvements are complete (dynamic house-radius, adaptive
  ring-colour thresholds, per-page stone-colour calibration).
- Ghost stone columns (displaced-stone origin positions) are captured and included in
  the output schema.

### Known gaps (not blocking training)

| Gap | Impact | Priority |
|---|---|---|
| **Incremental processing** — every run re-scrapes all ~260 events | High runtime cost when adding new tournaments | Medium |
| **Mixed-gender PDFs** — a small subset bundles men's and women's draws in one file | Those events get the wrong `gender` label | Low |

### Validation confidence

The schema is validated by the existing test suite (`test_extract_shot_data.py`,
`test_generate_board_image.py`) and visually via `generate_board_image.py`.  For a
higher-confidence audit, run `verify_stone_tracking.py` on the full parquet and inspect
the displacement histogram — a near-zero median displacement confirms stone detection is
internally consistent.

---

## Q3 — Is the stone tracking evaluation (Phase 2) sufficient?

**Partially.  The metric suite is good; the manual ground-truth check is still missing.**

### What is implemented

`track_stones.py` exports seven evaluation helpers covering complementary failure modes:

| Helper | Failure mode caught |
|---|---|
| `evaluate_tracking` | Overall new-stone rate vs. theoretical expectation |
| `displacement_distribution` | Tracking threshold calibration |
| `id_continuity_rate` | Stationary stones being re-IDed |
| `slot_swap_rate` | Two stones swapping IDs between shots |
| `cap_pressure_rate` | Borderline rejections at the distance cap |
| `displacement_symmetry` | Separation between still-stones and shot-stones |
| `stone_count_consistency_rate` | Upstream detection count anomalies |

`verify_stone_tracking.py` runs 7 structural checks (schema, first-shot invariant, ID
uniqueness/continuity, displacement plausibility, new-stone count) and optionally
produces visual board images with displacement arrows.

### What is still missing

| Option | Description | Status |
|---|---|---|
| **Option 1 — Manual ground truth** | Annotate 20–30 ends by hand; measure assignment accuracy | Not started |
| **Option 2 — Physics-based check** | Count shots where `new_IDs ≠ 1`; flagged as `delivery_anomaly_rate` | Not started |

**Recommendation for next sprint:**
Implement Option 2 first — it requires no human annotation and gives a per-shot binary
signal that can be computed in minutes.  Option 1 is the gold standard but is a
significant manual effort; schedule it as a one-time calibration once the tracker is
considered stable.

### Confidence level today

The displacement histogram (`near_zero_fraction > 0.90`, `threshold_zone_fraction < 0.05`)
combined with the `id_continuity_rate` and `slot_swap_rate` metrics provides strong
evidence that the greedy tracker is well-calibrated for the standard case.  Multi-stone
take-outs remain the unvalidated edge case; the Hungarian algorithm should handle these
better and can be selected via `--method hungarian`.

---

## Q4 — Is the data schema appropriate for a GNN (Phase 3)?

**Yes — the schema is now complete for training.**

All previously identified critical gaps have been addressed:

| Item | Status |
|---|---|
| `is_shot_stone` flag per stone slot | ✅ Added in `track_stones.py` |
| `shooting_team_is_team1` pre-computed | ✅ Added in `build_features.py` |
| `hammer_team_code`, `shooting_team_has_hammer`, score columns | ✅ Added in `build_features.py` |
| Ghost stone columns | ✅ Added in `extract_shot_data.py` |
| Stone IDs + `prev_x` / `prev_y` displacement columns | ✅ Added in `track_stones.py` |

The one remaining data-pipeline prerequisite before training is writing
`build_graph_dataset.py` — the bridge that converts the flat parquet into PyTorch
Geometric `Data` objects.  See `docs/gnn_data_readiness.md` for the full assessment.

---

## Immediate Next Steps

### Priority 1 — Tracking validation (1–2 days)

Run the full metric suite on the production parquet and record the results:

```bash
python track_stones.py --method greedy --evaluate --displacement
```

Document the output in `docs/tracking_metrics.md` (add a "Observed values" column).
If `spurious_id_rate > 0.02` or `id_continuity_rate < 0.95`, retune
`STONE_TRACK_MAX_DIST` and re-track before proceeding.

### Priority 2 — Physics-based delivery check (half day)

Implement Option 2 from `docs/tracking_evaluation.md` as a new helper
`delivery_anomaly_rate(tracked_df)` in `track_stones.py`.  Add it to the CLI
`--evaluate` output and to `verify_stone_tracking.py`.

### Priority 3 — `build_graph_dataset.py` (2–3 days)

Write the conversion script that turns each shot row into a PyTorch Geometric `Data`
object:

- Nodes: one per stone (both teams), features = `(x, y, dist, angle, team_flag, is_shot_stone)`
- Edges: fully-connected within each team; cross-team edges by proximity; temporal
  displacement edges using `stone_id` correspondence between `S_{t-1}` and `S_t`
- Graph-level features: `shot_number`, `end_number`, `score_diff_before`,
  `shooting_team_has_hammer`, `turn`
- Labels: `accuracy`, `shot_type`, `team1_score_this_end`, `team2_score_this_end`

Test with a small subset first (`output_test/shot_locations.parquet`).

### Priority 4 — GNN model skeleton (3–5 days)

Implement the Siamese GNN described in `docs/pipeline_overview.md`:

1. Two-branch GCN encoder (one branch for `S_{t-1}`, one for `S_t`)
2. Change embedding = concat or difference of the two branch outputs
3. Multi-task output heads: shot accuracy (MSE), shot type (cross-entropy), end score (MSE)
4. Multi-task loss with homoscedastic uncertainty weighting (Kendall 2018)
5. Training loop with event-level train/val/test split

Start with a simple 2-layer GCN; tune depth and hidden dimension after confirming the
pipeline works end-to-end.

### Priority 5 — Incremental scraping (1 day, can be done in parallel)

Add a `--skip-existing` flag to `extract_shot_data.py` that checks whether an event's
output already exists in the output directory before downloading and parsing its PDF.
This cuts re-scrape time from hours to minutes when only new events are added.
