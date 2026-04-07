# GNN Data Readiness Assessment

**Date:** 2026-04-07

Assessment of the current data pipeline state against the requirements of a Graph Neural
Network (GNN) model for curling shot prediction and evaluation.

---

## What's Solid

| Feature | Status |
|---|---|
| Board state (x, y, dist, angle per stone, both teams, all 16 shots) | ✅ Ready |
| Normalised coordinates (house-radius units, consistent orientation) | ✅ Ready |
| Stone IDs + displacement vectors after `track_stones.py` | ✅ Ready |
| Shot metadata (shot_type, accuracy, turn) as prediction targets | ✅ Ready |
| Parquet format, event-level train/val/test split boundary defined | ✅ Ready |
| Separation of raw detection from tracking (algorithms are swappable) | ✅ Ready |
| `is_shot_stone` flag per stone slot | ✅ Ready (`track_stones.py`) |
| `shooting_team_is_team1` pre-computed | ✅ Ready (`build_features.py`) |
| `hammer_team_code`, `shooting_team_has_hammer`, score columns pre-joined | ✅ Ready (`build_features.py`) |
| Ghost stone columns (origin positions of displaced stones) | ✅ Ready (`extract_shot_data.py`) |

---

## Remaining Gap — Address Before Training

### 1. Tracking quality under take-outs is unvalidated

**Priority: High**

`stone_id` columns are what temporal GNN edges are built on. ID fragmentation under
double/triple take-outs (where two or three stones move simultaneously) is unknown. The
displacement distribution diagnostic gives signal about threshold calibration but does not
confirm ID correctness under multi-stone events.

Incorrect IDs produce incorrect temporal edges — training examples where the "same stone"
at $t-1$ and $t$ is actually a different physical stone, silently poisoning the training
signal.

**Implementation:** Run the full suite of evaluation helpers from `track_stones.py`:

```python
from track_stones import (
    evaluate_tracking, displacement_distribution,
    id_continuity_rate, slot_swap_rate, cap_pressure_rate,
    displacement_symmetry, stone_count_consistency_rate,
)
```

If `spurious_id_rate > 0.02` or `id_continuity_rate < 0.95`, retune `STONE_TRACK_MAX_DIST`
before proceeding. See `docs/tracking_evaluation.md` and `docs/tracking_metrics.md`.

---

## Medium Priority

### 2. Distance-sorted slots will confuse temporal learning if not handled carefully

Stones within each team are sorted by distance in every shot row. The same physical stone
can appear in slot `team1_stone1` on shot 5 and `team1_stone3` on shot 6. The `_id`
columns are the only ground truth for correspondence.

Any graph construction code that row-shifts the flat table to construct $S_{t-1}$ without
first aligning by `stone_id` will silently produce wrong displacement vectors. This must
be documented prominently in `build_graph_dataset.py` and enforced by construction
(always iterate by stone ID, not by slot index).

### 3. No graph export format exists

The pipeline produces a flat parquet table. To feed a GNN (PyTorch Geometric / DGL)
you need:
- Node feature matrices per sample
- Edge index tensors per sample
- Graph-level feature vectors per sample
- A `Dataset` / `DataLoader` class

A `build_graph_dataset.py` step is a meaningful prerequisite before model training.
Edge construction strategy, virtual node design, and feature normalisation choices all
belong here.

---

## What Can Be Deferred to Training Code

These do not require changes to the scraping or tracking pipeline:

| Item | Where to handle |
|---|---|
| Edge construction (spatial proximity, team membership, temporal links) | `build_graph_dataset.py` |
| Ring zone categorical features (button / 4-ft / 8-ft / 12-ft) — derivable from `dist` | Graph construction |
| Feature standardisation / normalisation | Training transforms |
| Class imbalance in `shot_type` (weighted loss, oversampling) | Training loop |
| Player historical statistics | Separate data collection |
| Clock pressure features — only 74/260 events have time data | Optional modality with masking |

---

## Recommended Implementation Order

| Step | Task | Where |
|:---:|---|---|
| 1 | Run full tracking evaluation on the production dataset; retune cap if needed | CLI: `python track_stones.py --evaluate --displacement` |
| 2 | Write `build_graph_dataset.py` — converts enriched parquet into PyG `InMemoryDataset` | New file |
| 3 | Implement GNN model, multi-task heads, training loop | New `ml/` package |
