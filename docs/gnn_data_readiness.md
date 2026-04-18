# GNN Data Readiness Assessment

**Date:** 2026-04-18

Assessment of the current data pipeline state against the requirements of a Graph Neural
Network (GNN) model for curling shot prediction and evaluation.

---

## What's Solid

| Feature | Status |
|---|---|
| Board state (x, y, dist, angle per stone, both teams, all 16 shots) | ✅ Ready |
| Normalised coordinates (house-radius units, consistent orientation) | ✅ Ready |
| Stone IDs + displacement vectors after `track_stones.py` | ✅ Ready |
| `is_shot_stone` flag per stone slot | ✅ Ready |
| Ghost-to-stone ID matching (`team{N}_ghost{K}_stone_id`) | ✅ Ready |
| Game-context columns (`shooting_team_has_hammer`, `score_diff_before`, etc.) | ✅ Ready |
| Shot metadata (shot_type, accuracy, turn) as prediction targets | ✅ Ready |
| Parquet format, event-level train/val/test split boundary defined | ✅ Ready |
| Separation of raw detection from tracking (algorithms are swappable) | ✅ Ready |

---

## Critical Gaps — Address Before Training

### 1. `is_shot_stone` flag ✅ Implemented

`apply_tracking()` in `track_stones.py` writes `team{N}_stone{S}_is_shot_stone` for
every stone slot.  `True` = this stone was just delivered; `False` = carried from the
previous shot; `NaN` = ambiguous (first shot of end, or zero/multiple newly placed
stones in one shot).

### 2. `shooting_team_is_team1` ✅ Implemented

`build_features.py` pre-joins `shooting_team_is_team1`, `shooting_team_has_hammer`,
`score_diff_before`, and raw scores from `ends.csv` onto `shot_locations.parquet`.

### 3. Game-context columns ✅ Implemented

See §2.  All four columns required by the GNN are now pre-joined.

---

## Medium Priority

### 4. Ghost-to-stone matching ✅ Implemented

`apply_tracking()` now writes `team{N}_ghost{K}_stone_id` for every ghost slot when
ghost coordinate columns are present in the input DataFrame.  The match pool for each
ghost is the full `prev_state` snapshot from the preceding shot (covers both
still-in-play displaced stones and knocked-out stones).  Matching distance threshold:
`GHOST_MATCH_MAX_DIST = 0.20` normalised units.  `NaN` when no prior state exists
(first shot of end) or when no previous stone falls within the threshold.

### 5. Distance-sorted slots will confuse temporal learning if not handled carefully

Stones within each team are sorted by distance in every shot row. The same physical stone
can appear in slot `team1_stone1` on shot 5 and `team1_stone3` on shot 6. The `_id`
columns are the only ground truth for correspondence.

Any graph construction code that row-shifts the flat table to construct $S_{t-1}$ without
first aligning by `stone_id` will silently produce wrong displacement vectors. This must
be documented prominently in `build_graph_dataset.py` and enforced by construction
(always iterate by stone ID, not by slot index).

### 6. No graph export format exists

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

| Step | Task | Status | Where |
|:---:|---|---|---|
| 1 | Add `is_shot_stone` flag column per stone slot | ✅ Done | `track_stones.py` / `apply_tracking()` |
| 2 | Pre-join `shooting_team_is_team1`, `shooting_team_has_hammer`, `score_diff_before`, raw scores onto `shot_locations` | ✅ Done | `build_features.py` |
| 3 | Add `ghost_stone_id` matching — link ghost rings to the stone that was displaced | ✅ Done | `track_stones.py` / `apply_tracking()` |
| 4 | Run displacement distribution diagnostic on real dataset; retune `STONE_TRACK_MAX_DIST` if needed | ⬜ Pending | CLI: `python track_stones.py --displacement` |
| 5 | Write `build_graph_dataset.py` — converts enriched parquet into PyG `InMemoryDataset` | ⬜ Pending | New file |

Steps 1–3 are data pipeline changes and are complete.  Step 4 is a one-time calibration
run.  Step 5 is the bridge into model training.
