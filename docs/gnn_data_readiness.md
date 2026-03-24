# GNN Data Readiness Assessment

**Date:** 2026-03-24

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

---

## Critical Gaps — Address Before Training

### 1. `is_shot_stone` flag is missing

**Priority: High**

The stone that was just delivered is the single most informative node in every graph.
With tracking in place it is fully derivable: at `shot_number > 1`, the stone with
`prev_x = NaN` that was absent from shot $N-1$ is the delivered stone. `apply_tracking()`
already computes this implicitly — it just does not write a flag column.

Edge cases to handle:
- `shot_number = 1`: all stones have `prev_x = NaN` (first delivery of the end). Store
  `NaN` / `None` to indicate ambiguity rather than marking every stone as the shot stone.
- Take-out: simultaneously produces one new stone (delivered) and removes one (hit out).
  The delivered stone is identifiable as the one with no matching stone in `prev_state`
  *after* the outgoing stone has been accounted for.

Without this flag, the GNN has to infer which stone was just thrown from the board state
alone, which adds noise to every training example.

**Implementation:** Add `team{N}_stone{S}_is_shot_stone` columns (bool/NaN) in
`track_stones.py` during `apply_tracking()`.

---

### 2. `shooting_team_is_team1` is not pre-computed

**Priority: High**

The GNN needs to encode each stone as either **"shooting team"** or **"opponent"** — a
more natural and permutation-stable node feature than an absolute `team1`/`team2` label.
Resolving it requires joining `team_code` (in `shot_locations`) against `team1_code` /
`team2_code` (in `ends.csv`). This join is not pre-computed.

**Implementation:** Add a boolean column `shooting_team_is_team1` to `shot_locations`
during the feature enrichment step; `True` when `team_code == team1_code` for that end.

---

### 3. Game-context columns are not in `shot_locations`

**Priority: High**

`hammer_team_code`, `team1_score_before`, `team2_score_before` are in `ends.csv` only.
Hammer determines the strategy of almost every shot in curling and is a critical input.
Requiring a join at training time is friction and a source of bugs (wrong key, forgetting
conceded-end handling, etc.).

**Columns to pre-join onto `shot_locations`:**
- `hammer_team_code` — which team has last-stone advantage this end
- `shooting_team_has_hammer` — derived bool: `team_code == hammer_team_code`
- `score_diff_before` — derived: `team1_score_before - team2_score_before`
- `team1_score_before`, `team2_score_before` — raw scores for completeness

**Implementation:** Add a `build_features.py` (or extend `track_stones.py`) step that
reads both `shot_locations_raw.parquet` and `ends.csv`, joins the four columns above,
and writes the enriched `shot_locations.parquet`.

---

### 4. Tracking quality under take-outs is unvalidated

**Priority: High**

`stone_id` columns are what temporal GNN edges are built on. ID fragmentation under
double/triple take-outs (where two or three stones move simultaneously) is unknown. The
displacement distribution diagnostic gives signal about threshold calibration but does not
confirm ID correctness under multi-stone events.

Incorrect IDs produce incorrect temporal edges — training examples where the "same stone"
at $t-1$ and $t$ is actually a different physical stone, silently poisoning the training
signal.

**Implementation:** Run `python track_stones.py --displacement` on the real dataset and
inspect the distribution. If the threshold zone fraction exceeds 5%, retune
`STONE_TRACK_MAX_DIST` before proceeding. See `docs/tracking_evaluation.md` for full
diagnostic procedure.

---

## Medium Priority

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

| Step | Task | Where |
|:---:|---|---|
| 1 | Add `is_shot_stone` flag column per stone slot | `track_stones.py` / `apply_tracking()` |
| 2 | Pre-join `shooting_team_is_team1`, `shooting_team_has_hammer`, `score_diff_before`, raw scores onto `shot_locations` | New `build_features.py` or extend `track_stones.py` |
| 3 | Run displacement distribution diagnostic on real dataset; retune `STONE_TRACK_MAX_DIST` if needed | CLI: `python track_stones.py --displacement` |
| 4 | Write `build_graph_dataset.py` — converts enriched parquet into PyG `InMemoryDataset` | New file |

Steps 1 and 2 are data pipeline changes. Step 3 is a one-time calibration run. Step 4 is
the bridge into model training.
