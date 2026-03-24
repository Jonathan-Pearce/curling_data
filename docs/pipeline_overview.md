# ML Pipeline Overview

**Date:** 2026-03-24

End-to-end description of the three-phase machine learning pipeline for curling shot
analysis.  The goal is to train a Graph Neural Network (GNN) that, given the board state
before and after a shot, can predict shot type, shot accuracy, and end scores.

---

## Phase 1 — Raw Data Extraction

### What it does

Extracts structured shot-by-shot data from curling tournament PDF result books hosted on
[curlit.com](https://curlit.com/results).  PDFs are fetched at runtime — no large files
are stored in the repository.

### Scripts

| Script | Role |
|---|---|
| `scrape_results.py` | Scrapes `curlit.com/results` to build an index of all tournament PDF URLs |
| `extract_shot_data.py` | Downloads each PDF, renders pages at 300 DPI, and parses text + board images |
| `generate_board_image.py` | Reconstructs board diagrams from extracted data for QA comparison |

### How extraction works

1. **URL scraping** — `scrape_results.py` fetches the curlit.com results index and writes
   `output/result_urls.csv` with one row per tournament (name, year, location, gender,
   PDF URL).

2. **PDF processing** — `extract_shot_data.py` processes each result book:
   - Uses **pdfplumber** to extract text: match metadata, end scores, shot metadata
     (player, shot type, turn direction, accuracy percentage).
   - Renders each board diagram as a 300 DPI image using **pdfplumber** + **Pillow**.
   - Detects stone positions via **OpenCV** colour-based segmentation: red stones → team1,
     yellow stones → team2.
   - Normalises detected stone positions to house-radius units (1 unit = 6 ft = the 12-ft
     ring radius).  The house centre is placed at `(0, 0)`.

3. **Outputs** — Six CSV files are written to `output/`:

   | File | Grain | Key columns |
   |---|---|---|
   | `events.csv` | One row per tournament | `event_id`, `year`, `gender` |
   | `matches.csv` | One row per match | `event_id`, `match_id`, round, teams |
   | `teams.csv` | One row per team | `team_code`, country/name |
   | `players.csv` | One row per player | `player_id`, name |
   | `ends.csv` | One row per end | scores, hammer, thinking time |
   | `shot_locations_raw.csv` | One row per shot | stone x/y positions (raw, no tracking) |

   At this stage the shot table has stone position columns but **no stone IDs or
   previous-position columns** — those are added in Phase 2.

### Current state

- **260 events** processed (2013 – present); covers men's, women's, mixed, and mixed
  doubles World Curling Federation events.
- Stone detection uses hardcoded colour thresholds calibrated for the curlit.com PDF
  rendering.  Detection is solid for the standard board diagram format but has no
  fallback for layout variants.
- The raw output `shot_locations_raw.csv` / `shot_locations_raw.parquet` is the handoff
  point to Phase 2.

---

## Phase 2 — Data Transformation and ML Preparation

### What it does

Transforms the raw extracted data into a clean, enriched parquet file that is the direct
input to GNN training.  This phase covers stone tracking, data validation, and feature
engineering.

### Scripts

| Script | Role |
|---|---|
| `track_stones.py` | Assigns stable stone IDs within each end and populates `prev_x` / `prev_y` displacement columns |
| `build_features.py` | Joins game-context columns from `ends.csv` onto the shot table |
| `verify_stone_tracking.py` | Post-tracking validation: checks ID stability, displacement distributions, visual spot-checks |

### Step 2a — Stone tracking (`track_stones.py`)

The raw data records stone positions per shot but has no concept of identity — the same
physical stone may appear in a different slot on the next shot.  `track_stones.py` adds:

- `team{N}_stone{S}_id` — a stable integer ID for each physical stone, consistent across
  all shots in an end (resets at end boundaries).
- `team{N}_stone{S}_prev_x` / `_prev_y` — the position of the same stone at the previous
  shot (NULL if the stone was newly placed this shot, or on `shot_number = 1`).

Two tracking algorithms are implemented and can be compared:

| Algorithm | Description |
|---|---|
| `greedy` | Greedy nearest-neighbour: sort all candidate (prev → curr) pairs by distance, assign in order.  O(n²) per shot. |
| `hungarian` | Minimum-cost bipartite matching via the Hungarian algorithm.  Globally optimal assignment under the same distance cap.  O(n³) per shot; negligible in practice since n ≤ 8. |

The distance cap is `STONE_TRACK_MAX_DIST = 0.13` normalised units.  A stone that cannot
be matched to any previous stone within this threshold is treated as a new placement and
gets a fresh ID.

**Output:** `output/shot_locations.parquet` (tracked; no game-context columns yet).

### Step 2b — Feature enrichment (`build_features.py`)

Joins the following game-context columns from `ends.csv` onto the shot table so that
downstream training code does not need to manage its own merge:

| Column | Source | Notes |
|---|---|---|
| `shooting_team_is_team1` | derived | `True` when `team_code == team1_code` for that end |
| `hammer_team_code` | `ends.csv` | Which team holds last-stone advantage |
| `shooting_team_has_hammer` | derived | `True` when `team_code == hammer_team_code` |
| `team1_score_before` | `ends.csv` | Cumulative score entering this end |
| `team2_score_before` | `ends.csv` | Cumulative score entering this end |
| `score_diff_before` | derived | `team1_score_before - team2_score_before` |

**Output:** `output/shot_locations.parquet` overwritten in-place with the enriched schema.

### Step 2c — Validation (`verify_stone_tracking.py`)

Runtime checks run after tracking to catch regressions:

- Stone ID stability within ends (no duplicates, no gaps).
- Displacement distribution — identifies stones near the tracking threshold
  (`STONE_TRACK_MAX_DIST`) that may be incorrectly assigned.
- Visual spot-check: renders board diagrams from extracted data and compares against
  the source PDF for selected ends.

### Data quality constraints

These constraints must be applied when constructing training samples:

| Constraint | Reason |
|---|---|
| Exclude `gender = 'mxd'` events | Mixed doubles events have `ends.csv` rows but no `shot_locations` rows |
| Exclude conceded ends (`score_this_end = "X"`) | Score label is not numeric; board state may be partial |
| Exclude rows with zero total stones (`team1_stones_in_play + team2_stones_in_play == 0`) | No graph can be constructed |
| `shot_number = 1` has `prev_x = NULL` for all stones | No preceding board state; treat S_{t-1} as an empty graph |
| Stone slots are distance-sorted, not ID-stable | Always align by `stone_id` when constructing displacement vectors; never row-shift slot indices |

### Current state

- Stone tracking is implemented for both `greedy` and `hungarian` algorithms.
- `build_features.py` is implemented and tested; the enriched parquet is the current
  state of `output/shot_locations.parquet`.
- **Known gap:** `is_shot_stone` flag columns are not yet added.  These would identify
  which stone was just delivered — the single most informative node in each training
  graph.  See `docs/gnn_data_readiness.md` for implementation notes.
- **Known gap:** Tracking quality under multi-stone take-outs (double/triple take-outs)
  is unvalidated.  ID fragmentation in these cases is unknown and could silently
  corrupt temporal GNN edges.
- Usable training data (4-person events, after filters): ~185 events, ~3,800 matches,
  ~26,000 ends, ~415,000 shots.

---

## Phase 3 — GNN Training

### What it does

Trains a Graph Neural Network on the enriched `shot_locations.parquet` to predict:

| Head | Target | Type | Loss |
|---|---|---|---|
| Shot type | `shot_type` | Categorical | Cross-entropy |
| Shot accuracy | `accuracy` | Float 0–100 | MSE |
| End score (team1) | `team1_score_this_end` | Int 0–8+ | MSE |
| End score (team2) | `team2_score_this_end` | Int 0–8+ | MSE |

### Architecture (planned)

The model takes a **Siamese** (S_{t-1}, S_t) board-state pair as input, where:

- **S_t** = board state after the shot was thrown (the current row's stone positions)
- **S_{t-1}** = board state before the shot (the previous row's stone positions; empty
  graph for `shot_number = 1`)

Each board state is encoded as a graph:
- **Nodes** = individual stones on the board; node features include `(x, y, dist, angle)`,
  team membership relative to the shooting team, `is_shot_stone` flag (once implemented).
- **Edges** = spatial proximity, team membership, and temporal links via `stone_id`
  between S_{t-1} and S_t.
- **Graph-level features (M)** = shot number, end number, score differential, hammer
  status, thinking time (when available).

The shot-type and accuracy heads read from the **change embedding** (difference between
S_{t-1} and S_t encodings).  The end-score head reads from the **S_t embedding** only
(board state after the shot).

### Feature normalisation and train/val/test split

Split at **event level** to prevent data leakage (team and player identities are shared
across matches in the same event):

```python
train_events = events[events["year"] <= 2023]["event_id"]
val_events   = events[events["year"] == 2024]["event_id"]
test_events  = events[events["year"] >= 2025]["event_id"]
```

### Current state

Phase 3 has **not yet started**.  The following steps are required before training can
begin:

| Step | Task | Status |
|:---:|---|---|
| 1 | Add `is_shot_stone` flag columns in `track_stones.py` | Not started |
| 2 | Write `build_graph_dataset.py` — converts enriched parquet to PyTorch Geometric `InMemoryDataset` | Not started |
| 3 | Define edge construction strategy (spatial proximity, team membership, temporal links) | Not started |
| 4 | Implement GNN model, multi-task heads, training loop | Not started |
| 5 | Validate tracking quality under multi-stone take-outs before training | Not started |

See `docs/gnn_data_readiness.md` for a full assessment of data readiness and remaining
gaps.

---

## End-to-End Run Order

```
python scrape_results.py                  # Phase 1a — build URL index
python extract_shot_data.py               # Phase 1b — extract all PDFs → raw CSVs
python track_stones.py [--method hungarian]  # Phase 2a — add stone IDs + prev positions
python build_features.py                  # Phase 2b — join game-context columns
python verify_stone_tracking.py           # Phase 2c — validate tracking output
# Phase 3: build_graph_dataset.py + model training (not yet implemented)
```

## File Handoff Summary

```
curlit.com PDFs
      │
      ▼ extract_shot_data.py
output/shot_locations_raw.parquet   (raw stone positions, no tracking)
output/ends.csv
output/events.csv
output/matches.csv  ...
      │
      ▼ track_stones.py
output/shot_locations.parquet       (stone IDs + prev_x/prev_y added)
      │
      ▼ build_features.py
output/shot_locations.parquet       (enriched: hammer, scores, team flags)
      │
      ▼ build_graph_dataset.py  [NOT YET IMPLEMENTED]
PyTorch Geometric Dataset
      │
      ▼ GNN training  [NOT YET IMPLEMENTED]
```
