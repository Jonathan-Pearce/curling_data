---
description: "Use when building the ML pipeline, GNN, Siamese network, PyTorch model, or any machine learning feature for curling shot analysis. Covers data schema, feature construction, label extraction, filtering rules, and known data quality constraints for end score regression, shot accuracy regression, and shot type classification."
---

# Data Context: Curling ML Pipeline

## Source Files

| File | Rows (approx.) | Grain |
|------|---------------|-------|
| `output/shot_locations.parquet` | ~700k | One row per shot thrown |
| `output/ends.csv` | ~30k | One row per end played |
| `output/matches.csv` | ~4.5k | One row per match |
| `output/events.csv` | 260 | One row per event (tournament) |

Always read `shot_locations` from the **parquet** file, not any CSV. The CSV is gitignored.

---

## shot_locations.parquet — Schema

### Identity / grouping columns

| Column | Type | Notes |
|--------|------|-------|
| `event_id` | int | FK → events.csv |
| `match_id` | int | Local to event; (event_id, match_id) is globally unique |
| `end_number` | int | 1-based |
| `shot_number` | int | 1–16 within the end |
| `team_code` | str | Team that threw this shot |
| `player_id` | str | Unique player identifier |
| `player_name` | str | Display name |

### Shot metadata (prediction targets and context)

| Column | Type | Notes |
|--------|------|-------|
| `shot_type` | str | e.g. `Draw`, `Guard`, `Take-out`, `Raise`, `Hit and Roll`, `Wick / Soft Peeling`. Classification target. |
| `turn` | str | `In-turn`, `Out-turn`, `Not considered`. Classification sub-target or context feature. All "Through" penalty shots have `turn = "Not considered"`. |
| `accuracy` | str/float | 0–100 (percentage). Regression target. |
| `house_orientation` | str | `top` or `bottom`; stone coords are already normalised to a consistent frame — this column is informational only. |

### Board state columns

| Column | Type | Notes |
|--------|------|-------|
| `team1_stones_in_play` | int | Number of team1 stones visible in the shot diagram (0–8) |
| `team2_stones_in_play` | int | Same for team2 |
| `team1_stone{N}_x` | float/NULL | N = 1…8; x position normalised to house-radius units |
| `team1_stone{N}_y` | float/NULL | y position; positive = toward delivery end (hog line) |
| `team1_stone{N}_dist` | float/NULL | Euclidean distance from house centre in house-radius units |
| `team1_stone{N}_angle` | float/NULL | Angle in degrees (atan2) from house centre |
| `team1_stone{N}_id` | int/NULL | Stable integer ID for this stone within its end. Consistent across all shot rows of the same end; resets at end 1. Use to correlate the same physical stone across shots. |
| `team1_stone{N}_prev_x` | float/NULL | x position of this stone at the previous shot (NULL if stone was newly placed this shot). |
| `team1_stone{N}_prev_y` | float/NULL | y position at previous shot (NULL if newly placed). |
| `team2_stone{N}_*` | float/NULL | Same columns for team2 (N = 1…8) |

Stones within each team are **sorted ascending by distance** from the house centre.
NULL entries beyond `team{N}_stones_in_play` are empty cells, not zeroes.

**Colour-to-team mapping**: team1 = red stones, team2 = yellow stones. This is hardcoded in the scraper; the mapping is consistent across all events in the current dataset.

**Stone ID tracking:** `stone_id` values are assigned by a greedy nearest-neighbour matcher (`STONE_TRACK_MAX_DIST = 0.13` normalised units). A stone not matched to any previous stone (new placement or displacement beyond threshold) receives a fresh ID. IDs are scoped to a single end and shared across both teams, so team1 and team2 IDs within the same end are always distinct. **Shot 1 of every end** has `prev_x = NULL` for all stones (no prior state). Use `shot_number == 1` as the sentinel for the end boundary.

The displacement vector for a stone that stayed in play is:
```python
dx = row["teamN_stoneM_x"] - row["teamN_stoneM_prev_x"]
dy = row["teamN_stoneM_y"] - row["teamN_stoneM_prev_y"]
```
A stone with `prev_x = NULL` was newly placed or moved beyond the tracking threshold (treat as a new stone for GNN purposes).

---

## ends.csv — Schema (for labels and context)

| Column | Type | Notes |
|--------|------|-------|
| `event_id` | int | |
| `match_id` | int | |
| `end_number` | int | |
| `team1_code` | str | |
| `team2_code` | str | |
| `team1_score_before` | int | Cumulative score entering this end |
| `team2_score_before` | int | |
| `team1_score_this_end` | str | **⚠ May be `"X"` for conceded ends** — see below |
| `team2_score_this_end` | str | Same caveat |
| `team1_score_after` | int | |
| `team2_score_after` | int | |
| `hammer_team_code` | str | Team with last-stone advantage; blank for conceded ends |
| `team1_time_left` | str/NULL | Thinking-time clock remaining (MM:SS); NULL if event has no clock data |
| `team2_time_left` | str/NULL | |

---

## Critical Data Quality Constraints

### 1. Conceded ends — `score_this_end = "X"`
When a team concedes during an end, `team1_score_this_end` and/or `team2_score_this_end`
is written as the string `"X"` (not an integer). Always cast to numeric with
`pd.to_numeric(..., errors='coerce')` and decide how to handle NULLs (drop or
special-case). These rows have valid board-state data up to the last shot thrown.

### 2. Mixed Doubles events — no shot position data
Events with `gender = 'mxd'` in `events.csv` have `ends.csv` rows but **zero rows
in `shot_locations`**. Filter them out before joining.

### 3. No S_{t-1} for shot_number = 1
The first shot of every end has no preceding board state. Represent S_{t-1} as
an empty graph (zero stones) or zero-vector embedding with a boolean mask feature
`is_first_shot = True`.

### 4. Thinking-time availability
Only events where `events.has_time_data = True` (74/260 events) have non-NULL
`time_left` values. Do not use time features as hard inputs unless you filter to
those events or treat NULL as a missing modality.

---

## Constructing S_{t-1} and S_t (the Siamese pair)

The shot_locations table records the board state **after** each shot. To form the
(S_{t-1}, S_t) pair for shot N:

```python
# Within a group keyed by (event_id, match_id, end_number), sorted by shot_number:
# S_t   = the current row's stone positions (after shot N was thrown)
# S_{t-1} = the previous row's stone positions (after shot N-1 was thrown)
#          = empty board (no stones) if shot_number == 1
```

Key groupby key: `["event_id", "match_id", "end_number"]`, sort by `shot_number`.

---

## Joining Labels onto Shots

```python
ends = pd.read_csv("output/ends.csv")
shots = pd.read_parquet("output/shot_locations.parquet")

# Coerce targets
ends["team1_score_this_end"] = pd.to_numeric(ends["team1_score_this_end"], errors="coerce")
ends["team2_score_this_end"] = pd.to_numeric(ends["team2_score_this_end"], errors="coerce")

shots = shots.merge(
    ends[[
        "event_id", "match_id", "end_number",
        "team1_code", "team2_code",
        "team1_score_before", "team2_score_before",
        "team1_score_this_end", "team2_score_this_end",
        "hammer_team_code",
    ]],
    on=["event_id", "match_id", "end_number"],
    how="left",
)
```

---

## Recommended Pre-training Filters

Apply all of the following to get a clean 4-person training set:

```python
# 1. Exclude Mixed Doubles events
mxd_events = events.loc[events["gender"] == "mxd", "event_id"]
shots = shots[~shots["event_id"].isin(mxd_events)]

# 2. Exclude conceded ends (NaN score after coercion)
shots = shots[shots["team1_score_this_end"].notna()]

# 3. Exclude rows with no stones on the board at all
shots = shots[(shots["team1_stones_in_play"] + shots["team2_stones_in_play"]) > 0]
```

---

## Metadata Features Available per Shot

After the join above, the following are available as metadata inputs M:

| Feature | Source | Notes |
|---------|--------|-------|
| `shot_number` | shots | 1–16; proxy for end progress |
| `end_number` | shots | 1–10 (or extra ends) |
| `hammer_team_code` | ends | Which team has hammer this end |
| `team1_score_before` | ends | Score entering the end |
| `team2_score_before` | ends | Score entering the end |
| `score_diff_before` | derived | `team1_score_before - team2_score_before` |
| `shooting_team_has_hammer` | derived | bool: `team_code == hammer_team_code` |
| `team1_time_left` | ends | MM:SS string; parse to seconds; NULL if no clock |

---

## Target Variable Summary

| Head | Column | Type | Loss |
|------|--------|------|------|
| End score (team1) | `team1_score_this_end` | int (0–8+) | MSE |
| End score (team2) | `team2_score_this_end` | int (0–8+) | MSE |
| Shot accuracy | `accuracy` (shots) | float 0–100 | MSE |
| Shot type | `shot_type` (shots) | categorical | Cross-entropy |
| Turn | `turn` (shots) | categorical | Cross-entropy (optional) |

All three heads share the same (S_{t-1}, S_t, M) input. The end-score head
should only read from the S_t embedding (board state after shot), not the change
embedding, per the architecture in [issue #11](https://github.com/Jonathan-Pearce/curling_data/issues/11).

---

## Train / Validation / Test Split

Split at the **event level** (not match or end level) to prevent leakage.
Use year as a natural boundary: e.g. train on events up to 2023, validate on 2024,
test on 2025–2026.

```python
train_events = events[events["year"] <= 2023]["event_id"]
val_events   = events[events["year"] == 2024]["event_id"]
test_events  = events[events["year"] >= 2025]["event_id"]
```

---

## Scale of Usable Data (4-person events only, after filters)

- Events: ~185 (of 260 total)  
- Matches: ~3,800  
- Ends: ~26,000  
- Shots: ~415,000  

Shot accuracy (`accuracy`) and `shot_type` are populated for all shot rows.
Stone position columns are populated for all 16-stone ends (the vast majority).
