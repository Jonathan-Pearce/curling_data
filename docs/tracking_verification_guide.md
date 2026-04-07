# Stone Tracking Verification Guide

**Date:** 2026-04-07
**Relates to:** `track_stones.py` — `apply_tracking()`, `verify_stone_tracking.py`

---

## Background

The `apply_tracking()` function in `track_stones.py` assigns each detected stone a stable
`stone_id` that persists across all shots within an end. It also populates
`prev_x` / `prev_y` (the stone's position from the previous shot) and
`is_shot_stone` (True for the one newly delivered stone on each shot).

After any re-scrape and re-run of `track_stones.py` that produces a fresh
`shot_locations.parquet`, you should run the verification script to confirm the tracking
is behaving as expected.

---

## Running the Verification Script

The script `verify_stone_tracking.py` at the repo root performs all checks described
below and prints a pass/fail/warning summary.

### Basic run (whole parquet)

```bash
python verify_stone_tracking.py output/shot_locations.parquet
```

### Filter to one event or match

Useful when diagnosing a specific tournament or match:

```bash
python verify_stone_tracking.py output/shot_locations.parquet --event 1
python verify_stone_tracking.py output/shot_locations.parquet --event 1 --match 3
```

### Generate visual board images with displacement arrows

Produces one PNG per shot for a selected end, showing a small grey arrow from each
stone's previous position to its current position:

```bash
python verify_stone_tracking.py output/shot_locations.parquet \
    --event 1 --match 1 \
    --visualise --vis-end 3 --output-dir verify_out/
```

Open the resulting PNGs (named `e1_m1_end3_shot01.png` … `e1_m1_end3_shot16.png`) and
compare them against the corresponding PDF diagrams to visually confirm stones are being
tracked correctly.

---

## What Each Check Verifies

### Check 1 — Schema

**What it checks:**
All tracking columns are present in the parquet file:
`team{1,2}_stone{1–8}_{id, prev_x, prev_y, is_shot_stone}` (64 columns total).

**What failure means:**
The parquet was produced by the old version of `extract_shot_data.py` (before the
tracking code was separated into `track_stones.py`). Re-run `track_stones.py` and
`build_features.py` on the raw parquet.

**Expected result:** `PASS`

---

### Check 2 — First-shot invariant

**What it checks:**
Every row where `shot_number == 1` has `NULL` for all `prev_x` and `prev_y` columns.

**Why this matters:**
Shot 1 is the first shot of an end. No prior board state exists, so there are no
previous positions to propagate. If any shot-1 row has a non-NULL `prev_x`, the
tracking state is leaking across end boundaries — a serious bug.

**What failure means:**
The tracking state is leaking across end boundaries — a serious bug.  Check that the
`track_state = {1: [], 2: []}` initialisation inside `apply_tracking()` resets at each
new (event_id, match_id, end_number) group boundary.

**Expected result:** `PASS`

---

### Check 3 — Non-first-shot coverage

**What it checks:**
On shots 2–16, the fraction of active stone slots that have `prev_x` populated
(i.e. were successfully matched to a previous-shot stone).

**Interpreting the match rate:**
In a typical end, 15 out of 16 shots have between 1 and 15 stones already on the
board from the previous shot. Most of those stones didn't move. The expected match
rate is therefore well above 80%. The check flags if the rate falls below 50%.

| Rate | Interpretation |
|---|---|
| > 85% | Healthy — most stationary stones are being matched |
| 60–85% | Acceptable — possible noisier PDF or tighter crops |
| 50–60% | Warning — investigate; may indicate coord noise or threshold too tight |
| < 50% | Likely problem — tracking is not working for most stones |

**If the rate is low:**
- The current threshold is already `0.13`. If the rate is still low, check that the
  normalised coordinate system hasn't changed between scrape runs
  (e.g. `HOUSE_RADIUS` constant should not have changed).
- If the rate is below 60%, consider raising to `0.15`, but check Check 6/7 carefully
  to ensure the wider window isn't mis-matching adjacent stones.

**Expected result:** `PASS (match rate > 50%)`

---

### Check 4 — ID uniqueness within an end

**What it checks:**
No `stone_id` integer value appears for both `team1` and `team2` in the same end.
Stone IDs are assigned from a shared counter, so team1 and team2 IDs are always
distinct.

**What failure means:**
The `end_stone_id_counter` is not shared between the two `_match_stones_to_state()`
calls for team1 and team2 within the same shot. Check that both calls pass the same
`end_stone_id_counter` list object, and that it is not re-initialised between the team1
and team2 calls.

**Expected result:** `PASS`

---

### Check 5 — ID continuity

**What it checks:**
A `stone_id` that is present at shot N and also at shot M (M > N) does not have a gap
of more than 1 shot in between. Once a stone leaves play its ID should not reappear.

**Warning vs. failure:**
This check produces warnings rather than hard failures because there are legitimate
edge cases:
- A stone very close to the hog line may be detected on one shot and not the next
  due to the crop boundary, then re-detected if it was nudged back in.
- At very low stone counts, two stones close together may swap slots and generate
  apparent gaps.

A small number of warnings (< 1% of stone-shots) is acceptable. A high warning count
suggests coordinate noise or `STONE_TRACK_MAX_DIST` is too tight.

**Expected result:** `PASS` or low-count `WARN`

---

### Check 6 — Matched-stone displacement

**What it checks:**
For every stone with a non-NULL `prev_x`, compute the Euclidean distance from
`(prev_x, prev_y)` to `(x, y)`. A matched stone should have moved only due to
rendering noise, so this distance should be very small (< 0.05 normalised units).

The check prints a histogram of displacements in addition to flagging values above the
warning threshold (`0.05` normalised units = 5% of the 12-foot ring radius ≈ 3.6 inches
at real scale).

**Reading the histogram:**

```
0–0.01                  450,000  ███████████████████████████
0.01–0.02                12,000  ████
0.02–0.03                 2,000  █
0.03–0.05                   800
0.05–0.10 (WARN)            120
>0.10 (SUSPICIOUS)           15
```

The vast majority of matched stones should fall in the 0–0.01 bin (pure rendering
noise). Values in 0.01–0.05 are acceptable. Values above 0.05 indicate either:
1. A stone was displaced by a hit but remained within the tracking threshold — the
   matcher connected it to a nearby stone that should have been treated as removed.
2. The render DPI changed between scrapes, shifting all coordinates slightly.

**What counts > 0.13 mean:**
A stone was matched despite being more than 13% of the house radius away from its
previous position. This should not occur because `STONE_TRACK_MAX_DIST = 0.13` is
the hard matching cutoff. If you see counts here, the threshold constant in the
parquet file was different from `STONE_TRACK_MAX_DIST` in this script — check that
both are `0.13`.

**Expected result:** `PASS` with histogram heavily skewed toward 0–0.01

---

### Check 7 — New stone count

**What it checks:**
On each shot with `shot_number > 1`, counts stone slots where `x` is present
(active stone) but `prev_x` is NULL (newly placed this shot). Expects ≤ 1 new stone
per team per shot.

**Why more than 1 would be a problem:**
In normal play, one stone is delivered per shot. A team cannot have two newly placed
stones in a single shot. If two slots have NULL `prev_x` for the same team on the same
shot, it means the tracker failed to match 2 existing stones to their prior positions —
likely because a render artifact or temporary disappearance confused the matcher.

A warning here does not necessarily mean the data is wrong (e.g. a stone knocked out of
bounds that was simultaneously replaced), but it warrants manual inspection.

**Expected result:** `PASS` or low-count `WARN`

---

## Interpreting the Overall Summary

```
SUMMARY:  0 FAIL(s)   0 WARNING(s)
RESULT: PASS — tracking appears consistent.
```

| Result | Meaning |
|---|---|
| `PASS` | All checks clean. Data is safe to use. |
| `PASS WITH WARNINGS` | Structurally valid; review specific warnings. Typically harmless. |
| `FAIL` | Data should not be used until the issue is resolved. |

---

## Threshold Tuning Reference

The key constant `STONE_TRACK_MAX_DIST = 0.13` (in `extract_shot_data.py`) controls
the matching distance. If checks suggest it needs adjusting:

| Symptom | Adjustment |
|---|---|
| Check 3 match rate is low (< 70%) | Raise to `0.15`; re-check Check 6/7 for new mis-matches |
| Check 5 many reappearances | Raise to `0.15` |
| Check 6 counts in `> 0.13` bin | Investigate — should not happen if code and script agree |
| Check 7 many `> 1 new stone` shots | May indicate stones lost to crop edge; consider adjusting `STONE_Y_MAX_FRAC` |

Physical reference: 1 normalised unit = `HOUSE_RADIUS` = 6 feet. So:
- `0.10` ≈ 7.2 inches — comfortable above stone-detection jitter
- `0.13` ≈ 9.4 inches — current value (approaches the radius of a real stone ≈ 5.6 inches)
- `0.15` ≈ 10.8 inches — upper safe limit before adjacent-stone confusion becomes likely

---

## Manual Visual Checklist

When `--visualise` is used, open the generated PNG sequence and verify:

1. **Shot 1** — no arrows (no `prev_x`). All stones appear as clean circles.
2. **Shot 2** — one circle has no arrow (the delivered stone). All previously placed
   stones from shot 1 have a short grey arrow pointing from their shot-1 position
   (which should be nearly identical, confirming a correct match).
3. **Any take-out shot** — identified by `shot_type` containing "Take-out". The
   removed stone(s) disappear from the board. The shot stone has no arrow (NULL `prev_x`
   — it is newly arrived from the hack). The remaining stones should all have short
   arrows confirming they stayed near the same position.
4. **End consistency** — by shot 16, each team should have a maximum of 8 stones.
   Each stone's arrow should point to almost the same location (near-zero displacement),
   confirming stationary stones are tracked correctly across the full end.

---

## Limitations

- **Tracking accuracy on take-outs and raises:** When a stone is displaced, it may
  move to a new position that is within the matching threshold of a different stone.
  The greedy matcher always takes the minimum-distance pair, so a stone that moves a
  small distance will usually be matched correctly, but in crowded scenarios near the
  house there may be swaps. These show up as unusually large displacements in Check 6.

- **PDF rendering variation:** Different event PDFs render at slightly different
  effective resolutions even at a fixed DPI setting, so the same physical stone position
  may produce coordinates that vary by 0.02–0.04 units between events. The tracking
  threshold is tuned to absorb this.

- **Conceded/partial ends:** Ends with fewer than 16 shot images are written with score
  data only and no stone positions. The tracker state is not initialised for these ends,
  so they produce no tracking rows and are correctly absent from all checks above.
