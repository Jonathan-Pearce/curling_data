# Stone Tracking Quality Metrics

**Date:** 2026-04-07

Reference for all metrics available in `track_stones.py` for evaluating stone-tracking
quality.  Metrics are split across two functions:

- `evaluate_tracking(tracked_df)` — aggregate headline metrics.
- `displacement_distribution(tracked_df)` — detailed statistics on matched-link
  displacement magnitudes.

Additional targeted metrics below are computed by dedicated functions and can be combined
with `evaluate_tracking` for a full picture.

---

## `evaluate_tracking` metrics

| Metric | Ideal | Description |
|---|---|---|
| `total_ends` | — | Number of unique (event_id, match_id, end_number) groups. |
| `total_stone_appearances` | — | Total non-null stone slots across all shots. |
| `expected_new_stone_rate` | ~0.25 | Theoretical fraction of mid-end appearances that should be newly placed: `n_mid_end_shots / total_mid_end_appearances`. Equals `1 / avg_stones_on_board`. |
| `new_stone_rate_mid_end` | ≈ `expected` | Measured fraction of mid-end appearances where `prev_x` is NaN. Should equal `expected_new_stone_rate` when tracking is correct. |
| `spurious_id_rate` | 0.0 | Excess new IDs above the theoretical delivery rate: `max(0, new_mid_end − n_mid_end_shots) / total_mid_end`. Captures only stone appearances incorrectly treated as new. Takeout-heavy ends do **not** inflate this metric. Values above ~0.02 warrant investigation. |

### Why `fragmentation_index` was removed

The previous `fragmentation_index = total_ids_assigned / min_ids_needed` compared total
unique IDs against the peak `stones_in_play` per end.  In a takeout-heavy end a team may
deliver 8 stones, each immediately removed, cycling through 8 distinct physical stones —
that generates 8 correct IDs but a `min_ids_needed` of 1 (only 1 stone ever in play at a
time), yielding `fragmentation_index = 8`.  The metric falsely penalised correct
tracking.  `spurious_id_rate` avoids this by anchoring the baseline to shot count rather
than peak occupancy.

---

## `displacement_distribution` metrics

These describe the distribution of displacement magnitudes for **matched** stone links
(i.e. stones that were successfully assigned a previous-shot position).

| Metric | Ideal | Description |
|---|---|---|
| `total_links` | — | Total matched links used. |
| `threshold_zone_fraction` | < 0.05 | Fraction of links with displacement in `[0.8 × cap, cap)`. A peak here means the cap bisects a dense population; values above ~0.05 suggest cap retuning. |
| `near_zero_fraction` | > 0.90 | Fraction of links with displacement < 0.05 units. Stationary stones (rendering jitter only) should dominate. |
| `median_displacement` | < 0.01 | Typical matched-link displacement. Very small values confirm the cap is well above the noise floor. |
| `p95_displacement` | < cap | 95th-percentile displacement; characterises genuinely moving stones (hits, draws) that were still within tracking range. |

---

## Additional targeted metrics

### `id_continuity_rate(tracked_df)`

**What it measures:** For every stone that is occupied on two consecutive shots of the
same end *and* whose position didn't change by enough to indicate a delivery, does it
retain the same `_id`?  Drops below 1.0 only when a stone that should be stationary gets
re-IDed — the purest signal of noise-driven tracking failure.

**How it works:** For each consecutive (shot N, shot N+1) pair in an end, find all stone
slots where both shots have a non-null x/y and the stone was not the shot stone at shot
N+1 (`is_shot_stone != 1.0`).  These stones should not have changed identity.  Count the
fraction where the `_id` value is the same in both rows.

| Value | Interpretation |
|---|---|
| 1.0 | Perfect — no spurious re-IDs on non-delivered stones. |
| 0.95–0.99 | Acceptable; minor jitter at the cap boundary. |
| < 0.95 | Investigate cap calibration or detection noise. |

---

### `slot_swap_rate(tracked_df)`

**What it measures:** After tracking, do two stones of the same team ever exchange their
`_id` values between consecutive shots?  A stone cannot physically pass through another
stone in curling, so an ID swap between two stones that were both present at shot N is
always a tracking error.

**How it works:** For each consecutive shot pair within an end, collect `(slot, id)` for
all occupied stones of each team.  An ID swap occurs when two stones A and B have IDs
(a, b) at shot N and (b, a) at shot N+1 while both are present at both shots.

| Value | Interpretation |
|---|---|
| 0.0 | Perfect — no impossible stone crossings detected. |
| > 0.0 | At least one swap per N shot pairs; likely indicates greedy assignment errors when two stones are spatially close.  Hungarian tracking should eliminate these. |

---

### `cap_pressure_rate(tracked_df)`

**What it measures:** Of all mid-end stone appearances that received a **new** ID (treated
as a fresh delivery), what fraction have a nearest previous-shot stone in the
just-outside-cap zone `[cap, cap + 0.05)`?  This reveals matches that were rejected only
because they fell fractionally outside the cap — candidate false rejections.

**How it works:** For each mid-end new-ID appearance, compute the distance to every stone
in the previous shot state for the same team.  If the minimum distance lands in
`[cap, cap + 0.05)`, increment the pressure counter.

| Value | Interpretation |
|---|---|
| ~0.0 | Cap is well-placed; unmatched stones have no near neighbour. |
| > 0.10 | More than 10 % of new-ID events have a candidate just outside the cap — consider raising `STONE_TRACK_MAX_DIST`. |

---

### `displacement_symmetry(tracked_df)`

**What it measures:** Compares the displacement distribution for stones flagged
`is_shot_stone = 1.0` (the delivered stone) against stones flagged `0.0` (already on the
board).  The two populations should be clearly separated: still-stones have near-zero
displacement; shot-stones can be anywhere.

**Reported values:**
- `still_stone_median_displacement` — should be near zero (< 0.01).
- `shot_stone_median_displacement` — no fixed ideal; characterises typical stone delivery
  distance.
- `still_stone_p95_displacement` — a value approaching the cap signals noise issues.
- `separation_ratio` — `shot_stone_median / still_stone_median`; higher is better
  (clearer separation).  Values below ~5 suggest `is_shot_stone` detection quality
  problems or excessive still-stone jitter.

---

### `stone_count_consistency_rate(tracked_df)`

**What it measures:** Between consecutive shots, the count of each team's stones should
change by at most ±1 (one stone delivered or one taken out).  A jump of ±2 or more is
almost always an extraction error (detector hallucinated or dropped a stone), not a
tracking error.  This metric counts the fraction of consecutive-shot pairs where
`|Δstones_in_play| ≤ 1` for both teams.

| Value | Interpretation |
|---|---|
| 1.0 | Perfect stone-count continuity. |
| < 0.99 | At least 1 % of transitions have anomalous count jumps — report which events/ends are affected to guide upstream extraction fixes. |

---

## Interpreting metrics together

| Symptom | Likely cause | Primary metric |
|---|---|---|
| `spurious_id_rate > 0.02` | Cap too tight; noise exceeding cap | `cap_pressure_rate`, `threshold_zone_fraction` |
| `id_continuity_rate < 0.95` | Stationary stones being re-IDed | `still_stone_p95_displacement` |
| `slot_swap_rate > 0` | Wrong greedy assignment between nearby stones | Compare greedy vs. Hungarian `link_agreement_rate` |
| `stone_count_consistency_rate < 0.99` | Upstream detection error | Inspect affected ends in `verify_stone_tracking.py` |
| `separation_ratio < 5` | `is_shot_stone` flag quality issue | Check `spurious_id_rate` and takeout handling |
