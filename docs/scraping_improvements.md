# Scraping Pipeline Improvements — Issue 8 Extensions

**Date:** 2026-03-22

---

## Overview

Three improvements to the stone-detection pipeline are proposed, building on the ghost-stone
fix introduced in Issue 8. Each is summarised below with an assessment of implementation
feasibility and the expected impact on a downstream Graph Neural Network (GNN) model.

---

## Improvement 1 — Ghost Stone Position Recording

**Status: ✅ Complete**

### What it is

Rather than simply discarding outline-ring contours (fill ratio < `STONE_MIN_FILL_RATIO`),
collect their centroid coordinates as a separate "ghost" stone list and write them to new
schema columns alongside the existing active stone positions.

Ghost markers appear in shot diagrams as outline rings at the position where a stone
*was before the current shot was thrown*. They confirm that a displacement event occurred
and provide the origin position of the displaced stone.

### Implementation

The `_extract()` closure in `_detect_stones_in_crop()` now returns `(filled, ghosts)` instead
of a single list. The fill-ratio threshold still separates active stones from outline rings,
but ghost ring centroids are computed via pixel-moment centroid (same as active stones) and
recorded instead of discarded. `_detect_stones_in_crop()` returns a 5-tuple:
`(red_filled, red_ghosts, yellow_filled, yellow_ghosts, orientation)`.

Schema additions (in `shot_locations_raw.csv` and `shot_locations.parquet`):

| Column | Type | Notes |
|--------|------|-------|
| `team{N}_ghosts_in_play` | int | Count of ghost ring contours detected (0–8) |
| `team{N}_ghost{M}_x` | float/NULL | M = 1…8; normalised x position |
| `team{N}_ghost{M}_y` | float/NULL | Normalised y position |
| `team{N}_ghost{M}_dist` | float/NULL | Distance from house centre |
| `team{N}_ghost{M}_angle` | float/NULL | Angle from house centre (degrees) |

**Caveat:** Not all event PDF templates emit coloured ghost rings for displaced stones.
Grey-outline variants are already rejected upstream by the HSV saturation filter and will
produce no ghost data — those rows will carry NULL ghost columns, not incorrect ones.
Yellow ghost outlines tend to be thinner than red and may sit closer to the rejection
threshold; `STONE_MIN_FILL_RATIO` may need tuning per event template.

---

## Improvement 2 — Shot Stone Identification via Centre Dot / Bold Outline

### What it is

Each shot diagram marks the stone that was just delivered with either a small dark
centre dot or a bolder outline compared to other stones of the same colour. Detecting
this marker would allow the scraper to record which stone in the board state is the
shot stone, adding a `shot_stone_index` column to `shot_locations`.

### Implementation considerations

- **Centre dot:** at current render DPI the dot is approximately 3–5 pixels. Detection
  via a greyscale threshold on a small window at the contour centroid is feasible when
  the dot is large and crisp, but unreliable in noisy or lower-resolution crops.
- **Bold outline:** requires estimating per-stone contour boundary thickness and
  comparing across all stones on the same frame — a relative comparison vulnerable to
  atypically thin outlines elsewhere on the board producing false positives.
- **Cross-event inconsistency:** some events use the centre dot, some use the bold
  outline, some use neither. No single detection method achieves universal coverage,
  meaning any resulting `shot_stone_index` column would have a high NULL rate and
  uncertain reliability.

**Effort:** Medium–High.  **Reliability:** Low.

**Recommendation: Defer.** Once sequential tracking (Improvement 3) is implemented, the
delivered stone can be identified algorithmically as the stone present at shot N that was
absent in the active-stone set at shot N-1 — no image feature detection required, and
with higher reliability.

---

## Improvement 3 — Sequential Stone Coordinate Tracking Within an End

### What it is

Rather than treating each shot diagram independently, maintain a stone-identity state
across shots within an end. After detecting stones at shot N, match them to their
counterparts at shot N-1 using a minimum-distance assignment. Propagate unchanged
coordinates forward; flag stones that moved or left play.

The outcome is a stable `stone_id` per physical stone that persists across all shots of
an end, making frame-to-frame stone positions directly comparable.

### Algorithm

1. **Simple case (no hit):** Total stone count increases by exactly one. All N-1 stones
   are still present at nearly the same positions. Greedy nearest-neighbour matching
   propagates coordinates; the unmatched new stone is the delivered stone.

2. **Hit case:** Total stone count changes in a way that doesn't fit the simple model.
   Apply the **Hungarian algorithm** (`scipy.optimize.linear_sum_assignment`) to find
   a minimum-cost bijection between old and new positions under a distance cap (~0.1
   normalised units). Unmatched old positions = stones removed from play; unmatched
   new positions = newly arrived stones (delivered or displaced into view).

3. **End boundary:** State resets to an empty board at the start of each new end.

Schema additions:

| Column | Type | Notes |
|--------|------|-------|
| `team{N}_stone{M}_id` | int/NULL | Stable ID within the end; consistent across shot rows |
| `team{N}_stone{M}_prev_x` | float/NULL | Position of this stone at shot N-1 (NULL if newly arrived) |
| `team{N}_stone{M}_prev_y` | float/NULL | |

**Effort:** High (algorithmic; requires robust distance-threshold tuning and test
coverage across edge cases: simultaneous displacements, raises, triple take-outs).

**Key challenge:** coordinate noise of ±0.01–0.03 normalised units from rendering
variation means the matching distance threshold must be tuned empirically. Ghost
positions from Improvement 1 provide cross-validation signal for the raise case — where
two stones move simultaneously — making the two improvements complementary.

---

## Recommended Implementation Order

| Priority | Improvement | Reason |
|:---:|---|---|
| 1 | **3 — Sequential tracking** | Prerequisite for stable node identity in a GNN; also delivers shot-stone identification as a free by-product |
| 2 | **1 — Ghost stone recording** | Adds displacement-origin features; value is maximised once stable IDs from Improvement 3 allow ghost positions to be matched to specific moved stones |
| 3 | **2 — Centre dot / bold outline** | Superseded by Improvement 3; defer indefinitely |

---

## GNN Impact

### Why the current data is limiting for a GNN

The current `shot_locations` schema sorts stones by distance from the house centre on
every shot independently. A stone that does not move between shot N-1 and shot N may
appear as `stone1` in one frame and `stone3` in the next simply because another stone
landed closer. Node identity is therefore unstable across the shot sequence, which
directly harms any architecture that propagates information across timesteps.

### Improvement 3 — impact on GNN (high)

Stable `stone_id` values across shots within an end are a prerequisite for:

- **Temporal GNNs** (recurrent GNN, GraphTransformer over the shot sequence): these
  architectures require that node $i$ at time $t$ refers to the same physical object as
  node $i$ at time $t-1$ to meaningfully accumulate hidden states.
- **Reliable $S_{t-1}$ construction**: the Siamese pair $(S_{t-1}, S_t)$ required for
  shot-level prediction currently relies on a row-shift over distance-sorted columns.
  With stable IDs and propagated `prev_x/prev_y` coordinates, the pre-shot board state
  is exactly the tracked position of each stone rather than an approximation.
- **Displacement edges**: an edge type connecting a stone's position at $t-1$ to its
  matched position at $t$ can encode stone trajectory as an explicit relational feature,
  useful for predicting shot outcome and accuracy.
- **Shot stone identification**: the newly arrived stone at shot $N$ (the unmatched node
  in the Hungarian assignment) is the delivered stone — a zero-cost by-product of
  tracking.

### Improvement 1 — impact on GNN (moderate, conditional)

Once stable node identities are established by Improvement 3, ghost positions can be
attached to the corresponding active stone node as additional features:

- **Node feature `was_displaced_this_shot`** (binary): the model can learn that stones
  recently displaced are more likely to be near the edge of the house, relevant for
  guard strategy evaluation.
- **Ghost position as origin feature** `(ghost_x, ghost_y)`: the displacement vector
  $(x_t - x_{\text{ghost}}, y_t - y_{\text{ghost}})$ encodes how far and in what
  direction a stone moved, a discriminative signal for take-out vs. raise vs.
  hit-and-roll classification.
- **Without Improvement 3**, ghost nodes cannot be reliably assigned to specific active
  nodes, reducing their value to a board-level count signal only.

### Summary table

| Feature unlocked | Improvement 3 | Improvement 1 | Improvement 2 |
|---|:---:|:---:|:---:|
| Stable node identity across shot sequence | ✅ | — | — |
| Temporal / recurrent GNN support | ✅ | — | — |
| Accurate $S_{t-1}$ board state | ✅ | — | — |
| Displacement-vector edge features | ✅ + ✅ | ✅ | — |
| Shot stone identification | ✅ (derived) | — | ⚠️ unreliable |
| `was_displaced` binary node feature | — | ✅ | — |
| Ghost-origin node feature vector | — | ✅ | — |
