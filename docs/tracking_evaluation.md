# Tracking Method Evaluation

**Date:** 2026-03-24

This document describes three approaches for evaluating the quality of sequential stone
tracking when no ground-truth stone identities are available.  Option 3 is implemented
in `track_stones.py`; options 1 and 2 are documented here as future work.

---

## Option 1 — Manual Ground-Truth Sample

### What it is

Annotate 20–30 ends by hand: for each shot, record which physical stone corresponds to
which detected position slot.  Then measure **assignment accuracy** — the fraction of
matched links where the tracker's assignment matches the human label.

### How to use it

Run `apply_tracking()` over the annotated ends, then compare each
`team{N}_stone{S}_id` against the corresponding ground-truth label.  A correct match
means the tracker assigned the same persistent ID to the stone a human would have.

### Strengths

- The only method that directly measures whether assignments are **correct**, not just
  internally consistent.
- Validates the distance threshold `STONE_TRACK_MAX_DIST` empirically: if accuracy
  drops at a particular threshold value, the threshold is miscalibrated.
- Small sample (20–30 ends) is sufficient to detect systematic failures such as ID
  swaps on take-out shots.

### Limitations

- Requires human effort for annotation.  Ends with multi-stone take-outs are especially
  tedious since several stones change position simultaneously.
- Results are only as reliable as the annotations; ambiguous cases (e.g., two stones at
  nearly the same position) introduce label noise.

### Status

Not yet implemented.  Recommended as a one-time calibration exercise after the tracker
is otherwise stable.

---

## Option 2 — Physics-Based Sanity Check

### What it is

Between shot N and shot N+1 within an end, **exactly one new stone is delivered** (the
shot stone).  A correctly tracking scraper should therefore create exactly one new stone
ID on every non-first shot.  Any shot where `new IDs created ≠ 1` is either a genuine
edge case (a stone leaves play because it was hit out, or the shot stone fails to stop
in the house) or a tracking error.

The "expected new ID count = 1" rule gives a per-shot binary signal without any
annotation.

### Metric

```
delivery_anomaly_rate = shots where new IDs ≠ 1 / total non-first shots
```

A low rate means the tracker is creating IDs at the right frequency.  It cannot
distinguish between the valid edge cases and genuine errors, but a rate above ~10–15%
is a clear signal of miscalibration.

### Limitations

- Counts are **necessary but not sufficient** for correctness.  A tracker that creates
  exactly one new ID at every shot but always links it to the wrong stone will score
  well here.
- Multi-stone take-outs (doubles, triples) legitimately produce counts ≠ 1 because the
  removed stone IDs disappear and a new one arrives; these are valid events, not errors.
- The metric degrades gracefully (the take-out rate in competitive curling is roughly
  40–50% of shots), so anomaly rate must be interpreted relative to expected take-out
  frequency.

### Status

Not yet implemented.

---

## Option 3 — Displacement Magnitude Distribution (Implemented)

### What it is

For every matched link (a stone present at both shot N−1 and shot N), compute the
**displacement magnitude**:

$$d = \sqrt{(\Delta x)^2 + (\Delta y)^2}$$

where coordinates are normalised to house-radius units.  Examine the resulting
distribution.

### Why this is diagnostic without ground truth

A well-calibrated tracker should exhibit a **bimodal** displacement distribution:

- **Mode 1 — near-zero** (roughly $d < 0.05$): stones that did not move between shots.
  Rendering noise produces small jitter; the rendered PDF is static per shot so
  unperturbed stones should cluster tightly near 0.
- **Mode 2 — large** ($d \gg 0.13$): stones that were hit and moved a long way.  These
  are real displacements caught *within* the matching threshold because the stone stayed
  close enough.

The key diagnostic signal is the **threshold zone**: the band $[0.8 \times T, T]$ where
$T$ = `STONE_TRACK_MAX_DIST = 0.13`.  Matches in this band are the most uncertain ones
the tracker accepted.

| Pattern | Interpretation |
|---------|---------------|
| Few matches near $T$  | Threshold is in a clean gap; well-calibrated |
| Peak of matches near $T$ | Threshold sits inside a dense region; raising it would match more stones but risk linking wrong stones; lowering it would fragment more IDs |
| Bimodal gap clearly present | Strong evidence the bimodal structure assumed above holds for this dataset |

### Function

```python
from track_stones import displacement_distribution

result = displacement_distribution(tracked_df)
```

Returns a dict with:

| Key | Type | Description |
|-----|------|-------------|
| `displacements` | `np.ndarray` | Every matched-link displacement magnitude |
| `total_links` | int | Total matched links (non-NaN `prev_x`) across all shots |
| `threshold_zone_count` | int | Links in $[0.8T, T)$ |
| `threshold_zone_fraction` | float | `threshold_zone_count / total_links` |
| `median_displacement` | float | Median of all displacements |
| `p95_displacement` | float | 95th percentile (shows tail of hit distances) |
| `near_zero_fraction` | float | Fraction of links with $d < 0.05$ |

A `threshold_zone_fraction` above ~5% is a warning sign that the threshold is not
sitting in a clean gap.

### CLI

```
python track_stones.py --method greedy --evaluate --displacement
```

Prints the scalar summary metrics.  The raw `displacements` array is available via the
Python API for histogram plotting in a notebook.

### Limitations

- The assumed bimodal structure only holds if most stones either stay put or are hit
  hard.  Soft raises and nudges produce intermediate displacements that blur the gap.
- Coordinate noise of ±0.01–0.03 units widens Mode 1.  If noise is larger than
  expected (e.g., PDF rendering artefacts), the mode can bleed into the threshold zone
  even for an otherwise correct tracker.
- This metric validates the **threshold** more than the **algorithm**: both greedy and
  Hungarian use the same distance cap, so they produce near-identical displacement
  distributions.  The diagnostic is most useful for threshold tuning, not for choosing
  between algorithms.
