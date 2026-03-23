# Stone Location Accuracy — Proposed Improvements

**Date:** 2026-03-23

This document tracks proposed improvements to the stone detection pipeline in
`extract_shot_data.py` to improve the precision of stone coordinate extraction
and cross-event consistency.

---

## Improvement 1 — Dynamic `HOUSE_RADIUS` Detection

**Status: ✅ Complete**

### Problem

`HOUSE_RADIUS = 112` is a fixed constant calibrated against one specific event's
rendering. `_detect_house_center()` detects the center `(cx, cy)` of the 12-foot
ring from its pixel centroid, but the radius used to normalise all coordinates is
never measured — it is assumed. If a PDF renders the rink image at a different
scale (even 5–10%), all normalised coordinates will be *systematically biased*
across the entire event with no warning signal.

### Suggested Fix

After locating candidate ring pixels, fit the outer circle of the 12-foot ring to
the point cloud — e.g. compute the mean distance from the detected centroid to each
ring pixel, or fit a bounding ellipse. Store the per-crop `detected_radius` and use
it as the normalisation divisor instead of the constant. Fall back to
`HOUSE_RADIUS = 112` only if the detected value is implausible (outside ±25% of
the expected range).

**Impact:** High — removes a systematic per-event scaling error that currently has
no warning signal.  
**Effort:** Medium

---

## Improvement 2 — Adaptive House Ring Color Thresholds

**Status: ✅ Complete**

### Problem

`HOUSE_RING_LOWER = [100, 25, 140]` / `HOUSE_RING_UPPER = [140, 160, 255]` are
calibrated for the blue/lilac ring found in most World Curling events. When the
ring mask returns fewer than `HOUSE_RING_MIN_AREA = 1500` pixels the entire house
detection silently falls back to the hardcoded `HOUSE_CX = 161, HOUSE_CY = 171`
constants. Older events, B/C-division PDFs, and events from different national
organisations can use different ring colours, meaning every shot in such an event
uses the wrong fixed center.

### Suggested Fix

- Widen the detection attempt with multiple HSV ranges (e.g. also try a red/pink
  range for the 4-foot ring as a secondary geometric structure for locating the centre).
- Alternatively, detect the **button** (the small white disc at the centre) — its
  size and shape are tightly constrained and it appears in virtually every diagram.
- Emit a **per-page warning** whenever the fallback fires, logging the event ID and
  page index so problem events can be identified and custom per-event thresholds
  applied without a code change.

**Impact:** High — silent fallback currently affects any non-standard event without
any indication.  
**Effort:** Medium

---

## Improvement 3 — Per-Page Stone Color Calibration from Team Indicator Dots

**Status: ✅ Complete** *(yellow range calibrated; red kept static as it is stable across events)*

### Problem

Six HSV constants (`RED_LOWER_1/2`, `RED_UPPER_1/2`, `YELLOW_LOWER`, `YELLOW_UPPER`)
are used for all events. Older PDFs, events from different federations, and events
with different print profiles can render stone colours with shifted hue, reduced
saturation, or different value ranges. A stone falling just outside the static range
is silently dropped with no error.

### Suggested Fix

Before processing the 16 shot crops on each page, sample the colour of the **team
indicator images** (the small coloured dots that already appear on the page and
identify which team is red/yellow). Use the sampled HSV values as the centre of the
detection range for that specific page with a fixed tolerance window. This
self-calibrates the thresholds per page automatically with no manual tuning per
event.

As a fallback, widen the `YELLOW_UPPER` hue bound to ~40 and reduce the saturation
minimum slightly, relying on the area/fill-ratio guards to reject false positives.

**Impact:** High — cross-event generalization for stone detection.  
**Effort:** Low–Medium

---

## Improvement 4 — Scale Area Bounds with Detected Rink Size

**Status: ✅ Complete** *(implemented as a natural dependency of #1)*

### Problem

`STONE_MIN_AREA = 80` and `STONE_MAX_AREA = 600` pixels were calibrated for
`HOUSE_RADIUS = 112 px`. Stone pixel area scales as the square of the linear
scale factor. For a PDF rendering the rink at radius = 140 px (~25% larger), the
expected stone area range becomes ~125–940 px — stones clearly visible in the image
will be silently dropped because they exceed `STONE_MAX_AREA = 600`.

### Suggested Fix

Scale the area bounds proportionally to the square of the detected house radius
(from Improvement 1):

```python
scale_sq = (detected_radius / HOUSE_RADIUS) ** 2
effective_min_area = STONE_MIN_AREA * scale_sq
effective_max_area = STONE_MAX_AREA * scale_sq
```

**Impact:** Medium — prevents silent stone drops in larger-scale renderings.  
**Effort:** Low (depends on Improvement 1 landing first)

---

## Improvement 5 — Remove `int()` Truncation from House Centre Centroid

**Status: ✅ Complete** *(implemented inside #1)*

### Problem

In `_detect_house_center()`:
```python
cx = int(M["m10"] / M["m00"])
cy = int(M["m01"] / M["m00"])
```
The centroid is truncated to integer pixels before being used as the normalisation
origin. At `HOUSE_RADIUS = 112 px` a 1-pixel error translates to ~0.009 normalised
units — a systematic offset shared by every stone on the page.

### Suggested Fix

Keep the centroid as a float: remove the `int()` wrappers and use float division
in the normalisation step in `_detect_stones_in_crop()`. The improvement is small
(~0.005 units) but is essentially free.

**Impact:** Low (but free).  
**Effort:** Low

---

## Improvement 6 — Use Filled-Pixel Centroid Instead of Contour Polygon Moments

**Status: ✅ Complete**

### Problem

`M = cv2.moments(c)` computes moments from the contour polygon. For noisy or
slightly non-circular blobs, the contour may not perfectly represent the filled-
pixel distribution. The `mask` variable is already available in `_extract()` at
the point where the centroid is computed.

### Suggested Fix

Compute the centroid from the actual filled pixels in the stone's bounding box
rather than from the contour polygon:

```python
M = cv2.moments(mask[y:y + h_c, x:x + w_c])
# then offset back to full-image coordinates:
sx = x + M["m10"] / M["m00"]
sy = y + M["m01"] / M["m00"]
```

This is more robust to irregular contour shapes caused by rendering artifacts.

**Impact:** Low–Medium.  
**Effort:** Low

---

## Improvement 7 — Morphological Preprocessing Before Contour Detection

**Status: ✅ Complete**

### Problem

Raw HSV thresholding is applied directly to the rendered page without any
cleanup. In compressed PDFs, stone blobs can have small holes, noisy edges, or
broken outlines after thresholding, degrading contour quality and centroid accuracy.

### Suggested Fix

Apply a small morphological close before `findContours` on each colour mask:

```python
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)
mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_CLOSE, kernel)
```

This fills minor holes in solid stones and merges fragmented blobs without
meaningfully displacing clean detections.

**Impact:** Low–Medium.  
**Effort:** Low

---

## Improvement 8 — Validate Shot Image Grid Layout

**Status: ✅ Complete**

### Problem

`_get_shot_images()` identifies shot images purely by size (`width > 50` and
`height > 100`). Some PDFs include logos or decorative images that pass this
filter, potentially inserting a spurious image into the list and silently
misaligning all subsequent shot assignments. The `len(shot_images) != 16` guard
only catches cases where the total count is wrong — if exactly 17 images are
returned and one is excluded incorrectly, the guard passes with a misaligned list.

### Suggested Fix

After filtering candidates to 16, further validate the expected **3-row, 6-6-4 grid
layout**: x-positions should form ~6 distinct clusters and y-positions should form
~3 clusters. Images that don't fit the grid (e.g. a logo at an unexpected x/y
position) should be excluded before the `len == 16` check rather than after.

**Impact:** Medium — prevents subtle misalignment bugs.  
**Effort:** Medium

---

## Improvement 9 — Page-Level Orientation Consistency Check

**Status: ✅ Complete**

### Problem

House orientation is determined independently per crop as
`"top" if cy <= h / 2 else "bottom"`. When house detection falls back to
`HOUSE_CY = 171` (crop height ~686), the condition `171 <= 343` evaluates to
`"top"` — correct for most events but wrong for events where the house is
consistently at the bottom. Widespread fallback in such an event would invert all
y-coordinates silently.

### Suggested Fix

Determine orientation **once at the end-page level** using the first shot diagram
(or the layout of score labels vs. the detected house position), then assert that
all 16 crops in the same end agree. Emit a warning if any crop disagrees with the
page-level orientation, which would indicate a detection failure rather than a
genuine layout switch.

**Impact:** Medium — prevents systematically inverted y-coordinates for entire events.  
**Effort:** Low

---

## Improvement 10 — Per-Event Calibration Diagnostic Pass

**Status: ✅ Complete**

### Problem

There is currently no mechanism to detect that a given event's stone positions are
systematically wrong (e.g. all coordinates offset by 0.1 units because the ring
was not detected). Known geometric reference points that could be exploited:
- The **back line** at `y = -1.0` and the **tee line** at `(0, 0)` are physically
  fixed.
- **Ring boundaries** (4-foot at `0.333`, 8-foot at `0.667`, 12-foot at `1.0`) can
  be detected as colour transitions independent of the main extraction pipeline.
- Stone counts should follow game-logic constraints (non-decreasing within an end
  except after a take-out).

### Suggested Fix

Add an optional per-event calibration diagnostic (run separately from the main
extraction) that:
1. Renders the first shot diagram of each event.
2. Attempts to detect all ring boundaries and computes the implied `HOUSE_RADIUS`
   and centre.
3. Compares detected stone distributions against expected game-state logic (e.g.
   stones detected far outside the house on shot 1 are likely false positives).
4. Logs any event where the detected radius deviates from the nominal value by
   more than 10%, making it easy to add per-event override constants to a config
   dict.

**Impact:** Medium (long-term tooling value).  
**Effort:** Medium–High

---

## Summary

| # | Issue | Impact | Effort | Status |
|---|---|---|---|---|
| 1 | `HOUSE_RADIUS` not detected — systematic per-event scaling bias | High | Medium | ✅ |
| 2 | House ring colour thresholds fail on non-standard PDFs → silent fallback | High | Medium | ✅ |
| 3 | Stone HSV ranges are global → mis-detection for non-standard events | High | Low–Medium | ✅ |
| 4 | Area bounds don't scale with rink size → silent stone drops | Medium | Low | ✅ |
| 5 | `int()` truncation discards sub-pixel centre accuracy | Low | Low | ✅ |
| 6 | Contour moments vs. filled-pixel centroid | Low–Medium | Low | ✅ |
| 7 | No morphological preprocessing → noisy contours | Low–Medium | Low | ✅ |
| 8 | Shot image grid validation absent → silent misalignment | Medium | Medium | ✅ |
| 9 | Orientation fallback can invert all y-coordinates for an event | Medium | Low | ✅ |
| 10 | No per-event calibration diagnostic | Medium (long-term) | Medium–High | ✅ |

### Recommended implementation order

1. **#5, #6, #7** — Quick wins; minimal risk, no dependencies.
2. **#1** — Unlocks #4 as a follow-on; highest overall impact.
3. **#3** — Per-page colour calibration; high cross-event value with low tuning burden.
4. **#2 + #9** — Reduce silent-fallback surface area; add logging for problem events.
5. **#8** — Grid validation; guards against subtle misalignment.
6. **#4** — Straightforward once #1 is in place.
7. **#10** — Calibration diagnostic; useful long-term tooling.
