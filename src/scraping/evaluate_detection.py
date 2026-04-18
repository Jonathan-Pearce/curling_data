"""Evaluate stone detection accuracy against the original PDF crops.

For each shot crop extracted from a PDF page, two sets of stone positions
are computed independently:

  **ground truth** — a permissive blob extraction that applies only a loose
    minimum area filter and a slightly relaxed fill-ratio threshold.  No area
    cap; no deduplication.  These represent all stone-like blobs that are
    visible in the PDF image.

  **detected** — the full production pipeline output from
    ``_detect_stones_in_crop``, which applies area bounds, fill-ratio
    filtering, deduplication, and centroid refinement.

The two sets are matched using a greedy nearest-neighbour assignment under a
match radius of ``MATCH_RADIUS`` normalised units.  A matched pair is a True
Positive (TP); an unmatched detection is a False Positive (FP); an unmatched
ground-truth blob is a False Negative (FN).

Metrics reported per PDF run:
  precision      — TP / (TP + FP)
  recall         — TP / (TP + FN)
  f1             — harmonic mean of precision and recall
  position_error — median and p95 normalised-unit centroid error across TPs
  miss_rate      — FN / total_ground_truth  (= 1 − recall)

Active stones and ghost rings are evaluated separately.

Usage::

    python src/evaluate_detection.py path/to/event.pdf
    python src/evaluate_detection.py path/to/event.pdf --max-ends 3
    python src/evaluate_detection.py path/to/event.pdf --overlay-dir eval_out/
    python src/evaluate_detection.py path/to/event.pdf --csv-out eval_out/results.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Allow ``python src/scraping/evaluate_detection.py`` to resolve sibling packages.
if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scraping.extract_shot_data import (
    _open_pdf,
    find_shot_pages,
    _get_shot_images,
    render_page_image,
    crop_shot_image,
    _detect_house_center,
    _detect_stones_in_crop,
    _calibrate_stone_colors,
    parse_end_line,
    RED_LOWER_1, RED_UPPER_1,
    RED_LOWER_2, RED_UPPER_2,
    YELLOW_LOWER, YELLOW_UPPER,
    STONE_MIN_AREA,
    STONE_MIN_FILL_RATIO,
    STONE_Y_MIN_PX, STONE_Y_MAX_FRAC,
    HOUSE_RADIUS,
    STONE_DEDUP_RADIUS,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Radius (normalised units) within which a detected stone and a ground-truth
# blob are considered the same physical stone.  Using the same value as the
# production deduplication radius is appropriate: two positions closer than
# this are practically indistinguishable.
MATCH_RADIUS = STONE_DEDUP_RADIUS  # 0.08 normalised units

# Ground-truth extraction uses a looser area lower bound (50 % of the
# production minimum) to capture stones near the filtering boundary.
# The ghost/filled classification boundary deliberately uses the same value
# as the production pipeline (STONE_MIN_FILL_RATIO) so that both GT and
# production agree on which blobs are filled stones vs ghost rings.
# Using a different boundary here creates a disagreement zone where the
# same blob is simultaneously a filled FN and a ghost FP.
GT_AREA_FACTOR = 0.5       # min_area multiplier (only difference from production)

# Overlay drawing colours (BGR)
_COLOUR_TP = (0, 220, 0)       # green  — matched detection
_COLOUR_FP = (0, 80, 255)      # orange — false positive (hallucinated)
_COLOUR_FN = (255, 60, 60)     # blue   — false negative (missed)
_COLOUR_GT_OUTLINE = (200, 200, 200)   # light grey circle for GT position


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_colour_masks(crop_bgr, color_ranges=None):
    """Return (mask_red, mask_yellow) for a crop, after morphological close.

    Applies the same HSV thresholding, play-field masking, and morphological
    close as the production pipeline so that the ground-truth extraction sees
    exactly the same pixel set as the detector.
    """
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    h, w = crop_bgr.shape[:2]

    if color_ranges is not None:
        red_lo1, red_hi1, red_lo2, red_hi2, yel_lo, yel_hi = color_ranges
    else:
        red_lo1, red_hi1 = RED_LOWER_1, RED_UPPER_1
        red_lo2, red_hi2 = RED_LOWER_2, RED_UPPER_2
        yel_lo, yel_hi = YELLOW_LOWER, YELLOW_UPPER

    play_mask = np.zeros((h, w), dtype=np.uint8)
    play_mask[STONE_Y_MIN_PX:int(h * STONE_Y_MAX_FRAC), :] = 255

    mask_red = (
        cv2.inRange(hsv, red_lo1, red_hi1) | cv2.inRange(hsv, red_lo2, red_hi2)
    ) & play_mask
    mask_yellow = cv2.inRange(hsv, yel_lo, yel_hi) & play_mask

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)
    mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_CLOSE, kernel)

    return mask_red, mask_yellow


def _permissive_blobs(mask, house_cx, house_cy, house_radius, orientation, scale_sq):
    """Extract all stone-like blobs with loose filters (ground-truth extraction).

    Uses ``GT_AREA_FACTOR * STONE_MIN_AREA * scale_sq`` as the area lower
    bound and no area upper bound.  ``STONE_MIN_FILL_RATIO`` is used as the
    ghost/filled boundary — the same value as the production pipeline — so
    that both extractors agree on which blobs are filled stones vs ghost
    rings.  The only intentional difference from production is the looser
    area lower bound.

    Returns
    -------
    filled : list of (nx, ny, dist, angle)
        Blobs classified as filled stones (fill_ratio >= STONE_MIN_FILL_RATIO).
    ghosts : list of (nx, ny, dist, angle)
        Blobs classified as ghost outline rings (fill_ratio < STONE_MIN_FILL_RATIO).
    """
    min_area = STONE_MIN_AREA * scale_sq * GT_AREA_FACTOR

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    filled = []
    ghosts = []

    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue

        x, y, w_c, h_c = cv2.boundingRect(c)
        stone_mask = mask[y:y + h_c, x:x + w_c]
        coloured_px = cv2.countNonZero(stone_mask)
        fill_ratio = coloured_px / area if area > 0 else 0.0

        # Centroid from filled pixels (same as production: improvement #6)
        Mpx = cv2.moments(stone_mask)
        if Mpx["m00"] > 0:
            sx = x + Mpx["m10"] / Mpx["m00"]
            sy = y + Mpx["m01"] / Mpx["m00"]
        else:
            Mc = cv2.moments(c)
            if Mc["m00"] == 0:
                continue
            sx = Mc["m10"] / Mc["m00"]
            sy = Mc["m01"] / Mc["m00"]

        if orientation == "top":
            nx = (sx - house_cx) / house_radius
            ny = (sy - house_cy) / house_radius
        else:
            nx = (house_cx - sx) / house_radius
            ny = (house_cy - sy) / house_radius

        dist = math.sqrt(nx * nx + ny * ny)
        angle = math.degrees(math.atan2(ny, nx))

        if fill_ratio >= STONE_MIN_FILL_RATIO:
            filled.append((nx, ny, dist, angle))
        else:
            ghosts.append((nx, ny, dist, angle))

    filled.sort(key=lambda s: s[2])
    ghosts.sort(key=lambda s: s[2])
    return filled, ghosts


def _match_sets(gt_stones, det_stones, match_radius=MATCH_RADIUS):
    """Greedy nearest-neighbour matching between two position lists.

    Both lists contain tuples whose first two elements are (nx, ny) in
    normalised coordinates.

    Returns
    -------
    tp_pairs : list of (gt_idx, det_idx, distance)
        Each matched pair, sorted ascending by distance.
    fp_indices : set of int
        Detected stone indices with no ground-truth match.
    fn_indices : set of int
        Ground-truth indices with no detected match.
    """
    pairs = []
    for gi, gs in enumerate(gt_stones):
        for di, ds in enumerate(det_stones):
            d = math.sqrt((gs[0] - ds[0]) ** 2 + (gs[1] - ds[1]) ** 2)
            if d < match_radius:
                pairs.append((d, gi, di))
    pairs.sort()

    matched_gt = set()
    matched_det = set()
    tp_pairs = []

    for d, gi, di in pairs:
        if gi in matched_gt or di in matched_det:
            continue
        tp_pairs.append((gi, di, d))
        matched_gt.add(gi)
        matched_det.add(di)

    fp_indices = set(range(len(det_stones))) - matched_det
    fn_indices = set(range(len(gt_stones))) - matched_gt
    return tp_pairs, fp_indices, fn_indices


# ---------------------------------------------------------------------------
# Public API — single crop
# ---------------------------------------------------------------------------

def evaluate_shot_crop(crop_bgr, color_ranges=None):
    """Evaluate detection accuracy for a single shot crop.

    Runs both the permissive ground-truth extraction and the production
    pipeline on the same BGR crop image and returns a metrics dict.

    Parameters
    ----------
    crop_bgr : np.ndarray
        BGR image of a single shot diagram crop.
    color_ranges : tuple or None
        Per-page calibrated HSV bounds from ``_calibrate_stone_colors``.
        Pass ``None`` to use the global static constants.

    Returns
    -------
    dict with keys:

    ``gt_red``, ``gt_yellow``
        Ground-truth active stone counts per colour.
    ``gt_ghost_red``, ``gt_ghost_yellow``
        Ground-truth ghost ring counts per colour.
    ``det_red``, ``det_yellow``
        Production-detected active stone counts per colour.
    ``det_ghost_red``, ``det_ghost_yellow``
        Production-detected ghost ring counts per colour.
    ``tp_red``, ``fp_red``, ``fn_red``
        True positive / false positive / false negative counts for red stones.
    ``tp_yellow``, ``fp_yellow``, ``fn_yellow``
        Equivalent for yellow stones.
    ``tp_ghost_red``, ``fp_ghost_red``, ``fn_ghost_red``
        Equivalent for red ghost rings.
    ``tp_ghost_yellow``, ``fp_ghost_yellow``, ``fn_ghost_yellow``
        Equivalent for yellow ghost rings.
    ``position_errors``
        List of normalised-unit centroid distances for all active-stone TPs.
    ``house_cx``, ``house_cy``, ``house_radius``, ``orientation``
        Detected house parameters (for back-projection in overlay rendering).
    """
    house_cx, house_cy, house_radius, orientation = _detect_house_center(crop_bgr)
    scale_sq = (house_radius / HOUSE_RADIUS) ** 2

    # Production detections
    red_det, red_ghost_det, yel_det, yel_ghost_det, _ = _detect_stones_in_crop(
        crop_bgr, color_ranges
    )

    # Ground-truth blobs via permissive extraction on the same masks
    mask_red, mask_yellow = _build_colour_masks(crop_bgr, color_ranges)
    gt_red, gt_red_ghosts = _permissive_blobs(
        mask_red, house_cx, house_cy, house_radius, orientation, scale_sq
    )
    gt_yel, gt_yel_ghosts = _permissive_blobs(
        mask_yellow, house_cx, house_cy, house_radius, orientation, scale_sq
    )

    def _eval(gt_list, det_list):
        tp_pairs, fp_idx, fn_idx = _match_sets(gt_list, det_list)
        errs = [d for (_, _, d) in tp_pairs]
        return len(tp_pairs), len(fp_idx), len(fn_idx), errs, tp_pairs, fp_idx, fn_idx

    tp_r, fp_r, fn_r, errs_r, tp_pairs_r, fp_idx_r, fn_idx_r = _eval(gt_red, red_det)
    tp_y, fp_y, fn_y, errs_y, tp_pairs_y, fp_idx_y, fn_idx_y = _eval(gt_yel, yel_det)
    tp_gr, fp_gr, fn_gr, errs_gr, _, _, _ = _eval(gt_red_ghosts, red_ghost_det)
    tp_gy, fp_gy, fn_gy, errs_gy, _, _, _ = _eval(gt_yel_ghosts, yel_ghost_det)

    return {
        "gt_red": len(gt_red),
        "gt_yellow": len(gt_yel),
        "gt_ghost_red": len(gt_red_ghosts),
        "gt_ghost_yellow": len(gt_yel_ghosts),
        "det_red": len(red_det),
        "det_yellow": len(yel_det),
        "det_ghost_red": len(red_ghost_det),
        "det_ghost_yellow": len(yel_ghost_det),
        "tp_red": tp_r,   "fp_red": fp_r,   "fn_red": fn_r,
        "tp_yellow": tp_y, "fp_yellow": fp_y, "fn_yellow": fn_y,
        "tp_ghost_red": tp_gr, "fp_ghost_red": fp_gr, "fn_ghost_red": fn_gr,
        "tp_ghost_yellow": tp_gy, "fp_ghost_yellow": fp_gy, "fn_ghost_yellow": fn_gy,
        "position_errors": errs_r + errs_y,
        # Store raw lists for overlay rendering
        "_gt_red": gt_red,
        "_gt_yellow": gt_yel,
        "_det_red": red_det,
        "_det_yellow": yel_det,
        "_tp_pairs_red": tp_pairs_r,
        "_tp_pairs_yellow": tp_pairs_y,
        "_fp_idx_red": fp_idx_r,
        "_fp_idx_yellow": fp_idx_y,
        "_fn_idx_red": fn_idx_r,
        "_fn_idx_yellow": fn_idx_y,
        "house_cx": house_cx,
        "house_cy": house_cy,
        "house_radius": house_radius,
        "orientation": orientation,
    }


# ---------------------------------------------------------------------------
# Overlay rendering
# ---------------------------------------------------------------------------

def _norm_to_crop_pixel(nx, ny, house_cx, house_cy, house_radius, orientation):
    """Back-project normalised coordinates to pixel coordinates in the crop."""
    if orientation == "top":
        px = nx * house_radius + house_cx
        py = ny * house_radius + house_cy
    else:
        px = house_cx - nx * house_radius
        py = house_cy - ny * house_radius
    return int(round(px)), int(round(py))


def draw_evaluation_overlay(crop_bgr, result):
    """Draw a detection evaluation overlay on top of the original crop.

    Annotates:

    - **Green circle** — True Positive (GT blob matched by a detection)
    - **Orange cross** — False Positive (detected stone with no GT blob)
    - **Red cross** — False Negative (GT blob not detected)
    - **Grey dashed ring** — ground truth centroid position for TPs

    Parameters
    ----------
    crop_bgr : np.ndarray
        The original shot crop (BGR).
    result : dict
        Return value of :func:`evaluate_shot_crop`.

    Returns
    -------
    PIL.Image.Image
        The annotated crop image (RGB).
    """
    hx = result["house_cx"]
    hy = result["house_cy"]
    hr = result["house_radius"]
    ori = result["orientation"]

    stone_r_px = max(4, int(hr * 0.078))  # STONE_RADIUS in pixels

    # Convert to PIL for drawing
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)

    def _px(nx, ny):
        return _norm_to_crop_pixel(nx, ny, hx, hy, hr, ori)

    def _circle(stone, colour, width=2, radius_px=None):
        px, py = _px(stone[0], stone[1])
        r = radius_px if radius_px is not None else stone_r_px
        draw.ellipse(
            [px - r, py - r, px + r, py + r],
            outline=colour, width=width
        )

    def _cross(stone, colour, size=6, width=2):
        px, py = _px(stone[0], stone[1])
        draw.line([(px - size, py - size), (px + size, py + size)],
                  fill=colour, width=width)
        draw.line([(px - size, py + size), (px + size, py - size)],
                  fill=colour, width=width)

    gt_r = result["_gt_red"]
    gt_y = result["_gt_yellow"]
    det_r = result["_det_red"]
    det_y = result["_det_yellow"]
    tp_pairs_r = result["_tp_pairs_red"]
    tp_pairs_y = result["_tp_pairs_yellow"]
    fp_idx_r = result["_fp_idx_red"]
    fp_idx_y = result["_fp_idx_yellow"]
    fn_idx_r = result["_fn_idx_red"]
    fn_idx_y = result["_fn_idx_yellow"]

    # Grey rings for all GT positions (background layer)
    for s in gt_r:
        _circle(s, (160, 160, 160), width=1, radius_px=stone_r_px + 3)
    for s in gt_y:
        _circle(s, (160, 160, 160), width=1, radius_px=stone_r_px + 3)

    # True Positives — green circle at detected position
    tp_det_r = {di for (_, di, _) in tp_pairs_r}
    tp_det_y = {di for (_, di, _) in tp_pairs_y}
    for di in tp_det_r:
        _circle(det_r[di], (0, 200, 0), width=2)
    for di in tp_det_y:
        _circle(det_y[di], (0, 200, 0), width=2)

    # False Positives — orange cross at detected position
    for di in fp_idx_r:
        _cross(det_r[di], (255, 120, 0), size=7, width=3)
    for di in fp_idx_y:
        _cross(det_y[di], (255, 120, 0), size=7, width=3)

    # False Negatives — red cross at GT position
    for gi in fn_idx_r:
        _cross(gt_r[gi], (220, 30, 30), size=7, width=3)
    for gi in fn_idx_y:
        _cross(gt_y[gi], (220, 30, 30), size=7, width=3)

    return img


# ---------------------------------------------------------------------------
# Public API — full PDF evaluation
# ---------------------------------------------------------------------------

def evaluate_pdf(pdf_path, max_ends=None, verbose=True, overlay_dir=None):
    """Evaluate detection accuracy across all shot crops in a PDF.

    Parameters
    ----------
    pdf_path : str
        Local file path or HTTP(S) URL to a tournament result PDF.
    max_ends : int or None
        If set, stop after this many ends (pages) have been evaluated.
        Useful for a quick smoke-test on a subset of the PDF.
    verbose : bool
        Whether to print per-end progress to stdout.
    overlay_dir : str or None
        If set, save overlay images for every shot to this directory.
        Each image is named ``end{E:02d}_shot{S:02d}.png``.

    Returns
    -------
    summary : dict
        Aggregate metrics across the entire evaluated run.
    shot_records : list of dict
        One dict per evaluated shot with columns suitable for a CSV file.
    """
    if overlay_dir:
        os.makedirs(overlay_dir, exist_ok=True)

    pdf = _open_pdf(pdf_path)
    shot_pages = find_shot_pages(pdf)

    if not shot_pages:
        print("No shot pages found in this PDF.")
        pdf.close()
        return {}, []

    shot_records = []
    ends_evaluated = 0

    for page_idx in shot_pages:
        if max_ends is not None and ends_evaluated >= max_ends:
            break

        page = pdf.pages[page_idx]
        page_text = page.extract_text() or ""
        end_info = parse_end_line(page_text)
        end_number = end_info["end_number"] if end_info else (ends_evaluated + 1)

        shot_images = _get_shot_images(page)
        if len(shot_images) != 16:
            if verbose:
                print(f"  End {end_number}: skipping ({len(shot_images)} shot images, expected 16)")
            continue

        ends_evaluated += 1
        if verbose:
            print(f"  Evaluating end {end_number} (page {page_idx}) …", end="", flush=True)

        page_bgr = render_page_image(page)
        color_ranges = _calibrate_stone_colors(page_bgr, page)

        end_tp = end_fp = end_fn = 0

        for shot_idx in range(16):
            shot_num = shot_idx + 1
            crop = crop_shot_image(page_bgr, shot_images[shot_idx])
            result = evaluate_shot_crop(crop, color_ranges)

            tp = result["tp_red"] + result["tp_yellow"]
            fp = result["fp_red"] + result["fp_yellow"]
            fn = result["fn_red"] + result["fn_yellow"]
            end_tp += tp
            end_fp += fp
            end_fn += fn

            record = {
                "end_number": end_number,
                "shot_number": shot_num,
                "gt_red": result["gt_red"],
                "gt_yellow": result["gt_yellow"],
                "det_red": result["det_red"],
                "det_yellow": result["det_yellow"],
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "gt_ghost_red": result["gt_ghost_red"],
                "gt_ghost_yellow": result["gt_ghost_yellow"],
                "det_ghost_red": result["det_ghost_red"],
                "det_ghost_yellow": result["det_ghost_yellow"],
                "tp_ghost": result["tp_ghost_red"] + result["tp_ghost_yellow"],
                "fp_ghost": result["fp_ghost_red"] + result["fp_ghost_yellow"],
                "fn_ghost": result["fn_ghost_red"] + result["fn_ghost_yellow"],
                "pos_err_median": (
                    round(float(np.median(result["position_errors"])), 4)
                    if result["position_errors"] else None
                ),
                "pos_err_p95": (
                    round(float(np.percentile(result["position_errors"], 95)), 4)
                    if result["position_errors"] else None
                ),
                "house_radius": round(result["house_radius"], 1),
                "orientation": result["orientation"],
            }
            shot_records.append(record)

            if overlay_dir:
                overlay_img = draw_evaluation_overlay(crop, result)
                fname = f"end{end_number:02d}_shot{shot_num:02d}.png"
                overlay_img.save(os.path.join(overlay_dir, fname))

        if verbose:
            prec = end_tp / max(end_tp + end_fp, 1) * 100
            rec = end_tp / max(end_tp + end_fn, 1) * 100
            print(
                f"  GT={end_tp + end_fn}  det={end_tp + end_fp}  "
                f"TP={end_tp}  FP={end_fp}  FN={end_fn}  "
                f"prec={prec:.1f}%  rec={rec:.1f}%"
            )

    pdf.close()

    summary = _aggregate(shot_records, pdf_path, ends_evaluated)
    return summary, shot_records


def _aggregate(shot_records, pdf_path, ends_evaluated):
    """Compute aggregate metrics from per-shot records."""
    if not shot_records:
        return {}

    total_gt = sum(r["gt_red"] + r["gt_yellow"] for r in shot_records)
    total_det = sum(r["det_red"] + r["det_yellow"] for r in shot_records)
    total_tp = sum(r["tp"] for r in shot_records)
    total_fp = sum(r["fp"] for r in shot_records)
    total_fn = sum(r["fn"] for r in shot_records)

    total_gt_ghost = sum(r["gt_ghost_red"] + r["gt_ghost_yellow"] for r in shot_records)
    total_tp_ghost = sum(r["tp_ghost"] for r in shot_records)
    total_fn_ghost = sum(r["fn_ghost"] for r in shot_records)
    total_fp_ghost = sum(r["fp_ghost"] for r in shot_records)

    all_errors = [
        r["pos_err_median"] for r in shot_records if r["pos_err_median"] is not None
    ]

    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    ghost_recall = total_tp_ghost / max(total_tp_ghost + total_fn_ghost, 1)
    ghost_precision = total_tp_ghost / max(total_tp_ghost + total_fp_ghost, 1)

    shots_with_fp = sum(1 for r in shot_records if r["fp"] > 0)
    shots_with_fn = sum(1 for r in shot_records if r["fn"] > 0)

    return {
        "pdf": os.path.basename(pdf_path),
        "ends_evaluated": ends_evaluated,
        "shots_evaluated": len(shot_records),
        # Active stones
        "total_gt_stones": total_gt,
        "total_det_stones": total_det,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2),
        "f1": round(f1 * 100, 2),
        "pos_err_median_norm": round(float(np.median(all_errors)), 4) if all_errors else None,
        "pos_err_p95_norm": round(float(np.percentile(all_errors, 95)), 4) if all_errors else None,
        "shots_with_fp": shots_with_fp,
        "shots_with_fn": shots_with_fn,
        # Ghost rings
        "total_gt_ghosts": total_gt_ghost,
        "total_tp_ghost": total_tp_ghost,
        "total_fp_ghost": total_fp_ghost,
        "total_fn_ghost": total_fn_ghost,
        "ghost_precision": round(ghost_precision * 100, 2),
        "ghost_recall": round(ghost_recall * 100, 2),
    }


def print_summary(summary):
    """Print a formatted summary of aggregate evaluation metrics."""
    if not summary:
        print("No results to display.")
        return

    w = 56
    print()
    print("=" * w)
    print(f"  Detection Evaluation — {summary['pdf']}")
    print("=" * w)
    print(f"  Ends evaluated        : {summary['ends_evaluated']}")
    print(f"  Shots evaluated       : {summary['shots_evaluated']}")
    print()
    print("  ── Active stones ──────────────────────────────────")
    print(f"  Ground-truth blobs    : {summary['total_gt_stones']:>7,}")
    print(f"  Detected              : {summary['total_det_stones']:>7,}")
    print(f"  True Positives        : {summary['total_tp']:>7,}")
    print(f"  False Positives (FP)  : {summary['total_fp']:>7,}  (hallucinated)")
    print(f"  False Negatives (FN)  : {summary['total_fn']:>7,}  (missed)")
    print(f"  Precision             : {summary['precision']:>6.2f} %")
    print(f"  Recall                : {summary['recall']:>6.2f} %")
    print(f"  F1                    : {summary['f1']:>6.2f} %")
    if summary["pos_err_median_norm"] is not None:
        print(f"  Position error median : {summary['pos_err_median_norm']:.4f} norm. units")
        print(f"  Position error p95    : {summary['pos_err_p95_norm']:.4f} norm. units")
    print(f"  Shots with FP         : {summary['shots_with_fp']:>7,}")
    print(f"  Shots with FN         : {summary['shots_with_fn']:>7,}")
    print()
    print("  ── Ghost rings ────────────────────────────────────")
    print(f"  Ground-truth ghosts   : {summary['total_gt_ghosts']:>7,}")
    print(f"  Ghost Precision       : {summary['ghost_precision']:>6.2f} %")
    print(f"  Ghost Recall          : {summary['ghost_recall']:>6.2f} %")
    print("=" * w)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate stone detection accuracy against original PDF crops."
    )
    parser.add_argument(
        "pdf",
        help="Path to a result PDF (local file or HTTP(S) URL).",
    )
    parser.add_argument(
        "--max-ends",
        type=int,
        default=None,
        metavar="N",
        help="Evaluate only the first N ends (pages). Useful for a quick smoke-test.",
    )
    parser.add_argument(
        "--overlay-dir",
        default=None,
        metavar="DIR",
        help=(
            "Save annotated overlay images to DIR. "
            "Each shot is saved as end<E>_shot<S>.png with TP/FP/FN markers."
        ),
    )
    parser.add_argument(
        "--csv-out",
        default=None,
        metavar="PATH",
        help="Write per-shot evaluation records to a CSV file.",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        metavar="PATH",
        help="Write aggregate summary metrics to a JSON file (for CI collection).",
    )
    args = parser.parse_args()

    print(f"Evaluating: {args.pdf}")
    if args.max_ends:
        print(f"(Limited to first {args.max_ends} ends)")
    print()

    summary, shot_records = evaluate_pdf(
        args.pdf,
        max_ends=args.max_ends,
        verbose=True,
        overlay_dir=args.overlay_dir,
    )

    print_summary(summary)

    if args.csv_out and shot_records:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv_out)), exist_ok=True)
        fieldnames = list(shot_records[0].keys())
        with open(args.csv_out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(shot_records)
        print(f"Per-shot results written to {args.csv_out}")

    if args.json_out and summary:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Aggregate metrics written to {args.json_out}")

    if args.overlay_dir:
        print(f"Overlay images saved to {args.overlay_dir}/")


if __name__ == "__main__":
    main()
