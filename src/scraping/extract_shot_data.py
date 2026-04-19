"""
Curling Shot Data Extractor

Parses curling tournament PDFs to extract match, end, and shot-level data
including stone positions detected via OpenCV color-based segmentation.

PDFs can be provided as local file paths or HTTP(S) URLs.  When no arguments
are supplied, the script reads ``result_urls.csv`` (produced by
``scrape_results.py``) and processes every result-book PDF dated 2013 or
later.

Usage:
    python extract_shot_data.py [<pdf_or_url> ...] [--output-dir <dir>]
                                [--results-csv <path>] [--min-year <year>]

Outputs CSV files:
    - events.csv
    - matches.csv
    - teams.csv
    - players.csv
    - ends.csv
    - shot_locations_raw.csv   (no tracking columns; run track_stones.py to produce
                               the tracked shot_locations.parquet used by the ML pipeline)
"""

import argparse
import collections
import csv
import gc
import io
import json
import math
import multiprocessing
import os
import re
import shutil
import tempfile
import urllib.error
import urllib.request

import cv2
import numpy as np
import pandas as pd
import pdfplumber
import pyarrow as pa
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Default PDF URLs (curlit.com result books)
# ---------------------------------------------------------------------------
DEFAULT_PDF_URLS = [
    "https://curlit.com/PDF/ECC2025_ResultsBook_Men_A-Division.pdf",
    "https://curlit.com/PDF/WMCC2023_ResultsBook.pdf",
]

DEFAULT_RESULTS_CSV = os.path.join("output", "result_urls.csv")
DEFAULT_MIN_YEAR = 2013

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RENDER_DPI = 300
PDF_POINTS_PER_INCH = 72
SCALE = RENDER_DPI / PDF_POINTS_PER_INCH

# House geometry at 300 DPI (calibrated by measuring the 12-foot ring radius
# in the rendered PDF crop; original empirical value was 89, corrected to 112
# after visual comparison against generate_board_image.py output).
HOUSE_CX = 161  # pixels – centre-x of the house in each shot crop
HOUSE_CY = 171  # pixels – centre-y
HOUSE_RADIUS = 112  # pixels – radius of the 12-foot ring

# Stone detection thresholds (HSV)
RED_LOWER_1 = np.array([0, 100, 80])
RED_UPPER_1 = np.array([10, 255, 255])
RED_LOWER_2 = np.array([160, 100, 80])
RED_UPPER_2 = np.array([180, 255, 255])
YELLOW_LOWER = np.array([10, 100, 100])
YELLOW_UPPER = np.array([35, 255, 255])

STONE_MIN_AREA = 80
STONE_MAX_AREA = 600

# Minimum ratio of colored pixels to contour area for a detected blob to be
# accepted as a filled stone.  A solid disc has ratio ≈ 1.0.  Outline-only
# "ghost" markers (showing where a stone was before the shot) appear as an
# annular ring; cv2.contourArea() on the outer boundary returns approximately
# the full disc area while the actual colored pixels are only the ring,
# giving a ratio well below this threshold.
STONE_MIN_FILL_RATIO = 0.45

# HSV colour range for the 12-foot ring (blue/lilac) used to detect the
# house centre position and image orientation.  The ring colour in the PDF
# is approximately RGB (170, 170, 230) → HSV H≈120, S≈66, V≈230.
HOUSE_RING_LOWER = np.array([100, 25, 140])  # HSV lower bound
HOUSE_RING_UPPER = np.array([140, 160, 255])  # HSV upper bound
HOUSE_RING_MIN_AREA = 1500  # minimum blue pixels required to trust detection

# Broader fallback HSV range for 12-foot ring detection when the primary narrow
# range collects insufficient pixels (e.g. low-saturation or differently-coloured
# event PDFs).  Tried automatically before falling back to fixed constants.
HOUSE_RING_LOWER_BROAD = np.array([80, 15, 100])   # HSV lower bound (broad)
HOUSE_RING_UPPER_BROAD = np.array([160, 180, 255])  # HSV upper bound (broad)

# Per-page stone colour calibration settings (improvement #3).
# When a team indicator image is found on the page, its dominant HSV hue is
# used as the centre of a detection window of this half-width.
STONE_COLOR_HUE_TOL = 12      # hue half-window (OpenCV 0-180 scale)
STONE_COLOR_CAL_SAT_MIN = 60  # minimum saturation to include in indicator sample
STONE_COLOR_CAL_VAL_MIN = 60  # minimum value (brightness) for indicator sample

# Vertical pixel margins for stone detection (exclude score-indicator dots
# at the very top and shot-label text at the bottom of each crop).
# With HOUSE_RADIUS=112 and HOUSE_CY=171, the hog line sits at pixel
# 171 + 3.5×112 = 563.  The crop height ≈ 686 px, so the old value of
# 0.82 placed the cutoff exactly at the hog line, silently dropping any
# stone close to it.  0.93 extends detection to ~y=4.2 (25 ft) while
# still clearing the text label at the very bottom of the crop.
STONE_Y_MIN_PX = 15
STONE_Y_MAX_FRAC = 0.93  # fraction of crop height

MAX_STONES_PER_TEAM = 8

# Maximum Euclidean distance (normalised to house-radius units) within which
# two stone detections — in consecutive shot diagrams of the same end — are
# considered the same physical stone.  Rendering noise produces position
# jitter of ±0.01–0.03 units; genuine displacement by a hit travels >> 0.13
# units, so this threshold cleanly separates noise from true movement.
STONE_TRACK_MAX_DIST = 0.13

# Minimum separation (normalised units) between two accepted active-stone
# detections.  When two contours are closer than this, only the one nearer
# to the house centre is kept (improvement #13: deduplication of overlapping
# detections caused by PDF rendering splitting one stone into two blobs).
STONE_DEDUP_RADIUS = 0.08

# ---------------------------------------------------------------------------
# URL / PDF helpers
# ---------------------------------------------------------------------------


def _is_url(source):
    """Return True if *source* looks like an HTTP(S) URL."""
    return source.startswith(("http://", "https://"))


def _open_pdf(source):
    """Open a PDF from a local path or URL, returning a pdfplumber PDF object.

    For URL sources, the content is streamed to a temporary file on disk so
    that the PDF bytes do not occupy Python's heap.  The caller must close
    the returned object via :func:`_close_pdf` to ensure the temp file is
    removed.
    """
    if _is_url(source):
        tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        tmp_path = tmp.name
        try:
            with urllib.request.urlopen(source, timeout=60) as response:
                shutil.copyfileobj(response, tmp)
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
            tmp.close()
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise RuntimeError(f"Failed to download PDF from {source}: {exc}") from exc
        tmp.close()
        try:
            pdf = pdfplumber.open(tmp_path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
        pdf._tmp_path = tmp_path  # stash for cleanup in _close_pdf
        return pdf
    return pdfplumber.open(source)


def _close_pdf(pdf):
    """Close *pdf* and remove the associated temp file, if any."""
    tmp_path = getattr(pdf, "_tmp_path", None)
    pdf.close()
    if tmp_path is not None:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def load_result_urls(csv_path, min_year=DEFAULT_MIN_YEAR):
    """Load result-book URLs from *csv_path*, keeping only rows with year >= *min_year*.

    Returns a list of dicts, each with keys:
        result_book_url, tournament_name, year, location, gender,
        result_summary_url
    """
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                year = int(row["year"])
            except (ValueError, KeyError):
                continue
            if year >= min_year and row.get("result_book_url"):
                rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# PDF helpers
# ---------------------------------------------------------------------------


def find_shot_pages(pdf):
    """Return list of 0-based page indices containing 'Game - Shot by Shot'."""
    pages = []
    for i, page in enumerate(pdf.pages):
        text = page.extract_text() or ""
        if "Game - Shot by Shot" in text:
            pages.append(i)
    return pages


def _keep_regular_columns(images, n):
    """Select *n* images from *images* (sorted ascending by ``x0``) that form
    the most evenly-spaced column layout.

    When there is exactly one extra image, tries every possible single-image
    removal and returns the subset whose column spacing has the smallest
    variance.  Returns ``None`` when more than one extra image is present
    (caller falls back to the original candidate list).
    """
    excess = len(images) - n
    if excess <= 0:
        return images[:n]
    if excess > 1:
        return None  # Too many extras; caller must fall back

    best_var = float("inf")
    best_subset = None
    for skip in range(len(images)):
        subset = [img for i, img in enumerate(images) if i != skip]
        xs = [img["x0"] for img in subset]
        if len(xs) < 2:
            continue
        mean_spacing = (xs[-1] - xs[0]) / (len(xs) - 1)
        spacings = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
        var = sum((s - mean_spacing) ** 2 for s in spacings)
        if var < best_var:
            best_var = var
            best_subset = subset
    return best_subset


def _filter_shot_images_by_grid(candidates):
    """Filter candidate shot images to the expected 3-row, 6-6-4 grid layout.

    Uses the two largest vertical gaps between consecutive ``top`` values to
    split all candidates into three row groups, then validates (and optionally
    trims) each row against its expected column count (6, 6, 4).  Returns
    *candidates* unchanged when a clean 16-image grid cannot be recovered so
    that the caller can fall back to the size-only result (improvement #8).
    """
    sorted_by_y = sorted(candidates, key=lambda img: img["top"])
    tops = [img["top"] for img in sorted_by_y]

    if len(tops) < 3:
        return candidates

    # Find the two largest vertical gaps to identify row boundaries.
    gaps = [(tops[i + 1] - tops[i], i) for i in range(len(tops) - 1)]
    gap_indices = sorted(
        [g[1] for g in sorted(gaps, key=lambda g: -g[0])[:2]]
    )
    if len(gap_indices) < 2:
        return candidates

    split1, split2 = gap_indices
    row_groups = [
        sorted_by_y[: split1 + 1],
        sorted_by_y[split1 + 1 : split2 + 1],
        sorted_by_y[split2 + 1 :],
    ]
    expected_counts = [6, 6, 4]

    grid_images = []
    for row, expected in zip(row_groups, expected_counts):
        row_sorted = sorted(row, key=lambda img: img["x0"])
        if len(row_sorted) == expected:
            grid_images.extend(row_sorted)
        elif len(row_sorted) > expected:
            kept = _keep_regular_columns(row_sorted, expected)
            if kept is None:
                return candidates  # Too many extras; cannot resolve safely
            grid_images.extend(kept)
        else:
            return candidates  # Fewer images than expected; cannot resolve

    return grid_images if len(grid_images) == 16 else candidates


def _get_shot_images(page):
    """Return the 16 shot-diagram image metadata objects from *page*.

    Filters candidate images by minimum size and then, when more than 16
    candidates pass, validates the expected 3-row 6-6-4 grid layout to
    exclude spurious images such as logos or decorative graphics that would
    otherwise silently misalign subsequent shot assignments (improvement #8).
    """
    candidates = [
        img
        for img in page.images
        if img["width"] > 50 and img["height"] > 100
    ]
    if len(candidates) > 16:
        candidates = _filter_shot_images_by_grid(candidates)
    return candidates


# ---------------------------------------------------------------------------
# Text / metadata parsing
# ---------------------------------------------------------------------------

_END_LINE_RE = re.compile(
    r"End\s+(\d+)\s+"
    r"(\w+)\s*-\s*(.+?)\s+"
    r"(\d+)\s*\+\s*(\d+|X)\s*\(this end\)\s*=\s*(\d+)\s+"
    r"(\w+)\s*-\s*(.+?)\s+"
    r"(\d+)\s*\+\s*(\d+|X)\s*\(this end\)\s*=\s*(\d+)"
)

_MATCH_HEADER_RE = re.compile(
    r"([A-Z]{3}\s+\d{1,2}\s+[A-Z]{3}\s+\d{4})\s+(.*)"
)

_ACCURACY_RE = re.compile(r"(\d+)%")

_TURN_CHARS = {"↺": "Counter-clockwise", "↻": "Clockwise"}


def parse_match_header(page_text):
    """Extract date, round description and start time from page header."""
    lines = page_text.split("\n")
    date_str = round_name = start_time = ""
    for line in lines:
        m = _MATCH_HEADER_RE.match(line.strip())
        if m:
            date_str = m.group(1).strip()
            round_name = m.group(2).strip()
        if "Start Time" in line:
            start_time = line.split("Start Time")[-1].strip()
    return date_str, round_name, start_time


def parse_end_line(page_text):
    """Parse the 'End N  TEAM1 …  TEAM2 …' line.

    Returns dict with keys: end_number, team1_code, team1_name,
    team1_score_before, team1_score_this_end, team1_score_after,
    team2_code, team2_name, team2_score_before, team2_score_this_end,
    team2_score_after.
    """
    m = _END_LINE_RE.search(page_text)
    if not m:
        return None
    def _int_or_x(val):
        return val if val == "X" else int(val)

    return {
        "end_number": int(m.group(1)),
        "team1_code": m.group(2),
        "team1_name": m.group(3).strip(),
        "team1_score_before": int(m.group(4)),
        "team1_score_this_end": _int_or_x(m.group(5)),
        "team1_score_after": int(m.group(6)),
        "team2_code": m.group(7),
        "team2_name": m.group(8).strip(),
        "team2_score_before": int(m.group(9)),
        "team2_score_this_end": _int_or_x(m.group(10)),
        "team2_score_after": int(m.group(11)),
    }


def parse_score_box(page_text):
    """Parse Total Score and Time left from the score box at the bottom."""
    lines = page_text.split("\n")
    total_scores = {}
    time_left = {}
    for i, line in enumerate(lines):
        if "Total Score" in line:
            parts = line.split()
            idx = parts.index("Score") + 1
            if idx + 1 < len(parts):
                total_scores["team1"] = parts[idx]
                total_scores["team2"] = parts[idx + 1]
        if "Time left" in line:
            parts = line.split()
            idx = parts.index("left") + 1
            if idx + 1 < len(parts):
                time_left["team1"] = parts[idx]
                time_left["team2"] = parts[idx + 1]
    return total_scores, time_left


def _extract_shot_metadata_from_words(words, shot_images):
    """Extract per-shot metadata using word positions aligned with shot images.

    Returns list of 16 dicts with keys:
        team_code, player_name, shot_type, turn, accuracy
    """
    shots = [
        {"team_code": "", "player_name": "", "shot_type": "", "turn": "", "accuracy": ""}
        for _ in range(16)
    ]

    # Shots are arranged in 3 rows: 6 + 6 + 4.
    # Within each row, shots are in columns aligned with the shot images.
    row_definitions = [
        (0, 6),   # row 0: shots 0-5  (images 0-5)
        (6, 12),  # row 1: shots 6-11 (images 6-11)
        (12, 16), # row 2: shots 12-15 (images 12-15)
    ]

    # Build per-row column x-ranges and y-ranges for text lines
    row_col_ranges = []  # [(x0, x1)] per column within each row
    name_y_ranges = []   # (y_lo, y_hi) for player-name line
    type_y_ranges = []   # (y_lo, y_hi) for shot-type line

    for start, end in row_definitions:
        cols = [(shot_images[i]["x0"], shot_images[i]["x1"]) for i in range(start, end)]
        row_col_ranges.append(cols)
        img_bottom = shot_images[start]["bottom"]
        name_y_ranges.append((img_bottom - 2, img_bottom + 6))
        type_y_ranges.append((img_bottom + 6, img_bottom + 25))

    def _find_col_in_row(x, row_idx):
        """Return column index within the row, or None."""
        for ci, (x0, x1) in enumerate(row_col_ranges[row_idx]):
            if x0 - 5 <= x <= x1 + 5:
                return ci
        return None

    for w in words:
        top = w["top"]
        text = w["text"]
        x0 = w["x0"]

        # Determine which row and whether it's a name or type line
        row_idx = None
        is_type_row = False
        for ri, (y_lo, y_hi) in enumerate(name_y_ranges):
            if y_lo <= top <= y_hi:
                row_idx = ri
                break
        if row_idx is None:
            for ri, (y_lo, y_hi) in enumerate(type_y_ranges):
                if y_lo <= top <= y_hi:
                    row_idx = ri
                    is_type_row = True
                    break
        if row_idx is None:
            continue

        col_in_row = _find_col_in_row(x0, row_idx)
        if col_in_row is None:
            continue

        shot_idx = row_definitions[row_idx][0] + col_in_row
        if shot_idx >= 16:
            continue

        shot = shots[shot_idx]

        if not is_type_row:
            # Player name line: "TEAM: PLAYER_LAST PLAYER_FIRST"
            if ":" in text and len(text) <= 5:
                shot["team_code"] = text.rstrip(":")
            elif shot["team_code"] and not shot["player_name"]:
                shot["player_name"] = text
            elif shot["player_name"]:
                shot["player_name"] += " " + text
        else:
            # Shot type / accuracy line
            if text in _TURN_CHARS:
                shot["turn"] = _TURN_CHARS[text]
            elif text.startswith("↺") or text.startswith("↻"):
                shot["turn"] = _TURN_CHARS.get(text[0], text[0])
                acc = _ACCURACY_RE.search(text)
                if acc:
                    shot["accuracy"] = acc.group(1)
            elif "%" in text:
                acc = _ACCURACY_RE.search(text)
                if acc:
                    shot["accuracy"] = acc.group(1)
            elif text == "-":
                shot["turn"] = "Not considered"
            else:
                if shot["shot_type"]:
                    # Skip single lowercase letters and single digits that
                    # bleed in from adjacent PDF table cells (e.g. "Raise f",
                    # "Guard 4").  All legitimate multi-word continuation
                    # tokens ("and", "Roll", "Time-out", "Measurement", …)
                    # are longer than one character.
                    if not (len(text) == 1 and (text.islower() or text.isdigit())):
                        shot["shot_type"] += " " + text
                else:
                    shot["shot_type"] = text

    # Penalty "Through" shots (hog-line violation, FGZ violation, burned
    # stone, etc.) are removed from play — the PDF has no turn indicator for
    # them.  "Not considered" is the correct semantic value.
    for shot in shots:
        if not shot["turn"] and shot["shot_type"].startswith("Through"):
            shot["turn"] = "Not considered"

    return shots


# ---------------------------------------------------------------------------
# Stone detection (OpenCV)
# ---------------------------------------------------------------------------


def _detect_house_center(crop_bgr):
    """Detect the house centre position and 12-foot ring radius in a shot crop.

    The 12-foot ring has a distinctive blue/lilac colour that is isolated via
    HSV thresholding.  The centroid of all matching pixels gives the house
    centre.  Comparing that centroid to the vertical midpoint of the crop
    reveals the orientation: ``'top'`` when the house is in the upper half
    (guard zone below), ``'bottom'`` when the house is in the lower half
    (guard zone above).

    Two colour ranges are tried in order:

    1. The primary narrow range ``HOUSE_RING_LOWER / HOUSE_RING_UPPER``
       (tuned for the standard blue/lilac 12-foot ring).
    2. A broader fallback ``HOUSE_RING_LOWER_BROAD / HOUSE_RING_UPPER_BROAD``
       that is less sensitive to white-balance or print-colour variation.

    If both colour-mask attempts fail, a grayscale circle detector
    (``cv2.HoughCircles``) is used as a secondary fallback.  This handles
    event PDFs where the house rings are rendered in grayscale instead of the
    usual blue/lilac.

    Returns
    -------
    tuple (cx, cy, radius, orientation)
        ``cx``, ``cy`` are the sub-pixel house centre coordinates (float —
        improvement #5: no ``int()`` truncation).
        ``radius`` is the estimated 12-foot ring radius in pixels (float —
        improvement #1: derived from the median distance of detected ring
        pixels to the centroid rather than the fixed ``HOUSE_RADIUS`` constant).
        ``orientation`` is ``'top'`` or ``'bottom'``.
        Falls back to ``(HOUSE_CX, HOUSE_CY, HOUSE_RADIUS, 'top')`` only when
        both colour-mask and circle-detection fallbacks fail (improvement #2:
        logs a warning).
    """
    h, w = crop_bgr.shape[:2]
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)

    # Try primary colour range first, then the broader fallback (#2)
    ring_mask = M = None
    for lower, upper in [
        (HOUSE_RING_LOWER, HOUSE_RING_UPPER),
        (HOUSE_RING_LOWER_BROAD, HOUSE_RING_UPPER_BROAD),
    ]:
        candidate = cv2.inRange(hsv, lower, upper)
        m = cv2.moments(candidate)
        if m["m00"] >= HOUSE_RING_MIN_AREA:
            ring_mask, M = candidate, m
            break

    if ring_mask is None:
        # Colour-mask detection failed; try a grayscale circle detector as a
        # secondary fallback for events that render rings without blue colour.
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(40, h // 8),
            param1=100,
            param2=30,
            minRadius=int(HOUSE_RADIUS * 0.5),
            maxRadius=int(HOUSE_RADIUS * 1.6),
        )
        if circles is not None and len(circles[0]) > 0:
            circle_candidates = np.asarray(circles[0], dtype=float)

            # Choose the candidate closest to nominal radius and to either the
            # expected top-house position or its vertical mirror (bottom-house).
            top_y = HOUSE_CY
            bottom_y = h - HOUSE_CY

            def _score(c):
                cx_c, cy_c, r_c = c
                dy = min(abs(cy_c - top_y), abs(cy_c - bottom_y))
                return abs(r_c - HOUSE_RADIUS) + 0.25 * abs(cx_c - HOUSE_CX) + 0.25 * dy

            cx, cy, detected_radius = min(circle_candidates, key=_score)
            if not (HOUSE_RADIUS * 0.75 <= detected_radius <= HOUSE_RADIUS * 1.25):
                detected_radius = HOUSE_RADIUS
            orientation = "top" if cy <= h / 2 else "bottom"
            return float(cx), float(cy), float(detected_radius), orientation

        # Both colour-mask and circle detection failed – log a warning and
        # fall back to fixed calibration constants.
        print(
            "WARNING: house ring detection failed for a crop; "
            "stone coordinates may be inaccurate (using calibrated fallback)."
        )
        return HOUSE_CX, HOUSE_CY, HOUSE_RADIUS, "top"

    # Sub-pixel centroid (improvement #5: no int() truncation)
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]

    # Estimate the 12-foot ring radius from the median distance of ring pixels
    # to the detected centroid (improvement #1: dynamic radius per crop).
    ys, xs = np.where(ring_mask > 0)
    pixel_dists = np.hypot(xs.astype(float) - cx, ys.astype(float) - cy)
    median_radius = float(np.median(pixel_dists))
    del ys, xs, pixel_dists  # Free coordinate arrays now that median is computed.

    # Sanity-check: reject implausible estimates (outside ±25 % of the
    # calibrated constant) and fall back to the constant for that crop.
    if HOUSE_RADIUS * 0.75 <= median_radius <= HOUSE_RADIUS * 1.25:
        detected_radius = median_radius
    else:
        detected_radius = HOUSE_RADIUS

    orientation = "top" if cy <= h / 2 else "bottom"
    return cx, cy, detected_radius, orientation


def _calibrate_stone_colors(page_bgr, page):
    """Estimate per-page HSV stone-detection ranges from team indicator images.

    Each shot page contains small coloured indicator images that identify which
    team plays red stones and which plays yellow stones.  Sampling the dominant
    hue from those indicators allows the stone-detection thresholds to
    self-calibrate for events that render stones in slightly different shades
    (improvements #3 and #11).

    Both yellow and red ranges are calibrated from their respective indicator
    images.  The median (not mean) of sampled hues is used so that a single
    outlier indicator cannot shift the window off the true stone colour
    (improvement #14).  Red hue wraps at the 0/180 boundary in OpenCV HSV;
    two sub-ranges are built to straddle whichever side the sampled hue falls
    on (improvement #11).  Falls back to static constants per-channel if no
    indicator images of that colour can be found or sampled reliably.

    Parameters
    ----------
    page_bgr : np.ndarray
        Full rendered page image (BGR, 300 DPI).
    page : pdfplumber Page
        Corresponding pdfplumber page object (used to access ``page.images``).

    Returns
    -------
    tuple (red_lower1, red_upper1, red_lower2, red_upper2,
           yellow_lower, yellow_upper)
        Per-page HSV bounds as ``np.uint8`` arrays.
    """
    _static = (
        RED_LOWER_1, RED_UPPER_1, RED_LOWER_2, RED_UPPER_2,
        YELLOW_LOWER, YELLOW_UPPER,
    )

    # Team indicator images are small roughly-square raster images embedded in
    # the PDF (typically 20–50 PDF points wide and tall).
    candidates = [
        img for img in page.images
        if 15 < img.get("width", 0) < 55
        and 15 < img.get("height", 0) < 55
        and abs(img.get("width", 0) - img.get("height", 0)) < 10
    ]
    if not candidates:
        return _static

    h_pg, w_pg = page_bgr.shape[:2]
    yellow_hues = []
    red_hues = []

    for img_meta in candidates:
        x0 = max(0, int(img_meta["x0"] * SCALE))
        y0 = max(0, int(img_meta["top"] * SCALE))
        x1 = min(w_pg, int(img_meta["x1"] * SCALE))
        y1 = min(h_pg, int(img_meta["bottom"] * SCALE))
        if x1 <= x0 or y1 <= y0:
            continue
        crop = page_bgr[y0:y1, x0:x1]
        if crop.size == 0:
            continue

        hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        # Only sample sufficiently saturated and bright pixels.
        colour_mask = (
            (hsv_crop[:, :, 1] >= STONE_COLOR_CAL_SAT_MIN)
            & (hsv_crop[:, :, 2] >= STONE_COLOR_CAL_VAL_MIN)
        )
        hues = hsv_crop[:, :, 0][colour_mask]
        if len(hues) < 10:
            continue

        median_hue = float(np.median(hues))
        # Classify: yellow/orange range H 10–50; red range H ≤ 15 or H ≥ 165
        if 10 <= median_hue <= 50:
            yellow_hues.append(median_hue)
        elif median_hue <= 15 or median_hue >= 165:
            red_hues.append(median_hue)

    if not yellow_hues and not red_hues:
        return _static

    # ---- Yellow calibration (improvement #3) --------------------------------
    # Use median instead of mean so a single outlier indicator cannot skew
    # the centre hue off target (improvement #14).
    if yellow_hues:
        y_h = float(np.median(yellow_hues))
        yellow_lower = np.array(
            [max(0, int(y_h - STONE_COLOR_HUE_TOL)), int(YELLOW_LOWER[1]), int(YELLOW_LOWER[2])],
            dtype=np.uint8,
        )
        yellow_upper = np.array(
            [min(180, int(y_h + STONE_COLOR_HUE_TOL)), int(YELLOW_UPPER[1]), 255],
            dtype=np.uint8,
        )
    else:
        yellow_lower, yellow_upper = YELLOW_LOWER, YELLOW_UPPER

    # ---- Red calibration (improvement #11) ----------------------------------
    # Red hue wraps at the 0/180 boundary in OpenCV HSV.  Build two sub-ranges
    # that straddle whichever side of the boundary the sampled hue falls on.
    if red_hues:
        r_h = float(np.median(red_hues))
        r_lo_val = int(r_h - STONE_COLOR_HUE_TOL)
        r_hi_val = int(r_h + STONE_COLOR_HUE_TOL)
        # Primary sub-range (clamped to [0, 180])
        red_lower_1 = np.array(
            [max(0, r_lo_val), int(RED_LOWER_1[1]), int(RED_LOWER_1[2])],
            dtype=np.uint8,
        )
        red_upper_1 = np.array(
            [min(180, r_hi_val), int(RED_UPPER_1[1]), 255],
            dtype=np.uint8,
        )
        # Wrap-around sub-range: covers the opposite side of the 0/180 boundary
        if r_lo_val < 0:
            # r_h is near 0; wrap covers the top end (near 180)
            red_lower_2 = np.array(
                [180 + r_lo_val, int(RED_LOWER_2[1]), int(RED_LOWER_2[2])],
                dtype=np.uint8,
            )
            red_upper_2 = np.array([180, int(RED_UPPER_2[1]), 255], dtype=np.uint8)
        elif r_hi_val > 180:
            # r_h is near 180; wrap covers the bottom end (near 0)
            red_lower_2 = np.array([0, int(RED_LOWER_2[1]), int(RED_LOWER_2[2])], dtype=np.uint8)
            red_upper_2 = np.array(
                [r_hi_val - 180, int(RED_UPPER_2[1]), 255],
                dtype=np.uint8,
            )
        else:
            # No wraparound needed; keep static secondary range as fallback
            red_lower_2, red_upper_2 = RED_LOWER_2, RED_UPPER_2
    else:
        red_lower_1, red_upper_1 = RED_LOWER_1, RED_UPPER_1
        red_lower_2, red_upper_2 = RED_LOWER_2, RED_UPPER_2

    return (red_lower_1, red_upper_1, red_lower_2, red_upper_2, yellow_lower, yellow_upper)


def _detect_stones_in_crop(crop_bgr, color_ranges=None):
    """Detect red and yellow stones in a single shot crop image.

    The house centre and ring radius are detected dynamically via
    :func:`_detect_house_center` so that both orientations (house at top or
    bottom) and per-event scale variations are handled correctly.

    Parameters
    ----------
    crop_bgr : np.ndarray
        BGR image of a single shot diagram crop.
    color_ranges : tuple or None
        Optional per-page calibrated HSV bounds returned by
        :func:`_calibrate_stone_colors` (improvement #3).  When ``None`` the
        global static constants are used.

    Returns
    -------
    tuple (red_stones, red_ghosts, yellow_stones, yellow_ghosts, orientation)
        ``red_stones`` / ``yellow_stones`` are active-stone lists; each entry
        is ``(norm_x, norm_y, distance, angle_deg)`` sorted ascending by
        distance from the house centre.  ``red_ghosts`` / ``yellow_ghosts``
        contain the same tuple format for outline-ring ghost positions
        (improvement #12).  Coordinates are normalised to ``house_radius``
        units (1.0 = 12-foot ring boundary) using the dynamically detected
        radius (improvement #1).  ``orientation`` is ``'top'`` or ``'bottom'``.
    """
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    h, w = crop_bgr.shape[:2]

    # Detect house centre, ring radius, and orientation (improvements #1, #5)
    house_cx, house_cy, house_radius, orientation = _detect_house_center(crop_bgr)

    # Scale stone area bounds to match the detected rink scale (improvement #4)
    scale_sq = (house_radius / HOUSE_RADIUS) ** 2
    min_area = STONE_MIN_AREA * scale_sq
    max_area = STONE_MAX_AREA * scale_sq

    # Restrict detection to the playing-field portion of the crop
    y_max = int(h * STONE_Y_MAX_FRAC)
    play_mask = np.zeros((h, w), dtype=np.uint8)
    play_mask[STONE_Y_MIN_PX:y_max, :] = 255

    # Use per-page calibrated colour ranges if provided, else static constants
    # (improvement #3)
    if color_ranges is not None:
        red_lo1, red_hi1, red_lo2, red_hi2, yel_lo, yel_hi = color_ranges
    else:
        red_lo1, red_hi1 = RED_LOWER_1, RED_UPPER_1
        red_lo2, red_hi2 = RED_LOWER_2, RED_UPPER_2
        yel_lo, yel_hi = YELLOW_LOWER, YELLOW_UPPER

    mask_red = (
        cv2.inRange(hsv, red_lo1, red_hi1)
        | cv2.inRange(hsv, red_lo2, red_hi2)
    ) & play_mask

    mask_yellow = cv2.inRange(hsv, yel_lo, yel_hi) & play_mask

    # Morphological close to fill small holes and merge fragmented blobs
    # caused by PDF compression artefacts (improvement #7)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)
    mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_CLOSE, kernel)
    del hsv, play_mask  # No longer needed; free before per-contour work.

    def _extract(mask):
        """Return (filled_stones, ghost_stones) for one colour mask.

        ``filled_stones`` are active stones (high fill ratio); ``ghost_stones``
        are outline-ring contours marking the pre-shot position of displaced
        stones (improvement #12).  Both lists contain
        ``(nx, ny, dist, angle)`` tuples sorted ascending by distance.
        Duplicate active-stone detections closer than ``STONE_DEDUP_RADIUS``
        are collapsed to the nearer detection (improvement #13).
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = []
        ghosts = []
        for c in contours:
            area = cv2.contourArea(c)
            if not (min_area < area < max_area):
                continue

            x, y, w_c, h_c = cv2.boundingRect(c)
            stone_mask = mask[y:y + h_c, x:x + w_c]
            colored_pixels = cv2.countNonZero(stone_mask)
            fill_ratio = colored_pixels / area

            # Compute pixel-based centroid — works for both filled stones
            # (improvement #6) and outline rings (ghost positions).
            Mpx = cv2.moments(stone_mask)
            if Mpx["m00"] > 0:
                sx = x + Mpx["m10"] / Mpx["m00"]
                sy = y + Mpx["m01"] / Mpx["m00"]
            else:
                # Fallback to contour polygon moments
                Mc = cv2.moments(c)
                if Mc["m00"] == 0:
                    continue
                sx = Mc["m10"] / Mc["m00"]
                sy = Mc["m01"] / Mc["m00"]

            # Normalise coordinates to house-radius units (improvement #1).
            # Bottom orientation mirrors both axes relative to the standard view.
            if orientation == "top":
                nx = (sx - house_cx) / house_radius
                ny = (sy - house_cy) / house_radius
            else:
                nx = (house_cx - sx) / house_radius
                ny = (house_cy - sy) / house_radius
            dist = math.sqrt(nx * nx + ny * ny)
            angle = math.degrees(math.atan2(ny, nx))

            if fill_ratio < STONE_MIN_FILL_RATIO:
                # Outline-only ring → record as ghost position (improvement #12)
                ghosts.append((nx, ny, dist, angle))
            else:
                filled.append((nx, ny, dist, angle))

        # Sort both lists ascending by distance from the house centre.
        filled.sort(key=lambda s: s[2])
        ghosts.sort(key=lambda s: s[2])

        # Deduplicate active stones: if two detections are within
        # STONE_DEDUP_RADIUS of each other, keep only the closer one.
        # Because ``filled`` is already sorted ascending by distance, the
        # first entry in each near-pair is always the closer stone
        # (improvement #13).
        deduped = []
        for stone in filled:
            if not any(
                math.sqrt((stone[0] - s[0]) ** 2 + (stone[1] - s[1]) ** 2)
                < STONE_DEDUP_RADIUS
                for s in deduped
            ):
                deduped.append(stone)
        filled = deduped

        return filled, ghosts

    red_filled, red_ghosts = _extract(mask_red)
    del mask_red
    yellow_filled, yellow_ghosts = _extract(mask_yellow)
    del mask_yellow
    return red_filled, red_ghosts, yellow_filled, yellow_ghosts, orientation


def _match_stones_to_state(prev_state, curr_stones, stone_id_counter):
    """Match newly detected stones to the previous end state.

    Uses a greedy nearest-neighbour assignment (sorted by ascending pair
    distance) to propagate stable stone IDs across consecutive shot diagrams
    within a single end.  Each physical stone keeps the same ID from the shot
    it first appears until it leaves play.

    Parameters
    ----------
    prev_state : list of (stone_id, px, py)
        Tracked stones from the previous shot in this end.  Empty list for
        the first shot (no prior state).
    curr_stones : list of (nx, ny, dist, angle)
        Raw detections from the current shot, sorted ascending by distance
        from the house centre (the output of ``_detect_stones_in_crop``).
    stone_id_counter : list of [int]
        Single-element mutable list holding the next available stone ID
        within the current end.  Modified in-place when new IDs are assigned.

    Returns
    -------
    matched : list of (stone_id, nx, ny, dist, angle, prev_x, prev_y)
        Each detected stone annotated with its stable ID and, when matched to
        a previous stone, the coordinates it held at the previous shot.
        ``prev_x`` / ``prev_y`` are ``None`` for stones newly placed this
        shot (no match found within ``STONE_TRACK_MAX_DIST``).  The list is
        sorted ascending by distance (preserving the distance-sorted column
        layout).
    new_state : list of (stone_id, px, py)
        Updated tracking state to carry forward to the next shot.
    """
    # Build all candidate pairs (prev_index, curr_index, distance) that fall
    # within the matching threshold, then sort by ascending distance so that
    # the greedy pass always resolves the most-confident assignments first.
    pairs = []
    for pi, (sid, px, py) in enumerate(prev_state):
        for ci, (nx, ny, _dist, _angle) in enumerate(curr_stones):
            d = math.sqrt((nx - px) ** 2 + (ny - py) ** 2)
            if d < STONE_TRACK_MAX_DIST:
                pairs.append((d, pi, ci))
    pairs.sort()

    assignments = {}  # curr_index -> (stone_id, prev_x, prev_y)
    used_prev = set()
    used_curr = set()
    for _d, pi, ci in pairs:
        if pi in used_prev or ci in used_curr:
            continue
        sid, px, py = prev_state[pi]
        assignments[ci] = (sid, px, py)
        used_prev.add(pi)
        used_curr.add(ci)

    matched = []
    new_state = []
    for ci, (nx, ny, dist, angle) in enumerate(curr_stones):
        if ci in assignments:
            sid, prev_x, prev_y = assignments[ci]
        else:
            # Stone newly placed this shot — assign a fresh ID.
            sid = stone_id_counter[0]
            stone_id_counter[0] += 1
            prev_x, prev_y = None, None
        matched.append((sid, nx, ny, dist, angle, prev_x, prev_y))
        new_state.append((sid, nx, ny))

    # Keep distance-sorted order (dist is index 3 in the matched tuple).
    matched.sort(key=lambda s: s[3])
    return matched, new_state


def render_page_image(page):
    """Render a PDF page to a numpy BGR array at RENDER_DPI."""
    pil_img = page.to_image(resolution=RENDER_DPI).original
    arr = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def crop_shot_image(page_bgr, shot_img_meta):
    """Crop a single shot diagram from the rendered page image."""
    x0 = int(shot_img_meta["x0"] * SCALE)
    y0 = int(shot_img_meta["top"] * SCALE)
    x1 = int(shot_img_meta["x1"] * SCALE)
    y1 = int(shot_img_meta["bottom"] * SCALE)
    return page_bgr[y0:y1, x0:x1]


# ---------------------------------------------------------------------------
# Match grouping
# ---------------------------------------------------------------------------


def group_pages_into_matches(pdf, shot_page_indices):
    """Group consecutive shot pages into matches.

    A new match starts when the date+round header changes, or when the
    team matchup changes, or when the end number resets to 1 (after > 1).

    Returns list of lists of page indices, one sub-list per match.
    """
    matches = []
    current_match = []
    prev_key = None

    for pi in shot_page_indices:
        page = pdf.pages[pi]
        text = page.extract_text() or ""
        date_str, round_name, _ = parse_match_header(text)
        end_info = parse_end_line(text)
        end_num = end_info["end_number"] if end_info else 1
        teams = (end_info["team1_code"], end_info["team2_code"]) if end_info else ("", "")
        key = (date_str, round_name, teams)

        if prev_key is not None and (key != prev_key or (end_num == 1 and current_match)):
            matches.append(current_match)
            current_match = []

        current_match.append(pi)
        prev_key = key

    if current_match:
        matches.append(current_match)

    return matches


# ---------------------------------------------------------------------------
# Main extraction pipeline
# ---------------------------------------------------------------------------


def extract_event(pdf_path, event_id, _on_match_data=None):
    """Extract data from a single PDF and return raw data structures.

    *pdf_path* can be a local file path **or** an HTTP(S) URL.

    Returns (matches_rows, teams_dict, players_dict, ends_rows, shots_rows)
    with *event_id* embedded in every row.

    Parameters
    ----------
    _on_match_data : callable or None
        If supplied, called as
        ``_on_match_data(match_row, ends, shots, players_dict)`` after each
        match is fully processed.  When set, shots and ends are **not**
        accumulated in the returned lists, reducing peak memory for large
        events.  ``players_dict`` is the cumulative player registry for the
        event (all players seen in matches processed so far).
    """
    pdf = _open_pdf(pdf_path)
    shot_page_indices = find_shot_pages(pdf)
    match_groups = group_pages_into_matches(pdf, shot_page_indices)

    # Accumulators
    matches_rows = []
    teams_dict = {}  # code -> {name, players set}
    players_dict = {}  # (event_id, code, name) -> id
    ends_rows = []
    shots_rows = []

    player_id_counter = 0
    match_id_counter = 0

    def get_or_create_player(team_code, player_name):
        nonlocal player_id_counter
        key = (event_id, team_code, player_name)
        if key not in players_dict:
            player_id_counter += 1
            players_dict[key] = player_id_counter
        return players_dict[key]

    def get_or_create_team(code, name):
        if code not in teams_dict:
            teams_dict[code] = {"name": name, "players": set()}
        elif name and not teams_dict[code]["name"]:
            teams_dict[code]["name"] = name
        return code

    total_matches = len(match_groups)
    for mi, match_pages in enumerate(match_groups):
        match_id_counter += 1
        match_id = match_id_counter
        print(f"  Processing match {match_id}/{total_matches} ({len(match_pages)} ends) …")

        # Parse first page for match-level info
        first_page = pdf.pages[match_pages[0]]
        first_text = first_page.extract_text() or ""
        date_str, round_name, start_time = parse_match_header(first_text)
        first_end = parse_end_line(first_text)
        if not first_end:
            continue

        team1_code = get_or_create_team(first_end["team1_code"], first_end["team1_name"])
        team2_code = get_or_create_team(first_end["team2_code"], first_end["team2_name"])

        # Determine final score from the last end page
        last_page = pdf.pages[match_pages[-1]]
        last_text = last_page.extract_text() or ""
        last_end = parse_end_line(last_text)
        total_scores, time_left = parse_score_box(last_text)

        final_score_1 = total_scores.get("team1", "")
        final_score_2 = total_scores.get("team2", "")

        # Fallback: if the score box didn't parse (e.g. CWC PDF format),
        # derive final score from the last end's cumulative score_after.
        if (not final_score_1 or not final_score_2) and last_end:
            if not final_score_1:
                final_score_1 = last_end["team1_score_after"]
            if not final_score_2:
                final_score_2 = last_end["team2_score_after"]

        match_row = {
            "event_id": event_id,
            "match_id": match_id,
            "date": date_str,
            "round": round_name,
            "start_time": start_time,
            "team1_code": team1_code,
            "team2_code": team2_code,
            "team1_final_score": final_score_1,
            "team2_final_score": final_score_2,
        }
        # Per-match accumulators — freed after each match when _on_match_data
        # is set, so only one match's worth of data is live at a time.
        match_ends = []
        match_shots = []

        # Process each end (page)
        for page_idx in match_pages:
            page = pdf.pages[page_idx]
            page_text = page.extract_text() or ""
            end_info = parse_end_line(page_text)
            if not end_info:
                continue

            end_number = end_info["end_number"]
            _, end_time_left = parse_score_box(page_text)

            # Determine hammer: odd-numbered shots belong to team that
            # throws first; the team throwing last (shot 16) has the hammer.
            # We will determine this from the shot metadata below.

            shot_images = _get_shot_images(page)
            if len(shot_images) != 16:
                # Partial/conceded end: record the end score summary but skip
                # shot-level processing (we can't reliably assign 16 stone
                # positions without exactly 16 diagrams).  hammer_team_code is
                # left blank because shot 16 wasn't thrown.
                match_ends.append({
                    "event_id": event_id,
                    "match_id": match_id,
                    "end_number": end_number,
                    "team1_code": end_info["team1_code"],
                    "team2_code": end_info["team2_code"],
                    "team1_score_before": end_info["team1_score_before"],
                    "team2_score_before": end_info["team2_score_before"],
                    "team1_score_this_end": end_info["team1_score_this_end"],
                    "team2_score_this_end": end_info["team2_score_this_end"],
                    "team1_score_after": end_info["team1_score_after"],
                    "team2_score_after": end_info["team2_score_after"],
                    "hammer_team_code": "",
                    "team1_time_left": end_time_left.get("team1", ""),
                    "team2_time_left": end_time_left.get("team2", ""),
                })
                continue

            # Extract shot metadata from text
            words = page.extract_words()
            shot_metas = _extract_shot_metadata_from_words(words, shot_images)

            # Determine hammer from shot 16 (last stone)
            hammer_team = shot_metas[15]["team_code"] if shot_metas[15]["team_code"] else ""

            match_ends.append({
                "event_id": event_id,
                "match_id": match_id,
                "end_number": end_number,
                "team1_code": end_info["team1_code"],
                "team2_code": end_info["team2_code"],
                "team1_score_before": end_info["team1_score_before"],
                "team2_score_before": end_info["team2_score_before"],
                "team1_score_this_end": end_info["team1_score_this_end"],
                "team2_score_this_end": end_info["team2_score_this_end"],
                "team1_score_after": end_info["team1_score_after"],
                "team2_score_after": end_info["team2_score_after"],
                "hammer_team_code": hammer_team,
                "team1_time_left": end_time_left.get("team1", ""),
                "team2_time_left": end_time_left.get("team2", ""),
            })

            # Render page and detect stones for each shot
            page_bgr = render_page_image(page)

            # Calibrate per-page stone colour ranges from the team indicator
            # images (improvement #3); falls back to static constants silently.
            color_ranges = _calibrate_stone_colors(page_bgr, page)

            # Collect per-crop orientations for the page-level consistency
            # check (improvement #9).
            page_orientations = []

            for shot_idx in range(16):
                shot_num = shot_idx + 1
                meta = shot_metas[shot_idx]
                crop = crop_shot_image(page_bgr, shot_images[shot_idx])

                red_stones, red_ghosts, yellow_stones, yellow_ghosts, house_orientation = _detect_stones_in_crop(crop, color_ranges)
                page_orientations.append(house_orientation)

                # Register player
                player_name = meta["player_name"]
                team_code = meta["team_code"]
                player_id = ""
                if team_code and player_name:
                    player_id = get_or_create_player(team_code, player_name)
                    teams_dict.setdefault(team_code, {"name": "", "players": set()})
                    teams_dict[team_code]["players"].add(player_name)

                row = {
                    "event_id": event_id,
                    "match_id": match_id,
                    "end_number": end_number,
                    "shot_number": shot_num,
                    "team_code": team_code,
                    "player_id": player_id,
                    "player_name": player_name,
                    "shot_type": meta["shot_type"],
                    "turn": meta["turn"],
                    "accuracy": meta["accuracy"],
                    "house_orientation": house_orientation,
                    "team1_stones_in_play": len(red_stones),
                    "team2_stones_in_play": len(yellow_stones),
                }

                # Stone positions – up to 8 per team, sorted by distance.
                for ti, raw in enumerate([red_stones, yellow_stones], start=1):
                    prefix = f"team{ti}"
                    for si in range(MAX_STONES_PER_TEAM):
                        if si < len(raw):
                            nx, ny, dist, angle = raw[si]
                            row[f"{prefix}_stone{si+1}_x"] = round(nx, 3)
                            row[f"{prefix}_stone{si+1}_y"] = round(ny, 3)
                            row[f"{prefix}_stone{si+1}_dist"] = round(dist, 3)
                            row[f"{prefix}_stone{si+1}_angle"] = round(angle, 1)
                        else:
                            row[f"{prefix}_stone{si+1}_x"] = ""
                            row[f"{prefix}_stone{si+1}_y"] = ""
                            row[f"{prefix}_stone{si+1}_dist"] = ""
                            row[f"{prefix}_stone{si+1}_angle"] = ""

                # Ghost stone positions – outline rings marking pre-shot
                # positions of displaced stones (improvement #12).
                for ti, ghost_list in enumerate([red_ghosts, yellow_ghosts], start=1):
                    prefix = f"team{ti}"
                    row[f"{prefix}_ghosts_in_play"] = len(ghost_list)
                    for gi in range(MAX_STONES_PER_TEAM):
                        if gi < len(ghost_list):
                            nx, ny, dist, angle = ghost_list[gi]
                            row[f"{prefix}_ghost{gi+1}_x"] = round(nx, 3)
                            row[f"{prefix}_ghost{gi+1}_y"] = round(ny, 3)
                            row[f"{prefix}_ghost{gi+1}_dist"] = round(dist, 3)
                            row[f"{prefix}_ghost{gi+1}_angle"] = round(angle, 1)
                        else:
                            row[f"{prefix}_ghost{gi+1}_x"] = ""
                            row[f"{prefix}_ghost{gi+1}_y"] = ""
                            row[f"{prefix}_ghost{gi+1}_dist"] = ""
                            row[f"{prefix}_ghost{gi+1}_angle"] = ""

                match_shots.append(row)

            del page_bgr  # Free the full-page rendered image (~78 MB) before continuing.

            # Page-level orientation consistency check (improvement #9).
            # All 16 crops on the same end page should agree on orientation;
            # a disagreement most likely indicates a detection fallback firing
            # with the wrong orientation for some crops, which would silently
            # invert stone y-coordinates for those shots.
            if len(set(page_orientations)) > 1:
                orient_counts = collections.Counter(page_orientations)
                majority, _ = orient_counts.most_common(1)[0]
                disagreeing = sum(1 for o in page_orientations if o != majority)
                print(
                    f"WARNING: inconsistent house orientation on event "
                    f"{event_id}, match {match_id}, end {end_number}: "
                    f"{disagreeing}/16 crop(s) disagree with majority "
                    f"'{majority}'. Stone y-coordinates may be inverted "
                    "for affected shots."
                )

        # Either stream data to caller (memory-efficient) or accumulate for
        # the return value (backward-compatible with tests and direct callers).
        if _on_match_data is not None:
            _on_match_data(match_row, match_ends, match_shots, players_dict)
            del match_shots, match_ends  # Release shot/end memory immediately.
        else:
            matches_rows.append(match_row)
            ends_rows.extend(match_ends)
            shots_rows.extend(match_shots)

    _close_pdf(pdf)

    return matches_rows, teams_dict, players_dict, ends_rows, shots_rows


def run_calibration_diagnostic(pdf_paths, verbose=True):
    """Run a per-event calibration diagnostic pass (improvement #10).

    For each PDF, renders the first shot diagram of the first shot page,
    detects the house centre and 12-foot ring radius via
    :func:`_detect_house_center`, and checks whether the detected radius
    deviates from :data:`HOUSE_RADIUS` by more than 10 %.  Also examines
    shot 1 stone detections for stones far outside the house, which are a
    signal of miscalibrated HSV thresholds or a failed house-ring detection.

    Intended to be run *separately* from the main extraction (not called
    by :func:`extract_all`) as a pre-flight check before processing a large
    batch of PDFs.

    Parameters
    ----------
    pdf_paths : list[str]
        Local file paths or HTTP(S) URLs to tournament result PDFs.
    verbose : bool
        When ``True`` (default) print a per-event summary to stdout.

    Returns
    -------
    list[dict]
        One dict per event with keys ``event``, ``pdf``, ``status``,
        ``detected_radius``, ``deviation_pct``, ``house_cx``, ``house_cy``,
        ``orientation``, ``shot1_red``, ``shot1_yellow``, ``shot1_suspect``.
        ``status`` is ``'ok'``, ``'radius_deviation'``, ``'suspect_stones'``,
        ``'no_shot_pages'``, ``'no_shot_images'``, or ``'error'``.
    """
    RADIUS_DEVIATION_THRESHOLD = 0.10   # flag events deviating > 10 %
    SUSPECT_DIST_THRESHOLD = 1.5        # normalised units (12-ft ring = 1.0)

    results = []
    n = len(pdf_paths)

    for event_idx, pdf_path in enumerate(pdf_paths, start=1):
        event_name = os.path.splitext(os.path.basename(pdf_path))[0]
        if verbose:
            print(f"\n[{event_idx}/{n}] {event_name}")
            print(f"  PDF: {pdf_path}")

        try:
            pdf = _open_pdf(pdf_path)
        except RuntimeError as exc:
            if verbose:
                print(f"  ERROR: {exc}")
            results.append({"event": event_name, "pdf": pdf_path,
                             "status": "error", "error": str(exc)})
            continue

        shot_pages = find_shot_pages(pdf)
        if not shot_pages:
            if verbose:
                print("  No shot pages found.")
            _close_pdf(pdf)
            results.append({"event": event_name, "pdf": pdf_path,
                             "status": "no_shot_pages"})
            continue

        first_page = pdf.pages[shot_pages[0]]
        shot_images = _get_shot_images(first_page)
        if not shot_images:
            if verbose:
                print("  No shot images found on first shot page.")
            _close_pdf(pdf)
            results.append({"event": event_name, "pdf": pdf_path,
                             "status": "no_shot_images"})
            continue

        page_bgr = render_page_image(first_page)
        first_crop = crop_shot_image(page_bgr, shot_images[0])

        # Detect house centre and ring radius.
        cx, cy, detected_radius, orientation = _detect_house_center(first_crop)
        deviation_pct = abs(detected_radius - HOUSE_RADIUS) / HOUSE_RADIUS
        radius_ok = deviation_pct <= RADIUS_DEVIATION_THRESHOLD

        # Check shot 1 for stones unexpectedly far outside the house.
        color_ranges = _calibrate_stone_colors(page_bgr, first_page)
        del page_bgr  # No longer needed after calibration; free the ~78 MB image.
        red_stones, _red_ghosts, yellow_stones, _yellow_ghosts, _ = _detect_stones_in_crop(first_crop, color_ranges)
        all_stones = red_stones + yellow_stones
        suspect_stones = [s for s in all_stones if s[2] > SUSPECT_DIST_THRESHOLD]

        if not radius_ok:
            status = "radius_deviation"
        elif suspect_stones:
            status = "suspect_stones"
        else:
            status = "ok"

        if verbose:
            print(
                f"  Detected radius : {detected_radius:.1f} px  "
                f"(nominal {HOUSE_RADIUS} px, deviation {deviation_pct * 100:.1f} %)"
            )
            print(f"  House centre    : ({cx:.1f}, {cy:.1f})  orientation: {orientation}")
            print(
                f"  Shot 1 stones   : {len(red_stones)} red, "
                f"{len(yellow_stones)} yellow  "
                f"(suspect far-outside: {len(suspect_stones)})"
            )
            if not radius_ok:
                print(
                    f"  WARNING: radius deviation > {RADIUS_DEVIATION_THRESHOLD * 100:.0f} % — "
                    "consider adding a per-event override config:"
                )
                print(
                    f'    {{"house_radius": {detected_radius:.0f}, '
                    f'"house_cx": {cx:.0f}, "house_cy": {cy:.0f}}}'
                )
            if suspect_stones:
                print(
                    f"  WARNING: {len(suspect_stones)} stone(s) detected "
                    "far outside the house — check HSV thresholds or house "
                    "ring detection for this event."
                )

        results.append({
            "event": event_name,
            "pdf": pdf_path,
            "status": status,
            "detected_radius": round(detected_radius, 1),
            "deviation_pct": round(deviation_pct * 100, 1),
            "house_cx": round(cx, 1),
            "house_cy": round(cy, 1),
            "orientation": orientation,
            "shot1_red": len(red_stones),
            "shot1_yellow": len(yellow_stones),
            "shot1_suspect": len(suspect_stones),
        })
        _close_pdf(pdf)

    if verbose:
        ok_count = sum(1 for r in results if r.get("status") == "ok")
        warn_count = len(results) - ok_count
        print(f"\n{'=' * 60}")
        print(f"Calibration summary: {len(results)} event(s), "
              f"{ok_count} OK, {warn_count} warning(s)")
        print(f"{'=' * 60}")
        for r in results:
            if r.get("status") != "ok":
                print(f"  [{r['status'].upper()}] {r['event']}")

    return results


# ---------------------------------------------------------------------------
# Subprocess worker + CSV fragment helpers
# ---------------------------------------------------------------------------

def _append_csv_fragment(src_path, dst_file):
    """Append the rows of *src_path* to the open file *dst_file*, skipping
    its header line so the main file header is not duplicated."""
    with open(src_path, newline="", encoding="utf-8") as src:
        next(src, None)  # skip header
        shutil.copyfileobj(src, dst_file)


def _event_subprocess_target(
    event_id, pdf_path, meta, tmp_dir, player_id_offset,
    shots_all_fields, shots_numeric_cols, child_conn,
):
    """Process one event in an isolated subprocess.

    Writes per-event CSV/parquet fragments to *tmp_dir* and sends a compact
    result dict back to the parent via *child_conn*, then exits.  When the
    subprocess exits the OS reclaims all memory it consumed — Python heap,
    glibc pages held by numpy/OpenCV, pdfplumber/PIL caches — none of which
    would be returned to the OS if the work were done in the parent process.

    Parameters
    ----------
    player_id_offset : int
        Number of players already assigned globally by previous events.
        New players in this event receive IDs starting from
        ``player_id_offset + 1``.
    child_conn : multiprocessing.Connection
        Write-end of a ``Pipe``; the function sends one dict then closes it.
    """
    event_name = meta.get(
        "tournament_name",
        os.path.splitext(os.path.basename(pdf_path))[0],
    )

    matches_fields = [
        "event_id", "match_id", "date", "round", "start_time",
        "team1_code", "team2_code",
        "team1_final_score", "team2_final_score",
    ]
    ends_fields = [
        "event_id", "match_id", "end_number",
        "team1_code", "team2_code",
        "team1_score_before", "team2_score_before",
        "team1_score_this_end", "team2_score_this_end",
        "team1_score_after", "team2_score_after",
        "hammer_team_code",
        "team1_time_left", "team2_time_left",
    ]

    tmp_matches = os.path.join(tmp_dir, f"event_{event_id}_matches.parquet")
    tmp_ends = os.path.join(tmp_dir, f"event_{event_id}_ends.parquet")
    tmp_shots_parquet = os.path.join(tmp_dir, f"event_{event_id}_shots.parquet")

    shots_pq_writer = None
    matches_rows = []
    ends_rows = []
    n_matches = n_ends = n_shots = 0
    players_dict_global = {}          # (event_id, tc, pn) -> global_id
    player_counter = [player_id_offset]  # mutable int shared with closure
    has_time = False
    error = None
    teams_dict_result = {}

    try:
        def _on_match(match_row, ends, shots, event_players):
            nonlocal shots_pq_writer, n_matches, n_ends, n_shots, has_time

            # Assign global IDs to players first seen in this match.
            for key in event_players:
                if key not in players_dict_global:
                    player_counter[0] += 1
                    players_dict_global[key] = player_counter[0]

            # Remap player IDs in shots to global IDs.
            for row in shots:
                tc = row["team_code"]
                pn = row["player_name"]
                if tc and pn:
                    gid = players_dict_global.get((event_id, tc, pn))
                    if gid is not None:
                        row["player_id"] = gid

            if any(r.get("team1_time_left") or r.get("team2_time_left")
                   for r in ends):
                has_time = True

            matches_rows.append(match_row)
            ends_rows.extend(ends)
            n_matches += 1
            n_ends += len(ends)
            n_shots += len(shots)

            # Stream this match's shots to the temp parquet as a small batch.
            if shots:
                df_b = pd.DataFrame(shots, columns=shots_all_fields)
                df_b[shots_numeric_cols] = df_b[shots_numeric_cols].replace("", None)
                for col in shots_numeric_cols:
                    df_b[col] = pd.to_numeric(df_b[col], errors="coerce")
                for col in ("team1_ghosts_in_play", "team2_ghosts_in_play"):
                    df_b[col] = (
                        pd.to_numeric(df_b[col], errors="coerce").astype("Int64")
                    )
                tbl = pa.Table.from_pandas(df_b, preserve_index=False)
                del df_b
                if shots_pq_writer is None:
                    shots_pq_writer = pq.ParquetWriter(tmp_shots_parquet, tbl.schema)
                shots_pq_writer.write_table(tbl)
                del tbl

        _, teams_dict_result, _, _, _ = extract_event(
            pdf_path, event_id, _on_match_data=_on_match
        )

        # Write accumulated matches and ends to parquet (small per event).
        if matches_rows:
            df_m = pd.DataFrame(matches_rows, columns=matches_fields)
            for col in ("team1_final_score", "team2_final_score"):
                df_m[col] = pd.to_numeric(df_m[col], errors="coerce").astype("Int64")
            df_m.to_parquet(tmp_matches, index=False)
        if ends_rows:
            df_e = pd.DataFrame(ends_rows, columns=ends_fields)
            _ends_int_cols = [
                "team1_score_before", "team2_score_before",
                "team1_score_this_end", "team2_score_this_end",
                "team1_score_after", "team2_score_after",
            ]
            for col in _ends_int_cols:
                df_e[col] = pd.to_numeric(df_e[col], errors="coerce").astype("Int64")
            df_e.to_parquet(tmp_ends, index=False)

    except Exception as exc:  # noqa: BLE001
        error = str(exc)

    finally:
        if shots_pq_writer is not None:
            shots_pq_writer.close()

    event_row = {
        "event_id": event_id,
        "event_name": event_name,
        "pdf_file": os.path.basename(pdf_path),
    }
    for col in ("year", "location", "gender"):
        if meta.get(col):
            event_row[col] = meta[col]

    gc.collect()  # encourage Python to release any lingering references

    child_conn.send({
        "error": error,
        "event_row": event_row,
        "teams_dict": teams_dict_result,
        "players_dict": players_dict_global,
        "has_time": has_time,
        "n_matches": n_matches,
        "n_ends": n_ends,
        "n_shots": n_shots,
        "tmp_matches": tmp_matches,
        "tmp_ends": tmp_ends,
        "tmp_shots_parquet": tmp_shots_parquet,
    })
    child_conn.close()


def _read_events_parquet(path):
    """Read an existing events.parquet back into (events_rows, events_with_time_set)."""
    events_rows = []
    events_with_time = set()
    df = pd.read_parquet(path)
    for _, row in df.iterrows():
        r = row.where(pd.notna(row), other=None).to_dict()
        if r.get("event_id") is not None:
            try:
                r["event_id"] = int(r["event_id"])
            except (ValueError, TypeError):
                pass
        if r.get("year") is not None:
            try:
                r["year"] = int(r["year"])
            except (ValueError, TypeError):
                pass
        if r.get("has_time_data"):
            events_with_time.add(r["event_id"])
        events_rows.append(r)
    return events_rows, events_with_time


def _read_players_parquet(path):
    """Read an existing players.parquet back into {(event_id, team_code, player_name): player_id}."""
    players_dict = {}
    df = pd.read_parquet(path)
    for _, row in df.iterrows():
        try:
            eid = int(row["event_id"])
            pid = int(row["player_id"])
        except (KeyError, ValueError, TypeError):
            continue
        key = (eid, str(row.get("team_code") or ""), str(row.get("player_name") or ""))
        players_dict[key] = pid
    return players_dict


def _read_teams_parquet(path):
    """Read an existing teams.parquet back into {(event_id, team_code): {name, players}}."""
    teams_dict = {}
    df = pd.read_parquet(path)
    player_cols = [c for c in df.columns if c.startswith("player") and c.endswith("_name")]
    for _, row in df.iterrows():
        try:
            eid = int(row["event_id"])
        except (KeyError, ValueError, TypeError):
            continue
        code = str(row.get("team_code") or "")
        players = {str(row[c]) for c in player_cols if pd.notna(row.get(c)) and row.get(c)}
        teams_dict[(eid, code)] = {
            "name": str(row.get("team_name") or ""),
            "players": players,
        }
    return teams_dict


def extract_all(pdf_paths, output_dir="output", event_metadata=None,
                start_index=0, batch_size=None, checkpoint_file=None):
    """Run the full extraction pipeline for one or more PDFs and write CSV tables.

    Parameters
    ----------
    pdf_paths : str or list[str]
        Path to a single PDF, an HTTP(S) URL, or a list of paths/URLs.
    output_dir : str
        Directory for output CSV files.
    event_metadata : list[dict] or None
        Optional per-event metadata dicts (one per pdf_path) with keys such as
        ``tournament_name``, ``year``, ``location``, ``gender``.  When
        provided, the extra columns are included in ``events.csv``.
    start_index : int
        0-based index of the first event to process.  Use with ``batch_size``
        to process the full list in multiple runs.
    batch_size : int or None
        Maximum number of events to process in this run.  ``None`` means
        process all events from ``start_index`` to the end of the list.
    checkpoint_file : str or None
        Path to the JSON checkpoint file.  Updated after every event so a
        killed run can be resumed.  Defaults to
        ``{output_dir}/.checkpoint.json``.
    """
    if isinstance(pdf_paths, str):
        pdf_paths = [pdf_paths]

    if event_metadata is None:
        event_metadata = [{}] * len(pdf_paths)

    # ---- Determine the slice to process this run ---------------------------
    total_events = len(pdf_paths)
    end_index = total_events if batch_size is None else min(start_index + batch_size, total_events)
    batch_paths = pdf_paths[start_index:end_index]
    batch_meta = event_metadata[start_index:end_index]

    if not batch_paths:
        print(f"No events to process (start_index={start_index} >= {total_events}).")
        return

    append_mode = start_index > 0  # True when we are resuming / continuing a previous batch

    os.makedirs(output_dir, exist_ok=True)

    if checkpoint_file is None:
        checkpoint_file = os.path.join(output_dir, ".checkpoint.json")

    _, stone_fields, ghost_fields, shots_all_fields = _shot_fields()
    shots_numeric_cols = stone_fields + [
        c for c in ghost_fields if "ghosts_in_play" not in c
    ]

    matches_path = os.path.join(output_dir, "matches.parquet")
    ends_path = os.path.join(output_dir, "ends.parquet")
    shots_path = os.path.join(output_dir, "shot_locations_raw.parquet")

    matches_fields = [
        "event_id", "match_id", "date", "round", "start_time",
        "team1_code", "team2_code",
        "team1_final_score", "team2_final_score",
    ]
    ends_fields = [
        "event_id", "match_id", "end_number",
        "team1_code", "team2_code",
        "team1_score_before", "team2_score_before",
        "team1_score_this_end", "team2_score_this_end",
        "team1_score_after", "team2_score_after",
        "hammer_team_code",
        "team1_time_left", "team2_time_left",
    ]

    # ---- Restore small-table state from previous batches (if resuming) -----
    events_rows = []
    all_teams_dict = {}    # (event_id, code) -> {name, players set}
    all_players_dict = {}  # (event_id, code, name) -> global_id
    global_player_id = 0
    events_with_time = set()
    total_matches_count = total_ends_count = total_shots_count = 0

    if append_mode:
        # Restore counters from checkpoint so the summary at the end is accurate.
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, encoding="utf-8") as _f:
                _ckpt = json.load(_f)
            global_player_id = _ckpt.get("global_player_id", 0)
            total_matches_count = _ckpt.get("total_matches", 0)
            total_ends_count = _ckpt.get("total_ends", 0)
            total_shots_count = _ckpt.get("total_shots", 0)

        # Restore small tables from the parquet files written by the previous batch.
        _ev_path = os.path.join(output_dir, "events.parquet")
        _pl_path = os.path.join(output_dir, "players.parquet")
        _tm_path = os.path.join(output_dir, "teams.parquet")
        if os.path.exists(_ev_path):
            events_rows, events_with_time = _read_events_parquet(_ev_path)
        if os.path.exists(_pl_path):
            all_players_dict = _read_players_parquet(_pl_path)
        if os.path.exists(_tm_path):
            all_teams_dict = _read_teams_parquet(_tm_path)

        print(f"Resuming from event index {start_index} "
              f"(events {start_index+1}–{end_index} of {total_events}). "
              f"Appending to existing output files.")
    else:
        print(f"Processing events 1–{end_index} of {total_events}.")

    # Temp directory for per-event CSV/parquet fragments.
    tmp_dir = tempfile.mkdtemp(prefix="curling_extract_")

    # Each event is processed in a fresh subprocess (spawn context = clean
    # interpreter).  When the subprocess exits the OS reclaims ALL memory it
    # consumed: Python heap high-watermark, glibc pages held by numpy/OpenCV,
    # pdfplumber/PIL internal caches.  The parent process therefore stays at a
    # low, stable memory footprint regardless of how many events are processed.
    mp_ctx = multiprocessing.get_context("spawn")

    # Parquet writers for the three large streaming tables (opened lazily on
    # first batch so the schema is inferred from real data).
    pq_writers = {"matches": None, "ends": None, "shots": None}
    pq_paths = {"matches": matches_path, "ends": ends_path, "shots": shots_path}

    # In append mode, parquet files cannot be extended in-place.  Rename each
    # existing file to a .prev sidecar; the parent will re-stream its contents
    # at the top of the new file before appending the current batch's data.
    prev_parquets = {}
    if append_mode:
        for key, path in pq_paths.items():
            if os.path.exists(path):
                prev = path + ".prev"
                os.rename(path, prev)
                prev_parquets[key] = prev

    try:
        # Re-stream all previous-batch data so each output file is always
        # a complete, self-contained dataset after every run.
        for key, prev_path in prev_parquets.items():
            if os.path.exists(prev_path):
                _pf = pq.ParquetFile(prev_path)
                for _b in _pf.iter_batches(batch_size=1000):
                    if pq_writers[key] is None:
                        pq_writers[key] = pq.ParquetWriter(pq_paths[key], _pf.schema_arrow)
                    pq_writers[key].write_batch(_b)
                del _pf

        for event_id, (pdf_path, meta) in enumerate(
            zip(batch_paths, batch_meta), start=start_index + 1
        ):
            event_name = meta.get(
                "tournament_name",
                os.path.splitext(os.path.basename(pdf_path))[0],
            )
            print(f"\n{'='*60}")
            print(f"Event {event_id}/{total_events}: {event_name}")
            print(f"  PDF: {pdf_path}")
            print(f"{'='*60}")

            # Spawn a subprocess that processes this event and writes its data
            # to per-event temp files, then sends back compact metadata.
            parent_conn, child_conn = mp_ctx.Pipe(duplex=False)
            p = mp_ctx.Process(
                target=_event_subprocess_target,
                args=(
                    event_id, pdf_path, meta, tmp_dir, global_player_id,
                    shots_all_fields, shots_numeric_cols, child_conn,
                ),
                daemon=False,
            )
            p.start()
            child_conn.close()  # parent never writes to the child end

            try:
                result = parent_conn.recv()
            except EOFError:
                result = {
                    "error": "subprocess exited without result (possibly OOM-killed)"
                }
            finally:
                parent_conn.close()

            p.join()

            if result.get("error"):
                print(f"  ERROR: {result['error']}")
                events_rows.append({
                    "event_id": event_id,
                    "event_name": event_name,
                    "pdf_file": os.path.basename(pdf_path),
                })
            else:
                # Update global registries from subprocess result (all small).
                all_players_dict.update(result["players_dict"])
                if result["players_dict"]:
                    global_player_id = max(result["players_dict"].values())

                if result["has_time"]:
                    events_with_time.add(event_id)

                events_rows.append(result["event_row"])

                for code, info in result["teams_dict"].items():
                    key = (event_id, code)
                    if key not in all_teams_dict:
                        all_teams_dict[key] = {
                            "name": info["name"],
                            "players": set(info["players"]),
                        }
                    else:
                        all_teams_dict[key]["players"].update(info["players"])
                        if info["name"] and not all_teams_dict[key]["name"]:
                            all_teams_dict[key]["name"] = info["name"]

                total_matches_count += result["n_matches"]
                total_ends_count += result["n_ends"]
                total_shots_count += result["n_shots"]

                # Stream each subprocess temp parquet fragment into the
                # corresponding main parquet writer.
                for key, tmp_path in [
                    ("matches", result["tmp_matches"]),
                    ("ends", result["tmp_ends"]),
                    ("shots", result["tmp_shots_parquet"]),
                ]:
                    if os.path.exists(tmp_path):
                        _pf = pq.ParquetFile(tmp_path)
                        for _b in _pf.iter_batches(batch_size=1000):
                            if pq_writers[key] is None:
                                pq_writers[key] = pq.ParquetWriter(
                                    pq_paths[key], _pf.schema_arrow
                                )
                            pq_writers[key].write_batch(_b)
                        del _pf
                        try:
                            os.unlink(tmp_path)
                        except OSError:
                            pass

            # ---- Checkpoint: record progress after every event -------------
            _ckpt_data = {
                "next_index": event_id,
                "global_player_id": global_player_id,
                "total_matches": total_matches_count,
                "total_ends": total_ends_count,
                "total_shots": total_shots_count,
            }
            with open(checkpoint_file, "w", encoding="utf-8") as _f:
                json.dump(_ckpt_data, _f)

    finally:
        for writer in pq_writers.values():
            if writer is not None:
                writer.close()
        shutil.rmtree(tmp_dir, ignore_errors=True)
        for prev_path in prev_parquets.values():
            if os.path.exists(prev_path):
                os.unlink(prev_path)

    # ---- Write small tables at the end -------------------------------------
    for event_row in events_rows:
        event_row["has_time_data"] = event_row.get("event_id") in events_with_time

    _write_events_parquet(os.path.join(output_dir, "events.parquet"), events_rows)
    _write_teams_parquet(os.path.join(output_dir, "teams.parquet"), all_teams_dict)
    _write_players_parquet(os.path.join(output_dir, "players.parquet"), all_players_dict)

    batch_event_count = end_index - start_index
    print(f"\nBatch complete – processed events {start_index+1}–{end_index} of {total_events}. "
          f"Running totals: {len(events_rows)} events, {total_matches_count} matches, "
          f"{total_ends_count} ends, {total_shots_count} shots in {output_dir}/")
    if end_index < total_events:
        print(f"  {total_events - end_index} events remaining. "
              f"Run again with --resume to continue, or "
              f"--start-event {end_index} to specify manually.")


# ---------------------------------------------------------------------------
# Parquet writers for small lookup tables
# ---------------------------------------------------------------------------

def _write_events_parquet(path, rows):
    fields = ["event_id", "event_name", "year", "location", "gender", "pdf_file", "has_time_data"]
    df = pd.DataFrame(rows, columns=fields)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["event_id"] = pd.to_numeric(df["event_id"], errors="coerce").astype("Int64")
    df.to_parquet(path, index=False)


def _write_matches_parquet(path, rows):
    fields = [
        "event_id", "match_id", "date", "round", "start_time",
        "team1_code", "team2_code",
        "team1_final_score", "team2_final_score",
    ]
    df = pd.DataFrame(rows, columns=fields)
    for col in ("event_id", "match_id", "team1_final_score", "team2_final_score"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    df.to_parquet(path, index=False)


def _write_teams_parquet(path, teams_dict):
    rows = []
    for (event_id, code), info in sorted(teams_dict.items()):
        player_list = sorted(info["players"])
        row = {"event_id": event_id, "team_code": code, "team_name": info["name"]}
        for i, p in enumerate(player_list, start=1):
            row[f"player{i}_name"] = p
        rows.append(row)
    if not rows:
        pd.DataFrame(columns=["event_id", "team_code", "team_name"]).to_parquet(path, index=False)
        return
    max_p = max((len(r) - 3 for r in rows), default=0)
    fields = ["event_id", "team_code", "team_name"] + [f"player{i}_name" for i in range(1, max_p + 1)]
    df = pd.DataFrame(rows, columns=fields)
    df["event_id"] = pd.to_numeric(df["event_id"], errors="coerce").astype("Int64")
    df.to_parquet(path, index=False)


def _write_players_parquet(path, players_dict):
    rows = [
        {"player_id": pid, "event_id": eid, "team_code": tc, "player_name": pn}
        for (eid, tc, pn), pid in sorted(players_dict.items(), key=lambda x: x[1])
    ]
    df = pd.DataFrame(rows, columns=["player_id", "event_id", "team_code", "player_name"])
    for col in ("player_id", "event_id"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    df.to_parquet(path, index=False)


def _write_ends_parquet(path, rows):
    fields = [
        "event_id", "match_id", "end_number",
        "team1_code", "team2_code",
        "team1_score_before", "team2_score_before",
        "team1_score_this_end", "team2_score_this_end",
        "team1_score_after", "team2_score_after",
        "hammer_team_code",
        "team1_time_left", "team2_time_left",
    ]
    pd.DataFrame(rows, columns=fields).to_parquet(path, index=False)


def _shot_fields():
    """Return (base_fields, stone_fields, ghost_fields, all_fields) for shot CSV/parquet."""
    base = [
        "event_id", "match_id", "end_number", "shot_number",
        "team_code", "player_id", "player_name",
        "shot_type", "turn", "accuracy",
        "house_orientation",
        "team1_stones_in_play", "team2_stones_in_play",
    ]
    stone = []
    for ti in (1, 2):
        for si in range(1, MAX_STONES_PER_TEAM + 1):
            prefix = f"team{ti}_stone{si}"
            stone += [f"{prefix}_x", f"{prefix}_y", f"{prefix}_dist", f"{prefix}_angle"]
    # Ghost stone fields – outline-ring positions for displaced stones
    # (improvement #12).  team{N}_ghosts_in_play is always an integer count;
    # coordinate columns are NULL when no ghost was detected for that slot.
    ghost = []
    for ti in (1, 2):
        ghost.append(f"team{ti}_ghosts_in_play")
        for gi in range(1, MAX_STONES_PER_TEAM + 1):
            prefix = f"team{ti}_ghost{gi}"
            ghost += [f"{prefix}_x", f"{prefix}_y", f"{prefix}_dist", f"{prefix}_angle"]
    return base, stone, ghost, base + stone + ghost


def _write_shots_parquet(path, rows):
    """Write a list of shot dicts to a parquet file with correct numeric dtypes."""
    _, stone_fields, ghost_fields, all_fields = _shot_fields()
    numeric_cols = stone_fields + [c for c in ghost_fields if c not in
                                   ("team1_ghosts_in_play", "team2_ghosts_in_play")]
    df = pd.DataFrame(rows, columns=all_fields)
    df[numeric_cols] = df[numeric_cols].replace("", None)
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("team1_ghosts_in_play", "team2_ghosts_in_play"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    df.to_parquet(path, index=False)


def _write_csv(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Extract curling shot data from PDF")
    parser.add_argument("pdfs", nargs="*", default=None,
                        help="Path(s) or URL(s) to tournament results PDF(s). "
                             "If omitted, URLs are read from --results-csv.")
    parser.add_argument("--output-dir", default="output",
                        help="Directory for output CSV files (default: output)")
    parser.add_argument("--results-csv", default=DEFAULT_RESULTS_CSV,
                        help="Path to result_urls.csv produced by scrape_results.py "
                             "(default: output/result_urls.csv). Used when no "
                             "positional PDF arguments are provided.")
    parser.add_argument("--min-year", type=int, default=DEFAULT_MIN_YEAR,
                        help="Minimum event year to include when reading from "
                             "--results-csv (default: 2013).")
    parser.add_argument("--batch-size", type=int, default=None, metavar="N",
                        help="Process at most N events per run.  Run again with "
                             "--resume (or --start-event) to continue.")
    parser.add_argument("--resume", action="store_true",
                        help="Read the checkpoint file to find where the previous "
                             "run stopped and continue from there.")
    parser.add_argument("--start-event", type=int, default=None, metavar="INDEX",
                        help="0-based index of the first event to process.  "
                             "Overrides the checkpoint when combined with --resume.")
    parser.add_argument("--checkpoint", default=None, metavar="FILE",
                        help="Path to the JSON checkpoint file "
                             "(default: <output-dir>/.checkpoint.json).")
    parser.add_argument("--calibrate", action="store_true",
                        help="Run the per-event calibration diagnostic instead "
                             "of the full extraction (improvement #10). Prints "
                             "a per-event report and flags any event where the "
                             "detected house-ring radius deviates from the "
                             "nominal value by more than 10%%.")
    args = parser.parse_args()

    if args.pdfs:
        # Explicit PDFs given on the command line
        pdf_sources = args.pdfs
        metadata = None

        for src in pdf_sources:
            if not _is_url(src) and not os.path.isfile(src):
                parser.error(f"PDF not found: {src}")
    else:
        # Read from result_urls.csv
        if not os.path.isfile(args.results_csv):
            parser.error(
                f"Results CSV not found: {args.results_csv}. "
                "Run scrape_results.py first, or provide PDF paths directly."
            )
        rows = load_result_urls(args.results_csv, min_year=args.min_year)
        if not rows:
            parser.error(
                f"No result-book URLs with year >= {args.min_year} found "
                f"in {args.results_csv}."
            )
        pdf_sources = [r["result_book_url"] for r in rows]
        metadata = rows

    if args.calibrate:
        print(f"Running calibration diagnostic on {len(pdf_sources)} PDF(s) …")
        run_calibration_diagnostic(pdf_sources)
        return

    # ---- Determine start_index for this run --------------------------------
    checkpoint_file = args.checkpoint or os.path.join(args.output_dir, ".checkpoint.json")
    start_index = 0

    if args.resume and os.path.exists(checkpoint_file):
        with open(checkpoint_file, encoding="utf-8") as f:
            ckpt = json.load(f)
        start_index = ckpt.get("next_index", 0)
        print(f"Checkpoint found: resuming from event index {start_index} "
              f"({len(pdf_sources) - start_index} events remaining).")

    if args.start_event is not None:
        start_index = args.start_event  # explicit override

    if start_index >= len(pdf_sources):
        print(f"All {len(pdf_sources)} events already processed according to checkpoint.")
        return

    print(f"Extracting shot data from {len(pdf_sources)} PDF(s) …")
    extract_all(
        pdf_sources,
        args.output_dir,
        event_metadata=metadata,
        start_index=start_index,
        batch_size=args.batch_size,
        checkpoint_file=checkpoint_file,
    )


if __name__ == "__main__":
    main()
