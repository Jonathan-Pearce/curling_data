"""
Curling Shot Data Extractor

Parses curling tournament PDFs to extract match, end, and shot-level data
including stone positions detected via OpenCV color-based segmentation.

PDFs can be provided as local file paths or HTTP(S) URLs.  When no arguments
are supplied, the script downloads and processes the default result-book PDFs
from curlit.com.

Usage:
    python extract_shot_data.py [<pdf_or_url> ...] [--output-dir <dir>]

Outputs CSV files:
    - events.csv
    - matches.csv
    - teams.csv
    - players.csv
    - ends.csv
    - shot_locations.csv
"""

import argparse
import csv
import io
import math
import os
import re
import urllib.request

import cv2
import numpy as np
import pdfplumber

# ---------------------------------------------------------------------------
# Default PDF URLs (curlit.com result books)
# ---------------------------------------------------------------------------
DEFAULT_PDF_URLS = [
    "https://curlit.com/PDF/ECC2025_ResultsBook_Men_A-Division.pdf",
    "https://curlit.com/PDF/WMCC2023_ResultsBook.pdf",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RENDER_DPI = 300
PDF_POINTS_PER_INCH = 72
SCALE = RENDER_DPI / PDF_POINTS_PER_INCH

# House geometry at 300 DPI (empirically calibrated from the PDF)
HOUSE_CX = 161  # pixels – centre-x of the house in each shot crop
HOUSE_CY = 171  # pixels – centre-y
HOUSE_RADIUS = 89  # pixels – radius of the 12-foot ring

# Stone detection thresholds (HSV)
RED_LOWER_1 = np.array([0, 100, 80])
RED_UPPER_1 = np.array([10, 255, 255])
RED_LOWER_2 = np.array([160, 100, 80])
RED_UPPER_2 = np.array([180, 255, 255])
YELLOW_LOWER = np.array([10, 100, 100])
YELLOW_UPPER = np.array([35, 255, 255])

STONE_MIN_AREA = 80
STONE_MAX_AREA = 600

# Vertical pixel margins for stone detection (exclude score-indicator dots
# at the very top and shot-label text at the bottom of each crop).
STONE_Y_MIN_PX = 15
STONE_Y_MAX_FRAC = 0.82  # fraction of crop height

MAX_STONES_PER_TEAM = 8

# ---------------------------------------------------------------------------
# URL / PDF helpers
# ---------------------------------------------------------------------------


def _is_url(source):
    """Return True if *source* looks like an HTTP(S) URL."""
    return source.startswith(("http://", "https://"))


def _open_pdf(source):
    """Open a PDF from a local path or URL, returning a pdfplumber PDF object."""
    if _is_url(source):
        response = urllib.request.urlopen(source)
        data = response.read()
        return pdfplumber.open(io.BytesIO(data))
    return pdfplumber.open(source)


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


def _get_shot_images(page):
    """Return the 16 shot-diagram image metadata objects from *page*."""
    return [
        img
        for img in page.images
        if img["width"] > 50 and img["height"] > 100
    ]


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
                    shot["shot_type"] += " " + text
                else:
                    shot["shot_type"] = text

    return shots


# ---------------------------------------------------------------------------
# Stone detection (OpenCV)
# ---------------------------------------------------------------------------


def _detect_stones_in_crop(crop_bgr):
    """Detect red and yellow stones in a single shot crop image.

    Returns (red_stones, yellow_stones) where each is a list of
    (norm_x, norm_y, distance, angle_deg) tuples sorted ascending by
    distance from the house centre.  Coordinates are normalised so that
    the 12-foot ring has radius = 1.0.
    """
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    h, w = crop_bgr.shape[:2]

    # Restrict detection to the playing-field portion of the crop
    y_max = int(h * STONE_Y_MAX_FRAC)
    play_mask = np.zeros((h, w), dtype=np.uint8)
    play_mask[STONE_Y_MIN_PX:y_max, :] = 255

    mask_red = (
        cv2.inRange(hsv, RED_LOWER_1, RED_UPPER_1)
        | cv2.inRange(hsv, RED_LOWER_2, RED_UPPER_2)
    ) & play_mask

    mask_yellow = cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER) & play_mask

    def _extract(mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        stones = []
        for c in contours:
            area = cv2.contourArea(c)
            if STONE_MIN_AREA < area < STONE_MAX_AREA:
                M = cv2.moments(c)
                if M["m00"] > 0:
                    sx = M["m10"] / M["m00"]
                    sy = M["m01"] / M["m00"]
                    nx = (sx - HOUSE_CX) / HOUSE_RADIUS
                    ny = (sy - HOUSE_CY) / HOUSE_RADIUS
                    dist = math.sqrt(nx * nx + ny * ny)
                    angle = math.degrees(math.atan2(ny, nx))
                    stones.append((nx, ny, dist, angle))
        stones.sort(key=lambda s: s[2])
        return stones

    return _extract(mask_red), _extract(mask_yellow)


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


def extract_event(pdf_path, event_id):
    """Extract data from a single PDF and return raw data structures.

    *pdf_path* can be a local file path **or** an HTTP(S) URL.

    Returns (matches_rows, teams_dict, players_dict, ends_rows, shots_rows)
    with *event_id* embedded in every row.
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

        matches_rows.append({
            "event_id": event_id,
            "match_id": match_id,
            "date": date_str,
            "round": round_name,
            "start_time": start_time,
            "team1_code": team1_code,
            "team2_code": team2_code,
            "team1_final_score": final_score_1,
            "team2_final_score": final_score_2,
        })

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
                continue

            # Extract shot metadata from text
            words = page.extract_words()
            shot_metas = _extract_shot_metadata_from_words(words, shot_images)

            # Determine hammer from shot 16 (last stone)
            hammer_team = shot_metas[15]["team_code"] if shot_metas[15]["team_code"] else ""

            ends_rows.append({
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

            for shot_idx in range(16):
                shot_num = shot_idx + 1
                meta = shot_metas[shot_idx]
                crop = crop_shot_image(page_bgr, shot_images[shot_idx])

                red_stones, yellow_stones = _detect_stones_in_crop(crop)

                # Map red/yellow to team1/team2 using the team color indicator
                # images on the page.  In the PDF the first small indicator
                # (image index 3 in page.images, a 31×31 red dot) is team1
                # and the second (image index 2, a 31×31 yellow dot) is team2.
                # So team1 = red, team2 = yellow.
                team1_stones = red_stones
                team2_stones = yellow_stones

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
                    "team1_stones_in_play": len(team1_stones),
                    "team2_stones_in_play": len(team2_stones),
                }

                # Stone positions – up to 8 per team, sorted by distance
                for ti, stones in enumerate([team1_stones, team2_stones], start=1):
                    prefix = f"team{ti}"
                    for si in range(MAX_STONES_PER_TEAM):
                        if si < len(stones):
                            nx, ny, dist, angle = stones[si]
                            row[f"{prefix}_stone{si+1}_x"] = round(nx, 3)
                            row[f"{prefix}_stone{si+1}_y"] = round(ny, 3)
                            row[f"{prefix}_stone{si+1}_dist"] = round(dist, 3)
                            row[f"{prefix}_stone{si+1}_angle"] = round(angle, 1)
                        else:
                            row[f"{prefix}_stone{si+1}_x"] = ""
                            row[f"{prefix}_stone{si+1}_y"] = ""
                            row[f"{prefix}_stone{si+1}_dist"] = ""
                            row[f"{prefix}_stone{si+1}_angle"] = ""

                shots_rows.append(row)

    pdf.close()

    return matches_rows, teams_dict, players_dict, ends_rows, shots_rows


def extract_all(pdf_paths, output_dir="output"):
    """Run the full extraction pipeline for one or more PDFs and write CSV tables.

    Parameters
    ----------
    pdf_paths : str or list[str]
        Path to a single PDF, an HTTP(S) URL, or a list of paths/URLs.
    output_dir : str
        Directory for output CSV files.
    """
    if isinstance(pdf_paths, str):
        pdf_paths = [pdf_paths]

    os.makedirs(output_dir, exist_ok=True)

    # Accumulators across all events
    events_rows = []
    all_matches = []
    all_teams_dict = {}   # (event_id, code) -> {name, players set}
    all_players_dict = {} # (event_id, code, name) -> id
    all_ends = []
    all_shots = []

    global_player_id = 0

    for event_id, pdf_path in enumerate(pdf_paths, start=1):
        event_name = os.path.splitext(os.path.basename(pdf_path))[0]
        print(f"\n{'='*60}")
        print(f"Event {event_id}: {event_name}")
        print(f"  PDF: {pdf_path}")
        print(f"{'='*60}")

        events_rows.append({
            "event_id": event_id,
            "event_name": event_name,
            "pdf_file": os.path.basename(pdf_path),
        })

        matches, teams_dict, players_dict, ends, shots = extract_event(pdf_path, event_id)

        all_matches.extend(matches)
        all_ends.extend(ends)
        all_shots.extend(shots)

        # Merge teams scoped by event_id
        for code, info in teams_dict.items():
            key = (event_id, code)
            if key not in all_teams_dict:
                all_teams_dict[key] = {"name": info["name"], "players": set(info["players"])}
            else:
                all_teams_dict[key]["players"].update(info["players"])
                if info["name"] and not all_teams_dict[key]["name"]:
                    all_teams_dict[key]["name"] = info["name"]

        # Re-number player IDs globally
        for (eid, tc, pn), local_id in players_dict.items():
            if (eid, tc, pn) not in all_players_dict:
                global_player_id += 1
                all_players_dict[(eid, tc, pn)] = global_player_id

    # Remap player IDs to global IDs in shots
    local_to_global = {}
    for (eid, tc, pn), gid in all_players_dict.items():
        local_to_global[(eid, tc, pn)] = gid

    for row in all_shots:
        eid = row["event_id"]
        tc = row["team_code"]
        pn = row["player_name"]
        if tc and pn:
            row["player_id"] = local_to_global.get((eid, tc, pn), row["player_id"])

    # ---- Write CSVs --------------------------------------------------------
    _write_events_csv(os.path.join(output_dir, "events.csv"), events_rows)
    _write_matches_csv(os.path.join(output_dir, "matches.csv"), all_matches)
    _write_teams_csv(os.path.join(output_dir, "teams.csv"), all_teams_dict)
    _write_players_csv(os.path.join(output_dir, "players.csv"), all_players_dict)
    _write_ends_csv(os.path.join(output_dir, "ends.csv"), all_ends)
    _write_shots_csv(os.path.join(output_dir, "shot_locations.csv"), all_shots)

    total_matches = len(all_matches)
    total_ends = len(all_ends)
    total_shots = len(all_shots)
    print(f"\nDone – {len(events_rows)} events, {total_matches} matches, "
          f"{total_ends} ends, {total_shots} shots written to {output_dir}/")


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------

def _write_events_csv(path, rows):
    fields = ["event_id", "event_name", "pdf_file"]
    _write_csv(path, fields, rows)


def _write_matches_csv(path, rows):
    fields = [
        "event_id", "match_id", "date", "round", "start_time",
        "team1_code", "team2_code",
        "team1_final_score", "team2_final_score",
    ]
    _write_csv(path, fields, rows)


def _write_teams_csv(path, teams_dict):
    rows = []
    for (event_id, code), info in sorted(teams_dict.items()):
        player_list = sorted(info["players"])
        row = {"event_id": event_id, "team_code": code, "team_name": info["name"]}
        for i, p in enumerate(player_list, start=1):
            row[f"player{i}_name"] = p
        rows.append(row)
    # Determine max players across all teams
    max_p = max((len(r) - 3 for r in rows), default=0)
    fields = ["event_id", "team_code", "team_name"] + [f"player{i}_name" for i in range(1, max_p + 1)]
    _write_csv(path, fields, rows)


def _write_players_csv(path, players_dict):
    rows = [
        {"player_id": pid, "event_id": eid, "team_code": tc, "player_name": pn}
        for (eid, tc, pn), pid in sorted(players_dict.items(), key=lambda x: x[1])
    ]
    _write_csv(path, ["player_id", "event_id", "team_code", "player_name"], rows)


def _write_ends_csv(path, rows):
    fields = [
        "event_id", "match_id", "end_number",
        "team1_code", "team2_code",
        "team1_score_before", "team2_score_before",
        "team1_score_this_end", "team2_score_this_end",
        "team1_score_after", "team2_score_after",
        "hammer_team_code",
        "team1_time_left", "team2_time_left",
    ]
    _write_csv(path, fields, rows)


def _write_shots_csv(path, rows):
    base_fields = [
        "event_id", "match_id", "end_number", "shot_number",
        "team_code", "player_id", "player_name",
        "shot_type", "turn", "accuracy",
        "team1_stones_in_play", "team2_stones_in_play",
    ]
    stone_fields = []
    for ti in (1, 2):
        for si in range(1, MAX_STONES_PER_TEAM + 1):
            prefix = f"team{ti}_stone{si}"
            stone_fields += [f"{prefix}_x", f"{prefix}_y",
                             f"{prefix}_dist", f"{prefix}_angle"]
    _write_csv(path, base_fields + stone_fields, rows)


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
                             "If omitted, the default PDF URLs are used.")
    parser.add_argument("--output-dir", default="output",
                        help="Directory for output CSV files (default: output)")
    args = parser.parse_args()

    pdf_sources = args.pdfs if args.pdfs else DEFAULT_PDF_URLS

    for src in pdf_sources:
        if not _is_url(src) and not os.path.isfile(src):
            parser.error(f"PDF not found: {src}")

    print(f"Extracting shot data from {len(pdf_sources)} PDF(s) …")
    extract_all(pdf_sources, args.output_dir)


if __name__ == "__main__":
    main()
