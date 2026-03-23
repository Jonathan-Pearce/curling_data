"""Tests for extract_shot_data module."""

import math
import os
import csv
import urllib.error
import urllib.request
from unittest.mock import patch

import pytest

from extract_shot_data import (
    parse_end_line,
    parse_match_header,
    parse_score_box,
    find_shot_pages,
    _get_shot_images,
    _filter_shot_images_by_grid,
    _keep_regular_columns,
    _extract_shot_metadata_from_words,
    _detect_stones_in_crop,
    _match_stones_to_state,
    group_pages_into_matches,
    extract_all,
    extract_event,
    _open_pdf,
    _is_url,
    load_result_urls,
    run_calibration_diagnostic,
    DEFAULT_PDF_URLS,
    DEFAULT_MIN_YEAR,
    STONE_TRACK_MAX_DIST,
)

PDF_URL = "https://curlit.com/PDF/ECC2025_ResultsBook_Men_A-Division.pdf"
WMCC_PDF_URL = "https://curlit.com/PDF/WMCC2023_ResultsBook.pdf"


def _url_accessible(url):
    """Return True if a HEAD request to *url* succeeds."""
    try:
        req = urllib.request.Request(url, method="HEAD")
        urllib.request.urlopen(req, timeout=10)
        return True
    except (urllib.error.URLError, urllib.error.HTTPError, OSError):
        return False


PDF_AVAILABLE = _url_accessible(PDF_URL)
WMCC_PDF_AVAILABLE = _url_accessible(WMCC_PDF_URL)


# ---------------------------------------------------------------------------
# Unit tests for text parsing (no PDF required)
# ---------------------------------------------------------------------------

class TestParseEndLine:
    def test_normal_end(self):
        text = "End 7 SUI - Switzerland 1 + 0 (this end) = 1 SWE - Sweden 2 + 1 (this end) = 3"
        result = parse_end_line(text)
        assert result is not None
        assert result["end_number"] == 7
        assert result["team1_code"] == "SUI"
        assert result["team1_name"] == "Switzerland"
        assert result["team1_score_before"] == 1
        assert result["team1_score_this_end"] == 0
        assert result["team1_score_after"] == 1
        assert result["team2_code"] == "SWE"
        assert result["team2_name"] == "Sweden"
        assert result["team2_score_before"] == 2
        assert result["team2_score_this_end"] == 1
        assert result["team2_score_after"] == 3

    def test_x_score(self):
        text = "End 10 AUT - Austria 5 + X (this end) = 5 SWE - Sweden 7 + X (this end) = 7"
        result = parse_end_line(text)
        assert result is not None
        assert result["end_number"] == 10
        assert result["team1_score_this_end"] == "X"
        assert result["team2_score_this_end"] == "X"
        assert result["team1_score_after"] == 5
        assert result["team2_score_after"] == 7

    def test_no_match(self):
        assert parse_end_line("Some random text") is None

    def test_first_end(self):
        text = "End 1 SUI - Switzerland 0 + 0 (this end) = 0 SWE - Sweden 0 + 1 (this end) = 1"
        result = parse_end_line(text)
        assert result is not None
        assert result["end_number"] == 1
        assert result["team1_score_before"] == 0
        assert result["team2_score_this_end"] == 1


class TestParseMatchHeader:
    def test_gold_medal(self):
        text = "SAT 29 NOV 2025 Gold Medal Game\nStart Time 15:00\nOther lines"
        date, round_name, start_time = parse_match_header(text)
        assert date == "SAT 29 NOV 2025"
        assert round_name == "Gold Medal Game"
        assert start_time == "15:00"

    def test_round_robin(self):
        text = "THU 27 NOV 2025 Round Robin Session 9 - Sheet A\nStart Time 14:00"
        date, round_name, start_time = parse_match_header(text)
        assert date == "THU 27 NOV 2025"
        assert round_name == "Round Robin Session 9 - Sheet A"
        assert start_time == "14:00"

    def test_no_match(self):
        date, round_name, start_time = parse_match_header("no header here")
        assert date == ""
        assert round_name == ""
        assert start_time == ""


class TestParseScoreBox:
    def test_normal(self):
        text = "Total Score 1 3\nTime left 11:41 9:27"
        scores, times = parse_score_box(text)
        assert scores["team1"] == "1"
        assert scores["team2"] == "3"
        assert times["team1"] == "11:41"
        assert times["team2"] == "9:27"

    def test_no_score_box(self):
        scores, times = parse_score_box("no score info")
        assert scores == {}
        assert times == {}


class TestUrlHelpers:
    def test_is_url_http(self):
        assert _is_url("http://example.com/file.pdf")

    def test_is_url_https(self):
        assert _is_url("https://curlit.com/PDF/ECC2025_ResultsBook_Men_A-Division.pdf")

    def test_is_url_local_path(self):
        assert not _is_url("/some/local/path.pdf")

    def test_is_url_relative_path(self):
        assert not _is_url("data/pdfs/file.pdf")

    def test_default_pdf_urls(self):
        assert len(DEFAULT_PDF_URLS) == 2
        for url in DEFAULT_PDF_URLS:
            assert _is_url(url)


class TestLoadResultUrls:
    def test_filters_by_min_year(self, tmp_path):
        csv_path = str(tmp_path / "result_urls.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "tournament_name", "year", "location", "gender",
                "result_book_url", "result_summary_url",
            ])
            writer.writeheader()
            writer.writerow({
                "tournament_name": "Old Event", "year": "2012",
                "location": "City", "gender": "m",
                "result_book_url": "https://example.com/old.pdf",
                "result_summary_url": "",
            })
            writer.writerow({
                "tournament_name": "New Event", "year": "2013",
                "location": "Town", "gender": "w",
                "result_book_url": "https://example.com/new.pdf",
                "result_summary_url": "",
            })
            writer.writerow({
                "tournament_name": "Recent Event", "year": "2024",
                "location": "Place", "gender": "mx",
                "result_book_url": "https://example.com/recent.pdf",
                "result_summary_url": "",
            })

        rows = load_result_urls(csv_path, min_year=2013)
        assert len(rows) == 2
        assert rows[0]["tournament_name"] == "New Event"
        assert rows[1]["tournament_name"] == "Recent Event"

    def test_skips_invalid_year(self, tmp_path):
        csv_path = str(tmp_path / "result_urls.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "tournament_name", "year", "location", "gender",
                "result_book_url", "result_summary_url",
            ])
            writer.writeheader()
            writer.writerow({
                "tournament_name": "Bad Year", "year": "unknown",
                "location": "City", "gender": "m",
                "result_book_url": "https://example.com/bad.pdf",
                "result_summary_url": "",
            })

        rows = load_result_urls(csv_path, min_year=2013)
        assert len(rows) == 0

    def test_skips_empty_url(self, tmp_path):
        csv_path = str(tmp_path / "result_urls.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "tournament_name", "year", "location", "gender",
                "result_book_url", "result_summary_url",
            ])
            writer.writeheader()
            writer.writerow({
                "tournament_name": "No URL", "year": "2024",
                "location": "City", "gender": "m",
                "result_book_url": "",
                "result_summary_url": "",
            })

        rows = load_result_urls(csv_path, min_year=2013)
        assert len(rows) == 0

    def test_default_min_year(self):
        assert DEFAULT_MIN_YEAR == 2013

    def test_with_actual_csv(self):
        csv_path = os.path.join("output", "result_urls.csv")
        if not os.path.isfile(csv_path):
            pytest.skip("result_urls.csv not present")
        rows = load_result_urls(csv_path)
        assert len(rows) > 0
        for row in rows:
            assert int(row["year"]) >= 2013
            assert row["result_book_url"]



class TestShotMetadataWords:
    """Unit tests for _extract_shot_metadata_from_words."""

    def _make_shot_images(self, n=16):
        """Return *n* minimal shot-image dicts in a single row layout."""
        # Each image is 30px wide, spaced 5px apart, bottom at y=100
        imgs = []
        for i in range(n):
            x0 = i * 35
            imgs.append({"x0": x0, "x1": x0 + 30, "bottom": 100})
        return imgs

    def test_returns_16_shots(self):
        imgs = self._make_shot_images()
        shots = _extract_shot_metadata_from_words([], imgs)
        assert len(shots) == 16

    def test_empty_words_empty_shots(self):
        imgs = self._make_shot_images()
        shots = _extract_shot_metadata_from_words([], imgs)
        for s in shots:
            assert s["team_code"] == ""
            assert s["turn"] == ""
            assert s["shot_type"] == ""

    def test_through_shot_gets_not_considered_turn(self):
        """Through-violation shots have no turn indicator in the PDF.
        The post-processing guard should set turn = 'Not considered'."""
        imgs = self._make_shot_images()
        # Inject a 'Through Hog line violation' word into shot 0's type row.
        # type row y-range: bottom + 6 to bottom + 25 → 106–125
        words = [
            {"x0": 5, "top": 110, "text": "Through"},
            {"x0": 5, "top": 110, "text": "Hog"},
            {"x0": 5, "top": 110, "text": "line"},
            {"x0": 5, "top": 110, "text": "violation"},
        ]
        shots = _extract_shot_metadata_from_words(words, imgs)
        assert shots[0]["shot_type"] == "Through Hog line violation"
        assert shots[0]["turn"] == "Not considered"

    def test_through_with_existing_turn_not_overwritten(self):
        """If a Through shot somehow already has a turn value, it is kept."""
        imgs = self._make_shot_images()
        # Simulate ↺ and 'Through' in the same type row for shot 0
        words = [
            {"x0": 5, "top": 110, "text": "↺"},
            {"x0": 5, "top": 110, "text": "Through"},
        ]
        shots = _extract_shot_metadata_from_words(words, imgs)
        # The ↺ sets turn first; the guard should not overwrite it
        assert shots[0]["turn"] == "Counter-clockwise"

    def test_normal_shot_turn_unchanged(self):
        """Non-Through shots with no turn token should have empty turn."""
        imgs = self._make_shot_images()
        words = [
            {"x0": 5, "top": 110, "text": "Draw"},
        ]
        shots = _extract_shot_metadata_from_words(words, imgs)
        assert shots[0]["shot_type"] == "Draw"
        assert shots[0]["turn"] == ""  # no turn token — stays empty


class TestStoneTracking:
    """Unit tests for _match_stones_to_state sequential stone tracking."""

    def _raw(self, *coords):
        """Build a list of raw stone tuples (nx, ny, dist, angle) from (nx, ny) pairs."""
        result = []
        for nx, ny in coords:
            dist = math.sqrt(nx ** 2 + ny ** 2)
            angle = math.degrees(math.atan2(ny, nx))
            result.append((nx, ny, dist, angle))
        return result

    def test_empty_state_all_new_ids(self):
        """With no previous state, every stone gets a fresh ID."""
        curr = self._raw((0.1, 0.2), (0.3, 0.4))
        counter = [1]
        matched, new_state = _match_stones_to_state([], curr, counter)
        assert len(matched) == 2
        ids = [s[0] for s in matched]
        assert ids == [1, 2]
        assert counter[0] == 3  # consumed 2 IDs
        # No previous position for newly placed stones
        for sid, nx, ny, dist, angle, prev_x, prev_y in matched:
            assert prev_x is None
            assert prev_y is None

    def test_same_positions_ids_propagated(self):
        """Stones at identical positions between shots keep their IDs."""
        prev_state = [(5, 0.1, 0.2), (6, 0.5, 0.5)]
        curr = self._raw((0.1, 0.2), (0.5, 0.5))
        counter = [10]
        matched, new_state = _match_stones_to_state(prev_state, curr, counter)
        matched_ids = {(round(s[1], 3), round(s[2], 3)): s[0] for s in matched}
        assert matched_ids[(0.1, 0.2)] == 5
        assert matched_ids[(0.5, 0.5)] == 6
        # Counter not advanced — no new IDs needed
        assert counter[0] == 10

    def test_new_stone_added(self):
        """When a new stone appears, existing stones keep IDs; new one gets next ID."""
        prev_state = [(3, 0.1, 0.2)]
        # Two stones now: old one plus a newly placed stone
        curr = self._raw((0.1, 0.2), (0.8, 0.0))
        counter = [7]
        matched, new_state = _match_stones_to_state(prev_state, curr, counter)
        assert len(matched) == 2
        by_pos = {(round(s[1], 3), round(s[2], 3)): s for s in matched}
        # Existing stone keeps ID 3; prev_x/prev_y populated
        old = by_pos[(0.1, 0.2)]
        assert old[0] == 3
        assert old[5] == pytest.approx(0.1)
        assert old[6] == pytest.approx(0.2)
        # New stone gets ID 7
        new = by_pos[(0.8, 0.0)]
        assert new[0] == 7
        assert new[5] is None
        assert new[6] is None
        assert counter[0] == 8

    def test_stone_removed_absent_from_new_state(self):
        """A stone that leaves play is simply absent from new_state."""
        prev_state = [(1, 0.0, 0.0), (2, 0.5, 0.5)]
        # Only one stone remains
        curr = self._raw((0.5, 0.5),)
        counter = [10]
        matched, new_state = _match_stones_to_state(prev_state, curr, counter)
        assert len(matched) == 1
        assert matched[0][0] == 2  # stone 2 survived
        assert len(new_state) == 1
        # Counter unchanged — the remaining stone was matched
        assert counter[0] == 10

    def test_stone_beyond_threshold_treated_as_new(self):
        """A stone that moves beyond STONE_TRACK_MAX_DIST is treated as removed +
        a new stone placed, not as a continued stone."""
        far = STONE_TRACK_MAX_DIST + 0.05
        prev_state = [(1, 0.0, 0.0)]
        curr = self._raw((far, 0.0),)  # displaced well past threshold
        counter = [2]
        matched, new_state = _match_stones_to_state(prev_state, curr, counter)
        # Should not match: prev stone 1 is gone; current detection is new
        assert matched[0][0] == 2   # new ID assigned
        assert matched[0][5] is None  # no prev_x
        assert counter[0] == 3

    def test_stone_within_threshold_matched(self):
        """A stone slightly shifted (noise) within the threshold is matched."""
        shift = STONE_TRACK_MAX_DIST - 0.01
        prev_state = [(4, 0.0, 0.0)]
        curr = self._raw((shift, 0.0),)
        counter = [9]
        matched, _ = _match_stones_to_state(prev_state, curr, counter)
        assert matched[0][0] == 4   # same ID
        assert matched[0][5] == pytest.approx(0.0)  # prev_x
        assert matched[0][6] == pytest.approx(0.0)  # prev_y
        assert counter[0] == 9  # no new ID consumed

    def test_new_state_carries_current_positions(self):
        """new_state reflects the current (post-shot) stone positions."""
        prev_state = [(1, 0.0, 0.0)]
        curr = self._raw((0.01, 0.01),)  # tiny noise shift — still matched
        counter = [2]
        _, new_state = _match_stones_to_state(prev_state, curr, counter)
        assert len(new_state) == 1
        sid, px, py = new_state[0]
        assert sid == 1
        assert px == pytest.approx(0.01)
        assert py == pytest.approx(0.01)

    def test_counter_shared_across_teams(self):
        """Passing the same counter to two teams gives non-overlapping IDs."""
        counter = [1]
        curr_t1 = self._raw((0.1, 0.0), (0.2, 0.0))
        curr_t2 = self._raw((0.4, 0.0),)
        matched_t1, _ = _match_stones_to_state([], curr_t1, counter)
        matched_t2, _ = _match_stones_to_state([], curr_t2, counter)
        ids_t1 = {s[0] for s in matched_t1}
        ids_t2 = {s[0] for s in matched_t2}
        assert ids_t1 == {1, 2}
        assert ids_t2 == {3}
        assert ids_t1.isdisjoint(ids_t2)


# ---------------------------------------------------------------------------
# Integration tests (require PDF)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not PDF_AVAILABLE, reason="PDF URL not accessible")
class TestWithPDF:
    @pytest.fixture(autouse=True)
    def setup_pdf(self):
        self.pdf = _open_pdf(PDF_URL)
        yield
        self.pdf.close()

    def test_find_shot_pages(self):
        pages = find_shot_pages(self.pdf)
        assert len(pages) > 400
        assert pages[0] == 7

    def test_get_shot_images(self):
        page = self.pdf.pages[7]
        imgs = _get_shot_images(page)
        assert len(imgs) == 16

    def test_shot_metadata_extraction_end1(self):
        page = self.pdf.pages[7]
        words = page.extract_words()
        shot_imgs = _get_shot_images(page)
        shots = _extract_shot_metadata_from_words(words, shot_imgs)
        assert len(shots) == 16
        # Shot 1 of End 1, Gold Medal
        assert shots[0]["team_code"] == "SWE"
        assert shots[0]["player_name"] == "SUNDGREN C"
        assert shots[0]["shot_type"] == "Front"
        assert shots[0]["accuracy"] == "100"
        # Shot 16
        assert shots[15]["team_code"] == "SUI"
        assert "SCHWARZ-VAN BERKEL" in shots[15]["player_name"]

    def test_shot_metadata_extraction_end7(self):
        page = self.pdf.pages[13]
        words = page.extract_words()
        shot_imgs = _get_shot_images(page)
        shots = _extract_shot_metadata_from_words(words, shot_imgs)
        # Shot 6: SWE: WRANAA R, Double Take-out
        assert shots[5]["team_code"] == "SWE"
        assert shots[5]["player_name"] == "WRANAA R"
        assert shots[5]["shot_type"] == "Double Take-out"
        # Shot 15: SUI: SCHWARZ-VAN BERKEL, Promotion Take-out
        assert shots[14]["shot_type"] == "Promotion Take-out"

    def test_group_pages_into_matches(self):
        pages = find_shot_pages(self.pdf)
        matches = group_pages_into_matches(self.pdf, pages)
        assert len(matches) == 49

    def test_stone_detection(self):
        import numpy as np
        import cv2
        from extract_shot_data import crop_shot_image

        page = self.pdf.pages[13]
        shot_imgs = _get_shot_images(page)
        pil_img = page.to_image(resolution=300).original
        arr = np.array(pil_img.convert("RGB"))
        page_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

        crop = crop_shot_image(page_bgr, shot_imgs[11])  # shot 12

        red_stones, yellow_stones, _ = _detect_stones_in_crop(crop)
        # Shot 12 should have several stones visible
        assert len(red_stones) >= 3
        assert len(yellow_stones) >= 3
        # Stones should be sorted by distance
        for i in range(1, len(red_stones)):
            assert red_stones[i][2] >= red_stones[i - 1][2]
        for i in range(1, len(yellow_stones)):
            assert yellow_stones[i][2] >= yellow_stones[i - 1][2]

    def test_full_extraction(self, tmp_path):
        output_dir = str(tmp_path / "output")
        extract_all(PDF_URL, output_dir)

        for fname in ["events.csv", "matches.csv", "teams.csv", "players.csv", "ends.csv", "shot_locations.csv"]:
            fpath = os.path.join(output_dir, fname)
            assert os.path.isfile(fpath), f"{fname} not created"

        with open(os.path.join(output_dir, "events.csv")) as f:
            events = list(csv.DictReader(f))
        assert len(events) == 1
        assert events[0]["event_id"] == "1"
        assert events[0]["pdf_file"] == "ECC2025_ResultsBook_Men_A-Division.pdf"

        with open(os.path.join(output_dir, "matches.csv")) as f:
            matches = list(csv.DictReader(f))
        assert len(matches) == 49
        assert matches[0]["event_id"] == "1"
        assert matches[0]["team1_code"] == "SUI"
        assert matches[0]["team2_code"] == "SWE"

        with open(os.path.join(output_dir, "ends.csv")) as f:
            ends = list(csv.DictReader(f))
        assert len(ends) == 437
        assert ends[0]["event_id"] == "1"

        with open(os.path.join(output_dir, "shot_locations.csv")) as f:
            shots = list(csv.DictReader(f))
        assert len(shots) == 6992
        assert shots[0]["event_id"] == "1"

        with open(os.path.join(output_dir, "teams.csv")) as f:
            teams = list(csv.DictReader(f))
        assert len(teams) == 10
        assert teams[0]["event_id"] == "1"

        with open(os.path.join(output_dir, "players.csv")) as f:
            players = list(csv.DictReader(f))
        assert players[0]["event_id"] == "1"


@pytest.mark.skipif(not (PDF_AVAILABLE and WMCC_PDF_AVAILABLE), reason="Both PDF URLs required")
class TestMultiEvent:
    def test_multi_event_extraction(self, tmp_path):
        output_dir = str(tmp_path / "output")
        extract_all([PDF_URL, WMCC_PDF_URL], output_dir)

        with open(os.path.join(output_dir, "events.csv")) as f:
            events = list(csv.DictReader(f))
        assert len(events) == 2
        assert events[0]["event_id"] == "1"
        assert events[1]["event_id"] == "2"

        with open(os.path.join(output_dir, "matches.csv")) as f:
            matches = list(csv.DictReader(f))
        event1_matches = [m for m in matches if m["event_id"] == "1"]
        event2_matches = [m for m in matches if m["event_id"] == "2"]
        assert len(event1_matches) == 49
        assert len(event2_matches) > 0

        with open(os.path.join(output_dir, "teams.csv")) as f:
            teams = list(csv.DictReader(f))
        event1_teams = [t for t in teams if t["event_id"] == "1"]
        event2_teams = [t for t in teams if t["event_id"] == "2"]
        assert len(event1_teams) == 10
        assert len(event2_teams) > 0


# ---------------------------------------------------------------------------
# Improvement 8 — Shot image grid validation
# ---------------------------------------------------------------------------

def _make_grid_image(x0, top, width=60, height=110):
    """Return a minimal image-metadata dict like pdfplumber produces."""
    return {"x0": x0, "x1": x0 + width, "top": top, "bottom": top + height,
            "width": width, "height": height}


def _make_standard_grid():
    """Return a list of 16 image dicts arranged in the standard 6-6-4 layout."""
    images = []
    col_xs = [10 + i * 70 for i in range(6)]  # 6 columns
    row_tops = [50, 200, 350]                  # 3 rows
    counts = [6, 6, 4]
    for row_top, count in zip(row_tops, counts):
        for ci in range(count):
            images.append(_make_grid_image(col_xs[ci], row_top))
    return images


class TestKeepRegularColumns:
    """Unit tests for _keep_regular_columns."""

    def test_exact_count_returns_input(self):
        images = [_make_grid_image(i * 70, 50) for i in range(6)]
        result = _keep_regular_columns(images, 6)
        assert result == images

    def test_one_extra_removes_outlier(self):
        # 6 evenly-spaced images plus one extra squeezed between cols 0 and 1
        images = [_make_grid_image(i * 70, 50) for i in range(6)]
        outlier = _make_grid_image(10, 50)        # x0=10, very close to first
        all_imgs = sorted(images + [outlier], key=lambda i: i["x0"])
        result = _keep_regular_columns(all_imgs, 6)
        assert result is not None
        assert len(result) == 6
        # The outlier at x0=10 should be excluded; the regular grid stays
        result_xs = sorted(img["x0"] for img in result)
        assert result_xs == sorted(img["x0"] for img in images)

    def test_two_extras_returns_none(self):
        images = [_make_grid_image(i * 70, 50) for i in range(8)]
        result = _keep_regular_columns(images, 6)
        assert result is None


class TestFilterShotImagesByGrid:
    """Unit tests for _filter_shot_images_by_grid (improvement #8)."""

    def test_exact_16_unchanged(self):
        images = _make_standard_grid()
        result = _filter_shot_images_by_grid(images)
        assert len(result) == 16

    def test_logo_in_row1_excluded(self):
        """An extra image in row 1 that disrupts column spacing is removed."""
        images = _make_standard_grid()
        # Insert a logo-like image at an x position that breaks row 1's spacing
        logo = _make_grid_image(x0=5, top=50)   # same y-row as row1, awkward x
        candidates = images + [logo]
        result = _filter_shot_images_by_grid(candidates)
        assert len(result) == 16
        # The logo should not appear in the result
        assert logo not in result

    def test_logo_in_separate_row_no_resolve(self):
        """An extra image forming its own 4th row cannot be resolved; original returned."""
        images = _make_standard_grid()
        logo = _make_grid_image(x0=100, top=600)  # far below all 3 rows
        candidates = images + [logo]
        result = _filter_shot_images_by_grid(candidates)
        # Cannot split into 3 clean rows — should return candidates unchanged
        assert set(id(i) for i in result) == set(id(i) for i in candidates)

    def test_too_few_candidates_unchanged(self):
        images = _make_standard_grid()[:14]  # only 14
        result = _filter_shot_images_by_grid(images)
        assert result == images

    def test_get_shot_images_still_returns_16_for_normal_page(self):
        """_get_shot_images returns the same 16 images from a mock page."""
        images = _make_standard_grid()

        class FakePage:
            pass

        page = FakePage()
        page.images = images
        result = _get_shot_images(page)
        assert len(result) == 16


# ---------------------------------------------------------------------------
# Improvement 9 — Page-level orientation consistency warning
# ---------------------------------------------------------------------------

class TestOrientationConsistencyWarning:
    """Tests that a warning is printed when shot crops disagree on orientation."""

    def test_warning_on_mixed_orientations(self, capsys):
        """extract_event emits a WARNING when house orientations are inconsistent
        across the 16 crops of an end page."""
        # Build a mock PDF chain so we don't need a real file.
        import extract_shot_data as esd
        import numpy as np

        # Patch everything needed to run a minimal end-page pass.
        # _detect_stones_in_crop is called 16 times — we make 15 return 'top'
        # and 1 return 'bottom' to trigger the inconsistency warning.
        call_count = [0]

        def fake_detect(crop, color_ranges=None):
            call_count[0] += 1
            orient = "bottom" if call_count[0] == 3 else "top"
            return [], [], orient

        fake_page_bgr = np.zeros((700, 400, 3), dtype=np.uint8)

        def fake_render(page):
            return fake_page_bgr

        def fake_crop(page_bgr, meta):
            return np.zeros((100, 60, 3), dtype=np.uint8)

        # A minimal shot image list (16 items).
        shot_imgs = [_make_grid_image(i * 25, 50) for i in range(16)]

        def fake_get_shot_images(page):
            return shot_imgs

        def fake_calibrate(page_bgr, page):
            return None

        def fake_shot_meta(words, imgs):
            return [
                {"team_code": "T1", "player_name": "PLAYER A",
                 "shot_type": "Draw", "turn": "Clockwise", "accuracy": "90"}
                if i % 2 == 0 else
                {"team_code": "T2", "player_name": "PLAYER B",
                 "shot_type": "Guard", "turn": "Counter-clockwise", "accuracy": "85"}
                for i in range(16)
            ]

        # Minimal pdfplumber-like PDF stub
        class FakePage:
            def extract_text(self):
                return (
                    "SAT 01 JAN 2025 Round Robin\n"
                    "Start Time 10:00\n"
                    "End 1 T1 - Team One 0 + 1 (this end) = 1 "
                    "T2 - Team Two 0 + 0 (this end) = 0\n"
                    "Total Score 1 0\nTime left 30:00 30:00"
                )
            def extract_words(self):
                return []
            @property
            def images(self):
                return shot_imgs

        class FakePDF:
            pages = [FakePage()]
            def close(self):
                pass

        def fake_open(source):
            return FakePDF()

        def fake_find_shot_pages(pdf):
            return [0]

        def fake_group_pages(pdf, indices):
            return [[0]]

        def fake_match_stones(prev, curr, counter):
            return [], []

        with (
            patch.object(esd, "_open_pdf", side_effect=fake_open),
            patch.object(esd, "find_shot_pages", side_effect=fake_find_shot_pages),
            patch.object(esd, "group_pages_into_matches", side_effect=fake_group_pages),
            patch.object(esd, "_get_shot_images", side_effect=fake_get_shot_images),
            patch.object(esd, "render_page_image", side_effect=fake_render),
            patch.object(esd, "crop_shot_image", side_effect=fake_crop),
            patch.object(esd, "_calibrate_stone_colors", side_effect=fake_calibrate),
            patch.object(esd, "_extract_shot_metadata_from_words", side_effect=fake_shot_meta),
            patch.object(esd, "_detect_stones_in_crop", side_effect=fake_detect),
            patch.object(esd, "_match_stones_to_state", side_effect=fake_match_stones),
        ):
            esd.extract_event("fake.pdf", event_id=1)

        captured = capsys.readouterr()
        assert "WARNING" in captured.out
        assert "inconsistent house orientation" in captured.out.lower()

    def test_no_warning_on_uniform_orientations(self, capsys):
        """No warning is emitted when all crops agree on orientation."""
        import extract_shot_data as esd
        import numpy as np

        def fake_detect(crop, color_ranges=None):
            return [], [], "top"   # always top

        fake_page_bgr = np.zeros((700, 400, 3), dtype=np.uint8)
        shot_imgs = [_make_grid_image(i * 25, 50) for i in range(16)]

        class FakePage:
            def extract_text(self):
                return (
                    "SAT 01 JAN 2025 Round Robin\n"
                    "Start Time 10:00\n"
                    "End 1 T1 - Team One 0 + 1 (this end) = 1 "
                    "T2 - Team Two 0 + 0 (this end) = 0\n"
                    "Total Score 1 0\nTime left 30:00 30:00"
                )
            def extract_words(self):
                return []
            @property
            def images(self):
                return shot_imgs

        class FakePDF:
            pages = [FakePage()]
            def close(self):
                pass

        with (
            patch.object(esd, "_open_pdf", lambda _: FakePDF()),
            patch.object(esd, "find_shot_pages", lambda _: [0]),
            patch.object(esd, "group_pages_into_matches", lambda _p, _i: [[0]]),
            patch.object(esd, "_get_shot_images", lambda _: shot_imgs),
            patch.object(esd, "render_page_image", lambda _: fake_page_bgr),
            patch.object(esd, "crop_shot_image", lambda _b, _m: np.zeros((100, 60, 3), dtype=np.uint8)),
            patch.object(esd, "_calibrate_stone_colors", lambda _b, _p: None),
            patch.object(esd, "_extract_shot_metadata_from_words", lambda _w, _i: [
                {"team_code": "T1", "player_name": "P", "shot_type": "Draw",
                 "turn": "Clockwise", "accuracy": "90"} if k % 2 == 0 else
                {"team_code": "T2", "player_name": "Q", "shot_type": "Guard",
                 "turn": "Counter-clockwise", "accuracy": "80"}
                for k in range(16)
            ]),
            patch.object(esd, "_detect_stones_in_crop", side_effect=fake_detect),
            patch.object(esd, "_match_stones_to_state", lambda *a: ([], [])),
        ):
            esd.extract_event("fake.pdf", event_id=1)

        captured = capsys.readouterr()
        assert "inconsistent house orientation" not in captured.out.lower()


# ---------------------------------------------------------------------------
# Improvement 10 — Per-event calibration diagnostic
# ---------------------------------------------------------------------------

class TestRunCalibrationDiagnostic:
    """Unit tests for run_calibration_diagnostic (improvement #10)."""

    def _make_minimal_pdf_mocks(self, detected_radius=112, suspect=False):
        """Return a set of patches for a minimal successful calibration run."""
        import numpy as np
        import extract_shot_data as esd

        fake_bgr = np.zeros((700, 400, 3), dtype=np.uint8)
        shot_imgs = [_make_grid_image(i * 25, 50) for i in range(16)]

        class FakePage:
            @property
            def images(self):
                return shot_imgs

        class FakePDF:
            pages = [FakePage()]
            def close(self): pass

        dist = 2.0 if suspect else 0.3
        stones = [(0.0, 0.0, dist, 0.0)]

        return {
            "_open_pdf": lambda _: FakePDF(),
            "find_shot_pages": lambda _: [0],
            "_get_shot_images": lambda _: shot_imgs,
            "render_page_image": lambda _: fake_bgr,
            "crop_shot_image": lambda _b, _m: np.zeros((100, 60, 3), dtype=np.uint8),
            "_detect_house_center": lambda _: (161.0, 171.0, float(detected_radius), "top"),
            "_calibrate_stone_colors": lambda _b, _p: None,
            "_detect_stones_in_crop": lambda _c, _r=None: (stones, [], "top"),
        }

    def test_ok_result_for_good_event(self):
        import extract_shot_data as esd

        patches = self._make_minimal_pdf_mocks(detected_radius=112, suspect=False)
        with (
            patch.object(esd, "_open_pdf", patches["_open_pdf"]),
            patch.object(esd, "find_shot_pages", patches["find_shot_pages"]),
            patch.object(esd, "_get_shot_images", patches["_get_shot_images"]),
            patch.object(esd, "render_page_image", patches["render_page_image"]),
            patch.object(esd, "crop_shot_image", patches["crop_shot_image"]),
            patch.object(esd, "_detect_house_center", patches["_detect_house_center"]),
            patch.object(esd, "_calibrate_stone_colors", patches["_calibrate_stone_colors"]),
            patch.object(esd, "_detect_stones_in_crop", patches["_detect_stones_in_crop"]),
        ):
            results = run_calibration_diagnostic(["fake_event.pdf"], verbose=False)

        assert len(results) == 1
        assert results[0]["status"] == "ok"
        assert results[0]["detected_radius"] == 112.0
        assert results[0]["deviation_pct"] == 0.0

    def test_radius_deviation_flagged(self):
        import extract_shot_data as esd

        # 140 px is 25% above the nominal 112 — exceeds 10% threshold
        patches = self._make_minimal_pdf_mocks(detected_radius=140, suspect=False)
        with (
            patch.object(esd, "_open_pdf", patches["_open_pdf"]),
            patch.object(esd, "find_shot_pages", patches["find_shot_pages"]),
            patch.object(esd, "_get_shot_images", patches["_get_shot_images"]),
            patch.object(esd, "render_page_image", patches["render_page_image"]),
            patch.object(esd, "crop_shot_image", patches["crop_shot_image"]),
            patch.object(esd, "_detect_house_center", patches["_detect_house_center"]),
            patch.object(esd, "_calibrate_stone_colors", patches["_calibrate_stone_colors"]),
            patch.object(esd, "_detect_stones_in_crop", patches["_detect_stones_in_crop"]),
        ):
            results = run_calibration_diagnostic(["fake_event.pdf"], verbose=False)

        assert results[0]["status"] == "radius_deviation"
        assert results[0]["deviation_pct"] > 10.0

    def test_suspect_stones_flagged(self):
        import extract_shot_data as esd

        # Radius OK but a stone is detected far outside the house
        patches = self._make_minimal_pdf_mocks(detected_radius=112, suspect=True)
        with (
            patch.object(esd, "_open_pdf", patches["_open_pdf"]),
            patch.object(esd, "find_shot_pages", patches["find_shot_pages"]),
            patch.object(esd, "_get_shot_images", patches["_get_shot_images"]),
            patch.object(esd, "render_page_image", patches["render_page_image"]),
            patch.object(esd, "crop_shot_image", patches["crop_shot_image"]),
            patch.object(esd, "_detect_house_center", patches["_detect_house_center"]),
            patch.object(esd, "_calibrate_stone_colors", patches["_calibrate_stone_colors"]),
            patch.object(esd, "_detect_stones_in_crop", patches["_detect_stones_in_crop"]),
        ):
            results = run_calibration_diagnostic(["fake_event.pdf"], verbose=False)

        assert results[0]["status"] == "suspect_stones"
        assert results[0]["shot1_suspect"] >= 1

    def test_download_failure_returns_error_status(self):
        import extract_shot_data as esd

        def bad_open(_):
            raise RuntimeError("404 Not Found")

        with patch.object(esd, "_open_pdf", side_effect=bad_open):
            results = run_calibration_diagnostic(
                ["https://example.com/missing.pdf"], verbose=False
            )

        assert len(results) == 1
        assert results[0]["status"] == "error"

    def test_no_shot_pages_status(self):
        import extract_shot_data as esd
        import numpy as np

        class FakePDF:
            pages = []
            def close(self): pass

        with (
            patch.object(esd, "_open_pdf", lambda _: FakePDF()),
            patch.object(esd, "find_shot_pages", lambda _: []),
        ):
            results = run_calibration_diagnostic(["fake.pdf"], verbose=False)

        assert results[0]["status"] == "no_shot_pages"

    def test_verbose_output_contains_summary(self, capsys):
        import extract_shot_data as esd

        patches = self._make_minimal_pdf_mocks(detected_radius=112, suspect=False)
        with (
            patch.object(esd, "_open_pdf", patches["_open_pdf"]),
            patch.object(esd, "find_shot_pages", patches["find_shot_pages"]),
            patch.object(esd, "_get_shot_images", patches["_get_shot_images"]),
            patch.object(esd, "render_page_image", patches["render_page_image"]),
            patch.object(esd, "crop_shot_image", patches["crop_shot_image"]),
            patch.object(esd, "_detect_house_center", patches["_detect_house_center"]),
            patch.object(esd, "_calibrate_stone_colors", patches["_calibrate_stone_colors"]),
            patch.object(esd, "_detect_stones_in_crop", patches["_detect_stones_in_crop"]),
        ):
            run_calibration_diagnostic(["fake_event.pdf"], verbose=True)

        captured = capsys.readouterr()
        assert "Calibration summary" in captured.out
        assert "fake_event" in captured.out
