"""Tests for extract_shot_data module."""

import math
import os
import csv
import urllib.error
import urllib.request

import pytest

from extract_shot_data import (
    parse_end_line,
    parse_match_header,
    parse_score_box,
    find_shot_pages,
    _get_shot_images,
    _extract_shot_metadata_from_words,
    _detect_stones_in_crop,
    _match_stones_to_state,
    group_pages_into_matches,
    extract_all,
    extract_event,
    _open_pdf,
    _is_url,
    load_result_urls,
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

        red_stones, yellow_stones = _detect_stones_in_crop(crop)
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
