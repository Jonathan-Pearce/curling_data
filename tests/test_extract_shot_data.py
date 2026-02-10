"""Tests for extract_shot_data module."""

import math
import os
import csv

import pytest

from extract_shot_data import (
    parse_end_line,
    parse_match_header,
    parse_score_box,
    find_shot_pages,
    _get_shot_images,
    _extract_shot_metadata_from_words,
    _detect_stones_in_crop,
    group_pages_into_matches,
    extract_all,
    extract_event,
)

PDF_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "pdfs", "ECC2025_ResultsBook_Men_A-Division.pdf")
PDF_EXISTS = os.path.isfile(PDF_PATH)

WMCC_PDF_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "pdfs", "WMCC2023_ResultsBook.pdf")
WMCC_PDF_EXISTS = os.path.isfile(WMCC_PDF_PATH)


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


# ---------------------------------------------------------------------------
# Integration tests (require PDF)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not PDF_EXISTS, reason="PDF not available")
class TestWithPDF:
    @pytest.fixture(autouse=True)
    def setup_pdf(self):
        import pdfplumber
        self.pdf = pdfplumber.open(PDF_PATH)
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
        extract_all(PDF_PATH, output_dir)

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


@pytest.mark.skipif(not (PDF_EXISTS and WMCC_PDF_EXISTS), reason="Both PDFs required")
class TestMultiEvent:
    def test_multi_event_extraction(self, tmp_path):
        output_dir = str(tmp_path / "output")
        extract_all([PDF_PATH, WMCC_PDF_PATH], output_dir)

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
