"""Tests for generate_board_image module."""

import os
import csv
import math

import pytest
from PIL import Image

from generate_board_image import (
    generate_board_image,
    generate_board_image_from_csv,
    _extract_stones,
    _norm_to_pixel,
    _norm_radius_to_pixels,
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
    X_MIN,
    X_MAX,
    Y_MIN,
    Y_MAX,
    MAX_STONES_PER_TEAM,
)


# ---------------------------------------------------------------------------
# Coordinate mapping tests
# ---------------------------------------------------------------------------

class TestNormToPixel:
    def test_centre_maps_correctly(self):
        px, py = _norm_to_pixel(0, 0)
        expected_px = (0 - X_MIN) / (X_MAX - X_MIN) * IMAGE_WIDTH
        expected_py = (0 - Y_MIN) / (Y_MAX - Y_MIN) * IMAGE_HEIGHT
        assert abs(px - expected_px) < 1
        assert abs(py - expected_py) < 1

    def test_top_left_corner(self):
        # Y_MIN (back-line end) maps to the top-left of the image.
        px, py = _norm_to_pixel(X_MIN, Y_MIN)
        assert abs(px) < 1
        assert abs(py) < 1

    def test_bottom_right_corner(self):
        # Y_MAX (hog-line end) maps to the bottom-right of the image.
        px, py = _norm_to_pixel(X_MAX, Y_MAX)
        assert abs(px - IMAGE_WIDTH) < 1
        assert abs(py - IMAGE_HEIGHT) < 1


class TestNormRadiusToPixels:
    def test_twelve_foot_ring(self):
        r = _norm_radius_to_pixels(1.0)
        expected = 1.0 / (X_MAX - X_MIN) * IMAGE_WIDTH
        assert abs(r - expected) < 0.01

    def test_zero_radius(self):
        assert _norm_radius_to_pixels(0) == 0


# ---------------------------------------------------------------------------
# Stone extraction tests
# ---------------------------------------------------------------------------

class TestExtractStones:
    def test_basic_extraction(self):
        row = {
            "team1_stone1_x": "0.1", "team1_stone1_y": "0.2",
            "team1_stone2_x": "-0.5", "team1_stone2_y": "1.3",
            "team1_stone3_x": "", "team1_stone3_y": "",
        }
        stones = _extract_stones(row, 1)
        assert len(stones) == 2
        assert stones[0] == (0.1, 0.2)
        assert stones[1] == (-0.5, 1.3)

    def test_empty_data(self):
        row = {}
        stones = _extract_stones(row, 1)
        assert stones == []

    def test_all_empty_values(self):
        row = {f"team1_stone{i}_x": "" for i in range(1, 9)}
        row.update({f"team1_stone{i}_y": "" for i in range(1, 9)})
        assert _extract_stones(row, 1) == []

    def test_team2_extraction(self):
        row = {"team2_stone1_x": "0.5", "team2_stone1_y": "-0.3"}
        stones = _extract_stones(row, 2)
        assert len(stones) == 1
        assert stones[0] == (0.5, -0.3)

    def test_invalid_values_skipped(self):
        row = {
            "team1_stone1_x": "bad", "team1_stone1_y": "0.2",
            "team1_stone2_x": "0.3", "team1_stone2_y": "0.4",
        }
        stones = _extract_stones(row, 1)
        assert len(stones) == 1
        assert stones[0] == (0.3, 0.4)


# ---------------------------------------------------------------------------
# Board image generation tests
# ---------------------------------------------------------------------------

class TestGenerateBoardImage:
    def test_returns_image(self):
        shot = {"team1_stone1_x": "0.1", "team1_stone1_y": "0.2"}
        img = generate_board_image(shot)
        assert isinstance(img, Image.Image)
        assert img.size == (IMAGE_WIDTH, IMAGE_HEIGHT)
        assert img.mode == "RGB"

    def test_custom_dimensions(self):
        shot = {}
        img = generate_board_image(shot, image_width=200, image_height=250)
        assert img.size == (200, 250)

    def test_empty_board(self):
        img = generate_board_image({})
        assert isinstance(img, Image.Image)

    def test_board_with_stones(self):
        shot = {
            "team1_stone1_x": "0.0", "team1_stone1_y": "0.0",
            "team2_stone1_x": "0.5", "team2_stone1_y": "1.0",
        }
        img = generate_board_image(shot)
        assert isinstance(img, Image.Image)

    def test_stones_outside_house(self):
        shot = {
            "team1_stone1_x": "0.0", "team1_stone1_y": "2.0",
            "team2_stone1_x": "-0.8", "team2_stone1_y": "1.8",
        }
        img = generate_board_image(shot)
        assert isinstance(img, Image.Image)

    def test_full_house(self):
        """Test with maximum stones for both teams."""
        shot = {}
        for i in range(1, MAX_STONES_PER_TEAM + 1):
            shot[f"team1_stone{i}_x"] = str(round(0.1 * i, 3))
            shot[f"team1_stone{i}_y"] = str(round(0.1 * i, 3))
            shot[f"team2_stone{i}_x"] = str(round(-0.1 * i, 3))
            shot[f"team2_stone{i}_y"] = str(round(0.1 * i, 3))
        img = generate_board_image(shot)
        assert isinstance(img, Image.Image)

    def test_house_rings_are_drawn(self):
        """The centre of the board should not be plain white."""
        img = generate_board_image({})
        hx, hy = _norm_to_pixel(0, 0)
        # Sample a pixel on the 12-foot ring (to the right of centre)
        ring_x = int(hx + _norm_radius_to_pixels(0.85))
        ring_y = int(hy)
        if 0 <= ring_x < img.width and 0 <= ring_y < img.height:
            pixel = img.getpixel((ring_x, ring_y))
            # Should be the blue ring colour, not white
            assert pixel != (255, 255, 255)


# ---------------------------------------------------------------------------
# CSV lookup tests
# ---------------------------------------------------------------------------

class TestGenerateBoardImageFromCsv:
    def test_valid_lookup(self, tmp_path):
        csv_path = str(tmp_path / "shots.csv")
        fieldnames = [
            "event_id", "match_id", "end_number", "shot_number",
            "team_code", "player_id", "player_name",
            "shot_type", "turn", "accuracy",
            "team1_stones_in_play", "team2_stones_in_play",
        ]
        for ti in (1, 2):
            for si in range(1, MAX_STONES_PER_TEAM + 1):
                prefix = f"team{ti}_stone{si}"
                fieldnames += [f"{prefix}_x", f"{prefix}_y",
                               f"{prefix}_dist", f"{prefix}_angle"]

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            row = {fn: "" for fn in fieldnames}
            row.update({
                "event_id": "1", "match_id": "2",
                "end_number": "3", "shot_number": "4",
                "team1_stone1_x": "0.1", "team1_stone1_y": "0.2",
                "team2_stone1_x": "-0.3", "team2_stone1_y": "0.5",
            })
            writer.writerow(row)

        img = generate_board_image_from_csv(csv_path, 1, 2, 3, 4)
        assert isinstance(img, Image.Image)

    def test_missing_row_raises(self, tmp_path):
        csv_path = str(tmp_path / "shots.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["event_id", "match_id", "end_number",
                               "shot_number"],
            )
            writer.writeheader()
            writer.writerow({
                "event_id": "1", "match_id": "1",
                "end_number": "1", "shot_number": "1",
            })

        with pytest.raises(ValueError, match="No shot found"):
            generate_board_image_from_csv(csv_path, 9, 9, 9, 9)

    def test_with_actual_csv(self):
        csv_path = os.path.join("output", "shot_locations.csv")
        if not os.path.isfile(csv_path):
            pytest.skip("shot_locations.csv not present")
        img = generate_board_image_from_csv(csv_path, 1, 1, 1, 5)
        assert isinstance(img, Image.Image)
        assert img.size == (IMAGE_WIDTH, IMAGE_HEIGHT)
