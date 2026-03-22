"""Tests for verify_stone_tracking module."""

import math
import os

import pandas as pd
import pytest

from verify_stone_tracking import (
    _check_columns_present,
    _check_first_shot_invariant,
    _check_matched_displacement,
    _check_id_uniqueness,
    _check_id_continuity,
    _check_new_stone_count,
    STONE_TRACK_MAX_DIST,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MAX_SLOTS = 8

def _base_stone_cols():
    """Return the full set of tracking column names for both teams, all slots."""
    cols = {}
    for team in (1, 2):
        for slot in range(1, MAX_SLOTS + 1):
            p = f"team{team}_stone{slot}"
            for field in ("x", "y", "dist", "angle", "id", "prev_x", "prev_y"):
                cols[f"{p}_{field}"] = None
    return cols


def _make_df(rows):
    """Build a DataFrame from a list of dicts, filling missing tracking cols with NaN."""
    template = _base_stone_cols()
    filled = []
    for row in rows:
        r = {**template, **row}
        filled.append(r)
    df = pd.DataFrame(filled)
    # Convert numeric tracking cols to float (matching parquet behaviour)
    for col in df.columns:
        if any(col.endswith(s) for s in ("_x", "_y", "_dist", "_angle", "_id",
                                          "_prev_x", "_prev_y")):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _identity_cols():
    """Return minimal identity columns for a shot row."""
    return {
        "event_id": 1,
        "match_id": 1,
        "end_number": 1,
        "shot_number": 1,
        "team_code": "TST",
        "team1_stones_in_play": 0,
        "team2_stones_in_play": 0,
    }


# ---------------------------------------------------------------------------
# TestCheckColumnsPresent
# ---------------------------------------------------------------------------

class TestCheckColumnsPresent:
    def test_all_present(self):
        df = _make_df([{**_identity_cols()}])
        errors = _check_columns_present(df)
        assert errors == []

    def test_missing_one_id_col(self):
        df = _make_df([{**_identity_cols()}])
        df = df.drop(columns=["team1_stone1_id"])
        errors = _check_columns_present(df)
        assert any("team1_stone1_id" in e for e in errors)

    def test_missing_prev_col(self):
        df = _make_df([{**_identity_cols()}])
        df = df.drop(columns=["team2_stone3_prev_x"])
        errors = _check_columns_present(df)
        assert any("team2_stone3_prev_x" in e for e in errors)


# ---------------------------------------------------------------------------
# TestCheckFirstShotInvariant
# ---------------------------------------------------------------------------

class TestCheckFirstShotInvariant:
    def test_shot1_all_null_passes(self):
        row = {**_identity_cols(), "shot_number": 1}
        # All prev_x / prev_y remain NaN (from template)
        df = _make_df([row])
        errors = _check_first_shot_invariant(df)
        assert errors == []

    def test_shot1_with_prev_x_fails(self):
        row = {
            **_identity_cols(),
            "shot_number": 1,
            "team1_stone1_x": 0.1,
            "team1_stone1_y": 0.2,
            "team1_stone1_id": 1,
            "team1_stone1_prev_x": 0.1,   # Should be NULL for shot 1
            "team1_stone1_prev_y": 0.2,
        }
        df = _make_df([row])
        errors = _check_first_shot_invariant(df)
        assert any("FAIL" in e for e in errors)

    def test_shot2_with_prev_x_is_fine(self):
        """Non-shot-1 rows with prev_x populated should not trigger check 2."""
        row = {
            **_identity_cols(),
            "shot_number": 2,
            "team1_stone1_x": 0.1,
            "team1_stone1_y": 0.2,
            "team1_stone1_id": 1,
            "team1_stone1_prev_x": 0.1,
            "team1_stone1_prev_y": 0.2,
        }
        df = _make_df([row])
        errors = _check_first_shot_invariant(df)
        assert errors == []


# ---------------------------------------------------------------------------
# TestCheckMatchedDisplacement
# ---------------------------------------------------------------------------

class TestCheckMatchedDisplacement:
    def test_zero_displacement_no_warning(self):
        row = {
            **_identity_cols(),
            "shot_number": 2,
            "team1_stone1_x": 0.1,
            "team1_stone1_y": 0.2,
            "team1_stone1_id": 1,
            "team1_stone1_prev_x": 0.1,
            "team1_stone1_prev_y": 0.2,
        }
        df = _make_df([row])
        errors, detail = _check_matched_displacement(df)
        assert errors == []
        assert not detail.empty
        assert detail["displacement"].iloc[0] == pytest.approx(0.0)

    def test_small_noise_displacement_no_warning(self):
        row = {
            **_identity_cols(),
            "shot_number": 2,
            "team1_stone1_x": 0.102,
            "team1_stone1_y": 0.201,
            "team1_stone1_id": 1,
            "team1_stone1_prev_x": 0.100,
            "team1_stone1_prev_y": 0.200,
        }
        df = _make_df([row])
        errors, detail = _check_matched_displacement(df)
        assert errors == []
        expected_disp = math.sqrt(0.002**2 + 0.001**2)
        assert detail["displacement"].iloc[0] == pytest.approx(expected_disp, abs=1e-6)

    def test_large_displacement_warns(self):
        """Displacement > NOISE_WARN_DIST should produce a warning."""
        row = {
            **_identity_cols(),
            "shot_number": 2,
            "team1_stone1_x": 0.3,
            "team1_stone1_y": 0.2,
            "team1_stone1_id": 1,
            "team1_stone1_prev_x": 0.1,   # displacement = 0.2 (above 0.05)
            "team1_stone1_prev_y": 0.2,
        }
        df = _make_df([row])
        errors, _ = _check_matched_displacement(df)
        assert any("WARN" in e for e in errors)

    def test_shot1_rows_excluded(self):
        """Shot-1 rows should not appear in the displacement detail (no prev)."""
        row = {
            **_identity_cols(),
            "shot_number": 1,
            "team1_stone1_x": 0.1,
            "team1_stone1_y": 0.2,
            "team1_stone1_id": 1,
            # prev_x / prev_y are NaN — omitted from _make_df
        }
        df = _make_df([row])
        _, detail = _check_matched_displacement(df)
        assert detail.empty


# ---------------------------------------------------------------------------
# TestCheckIdUniqueness
# ---------------------------------------------------------------------------

class TestCheckIdUniqueness:
    def test_non_overlapping_ids_pass(self):
        # team1 stone gets ID 1, team2 stone gets ID 2 — no overlap
        rows = [
            {**_identity_cols(), "shot_number": 1,
             "team1_stone1_id": 1.0, "team2_stone1_id": 2.0},
        ]
        df = _make_df(rows)
        errors = _check_id_uniqueness(df)
        assert errors == []

    def test_overlapping_ids_fail(self):
        # Both teams have stone_id == 5 in the same end
        rows = [
            {**_identity_cols(), "shot_number": 1,
             "team1_stone1_id": 5.0, "team2_stone1_id": 5.0},
        ]
        df = _make_df(rows)
        errors = _check_id_uniqueness(df)
        assert any("FAIL" in e for e in errors)

    def test_same_id_different_ends_passes(self):
        """IDs reset each end, so id=1 for team1 in end 1 and id=1 for team2
        in end 2 should not clash."""
        rows = [
            {"event_id": 1, "match_id": 1, "end_number": 1, "shot_number": 1,
             "team_code": "T", "team1_stones_in_play": 1, "team2_stones_in_play": 0,
             "team1_stone1_id": 1.0},
            {"event_id": 1, "match_id": 1, "end_number": 2, "shot_number": 1,
             "team_code": "T", "team1_stones_in_play": 0, "team2_stones_in_play": 1,
             "team2_stone1_id": 1.0},
        ]
        df = _make_df(rows)
        errors = _check_id_uniqueness(df)
        assert errors == []


# ---------------------------------------------------------------------------
# TestCheckIdContinuity
# ---------------------------------------------------------------------------

class TestCheckIdContinuity:
    def test_continuous_presence_no_warning(self):
        """Stone id=1 present at shots 1, 2, 3 — no gaps."""
        rows = [
            {**_identity_cols(), "shot_number": 1, "team1_stone1_id": 1.0},
            {**_identity_cols(), "shot_number": 2, "team1_stone1_id": 1.0},
            {**_identity_cols(), "shot_number": 3, "team1_stone1_id": 1.0},
        ]
        df = _make_df(rows)
        errors = _check_id_continuity(df)
        assert errors == []

    def test_gap_produces_warning(self):
        """Stone id=1 present at shots 1 and 3 but absent at 2 — should warn."""
        rows = [
            {**_identity_cols(), "shot_number": 1, "team1_stone1_id": 1.0},
            # shot 2 — stone absent
            {**_identity_cols(), "shot_number": 2},
            {**_identity_cols(), "shot_number": 3, "team1_stone1_id": 1.0},
        ]
        df = _make_df(rows)
        errors = _check_id_continuity(df)
        assert any("WARN" in e for e in errors)

    def test_stone_leaves_play_no_warning(self):
        """Stone id=1 present at shots 1 and 2 only (leaves play) — no gap warning."""
        rows = [
            {**_identity_cols(), "shot_number": 1, "team1_stone1_id": 1.0},
            {**_identity_cols(), "shot_number": 2, "team1_stone1_id": 1.0},
            {**_identity_cols(), "shot_number": 3},
        ]
        df = _make_df(rows)
        errors = _check_id_continuity(df)
        assert errors == []


# ---------------------------------------------------------------------------
# TestCheckNewStoneCount
# ---------------------------------------------------------------------------

class TestCheckNewStoneCount:
    def test_one_new_stone_per_team_passes(self):
        """One stone with NULL prev_x per team on a non-first shot is expected."""
        row = {
            **_identity_cols(),
            "shot_number": 2,
            # team1: one stone with prev (slot 1), one new stone (slot 2)
            "team1_stone1_x": 0.1, "team1_stone1_prev_x": 0.1,
            "team1_stone2_x": 0.5,  # prev_x remains NaN = newly placed
            # team2: one new stone (slot 1)
            "team2_stone1_x": 0.3,  # prev_x remains NaN = newly placed
        }
        df = _make_df([row])
        errors = _check_new_stone_count(df)
        assert errors == []

    def test_two_new_stones_same_team_warns(self):
        """Two stones with NULL prev_x for the same team on one shot should warn."""
        row = {
            **_identity_cols(),
            "shot_number": 2,
            "team1_stone1_x": 0.1,  # newly placed (NaN prev_x)
            "team1_stone2_x": 0.5,  # also newly placed (NaN prev_x)
        }
        df = _make_df([row])
        errors = _check_new_stone_count(df)
        assert any("WARN" in e for e in errors)

    def test_shot1_excluded_from_check(self):
        """Check 7 only applies to shot_number > 1; shot 1 always has all NULL prev."""
        row = {
            **_identity_cols(),
            "shot_number": 1,
            "team1_stone1_x": 0.1,
            "team1_stone2_x": 0.5,
            # Both newly placed, but this is shot 1 — should not warn
        }
        df = _make_df([row])
        errors = _check_new_stone_count(df)
        assert errors == []
