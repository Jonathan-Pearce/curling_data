"""Tests for build_features module — game-context enrichment."""

import numpy as np
import pandas as pd
import pytest

from ml.build_features import enrich_features


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _shots_df(rows):
    """Build a minimal shot-locations DataFrame from a list of dicts."""
    defaults = {
        "event_id": 1,
        "match_id": 1,
        "end_number": 1,
        "shot_number": 1,
        "team_code": "AAA",
        "player_id": "",
        "player_name": "",
    }
    return pd.DataFrame([{**defaults, **r} for r in rows])


def _ends_df(rows):
    """Build a minimal ends DataFrame from a list of dicts."""
    defaults = {
        "event_id": 1,
        "match_id": 1,
        "end_number": 1,
        "team1_code": "AAA",
        "team2_code": "BBB",
        "hammer_team_code": "AAA",
        "team1_score_before": 0,
        "team2_score_before": 0,
    }
    return pd.DataFrame([{**defaults, **r} for r in rows])


# ---------------------------------------------------------------------------
# Column presence
# ---------------------------------------------------------------------------

class TestEnrichFeaturesColumns:

    def test_adds_all_expected_columns(self):
        shots = _shots_df([{}])
        ends = _ends_df([{}])
        result = enrich_features(shots, ends)
        for col in (
            "shooting_team_is_team1",
            "hammer_team_code",
            "shooting_team_has_hammer",
            "team1_score_before",
            "team2_score_before",
            "score_diff_before",
        ):
            assert col in result.columns, f"Missing column: {col}"

    def test_does_not_expose_team1_code_team2_code(self):
        """Helper join columns team1_code/team2_code must be dropped."""
        shots = _shots_df([{}])
        ends = _ends_df([{}])
        result = enrich_features(shots, ends)
        assert "team1_code" not in result.columns
        assert "team2_code" not in result.columns

    def test_original_shot_columns_preserved(self):
        shots = _shots_df([{"shot_number": 3, "player_id": "P1"}])
        ends = _ends_df([{}])
        result = enrich_features(shots, ends)
        assert result.loc[0, "shot_number"] == 3
        assert result.loc[0, "player_id"] == "P1"

    def test_row_count_unchanged(self):
        shots = _shots_df([{}, {"shot_number": 2}, {"shot_number": 3}])
        ends = _ends_df([{}])
        result = enrich_features(shots, ends)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# shooting_team_is_team1
# ---------------------------------------------------------------------------

class TestShootingTeamIsTeam1:

    def test_true_when_team_code_matches_team1(self):
        shots = _shots_df([{"team_code": "RED"}])
        ends = _ends_df([{"team1_code": "RED", "team2_code": "YEL"}])
        result = enrich_features(shots, ends)
        assert result.loc[0, "shooting_team_is_team1"] == True

    def test_false_when_team_code_matches_team2(self):
        shots = _shots_df([{"team_code": "YEL"}])
        ends = _ends_df([{"team1_code": "RED", "team2_code": "YEL"}])
        result = enrich_features(shots, ends)
        assert result.loc[0, "shooting_team_is_team1"] == False

    def test_both_values_in_same_end(self):
        shots = _shots_df([
            {"shot_number": 1, "team_code": "RED"},
            {"shot_number": 2, "team_code": "YEL"},
        ])
        ends = _ends_df([{"team1_code": "RED", "team2_code": "YEL"}])
        result = enrich_features(shots, ends)
        assert result.loc[0, "shooting_team_is_team1"] == True
        assert result.loc[1, "shooting_team_is_team1"] == False


# ---------------------------------------------------------------------------
# shooting_team_has_hammer
# ---------------------------------------------------------------------------

class TestShootingTeamHasHammer:

    def test_true_when_shooting_team_has_hammer(self):
        shots = _shots_df([{"team_code": "AAA"}])
        ends = _ends_df([{"hammer_team_code": "AAA"}])
        result = enrich_features(shots, ends)
        assert result.loc[0, "shooting_team_has_hammer"] == True

    def test_false_when_opponent_has_hammer(self):
        shots = _shots_df([{"team_code": "AAA"}])
        ends = _ends_df([{"hammer_team_code": "BBB"}])
        result = enrich_features(shots, ends)
        assert result.loc[0, "shooting_team_has_hammer"] == False

    def test_hammer_team_code_column_preserved(self):
        shots = _shots_df([{}])
        ends = _ends_df([{"hammer_team_code": "AAA"}])
        result = enrich_features(shots, ends)
        assert result.loc[0, "hammer_team_code"] == "AAA"


# ---------------------------------------------------------------------------
# score columns
# ---------------------------------------------------------------------------

class TestScoreColumns:

    def test_score_diff_before_positive(self):
        shots = _shots_df([{}])
        ends = _ends_df([{"team1_score_before": 3, "team2_score_before": 1}])
        result = enrich_features(shots, ends)
        assert result.loc[0, "score_diff_before"] == 2

    def test_score_diff_before_negative(self):
        shots = _shots_df([{}])
        ends = _ends_df([{"team1_score_before": 1, "team2_score_before": 4}])
        result = enrich_features(shots, ends)
        assert result.loc[0, "score_diff_before"] == -3

    def test_score_diff_before_zero(self):
        shots = _shots_df([{}])
        ends = _ends_df([{"team1_score_before": 0, "team2_score_before": 0}])
        result = enrich_features(shots, ends)
        assert result.loc[0, "score_diff_before"] == 0

    def test_raw_scores_present(self):
        shots = _shots_df([{}])
        ends = _ends_df([{"team1_score_before": 5, "team2_score_before": 3}])
        result = enrich_features(shots, ends)
        assert result.loc[0, "team1_score_before"] == 5
        assert result.loc[0, "team2_score_before"] == 3


# ---------------------------------------------------------------------------
# Join behaviour
# ---------------------------------------------------------------------------

class TestJoinBehaviour:

    def test_multi_end_join(self):
        """Shots from different ends get context from the correct end row."""
        shots = _shots_df([
            {"end_number": 1, "team_code": "AAA"},
            {"end_number": 2, "team_code": "BBB"},
        ])
        ends = _ends_df([
            {"end_number": 1, "team1_code": "AAA", "team2_code": "BBB",
             "hammer_team_code": "AAA", "team1_score_before": 0, "team2_score_before": 0},
            {"end_number": 2, "team1_code": "AAA", "team2_code": "BBB",
             "hammer_team_code": "BBB", "team1_score_before": 2, "team2_score_before": 0},
        ])
        result = enrich_features(shots, ends)
        assert result.loc[0, "shooting_team_has_hammer"] == True
        assert result.loc[1, "shooting_team_has_hammer"] == False
        assert result.loc[1, "team1_score_before"] == 2

    def test_unmatched_end_produces_nan(self):
        """Shot with no matching end row gets NaN for context columns."""
        shots = _shots_df([{"end_number": 99}])
        ends = _ends_df([{"end_number": 1}])
        result = enrich_features(shots, ends)
        assert pd.isna(result.loc[0, "hammer_team_code"])
        assert pd.isna(result.loc[0, "team1_score_before"])

    def test_does_not_duplicate_rows(self):
        """A single ends row matching multiple shot rows must not inflate row count."""
        shots = _shots_df([
            {"shot_number": 1},
            {"shot_number": 2},
            {"shot_number": 3},
        ])
        ends = _ends_df([{}])
        result = enrich_features(shots, ends)
        assert len(result) == 3

    def test_multi_match_multi_end(self):
        """Distinct (event_id, match_id, end_number) keys joined correctly."""
        shots = _shots_df([
            {"event_id": 1, "match_id": 1, "end_number": 1, "team_code": "A"},
            {"event_id": 1, "match_id": 2, "end_number": 1, "team_code": "C"},
        ])
        ends = _ends_df([
            {"event_id": 1, "match_id": 1, "end_number": 1,
             "team1_code": "A", "team2_code": "B",
             "hammer_team_code": "A", "team1_score_before": 0, "team2_score_before": 0},
            {"event_id": 1, "match_id": 2, "end_number": 1,
             "team1_code": "C", "team2_code": "D",
             "hammer_team_code": "D", "team1_score_before": 1, "team2_score_before": 2},
        ])
        result = enrich_features(shots, ends)
        assert result.loc[0, "shooting_team_is_team1"] == True
        assert result.loc[1, "shooting_team_is_team1"] == True
        assert result.loc[0, "shooting_team_has_hammer"] == True
        assert result.loc[1, "shooting_team_has_hammer"] == False
        assert result.loc[1, "score_diff_before"] == -1
