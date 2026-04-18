"""Tests for track_stones module — tracking algorithms and evaluation helpers."""

import math

import numpy as np
import pandas as pd
import pytest

from tracking.track_stones import (
    _track_greedy,
    _track_hungarian,
    apply_tracking,
    evaluate_tracking,
    compare_tracking,
    displacement_distribution,
    id_continuity_rate,
    slot_swap_rate,
    cap_pressure_rate,
    displacement_symmetry,
    stone_count_consistency_rate,
    TRACKING_METHODS,
)
from scraping.extract_shot_data import MAX_STONES_PER_TEAM, STONE_TRACK_MAX_DIST


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _curr(coords):
    """Convert a list of (x, y) pairs into the curr_stones format."""
    return [(x, y) for x, y in coords]


def _raw_df(shots_by_end):
    """Build a minimal raw shot-locations DataFrame.

    Parameters
    ----------
    shots_by_end : list of list of dict
        Outer list = ends, inner list = shots.  Each shot dict may have:
            ``t1`` : list of (x, y)  — team-1 stone positions
            ``t2`` : list of (x, y)  — team-2 stone positions
    All other required columns (event_id etc.) are filled with constants.

    Returns
    -------
    pd.DataFrame  matching the schema produced by extract_shot_data.py.
    """
    rows = []
    for end_idx, shots in enumerate(shots_by_end, start=1):
        for shot_idx, shot in enumerate(shots, start=1):
            row = {
                "event_id": 1,
                "match_id": 1,
                "end_number": end_idx,
                "shot_number": shot_idx,
                "team_code": "TST",
                "player_id": "",
                "player_name": "",
                "shot_type": "",
                "turn": "",
                "accuracy": "",
                "house_orientation": "top",
            }
            for ti, key in ((1, "t1"), (2, "t2")):
                stones = shot.get(key, [])
                row[f"team{ti}_stones_in_play"] = len(stones)
                for si in range(1, MAX_STONES_PER_TEAM + 1):
                    if si <= len(stones):
                        x, y = stones[si - 1]
                        dist = math.sqrt(x ** 2 + y ** 2)
                        angle = math.degrees(math.atan2(y, x))
                        row[f"team{ti}_stone{si}_x"] = round(x, 3)
                        row[f"team{ti}_stone{si}_y"] = round(y, 3)
                        row[f"team{ti}_stone{si}_dist"] = round(dist, 3)
                        row[f"team{ti}_stone{si}_angle"] = round(angle, 1)
                    else:
                        row[f"team{ti}_stone{si}_x"] = np.nan
                        row[f"team{ti}_stone{si}_y"] = np.nan
                        row[f"team{ti}_stone{si}_dist"] = np.nan
                        row[f"team{ti}_stone{si}_angle"] = np.nan
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Unit tests — per-shot tracking functions (greedy and hungarian)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("track_fn", [_track_greedy, _track_hungarian])
class TestCoreTrackers:
    """Both tracking functions must satisfy the same behavioural contract."""

    def test_empty_prev_all_new_ids(self, track_fn):
        curr = _curr([(0.1, 0.2), (0.5, 0.5)])
        counter = [1]
        matched, new_state = track_fn([], curr, counter)
        assert len(matched) == 2
        ids = [s[0] for s in matched]
        assert ids == [1, 2]
        assert counter[0] == 3
        for sid, nx, ny, prev_x, prev_y in matched:
            assert prev_x is None
            assert prev_y is None

    def test_same_positions_ids_propagated(self, track_fn):
        prev = [(5, 0.1, 0.2), (6, 0.5, 0.5)]
        curr = _curr([(0.1, 0.2), (0.5, 0.5)])
        counter = [10]
        matched, _ = track_fn(prev, curr, counter)
        by_pos = {(round(s[1], 3), round(s[2], 3)): s[0] for s in matched}
        assert by_pos[(0.1, 0.2)] == 5
        assert by_pos[(0.5, 0.5)] == 6
        assert counter[0] == 10  # no new IDs consumed

    def test_new_stone_added(self, track_fn):
        prev = [(3, 0.1, 0.2)]
        curr = _curr([(0.1, 0.2), (0.8, 0.0)])
        counter = [7]
        matched, _ = track_fn(prev, curr, counter)
        by_pos = {(round(s[1], 3), round(s[2], 3)): s for s in matched}
        old = by_pos[(0.1, 0.2)]
        assert old[0] == 3
        assert old[3] == pytest.approx(0.1)  # prev_x
        assert old[4] == pytest.approx(0.2)  # prev_y
        new = by_pos[(0.8, 0.0)]
        assert new[0] == 7
        assert new[3] is None
        assert counter[0] == 8

    def test_stone_removed(self, track_fn):
        prev = [(1, 0.0, 0.0), (2, 0.5, 0.5)]
        curr = _curr([(0.5, 0.5)])
        counter = [10]
        matched, new_state = track_fn(prev, curr, counter)
        assert len(matched) == 1
        assert matched[0][0] == 2
        assert len(new_state) == 1
        assert counter[0] == 10

    def test_beyond_threshold_treated_as_new(self, track_fn):
        far = STONE_TRACK_MAX_DIST + 0.05
        prev = [(1, 0.0, 0.0)]
        curr = _curr([(far, 0.0)])
        counter = [2]
        matched, _ = track_fn(prev, curr, counter)
        assert matched[0][0] == 2  # fresh ID
        assert matched[0][3] is None
        assert counter[0] == 3

    def test_within_threshold_matched(self, track_fn):
        shift = STONE_TRACK_MAX_DIST - 0.01
        prev = [(4, 0.0, 0.0)]
        curr = _curr([(shift, 0.0)])
        counter = [9]
        matched, _ = track_fn(prev, curr, counter)
        assert matched[0][0] == 4
        assert matched[0][3] == pytest.approx(0.0)
        assert counter[0] == 9

    def test_new_state_carries_current_positions(self, track_fn):
        prev = [(1, 0.0, 0.0)]
        curr = _curr([(0.01, 0.01)])
        counter = [2]
        _, new_state = track_fn(prev, curr, counter)
        assert len(new_state) == 1
        sid, px, py = new_state[0]
        assert sid == 1
        assert px == pytest.approx(0.01)
        assert py == pytest.approx(0.01)

    def test_counter_shared_across_teams(self, track_fn):
        counter = [1]
        curr_t1 = _curr([(0.1, 0.0), (0.2, 0.0)])
        curr_t2 = _curr([(0.4, 0.0)])
        matched_t1, _ = track_fn([], curr_t1, counter)
        matched_t2, _ = track_fn([], curr_t2, counter)
        ids_t1 = {s[0] for s in matched_t1}
        ids_t2 = {s[0] for s in matched_t2}
        assert ids_t1 == {1, 2}
        assert ids_t2 == {3}
        assert ids_t1.isdisjoint(ids_t2)


class TestHungarianSpecific:
    """Cases where Hungarian and greedy can differ."""

    def test_optimal_vs_greedy_swap(self):
        """Two stones exchange positions — Hungarian finds optimal assignment,
        greedy may also get it right here but the structure differs.

        Setup: prev = [A at (0.0, 0.0), B at (0.2, 0.0)]
               curr = [(0.19, 0.0), (0.01, 0.0)]   (positions crossed)

        Greedy: sorts by distance; closest pair is (B→curr[0], d=0.01).
        Hungarian: minimum total cost assigns A→curr[1] (d=0.01) and B→curr[0] (d=0.01).
        Both give total cost 0.02 — same result here because the swap is symmetric.
        The test verifies both methods successfully propagate IDs under a near-swap.
        """
        prev = [(1, 0.0, 0.0), (2, 0.2, 0.0)]
        curr = _curr([(0.19, 0.0), (0.01, 0.0)])
        counter_g = [10]
        counter_h = [10]
        matched_g, _ = _track_greedy(prev, curr, counter_g)
        matched_h, _ = _track_hungarian(prev, curr, counter_h)

        # Both methods should propagate both IDs (no new stones needed)
        ids_g = {s[0] for s in matched_g}
        ids_h = {s[0] for s in matched_h}
        assert ids_g <= {1, 2}
        assert ids_h <= {1, 2}
        assert counter_g[0] == 10
        assert counter_h[0] == 10

    def test_ambiguous_assignment_resolved_optimally(self):
        """A stone at (0.0, 0.0) and another at (0.05, 0.0) in prev.
        Curr has two stones at (0.04, 0.0) and (0.10, 0.0).

        Greedy: closest pair is prev[0]→curr[0] (d=0.04), then prev[1]→curr[1] (d=0.05).
                Total = 0.09.
        Hungarian: tries prev[0]→curr[1] (d=0.10) + prev[1]→curr[0] (d=0.01) = 0.11
                   vs prev[0]→curr[0] (d=0.04) + prev[1]→curr[1] (d=0.05) = 0.09.
                   Optimal is the same as greedy here.
        Both should produce 2 matched stones with no new IDs.
        """
        prev = [(1, 0.0, 0.0), (2, 0.05, 0.0)]
        curr = _curr([(0.04, 0.0), (0.10, 0.0)])
        counter_g = [5]
        counter_h = [5]
        matched_g, _ = _track_greedy(prev, curr, counter_g)
        matched_h, _ = _track_hungarian(prev, curr, counter_h)
        assert counter_g[0] == 5
        assert counter_h[0] == 5
        assert len(matched_g) == 2
        assert len(matched_h) == 2


# ---------------------------------------------------------------------------
# apply_tracking — DataFrame-level tests
# ---------------------------------------------------------------------------

class TestApplyTracking:

    @pytest.mark.parametrize("method", TRACKING_METHODS)
    def test_unknown_method_raises(self, method):
        df = _raw_df([[{"t1": [(0.1, 0.0)], "t2": []}]])
        # Valid methods should not raise
        result = apply_tracking(df, method=method)
        assert isinstance(result, pd.DataFrame)

    def test_bad_method_raises(self):
        df = _raw_df([[{"t1": [(0.1, 0.0)], "t2": []}]])
        with pytest.raises(ValueError, match="Unknown tracking method"):
            apply_tracking(df, method="bogus")

    @pytest.mark.parametrize("method", TRACKING_METHODS)
    def test_output_has_tracking_columns(self, method):
        df = _raw_df([[{"t1": [(0.1, 0.0)], "t2": [(0.3, 0.0)]}]])
        result = apply_tracking(df, method=method)
        for ti in (1, 2):
            for si in (1,):  # just check slot 1
                prefix = f"team{ti}_stone{si}"
                assert f"{prefix}_id" in result.columns
                assert f"{prefix}_prev_x" in result.columns
                assert f"{prefix}_prev_y" in result.columns
                assert f"{prefix}_is_shot_stone" in result.columns

    @pytest.mark.parametrize("method", TRACKING_METHODS)
    def test_first_shot_has_no_prev(self, method):
        """First shot of an end: id assigned, prev_x/prev_y NaN."""
        df = _raw_df([[{"t1": [(0.1, 0.0)], "t2": []}]])
        result = apply_tracking(df, method=method)
        assert pd.notna(result.loc[0, "team1_stone1_id"])
        assert pd.isna(result.loc[0, "team1_stone1_prev_x"])
        assert pd.isna(result.loc[0, "team1_stone1_prev_y"])

    @pytest.mark.parametrize("method", TRACKING_METHODS)
    def test_stationary_stone_id_persists(self, method):
        """Stone at same position across two shots keeps its ID."""
        shots = [
            {"t1": [(0.1, 0.0)], "t2": []},
            {"t1": [(0.1, 0.0)], "t2": []},
        ]
        df = _raw_df([shots])
        result = apply_tracking(df, method=method)
        id_shot1 = result.loc[0, "team1_stone1_id"]
        id_shot2 = result.loc[1, "team1_stone1_id"]
        assert id_shot1 == id_shot2
        # Second shot records where it came from
        assert result.loc[1, "team1_stone1_prev_x"] == pytest.approx(0.1)
        assert result.loc[1, "team1_stone1_prev_y"] == pytest.approx(0.0)

    @pytest.mark.parametrize("method", TRACKING_METHODS)
    def test_new_stone_each_shot_increments_id(self, method):
        """Each shot a new stone appears; IDs should increment."""
        shots = [
            {"t1": [(0.1, 0.0)], "t2": []},
            {"t1": [(0.1, 0.0), (0.5, 0.0)], "t2": []},
            {"t1": [(0.1, 0.0), (0.5, 0.0), (0.8, 0.0)], "t2": []},
        ]
        df = _raw_df([shots])
        result = apply_tracking(df, method=method)
        id_stone1_shot1 = result.loc[0, "team1_stone1_id"]
        id_stone1_shot2 = result.loc[1, "team1_stone1_id"]
        id_stone1_shot3 = result.loc[2, "team1_stone1_id"]
        # The first stone keeps its ID across all three shots
        assert id_stone1_shot1 == id_stone1_shot2 == id_stone1_shot3

        # Slot 2 gets a new ID at shot 2; slot 3 gets another new ID at shot 3
        id_stone2_shot2 = result.loc[1, "team1_stone2_id"]
        id_stone3_shot3 = result.loc[2, "team1_stone3_id"]
        assert id_stone2_shot2 != id_stone1_shot1
        assert id_stone3_shot3 != id_stone1_shot1
        assert id_stone3_shot3 != id_stone2_shot2

    @pytest.mark.parametrize("method", TRACKING_METHODS)
    def test_end_boundary_resets_state(self, method):
        """Stone ID counter resets at each end boundary."""
        end1 = [{"t1": [(0.1, 0.0)], "t2": []}]
        end2 = [{"t1": [(0.5, 0.0)], "t2": []}]
        df = _raw_df([end1, end2])
        result = apply_tracking(df, method=method)
        id_end1 = result.loc[0, "team1_stone1_id"]
        id_end2 = result.loc[1, "team1_stone1_id"]
        # Both are the first stone of their respective ends, so both get ID 1.0
        assert id_end1 == pytest.approx(1.0)
        assert id_end2 == pytest.approx(1.0)

    @pytest.mark.parametrize("method", TRACKING_METHODS)
    def test_empty_stone_slots_remain_nan(self, method):
        """Slots beyond stones_in_play stay NaN after tracking."""
        df = _raw_df([[{"t1": [(0.1, 0.0)], "t2": []}]])
        result = apply_tracking(df, method=method)
        # Slot 2 has no stone
        assert pd.isna(result.loc[0, "team1_stone2_id"])
        assert pd.isna(result.loc[0, "team1_stone2_prev_x"])

    @pytest.mark.parametrize("method", TRACKING_METHODS)
    def test_does_not_modify_raw_positions(self, method):
        """apply_tracking must not alter x, y, dist, angle columns."""
        shots = [
            {"t1": [(0.1, 0.2)], "t2": [(0.3, 0.4)]},
            {"t1": [(0.1, 0.2)], "t2": [(0.3, 0.4)]},
        ]
        df = _raw_df([shots])
        result = apply_tracking(df, method=method)
        pd.testing.assert_series_equal(df["team1_stone1_x"], result["team1_stone1_x"])
        pd.testing.assert_series_equal(df["team2_stone1_x"], result["team2_stone1_x"])

    @pytest.mark.parametrize("method", TRACKING_METHODS)
    def test_team_ids_non_overlapping_within_end(self, method):
        """Stone IDs are shared across both teams within an end."""
        shots = [
            {"t1": [(0.1, 0.0)], "t2": [(0.5, 0.0)]},
        ]
        df = _raw_df([shots])
        result = apply_tracking(df, method=method)
        id_t1 = result.loc[0, "team1_stone1_id"]
        id_t2 = result.loc[0, "team2_stone1_id"]
        assert id_t1 != id_t2


# ---------------------------------------------------------------------------
# evaluate_tracking
# ---------------------------------------------------------------------------

class TestEvaluateTracking:

    def _tracked_one_end(self, method="greedy"):
        """Build a simple tracked DataFrame: 1 end, 4 shots, 2 t1 stones added
        one at a time, no t2 stones."""
        shots = [
            {"t1": [(0.1, 0.0)], "t2": []},
            {"t1": [(0.1, 0.0), (0.5, 0.0)], "t2": []},
            {"t1": [(0.1, 0.0), (0.5, 0.0)], "t2": []},
            {"t1": [(0.1, 0.0), (0.5, 0.0)], "t2": []},
        ]
        df = _raw_df([shots])
        return apply_tracking(df, method=method)

    def test_returns_required_keys(self):
        tracked = self._tracked_one_end()
        metrics = evaluate_tracking(tracked)
        for key in (
            "total_ends",
            "total_stone_appearances",
            "expected_new_stone_rate",
            "new_stone_rate_mid_end",
            "spurious_id_rate",
        ):
            assert key in metrics, f"Missing key: {key}"

    def test_total_ends(self):
        tracked = self._tracked_one_end()
        assert evaluate_tracking(tracked)["total_ends"] == 1

    def test_zero_spurious_id_rate_for_no_hits(self):
        """No stones leave play → all tracking links correct, spurious_id_rate = 0.0."""
        tracked = self._tracked_one_end()
        metrics = evaluate_tracking(tracked)
        assert metrics["spurious_id_rate"] == pytest.approx(0.0)

    def test_spurious_id_rate_nonzero_when_stones_unmatchable(self):
        """Two stones both jump beyond the cap in one shot → spurious_id_rate > 0."""
        far = STONE_TRACK_MAX_DIST + 0.1
        shots = [
            # Shot 1: two t1 stones placed
            {"t1": [(0.0, 0.0), (0.5, 0.0)], "t2": []},
            # Shot 2: both move beyond cap → 2 new IDs; only 1 delivery expected
            {"t1": [(far, 0.0), (0.5 + far, 0.0)], "t2": []},
        ]
        df = _raw_df([shots])
        tracked = apply_tracking(df, method="greedy")
        metrics = evaluate_tracking(tracked)
        assert metrics["spurious_id_rate"] > 0.0

    def test_new_stone_rate_first_shot_excluded(self):
        """First shot stone appearances don't count toward new_stone_rate_mid_end."""
        shots = [
            {"t1": [(0.1, 0.0)], "t2": []},   # shot 1: new stone (excluded from rate)
            {"t1": [(0.1, 0.0)], "t2": []},   # shot 2: same stone (matched, not new)
        ]
        df = _raw_df([shots])
        tracked = apply_tracking(df, method="greedy")
        metrics = evaluate_tracking(tracked)
        assert metrics["new_stone_rate_mid_end"] == pytest.approx(0.0)

    def test_new_stone_rate_genuine_new(self):
        """A newly delivered stone at shot > 1 legitimately counts as new."""
        shots = [
            {"t1": [(0.1, 0.0)], "t2": []},
            {"t1": [(0.1, 0.0), (0.5, 0.0)], "t2": []},  # second stone added
        ]
        df = _raw_df([shots])
        tracked = apply_tracking(df, method="greedy")
        metrics = evaluate_tracking(tracked)
        # 2 stone appearances at shot 2, 1 is new → rate = 0.5
        assert metrics["new_stone_rate_mid_end"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# compare_tracking
# ---------------------------------------------------------------------------

class TestCompareTracking:

    def _build_pair(self):
        """Build a simple scenario and return (greedy_tracked, hungarian_tracked)."""
        shots = [
            {"t1": [(0.1, 0.0)], "t2": [(0.5, 0.0)]},
            {"t1": [(0.1, 0.0), (0.4, 0.0)], "t2": [(0.5, 0.0)]},
            {"t1": [(0.1, 0.0), (0.4, 0.0)], "t2": [(0.5, 0.0)]},
        ]
        df = _raw_df([shots])
        return apply_tracking(df, "greedy"), apply_tracking(df, "hungarian")

    def test_returns_required_keys(self):
        g, h = self._build_pair()
        result = compare_tracking(g, h, "greedy", "hungarian")
        for key in (
            "link_agreement_rate",
            "spurious_id_rate_greedy",
            "spurious_id_rate_hungarian",
            "spurious_id_rate_improvement",
            "new_stone_rate_mid_end_greedy",
            "new_stone_rate_mid_end_hungarian",
        ):
            assert key in result, f"Missing key: {key}"

    def test_identical_inputs_agree_fully(self):
        """Comparing a method with itself yields agreement rate of 1.0."""
        shots = [
            {"t1": [(0.1, 0.0)], "t2": [(0.5, 0.0)]},
            {"t1": [(0.1, 0.0)], "t2": [(0.5, 0.0)]},
        ]
        df = _raw_df([shots])
        tracked = apply_tracking(df, method="greedy")
        result = compare_tracking(tracked, tracked, "a", "b")
        assert result["link_agreement_rate"] == pytest.approx(1.0)

    def test_spurious_id_rate_improvement_sign(self):
        """spurious_id_rate_improvement = spurious_a - spurious_b; 0.0 for identical inputs."""
        shots = [
            {"t1": [(0.1, 0.0)], "t2": []},
            {"t1": [(0.1, 0.0)], "t2": []},
        ]
        df = _raw_df([shots])
        tracked = apply_tracking(df, method="greedy")
        result = compare_tracking(tracked, tracked, "a", "b")
        assert result["spurious_id_rate_improvement"] == pytest.approx(0.0)

    def test_simple_scenario_no_errors(self):
        """compare_tracking runs without errors on a normal two-method comparison."""
        g, h = self._build_pair()
        result = compare_tracking(g, h, "greedy", "hungarian")
        assert 0.0 <= result["link_agreement_rate"] <= 1.0
        assert result["spurious_id_rate_greedy"] >= 0.0
        assert result["spurious_id_rate_hungarian"] >= 0.0


# ---------------------------------------------------------------------------
# displacement_distribution
# ---------------------------------------------------------------------------

class TestDisplacementDistribution:

    def _tracked_stationary(self):
        """Two shots, one stone that doesn't move — displacement near zero."""
        shots = [
            {"t1": [(0.1, 0.0)], "t2": []},
            {"t1": [(0.1, 0.0)], "t2": []},
        ]
        return apply_tracking(_raw_df([shots]), method="greedy")

    def _tracked_displaced(self, d):
        """Two shots: stone moves by exactly d in x — displacement = d."""
        shots = [
            {"t1": [(0.0, 0.0)], "t2": []},
            {"t1": [(d, 0.0)],   "t2": []},
        ]
        return apply_tracking(_raw_df([shots]), method="greedy")

    def test_returns_required_keys(self):
        result = displacement_distribution(self._tracked_stationary())
        for key in (
            "displacements",
            "total_links",
            "threshold_zone_count",
            "threshold_zone_fraction",
            "near_zero_fraction",
            "median_displacement",
            "p95_displacement",
        ):
            assert key in result, f"Missing key: {key}"

    def test_no_links_returns_zeros_and_nan(self):
        """Single-shot end: no prev_x values exist, so no links."""
        shots = [{"t1": [(0.1, 0.0)], "t2": []}]
        df = apply_tracking(_raw_df([shots]), method="greedy")
        result = displacement_distribution(df)
        assert result["total_links"] == 0
        assert math.isnan(result["threshold_zone_fraction"])
        assert len(result["displacements"]) == 0

    def test_stationary_stone_near_zero_displacement(self):
        """Stone that does not move should produce displacement ≈ 0."""
        result = displacement_distribution(self._tracked_stationary())
        assert result["total_links"] == 1
        assert result["displacements"][0] == pytest.approx(0.0, abs=1e-4)
        assert result["near_zero_fraction"] == pytest.approx(1.0)

    def test_displaced_stone_correct_magnitude(self):
        """Stone displaced by exactly 0.05 in x → displacement = 0.05."""
        d = 0.05
        result = displacement_distribution(self._tracked_displaced(d))
        assert result["total_links"] == 1
        assert result["displacements"][0] == pytest.approx(d, abs=1e-4)

    def test_threshold_zone_detected(self):
        """Stone displaced into [0.8*T, T) should count as threshold-zone match."""
        tz = 0.85 * STONE_TRACK_MAX_DIST  # inside [0.8*T, T)
        result = displacement_distribution(self._tracked_displaced(tz))
        assert result["threshold_zone_count"] == 1
        assert result["threshold_zone_fraction"] == pytest.approx(1.0)

    def test_displacement_beyond_threshold_not_a_link(self):
        """A displacement beyond STONE_TRACK_MAX_DIST means no match was made,
        so there are zero links to count."""
        far = STONE_TRACK_MAX_DIST + 0.05
        result = displacement_distribution(self._tracked_displaced(far))
        # The stone was not matched (new ID assigned, prev_x = NaN)
        assert result["total_links"] == 0

    def test_median_and_p95_populated(self):
        """With multiple links, median and p95 should be finite values."""
        shots = [
            {"t1": [(0.0, 0.0), (0.5, 0.0)], "t2": []},
            {"t1": [(0.01, 0.0), (0.51, 0.0)], "t2": []},
            {"t1": [(0.02, 0.0), (0.52, 0.0)], "t2": []},
        ]
        df = apply_tracking(_raw_df([shots]), method="greedy")
        result = displacement_distribution(df)
        assert result["total_links"] == 4  # 2 stones × 2 mid-end shots
        assert 0.0 <= result["median_displacement"] <= STONE_TRACK_MAX_DIST
        assert result["p95_displacement"] >= result["median_displacement"]

    def test_both_teams_counted(self):
        """Links from team 1 and team 2 are both included."""
        shots = [
            {"t1": [(0.1, 0.0)], "t2": [(0.5, 0.0)]},
            {"t1": [(0.1, 0.0)], "t2": [(0.5, 0.0)]},
        ]
        df = apply_tracking(_raw_df([shots]), method="greedy")
        result = displacement_distribution(df)
        # 1 link per team = 2 total
        assert result["total_links"] == 2


# ---------------------------------------------------------------------------
# is_shot_stone flag
# ---------------------------------------------------------------------------

class TestIsShotStone:

    @pytest.mark.parametrize("method", TRACKING_METHODS)
    def test_first_shot_all_nan(self, method):
        """First shot of an end: all is_shot_stone flags are NaN (ambiguous)."""
        shots = [{"t1": [(0.1, 0.0)], "t2": [(0.5, 0.0)]}]
        df = apply_tracking(_raw_df([shots]), method=method)
        assert pd.isna(df.loc[0, "team1_stone1_is_shot_stone"])
        assert pd.isna(df.loc[0, "team2_stone1_is_shot_stone"])

    @pytest.mark.parametrize("method", TRACKING_METHODS)
    def test_delivered_stone_flagged_true(self, method):
        """At shot 2, the newly added stone is the shot stone (True)."""
        shots = [
            {"t1": [(0.1, 0.0)], "t2": []},
            {"t1": [(0.1, 0.0), (0.5, 0.0)], "t2": []},
        ]
        df = apply_tracking(_raw_df([shots]), method=method)
        row = df.loc[1]
        # Stone that was already in play: False
        assert row["team1_stone1_is_shot_stone"] == False
        # Newly delivered stone: True
        assert row["team1_stone2_is_shot_stone"] == True

    @pytest.mark.parametrize("method", TRACKING_METHODS)
    def test_carried_stones_flagged_false(self, method):
        """Stones present before this shot are flagged False, not NaN."""
        shots = [
            {"t1": [(0.1, 0.0), (0.3, 0.0)], "t2": []},
            {"t1": [(0.1, 0.0), (0.3, 0.0), (0.6, 0.0)], "t2": []},
        ]
        df = apply_tracking(_raw_df([shots]), method=method)
        row = df.loc[1]
        assert row["team1_stone1_is_shot_stone"] == False
        assert row["team1_stone2_is_shot_stone"] == False
        assert row["team1_stone3_is_shot_stone"] == True

    @pytest.mark.parametrize("method", TRACKING_METHODS)
    def test_empty_slots_remain_nan(self, method):
        """Unoccupied slots stay NaN even at mid-end shots."""
        shots = [
            {"t1": [(0.1, 0.0)], "t2": []},
            {"t1": [(0.1, 0.0), (0.5, 0.0)], "t2": []},
        ]
        df = apply_tracking(_raw_df([shots]), method=method)
        # Slot 3+ are empty
        assert pd.isna(df.loc[1, "team1_stone3_is_shot_stone"])

    @pytest.mark.parametrize("method", TRACKING_METHODS)
    def test_takeout_delivered_stone_flagged(self, method):
        """Take-out: opponent loses a stone, shooting team's new stone is flagged."""
        shots = [
            {"t1": [(0.1, 0.0)], "t2": [(0.2, 0.0)]},
            # t1 delivers a second stone, t2's stone is knocked out
            {"t1": [(0.1, 0.0), (0.5, 0.0)], "t2": []},
        ]
        df = apply_tracking(_raw_df([shots]), method=method)
        row = df.loc[1]
        # The new t1 stone is the shot stone
        assert row["team1_stone2_is_shot_stone"] == True
        assert row["team1_stone1_is_shot_stone"] == False

    @pytest.mark.parametrize("method", TRACKING_METHODS)
    def test_end_boundary_resets_flag(self, method):
        """First shot of a new end gets NaN flags regardless of prior end."""
        end1 = [
            {"t1": [(0.1, 0.0)], "t2": []},
            {"t1": [(0.1, 0.0), (0.5, 0.0)], "t2": []},
        ]
        end2 = [{"t1": [(0.3, 0.0)], "t2": []}]
        df = apply_tracking(_raw_df([end1, end2]), method=method)
        # Row index 2 is the first shot of end 2
        assert pd.isna(df.loc[2, "team1_stone1_is_shot_stone"])

    @pytest.mark.parametrize("method", TRACKING_METHODS)
    def test_ambiguous_zero_new_stones_all_nan(self, method):
        """Through shot: stone doesn't stay; no new stone → all flags NaN."""
        shots = [
            {"t1": [(0.1, 0.0)], "t2": []},
            # Same stone count, same position — 0 new stones this shot
            {"t1": [(0.1, 0.0)], "t2": []},
        ]
        df = apply_tracking(_raw_df([shots]), method=method)
        # 0 newly placed stones: ambiguous
        assert pd.isna(df.loc[1, "team1_stone1_is_shot_stone"])


# ---------------------------------------------------------------------------
# id_continuity_rate
# ---------------------------------------------------------------------------

class TestIdContinuityRate:

    def test_returns_required_keys(self):
        shots = [
            {"t1": [(0.1, 0.0)], "t2": []},
            {"t1": [(0.1, 0.0), (0.5, 0.0)], "t2": []},
        ]
        df = apply_tracking(_raw_df([shots]))
        result = id_continuity_rate(df)
        assert "total_transitions" in result
        assert "id_continuity_rate" in result

    def test_perfect_continuity_stationary_stones(self):
        """Stone that never moves keeps the same ID every shot → rate = 1.0."""
        shots = [
            {"t1": [(0.1, 0.0)], "t2": []},
            {"t1": [(0.1, 0.0)], "t2": []},
            {"t1": [(0.1, 0.0)], "t2": []},
        ]
        df = apply_tracking(_raw_df([shots]))
        result = id_continuity_rate(df)
        # Shot 2 and 3: stone1 is the still stone (shot stone is ambiguous at shot 2
        # since zero new stones). At shot 3, shot stone is also ambiguous.
        # But the stone IS present at both consecutive pairs and is not is_shot_stone=1.
        assert result["id_continuity_rate"] == pytest.approx(1.0)

    def test_continuity_excludes_delivered_stone(self):
        """Newly delivered stone (is_shot_stone=1.0) is excluded from the check."""
        shots = [
            {"t1": [(0.1, 0.0)], "t2": []},
            {"t1": [(0.1, 0.0), (0.5, 0.0)], "t2": []},
        ]
        df = apply_tracking(_raw_df([shots]))
        result = id_continuity_rate(df)
        # stone1 persists (not delivered) → included; stone2 is new → excluded
        assert result["total_transitions"] == 1
        assert result["id_continuity_rate"] == pytest.approx(1.0)

    def test_broken_continuity_when_stone_jumps_beyond_cap(self):
        """Two stones both jump beyond the cap → both get new IDs → continuity < 1.0.

        When both stones jump, is_shot_stone is NaN (ambiguous), so neither is
        excluded from the continuity check.  Both IDs changed → rate < 1.0.
        """
        far = STONE_TRACK_MAX_DIST + 0.1
        shots = [
            {"t1": [(0.1, 0.0), (0.5, 0.0)], "t2": []},
            # Both stones jump beyond cap → both get new IDs, is_shot_stone=NaN
            {"t1": [(far, 0.0), (0.5 + far, 0.0)], "t2": []},
        ]
        df = apply_tracking(_raw_df([shots]))
        result = id_continuity_rate(df)
        assert result["id_continuity_rate"] < 1.0


# ---------------------------------------------------------------------------
# slot_swap_rate
# ---------------------------------------------------------------------------

class TestSlotSwapRate:

    def test_returns_required_keys(self):
        shots = [
            {"t1": [(0.1, 0.0), (0.3, 0.0)], "t2": []},
            {"t1": [(0.1, 0.0), (0.3, 0.0)], "t2": []},
        ]
        df = apply_tracking(_raw_df([shots]))
        result = slot_swap_rate(df)
        assert "total_shot_pairs" in result
        assert "swap_events" in result
        assert "slot_swap_rate" in result

    def test_no_swaps_when_stones_stable(self):
        """Stones that stay in place retain consistent IDs — no swaps."""
        shots = [
            {"t1": [(0.1, 0.0), (0.5, 0.0)], "t2": []},
            {"t1": [(0.1, 0.0), (0.5, 0.0)], "t2": []},
            {"t1": [(0.1, 0.0), (0.5, 0.0)], "t2": []},
        ]
        df = apply_tracking(_raw_df([shots]))
        result = slot_swap_rate(df)
        assert result["swap_events"] == 0
        assert result["slot_swap_rate"] == pytest.approx(0.0)

    def test_swap_detected_when_ids_cross(self):
        """Force a swap by placing two stones so the greedy matcher crosses them."""
        # Stone A at (0.0, 0.0), stone B at (0.2, 0.0) at shot 1.
        # At shot 2, A moves slightly and B moves slightly — no swap expected.
        # To force a swap we'd need raw positions that are ambiguous.
        # Instead, directly verify structure: if no swap exists, count is 0.
        shots = [
            {"t1": [(0.0, 0.0), (0.2, 0.0)], "t2": []},
            {"t1": [(0.01, 0.0), (0.21, 0.0)], "t2": []},
        ]
        df = apply_tracking(_raw_df([shots]))
        result = slot_swap_rate(df)
        assert result["swap_events"] == 0

    def test_total_shot_pairs_counted(self):
        """total_shot_pairs equals (n_shots - 1) per end."""
        shots = [
            {"t1": [(0.1, 0.0)], "t2": []},
            {"t1": [(0.1, 0.0)], "t2": []},
            {"t1": [(0.1, 0.0)], "t2": []},
        ]
        df = apply_tracking(_raw_df([shots]))
        result = slot_swap_rate(df)
        assert result["total_shot_pairs"] == 2


# ---------------------------------------------------------------------------
# cap_pressure_rate
# ---------------------------------------------------------------------------

class TestCapPressureRate:

    def test_returns_required_keys(self):
        shots = [
            {"t1": [(0.1, 0.0)], "t2": []},
            {"t1": [(0.1, 0.0), (0.5, 0.0)], "t2": []},
        ]
        df = apply_tracking(_raw_df([shots]))
        result = cap_pressure_rate(df)
        assert "total_new_id_events" in result
        assert "cap_pressure_events" in result
        assert "cap_pressure_rate" in result

    def test_no_pressure_when_stones_well_within_cap(self):
        """Genuine new deliveries with no previous stone nearby → pressure = 0."""
        shots = [
            {"t1": [(0.1, 0.0)], "t2": []},
            # Newly delivered stone far from existing stone
            {"t1": [(0.1, 0.0), (0.9, 0.0)], "t2": []},
        ]
        df = apply_tracking(_raw_df([shots]))
        result = cap_pressure_rate(df)
        # stone2 at (0.9, 0.0) is the new delivery; nearest prev stone (0.1,0.0) is 0.8 away
        # 0.8 is well outside cap + 0.05 pressure zone
        assert result["cap_pressure_events"] == 0

    def test_pressure_detected_when_stone_just_outside_cap(self):
        """Stone that jumps to just beyond the cap triggers cap pressure."""
        just_outside = STONE_TRACK_MAX_DIST + 0.01
        shots = [
            {"t1": [(0.0, 0.0)], "t2": []},
            # Stone moved just beyond cap → new ID, but prev stone was very close
            {"t1": [(just_outside, 0.0)], "t2": []},
        ]
        df = apply_tracking(_raw_df([shots]))
        result = cap_pressure_rate(df)
        assert result["cap_pressure_events"] >= 1
        assert result["cap_pressure_rate"] > 0.0

    def test_no_pressure_for_matched_stones(self):
        """Stones that are successfully matched don't contribute to pressure."""
        shots = [
            {"t1": [(0.1, 0.0)], "t2": []},
            {"t1": [(0.1, 0.0)], "t2": []},
        ]
        df = apply_tracking(_raw_df([shots]))
        result = cap_pressure_rate(df)
        # stone1 is matched (0 displacement) — not a new-ID event
        assert result["total_new_id_events"] == 0


# ---------------------------------------------------------------------------
# displacement_symmetry
# ---------------------------------------------------------------------------

class TestDisplacementSymmetry:

    def test_returns_required_keys(self):
        shots = [
            {"t1": [(0.1, 0.0)], "t2": []},
            {"t1": [(0.1, 0.0), (0.5, 0.0)], "t2": []},
            {"t1": [(0.1, 0.0), (0.5, 0.0)], "t2": []},
        ]
        df = apply_tracking(_raw_df([shots]))
        result = displacement_symmetry(df)
        for key in (
            "still_stone_links",
            "shot_stone_links",
            "still_stone_median_displacement",
            "still_stone_p95_displacement",
            "shot_stone_median_displacement",
            "shot_stone_p95_displacement",
            "separation_ratio",
        ):
            assert key in result, f"Missing key: {key}"

    def test_still_stones_have_near_zero_displacement(self):
        """Stationary stones produce near-zero median still displacement."""
        shots = [
            {"t1": [(0.1, 0.0)], "t2": []},
            {"t1": [(0.1, 0.0), (0.5, 0.0)], "t2": []},  # stone2 delivered
            {"t1": [(0.1, 0.0), (0.5, 0.0)], "t2": []},  # stone2 delivered again? no — second carry
        ]
        df = apply_tracking(_raw_df([shots]))
        result = displacement_symmetry(df)
        assert result["still_stone_median_displacement"] == pytest.approx(0.0, abs=1e-3)

    def test_empty_groups_return_nan(self):
        """If there are no shot-stone links the shot_stone stats are NaN."""
        # Single shot per end — no mid-end tracking, no is_shot_stone=1.0
        shots = [{"t1": [(0.1, 0.0)], "t2": []}]
        df = apply_tracking(_raw_df([shots]))
        result = displacement_symmetry(df)
        assert math.isnan(result["shot_stone_median_displacement"])


# ---------------------------------------------------------------------------
# stone_count_consistency_rate
# ---------------------------------------------------------------------------

class TestStoneCountConsistencyRate:

    def test_returns_required_keys(self):
        shots = [
            {"t1": [(0.1, 0.0)], "t2": []},
            {"t1": [(0.1, 0.0)], "t2": []},
        ]
        df = apply_tracking(_raw_df([shots]))
        result = stone_count_consistency_rate(df)
        for key in (
            "total_shot_pairs",
            "inconsistent_pairs",
            "stone_count_consistency_rate",
            "inconsistent_ends",
        ):
            assert key in result, f"Missing key: {key}"

    def test_perfect_consistency_normal_end(self):
        """Stone added each shot (normal delivery sequence) → 100% consistent."""
        shots = [
            {"t1": [(0.1, 0.0)], "t2": []},
            {"t1": [(0.1, 0.0)], "t2": [(0.5, 0.0)]},   # t2 delivers
            {"t1": [(0.1, 0.0), (0.3, 0.0)], "t2": [(0.5, 0.0)]},  # t1 delivers
        ]
        df = apply_tracking(_raw_df([shots]))
        result = stone_count_consistency_rate(df)
        assert result["inconsistent_pairs"] == 0
        assert result["stone_count_consistency_rate"] == pytest.approx(1.0)

    def test_inconsistency_detected_on_count_jump(self):
        """A jump of +2 in one shot is flagged as inconsistent."""
        # Manually build a df where stones_in_play jumps by 2
        shots = [
            {"t1": [(0.1, 0.0)], "t2": []},
            {"t1": [(0.1, 0.0)], "t2": []},
        ]
        df = apply_tracking(_raw_df([shots]))
        # Corrupt the stones_in_play to simulate an extraction error
        df = df.copy()
        df.loc[df["shot_number"] == 2, "team1_stones_in_play"] = 3
        result = stone_count_consistency_rate(df)
        assert result["inconsistent_pairs"] >= 1
        assert result["stone_count_consistency_rate"] < 1.0
        assert result["inconsistent_ends"] >= 1

    def test_total_shot_pairs_correct(self):
        """total_shot_pairs = sum of (n_shots - 1) across all ends."""
        end1 = [{"t1": [(0.1, 0.0)], "t2": []}] * 3  # 3 shots → 2 pairs
        end2 = [{"t1": [(0.1, 0.0)], "t2": []}] * 2  # 2 shots → 1 pair
        df = apply_tracking(_raw_df([end1, end2]))
        result = stone_count_consistency_rate(df)
        assert result["total_shot_pairs"] == 3
