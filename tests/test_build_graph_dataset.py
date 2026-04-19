"""Tests for build_graph_dataset — graph construction for GNN training."""

import math
import os

import numpy as np
import pandas as pd
import pytest

from ml.build_graph_dataset import (
    N_EDGE_FEATURES,
    N_GRAPH_FEATURES,
    N_NODE_FEATURES,
    SHOT_TYPE_CLASSES,
    SHOT_TYPE_INDEX,
    SPATIAL_EDGE_DIST,
    assign_splits,
    build_all_graphs,
    build_board_state,
    build_graph_features,
    build_shot_graph,
    build_spatial_edges,
    build_temporal_edges,
    extract_labels,
    load_graphs,
    save_graphs,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _row(**kwargs) -> pd.Series:
    """Minimal shot row with sensible defaults."""
    defaults = {
        "event_id": 1,
        "match_id": 1,
        "end_number": 1,
        "shot_number": 1,
        "team_code": "AAA",
        "team1_stones_in_play": 0,
        "team2_stones_in_play": 0,
        "shot_type": "Draw",
        "accuracy": 75.0,
        "team1_score_this_end": 1.0,
        "team2_score_this_end": 0.0,
        "shooting_team_has_hammer": True,
        "shooting_team_is_team1": True,
        "score_diff_before": 0,
    }
    return pd.Series({**defaults, **kwargs})


def _add_stone(d: dict, team: int, slot: int, x: float, y: float,
               dist: float = 0.5, angle: float = 0.0,
               stone_id=None, prev_x=None, prev_y=None,
               is_shot_stone=False) -> dict:
    """Add stone columns to a row-dict in-place and return it."""
    prefix = f"team{team}_stone{slot}"
    d[f"{prefix}_x"] = x
    d[f"{prefix}_y"] = y
    d[f"{prefix}_dist"] = dist
    d[f"{prefix}_angle"] = angle
    d[f"{prefix}_id"] = stone_id
    d[f"{prefix}_prev_x"] = prev_x
    d[f"{prefix}_prev_y"] = prev_y
    d[f"{prefix}_is_shot_stone"] = is_shot_stone
    return d


def _shots_df(rows: list[dict]) -> pd.DataFrame:
    """Build a DataFrame from a list of row dicts with sensible defaults."""
    base = {
        "event_id": 1, "match_id": 1, "end_number": 1, "shot_number": 1,
        "team_code": "AAA",
        "team1_stones_in_play": 0, "team2_stones_in_play": 0,
        "shot_type": "Draw", "accuracy": 75.0,
        "team1_score_this_end": 1.0, "team2_score_this_end": 0.0,
        "shooting_team_has_hammer": True, "shooting_team_is_team1": True,
        "score_diff_before": 0,
    }
    return pd.DataFrame([{**base, **r} for r in rows])


# ---------------------------------------------------------------------------
# NODE_FEATURE_NAMES / constants
# ---------------------------------------------------------------------------

class TestConstants:

    def test_n_node_features_matches_list(self):
        from ml.build_graph_dataset import NODE_FEATURE_NAMES
        assert N_NODE_FEATURES == len(NODE_FEATURE_NAMES)

    def test_n_edge_features_matches_list(self):
        from ml.build_graph_dataset import EDGE_FEATURE_NAMES
        assert N_EDGE_FEATURES == len(EDGE_FEATURE_NAMES)

    def test_n_graph_features_matches_list(self):
        from ml.build_graph_dataset import GRAPH_FEATURE_NAMES
        assert N_GRAPH_FEATURES == len(GRAPH_FEATURE_NAMES)

    def test_shot_type_index_covers_all_classes(self):
        assert set(SHOT_TYPE_INDEX.keys()) == set(SHOT_TYPE_CLASSES)

    def test_shot_type_index_values_are_unique_and_contiguous(self):
        vals = sorted(SHOT_TYPE_INDEX.values())
        assert vals == list(range(len(SHOT_TYPE_CLASSES)))


# ---------------------------------------------------------------------------
# build_board_state
# ---------------------------------------------------------------------------

class TestBuildBoardState:

    def test_empty_board_returns_zero_rows(self):
        row = _row(team1_stones_in_play=0, team2_stones_in_play=0)
        x, ids = build_board_state(row)
        assert x.shape == (0, N_NODE_FEATURES)
        assert ids == []

    def test_single_stone_shape(self):
        d = {"team1_stones_in_play": 1, "team2_stones_in_play": 0}
        _add_stone(d, 1, 1, x=0.1, y=0.2, dist=0.22, angle=30.0, stone_id=1)
        x, ids = build_board_state(_row(**d))
        assert x.shape == (1, N_NODE_FEATURES)
        assert ids == [1]

    def test_two_stones_both_teams(self):
        d = {"team1_stones_in_play": 1, "team2_stones_in_play": 1}
        _add_stone(d, 1, 1, x=0.1, y=0.2, stone_id=1)
        _add_stone(d, 2, 1, x=-0.1, y=0.3, stone_id=2)
        x, ids = build_board_state(_row(**d))
        assert x.shape == (2, N_NODE_FEATURES)
        assert set(ids) == {1, 2}

    def test_stones_sorted_by_stone_id_not_slot_order(self):
        """Stone with higher slot index but lower stone_id must appear first."""
        d = {"team1_stones_in_play": 2, "team2_stones_in_play": 0}
        # slot 1 has stone_id=5, slot 2 has stone_id=2
        _add_stone(d, 1, 1, x=0.5, y=0.0, stone_id=5)
        _add_stone(d, 1, 2, x=0.2, y=0.0, stone_id=2)
        x, ids = build_board_state(_row(**d))
        assert ids == [2, 5], "Nodes must be ordered by stone_id"
        assert x[0, 0] == pytest.approx(0.2)  # stone_id 2 has x=0.2
        assert x[1, 0] == pytest.approx(0.5)  # stone_id 5 has x=0.5

    def test_team_feature_is_zero_for_team1(self):
        d = {"team1_stones_in_play": 1, "team2_stones_in_play": 0}
        _add_stone(d, 1, 1, x=0.0, y=0.0, stone_id=1)
        x, _ = build_board_state(_row(**d))
        team_col = 5  # index of "team" in NODE_FEATURE_NAMES
        assert x[0, team_col] == pytest.approx(0.0)

    def test_team_feature_is_one_for_team2(self):
        d = {"team1_stones_in_play": 0, "team2_stones_in_play": 1}
        _add_stone(d, 2, 1, x=0.0, y=0.0, stone_id=1)
        x, _ = build_board_state(_row(**d))
        team_col = 5
        assert x[0, team_col] == pytest.approx(1.0)

    def test_displacement_zero_when_prev_is_null(self):
        d = {"team1_stones_in_play": 1, "team2_stones_in_play": 0}
        _add_stone(d, 1, 1, x=0.3, y=0.4, stone_id=1, prev_x=None, prev_y=None)
        x, _ = build_board_state(_row(**d))
        dx_col, dy_col, is_new_col = 6, 7, 8
        assert x[0, dx_col] == pytest.approx(0.0)
        assert x[0, dy_col] == pytest.approx(0.0)
        assert x[0, is_new_col] == pytest.approx(1.0)

    def test_displacement_computed_when_prev_available(self):
        d = {"team1_stones_in_play": 1, "team2_stones_in_play": 0}
        _add_stone(d, 1, 1, x=0.3, y=0.4, stone_id=1, prev_x=0.1, prev_y=0.2)
        x, _ = build_board_state(_row(**d))
        dx_col, dy_col, is_new_col = 6, 7, 8
        assert x[0, dx_col] == pytest.approx(0.2)
        assert x[0, dy_col] == pytest.approx(0.2)
        assert x[0, is_new_col] == pytest.approx(0.0)

    def test_is_shot_stone_true_maps_to_one(self):
        d = {"team1_stones_in_play": 1, "team2_stones_in_play": 0}
        _add_stone(d, 1, 1, x=0.0, y=0.0, stone_id=1, is_shot_stone=True)
        x, _ = build_board_state(_row(**d))
        assert x[0, 9] == pytest.approx(1.0)

    def test_is_shot_stone_false_maps_to_zero(self):
        d = {"team1_stones_in_play": 1, "team2_stones_in_play": 0}
        _add_stone(d, 1, 1, x=0.0, y=0.0, stone_id=1, is_shot_stone=False)
        x, _ = build_board_state(_row(**d))
        assert x[0, 9] == pytest.approx(0.0)

    def test_is_shot_stone_missing_maps_to_half(self):
        """None / not present → ambiguous → 0.5."""
        d = {"team1_stones_in_play": 1, "team2_stones_in_play": 0}
        _add_stone(d, 1, 1, x=0.0, y=0.0, stone_id=1, is_shot_stone=None)
        x, _ = build_board_state(_row(**d))
        assert x[0, 9] == pytest.approx(0.5)

    def test_angle_sin_cos_encoding(self):
        """90° → sin=1, cos≈0."""
        d = {"team1_stones_in_play": 1, "team2_stones_in_play": 0}
        _add_stone(d, 1, 1, x=0.0, y=0.5, angle=90.0, stone_id=1)
        x, _ = build_board_state(_row(**d))
        sin_col, cos_col = 3, 4
        assert x[0, sin_col] == pytest.approx(math.sin(math.radians(90)), abs=1e-5)
        assert x[0, cos_col] == pytest.approx(math.cos(math.radians(90)), abs=1e-5)

    def test_dtype_is_float32(self):
        d = {"team1_stones_in_play": 1, "team2_stones_in_play": 0}
        _add_stone(d, 1, 1, x=0.1, y=0.1, stone_id=1)
        x, _ = build_board_state(_row(**d))
        assert x.dtype == np.float32

    def test_none_stone_id_sorts_last(self):
        """Stones without a tracked ID should appear after those with IDs."""
        d = {"team1_stones_in_play": 2, "team2_stones_in_play": 0}
        _add_stone(d, 1, 1, x=0.1, y=0.0, stone_id=None)
        _add_stone(d, 1, 2, x=0.2, y=0.0, stone_id=3)
        _, ids = build_board_state(_row(**d))
        assert ids[0] == 3
        assert ids[1] is None


# ---------------------------------------------------------------------------
# build_spatial_edges
# ---------------------------------------------------------------------------

class TestBuildSpatialEdges:

    def _make_nodes(self, positions: list[tuple[float, float]]) -> np.ndarray:
        """Make a minimal node feature matrix with the given (x, y) positions."""
        n = len(positions)
        features = np.zeros((n, N_NODE_FEATURES), dtype=np.float32)
        for i, (x, y) in enumerate(positions):
            features[i, 0] = x
            features[i, 1] = y
        return features

    def test_empty_board_returns_empty_edges(self):
        nodes = np.zeros((0, N_NODE_FEATURES), dtype=np.float32)
        ei, ea = build_spatial_edges(nodes)
        assert ei.shape == (2, 0)
        assert ea.shape == (0, N_EDGE_FEATURES)

    def test_single_stone_returns_empty_edges(self):
        nodes = self._make_nodes([(0.0, 0.0)])
        ei, ea = build_spatial_edges(nodes)
        assert ei.shape == (2, 0)

    def test_two_stones_within_threshold_connected_bidirectionally(self):
        nodes = self._make_nodes([(0.0, 0.0), (0.5, 0.0)])
        ei, ea = build_spatial_edges(nodes, dist_threshold=1.0)
        assert ei.shape[0] == 2
        assert ei.shape[1] == 2  # two directed edges (0→1 and 1→0)
        sources = set(ei[0].tolist())
        targets = set(ei[1].tolist())
        assert {0, 1} == sources
        assert {0, 1} == targets

    def test_two_stones_beyond_threshold_not_connected(self):
        nodes = self._make_nodes([(0.0, 0.0), (3.0, 0.0)])
        ei, ea = build_spatial_edges(nodes, dist_threshold=1.0)
        assert ei.shape[1] == 0

    def test_edge_attr_shape(self):
        nodes = self._make_nodes([(0.0, 0.0), (0.5, 0.0), (-0.3, 0.4)])
        ei, ea = build_spatial_edges(nodes, dist_threshold=2.0)
        assert ea.shape[0] == ei.shape[1]
        assert ea.shape[1] == N_EDGE_FEATURES

    def test_edge_attr_distance_is_positive(self):
        nodes = self._make_nodes([(0.0, 0.0), (0.5, 0.0)])
        _, ea = build_spatial_edges(nodes, dist_threshold=1.0)
        assert (ea[:, 2] > 0).all()

    def test_edge_attr_rel_x_is_antisymmetric(self):
        """rel_x of edge 0→1 should equal -(rel_x of edge 1→0)."""
        nodes = self._make_nodes([(0.0, 0.0), (0.5, 0.0)])
        ei, ea = build_spatial_edges(nodes, dist_threshold=1.0)
        # Find which edge is 0→1 and which is 1→0
        idx_01 = int(np.where((ei[0] == 0) & (ei[1] == 1))[0][0])
        idx_10 = int(np.where((ei[0] == 1) & (ei[1] == 0))[0][0])
        assert ea[idx_01, 0] == pytest.approx(-ea[idx_10, 0], abs=1e-5)

    def test_edge_index_dtype_is_int64(self):
        nodes = self._make_nodes([(0.0, 0.0), (0.5, 0.0)])
        ei, _ = build_spatial_edges(nodes, dist_threshold=1.0)
        assert ei.dtype == np.int64

    def test_edge_attr_dtype_is_float32(self):
        nodes = self._make_nodes([(0.0, 0.0), (0.5, 0.0)])
        _, ea = build_spatial_edges(nodes, dist_threshold=1.0)
        assert ea.dtype == np.float32


# ---------------------------------------------------------------------------
# build_temporal_edges
# ---------------------------------------------------------------------------

class TestBuildTemporalEdges:

    def test_no_overlap_returns_empty(self):
        ei = build_temporal_edges([1, 2], [3, 4])
        assert ei.shape == (2, 0)

    def test_empty_prev_returns_empty(self):
        ei = build_temporal_edges([], [1, 2])
        assert ei.shape == (2, 0)

    def test_empty_curr_returns_empty(self):
        ei = build_temporal_edges([1, 2], [])
        assert ei.shape == (2, 0)

    def test_full_overlap_one_to_one(self):
        """Each stone in prev matches exactly one in curr."""
        ei = build_temporal_edges([1, 2], [1, 2])
        assert ei.shape == (2, 2)
        # Stone 1: prev_idx=0 → curr_idx=0
        # Stone 2: prev_idx=1 → curr_idx=1
        edges = set(map(tuple, ei.T.tolist()))
        assert (0, 0) in edges
        assert (1, 1) in edges

    def test_partial_overlap(self):
        """Stone 2 disappeared; stone 3 appeared. Only stone 1 has a link."""
        ei = build_temporal_edges([1, 2], [1, 3])
        assert ei.shape == (2, 1)
        assert ei[0, 0] == 0  # stone 1 was at prev_idx 0
        assert ei[1, 0] == 0  # stone 1 is at curr_idx 0

    def test_reordered_ids_match_by_id_not_position(self):
        """prev=[2,1], curr=[1,2] — matches must use ID, not index position."""
        ei = build_temporal_edges([2, 1], [1, 2])
        edges = set(map(tuple, ei.T.tolist()))
        # Stone 1: prev_idx=1, curr_idx=0
        assert (1, 0) in edges
        # Stone 2: prev_idx=0, curr_idx=1
        assert (0, 1) in edges

    def test_none_ids_do_not_form_edges(self):
        ei = build_temporal_edges([None, 1], [None, 1])
        # Only stone_id=1 can match
        assert ei.shape == (2, 1)

    def test_dtype_is_int64(self):
        ei = build_temporal_edges([1], [1])
        assert ei.dtype == np.int64


# ---------------------------------------------------------------------------
# build_graph_features
# ---------------------------------------------------------------------------

class TestBuildGraphFeatures:

    def test_shape(self):
        u = build_graph_features(_row())
        assert u.shape == (N_GRAPH_FEATURES,)

    def test_dtype_is_float32(self):
        u = build_graph_features(_row())
        assert u.dtype == np.float32

    def test_shot_number_normalised(self):
        u = build_graph_features(_row(shot_number=8))
        shot_norm_idx = 0
        assert u[shot_norm_idx] == pytest.approx(8 / 16.0)

    def test_end_number_normalised(self):
        u = build_graph_features(_row(end_number=5))
        end_norm_idx = 1
        assert u[end_norm_idx] == pytest.approx(5 / 10.0)

    def test_stone_counts_reflect_row(self):
        u = build_graph_features(_row(team1_stones_in_play=3, team2_stones_in_play=2))
        assert u[2] == pytest.approx(3.0)
        assert u[3] == pytest.approx(2.0)

    def test_hammer_true_maps_to_one(self):
        u = build_graph_features(_row(shooting_team_has_hammer=True))
        assert u[4] == pytest.approx(1.0)

    def test_hammer_false_maps_to_zero(self):
        u = build_graph_features(_row(shooting_team_has_hammer=False))
        assert u[4] == pytest.approx(0.0)

    def test_none_values_default_to_zero(self):
        u = build_graph_features(_row(
            shooting_team_has_hammer=None,
            score_diff_before=None,
            shooting_team_is_team1=None,
        ))
        assert u[4] == pytest.approx(0.0)
        assert u[5] == pytest.approx(0.0)
        assert u[6] == pytest.approx(0.0)

    def test_score_diff_preserved(self):
        u = build_graph_features(_row(score_diff_before=-3))
        assert u[5] == pytest.approx(-3.0)


# ---------------------------------------------------------------------------
# extract_labels
# ---------------------------------------------------------------------------

class TestExtractLabels:

    def test_known_shot_type_produces_correct_index(self):
        for i, shot_type in enumerate(SHOT_TYPE_CLASSES):
            labels = extract_labels(_row(shot_type=shot_type))
            assert labels["shot_type_idx"] == i

    def test_unknown_shot_type_returns_minus_one(self):
        labels = extract_labels(_row(shot_type="Bogus"))
        assert labels["shot_type_idx"] == -1

    def test_none_shot_type_returns_minus_one(self):
        labels = extract_labels(_row(shot_type=None))
        assert labels["shot_type_idx"] == -1

    def test_accuracy_float_preserved(self):
        labels = extract_labels(_row(accuracy=82.5))
        assert labels["accuracy"] == pytest.approx(82.5)

    def test_accuracy_none_is_nan(self):
        labels = extract_labels(_row(accuracy=None))
        assert math.isnan(labels["accuracy"])

    def test_accuracy_empty_string_is_nan(self):
        labels = extract_labels(_row(accuracy=""))
        assert math.isnan(labels["accuracy"])

    def test_end_scores_preserved(self):
        labels = extract_labels(_row(team1_score_this_end=2.0, team2_score_this_end=1.0))
        assert labels["team1_score_this_end"] == pytest.approx(2.0)
        assert labels["team2_score_this_end"] == pytest.approx(1.0)

    def test_end_scores_nan_when_null(self):
        labels = extract_labels(_row(team1_score_this_end=float("nan"), team2_score_this_end=None))
        assert math.isnan(labels["team1_score_this_end"])
        assert math.isnan(labels["team2_score_this_end"])

    def test_all_required_keys_present(self):
        labels = extract_labels(_row())
        assert set(labels.keys()) == {
            "shot_type_idx", "accuracy",
            "team1_score_this_end", "team2_score_this_end",
        }


# ---------------------------------------------------------------------------
# build_shot_graph
# ---------------------------------------------------------------------------

class TestBuildShotGraph:

    def _row_with_one_stone(self, team, slot, x, y, stone_id, **extra):
        d = {f"team{team}_stones_in_play": 1}
        _add_stone(d, team, slot, x=x, y=y, stone_id=stone_id)
        return _row(**{**d, **extra})

    def test_required_keys_present(self):
        curr = self._row_with_one_stone(1, 1, 0.0, 0.0, stone_id=1)
        g = build_shot_graph(None, curr)
        required = {
            "x_t", "x_prev",
            "edge_index_t", "edge_attr_t",
            "edge_index_prev", "edge_attr_prev",
            "edge_index_temp", "u",
            "shot_type_idx", "accuracy",
            "team1_score_this_end", "team2_score_this_end",
            "event_id", "match_id", "end_number", "shot_number",
        }
        assert required.issubset(g.keys())

    def test_first_shot_has_empty_x_prev(self):
        curr = self._row_with_one_stone(1, 1, 0.0, 0.0, stone_id=1, shot_number=1)
        g = build_shot_graph(None, curr)
        assert g["x_prev"].shape == (0, N_NODE_FEATURES)
        assert g["edge_index_prev"].shape == (2, 0)

    def test_non_first_shot_x_prev_has_nodes(self):
        prev = self._row_with_one_stone(1, 1, 0.0, 0.0, stone_id=1, shot_number=1)
        curr = self._row_with_one_stone(1, 1, 0.1, 0.0, stone_id=1, shot_number=2)
        g = build_shot_graph(prev, curr)
        assert g["x_prev"].shape[0] == 1

    def test_temporal_edges_connect_matching_stone_ids(self):
        prev = self._row_with_one_stone(1, 1, 0.0, 0.0, stone_id=5, shot_number=1)
        d = {"team1_stones_in_play": 1}
        _add_stone(d, 1, 1, x=0.1, y=0.0, stone_id=5, prev_x=0.0, prev_y=0.0)
        curr = _row(shot_number=2, **d)
        g = build_shot_graph(prev, curr)
        # One temporal edge: prev node 0 → curr node 0
        assert g["edge_index_temp"].shape == (2, 1)
        assert g["edge_index_temp"][0, 0] == 0  # prev idx
        assert g["edge_index_temp"][1, 0] == 0  # curr idx

    def test_temporal_edges_empty_when_stone_knocked_out(self):
        """Stone in S_{t-1} does not appear in S_t — no temporal edge."""
        prev = self._row_with_one_stone(1, 1, 0.0, 0.0, stone_id=3, shot_number=1)
        curr = self._row_with_one_stone(1, 1, 0.5, 0.0, stone_id=7, shot_number=2)
        g = build_shot_graph(prev, curr)
        assert g["edge_index_temp"].shape == (2, 0)

    def test_x_t_has_correct_shape(self):
        d = {"team1_stones_in_play": 2, "team2_stones_in_play": 1}
        _add_stone(d, 1, 1, x=0.0, y=0.0, stone_id=1)
        _add_stone(d, 1, 2, x=0.2, y=0.0, stone_id=2)
        _add_stone(d, 2, 1, x=-0.1, y=0.0, stone_id=3)
        curr = _row(**d)
        g = build_shot_graph(None, curr)
        assert g["x_t"].shape == (3, N_NODE_FEATURES)

    def test_metadata_fields_correct(self):
        curr = self._row_with_one_stone(1, 1, 0.0, 0.0, stone_id=1,
                                        event_id=42, match_id=7,
                                        end_number=3, shot_number=5)
        g = build_shot_graph(None, curr)
        assert g["event_id"] == 42
        assert g["match_id"] == 7
        assert g["end_number"] == 3
        assert g["shot_number"] == 5

    def test_u_shape(self):
        curr = self._row_with_one_stone(1, 1, 0.0, 0.0, stone_id=1)
        g = build_shot_graph(None, curr)
        assert g["u"].shape == (N_GRAPH_FEATURES,)


# ---------------------------------------------------------------------------
# build_all_graphs
# ---------------------------------------------------------------------------

class TestBuildAllGraphs:

    def test_one_end_two_shots_produces_two_graphs(self):
        rows = [
            {"shot_number": 1, "team1_stones_in_play": 1, "team2_stones_in_play": 0,
             **{k: v for k, v in _add_stone({}, 1, 1, x=0.0, y=0.0, stone_id=1).items()}},
            {"shot_number": 2, "team1_stones_in_play": 1, "team2_stones_in_play": 0,
             **{k: v for k, v in _add_stone({}, 1, 1, x=0.1, y=0.0, stone_id=1).items()}},
        ]
        df = _shots_df(rows)
        graphs = build_all_graphs(df)
        assert len(graphs) == 2

    def test_first_shot_in_end_has_empty_x_prev(self):
        rows = [{"shot_number": 1, "team1_stones_in_play": 1, "team2_stones_in_play": 0,
                 **_add_stone({}, 1, 1, x=0.0, y=0.0, stone_id=1)}]
        df = _shots_df(rows)
        graphs = build_all_graphs(df)
        assert graphs[0]["x_prev"].shape[0] == 0

    def test_second_shot_has_x_prev_from_first_shot(self):
        d1 = {"shot_number": 1, "team1_stones_in_play": 1, "team2_stones_in_play": 0}
        _add_stone(d1, 1, 1, x=0.0, y=0.0, stone_id=1)
        d2 = {"shot_number": 2, "team1_stones_in_play": 1, "team2_stones_in_play": 0}
        _add_stone(d2, 1, 1, x=0.1, y=0.0, stone_id=1)
        df = _shots_df([d1, d2])
        graphs = build_all_graphs(df)
        assert graphs[1]["x_prev"].shape[0] == 1

    def test_end_boundary_resets_prev_row(self):
        """First shot of end 2 must have empty x_prev even though end 1 had data."""
        d1 = {"end_number": 1, "shot_number": 1,
              "team1_stones_in_play": 1, "team2_stones_in_play": 0}
        _add_stone(d1, 1, 1, x=0.0, y=0.0, stone_id=1)
        d2 = {"end_number": 2, "shot_number": 1,
              "team1_stones_in_play": 1, "team2_stones_in_play": 0}
        _add_stone(d2, 1, 1, x=0.2, y=0.0, stone_id=10)
        df = _shots_df([d1, d2])
        graphs = build_all_graphs(df)
        assert graphs[0]["x_prev"].shape[0] == 0  # end1 shot1
        assert graphs[1]["x_prev"].shape[0] == 0  # end2 shot1

    def test_shots_processed_in_shot_number_order_despite_df_order(self):
        """DataFrames not in shot order should still be processed correctly."""
        d1 = {"shot_number": 2, "team1_stones_in_play": 1, "team2_stones_in_play": 0}
        _add_stone(d1, 1, 1, x=0.1, y=0.0, stone_id=1)
        d2 = {"shot_number": 1, "team1_stones_in_play": 1, "team2_stones_in_play": 0}
        _add_stone(d2, 1, 1, x=0.0, y=0.0, stone_id=1)
        df = _shots_df([d1, d2])  # shot 2 is listed first
        graphs = build_all_graphs(df)
        # After sorting, shot 1 is processed first → its graph has no x_prev
        shot1_graph = next(g for g in graphs if g["shot_number"] == 1)
        shot2_graph = next(g for g in graphs if g["shot_number"] == 2)
        assert shot1_graph["x_prev"].shape[0] == 0
        assert shot2_graph["x_prev"].shape[0] == 1

    def test_two_ends_processed_independently(self):
        rows = []
        for end in (1, 2):
            for shot in (1, 2):
                d = {"end_number": end, "shot_number": shot,
                     "team1_stones_in_play": 1, "team2_stones_in_play": 0}
                _add_stone(d, 1, 1, x=float(shot) * 0.1, y=0.0, stone_id=shot)
                rows.append(d)
        df = _shots_df(rows)
        graphs = build_all_graphs(df)
        assert len(graphs) == 4
        # Both shot-1 graphs (one per end) must have empty x_prev
        shot1_graphs = [g for g in graphs if g["shot_number"] == 1]
        for g in shot1_graphs:
            assert g["x_prev"].shape[0] == 0


# ---------------------------------------------------------------------------
# assign_splits
# ---------------------------------------------------------------------------

class TestAssignSplits:

    def _events_df(self, rows: list[dict]) -> pd.DataFrame:
        return pd.DataFrame(rows)

    def test_old_event_assigned_train(self):
        graphs = [{"event_id": 1}]
        events = self._events_df([{"event_id": 1, "year": "2022"}])
        result = assign_splits(graphs, events)
        assert result[0]["split"] == "train"

    def test_boundary_year_assigned_train(self):
        graphs = [{"event_id": 1}]
        events = self._events_df([{"event_id": 1, "year": "2023"}])
        result = assign_splits(graphs, events)
        assert result[0]["split"] == "train"

    def test_val_year_assigned_val(self):
        graphs = [{"event_id": 1}]
        events = self._events_df([{"event_id": 1, "year": "2024"}])
        result = assign_splits(graphs, events)
        assert result[0]["split"] == "val"

    def test_future_year_assigned_test(self):
        graphs = [{"event_id": 1}]
        events = self._events_df([{"event_id": 1, "year": "2025"}])
        result = assign_splits(graphs, events)
        assert result[0]["split"] == "test"

    def test_unknown_event_defaults_to_train(self):
        graphs = [{"event_id": 99}]
        events = self._events_df([{"event_id": 1, "year": "2024"}])
        result = assign_splits(graphs, events)
        assert result[0]["split"] == "train"

    def test_multiple_events_split_correctly(self):
        graphs = [{"event_id": 1}, {"event_id": 2}, {"event_id": 3}]
        events = self._events_df([
            {"event_id": 1, "year": "2023"},
            {"event_id": 2, "year": "2024"},
            {"event_id": 3, "year": "2025"},
        ])
        result = assign_splits(graphs, events)
        assert result[0]["split"] == "train"
        assert result[1]["split"] == "val"
        assert result[2]["split"] == "test"


# ---------------------------------------------------------------------------
# save_graphs / load_graphs round-trip
# ---------------------------------------------------------------------------

class TestSaveLoadGraphs:

    def test_round_trip_preserves_dict_structure(self, tmp_path):
        graphs = [
            {
                "x_t": np.zeros((2, N_NODE_FEATURES), dtype=np.float32),
                "x_prev": np.zeros((0, N_NODE_FEATURES), dtype=np.float32),
                "event_id": 1, "shot_number": 1, "split": "train",
            }
        ]
        path = str(tmp_path / "graphs.pkl.gz")
        save_graphs(graphs, path)
        loaded = load_graphs(path)
        assert len(loaded) == 1
        np.testing.assert_array_equal(loaded[0]["x_t"], graphs[0]["x_t"])
        assert loaded[0]["event_id"] == 1
        assert loaded[0]["split"] == "train"

    def test_round_trip_multiple_graphs(self, tmp_path):
        graphs = [{"shot_number": i, "val": np.array([i], dtype=np.float32)}
                  for i in range(10)]
        path = str(tmp_path / "g.pkl.gz")
        save_graphs(graphs, path)
        loaded = load_graphs(path)
        assert len(loaded) == 10
        for i, g in enumerate(loaded):
            assert g["shot_number"] == i
            np.testing.assert_array_equal(g["val"], [i])

    def test_save_creates_output_directory(self, tmp_path):
        graphs = [{"x": np.zeros((1, 1))}]
        path = str(tmp_path / "subdir" / "graphs.pkl.gz")
        save_graphs(graphs, path)
        assert os.path.isfile(path)
