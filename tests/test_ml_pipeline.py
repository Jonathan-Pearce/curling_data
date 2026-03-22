"""Unit tests for the Siamese GNN curling ML pipeline.

Tests cover:
- Configuration defaults
- Data helpers (base shot type extraction, graph construction)
- Model forward pass shapes
- Multi-task loss computation and gradient flow
- Dataset construction and collation
- End-to-end mini training loop
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import torch

from ml.config import BASE_SHOT_TYPES, TURN_CLASSES, PipelineConfig
from ml.data import (
    CurlingShotDataset,
    _board_state_to_node_features,
    build_fully_connected_edge_index,
    collate_curling_batch,
    extract_base_shot_type,
)
from ml.loss import MultiTaskLoss
from ml.model import GCNLayer, GNNEncoder, SiameseGNN


# ─── Config tests ────────────────────────────────────────────────────────────


class TestConfig:
    def test_defaults(self):
        cfg = PipelineConfig()
        assert cfg.device == "cpu"
        assert cfg.gnn_hidden_dim == 64
        assert cfg.num_shot_types == len(BASE_SHOT_TYPES)
        assert cfg.batch_size > 0

    def test_shot_types_non_empty(self):
        assert len(BASE_SHOT_TYPES) >= 10
        assert "Draw" in BASE_SHOT_TYPES
        assert "Take-out" in BASE_SHOT_TYPES

    def test_turn_classes(self):
        assert len(TURN_CLASSES) == 3
        assert "Clockwise" in TURN_CLASSES


# ─── Data helper tests ───────────────────────────────────────────────────────


class TestExtractBaseShotType:
    def test_exact_match(self):
        assert extract_base_shot_type("Draw") == "Draw"
        assert extract_base_shot_type("Take-out") == "Take-out"
        assert extract_base_shot_type("Hit and Roll") == "Hit and Roll"

    def test_with_annotation(self):
        assert extract_base_shot_type("Draw Time-out") == "Draw"
        assert extract_base_shot_type("Take-out Free Guard Zone violation") == "Take-out"
        assert extract_base_shot_type("Guard Did not pass the hog line") == "Guard"
        assert extract_base_shot_type("Wick / Soft Peeling Split") == "Wick / Soft Peeling"

    def test_unknown_type(self):
        assert extract_base_shot_type("no statistics") is None
        assert extract_base_shot_type("not played") is None

    def test_longer_match_preferred(self):
        # "Double Take-out" should not match just "Take-out"
        assert extract_base_shot_type("Double Take-out Time-out") == "Double Take-out"
        assert extract_base_shot_type("Promotion Take-out Measurement") == "Promotion Take-out"
        assert extract_base_shot_type("Hit and Roll Free Guard Zone violation") == "Hit and Roll"


class TestBoardStateToNodeFeatures:
    def _make_row(self, t1_count, t2_count):
        """Create a minimal Series with stone columns."""
        data = {
            "team1_stones_in_play": t1_count,
            "team2_stones_in_play": t2_count,
        }
        for i in range(1, 9):
            for team in ("team1", "team2"):
                for col in ("x", "y", "dist", "angle"):
                    key = f"{team}_stone{i}_{col}"
                    count = t1_count if team == "team1" else t2_count
                    data[key] = float(i) if i <= count else None
        return pd.Series(data)

    def test_empty_board(self):
        row = self._make_row(0, 0)
        feats, n = _board_state_to_node_features(row, 0, 0)
        assert n == 0
        assert feats.shape == (0, 5)

    def test_single_team(self):
        row = self._make_row(3, 0)
        feats, n = _board_state_to_node_features(row, 3, 0)
        assert n == 3
        assert feats.shape == (3, 5)
        # All should be team1 (is_team1 == 1.0)
        assert (feats[:, 4] == 1.0).all()

    def test_both_teams(self):
        row = self._make_row(2, 3)
        feats, n = _board_state_to_node_features(row, 2, 3)
        assert n == 5
        assert feats.shape == (5, 5)
        # First 2 are team1, next 3 are team2
        assert (feats[:2, 4] == 1.0).all()
        assert (feats[2:, 4] == 0.0).all()


class TestEdgeIndex:
    def test_zero_nodes(self):
        ei = build_fully_connected_edge_index(0)
        assert ei.shape == (2, 0)

    def test_one_node(self):
        ei = build_fully_connected_edge_index(1)
        assert ei.shape == (2, 0)

    def test_two_nodes(self):
        ei = build_fully_connected_edge_index(2)
        assert ei.shape == (2, 2)  # 0→1 and 1→0

    def test_fully_connected(self):
        n = 5
        ei = build_fully_connected_edge_index(n)
        assert ei.shape == (2, n * (n - 1))
        # No self-loops
        assert (ei[0] != ei[1]).all()


# ─── Model tests ─────────────────────────────────────────────────────────────


class TestGCNLayer:
    def test_output_shape(self):
        layer = GCNLayer(5, 16, dropout=0.0)
        x = torch.randn(4, 6, 5)
        mask = torch.ones(4, 6)
        out = layer(x, mask)
        assert out.shape == (4, 6, 16)

    def test_padding_masked(self):
        layer = GCNLayer(5, 16, dropout=0.0)
        x = torch.randn(2, 4, 5)
        mask = torch.zeros(2, 4)
        mask[0, :2] = 1.0  # Only 2 valid nodes in sample 0
        mask[1, :3] = 1.0
        out = layer(x, mask)
        # Padding positions should be zero
        assert (out[0, 2:] == 0).all()
        assert (out[1, 3:] == 0).all()


class TestGNNEncoder:
    def test_output_shape(self):
        cfg = PipelineConfig(gnn_hidden_dim=32, gnn_num_layers=2)
        encoder = GNNEncoder(cfg)
        x = torch.randn(3, 5, 5)
        mask = torch.ones(3, 5)
        emb = encoder(x, mask)
        assert emb.shape == (3, 32)

    def test_empty_graph_embedding(self):
        """An empty graph (all-zero mask) should produce a finite embedding."""
        cfg = PipelineConfig(gnn_hidden_dim=16, gnn_num_layers=2)
        encoder = GNNEncoder(cfg)
        x = torch.zeros(1, 3, 5)
        mask = torch.zeros(1, 3)
        emb = encoder(x, mask)
        assert emb.shape == (1, 16)
        assert torch.isfinite(emb).all()


class TestSiameseGNN:
    @pytest.fixture()
    def model_and_cfg(self):
        cfg = PipelineConfig(
            gnn_hidden_dim=16,
            gnn_num_layers=2,
            metadata_embed_dim=8,
            head_hidden_dim=16,
        )
        model = SiameseGNN(cfg)
        return model, cfg

    def test_forward_shapes(self, model_and_cfg):
        model, cfg = model_and_cfg
        B = 4
        prev_nodes = torch.randn(B, 3, 5)
        prev_mask = torch.ones(B, 3)
        cur_nodes = torch.randn(B, 5, 5)
        cur_mask = torch.ones(B, 5)
        metadata = torch.randn(B, cfg.metadata_features)
        is_first = torch.zeros(B)

        out = model(prev_nodes, prev_mask, cur_nodes, cur_mask, metadata, is_first)
        assert out["end_score"].shape == (B, 2)
        assert out["accuracy"].shape == (B, 1)
        assert out["shot_type"].shape == (B, cfg.num_shot_types)

    def test_first_shot_handling(self, model_and_cfg):
        """First shot: empty prev graph should still produce valid outputs."""
        model, cfg = model_and_cfg
        B = 2
        prev_nodes = torch.zeros(B, 1, 5)
        prev_mask = torch.zeros(B, 1)  # No valid prev nodes
        cur_nodes = torch.randn(B, 4, 5)
        cur_mask = torch.ones(B, 4)
        metadata = torch.randn(B, cfg.metadata_features)
        is_first = torch.ones(B)

        out = model(prev_nodes, prev_mask, cur_nodes, cur_mask, metadata, is_first)
        for key in ("end_score", "accuracy", "shot_type"):
            assert torch.isfinite(out[key]).all(), f"NaN/Inf in {key}"


# ─── Loss tests ──────────────────────────────────────────────────────────────


class TestMultiTaskLoss:
    def test_basic_forward(self):
        criterion = MultiTaskLoss()
        B, C = 8, 12
        pred_end = torch.randn(B, 2)
        target_end = torch.randn(B, 2)
        pred_acc = torch.randn(B, 1)
        target_acc = torch.rand(B) * 100
        acc_mask = torch.ones(B)
        pred_type = torch.randn(B, C)
        target_type = torch.randint(0, C, (B,))
        type_mask = torch.ones(B)

        losses = criterion(
            pred_end, target_end, pred_acc, target_acc, acc_mask,
            pred_type, target_type, type_mask,
        )
        assert "total" in losses
        assert losses["total"].requires_grad

    def test_gradients_flow_to_log_vars(self):
        criterion = MultiTaskLoss()
        B, C = 4, 12
        pred_end = torch.randn(B, 2, requires_grad=True)
        target_end = torch.randn(B, 2)
        pred_acc = torch.randn(B, 1, requires_grad=True)
        target_acc = torch.rand(B) * 100
        pred_type = torch.randn(B, C, requires_grad=True)
        target_type = torch.randint(0, C, (B,))

        losses = criterion(
            pred_end, target_end,
            pred_acc, target_acc, torch.ones(B),
            pred_type, target_type, torch.ones(B),
        )
        losses["total"].backward()
        assert criterion.log_var_end.grad is not None
        assert criterion.log_var_acc.grad is not None
        assert criterion.log_var_cls.grad is not None

    def test_masked_accuracy(self):
        """When all accuracy labels are invalid, accuracy loss should be 0."""
        criterion = MultiTaskLoss()
        B, C = 4, 12
        losses = criterion(
            torch.randn(B, 2), torch.randn(B, 2),
            torch.randn(B, 1), torch.ones(B) * -1, torch.zeros(B),  # all masked
            torch.randn(B, C), torch.randint(0, C, (B,)), torch.ones(B),
        )
        assert losses["accuracy"].item() == 0.0

    def test_masked_shot_type(self):
        """When all shot type labels are invalid, cls loss should be 0."""
        criterion = MultiTaskLoss()
        B, C = 4, 12
        losses = criterion(
            torch.randn(B, 2), torch.randn(B, 2),
            torch.randn(B, 1), torch.rand(B) * 100, torch.ones(B),
            torch.randn(B, C), torch.full((B,), -1, dtype=torch.long), torch.zeros(B),
        )
        assert losses["shot_type"].item() == 0.0


# ─── Dataset & collation tests ──────────────────────────────────────────────


def _make_mini_dataframe(n_ends: int = 2, shots_per_end: int = 4) -> pd.DataFrame:
    """Create a small synthetic DataFrame mimicking preprocessed shots."""
    rows = []
    for end in range(1, n_ends + 1):
        for shot in range(1, shots_per_end + 1):
            t1 = min(shot // 2, 3)
            t2 = min((shot - 1) // 2, 3)
            row = {
                "event_id": 1,
                "match_id": 1,
                "end_number": end,
                "shot_number": shot,
                "team_code": "A" if shot % 2 == 1 else "B",
                "team1_stones_in_play": t1,
                "team2_stones_in_play": t2,
                "accuracy": float(50 + shot * 5),
                "shot_type_label": shot % len(BASE_SHOT_TYPES),
                "team1_score_this_end": 2.0,
                "team2_score_this_end": 1.0,
                "score_diff_before": 0,
                "shooting_team_has_hammer": shot % 2,
            }
            # Fill stone columns
            for i in range(1, 9):
                for team in ("team1", "team2"):
                    count = t1 if team == "team1" else t2
                    for col in ("x", "y", "dist", "angle"):
                        key = f"{team}_stone{i}_{col}"
                        row[key] = float(i) * 0.1 if i <= count else None
            rows.append(row)
    return pd.DataFrame(rows)


class TestCurlingShotDataset:
    def test_length(self):
        df = _make_mini_dataframe(n_ends=2, shots_per_end=4)
        ds = CurlingShotDataset(df)
        assert len(ds) == 8  # 2 ends × 4 shots

    def test_first_shot_flag(self):
        df = _make_mini_dataframe(n_ends=1, shots_per_end=3)
        ds = CurlingShotDataset(df)
        assert ds[0]["is_first_shot"] == 1.0
        assert ds[1]["is_first_shot"] == 0.0

    def test_item_keys(self):
        df = _make_mini_dataframe(n_ends=1, shots_per_end=2)
        ds = CurlingShotDataset(df)
        item = ds[1]
        expected_keys = {
            "nodes_prev", "edge_index_prev", "num_nodes_prev",
            "nodes_cur", "edge_index_cur", "num_nodes_cur",
            "metadata", "is_first_shot",
            "end_score_team1", "end_score_team2",
            "accuracy", "shot_type_label",
        }
        assert set(item.keys()) == expected_keys


class TestCollateBatch:
    def test_collate_shapes(self):
        df = _make_mini_dataframe(n_ends=1, shots_per_end=4)
        ds = CurlingShotDataset(df)
        batch = collate_curling_batch([ds[i] for i in range(4)])

        B = 4
        assert batch["cur_nodes"].shape[0] == B
        assert batch["metadata"].shape == (B, 6)
        assert batch["end_score_team1"].shape == (B,)
        assert batch["accuracy"].shape == (B,)
        assert batch["shot_type_label"].shape == (B,)

    def test_collate_masks(self):
        df = _make_mini_dataframe(n_ends=1, shots_per_end=3)
        ds = CurlingShotDataset(df)
        batch = collate_curling_batch([ds[0], ds[2]])
        # First shot has fewer stones than third
        assert batch["cur_mask"].shape[0] == 2


# ─── Integration: mini training step ────────────────────────────────────────


class TestMiniTrainingStep:
    def test_one_step_no_crash(self):
        """Verify one forward + backward pass completes without error."""
        cfg = PipelineConfig(
            gnn_hidden_dim=8,
            gnn_num_layers=2,
            metadata_embed_dim=4,
            head_hidden_dim=8,
        )
        model = SiameseGNN(cfg)
        criterion = MultiTaskLoss()
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(criterion.parameters()), lr=1e-3
        )

        df = _make_mini_dataframe(n_ends=2, shots_per_end=4)
        ds = CurlingShotDataset(df)
        batch = collate_curling_batch([ds[i] for i in range(min(4, len(ds)))])

        preds = model(
            batch["prev_nodes"],
            batch["prev_mask"],
            batch["cur_nodes"],
            batch["cur_mask"],
            batch["metadata"],
            batch["is_first_shot"],
        )

        target_end = torch.stack(
            [batch["end_score_team1"], batch["end_score_team2"]], dim=-1
        )
        acc_mask = (batch["accuracy"] >= 0).float()
        type_mask = (batch["shot_type_label"] >= 0).float()

        losses = criterion(
            preds["end_score"], target_end,
            preds["accuracy"], batch["accuracy"], acc_mask,
            preds["shot_type"], batch["shot_type_label"], type_mask,
        )

        optimizer.zero_grad()
        losses["total"].backward()
        optimizer.step()

        # Weights should have changed
        assert losses["total"].item() > 0

    def test_loss_decreases_over_steps(self):
        """A few steps on the same batch should reduce loss."""
        cfg = PipelineConfig(
            gnn_hidden_dim=16,
            gnn_num_layers=2,
            metadata_embed_dim=8,
            head_hidden_dim=16,
        )
        model = SiameseGNN(cfg)
        criterion = MultiTaskLoss()
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(criterion.parameters()), lr=1e-2
        )

        df = _make_mini_dataframe(n_ends=1, shots_per_end=4)
        ds = CurlingShotDataset(df)
        batch = collate_curling_batch([ds[i] for i in range(len(ds))])

        target_end = torch.stack(
            [batch["end_score_team1"], batch["end_score_team2"]], dim=-1
        )
        acc_mask = (batch["accuracy"] >= 0).float()
        type_mask = (batch["shot_type_label"] >= 0).float()

        losses_history = []
        for _ in range(20):
            preds = model(
                batch["prev_nodes"],
                batch["prev_mask"],
                batch["cur_nodes"],
                batch["cur_mask"],
                batch["metadata"],
                batch["is_first_shot"],
            )
            losses = criterion(
                preds["end_score"], target_end,
                preds["accuracy"], batch["accuracy"], acc_mask,
                preds["shot_type"], batch["shot_type_label"], type_mask,
            )
            optimizer.zero_grad()
            losses["total"].backward()
            optimizer.step()
            losses_history.append(losses["total"].item())

        # Loss should decrease (last < first)
        assert losses_history[-1] < losses_history[0]
