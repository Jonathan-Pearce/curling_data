"""Unit tests for the Siamese GNN curling pipeline."""

import math

import pytest
import torch

from gnn_pipeline.config import BASE_SHOT_TYPES, GNNConfig
from gnn_pipeline.data import (
    CurlingDataset,
    _classify_shot_type,
    _encode_turn,
    _parse_stone_positions,
)
from gnn_pipeline.graph import build_board_graph
from gnn_pipeline.loss import MultiTaskLoss
from gnn_pipeline.model import GNNEncoder, GraphConvLayer, SiameseGNN
from gnn_pipeline.train import collate_fn


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_config(**overrides) -> GNNConfig:
    defaults = dict(
        max_stones=4,
        gnn_hidden_dim=8,
        gnn_num_layers=2,
        graph_emb_dim=8,
        comparison_dim=16,
        head_hidden_dim=8,
        dropout=0.0,
    )
    defaults.update(overrides)
    return GNNConfig(**defaults)


def _dummy_stones(n: int = 3) -> list:
    return [
        {"x": float(i), "y": float(i) + 0.5, "dist": float(i) + 1.0,
         "angle": float(i) * 10.0, "team": 1 if i % 2 == 0 else 2}
        for i in range(n)
    ]


# ── Config tests ─────────────────────────────────────────────────────────


class TestConfig:
    def test_defaults(self):
        cfg = GNNConfig()
        assert cfg.max_stones == 16
        assert cfg.num_shot_types == 12
        assert cfg.device == "cpu"

    def test_shot_types_length(self):
        assert len(BASE_SHOT_TYPES) == 12


# ── Graph construction tests ─────────────────────────────────────────────


class TestBuildBoardGraph:
    def test_empty_board(self):
        nf, adj, mask = build_board_graph([], max_stones=4)
        assert nf.shape == (4, 6)
        assert adj.shape == (4, 4)
        assert mask.shape == (4,)
        assert not mask.any()

    def test_single_stone(self):
        stones = [{"x": 1.0, "y": 2.0, "dist": 2.24, "angle": 63.4, "team": 1}]
        nf, adj, mask = build_board_graph(stones, max_stones=4)
        assert mask[0].item()
        assert not mask[1:].any()
        assert nf[0, 0].item() == pytest.approx(1.0)
        assert nf[0, 4].item() == 1.0  # is_team1
        assert nf[0, 5].item() == 0.0  # is_team2
        assert adj[0, 0].item() == pytest.approx(1.0)

    def test_multiple_stones(self):
        stones = _dummy_stones(3)
        nf, adj, mask = build_board_graph(stones, max_stones=4)
        assert mask[:3].all()
        assert not mask[3]
        # Adjacency should be normalised for existing nodes
        assert adj[0, :3].sum().item() == pytest.approx(1.0, abs=1e-5)

    def test_excess_stones_clipped(self):
        stones = _dummy_stones(6)
        nf, adj, mask = build_board_graph(stones, max_stones=4)
        assert mask.sum().item() == 4


# ── Data helpers ─────────────────────────────────────────────────────────


class TestClassifyShotType:
    def test_exact_match(self):
        assert _classify_shot_type("Draw") == BASE_SHOT_TYPES.index("Draw")

    def test_with_annotation(self):
        assert _classify_shot_type("Draw Time-out") == BASE_SHOT_TYPES.index("Draw")

    def test_compound_type(self):
        assert _classify_shot_type("Hit and Roll") == BASE_SHOT_TYPES.index("Hit and Roll")
        assert _classify_shot_type("Hit and Roll Burned stone") == BASE_SHOT_TYPES.index("Hit and Roll")

    def test_unknown(self):
        assert _classify_shot_type("Unknown Shot") is None


class TestEncodeTurn:
    def test_clockwise(self):
        assert _encode_turn("Clockwise") == 0.0

    def test_counter_clockwise(self):
        assert _encode_turn("Counter-clockwise") == 1.0

    def test_other(self):
        assert _encode_turn("Not considered") == 0.5
        assert _encode_turn("") == 0.5


class TestParseStonePositions:
    def test_parses_present_stones(self):
        row = {
            "team1_stone1_x": "0.5", "team1_stone1_y": "1.0",
            "team1_stone1_dist": "1.1", "team1_stone1_angle": "63.0",
            "team2_stone1_x": "-0.3", "team2_stone1_y": "2.0",
            "team2_stone1_dist": "2.02", "team2_stone1_angle": "99.0",
        }
        stones = _parse_stone_positions(row)
        assert len(stones) == 2
        assert stones[0]["team"] == 1
        assert stones[1]["team"] == 2

    def test_skips_empty_fields(self):
        row = {"team1_stone1_x": "0.5", "team1_stone1_y": "", "team1_stone1_dist": "1", "team1_stone1_angle": "1"}
        assert _parse_stone_positions(row) == []


# ── Model tests ──────────────────────────────────────────────────────────


class TestGraphConvLayer:
    def test_output_shape(self):
        layer = GraphConvLayer(6, 8)
        nf = torch.randn(2, 4, 6)
        adj = torch.ones(2, 4, 4) / 4
        mask = torch.ones(2, 4, dtype=torch.bool)
        out = layer(nf, adj, mask)
        assert out.shape == (2, 4, 8)

    def test_mask_zeros_padded(self):
        layer = GraphConvLayer(6, 8)
        nf = torch.randn(1, 4, 6)
        adj = torch.zeros(1, 4, 4)
        adj[0, :2, :2] = 0.5
        mask = torch.tensor([[True, True, False, False]])
        out = layer(nf, adj, mask)
        assert (out[0, 2:] == 0).all()


class TestGNNEncoder:
    def test_output_shape(self):
        cfg = _make_config()
        enc = GNNEncoder(cfg)
        nf = torch.randn(3, cfg.max_stones, cfg.node_feature_dim)
        adj = torch.ones(3, cfg.max_stones, cfg.max_stones) / cfg.max_stones
        mask = torch.ones(3, cfg.max_stones, dtype=torch.bool)
        out = enc(nf, adj, mask)
        assert out.shape == (3, cfg.graph_emb_dim)


class TestSiameseGNN:
    @pytest.fixture()
    def model_and_config(self):
        cfg = _make_config()
        model = SiameseGNN(cfg)
        return model, cfg

    def test_forward_shape(self, model_and_config):
        model, cfg = model_and_config
        B, N = 2, cfg.max_stones
        prev_nf = torch.randn(B, N, cfg.node_feature_dim)
        prev_adj = torch.ones(B, N, N) / N
        prev_mask = torch.ones(B, N, dtype=torch.bool)
        prev_null = torch.tensor([True, False])
        curr_nf = torch.randn(B, N, cfg.node_feature_dim)
        curr_adj = torch.ones(B, N, N) / N
        curr_mask = torch.ones(B, N, dtype=torch.bool)
        meta = torch.randn(B, cfg.metadata_dim)

        acc, types, score = model(
            prev_nf, prev_adj, prev_mask, prev_null,
            curr_nf, curr_adj, curr_mask, meta,
        )
        assert acc.shape == (B,)
        assert types.shape == (B, cfg.num_shot_types)
        assert score.shape == (B,)

    def test_null_embedding_used(self, model_and_config):
        """When prev_is_null is True the null embedding should be used."""
        model, cfg = model_and_config
        B, N = 1, cfg.max_stones
        zeros = torch.zeros(B, N, cfg.node_feature_dim)
        adj = torch.zeros(B, N, N)
        mask = torch.zeros(B, N, dtype=torch.bool)
        meta = torch.zeros(B, cfg.metadata_dim)

        # Both NULL
        acc1, _, _ = model(zeros, adj, mask, torch.tensor([True]),
                           zeros, adj, mask, meta)
        # Not NULL (identical zero input but flag differs)
        acc2, _, _ = model(zeros, adj, mask, torch.tensor([False]),
                           zeros, adj, mask, meta)
        # Outputs should differ because the null-embedding path is different
        assert not torch.allclose(acc1, acc2)


# ── Loss tests ───────────────────────────────────────────────────────────


class TestMultiTaskLoss:
    def test_all_masked(self):
        criterion = MultiTaskLoss()
        B = 4
        acc_pred = torch.randn(B)
        type_logits = torch.randn(B, 12)
        score_pred = torch.randn(B)
        no_mask = torch.zeros(B, dtype=torch.bool)

        loss, details = criterion(
            acc_pred, torch.zeros(B), no_mask,
            type_logits, torch.zeros(B, dtype=torch.long), no_mask,
            score_pred, torch.zeros(B), no_mask,
        )
        assert loss.item() == pytest.approx(0.0)

    def test_gradients_flow(self):
        criterion = MultiTaskLoss()
        B = 4
        acc_pred = torch.randn(B, requires_grad=True)
        type_logits = torch.randn(B, 12, requires_grad=True)
        score_pred = torch.randn(B, requires_grad=True)
        full_mask = torch.ones(B, dtype=torch.bool)

        loss, _ = criterion(
            acc_pred, torch.rand(B), full_mask,
            type_logits, torch.randint(0, 12, (B,)), full_mask,
            score_pred, torch.rand(B), full_mask,
        )
        loss.backward()
        assert acc_pred.grad is not None
        assert type_logits.grad is not None
        assert score_pred.grad is not None

    def test_learnable_weights(self):
        criterion = MultiTaskLoss()
        assert criterion.log_var_accuracy.requires_grad
        assert criterion.log_var_shot_type.requires_grad
        assert criterion.log_var_end_score.requires_grad


# ── Dataset / collate tests ──────────────────────────────────────────────


class TestCurlingDataset:
    def test_basic_sample(self):
        nf, adj, mask = build_board_graph(_dummy_stones(2), max_stones=4)
        sample = {
            "prev_node_feat": nf, "prev_adj": adj, "prev_mask": mask,
            "prev_is_null": True,
            "curr_node_feat": nf, "curr_adj": adj, "curr_mask": mask,
            "metadata": torch.zeros(6),
            "accuracy": 0.75, "shot_type": 2, "end_score": 1.0,
        }
        ds = CurlingDataset([sample])
        assert len(ds) == 1
        item = ds[0]
        assert item["prev_is_null"].item()
        assert item["accuracy"].item() == pytest.approx(0.75)
        assert item["shot_type"].item() == 2

    def test_missing_targets(self):
        nf, adj, mask = build_board_graph([], max_stones=4)
        sample = {
            "prev_node_feat": nf, "prev_adj": adj, "prev_mask": mask,
            "prev_is_null": False,
            "curr_node_feat": nf, "curr_adj": adj, "curr_mask": mask,
            "metadata": torch.zeros(6),
            "accuracy": None, "shot_type": None, "end_score": None,
        }
        ds = CurlingDataset([sample])
        item = ds[0]
        assert math.isnan(item["accuracy"].item())
        assert item["shot_type"].item() == -1
        assert math.isnan(item["end_score"].item())


class TestCollate:
    def test_batch_stacking(self):
        nf, adj, mask = build_board_graph(_dummy_stones(2), max_stones=4)
        sample = {
            "prev_node_feat": nf, "prev_adj": adj, "prev_mask": mask,
            "prev_is_null": torch.tensor(False),
            "curr_node_feat": nf, "curr_adj": adj, "curr_mask": mask,
            "metadata": torch.zeros(6),
            "accuracy": torch.tensor(0.5),
            "shot_type": torch.tensor(1, dtype=torch.long),
            "end_score": torch.tensor(0.0),
        }
        batch = collate_fn([sample, sample])
        assert batch["prev_node_feat"].shape[0] == 2
        assert batch["metadata"].shape == (2, 6)
