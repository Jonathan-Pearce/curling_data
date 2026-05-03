"""Unit tests for the Siamese GNN: model.py, loss.py, and train.py."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from ml.build_graph_dataset import N_EDGE_FEATURES, N_GRAPH_FEATURES, N_NODE_FEATURES
from ml.loss import HomoscedasticLoss, compute_loss
from ml.model import (
    CurlingGNN,
    GNNLayer,
    GraphBatch,
    GraphEncoder,
    MLP,
    N_SHOT_TYPES,
)
from ml.train import CurlingGraphDataset, collate_graphs


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_batch(
    n_t: int = 4,
    n_prev: int = 3,
    n_edges_t: int = 4,
    n_edges_prev: int = 2,
    B: int = 1,
    seed: int = 0,
) -> GraphBatch:
    """Create a minimal ``GraphBatch`` for unit tests."""
    torch.manual_seed(seed)
    x_t = torch.randn(n_t, N_NODE_FEATURES)
    x_prev = torch.randn(n_prev, N_NODE_FEATURES)

    if n_edges_t > 0 and n_t >= 2:
        ei_t = torch.randint(0, n_t, (2, n_edges_t))
        ea_t = torch.randn(n_edges_t, N_EDGE_FEATURES)
    else:
        ei_t = torch.zeros(2, 0, dtype=torch.long)
        ea_t = torch.zeros(0, N_EDGE_FEATURES)

    if n_edges_prev > 0 and n_prev >= 2:
        ei_p = torch.randint(0, n_prev, (2, n_edges_prev))
        ea_p = torch.randn(n_edges_prev, N_EDGE_FEATURES)
    else:
        ei_p = torch.zeros(2, 0, dtype=torch.long)
        ea_p = torch.zeros(0, N_EDGE_FEATURES)

    return GraphBatch(
        x_t=x_t,
        edge_index_t=ei_t,
        edge_attr_t=ea_t,
        batch_t=torch.zeros(n_t, dtype=torch.long),
        x_prev=x_prev,
        edge_index_prev=ei_p,
        edge_attr_prev=ea_p,
        batch_prev=torch.zeros(n_prev, dtype=torch.long),
        u=torch.randn(B, N_GRAPH_FEATURES),
        batch_size=B,
    )


def _make_graph_dict(
    n_t: int = 3,
    n_prev: int = 2,
    shot_type_idx: int = 0,
    accuracy: float = 75.0,
    t1_score: float = 2.0,
    t2_score: float = 1.0,
    seed: int = 0,
) -> dict:
    """Create a minimal graph dict (like ``build_shot_graph`` output)."""
    np.random.seed(seed)
    return {
        "x_t": np.random.randn(n_t, N_NODE_FEATURES).astype(np.float32),
        "x_prev": np.random.randn(n_prev, N_NODE_FEATURES).astype(np.float32),
        "edge_index_t": np.zeros((2, 0), dtype=np.int64),
        "edge_attr_t": np.zeros((0, N_EDGE_FEATURES), dtype=np.float32),
        "edge_index_prev": np.zeros((2, 0), dtype=np.int64),
        "edge_attr_prev": np.zeros((0, N_EDGE_FEATURES), dtype=np.float32),
        "edge_index_temp": np.zeros((2, 0), dtype=np.int64),
        "u": np.random.randn(N_GRAPH_FEATURES).astype(np.float32),
        "shot_type_idx": shot_type_idx,
        "accuracy": accuracy,
        "team1_score_this_end": t1_score,
        "team2_score_this_end": t2_score,
        "event_id": 1,
        "match_id": 1,
        "end_number": 1,
        "shot_number": 1,
    }


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------


class TestMLP:

    def test_output_shape(self):
        mlp = MLP([16, 32, 8])
        x = torch.randn(5, 16)
        assert mlp(x).shape == (5, 8)

    def test_two_layer_shape(self):
        mlp = MLP([10, 20])
        x = torch.randn(3, 10)
        assert mlp(x).shape == (3, 20)

    def test_activate_last_adds_relu(self):
        """With activate_last=True, outputs are non-negative (ReLU applied)."""
        torch.manual_seed(0)
        mlp = MLP([4, 8], activate_last=True)
        x = torch.randn(100, 4)
        out = mlp(x)
        assert (out >= 0).all(), "ReLU not applied after last linear layer"

    def test_no_activate_last_allows_negative(self):
        """Without activate_last, negative outputs are possible."""
        torch.manual_seed(0)
        mlp = MLP([4, 8], activate_last=False)
        x = torch.randn(100, 4)
        out = mlp(x)
        assert (out < 0).any(), "Expected some negative values without final activation"

    def test_dropout_reduces_variance_in_training(self):
        """With high dropout, repeated passes on a 3-layer MLP should differ."""
        mlp = MLP([8, 16, 16], dropout=0.9)
        mlp.train()
        x = torch.ones(1, 8)
        out1 = mlp(x)
        out2 = mlp(x)
        # At least one output should differ due to stochastic dropout
        assert not torch.allclose(out1, out2)

    def test_gradient_flows(self):
        mlp = MLP([4, 8, 1])
        x = torch.randn(3, 4, requires_grad=False)
        out = mlp(x).sum()
        out.backward()
        for p in mlp.parameters():
            assert p.grad is not None


# ---------------------------------------------------------------------------
# GNNLayer
# ---------------------------------------------------------------------------


class TestGNNLayer:

    def test_output_shape_same_dims(self):
        layer = GNNLayer(node_in=16, node_out=16, edge_in=3)
        x = torch.randn(5, 16)
        ei = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        ea = torch.randn(3, 3)
        out = layer(x, ei, ea)
        assert out.shape == (5, 16)

    def test_output_shape_different_dims(self):
        layer = GNNLayer(node_in=10, node_out=32, edge_in=3)
        x = torch.randn(4, 10)
        ei = torch.zeros(2, 0, dtype=torch.long)
        ea = torch.zeros(0, 3)
        out = layer(x, ei, ea)
        assert out.shape == (4, 32)

    def test_empty_graph_returns_zero_shape(self):
        layer = GNNLayer(node_in=16, node_out=16, edge_in=3)
        x = torch.zeros(0, 16)
        ei = torch.zeros(2, 0, dtype=torch.long)
        ea = torch.zeros(0, 3)
        out = layer(x, ei, ea)
        assert out.shape == (0, 16)

    def test_no_edges_works(self):
        """Nodes with no edges should still produce valid output."""
        layer = GNNLayer(node_in=8, node_out=8, edge_in=3)
        x = torch.randn(3, 8)
        ei = torch.zeros(2, 0, dtype=torch.long)
        ea = torch.zeros(0, 3)
        out = layer(x, ei, ea)
        assert out.shape == (3, 8)
        assert not torch.isnan(out).any()

    def test_gradient_flows_through_edges(self):
        layer = GNNLayer(node_in=8, node_out=8, edge_in=3)
        x = torch.randn(4, 8, requires_grad=True)
        ei = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        ea = torch.randn(2, 3)
        out = layer(x, ei, ea).sum()
        out.backward()
        assert x.grad is not None

    def test_out_channels_property(self):
        layer = GNNLayer(node_in=10, node_out=24, edge_in=3)
        assert layer.out_channels == 24


# ---------------------------------------------------------------------------
# GraphEncoder
# ---------------------------------------------------------------------------


class TestGraphEncoder:

    def test_output_shape(self):
        enc = GraphEncoder(out_dim=64)
        x = torch.randn(5, N_NODE_FEATURES)
        ei = torch.zeros(2, 0, dtype=torch.long)
        ea = torch.zeros(0, N_EDGE_FEATURES)
        batch = torch.zeros(5, dtype=torch.long)
        out = enc(x, ei, ea, batch, batch_size=1)
        assert out.shape == (1, 64)

    def test_empty_graph_returns_zeros(self):
        enc = GraphEncoder(out_dim=64)
        x = torch.zeros(0, N_NODE_FEATURES)
        ei = torch.zeros(2, 0, dtype=torch.long)
        ea = torch.zeros(0, N_EDGE_FEATURES)
        batch = torch.zeros(0, dtype=torch.long)
        out = enc(x, ei, ea, batch, batch_size=2)
        assert out.shape == (2, 64)
        assert torch.all(out == 0.0)

    def test_batch_of_two_independent_outputs(self):
        """Two graphs in a batch should produce independent embeddings."""
        enc = GraphEncoder(out_dim=32)
        # Graph 0: 2 nodes; Graph 1: 3 nodes
        x = torch.randn(5, N_NODE_FEATURES)
        batch = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long)
        ei = torch.zeros(2, 0, dtype=torch.long)
        ea = torch.zeros(0, N_EDGE_FEATURES)
        out = enc(x, ei, ea, batch, batch_size=2)
        assert out.shape == (2, 32)

    def test_mixed_batch_empty_and_nonempty(self):
        """Graph 0 has nodes, graph 1 is empty — graph 1 embedding must be zero."""
        enc = GraphEncoder(out_dim=32)
        x = torch.randn(2, N_NODE_FEATURES)
        batch = torch.tensor([0, 0], dtype=torch.long)
        ei = torch.zeros(2, 0, dtype=torch.long)
        ea = torch.zeros(0, N_EDGE_FEATURES)
        out = enc(x, ei, ea, batch, batch_size=2)
        assert out.shape == (2, 32)
        # Graph 1 has no nodes: output must be exactly zero (masking enforced in encoder)
        assert torch.all(out[1] == 0.0), (
            "Empty graph should produce a zero embedding to avoid bias leakage"
        )

    def test_out_dim_property(self):
        enc = GraphEncoder(out_dim=96)
        assert enc.out_dim == 96

    def test_gradient_flows(self):
        enc = GraphEncoder(out_dim=16)
        x = torch.randn(3, N_NODE_FEATURES, requires_grad=True)
        ei = torch.zeros(2, 0, dtype=torch.long)
        ea = torch.zeros(0, N_EDGE_FEATURES)
        batch = torch.zeros(3, dtype=torch.long)
        out = enc(x, ei, ea, batch, batch_size=1).sum()
        out.backward()
        assert x.grad is not None


# ---------------------------------------------------------------------------
# GraphBatch
# ---------------------------------------------------------------------------


class TestGraphBatch:

    def test_to_method_moves_tensors(self):
        batch = _make_batch(n_t=3, n_prev=2, B=1)
        moved = batch.to(torch.device("cpu"))
        assert moved.x_t.device == batch.x_t.device
        assert moved.batch_size == batch.batch_size

    def test_batch_size_preserved(self):
        batch = _make_batch(B=4, n_t=8, n_prev=6)
        assert batch.batch_size == 4


# ---------------------------------------------------------------------------
# CurlingGNN
# ---------------------------------------------------------------------------


class TestCurlingGNN:

    def _model(self, **kwargs) -> CurlingGNN:
        defaults = dict(node_hidden=16, graph_out=32, head_hidden=32, n_gnn_layers=2, dropout=0.0)
        defaults.update(kwargs)
        return CurlingGNN(**defaults)

    def test_forward_output_keys(self):
        model = self._model()
        batch = _make_batch(n_t=3, n_prev=2, B=1)
        out = model(batch)
        assert set(out.keys()) == {"accuracy", "shot_type", "end_scores"}

    def test_accuracy_shape(self):
        model = self._model()
        batch = _make_batch(n_t=3, n_prev=2, B=2)
        out = model(batch)
        assert out["accuracy"].shape == (2, 1)

    def test_shot_type_shape(self):
        model = self._model()
        batch = _make_batch(n_t=3, n_prev=2, B=2)
        out = model(batch)
        assert out["shot_type"].shape == (2, N_SHOT_TYPES)

    def test_end_scores_shape(self):
        model = self._model()
        batch = _make_batch(n_t=3, n_prev=2, B=2)
        out = model(batch)
        assert out["end_scores"].shape == (2, 2)

    def test_first_shot_empty_prev(self):
        """Empty S_{t-1} (shot_number=1) should not crash and produce valid output."""
        model = self._model()
        # S_{t-1} has 0 nodes
        batch = _make_batch(n_t=3, n_prev=0, n_edges_prev=0, B=1)
        out = model(batch)
        assert not torch.isnan(out["accuracy"]).any()
        assert not torch.isnan(out["shot_type"]).any()
        assert not torch.isnan(out["end_scores"]).any()

    def test_gradient_flows_from_all_heads(self):
        """Backward pass should populate gradients for all encoder parameters."""
        model = self._model()
        batch = _make_batch(n_t=3, n_prev=2, B=2)
        out = model(batch)
        loss = out["accuracy"].sum() + out["shot_type"].sum() + out["end_scores"].sum()
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_shared_encoder_same_weights(self):
        """The Siamese encoder must have exactly one copy of weights."""
        model = self._model()
        # There should be only one 'encoder' sub-module, not two
        encoder_modules = [n for n, _ in model.named_modules() if n == "encoder"]
        assert len(encoder_modules) == 1

    def test_batch_size_one_vs_two_consistent(self):
        """Processing two graphs together should give same results as separately."""
        torch.manual_seed(42)
        model = self._model()
        model.eval()

        batch1 = _make_batch(n_t=2, n_prev=1, B=1, seed=1)
        batch2 = _make_batch(n_t=3, n_prev=2, B=1, seed=2)

        out1 = model(batch1)
        out2 = model(batch2)

        # Build a combined batch manually
        u_combined = torch.cat([batch1.u, batch2.u], dim=0)
        combined = GraphBatch(
            x_t=torch.cat([batch1.x_t, batch2.x_t]),
            edge_index_t=torch.cat(
                [batch1.edge_index_t,
                 batch2.edge_index_t + batch1.x_t.size(0)], dim=1
            ),
            edge_attr_t=torch.cat([batch1.edge_attr_t, batch2.edge_attr_t]),
            batch_t=torch.cat([
                torch.zeros(batch1.x_t.size(0), dtype=torch.long),
                torch.ones(batch2.x_t.size(0), dtype=torch.long),
            ]),
            x_prev=torch.cat([batch1.x_prev, batch2.x_prev]),
            edge_index_prev=torch.cat(
                [batch1.edge_index_prev,
                 batch2.edge_index_prev + batch1.x_prev.size(0)], dim=1
            ),
            edge_attr_prev=torch.cat([batch1.edge_attr_prev, batch2.edge_attr_prev]),
            batch_prev=torch.cat([
                torch.zeros(batch1.x_prev.size(0), dtype=torch.long),
                torch.ones(batch2.x_prev.size(0), dtype=torch.long),
            ]),
            u=u_combined,
            batch_size=2,
        )
        out_combined = model(combined)

        # Results for graph 0 in combined batch should match single-graph result
        assert torch.allclose(out_combined["accuracy"][0], out1["accuracy"][0], atol=1e-5)
        assert torch.allclose(out_combined["shot_type"][0], out1["shot_type"][0], atol=1e-5)
        assert torch.allclose(out_combined["end_scores"][0], out1["end_scores"][0], atol=1e-5)


# ---------------------------------------------------------------------------
# HomoscedasticLoss
# ---------------------------------------------------------------------------


class TestHomoscedasticLoss:

    def test_output_is_scalar(self):
        loss_fn = HomoscedasticLoss(n_tasks=3)
        l0 = torch.tensor(1.0, requires_grad=True)
        l1 = torch.tensor(2.0, requires_grad=True)
        l2 = torch.tensor(0.5, requires_grad=True)
        total = loss_fn(l0, l1, l2)
        assert total.shape == ()

    def test_initial_weights_are_one(self):
        """All log_vars initialised to 0 → σ² = exp(0) = 1."""
        loss_fn = HomoscedasticLoss(n_tasks=3)
        weights = loss_fn.task_weights
        assert torch.allclose(weights, torch.ones(3), atol=1e-6)

    def test_log_vars_are_learnable(self):
        loss_fn = HomoscedasticLoss(n_tasks=3)
        assert loss_fn.log_vars.requires_grad

    def test_backward_updates_log_vars(self):
        loss_fn = HomoscedasticLoss(n_tasks=3)
        optimizer = torch.optim.SGD(loss_fn.parameters(), lr=0.1)
        l0 = torch.tensor(2.0)
        l1 = torch.tensor(3.0)
        l2 = torch.tensor(1.5)
        total = loss_fn(l0, l1, l2)
        total.backward()
        optimizer.step()
        # log_vars should have changed from their initial value of 0
        assert not torch.allclose(loss_fn.log_vars, torch.zeros(3))

    def test_task_weights_property_is_detached(self):
        loss_fn = HomoscedasticLoss(n_tasks=3)
        weights = loss_fn.task_weights
        assert not weights.requires_grad

    def test_zero_losses_yields_finite_total(self):
        loss_fn = HomoscedasticLoss(n_tasks=3)
        total = loss_fn(
            torch.tensor(0.0),
            torch.tensor(0.0),
            torch.tensor(0.0),
        )
        assert torch.isfinite(total)


# ---------------------------------------------------------------------------
# compute_loss
# ---------------------------------------------------------------------------


class TestComputeLoss:

    def _preds(self, B: int = 4) -> dict[str, torch.Tensor]:
        torch.manual_seed(0)
        return {
            "accuracy": torch.randn(B, 1),
            "shot_type": torch.randn(B, N_SHOT_TYPES),
            "end_scores": torch.randn(B, 2),
        }

    def _labels(
        self,
        B: int = 4,
        acc_nan: bool = False,
        type_unknown: bool = False,
        score_nan: bool = False,
    ) -> dict[str, torch.Tensor]:
        y_acc = torch.full((B,), float("nan") if acc_nan else 75.0)
        y_type = torch.full((B,), -1 if type_unknown else 2, dtype=torch.long)
        y_t1 = torch.full((B,), float("nan") if score_nan else 2.0)
        y_t2 = torch.full((B,), float("nan") if score_nan else 1.0)
        return {
            "y_accuracy": y_acc,
            "y_shot_type": y_type,
            "y_team1_score": y_t1,
            "y_team2_score": y_t2,
        }

    def test_returns_scalar_and_metrics_dict(self):
        loss_fn = HomoscedasticLoss()
        total, metrics = compute_loss(self._preds(), self._labels(), loss_fn)
        assert total.shape == ()
        assert set(metrics.keys()) == {"loss_accuracy", "loss_end_score", "loss_shot_type"}

    def test_all_nan_accuracy_does_not_crash(self):
        loss_fn = HomoscedasticLoss()
        total, metrics = compute_loss(
            self._preds(), self._labels(acc_nan=True), loss_fn
        )
        assert torch.isfinite(total)

    def test_all_unknown_shot_type_does_not_crash(self):
        loss_fn = HomoscedasticLoss()
        total, metrics = compute_loss(
            self._preds(), self._labels(type_unknown=True), loss_fn
        )
        assert torch.isfinite(total)

    def test_all_nan_scores_does_not_crash(self):
        loss_fn = HomoscedasticLoss()
        total, metrics = compute_loss(
            self._preds(), self._labels(score_nan=True), loss_fn
        )
        assert torch.isfinite(total)

    def test_backward_works(self):
        loss_fn = HomoscedasticLoss()
        preds = self._preds()
        labels = self._labels()
        total, _ = compute_loss(preds, labels, loss_fn)
        assert total.grad_fn is not None, "total loss should be part of autograd graph"
        total.backward()
        # HomoscedasticLoss log_vars must receive gradients
        assert loss_fn.log_vars.grad is not None

    def test_loss_accuracy_reflects_mse(self):
        """When predicted accuracy equals target, MSE should be ~0."""
        loss_fn = HomoscedasticLoss()
        B = 4
        preds = {
            "accuracy": torch.full((B, 1), 75.0),
            "shot_type": torch.randn(B, N_SHOT_TYPES),
            "end_scores": torch.randn(B, 2),
        }
        labels = {
            "y_accuracy": torch.full((B,), 75.0),
            "y_shot_type": torch.ones(B, dtype=torch.long),
            "y_team1_score": torch.full((B,), 2.0),
            "y_team2_score": torch.full((B,), 1.0),
        }
        _, metrics = compute_loss(preds, labels, loss_fn)
        assert metrics["loss_accuracy"] == pytest.approx(0.0, abs=1e-5)

    def test_partial_nan_accuracy_uses_valid_samples(self):
        """Only non-NaN accuracy rows should contribute to the accuracy loss."""
        loss_fn = HomoscedasticLoss()
        B = 4
        y_acc = torch.tensor([float("nan"), 80.0, float("nan"), 60.0])
        preds = {
            "accuracy": torch.tensor([[80.0], [80.0], [60.0], [60.0]]),
            "shot_type": torch.randn(B, N_SHOT_TYPES),
            "end_scores": torch.randn(B, 2),
        }
        labels = {
            "y_accuracy": y_acc,
            "y_shot_type": torch.ones(B, dtype=torch.long),
            "y_team1_score": torch.ones(B),
            "y_team2_score": torch.zeros(B),
        }
        _, metrics = compute_loss(preds, labels, loss_fn)
        # Only rows 1 and 3 are valid: pred[1]=80 vs 80 → 0; pred[3]=60 vs 60 → 0
        assert metrics["loss_accuracy"] == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# collate_graphs
# ---------------------------------------------------------------------------


class TestCollateGraphs:

    def test_returns_graph_batch_and_labels(self):
        graphs = [_make_graph_dict(seed=i) for i in range(3)]
        batch, labels = collate_graphs(graphs)
        assert isinstance(batch, GraphBatch)
        assert batch.batch_size == 3

    def test_label_shapes(self):
        B = 4
        graphs = [_make_graph_dict(seed=i) for i in range(B)]
        batch, labels = collate_graphs(graphs)
        for key in ("y_accuracy", "y_shot_type", "y_team1_score", "y_team2_score"):
            assert key in labels
            assert labels[key].shape == (B,)

    def test_node_concatenation_shape(self):
        g1 = _make_graph_dict(n_t=2, n_prev=1, seed=0)
        g2 = _make_graph_dict(n_t=3, n_prev=2, seed=1)
        batch, _ = collate_graphs([g1, g2])
        assert batch.x_t.shape == (5, N_NODE_FEATURES)
        assert batch.x_prev.shape == (3, N_NODE_FEATURES)

    def test_batch_vectors_correct(self):
        g1 = _make_graph_dict(n_t=2, n_prev=0, seed=0)
        g2 = _make_graph_dict(n_t=3, n_prev=2, seed=1)
        batch, _ = collate_graphs([g1, g2])
        # batch_t: first 2 nodes → 0, last 3 → 1
        assert batch.batch_t.tolist() == [0, 0, 1, 1, 1]
        # batch_prev: first 0 nodes → (none), last 2 → 1
        assert batch.batch_prev.tolist() == [1, 1]

    def test_u_shape(self):
        B = 3
        graphs = [_make_graph_dict(seed=i) for i in range(B)]
        batch, _ = collate_graphs(graphs)
        assert batch.u.shape == (B, N_GRAPH_FEATURES)

    def test_nan_accuracy_preserved(self):
        g = _make_graph_dict(accuracy=float("nan"), seed=0)
        _, labels = collate_graphs([g])
        assert math.isnan(labels["y_accuracy"][0].item())

    def test_shot_type_idx_minus_one_preserved(self):
        g = _make_graph_dict(shot_type_idx=-1, seed=0)
        _, labels = collate_graphs([g])
        assert labels["y_shot_type"][0].item() == -1

    def test_edge_index_offset_correct(self):
        """Edge indices in second graph should be offset by number of nodes in first."""
        # g1: 3 nodes, 1 edge 0→1
        g1 = _make_graph_dict(n_t=3, n_prev=0, seed=0)
        g1["edge_index_t"] = np.array([[0], [1]], dtype=np.int64)
        g1["edge_attr_t"] = np.zeros((1, N_EDGE_FEATURES), dtype=np.float32)

        # g2: 2 nodes, 1 edge 0→1 (should become 3→4 after offset)
        g2 = _make_graph_dict(n_t=2, n_prev=0, seed=1)
        g2["edge_index_t"] = np.array([[0], [1]], dtype=np.int64)
        g2["edge_attr_t"] = np.zeros((1, N_EDGE_FEATURES), dtype=np.float32)

        batch, _ = collate_graphs([g1, g2])
        # Combined edge_index_t should be [[0, 3], [1, 4]]
        ei = batch.edge_index_t
        assert ei.shape == (2, 2)
        assert ei[0, 0].item() == 0
        assert ei[1, 0].item() == 1
        assert ei[0, 1].item() == 3
        assert ei[1, 1].item() == 4

    def test_empty_prev_in_batch(self):
        """Graphs with empty S_{t-1} should produce correct batch_prev."""
        g1 = _make_graph_dict(n_t=2, n_prev=0, seed=0)
        g2 = _make_graph_dict(n_t=2, n_prev=0, seed=1)
        batch, _ = collate_graphs([g1, g2])
        assert batch.x_prev.shape == (0, N_NODE_FEATURES)
        assert batch.batch_prev.shape == (0,)


# ---------------------------------------------------------------------------
# CurlingGraphDataset
# ---------------------------------------------------------------------------


class TestCurlingGraphDataset:

    def test_len(self):
        graphs = [_make_graph_dict(seed=i) for i in range(5)]
        ds = CurlingGraphDataset(graphs)
        assert len(ds) == 5

    def test_getitem_returns_dict(self):
        graphs = [_make_graph_dict(seed=0)]
        ds = CurlingGraphDataset(graphs)
        item = ds[0]
        assert isinstance(item, dict)
        assert "x_t" in item


# ---------------------------------------------------------------------------
# Integration: forward pass + loss backward
# ---------------------------------------------------------------------------


class TestIntegration:

    def test_full_forward_and_backward(self):
        """A complete mini-batch: collate → model → loss → backward."""
        torch.manual_seed(42)
        np.random.seed(42)

        # Build graphs with edges so every GNN sub-network receives gradient signal
        graphs = []
        for i in range(4):
            g = _make_graph_dict(n_t=3, n_prev=2, seed=i)
            g["edge_index_t"] = np.array([[0, 1, 2], [1, 2, 0]], dtype=np.int64)
            g["edge_attr_t"] = np.random.randn(3, N_EDGE_FEATURES).astype(np.float32)
            g["edge_index_prev"] = np.array([[0, 1], [1, 0]], dtype=np.int64)
            g["edge_attr_prev"] = np.random.randn(2, N_EDGE_FEATURES).astype(np.float32)
            graphs.append(g)

        batch, labels = collate_graphs(graphs)

        model = CurlingGNN(
            node_hidden=16, graph_out=32, head_hidden=32, n_gnn_layers=2, dropout=0.0
        )
        homo_loss = HomoscedasticLoss(n_tasks=3)

        preds = model(batch)
        total, metrics = compute_loss(preds, labels, homo_loss)

        assert torch.isfinite(total)
        total.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_first_shot_batch_forward(self):
        """All graphs have empty S_{t-1} — first shot of each end."""
        torch.manual_seed(0)
        graphs = [_make_graph_dict(n_t=4, n_prev=0, seed=i) for i in range(3)]
        batch, labels = collate_graphs(graphs)

        model = CurlingGNN(
            node_hidden=16, graph_out=32, head_hidden=32, n_gnn_layers=2, dropout=0.0
        )
        homo_loss = HomoscedasticLoss(n_tasks=3)

        preds = model(batch)
        total, _ = compute_loss(preds, labels, homo_loss)
        assert torch.isfinite(total)

    def test_all_labels_missing(self):
        """All labels are NaN/unknown — loss should still be finite."""
        graphs = [
            _make_graph_dict(
                accuracy=float("nan"),
                shot_type_idx=-1,
                t1_score=float("nan"),
                t2_score=float("nan"),
                seed=i,
            )
            for i in range(2)
        ]
        batch, labels = collate_graphs(graphs)

        model = CurlingGNN(
            node_hidden=16, graph_out=32, head_hidden=32, n_gnn_layers=2, dropout=0.0
        )
        homo_loss = HomoscedasticLoss(n_tasks=3)

        preds = model(batch)
        total, _ = compute_loss(preds, labels, homo_loss)
        assert torch.isfinite(total)
