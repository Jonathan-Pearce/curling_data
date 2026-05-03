"""Training script for the Siamese GNN curling model.

Reads the ``graphs.pkl.gz`` file produced by ``build_graph_dataset.py``,
splits by the ``split`` key (assigned by ``assign_splits``), and trains the
:class:`~ml.model.CurlingGNN` with the
:class:`~ml.loss.HomoscedasticLoss`.

Usage::

    python -m ml.train \\
        --graphs  output/graphs.pkl.gz \\
        --output  output/model.pt \\
        --epochs  50 \\
        --batch-size 32 \\
        --lr 1e-3

The best checkpoint (lowest sum of raw validation losses) is saved to
``--output``.  Per-epoch metrics are printed to stdout.

Importable API::

    from ml.train import (
        CurlingGraphDataset,
        collate_graphs,
        train_one_epoch,
        evaluate,
        train,
    )
"""

from __future__ import annotations

import argparse
import math
import os
import random
from collections import defaultdict
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from ml.build_graph_dataset import (
    N_EDGE_FEATURES,
    N_GRAPH_FEATURES,
    N_NODE_FEATURES,
    load_graphs,
)
from ml.loss import HomoscedasticLoss, compute_loss
from ml.model import CurlingGNN, GraphBatch

__all__ = [
    "CurlingGraphDataset",
    "collate_graphs",
    "train_one_epoch",
    "evaluate",
    "train",
]


# ---------------------------------------------------------------------------
# Dataset and collation
# ---------------------------------------------------------------------------


class CurlingGraphDataset(Dataset):
    """PyTorch ``Dataset`` wrapping a list of graph dicts.

    Parameters
    ----------
    graphs : list[dict]
        Output of :func:`~ml.build_graph_dataset.build_all_graphs`, optionally
        annotated with a ``split`` key by
        :func:`~ml.build_graph_dataset.assign_splits`.
    """

    def __init__(self, graphs: list[dict]) -> None:
        self.graphs = graphs

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> dict:
        return self.graphs[idx]


def collate_graphs(
    graph_list: list[dict],
) -> tuple[GraphBatch, dict[str, torch.Tensor]]:
    """Collate a list of graph dicts into a :class:`GraphBatch` + labels.

    Node feature tensors from all graphs are concatenated along the node
    dimension.  Edge indices are offset by the cumulative node count so
    that they index into the concatenated node matrix.  A ``batch`` vector
    maps each node back to its graph index.

    Parameters
    ----------
    graph_list : list[dict]
        Output of :func:`~ml.build_graph_dataset.build_shot_graph`.

    Returns
    -------
    batch : GraphBatch
    labels : dict with tensors
        ``y_accuracy`` (B,), ``y_shot_type`` (B,),
        ``y_team1_score`` (B,), ``y_team2_score`` (B,).
    """
    B = len(graph_list)

    x_t_list: list[torch.Tensor] = []
    x_prev_list: list[torch.Tensor] = []
    ei_t_list: list[torch.Tensor] = []
    ea_t_list: list[torch.Tensor] = []
    ei_p_list: list[torch.Tensor] = []
    ea_p_list: list[torch.Tensor] = []
    bt_list: list[torch.Tensor] = []
    bp_list: list[torch.Tensor] = []
    u_list: list[torch.Tensor] = []

    n_t_cum = 0  # cumulative S_t  node count (for edge-index offsetting)
    n_p_cum = 0  # cumulative S_{t-1} node count

    for i, g in enumerate(graph_list):
        xt = torch.as_tensor(g["x_t"], dtype=torch.float32)
        xp = torch.as_tensor(g["x_prev"], dtype=torch.float32)
        n_t = xt.size(0)
        n_p = xp.size(0)

        x_t_list.append(xt)
        x_prev_list.append(xp)

        ei_t = torch.as_tensor(g["edge_index_t"], dtype=torch.long)
        ea_t = torch.as_tensor(g["edge_attr_t"], dtype=torch.float32)
        if ei_t.size(1) > 0:
            ei_t = ei_t + n_t_cum

        ei_p = torch.as_tensor(g["edge_index_prev"], dtype=torch.long)
        ea_p = torch.as_tensor(g["edge_attr_prev"], dtype=torch.float32)
        if ei_p.size(1) > 0:
            ei_p = ei_p + n_p_cum

        ei_t_list.append(ei_t)
        ea_t_list.append(ea_t)
        ei_p_list.append(ei_p)
        ea_p_list.append(ea_p)

        bt_list.append(torch.full((n_t,), i, dtype=torch.long))
        bp_list.append(torch.full((n_p,), i, dtype=torch.long))

        u_list.append(torch.as_tensor(g["u"], dtype=torch.float32))

        n_t_cum += n_t
        n_p_cum += n_p

    # Concatenate along the node / edge dimension
    x_t_all   = torch.cat(x_t_list,   dim=0)  # (total_nodes_t, F_node)
    x_prev_all = torch.cat(x_prev_list, dim=0)  # (total_nodes_p, F_node)

    ei_t_all = torch.cat(ei_t_list, dim=1)  # (2, total_edges_t)
    ea_t_all = torch.cat(ea_t_list, dim=0)  # (total_edges_t, F_edge)
    ei_p_all = torch.cat(ei_p_list, dim=1)  # (2, total_edges_p)
    ea_p_all = torch.cat(ea_p_list, dim=0)  # (total_edges_p, F_edge)

    bt_all = torch.cat(bt_list, dim=0)  # (total_nodes_t,)
    bp_all = torch.cat(bp_list, dim=0)  # (total_nodes_p,)

    u_all = torch.stack(u_list, dim=0)  # (B, N_GRAPH_FEATURES)

    graph_batch = GraphBatch(
        x_t=x_t_all,
        edge_index_t=ei_t_all,
        edge_attr_t=ea_t_all,
        batch_t=bt_all,
        x_prev=x_prev_all,
        edge_index_prev=ei_p_all,
        edge_attr_prev=ea_p_all,
        batch_prev=bp_all,
        u=u_all,
        batch_size=B,
    )

    def _safe_float(val: object) -> float:
        """Convert *val* to float; return NaN on failure."""
        try:
            v = float(val)  # type: ignore[arg-type]
            return v
        except (TypeError, ValueError):
            return float("nan")

    labels: dict[str, torch.Tensor] = {
        "y_accuracy": torch.tensor(
            [_safe_float(g["accuracy"]) for g in graph_list],
            dtype=torch.float32,
        ),
        "y_shot_type": torch.tensor(
            [int(g["shot_type_idx"]) for g in graph_list],
            dtype=torch.long,
        ),
        "y_team1_score": torch.tensor(
            [_safe_float(g["team1_score_this_end"]) for g in graph_list],
            dtype=torch.float32,
        ),
        "y_team2_score": torch.tensor(
            [_safe_float(g["team2_score_this_end"]) for g in graph_list],
            dtype=torch.float32,
        ),
    }

    return graph_batch, labels


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_one_epoch(
    model: CurlingGNN,
    homo_loss: HomoscedasticLoss,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    """Run one full training epoch.

    Returns
    -------
    dict[str, float]
        Average ``total_loss``, ``loss_accuracy``, ``loss_end_score``,
        ``loss_shot_type`` over all batches.
    """
    model.train()
    homo_loss.train()
    totals: dict[str, float] = defaultdict(float)
    n_batches = 0

    for graph_batch, labels in loader:
        graph_batch = graph_batch.to(device)
        labels = {k: v.to(device) for k, v in labels.items()}

        optimizer.zero_grad()
        preds = model(graph_batch)
        loss, metrics = compute_loss(preds, labels, homo_loss)
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(homo_loss.parameters()),
            max_norm=1.0,
        )
        optimizer.step()

        totals["total_loss"] += loss.item()
        for k, v in metrics.items():
            totals[k] += v
        n_batches += 1

    if n_batches == 0:
        return {}
    return {k: v / n_batches for k, v in totals.items()}


@torch.no_grad()
def evaluate(
    model: CurlingGNN,
    homo_loss: HomoscedasticLoss,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate the model on a data loader and return a metrics dict.

    Metrics
    -------
    loss_accuracy, loss_end_score, loss_shot_type
        Raw per-task MSE / cross-entropy values.
    mae_accuracy
        Mean absolute error of shot accuracy prediction (percentage points).
    mae_end_score
        MAE of end-score prediction (averaged over team1 and team2).
    acc_shot_type
        Top-1 accuracy for shot-type classification.
    """
    model.eval()
    homo_loss.eval()
    totals: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)

    for graph_batch, labels in loader:
        graph_batch = graph_batch.to(device)
        labels = {k: v.to(device) for k, v in labels.items()}

        preds = model(graph_batch)
        _, metrics = compute_loss(preds, labels, homo_loss)

        B = graph_batch.batch_size
        for k, v in metrics.items():
            totals[k] += v * B
            counts[k] += B

        # MAE — accuracy
        y_acc = labels["y_accuracy"]
        acc_mask = ~torch.isnan(y_acc)
        if acc_mask.any():
            mae_acc = (
                (preds["accuracy"].squeeze(-1)[acc_mask] - y_acc[acc_mask])
                .abs()
                .mean()
                .item()
            )
            n = int(acc_mask.sum().item())
            totals["mae_accuracy"] += mae_acc * n
            counts["mae_accuracy"] += n

        # MAE — end scores
        y_t1, y_t2 = labels["y_team1_score"], labels["y_team2_score"]
        sc_mask = ~(torch.isnan(y_t1) | torch.isnan(y_t2))
        if sc_mask.any():
            ps = preds["end_scores"][sc_mask]
            ts = torch.stack([y_t1[sc_mask], y_t2[sc_mask]], dim=-1)
            mae_sc = (ps - ts).abs().mean().item()
            n = int(sc_mask.sum().item())
            totals["mae_end_score"] += mae_sc * n
            counts["mae_end_score"] += n

        # Top-1 accuracy — shot type
        y_type = labels["y_shot_type"]
        tp_mask = y_type >= 0
        if tp_mask.any():
            correct = (
                preds["shot_type"][tp_mask].argmax(-1) == y_type[tp_mask]
            ).sum().item()
            n = int(tp_mask.sum().item())
            totals["acc_shot_type"] += correct
            counts["acc_shot_type"] += n

    result: dict[str, float] = {}
    for k in totals:
        c = counts.get(k, 0)
        result[k] = totals[k] / c if c > 0 else float("nan")
    return result


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def train(
    graphs_path: str,
    output_path: str,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    node_hidden: int = 64,
    graph_out: int = 128,
    head_hidden: int = 128,
    n_gnn_layers: int = 3,
    dropout: float = 0.1,
    seed: int = 42,
    device_str: str = "cpu",
) -> CurlingGNN:
    """Train the Siamese GNN model and save the best checkpoint.

    Graphs are split by the ``split`` field: ``"train"``, ``"val"``, or
    ``"test"``.  The checkpoint with the lowest sum of raw validation losses
    (accuracy MSE + end-score MSE + shot-type CE) is saved.

    Parameters
    ----------
    graphs_path : str
        Path to ``graphs.pkl.gz`` produced by ``build_graph_dataset.py``.
    output_path : str
        Destination for the saved ``.pt`` checkpoint.
    epochs : int
    batch_size : int
    lr : float
        Initial learning rate (Adam optimizer).
    weight_decay : float
    node_hidden : int
        Hidden dimension in GNN layers.
    graph_out : int
        Output dimension of ``GraphEncoder``.
    head_hidden : int
        Hidden dimension in prediction heads and ``ChangeNet``.
    n_gnn_layers : int
        Number of GNN message-passing layers.
    dropout : float
    seed : int
    device_str : str
        PyTorch device string, e.g. ``"cpu"`` or ``"cuda:0"``.

    Returns
    -------
    CurlingGNN
        The trained model loaded with the best checkpoint weights.
    """
    _set_seed(seed)
    device = torch.device(device_str)

    print(f"Loading graphs from {graphs_path} …")
    all_graphs = load_graphs(graphs_path)
    print(f"  {len(all_graphs):,} graphs loaded.")

    train_graphs = [g for g in all_graphs if g.get("split", "train") == "train"]
    val_graphs = [g for g in all_graphs if g.get("split") == "val"]
    print(f"  {len(train_graphs):,} train  /  {len(val_graphs):,} val")

    train_ds = CurlingGraphDataset(train_graphs)
    val_ds = CurlingGraphDataset(val_graphs)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_graphs,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_graphs,
        num_workers=0,
    )

    model = CurlingGNN(
        node_hidden=node_hidden,
        graph_out=graph_out,
        head_hidden=head_hidden,
        n_gnn_layers=n_gnn_layers,
        dropout=dropout,
    ).to(device)

    homo_loss = HomoscedasticLoss(n_tasks=3).to(device)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(homo_loss.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )

    best_val_loss: float = math.inf
    best_state: Optional[dict] = None

    for epoch in range(1, epochs + 1):
        tr_metrics = train_one_epoch(model, homo_loss, train_loader, optimizer, device)
        val_metrics = evaluate(model, homo_loss, val_loader, device)
        scheduler.step()

        val_total = sum(
            val_metrics.get(k, 0.0)
            for k in ("loss_accuracy", "loss_end_score", "loss_shot_type")
        )
        if val_total < best_val_loss:
            best_val_loss = val_total
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            tr_total = tr_metrics.get("total_loss", float("nan"))
            va_mae_acc = val_metrics.get("mae_accuracy", float("nan"))
            va_mae_sc = val_metrics.get("mae_end_score", float("nan"))
            va_type_acc = val_metrics.get("acc_shot_type", float("nan"))
            weights = homo_loss.task_weights.tolist()
            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"train_loss={tr_total:.4f} | "
                f"val_mae_acc={va_mae_acc:.2f} | "
                f"val_mae_score={va_mae_sc:.3f} | "
                f"val_type_acc={va_type_acc:.3f} | "
                f"σ²={[f'{w:.3f}' for w in weights]}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "homo_loss_state_dict": homo_loss.state_dict(),
            "config": {
                "node_hidden": node_hidden,
                "graph_out": graph_out,
                "head_hidden": head_hidden,
                "n_gnn_layers": n_gnn_layers,
                "dropout": dropout,
            },
        },
        output_path,
    )
    print(f"Model saved to {output_path}")
    return model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the Siamese GNN curling model."
    )
    parser.add_argument(
        "--graphs",
        default=os.path.join("output", "graphs.pkl.gz"),
        help="Path to graphs.pkl.gz (default: output/graphs.pkl.gz)",
    )
    parser.add_argument(
        "--output",
        default=os.path.join("output", "model.pt"),
        help="Output checkpoint path (default: output/model.pt)",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--node-hidden", type=int, default=64)
    parser.add_argument("--graph-out", type=int, default=128)
    parser.add_argument("--head-hidden", type=int, default=128)
    parser.add_argument("--gnn-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="cpu",
        help="PyTorch device string, e.g. 'cpu' or 'cuda:0' (default: cpu)",
    )
    args = parser.parse_args()

    train(
        graphs_path=args.graphs,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        node_hidden=args.node_hidden,
        graph_out=args.graph_out,
        head_hidden=args.head_hidden,
        n_gnn_layers=args.gnn_layers,
        dropout=args.dropout,
        seed=args.seed,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
