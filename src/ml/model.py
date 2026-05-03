"""Siamese Graph Neural Network for curling shot analysis.

Architecture
------------

Each shot is represented as a tuple ``(S_{t-1}, S_t)`` where ``S_{t-1}`` is
the board state *before* the shot was thrown and ``S_t`` is the state
*after*.  Both are variable-size graphs (one node per stone on the board).

                 S_{t-1}            S_t
                    │                │
              GraphEncoder     GraphEncoder   ← shared weights (Siamese)
                    │                │
                 e_prev            e_t
                    └──────┬────────┘
                           │  u  (metadata)
                      ChangeNet
                           │
                      change_emb          e_t   u
                    ┌──────┴──────┐         │
              AccHead        TypeHead   ScoreHead
            (accuracy)    (shot_type)  (end_scores)

All computation runs on CPU by default.  Move the model and ``GraphBatch``
to a GPU device by calling ``model.to(device)`` and ``batch.to(device)``
before the forward pass; no device-specific logic is hardcoded.

Importable API::

    from ml.model import CurlingGNN, GraphBatch
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ml.build_graph_dataset import (
    N_EDGE_FEATURES,
    N_GRAPH_FEATURES,
    N_NODE_FEATURES,
    SHOT_TYPE_CLASSES,
)

#: Number of shot-type classes (Classification head output size).
N_SHOT_TYPES: int = len(SHOT_TYPE_CLASSES)

__all__ = [
    "MLP",
    "GNNLayer",
    "GraphEncoder",
    "GraphBatch",
    "CurlingGNN",
    "N_SHOT_TYPES",
]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


class MLP(nn.Module):
    """Fully-connected feed-forward network (Linear → ReLU stacks).

    Parameters
    ----------
    dims : list[int]
        Layer sizes, including input and output.  Must have at least 2 entries.
    dropout : float
        Dropout probability applied after every hidden activation.
    activate_last : bool
        When ``True``, a ReLU is appended after the final linear layer.
    """

    def __init__(
        self,
        dims: list[int],
        dropout: float = 0.0,
        activate_last: bool = False,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            is_last = i == len(dims) - 2
            if not is_last or activate_last:
                layers.append(nn.ReLU())
                if dropout > 0.0:
                    layers.append(nn.Dropout(p=dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.net(x)


def _global_mean_pool(
    x: torch.Tensor,
    batch: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    """Mean-pool node features to graph-level embeddings.

    Parameters
    ----------
    x : torch.Tensor, shape (total_nodes, D)
    batch : torch.Tensor, shape (total_nodes,), dtype long
        Maps each node to its graph index (0-based).
    batch_size : int

    Returns
    -------
    torch.Tensor, shape (batch_size, D)
        Zero vector for graphs with no nodes.
    """
    D = x.size(1)
    out = torch.zeros(batch_size, D, device=x.device, dtype=x.dtype)
    if x.size(0) == 0:
        return out
    count = torch.zeros(batch_size, 1, device=x.device, dtype=x.dtype)
    idx = batch.unsqueeze(1).expand_as(x)
    out.scatter_add_(0, idx, x)
    count.scatter_add_(
        0,
        batch.unsqueeze(1),
        torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype),
    )
    return out / count.clamp(min=1.0)


# ---------------------------------------------------------------------------
# GNN Layer
# ---------------------------------------------------------------------------


class GNNLayer(nn.Module):
    """Message-passing GNN layer with edge features.

    Implements a simplified MPNN (Gilmer et al. 2017) with mean aggregation:

    .. code-block:: text

        message_{ij}  = MLP([h_i, h_j, e_{ij}])
        aggregate_i   = mean({message_{ij} : j ∈ N(i)})
        h_i'          = LayerNorm(MLP([h_i, aggregate_i]) + proj(h_i))

    The residual projection ``proj`` is a ``Linear`` when ``node_in !=
    node_out``, and an identity otherwise.

    Parameters
    ----------
    node_in : int
        Input node-feature dimension.
    node_out : int
        Output node-feature dimension.
    edge_in : int
        Edge-feature dimension.
    hidden : int
        Hidden dimension of the message MLP.
    """

    def __init__(
        self,
        node_in: int,
        node_out: int,
        edge_in: int,
        hidden: int = 64,
    ) -> None:
        super().__init__()
        self._node_out = node_out

        # Message MLP: [src_feat || dst_feat || edge_feat] → message
        self.msg_mlp = MLP(
            [node_in * 2 + edge_in, hidden, node_out],
            activate_last=True,
        )

        # Update MLP: [node_feat || agg_message] → new_node_feat
        self.upd_mlp = MLP([node_in + node_out, node_out])

        self.norm = nn.LayerNorm(node_out)

        # Residual projection when dimensions differ
        self.res_proj: nn.Module = (
            nn.Linear(node_in, node_out, bias=False)
            if node_in != node_out
            else nn.Identity()
        )

    @property
    def out_channels(self) -> int:
        """Output node-feature dimension."""
        return self._node_out

    def forward(
        self,
        x: torch.Tensor,           # (N, node_in)
        edge_index: torch.Tensor,  # (2, E)
        edge_attr: torch.Tensor,   # (E, edge_in)
    ) -> torch.Tensor:
        N = x.size(0)

        if N == 0:
            return torch.zeros(0, self._node_out, device=x.device, dtype=x.dtype)

        if edge_index.size(1) == 0:
            # No edges — aggregate is zero for all nodes
            agg = torch.zeros(N, self._node_out, device=x.device, dtype=x.dtype)
        else:
            src, dst = edge_index[0], edge_index[1]
            msg_input = torch.cat([x[src], x[dst], edge_attr], dim=-1)
            msgs = self.msg_mlp(msg_input)  # (E, node_out)

            # Mean aggregation at destination nodes
            agg = torch.zeros(N, self._node_out, device=x.device, dtype=x.dtype)
            cnt = torch.zeros(N, 1, device=x.device, dtype=x.dtype)
            agg.scatter_add_(0, dst.unsqueeze(1).expand_as(msgs), msgs)
            cnt.scatter_add_(
                0,
                dst.unsqueeze(1),
                torch.ones(dst.size(0), 1, device=x.device, dtype=x.dtype),
            )
            agg = agg / cnt.clamp(min=1.0)

        updated = self.upd_mlp(torch.cat([x, agg], dim=-1))
        return self.norm(updated + self.res_proj(x))


# ---------------------------------------------------------------------------
# Graph Encoder
# ---------------------------------------------------------------------------


class GraphEncoder(nn.Module):
    """Multi-layer GNN with global mean pooling.

    Encodes a variable-size graph into a fixed-size graph-level embedding.
    Empty graphs (zero nodes) produce a zero-filled embedding.

    Parameters
    ----------
    node_in : int
        Input node-feature dimension (default: ``N_NODE_FEATURES``).
    edge_in : int
        Edge-feature dimension (default: ``N_EDGE_FEATURES``).
    hidden : int
        Hidden dimension within GNN layers.
    n_layers : int
        Number of GNN message-passing layers.
    out_dim : int
        Dimension of the output graph-level embedding.
    """

    def __init__(
        self,
        node_in: int = N_NODE_FEATURES,
        edge_in: int = N_EDGE_FEATURES,
        hidden: int = 64,
        n_layers: int = 3,
        out_dim: int = 128,
    ) -> None:
        super().__init__()
        self._out_dim = out_dim

        # Initial node embedding
        self.node_embed = MLP([node_in, hidden, hidden], activate_last=True)

        # GNN layers (all share the same hidden dimension)
        self.gnn_layers = nn.ModuleList(
            [GNNLayer(hidden, hidden, edge_in) for _ in range(n_layers)]
        )

        # Project pooled representation to output dimension
        self.output_proj = MLP([hidden, out_dim])

    @property
    def out_dim(self) -> int:
        """Output embedding dimension."""
        return self._out_dim

    def forward(
        self,
        x: torch.Tensor,           # (total_nodes, node_in)
        edge_index: torch.Tensor,  # (2, total_edges)
        edge_attr: torch.Tensor,   # (total_edges, edge_in)
        batch: torch.Tensor,       # (total_nodes,) graph index per node
        batch_size: int,
    ) -> torch.Tensor:
        """Encode a batched set of graphs.

        Returns
        -------
        torch.Tensor, shape (batch_size, out_dim)
        """
        if x.size(0) == 0:
            # All graphs in this batch are empty (no nodes)
            return torch.zeros(
                batch_size, self._out_dim, device=x.device, dtype=x.dtype
            )

        h = self.node_embed(x)

        for layer in self.gnn_layers:
            h = layer(h, edge_index, edge_attr)

        pooled = _global_mean_pool(h, batch, batch_size)  # (batch_size, hidden)
        result = self.output_proj(pooled)                  # (batch_size, out_dim)

        # Graphs with no nodes have zero pooled features.  Masking the output
        # here prevents bias terms in output_proj from producing non-zero
        # "phantom" embeddings for empty board states (e.g. S_{t-1} on shot 1).
        node_counts = torch.zeros(batch_size, device=x.device)
        node_counts.scatter_add_(0, batch, torch.ones(x.size(0), device=x.device))
        result = result * (node_counts > 0).float().unsqueeze(1)

        return result


# ---------------------------------------------------------------------------
# Graph Batch container
# ---------------------------------------------------------------------------


@dataclass
class GraphBatch:
    """Batched graph data for the Siamese GNN forward pass.

    Created by :func:`ml.train.collate_graphs`.

    Attributes
    ----------
    x_t : torch.Tensor, shape (total_nodes_t, N_NODE_FEATURES)
        Concatenated node features for S_t (current board state).
    edge_index_t : torch.Tensor, shape (2, total_edges_t), dtype long
        Graph-offset edge indices for S_t.
    edge_attr_t : torch.Tensor, shape (total_edges_t, N_EDGE_FEATURES)
        Edge features for S_t.
    batch_t : torch.Tensor, shape (total_nodes_t,), dtype long
        Maps each S_t node to its graph index in [0, batch_size).
    x_prev, edge_index_prev, edge_attr_prev, batch_prev
        Same fields for S_{t-1} (previous board state).
    u : torch.Tensor, shape (batch_size, N_GRAPH_FEATURES)
        Graph-level metadata features (one row per graph).
    batch_size : int
        Number of graphs in this batch.
    """

    # Current board state (S_t)
    x_t: torch.Tensor
    edge_index_t: torch.Tensor
    edge_attr_t: torch.Tensor
    batch_t: torch.Tensor

    # Previous board state (S_{t-1})
    x_prev: torch.Tensor
    edge_index_prev: torch.Tensor
    edge_attr_prev: torch.Tensor
    batch_prev: torch.Tensor

    # Graph-level metadata
    u: torch.Tensor  # (batch_size, N_GRAPH_FEATURES)

    batch_size: int

    def to(self, device: torch.device) -> "GraphBatch":
        """Return a new ``GraphBatch`` with all tensors moved to *device*."""
        return GraphBatch(
            x_t=self.x_t.to(device),
            edge_index_t=self.edge_index_t.to(device),
            edge_attr_t=self.edge_attr_t.to(device),
            batch_t=self.batch_t.to(device),
            x_prev=self.x_prev.to(device),
            edge_index_prev=self.edge_index_prev.to(device),
            edge_attr_prev=self.edge_attr_prev.to(device),
            batch_prev=self.batch_prev.to(device),
            u=self.u.to(device),
            batch_size=self.batch_size,
        )


# ---------------------------------------------------------------------------
# Full Siamese GNN
# ---------------------------------------------------------------------------


class CurlingGNN(nn.Module):
    """Siamese GNN for multi-task curling shot prediction.

    The model accepts a :class:`GraphBatch` containing paired board states
    ``(S_{t-1}, S_t)`` plus graph-level metadata ``u``, and returns three
    predictions per sample:

    * **accuracy** — predicted shot accuracy (regression, 0–100).
    * **shot_type** — per-class logits (6-class classification).
    * **end_scores** — predicted ``(team1_score, team2_score)`` for this end
      (regression).

    The ``accuracy`` and ``shot_type`` heads consume the *change embedding*
    derived from both board states.  The ``end_scores`` head uses only the
    S_t embedding together with metadata, as specified in the issue.

    Parameters
    ----------
    node_hidden : int
        Hidden dimension within the GNN layers.
    graph_out : int
        Dimension of the per-graph embedding from ``GraphEncoder``.
    head_hidden : int
        Hidden dimension in the output heads and ``ChangeNet``.
    n_gnn_layers : int
        Number of MPNN message-passing layers in ``GraphEncoder``.
    dropout : float
        Dropout probability in heads and ``ChangeNet``.
    """

    def __init__(
        self,
        node_hidden: int = 64,
        graph_out: int = 128,
        head_hidden: int = 128,
        n_gnn_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Shared (Siamese) encoder — same weights used for S_t and S_{t-1}
        self.encoder = GraphEncoder(
            node_in=N_NODE_FEATURES,
            edge_in=N_EDGE_FEATURES,
            hidden=node_hidden,
            n_layers=n_gnn_layers,
            out_dim=graph_out,
        )

        # ChangeNet: fuses e_t, e_prev, (e_t − e_prev), and metadata u
        change_in = graph_out * 3 + N_GRAPH_FEATURES
        self.change_net = MLP(
            [change_in, head_hidden, head_hidden],
            dropout=dropout,
            activate_last=True,
        )

        # Accuracy head (regression): change_emb → scalar
        self.acc_head = nn.Sequential(
            nn.Linear(head_hidden, head_hidden // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(head_hidden // 2, 1),
        )

        # Shot-type head (classification): change_emb → N_SHOT_TYPES logits
        self.type_head = nn.Sequential(
            nn.Linear(head_hidden, head_hidden // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(head_hidden // 2, N_SHOT_TYPES),
        )

        # End-score head (regression): e_t + u → (team1_score, team2_score)
        score_in = graph_out + N_GRAPH_FEATURES
        self.score_head = nn.Sequential(
            nn.Linear(score_in, head_hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(head_hidden, 2),
        )

    def forward(self, batch: "GraphBatch") -> dict[str, torch.Tensor]:
        """Run the full Siamese GNN forward pass.

        Parameters
        ----------
        batch : GraphBatch

        Returns
        -------
        dict with keys:
            ``accuracy``   — shape (B, 1)            — shot accuracy prediction
            ``shot_type``  — shape (B, N_SHOT_TYPES) — shot-type logits
            ``end_scores`` — shape (B, 2)            — (team1_score, team2_score)
        """
        B = batch.batch_size

        # Encode S_t and S_{t-1} with the shared (Siamese) encoder
        e_t = self.encoder(
            batch.x_t,
            batch.edge_index_t,
            batch.edge_attr_t,
            batch.batch_t,
            B,
        )  # (B, graph_out)

        e_prev = self.encoder(
            batch.x_prev,
            batch.edge_index_prev,
            batch.edge_attr_prev,
            batch.batch_prev,
            B,
        )  # (B, graph_out) — zero rows for first shots (empty S_{t-1})

        # Change embedding: captures the delta between the two board states
        change_input = torch.cat([e_t, e_prev, e_t - e_prev, batch.u], dim=-1)
        change_emb = self.change_net(change_input)  # (B, head_hidden)

        accuracy = self.acc_head(change_emb)                                  # (B, 1)
        shot_type = self.type_head(change_emb)                                # (B, N_SHOT_TYPES)
        end_scores = self.score_head(torch.cat([e_t, batch.u], dim=-1))       # (B, 2)

        return {
            "accuracy": accuracy,
            "shot_type": shot_type,
            "end_scores": end_scores,
        }
