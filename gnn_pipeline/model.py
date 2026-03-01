"""Siamese Graph Neural Network for multi-task curling shot prediction.

Architecture
------------
* **Twin GNN encoder** (shared weights) maps each board-state graph to a
  fixed-size embedding.
* A **comparison layer** takes ``[e_{t-1}, e_t, e_t - e_{t-1}]`` and
  produces a *change embedding* capturing what the shot did.
* Three output heads:
  - *Shot accuracy* (regression) – from change embedding + metadata.
  - *Shot type* (classification) – from change embedding + metadata.
  - *End score* (regression) – from ``e_t`` + metadata.

For the first shot of an end (``S_{t-1}`` is NULL), a learned
``null_embedding`` replaces the missing previous-state embedding.
"""

import torch
import torch.nn as nn

from gnn_pipeline.config import GNNConfig


# ── GNN building blocks ──────────────────────────────────────────────────


class GraphConvLayer(nn.Module):
    """Single message-passing layer.

    Each node aggregates its neighbours' features via the (normalised)
    adjacency matrix, concatenates with its own features, and applies a
    linear projection + ReLU.
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim * 2, out_dim)
        self.activation = nn.ReLU()

    def forward(
        self,
        node_features: torch.Tensor,
        adj: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        node_features : [B, N, F_in]
        adj           : [B, N, N]
        mask          : [B, N]  (bool)

        Returns
        -------
        [B, N, out_dim]
        """
        agg = torch.bmm(adj, node_features)  # [B, N, F_in]
        combined = torch.cat([node_features, agg], dim=-1)  # [B, N, 2*F_in]
        out = self.activation(self.linear(combined))
        out = out * mask.unsqueeze(-1).float()
        return out


class GNNEncoder(nn.Module):
    """Stack of ``GraphConvLayer``s followed by masked mean-pooling."""

    def __init__(self, config: GNNConfig) -> None:
        super().__init__()
        dims = [config.node_feature_dim] + [config.gnn_hidden_dim] * config.gnn_num_layers
        self.layers = nn.ModuleList(
            [GraphConvLayer(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )
        self.output_proj = nn.Linear(config.gnn_hidden_dim, config.graph_emb_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        node_features: torch.Tensor,
        adj: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return a graph-level embedding ``[B, graph_emb_dim]``."""
        h = node_features
        for layer in self.layers:
            h = layer(h, adj, mask)
            h = self.dropout(h)

        # Masked mean-pooling
        mask_f = mask.unsqueeze(-1).float()
        pooled = (h * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)

        return self.output_proj(pooled)


# ── Siamese model ────────────────────────────────────────────────────────


class SiameseGNN(nn.Module):
    """Siamese GNN with multi-head output for curling shot analysis."""

    def __init__(self, config: GNNConfig) -> None:
        super().__init__()
        self.config = config

        # Shared twin encoder
        self.gnn_encoder = GNNEncoder(config)

        # Learned embedding for NULL previous state
        self.null_embedding = nn.Parameter(torch.randn(config.graph_emb_dim))

        # Comparison layer: [e_prev, e_t, e_t - e_prev] → change emb
        self.comparison_layer = nn.Sequential(
            nn.Linear(config.graph_emb_dim * 3, config.comparison_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )

        # Metadata projection
        self.metadata_proj = nn.Linear(config.metadata_dim, config.comparison_dim)

        # ── Output heads ──
        head_in = config.comparison_dim * 2  # change_emb ∥ meta_emb

        self.accuracy_head = nn.Sequential(
            nn.Linear(head_in, config.head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.head_hidden_dim, 1),
        )

        self.shot_type_head = nn.Sequential(
            nn.Linear(head_in, config.head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.head_hidden_dim, config.num_shot_types),
        )

        # End-score head operates on e_t + metadata only
        self.end_score_head = nn.Sequential(
            nn.Linear(config.graph_emb_dim + config.comparison_dim, config.head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.head_hidden_dim, 1),
        )

    def forward(
        self,
        prev_node_feat: torch.Tensor,
        prev_adj: torch.Tensor,
        prev_mask: torch.Tensor,
        prev_is_null: torch.Tensor,
        curr_node_feat: torch.Tensor,
        curr_adj: torch.Tensor,
        curr_mask: torch.Tensor,
        metadata: torch.Tensor,
    ) -> tuple:
        """
        Parameters
        ----------
        prev_node_feat : [B, N, F]   – previous board-state node features
        prev_adj       : [B, N, N]
        prev_mask      : [B, N]  (bool)
        prev_is_null   : [B]     (bool – True when S_{t-1} is absent)
        curr_node_feat : [B, N, F]   – current board-state node features
        curr_adj       : [B, N, N]
        curr_mask      : [B, N]  (bool)
        metadata       : [B, metadata_dim]

        Returns
        -------
        accuracy_pred    : [B]
        shot_type_logits : [B, num_shot_types]
        end_score_pred   : [B]
        """
        # Encode both states with the shared GNN
        e_t = self.gnn_encoder(curr_node_feat, curr_adj, curr_mask)
        e_prev = self.gnn_encoder(prev_node_feat, prev_adj, prev_mask)

        # Substitute null embedding where S_{t-1} is absent
        null_emb = self.null_embedding.unsqueeze(0).expand_as(e_prev)
        is_null = prev_is_null.unsqueeze(-1).float()
        e_prev = e_prev * (1.0 - is_null) + null_emb * is_null

        # Change embedding
        change_emb = self.comparison_layer(
            torch.cat([e_prev, e_t, e_t - e_prev], dim=-1)
        )

        # Metadata
        meta_emb = self.metadata_proj(metadata)

        # Heads
        combined = torch.cat([change_emb, meta_emb], dim=-1)
        accuracy_pred = self.accuracy_head(combined).squeeze(-1)
        shot_type_logits = self.shot_type_head(combined)

        end_input = torch.cat([e_t, meta_emb], dim=-1)
        end_score_pred = self.end_score_head(end_input).squeeze(-1)

        return accuracy_pred, shot_type_logits, end_score_pred
