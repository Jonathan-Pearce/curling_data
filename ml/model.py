"""Siamese GNN model for multi-task curling shot prediction.

Architecture
------------
1. A shared **GNN encoder** processes each board-state graph (S_{t-1}, S_t)
   into a fixed-size embedding via message-passing + global mean pooling.
2. A **comparison layer** combines the two embeddings into a *change embedding*.
3. Three prediction heads consume appropriate embeddings:

   * **End score** (team1 & team2): S_t embedding only → regression.
   * **Shot accuracy**: change embedding → regression.
   * **Shot type**: change embedding → classification logits.

The GNN uses simple GCN-style message passing implemented in pure PyTorch
(no external GNN library required).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import PipelineConfig


# ── GCN-style message-passing layer ─────────────────────────────────────────

class GCNLayer(nn.Module):
    """Single graph-convolution layer (GCN variant).

    For each node *i*::

        h_i' = ReLU( W_self · h_i  +  W_neigh · mean_{j ∈ N(i)} h_j )

    Operates on **padded** batched node tensors ``(B, N, F)`` with a
    boolean ``mask (B, N)`` that marks valid (non-padding) nodes.
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.W_self = nn.Linear(in_dim, out_dim)
        self.W_neigh = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, N, F_in)
        mask : (B, N)   1.0 for real nodes, 0.0 for padding.

        Returns
        -------
        (B, N, F_out)
        """
        # Compute neighbour mean using masked sum.
        # Expand mask for broadcasting:  (B, N, 1)
        mask_3d = mask.unsqueeze(-1)  # (B, N, 1)

        # Masked node features
        x_masked = x * mask_3d  # zero-out padding

        # Sum all node features (broadcast across N) → (B, 1, F)
        agg_sum = x_masked.sum(dim=1, keepdim=True).expand_as(x)

        # Subtract self to get neighbour sum, then divide by #neighbours
        neigh_sum = agg_sum - x_masked  # (B, N, F)
        neigh_count = mask.sum(dim=1, keepdim=True).unsqueeze(-1) - 1  # (B,1,1)
        neigh_count = neigh_count.clamp(min=1)
        neigh_mean = neigh_sum / neigh_count

        out = self.W_self(x) + self.W_neigh(neigh_mean)
        out = F.relu(out)
        out = self.dropout(out)
        # Re-apply mask so padding stays zero
        return out * mask_3d


# ── GNN Encoder ──────────────────────────────────────────────────────────────

class GNNEncoder(nn.Module):
    """Stack of GCN layers + global mean pooling → graph embedding."""

    def __init__(self, cfg: PipelineConfig) -> None:
        super().__init__()
        layers = []
        in_dim = cfg.node_feature_dim
        for _ in range(cfg.gnn_num_layers):
            layers.append(GCNLayer(in_dim, cfg.gnn_hidden_dim, cfg.gnn_dropout))
            in_dim = cfg.gnn_hidden_dim
        self.layers = nn.ModuleList(layers)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, N, node_feature_dim)
        mask : (B, N)

        Returns
        -------
        graph_emb : (B, gnn_hidden_dim)
        """
        for layer in self.layers:
            x = layer(x, mask)
        # Global mean pool over valid nodes
        mask_3d = mask.unsqueeze(-1)  # (B, N, 1)
        node_sum = (x * mask_3d).sum(dim=1)  # (B, H)
        node_count = mask.sum(dim=1, keepdim=True).clamp(min=1)  # (B, 1)
        return node_sum / node_count  # (B, H)


# ── Siamese GNN Model ───────────────────────────────────────────────────────

class SiameseGNN(nn.Module):
    """Multi-task Siamese GNN for curling shot analysis.

    Parameters
    ----------
    cfg : PipelineConfig
        Full pipeline configuration.
    """

    def __init__(self, cfg: PipelineConfig) -> None:
        super().__init__()
        self.cfg = cfg
        H = cfg.gnn_hidden_dim

        # Shared GNN encoder (weight-tied for S_{t-1} and S_t)
        self.gnn = GNNEncoder(cfg)

        # Metadata embedding
        self.meta_mlp = nn.Sequential(
            nn.Linear(cfg.metadata_features + 1, cfg.metadata_embed_dim),  # +1 for is_first_shot
            nn.ReLU(),
        )

        # Comparison layer: combines prev + cur embeddings + metadata
        comparison_in = H * 2 + cfg.metadata_embed_dim
        self.comparison = nn.Sequential(
            nn.Linear(comparison_in, cfg.head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.gnn_dropout),
        )

        # ── Prediction heads ────────────────────────────────────────────

        # End score head (uses S_t embedding only)
        end_score_in = H + cfg.metadata_embed_dim
        self.end_score_head = nn.Sequential(
            nn.Linear(end_score_in, cfg.head_hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.head_hidden_dim, 2),  # team1_score, team2_score
        )

        # Accuracy head (uses change embedding)
        self.accuracy_head = nn.Sequential(
            nn.Linear(cfg.head_hidden_dim, cfg.head_hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.head_hidden_dim, 1),
        )

        # Shot type head (uses change embedding)
        self.shot_type_head = nn.Sequential(
            nn.Linear(cfg.head_hidden_dim, cfg.head_hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.head_hidden_dim, cfg.num_shot_types),
        )

    def forward(
        self,
        prev_nodes: torch.Tensor,
        prev_mask: torch.Tensor,
        cur_nodes: torch.Tensor,
        cur_mask: torch.Tensor,
        metadata: torch.Tensor,
        is_first_shot: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Returns
        -------
        dict with keys:
            ``end_score``   – (B, 2)
            ``accuracy``    – (B, 1)
            ``shot_type``   – (B, num_shot_types)
        """
        # Encode S_t
        emb_cur = self.gnn(cur_nodes, cur_mask)    # (B, H)

        # Encode S_{t-1}  (zeros for first shot → zero embedding after pool)
        emb_prev = self.gnn(prev_nodes, prev_mask)  # (B, H)

        # Metadata
        meta_input = torch.cat(
            [metadata, is_first_shot.unsqueeze(-1)], dim=-1
        )  # (B, M+1)
        meta_emb = self.meta_mlp(meta_input)  # (B, meta_embed_dim)

        # ── Comparison / change embedding ───────────────────────────────
        combined = torch.cat([emb_prev, emb_cur, meta_emb], dim=-1)
        change_emb = self.comparison(combined)  # (B, head_hidden_dim)

        # ── End score head (S_t embedding + metadata only) ──────────────
        end_input = torch.cat([emb_cur, meta_emb], dim=-1)
        end_score = self.end_score_head(end_input)  # (B, 2)

        # ── Accuracy head ───────────────────────────────────────────────
        accuracy = self.accuracy_head(change_emb)  # (B, 1)

        # ── Shot type head ──────────────────────────────────────────────
        shot_type_logits = self.shot_type_head(change_emb)  # (B, C)

        return {
            "end_score": end_score,
            "accuracy": accuracy,
            "shot_type": shot_type_logits,
        }
