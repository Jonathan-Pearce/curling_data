"""Siamese GNN pipeline for curling shot analysis."""

from gnn_pipeline.config import GNNConfig
from gnn_pipeline.graph import build_board_graph
from gnn_pipeline.model import SiameseGNN
from gnn_pipeline.loss import MultiTaskLoss
from gnn_pipeline.data import CurlingDataset, load_and_preprocess

__all__ = [
    "GNNConfig",
    "build_board_graph",
    "SiameseGNN",
    "MultiTaskLoss",
    "CurlingDataset",
    "load_and_preprocess",
]
