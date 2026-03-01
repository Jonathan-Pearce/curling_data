"""Configuration for the Siamese GNN pipeline."""

from dataclasses import dataclass, field
from typing import List


# Base shot types extracted from the raw shot_type column (annotations stripped).
BASE_SHOT_TYPES: List[str] = [
    "Clearing",
    "Double Take-out",
    "Draw",
    "Freeze",
    "Front",
    "Guard",
    "Hit and Roll",
    "Promotion Take-out",
    "Raise",
    "Take-out",
    "Through",
    "Wick / Soft Peeling",
]


@dataclass
class GNNConfig:
    """Hyperparameters and constants for the Siamese GNN pipeline."""

    # Graph ----------------------------------------------------------------
    max_stones: int = 16  # 8 per team
    node_feature_dim: int = 6  # x, y, dist, angle, is_team1, is_team2
    metadata_dim: int = 6  # see data.py for details

    # GNN encoder ----------------------------------------------------------
    gnn_hidden_dim: int = 64
    gnn_num_layers: int = 3
    graph_emb_dim: int = 64

    # Comparison / heads ---------------------------------------------------
    comparison_dim: int = 128
    head_hidden_dim: int = 64
    dropout: float = 0.1

    # Shot-type classification ---------------------------------------------
    base_shot_types: List[str] = field(default_factory=lambda: list(BASE_SHOT_TYPES))
    num_shot_types: int = field(init=False)

    # Training -------------------------------------------------------------
    learning_rate: float = 1e-3
    batch_size: int = 64
    num_epochs: int = 50
    weight_decay: float = 1e-4
    train_ratio: float = 0.8
    val_ratio: float = 0.1

    # Device (kept configurable; defaults to CPU) --------------------------
    device: str = "cpu"

    def __post_init__(self) -> None:
        self.num_shot_types = len(self.base_shot_types)
