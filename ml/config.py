"""Configuration and hyperparameters for the Siamese GNN pipeline."""

from dataclasses import dataclass, field
from typing import List


# ── Base shot types used for classification ──────────────────────────────────
# Ordered list; index is the class label.
BASE_SHOT_TYPES: List[str] = [
    "Draw",
    "Take-out",
    "Hit and Roll",
    "Guard",
    "Raise",
    "Wick / Soft Peeling",
    "Through",
    "Front",
    "Clearing",
    "Double Take-out",
    "Promotion Take-out",
    "Freeze",
]

# Turn categories for optional classification head
TURN_CLASSES: List[str] = [
    "Clockwise",
    "Counter-clockwise",
    "Not considered",
]

# ── Node / graph feature dimensions ─────────────────────────────────────────
NODE_FEATURE_DIM = 5  # [x, y, distance, angle, is_team1]
MAX_STONES = 16       # 8 per team


@dataclass
class PipelineConfig:
    """All tuneable hyper-parameters in one place."""

    # ── Data ────────────────────────────────────────────────────────────────
    shots_path: str = "output/shot_locations.parquet"
    ends_path: str = "output/ends.csv"
    events_path: str = "output/events.csv"
    train_year_max: int = 2023
    val_year: int = 2024
    test_year_min: int = 2025

    # ── Graph construction ──────────────────────────────────────────────────
    node_feature_dim: int = NODE_FEATURE_DIM
    max_stones: int = MAX_STONES

    # ── GNN architecture ────────────────────────────────────────────────────
    gnn_hidden_dim: int = 64
    gnn_num_layers: int = 3
    gnn_dropout: float = 0.1

    # ── Metadata MLP ────────────────────────────────────────────────────────
    metadata_features: int = 6  # shot_number, end_number, score_diff,
    #                             has_hammer, team1_stones, team2_stones
    metadata_embed_dim: int = 16

    # ── Prediction heads ────────────────────────────────────────────────────
    head_hidden_dim: int = 64
    num_shot_types: int = len(BASE_SHOT_TYPES)
    num_turn_classes: int = len(TURN_CLASSES)

    # ── Training ────────────────────────────────────────────────────────────
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 30
    patience: int = 5  # early-stopping patience (epochs)

    # ── Device ──────────────────────────────────────────────────────────────
    device: str = "cpu"

    # ── Reproducibility ─────────────────────────────────────────────────────
    seed: int = 42

    # ── Output ──────────────────────────────────────────────────────────────
    model_save_path: str = "ml/checkpoints/best_model.pt"

    # ── Shot type mapping ───────────────────────────────────────────────────
    base_shot_types: List[str] = field(default_factory=lambda: list(BASE_SHOT_TYPES))
