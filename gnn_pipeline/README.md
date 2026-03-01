# Siamese GNN Pipeline for Curling Shot Analysis

CPU-only Graph ML pipeline that predicts multiple curling shot outcomes using a
Siamese Graph Neural Network, implemented in PyTorch.

## Objectives

| Task | Type | Target column(s) | Data file |
|---|---|---|---|
| End score | Regression | `team1_score_this_end`, `team2_score_this_end` | `ends.csv` |
| Shot accuracy | Regression | `accuracy` | `shot_locations.csv` |
| Shot type | Classification | `shot_type` | `shot_locations.csv` |

## Architecture

```
S_{t-1} ──► Twin GNN ──► e_{t-1} ─┐
                                    ├─► Comparison ──► change_emb ──► Accuracy Head  (MSE)
S_t ────► Twin GNN ──► e_t ───────┤                                ► Shot Type Head (CE)
                                    │
                                    └─► End Score Head (MSE)
```

* **Twin GNN encoder** (shared weights) maps each board-state graph to a
  fixed-size embedding via stacked message-passing layers and masked
  mean-pooling.
* A **comparison layer** concatenates `[e_{t-1}, e_t, e_t − e_{t-1}]` to
  produce a *change embedding*.
* Three **output heads** predict shot accuracy (regression), shot type
  (12-class classification), and end score (regression).
* For the first shot of an end (`S_{t-1}` is NULL), a learned
  `null_embedding` is substituted.

## Loss Function

Composite loss using **Homoscedastic Uncertainty Weighting** (Kendall et al.
2018) with three learnable log-variance parameters that automatically balance
gradients across MSE and cross-entropy objectives.

## Graph Representation

Each board state is a fully-connected graph where every stone is a node.

| Feature | Description |
|---|---|
| `x`, `y` | Stone position (normalised to house centre) |
| `dist` | Distance from house centre |
| `angle` | Angle from house centre |
| `is_team1` | 1.0 if stone belongs to team 1 |
| `is_team2` | 1.0 if stone belongs to team 2 |

Up to 16 nodes (8 per team), padded with zeros and masked.

## Metadata Vector

| Index | Feature | Encoding |
|---|---|---|
| 0 | Shooting team | 1.0 = team 1, 0.0 = team 2 |
| 1 | Shot number | Normalised (÷ 16) |
| 2 | End number | Normalised (÷ 12) |
| 3 | Score differential | (team1 − team2) ÷ 10 |
| 4 | Has hammer | 1.0 if shooting team has hammer |
| 5 | Turn direction | 0.0 = CW, 1.0 = CCW, 0.5 = other |

## Usage

```bash
# Install dependencies
pip install torch

# Train (from repository root)
python -m gnn_pipeline.train \
    --shots_csv output/shot_locations.csv \
    --ends_csv output/ends.csv \
    --epochs 50 \
    --batch_size 64

# Run tests
pytest tests/test_gnn_pipeline.py -v
```

## Module Structure

```
gnn_pipeline/
├── __init__.py   # Public API
├── config.py     # GNNConfig dataclass
├── data.py       # CSV loading, preprocessing, CurlingDataset
├── graph.py      # Board-state → graph construction
├── loss.py       # MultiTaskLoss (uncertainty weighting)
├── model.py      # SiameseGNN, GNNEncoder, GraphConvLayer
├── train.py      # Training loop and CLI
└── README.md     # This file
```

## References

* Kendall, A., Gal, Y., & Cipolla, R. (2018). *Multi-Task Learning Using
  Uncertainty to Weigh Losses for Scene Geometry and Semantics*.
  [arXiv:1705.07115](https://arxiv.org/abs/1705.07115)
