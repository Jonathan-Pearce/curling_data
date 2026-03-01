"""Data loading and preprocessing for the Siamese GNN pipeline.

This module reads ``ends.csv`` and ``shot_locations.csv``, constructs
paired board-state graphs ``(S_{t-1}, S_t)`` for every shot, and exposes
the result as a :class:`CurlingDataset` compatible with
:class:`torch.utils.data.DataLoader`.

Metadata vector layout (6 elements)
------------------------------------
0. ``is_team1_shooting``  – 1.0 if the shooting team is team 1, else 0.0
1. ``shot_number_norm``   – shot number / 16 (normalised)
2. ``end_number_norm``    – end number / 12 (normalised)
3. ``score_diff_norm``    – (team1_score - team2_score) / 10
4. ``is_hammer``          – 1.0 if shooting team has hammer
5. ``turn_enc``           – 0.0 clockwise, 1.0 counter-clockwise, 0.5 other
"""

import csv
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from gnn_pipeline.config import BASE_SHOT_TYPES, GNNConfig
from gnn_pipeline.graph import build_board_graph

# Mapping from raw shot_type text to base type index.
_SHOT_TYPE_INDEX = {st: i for i, st in enumerate(BASE_SHOT_TYPES)}


# ── Helpers ──────────────────────────────────────────────────────────────


def _classify_shot_type(raw: str) -> Optional[int]:
    """Return the base shot-type index, or *None* if unrecognised."""
    for base in BASE_SHOT_TYPES:
        if raw == base or raw.startswith(base + " "):
            return _SHOT_TYPE_INDEX[base]
    return None


def _encode_turn(raw: str) -> float:
    """Encode the turn/rotation as a scalar feature."""
    if raw == "Clockwise":
        return 0.0
    if raw == "Counter-clockwise":
        return 1.0
    return 0.5  # "Not considered" or missing


def _parse_stone_positions(row: Dict[str, str]) -> List[Dict]:
    """Extract stone positions from a shot_locations CSV row."""
    stones: List[Dict] = []
    for team in (1, 2):
        for idx in range(1, 9):
            prefix = f"team{team}_stone{idx}"
            x = row.get(f"{prefix}_x", "")
            y = row.get(f"{prefix}_y", "")
            dist = row.get(f"{prefix}_dist", "")
            angle = row.get(f"{prefix}_angle", "")
            if x and y and dist and angle:
                stones.append(
                    {
                        "x": float(x),
                        "y": float(y),
                        "dist": float(dist),
                        "angle": float(angle),
                        "team": team,
                    }
                )
    return stones


# ── Public loading function ──────────────────────────────────────────────


def load_and_preprocess(
    shots_csv: str,
    ends_csv: str,
    config: Optional[GNNConfig] = None,
) -> List[Dict]:
    """Load CSVs and return a list of sample dicts ready for the dataset.

    Each dict contains tensors for both board states, metadata, and
    target values.
    """
    if config is None:
        config = GNNConfig()

    # --- Load ends lookup -------------------------------------------------
    end_scores: Dict[Tuple[str, str, str], Tuple[float, float, str]] = {}
    with open(ends_csv, newline="") as f:
        for row in csv.DictReader(f):
            key = (row["event_id"], row["match_id"], row["end_number"])
            t1 = row["team1_score_this_end"]
            t2 = row["team2_score_this_end"]
            if t1 == "X" or t2 == "X":
                continue
            hammer = row.get("hammer_team_code", "")
            end_scores[key] = (float(t1), float(t2), hammer)

    # --- Load shots -------------------------------------------------------
    with open(shots_csv, newline="") as f:
        all_shots = list(csv.DictReader(f))

    # Group by (event_id, match_id, end_number) to find previous shot.
    grouped: Dict[Tuple[str, str, str], List[Dict[str, str]]] = {}
    for row in all_shots:
        key = (row["event_id"], row["match_id"], row["end_number"])
        grouped.setdefault(key, []).append(row)

    # --- Build samples ----------------------------------------------------
    samples: List[Dict] = []
    for key, shots in grouped.items():
        shots.sort(key=lambda r: int(r["shot_number"]))
        end_info = end_scores.get(key)

        for i, row in enumerate(shots):
            # Current board state (S_t)
            curr_stones = _parse_stone_positions(row)
            curr_nf, curr_adj, curr_mask = build_board_graph(
                curr_stones, config.max_stones
            )

            # Previous board state (S_{t-1})
            is_first = i == 0
            if is_first:
                prev_stones: List[Dict] = []
            else:
                prev_stones = _parse_stone_positions(shots[i - 1])
            prev_nf, prev_adj, prev_mask = build_board_graph(
                prev_stones, config.max_stones
            )

            # Metadata
            # Whether the shooting team matches the first shooter of this end
            is_first_team = 1.0 if row["team_code"] == shots[0].get("team_code", "") else 0.0
            shot_num = int(row["shot_number"])
            end_num = int(row["end_number"])

            is_hammer = 0.0
            if end_info is not None:
                hammer_code = end_info[2]
                is_hammer = 1.0 if row["team_code"] == hammer_code else 0.0

            turn_enc = _encode_turn(row.get("turn", ""))

            metadata = torch.tensor(
                [
                    is_first_team,
                    shot_num / 16.0,
                    end_num / 12.0,
                    0.0,  # reserved for score context
                    is_hammer,
                    turn_enc,
                ],
                dtype=torch.float,
            )

            # Targets
            acc_raw = row.get("accuracy", "")
            accuracy = float(acc_raw) / 100.0 if acc_raw else None

            type_idx = _classify_shot_type(row.get("shot_type", ""))

            end_score: Optional[float] = None
            if end_info is not None:
                # Net score from team1 perspective
                end_score = end_info[0] - end_info[1]

            samples.append(
                {
                    "prev_node_feat": prev_nf,
                    "prev_adj": prev_adj,
                    "prev_mask": prev_mask,
                    "prev_is_null": is_first,
                    "curr_node_feat": curr_nf,
                    "curr_adj": curr_adj,
                    "curr_mask": curr_mask,
                    "metadata": metadata,
                    "accuracy": accuracy,
                    "shot_type": type_idx,
                    "end_score": end_score,
                }
            )

    return samples


# ── Dataset ──────────────────────────────────────────────────────────────


class CurlingDataset(Dataset):
    """PyTorch dataset wrapping preprocessed curling samples."""

    def __init__(self, samples: List[Dict]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]
        return {
            "prev_node_feat": s["prev_node_feat"],
            "prev_adj": s["prev_adj"],
            "prev_mask": s["prev_mask"],
            "prev_is_null": torch.tensor(s["prev_is_null"], dtype=torch.bool),
            "curr_node_feat": s["curr_node_feat"],
            "curr_adj": s["curr_adj"],
            "curr_mask": s["curr_mask"],
            "metadata": s["metadata"],
            # Targets – use -1 / NaN as sentinel for missing
            "accuracy": torch.tensor(
                s["accuracy"] if s["accuracy"] is not None else float("nan"),
                dtype=torch.float,
            ),
            "shot_type": torch.tensor(
                s["shot_type"] if s["shot_type"] is not None else -1,
                dtype=torch.long,
            ),
            "end_score": torch.tensor(
                s["end_score"] if s["end_score"] is not None else float("nan"),
                dtype=torch.float,
            ),
        }
