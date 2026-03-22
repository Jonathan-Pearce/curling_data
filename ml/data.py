"""Data loading, preprocessing, and graph construction for curling shots.

Responsibilities
----------------
* Load ``shot_locations.parquet``, ``ends.csv``, and ``events.csv``.
* Apply quality filters (mixed-doubles, conceded ends, empty boards).
* Construct (S_{t-1}, S_t) pairs with metadata features.
* Convert board states into graph representations (node features + adjacency).
* Provide a ``CurlingShotDataset`` compatible with ``torch.utils.data.DataLoader``.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import BASE_SHOT_TYPES, TURN_CLASSES, PipelineConfig

# ── helpers ──────────────────────────────────────────────────────────────────

_BASE_TYPE_PATTERN = re.compile(
    r"^(" + "|".join(re.escape(t) for t in sorted(BASE_SHOT_TYPES, key=len, reverse=True)) + r")\b",
)


def extract_base_shot_type(raw: str) -> Optional[str]:
    """Return the canonical base shot type from a raw ``shot_type`` string.

    Examples
    --------
    >>> extract_base_shot_type("Take-out Free Guard Zone violation")
    'Take-out'
    >>> extract_base_shot_type("no statistics")  # not a known type
    """
    m = _BASE_TYPE_PATTERN.match(raw)
    return m.group(1) if m else None


# ── stone features ───────────────────────────────────────────────────────────

def _board_state_to_node_features(
    row: pd.Series,
    team1_count: int,
    team2_count: int,
) -> Tuple[torch.Tensor, int]:
    """Convert one row's stone columns into a (num_stones, 5) tensor.

    Node features: ``[x, y, distance, angle, is_team1]``
    Returns ``(node_features, num_stones)``.  If *num_stones* == 0 the tensor
    has shape ``(0, 5)``.
    """
    features: List[List[float]] = []
    for i in range(1, team1_count + 1):
        x = row.get(f"team1_stone{i}_x")
        y = row.get(f"team1_stone{i}_y")
        d = row.get(f"team1_stone{i}_dist")
        a = row.get(f"team1_stone{i}_angle")
        if pd.notna(x):
            features.append([float(x), float(y), float(d), float(a), 1.0])
    for i in range(1, team2_count + 1):
        x = row.get(f"team2_stone{i}_x")
        y = row.get(f"team2_stone{i}_y")
        d = row.get(f"team2_stone{i}_dist")
        a = row.get(f"team2_stone{i}_angle")
        if pd.notna(x):
            features.append([float(x), float(y), float(d), float(a), 0.0])
    num_stones = len(features)
    if num_stones == 0:
        return torch.zeros(0, 5), 0
    return torch.tensor(features, dtype=torch.float32), num_stones


def build_fully_connected_edge_index(num_nodes: int) -> torch.Tensor:
    """Return a ``(2, E)`` edge-index tensor for a fully-connected graph.

    Self-loops are excluded.  Returns shape ``(2, 0)`` when ``num_nodes < 2``.
    """
    if num_nodes < 2:
        return torch.zeros(2, 0, dtype=torch.long)
    src = []
    dst = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                src.append(i)
                dst.append(j)
    return torch.tensor([src, dst], dtype=torch.long)


# ── data loading & preprocessing ─────────────────────────────────────────────

def load_and_preprocess(cfg: PipelineConfig) -> pd.DataFrame:
    """Load source tables, apply filters, construct S_{t-1}/S_t pairs.

    Returns a DataFrame with one row per shot, augmented with end-score
    labels, metadata, and S_{t-1} index pointers.
    """
    shots = pd.read_parquet(cfg.shots_path)
    ends = pd.read_csv(cfg.ends_path)
    events = pd.read_csv(cfg.events_path)

    # ── 1. Exclude Mixed Doubles events (no shot location data) ─────────
    mxd_events = events.loc[events["gender"] == "mxd", "event_id"]
    shots = shots[~shots["event_id"].isin(mxd_events)]

    # ── 2. Coerce end scores and drop conceded ends ─────────────────────
    ends["team1_score_this_end"] = pd.to_numeric(
        ends["team1_score_this_end"], errors="coerce"
    )
    ends["team2_score_this_end"] = pd.to_numeric(
        ends["team2_score_this_end"], errors="coerce"
    )

    # ── 3. Join end-level labels onto shots ─────────────────────────────
    shots = shots.merge(
        ends[
            [
                "event_id",
                "match_id",
                "end_number",
                "team1_code",
                "team2_code",
                "team1_score_before",
                "team2_score_before",
                "team1_score_this_end",
                "team2_score_this_end",
                "hammer_team_code",
            ]
        ],
        on=["event_id", "match_id", "end_number"],
        how="left",
    )

    # Drop rows where end score is NaN (conceded)
    shots = shots[shots["team1_score_this_end"].notna()]

    # ── 4. Exclude empty boards ─────────────────────────────────────────
    shots = shots[
        (shots["team1_stones_in_play"] + shots["team2_stones_in_play"]) > 0
    ]

    # ── 5. Parse accuracy to float ──────────────────────────────────────
    shots["accuracy"] = pd.to_numeric(shots["accuracy"], errors="coerce")

    # ── 6. Extract base shot type & encode ──────────────────────────────
    shots["base_shot_type"] = shots["shot_type"].apply(extract_base_shot_type)
    type_to_idx = {t: i for i, t in enumerate(BASE_SHOT_TYPES)}
    shots["shot_type_label"] = shots["base_shot_type"].map(type_to_idx)

    # ── 7. Encode turn ──────────────────────────────────────────────────
    turn_to_idx = {t: i for i, t in enumerate(TURN_CLASSES)}
    shots["turn_label"] = shots["turn"].map(turn_to_idx)

    # ── 8. Derived metadata features ────────────────────────────────────
    shots["score_diff_before"] = (
        shots["team1_score_before"] - shots["team2_score_before"]
    )
    shots["shooting_team_has_hammer"] = (
        shots["team_code"] == shots["hammer_team_code"]
    ).astype(int)

    # ── 9. Sort and assign prev-row index for S_{t-1} ──────────────────
    shots = shots.sort_values(
        ["event_id", "match_id", "end_number", "shot_number"]
    ).reset_index(drop=True)

    # Join event year for train/val/test split
    shots = shots.merge(
        events[["event_id", "year"]], on="event_id", how="left"
    )

    return shots


# ── Dataset ──────────────────────────────────────────────────────────────────

class CurlingShotDataset(Dataset):
    """PyTorch dataset yielding (S_{t-1}, S_t, metadata, labels) tuples.

    Each item is a dictionary with keys:

    * ``nodes_prev``  – (N_{t-1}, 5) node features for S_{t-1}
    * ``edge_index_prev`` – (2, E_{t-1}) edge indices
    * ``num_nodes_prev`` – int
    * ``nodes_cur``   – (N_t, 5) node features for S_t
    * ``edge_index_cur``  – (2, E_t) edge indices
    * ``num_nodes_cur`` – int
    * ``metadata``    – (M,) float tensor of contextual features
    * ``is_first_shot`` – bool (1.0 / 0.0)
    * ``end_score_team1`` – float
    * ``end_score_team2`` – float
    * ``accuracy``    – float  (NaN → -1 sentinel)
    * ``shot_type_label`` – long  (-1 if unknown)
    """

    def __init__(self, df: pd.DataFrame) -> None:
        # Group by end to find previous-shot rows
        self._records: List[Dict] = []
        grouped = df.groupby(["event_id", "match_id", "end_number"])
        for _key, grp in grouped:
            grp = grp.sort_values("shot_number").reset_index(drop=True)
            for i, row in grp.iterrows():
                is_first = row["shot_number"] == 1
                prev_row = grp.iloc[grp.index.get_loc(i) - 1] if not is_first else None
                self._records.append(self._make_record(row, prev_row, is_first))

    # ── internal helpers ────────────────────────────────────────────────

    @staticmethod
    def _make_record(
        cur_row: pd.Series,
        prev_row: Optional[pd.Series],
        is_first: bool,
    ) -> Dict:
        # Current state S_t
        nodes_cur, n_cur = _board_state_to_node_features(
            cur_row,
            int(cur_row["team1_stones_in_play"]),
            int(cur_row["team2_stones_in_play"]),
        )
        edge_index_cur = build_fully_connected_edge_index(n_cur)

        # Previous state S_{t-1}
        if is_first or prev_row is None:
            nodes_prev = torch.zeros(0, 5)
            edge_index_prev = torch.zeros(2, 0, dtype=torch.long)
            n_prev = 0
        else:
            nodes_prev, n_prev = _board_state_to_node_features(
                prev_row,
                int(prev_row["team1_stones_in_play"]),
                int(prev_row["team2_stones_in_play"]),
            )
            edge_index_prev = build_fully_connected_edge_index(n_prev)

        # Metadata vector
        metadata = torch.tensor(
            [
                float(cur_row["shot_number"]),
                float(cur_row["end_number"]),
                float(cur_row.get("score_diff_before", 0)),
                float(cur_row.get("shooting_team_has_hammer", 0)),
                float(cur_row["team1_stones_in_play"]),
                float(cur_row["team2_stones_in_play"]),
            ],
            dtype=torch.float32,
        )

        # Labels
        acc = cur_row.get("accuracy", float("nan"))
        accuracy = float(acc) if pd.notna(acc) else -1.0

        stl = cur_row.get("shot_type_label", -1)
        shot_type_label = int(stl) if pd.notna(stl) else -1

        return {
            "nodes_prev": nodes_prev,
            "edge_index_prev": edge_index_prev,
            "num_nodes_prev": n_prev,
            "nodes_cur": nodes_cur,
            "edge_index_cur": edge_index_cur,
            "num_nodes_cur": n_cur,
            "metadata": metadata,
            "is_first_shot": 1.0 if is_first else 0.0,
            "end_score_team1": float(cur_row["team1_score_this_end"]),
            "end_score_team2": float(cur_row["team2_score_this_end"]),
            "accuracy": accuracy,
            "shot_type_label": shot_type_label,
        }

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> Dict:
        return self._records[idx]


# ── collate function ─────────────────────────────────────────────────────────

def collate_curling_batch(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate that pads variable-size graphs into batched tensors.

    Each graph is represented with padded node features (up to *max_nodes*
    in the batch) and a ``batch_mask`` indicating valid nodes.
    """
    B = len(batch)

    def _pad_graphs(key_nodes: str, key_edges: str, key_num: str):
        max_n = max(b[key_num] for b in batch) or 1  # at least 1 for shape
        padded = torch.zeros(B, max_n, 5)
        masks = torch.zeros(B, max_n)
        # Collect edge indices with batch offsets
        all_edge_src: List[torch.Tensor] = []
        all_edge_dst: List[torch.Tensor] = []
        batch_idx = torch.zeros(B * max_n, dtype=torch.long)
        for i, b in enumerate(batch):
            n = b[key_num]
            if n > 0:
                padded[i, :n] = b[key_nodes]
                masks[i, :n] = 1.0
            ei = b[key_edges]
            if ei.shape[1] > 0:
                offset = i * max_n
                all_edge_src.append(ei[0] + offset)
                all_edge_dst.append(ei[1] + offset)
            batch_idx[i * max_n: i * max_n + n] = i
        if all_edge_src:
            edge_index = torch.cat(
                [torch.cat(all_edge_src).unsqueeze(0),
                 torch.cat(all_edge_dst).unsqueeze(0)], dim=0
            )
        else:
            edge_index = torch.zeros(2, 0, dtype=torch.long)
        return padded, masks, edge_index, batch_idx, max_n

    prev_nodes, prev_mask, prev_ei, prev_bi, prev_mn = _pad_graphs(
        "nodes_prev", "edge_index_prev", "num_nodes_prev"
    )
    cur_nodes, cur_mask, cur_ei, cur_bi, cur_mn = _pad_graphs(
        "nodes_cur", "edge_index_cur", "num_nodes_cur"
    )

    return {
        "prev_nodes": prev_nodes,          # (B, max_n_prev, 5)
        "prev_mask": prev_mask,            # (B, max_n_prev)
        "prev_edge_index": prev_ei,        # (2, E_total_prev)
        "prev_batch_idx": prev_bi,         # (B*max_n_prev,)
        "prev_max_nodes": prev_mn,
        "cur_nodes": cur_nodes,            # (B, max_n_cur, 5)
        "cur_mask": cur_mask,              # (B, max_n_cur)
        "cur_edge_index": cur_ei,          # (2, E_total_cur)
        "cur_batch_idx": cur_bi,           # (B*max_n_cur,)
        "cur_max_nodes": cur_mn,
        "metadata": torch.stack([b["metadata"] for b in batch]),        # (B, M)
        "is_first_shot": torch.tensor(
            [b["is_first_shot"] for b in batch], dtype=torch.float32
        ),                                                               # (B,)
        "end_score_team1": torch.tensor(
            [b["end_score_team1"] for b in batch], dtype=torch.float32
        ),                                                               # (B,)
        "end_score_team2": torch.tensor(
            [b["end_score_team2"] for b in batch], dtype=torch.float32
        ),                                                               # (B,)
        "accuracy": torch.tensor(
            [b["accuracy"] for b in batch], dtype=torch.float32
        ),                                                               # (B,)
        "shot_type_label": torch.tensor(
            [b["shot_type_label"] for b in batch], dtype=torch.long
        ),                                                               # (B,)
    }


# ── split helper ─────────────────────────────────────────────────────────────

def split_by_year(
    df: pd.DataFrame,
    cfg: PipelineConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Event-level train / val / test split based on year boundaries."""
    train = df[df["year"] <= cfg.train_year_max]
    val = df[df["year"] == cfg.val_year]
    test = df[df["year"] >= cfg.test_year_min]
    return train, val, test
