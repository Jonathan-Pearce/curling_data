"""Convert enriched shot_locations.parquet into graph samples for GNN training.

Architecture context
--------------------
The GNN uses a Siamese architecture:

  - S_{t-1}: board state after the previous shot (empty for shot_number == 1)
  - S_t:     board state after the current shot

Each board state is encoded as a node-attributed graph:

  - Nodes: stones on the board (up to 16 total, both teams), sorted by stone_id.
  - Node features: see ``NODE_FEATURE_NAMES``
  - Edges: bidirectional spatial proximity edges within ``SPATIAL_EDGE_DIST``

Graph-level features capture shooting context: see ``GRAPH_FEATURE_NAMES``.

Prediction targets (per shot):
  - shot_type_idx : int class index (multi-class classification; -1 if unknown)
  - accuracy      : float 0–100 (regression; NaN if missing)
  - team1_score_this_end : float (end-score regression; NaN if missing/conceded)
  - team2_score_this_end : float (end-score regression; NaN if missing/conceded)

CRITICAL — stone slot ordering
--------------------------------
Stones within each team are sorted by distance from the house centre in every
shot row. The same physical stone can appear in slot ``team1_stone1`` on shot 5
and ``team1_stone3`` on shot 6. The ``_id`` columns are the only ground truth
for cross-shot correspondence. This module always sorts nodes by ``stone_id``
(not by slot index) so temporal node correspondence is stable across shots.
Never row-shift the flat parquet to construct S_{t-1} without first aligning
by ``stone_id``.

Usage::

    python build_graph_dataset.py [--shots  <path>]
                                   [--ends   <path>]
                                   [--events <path>]
                                   [--output <path>]
                                   [--edge-dist <float>]

Output:
    Gzip-compressed pickle file (list of graph dicts) at ``--output``.
    Load with ``load_graphs(path)`` or, when PyTorch Geometric is available,
    via ``CurlingDataset(path, split='train')``.

Importable API::

    from ml.build_graph_dataset import (
        build_board_state,
        build_spatial_edges,
        build_temporal_edges,
        build_graph_features,
        build_shot_graph,
        build_all_graphs,
        assign_splits,
        load_and_filter,
        save_graphs,
        load_graphs,
        NODE_FEATURE_NAMES,
        GRAPH_FEATURE_NAMES,
        SHOT_TYPE_CLASSES,
        SPATIAL_EDGE_DIST,
    )
"""

from __future__ import annotations

import argparse
import gzip
import os
import pickle
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

#: Node feature names, in the order they appear in the matrix columns.
NODE_FEATURE_NAMES: list[str] = [
    "x",             # normalised x position (house-radius units)
    "y",             # normalised y position
    "dist",          # Euclidean distance from house centre (house-radius units)
    "angle_sin",     # sin(angle_radians) — circular encoding
    "angle_cos",     # cos(angle_radians) — circular encoding
    "team",          # 0.0 = team1 (red), 1.0 = team2 (yellow)
    "dx",            # x displacement since previous shot (0.0 if newly placed)
    "dy",            # y displacement since previous shot (0.0 if newly placed)
    "is_new",        # 1.0 if newly placed this shot (prev_x was NULL), else 0.0
    "is_shot_stone", # 1.0 delivered / 0.0 carried / 0.5 ambiguous or first shot
]
N_NODE_FEATURES: int = len(NODE_FEATURE_NAMES)

#: Per-edge feature names (source → target).
EDGE_FEATURE_NAMES: list[str] = [
    "rel_x",  # target_x − source_x
    "rel_y",  # target_y − source_y
    "dist",   # Euclidean distance between the two stones
]
N_EDGE_FEATURES: int = len(EDGE_FEATURE_NAMES)

#: Graph-level feature names.
GRAPH_FEATURE_NAMES: list[str] = [
    "shot_number_norm",          # shot_number / 16.0  (0.0625 … 1.0)
    "end_number_norm",           # end_number / 10.0   (may exceed 1.0 in extra ends)
    "n_team1_stones",            # number of team1 stones in S_t (0–8, raw)
    "n_team2_stones",            # number of team2 stones in S_t (0–8, raw)
    "shooting_team_has_hammer",  # 0.0 or 1.0
    "score_diff_before",         # team1_score_before − team2_score_before (raw int)
    "shooting_team_is_team1",    # 0.0 or 1.0
]
N_GRAPH_FEATURES: int = len(GRAPH_FEATURE_NAMES)

#: Ordered list of shot-type class labels (index = class integer).
SHOT_TYPE_CLASSES: list[str] = [
    "Draw",
    "Guard",
    "Take-out",
    "Raise",
    "Hit and Roll",
    "Wick / Soft Peeling",
]

#: Maps shot-type string → integer class index.
SHOT_TYPE_INDEX: dict[str, int] = {c: i for i, c in enumerate(SHOT_TYPE_CLASSES)}

#: Default spatial edge distance threshold in normalised house-radius units.
#: Stone pairs further apart than this are not connected.
SPATIAL_EDGE_DIST: float = 2.0

# Denominator for end-number normalisation.
_END_NORM: float = 10.0

# Gender code for Mixed Doubles events (no shot position data).
_MXD_GENDER: str = "mxd"

# Default paths
DEFAULT_SHOTS_PATH: str = os.path.join("output", "shot_locations.parquet")
DEFAULT_ENDS_PATH: str = os.path.join("output", "ends.csv")
DEFAULT_EVENTS_PATH: str = os.path.join("output", "events.csv")
DEFAULT_OUTPUT_PATH: str = os.path.join("output", "graphs.pkl.gz")


# ---------------------------------------------------------------------------
# Loading and filtering
# ---------------------------------------------------------------------------

def load_and_filter(
    shots_path: str = DEFAULT_SHOTS_PATH,
    ends_path: str = DEFAULT_ENDS_PATH,
    events_path: str = DEFAULT_EVENTS_PATH,
) -> pd.DataFrame:
    """Load and join shot, end, and event data, then apply pre-training filters.

    Filters applied (per data quality audit and ml-pipeline-data-context):

    1. Exclude Mixed Doubles events — no shot position data.
    2. Exclude conceded ends — ``team{N}_score_this_end == "X"`` coerces to NaN.
    3. Exclude zero-stone rows — board is empty (unusable for GNN).

    Parameters
    ----------
    shots_path : str
        Path to the enriched ``shot_locations.parquet`` (output of
        ``build_features.py``).
    ends_path : str
        Path to ``ends.csv``.
    events_path : str
        Path to ``events.csv``.

    Returns
    -------
    pd.DataFrame
        Filtered shot rows with end-level score labels joined on.
    """
    shots = pd.read_parquet(shots_path)
    ends = pd.read_csv(ends_path)
    events = pd.read_csv(events_path)

    # Coerce end scores; "X" (conceded) → NaN
    ends["team1_score_this_end"] = pd.to_numeric(
        ends["team1_score_this_end"], errors="coerce"
    )
    ends["team2_score_this_end"] = pd.to_numeric(
        ends["team2_score_this_end"], errors="coerce"
    )

    end_label_cols = [
        "event_id", "match_id", "end_number",
        "team1_code", "team2_code",
        "team1_score_this_end", "team2_score_this_end",
    ]
    shots = shots.merge(
        ends[end_label_cols],
        on=["event_id", "match_id", "end_number"],
        how="left",
    )

    # Filter 1: exclude Mixed Doubles events
    mxd_ids = events.loc[events["gender"] == _MXD_GENDER, "event_id"]
    shots = shots[~shots["event_id"].isin(mxd_ids)].copy()

    # Filter 2: exclude conceded ends
    shots = shots[shots["team1_score_this_end"].notna()].copy()

    # Filter 3: exclude zero-stone rows
    shots = shots[
        (shots["team1_stones_in_play"] + shots["team2_stones_in_play"]) > 0
    ].copy()

    return shots.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Node feature extraction
# ---------------------------------------------------------------------------

def _extract_team_stones(row: pd.Series, team: int) -> list[dict]:
    """Extract stone feature dicts for *team* (1 or 2) from a single shot row.

    Each dict has keys:
      ``stone_id``: int or None — stable identity within the end.
      ``features``: float32 array of length ``N_NODE_FEATURES``.

    IMPORTANT: the caller is responsible for sorting these by stone_id before
    constructing the node feature matrix. Never use slot index as node identity
    across shots; always use stone_id.
    """
    prefix = f"team{team}_stone"
    count = int(row.get(f"team{team}_stones_in_play") or 0)
    team_val = float(team - 1)  # 0.0 = team1, 1.0 = team2

    stones: list[dict] = []
    for slot in range(1, count + 1):
        x = row.get(f"{prefix}{slot}_x")
        if pd.isna(x):
            continue

        y = row.get(f"{prefix}{slot}_y") or 0.0
        dist = row.get(f"{prefix}{slot}_dist") or 0.0
        angle_deg = row.get(f"{prefix}{slot}_angle") or 0.0
        stone_id = row.get(f"{prefix}{slot}_id")
        prev_x = row.get(f"{prefix}{slot}_prev_x")
        prev_y = row.get(f"{prefix}{slot}_prev_y")

        is_new_raw = pd.isna(prev_x)
        is_new = 1.0 if is_new_raw else 0.0
        dx = 0.0 if is_new_raw else float(x) - float(prev_x)
        dy = 0.0 if is_new_raw else float(y) - float(prev_y)

        is_shot_raw = row.get(f"{prefix}{slot}_is_shot_stone")
        if is_shot_raw is True or is_shot_raw == 1.0:
            is_shot_val = 1.0
        elif is_shot_raw is False or is_shot_raw == 0.0:
            is_shot_val = 0.0
        else:
            is_shot_val = 0.5  # ambiguous / not available

        angle_rad = float(np.deg2rad(float(angle_deg)))

        features = np.array(
            [
                float(x),
                float(y),
                float(dist),
                float(np.sin(angle_rad)),
                float(np.cos(angle_rad)),
                team_val,
                dx,
                dy,
                is_new,
                is_shot_val,
            ],
            dtype=np.float32,
        )
        stones.append({"stone_id": stone_id, "features": features})

    return stones


def _stone_id_sort_key(stone: dict):
    """Sort key: valid int IDs first (ascending), then None/NaN last."""
    sid = stone["stone_id"]
    if sid is None or (isinstance(sid, float) and np.isnan(sid)):
        return (1, 0)
    return (0, int(sid))


def build_board_state(row: pd.Series) -> tuple[np.ndarray, list]:
    """Extract the node feature matrix for all stones in a single shot row.

    Combines both teams' stones and sorts by stone_id so that the same physical
    stone always occupies the same node index when its ID is tracked — regardless
    of which distance-sorted slot it ends up in.

    Parameters
    ----------
    row : pd.Series
        One row from the filtered shot_locations DataFrame.

    Returns
    -------
    node_features : np.ndarray, shape (n_stones, N_NODE_FEATURES)
        One row per stone, sorted by stone_id.  Empty (0, N_NODE_FEATURES)
        array if there are no stones.
    stone_ids : list
        Ordered stone_id values corresponding to each node row.
        Entries may be None or NaN for untracked stones.
    """
    stones = _extract_team_stones(row, 1) + _extract_team_stones(row, 2)
    if not stones:
        return np.zeros((0, N_NODE_FEATURES), dtype=np.float32), []

    stones.sort(key=_stone_id_sort_key)
    stone_ids = [s["stone_id"] for s in stones]
    node_features = np.stack([s["features"] for s in stones], axis=0)
    return node_features, stone_ids


# ---------------------------------------------------------------------------
# Edge construction
# ---------------------------------------------------------------------------

def build_spatial_edges(
    node_features: np.ndarray,
    dist_threshold: float = SPATIAL_EDGE_DIST,
) -> tuple[np.ndarray, np.ndarray]:
    """Build bidirectional spatial proximity edges for a board state.

    Every pair of stones within *dist_threshold* house-radius units is connected
    with a directed edge in each direction (source→target and target→source).

    Parameters
    ----------
    node_features : np.ndarray, shape (n, N_NODE_FEATURES)
        Node feature matrix.  Columns 0 and 1 are x and y positions.
    dist_threshold : float
        Maximum Euclidean distance between stones to form an edge.

    Returns
    -------
    edge_index : np.ndarray, shape (2, E), dtype int64
        Each column is one directed edge (source, target).
    edge_attr : np.ndarray, shape (E, N_EDGE_FEATURES), dtype float32
        Per-edge features: [rel_x, rel_y, distance].
    """
    n = node_features.shape[0]
    _empty = (
        np.zeros((2, 0), dtype=np.int64),
        np.zeros((0, N_EDGE_FEATURES), dtype=np.float32),
    )
    if n < 2:
        return _empty

    xy = node_features[:, :2]  # (n, 2)

    # Pairwise relative positions: diff[i, j] = xy[j] - xy[i]
    diff = xy[np.newaxis, :, :] - xy[:, np.newaxis, :]  # (n, n, 2)
    dist_matrix = np.linalg.norm(diff, axis=-1)          # (n, n)

    mask = (dist_matrix < dist_threshold) & (dist_matrix > 0.0)
    src, dst = np.where(mask)
    if src.size == 0:
        return _empty

    edge_index = np.stack([src, dst], axis=0).astype(np.int64)
    rel_xy = diff[src, dst, :]                       # (E, 2)
    dists  = dist_matrix[src, dst][:, np.newaxis]    # (E, 1)
    edge_attr = np.concatenate([rel_xy, dists], axis=1).astype(np.float32)

    return edge_index, edge_attr


def build_temporal_edges(prev_ids: list, curr_ids: list) -> np.ndarray:
    """Build temporal correspondence edges from S_{t-1} nodes to S_t nodes.

    Matches nodes by stone_id: if a stone with ID X exists in both S_{t-1} and
    S_t, a directed edge (prev_node_index → curr_node_index) is created.

    Parameters
    ----------
    prev_ids : list
        Ordered stone_id values for S_{t-1} nodes (from ``build_board_state``).
    curr_ids : list
        Ordered stone_id values for S_t nodes.

    Returns
    -------
    np.ndarray, shape (2, E), dtype int64
        Row 0 = index in prev_ids, row 1 = matching index in curr_ids.
        Empty (2, 0) array if no matches.
    """
    id_to_prev_idx: dict[int, int] = {}
    for i, sid in enumerate(prev_ids):
        if sid is not None and not (isinstance(sid, float) and np.isnan(sid)):
            id_to_prev_idx[int(sid)] = i

    edges: list[list[int]] = []
    for curr_idx, sid in enumerate(curr_ids):
        if sid is not None and not (isinstance(sid, float) and np.isnan(sid)):
            prev_idx = id_to_prev_idx.get(int(sid))
            if prev_idx is not None:
                edges.append([prev_idx, curr_idx])

    if not edges:
        return np.zeros((2, 0), dtype=np.int64)

    return np.array(edges, dtype=np.int64).T


# ---------------------------------------------------------------------------
# Graph-level features
# ---------------------------------------------------------------------------

def build_graph_features(row: pd.Series) -> np.ndarray:
    """Build the graph-level feature vector for a single shot.

    Returns
    -------
    np.ndarray, shape (N_GRAPH_FEATURES,), dtype float32
    """
    n1 = int(row.get("team1_stones_in_play") or 0)
    n2 = int(row.get("team2_stones_in_play") or 0)

    def _f(val, default: float = 0.0) -> float:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return default
        return float(val)

    return np.array(
        [
            float(row.get("shot_number", 1)) / 16.0,
            float(row.get("end_number", 1)) / _END_NORM,
            float(n1),
            float(n2),
            _f(row.get("shooting_team_has_hammer")),
            _f(row.get("score_diff_before")),
            _f(row.get("shooting_team_is_team1")),
        ],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Label extraction
# ---------------------------------------------------------------------------

def extract_labels(row: pd.Series) -> dict:
    """Extract all prediction targets for a single shot row.

    Returns
    -------
    dict with keys:
        ``shot_type_idx``        : int, -1 if unrecognised or missing.
        ``accuracy``             : float, NaN if missing.
        ``team1_score_this_end`` : float, NaN if missing or conceded.
        ``team2_score_this_end`` : float, NaN if missing or conceded.
    """
    shot_type_raw = row.get("shot_type")
    shot_type_idx = (
        SHOT_TYPE_INDEX.get(str(shot_type_raw), -1)
        if shot_type_raw and not pd.isna(shot_type_raw)
        else -1
    )

    accuracy_raw = row.get("accuracy")
    try:
        accuracy = (
            float(accuracy_raw)
            if accuracy_raw not in (None, "") and not pd.isna(accuracy_raw)
            else float("nan")
        )
    except (TypeError, ValueError):
        accuracy = float("nan")

    def _score(val) -> float:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return float("nan")
        return float(val)

    return {
        "shot_type_idx": shot_type_idx,
        "accuracy": accuracy,
        "team1_score_this_end": _score(row.get("team1_score_this_end")),
        "team2_score_this_end": _score(row.get("team2_score_this_end")),
    }


# ---------------------------------------------------------------------------
# Per-shot graph construction
# ---------------------------------------------------------------------------

def build_shot_graph(
    prev_row: Optional[pd.Series],
    curr_row: pd.Series,
    spatial_edge_dist: float = SPATIAL_EDGE_DIST,
) -> dict:
    """Build the complete graph sample dict for a single shot.

    Parameters
    ----------
    prev_row : pd.Series or None
        Shot row for shot N-1.  Pass ``None`` for the first shot of an end
        (S_{t-1} will be an empty board with zero nodes).
    curr_row : pd.Series
        Shot row for shot N.  This defines S_t.
    spatial_edge_dist : float
        Distance threshold in house-radius units for spatial proximity edges.

    Returns
    -------
    dict with numpy array values:
        ``x_t``             (n_t,   N_NODE_FEATURES)    — S_t node features
        ``x_prev``          (n_prev, N_NODE_FEATURES)   — S_{t-1} node features
        ``edge_index_t``    (2, E_t)                    — S_t spatial edges
        ``edge_attr_t``     (E_t,   N_EDGE_FEATURES)    — S_t edge features
        ``edge_index_prev`` (2, E_prev)                 — S_{t-1} spatial edges
        ``edge_attr_prev``  (E_prev, N_EDGE_FEATURES)   — S_{t-1} edge features
        ``edge_index_temp`` (2, E_temp)                 — temporal edges (prev→curr)
        ``u``               (N_GRAPH_FEATURES,)         — graph-level features
    And scalar labels / metadata:
        ``shot_type_idx``, ``accuracy``,
        ``team1_score_this_end``, ``team2_score_this_end``,
        ``event_id``, ``match_id``, ``end_number``, ``shot_number``
    """
    # S_t — current board state
    x_t, ids_t = build_board_state(curr_row)
    edge_index_t, edge_attr_t = build_spatial_edges(x_t, spatial_edge_dist)

    # S_{t-1} — previous board state (empty for shot 1)
    if prev_row is not None:
        x_prev, ids_prev = build_board_state(prev_row)
    else:
        x_prev = np.zeros((0, N_NODE_FEATURES), dtype=np.float32)
        ids_prev = []
    edge_index_prev, edge_attr_prev = build_spatial_edges(x_prev, spatial_edge_dist)

    # Temporal correspondence edges (S_{t-1} nodes → S_t nodes, matched by stone_id)
    edge_index_temp = build_temporal_edges(ids_prev, ids_t)

    # Graph-level feature vector
    u = build_graph_features(curr_row)

    # Labels
    labels = extract_labels(curr_row)

    return {
        "x_t": x_t,
        "x_prev": x_prev,
        "edge_index_t": edge_index_t,
        "edge_attr_t": edge_attr_t,
        "edge_index_prev": edge_index_prev,
        "edge_attr_prev": edge_attr_prev,
        "edge_index_temp": edge_index_temp,
        "u": u,
        # Labels
        "shot_type_idx": labels["shot_type_idx"],
        "accuracy": labels["accuracy"],
        "team1_score_this_end": labels["team1_score_this_end"],
        "team2_score_this_end": labels["team2_score_this_end"],
        # Metadata
        "event_id": int(curr_row["event_id"]),
        "match_id": int(curr_row["match_id"]),
        "end_number": int(curr_row["end_number"]),
        "shot_number": int(curr_row["shot_number"]),
    }


# ---------------------------------------------------------------------------
# Full dataset construction
# ---------------------------------------------------------------------------

def build_all_graphs(
    shots_df: pd.DataFrame,
    spatial_edge_dist: float = SPATIAL_EDGE_DIST,
) -> list[dict]:
    """Convert every shot in *shots_df* into a graph sample dict.

    Groups rows by ``(event_id, match_id, end_number)`` and sorts within each
    group by ``shot_number``.  The previous shot's row within the same end is
    passed as ``prev_row``; ``None`` is passed for the first shot of each end.

    Parameters
    ----------
    shots_df : pd.DataFrame
        Filtered, enriched shot-locations data (output of ``load_and_filter``).
    spatial_edge_dist : float
        Spatial edge distance threshold forwarded to ``build_shot_graph``.

    Returns
    -------
    list[dict]
        One graph dict per shot, in end-then-shot order.
    """
    graphs: list[dict] = []
    group_keys = ["event_id", "match_id", "end_number"]

    for _, end_group in shots_df.groupby(group_keys, sort=True):
        end_group = end_group.sort_values("shot_number")
        prev_row: Optional[pd.Series] = None
        for _, curr_row in end_group.iterrows():
            graphs.append(build_shot_graph(prev_row, curr_row, spatial_edge_dist))
            prev_row = curr_row

    return graphs


def assign_splits(
    graphs: list[dict],
    events_df: pd.DataFrame,
    train_max_year: int = 2023,
    val_year: int = 2024,
) -> list[dict]:
    """Annotate each graph dict with a ``split`` key.

    Split strategy (event-level by year):
      - ``year <= train_max_year`` → ``"train"``
      - ``year == val_year``       → ``"val"``
      - ``year > val_year``        → ``"test"``
      - No year data found         → ``"train"`` (conservative fallback)

    Parameters
    ----------
    graphs : list[dict]
        Output of ``build_all_graphs``.
    events_df : pd.DataFrame
        events.csv DataFrame.  Must contain ``event_id`` and ``year`` columns.
    train_max_year : int
        Events up to and including this year become the training set.
    val_year : int
        Events in exactly this year become the validation set.
    """
    event_to_year: dict[int, int] = {}
    for _, row in events_df[["event_id", "year"]].iterrows():
        try:
            event_to_year[int(row["event_id"])] = int(row["year"])
        except (ValueError, TypeError):
            pass

    for g in graphs:
        year = event_to_year.get(g["event_id"])
        if year is None or year <= train_max_year:
            g["split"] = "train"
        elif year == val_year:
            g["split"] = "val"
        else:
            g["split"] = "test"

    return graphs


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def save_graphs(graphs: list[dict], output_path: str) -> None:
    """Serialise *graphs* to a gzip-compressed pickle file.

    The format is a Python list of dicts with numpy array values.  Compatible
    with ``load_graphs`` and ``CurlingDataset`` (when PyG is installed).

    Parameters
    ----------
    graphs : list[dict]
        Output of ``build_all_graphs`` (optionally annotated by
        ``assign_splits``).
    output_path : str
        Destination file path (conventionally ``*.pkl.gz``).
    """
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with gzip.open(output_path, "wb") as fh:
        pickle.dump(graphs, fh, protocol=4)


def load_graphs(path: str) -> list[dict]:
    """Load graph dicts from a file created by ``save_graphs``."""
    with gzip.open(path, "rb") as fh:
        return pickle.load(fh)


# ---------------------------------------------------------------------------
# Optional PyTorch Geometric wrapper
# ---------------------------------------------------------------------------

try:
    import torch
    from torch_geometric.data import Data, InMemoryDataset

    class CurlingDataset(InMemoryDataset):  # type: ignore[misc]
        """PyTorch Geometric ``InMemoryDataset`` backed by a saved graphs file.

        Requires ``torch`` and ``torch_geometric`` to be installed.

        Parameters
        ----------
        graphs_path : str
            Path to the ``.pkl.gz`` file produced by ``save_graphs``.
        split : str or None
            When given (``"train"``, ``"val"``, or ``"test"``), only graphs
            whose ``split`` key matches are included.  Pass ``None`` to load
            all graphs regardless of split.

        Example::

            dataset = CurlingDataset("output/graphs.pkl.gz", split="train")
            loader  = DataLoader(dataset, batch_size=32, shuffle=True)
            data    = dataset[0]   # torch_geometric.data.Data
        """

        def __init__(
            self,
            graphs_path: str,
            split: Optional[str] = None,
            transform=None,
            pre_transform=None,
        ):
            self._graphs_path = graphs_path
            self._split = split
            super().__init__(
                root=None, transform=transform, pre_transform=pre_transform
            )
            self._load()

        def _load(self) -> None:
            graphs = load_graphs(self._graphs_path)
            if self._split is not None:
                graphs = [g for g in graphs if g.get("split") == self._split]
            data_list = [_graph_dict_to_pyg(g) for g in graphs]
            self.data, self.slices = self.collate(data_list)

        @property
        def num_node_features(self) -> int:
            return N_NODE_FEATURES

        @property
        def num_graph_features(self) -> int:
            return N_GRAPH_FEATURES

    def _graph_dict_to_pyg(g: dict) -> "Data":
        """Convert a graph sample dict to a PyG ``Data`` object."""

        def _t(arr, dtype=torch.float32) -> "torch.Tensor":
            return torch.tensor(np.asarray(arr), dtype=dtype)

        accuracy = g["accuracy"]
        t1 = g["team1_score_this_end"]
        t2 = g["team2_score_this_end"]

        return Data(
            # Current board state S_t
            x=_t(g["x_t"]),
            edge_index=_t(g["edge_index_t"], dtype=torch.long),
            edge_attr=_t(g["edge_attr_t"]),
            # Previous board state S_{t-1}
            x_prev=_t(g["x_prev"]),
            edge_index_prev=_t(g["edge_index_prev"], dtype=torch.long),
            edge_attr_prev=_t(g["edge_attr_prev"]),
            # Temporal correspondence edges (prev node → curr node)
            edge_index_temp=_t(g["edge_index_temp"], dtype=torch.long),
            # Graph-level features — unsqueeze to (1, N_GRAPH_FEATURES) for batching
            u=_t(g["u"]).unsqueeze(0),
            # Prediction targets
            y_shot_type=torch.tensor([g["shot_type_idx"]], dtype=torch.long),
            y_accuracy=torch.tensor(
                [float("nan") if np.isnan(accuracy) else accuracy],
                dtype=torch.float32,
            ),
            y_team1_score=torch.tensor(
                [float("nan") if np.isnan(t1) else t1], dtype=torch.float32
            ),
            y_team2_score=torch.tensor(
                [float("nan") if np.isnan(t2) else t2], dtype=torch.float32
            ),
            # Metadata
            event_id=g["event_id"],
            match_id=g["match_id"],
            end_number=g["end_number"],
            shot_number=g["shot_number"],
        )

except ImportError:
    pass  # torch / torch_geometric not installed; CurlingDataset unavailable


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert enriched shot_locations.parquet into graph samples "
            "for GNN training."
        )
    )
    parser.add_argument(
        "--shots",
        default=DEFAULT_SHOTS_PATH,
        help=f"Enriched shot_locations.parquet (default: {DEFAULT_SHOTS_PATH})",
    )
    parser.add_argument(
        "--ends",
        default=DEFAULT_ENDS_PATH,
        help=f"ends.csv (default: {DEFAULT_ENDS_PATH})",
    )
    parser.add_argument(
        "--events",
        default=DEFAULT_EVENTS_PATH,
        help=f"events.csv (default: {DEFAULT_EVENTS_PATH})",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output path for graphs.pkl.gz (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--edge-dist",
        type=float,
        default=SPATIAL_EDGE_DIST,
        help=(
            f"Spatial edge distance threshold in house-radius units "
            f"(default: {SPATIAL_EDGE_DIST})"
        ),
    )
    args = parser.parse_args()

    for path in (args.shots, args.ends, args.events):
        if not os.path.isfile(path):
            parser.error(f"Input file not found: {path}")

    print(f"Loading and filtering shots from {args.shots} …")
    shots_df = load_and_filter(args.shots, args.ends, args.events)
    print(f"  {len(shots_df):,} shots after filtering.")

    events_df = pd.read_csv(args.events)

    print(f"Building graphs (spatial edge dist = {args.edge_dist}) …")
    graphs = build_all_graphs(shots_df, spatial_edge_dist=args.edge_dist)
    print(f"  {len(graphs):,} graph samples built.")

    graphs = assign_splits(graphs, events_df)
    for split in ("train", "val", "test"):
        n = sum(1 for g in graphs if g.get("split") == split)
        print(f"  {split}: {n:,}")

    print(f"Saving to {args.output} …")
    save_graphs(graphs, args.output)
    print("Done.")


if __name__ == "__main__":
    main()
