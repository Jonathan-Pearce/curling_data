"""Graph construction from curling board states.

Each board state is represented as a fully-connected graph where every stone
on the sheet is a node.  Node features encode position, distance/angle from
the house centre, and team ownership.  An adjacency matrix and a boolean
node mask handle padding up to ``max_stones`` (default 16 = 8 per team).
"""

from typing import Dict, List

import torch


def build_board_graph(
    stone_positions: List[Dict],
    max_stones: int = 16,
) -> tuple:
    """Convert a list of stone dicts into padded graph tensors.

    Parameters
    ----------
    stone_positions:
        Each dict must contain keys ``x``, ``y``, ``dist``, ``angle``
        (all float) and ``team`` (1 or 2).
    max_stones:
        Fixed node-count for padding.

    Returns
    -------
    node_features : Tensor [max_stones, 6]
    adj           : Tensor [max_stones, max_stones]  (row-normalised)
    mask          : Tensor [max_stones]  (bool – True for real nodes)
    """
    num_stones = min(len(stone_positions), max_stones)

    node_features = torch.zeros(max_stones, 6)
    mask = torch.zeros(max_stones, dtype=torch.bool)

    for i in range(num_stones):
        s = stone_positions[i]
        node_features[i] = torch.tensor(
            [
                s["x"],
                s["y"],
                s["dist"],
                s["angle"],
                1.0 if s["team"] == 1 else 0.0,
                1.0 if s["team"] == 2 else 0.0,
            ]
        )
        mask[i] = True

    # Fully-connected adjacency among existing nodes (with self-loops).
    adj = torch.zeros(max_stones, max_stones)
    if num_stones > 0:
        adj[:num_stones, :num_stones] = 1.0
        # Row-normalise so each row sums to 1.
        adj[:num_stones, :num_stones] /= num_stones

    return node_features, adj, mask
