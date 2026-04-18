"""Sequential stone-tracking post-processor for curling shot data.

Reads the raw (untracked) shot locations written by ``extract_shot_data.py``
(``shot_locations_raw.parquet`` or ``shot_locations_raw.csv``) and applies a
sequential stone-tracking algorithm that assigns stable stone IDs within each
end and populates ``prev_x`` / ``prev_y`` displacement columns.

Two algorithms are provided so they can be compared:

``greedy``
    Greedy nearest-neighbour: build all (prev, curr) candidate pairs that fall
    within ``STONE_TRACK_MAX_DIST``, sort ascending by distance, then assign
    each pair in that order skipping already-used nodes.  O(n²) per shot.
    Equivalent to the tracker that was previously embedded in
    ``extract_shot_data.py``.

``hungarian``
    Minimum-cost bipartite matching via the Hungarian algorithm
    (``scipy.optimize.linear_sum_assignment``).  Finds the globally optimal
    assignment under the same distance cap, which can differ from the greedy
    result when two stones are close in distance to the same previous stone.
    O(n³) per shot; in practice n ≤ 8 so this is negligible.

Usage::

    python track_stones.py [--method greedy|hungarian]
                           [--input  <path>]
                           [--output-dir <dir>]

Outputs:
    ``shot_locations.csv``
    ``shot_locations.parquet``  (ML-ready; schema matches the old parquet)

Evaluation helpers (importable)::

    from track_stones import (
        apply_tracking,
        evaluate_tracking,
        compare_tracking,
        displacement_distribution,
        id_continuity_rate,
        slot_swap_rate,
        cap_pressure_rate,
        displacement_symmetry,
        stone_count_consistency_rate,
        delivery_anomaly_rate,
    )
"""

import argparse
import math
import os
import sys

import numpy as np
import pandas as pd

# Allow ``python src/tracking/track_stones.py`` to resolve sibling packages.
if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scraping.extract_shot_data import MAX_STONES_PER_TEAM, STONE_TRACK_MAX_DIST

TRACKING_METHODS = ("greedy", "hungarian")

DEFAULT_INPUT = os.path.join("output", "shot_locations_raw.parquet")
DEFAULT_OUTPUT_DIR = "output"

# ---------------------------------------------------------------------------
# Core per-shot tracking functions
#
# Each function shares the same signature:
#
#   track_fn(prev_state, curr_stones, stone_id_counter) -> (matched, new_state)
#
# prev_state   : list of (stone_id, px, py)   — state carried from previous shot
# curr_stones  : list of (nx, ny)             — distance-sorted raw detections
# stone_id_counter : [int]                    — mutable single-element list;
#                                               incremented when a new ID is issued
#
# Returns:
#   matched   : list of (stone_id, nx, ny, prev_x, prev_y)  indexed by curr slot
#   new_state : list of (stone_id, nx, ny)                  for next shot
# ---------------------------------------------------------------------------

def _track_greedy(prev_state, curr_stones, stone_id_counter):
    """Greedy nearest-neighbour matching.

    Builds all candidate (prev_index, curr_index) pairs within
    ``STONE_TRACK_MAX_DIST``, sorts them ascending by distance, then
    assigns each pair greedily, skipping any node that has already been
    used by a closer pair.
    """
    pairs = []
    for pi, (sid, px, py) in enumerate(prev_state):
        for ci, (nx, ny) in enumerate(curr_stones):
            d = math.sqrt((nx - px) ** 2 + (ny - py) ** 2)
            if d < STONE_TRACK_MAX_DIST:
                pairs.append((d, pi, ci))
    pairs.sort()

    assignments = {}  # curr_index -> (stone_id, prev_x, prev_y)
    used_prev = set()
    used_curr = set()
    for _d, pi, ci in pairs:
        if pi in used_prev or ci in used_curr:
            continue
        sid, px, py = prev_state[pi]
        assignments[ci] = (sid, px, py)
        used_prev.add(pi)
        used_curr.add(ci)

    matched = []
    new_state = []
    for ci, (nx, ny) in enumerate(curr_stones):
        if ci in assignments:
            sid, prev_x, prev_y = assignments[ci]
        else:
            sid = stone_id_counter[0]
            stone_id_counter[0] += 1
            prev_x, prev_y = None, None
        matched.append((sid, nx, ny, prev_x, prev_y))
        new_state.append((sid, nx, ny))

    return matched, new_state


def _track_hungarian(prev_state, curr_stones, stone_id_counter):
    """Optimal bipartite matching via the Hungarian algorithm.

    Constructs a cost matrix of pairwise distances, sets entries beyond
    ``STONE_TRACK_MAX_DIST`` to a large sentinel, and calls
    ``scipy.optimize.linear_sum_assignment`` to find the globally minimum-
    cost assignment.  Pairs whose cost exceeds the threshold are discarded
    (unmatched on both sides).
    """
    from scipy.optimize import linear_sum_assignment  # optional dependency

    if not prev_state or not curr_stones:
        matched = []
        new_state = []
        for nx, ny in curr_stones:
            sid = stone_id_counter[0]
            stone_id_counter[0] += 1
            matched.append((sid, nx, ny, None, None))
            new_state.append((sid, nx, ny))
        return matched, new_state

    n_prev = len(prev_state)
    n_curr = len(curr_stones)
    INF = 1e9
    cost = np.full((n_prev, n_curr), INF)
    for pi, (sid, px, py) in enumerate(prev_state):
        for ci, (nx, ny) in enumerate(curr_stones):
            d = math.sqrt((nx - px) ** 2 + (ny - py) ** 2)
            if d < STONE_TRACK_MAX_DIST:
                cost[pi, ci] = d

    row_ind, col_ind = linear_sum_assignment(cost)

    assignments = {}  # curr_index -> (stone_id, prev_x, prev_y)
    for pi, ci in zip(row_ind, col_ind):
        if cost[pi, ci] < STONE_TRACK_MAX_DIST:
            sid, px, py = prev_state[pi]
            assignments[ci] = (sid, px, py)

    matched = []
    new_state = []
    for ci, (nx, ny) in enumerate(curr_stones):
        if ci in assignments:
            sid, prev_x, prev_y = assignments[ci]
        else:
            sid = stone_id_counter[0]
            stone_id_counter[0] += 1
            prev_x, prev_y = None, None
        matched.append((sid, nx, ny, prev_x, prev_y))
        new_state.append((sid, nx, ny))

    return matched, new_state


_TRACK_FN = {
    "greedy": _track_greedy,
    "hungarian": _track_hungarian,
}

# Maximum distance (normalised house-radius units) within which a ghost ring
# is matched to a stone's previous-shot position.  Slightly larger than
# STONE_TRACK_MAX_DIST to absorb ghost-detection pixel imprecision.
GHOST_MATCH_MAX_DIST = 0.20

# ---------------------------------------------------------------------------
# DataFrame-level tracking
# ---------------------------------------------------------------------------

def apply_tracking(raw_df, method="greedy"):
    """Apply sequential stone tracking to a raw shot-locations DataFrame.

    Processes each (event_id, match_id, end_number) group independently,
    resetting stone IDs and tracking state at each end boundary.  Within an
    end, shots are processed in ascending ``shot_number`` order.

    Parameters
    ----------
    raw_df : pd.DataFrame
        Untracked shot locations as written by ``extract_shot_data.py``.
        Must contain ``event_id``, ``match_id``, ``end_number``,
        ``shot_number``, ``team{N}_stones_in_play``, and
        ``team{N}_stone{S}_x`` / ``_y`` columns for N ∈ {1, 2}, S ∈ 1–8.
    method : str
        ``"greedy"`` or ``"hungarian"``.

    Returns
    -------
    pd.DataFrame
        Copy of *raw_df* with four tracking columns appended per stone slot:
        ``team{N}_stone{S}_id``, ``team{N}_stone{S}_prev_x``,
        ``team{N}_stone{S}_prev_y``, ``team{N}_stone{S}_is_shot_stone``.
        Values are ``NaN`` for empty slots and for the first shot of each end
        (no prior state).  ``is_shot_stone`` is ``True`` for the one stone
        that was just delivered, ``False`` for all other occupied stones, and
        ``NaN`` when the delivered stone cannot be identified unambiguously
        (first shot of end, or tracking fragmentation produced zero or
        multiple newly-placed stones in a single shot).

        If *raw_df* contains ghost columns (``team{N}_ghost{K}_x`` /
        ``team{N}_ghost{K}_y`` and ``team{N}_ghosts_in_play``), one
        additional column per ghost slot is appended:
        ``team{N}_ghost{K}_stone_id`` — the ID of the stone that occupied
        the ghost position at the preceding shot.  Matched by nearest
        previous-shot position within ``GHOST_MATCH_MAX_DIST``.  ``NaN``
        when no prior state exists (first shot of end) or no match falls
        within the threshold.
    """
    if method not in _TRACK_FN:
        raise ValueError(
            f"Unknown tracking method {method!r}. Choose from {TRACKING_METHODS}."
        )
    track_fn = _TRACK_FN[method]

    df = raw_df.copy().reset_index(drop=True)

    # Pre-allocate tracking columns as float NaN so dtype is consistent.
    tracking_cols = []
    for ti in (1, 2):
        for si in range(1, MAX_STONES_PER_TEAM + 1):
            prefix = f"team{ti}_stone{si}"
            for suffix in ("_id", "_prev_x", "_prev_y", "_is_shot_stone"):
                col = prefix + suffix
                df[col] = np.nan
                tracking_cols.append(col)

    # Pre-allocate ghost_stone_id columns only if ghost coordinate columns exist.
    _has_ghosts = f"team1_ghost1_x" in df.columns
    if _has_ghosts:
        for ti in (1, 2):
            for gi in range(1, MAX_STONES_PER_TEAM + 1):
                df[f"team{ti}_ghost{gi}_stone_id"] = np.nan

    group_keys = ["event_id", "match_id", "end_number"]
    for _, end_idx in df.groupby(group_keys, sort=True).groups.items():
        end_view = df.loc[end_idx].sort_values("shot_number")
        track_state = {1: [], 2: []}
        stone_id_counter = [1]

        for row_pos in end_view.index:
            row = df.loc[row_pos]

            # Snapshot the previous-shot positions for both teams before updating
            # track_state.  Used below for ghost-to-stone matching.
            prev_state_before = {ti: list(track_state[ti]) for ti in (1, 2)}

            matched_by_team = {}
            for ti in (1, 2):
                n = int(row[f"team{ti}_stones_in_play"] or 0)
                curr_stones = []
                for si in range(1, n + 1):
                    x = row[f"team{ti}_stone{si}_x"]
                    y = row[f"team{ti}_stone{si}_y"]
                    if pd.notna(x) and pd.notna(y):
                        curr_stones.append((float(x), float(y)))

                matched, track_state[ti] = track_fn(
                    track_state[ti], curr_stones, stone_id_counter
                )
                matched_by_team[ti] = matched

                for slot_idx, (sid, nx, ny, prev_x, prev_y) in enumerate(matched):
                    si = slot_idx + 1
                    prefix = f"team{ti}_stone{si}"
                    df.at[row_pos, f"{prefix}_id"] = float(sid)
                    if prev_x is not None:
                        df.at[row_pos, f"{prefix}_prev_x"] = round(prev_x, 3)
                    if prev_y is not None:
                        df.at[row_pos, f"{prefix}_prev_y"] = round(prev_y, 3)

            # Assign is_shot_stone flag.
            # First shot of end: no prior state, all flags remain NaN.
            # Mid-end: the one newly placed stone (prev_x is None) is the
            # delivered stone.  If exactly one such stone exists across both
            # teams, flag it True and all other occupied stones False.
            # If zero or more than one: ambiguous — leave NaN.
            if int(row["shot_number"]) > 1:
                new_slots = [
                    (ti, slot_idx + 1)
                    for ti in (1, 2)
                    for slot_idx, (_sid, _nx, _ny, prev_x, _prev_y)
                    in enumerate(matched_by_team[ti])
                    if prev_x is None
                ]
                if len(new_slots) == 1:
                    shot_ti, shot_si = new_slots[0]
                    for ti in (1, 2):
                        for slot_idx in range(len(matched_by_team[ti])):
                            si = slot_idx + 1
                            is_shot = 1.0 if (ti == shot_ti and si == shot_si) else 0.0
                            df.at[row_pos, f"team{ti}_stone{si}_is_shot_stone"] = is_shot

            # Ghost-to-stone matching.
            # For each ghost ring of team N at position (gx, gy), find the
            # stone from the previous shot whose position was closest to that
            # ghost position.  The match pool is prev_state_before[ti], which
            # includes both stones that are still on the board (prev_x/prev_y
            # records their prior position) and stones that were knocked out
            # (they appear in prev_state_before but not in matched_by_team).
            # First shot of end: prev_state_before is empty, all NaN.
            if _has_ghosts and int(row["shot_number"]) > 1:
                for ti in (1, 2):
                    ghosts_count = row[f"team{ti}_ghosts_in_play"]
                    ng = int(ghosts_count) if pd.notna(ghosts_count) else 0
                    prev_positions = prev_state_before[ti]  # [(sid, px, py), ...]
                    for gi in range(1, ng + 1):
                        gx = row[f"team{ti}_ghost{gi}_x"]
                        gy = row[f"team{ti}_ghost{gi}_y"]
                        if pd.isna(gx) or pd.isna(gy):
                            continue
                        gx, gy = float(gx), float(gy)
                        best_sid = None
                        best_dist = GHOST_MATCH_MAX_DIST
                        for sid, px, py in prev_positions:
                            d = math.sqrt((gx - px) ** 2 + (gy - py) ** 2)
                            if d < best_dist:
                                best_dist = d
                                best_sid = sid
                        if best_sid is not None:
                            df.at[row_pos, f"team{ti}_ghost{gi}_stone_id"] = float(
                                best_sid
                            )

    return df


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_tracking(tracked_df):
    """Compute quality metrics for a tracked shot-locations DataFrame.

    All metrics are computed independently of any ground-truth labels.

    Parameters
    ----------
    tracked_df : pd.DataFrame
        Output of :func:`apply_tracking` — must contain ``_id``, ``_prev_x``,
        ``_prev_y`` columns for every stone slot.

    Returns
    -------
    dict
        ``total_ends``
            Number of unique (event_id, match_id, end_number) groups.
        ``total_stone_appearances``
            Total non-null stone slots across all shots (sum of
            ``team{N}_stones_in_play``).
        ``expected_new_stone_rate``
            Theoretical fraction of mid-end stone slot appearances that should
            be newly placed: ``n_mid_end_shots / total_mid_end_appearances``.
            In 4-person curling each shot delivers exactly one stone, so this
            equals ``1 / avg_stones_on_board`` (typically ~0.25).
        ``new_stone_rate_mid_end``
            Measured fraction of non-first-shot stone appearances where the
            stone received a new ID (``prev_x`` is NaN).  Should be close to
            ``expected_new_stone_rate`` for a well-calibrated tracker.
        ``spurious_id_rate``
            Excess new-stone events above the expected delivery rate:
            ``max(0, (new_mid_end - n_mid_end_shots) / total_mid_end)``.
            Each shot delivers exactly one stone, so ``n_mid_end_shots`` is
            the expected number of new IDs.  Any excess means stones already
            on the board failed to match and were re-IDed.  0.0 is ideal;
            values above ~0.02 warrant investigation.

            Note: a takeout-heavy end (high stone churn) correctly produces
            many new IDs — one per delivery — and does not inflate this metric.
    """
    df = tracked_df
    group_keys = ["event_id", "match_id", "end_number"]

    total_appearances = 0
    new_mid_end = 0
    total_mid_end = 0
    n_mid_end_shots = 0

    for (event_id, match_id, end_number), end_df in df.groupby(group_keys, sort=True):
        end_df_sorted = end_df.sort_values("shot_number")
        first_shot = end_df_sorted["shot_number"].min()
        mid_df = end_df_sorted[end_df_sorted["shot_number"] > first_shot]
        n_mid_end_shots += len(mid_df)

        for ti in (1, 2):
            # New-stone rate mid-end (shot_number > first in end)
            for si in range(1, MAX_STONES_PER_TEAM + 1):
                x_col = f"team{ti}_stone{si}_x"
                prev_col = f"team{ti}_stone{si}_prev_x"
                if x_col not in df.columns or prev_col not in df.columns:
                    break
                occupied = mid_df[x_col].notna()
                total_mid_end += int(occupied.sum())
                new_mid_end += int((occupied & mid_df[prev_col].isna()).sum())

        for ti in (1, 2):
            for si in range(1, MAX_STONES_PER_TEAM + 1):
                col = f"team{ti}_stone{si}_x"
                if col in df.columns:
                    total_appearances += int(end_df_sorted[col].notna().sum())

    expected_new_rate = (
        n_mid_end_shots / total_mid_end if total_mid_end > 0 else float("nan")
    )
    new_rate = (
        new_mid_end / total_mid_end if total_mid_end > 0 else float("nan")
    )
    spurious = (
        max(0.0, (new_mid_end - n_mid_end_shots) / total_mid_end)
        if total_mid_end > 0
        else float("nan")
    )

    return {
        "total_ends": int(df.groupby(group_keys).ngroups),
        "total_stone_appearances": total_appearances,
        "expected_new_stone_rate": round(expected_new_rate, 4),
        "new_stone_rate_mid_end": round(new_rate, 4),
        "spurious_id_rate": round(spurious, 4),
    }


def compare_tracking(tracked_a, tracked_b, method_a="A", method_b="B"):
    """Compare two tracked DataFrames produced from the same raw data.

    The key comparison is **link agreement**: for each stone slot at each
    non-first shot of an end, do both methods agree on which previous-shot
    position the stone came from (i.e., do they assign the same
    ``prev_x`` / ``prev_y``)?

    Parameters
    ----------
    tracked_a, tracked_b : pd.DataFrame
        Outputs of :func:`apply_tracking` applied to the same raw DataFrame
        with different ``method`` arguments.
    method_a, method_b : str
        Names used in the returned summary dict keys.

    Returns
    -------
    dict
        ``link_agreement_rate``
            Fraction of mid-end stone appearances where both methods assigned
            the same origin (``prev_x`` and ``prev_y`` match within 1e-4
            tolerance, or both are NaN meaning both treated the stone as new).
        ``spurious_id_rate_{method_a}``
        ``spurious_id_rate_{method_b}``
            Spurious ID rate from :func:`evaluate_tracking` for each.
        ``spurious_id_rate_improvement``
            ``spurious_a − spurious_b``.  Positive means method B produces
            fewer spurious re-IDs.
        ``new_stone_rate_mid_end_{method_a}``
        ``new_stone_rate_mid_end_{method_b}``
    """
    merge_keys = ["event_id", "match_id", "end_number", "shot_number"]

    # Merge on identity columns so we compare same shots
    tracking_cols_a = {}
    tracking_cols_b = {}
    for ti in (1, 2):
        for si in range(1, MAX_STONES_PER_TEAM + 1):
            prefix = f"team{ti}_stone{si}"
            for suffix in ("_prev_x", "_prev_y"):
                tracking_cols_a[f"{prefix}{suffix}"] = f"{prefix}{suffix}_a"
                tracking_cols_b[f"{prefix}{suffix}"] = f"{prefix}{suffix}_b"

    df_a = tracked_a[merge_keys + list(tracking_cols_a.keys())].rename(
        columns=tracking_cols_a
    )
    df_b = tracked_b[merge_keys + list(tracking_cols_b.keys())].rename(
        columns=tracking_cols_b
    )
    merged = df_a.merge(df_b, on=merge_keys, how="inner")

    # Only compare mid-end shots (exclude first shot of each end)
    first_shots = (
        merged.groupby(["event_id", "match_id", "end_number"])["shot_number"]
        .transform("min")
    )
    mid = merged[merged["shot_number"] > first_shots]

    total = 0
    agreed = 0
    tol = 1e-4
    for ti in (1, 2):
        for si in range(1, MAX_STONES_PER_TEAM + 1):
            x_col = f"team{ti}_stone{si}_x"
            if x_col not in tracked_a.columns:
                break
            # Merge also lacks x; proxy: check prev_x_a exists
            px_a = mid.get(f"team{ti}_stone{si}_prev_x_a")
            px_b = mid.get(f"team{ti}_stone{si}_prev_x_b")
            py_a = mid.get(f"team{ti}_stone{si}_prev_y_a")
            py_b = mid.get(f"team{ti}_stone{si}_prev_y_b")
            if px_a is None:
                break

            # Rows where at least one method has a value (stone occupied)
            occupied = px_a.notna() | px_b.notna()
            total += int(occupied.sum())

            both_nan = px_a.isna() & px_b.isna()
            both_match = (
                (px_a - px_b).abs() < tol
            ) & (
                (py_a - py_b).abs() < tol
            )
            agreed += int((both_nan | both_match)[occupied].sum())

    metrics_a = evaluate_tracking(tracked_a)
    metrics_b = evaluate_tracking(tracked_b)

    agreement = agreed / total if total > 0 else float("nan")
    spurious_a = metrics_a["spurious_id_rate"]
    spurious_b = metrics_b["spurious_id_rate"]

    return {
        "link_agreement_rate": round(agreement, 4),
        f"spurious_id_rate_{method_a}": spurious_a,
        f"spurious_id_rate_{method_b}": spurious_b,
        "spurious_id_rate_improvement": round(spurious_a - spurious_b, 4),
        f"new_stone_rate_mid_end_{method_a}": metrics_a["new_stone_rate_mid_end"],
        f"new_stone_rate_mid_end_{method_b}": metrics_b["new_stone_rate_mid_end"],
    }


def displacement_distribution(tracked_df):
    """Compute the distribution of matched-link displacement magnitudes.

    For every stone slot at every non-first shot of an end where the stone was
    successfully matched to its previous position (``prev_x`` is not NaN),
    computes:

        d = sqrt((x - prev_x)^2 + (y - prev_y)^2)

    in normalised house-radius units and examines the resulting distribution.

    A well-calibrated tracker produces a **bimodal** distribution:

    - Mode 1 near zero — stones that did not move (rendering jitter only).
    - Mode 2 large — stones hit hard enough to displace noticeably but still
      matched within ``STONE_TRACK_MAX_DIST``.

    The **threshold zone** ``[0.8 * T, T)`` where ``T = STONE_TRACK_MAX_DIST``
    is the most diagnostic region: a clean gap here means the threshold sits
    between the two modes and is well-calibrated.  A peak in this band means
    the threshold is inside a dense region and may need retuning.

    Parameters
    ----------
    tracked_df : pd.DataFrame
        Output of :func:`apply_tracking`.  Must contain ``_x``, ``_prev_x``,
        ``_y``, ``_prev_y`` columns for each stone slot.

    Returns
    -------
    dict
        ``displacements`` : np.ndarray
            Every matched-link displacement magnitude (length = total_links).
        ``total_links`` : int
            Total matched links across all shots and both teams.
        ``threshold_zone_count`` : int
            Links with displacement in ``[0.8 * STONE_TRACK_MAX_DIST,
            STONE_TRACK_MAX_DIST)``.
        ``threshold_zone_fraction`` : float
            ``threshold_zone_count / total_links``.  Values above ~0.05 are a
            warning that the threshold sits inside a dense region of the
            distribution.
        ``near_zero_fraction`` : float
            Fraction of links with displacement < 0.05 (stationary stones,
            noise-only movement).
        ``median_displacement`` : float
            Median of all displacement magnitudes.
        ``p95_displacement`` : float
            95th-percentile displacement (characterises the tail of hard hits).
    """
    displacements = []

    for ti in (1, 2):
        for si in range(1, MAX_STONES_PER_TEAM + 1):
            x_col = f"team{ti}_stone{si}_x"
            y_col = f"team{ti}_stone{si}_y"
            px_col = f"team{ti}_stone{si}_prev_x"
            py_col = f"team{ti}_stone{si}_prev_y"
            if x_col not in tracked_df.columns or px_col not in tracked_df.columns:
                break
            matched = tracked_df[[x_col, y_col, px_col, py_col]].dropna()
            dx = matched[x_col].to_numpy() - matched[px_col].to_numpy()
            dy = matched[y_col].to_numpy() - matched[py_col].to_numpy()
            displacements.append(np.sqrt(dx ** 2 + dy ** 2))

    if displacements:
        all_d = np.concatenate(displacements)
    else:
        all_d = np.array([])

    total = len(all_d)
    threshold_lo = 0.8 * STONE_TRACK_MAX_DIST

    if total == 0:
        return {
            "displacements": all_d,
            "total_links": 0,
            "threshold_zone_count": 0,
            "threshold_zone_fraction": float("nan"),
            "near_zero_fraction": float("nan"),
            "median_displacement": float("nan"),
            "p95_displacement": float("nan"),
        }

    tz_mask = (all_d >= threshold_lo) & (all_d < STONE_TRACK_MAX_DIST)
    nz_mask = all_d < 0.05

    return {
        "displacements": all_d,
        "total_links": total,
        "threshold_zone_count": int(tz_mask.sum()),
        "threshold_zone_fraction": round(float(tz_mask.sum()) / total, 4),
        "near_zero_fraction": round(float(nz_mask.sum()) / total, 4),
        "median_displacement": round(float(np.median(all_d)), 4),
        "p95_displacement": round(float(np.percentile(all_d, 95)), 4),
    }


def id_continuity_rate(tracked_df):
    """Fraction of non-shot-stone transitions where stone ID is preserved.

    For every consecutive (shot N, shot N+1) pair within an end, finds stone
    slots that are occupied at both shots **and** were not the delivered stone
    at shot N+1 (``is_shot_stone != 1.0``).  These stones should not change
    identity.  Returns the fraction where the ``_id`` is identical.

    Parameters
    ----------
    tracked_df : pd.DataFrame
        Output of :func:`apply_tracking`.

    Returns
    -------
    dict
        ``total_transitions``
            Total (stone, shot-pair) observations considered.
        ``id_continuity_rate``
            Fraction of transitions where the ID was preserved.  1.0 is
            perfect; values below 0.95 indicate excessive re-IDing of
            stationary stones.
    """
    df = tracked_df
    group_keys = ["event_id", "match_id", "end_number"]
    total = 0
    preserved = 0

    for _, end_df in df.groupby(group_keys, sort=True):
        end_df = end_df.sort_values("shot_number").reset_index(drop=True)
        for i in range(len(end_df) - 1):
            row_curr = end_df.iloc[i]
            row_next = end_df.iloc[i + 1]
            for ti in (1, 2):
                for si in range(1, MAX_STONES_PER_TEAM + 1):
                    x_col = f"team{ti}_stone{si}_x"
                    id_col = f"team{ti}_stone{si}_id"
                    ss_col = f"team{ti}_stone{si}_is_shot_stone"
                    if x_col not in df.columns or id_col not in df.columns:
                        break
                    # Both shots must have this stone occupied
                    if pd.isna(row_curr.get(x_col)) or pd.isna(row_next.get(x_col)):
                        continue
                    # Exclude: stone was delivered at the next shot
                    if row_next.get(ss_col) == 1.0:
                        continue
                    total += 1
                    if row_curr.get(id_col) == row_next.get(id_col):
                        preserved += 1

    rate = preserved / total if total > 0 else float("nan")
    return {
        "total_transitions": total,
        "id_continuity_rate": round(rate, 4),
    }


def slot_swap_rate(tracked_df):
    """Rate of impossible stone-ID swaps between consecutive shots.

    A swap occurs when two stones of the same team both present at shot N and
    shot N+1 exchange their ``_id`` values.  This is physically impossible in
    curling (stones cannot pass through each other) and always indicates a
    tracking assignment error.

    Parameters
    ----------
    tracked_df : pd.DataFrame
        Output of :func:`apply_tracking`.

    Returns
    -------
    dict
        ``total_shot_pairs``
            Number of consecutive shot pairs examined across all ends.
        ``swap_events``
            Total number of observed ID swaps (each swap = one pair of
            stones exchanging IDs in one shot transition).
        ``slot_swap_rate``
            ``swap_events / total_shot_pairs``.  0.0 is perfect.
    """
    df = tracked_df
    group_keys = ["event_id", "match_id", "end_number"]
    total_pairs = 0
    swap_events = 0

    for _, end_df in df.groupby(group_keys, sort=True):
        end_df = end_df.sort_values("shot_number").reset_index(drop=True)
        for i in range(len(end_df) - 1):
            row_a = end_df.iloc[i]
            row_b = end_df.iloc[i + 1]
            total_pairs += 1
            for ti in (1, 2):
                # Collect (slot, id) for occupied stones at both shots
                ids_a = {}
                ids_b = {}
                for si in range(1, MAX_STONES_PER_TEAM + 1):
                    x_col = f"team{ti}_stone{si}_x"
                    id_col = f"team{ti}_stone{si}_id"
                    if x_col not in df.columns or id_col not in df.columns:
                        break
                    if pd.notna(row_a.get(x_col)) and pd.notna(row_a.get(id_col)):
                        ids_a[si] = int(row_a[id_col])
                    if pd.notna(row_b.get(x_col)) and pd.notna(row_b.get(id_col)):
                        ids_b[si] = int(row_b[id_col])
                # Find slots present at both shots
                common_slots = set(ids_a) & set(ids_b)
                if len(common_slots) < 2:
                    continue
                # Build reverse maps: id -> slot
                rev_a = {v: k for k, v in ids_a.items() if k in common_slots}
                rev_b = {v: k for k, v in ids_b.items() if k in common_slots}
                common_ids = set(rev_a) & set(rev_b)
                # Count swaps: id X moved to slot of id Y, and id Y moved to slot of id X
                checked = set()
                for id_x in common_ids:
                    if id_x in checked:
                        continue
                    slot_x_in_a = rev_a[id_x]
                    slot_x_in_b = rev_b[id_x]
                    if slot_x_in_a == slot_x_in_b:
                        continue
                    # Check if whatever was in slot_x_in_b at shot A moved to slot_x_in_a
                    id_y = ids_a.get(slot_x_in_b)
                    if id_y is not None and ids_b.get(slot_x_in_a) == id_y:
                        swap_events += 1
                        checked.add(id_x)
                        checked.add(id_y)

    rate = swap_events / total_pairs if total_pairs > 0 else float("nan")
    return {
        "total_shot_pairs": total_pairs,
        "swap_events": swap_events,
        "slot_swap_rate": round(rate, 4),
    }


def cap_pressure_rate(tracked_df):
    """Fraction of new-ID events whose nearest previous stone is just outside the cap.

    For each mid-end stone appearance that received a new ID (treated as a
    fresh delivery), computes the distance to every stone in the previous shot
    for the same team.  If the nearest distance falls in the zone
    ``[cap, cap + 0.05)`` the match was rejected only because the cap was
    exceeded by a small margin — a candidate false rejection.

    High values suggest ``STONE_TRACK_MAX_DIST`` should be raised.

    Parameters
    ----------
    tracked_df : pd.DataFrame
        Output of :func:`apply_tracking`.

    Returns
    -------
    dict
        ``total_new_id_events``
            Total mid-end stone appearances that received a new ID.
        ``cap_pressure_events``
            Subset where the nearest previous stone is in ``[cap, cap+0.05)``.
        ``cap_pressure_rate``
            ``cap_pressure_events / total_new_id_events``.  Values above ~0.10
            suggest the cap should be increased.
    """
    df = tracked_df
    group_keys = ["event_id", "match_id", "end_number"]
    pressure_zone = 0.05
    cap = STONE_TRACK_MAX_DIST
    total_new = 0
    cap_pressure = 0

    for _, end_df in df.groupby(group_keys, sort=True):
        end_df = end_df.sort_values("shot_number").reset_index(drop=True)
        first_shot = end_df["shot_number"].min()
        mid_df = end_df[end_df["shot_number"] > first_shot]

        for i, (_, row) in enumerate(mid_df.iterrows()):
            # Get the previous shot row
            shot_idx = end_df[end_df["shot_number"] == row["shot_number"]].index[0]
            if shot_idx == 0:
                continue
            prev_row = end_df.iloc[shot_idx - 1]

            for ti in (1, 2):
                # Collect previous-shot stone positions for this team
                prev_positions = []
                for si in range(1, MAX_STONES_PER_TEAM + 1):
                    px_col = f"team{ti}_stone{si}_x"
                    py_col = f"team{ti}_stone{si}_y"
                    if px_col not in df.columns:
                        break
                    px = prev_row.get(px_col)
                    py = prev_row.get(py_col)
                    if pd.notna(px) and pd.notna(py):
                        prev_positions.append((float(px), float(py)))

                for si in range(1, MAX_STONES_PER_TEAM + 1):
                    x_col = f"team{ti}_stone{si}_x"
                    y_col = f"team{ti}_stone{si}_y"
                    prev_x_col = f"team{ti}_stone{si}_prev_x"
                    if x_col not in df.columns or prev_x_col not in df.columns:
                        break
                    nx = row.get(x_col)
                    ny = row.get(y_col)
                    prev_x = row.get(prev_x_col)
                    # Only consider mid-end new-ID stones
                    if pd.isna(nx) or pd.notna(prev_x):
                        continue
                    total_new += 1
                    if not prev_positions:
                        continue
                    min_dist = min(
                        math.sqrt((nx - px) ** 2 + (ny - py) ** 2)
                        for px, py in prev_positions
                    )
                    if cap <= min_dist < cap + pressure_zone:
                        cap_pressure += 1

    rate = cap_pressure / total_new if total_new > 0 else float("nan")
    return {
        "total_new_id_events": total_new,
        "cap_pressure_events": cap_pressure,
        "cap_pressure_rate": round(rate, 4),
    }


def displacement_symmetry(tracked_df):
    """Compare displacement distributions for shot stones vs. still stones.

    Uses the ``is_shot_stone`` flag to separate matched-link displacements
    into two groups: the delivered stone (``is_shot_stone == 1.0``) and all
    other stones already on the board (``is_shot_stone == 0.0``).

    A well-tracked dataset should show clear separation: still-stones cluster
    near zero (rendering jitter only) while shot-stones span a wider range.

    Parameters
    ----------
    tracked_df : pd.DataFrame
        Output of :func:`apply_tracking`.  Requires ``_is_shot_stone``,
        ``_prev_x``, ``_prev_y`` columns.

    Returns
    -------
    dict
        ``still_stone_median_displacement``
            Should be near zero (< 0.01).
        ``still_stone_p95_displacement``
            A value close to the cap signals noise issues.
        ``shot_stone_median_displacement``
            No fixed ideal; reflects typical stone delivery distance.
        ``shot_stone_p95_displacement``
            95th-percentile delivery displacement.
        ``separation_ratio``
            ``shot_stone_median / still_stone_median``.  Higher is better;
            values below ~5 suggest ``is_shot_stone`` quality problems or
            excessive still-stone jitter.
        ``still_stone_links``
        ``shot_stone_links``
            Number of observations in each group.
    """
    df = tracked_df
    still_disps = []
    shot_disps = []

    for ti in (1, 2):
        for si in range(1, MAX_STONES_PER_TEAM + 1):
            x_col = f"team{ti}_stone{si}_x"
            y_col = f"team{ti}_stone{si}_y"
            px_col = f"team{ti}_stone{si}_prev_x"
            py_col = f"team{ti}_stone{si}_prev_y"
            ss_col = f"team{ti}_stone{si}_is_shot_stone"
            if not all(c in df.columns for c in [x_col, px_col, ss_col]):
                break
            matched = df[df[px_col].notna() & df[x_col].notna()].copy()
            if matched.empty:
                continue
            dx = matched[x_col].to_numpy() - matched[px_col].to_numpy()
            dy = matched[y_col].to_numpy() - matched[py_col].to_numpy()
            d = np.sqrt(dx ** 2 + dy ** 2)
            ss = matched[ss_col].to_numpy()
            still_disps.append(d[ss == 0.0])
            shot_disps.append(d[ss == 1.0])

    still = np.concatenate(still_disps) if still_disps else np.array([])
    shot = np.concatenate(shot_disps) if shot_disps else np.array([])

    def _stats(arr):
        if len(arr) == 0:
            return float("nan"), float("nan")
        return round(float(np.median(arr)), 4), round(float(np.percentile(arr, 95)), 4)

    still_med, still_p95 = _stats(still)
    shot_med, shot_p95 = _stats(shot)

    if still_med and still_med > 0:
        sep = round(shot_med / still_med, 2) if not math.isnan(shot_med) else float("nan")
    else:
        sep = float("nan")

    return {
        "still_stone_links": len(still),
        "shot_stone_links": len(shot),
        "still_stone_median_displacement": still_med,
        "still_stone_p95_displacement": still_p95,
        "shot_stone_median_displacement": shot_med,
        "shot_stone_p95_displacement": shot_p95,
        "separation_ratio": sep,
    }


def stone_count_consistency_rate(tracked_df):
    """Fraction of consecutive shot pairs where stone count changes by at most 1.

    Between consecutive shots, each team's ``stones_in_play`` should change
    by at most ±1 (one stone delivered or one taken out).  A jump of ±2 or
    more almost always reflects an upstream extraction error (the detector
    hallucinated or dropped a stone), not a tracking error.

    Parameters
    ----------
    tracked_df : pd.DataFrame
        Output of :func:`apply_tracking`.  Requires
        ``team{N}_stones_in_play`` columns.

    Returns
    -------
    dict
        ``total_shot_pairs``
            Consecutive shot pairs examined.
        ``inconsistent_pairs``
            Pairs where ``|Δstones_in_play| > 1`` for at least one team.
        ``stone_count_consistency_rate``
            ``1 - inconsistent_pairs / total_shot_pairs``.  1.0 is perfect.
        ``inconsistent_ends``
            Number of distinct ends containing at least one inconsistent pair.
    """
    df = tracked_df
    group_keys = ["event_id", "match_id", "end_number"]
    total_pairs = 0
    inconsistent_pairs = 0
    inconsistent_ends = 0

    for _, end_df in df.groupby(group_keys, sort=True):
        end_df = end_df.sort_values("shot_number").reset_index(drop=True)
        end_has_inconsistency = False
        for i in range(len(end_df) - 1):
            row_a = end_df.iloc[i]
            row_b = end_df.iloc[i + 1]
            total_pairs += 1
            bad = False
            for ti in (1, 2):
                sip_col = f"team{ti}_stones_in_play"
                if sip_col not in df.columns:
                    continue
                delta = abs(int(row_b[sip_col]) - int(row_a[sip_col]))
                if delta > 1:
                    bad = True
                    break
            if bad:
                inconsistent_pairs += 1
                end_has_inconsistency = True
        if end_has_inconsistency:
            inconsistent_ends += 1

    rate = (
        1.0 - inconsistent_pairs / total_pairs if total_pairs > 0 else float("nan")
    )
    return {
        "total_shot_pairs": total_pairs,
        "inconsistent_pairs": inconsistent_pairs,
        "stone_count_consistency_rate": round(rate, 4),
        "inconsistent_ends": inconsistent_ends,
    }


def delivery_anomaly_rate(tracked_df):
    """Fraction of non-first shots where new stone ID count is not exactly 1.

    In standard 4-person curling each non-first shot delivers exactly one
    stone, so exactly one stone slot should receive a new ID (``prev_x`` is
    NaN) per shot across both teams combined.  A count ≠ 1 indicates either:

    - A tracking miss (stone failed to match, received spurious new ID).
    - A stone disappearing off-screen without a corresponding delivery
      (stone left play but no new stone was placed — legitimate in short
      ends / conceded ends).
    - Multi-stone confusion from heavy take-outs (rare edge cases where the
      same shot clears and replaces multiple stones simultaneously).

    This is a necessary-but-not-sufficient signal: a near-zero rate confirms
    the tracker creates IDs at the right *frequency* but cannot confirm the
    assignments to individual stones are correct.

    Parameters
    ----------
    tracked_df : pd.DataFrame
        Output of :func:`apply_tracking`.  Requires ``_x`` and ``_prev_x``
        columns for all stone slots and an ``is_shot_stone`` column.

    Returns
    -------
    dict
        ``total_non_first_shots``
            Number of non-first shots examined.
        ``delivery_anomaly_count``
            Shots where new-ID count ≠ 1.
        ``delivery_anomaly_rate``
            ``delivery_anomaly_count / total_non_first_shots``.  Near-zero
            is ideal; values above ~0.10 suggest cap miscalibration.
        ``anomaly_examples``
            Up to 5 representative anomalous shots for inspection (dicts
            with ``event_id``, ``match_id``, ``end_number``,
            ``shot_number``, ``new_ids_created``).
    """
    df = tracked_df
    group_keys = ["event_id", "match_id", "end_number"]
    total_shots = 0
    anomalous = 0
    examples = []

    for (event_id, match_id, end_number), end_df in df.groupby(group_keys, sort=True):
        end_df = end_df.sort_values("shot_number").reset_index(drop=True)
        first_shot = end_df["shot_number"].min()
        mid_df = end_df[end_df["shot_number"] > first_shot]

        for _, row in mid_df.iterrows():
            total_shots += 1
            new_ids = 0
            for ti in (1, 2):
                for si in range(1, MAX_STONES_PER_TEAM + 1):
                    x_col = f"team{ti}_stone{si}_x"
                    prev_col = f"team{ti}_stone{si}_prev_x"
                    if x_col not in df.columns or prev_col not in df.columns:
                        break
                    if pd.notna(row.get(x_col)) and pd.isna(row.get(prev_col)):
                        new_ids += 1
            if new_ids != 1:
                anomalous += 1
                if len(examples) < 5:
                    examples.append({
                        "event_id": int(event_id),
                        "match_id": int(match_id),
                        "end_number": int(end_number),
                        "shot_number": int(row["shot_number"]),
                        "new_ids_created": new_ids,
                    })

    rate = anomalous / total_shots if total_shots > 0 else float("nan")
    return {
        "total_non_first_shots": total_shots,
        "delivery_anomaly_count": anomalous,
        "delivery_anomaly_rate": round(rate, 4),
        "anomaly_examples": examples,
    }


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _read_raw(path):
    """Read raw shot locations from parquet or CSV."""
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _write_tracked(df, output_dir, method):
    """Write the tracked DataFrame to shot_locations.csv and .parquet."""
    os.makedirs(output_dir, exist_ok=True)

    # Column order: base fields, raw stone fields, then tracking fields
    base_and_raw = [c for c in df.columns if not any(
        c.endswith(s) for s in ("_id", "_prev_x", "_prev_y", "_is_shot_stone")
    )]
    tracking = [c for c in df.columns if any(
        c.endswith(s) for s in ("_id", "_prev_x", "_prev_y", "_is_shot_stone")
    )]
    ordered = base_and_raw + tracking
    df = df[[c for c in ordered if c in df.columns]]

    csv_path = os.path.join(output_dir, "shot_locations.csv")
    parquet_path = os.path.join(output_dir, "shot_locations.parquet")

    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)
    print(f"Written {len(df):,} rows to {csv_path} and {parquet_path}")
    print(f"  Tracking method: {method}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Apply sequential stone tracking to raw shot-location data."
    )
    parser.add_argument(
        "--method",
        choices=TRACKING_METHODS,
        default="greedy",
        help="Tracking algorithm (default: greedy)",
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help=f"Path to shot_locations_raw.parquet or .csv (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for output files (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Print evaluation metrics after tracking.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help=(
            "Run both greedy and hungarian, print comparison metrics, "
            "and write the --method result. Requires scipy."
        ),
    )
    parser.add_argument(
        "--displacement",
        action="store_true",
        help=(
            "Print displacement magnitude distribution metrics after tracking. "
            "Diagnoses whether STONE_TRACK_MAX_DIST is well-calibrated."
        ),
    )
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        parser.error(f"Input file not found: {args.input}")

    print(f"Reading {args.input} …")
    raw_df = _read_raw(args.input)
    print(f"  {len(raw_df):,} shots loaded.")

    if args.compare:
        print("\nApplying greedy tracking …")
        tracked_greedy = apply_tracking(raw_df, method="greedy")
        print("Applying hungarian tracking …")
        tracked_hungarian = apply_tracking(raw_df, method="hungarian")
        print("\nComparison results:")
        results = compare_tracking(
            tracked_greedy, tracked_hungarian,
            method_a="greedy", method_b="hungarian"
        )
        for k, v in results.items():
            print(f"  {k}: {v}")
        # Write the requested method's output
        tracked = tracked_greedy if args.method == "greedy" else tracked_hungarian
    else:
        print(f"\nApplying {args.method} tracking …")
        tracked = apply_tracking(raw_df, method=args.method)

    if args.evaluate:
        print("\nEvaluation metrics:")
        metrics = evaluate_tracking(tracked)
        for k, v in metrics.items():
            print(f"  {k}: {v}")

    if args.displacement:
        print("\nDisplacement distribution metrics:")
        disp = displacement_distribution(tracked)
        for k, v in disp.items():
            if k == "displacements":
                print(f"  displacements: array of {len(v)} values")
            else:
                print(f"  {k}: {v}")
        pct = disp["threshold_zone_fraction"]
        if not (pct != pct):  # NaN check
            if pct > 0.05:
                print(
                    f"  WARNING: {pct:.1%} of matched links fall in the threshold zone "
                    f"[{0.8 * STONE_TRACK_MAX_DIST:.3f}, {STONE_TRACK_MAX_DIST:.3f}). "
                    "Consider retuning STONE_TRACK_MAX_DIST."
                )
            else:
                print(
                    f"  OK: threshold zone fraction {pct:.1%} is below 5% — "
                    "threshold appears well-calibrated."
                )

    _write_tracked(tracked, args.output_dir, args.method)


if __name__ == "__main__":
    main()
