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
    )
"""

import argparse
import math
import os

import numpy as np
import pandas as pd

from extract_shot_data import MAX_STONES_PER_TEAM, STONE_TRACK_MAX_DIST

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

    group_keys = ["event_id", "match_id", "end_number"]
    for _, end_idx in df.groupby(group_keys, sort=True).groups.items():
        end_view = df.loc[end_idx].sort_values("shot_number")
        track_state = {1: [], 2: []}
        stone_id_counter = [1]

        for row_pos in end_view.index:
            row = df.loc[row_pos]
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
                            is_shot = bool(ti == shot_ti and si == shot_si)
                            df.at[row_pos, f"team{ti}_stone{si}_is_shot_stone"] = is_shot

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
        ``total_ids_assigned``
            Total unique stone IDs assigned, summed across all ends and teams.
            Each end resets the counter, so IDs are counted per-end.
        ``min_ids_needed``
            Theoretical minimum IDs: for each end and team, the maximum
            ``stones_in_play`` value seen in that end.  A perfect tracker
            never creates more IDs than this.
        ``fragmentation_index``
            ``total_ids_assigned / min_ids_needed``.  1.0 is perfect;
            higher values indicate unnecessary ID creation.
        ``new_stone_rate_mid_end``
            Fraction of non-first-shot stone appearances where the stone
            received a new ID (``prev_x`` is NaN), i.e. was not matched to
            any stone from the previous shot.  Lower is generally better,
            though genuine new arrivals (deliveries) legitimately produce
            one new ID per shot.
    """
    df = tracked_df
    group_keys = ["event_id", "match_id", "end_number"]

    total_ids_assigned = 0
    min_ids_needed = 0
    total_appearances = 0
    new_mid_end = 0
    total_mid_end = 0

    for (event_id, match_id, end_number), end_df in df.groupby(group_keys, sort=True):
        end_df_sorted = end_df.sort_values("shot_number")
        first_shot = end_df_sorted["shot_number"].min()

        for ti in (1, 2):
            # Unique IDs used for this team in this end
            id_vals = set()
            for si in range(1, MAX_STONES_PER_TEAM + 1):
                col = f"team{ti}_stone{si}_id"
                if col in df.columns:
                    vals = end_df_sorted[col].dropna()
                    id_vals.update(int(v) for v in vals)
            total_ids_assigned += len(id_vals)

            # Minimum IDs needed = max stones in play at any shot
            sip_col = f"team{ti}_stones_in_play"
            min_ids_needed += int(end_df_sorted[sip_col].max())

            # New-stone rate mid-end (shot_number > first in end)
            mid_df = end_df_sorted[end_df_sorted["shot_number"] > first_shot]
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

    fragmentation = (
        total_ids_assigned / min_ids_needed if min_ids_needed > 0 else float("nan")
    )
    new_rate = (
        new_mid_end / total_mid_end if total_mid_end > 0 else float("nan")
    )

    return {
        "total_ends": int(df.groupby(group_keys).ngroups),
        "total_stone_appearances": total_appearances,
        "total_ids_assigned": total_ids_assigned,
        "min_ids_needed": min_ids_needed,
        "fragmentation_index": round(fragmentation, 4),
        "new_stone_rate_mid_end": round(new_rate, 4),
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
        ``fragmentation_index_{method_a}``
        ``fragmentation_index_{method_b}``
            Fragmentation index from :func:`evaluate_tracking` for each.
        ``fragmentation_improvement``
            ``frag_a − frag_b``.  Positive means method B is better
            (less fragmentation).
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
    frag_a = metrics_a["fragmentation_index"]
    frag_b = metrics_b["fragmentation_index"]

    return {
        "link_agreement_rate": round(agreement, 4),
        f"fragmentation_index_{method_a}": frag_a,
        f"fragmentation_index_{method_b}": frag_b,
        "fragmentation_improvement": round(frag_a - frag_b, 4),
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
