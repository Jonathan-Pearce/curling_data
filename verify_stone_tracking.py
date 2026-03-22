"""
Stone Tracking Verification Script
===================================
Validates that the sequential stone-tracking data written by extract_shot_data.py
is internally consistent and physically plausible.

Run after scraping to verify any parquet file::

    python verify_stone_tracking.py output/shot_locations.parquet
    python verify_stone_tracking.py output/shot_locations.parquet --event 1 --match 2
    python verify_stone_tracking.py output/shot_locations.parquet --visualise --output-dir verify_out

Checks performed
----------------
1.  Schema — new tracking columns (_id, _prev_x, _prev_y) are present and numeric.
2.  First-shot invariant — shot_number == 1 rows have NULL prev_x / prev_y for all
    stones (no prior state exists at the start of an end).
3.  Non-first-shot coverage — matched stones on shot N > 1 have prev_x populated;
    newly placed stones (stone count increase) have prev_x = NULL.
4.  ID uniqueness within an end — no stone_id appears for both team1 and team2 in
    the same end.  IDs are distinct integers > 0.
5.  ID continuity — once a stone_id appears for a team it remains present in all
    subsequent shots of the same end until the stone leaves play.  It never
    reappears after being absent.
6.  Matched-stone displacement — the distance from (prev_x, prev_y) to (x, y) for
    matched stones is << STONE_TRACK_MAX_DIST (0.10). Distances above 0.05 are
    flagged as suspicious rendering noise.
7.  New-stone count — on each shot the number of stones with NULL prev is ≤ 2 (at
    most one per team), and approximately one per team on shots where a stone was
    first placed.
8.  (Optional) Visual board images showing displacement arrows for a selected end,
    allowing manual comparison to the observed PDF diagrams.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from collections import defaultdict

import pandas as pd

# ---------------------------------------------------------------------------
# Constants (mirrors extract_shot_data.py)
# ---------------------------------------------------------------------------
MAX_STONES_PER_TEAM = 8
STONE_TRACK_MAX_DIST = 0.10   # matching threshold used during scraping
NOISE_WARN_DIST = 0.05        # displacement above this is flagged as suspicious

STONE_COLS_PER_TEAM = ["x", "y", "dist", "angle", "id", "prev_x", "prev_y"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stone_col(team: int, slot: int, field: str) -> str:
    return f"team{team}_stone{slot}_{field}"


def _iter_stone_slots(df: pd.DataFrame, team: int):
    """Yield (slot, x_col, y_col, id_col, prev_x_col, prev_y_col) for all 8 slots."""
    for slot in range(1, MAX_STONES_PER_TEAM + 1):
        yield (
            slot,
            _stone_col(team, slot, "x"),
            _stone_col(team, slot, "y"),
            _stone_col(team, slot, "id"),
            _stone_col(team, slot, "prev_x"),
            _stone_col(team, slot, "prev_y"),
        )


def _check_columns_present(df: pd.DataFrame) -> list[str]:
    """Check 1: all tracking columns exist in the dataframe."""
    errors = []
    for team in (1, 2):
        for slot in range(1, MAX_STONES_PER_TEAM + 1):
            for field in ("id", "prev_x", "prev_y"):
                col = _stone_col(team, slot, field)
                if col not in df.columns:
                    errors.append(f"Missing column: {col}")
    return errors


def _check_first_shot_invariant(df: pd.DataFrame) -> list[str]:
    """Check 2: shot 1 of every end must have NULL prev_x and prev_y for all stones."""
    errors = []
    shot1 = df[df["shot_number"] == 1]
    for team in (1, 2):
        for slot in range(1, MAX_STONES_PER_TEAM + 1):
            prev_x_col = _stone_col(team, slot, "prev_x")
            prev_y_col = _stone_col(team, slot, "prev_y")
            if prev_x_col not in df.columns:
                continue
            bad_x = shot1[prev_x_col].notna().sum()
            bad_y = shot1[prev_y_col].notna().sum()
            if bad_x:
                errors.append(
                    f"Check 2 FAIL: {bad_x} shot-1 rows have non-NULL "
                    f"{prev_x_col} (should always be NULL at end start)"
                )
            if bad_y:
                errors.append(
                    f"Check 2 FAIL: {bad_y} shot-1 rows have non-NULL "
                    f"{prev_y_col} (should always be NULL at end start)"
                )
    return errors


def _check_matched_displacement(df: pd.DataFrame) -> tuple[list[str], pd.DataFrame]:
    """Check 6: displacement from prev to current for matched stones.

    Returns (errors, detail_df) where detail_df has one row per stone-slot
    with non-NULL prev and columns (event_id, match_id, end_number,
    shot_number, team, slot, dx, dy, displacement).
    """
    errors = []
    records = []
    non_first = df[df["shot_number"] > 1]

    for team in (1, 2):
        for slot in range(1, MAX_STONES_PER_TEAM + 1):
            x_col = _stone_col(team, slot, "x")
            y_col = _stone_col(team, slot, "y")
            prev_x_col = _stone_col(team, slot, "prev_x")
            prev_y_col = _stone_col(team, slot, "prev_y")
            if prev_x_col not in df.columns:
                continue

            matched = non_first[non_first[prev_x_col].notna()].copy()
            if matched.empty:
                continue

            matched = matched.assign(
                dx=matched[x_col] - matched[prev_x_col],
                dy=matched[y_col] - matched[prev_y_col],
            )
            matched["displacement"] = (
                matched["dx"] ** 2 + matched["dy"] ** 2
            ).pow(0.5)
            matched["team"] = team
            matched["slot"] = slot

            suspicious = matched[matched["displacement"] > NOISE_WARN_DIST]
            if not suspicious.empty:
                errors.append(
                    f"Check 6 WARN: team{team} slot {slot} — "
                    f"{len(suspicious)} matched stones have displacement "
                    f"> {NOISE_WARN_DIST:.2f} (max={suspicious['displacement'].max():.3f}). "
                    f"These may indicate a mis-match or high rendering noise."
                )

            records.append(matched[
                ["event_id", "match_id", "end_number", "shot_number",
                 "team", "slot", "dx", "dy", "displacement"]
            ])

    detail_df = pd.concat(records, ignore_index=True) if records else pd.DataFrame()
    return errors, detail_df


def _check_id_uniqueness(df: pd.DataFrame) -> list[str]:
    """Check 4: no stone_id should appear for both team1 and team2 within the same end."""
    errors = []
    end_key = ["event_id", "match_id", "end_number"]

    for team in (1, 2):
        for slot in range(1, MAX_STONES_PER_TEAM + 1):
            id_col = _stone_col(team, slot, "id")
            if id_col not in df.columns:
                continue

    # Collect all (end_key, stone_id, team) triples
    rows = []
    for team in (1, 2):
        for slot in range(1, MAX_STONES_PER_TEAM + 1):
            id_col = _stone_col(team, slot, "id")
            if id_col not in df.columns:
                continue
            sub = df[df[id_col].notna()][end_key + [id_col]].copy()
            sub["team"] = team
            sub = sub.rename(columns={id_col: "stone_id"})
            rows.append(sub)

    if not rows:
        return errors

    all_ids = pd.concat(rows, ignore_index=True).drop_duplicates(
        subset=end_key + ["stone_id", "team"]
    )
    # Flag any stone_id that appears for both team1 and team2 in the same end
    grouped = all_ids.groupby(end_key + ["stone_id"])["team"].nunique()
    overlap = grouped[grouped > 1]
    if len(overlap):
        errors.append(
            f"Check 4 FAIL: {len(overlap)} (end, stone_id) pairs appear for "
            f"both team1 and team2. Stone IDs must be globally unique within "
            f"an end. Sample:\n{overlap.head(5)}"
        )
    return errors


def _check_id_continuity(df: pd.DataFrame) -> list[str]:
    """Check 5: a stone that was present at shot N should not reappear at shot M > N+k
    after being absent — IDs should not be re-used once a stone leaves play."""
    errors = []
    end_key = ["event_id", "match_id", "end_number"]

    # Build per-end set of (team, stone_id) -> sorted list of shot_numbers present
    presence: dict = defaultdict(list)
    for team in (1, 2):
        for slot in range(1, MAX_STONES_PER_TEAM + 1):
            id_col = _stone_col(team, slot, "id")
            if id_col not in df.columns:
                continue
            present = df[df[id_col].notna()].copy()
            present["_team"] = team
            for _, row in present.iterrows():
                key = (
                    row["event_id"], row["match_id"], row["end_number"],
                    team, int(row[id_col]),
                )
                presence[key].append(int(row["shot_number"]))

    reappearances = 0
    sample = []
    for key, shots in presence.items():
        shots_sorted = sorted(set(shots))
        # A stone can leave play (miss a shot) and a re-detection at a later
        # shot is a reappearance.  We flag gaps > 1 shot.
        for i in range(len(shots_sorted) - 1):
            gap = shots_sorted[i + 1] - shots_sorted[i]
            if gap > 1:
                reappearances += 1
                if len(sample) < 5:
                    sample.append(
                        f"event={key[0]} match={key[1]} end={key[2]} "
                        f"team={key[3]} stone_id={key[4]} "
                        f"absent between shots {shots_sorted[i]}–{shots_sorted[i+1]}"
                    )

    if reappearances:
        errors.append(
            f"Check 5 WARN: {reappearances} stone IDs reappear after being absent "
            f"for ≥1 shot (possible re-use or detection artifact).\n"
            + "\n".join(f"  {s}" for s in sample)
        )
    return errors


def _check_new_stone_count(df: pd.DataFrame) -> list[str]:
    """Check 7: on any single shot, at most 2 stones should have NULL prev
    (one newly placed stone per team is expected; more suggests tracking errors)."""
    errors = []
    non_first = df[df["shot_number"] > 1].copy()

    for team in (1, 2):
        prev_x_cols = [
            _stone_col(team, slot, "prev_x")
            for slot in range(1, MAX_STONES_PER_TEAM + 1)
            if _stone_col(team, slot, "prev_x") in df.columns
        ]
        x_cols = [
            _stone_col(team, slot, "x")
            for slot in range(1, MAX_STONES_PER_TEAM + 1)
            if _stone_col(team, slot, "x") in df.columns
        ]
        if not prev_x_cols:
            continue

        # Count new stones per shot: slots where x is not NULL but prev_x is NULL
        # (i.e. stone is present but has no previous match)
        null_prev = non_first[prev_x_cols].isna()
        has_x = non_first[x_cols].notna()
        # Align by slot index
        new_stones_per_slot = (null_prev.values & has_x.values)
        new_count_per_shot = new_stones_per_slot.sum(axis=1)
        bad = (new_count_per_shot > 1).sum()
        if bad:
            errors.append(
                f"Check 7 WARN: team{team} has {bad} shots where >1 stone slot "
                f"has NULL prev_x while also having an active stone (expected ≤1 "
                f"new stone per team per shot)."
            )
    return errors


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def _displacement_stats(detail_df: pd.DataFrame) -> str:
    if detail_df.empty:
        return "  (no matched stones found)"
    d = detail_df["displacement"]
    return (
        f"  n={len(d):,}  "
        f"mean={d.mean():.4f}  "
        f"median={d.median():.4f}  "
        f"p95={d.quantile(0.95):.4f}  "
        f"max={d.max():.4f}  "
        f"(threshold={STONE_TRACK_MAX_DIST:.2f})"
    )


def _coverage_stats(df: pd.DataFrame) -> str:
    total = len(df)
    shot1 = (df["shot_number"] == 1).sum()
    non_first = total - shot1
    t1_id_col = _stone_col(1, 1, "id")
    id_present = df[t1_id_col].notna().sum() if t1_id_col in df.columns else 0
    return (
        f"  total rows={total:,}  "
        f"shot-1 rows={shot1:,}  "
        f"non-first rows={non_first:,}  "
        f"stone-1-team1 IDs assigned={id_present:,}"
    )


# ---------------------------------------------------------------------------
# Visual verification (optional)
# ---------------------------------------------------------------------------

def _generate_tracking_images(df: pd.DataFrame, output_dir: str,
                               event_id: int, match_id: int, end_number: int):
    """Render a board image per shot for one end with displacement arrows
    showing where each stone came from (prev_x, prev_y → x, y).

    Requires Pillow (already a project dependency).
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
        import generate_board_image as gbi
    except ImportError as e:
        print(f"  [visual] Skipping: {e}")
        return

    os.makedirs(output_dir, exist_ok=True)
    end_df = df[
        (df["event_id"] == event_id)
        & (df["match_id"] == match_id)
        & (df["end_number"] == end_number)
    ].sort_values("shot_number")

    if end_df.empty:
        print(f"  [visual] No rows for event={event_id} match={match_id} end={end_number}")
        return

    for _, row_s in end_df.iterrows():
        row = {k: ("" if pd.isna(v) else v) for k, v in row_s.items()}
        img = gbi.generate_board_image(row)
        draw = ImageDraw.Draw(img)

        # Draw displacement arrows for matched stones
        for team, fill in ((1, (180, 20, 20, 200)), (2, (160, 130, 20, 200))):
            for slot in range(1, MAX_STONES_PER_TEAM + 1):
                x_col = _stone_col(team, slot, "x")
                y_col = _stone_col(team, slot, "y")
                px_col = _stone_col(team, slot, "prev_x")
                py_col = _stone_col(team, slot, "prev_y")
                if any(c not in row_s.index for c in (x_col, y_col, px_col, py_col)):
                    continue
                cx = row_s.get(x_col)
                cy = row_s.get(y_col)
                px = row_s.get(px_col)
                py = row_s.get(py_col)
                if any(pd.isna(v) for v in (cx, cy, px, py)):
                    continue
                # Convert normalised coords to pixels
                x1, y1 = gbi._norm_to_pixel(float(px), float(py))
                x2, y2 = gbi._norm_to_pixel(float(cx), float(cy))
                # Dashed arrow: origin circle → current position
                draw.ellipse(
                    (x1 - 4, y1 - 4, x1 + 4, y1 + 4),
                    outline=(80, 80, 80), width=1,
                )
                draw.line([(x1, y1), (x2, y2)], fill=(80, 80, 80), width=1)

        shot_num = int(row_s["shot_number"])
        fname = (
            f"e{event_id}_m{match_id}_end{end_number}_shot{shot_num:02d}.png"
        )
        img.save(os.path.join(output_dir, fname))

    print(
        f"  [visual] Saved {len(end_df)} board images to {output_dir}/ "
        f"(event={event_id}, match={match_id}, end={end_number})"
    )


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_checks(parquet_path: str,
               event_id: int | None = None,
               match_id: int | None = None,
               visualise: bool = False,
               vis_end: int = 1,
               output_dir: str = "verify_out") -> bool:
    """Load *parquet_path*, optionally filter to one match, and run all checks.

    Returns True if no FAILs were found (warnings are still printed).
    """
    print(f"\n{'='*60}")
    print(f"Stone Tracking Verification")
    print(f"File: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns.")

    if event_id is not None:
        df = df[df["event_id"] == event_id]
        print(f"Filtered to event_id={event_id}: {len(df):,} rows remaining.")
    if match_id is not None:
        df = df[df["match_id"] == match_id]
        print(f"Filtered to match_id={match_id}: {len(df):,} rows remaining.")

    if df.empty:
        print("ERROR: No rows after filtering.")
        return False

    all_errors = []
    all_warnings = []

    # ---- Check 1: Schema ---------------------------------------------------
    print(f"\n{'─'*40}")
    print("Check 1: Schema — new tracking columns present")
    schema_errors = _check_columns_present(df)
    if schema_errors:
        print(f"  FAIL ({len(schema_errors)} errors):")
        for e in schema_errors:
            print(f"    {e}")
        all_errors.extend(schema_errors)
        print(
            "\n  Cannot continue — parquet was produced without tracking code.\n"
            "  Re-scrape with the updated extract_shot_data.py first."
        )
        return False
    else:
        # Report id column dtype
        id_col = _stone_col(1, 1, "id")
        print(f"  PASS — all _id / _prev_x / _prev_y columns present.")
        print(f"  Sample dtype: {id_col} = {df[id_col].dtype}")

    # ---- Check 2: First-shot invariant ------------------------------------
    print(f"\n{'─'*40}")
    print("Check 2: First-shot invariant — shot 1 has NULL prev_x/prev_y")
    errs = _check_first_shot_invariant(df)
    fails = [e for e in errs if "FAIL" in e]
    if fails:
        all_errors.extend(fails)
        for e in fails:
            print(f"  {e}")
    else:
        print(f"  PASS")

    # ---- Check 3: Non-first-shot prev coverage ----------------------------
    print(f"\n{'─'*40}")
    print("Check 3: Non-first-shot coverage — matched stones have prev values")
    non_first = df[df["shot_number"] > 1]
    total_active = 0
    total_matched = 0
    for team in (1, 2):
        for slot in range(1, MAX_STONES_PER_TEAM + 1):
            x_col = _stone_col(team, slot, "x")
            px_col = _stone_col(team, slot, "prev_x")
            if x_col not in df.columns or px_col not in df.columns:
                continue
            active = non_first[x_col].notna().sum()
            matched = non_first[px_col].notna().sum()
            total_active += active
            total_matched += matched
    if total_active == 0:
        print("  SKIP — no active stones on non-first shots found.")
    else:
        match_pct = 100.0 * total_matched / total_active
        print(f"  Active stone slots (shot > 1): {total_active:,}")
        print(f"  Of which have prev_x populated: {total_matched:,} ({match_pct:.1f}%)")
        # Expect most stones to be matched (only ~1/N are new each shot)
        # A match_pct below 50% would indicate a tracking problem.
        if match_pct < 50.0:
            msg = (
                f"Check 3 WARN: only {match_pct:.1f}% of active stones have "
                f"prev_x. Expected > 50% if tracking is working. This could "
                f"mean STONE_TRACK_MAX_DIST is too tight or coords are noisy."
            )
            print(f"  {msg}")
            all_warnings.append(msg)
        else:
            print("  PASS (match rate > 50%)")

    # ---- Check 4: ID uniqueness -------------------------------------------
    print(f"\n{'─'*40}")
    print("Check 4: ID uniqueness — no stone_id shared between team1 and team2 in same end")
    errs = _check_id_uniqueness(df)
    fails = [e for e in errs if "FAIL" in e]
    if fails:
        all_errors.extend(fails)
        for e in fails:
            print(f"  {e}")
    else:
        print("  PASS")

    # ---- Check 5: ID continuity -------------------------------------------
    print(f"\n{'─'*40}")
    print("Check 5: ID continuity — stone IDs do not reappear after being absent")
    errs = _check_id_continuity(df)
    warns = [e for e in errs if "WARN" in e]
    if warns:
        all_warnings.extend(warns)
        for w in warns:
            print(f"  {w}")
    else:
        print("  PASS")

    # ---- Check 6: Displacement plausibility -------------------------------
    print(f"\n{'─'*40}")
    print("Check 6: Matched-stone displacement (prev → current distance)")
    disp_errors, detail_df = _check_matched_displacement(df)
    warns = [e for e in disp_errors if "WARN" in e]
    if warns:
        all_warnings.extend(warns)
        for w in warns:
            print(f"  {w}")
    else:
        print("  PASS — no suspicious displacements above warning threshold.")
    print(f"  Displacement stats:{_displacement_stats(detail_df)}")

    print(f"\n  Displacement distribution (matched stones, shot > 1):")
    if not detail_df.empty:
        bins = [0, 0.01, 0.02, 0.03, 0.05, 0.10, float("inf")]
        labels = ["0–0.01", "0.01–0.02", "0.02–0.03", "0.03–0.05",
                  "0.05–0.10", ">0.10 (SUSPICIOUS)"]
        hist = pd.cut(detail_df["displacement"], bins=bins, labels=labels,
                      right=False).value_counts().sort_index()
        max_count = hist.max()
        for label, count in hist.items():
            bar = "█" * int(30 * count / max_count) if max_count > 0 else ""
            print(f"    {label:25s} {count:8,}  {bar}")

    # ---- Check 7: New-stone count -----------------------------------------
    print(f"\n{'─'*40}")
    print("Check 7: New stone count per shot (NULL prev with active stone ≤ 1 per team)")
    errs = _check_new_stone_count(df)
    warns = [e for e in errs if "WARN" in e]
    if warns:
        all_warnings.extend(warns)
        for w in warns:
            print(f"  {w}")
    else:
        print("  PASS")

    # ---- Summary ----------------------------------------------------------
    print(f"\n{'─'*40}")
    print(f"Coverage stats:{_coverage_stats(df)}")
    print(f"\n{'='*60}")
    print(f"SUMMARY:  {len(all_errors)} FAIL(s)   {len(all_warnings)} WARNING(s)")
    if all_errors:
        print("RESULT: FAIL — re-scrape or check tracking code.")
    elif all_warnings:
        print("RESULT: PASS WITH WARNINGS — review warnings above.")
    else:
        print("RESULT: PASS — tracking appears consistent.")
    print(f"{'='*60}\n")

    # ---- Optional visual output -------------------------------------------
    if visualise and not all_errors:
        if event_id is None:
            vis_event = int(df["event_id"].iloc[0])
        else:
            vis_event = event_id
        if match_id is None:
            vis_match = int(df[df["event_id"] == vis_event]["match_id"].iloc[0])
        else:
            vis_match = match_id
        print(
            f"Generating visual tracking images for "
            f"event={vis_event} match={vis_match} end={vis_end} …"
        )
        _generate_tracking_images(df, output_dir, vis_event, vis_match, vis_end)

    return len(all_errors) == 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Verify sequential stone tracking in shot_locations.parquet."
    )
    parser.add_argument(
        "parquet",
        help="Path to shot_locations.parquet to verify.",
    )
    parser.add_argument(
        "--event", type=int, default=None,
        help="Filter to a specific event_id before running checks.",
    )
    parser.add_argument(
        "--match", type=int, default=None,
        help="Filter to a specific match_id (requires --event).",
    )
    parser.add_argument(
        "--visualise", action="store_true",
        help="Generate per-shot board images with displacement arrows.",
    )
    parser.add_argument(
        "--vis-end", type=int, default=1,
        help="End number to visualise (default: 1). Only used with --visualise.",
    )
    parser.add_argument(
        "--output-dir", default="verify_out",
        help="Directory for visual output images (default: verify_out).",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.parquet):
        print(f"ERROR: File not found: {args.parquet}")
        sys.exit(1)

    ok = run_checks(
        parquet_path=args.parquet,
        event_id=args.event,
        match_id=args.match,
        visualise=args.visualise,
        vis_end=args.vis_end,
        output_dir=args.output_dir,
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
