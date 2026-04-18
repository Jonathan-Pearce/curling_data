"""Feature enrichment step for the curling ML pipeline.

Reads the tracked shot-locations parquet (output of ``track_stones.py``) and
``ends.csv``, then joins game-context columns that are required by the GNN but
were previously only in ``ends.csv``.

Columns added
-------------
``shooting_team_is_team1`` : bool
    True when the row's ``team_code`` equals ``team1_code`` for that end.
``hammer_team_code`` : str
    Which team holds last-stone advantage this end.
``shooting_team_has_hammer`` : bool
    True when ``team_code == hammer_team_code``.
``team1_score_before`` : int
    Cumulative score for team1 entering this end.
``team2_score_before`` : int
    Cumulative score for team2 entering this end.
``score_diff_before`` : int
    ``team1_score_before - team2_score_before`` entering this end.

Usage::

    python build_features.py [--shots  <path>]
                              [--ends   <path>]
                              [--output <path>]

Outputs:
    Enriched ``shot_locations.parquet`` (overwrites ``--output``).

Importable helper::

    from build_features import enrich_features
"""

import argparse
import os

import pandas as pd


DEFAULT_SHOTS_PATH = os.path.join("output", "shot_locations.parquet")
DEFAULT_ENDS_PATH = os.path.join("output", "ends.csv")
DEFAULT_OUTPUT_PATH = os.path.join("output", "shot_locations.parquet")

_END_COLS = [
    "event_id",
    "match_id",
    "end_number",
    "team1_code",
    "team2_code",
    "hammer_team_code",
    "team1_score_before",
    "team2_score_before",
]


def enrich_features(shots_df, ends_df):
    """Join game-context columns from *ends_df* onto *shots_df*.

    Parameters
    ----------
    shots_df : pd.DataFrame
        Tracked shot locations.  Must contain ``event_id``, ``match_id``,
        ``end_number``, and ``team_code``.
    ends_df : pd.DataFrame
        End-level data.  Must contain all columns listed in ``_END_COLS``.

    Returns
    -------
    pd.DataFrame
        Copy of *shots_df* with the six new context columns appended.
        Rows that do not find a matching end (e.g. mixed-doubles rows with no
        ends entry) receive NaN for all joined columns.
    """
    ends_slim = ends_df[_END_COLS].copy()

    join_keys = ["event_id", "match_id", "end_number"]
    result = shots_df.merge(ends_slim, on=join_keys, how="left")

    result["shooting_team_is_team1"] = result["team_code"] == result["team1_code"]
    result["shooting_team_has_hammer"] = result["team_code"] == result["hammer_team_code"]
    result["score_diff_before"] = (
        result["team1_score_before"] - result["team2_score_before"]
    )

    # Drop the helper columns that are now redundant (team codes already in ends.csv;
    # callers can join ends.csv directly if they need team1_code/team2_code).
    result = result.drop(columns=["team1_code", "team2_code"])

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Enrich shot_locations.parquet with game-context columns from ends.csv."
    )
    parser.add_argument(
        "--shots",
        default=DEFAULT_SHOTS_PATH,
        help=f"Path to tracked shot_locations.parquet (default: {DEFAULT_SHOTS_PATH})",
    )
    parser.add_argument(
        "--ends",
        default=DEFAULT_ENDS_PATH,
        help=f"Path to ends.csv (default: {DEFAULT_ENDS_PATH})",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output parquet path (default: {DEFAULT_OUTPUT_PATH})",
    )
    args = parser.parse_args()

    for path in (args.shots, args.ends):
        if not os.path.isfile(path):
            parser.error(f"Input file not found: {path}")

    print(f"Reading shots from {args.shots} …")
    shots_df = pd.read_parquet(args.shots)
    print(f"  {len(shots_df):,} rows loaded.")

    print(f"Reading ends from {args.ends} …")
    ends_df = pd.read_csv(args.ends)
    print(f"  {len(ends_df):,} ends loaded.")

    print("Enriching with game-context columns …")
    enriched = enrich_features(shots_df, ends_df)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    enriched.to_parquet(args.output, index=False)
    print(f"Written {len(enriched):,} rows to {args.output}")
    new_cols = [
        "shooting_team_is_team1",
        "hammer_team_code",
        "shooting_team_has_hammer",
        "team1_score_before",
        "team2_score_before",
        "score_diff_before",
    ]
    for col in new_cols:
        null_count = enriched[col].isna().sum()
        print(f"  {col}: {null_count:,} NaN values")


if __name__ == "__main__":
    main()
