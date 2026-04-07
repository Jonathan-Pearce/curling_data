"""
Investigate stone tracking fragmentation in output_test/shot_locations.parquet.
Run from the repo root:  python investigations/fragmentation_analysis.py
"""
import pandas as pd
import numpy as np

df = pd.read_parquet("output_test/shot_locations.parquet")

MAX_STONES = 8
group_keys = ["event_id", "match_id", "end_number"]

rows = []
for (eid, mid, enum), end_df in df.groupby(group_keys, sort=True):
    end_df = end_df.sort_values("shot_number")
    first_shot = end_df["shot_number"].min()
    for ti in (1, 2):
        id_vals = set()
        for si in range(1, MAX_STONES + 1):
            col = f"team{ti}_stone{si}_id"
            if col in df.columns:
                vals = end_df[col].dropna()
                id_vals.update(int(v) for v in vals)
        ids_assigned = len(id_vals)
        min_ids = int(end_df[f"team{ti}_stones_in_play"].max())

        mid_df = end_df[end_df["shot_number"] > first_shot]
        total_mid = 0
        new_mid = 0
        for si in range(1, MAX_STONES + 1):
            x_col = f"team{ti}_stone{si}_x"
            prev_col = f"team{ti}_stone{si}_prev_x"
            if x_col not in df.columns or prev_col not in df.columns:
                break
            occupied = mid_df[x_col].notna()
            total_mid += int(occupied.sum())
            new_mid += int((occupied & mid_df[prev_col].isna()).sum())

        frag = ids_assigned / min_ids if min_ids > 0 else np.nan
        new_rate = new_mid / total_mid if total_mid > 0 else np.nan
        rows.append({
            "event_id": eid, "match_id": mid, "end_number": enum,
            "team": ti, "ids_assigned": ids_assigned, "min_ids": min_ids,
            "frag": frag, "new_mid": new_mid, "total_mid": total_mid,
            "new_rate": new_rate, "n_shots": len(end_df)
        })

stats = pd.DataFrame(rows).dropna(subset=["frag"])

print("=== Fragmentation distribution (per end, per team) ===")
print(stats["frag"].describe().round(3))
print()

# Bucket breakdown
bins = [0, 1.05, 1.5, 2.0, 3.0, float("inf")]
labels = ["1.0 (perfect)", "1.01-1.5", "1.51-2.0", "2.01-3.0", ">3.0"]
stats["frag_bucket"] = pd.cut(stats["frag"], bins=bins, labels=labels)
print("=== Fragmentation bucket counts ===")
print(stats["frag_bucket"].value_counts().sort_index().to_string())
print()

print("=== Avg frag by end_number ===")
print(stats.groupby("end_number")["frag"].agg(["mean", "count"]).round(3).to_string())
print()

print("=== Avg frag by n_shots per end ===")
print(stats.groupby("n_shots")["frag"].agg(["mean", "count"]).round(3).to_string())
print()

print("=== Worst 20 ends (frag > 3, sorted desc) ===")
bad = stats[stats["frag"] > 3].sort_values("frag", ascending=False)
print(f"Count: {len(bad)} / {len(stats)}")
if len(bad) > 0:
    print(bad[["event_id", "match_id", "end_number", "team",
               "ids_assigned", "min_ids", "frag", "n_shots"]].head(20).to_string(index=False))
print()

# Look at a specific bad end in detail
if len(bad) > 0:
    worst = bad.iloc[0]
    print(f"=== Detail: event={worst.event_id} match={worst.match_id} end={worst.end_number} team={worst.team} ===")
    end_detail = df[
        (df["event_id"] == worst.event_id) &
        (df["match_id"] == worst.match_id) &
        (df["end_number"] == worst.end_number)
    ].sort_values("shot_number")
    ti = int(worst.team)
    cols = ["shot_number"] + [f"team{ti}_stone{si}_x" for si in range(1, 9)] + \
           [f"team{ti}_stone{si}_id" for si in range(1, 9)] + \
           [f"team{ti}_stone{si}_prev_x" for si in range(1, 9)]
    # Only show cols that exist
    cols = [c for c in cols if c in df.columns]
    # Trim to actually used stone slots
    max_si = int(end_detail[f"team{ti}_stones_in_play"].max())
    cols = ["shot_number"] + \
           [f"team{ti}_stone{si}_x" for si in range(1, max_si + 2)] + \
           [f"team{ti}_stone{si}_id" for si in range(1, max_si + 2)] + \
           [f"team{ti}_stone{si}_prev_x" for si in range(1, max_si + 2)]
    cols = [c for c in cols if c in df.columns]
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.width", 200)
    print(end_detail[cols].to_string(index=False))

# Displacement distribution for mid-end stones that got new IDs (potential false new placements)
print("\n=== Displacement magnitude distribution for matched stones ===")
disps = []
for ti in (1, 2):
    for si in range(1, MAX_STONES + 1):
        x_col = f"team{ti}_stone{si}_x"
        px_col = f"team{ti}_stone{si}_prev_x"
        y_col = f"team{ti}_stone{si}_y"
        py_col = f"team{ti}_stone{si}_prev_y"
        if not all(c in df.columns for c in [x_col, px_col, y_col, py_col]):
            break
        matched = df[df[px_col].notna() & df[x_col].notna()].copy()
        if len(matched) == 0:
            continue
        d = np.sqrt((matched[x_col] - matched[px_col])**2 + (matched[y_col] - matched[py_col])**2)
        disps.append(d)

if disps:
    all_disps = pd.concat(disps)
    print(all_disps.describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]).round(4))
    print(f"\nStones with displacement > 0.10 (near cap 0.13): {(all_disps > 0.10).sum()}")
    print(f"Stones with displacement > 0.12:                  {(all_disps > 0.12).sum()}")
    print(f"Total matched stones:                              {len(all_disps)}")
