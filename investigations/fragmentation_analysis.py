"""
Investigate stone tracking quality in output_test/shot_locations.parquet.

Uses the evaluation helpers from track_stones.py.  The old fragmentation_index
metric (total_ids / peak_stones_in_play) has been replaced by spurious_id_rate
which avoids false positives in takeout-heavy ends; see docs/tracking_metrics.md.

Run from the repo root:  python investigations/fragmentation_analysis.py
"""
import pandas as pd

from track_stones import (
    evaluate_tracking,
    displacement_distribution,
    id_continuity_rate,
    slot_swap_rate,
    cap_pressure_rate,
    stone_count_consistency_rate,
)

df = pd.read_parquet("output_test/shot_locations.parquet")

print("=== Headline tracking metrics ===")
metrics = evaluate_tracking(df)
for k, v in metrics.items():
    print(f"  {k}: {v}")
print()

print("=== Displacement distribution ===")
disp = displacement_distribution(df)
for k, v in disp.items():
    if k != "displacements":
        print(f"  {k}: {v}")
print()

print("=== id_continuity_rate ===")
print(f"  {id_continuity_rate(df):.4f}  (ideal: 1.0, acceptable: ≥ 0.95)")
print()

print("=== slot_swap_rate ===")
print(f"  {slot_swap_rate(df):.4f}  (ideal: 0.0)")
print()

print("=== cap_pressure_rate ===")
print(f"  {cap_pressure_rate(df):.4f}  (warn if > 0.10)")
print()

print("=== stone_count_consistency_rate ===")
print(f"  {stone_count_consistency_rate(df):.4f}  (ideal: 1.0, warn if < 0.99)")

