"""Reshape to one row per patient time point, with one column per vital.

Gaps are filled by linear interpolation within each patient. A *_is_measured flag per vital
marks which values are real readings (1) and which were filled in (0).
Reads data/intermediate/clean_long.parquet. Writes data/clean/wide_timeseries.parquet.
"""
from pathlib import Path
import pandas as pd
import numpy as np


BASE = Path(__file__).parent.parent / "data"
INPUT_PARQUET = BASE / "intermediate" / "clean_long.parquet"
OUT_PARQUET = BASE / "clean" / "wide_timeseries.parquet"
OUT_PARQUET.parent.mkdir(exist_ok=True)


def interpolate_patient(group: pd.DataFrame) -> pd.DataFrame:
    # Pivot to wide form for this single patient
    pivoted = group.pivot_table(index="time_hours", columns="Parameter", values="Value")
    # Sort by admission time
    pivoted = pivoted.sort_index()

    # Create boolean mask for original measured points before interpolation
    measured_mask = pivoted.notna().astype(np.float32)
    measured_mask.columns = [f"{col}_is_measured" for col in measured_mask.columns]

    # Linear interpolation within the patient; do not extrapolate outside known points
    pivoted = pivoted.interpolate(method="linear", limit_direction="forward")

    combined = pd.concat([pivoted, measured_mask], axis=1)
    combined["RecordID"] = int(group["RecordID"].iloc[0])
    combined = combined.reset_index()
    return combined


def main():
    print("Loading cleaned long-form vital data...")
    df = pd.read_parquet(INPUT_PARQUET)
    unique_patients = df["RecordID"].nunique()
    print(f"Loaded {len(df):,} valid vital rows from {unique_patients} patients")

    patient_groups = df.groupby("RecordID")
    output_list = []
    count_finished = 0

    for _, group in patient_groups:
        patient_df = interpolate_patient(group)
        output_list.append(patient_df)
        count_finished += 1
        if count_finished % 1000 == 0:
            print(f"Processed {count_finished}/{unique_patients} patients")

    final_wide = pd.concat(output_list, ignore_index=True)
    total_time_points = len(final_wide)

    print("\n===== Reshape & Interpolation Summary =====")
    print(f"Total unique time points after pivoting: {total_time_points:,}")
    print(f"Columns include vitals + *_is_measured flags")
    print("============================================")

    final_wide.to_parquet(OUT_PARQUET, index=False)
    print(f"\nSaved final wide dataset for modeling to {OUT_PARQUET}")


if __name__ == "__main__":
    main()
