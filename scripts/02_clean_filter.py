"""Clean the long table: keep 7 vitals and remove values outside physiological bounds.

Prints the row count after every step, so each number in the data report can be reproduced.
Reads data/intermediate/raw_long.parquet. Writes data/intermediate/clean_long.parquet.
"""
from pathlib import Path
import pandas as pd


BASE = Path(__file__).parent.parent / "data"
INPUT_PARQUET = BASE / "intermediate" / "raw_long.parquet"
OUT_PARQUET = BASE / "intermediate" / "clean_long.parquet"

# Define the vital parameters we keep for this forecasting task
TARGET_PARAMS = {
    "NIMAP",
    "HR",
    "NISysABP",
    "NIDiasABP",
    "RespRate",
    "Temp",
    "Urine",
}

# Clinically reasonable physiological bounds for each vital sign
PHYSIO_BOUNDS = {
    "NIMAP": (20, 160),
    "HR": (30, 220),
    "NISysABP": (40, 240),
    "NIDiasABP": (20, 140),
    "RespRate": (4, 60),
    "Temp": (32, 42),
    "Urine": (0, 2000),
}


def main():
    print("Loading raw long table...")
    df = pd.read_parquet(INPUT_PARQUET)
    total_rows_start = len(df)
    total_patients_start = df["RecordID"].nunique()
    print(f"Starting: {total_rows_start:,} rows, {total_patients_start} patients")

    # Step 1: Keep only the selected vital parameters
    df = df[df["Parameter"].isin(TARGET_PARAMS)].copy()
    rows_after_param_filter = len(df)

    # Step 2: Convert raw Value column to numeric; unparseable entries become NaN
    df["Value_num"] = pd.to_numeric(df["Value"], errors="coerce")

    # Step 3: Drop rows where original dataset uses -1 as an explicit missing marker
    df = df[~((df["Value_num"] == -1.0) & df["Value_num"].notna())].copy()
    rows_after_drop_neg1 = len(df)

    # Step 4: Restrict data to the first 48 hours after ICU admission
    df = df[(df["time_hours"] >= 0) & (df["time_hours"] <= 48)].copy()
    rows_after_time = len(df)

    # Step 5: Flag values outside physiological bounds and set them to NA
    def out_of_bounds(row):
        param = row["Parameter"]
        val = row["Value_num"]
        lo, hi = PHYSIO_BOUNDS[param]
        if pd.isna(val):
            return False
        return not (lo <= val <= hi)

    df["is_outlier"] = df.apply(out_of_bounds, axis=1)
    df.loc[df["is_outlier"], "Value_num"] = pd.NA
    rows_after_outlier = len(df)

    # Step 6: Remove rows that still have missing numeric vital values
    df = df[df["Value_num"].notna()].copy()
    rows_final = len(df)
    patients_final = df["RecordID"].nunique()

    # Print reproducible cleaning statistics for status check and report
    print("\n===== Cleaning Step Summary =====")
    print(f"Initial raw rows                 : {total_rows_start:,}")
    print(f"After keep only target vitals    : {rows_after_param_filter:,}")
    print(f"After drop native -1 missing      : {rows_after_drop_neg1:,}")
    print(f"After 0–48h time window filter    : {rows_after_time:,}")
    print(f"After mask physiologic outliers   : {rows_after_outlier:,}")
    print(f"After remove final NaN values     : {rows_final:,}")
    print(f"\nFinal patients remaining         : {patients_final}")
    print("==================================")

    # Prepare output table and save
    df_out = df[["RecordID", "time_hours", "Parameter", "Value_num"]].rename(
        columns={"Value_num": "Value"}
    )
    df_out.to_parquet(OUT_PARQUET, index=False)
    print(f"\nSaved cleaned long table to {OUT_PARQUET}")


if __name__ == "__main__":
    main()
