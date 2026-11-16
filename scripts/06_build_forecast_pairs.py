"""Define the forecasting task once, so every model is trained and scored on the same pairs.

A pair is a measured NIMAP reading at time t ("current") and the patient's next measured
NIMAP reading ("target"), kept only when that next reading comes 45-75 minutes later.
Reads data/clean/wide_final_split.parquet. Writes data/clean/forecast_pairs.parquet.
"""
from common import CLEAN, THRESHOLD, load_measured

MIN_GAP_MINUTES, MAX_GAP_MINUTES = 45, 75


def main():
    measured = load_measured()
    bp = measured.loc[measured["NIMAP"].notna(), ["RecordID", "time_hours", "split", "NIMAP"]]
    bp = bp.rename(columns={"NIMAP": "current"})

    by_patient = bp.groupby("RecordID")
    bp["target_hour"] = by_patient["time_hours"].shift(-1)
    bp["target"] = by_patient["current"].shift(-1)
    # Compare the gap in whole minutes: 45 min stored as hours is not exactly 0.75 in floating point.
    bp["gap_minutes"] = ((bp["target_hour"] - bp["time_hours"]) * 60).round()

    pairs = bp[bp["gap_minutes"].between(MIN_GAP_MINUTES, MAX_GAP_MINUTES)].reset_index(drop=True)
    pairs.to_parquet(CLEAN / "forecast_pairs.parquet", index=False)

    print("===== Forecast Pair Summary =====")
    print(f"Measured NIMAP readings          : {len(bp):,}")
    print(f"Pairs with next reading in {MIN_GAP_MINUTES}-{MAX_GAP_MINUTES} min: {len(pairs):,}")
    print(f"Patients represented             : {pairs['RecordID'].nunique():,}")
    for split in ["train", "val", "test"]:
        part = pairs[pairs["split"] == split]
        print(f"  {split:<5}: {len(part):>7,} pairs, {part['RecordID'].nunique():>5,} patients, "
              f"{(part['target'] < THRESHOLD).mean():.1%} hypotensive targets")
    print("=================================")
    print(f"Saved {CLEAN / 'forecast_pairs.parquet'}")


if __name__ == "__main__":
    main()
