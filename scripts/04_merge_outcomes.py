"""Add the patient-level outcome labels (mortality, length of stay, severity scores) to the wide table.

Reads data/clean/wide_timeseries.parquet and data/raw/outcomes/Outcomes-a.txt, Outcomes-b.txt.
Writes data/clean/wide_with_outcomes.parquet.
"""
from pathlib import Path
import pandas as pd


BASE = Path(__file__).parent.parent / "data"
WIDE_INPUT = BASE / "clean" / "wide_timeseries.parquet"
OUTCOME_A = BASE / "raw" / "outcomes" / "Outcomes-a.txt"
OUTCOME_B = BASE / "raw" / "outcomes" / "Outcomes-b.txt"
OUT_MERGED = BASE / "clean" / "wide_with_outcomes.parquet"



def load_outcome_file(filepath: Path) -> pd.DataFrame:
    # Read PhysioNet 2012 outcome file, use RecordID as key
   
    df = pd.read_csv(filepath)
    df["RecordID"] = df["RecordID"].astype(int)
    return df


def main():
    
    print("Loading wide time-series dataset...")
    df_wide = pd.read_parquet(WIDE_INPUT)
    print(f"Wide dataset: {len(df_wide):,} rows")

    print("Loading outcome labels from set-a and set-b")
    df_oa = load_outcome_file(OUTCOME_A)
    df_ob = load_outcome_file(OUTCOME_B)
    df_outcome = pd.concat([df_oa, df_ob], ignore_index=True)
    print(f"Outcome table: {len(df_outcome)} unique patients")

    # Merge patient static labels onto time series
    df_merged = pd.merge(
        left=df_wide,
        right=df_outcome,
        on="RecordID",
        how="left"
    )

    print("\n===== Merge Outcome Summary =====")
    print(f"Final merged dataset rows: {len(df_merged):,}")
    print(f"Patients with matched outcome: {df_merged['RecordID'].nunique()}")
    print("=================================")

    df_merged.to_parquet(OUT_MERGED, index=False)
    print(f"Saved merged dataset to {OUT_MERGED}")


if __name__ == "__main__":
    main()
