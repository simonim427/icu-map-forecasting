"""Split the patients into train / validation / test (70 / 15 / 15, seed 42).

The split is by patient, so all time points of one ICU stay end up in the same set.
Reads data/clean/wide_with_outcomes.parquet.
Writes data/clean/patient_split.csv and data/clean/wide_final_split.parquet (adds a `split` column).
"""
from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path(__file__).parent.parent / "data"
INPUT_MERGED = BASE / "clean" / "wide_with_outcomes.parquet"
OUT_SPLIT_TABLE = BASE / "clean" / "patient_split.csv"
OUT_FINAL_DATA = BASE / "clean" / "wide_final_split.parquet"

# Fixed random seed for reproducibility

SEED = 42

# Split ratio: 70% train, 15% validation, 15% test
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def main():
    np.random.seed(SEED)
    print("Loading merged dataset with outcome labels...")
    df = pd.read_parquet(INPUT_MERGED)

    # Get unique patient list
    patient_ids = df["RecordID"].unique()
    np.random.shuffle(patient_ids)
    n_total = len(patient_ids)

    n_train = int(n_total * TRAIN_RATIO)
    n_val = int(n_total * VAL_RATIO)
    n_test = n_total - n_train - n_val

    train_patients = patient_ids[:n_train]
    val_patients = patient_ids[n_train: n_train + n_val]
    test_patients = patient_ids[n_train + n_val:]

    # Build patient-split lookup table

    split_df = pd.DataFrame({
        "RecordID": np.concatenate([train_patients, val_patients, test_patients]),
        "split": ["train"] * n_train + ["val"] * n_val + ["test"] * n_test
    })

    # Merge split group back onto full time-series table
    df_final = pd.merge(df, split_df, on="RecordID", how="left")

    print("\n===== Patient Split Summary =====")
    print(f"Total patients for split: {n_total}")
    print(f"Train patients: {n_train}")
    print(f"Validation patients: {n_val}")
    print(f"Test patients: {n_test}")
    print(f"Total time-series rows: {len(df_final):,}")
    print("==================================")

    split_df.to_csv(OUT_SPLIT_TABLE, index=False)
    df_final.to_parquet(OUT_FINAL_DATA, index=False)
    print(f"Saved patient split lookup: {OUT_SPLIT_TABLE}")
    print(f"Saved final dataset with split tag: {OUT_FINAL_DATA}")


if __name__ == "__main__":
    main()
