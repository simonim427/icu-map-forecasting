"""Read the raw PhysioNet patient files (set-a and set-b) into one long table.

Reads data/raw/set-a and data/raw/set-b, one .txt file per patient.
Writes data/intermediate/raw_long.parquet (RecordID, time_hours, Parameter, Value).
"""
from pathlib import Path
import pandas as pd

BASE = Path(__file__).parent.parent / "data"
RAW_DIR = BASE / "raw"
OUT_PARQUET = BASE / "intermediate" / "raw_long.parquet"
OUT_PARQUET.parent.mkdir(exist_ok=True)


def parse_hhmm_to_hours(time_str: str) -> float:
    hh, mm = map(int, time_str.strip().split(":"))
    return hh + mm / 60.0


def main():
    all_rows = []
    file_paths = []
    # only set-a, set-b
    for subfolder in ["set-a", "set-b"]:
        folder = RAW_DIR / subfolder
        txt_files = sorted(folder.glob("*.txt"))  # sorted: same row order on every filesystem
        file_paths.extend(txt_files)

    n_files_loaded = len(file_paths)

    for fp in file_paths:
        record_id = int(fp.stem)
        df_patient = pd.read_csv(fp)
        df_patient["time_hours"] = df_patient["Time"].apply(parse_hhmm_to_hours)
        df_patient["RecordID"] = record_id
        df_patient = df_patient[["RecordID", "time_hours", "Parameter", "Value"]]
        all_rows.append(df_patient)

    df_raw = pd.concat(all_rows, ignore_index=True)
    n_raw_rows = len(df_raw)

    print("===== Ingestion Summary =====")
    print(f"n_files_loaded = {n_files_loaded}")
    print(f"n_raw_rows     = {n_raw_rows}")
    print("============================")

    df_raw.to_parquet(OUT_PARQUET, index=False)
    print(f"Saved raw long table to {OUT_PARQUET}")


if __name__ == "__main__":
    main()
