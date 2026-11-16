"""Trivial baseline: last observation carried forward (LOCF).

The forecast for one hour ahead is simply the most recent NIMAP reading, and the
hypotension alert fires when that reading is below 65 mmHg.
Writes results/metrics/locf.json and results/predictions/locf.parquet.
"""
from common import KEYS, evaluate_model, load_measured, load_pairs


def main():
    measured, pairs = load_measured(), load_pairs()
    # Only needed in the robustness test, when a patient's earlier readings were all removed.
    fallback = measured.loc[measured["split"] == "train", "NIMAP"].mean()

    def predict(measured, pairs):
        last = measured[KEYS].assign(last=measured.groupby("RecordID")["NIMAP"].ffill())
        return pairs[KEYS].merge(last, on=KEYS, how="left")["last"].fillna(fallback).to_numpy()

    evaluate_model("locf", predict, measured, pairs)


if __name__ == "__main__":
    main()
