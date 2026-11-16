"""XGBoost baseline: gradient-boosted trees on hand-made features of each patient's history.

Irregular sampling is handled through the features: for every vital the model sees the
most recent reading and how old that reading is. The model predicts the change from the
most recent NIMAP reading.
Writes results/metrics/xgboost.json and results/predictions/xgboost.parquet.
"""
import pandas as pd
from xgboost import XGBRegressor

from common import KEYS, SEED, VITALS, evaluate_model, load_measured, load_pairs


def build_features(measured: pd.DataFrame) -> pd.DataFrame:
    """One row per time point, using only readings at or before that time."""
    patient = measured["RecordID"]
    time = measured["time_hours"]
    feats = measured[KEYS].copy()
    feats["hour"] = time

    for v in VITALS:
        feats[f"{v}_last"] = measured.groupby("RecordID")[v].ffill()
        read_at = time.where(measured[v].notna()).groupby(patient).ffill()
        feats[f"{v}_age"] = time - read_at  # hours since that reading

    # Recent NIMAP history: computed over the NIMAP readings only, then carried forward.
    bp = measured.loc[measured["NIMAP"].notna(), ["RecordID", "time_hours", "NIMAP"]]
    values = bp.groupby("RecordID")["NIMAP"]
    history = pd.DataFrame({
        "NIMAP_prev": values.shift(1),
        "NIMAP_prev_gap": bp["time_hours"] - bp.groupby("RecordID")["time_hours"].shift(1),
        "NIMAP_mean3": values.rolling(3, min_periods=1).mean().reset_index(level=0, drop=True),
        "NIMAP_std3": values.rolling(3, min_periods=2).std().reset_index(level=0, drop=True),
        "NIMAP_mean_so_far": values.expanding().mean().reset_index(level=0, drop=True),
        "NIMAP_min_so_far": values.cummin(),
    })
    feats = feats.join(history)
    feats[history.columns] = feats.groupby("RecordID")[list(history.columns)].ffill()
    feats["NIMAP_change"] = feats["NIMAP_last"] - feats["NIMAP_prev"]
    return feats


def main():
    measured, pairs = load_measured(), load_pairs()
    fallback = measured.loc[measured["split"] == "train", "NIMAP"].mean()

    def features_for(measured, pairs):
        X = pairs[KEYS].merge(build_features(measured), on=KEYS, how="left").drop(columns=KEYS)
        base = X["NIMAP_last"].fillna(fallback).to_numpy()  # forecast = base + predicted change
        return X, base

    train, val = pairs[pairs["split"] == "train"], pairs[pairs["split"] == "val"]
    X_train, base_train = features_for(measured[measured["split"] == "train"], train)
    X_val, base_val = features_for(measured[measured["split"] == "val"], val)

    model = XGBRegressor(
        n_estimators=1000, learning_rate=0.03, max_depth=5, min_child_weight=10,
        subsample=0.8, colsample_bytree=0.8, early_stopping_rounds=50, random_state=SEED,
    )
    model.fit(X_train, train["target"] - base_train, eval_set=[(X_val, val["target"] - base_val)], verbose=False)
    print(f"XGBoost: {len(X_train):,} training pairs, {X_train.shape[1]} features, "
          f"stopped at {model.best_iteration + 1} trees")

    def predict(measured, pairs):
        X, base = features_for(measured, pairs)
        return base + model.predict(X)

    evaluate_model("xgboost", predict, measured, pairs)

    importance = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    print("\nTop features (gain share):")
    print(importance.head(8).round(3).to_string())


if __name__ == "__main__":
    main()
