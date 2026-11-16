"""Shared by every model script: data loading, the robustness perturbations, and the metrics.

Every model is scored by evaluate_model() on the same forecast pairs with the same
metrics, so the numbers in results/ are directly comparable.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
CLEAN = ROOT / "data" / "clean"
RESULTS = ROOT / "results"

VITALS = ["NIMAP", "HR", "NISysABP", "NIDiasABP", "RespRate", "Temp", "Urine"]
THRESHOLD = 65.0  # mmHg; NIMAP below this counts as hypotension
HORIZON = 1.0  # hours ahead that every model forecasts
SEED = 42
KEYS = ["RecordID", "time_hours"]

# Robustness test: how the inputs are degraded before the model sees them.
#   drop  = share of time points whose readings are removed (sparser sampling)
#   noise = Gaussian noise on every reading, as a fraction of that vital's standard deviation
CONDITIONS = {
    "clean": {"drop": 0.0, "noise": 0.0},
    "drop_30": {"drop": 0.3, "noise": 0.0},
    "drop_50": {"drop": 0.5, "noise": 0.0},
    "noise": {"drop": 0.0, "noise": 0.3},
}


def load_measured() -> pd.DataFrame:
    """One row per patient time point, keeping only real sensor readings.

    The data pipeline fills gaps by linear interpolation, which uses the *next* reading.
    Feeding those values to a forecaster would leak the future, so they are set to NaN here.
    """
    flags = [f"{v}_is_measured" for v in VITALS]
    df = pd.read_parquet(CLEAN / "wide_final_split.parquet", columns=KEYS + ["split"] + VITALS + flags)
    for v in VITALS:
        df[v] = df[v].where(df[f"{v}_is_measured"] == 1)
    return df[KEYS + ["split"] + VITALS].sort_values(KEYS).reset_index(drop=True)


def load_pairs() -> pd.DataFrame:
    """Forecast pairs written by 06_build_forecast_pairs.py."""
    return pd.read_parquet(CLEAN / "forecast_pairs.parquet")


def perturb(measured: pd.DataFrame, drop: float = 0.0, noise: float = 0.0, seed: int = SEED) -> pd.DataFrame:
    """Degrade the inputs for the robustness test. The targets are never touched."""
    rng = np.random.default_rng(seed)
    out = measured.copy()
    if drop > 0:
        out.loc[rng.random(len(out)) < drop, VITALS] = np.nan
    if noise > 0:
        for v in VITALS:
            out[v] = out[v] + rng.normal(0.0, noise * measured[v].std(), len(out))
    return out


def _precision_recall_f1(actual: np.ndarray, predicted: np.ndarray, prefix: str = "") -> dict:
    tp = int((actual & predicted).sum())
    precision = tp / predicted.sum() if predicted.sum() else 0.0
    recall = tp / actual.sum() if actual.sum() else 0.0
    f1 = 2 * precision * recall / (precision + recall) if tp else 0.0
    return {f"{prefix}precision": float(precision), f"{prefix}recall": float(recall), f"{prefix}f1": float(f1)}


def evaluate(target, pred, current, cutoff: float = THRESHOLD) -> dict:
    """Forecast error plus hypotension-alert quality. Same keys as Aruti's GRU-D output.

    target  = the NIMAP that was actually measured ~1 h later
    pred    = the model's forecast of it
    current = the NIMAP at the time the forecast was made
    An alert is raised when pred < cutoff.
      onset    = patient is fine now (current >= 65): does the model warn before the drop?
      recovery = patient is hypotensive now (current < 65): does the model see the recovery coming?
    """
    target, pred, current = (np.asarray(a, dtype=float) for a in (target, pred, current))
    alert = pred < cutoff
    hypotensive = target < THRESHOLD
    onset = current >= THRESHOLD
    out = {
        "n_pairs": int(len(target)),
        "mae": float(np.abs(pred - target).mean()),
        "rmse": float(np.sqrt(((pred - target) ** 2).mean())),
    }
    out |= _precision_recall_f1(hypotensive, alert)
    out["onset_n"] = int(onset.sum())
    out |= _precision_recall_f1(hypotensive[onset], alert[onset], "onset_")
    out["recovery_n"] = int((~onset).sum())
    out |= _precision_recall_f1(~hypotensive[~onset], ~alert[~onset], "recovery_")
    return out


def tune_cutoff(target, pred) -> float:
    """Alert cutoff (mmHg) that gives the best F1. Call this on validation data only."""
    cutoffs = np.arange(60.0, 75.01, 0.25)
    scores = [evaluate(target, pred, target, c)["f1"] for c in cutoffs]
    return float(cutoffs[int(np.argmax(scores))])


def evaluate_model(name: str, predict, measured: pd.DataFrame, pairs: pd.DataFrame, extra: dict | None = None) -> dict:
    """Score one model and save its results.

    predict(measured, pairs) must return one forecast per row of `pairs`, using only the
    readings in `measured` at or before each pair's time_hours.
    extra: optional model-specific entries to store in the metrics file as well.

    Writes results/metrics/<name>.json and results/predictions/<name>.parquet.
    """
    val, test = pairs[pairs["split"] == "val"], pairs[pairs["split"] == "test"]
    measured_val, measured_test = measured[measured["split"] == "val"], measured[measured["split"] == "test"]

    cutoff = tune_cutoff(val["target"], predict(measured_val, val))
    metrics = {"model": name, "tuned_cutoff": cutoff, "robustness": {}}
    for condition, settings in CONDITIONS.items():
        pred = predict(perturb(measured_test, **settings), test)
        metrics["robustness"][condition] = evaluate(test["target"], pred, test["current"])
        if condition == "clean":
            metrics["test"] = metrics["robustness"]["clean"]  # alert when forecast < 65 mmHg
            metrics["test_tuned"] = evaluate(test["target"], pred, test["current"], cutoff)  # alert at tuned cutoff
            predictions = test[KEYS + ["target_hour", "current", "target"]].assign(pred=pred)
    metrics |= extra or {}

    (RESULTS / "metrics").mkdir(parents=True, exist_ok=True)
    (RESULTS / "predictions").mkdir(parents=True, exist_ok=True)
    (RESULTS / "metrics" / f"{name}.json").write_text(json.dumps(metrics, indent=2) + "\n")
    predictions.to_parquet(RESULTS / "predictions" / f"{name}.parquet", index=False)

    print(f"\n[{name} - held-out test set (alert when forecast < {THRESHOLD:g} mmHg)]")
    print(json.dumps(metrics["test"], indent=2))
    print(f"[{name} - alert cutoff tuned on validation: {cutoff:g} mmHg]")
    print(json.dumps({k: round(metrics["test_tuned"][k], 4) for k in ("precision", "recall", "f1", "onset_recall")}))
    print(f"[{name} - test MAE under degraded inputs]")
    print(json.dumps({c: round(m["mae"], 3) for c, m in metrics["robustness"].items()}))
    return metrics
