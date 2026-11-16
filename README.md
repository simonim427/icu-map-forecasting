# Forecasting mean arterial pressure and hypotension with stability-aware Neural ODEs

Can a stability-aware Neural ODE forecast a patient's non-invasive mean arterial pressure
(NIMAP) one hour ahead, and warn of hypotension (NIMAP < 65 mmHg), more reliably than simple
baselines when ICU measurements are sparse, irregular, or noisy?

Data: PhysioNet Challenge 2012, set-a and set-b (8,000 ICU stays).

## Setup

Requires [uv](https://docs.astral.sh/uv/). From this folder:

```
uv sync
```

This creates `.venv` with Python 3.12 and the exact package versions pinned in `uv.lock`.

macOS only: XGBoost needs the OpenMP runtime. If `08_xgboost.py` fails to load
`libomp.dylib`, run `brew install libomp`.

## Regenerate every number and figure

```
uv run python run_all.py
```

This runs the numbered scripts in `scripts/` in order, starting from `data/raw`. It takes
about 10 minutes on a laptop CPU; almost all of it is training the Neural ODE and GRU-D. No GPU
and no API key are needed. To run a single step: `uv run python scripts/08_xgboost.py`.

| Step | Script | What it does | Writes |
|---|---|---|---|
| 1 | `01_ingest_raw.py` | Reads the 8,000 patient files into one long table | `data/intermediate/raw_long.parquet` |
| 2 | `02_clean_filter.py` | Keeps 7 vitals, removes values outside physiological bounds | `data/intermediate/clean_long.parquet` |
| 3 | `03_reshape_interpolate.py` | One row per patient time point, with `*_is_measured` flags | `data/clean/wide_timeseries.parquet` |
| 4 | `04_merge_outcomes.py` | Adds the patient-level outcome labels | `data/clean/wide_with_outcomes.parquet` |
| 5 | `05_patient_split.py` | 70/15/15 split by patient, seed 42 | `data/clean/wide_final_split.parquet`, `patient_split.csv` |
| 6 | `06_build_forecast_pairs.py` | Defines the one-hour-ahead forecasting task | `data/clean/forecast_pairs.parquet` |
| 7 | `07_locf_baseline.py` | Trivial baseline: last observation carried forward | `results/` |
| 8 | `08_xgboost.py` | Gradient-boosted trees on history features | `results/` |
| 9 | `09_neural_ode.py` | Main model: latent Neural ODE, plain and stability-aware | `results/` |
| 10 | `10_gru_d.py` | GRU-D baseline, a standard model for irregular time series | `results/` |
| 11 | `11_compare_models.py` | Comparison tables and figures | `results/comparison.md`, `model_comparison.png`, `robustness.png` |

Every model script writes `results/metrics/<model>.json` (all metrics) and
`results/predictions/<model>.parquet` (one row per test pair: current value, forecast, truth).

## The task, and the rules every model follows

These live in `scripts/common.py` and `scripts/06_build_forecast_pairs.py`, so all models are
trained and scored the same way.

- **Forecast pair:** a measured NIMAP reading at time t and the patient's next measured NIMAP
  reading, kept when that next reading comes 45 to 75 minutes later.
- **Split:** by patient (train / validation / test), never by row. Models are fitted on train,
  tuned on validation, and reported on test.
- **Inputs:** only real sensor readings at or before time t. The interpolated values in the
  wide tables are never used as inputs, because interpolation looks at the next reading and
  would leak the future.
- **Metrics:** MAE and RMSE of the forecast; precision, recall and F1 of the hypotension alert,
  also split into onset (patient fine now, drop coming) and recovery (patient hypotensive now).
- **Robustness test:** each model is also scored on the test set after removing 30% or 50% of
  each patient's time points, and after adding noise to every reading. This is how
  "stability-aware" is evaluated.

## Adding a model

Save it as a numbered script in `scripts/`, add any packages it needs with
`uv add <package>`, and score it through the shared function so it uses the same pairs,
metrics and robustness test as the others:

```python
from common import evaluate_model, load_measured, load_pairs

measured, pairs = load_measured(), load_pairs()
# ... train on the "train" split ...
# predict(measured, pairs) returns one forecast per row of `pairs`
evaluate_model("my_model", predict, measured, pairs)
```

Then add its name and label to `MODELS` in `scripts/11_compare_models.py`.

`scripts/10_gru_d.py` was integrated this way: the model and training loop are Aruti's, while
data loading, forecast pairs and scoring come from `common.py`.

## Data

Source: PhysioNet Challenge 2012, "Predicting mortality of ICU patients", set-a and set-b,
under the Open Data Commons Attribution licence (ODC-BY). `data/raw/` holds the files as
downloaded: one text file per ICU stay, plus the outcome tables.

Every count below is printed by the script named next to it.

| Stage | Count | Script |
|---|---|---|
| Patient files read | 8,000 files, 3,520,515 rows | `01_ingest_raw.py` |
| Rows for the 7 vitals kept (NIMAP, HR, NISysABP, NIDiasABP, RespRate, Temp, Urine) | 1,603,529 | `02_clean_filter.py` |
| Rows left after removing values outside physiological bounds | 1,601,672, from 7,873 patients | `02_clean_filter.py` |
| Patient time points in the wide table | 466,841 | `03_reshape_interpolate.py` |
| Patients in train / validation / test | 5,511 / 1,180 / 1,182 | `05_patient_split.py` |
| Measured NIMAP readings | 194,635 | `06_build_forecast_pairs.py` |
| Forecast pairs in train / validation / test | 84,868 / 17,882 / 18,113 | `06_build_forecast_pairs.py` |

The table the models use is `data/clean/wide_final_split.parquet`, one row per patient time point:

- `RecordID`, `time_hours` (hours since ICU admission, 0 to 48), `split` (train / val / test)
- one column per vital, and a matching `<vital>_is_measured` flag: 1 = real sensor reading,
  0 = no reading at that time. Where the flag is 0 the value is filled in, by linear
  interpolation between readings or by carrying the last reading forward; before a patient's
  first reading it stays empty. Value and flag are both empty for a patient with no reading of
  that vital at all. The models never use filled values.
- the PhysioNet outcome columns (`SAPS-I`, `SOFA`, `Length_of_stay`, `Survival`,
  `In-hospital_death`), which the models do not use

## Layout

```
data/raw/            PhysioNet files as downloaded (set-a, set-b, outcomes)
data/intermediate/   long tables written by steps 1-2
data/clean/          tables the models are trained on (steps 3-6)
scripts/             numbered pipeline steps + common.py (shared loading, metrics, robustness test)
results/             metrics, predictions, comparison table and figures (regenerated by run_all.py)
```

## Reproducibility notes

- All random seeds are fixed (42). The data pipeline and XGBoost give identical output on every run.
- The Neural ODE runs on CPU with a fixed seed. On a different machine or operating system its
  numbers can differ in the last digits because of floating-point differences.
