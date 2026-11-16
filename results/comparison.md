# Model comparison on the held-out test set

One-hour-ahead NIMAP forecast. Regenerate with `uv run python run_all.py`.

## Forecast error, and alerts raised when the forecast is below 65 mmHg

| Model | Pairs | MAE (mmHg) | RMSE (mmHg) | Precision | Recall | F1 | Onset precision | Onset recall | Onset F1 | Recovery precision | Recovery recall | Recovery F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LOCF (trivial baseline) | 18,113 | 7.63 | 10.79 | 0.581 | 0.591 | 0.586 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| XGBoost | 18,113 | 6.63 | 9.09 | 0.689 | 0.443 | 0.539 | 0.537 | 0.050 | 0.091 | 0.592 | 0.571 | 0.581 |
| GRU-D | 18,113 | 6.62 | 9.15 | 0.668 | 0.461 | 0.546 | 0.494 | 0.115 | 0.186 | 0.582 | 0.576 | 0.579 |
| Neural ODE (plain) | 18,113 | 6.61 | 9.06 | 0.673 | 0.453 | 0.542 | 0.484 | 0.092 | 0.154 | 0.585 | 0.578 | 0.581 |
| Neural ODE (stability-aware) | 18,113 | 6.66 | 9.12 | 0.672 | 0.441 | 0.533 | 0.460 | 0.074 | 0.127 | 0.579 | 0.578 | 0.578 |

## Alerts with the rule tuned on the validation set

| Model | Alert rule | Precision | Recall | F1 | Onset precision | Onset recall | Onset F1 | Recovery precision | Recovery recall | Recovery F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| LOCF (trivial baseline) | forecast < 66.25 mmHg | 0.538 | 0.653 | 0.590 | 0.318 | 0.152 | 0.206 | 0.000 | 0.000 | 0.000 |
| XGBoost | forecast < 68 mmHg | 0.576 | 0.649 | 0.611 | 0.397 | 0.250 | 0.307 | 0.705 | 0.245 | 0.363 |
| GRU-D | forecast < 67.5 mmHg | 0.548 | 0.674 | 0.605 | 0.373 | 0.357 | 0.365 | 0.649 | 0.269 | 0.380 |
| GRU-D | classifier head, p >= 0.62 | 0.460 | 0.692 | 0.553 | 0.291 | 0.591 | 0.390 | 0.590 | 0.475 | 0.527 |
| Neural ODE (plain) | forecast < 68 mmHg | 0.570 | 0.664 | 0.613 | 0.400 | 0.305 | 0.346 | 0.686 | 0.263 | 0.381 |
| Neural ODE (stability-aware) | forecast < 68.25 mmHg | 0.558 | 0.668 | 0.608 | 0.383 | 0.306 | 0.340 | 0.672 | 0.228 | 0.341 |

## Robustness: test MAE (mmHg) when the inputs are degraded

| Model | Clean inputs | 30% of time points removed | 50% of time points removed | Noisy readings (0.3 SD) | Increase at 50% removed |
|---|---|---|---|---|---|
| LOCF (trivial baseline) | 7.63 | 7.98 | 8.38 | 8.67 | +0.76 |
| XGBoost | 6.63 | 7.20 | 7.94 | 7.04 | +1.31 |
| GRU-D | 6.62 | 7.07 | 7.54 | 7.02 | +0.92 |
| Neural ODE (plain) | 6.61 | 7.37 | 8.99 | 6.81 | +2.37 |
| Neural ODE (stability-aware) | 6.66 | 7.01 | 7.42 | 6.86 | +0.77 |

Onset = the patient is at or above 65 mmHg now and the model must warn of the drop. Recovery = the patient is below 65 mmHg now and the model must predict the return above it.
