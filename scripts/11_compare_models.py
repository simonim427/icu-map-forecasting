"""Collect every model's metrics into one table (results/comparison.md) and two figures.

Reads results/metrics/<model>.json, written by the model scripts (steps 7-10).
Writes results/comparison.md, results/model_comparison.png and results/robustness.png.
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import CONDITIONS, RESULTS, THRESHOLD

# file name -> (label, line colour). Rows appear in this order.
MODELS = {
    "locf": ("LOCF (trivial baseline)", "#eda100"),
    "xgboost": ("XGBoost", "#1baf7a"),
    "gru_d": ("GRU-D", "#e87ba4"),
    "neural_ode_plain": ("Neural ODE (plain)", "#eb6834"),
    "neural_ode_stable": ("Neural ODE (stability-aware)", "#2a78d6"),
}
CONDITION_LABELS = {"clean": "Clean inputs", "drop_30": "30% of time points removed",
                    "drop_50": "50% of time points removed", "noise": "Noisy readings (0.3 SD)"}
SURFACE, INK, INK_SOFT, MUTED, GRID, AXIS, BLUE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#2a78d6"


def load_models() -> dict:
    models = {}
    for name, (label, _) in MODELS.items():
        path = RESULTS / "metrics" / f"{name}.json"
        if path.exists():
            models[name] = {"label": label, **json.loads(path.read_text())}
        else:
            print(f"Note: no results for {name} yet ({path.name} missing).")
    return models


def markdown_table(header: list, rows: list) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def write_tables(models: dict) -> None:
    alert_keys = ["precision", "recall", "f1", "onset_precision", "onset_recall", "onset_f1",
                  "recovery_precision", "recovery_recall", "recovery_f1"]
    alert_columns = ["Precision", "Recall", "F1", "Onset precision", "Onset recall", "Onset F1",
                     "Recovery precision", "Recovery recall", "Recovery F1"]

    def row(first_cells: list, result: dict) -> list:
        return first_cells + [f"{result[k]:.3f}" for k in alert_keys]

    test_rows, tuned_rows = [], []
    for m in models.values():
        r = m["test"]
        test_rows.append(row([m["label"], f"{r['n_pairs']:,}", f"{r['mae']:.2f}", f"{r['rmse']:.2f}"], r))
        tuned_rows.append(row([m["label"], f"forecast < {m['tuned_cutoff']:g} mmHg"], m["test_tuned"]))
        if "aux_classifier" in m:  # GRU-D also has a classifier head that raises alerts
            tuned_rows.append(row([m["label"], m["aux_classifier"]["rule"]], m["aux_classifier"]["test"]))

    robust = {n: m for n, m in models.items() if "robustness" in m}
    robust_rows = [[m["label"]] + [f"{m['robustness'][c]['mae']:.2f}" for c in CONDITIONS]
                   + [f"+{m['robustness']['drop_50']['mae'] - m['robustness']['clean']['mae']:.2f}"] for m in robust.values()]

    text = "\n\n".join([
        "# Model comparison on the held-out test set",
        "One-hour-ahead NIMAP forecast. Regenerate with `uv run python run_all.py`.",
        f"## Forecast error, and alerts raised when the forecast is below {THRESHOLD:g} mmHg",
        markdown_table(["Model", "Pairs", "MAE (mmHg)", "RMSE (mmHg)"] + alert_columns, test_rows),
        "## Alerts with the rule tuned on the validation set",
        markdown_table(["Model", "Alert rule"] + alert_columns, tuned_rows),
        "## Robustness: test MAE (mmHg) when the inputs are degraded",
        markdown_table(["Model"] + [CONDITION_LABELS[c] for c in CONDITIONS] + ["Increase at 50% removed"], robust_rows),
        "Onset = the patient is at or above 65 mmHg now and the model must warn of the drop. "
        "Recovery = the patient is below 65 mmHg now and the model must predict the return above it.",
    ]) + "\n"
    (RESULTS / "comparison.md").write_text(text)
    print(text)


def style(ax, title: str) -> None:
    ax.set_title(title, loc="left", fontsize=10, color=INK, pad=10)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.set_axisbelow(True)


def bar_panel(ax, labels: list, values: list, title: str, fmt: str, show_names: bool) -> None:
    style(ax, title)
    positions = range(len(labels))
    ax.barh(positions, values, height=0.4, color=BLUE)
    for y, v in zip(positions, values):
        ax.text(v, y, " " + fmt.format(v), va="center", ha="left", fontsize=9, color=INK)
    ax.set_yticks(positions, labels if show_names else [""] * len(labels))
    ax.tick_params(axis="y", labelcolor=INK_SOFT)
    ax.invert_yaxis()
    ax.set_xlim(0, max(values) * 1.22)
    ax.grid(axis="x", color=GRID, linewidth=0.8)


def figure_comparison(models: dict) -> None:
    labels = [m["label"] for m in models.values()]
    fig, axes = plt.subplots(1, 3, figsize=(12, 0.55 * len(labels) + 1.4), facecolor=SURFACE)
    bar_panel(axes[0], labels, [m["test"]["mae"] for m in models.values()],
              "Forecast error, MAE in mmHg (lower is better)", "{:.2f}", True)
    bar_panel(axes[1], labels, [m["test_tuned"]["f1"] for m in models.values()],
              "Hypotension alert F1 (higher is better)", "{:.2f}", False)
    bar_panel(axes[2], labels, [m["test_tuned"]["onset_f1"] for m in models.values()],
              "Onset F1: warning before a drop (higher is better)", "{:.2f}", False)
    fig.text(0.01, 0.01, "Held-out test patients, one hour ahead. Alert rule tuned on the validation set. "
             "Onset = patient at or above 65 mmHg now, below 65 mmHg an hour later.", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(RESULTS / "model_comparison.png", dpi=200)
    plt.close(fig)


def spread(values: list, gap: float) -> list:
    """Move label heights apart so that no two are closer than `gap`."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    placed = list(values)
    for prev, cur in zip(order, order[1:]):
        placed[cur] = max(placed[cur], placed[prev] + gap)
    return placed


def figure_robustness(models: dict) -> None:
    robust = {n: m for n, m in models.items() if "robustness" in m}
    if not robust:
        return
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.2), facecolor=SURFACE, gridspec_kw={"width_ratios": [1.25, 1]})

    # Left: error as more and more of the patient's time points are removed.
    style(left, "Sparser sampling: test MAE in mmHg as time points are removed")
    shares = [0, 30, 50]
    ends = [m["robustness"]["drop_50"]["mae"] for m in robust.values()]
    low = min(m["robustness"]["clean"]["mae"] for m in robust.values())
    label_heights = spread(ends, gap=(max(ends) - low) * 0.075)
    for (name, m), end, height in zip(robust.items(), ends, label_heights):
        maes = [m["robustness"][c]["mae"] for c in ("clean", "drop_30", "drop_50")]
        left.plot(shares, maes, color=MODELS[name][1], linewidth=2, marker="o", markersize=7,
                  markeredgecolor=SURFACE, markeredgewidth=2, solid_capstyle="round", label=m["label"])
        left.plot([50.8, 52.6], [end, height], color=AXIS, linewidth=0.8)
        left.text(53, height, f"{m['label']}  {end:.2f}", va="center", fontsize=9, color=INK_SOFT)
    left.set_xticks(shares, ["none removed", "30% removed", "50% removed"])
    left.set_xlim(-3, 88)
    left.grid(axis="y", color=GRID, linewidth=0.8)
    left.legend(loc="lower right", fontsize=8.5, frameon=False, labelcolor=INK_SOFT)

    # Right: error with clean readings versus noisy readings, per model.
    style(right, "Measurement noise: test MAE in mmHg, clean vs noisy readings")
    for y, m in enumerate(robust.values()):
        clean, noisy = m["robustness"]["clean"]["mae"], m["robustness"]["noise"]["mae"]
        right.plot([clean, noisy], [y, y], color=AXIS, linewidth=2, zorder=1)
        right.scatter([clean], [y], s=70, color="#86b6ef", edgecolor=SURFACE, linewidth=2, zorder=2, label="Clean readings" if y == 0 else None)
        right.scatter([noisy], [y], s=70, color="#1c5cab", edgecolor=SURFACE, linewidth=2, zorder=2, label="Noisy readings" if y == 0 else None)
        right.text(noisy, y, f"   {noisy:.2f}  (+{noisy - clean:.2f})", va="center", fontsize=9, color=INK)
    right.set_yticks(range(len(robust)), [m["label"] for m in robust.values()])
    right.tick_params(axis="y", labelcolor=INK_SOFT)
    right.set_ylim(len(robust) - 0.4, -0.9)
    span = [v for m in robust.values() for v in (m["robustness"]["clean"]["mae"], m["robustness"]["noise"]["mae"])]
    right.set_xlim(min(span) - 0.3, max(span) + (max(span) - min(span)) * 0.55)
    right.grid(axis="x", color=GRID, linewidth=0.8)
    right.legend(loc="upper left", ncol=2, fontsize=8.5, frameon=False, labelcolor=INK_SOFT, handletextpad=0.2)

    fig.tight_layout()
    fig.savefig(RESULTS / "robustness.png", dpi=200)
    plt.close(fig)


def main():
    plt.rcParams["font.sans-serif"] = ["Helvetica Neue", "Arial", "DejaVu Sans"]
    models = load_models()
    write_tables(models)
    figure_comparison(models)
    figure_robustness(models)
    print(f"Saved {RESULTS / 'comparison.md'}, model_comparison.png and robustness.png")


if __name__ == "__main__":
    main()
