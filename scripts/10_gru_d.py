"""GRU-D baseline (Che et al., 2018): a GRU whose inputs and hidden state decay toward a
default the longer a vital has gone unmeasured. Model and training by Aruti.

Like every other model script, it uses the shared forecast pairs and scoring from common.py.
Writes results/metrics/gru_d.json and results/predictions/gru_d.parquet.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from common import KEYS, THRESHOLD, VITALS, evaluate_model, load_measured, load_pairs

MASK_COLS: List[str] = [f"{v}_is_measured" for v in VITALS]
HYPO_THRESHOLD: float = THRESHOLD


def add_mask_columns(measured: pd.DataFrame) -> pd.DataFrame:
    """common.load_measured() marks missing readings as NaN; GRU-D wants explicit 0/1 mask columns."""
    masks = {m_col: measured[v].notna().astype(np.float32) for v, m_col in zip(VITALS, MASK_COLS)}
    return measured.assign(**masks)


@dataclass
class NormStats:
    mean: np.ndarray  # shape (D,)
    std: np.ndarray   # shape (D,)


def compute_train_norm_stats(df: pd.DataFrame) -> NormStats:
    """Compute per-vital mean and std strictly on train split & measured entries."""
    train_df = df[df["split"] == "train"]
    means = np.zeros(len(VITALS), dtype=np.float32)
    stds = np.ones(len(VITALS), dtype=np.float32)
    for i, (v, m_col) in enumerate(zip(VITALS, MASK_COLS)):
        obs = train_df.loc[train_df[m_col] == 1.0, v].dropna().to_numpy(dtype=np.float64)
        means[i] = float(np.mean(obs))
        s = float(np.std(obs))
        stds[i] = s if s > 1e-6 else 1.0
    return NormStats(mean=means, std=stds)


def build_patient_sequences(
    df: pd.DataFrame,
    stats: NormStats,
    pairs: pd.DataFrame,
) -> Dict[str, List[dict]]:
    """Per-patient GRU-D inputs.

    Which time points get a target comes from the shared pairs table
    (06_build_forecast_pairs.py), so GRU-D is trained and scored on exactly the same
    pairs as every other model.
    """
    df = df.sort_values(KEYS).reset_index(drop=True)
    df = df.merge(pairs[KEYS + ["target_hour", "target"]], on=KEYS, how="left")
    nimap_idx = VITALS.index("NIMAP")
    D = len(VITALS)

    splits: Dict[str, List[dict]] = {"train": [], "val": [], "test": []}

    record_ids = df["RecordID"].to_numpy()
    split_vals = df["split"].to_numpy()
    hours_all = df["time_hours"].to_numpy()
    pair_all = df["target"].notna().to_numpy()
    pair_target_all = df["target"].fillna(0.0).to_numpy(dtype=np.float32)
    pair_dt_all = (df["target_hour"] - df["time_hours"]).fillna(0.0).to_numpy(dtype=np.float32)
    times_all = df["time_hours"].to_numpy(dtype=np.float32)
    vals_all = df[VITALS].to_numpy(dtype=np.float32)
    masks_all = (df[MASK_COLS].to_numpy(dtype=np.float32) == 1.0).astype(np.float32)

    # Mask out interpolated/NaN entries and standardize observed entries
    vals_clean = np.where(masks_all == 1.0, (vals_all - stats.mean) / stats.std, 0.0).astype(np.float32)
    raw_nimap_all = vals_all[:, nimap_idx]

    change_idx = np.flatnonzero(record_ids[1:] != record_ids[:-1]) + 1
    starts = np.concatenate(([0], change_idx))
    ends = np.concatenate((change_idx, [len(record_ids)]))

    for s, e in zip(starts, ends):
        split_name = str(split_vals[s])
        if split_name not in splits:
            continue

        t = times_all[s:e]
        x = vals_clean[s:e]
        m = masks_all[s:e]
        raw_nimap = raw_nimap_all[s:e]
        T = len(t)

        delta_t = np.zeros((T, 1), dtype=np.float32)
        delta_x = np.zeros((T, D), dtype=np.float32)
        x_last = np.zeros((T, D), dtype=np.float32)

        running_last = np.zeros(D, dtype=np.float32)
        running_delta = np.zeros(D, dtype=np.float32)
        raw_last = np.zeros(T, dtype=np.float32)  # most recent real NIMAP reading, in mmHg
        running_raw = stats.mean[nimap_idx]  # training mean until the first reading

        for k in range(T):
            if k > 0:
                dt_step = float(t[k] - t[k - 1])
                delta_t[k, 0] = dt_step
                running_delta = np.where(m[k - 1] == 1.0, dt_step, running_delta + dt_step)
                delta_x[k] = running_delta
            running_last = np.where(m[k] == 1.0, x[k], running_last)
            x_last[k] = running_last
            if m[k, nimap_idx] == 1.0:
                running_raw = raw_nimap[k]
            raw_last[k] = running_raw

        target_mask = pair_all[s:e].copy()
        if target_mask.sum() == 0:
            continue
        target_dt = pair_dt_all[s:e].copy()
        target_nimap = pair_target_all[s:e].copy()
        # Current NIMAP is the reading at the pair's own time point, or the most recent
        # earlier reading when the robustness test has removed it.
        curr_nimap = np.where(target_mask, raw_last, 0.0).astype(np.float32)

        splits[split_name].append(
            {
                "RecordID": int(record_ids[s]),
                "time_hours": hours_all[s:e],
                "x": x,
                "m": m,
                "x_last": x_last,
                "delta_x": delta_x,
                "delta_t": delta_t,
                "target_mask": target_mask,
                "target_dt": target_dt,
                "target_nimap": target_nimap,
                "curr_nimap": curr_nimap,
            }
        )

    return splits


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_curr: np.ndarray,
    pred_hypo_override: Optional[np.ndarray] = None,
    threshold: float = HYPO_THRESHOLD,
) -> Dict[str, float]:
   
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))

    def cls_scores(true_bin: np.ndarray, pred_bin: np.ndarray) -> Tuple[float, float, float, int]:
        tp = int(np.sum((true_bin == 1) & (pred_bin == 1)))
        fp = int(np.sum((true_bin == 0) & (pred_bin == 1)))
        fn = int(np.sum((true_bin == 1) & (pred_bin == 0)))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        return prec, rec, f1, int(len(true_bin))

    true_hypo = (y_true < threshold).astype(int)
    if pred_hypo_override is not None:
        pred_hypo = pred_hypo_override.astype(int)
    else:
        pred_hypo = (y_pred < threshold).astype(int)

    p_all, r_all, f1_all, n_all = cls_scores(true_hypo, pred_hypo)

    # Onset: currently normotensive (curr >= 65), positive class = becomes hypotensive (next < 65)
    onset_idx = y_curr >= threshold
    p_on, r_on, f1_on, n_on = cls_scores(true_hypo[onset_idx], pred_hypo[onset_idx])

    # Recovery: currently hypotensive (curr < 65), positive class = recovers to normotensive (next >= 65)
    recov_idx = y_curr < threshold
    true_recov = (y_true[recov_idx] >= threshold).astype(int)
    pred_recov = (1 - pred_hypo[recov_idx]).astype(int)
    p_rec, r_rec, f1_rec, n_rec = cls_scores(true_recov, pred_recov)

    return {
        "n_pairs": n_all,
        "mae": mae,
        "rmse": rmse,
        "precision": p_all,
        "recall": r_all,
        "f1": f1_all,
        "onset_n": n_on,
        "onset_precision": p_on,
        "onset_recall": r_on,
        "onset_f1": f1_on,
        "recovery_n": n_rec,
        "recovery_precision": p_rec,
        "recovery_recall": r_rec,
        "recovery_f1": f1_rec,
    }


def evaluate_locf_baseline(sequences: List[dict]) -> Dict[str, float]:
    y_true = np.concatenate([s["target_nimap"][s["target_mask"]] for s in sequences])
    y_curr = np.concatenate([s["curr_nimap"][s["target_mask"]] for s in sequences])
    return compute_metrics(y_true=y_true, y_pred=y_curr, y_curr=y_curr)


def tune_prob_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_curr: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    best_thresh = 0.5
    best_f1 = -1.0
    for t in np.linspace(0.15, 0.85, 71):
        m = compute_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_curr=y_curr,
            pred_hypo_override=(y_prob >= t).astype(int),
        )
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_thresh = float(t)
    return best_thresh


if TORCH_AVAILABLE:
    class PatientSequenceDataset(Dataset):
        def __init__(self, sequences: List[dict]):
            self.sequences = sequences

        def __len__(self) -> int:
            return len(self.sequences)

        def __getitem__(self, idx: int) -> dict:
            return self.sequences[idx]


    def collate_patient_batch(batch: List[dict]) -> dict:
        B = len(batch)
        max_T = max(item["x"].shape[0] for item in batch)
        D = batch[0]["x"].shape[1]

        x = torch.zeros((B, max_T, D), dtype=torch.float32)
        m = torch.zeros((B, max_T, D), dtype=torch.float32)
        x_last = torch.zeros((B, max_T, D), dtype=torch.float32)
        delta_x = torch.zeros((B, max_T, D), dtype=torch.float32)
        delta_t = torch.zeros((B, max_T, 1), dtype=torch.float32)
        target_mask = torch.zeros((B, max_T), dtype=torch.bool)
        target_dt = torch.zeros((B, max_T), dtype=torch.float32)
        target_nimap = torch.zeros((B, max_T), dtype=torch.float32)
        curr_nimap = torch.zeros((B, max_T), dtype=torch.float32)
        lengths = torch.zeros(B, dtype=torch.long)

        for i, item in enumerate(batch):
            T = item["x"].shape[0]
            lengths[i] = T
            x[i, :T] = torch.from_numpy(item["x"])
            m[i, :T] = torch.from_numpy(item["m"])
            x_last[i, :T] = torch.from_numpy(item["x_last"])
            delta_x[i, :T] = torch.from_numpy(item["delta_x"])
            delta_t[i, :T] = torch.from_numpy(item["delta_t"])
            target_mask[i, :T] = torch.from_numpy(item["target_mask"])
            target_dt[i, :T] = torch.from_numpy(item["target_dt"])
            target_nimap[i, :T] = torch.from_numpy(item["target_nimap"])
            curr_nimap[i, :T] = torch.from_numpy(item["curr_nimap"])

        return {
            "x": x,
            "m": m,
            "x_last": x_last,
            "delta_x": delta_x,
            "delta_t": delta_t,
            "target_mask": target_mask,
            "target_dt": target_dt,
            "target_nimap": target_nimap,
            "curr_nimap": curr_nimap,
            "lengths": lengths,
        }


    class GRUDCell(nn.Module):

        def __init__(self, input_dim: int, hidden_dim: int):
            super().__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim

            self.w_dg_x = nn.Parameter(torch.zeros(input_dim))
            self.b_dg_x = nn.Parameter(torch.zeros(input_dim))
            self.decay_h = nn.Linear(1, hidden_dim)
            self.gru_cell = nn.GRUCell(input_dim * 2, hidden_dim)
            self.register_buffer("x_mean", torch.zeros(input_dim))

        def forward(
            self,
            x: torch.Tensor,
            m: torch.Tensor,
            x_last: torch.Tensor,
            delta_x: torch.Tensor,
            delta_t: torch.Tensor,
            h: torch.Tensor,
        ) -> torch.Tensor:
            gamma_x = torch.exp(-torch.relu(self.w_dg_x * delta_x + self.b_dg_x))
            x_hat = m * x + (1.0 - m) * (gamma_x * x_last + (1.0 - gamma_x) * self.x_mean)

            gamma_h = torch.exp(-torch.relu(self.decay_h(delta_t)))
            h_hat = gamma_h * h

            gru_in = torch.cat([x_hat, m], dim=-1)
            return self.gru_cell(gru_in, h_hat)


    class GRUDForecaster(nn.Module):
        """
        GRU-D encoder + shared trunk with:
            1) Residual continuous NIMAP forecasting head
            2) Auxiliary hypotension (< 65 mmHg) classification head
        """

        def __init__(
            self,
            input_dim: int = len(VITALS),
            hidden_dim: int = 128,
            dropout: float = 0.15,
            nimap_mean: float = 77.0,
            nimap_std: float = 15.5,
        ):
            super().__init__()
            self.hidden_dim = hidden_dim
            self.register_buffer("nimap_mean", torch.tensor(float(nimap_mean)))
            self.register_buffer("nimap_std", torch.tensor(float(nimap_std)))

            self.cell = GRUDCell(input_dim=input_dim, hidden_dim=hidden_dim)
            # Input to head: [h_k, target_dt, standardized curr_nimap]
            self.trunk = nn.Sequential(
                nn.Linear(hidden_dim + 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.reg_head = nn.Linear(hidden_dim, 1)
            self.cls_head = nn.Linear(hidden_dim, 1)

        def forward(
            self,
            x: torch.Tensor,
            m: torch.Tensor,
            x_last: torch.Tensor,
            delta_x: torch.Tensor,
            delta_t: torch.Tensor,
            target_dt: torch.Tensor,
            curr_nimap: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            B, T, _ = x.shape
            h = torch.zeros((B, self.hidden_dim), dtype=x.dtype, device=x.device)
            h_states = []
            for k in range(T):
                h = self.cell(
                    x[:, k, :],
                    m[:, k, :],
                    x_last[:, k, :],
                    delta_x[:, k, :],
                    delta_t[:, k, :],
                    h,
                )
                h_states.append(h)

            H = torch.stack(h_states, dim=1)  # (B, T, hidden_dim)
            curr_std = (curr_nimap - self.nimap_mean) / self.nimap_std
            head_in = torch.cat([H, target_dt.unsqueeze(-1), curr_std.unsqueeze(-1)], dim=-1)
            feat = self.trunk(head_in)

            delta_pred = self.reg_head(feat).squeeze(-1)
            logits = self.cls_head(feat).squeeze(-1)
            return curr_nimap + delta_pred, logits


    @torch.no_grad()
    def predict_split(
        model: GRUDForecaster,
        loader: DataLoader,
        device: torch.device,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        model.eval()
        all_true, all_pred, all_curr, all_prob = [], [], [], []
        for batch in loader:
            preds, logits = model(
                batch["x"].to(device),
                batch["m"].to(device),
                batch["x_last"].to(device),
                batch["delta_x"].to(device),
                batch["delta_t"].to(device),
                batch["target_dt"].to(device),
                batch["curr_nimap"].to(device),
            )
            probs = torch.sigmoid(logits)
            mask = batch["target_mask"].to(device)
            all_true.append(batch["target_nimap"].to(device)[mask].cpu().numpy())
            all_pred.append(preds[mask].cpu().numpy())
            all_curr.append(batch["curr_nimap"].to(device)[mask].cpu().numpy())
            all_prob.append(probs[mask].cpu().numpy())

        return (
            np.concatenate(all_true),
            np.concatenate(all_pred),
            np.concatenate(all_curr),
            np.concatenate(all_prob),
        )


def train_and_evaluate(args: argparse.Namespace) -> None:
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required to train GRU-D. Install via `uv add torch` or `pip install torch`.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    measured, pairs = load_measured(), load_pairs()
    df = add_mask_columns(measured)
    stats = compute_train_norm_stats(df)
    splits = build_patient_sequences(df, stats, pairs)

    for s in ["train", "val", "test"]:
        n_pairs = sum(int(item["target_mask"].sum()) for item in splits[s])
        print(f"  {s:5s}: {len(splits[s]):,} patients, {n_pairs:,} 1h pairs")

    locf_test = evaluate_locf_baseline(splits["test"])
    print("\n[LOCF Baseline - Test Set]")
    print(json.dumps(locf_test, indent=2))

    # Compute automatic pos_weight from training pairs if not overridden
    train_targets = np.concatenate([s["target_nimap"][s["target_mask"]] for s in splits["train"]])
    n_pos = float(np.sum(train_targets < HYPO_THRESHOLD))
    n_neg = float(np.sum(train_targets >= HYPO_THRESHOLD))
    auto_pos_weight = min(n_neg / max(n_pos, 1.0), 8.0) if args.pos_weight <= 0 else args.pos_weight
    print(f"\nTrain hypotension prevalence: {n_pos / len(train_targets):.3f} | pos_weight={auto_pos_weight:.2f}")

    train_loader = DataLoader(
        PatientSequenceDataset(splits["train"]),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_patient_batch,
    )
    val_loader = DataLoader(
        PatientSequenceDataset(splits["val"]),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_patient_batch,
    )
    test_loader = DataLoader(
        PatientSequenceDataset(splits["test"]),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_patient_batch,
    )

    nimap_idx = VITALS.index("NIMAP")
    model = GRUDForecaster(
        input_dim=len(VITALS),
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        nimap_mean=float(stats.mean[nimap_idx]),
        nimap_std=float(stats.std[nimap_idx]),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    reg_criterion = nn.SmoothL1Loss(beta=2.0, reduction="none")
    cls_criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(auto_pos_weight, device=device),
        reduction="none",
    )

    best_val_mae = math.inf
    best_state = None

    print(f"\nTraining GRU-D on {device} for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_pairs = 0

        for batch in train_loader:
            optimizer.zero_grad()
            preds, logits = model(
                batch["x"].to(device),
                batch["m"].to(device),
                batch["x_last"].to(device),
                batch["delta_x"].to(device),
                batch["delta_t"].to(device),
                batch["target_dt"].to(device),
                batch["curr_nimap"].to(device),
            )
            mask = batch["target_mask"].to(device)
            targets = batch["target_nimap"].to(device)[mask]
            curr = batch["curr_nimap"].to(device)[mask]
            pred_vals = preds[mask]
            logit_vals = logits[mask]

            true_hypo = (targets < HYPO_THRESHOLD).float()
            curr_hypo = (curr < HYPO_THRESHOLD).float()
            is_transition = (true_hypo != curr_hypo) | (true_hypo == 1.0)
            weights = torch.where(
                is_transition,
                torch.tensor(args.transition_weight, device=device),
                torch.tensor(1.0, device=device),
            )

            loss_reg = (reg_criterion(pred_vals, targets) * weights).mean()
            # Extra weight on onset/recovery transitions in classification as well
            cls_weights = torch.where(
                true_hypo != curr_hypo,
                torch.tensor(args.transition_weight, device=device),
                torch.tensor(1.0, device=device),
            )
            loss_cls = (cls_criterion(logit_vals, true_hypo) * cls_weights).mean()
            loss = loss_reg + args.cls_weight * loss_cls

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            n_batch_pairs = int(mask.sum().item())
            total_loss += float(loss.item()) * n_batch_pairs
            total_pairs += n_batch_pairs

        val_true, val_pred, val_curr, val_prob = predict_split(model, val_loader, device)
        val_thresh = tune_prob_threshold(val_true, val_prob, val_curr, val_pred)
        val_metrics = compute_metrics(
            y_true=val_true,
            y_pred=val_pred,
            y_curr=val_curr,
            pred_hypo_override=(val_prob >= val_thresh).astype(int),
        )
        scheduler.step(val_metrics["mae"])

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"Loss: {total_loss / max(total_pairs, 1):.4f} | "
            f"Val MAE: {val_metrics['mae']:.3f} | RMSE: {val_metrics['rmse']:.3f} | "
            f"F1@{val_thresh:.2f}: {val_metrics['f1']:.3f} | Onset F1: {val_metrics['onset_f1']:.3f}"
        )

        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    # Tune probability threshold strictly on validation set using best checkpoint
    val_true, val_pred, val_curr, val_prob = predict_split(model, val_loader, device)
    best_prob_thresh = tune_prob_threshold(val_true, val_prob, val_curr, val_pred)

    test_true, test_pred, test_curr, test_prob = predict_split(model, test_loader, device)
    test_metrics_cls = compute_metrics(
        y_true=test_true,
        y_pred=test_pred,
        y_curr=test_curr,
        pred_hypo_override=(test_prob >= best_prob_thresh).astype(int),
    )
    test_metrics_reg = compute_metrics(
        y_true=test_true,
        y_pred=test_pred,
        y_curr=test_curr,
        pred_hypo_override=None,
    )

    print(f"\n[GRU-D - Held-Out Test Set (Aux Classifier @ val-tuned p >= {best_prob_thresh:.2f})]")
    print(json.dumps(test_metrics_cls, indent=2))
    print("\n[GRU-D - Held-Out Test Set (Direct Regression < 65 mmHg)]")
    print(json.dumps(test_metrics_reg, indent=2))

    def run_model(measured: pd.DataFrame, pairs: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Forecast and hypotension probability for every row of `pairs`, in the same order."""
        sequences = sum(build_patient_sequences(add_mask_columns(measured), stats, pairs).values(), [])
        loader = DataLoader(
            PatientSequenceDataset(sequences),
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_patient_batch,
        )
        _, pred, _, prob = predict_split(model, loader, device)
        made = pd.DataFrame(
            {
                "RecordID": np.concatenate([np.full(int(s["target_mask"].sum()), s["RecordID"]) for s in sequences]),
                "time_hours": np.concatenate([s["time_hours"][s["target_mask"]] for s in sequences]),
                "pred": pred,
                "prob": prob,
            }
        )
        made = pairs[KEYS].merge(made, on=KEYS, how="left")
        return made["pred"].to_numpy(), made["prob"].to_numpy()

    # Same scoring as every other model (including the robustness test); writes
    # results/metrics/gru_d.json. The auxiliary classifier's alerts are stored alongside.
    evaluate_model(
        "gru_d",
        lambda measured, pairs: run_model(measured, pairs)[0],
        measured,
        pairs,
        extra={"aux_classifier": {"rule": f"classifier head, p >= {best_prob_thresh:.2f}", "test": test_metrics_cls}},
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate GRU-D baseline on PhysioNet 2012 NIMAP.")
    parser.add_argument("--hidden-dim", type=int, default=128, help="GRU-D hidden dimension")
    parser.add_argument("--dropout", type=float, default=0.15, help="Dropout rate in trunk")
    parser.add_argument("--batch-size", type=int, default=64, help="Patient batch size")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="AdamW learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay")
    parser.add_argument("--cls-weight", type=float, default=5.0, help="Weight for auxiliary BCE hypotension loss")
    parser.add_argument("--transition-weight", type=float, default=2.5, help="Loss weight multiplier for hypotension/transition pairs")
    parser.add_argument("--pos-weight", type=float, default=-1.0, help="Positive class weight for BCE (<=0 uses auto train ratio)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


if __name__ == "__main__":
    train_and_evaluate(parse_args())