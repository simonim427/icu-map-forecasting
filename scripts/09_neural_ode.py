"""Main model: latent Neural ODE with an ODE-RNN encoder (Rubanova et al., 2019).

Trains and scores two variants on CPU with fixed seeds: "plain" (a standard Neural ODE) and
"stable" (the stability-aware version). How they work is described on the LatentODE class.
Writes results/metrics and results/predictions for neural_ode_plain and neural_ode_stable.
"""
import copy

import numpy as np
import pandas as pd
import torch
from torch import nn
from torchdiffeq import odeint

from common import HORIZON, KEYS, SEED, VITALS, evaluate_model, load_measured, load_pairs

HIDDEN = 64
BATCH_SIZE = 64  # patients per batch
LEARNING_RATE = 2e-3
LR_DECAY = 0.8  # learning rate is multiplied by this after every epoch
MAX_EPOCHS = 20
PATIENCE = 5  # stop after this many epochs without a better validation MAE
MAX_GAP = 4.0  # hours; longer gaps between time points are integrated as 4 h
MAX_DROP, MAX_NOISE = 0.5, 0.3  # strongest degradation used while training the stable variant


class LatentODE(nn.Module):
    """How it works
      * A hidden state h summarises the patient so far.
      * Between time points h is not frozen: it follows a learned ODE, dh/dt = f(h), integrated
        over the real gap between measurements. This is how irregular sampling is handled.
      * At a time point that has readings, a GRU cell updates h with the new values.
      * To forecast, h is integrated one more hour ahead with the same ODE and decoded into
        the expected change from the most recent NIMAP reading.

    stable=False  f(h) = net(h). The standard Neural ODE (the proposal's fallback).
    stable=True   "Stability-aware", in two ways:
                  1. f(h) = tanh(net(h)) - h. The -h term pulls the state back, so h stays
                     bounded however long the gap between measurements is.
                  2. Every training batch is also seen in a degraded copy (time points removed,
                     noise added), so the model learns not to rely on clean, dense inputs.
    """

    def __init__(self, stable: bool):
        super().__init__()
        self.stable = stable
        self.net = nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, HIDDEN))
        self.cell = nn.GRUCell(2 * len(VITALS), HIDDEN)
        self.decoder = nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, 1))
        self.register_buffer("unit_interval", torch.tensor([0.0, 1.0]))

    def dynamics(self, h):
        if self.stable:
            return torch.tanh(self.net(h)) - h
        return self.net(h)

    def evolve(self, h, hours):
        """Integrate dh/dt = f(h) forward by `hours` (one value per row of h).

        Time is rescaled to s in [0, 1], so dh/ds = hours * f(h); this lets a single batched
        solve cover patients with different gaps. Fixed-step RK4, two steps per gap.
        """
        hours = hours.clamp(max=MAX_GAP).unsqueeze(-1)
        return odeint(lambda s, y: hours * self.dynamics(y), h, self.unit_interval,
                      method="rk4", options={"step_size": 0.5})[-1]

    def forward(self, dt, x, mask):
        """dt (B, L): hours since the previous time point. x, mask (B, L, vitals): readings
        (0 where missing) and which of them are real. Returns a forecast for every time point."""
        batch, length, _ = x.shape
        h = x.new_zeros(batch, HIDDEN)
        last = x.new_zeros(batch)  # most recent NIMAP reading; 0 is the training mean
        states, lasts = [], []
        for i in range(length):
            h = self.evolve(h, dt[:, i])
            updated = self.cell(torch.cat([x[:, i], mask[:, i]], dim=-1), h)
            h = torch.where(mask[:, i].any(dim=-1, keepdim=True), updated, h)  # update only if something was read
            last = torch.where(mask[:, i, 0] > 0, x[:, i, 0], last)  # column 0 is NIMAP
            states.append(h)
            lasts.append(last)
        h = torch.stack(states, dim=1).reshape(batch * length, HIDDEN)
        future = self.evolve(h, torch.full((batch * length,), HORIZON))
        change = self.decoder(future).reshape(batch, length)
        return torch.stack(lasts, dim=1) + change


def degrade(x, mask):
    """Training-time version of common.perturb: remove time points and add noise."""
    drop = torch.rand(x.shape[0], 1, 1) * MAX_DROP
    noise = torch.rand(x.shape[0], 1, 1) * MAX_NOISE
    kept = mask * (torch.rand(x.shape[0], x.shape[1], 1) > drop)
    return (x + noise * torch.randn_like(x)) * kept, kept


def to_tensors(measured: pd.DataFrame, mean: pd.Series, std: pd.Series):
    """Pack patients into padded arrays of shape (patients, longest sequence, ...)."""
    patient = measured.groupby("RecordID").ngroup().to_numpy()
    step = measured.groupby("RecordID").cumcount().to_numpy()
    shape = (patient.max() + 1, step.max() + 1)

    dt = np.zeros(shape, dtype=np.float32)
    x = np.zeros(shape + (len(VITALS),), dtype=np.float32)
    mask = np.zeros(shape + (len(VITALS),), dtype=np.float32)
    dt[patient, step] = measured.groupby("RecordID")["time_hours"].diff().fillna(0.0).to_numpy()
    x[patient, step] = ((measured[VITALS] - mean) / std).fillna(0.0).to_numpy()
    mask[patient, step] = measured[VITALS].notna().to_numpy()
    lengths = np.bincount(patient)
    return torch.from_numpy(dt), torch.from_numpy(x), torch.from_numpy(mask), patient, step, lengths


def train_and_evaluate(name: str, stable: bool, measured: pd.DataFrame, pairs: pd.DataFrame):
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    train_rows = measured[measured["split"] == "train"]
    mean, std = train_rows[VITALS].mean(), train_rows[VITALS].std()

    def predict(measured, pairs):
        dt, x, mask, patient, step, _ = to_tensors(measured, mean, std)
        model.eval()
        with torch.no_grad():
            pred = torch.cat([model(dt[i:i + 512], x[i:i + 512], mask[i:i + 512]) for i in range(0, len(x), 512)])
        pred = mean["NIMAP"] + std["NIMAP"] * pred.numpy()[patient, step]
        return pairs[KEYS].merge(measured[KEYS].assign(pred=pred), on=KEYS, how="left")["pred"].to_numpy()

    # Training tensors; `target` is NaN wherever a time point is not the start of a forecast pair.
    dt, x, mask, patient, step, lengths = to_tensors(train_rows, mean, std)
    targets = train_rows[KEYS].merge(pairs[KEYS + ["target"]], on=KEYS, how="left")["target"]
    target = np.full(dt.shape, np.nan, dtype=np.float32)
    target[patient, step] = ((targets - mean["NIMAP"]) / std["NIMAP"]).to_numpy()
    target = torch.from_numpy(target)
    is_pair = ~target.isnan()

    # Batches group patients of similar sequence length, which keeps padding (and run time) low.
    order = np.argsort(lengths)
    batches = [order[i:i + BATCH_SIZE] for i in range(0, len(order), BATCH_SIZE)]

    val_rows, val_pairs = measured[measured["split"] == "val"], pairs[pairs["split"] == "val"]
    model = LatentODE(stable)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=LR_DECAY)
    best_mae, best_state, epochs_since_best = float("inf"), None, 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for b in rng.permutation(len(batches)):
            idx = batches[b]
            n = lengths[idx].max()
            b_dt, b_x, b_mask, b_target, b_pair = dt[idx, :n], x[idx, :n], mask[idx, :n], target[idx, :n], is_pair[idx, :n]
            loss = ((model(b_dt, b_x, b_mask) - b_target)[b_pair] ** 2).mean()
            if stable:
                degraded = model(b_dt, *degrade(b_x, b_mask))
                loss = (loss + ((degraded - b_target)[b_pair] ** 2).mean()) / 2
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        val_mae = np.abs(predict(val_rows, val_pairs) - val_pairs["target"].to_numpy()).mean()
        print(f"{name} epoch {epoch:>2}: validation MAE {val_mae:.3f} mmHg", flush=True)
        if val_mae < best_mae:
            best_mae, best_state, epochs_since_best = val_mae, copy.deepcopy(model.state_dict()), 0
        else:
            epochs_since_best += 1
            if epochs_since_best >= PATIENCE:
                break

    model.load_state_dict(best_state)
    evaluate_model(name, predict, measured, pairs)


def main():
    torch.set_num_threads(1)  # small model: one thread is fastest and keeps runs repeatable
    measured, pairs = load_measured(), load_pairs()
    train_and_evaluate("neural_ode_plain", False, measured, pairs)
    train_and_evaluate("neural_ode_stable", True, measured, pairs)


if __name__ == "__main__":
    main()
