"""
PyTorch sequence models for hourly sepsis early warning.

Each model maps a window of shape (batch, time, features) plus a padding
mask of shape (batch, time) to a single logit per window.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader


def last_valid_index(mask: torch.Tensor) -> torch.Tensor:
    """For each row, index of the last real (non-padded) hour."""
    return (mask.sum(dim=1).long() - 1).clamp(min=0)


class LSTMClassifier(nn.Module):
    """2-layer LSTM -> dropout -> linear logit."""

    def __init__(self, n_features: int, hidden: int = 64, dropout: float = 0.3):
        super().__init__()
        # LSTM reads the 24-hour window from left to right and keeps a memory.
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )
        # Dropout randomly turns units off during training to reduce overfitting.
        self.drop = nn.Dropout(dropout)
        # One number (a logit) per window: sepsis vs not at the last hour.
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # lengths tell the LSTM to skip the left-padded zeros
        lengths = mask.sum(dim=1).clamp(min=1).cpu()
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_out, (h_n, _) = self.lstm(packed)
        # h_n[-1] is the last layer's hidden state for each patient-window
        h = self.drop(h_n[-1])
        return self.head(h).squeeze(-1)


class Chomp1d(nn.Module):
    """Remove extra timesteps on the right so a padded conv stays causal."""

    def __init__(self, chomp: int):
        super().__init__()
        self.chomp = chomp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, : -self.chomp].contiguous() if self.chomp > 0 else x


class TemporalBlock(nn.Module):
    """One residual TCN block: causal dilated conv -> ReLU -> dropout, twice."""

    def __init__(self, n_in: int, n_out: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        # Left-pad by (kernel-1)*dilation, then chomp, so output[t] uses only <= t
        pad = (kernel - 1) * dilation
        self.net = nn.Sequential(
            nn.utils.parametrizations.weight_norm(
                nn.Conv1d(n_in, n_out, kernel, padding=pad, dilation=dilation)
            ),
            Chomp1d(pad),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.utils.parametrizations.weight_norm(
                nn.Conv1d(n_out, n_out, kernel, padding=pad, dilation=dilation)
            ),
            Chomp1d(pad),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(n_in, n_out, 1) if n_in != n_out else nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.net(x) + self.downsample(x))


class TCNClassifier(nn.Module):
    """Stacked residual TCN blocks, then the last valid hour -> linear logit."""

    def __init__(
        self,
        n_features: int,
        channels: int = 64,
        kernel: int = 3,
        dilations: Optional[List[int]] = None,
        dropout: float = 0.3,
    ):
        super().__init__()
        if dilations is None:
            dilations = [1, 2, 4, 8]
        self.dilations = dilations
        self.kernel = kernel
        layers = []
        n_in = n_features
        for d in dilations:
            layers.append(TemporalBlock(n_in, channels, kernel, d, dropout))
            n_in = channels
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Linear(channels, 1)

    def receptive_field(self) -> int:
        """1 + (kernel-1) * sum(dilations) for this residual stack (two convs per block)."""
        # Each block has two convs of the same dilation
        return 1 + 2 * (self.kernel - 1) * int(sum(self.dilations))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Conv1d wants (batch, channels, time)
        h = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        idx = last_valid_index(mask)
        last = h[torch.arange(h.size(0), device=h.device), idx]
        return self.head(last).squeeze(-1)


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sine/cosine clocks so the Transformer knows hour order."""

    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, T, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerClassifier(nn.Module):
    """Linear projection -> positional encoding -> Transformer encoder -> last token."""

    def __init__(
        self,
        n_features: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.3,
        max_len: int = 64,
    ):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos = SinusoidalPositionalEncoding(d_model, max_len=max_len)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, return_attn: bool = False):
        h = self.pos(self.input_proj(x))
        t = x.size(1)
        causal = torch.triu(torch.ones(t, t, dtype=torch.bool, device=x.device), diagonal=1)
        # Avoid passing src_key_padding_mask when left-padding with a causal mask,
        # which would cause all -inf in the leftmost padded tokens and produce NaNs.
        h = self.encoder(h, mask=causal)
        idx = last_valid_index(mask)
        last = self.drop(h[torch.arange(h.size(0), device=h.device), idx])
        logit = self.head(last).squeeze(-1)
        if return_attn:
            return logit, h
        return logit


def train_sequence_model(
    model: nn.Module,
    train_ds,
    val_ds,
    device: torch.device,
    pos_weight: float,
    epochs: int = 10,
    patience: int = 3,
    lr: float = 1e-3,
    batch_size: int = 128,
    ckpt_path: Optional[Path] = None,
    clip_grad: float = 1.0,
) -> Dict:
    """
    Train with Adam + BCEWithLogitsLoss(pos_weight) and early stopping on val PR-AUC.

    pos_weight > 1 tells the loss to care more about the rare sepsis hours.
    """
    model = model.to(device)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    pw = torch.tensor([pos_weight], dtype=torch.float32, device=device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)

    history = {"train_loss": [], "val_loss": [], "val_pr_auc": []}
    best_pr, best_state, wait = -1.0, None, 0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for seq, mask, y, _ in train_loader:
            seq, mask, y = seq.to(device), mask.to(device), y.to(device)
            opt.zero_grad()
            logit = model(seq, mask)
            loss = crit(logit, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            opt.step()
            losses.append(loss.item())
        train_loss = float(np.mean(losses))

        model.eval()
        vlosses, probs, ys = [], [], []
        with torch.no_grad():
            for seq, mask, y, _ in val_loader:
                seq, mask, y = seq.to(device), mask.to(device), y.to(device)
                logit = model(seq, mask)
                vlosses.append(crit(logit, y).item())
                probs.append(torch.sigmoid(logit).cpu().numpy())
                ys.append(y.cpu().numpy())
        val_loss = float(np.mean(vlosses))
        y_true = np.concatenate(ys)
        y_prob = np.concatenate(probs)
        try:
            val_pr = float(average_precision_score(y_true, y_prob))
        except ValueError:
            val_pr = 0.0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_pr_auc"].append(val_pr)
        print(
            f"epoch {epoch:02d}  train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  val_pr_auc={val_pr:.4f}"
        )

        if val_pr > best_pr:
            best_pr = val_pr
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
            if ckpt_path is not None:
                ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"state_dict": best_state, "val_pr_auc": best_pr}, ckpt_path)
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch} (best val PR-AUC={best_pr:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    history["train_seconds"] = time.time() - t0
    history["best_val_pr_auc"] = best_pr
    return history


def mc_dropout_predict(model: nn.Module, dataset, device, n_passes: int = 30, batch_size: int = 256):
    """
    Monte Carlo dropout: keep dropout ON and run several forward passes.

    Mean = predicted risk. Std = uncertainty.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    # enable dropout
    model.train()
    all_passes = []
    with torch.no_grad():
        for _ in range(n_passes):
            probs = []
            for seq, mask, y, _ in loader:
                seq, mask = seq.to(device), mask.to(device)
                logit = model(seq, mask)
                probs.append(torch.sigmoid(logit).cpu().numpy())
            all_passes.append(np.concatenate(probs))
    stacked = np.stack(all_passes, axis=0)
    return stacked.mean(axis=0), stacked.std(axis=0)
