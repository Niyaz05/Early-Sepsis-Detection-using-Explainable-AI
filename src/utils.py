"""
Shared helpers for sepsis early-warning notebooks.

These functions keep splits, metrics, plots, and sequence loading consistent
so no notebook silently uses a different evaluation recipe.
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

SEED = 42


def get_project_root() -> Path:
    """Return PROJECT_ROOT. Robust to running in notebooks/ or from project root."""
    if (Path.cwd() / "data").exists():
        return Path.cwd()
    if (Path.cwd().parent / "data").exists():
        return Path.cwd().parent
    return Path(__file__).resolve().parents[1]



def ensure_dirs(root: Path) -> Dict[str, Path]:
    """Create the standard project folders and return their paths."""
    paths = {
        "root": root,
        "raw": root / "data" / "raw",
        "processed": root / "data" / "processed",
        "splits": root / "data" / "processed" / "splits",
        "tabular": root / "data" / "processed" / "tabular",
        "sequences": root / "data" / "processed" / "sequences",
        "models": root / "models",
        "figures": root / "results" / "figures",
        "tables": root / "results" / "tables",
        "results": root / "results",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def set_seed(seed: int = SEED) -> None:
    """Set python, numpy, and (if present) torch / xgboost seeds."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass
    try:
        import xgboost as xgb

        # xgboost uses random_state on the estimator; this is extra insurance
        _ = xgb
    except Exception:
        pass


def print_library_versions() -> None:
    """Print versions of the libraries this project depends on."""
    import importlib
    import sys

    print("python", sys.version.replace("\n", " "))
    for name in [
        "numpy",
        "pandas",
        "sklearn",
        "xgboost",
        "torch",
        "shap",
        "optuna",
        "matplotlib",
        "seaborn",
    ]:
        try:
            mod = importlib.import_module(name)
            print(f"{name:12s} {getattr(mod, '__version__', 'unknown')}")
        except Exception as exc:
            print(f"{name:12s} NOT INSTALLED ({exc})")


def get_torch_device():
    """Pick CUDA, then Apple MPS, otherwise CPU. Print the choice."""
    import torch

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("Using device:", device)
    return device


# ---------------------------------------------------------------------------
# Schema identification (do not hard-code until we inspect the CSV)
# ---------------------------------------------------------------------------

def identify_schema(df: pd.DataFrame) -> Dict:
    """
    Infer ID, time, target, vitals, labs, and demographics from column names.

    Returns a dict used by every later notebook. Ambiguous choices are listed
    under 'assumptions'.
    """
    cols = list(df.columns)
    lower = {c.lower(): c for c in cols}

    def pick(*candidates: str) -> Optional[str]:
        for cand in candidates:
            if cand.lower() in lower:
                return lower[cand.lower()]
        return None

    patient_id = pick("Patient_ID", "patient_id", "PatientID", "id")
    hour = pick("Hour", "ICULOS", "hour", "time")
    iculos = pick("ICULOS")
    target = pick("SepsisLabel", "sepsis_label", "Sepsis", "label")

    # PhysioNet Challenge 2019 vital-sign names
    vital_candidates = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2"]
    lab_candidates = [
        "BaseExcess",
        "HCO3",
        "FiO2",
        "pH",
        "PaCO2",
        "SaO2",
        "AST",
        "BUN",
        "Alkalinephos",
        "Calcium",
        "Chloride",
        "Creatinine",
        "Bilirubin_direct",
        "Glucose",
        "Lactate",
        "Magnesium",
        "Phosphate",
        "Potassium",
        "Bilirubin_total",
        "TroponinI",
        "Hct",
        "Hgb",
        "PTT",
        "WBC",
        "Fibrinogen",
        "Platelets",
    ]
    demo_candidates = ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime"]

    vitals = [c for c in vital_candidates if c in cols]
    labs = [c for c in lab_candidates if c in cols]
    demographics = [c for c in demo_candidates if c in cols]

    # Drop columns that are indexes or 100% empty
    drop_candidates = []
    for c in cols:
        if c.startswith("Unnamed"):
            drop_candidates.append(c)
        elif df[c].isna().all():
            drop_candidates.append(c)

    assumptions = [
        "This file matches the PhysioNet 2019 Sepsis Challenge column names.",
        "Patient_ID identifies a stay; rows are hourly observations.",
        "Hour is the time index within the stay (starts at 0).",
        "ICULOS is ICU length of stay in hours (not always Hour+1).",
        "SepsisLabel=1 means the sepsis-onset window at that hour "
        "(PhysioNet already shifts the clinical onset by up to 6 hours).",
        "Gender: 0 = female, 1 = male (PhysioNet coding).",
        "Unit1/Unit2 are ICU-type flags and are often missing.",
        "EtCO2 (if entirely missing) is dropped as unusable.",
        "Unnamed: 0 is a duplicate of Hour and is dropped.",
        "The last row of this Kaggle dump is a truncated empty record and is dropped.",
    ]

    schema = {
        "patient_id": patient_id,
        "hour": hour,
        "iculos": iculos,
        "target": target,
        "vitals": vitals,
        "labs": labs,
        "demographics": demographics,
        "drop_candidates": drop_candidates,
        "assumptions": assumptions,
    }
    return schema


def print_schema(schema: Dict) -> None:
    """Pretty-print the identified schema so it is visible in the notebook."""
    print("Identified schema")
    print("  patient_id :", schema["patient_id"])
    print("  hour       :", schema["hour"])
    print("  iculos     :", schema["iculos"])
    print("  target     :", schema["target"])
    print("  vitals     :", schema["vitals"])
    print("  labs       :", schema["labs"])
    print("  demographics:", schema["demographics"])
    print("  drop_candidates:", schema["drop_candidates"])
    print("Assumptions:")
    for a in schema["assumptions"]:
        print(" -", a)


def save_schema(schema: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(schema, f, indent=2)


def load_schema(path: Path) -> Dict:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Metrics (same functions for every model)
# ---------------------------------------------------------------------------

def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    """Return TP/FP/TN/FN. y_pred is already 0/1."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    """
    Standard test-set metrics.

    Accuracy is computed but is NOT a headline metric because sepsis rows are rare.
    PR-AUC and ROC-AUC do not depend on the threshold; the rest do.
    """
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)
    counts = confusion_counts(y_true, y_pred)
    tp, fp, tn, fn = counts["tp"], counts["fp"], counts["tn"], counts["fn"]

    spec = tn / (tn + fp) if (tn + fp) else 0.0
    rec = recall_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    try:
        roc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc = float("nan")
    try:
        pr = average_precision_score(y_true, y_prob)
    except ValueError:
        pr = float("nan")

    return {
        "roc_auc": float(roc),
        "pr_auc": float(pr),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "specificity": float(spec),
        "accuracy": float((tp + tn) / max(len(y_true), 1)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "threshold": float(threshold),
        **counts,
        "n": int(len(y_true)),
        "n_positive": int(y_true.sum()),
    }


def choose_threshold_max_f1(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    grid: Optional[np.ndarray] = None,
) -> Tuple[float, Dict[str, float]]:
    """
    Pick the decision threshold on VALIDATION only by maximizing F1.

    We search a grid of probabilities. Ties break toward the lower threshold
    (slightly more sensitive, which is preferred for a warning system).
    """
    from sklearn.metrics import f1_score

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    if grid is None:
        grid = np.unique(np.concatenate(([0.01, 0.99], np.linspace(0.05, 0.95, 19), np.quantile(y_prob, np.linspace(0.05, 0.95, 19)))))
        grid = np.clip(grid, 0.01, 0.99)

    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1 + 1e-12 or (abs(f1 - best_f1) <= 1e-12 and t < best_t):
            best_f1 = f1
            best_t = float(t)
    metrics = classification_metrics(y_true, y_prob, best_t)
    return best_t, metrics


def early_detection_metrics(
    df: pd.DataFrame,
    patient_col: str,
    hour_col: str,
    y_true_col: str = "y_true",
    y_prob_col: str = "y_prob",
    threshold: float = 0.5,
    min_lead_hours: int = 6,
) -> Dict[str, float]:
    """
    Early-detection statistics computed per PATIENT.

    Onset hour = first hour where the true label is 1.
    First alert = first hour where predicted risk >= threshold.
    Early detection time = onset - first_alert  (positive => warned before onset).

    False alarm rate per patient-day uses non-sepsis patients plus the hours
    before onset for sepsis patients. A patient-day is 24 hours of stay.
    """
    rows = []
    false_alarms = 0
    at_risk_hours = 0

    for pid, g in df.groupby(patient_col, sort=False):
        g = g.sort_values(hour_col)
        y = g[y_true_col].to_numpy().astype(int)
        p = g[y_prob_col].to_numpy().astype(float)
        hours = g[hour_col].to_numpy()
        pred = p >= threshold
        ever_sepsis = y.max() == 1

        if ever_sepsis:
            onset_idx = int(np.argmax(y == 1))
            onset_hour = int(hours[onset_idx])
            if pred.any():
                alert_idx = int(np.argmax(pred))
                alert_hour = int(hours[alert_idx])
                edt = onset_hour - alert_hour
                detected = True
            else:
                alert_hour = None
                edt = None
                detected = False
            # false alarms: alerts strictly before onset
            false_alarms += int(pred[:onset_idx].sum())
            at_risk_hours += int(onset_idx)
            rows.append(
                {
                    "patient_id": pid,
                    "ever_sepsis": 1,
                    "onset_hour": onset_hour,
                    "alert_hour": alert_hour,
                    "early_detection_time": edt,
                    "detected": int(detected),
                    "detected_6h": int(detected and edt is not None and edt >= min_lead_hours),
                }
            )
        else:
            false_alarms += int(pred.sum())
            at_risk_hours += int(len(g))
            rows.append(
                {
                    "patient_id": pid,
                    "ever_sepsis": 0,
                    "onset_hour": None,
                    "alert_hour": None,
                    "early_detection_time": None,
                    "detected": 0,
                    "detected_6h": 0,
                }
            )

    det = pd.DataFrame(rows)
    sepsis = det[det["ever_sepsis"] == 1]
    detected = sepsis[sepsis["detected"] == 1]
    edt = detected["early_detection_time"].astype(float)

    patient_days = at_risk_hours / 24.0 if at_risk_hours else np.nan
    return {
        "n_sepsis_patients": int(len(sepsis)),
        "pct_detected": float(sepsis["detected"].mean() * 100) if len(sepsis) else float("nan"),
        "pct_detected_6h_before": float(sepsis["detected_6h"].mean() * 100) if len(sepsis) else float("nan"),
        "edt_mean_hours": float(edt.mean()) if len(edt) else float("nan"),
        "edt_median_hours": float(edt.median()) if len(edt) else float("nan"),
        "false_alarms_per_patient_day": float(false_alarms / patient_days) if patient_days else float("nan"),
        "n_detected": int(len(detected)),
    }, det


def metrics_to_row(model_name: str, metrics: Dict, extra: Optional[Dict] = None) -> pd.DataFrame:
    row = {"model": model_name}
    row.update(metrics)
    if extra:
        row.update(extra)
    return pd.DataFrame([row])


def save_metrics_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_predictions(
    patient_id: np.ndarray,
    hour: np.ndarray,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    path: Path,
) -> pd.DataFrame:
    """Save test predictions in the common format used by the comparison notebook."""
    out = pd.DataFrame(
        {
            "patient_id": patient_id,
            "hour": hour,
            "y_true": np.asarray(y_true).astype(int),
            "y_prob": np.asarray(y_prob).astype(float),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return out


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def save_fig(fig, path: Path):
    """Save a matplotlib figure and keep it displayed in the notebook."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")


def plot_confusion_heatmap(counts: Dict[str, int], title: str, path: Path):
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = np.array([[counts["tn"], counts["fp"]], [counts["fn"], counts["tp"]]])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["True 0", "True 1"],
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    save_fig(fig, path)
    return fig


def plot_roc_pr(y_true, y_prob, title_prefix: str, path_roc: Path, path_pr: Path):
    from sklearn.metrics import (
        PrecisionRecallDisplay,
        RocCurveDisplay,
        average_precision_score,
        roc_auc_score,
    )
    import matplotlib.pyplot as plt

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_true, y_prob, ax=ax, name=title_prefix)
    ax.set_title(f"{title_prefix} ROC (AUC={roc_auc_score(y_true, y_prob):.3f})")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend()
    save_fig(fig, path_roc)

    fig2, ax2 = plt.subplots(figsize=(5, 4))
    PrecisionRecallDisplay.from_predictions(y_true, y_prob, ax=ax2, name=title_prefix)
    ax2.set_title(f"{title_prefix} PR (AP={average_precision_score(y_true, y_prob):.3f})")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.legend()
    save_fig(fig2, path_pr)
    return fig, fig2


# ---------------------------------------------------------------------------
# Sequence dataset
# ---------------------------------------------------------------------------

class WindowSequenceDataset:
    """
    PyTorch-compatible dataset of sliding windows over hourly patient rows.

    X_mm: memmap or ndarray (n_rows, n_features) stored in patient-contiguous order.
    patient_start[i] = first row index of the patient that owns row i.
    The window ending at row i uses rows [max(start, i-window+1), i] and pads the
    left side with zeros. mask=1 where a real hour exists.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        patient_start: np.ndarray,
        patient_id: np.ndarray,
        hour: np.ndarray,
        window: int = 24,
        indices: Optional[np.ndarray] = None,
    ):
        self.X = X
        self.y = y.astype(np.float32)
        self.patient_start = patient_start.astype(np.int64)
        self.patient_id = patient_id
        self.hour = hour
        self.window = int(window)
        self.indices = np.arange(len(y), dtype=np.int64) if indices is None else np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        import torch

        row = int(self.indices[i])
        start = int(self.patient_start[row])
        left = row - self.window + 1
        pad = 0
        if left < start:
            pad = start - left
            left = start
        seq = np.zeros((self.window, self.X.shape[1]), dtype=np.float32)
        mask = np.zeros((self.window,), dtype=np.float32)
        sl = self.X[left : row + 1]
        seq[pad:] = sl
        mask[pad:] = 1.0
        return (
            torch.from_numpy(seq),
            torch.from_numpy(mask),
            torch.tensor(self.y[row], dtype=torch.float32),
            torch.tensor(row, dtype=torch.int64),
        )


def downsample_indices(
    y: np.ndarray,
    max_neg_per_pos: float = 8.0,
    seed: int = SEED,
) -> np.ndarray:
    """Keep every positive row and a random subset of negatives (TRAIN only)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    n_keep = min(len(neg), int(max(1, len(pos) * max_neg_per_pos)))
    keep_neg = rng.choice(neg, size=n_keep, replace=False) if len(neg) else neg
    idx = np.concatenate([pos, keep_neg])
    rng.shuffle(idx)
    return idx


def load_sequence_split(seq_dir: Path, split: str):
    """Load memmap features plus row metadata for one split."""
    meta = np.load(seq_dir / f"{split}_meta.npz", allow_pickle=False)
    X = np.load(seq_dir / f"{split}_X.npy", mmap_mode="r")
    return {
        "X": X,
        "y": meta["y"],
        "patient_start": meta["patient_start"],
        "patient_id": meta["patient_id"],
        "hour": meta["hour"],
        "feature_names": meta["feature_names"].tolist() if "feature_names" in meta.files else None,
    }


def predict_torch_model(model, dataset, device, batch_size: int = 256) -> np.ndarray:
    """Run a trained model over a dataset and return sigmoid probabilities."""
    import torch
    from torch.utils.data import DataLoader

    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    probs = np.zeros(len(dataset), dtype=np.float32)
    offset = 0
    with torch.no_grad():
        for seq, mask, y, rows in loader:
            seq = seq.to(device)
            mask = mask.to(device)
            logit = model(seq, mask)
            p = torch.sigmoid(logit).detach().cpu().numpy().reshape(-1)
            n = len(p)
            probs[offset : offset + n] = p
            offset += n
    return probs


def count_parameters(model) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def file_size_mb(path: Path) -> float:
    if not path.exists():
        return float("nan")
    return round(path.stat().st_size / (1024 ** 2), 3)


def permutation_importance_torch(
    model,
    dataset,
    device,
    feature_names: Sequence[str],
    n_samples: int = 400,
    seed: int = SEED,
) -> pd.DataFrame:
    """
    Fallback explainability: shuffle one feature across time and measure PR-AUC drop.

    Used when SHAP Deep/Gradient explainers are unstable on sequential models.
    """
    from sklearn.metrics import average_precision_score
    import torch
    from torch.utils.data import DataLoader, Subset

    rng = np.random.default_rng(seed)
    n = min(n_samples, len(dataset))
    idx = rng.choice(len(dataset), size=n, replace=False)
    subset = Subset(dataset, idx.tolist())
    loader = DataLoader(subset, batch_size=64, shuffle=False)

    seqs, masks, ys = [], [], []
    with torch.no_grad():
        for seq, mask, y, _ in loader:
            seqs.append(seq)
            masks.append(mask)
            ys.append(y)
    seq = torch.cat(seqs).to(device)
    mask = torch.cat(masks).to(device)
    y_true = torch.cat(ys).cpu().numpy()

    model.eval()
    with torch.no_grad():
        base = torch.sigmoid(model(seq, mask)).cpu().numpy().reshape(-1)
    base_ap = average_precision_score(y_true, base)

    drops = []
    for f in range(seq.shape[-1]):
        seq_perm = seq.clone()
        # Shuffle that feature along the batch axis, keeping time order inside a row
        perm = torch.randperm(seq_perm.size(0), device=device)
        seq_perm[:, :, f] = seq_perm[perm, :, f]
        with torch.no_grad():
            p = torch.sigmoid(model(seq_perm, mask)).cpu().numpy().reshape(-1)
        ap = average_precision_score(y_true, p)
        drops.append({"feature": feature_names[f], "pr_auc_drop": float(base_ap - ap)})
    out = pd.DataFrame(drops).sort_values("pr_auc_drop", ascending=False)
    return out, base_ap


def time_step_importance_torch(
    model,
    dataset,
    device,
    n_samples: int = 400,
    seed: int = SEED,
) -> np.ndarray:
    """Zero out one time step at a time and measure mean |prob change|."""
    import torch
    from torch.utils.data import DataLoader, Subset

    rng = np.random.default_rng(seed)
    n = min(n_samples, len(dataset))
    idx = rng.choice(len(dataset), size=n, replace=False)
    subset = Subset(dataset, idx.tolist())
    loader = DataLoader(subset, batch_size=64, shuffle=False)
    seqs, masks = [], []
    with torch.no_grad():
        for seq, mask, y, _ in loader:
            seqs.append(seq)
            masks.append(mask)
    seq = torch.cat(seqs).to(device)
    mask = torch.cat(masks).to(device)
    T = seq.shape[1]
    model.eval()
    with torch.no_grad():
        base = torch.sigmoid(model(seq, mask))
    impact = np.zeros(T, dtype=np.float64)
    for t in range(T):
        seq_t = seq.clone()
        seq_t[:, t, :] = 0
        with torch.no_grad():
            p = torch.sigmoid(model(seq_t, mask))
        impact[t] = (p - base).abs().mean().item()
    return impact


def patient_bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    patient_id: np.ndarray,
    n_boot: int = 500,
    seed: int = SEED,
) -> Dict[str, Tuple[float, float, float]]:
    """
    Bootstrap 95% CIs for ROC-AUC and PR-AUC by resampling PATIENTS, not rows.

    Resampling rows would treat hours from the same stay as independent.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    patient_id = np.asarray(patient_id)
    patients = np.unique(patient_id)
    # map patient -> row indices
    index = pd.Series(np.arange(len(patient_id)), index=patient_id)
    groups = {pid: np.where(patient_id == pid)[0] for pid in patients}

    rocs, prs = [], []
    for _ in range(n_boot):
        sample = rng.choice(patients, size=len(patients), replace=True)
        rows = np.concatenate([groups[pid] for pid in sample])
        yt, yp = y_true[rows], y_prob[rows]
        if yt.min() == yt.max():
            continue
        rocs.append(roc_auc_score(yt, yp))
        prs.append(average_precision_score(yt, yp))
    def ci(arr):
        arr = np.asarray(arr)
        return float(arr.mean()), float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))

    return {"roc_auc": ci(rocs), "pr_auc": ci(prs)}
