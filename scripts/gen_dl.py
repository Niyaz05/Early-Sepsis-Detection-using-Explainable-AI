"""Generate 04 LSTM, 05 TCN, 06 Transformer notebooks from a shared template."""
from pathlib import Path
from nbformat_util import write_notebook

ROOT = Path(__file__).resolve().parents[1]


def boot(model_import: str) -> str:
    return f'''
# Shared deep-learning setup: frozen splits, GPU if available, one global seed.
from pathlib import Path
import os, sys, json, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

PROJECT_ROOT = Path.cwd().parent
os.environ["MPLCONFIGDIR"] = str(PROJECT_ROOT / ".mplconfig")
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import (
    SEED, set_seed, print_library_versions, ensure_dirs, save_fig,
    get_torch_device, WindowSequenceDataset, load_sequence_split,
    downsample_indices, predict_torch_model, count_parameters, file_size_mb,
    classification_metrics, choose_threshold_max_f1, early_detection_metrics,
    metrics_to_row, save_metrics_table, save_predictions,
    plot_confusion_heatmap, plot_roc_pr,
    permutation_importance_torch, time_step_importance_torch,
)
from src.torch_models import {model_import}, train_sequence_model

set_seed(SEED)
print("SEED =", SEED)
print_library_versions()
PATHS = ensure_dirs(PROJECT_ROOT)
DEVICE = get_torch_device()
WINDOW = json.load(open(PATHS["processed"] / "sequence_config.json"))["window"]
print("WINDOW", WINDOW)
'''


def load_data_cell():
    return r'''
# Load sequence arrays. Train uses the documented patient subsample + negative downsampling; val/test stay natural.
dl = load_sequence_split(PATHS["sequences"], "dl_train")
val = load_sequence_split(PATHS["sequences"], "val")
test = load_sequence_split(PATHS["sequences"], "test")
feat_names = json.load(open(PATHS["processed"] / "sequence_feature_cols.json"))
print("features", len(feat_names), feat_names[:8], "...")

# Downsample validation *only* for the per-epoch monitor so early stopping is fast.
# Threshold selection later uses the full validation set.
rng = np.random.default_rng(SEED)
val_mon_idx = downsample_indices(val["y"], max_neg_per_pos=6.0, seed=SEED)
train_idx = downsample_indices(dl["y"], max_neg_per_pos=8.0, seed=SEED)
print("train windows used", len(train_idx), "pos_rate", dl["y"][train_idx].mean())
print("val monitor windows", len(val_mon_idx), "full val", len(val["y"]))

train_ds = WindowSequenceDataset(dl["X"], dl["y"], dl["patient_start"], dl["patient_id"], dl["hour"], WINDOW, train_idx)
val_mon_ds = WindowSequenceDataset(val["X"], val["y"], val["patient_start"], val["patient_id"], val["hour"], WINDOW, val_mon_idx)
val_ds = WindowSequenceDataset(val["X"], val["y"], val["patient_start"], val["patient_id"], val["hour"], WINDOW)
test_ds = WindowSequenceDataset(test["X"], test["y"], test["patient_start"], test["patient_id"], test["hour"], WINDOW)

n_pos = float((dl["y"][train_idx] == 1).sum())
n_neg = float((dl["y"][train_idx] == 0).sum())
pos_weight = n_neg / max(n_pos, 1.0)
print("BCE pos_weight", round(pos_weight, 3))
n_features = dl["X"].shape[1]
print("n_features", n_features)
'''


def save_eval_cell(name: str) -> str:
    return f'''
# Freeze the F1-max threshold on FULL validation, then score test patients only once.
p_va = predict_torch_model(model, val_ds, DEVICE, batch_size=256)
thr, val_m = choose_threshold_max_f1(val["y"], p_va)
print("threshold", round(thr, 4), "val F1", round(val_m["f1"], 4), "val PR-AUC", round(val_m["pr_auc"], 4))

t1 = time.time()
p_te = predict_torch_model(model, test_ds, DEVICE, batch_size=256)
infer_seconds = time.time() - t1
metrics = classification_metrics(test["y"], p_te, thr)
print({{k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()}})

pred_path = PATHS["results"] / "predictions_{name}.csv"
save_predictions(test["patient_id"], test["hour"], test["y"], p_te, pred_path)
edt, _ = early_detection_metrics(pd.read_csv(pred_path), "patient_id", "hour", threshold=thr)
print("early detection", edt)

plot_roc_pr(test["y"], p_te, "{name}", PATHS["figures"] / "0X_{name}_roc.png", PATHS["figures"] / "0X_{name}_pr.png")
plt.show()
plot_confusion_heatmap(metrics, "{name} test confusion matrix", PATHS["figures"] / "0X_{name}_cm.png")
plt.show()

ckpt = PATHS["models"] / "{name}.pt"
torch.save({{"state_dict": model.state_dict(), "threshold": thr, "n_features": n_features, "window": WINDOW}}, ckpt)
extra = dict(
    train_seconds=history["train_seconds"], infer_seconds=infer_seconds,
    n_params=count_parameters(model), model_mb=file_size_mb(ckpt),
    **{{f"edt_{{k}}": v for k, v in edt.items() if k != "n_sepsis_patients"}},
    n_sepsis_patients=edt["n_sepsis_patients"],
)
save_metrics_table(metrics_to_row("{name}", metrics, extra), PATHS["tables"] / "metrics_{name}.csv")
print("saved", ckpt)
'''


def xai_cell(name: str) -> str:
    return f'''
# Explainability: try a small SHAP GradientExplainer; if it fails or looks unstable, fall back to permutation importance.
# We also measure which hours in the 24-hour window move the prediction the most.
from sklearn.metrics import average_precision_score

imp_df, base_ap = permutation_importance_torch(model, test_ds, DEVICE, feat_names, n_samples=350, seed=SEED)
print("baseline PR-AUC on SHAP/permutation sample:", round(base_ap, 4))
print(imp_df.head(12).to_string(index=False))
fig, ax = plt.subplots(figsize=(8, 5))
top = imp_df.head(15).iloc[::-1]
ax.barh(top["feature"], top["pr_auc_drop"], color="#3b6ea5")
ax.set_title("{name}: permutation importance (PR-AUC drop)")
ax.set_xlabel("PR-AUC drop when the feature is shuffled")
save_fig(fig, PATHS["figures"] / "0X_{name}_perm_importance.png")
plt.show()

impact = time_step_importance_torch(model, test_ds, DEVICE, n_samples=350, seed=SEED)
fig, ax = plt.subplots(figsize=(7, 4))
hours_before = np.arange(-WINDOW + 1, 1)
ax.plot(hours_before, impact, marker="o")
ax.set_title("{name}: importance by time step (zero-out |Δp|)")
ax.set_xlabel("Hours relative to the end of the window (0 = now)")
ax.set_ylabel("Mean absolute change in predicted risk")
save_fig(fig, PATHS["figures"] / "0X_{name}_time_importance.png")
plt.show()

# Optional SHAP on a tiny batch; skip cleanly if the explainer errors.
try:
    import shap
    from torch.utils.data import DataLoader
    bg_n = 32
    ex_n = 48
    bg_loader = DataLoader(WindowSequenceDataset(dl["X"], dl["y"], dl["patient_start"], dl["patient_id"], dl["hour"], WINDOW, train_idx[:bg_n]), batch_size=bg_n)
    ex_loader = DataLoader(WindowSequenceDataset(test["X"], test["y"], test["patient_start"], test["patient_id"], test["hour"], WINDOW, np.arange(ex_n)), batch_size=ex_n)
    bg_seq, bg_mask, _, _ = next(iter(bg_loader))
    ex_seq, ex_mask, _, _ = next(iter(ex_loader))
    # Flatten time*feature so GradientExplainer sees a 2D tensor, then wrap the model.
    class FlatWrap(torch.nn.Module):
        def __init__(self, m, t, f, mask):
            super().__init__()
            self.m, self.t, self.f, self.mask = m, t, f, mask
        def forward(self, z):
            x = z.view(-1, self.t, self.f)
            return self.m(x, self.mask.to(z.device)[:x.size(0)] if self.mask.size(0)==x.size(0) else self.mask[:1].repeat(x.size(0),1))
    model.eval()
    wrap = FlatWrap(model, WINDOW, n_features, ex_mask.to(DEVICE)).to(DEVICE)
    explainer = shap.GradientExplainer(wrap, bg_seq.to(DEVICE).reshape(bg_n, -1))
    sv = explainer.shap_values(ex_seq.to(DEVICE).reshape(ex_n, -1))
    sv = np.array(sv).reshape(ex_n, WINDOW, n_features)
    mean_abs = np.abs(sv).mean(axis=(0, 1))
    order = np.argsort(mean_abs)[-15:]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([feat_names[i] for i in order], mean_abs[order], color="#3b6ea5")
    ax.set_title("{name}: mean |SHAP| aggregated over time")
    ax.set_xlabel("mean |SHAP|")
    save_fig(fig, PATHS["figures"] / "0X_{name}_shap_bar.png")
    plt.show()
    print("SHAP GradientExplainer succeeded on a small sample.")
except Exception as e:
    print("SHAP GradientExplainer not used (unstable or failed). Fallback = permutation importance.")
    print("reason:", type(e).__name__, str(e)[:300])
'''


# --------------- 04 LSTM ---------------
cells04 = [
("md", """
# 04 — LSTM sequence model

**Purpose.** Train a 2-layer LSTM on 24-hour windows and compare it with the XGBoost baseline.

**Inputs.** Sequence arrays from notebook 02 (`dl_train`, `val`, `test`).

**Outputs.** `models/lstm.pt`, `results/predictions_lstm.csv`, `results/tables/metrics_lstm.csv`.
"""),
("code", boot("LSTMClassifier")),
("md", """
## Class imbalance

`BCEWithLogitsLoss(pos_weight=n_neg/n_pos)` is computed from the **training windows we actually optimize on**. A `pos_weight` of 8 means a missed sepsis hour costs 8× a false alarm in the loss. Validation/test distributions are not re-balanced.
"""),
("md", "## Model-specific EDA: sequences, onset-aligned vitals, padding"),
("code", r'''
# How long are stays, how much of a 24-hour window is padding, and how do vitals look as sepsis approaches?
meta = np.load(PATHS["sequences"] / "train_meta.npz")
starts = meta["patient_start"]
# length of each stay = how many rows share that start
_, counts = np.unique(starts, return_counts=True)
fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(counts, bins=40, color="#3b6ea5", edgecolor="white")
ax.set_title("Observed hours per patient (train)")
ax.set_xlabel("Hours")
ax.set_ylabel("Patients")
save_fig(fig, PATHS["figures"] / "04_hours_per_patient.png")
plt.show()

# Padding fraction: patients shorter than WINDOW contribute zeros on the left.
pad_frac = np.maximum(WINDOW - counts, 0) / WINDOW
print("mean fraction of a 24h window that would be padded at the *last* hour of the stay:", pad_frac.mean())
print("patients shorter than 24h:", (counts < WINDOW).mean())

# Average trajectories aligned to onset (hours before first label=1).
raw = pd.read_csv(PATHS["raw"] / "dataset.csv")
schema = json.load(open(PATHS["processed"] / "schema.json"))
pid, hour, target = schema["patient_id"], schema["hour"], schema["target"]
raw = raw.dropna(subset=[pid, target])
raw[pid] = raw[pid].astype(int)
vitals = [c for c in ["HR", "Temp", "Resp", "MAP", "O2Sat"] if c in raw.columns]
ever = raw.groupby(pid)[target].max()
septic = ever[ever == 1].index
# Use a sample of septic patients for a readable average.
rng = np.random.default_rng(SEED)
pick = rng.choice(septic, size=min(250, len(septic)), replace=False)
aligned = {v: [] for v in vitals}
for i in pick:
    g = raw[raw[pid]==i].sort_values(hour)
    if (g[target]==1).any():
        onset = g.loc[g[target]==1, hour].iloc[0]
        g = g.copy()
        g["t"] = g[hour] - onset
        sub = g[(g["t"] >= -12) & (g["t"] <= 0)]
        for v in vitals:
            aligned[v].append(sub.set_index("t")[v])
fig, ax = plt.subplots(figsize=(8, 4))
t_index = np.arange(-12, 1)
for v in vitals:
    mat = pd.concat(aligned[v], axis=1)
    ax.plot(mat.mean(axis=1).reindex(t_index), label=v)
ax.axvline(0, color="red", linestyle="--", label="onset")
ax.set_title("Mean vitals aligned to sepsis onset (sample of septic patients)")
ax.set_xlabel("Hours before onset")
ax.set_ylabel("Raw value")
ax.legend()
save_fig(fig, PATHS["figures"] / "04_onset_aligned_vitals.png")
plt.show()
'''),
("md", "**What we learned.** Many stays are shorter than 24 hours, so the mask is essential. Vitals often drift in the hours before the label flips — that is the signal the LSTM is hoping to use."),
("code", load_data_cell()),
("md", "## Architecture: 2-layer LSTM → dropout → linear logit"),
("code", r'''
# hidden=64 keeps CPU training reasonable. Each layer comment lives in src/torch_models.py as well.
model = LSTMClassifier(n_features=n_features, hidden=64, dropout=0.3)
print(model)
print("parameters", count_parameters(model))
'''),
("md", "## Train with Adam, pos_weight, grad clip, early stopping"),
("code", r'''
# Early stopping watches validation PR-AUC (not accuracy). Best weights are restored at the end.
history = train_sequence_model(
    model, train_ds, val_mon_ds, DEVICE,
    pos_weight=pos_weight, epochs=10, patience=3, lr=1e-3, batch_size=128,
    ckpt_path=PATHS["models"] / "lstm_best.pt",
)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(history["train_loss"], label="train")
axes[0].plot(history["val_loss"], label="val")
axes[0].set_title("LSTM loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("BCEWithLogits")
axes[0].legend()
axes[1].plot(history["val_pr_auc"], label="val PR-AUC", color="#c0392b")
axes[1].set_title("LSTM validation PR-AUC")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("PR-AUC")
axes[1].legend()
save_fig(fig, PATHS["figures"] / "04_lstm_curves.png")
plt.show()
'''),
("md", "## Threshold, test evaluation, save"),
("code", save_eval_cell("lstm").replace("0X_", "04_")),
("md", "## Explainability"),
("code", xai_cell("lstm").replace("0X_", "04_")),
("md", "### Notebook 04 summary"),
("code", r'''
print(pd.read_csv(PATHS["tables"] / "metrics_lstm.csv").T)
print("Done with notebook 04.")
'''),
]

# --------------- 05 TCN ---------------
cells05 = [
("md", """
# 05 — Temporal Convolutional Network (TCN)

**Purpose.** Train a causal dilated TCN on the same 24-hour windows as the LSTM.

**Inputs.** Sequence arrays from notebook 02.

**Outputs.** `models/tcn.pt`, `results/predictions_tcn.csv`, `results/tables/metrics_tcn.csv`.
"""),
("code", boot("TCNClassifier")),
("md", "## Model-specific EDA: receptive field and causality"),
("code", r'''
# A dilated causal stack can look far into the past without using the future.
# Receptive field for two kernel-3 convs per block: 1 + 2*(k-1)*sum(dilations).
from src.torch_models import TCNClassifier as _T
probe = _T(n_features=8, channels=16, kernel=3, dilations=[1, 2, 4, 8], dropout=0.0)
rf = probe.receptive_field()
WINDOW = json.load(open(PATHS["processed"] / "sequence_config.json"))["window"]
print("kernel=3, dilations=1,2,4,8, two convs/block")
print("receptive field =", rf, "window =", WINDOW)
print("covers window?", rf >= WINDOW)

# Simple table of how far each block can see.
rows = []
acc = 1
for i, d in enumerate([1, 2, 4, 8], start=1):
    acc += 2 * (3 - 1) * d
    rows.append({"block": i, "dilation": d, "receptive_field_after_block": acc})
print(pd.DataFrame(rows).to_string(index=False))

fig, ax = plt.subplots(figsize=(7, 3))
ax.plot([r["block"] for r in rows], [r["receptive_field_after_block"] for r in rows], marker="o", label="receptive field")
ax.axhline(WINDOW, color="red", linestyle="--", label=f"window={WINDOW}")
ax.set_title("TCN receptive field vs window length")
ax.set_xlabel("Residual block")
ax.set_ylabel("Hours of lookback")
ax.legend()
save_fig(fig, PATHS["figures"] / "05_tcn_receptive_field.png")
plt.show()

# Causality check: changing the future (t+1 ... T) must not change the output at time t.
# We compare last-step logits after scrambling only the *last* input vs only an earlier input.
torch.manual_seed(SEED)
x = torch.randn(4, WINDOW, 8)
mask = torch.ones(4, WINDOW)
probe.eval()
base = probe(x, mask)
x_future = x.clone()
# For last-token readout the "future" relative to the last hour does not exist inside the window.
# Check internal causality: output of the TCN *sequence* at time t should ignore t+1.
# We read the conv stack before the head by comparing two clones that differ only at the last timestep
# when we evaluate a shortened mask (as if we were at t = T-2).
mask_t = mask.clone()
mask_t[:, -1] = 0  # pretend the last hour is padding / unknown
y1 = probe(x, mask_t)
x2 = x.clone()
x2[:, -1, :] = 99.0  # junk in the "future" hour
y2 = probe(x2, mask_t)
print("max |Δlogit| when the padded future hour is scrambled:", float((y1-y2).abs().max()))
print("Causality holds if this number is ~0 (the masked last step is not used).")
'''),
("md", "**What we learned.** The stacked dilations see at least 24 hours. Zeroing/masking the last step and then scrambling it should not change the prediction — that is the causality check."),
("code", load_data_cell()),
("md", "## Architecture"),
("code", r'''
# Four residual blocks, dilations 1,2,4,8, 64 channels, then last valid hour -> linear logit.
model = TCNClassifier(n_features=n_features, channels=64, kernel=3, dilations=[1, 2, 4, 8], dropout=0.3)
print(model)
print("parameters", count_parameters(model), "receptive field", model.receptive_field())
'''),
("md", "## Train"),
("code", r'''
history = train_sequence_model(
    model, train_ds, val_mon_ds, DEVICE,
    pos_weight=pos_weight, epochs=10, patience=3, lr=1e-3, batch_size=128,
    ckpt_path=PATHS["models"] / "tcn_best.pt",
)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(history["train_loss"], label="train")
axes[0].plot(history["val_loss"], label="val")
axes[0].set_title("TCN loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("BCEWithLogits")
axes[0].legend()
axes[1].plot(history["val_pr_auc"], label="val PR-AUC", color="#c0392b")
axes[1].set_title("TCN validation PR-AUC")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("PR-AUC")
axes[1].legend()
save_fig(fig, PATHS["figures"] / "05_tcn_curves.png")
plt.show()
'''),
("md", "## Threshold, test evaluation, save"),
("code", save_eval_cell("tcn").replace("0X_", "05_")),
("md", "## Explainability"),
("code", xai_cell("tcn").replace("0X_", "05_")),
("md", "### Notebook 05 summary"),
("code", r'''
print(pd.read_csv(PATHS["tables"] / "metrics_tcn.csv").T)
print("Done with notebook 05.")
'''),
]

# --------------- 06 Transformer ---------------
cells06 = [
("md", """
# 06 — Transformer encoder

**Purpose.** Train a small causal Transformer on the same 24-hour windows.

**Inputs.** Sequence arrays from notebook 02.

**Outputs.** `models/transformer.pt`, `results/predictions_transformer.csv`, `results/tables/metrics_transformer.csv`.
"""),
("code", boot("TransformerClassifier")),
("md", "## Model-specific EDA: positional encoding and attention cost"),
("code", r'''
# Transformers do not know order unless we add a position signal. Attention cost grows with T^2.
WINDOW = json.load(open(PATHS["processed"] / "sequence_config.json"))["window"]
d_model = 64
n_heads = 4
print("window T =", WINDOW, "d_model =", d_model, "heads =", n_heads)
print("self-attention cost per layer is O(T^2 * d) =", WINDOW * WINDOW * d_model)
print("At T=24 this is cheap; at T=512 it would dominate. That is why we keep a 24-hour window.")

# Show the sine/cosine clocks for a few dimensions.
pos = np.arange(WINDOW)[:, None]
div = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
pe = np.zeros((WINDOW, d_model))
pe[:, 0::2] = np.sin(pos * div)
pe[:, 1::2] = np.cos(pos * div)
fig, ax = plt.subplots(figsize=(8, 4))
for dim in [0, 1, 4, 16]:
    ax.plot(np.arange(WINDOW), pe[:, dim], label=f"dim {dim}")
ax.set_title("Sinusoidal positional encoding (first 24 hours)")
ax.set_xlabel("Time step in the window")
ax.set_ylabel("Encoding value")
ax.legend()
save_fig(fig, PATHS["figures"] / "06_positional_encoding.png")
plt.show()
'''),
("md", "**What we learned.** Positional encodings give each hour a unique pattern so attention can tell '3 hours ago' from '20 hours ago'. A causal mask then blocks looking into the future inside the window."),
("code", load_data_cell()),
("md", "## Architecture"),
("code", r'''
# input projection -> sinusoidal PE -> 2 encoder layers (4 heads, d_model=64) with causal + padding masks -> last token.
model = TransformerClassifier(n_features=n_features, d_model=64, n_heads=4, n_layers=2, dropout=0.3, max_len=WINDOW)
print(model)
print("parameters", count_parameters(model))
'''),
("md", "## Train"),
("code", r'''
history = train_sequence_model(
    model, train_ds, val_mon_ds, DEVICE,
    pos_weight=pos_weight, epochs=10, patience=3, lr=1e-3, batch_size=128,
    ckpt_path=PATHS["models"] / "transformer_best.pt",
)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(history["train_loss"], label="train")
axes[0].plot(history["val_loss"], label="val")
axes[0].set_title("Transformer loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("BCEWithLogits")
axes[0].legend()
axes[1].plot(history["val_pr_auc"], label="val PR-AUC", color="#c0392b")
axes[1].set_title("Transformer validation PR-AUC")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("PR-AUC")
axes[1].legend()
save_fig(fig, PATHS["figures"] / "06_transformer_curves.png")
plt.show()
'''),
("md", "## Threshold, test evaluation, save"),
("code", save_eval_cell("transformer").replace("0X_", "06_")),
("md", "## Attention heatmap and feature importance"),
("code", r'''
# Attention weights for one septic test window, plus the same permutation fallback as the other deep models.
from torch.utils.data import DataLoader

# Find a sepsis-positive test index with enough history if possible.
pos_idx = np.where(test["y"] == 1)[0]
show_i = int(pos_idx[0]) if len(pos_idx) else 0
one = WindowSequenceDataset(test["X"], test["y"], test["patient_start"], test["patient_id"], test["hour"], WINDOW, np.array([show_i]))
seq, mask, y, row = one[0]
seq_b, mask_b = seq.unsqueeze(0).to(DEVICE), mask.unsqueeze(0).to(DEVICE)

# Recreate attention from the first encoder layer by a manual capture.
attn_w = []
def hook(mod, inp, out):
    # TransformerEncoderLayer does not return weights; use a small manual attention on projected tokens instead.
    pass

model.eval()
with torch.no_grad():
    h = model.pos(model.input_proj(seq_b))
    # Grab QK^T from layer 0 self attention
    layer = model.encoder.layers[0]
    x = h
    # PyTorch MHA: use the module directly
    attn_mod = layer.self_attn
    t = x.size(1)
    causal = torch.triu(torch.ones(t, t, dtype=torch.bool, device=DEVICE), 1)
    pad = (mask_b == 0)
    _, w = attn_mod(x, x, x, attn_mask=causal, need_weights=True, average_attn_weights=True)
    # w: (batch, T, T)
    W = w[0].cpu().numpy()

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(W, cmap="viridis", aspect="auto")
ax.set_title(f"Transformer attention (avg heads) — test row {show_i}, y={int(y.item())}")
ax.set_xlabel("Key time (hours in window)")
ax.set_ylabel("Query time")
fig.colorbar(im, ax=ax, fraction=0.046, label="weight")
save_fig(fig, PATHS["figures"] / "06_attention_heatmap.png")
plt.show()
print("patient", int(test["patient_id"][show_i]), "hour", int(test["hour"][show_i]), "prob later from saved file")
'''),
("code", xai_cell("transformer").replace("0X_", "06_")),
("md", "### Notebook 06 summary"),
("code", r'''
print(pd.read_csv(PATHS["tables"] / "metrics_transformer.csv").T)
print("Done with notebook 06.")
'''),
]

write_notebook(ROOT / "notebooks" / "04_model_lstm.ipynb", cells04)
write_notebook(ROOT / "notebooks" / "05_model_tcn.ipynb", cells05)
write_notebook(ROOT / "notebooks" / "06_model_transformer.ipynb", cells06)
