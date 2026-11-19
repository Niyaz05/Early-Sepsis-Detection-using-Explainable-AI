from pathlib import Path
from nbformat_util import write_notebook

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "02_preprocessing_and_splits.ipynb"

BOOT = r'''
# Reproducible setup: seed, versions, project paths. Notebooks live in notebooks/, so root is the parent folder.
from pathlib import Path
import os, sys, json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

PROJECT_ROOT = Path.cwd().parent
os.environ["MPLCONFIGDIR"] = str(PROJECT_ROOT / ".mplconfig")
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import (
    SEED, set_seed, print_library_versions, ensure_dirs,
    identify_schema, print_schema, save_schema,
)

set_seed(SEED)
print("SEED =", SEED)
print_library_versions()
PATHS = ensure_dirs(PROJECT_ROOT)
WINDOW = 24  # hours of history in each deep-learning window
print("WINDOW =", WINDOW)
'''

cells = [
("md", """
# 02 — Preprocessing and patient-level splits

**Purpose.** Clean the table, create a **patient-level** 70/15/15 split (created once, reused everywhere), impute without leaking future or cross-patient information, engineer tabular features, and build 24-hour sequence arrays.

**Inputs.** `data/raw/dataset.csv`

**Outputs.**
- `data/processed/splits/*_patient_ids.npy` (and a deep-learning train subsample)
- `data/processed/tabular/{train,val,test}.parquet`
- `data/processed/sequences/{train,val,test}_X.npy` plus `*_meta.npz`
- scaler / medians / column lists under `data/processed/`
"""),
("code", BOOT),
("md", "## Load data and drop unused columns"),
("code", r'''
# Read the raw file, drop the empty trailing row, then drop columns that cannot help a model.
df = pd.read_csv(PATHS["raw"] / "dataset.csv")
schema = identify_schema(df)
print_schema(schema)
pid, hour, target = schema["patient_id"], schema["hour"], schema["target"]
iculos = schema["iculos"]

n0 = len(df)
df = df.dropna(subset=[pid, target]).copy()
print(f"Dropped {n0 - len(df)} empty/corrupt rows (no patient id or label).")

drop_cols = []
if "Unnamed: 0" in df.columns:
    drop_cols.append("Unnamed: 0")  # exact duplicate of Hour
for c in df.columns:
    if df[c].isna().all() and c not in drop_cols:
        drop_cols.append(c)  # e.g. EtCO2 is 100% missing in this extract
df = df.drop(columns=drop_cols)
print("Dropped columns:", drop_cols)
print("Reason: Unnamed: 0 == Hour; fully-empty columns contain no information.")

df[pid] = df[pid].astype(int)
df = df.sort_values([pid, hour]).reset_index(drop=True)
schema["vitals"] = [c for c in schema["vitals"] if c in df.columns]
schema["labs"] = [c for c in schema["labs"] if c in df.columns]
save_schema(schema, PATHS["processed"] / "schema.json")
'''),
("md", "**What we learned.** After dropping the truncated row and empty columns we have a clean hourly panel indexed by patient and hour."),
("md", "## Patient-level 70 / 15 / 15 split (stratified on ever-sepsis)"),
("code", r'''
# Split *patients*, not rows. Stratify on whether the stay ever has SepsisLabel=1 so each split has a similar sepsis rate.
ever = df.groupby(pid)[target].max().astype(int)
ids = ever.index.to_numpy()
y_pat = ever.to_numpy()
train_ids, rest_ids, y_tr, y_rest = train_test_split(
    ids, y_pat, test_size=0.30, stratify=y_pat, random_state=SEED
)
val_ids, test_ids, y_va, y_te = train_test_split(
    rest_ids, y_rest, test_size=0.50, stratify=y_rest, random_state=SEED
)

def save_ids(name, arr):
    p = PATHS["splits"] / f"{name}_patient_ids.npy"
    np.save(p, arr)
    print(name, "patients", len(arr), "sepsis_rate", ever.loc[arr].mean(), "->", p)

save_ids("train", train_ids)
save_ids("val", val_ids)
save_ids("test", test_ids)

# Same deep-learning train subsample for LSTM/TCN/Transformer (full XGBoost train is kept).
dl_n = min(4000, len(train_ids))
train_ids_s, *_rest = train_test_split(
    train_ids, y_tr, train_size=dl_n, stratify=y_tr, random_state=SEED
)
np.save(PATHS["splits"] / "dl_train_patient_ids.npy", train_ids_s)
print("dl_train patients", len(train_ids_s), "sepsis_rate", ever.loc[train_ids_s].mean())

# Sanity: no patient in two splits
s_tr, s_va, s_te = set(train_ids), set(val_ids), set(test_ids)
assert s_tr.isdisjoint(s_va) and s_tr.isdisjoint(s_te) and s_va.isdisjoint(s_te)
print("Sanity check passed: no patient appears in two splits.")
print("overlap train/val", len(s_tr & s_va), "train/test", len(s_tr & s_te), "val/test", len(s_va & s_te))
'''),
("md", "**What we learned.** The split is frozen on disk. Later notebooks must load these IDs rather than splitting again."),
("md", "## Missingness indicators, hours since last lab, forward-fill, train-median"),
("code", r'''
# 1) Binary "was this measured at this hour?" flags — missingness is clinically informative.
# 2) Hours since last draw, computed from the *original* NaNs so we do not use future labs.
# 3) Forward-fill only inside a patient (never across patients, never backward).
# 4) Whatever is still missing gets the TRAIN median (no val/test statistics).

vitals = schema["vitals"]
labs = schema["labs"]
demo = [c for c in schema["demographics"] if c in df.columns]
sparse_for_indicator = [c for c in vitals + labs + ["Unit1", "Unit2"] if c in df.columns]
# Indicators from the raw NaN pattern, before filling.
for c in sparse_for_indicator:
    df[f"{c}_measured"] = df[c].notna().astype(np.int8)

# Hours since last measurement for labs (and for sparse vitals Temp/DBP if present).
hrs_since_cols = []
key_sparse = [c for c in labs + [x for x in ["Temp", "DBP"] if x in df.columns]]
for c in key_sparse:
    meas_hour = np.where(df[c].notna(), df[hour].to_numpy(), np.nan)
    tmp = pd.Series(meas_hour)
    last = tmp.groupby(df[pid]).ffill()
    name = f"{c}_hrs_since"
    df[name] = df[hour] - last
    hrs_since_cols.append(name)

value_cols = [c for c in vitals + labs if c in df.columns]
# Forward-fill within patient only.
df[value_cols] = df.groupby(pid, sort=False)[value_cols].ffill()

train_mask = df[pid].isin(train_ids)
medians = df.loc[train_mask, value_cols].median()
# Never-measured labs / hrs_since still NaN: fill with train median (hrs_since uses a cap).
df[value_cols] = df[value_cols].fillna(medians)
for c in hrs_since_cols:
    # If a lab was never drawn, treat "hours since" as 48 (a long gap, not zero).
    df[c] = df[c].fillna(48.0)

for c in ["Unit1", "Unit2"]:
    if c in df.columns:
        df[c] = df[c].fillna(0.0)

print("NaNs remaining in value cols:", int(df[value_cols].isna().sum().sum()))
print("Added", len(sparse_for_indicator), "measurement flags and", len(hrs_since_cols), "hrs-since columns.")
medians.to_json(PATHS["processed"] / "train_medians.json")
'''),
("md", """
**Class imbalance handling (used later, not here).** We will **not** oversample the test (or val) set. XGBoost gets `scale_pos_weight = n_neg / n_pos` on **train hours**. Deep models get `BCEWithLogitsLoss(pos_weight=n_neg/n_pos)` on **train windows**. That up-weights the rare positive class in the loss without fabricating test rows.
"""),
("md", "## Tabular features: rolling stats, deltas, ICU length of stay"),
("code", r'''
# Rolling 6-hour mean/std/min/max and 1h/6h deltas capture deterioration that a single snapshot misses.
# Computed after imputation so the windows are dense, still grouped by patient so we do not mix stays.
roll_src = [c for c in ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp",
                         "Lactate", "WBC", "Creatinine", "Glucose", "BUN", "Platelets"] if c in df.columns]
g = df.groupby(pid, sort=False)

engineered = []
rolled = g[roll_src]
for stat in ["mean", "std", "min", "max"]:
    part = getattr(rolled.rolling(6, min_periods=1), stat)()
    part = part.reset_index(level=0, drop=True)
    part.columns = [f"{c}_roll6_{stat}" for c in roll_src]
    df = pd.concat([df, part], axis=1)
    engineered += list(part.columns)

# std can be NaN on a 1-row window
for c in [col for col in df.columns if col.endswith("_roll6_std")]:
    df[c] = df[c].fillna(0.0)

for lag, tag in [(1, "d1"), (6, "d6")]:
    delta = g[roll_src].diff(lag)
    delta.columns = [f"{c}_{tag}" for c in roll_src]
    df = pd.concat([df, delta], axis=1)
    df[delta.columns] = df[delta.columns].fillna(0.0)
    engineered += list(delta.columns)

# ICU length of stay is already in ICULOS; keep Hour as well (time in the recorded stay).
print("engineered feature count:", len(engineered))
print("table shape after engineering:", df.shape)
'''),
("md", "**What we learned.** Rolling and delta features add a short history to each hour so a tree model can see trends without an explicit sequence encoder."),
("md", "## Scale on TRAIN only, then save the tabular representation"),
("code", r'''
# Fit a StandardScaler on train hours only, then transform val/test with those same mean/std values.
indicator_cols = [c for c in df.columns if c.endswith("_measured")]
id_like = [pid, hour, target]
continuous = [
    c for c in df.columns
    if c not in id_like + indicator_cols
    and pd.api.types.is_numeric_dtype(df[c])
]
# Gender and unit flags are 0/1 — leave them unscaled with the indicators.
binary_keep = [c for c in ["Gender", "Unit1", "Unit2"] if c in df.columns]
continuous = [c for c in continuous if c not in binary_keep]

scaler = StandardScaler()
scaler.fit(df.loc[train_mask, continuous])
df[continuous] = scaler.transform(df[continuous])
joblib.dump(
    {"scaler": scaler, "continuous": continuous, "indicators": indicator_cols, "binary": binary_keep},
    PATHS["processed"] / "tabular_scaler.joblib",
)

feature_cols = continuous + indicator_cols + binary_keep
# Drop any leftover non-features
feature_cols = [c for c in feature_cols if c in df.columns]
json.dump(feature_cols, open(PATHS["processed"] / "tabular_feature_cols.json", "w"), indent=2)

def slice_split(ids):
    part = df[df[pid].isin(ids)].copy()
    return part

for name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
    part = slice_split(ids)
    keep = [pid, hour, target] + feature_cols
    out = part[keep].rename(columns={pid: "patient_id", hour: "hour", target: "y"})
    path = PATHS["tabular"] / f"{name}.parquet"
    out.to_parquet(path, index=False)
    print(f"{name:5s} rows={len(out):7d}  patients={out.patient_id.nunique():5d}  "
          f"pos_rate={out.y.mean():.4f}  -> {path}")

# Demographics for fairness (unscaled originals from the raw file would be nicer;
# we stored scaled Age. Save raw age/gender from a reload of constants.)
raw = pd.read_csv(PATHS["raw"] / "dataset.csv")
raw = raw.dropna(subset=[pid, target])
raw[pid] = raw[pid].astype(int)
demo_tbl = raw.groupby(pid).agg(Age=("Age", "first"), Gender=("Gender", "first")).reset_index()
demo_tbl = demo_tbl.rename(columns={pid: "patient_id"})
demo_tbl.to_csv(PATHS["processed"] / "patient_demographics.csv", index=False)
print("saved demographics", len(demo_tbl))
'''),
("md", "## Sequence representation: 24-hour sliding windows as memmap-friendly arrays"),
("code", r'''
# Each row is one patient-hour of *current* scaled values + measurement flags (no rolling extras).
# Later, WindowSequenceDataset builds a window ending at that hour and pads the left with zeros + a mask.
seq_value = [c for c in vitals + labs if c in df.columns]
seq_extra = [c for c in [iculos, "Age", "Gender", "HospAdmTime"] if c in df.columns and c is not None]
seq_features = seq_value + indicator_cols + seq_extra
seq_features = [c for c in seq_features if c in df.columns]
json.dump(seq_features, open(PATHS["processed"] / "sequence_feature_cols.json", "w"), indent=2)
json.dump({"window": WINDOW}, open(PATHS["processed"] / "sequence_config.json", "w"))
print("n sequence features:", len(seq_features))
print(seq_features)

def write_sequence_split(name, ids):
    part = df[df[pid].isin(ids)].sort_values([pid, hour]).reset_index(drop=True)
    X = part[seq_features].to_numpy(dtype=np.float32)
    y = part[target].to_numpy(dtype=np.float32)
    patient_id = part[pid].to_numpy()
    hours = part[hour].to_numpy()
    # patient_start[i] = first row of the stay that owns row i (used for padding).
    starts = np.zeros(len(part), dtype=np.int64)
    groups = part.groupby(pid, sort=False).size().to_numpy()
    cursor = 0
    start_vals = []
    for n in groups:
        start_vals.extend([cursor] * int(n))
        cursor += int(n)
    starts[:] = np.array(start_vals, dtype=np.int64)
    np.save(PATHS["sequences"] / f"{name}_X.npy", X)
    np.savez(
        PATHS["sequences"] / f"{name}_meta.npz",
        y=y,
        patient_start=starts,
        patient_id=patient_id.astype(np.int64),
        hour=hours.astype(np.int64),
        feature_names=np.array(seq_features),
    )
    print(f"{name:8s} X={X.shape}  pos_rate={y.mean():.4f}  patients={part[pid].nunique()}")

write_sequence_split("train", train_ids)
write_sequence_split("val", val_ids)
write_sequence_split("test", test_ids)
write_sequence_split("dl_train", train_ids_s)
'''),
("md", "**What we learned.** Sequence tensors are stored per-row, not as a giant (N, 24, F) cube. Windows are built on the fly so RAM stays bounded. Deep models share the same `dl_train` patient subsample."),
("md", "## Class balance printout and final sanity checks"),
("code", r'''
# Confirm shapes, class balance, and that the scaler was not fit on val/test.
print("TABULAR")
for name in ["train", "val", "test"]:
    t = pd.read_parquet(PATHS["tabular"] / f"{name}.parquet")
    print(f"  {name:5s} {t.shape}  patients={t.patient_id.nunique()}  pos={t.y.mean():.4f}")

print("SEQUENCES")
for name in ["train", "val", "test", "dl_train"]:
    X = np.load(PATHS["sequences"] / f"{name}_X.npy", mmap_mode="r")
    meta = np.load(PATHS["sequences"] / f"{name}_meta.npz")
    print(f"  {name:8s} X={X.shape} y_pos={meta['y'].mean():.4f}")

tr = set(np.load(PATHS["splits"] / "train_patient_ids.npy"))
va = set(np.load(PATHS["splits"] / "val_patient_ids.npy"))
te = set(np.load(PATHS["splits"] / "test_patient_ids.npy"))
print("disjoint:", tr.isdisjoint(va) and tr.isdisjoint(te) and va.isdisjoint(te))
print("union covers all?", len(tr|va|te), "expected", df[pid].nunique())
print("WINDOW", WINDOW)
print("Done with notebook 02.")
'''),
("md", "### Notebook 02 complete. Downstream models must load these files and must not re-split."),
]

write_notebook(OUT, cells)
