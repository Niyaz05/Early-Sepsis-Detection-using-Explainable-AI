# Model Architecture, Hyperparameters, Training & Evaluation Documentation

This document provides a complete technical breakdown of the four sepsis early-warning models built in this project—covering exact dataset features, parameter counts, hyperparameters, training/validation dynamics, test-set evaluation, and SHAP explainability status.

---

## 1. Data Points & Feature Representations Used

From `data/raw/dataset.csv` (**546,122** valid hourly rows across **14,057** ICU patients after dropping 1 truncated trailing row, `Unnamed: 0`, and 100%-missing `EtCO2`), all models are evaluated on the exact same **patient-level stratified split** (70% train / 15% validation / 15% test; 0 overlapping patients):

| Split | Patients | Total Hourly Rows | Sepsis-Positive Hours | Row Prevalence |
|---|---:|---:|---:|---:|
| **Train (Full — XGBoost)** | 9,839 | 382,661 | 8,293 | 2.17% |
| **DL Train Subsample (LSTM/TCN/Transformer)** | 4,000 | 155,907 (30,159 windows after 8:1 neg downsampling) | 3,351 (11.11% in sampled train) | 2.15% (raw) |
| **Validation (Full)** | 2,109 | 82,478 (12,481 windows in per-epoch monitor) | 1,783 | 2.16% |
| **Test (Held-Out)** | 2,109 | 80,983 | 1,774 (186 sepsis patients) | 2.19% |

### A. Tabular Representation (Used by **XGBoost**) — **180 Features**
Stored in `data/processed/tabular/{train,val,test}.parquet`:
1. **Current Clinical Vitals (7):** `HR`, `O2Sat`, `Temp`, `SBP`, `MAP`, `DBP`, `Resp` (forward-filled within patient, remaining NaNs filled with train median, standardized on train).
2. **Current Lab Values (26):** `BaseExcess`, `HCO3`, `FiO2`, `pH`, `PaCO2`, `SaO2`, `AST`, `BUN`, `Alkalinephos`, `Calcium`, `Chloride`, `Creatinine`, `Bilirubin_direct`, `Glucose`, `Lactate`, `Magnesium`, `Phosphate`, `Potassium`, `Bilirubin_total`, `TroponinI`, `Hct`, `Hgb`, `PTT`, `WBC`, `Fibrinogen`, `Platelets`.
3. **Demographics & ICU Context (5):** `ICULOS`, `HospAdmTime`, `Age` (standardized), `Gender`, `Unit1`, `Unit2` (binary kept unscaled; total 6 demographic/stay columns including `ICULOS`).
4. **Measurement Indicators (`_measured`, 35 binary features):** Flags indicating whether each of the 7 vitals, 26 labs, `Unit1`, or `Unit2` was actually measured at that hour (capturing clinical ordering intent).
5. **Hours Since Last Measurement (`_hrs_since`, 28 features):** Time elapsed since the last recorded draw for all 26 labs + `Temp` and `DBP` (capped at `48.0` if never measured).
6. **6-Hour Rolling Statistics (`_roll6_{mean,std,min,max}`, 52 features):** Computed over the last 6 hours per patient for 13 key variables (`HR`, `O2Sat`, `Temp`, `SBP`, `MAP`, `DBP`, `Resp`, `Lactate`, `WBC`, `Creatinine`, `Glucose`, `BUN`, `Platelets`).
7. **Temporal Deltas (`_d1`, `_d6`, 26 features):** 1-hour and 6-hour value changes for the same 13 key variables.

### B. Sequence Representation (Used by **LSTM, TCN, Transformer**) — **72 Features × 24-Hour Window**
Stored in `data/processed/sequences/{dl_train,val,test}_X.npy` and accessed via `WindowSequenceDataset` (`src/utils.py`):
- **Shape per sample:** `(24, 72)` + binary padding `mask` of shape `(24,)` (left-padded with `0.0` for stays shorter than 24 hours; `mask = 1.0` for real observed hours).
- **Features included (72):**
  - 7 Vitals + 26 Labs (33 continuous)
  - 35 `_measured` binary flags
  - 4 Stay/Demographic features: `ICULOS`, `Age`, `Gender`, `HospAdmTime`
- *Note:* Deep models do **not** receive the 106 manual rolling/delta/`_hrs_since` features because they are designed to learn temporal dynamics directly across the 24-hour window.

---

## 2. Model Architectures, Parameters & Hyperparameters

### Model 1: XGBoost (`notebooks/03_model_xgboost.ipynb`)
- **Model Type:** Gradient Boosted Decision Trees (`xgboost.XGBClassifier`, `tree_method="hist"`).
- **Model Size / Complexity:** **65 boosted trees** (Optuna set `n_estimators=232`, early stopping stopped at `best_iteration=39`, saving 65 rounds), **0.134 MB** on disk.
- **Imbalance Handling:** `scale_pos_weight = 45.143` ($\text{neg}/\text{pos}$ ratio in full train set).
- **Hyperparameter Search:** 25 Optuna TPE trials on a 120,000-row train subsample, scored on validation PR-AUC (`early_stopping_rounds=20`), then refitted on all 382,661 train rows (`early_stopping_rounds=25`).
- **Best Hyperparameters:**
  - `max_depth`: `4`
  - `learning_rate`: `0.1684`
  - `subsample`: `0.8925`
  - `colsample_bytree`: `0.9773`
  - `min_child_weight`: `17`
  - `gamma`: `1.3386`
  - `reg_lambda`: `6.5716`
  - `eval_metric`: `"aucpr"`
- **Validation-Tuned Decision Threshold:** `0.80` (maximizes validation F1).

### Model 2: 2-Layer LSTM (`notebooks/04_model_lstm.ipynb`, `src/torch_models.py`)
- **Architecture:** `LSTM(72 → 64, num_layers=2, dropout=0.3)` using `pack_padded_sequence` (to skip left-padded zeros) → `Dropout(0.3)` → `Linear(64 → 1)`.
- **Trainable Parameters:** **68,673** (**0.265 MB** on disk).
- **Training Hyperparameters:**
  - Optimizer: `Adam(lr=1e-3)`
  - Loss: `BCEWithLogitsLoss(pos_weight=8.0)` (computed on the 8:1 downsampled `dl_train` windows)
  - Batch Size: `128` | Gradient Clipping: `1.0`
  - Max Epochs: `10` | Early Stopping Patience: `3` (monitored on validation PR-AUC)
- **Validation-Tuned Decision Threshold:** `0.90`.

### Model 3: Dilated Causal TCN (`notebooks/05_model_tcn.ipynb`, `src/torch_models.py`)
- **Architecture:** 4 stacked residual `TemporalBlock` layers (`channels=64`, `kernel_size=3`, `dilations=[1, 2, 4, 8]`, weight-normalized causal `Conv1d` + `Chomp1d` + `ReLU` + `Dropout(0.3)` + residual connection) → last valid time step readout → `Linear(64 → 1)`.
- **Receptive Field:** **61 hours** ($1 + 2 \times (3 - 1) \times (1+2+4+8) = 61 \ge 24$-hour window), verified strictly causal ($\max |\Delta \text{logit}| = 0.0$ when masked future steps are scrambled).
- **Trainable Parameters:** **105,601** (**0.413 MB** on disk).
- **Training Hyperparameters:** Same recipe as LSTM (`Adam`, `lr=1e-3`, `pos_weight=8.0`, `batch_size=128`, `clip_grad=1.0`, `epochs=10`, `patience=3`).
- **Validation-Tuned Decision Threshold:** `0.75`.

### Model 4: Causal Transformer Encoder (`notebooks/06_model_transformer.ipynb`, `src/torch_models.py`)
- **Architecture:** `Linear(72 → 64)` → `SinusoidalPositionalEncoding(d_model=64, max_len=24)` → `2 × TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=256, dropout=0.3)` with upper-triangular causal attention mask → last valid token readout → `Dropout(0.3)` → `Linear(64 → 1)`.
- **Trainable Parameters:** **104,705** (**0.415 MB** on disk).
- **Training Hyperparameters:** Same recipe as LSTM & TCN (`Adam`, `lr=1e-3`, `pos_weight=8.0`, `batch_size=128`, `clip_grad=1.0`, `epochs=10`, `patience=3`).
- **Validation-Tuned Decision Threshold:** `0.90`.

---

## 3. Training & Validation Dynamics (Epochs & Early Stopping)

All three PyTorch deep models were trained on CPU (`SEED=42`) using early stopping on validation PR-AUC (`val_mon_ds`: 12,481 windows with 6:1 negative downsampling). Each model peaked at **Epoch 1** on validation PR-AUC and triggered early stopping at **Epoch 4** (patience = 3) as validation loss began rising due to overfitting on the minority class:

| Model | Epoch 1 (Train Loss / Val Loss / Val PR-AUC) | Epoch 2 | Epoch 3 | Epoch 4 (Stopped) | Best Val PR-AUC (Monitor) | Full Val PR-AUC (82,478 rows) | Train Time | Infer Time (80,983 rows) |
|---|---|---|---|---|---|---|---|---|
| **XGBoost** | *41.8s Optuna search; 39 boosting rounds* | — | — | — | — | **0.1280** | **2.39 s** | **0.034 s** |
| **LSTM** | 0.9845 / 1.4308 / **0.3732** | 0.7747 / 1.8128 / 0.3258 | 0.6688 / 2.3976 / 0.3270 | 0.6005 / 3.0647 / 0.3001 | 0.3732 (Ep 1) | 0.0888 | 11.93 s | 2.50 s |
| **TCN** | 1.0095 / 1.4927 / **0.3737** | 0.8256 / 2.1585 / 0.3522 | 0.7345 / 2.4917 / 0.3272 | 0.6586 / 3.2136 / 0.3191 | 0.3737 (Ep 1) | 0.0868 | 92.29 s | 29.36 s |
| **Transformer** | 1.0466 / 1.2498 / **0.3930** | 0.9405 / 1.3472 / 0.3758 | 0.8903 / 1.5443 / 0.3710 | 0.8237 / 1.6445 / 0.3078 | 0.3930 (Ep 1) | 0.0923 | 22.08 s | 3.34 s |

*(Note: Full-validation PR-AUC is lower than the downsampled monitor PR-AUC because full validation preserves the natural 2.16% sepsis prevalence.)*

---

## 4. Test-Set Evaluation & What the Metrics Tell Us

Evaluated on the held-out test cohort (**2,109 patients**, **80,983 hourly rows**, **1,774 positive hours**, **186 sepsis patients**):

| Metric | XGBoost | Transformer | TCN | LSTM |
|---|---:|---:|---:|---:|
| **PR-AUC** *(Headline)* | **0.1250** | 0.0848 | 0.0836 | 0.0826 |
| **PR-AUC 95% Bootstrap CI** | **[0.093, 0.178]** | [0.069, 0.114] | [0.067, 0.109] | [0.068, 0.106] |
| **ROC-AUC** | **0.8206** | 0.7569 | 0.7425 | 0.7262 |
| **ROC-AUC 95% Bootstrap CI** | **[0.794, 0.843]** | [0.725, 0.792] | [0.710, 0.776] | [0.692, 0.761] |
| **F1-Score** | **0.2124** | 0.1684 | 0.1705 | 0.1622 |
| **Precision (PPV)** | **16.73%** | 14.91% | 12.11% | 14.28% |
| **Recall (Sensitivity)** | **29.09%** | 19.33% | 28.80% | 18.77% |
| **Specificity** | 96.76% | **97.53%** | 95.32% | 97.48% |
| **Brier Score** *(Lower = better)* | **0.1430** | 0.2096 | 0.1770 | 0.1583 |
| **Confusion Matrix (TP / FP / TN / FN)** | 516 / 2,568 / 76,641 / 1,258 | 343 / 1,957 / 77,252 / 1,431 | 511 / 3,708 / 75,501 / 1,263 | 333 / 1,999 / 77,210 / 1,441 |
| **Sepsis Patients Detected** | **45.70% (85/186)** | 24.19% (45/186) | 39.78% (74/186) | 23.66% (44/186) |
| **Detected ≥ 6h Before Onset** | **30.65%** | 17.74% | 26.88% | 16.13% |
| **Median Lead Time (Detected)** | 19.0 h | **25.0 h** | 24.0 h | 21.5 h |
| **False Alarms / Patient-Day** | 0.78 | **0.59** | 1.12 | 0.61 |

### Which Model Performs Best in Which Case?
1. **Best Overall Discrimination & Calibration (`XGBoost`):**
   - Achieves the highest PR-AUC (**0.1250**, ~5.7× random baseline of 0.0219), highest ROC-AUC (**0.8206** with non-overlapping 95% bootstrap CI `[0.794, 0.843]` vs deep models), best F1 (**0.2124**), and lowest Brier score (**0.1430**).
   - **Why:** In clinical tabular data with sparse labs (~85–99% missing), explicit domain features (6h rolling stats, 1h/6h deltas, `_measured` flags, and `_hrs_since` lab gaps) give gradient-boosted trees an immediate structural advantage over sequence models trained on raw hourly steps.
2. **Best Patient-Level Sensitivity & Early Detection Coverage (`XGBoost`, followed by `TCN`):**
   - XGBoost catches **45.70%** of septic patients (**30.65%** at least 6 hours before label onset).
   - Among deep models, **TCN** has the highest recall (**28.80%**) and patient detection rate (**39.78%**), nearly matching XGBoost's sensitivity, though at the cost of higher alarm fatigue (**1.12** false alarms/patient-day).
3. **Best Alarm Precision / Lowest False-Alarm Burden (`Transformer`):**
   - Among deep models, **Transformer** achieves the best PR-AUC (**0.0848**) and ROC-AUC (**0.7569**), highest specificity (**97.53%**), lowest false-alarm rate (**0.59** alerts/patient-day), and longest median lead time (**25.0 hours**) among detected patients—though it detects fewer total septic patients (24.19%) due to its conservative threshold (`0.90`).
4. **Best Compute Efficiency (`XGBoost`):**
   - Trains in **2.39 seconds** and scores 80,983 rows in **0.034 seconds** (vs. 92.3s train / 29.4s inference for TCN).

---

## 5. SHAP Explainability: Current Status & How It Works

### What Is Implemented Right Now
1. **XGBoost (`03_model_xgboost.ipynb`) — Exact `TreeExplainer` IS fully implemented:**
   - Uses `shap.TreeExplainer(model)` on a 2,000-row test sample.
   - Outputs three saved visualizations in `results/figures/`:
     - `03_shap_beeswarm.png`: Global impact + direction (how high vs. low feature values push sepsis risk up or down).
     - `03_shap_bar.png`: Mean `|SHAP|` feature importance ranking.
     - `03_shap_waterfall.png`: Local per-patient explanation for a true-positive sepsis hour (Patient 373, Hour 141, predicted risk `0.875`).
2. **Deep Learning Models (`04_model_lstm.ipynb`, `05_model_tcn.ipynb`, `06_model_transformer.ipynb`) — `GradientExplainer` attempted, fell back to Permutation + Time-Step Importance:**
   - All three deep notebooks attempt `shap.GradientExplainer` on a small sample (32 background windows, 48 explain windows) using a `FlatWrap` module.
   - During execution, `shap.GradientExplainer` raised `IndexError: too many indices for tensor of dimension 1` because:
     1. The PyTorch models return a **1D tensor** of shape `(batch,)` (due to `.squeeze(-1)`), whereas `shap 0.49.1`'s `GradientExplainer` expects a **2D tensor** `(batch, 1)` when indexing outputs.
     2. Additionally, for `LSTMClassifier`, `pack_padded_sequence` does not support double-backward/gradient hooks cleanly in all PyTorch/SHAP versions.
   - As designed in the fallback logic, all three deep notebooks cleanly switched to:
     - **Feature Permutation Importance** (`permutation_importance_torch` in `src/utils.py`): Shuffles one feature across the batch (preserving time order) on 350 test samples and measures the **PR-AUC drop**. Top features consistently include `ICULOS`, `Age`, `HR`, `Potassium`, `SaO2`, `Platelets`, and lab `_measured` flags.
     - **Time-Step Zero-Out Importance** (`time_step_importance_torch` in `src/utils.py`): Zeros out one hour at a time across the 24h window to measure mean $|\Delta p|$.
     - **Transformer Attention Heatmap** (`06_attention_heatmap.png`): Extracts multi-head self-attention weights over the 24h window for a septic patient.

### Should SHAP Be Fixed for the Deep Models?
- **For this project, it is not strictly necessary** because:
  - Your best-performing model (**XGBoost**) already has **exact, fast TreeExplainer SHAP** plots.
  - Deep models have complete, mathematically sound **Permutation Importance**, **Time-Step Importance**, and **Attention Heatmaps** (and `PermutationExplainer` / PR-AUC drop is often preferred over `GradientExplainer` for sequence models because it directly measures impact on PR-AUC rather than noisy input gradients).
- **If you want `GradientExplainer` to run on TCN and Transformer anyway**, it is a 2-line fix in `FlatWrap.forward()` inside `scripts/gen_dl.py`: return `self.m(...).unsqueeze(-1)` so the output tensor has shape `(batch, 1)` instead of `(batch,)`. (For LSTM, you would also replace `pack_padded_sequence` with masked indexing during SHAP or use `shap.KernelExplainer`.)
