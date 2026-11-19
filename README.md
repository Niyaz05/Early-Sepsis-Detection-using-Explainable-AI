# ICU Sepsis Early-Warning Machine Learning Project

A production-grade, reproducible clinical machine learning system for the hourly early prediction of sepsis in Intensive Care Unit (ICU) patients.

---

## 1. Project Overview & Clinical Context

Sepsis is a life-threatening organ dysfunction caused by a dysregulated host response to infection. In the ICU, every hour of delayed antimicrobial therapy increases mortality by 7.6%. This project develops an early-warning clinical surveillance system comparing traditional machine learning (**XGBoost**) against modern deep sequence models (**2-layer LSTM**, **Dilated Causal TCN**, and **Causal Transformer Encoder**).

Beyond raw performance rankings, the project investigates:
- **Clinical Lead Time**: Evaluating detection hours ahead of clinical onset.
- **Explainability (XAI)**: Exact TreeExplainer SHAP values, beeswarm visualizations, local waterfall plots, and temporal attention heatmaps.
- **Uncertainty Quantification**: Epistemic uncertainty estimation via Monte Carlo Dropout (30 stochastic passes), correlating risk confidence with clinical error rates.
- **Fairness & Health Equity**: Auditing demographic parity and equal opportunity differences across Gender and Age brackets (<40, 40-65, 65+).

---

## 2. Directory Structure

```
CSET343_project/
├── data/
│   ├── raw/
│   │   └── dataset.csv                      # Raw PhysioNet/Kaggle dataset (546k rows, 14,057 patients)
│   └── processed/
│       ├── schema.json                      # Inferred dataset schema & clinical assumptions
│       ├── train_medians.json               # Train-only medians for unbiased imputation
│       ├── tabular_scaler.joblib            # Train-fitted StandardScaler
│       ├── patient_demographics.csv         # Raw Age and Gender for fairness audits
│       ├── splits/
│       │   ├── train_patient_ids.npy        # 70% patient IDs (stratified)
│       │   ├── val_patient_ids.npy          # 15% patient IDs
│       │   ├── test_patient_ids.npy         # 15% patient IDs
│       │   └── dl_train_patient_ids.npy     # Deep learning patient subset (balanced compute)
│       ├── tabular/                         # Parquet files for XGBoost (raw + rolling/delta features)
│       │   ├── train.parquet
│       │   ├── val.parquet
│       │   └── test.parquet
│       └── sequences/                       # Memmap-friendly arrays for sliding window sequence models
│           ├── train_X.npy & train_meta.npz
│           ├── val_X.npy & val_meta.npz
│           ├── test_X.npy & test_meta.npz
│           └── dl_train_X.npy & dl_train_meta.npz
├── models/
│   ├── xgboost.json                         # XGBoost booster
│   ├── lstm.pt                              # PyTorch LSTM state dict
│   ├── tcn.pt                               # PyTorch TCN state dict
│   └── transformer.pt                       # PyTorch Transformer state dict
├── notebooks/
│   ├── 01_eda_overview.ipynb                # Comprehensive clinical EDA & schema discovery
│   ├── 02_preprocessing_and_splits.ipynb    # Leakage-free patient split, imputation, scaling, feature engineering
│   ├── 03_model_xgboost.ipynb               # Tabular baseline: Optuna tuning, scale_pos_weight, SHAP
│   ├── 04_model_lstm.ipynb                  # 2-layer LSTM sequence model on 24h sliding windows
│   ├── 05_model_tcn.ipynb                   # Dilated causal residual Temporal Convolutional Network
│   ├── 06_model_transformer.ipynb           # Causal Transformer encoder with sinusoidal positional encoding
│   ├── 07_model_comparison.ipynb           # Multi-model evaluation, ROC/PR overlay, calibration, bootstrap CIs
│   └── 08_innovation_xai_uncertainty_fairness.ipynb # Patient trajectories, MC Dropout uncertainty, fairness audit
├── results/
│   ├── figures/                             # High-resolution clinical figures (.png)
│   ├── tables/                              # Metric summaries, bootstrap CIs, fairness audits (.csv)
│   ├── predictions_xgboost.csv              # Test set predictions (patient_id, hour, y_true, y_prob)
│   ├── predictions_lstm.csv
│   ├── predictions_tcn.csv
│   └── predictions_transformer.csv
├── scripts/
│   ├── run_notebook.py                      # Robust in-process notebook runner
│   ├── gen_01.py .. gen_08.py               # Notebook generators
│   └── nbformat_util.py                     # Notebook formatting helper
├── src/
│   ├── __init__.py
│   ├── utils.py                             # Reusable metrics, schema, seed, plots, bootstrap, dataset helpers
│   └── torch_models.py                      # PyTorch LSTM, TCN, and Transformer architectures
└── requirements.txt                         # Exact pinned Python dependencies
```

---

## 3. Notebook Execution Order

The notebooks must be executed sequentially from 01 to 08:

1. **`01_eda_overview.ipynb`**:
   Loads `data/raw/dataset.csv`, identifies columns dynamically (vitals, labs, demographics), profiles missingness and extreme class imbalance, and saves `data/processed/schema.json`.
2. **`02_preprocessing_and_splits.ipynb`**:
   Partitions data by **PATIENT** (70% train / 15% validation / 15% test, stratified on ever-sepsis). Applies intra-patient forward-fill followed by train-median imputation. Adds binary `_measured` indicators. Engineers 6h rolling stats and 1h/6h deltas. Constructs 24-hour sliding sequence representations.
3. **`03_model_xgboost.ipynb`**:
   Trains XGBoost with `scale_pos_weight` and Optuna hyperparameter optimization. Selects optimal decision threshold on validation data. Generates exact TreeExplainer SHAP beeswarm, bar, and waterfall plots.
4. **`04_model_lstm.ipynb`**:
   Trains a 2-layer LSTM with BCE loss (`pos_weight`), gradient clipping, and early stopping. Evaluates temporal feature trajectories aligned to sepsis onset.
5. **`05_model_tcn.ipynb`**:
   Implements a causal dilated residual TCN (kernel 3, dilations 1, 2, 4, 8; receptive field = 61h). Verifies strict temporal causality and evaluates test performance.
6. **`06_model_transformer.ipynb`**:
   Implements a causal Transformer Encoder with sinusoidal positional encoding. Visualizes attention heatmaps for septic patient trajectories.
7. **`07_model_comparison.ipynb`**:
   Without retraining, aggregates predictions from all 4 models. Overlays ROC and PR curves, side-by-side confusion matrices, calibration curves, early detection lead-time boxplots, and 500-iteration patient-level bootstrap 95% confidence intervals.
8. **`08_innovation_xai_uncertainty_fairness.ipynb`**:
   Plots clinical trajectories with early warning lead times (TP, FP, FN, TN). Quantifies epistemic uncertainty via Monte Carlo Dropout (30 passes). Audits algorithmic fairness across Gender and Age brackets.

---

## 4. Benchmark Results Summary Table

Evaluated on the held-out test cohort (2,109 patients, 80,983 hourly observations, prevalence 2.19%):

| Metric | XGBoost (Tabular + Eng.) | Transformer Encoder | TCN (Dilated Causal) | LSTM (2-Layer Recurrent) |
|---|:---:|:---:|:---:|:---:|
| **PR-AUC (Headline Metric)** | **0.1250** | 0.0848 | 0.0836 | 0.0826 |
| **PR-AUC 95% Bootstrap CI** | **[0.093, 0.178]** | [0.069, 0.114] | [0.067, 0.109] | [0.068, 0.106] |
| **ROC-AUC** | **0.8206** | 0.7569 | 0.7425 | 0.7262 |
| **ROC-AUC 95% Bootstrap CI** | **[0.794, 0.843]** | [0.725, 0.792] | [0.710, 0.776] | [0.692, 0.761] |
| **Brier Score (Calibration)** | **0.1430** | 0.2096 | 0.1770 | 0.1583 |
| **F1-Score (Val-Tuned Thr)** | **0.2124** (thr=0.80) | 0.1684 (thr=0.90) | 0.1705 (thr=0.75) | 0.1622 (thr=0.90) |
| **Sensitivity (Recall)** | 29.09% | 19.33% | 28.81% | 18.77% |
| **Specificity** | 96.76% | **97.53%** | 95.32% | 97.48% |
| **Sepsis Patients Alerted** | **45.70%** (85/186) | 24.19% (45/186) | 39.78% (74/186) | 23.66% (44/186) |
| **Alerted ≥ 6h Before Onset** | **30.65%** | 17.74% | 26.88% | 16.13% |
| **Median Lead Time Ahead of Onset** | 19.0 hours | **25.0 hours** | 24.0 hours | 21.5 hours |
| **Alarm Burden (False Alarms / Patient-Day)**| 0.78 | **0.59** | 1.12 | 0.61 |
| **Training Duration** | **2.39 s** | 22.08 s | 92.29 s | 12.19 s |
| **Inference Time (80k rows)** | **0.034 s** | 3.34 s | 29.36 s | 2.50 s |

---

## 5. Key Clinical Insights

1. **Tabular Feature Engineering Trumps Raw Sequence Modeling:**
   XGBoost supplied with 6-hour rolling statistics (mean, std, min, max), 1h/6h deltas, and hours since last lab measurement significantly outperformed deep models trained on raw sliding windows.
2. **Clinical Lead Time & Life-Saving Interventions:**
   All models achieved median lead times between **19.0 and 25.0 hours** prior to the label onset for detected sepsis patients, easily meeting the clinical guideline requirement of ≥ 6 hours for culture collection and antibiotic initiation.
3. **Managing ICU Alarm Fatigue:**
   At the chosen validation thresholds, XGBoost and Transformer models produce fewer than **0.8 false alarms per patient-day**, protecting ICU clinical teams from alarm desensitization.
4. **Epistemic Uncertainty as Clinical Triage:**
   Monte Carlo Dropout demonstrated that predictions in the highest uncertainty quartile exhibit significantly higher error rates, indicating that alerts with wide confidence intervals should prompt manual diagnostic verification rather than automated escalation.
5. **Fairness and Health Equity:**
   Performance audits revealed equitable sensitivity across gender groups (Equal Opportunity Difference = 0.025) and comparable recall across adult and elderly cohorts (<40: 31.8%, 40-65: 28.8%, 65+: 28.7%).

