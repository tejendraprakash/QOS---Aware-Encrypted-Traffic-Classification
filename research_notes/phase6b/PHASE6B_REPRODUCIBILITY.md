# Phase 6B Reproducibility & Environment Report

**Project:** QoS-Aware Encrypted Network Traffic Classification for Next-Generation Networks  
**Phase:** 6B — Parallel Multi-Agent Early Traffic Classification  
**Date:** November 2026

---

## 1. Hardware & Execution Environment

All Phase 6B experiments were executed in a fully controlled local computing environment:
- **Operating System:** Windows 11 Pro 64-bit (x86_64)
- **Processor:** 12 Logical Cores (Intel / AMD Multi-Core Architecture)
- **System Memory:** 16 GB Total RAM (> 7 GB Available during training)
- **Acceleration:** PyTorch CPU execution (`torch.cuda.is_available() == False`)
- **Storage:** Local NVMe SSD

---

## 2. Software Stack & Exact Package Versions

To ensure exact bitwise reproducibility across independent research replications:
- **Python Runtime:** `3.13.5 (AMD64)`
- **scikit-learn:** `1.5.2`
- **xgboost:** `3.2.0`
- **PyTorch:** `2.6.0+cpu`
- **pandas:** `2.2.3`
- **numpy:** `2.1.3`

---

## 3. Deterministic Seed Controls

All stochastic processes were seeded deterministically prior to data partitioning, sample weighting, model initialization, and batch shuffling:
- `GLOBAL_SEED = 42`
- `random.seed(42)`
- `np.random.seed(42)`
- `torch.manual_seed(42)`
- `torch.cuda.manual_seed_all(42)`
- **Track B Cross-Validation Splitting Seed:** `random_state = 2` (5-fold StratifiedGroupKFold, matching Phase 5 and Phase 6A exactly).

---

## 4. Model Architectures & Hyperparameters

### 4.1 Random Forest
```python
RandomForestClassifier(
    n_estimators=100,
    criterion="gini",
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
```

### 4.2 XGBoost
```python
XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softprob",
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1
)
# sample_weight = training-fold inverse class frequencies
```

### 4.3 Packet MLP (EarlyPMLP)
- **Layers:** `Linear(in_features, 128) -> BatchNorm1d -> ReLU -> Dropout(0.3) -> Linear(128, 64) -> BatchNorm1d -> ReLU -> Dropout(0.3) -> Linear(64, 32) -> BatchNorm1d -> ReLU -> Dropout(0.2) -> Linear(32, 3)`
  - `in_features` $\in \{6, 12, 18, 24\}$ depending on horizon.
- **Optimizer:** `Adam(lr=0.001, weight_decay=1e-4)`
- **Scheduler:** `ReduceLROnPlateau(mode='min', factor=0.5, patience=5)`
- **Loss:** `CrossEntropyLoss(weight=train_class_weights_tensor)`
- **Batch Size:** 64
- **Max Epochs:** 100
- **Early Stopping:** Patience 15 epochs on validation loss.

---

## 5. Artifact Directory Replication Checklist

To replicate all 60 Phase 6B runs from scratch, execute:
```powershell
python scripts/run_phase6b_experiments.py
```
This produces all outputs in:
- [output/phase6b/early_5/](file:///c:/Users/saipr/OneDrive/Desktop/EncryptedTrafficResearch/output/phase6b/early_5/)
- [output/phase6b/early_10/](file:///c:/Users/saipr/OneDrive/Desktop/EncryptedTrafficResearch/output/phase6b/early_10/)
- [output/phase6b/early_15/](file:///c:/Users/saipr/OneDrive/Desktop/EncryptedTrafficResearch/output/phase6b/early_15/)
- [output/phase6b/early_20/](file:///c:/Users/saipr/OneDrive/Desktop/EncryptedTrafficResearch/output/phase6b/early_20/)
- [output/phase6b/horizon_comparison.csv](file:///c:/Users/saipr/OneDrive/Desktop/EncryptedTrafficResearch/output/phase6b/horizon_comparison.csv)
