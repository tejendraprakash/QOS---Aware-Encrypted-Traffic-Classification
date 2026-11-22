# Phase 6A Reproducibility & Environment Report

**Project:** QoS-Aware Encrypted Network Traffic Classification for Next-Generation Networks  
**Phase:** 6A — Comprehensive Full-Flow Machine Learning Baseline Experiment  
**Date:** November 2026

---

## 1. Hardware & Execution Environment

All Phase 6A experiments were executed in a fully controlled local computing environment:
- **Operating System:** Windows 11 Pro 64-bit (x86_64)
- **Processor:** 12 Logical Cores (Intel / AMD Multi-Core Architecture)
- **System Memory:** 16 GB Total RAM (> 7 GB Available during training)
- **Acceleration:** PyTorch CPU execution (`torch.cuda.is_available() == False`)
- **Storage:** Local NVMe SSD (Zero read/write I/O bottlenecks)

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

### Cross-Validation Splitting Seeds:
- **Track A (3-Fold StratifiedGroupKFold):** `random_state = 7` (verified to guarantee 1 email group per fold with 100% class coverage).
- **Track B (5-Fold StratifiedGroupKFold):** `random_state = 2` (verified to guarantee 100% class coverage with zero group leakage).

---

## 4. Model Hyperparameters for Re-execution

### 4.1 Random Forest
```python
RandomForestClassifier(
    n_estimators=100,
    criterion="gini",
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
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

### 4.3 Packet MLP (P-MLP)
- **Layers:** `Linear(77, 128) -> BatchNorm1d -> ReLU -> Dropout(0.3) -> Linear(128, 64) -> BatchNorm1d -> ReLU -> Dropout(0.3) -> Linear(64, 32) -> BatchNorm1d -> ReLU -> Dropout(0.2) -> Linear(32, C)`
- **Optimizer:** `Adam(lr=0.001, weight_decay=1e-4)`
- **Scheduler:** `ReduceLROnPlateau(mode='min', factor=0.5, patience=5)`
- **Loss:** `CrossEntropyLoss(weight=train_class_weights_tensor)`
- **Batch Size:** 64
- **Max Epochs:** 100
- **Early Stopping Patience:** 15 epochs on validation loss

---

## 5. Artifact Directory Replication Checklist

To replicate the entire Phase 6A experiment from scratch, execute:
```powershell
python scripts/run_phase6a_experiments.py
```
This produces all outputs in [output/phase6/track_a/](file:///c:/Users/saipr/OneDrive/Desktop/EncryptedTrafficResearch/output/phase6/track_a/) and [output/phase6/track_b/](file:///c:/Users/saipr/OneDrive/Desktop/EncryptedTrafficResearch/output/phase6/track_b/) identically.
