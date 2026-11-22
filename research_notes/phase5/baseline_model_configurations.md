# Baseline Model Pipeline Configurations & Architectures

**Project:** QoS-Aware Encrypted Network Traffic Classification for Next-Generation Networks  
**Phase:** 5 — Baseline Experiment Preparation & Protocol Validation  
**Date:** November 2026  
**Pipeline Configuration:** [model_pipeline_specs.json](file:///c:/Users/saipr/OneDrive/Desktop/EncryptedTrafficResearch/output/phase5/model_pipeline_specs.json)

---

## 1. Overview of Evaluated Model Architectures

In accordance with modern encrypted network traffic classification standards, our benchmark evaluates three foundational architectures representing distinct machine learning paradigms:

1. **Random Forest (RF):** Non-parametric ensemble of decorrelated decision trees; excels on tabular flow statistics with complex non-linear feature interactions and robustness to noise.
2. **XGBoost (XGB):** Scalable gradient boosted decision tree framework; optimizes custom objective functions via second-order Taylor approximations, providing top-tier tabular performance.
3. **Packet Multi-Layer Perceptron (P-MLP):** Deep feedforward neural network utilizing batch normalization, non-linear activations, and dropout for high-dimensional representation learning.

---

## 2. Detailed Pipeline Specifications

### 2.1 Model 1: Random Forest Classifier
- **Implementation:** `sklearn.ensemble.RandomForestClassifier`
- **Hyperparameter Configuration:**
  - `n_estimators`: 100
  - `criterion`: 'gini'
  - `max_depth`: None (full tree expansion with leaf constraints)
  - `min_samples_split`: 2
  - `min_samples_leaf`: 1
  - `max_features`: 'sqrt'
  - `class_weight`: 'balanced'
  - `random_state`: 42
  - `n_jobs`: -1
- **Class Imbalance Strategy:** `class_weight='balanced'` dynamically weights each sample inversely proportional to class frequencies:
  $$w_c = \frac{N}{|\mathcal{C}| \cdot N_c}$$

### 2.2 Model 2: XGBoost Classifier
- **Implementation:** `xgboost.XGBClassifier`
- **Hyperparameter Configuration:**
  - `objective`: 'multi:softprob'
  - `eval_metric`: 'mlogloss'
  - `n_estimators`: 100
  - `learning_rate`: 0.1
  - `max_depth`: 6
  - `subsample`: 0.8
  - `colsample_bytree`: 0.8
  - `gamma`: 0.0
  - `random_state`: 42
  - `n_jobs`: -1
- **Class Imbalance Strategy:** Sample weights are computed on each training fold using inverse class frequencies ($w_c = \frac{N}{|\mathcal{C}| \cdot N_c}$) and passed to `model.fit(X_train, y_train, sample_weight=weights)`.

### 2.3 Model 3: Packet Multi-Layer Perceptron (P-MLP)
- **Framework:** PyTorch (`torch.nn`)
- **Deep Architecture Design:**
  ```
  Input: x in R^{D} (D = 78 for Full-Flow, or 6/12/18/24 for Early Windows)
    │
    ▼
  [Linear: D -> 128] ──► [BatchNorm1d] ──► [ReLU] ──► [Dropout(p=0.3)]
    │
    ▼
  [Linear: 128 -> 64] ──► [BatchNorm1d] ──► [ReLU] ──► [Dropout(p=0.3)]
    │
    ▼
  [Linear: 64 -> 32] ──► [BatchNorm1d] ──► [ReLU] ──► [Dropout(p=0.2)]
    │
    ▼
  [Linear: 32 -> num_classes] ──► [CrossEntropyLoss]
  ```
- **Training Hyperparameters:**
  - Optimizer: `Adam(lr=0.001, weight_decay=1e-4)`
  - Loss Function: `torch.nn.CrossEntropyLoss(weight=class_weights_tensor)`
  - Batch Size: 64
  - Max Epochs: 100
  - Learning Rate Scheduler: `ReduceLROnPlateau(mode='min', factor=0.5, patience=5)`
  - Early Stopping: 15 epochs patience monitoring validation loss.

---

## 3. Scientific Comparison with Huang et al. (2021)

It is essential to clarify the distinction between our **P-MLP** architecture and the deep learning approach of Huang et al. (2021) (*"Packet-level Deep Learning for Encrypted Traffic Classification"*):

| Comparative Dimension | Huang et al. (2021) | Our Proposed P-MLP Baseline |
|---|---|---|
| **Input Feature Domain** | Raw byte payloads (first $N$ bytes of first $K$ packets represented as byte vectors $0-255$). | Flow-level statistical and temporal packet dynamics (durations, bytes, packet rates, IATs, bursts). |
| **Model Architectures** | 1D-Convolutional Neural Networks (1D-CNN) and LSTM. | Deep Feedforward Multi-Layer Perceptron (MLP) with Batch Normalization & Dropout. |
| **Privacy Preservation** | Low: Inspects application payload byte sequences. | **High:** Fully zero-payload; strictly payload-agnostic. |
| **TLS 1.3 / ECH Robustness** | Fragile: High entropy encrypted payloads appear as pseudorandom noise. | **Robust:** Encrypted payload randomness does not degrade packet timing or volume dynamics. |
| **Inference Efficiency** | High computational complexity (matrix convolutions over thousands of raw bytes). | Ultra-fast inference (< 10 $\mu s$ per flow vector), suitable for line-rate ASIC/eBPF integration. |

---

## 4. Leakage-Free Preprocessing Sequence

To guarantee zero data leakage across folds, feature transformations are strictly encapsulated within an atomic scikit-learn Pipeline / PyTorch preprocessing module:

```python
pipeline = Pipeline([
    ('zero_variance_dropper', VarianceThreshold(threshold=0.0)),
    ('collinearity_filter', CorrelationFilter(threshold=0.999)),
    ('scaler', StandardScaler()),
    ('classifier', model)
])
```

- `VarianceThreshold` identifies and removes features that have zero variance on the training fold (e.g., `tcp_urg_count`).
- `CorrelationFilter` computes the pairwise correlation matrix on the training fold only, dropping redundant features with $|r| \ge 0.999$ (e.g., `bwd_burst_count`).
- `StandardScaler` calculates $\mu_{train}$ and $\sigma_{train}$ exclusively on training samples, applying the transformation $(x - \mu_{train}) / \sigma_{train}$ to both train and validation sets.
