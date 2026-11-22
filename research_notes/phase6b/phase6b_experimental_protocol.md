# PHASE 6B EXPERIMENTAL PROTOCOL SPECIFICATION: EARLY TRAFFIC CLASSIFICATION BENCHMARK

**Project:** QoS-Aware Encrypted Network Traffic Classification for Next-Generation Networks  
**Document Code:** PROTOCOL-PHASE-6B-V1.0  
**Phase:** 6B — Early-Stage Encrypted Traffic Classification Benchmark  
**Role:** Principal Machine Learning Research Scientist, Computer Networks Researcher & Experimental Methodology Auditor  
**Date:** November 2026  
**Status:** Formally Specified, Audited, and Approved for Execution  
**Target Repository Location:** `research_notes/phase6b/phase6b_experimental_protocol.md`  
**Output Directories:** `output/phase6b/early_5/`, `output/phase6b/early_10/`, `output/phase6b/early_15/`, `output/phase6b/early_20/`, `output/phase6b/consolidated/`  

---

## 1. Executive Summary & Experimental Objectives

### 1.1 Research Context
In modern software-defined networking (SDN), 5G/6G cellular cores, and programmable data planes (e.g., P4 hardware switches), traffic classification cannot wait for flow termination. Full-flow classification (as benchmarked in Phase 6A) establishes theoretical performance ceilings, but full-flow features are unavailable during active session routing. In practice, quality-of-service (QoS) enforcement—including priority queuing, slice allocation, buffer reservation, and dynamic traffic policing—must occur within the first few packets of flow establishment before queuing latency degrades real-time applications.

Phase 6B evaluates the classification performance, algorithmic stability, representation capability, and inference latency of machine learning models operating strictly on **early packet horizons** ($N \in \{5, 10, 15, 20\}$ packets).

### 1.2 Protocol Mandates & Core Objectives
1. **Base Dataset Preservation:** Utilize Track B from Phase 5 (`output/phase5/track_b_flows.csv`, 10,770 flows, 3 QoS classes: Bulk Data, Streaming, Real-Time).
2. **Identical Cross-Validation Splits:** Enforce the identical 5-fold `StratifiedGroupKFold` split mapping (`output/phase5/track_b_folds.csv`) established in Phase 5 and executed in Phase 6A.
3. **Rigorous Model Alignment:** Evaluate the identical model architectures and hyperparameters utilized in Phase 6A:
   - Random Forest (RF, 100 trees, balanced class weighting)
   - XGBoost (XGB, 100 boosted trees, inverse-frequency sample weighting)
   - Packet Multi-Layer Perceptron (P-MLP, 3-layer deep architecture with BatchNorm, Dropout, Adam, and ReduceLROnPlateau)
4. **Horizon Matrix:** Conduct evaluation across 4 progressive packet horizons:
   - **Early-5:** 6 features ($N=5$)
   - **Early-10:** 12 features ($N=10$, cumulative)
   - **Early-15:** 18 features ($N=15$, cumulative)
   - **Early-20:** 24 features ($N=20$, cumulative)
5. **Exact Run Count:** 4 horizons $\times$ 3 model families $\times$ 5 folds = **60 independent model-fold executions**.
6. **Leakage-Free Feature Preprocessing:** Zero-variance feature dropping and feature standardisation (scaling) must be fitted strictly on the training partition ($X_{train}$) of each fold, never touching validation partitions ($X_{val}$).
7. **Comprehensive Metric Telemetry:** Capture Macro F1, Balanced Accuracy, Overall Accuracy, Weighted F1, per-class Precision/Recall/F1, raw & normalized confusion matrices, wall-clock training runtime, and per-flow prediction latency ($\mu s$/flow).
8. **Direct Baseline Comparability:** Directly align results against the Phase 6A full-flow baseline (78 features) to construct the empirical **Accuracy-vs-Packet Horizon Trade-off Curve**.

---

## 2. Network Engineering & QoS Rationale for Early Classification

### 2.1 The Line-Rate Switching Dilemma
In high-throughput IP networks, an average flow duration can span hundreds of seconds and tens of thousands of packets. If network routers or edge security appliances defer traffic prioritization until 50 or 100 packets have traversed the interface:
- **Head-of-Line (HoL) Blocking:** Latency-sensitive Real-Time voice/video packets get trapped in default best-effort FIFO queues behind heavy Bulk Data bursts.
- **Buffer Bloat & Packet Drop:** Interactive sessions experience jitter and packet drops before bandwidth reservations take effect.
- **Line-Rate Memory Constraints:** Commodity switch TCAM (Ternary Content-Addressable Memory) and SRAM cannot maintain deep flow state for millions of concurrent connections.

```
Flow Arrival (Packet 1)
       │
       ▼
 [Horizon N=5]   ──► Latency < 5 ms    ──► Immediate Queue Assignment (Strict Priority / Low Latency Queuing)
       │
       ▼
 [Horizon N=10]  ──► Latency 10–25 ms  ──► Handshake Verification & Dynamic Bandwidth Allocation
       │
       ▼
 [Horizon N=15]  ──► Latency 25–50 ms  ──► TLS Negotiation Confirmation & Slice Re-assignment
       │
       ▼
 [Horizon N=20]  ──► Latency 50–100 ms ──► Steady-State Policy Finalization
       │
       ▼
 [Full-Flow End] ──► Post-hoc audit only (Offline; impractical for real-time QoS control)
```

### 2.2 Objective of Early Feature Windows
The goal is to determine the earliest packet threshold $N^*$ where a classifier achieves performance parity ($\ge 90\%$ of full-flow Macro F1) while maintaining line-rate execution speed.

---

## 3. Base Dataset & Class Definitions

### 3.1 Base Dataset: Track B (Primary 3-Class Benchmark)
Phase 6B is conducted strictly on **Track B**, defined in [track_b_flows.csv](file:///c:/Users/saipr/OneDrive/Desktop/EncryptedTrafficResearch/output/phase5/track_b_flows.csv). Track B contains 10,770 flows derived from 29 capture groups.

#### Class Distribution
| QoS Class ID | QoS Class Name | Flow Count | Relative Share (%) | Capture Groups | Description & Target Network Handling |
|:---:|---|:---:|:---:|:---:|---|
| 0 | **Bulk Data** | 1,348 | 12.52% | 10 | Large data transfers (FTP, SFTP, FTPS, cloud storage). Bandwidth-heavy, jitter-tolerant; eligible for background/elastic queuing. |
| 1 | **Real-Time** | 8,132 | 75.51% | 10 | Interactive voice/video communication (VoIP, chat, conferencing). Strict low-latency, low-jitter priority queuing. |
| 2 | **Streaming** | 1,290 | 11.98% | 9 | Video/Audio streaming (YouTube, Vimeo, Spotify). High sustained bandwidth, buffered, moderate latency tolerance. |
| **Total** | **3 Classes** | **10,770** | **100.00%** | **29 Groups** | **Primary Benchmark Dataset** |

### 3.2 Methodological Justification for Excluding Browsing
As proven in Phase 4 feature compatibility audits and confirmed empirically in Phase 6A Track A:
- The ISCX VPN-nonVPN dataset's `Email` captures consist almost entirely of local host LLMNR (UDP port 5355) multicast background noise rather than interactive web traffic.
- In Phase 6A Track A, all models collapsed on the `Browsing` class (pooled recall 2.6% to 8.1%), with >62% to 82% falsely classified as Real-Time.
- Retaining corrupted browsing flows introduces severe label pollution. Track B represents the true, scientifically defensible benchmark for QoS traffic classification.

---

## 4. Cross-Validation Scheme & Group Disjointness Proof

### 4.1 Strict Group-KFold Partitioning
To prevent **session fingerprinting** and **host MTU leakage**, flows from the same capture file/session group must never appear in both training and validation folds simultaneously. 

Phase 6B strictly uses the pre-computed, verified fold mapping in [track_b_folds.csv](file:///c:/Users/saipr/OneDrive/Desktop/EncryptedTrafficResearch/output/phase5/track_b_folds.csv).

### 4.2 Group Disjointness Verification
For each fold $k \in \{0, 1, 2, 3, 4\}$, the set of validation capture groups $\mathcal{G}_{val}^{(k)}$ and training capture groups $\mathcal{G}_{train}^{(k)}$ satisfy:
$$\mathcal{G}_{train}^{(k)} \cap \mathcal{G}_{val}^{(k)} = \emptyset \quad \forall k \in \{0, 1, 2, 3, 4\}$$

#### Partition Details Across 5 Folds:
| Fold ID | Validation Flows | Training Flows | Validation Groups | Training Groups | Group Overlap | Class Completeness |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Fold 0** | 1,723 | 9,047 | 8 | 21 | **0 (None)** | 100% (3/3 classes present) |
| **Fold 1** | 1,397 | 9,373 | 5 | 24 | **0 (None)** | 100% (3/3 classes present) |
| **Fold 2** | 4,257 | 6,513 | 6 | 23 | **0 (None)** | 100% (3/3 classes present) |
| **Fold 3** | 2,132 | 8,638 | 5 | 24 | **0 (None)** | 100% (3/3 classes present) |
| **Fold 4** | 1,261 | 9,509 | 5 | 24 | **0 (None)** | 100% (3/3 classes present) |
| **Total** | **10,770** | — | **29 Groups** | — | **0 (Leak-Free)** | **Complete Coverage** |

Every flow in `track_b_flows.csv` is evaluated as an out-of-fold validation instance exactly once.

---

## 5. Early Packet Horizons & Feature Taxonomy

### 5.1 Early Feature Windows
Phase 6B evaluates four early packet horizons: $N \in \{5, 10, 15, 20\}$ packets. At each packet horizon $N$, 6 distinct behavioral features are extracted directly from flow packet headers without payload inspection.

#### Formal Feature Definitions per Horizon $N$:
1. `early_{N}_complete`: Binary flag ($\in \{0, 1\}$) indicating whether the flow reached at least $N$ packets during its entire lifetime ($1 = \text{True}$, $0 = \text{False}$).
2. `early_{N}_duration`: Elapsed time (in seconds) from the arrival timestamp of packet 1 to packet $N$ (or flow end time if total flow packets $< N$).
3. `early_{N}_bytes`: Cumulative bidirectional payload and header bytes transmitted across the first $N$ packets.
4. `early_{N}_fwd_pkts`: Count of forward-direction (client-to-server) packets observed within the first $N$ packets.
5. `early_{N}_bwd_pkts`: Count of backward-direction (server-to-client) packets observed within the first $N$ packets ($\text{early}_{N}\text{\_fwd\_pkts} + \text{early}_{N}\text{\_bwd\_pkts} = \min(\text{total\_packets}, N)$).
6. `early_{N}_avg_size`: Mean packet size in bytes across observed early packets ($\text{early}_{N}\text{\_bytes} / \min(\text{total\_packets}, N)$).

### 5.2 Cumulative Feature Progression
In line with standard sequential network observation, an SDN controller observing up to packet $N_2$ has access to all statistics measured at prior horizon $N_1 < N_2$. The feature sets are cumulative:

```
Early-5  (6 features)  : [Horizon 5]
Early-10 (12 features) : [Horizon 5] + [Horizon 10]
Early-15 (18 features) : [Horizon 5] + [Horizon 10] + [Horizon 15]
Early-20 (24 features) : [Horizon 5] + [Horizon 10] + [Horizon 15] + [Horizon 20]
```

#### Detailed Feature Set Specification Table:
| Horizon ID | Feature Count | Specific Feature Columns | Network Inspection Phase | Target Latency |
|---|:---:|---|---|:---:|
| **Early-5** | **6** | `early_5_complete`, `early_5_duration`, `early_5_bytes`, `early_5_fwd_pkts`, `early_5_bwd_pkts`, `early_5_avg_size` | TCP Handshake (SYN, SYN-ACK, ACK) + TLS ClientHello / initial burst | $< 5\text{ ms}$ |
| **Early-10** | **12** | Early-5 features + `early_10_complete`, `early_10_duration`, `early_10_bytes`, `early_10_fwd_pkts`, `early_10_bwd_pkts`, `early_10_avg_size` | TLS ServerHello, Certificate exchange, Key Exchange initial cipher negotiation | $10 - 25\text{ ms}$ |
| **Early-15** | **18** | Early-10 features + `early_15_complete`, `early_15_duration`, `early_15_bytes`, `early_15_fwd_pkts`, `early_15_bwd_pkts`, `early_15_avg_size` | TLS Handshake completion, Finished message, initial encrypted application records | $25 - 50\text{ ms}$ |
| **Early-20** | **24** | Early-15 features + `early_20_complete`, `early_20_duration`, `early_20_bytes`, `early_20_fwd_pkts`, `early_20_bwd_pkts`, `early_20_avg_size` | Post-handshake application burst framing, early request-response turn-around | $50 - 100\text{ ms}$ |
| *Full-Flow (Ref)* | *78* | Complete set of 78 ML-safe full-flow statistical and temporal features | Offline post-flow session lifetime | Post-flow |

---

## 6. Model Family Specifications & Hyperparameters

To ensure direct, unconfounded comparability with Phase 6A full-flow baselines, model architectures, hyperparameters, initialization seeds, and training procedures must be **strictly identical** to Phase 6A.

### 6.1 Random Forest Classifier (RF)
- **Library:** `scikit-learn.ensemble.RandomForestClassifier`
- **Number of Estimators (`n_estimators`):** 100 trees
- **Split Criterion (`criterion`):** `gini`
- **Max Depth (`max_depth`):** None (unconstrained, nodes expanded until pure or min samples reached)
- **Min Samples Split (`min_samples_split`):** 2
- **Min Samples Leaf (`min_samples_leaf`):** 1
- **Class Weighting (`class_weight`):** `"balanced"` (inversely proportional to class frequencies in training fold)
- **Random Seed (`random_state`):** 42
- **Parallel Workers (`n_jobs`):** -1 (all available CPU cores)

### 6.2 XGBoost Gradient Boosted Decision Trees (XGB)
- **Library:** `xgboost.XGBClassifier`
- **Number of Estimators (`n_estimators`):** 100 boosted rounds
- **Learning Rate (`learning_rate`):** 0.1
- **Max Tree Depth (`max_depth`):** 6
- **Subsample Ratio (`subsample`):** 0.8 (row subsampling per tree)
- **Feature Subsample Ratio (`colsample_bytree`):** 0.8 (feature subsampling per tree)
- **Objective (`objective`):** `"multi:softprob"` (multi-class probabilistic classification)
- **Evaluation Metric (`eval_metric`):** `"mlogloss"`
- **Class Balancing:** `sample_weight` computed strictly on training partition:
  $$w_c = \frac{N_{train}}{K \cdot N_c}$$
  where $N_{train}$ is total training flows, $K=3$ classes, and $N_c$ is flow count for class $c$.
- **Random Seed (`random_state`):** 42
- **Parallel Workers (`n_jobs`):** -1

### 6.3 Packet Multi-Layer Perceptron (P-MLP)
- **Framework:** `PyTorch` (`torch.nn`)
- **Network Architecture:**
  - Input Layer: `Linear(in_features, 128)`
  - Normalization & Non-linearity: `BatchNorm1d(128)` $\rightarrow$ `ReLU()`
  - Regularization: `Dropout(p=0.3)`
  - Hidden Layer 1: `Linear(128, 64)` $\rightarrow$ `BatchNorm1d(64)` $\rightarrow$ `ReLU()` $\rightarrow$ `Dropout(p=0.3)`
  - Hidden Layer 2: `Linear(64, 32)` $\rightarrow$ `BatchNorm1d(32)` $\rightarrow$ `ReLU()` $\rightarrow$ `Dropout(p=0.2)`
  - Output Layer: `Linear(32, 3)` (raw class logits)
- **Loss Function:** `nn.CrossEntropyLoss(weight=class_weights_tensor)`
  - Class weights calculated strictly on training fold using inverse class frequencies and moved to execution device.
- **Optimization Algorithm:** `optim.Adam(lr=0.001, weight_decay=1e-4)`
- **Learning Rate Scheduler:** `optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)`
- **Batch Size:** 64
- **Maximum Epochs:** 100
- **Early Stopping:** Patience of 15 epochs monitoring validation loss (`val_loss`). If validation loss does not strictly improve for 15 consecutive epochs, training terminates and the best model weights are restored.
- **Reproducibility:** Seed 42 set for Python `random`, `numpy`, `torch.manual_seed`, and `torch.cuda.manual_seed_all`. Device pinned to `torch.device("cpu")` (or CUDA if reproducible deterministic backend configured).

---

## 7. Feature Preprocessing & Leakage Prevention Rules

To guarantee strict scientific integrity and prevent data snooping:

### 7.1 Zero-Variance Feature Dropping
- In any cross-validation fold, features exhibiting zero variance ($\sigma = 0.0$) contain zero discriminative information and can cause numerical instabilities in neural architectures.
- **Rule:** The standard deviation $\sigma$ must be computed **strictly on the training fold** ($X_{train}$).
- Any column in $X_{train}$ where $\sigma == 0.0$ is identified and dropped from $X_{train}$.
- The exact identical list of dropped columns must then be dropped from $X_{val}$.
- **Never** compute standard deviation on the combined dataset or validation fold.

### 7.2 Collinearity Handling (Optional Check)
- If two early features exhibit $|r| \ge 0.999$, correlation is computed strictly on $X_{train}$ and the redundant column dropped. For early features, because $N$ expands cumulatively, features across different horizons may correlate; however, retaining distinct horizons reflects temporal evolution. If collinearity filtering is enabled, it must be fit strictly on $X_{train}$.

### 7.3 Feature Standardisation (Scaling for P-MLP)
- Artificial neural networks require zero-mean unit-variance input normalization for stable gradient descent through `BatchNorm1d` and `Linear` layers.
- **Rule:** `StandardScaler` must be instantiated and **fitted strictly on $X_{train}$**:
  $$\mu_{train} = \frac{1}{N_{train}}\sum_{i=1}^{N_{train}} x_i, \quad \sigma_{train} = \sqrt{\frac{1}{N_{train}}\sum_{i=1}^{N_{train}} (x_i - \mu_{train})^2}$$
- $X_{train}$ is transformed using $\mu_{train}$ and $\sigma_{train}$.
- $X_{val}$ is transformed using the stored $\mu_{train}$ and $\sigma_{train}$:
  $$X_{val\_scaled} = \frac{X_{val} - \mu_{train}}{\sigma_{train} + \epsilon}$$
- **Zero test/validation information is ever seen by the scaler.**

### 7.4 Sample & Class Weight Fitting
- Class frequency weights ($w_c$) must be derived strictly from the label distribution $y_{train}$ of that specific fold.

---

## 8. Experimental Matrix & Execution Plan

### 8.1 Experiment Space
The experimental design spans:
$$\mathcal{M}_{exp} = 4 \text{ Horizons} \times 3 \text{ Model Families} \times 5 \text{ Cross-Validation Folds} = \mathbf{60 \text{ Runs}}$$

### 8.2 Tabular Specification of the 60 Model-Fold Runs
| Run ID | Early Horizon | Feature Count | Model Family | Fold ID | Training Samples | Val Samples | Preprocessing Sequence |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **01–05** | Early-5 | 6 | Random Forest | 0–4 | ~9,000 | ~1,700 | Zero-var drop on train; balanced tree weights |
| **06–10** | Early-5 | 6 | XGBoost | 0–4 | ~9,000 | ~1,700 | Zero-var drop on train; inverse-freq sample weights |
| **11–15** | Early-5 | 6 | P-MLP | 0–4 | ~9,000 | ~1,700 | Zero-var drop on train; StandardScaler on train; Adam/CE |
| **16–20** | Early-10 | 12 | Random Forest | 0–4 | ~9,000 | ~1,700 | Zero-var drop on train; balanced tree weights |
| **21–25** | Early-10 | 12 | XGBoost | 0–4 | ~9,000 | ~1,700 | Zero-var drop on train; inverse-freq sample weights |
| **26–30** | Early-10 | 12 | P-MLP | 0–4 | ~9,000 | ~1,700 | Zero-var drop on train; StandardScaler on train; Adam/CE |
| **31–35** | Early-15 | 18 | Random Forest | 0–4 | ~9,000 | ~1,700 | Zero-var drop on train; balanced tree weights |
| **36–40** | Early-15 | 18 | XGBoost | 0–4 | ~9,000 | ~1,700 | Zero-var drop on train; inverse-freq sample weights |
| **41–45** | Early-15 | 18 | P-MLP | 0–4 | ~9,000 | ~1,700 | Zero-var drop on train; StandardScaler on train; Adam/CE |
| **46–50** | Early-20 | 24 | Random Forest | 0–4 | ~9,000 | ~1,700 | Zero-var drop on train; balanced tree weights |
| **51–55** | Early-20 | 24 | XGBoost | 0–4 | ~9,000 | ~1,700 | Zero-var drop on train; inverse-freq sample weights |
| **56–60** | Early-20 | 24 | P-MLP | 0–4 | ~9,000 | ~1,700 | Zero-var drop on train; StandardScaler on train; Adam/CE |

---

## 9. Evaluation Metrics, Measurement Standards & Telemetry

### 9.1 Primary Classification Metrics
In highly imbalanced QoS datasets (Track B has 75.5% Real-Time, 12.5% Bulk Data, 12.0% Streaming), raw overall accuracy is deeply deceptive (a dummy model predicting Real-Time gets 75.5% accuracy but fails completely for Bulk Data and Streaming).

1. **Macro-Averaged F1-Score ($\text{Macro } F_1$):** Primary optimization and ranking metric:
   $$\text{Macro } F_1 = \frac{1}{K} \sum_{c=1}^{K} F_{1, c} = \frac{1}{K} \sum_{c=1}^{K} \frac{2 \cdot P_c \cdot R_c}{P_c + R_c}$$
2. **Balanced Accuracy ($\text{BalAcc}$):** Primary sensitivity metric:
   $$\text{Balanced Accuracy} = \frac{1}{K} \sum_{c=1}^{K} \text{Recall}_c = \frac{1}{K} \sum_{c=1}^{K} \frac{TP_c}{TP_c + FN_c}$$

### 9.2 Secondary Classification Metrics
3. **Overall Accuracy ($\text{Acc}$):** $\frac{\sum_{c} TP_c}{N_{total}}$.
4. **Weighted F1-Score ($\text{Weighted } F_1$):** Weighted by class support.
5. **Per-Class Metrics:** Precision ($P_c$), Recall ($R_c$), and $F_{1, c}$ for every class $c \in \{\text{Bulk Data}, \text{Real-Time}, \text{Streaming}\}$.
6. **Confusion Matrices:**
   - Raw integer counts: $C_{i, j} = |\{x : y_{true} = i \land y_{pred} = j\}|$
   - Row-normalized fractions: $\tilde{C}_{i, j} = C_{i, j} / \sum_{k} C_{i, k}$ (direct class recall on diagonal).

### 9.3 System & Hardware Telemetry
7. **Training Wall-Clock Time ($T_{train}$):** Elapsed execution time in seconds to fit the model on $X_{train}$.
8. **Inference Latency Per Flow ($L_{inf}$):**
   $$L_{inf} = \frac{T_{predict}(X_{val})}{|X_{val}|} \times 10^6 \quad (\mu\text{s / flow})$$
   Measured in microseconds per flow instance. Critical for assessing SDN controller line-rate feasibilities.

---

## 10. Direct Baseline Comparability Protocol with Phase 6A Full-Flow Results

### 10.1 The Phase 6A Benchmark Anchor
To evaluate early classification performance degradation or retention, all Phase 6B runs must be benchmarked against the official Phase 6A Track B full-flow baseline results recorded in [PHASE6A_FULL_FLOW_BASELINE_REPORT.md](file:///c:/Users/saipr/OneDrive/Desktop/EncryptedTrafficResearch/research_notes/phase6/PHASE6A_FULL_FLOW_BASELINE_REPORT.md):

#### Phase 6A Full-Flow Benchmark Reference Table (78 Features, 5 Folds):
| Model Family | Mean Macro F1 ($\pm \sigma$) | Mean Bal Acc ($\pm \sigma$) | Mean Acc ($\pm \sigma$) | Pooled Macro F1 | Pooled Bal Acc | Pooled Acc | Inference Latency ($\mu s$/flow) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Random Forest** | 0.3970 $\pm$ 0.1831 | 0.5425 $\pm$ 0.1897 | 0.5145 $\pm$ 0.2248 | 0.4664 | 0.5051 | 0.6016 | 38.9 $\mu s$ |
| **XGBoost** | 0.4564 $\pm$ 0.1255 | 0.6072 $\pm$ 0.1640 | 0.5759 $\pm$ 0.1587 | 0.5452 | 0.6349 | 0.6329 | 3.4 $\mu s$ |
| **P-MLP** | 0.4069 $\pm$ 0.0384 | 0.5454 $\pm$ 0.0857 | 0.5684 $\pm$ 0.1915 | **0.5567** | **0.6823** | 0.6257 | **1.7 $\mu s$** |

### 10.2 Comparative Analytical Metrics to Compute
For every horizon $H \in \{\text{Early-5, Early-10, Early-15, Early-20}\}$ and model $M$:
1. **Absolute Performance Delta:**
   $$\Delta \text{Macro } F_1^{(H, M)} = \text{Macro } F_1^{(H, M)} - \text{Macro } F_1^{(\text{Full-Flow}, M)}$$
   $$\Delta \text{BalAcc}^{(H, M)} = \text{BalAcc}^{(H, M)} - \text{BalAcc}^{(\text{Full-Flow}, M)}$$
2. **Early Performance Retention Ratio ($\mathcal{R}_{ret}$):**
   $$\mathcal{R}_{ret}(H, M) = \frac{\text{Macro } F_1^{(H, M)}}{\text{Macro } F_1^{(\text{Full-Flow}, M)}} \times 100\%$$
3. **Latency Speedup Factor ($\mathcal{S}_{lat}$):**
   $$\mathcal{S}_{lat}(H, M) = \frac{L_{inf}^{(\text{Full-Flow}, M)}}{L_{inf}^{(H, M)}}$$
4. **Accuracy-vs-Horizon Trajectory Curve:**
   Mapping $N \in \{5, 10, 15, 20, \text{Full}\}$ against Pooled Macro F1 and Balanced Accuracy to detect:
   - Knee-of-the-curve packet threshold (the point of diminishing returns).
   - Early separation capability between Bulk Data and Real-Time.

---

## 11. Output Artifact Specifications & Directory Schema

The experiment execution engine will store all artifacts in a structured directory hierarchy under `output/phase6b/`:

```
output/phase6b/
├── early_5/
│   ├── fold_metrics.csv                # Metrics for all 3 models across 5 folds
│   ├── aggregate_metrics.csv           # Mean ± std and pooled out-of-fold metrics
│   ├── per_class_metrics.csv           # Class-specific precision, recall, F1 per model
│   ├── confusion_matrix_counts.csv     # Raw integer confusion matrices
│   ├── confusion_matrix_normalized.csv # Row-normalized recall confusion matrices
│   ├── out_of_fold_predictions.csv     # Per-flow predictions & predicted probabilities
│   ├── runtime_summary.csv             # Training duration and inference latency per flow
│   └── training_log.csv                # Epoch-by-epoch loss log for P-MLP
├── early_10/                           # (Identical schema for Early-10)
├── early_15/                           # (Identical schema for Early-15)
├── early_20/                           # (Identical schema for Early-20)
└── consolidated/
    ├── early_vs_full_comparison.csv    # Cross-horizon comparison vs Phase 6A baseline
    ├── horizon_trajectory_summary.csv  # Performance metrics across 5, 10, 15, 20, Full
    └── early_classification_report.json # Machine-readable master summary
```

### 11.1 Standardized CSV Column Schemas

#### Schema: `fold_metrics.csv`
- `horizon`: String (`Early-5`, `Early-10`, `Early-15`, `Early-20`)
- `model`: String (`Random_Forest`, `XGBoost`, `P_MLP`)
- `fold`: Integer (`0` through `4`)
- `val_samples`: Integer
- `val_groups`: Integer
- `active_features`: Integer (count of features after zero-variance drop)
- `macro_f1`: Float
- `balanced_accuracy`: Float
- `accuracy`: Float
- `weighted_f1`: Float
- `train_time_sec`: Float
- `pred_time_sec`: Float
- `latency_us_per_flow`: Float

#### Schema: `out_of_fold_predictions.csv`
- `flow_id`: String (unique flow identifier matching `track_b_flows.csv`)
- `fold`: Integer
- `capture_group`: String
- `true_class`: String (`Bulk Data`, `Real-Time`, `Streaming`)
- `rf_pred`: String
- `rf_prob_bulk`: Float, `rf_prob_realtime`: Float, `rf_prob_streaming`: Float
- `xgb_pred`: String
- `xgb_prob_bulk`: Float, `xgb_prob_realtime`: Float, `xgb_prob_streaming`: Float
- `pmlp_pred`: String
- `pmlp_prob_bulk`: Float, `pmlp_prob_realtime`: Float, `pmlp_prob_streaming`: Float

---

## 12. Reproducibility & Integrity Verification Protocol

To satisfy publication and engineering audit requirements:

### 12.1 Execution Pre-Flight Checklist
Before launching training, the execution script must programmatically assert:
- [x] Dataset row count is exactly 10,770 rows.
- [x] Folds file row count is exactly 10,770 rows.
- [x] Every `flow_id` matches exactly in order.
- [x] Ground-truth and metadata columns are completely excluded from candidate feature vectors.
- [x] Zero `NaN`, `Inf`, or `null` values exist in feature matrices.
- [x] In every fold, `val_groups` $\cap$ `train_groups` is strictly empty (zero group leakage).
- [x] All 3 classes exist in both training and validation sets for every fold.
- [x] `GLOBAL_SEED = 42` is set across all stochastic modules.

### 12.2 Experiment Script Reference
The executable script `scripts/run_phase6b_experiments.py` implements this protocol in full fidelity, maintaining identical software dependencies and hardware constraints as Phase 6A.
