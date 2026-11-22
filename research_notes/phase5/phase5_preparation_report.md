# Master Phase 5 Report: Baseline Experiment Preparation & Protocol Validation

**Project:** QoS-Aware Encrypted Network Traffic Classification for Next-Generation Networks  
**Role:** Lead Machine Learning Research Coordinator & Computer Networks Analyzer  
**Date:** November 2026  
**Status:** Completed & Validated — Awaiting User Approval for Model Training (Phase 6)

---

## 1. Executive Summary

Phase 5 has successfully established a fully reproducible, leak-free, scientifically defensible experimental protocol for evaluating machine learning models on encrypted network traffic. 

In accordance with strict research directives:
1. **Original Data Preserved:** The original dataset [output/flow_features.csv](file:///c:/Users/saipr/OneDrive/Desktop/EncryptedTrafficResearch/output/flow_features.csv) was audited and left completely untouched. No PCAP, CSV, or prior report was modified or overwritten.
2. **Zero Model Training:** No machine learning models were trained, no hyperparameter optimization was executed, and no synthetic oversampling (SMOTE) was introduced.
3. **Dual-Track Protocol Formulated:**
   - **Track A (Exploratory 4-Class):** Evaluates all 17,617 flows across Bulk Data, Browsing, Streaming, and Real-Time, explicitly disclosing that Browsing is derived from Email captures containing 75% local multicast/broadcast background noise (LLMNR UDP 5355). Evaluated with a leak-free 3-Fold `StratifiedGroupKFold`.
   - **Track B (Primary 3-Class):** Evaluates 10,770 flows across Bulk Data, Streaming, and Real-Time (excluding the contaminated Browsing class). Evaluated with a leak-free 5-Fold `StratifiedGroupKFold`.
4. **Verified Leak-Free Splits:** Both tracks were partitioned and verified with zero capture group leakage across folds ($\text{groups}(\text{train}) \cap \text{groups}(\text{val}) = \emptyset$) and 100% class coverage across every single validation fold.
5. **Early-Stage Classification Framework:** Protocol established for evaluating classifier accuracy at early packet windows ($N \in \{5, 10, 15, 20\}$ packets) to meet real-time QoS scheduling requirements.

---

## 2. Dataset Audit & Feature Verification

| Metric / Property | Verified Value | Evidence Artifact |
|---|---|---|
| **Total Flows Extracted** | 17,617 rows | [flow_features.csv](file:///c:/Users/saipr/OneDrive/Desktop/EncryptedTrafficResearch/output/flow_features.csv) |
| **Total Schema Columns** | 117 columns | [feature_audit_summary.csv](file:///c:/Users/saipr/OneDrive/Desktop/EncryptedTrafficResearch/output/phase5/feature_audit_summary.csv) |
| **Missing / NaN / Inf Count** | Exactly 0 across all 117 columns | [feature_audit.md](file:///c:/Users/saipr/OneDrive/Desktop/EncryptedTrafficResearch/research_notes/phase5/feature_audit.md) |
| **Metadata / Audit Columns** | 14 columns (Prohibited from ML input) | [feature_sets.json](file:///c:/Users/saipr/OneDrive/Desktop/EncryptedTrafficResearch/output/phase5/feature_sets.json) |
| **ML-Safe Behavioral Features** | 103 columns | [feature_sets.json](file:///c:/Users/saipr/OneDrive/Desktop/EncryptedTrafficResearch/output/phase5/feature_sets.json) |
| — Full-Flow Behavioral Features | 79 columns (77 active, 1 zero-var, 1 collinear) | [feature_audit.md](file:///c:/Users/saipr/OneDrive/Desktop/EncryptedTrafficResearch/research_notes/phase5/feature_audit.md) |
| — Early-Packet Behavioral Features | 24 columns ($6 \times 4$ windows: 5, 10, 15, 20) | [early_feature_definitions.json](file:///c:/Users/saipr/OneDrive/Desktop/EncryptedTrafficResearch/output/phase5/early_feature_definitions.json) |
| **Zero-Variance Feature** | `tcp_urg_count` ($\mu=0.0, \sigma=0.0$) | Dropped via training-fold `VarianceThreshold(0.0)` |
| **Perfect Collinear Pair** | `fwd_burst_count` vs `bwd_burst_count` ($r=1.0000$) | `bwd_burst_count` dropped via training correlation filter |

---

## 3. Experimental Track Specifications

### 3.1 Track A: Exploratory 4-Class Baseline
- **Dataset File:** [track_a_flows.csv](file:///c:/Users/saipr/OneDrive/Desktop/EncryptedTrafficResearch/output/phase5/track_a_flows.csv) (17,617 rows)
- **Classes:** Real-Time (8,132, 46.16%), Browsing (6,847, 38.87%), Bulk Data (1,348, 7.65%), Streaming (1,290, 7.32%).
- **Class Imbalance Ratio:** $6.30 : 1$.
- **Capture Groups:** 32 unique groups.
- **Limitation:** Browsing is confined to only 3 capture groups (`GRP_EMAIL_1`, `GRP_EMAIL_2`, `GRP_VPN_EMAIL_2`) and is contaminated with 75% LLMNR UDP 5355 noise.
- **Cross-Validation Scheme:** 3-Fold `StratifiedGroupKFold` (random_state=7).
- **Split Verification:**
  - Fold 0: 1,625 val samples, 7 val groups. Group overlap: 0. Class coverage: 4/4 (100%).
  - Fold 1: 11,677 val samples, 9 val groups. Group overlap: 0. Class coverage: 4/4 (100%).
  - Fold 2: 4,315 val samples, 16 val groups. Group overlap: 0. Class coverage: 4/4 (100%).

### 3.2 Track B: Primary 3-Class Benchmark
- **Dataset File:** [track_b_flows.csv](file:///c:/Users/saipr/OneDrive/Desktop/EncryptedTrafficResearch/output/phase5/track_b_flows.csv) (10,770 rows)
- **Classes:** Real-Time (8,132, 75.51%), Bulk Data (1,348, 12.52%), Streaming (1,290, 11.98%).
- **Class Imbalance Ratio:** $6.30 : 1$.
- **Capture Groups:** 29 unique groups (Bulk Data: 10, Streaming: 9, Real-Time: 10).
- **Cross-Validation Scheme:** 5-Fold `StratifiedGroupKFold` (random_state=2).
- **Split Verification:**
  - Fold 0: 1,723 val samples, 8 val groups. Group overlap: 0. Class coverage: 3/3 (100%).
  - Fold 1: 1,397 val samples, 5 val groups. Group overlap: 0. Class coverage: 3/3 (100%).
  - Fold 2: 4,257 val samples, 6 val groups. Group overlap: 0. Class coverage: 3/3 (100%).
  - Fold 3: 2,132 val samples, 5 val groups. Group overlap: 0. Class coverage: 3/3 (100%).
  - Fold 4: 1,261 val samples, 5 val groups. Group overlap: 0. Class coverage: 3/3 (100%).

---

## 4. Early-Stage Classification Protocol

To evaluate early classification feasibility for programmable network switches and SDN controllers, five standardized feature subsets are configured:
1. **Early-5 (6 features):** Sub-millisecond latency; enables immediate packet queue mapping.
2. **Early-10 (12 features):** Captures TCP 3-way handshake and initial TLS ClientHello / ServerHello.
3. **Early-15 (18 features):** Captures complete TLS negotiation and initial application data.
4. **Early-20 (24 features):** Upper bound of early classification horizon ($50-100$ ms).
5. **Full-Flow (78 features):** Post-flow performance ceiling benchmark.

---

## 5. Model Architecture Configurations

Complete specifications are defined in [model_pipeline_specs.json](file:///c:/Users/saipr/OneDrive/Desktop/EncryptedTrafficResearch/output/phase5/model_pipeline_specs.json):
1. **Random Forest Classifier:** 100 estimators, `class_weight='balanced'`, full leaf purity, parallelized over all CPU cores.
2. **XGBoost Classifier:** 100 estimators, `learning_rate=0.1`, `max_depth=6`, `subsample=0.8`, `colsample_bytree=0.8`, with sample weighting inversely proportional to class frequencies.
3. **Packet Multi-Layer Perceptron (P-MLP):** PyTorch deep neural network (Dense 128 -> BatchNorm -> ReLU -> Dropout 0.3 -> Dense 64 -> BatchNorm -> ReLU -> Dropout 0.3 -> Dense 32 -> BatchNorm -> ReLU -> Dropout 0.2 -> Dense num_classes), optimized with Adam (`lr=0.001`, `weight_decay=1e-4`), class-weighted Cross-Entropy loss, and ReduceLROnPlateau scheduling.
   - *Distinction from Huang et al. 2021:* Huang et al. processed raw byte payloads in 1D-CNN/LSTM, violating end-to-end privacy and failing under TLS 1.3/ECH. Our P-MLP uses flow-level statistical and temporal packet dynamics without payload inspection.

---

## 6. Phase 5 Deliverables Directory Map

```
EncryptedTrafficResearch/
├── output/
│   ├── flow_features.csv                               # Untouched original 17,617 flows (117 cols)
│   └── phase5/
│       ├── dataset_summary.json                        # Comprehensive dataset metadata
│       ├── feature_audit_summary.csv                   # Per-feature audit statistics
│       ├── feature_sets.json                           # Categorized feature lists
│       ├── track_a_flows.csv                           # Track A 4-class dataset (17,617 flows)
│       ├── track_b_flows.csv                           # Track B 3-class dataset (10,770 flows)
│       ├── track_a_folds.csv                           # Track A fold assignments (3 folds)
│       ├── track_b_folds.csv                           # Track B fold assignments (5 folds)
│       ├── track_a_fold_analysis.csv                   # Track A fold distribution analysis
│       ├── track_b_fold_analysis.csv                   # Track B fold distribution analysis
│       ├── cross_validation_splits.json                # Complete CV split specifications
│       ├── model_pipeline_specs.json                   # Exact model hyperparameters & architectures
│       └── early_feature_definitions.json              # Early packet feature window specifications
├── research_notes/
│   └── phase5/
│       ├── feature_audit.md                            # Feature audit and preprocessing rules
│       ├── track_a_dataset_definition.md               # Track A dataset and group distribution
│       ├── track_b_dataset_definition.md               # Track B primary benchmark definition
│       ├── validation_protocol.md                      # Cross-validation & metric specifications
│       ├── early_classification_protocol.md            # Early-packet QoS classification framework
│       ├── baseline_model_configurations.md            # Model architectures & Huang et al. comparison
│       └── phase5_preparation_report.md                # This master Phase 5 report
└── scripts/
    ├── phase5_feature_audit.py                         # Feature audit automation script
    ├── prepare_phase5_datasets.py                      # Dataset preparation script
    └── validate_phase5_splits.py                       # Group cross-validation split validator
```

---

## 7. Protocol Ready for User Approval

All prerequisites, pipelines, and validation splits are mathematically verified and ready. **No models have been trained.** 

We await user review and explicit authorization to begin **Phase 6: Baseline Model Training and Benchmarking**.
