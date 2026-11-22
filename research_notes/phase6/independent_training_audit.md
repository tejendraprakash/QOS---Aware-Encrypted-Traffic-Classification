# Independent Training & Artifact Audit: Phase 6A Baseline Experiments

**Project:** QoS-Aware Encrypted Traffic Classification for Next-Generation Networks  
**Audit Target:** Phase 6A Full-Flow Baseline Training Implementation & Experimental Artifacts  
**Auditor:** Independent Methodology & Experimental Integrity Auditor  
**Date:** November 2026  
**Audit Verdict:** **100% PASSED (Zero Discrepancies, Zero Leakage, Full Reproducibility)**  

---

## Executive Summary

An exhaustive, independent verification and forensic audit was conducted on the Phase 6A training implementation, execution script ([scripts/run_phase6a_experiments.py](file:///c:/Users/saipr/OneDrive/Desktop/EncryptedTrafficResearch/scripts/run_phase6a_experiments.py)), underlying input datasets from Phase 5, and all generated artifacts in [output/phase6/track_a/](file:///c:/Users/saipr/OneDrive/Desktop/EncryptedTrafficResearch/output/phase6/track_a/) and [output/phase6/track_b/](file:///c:/Users/saipr/OneDrive/Desktop/EncryptedTrafficResearch/output/phase6/track_b/).

Every step of the training pipeline—including metadata exclusions, group disjointness, fold assignments, fold-level dynamic preprocessing, class/sample weighting, model hyperparameters, training convergence logs, inference latencies, and out-of-fold prediction matrices—was independently audited and mathematically verified.

### Summary of Audit Checkpoints

| Checkpoint | Scope | Verified Requirement | Audit Finding | Status |
|---|---|---|---|:---:|
| **1. Execution Script Architecture** | [run_phase6a_experiments.py](file:///c:/Users/saipr/OneDrive/Desktop/EncryptedTrafficResearch/scripts/run_phase6a_experiments.py) | Clean modular architecture, deterministic seeding (`seed=42`), CPU execution | Fully modular, leak-free design, strict assertions | **PASSED** |
| **2. Metadata Exclusions** | Flow feature matrices | All 14 metadata/identity columns excluded from candidate and active matrices | 0 metadata leakage (0/14), 0 early-packet leakage (0/24) | **PASSED** |
| **3. Group Leakage** | Track A (3 folds), Track B (5 folds) | $\mathcal{G}_{train}^{(k)} \cap \mathcal{G}_{val}^{(k)} = \emptyset$ across all 8 folds | Strict zero overlap ($|overlap| = 0$) across every fold | **PASSED** |
| **4. Fold & Sample Counts** | Track A (17,617), Track B (10,770) | 1-to-1 `flow_id` alignment, exact row counts, 100% class representation per fold | Exact 1:1 match with original dataset flows; no missing classes | **PASSED** |
| **5. Preprocessing Snooping** | Feature filtering & scaling | `VarianceThreshold`, Pearson collinearity, `StandardScaler` fitted on train fold only | Verified dynamic fitting per fold; validation fold never snooped | **PASSED** |
| **6. Class & Sample Weights** | Loss weighting & boosting weights | Inverse class weights computed exclusively from train fold class counts | Verified fold-specific weights $w_c = \frac{N_{train}}{K \cdot N_{train,c}}$; no global snooping | **PASSED** |
| **7. Hyperparameters** | RF, XGBoost, P-MLP | Exact concordance with approved [model_pipeline_specs.json](file:///c:/Users/saipr/OneDrive/Desktop/EncryptedTrafficResearch/output/phase5/model_pipeline_specs.json) | 100% parameter match across all three model families | **PASSED** |
| **8. Artifact Integrity & Metrics** | OOF predictions, logs, runtimes | 100% prediction completeness, probability simplex, exact metric recalculation | 100% exact match on all metrics ($\Delta < 10^{-4}$); no missing predictions | **PASSED** |

---

## 1. Inspection of `scripts/run_phase6a_experiments.py`

Inspection of the primary training script ([scripts/run_phase6a_experiments.py](file:///c:/Users/saipr/OneDrive/Desktop/EncryptedTrafficResearch/scripts/run_phase6a_experiments.py)) confirms that the experimental pipeline was built with rigorous programmatic safeguards:

1. **Deterministic Random Seeds:**
   - Global seed constant `GLOBAL_SEED = 42` is defined at line 47.
   - The `set_seeds()` function resets Python `random.seed(42)`, `numpy.random.seed(42)`, `torch.manual_seed(42)`, and `torch.cuda.manual_seed_all(42)`.
   - `set_seeds()` is re-invoked immediately prior to every P-MLP fold training run (line 82), while `RandomForestClassifier` and `XGBClassifier` are explicitly constructed with `random_state=42`.
2. **Deterministic Computing Environment:**
   - Deep learning models were pinned to `torch.device("cpu")` (line 244), eliminating non-deterministic GPU kernel nondeterminism and ensuring CPU floating-point reproducibility.
3. **Programmatic Fail-Safe Assertions:**
   - Row alignment assertion between flows and folds: `assert len(df_flows) == len(df_folds)` and `assert (df_flows["flow_id"].values == df_folds["flow_id"].values).all()` (lines 182–183).
   - Missing value verification: `assert df_flows.isnull().sum().sum() == 0` (line 197).
   - Feature count assertion: `assert len(candidate_features) == 79` (line 207).
   - Real-time group overlap assertion: `assert len(overlap) == 0` (line 217).
   - Class coverage assertion: `assert val_classes == set(classes)` (line 219).
   - Prediction completeness assertion: `assert (oof_preds >= 0).all()` (line 440).

---

## 2. Metadata Exclusion Audit

A primary failure mode in network traffic classification research is leakage of session identifiers, IP addresses, port numbers, or timestamps, which allows machine learning models to "cheat" by memorizing socket pairs rather than learning behavioral flow dynamics.

We audited the dataset columns in [output/phase5/track_a_flows.csv](file:///c:/Users/saipr/OneDrive/Desktop/EncryptedTrafficResearch/output/phase5/track_a_flows.csv), [output/phase5/track_b_flows.csv](file:///c:/Users/saipr/OneDrive/Desktop/EncryptedTrafficResearch/output/phase5/track_b_flows.csv), and the active feature records in [output/phase6/track_a/feature_list.csv](file:///c:/Users/saipr/OneDrive/Desktop/EncryptedTrafficResearch/output/phase6/track_a/feature_list.csv) and [output/phase6/track_b/feature_list.csv](file:///c:/Users/saipr/OneDrive/Desktop/EncryptedTrafficResearch/output/phase6/track_b/feature_list.csv).

### Target 14 Metadata Columns
The 14 designated non-behavioral columns are:
1. `flow_id` (Unique identifier)
2. `capture_filename` (Source PCAP file)
3. `capture_group` (Session/session-group partition label)
4. `application` (Fine-grained ground-truth application name)
5. `is_vpn` (Binary VPN tunnel indicator)
6. `qos_class` (Ground-truth classification target)
7. `label_confidence` (Mapping confidence score)
8. `src_ip` (Source IPv4 address)
9. `src_port` (Source transport port)
10. `dst_ip` (Destination IPv4 address)
11. `dst_port` (Destination transport port)
12. `protocol_name` (Textual transport protocol name)
13. `flow_start_time` (Unix start epoch timestamp)
14. `flow_end_time` (Unix end epoch timestamp)

### Audit Verification Results

```
Total Columns in Flow Datasets: 117
├── Metadata & Target Columns: 14 (Excluded from training)
├── Candidate Full-Flow Behavioral Features: 79 (Evaluated in Phase 6A)
└── Early-Packet Features (early_*): 24 (Excluded from Phase 6A; reserved for Phase 6B)
```

1. **Candidate Exclusion:** Every one of the 14 metadata columns was confirmed absent from `candidate_features`.
2. **Early-Packet Feature Exclusion:** All 24 `early_*` features (`early_5_*`, `early_10_*`, `early_15_*`, `early_20_*`) were confirmed absent from `candidate_features`.
3. **Active Feature Inspection:** We parsed every line of `feature_list.csv` across all folds:
   - Track A (Folds 0, 1, 2): Zero metadata columns, zero early-packet features found in active features.
   - Track B (Folds 0, 1, 2, 3, 4): Zero metadata columns, zero early-packet features found in active features.

**Audit Finding:** 100% compliance. Models operated exclusively on statistical and temporal transport-layer behavioral features.

---

## 3. Group Leakage & Mathematical Disjointness Verification

In network traffic analysis, flows from the same capture session share identical temporal conditions, background noise, and host-level network configurations. Evaluating a model on flows from a capture group present in the training set results in severe optimistic bias.

Phase 6A utilized **StratifiedGroupKFold** to ensure complete session independence. We independently re-extracted the set of capture groups in the training partition $\mathcal{G}_{train}^{(k)}$ and validation partition $\mathcal{G}_{val}^{(k)}$ for every fold $k$, computing the exact set intersection $\mathcal{G}_{train}^{(k)} \cap \mathcal{G}_{val}^{(k)}$.

### Track A Group Disjointness Proof (3 Folds, 32 Capture Groups)

| Fold | Train Groups ($|\mathcal{G}_{train}|$) | Val Groups ($|\mathcal{G}_{val}|$) | Overlap ($\mathcal{G}_{train} \cap \mathcal{G}_{val}$) | Validation Group Assignments | Status |
|:---:|:---:|:---:|:---:|---|:---:|
| **Fold 0** | 25 | 7 | $\emptyset$ (0) | `GRP_FTPS_UP_2`, `GRP_HANGOUTS_AUDIO_1`, `GRP_SCP_1`, `GRP_SCP_6`, `GRP_SKYPE_VIDEO_1`, `GRP_VIMEO_1`, `GRP_VPN_EMAIL_2` | **STRICTLY DISJOINT** |
| **Fold 1** | 23 | 9 | $\emptyset$ (0) | `GRP_EMAIL_1`, `GRP_FB_AUDIO_1`, `GRP_NETFLIX_1`, `GRP_SCP_2`, `GRP_VOIPBUSTER_1`, `GRP_VPN_FB_AUDIO_2`, `GRP_VPN_HANGOUTS_AUDIO_1`, `GRP_VPN_SPOTIFY_A`, `GRP_VPN_YOUTUBE_A` | **STRICTLY DISJOINT** |
| **Fold 2** | 16 | 16 | $\emptyset$ (0) | `GRP_EMAIL_2`, `GRP_FB_VIDEO_1`, `GRP_FTPS_DOWN_1`, `GRP_HANGOUTS_VIDEO_1`, `GRP_SCP_3`, `GRP_SCP_4`, `GRP_SCP_5`, `GRP_SKYPE_AUDIO_1`, `GRP_SPOTIFY_1`, `GRP_VPN_FTPS_A`, `GRP_VPN_FTPS_B`, `GRP_VPN_NETFLIX_A`, `GRP_VPN_SKYPE_AUDIO_1`, `GRP_VPN_VIMEO_A`, `GRP_VPN_VOIPBUSTER_1`, `GRP_YOUTUBE_1` | **STRICTLY DISJOINT** |

$$\bigcup_{k=0}^{2} \mathcal{G}_{val}^{(k)} = \mathcal{G}_{total} \quad (7 + 9 + 16 = 32 \text{ groups}), \quad \mathcal{G}_{val}^{(i)} \cap \mathcal{G}_{val}^{(j)} = \emptyset \quad \forall i \neq j$$

### Track B Group Disjointness Proof (5 Folds, 29 Capture Groups)

| Fold | Train Groups ($|\mathcal{G}_{train}|$) | Val Groups ($|\mathcal{G}_{val}|$) | Overlap ($\mathcal{G}_{train} \cap \mathcal{G}_{val}$) | Validation Group Assignments | Status |
|:---:|:---:|:---:|:---:|---|:---:|
| **Fold 0** | 21 | 8 | $\emptyset$ (0) | `GRP_FTPS_UP_2`, `GRP_HANGOUTS_VIDEO_1`, `GRP_NETFLIX_1`, `GRP_SCP_6`, `GRP_SKYPE_VIDEO_1`, `GRP_VIMEO_1`, `GRP_VPN_SKYPE_AUDIO_1`, `GRP_YOUTUBE_1` | **STRICTLY DISJOINT** |
| **Fold 1** | 24 | 5 | $\emptyset$ (0) | `GRP_HANGOUTS_AUDIO_1`, `GRP_VPN_FTPS_A`, `GRP_VPN_NETFLIX_A`, `GRP_VPN_VIMEO_A`, `GRP_VPN_YOUTUBE_A` | **STRICTLY DISJOINT** |
| **Fold 2** | 23 | 6 | $\emptyset$ (0) | `GRP_FB_AUDIO_1`, `GRP_SCP_3`, `GRP_SPOTIFY_1`, `GRP_VPN_FTPS_B`, `GRP_VPN_HANGOUTS_AUDIO_1`, `GRP_VPN_VOIPBUSTER_1` | **STRICTLY DISJOINT** |
| **Fold 3** | 24 | 5 | $\emptyset$ (0) | `GRP_SCP_1`, `GRP_SCP_5`, `GRP_VOIPBUSTER_1`, `GRP_VPN_FB_AUDIO_2`, `GRP_VPN_SPOTIFY_A` | **STRICTLY DISJOINT** |
| **Fold 4** | 24 | 5 | $\emptyset$ (0) | `GRP_FB_VIDEO_1`, `GRP_FTPS_DOWN_1`, `GRP_SCP_2`, `GRP_SCP_4`, `GRP_SKYPE_AUDIO_1` | **STRICTLY DISJOINT** |

$$\bigcup_{k=0}^{4} \mathcal{G}_{val}^{(k)} = \mathcal{G}_{total} \quad (8 + 5 + 6 + 5 + 5 = 29 \text{ groups}), \quad \mathcal{G}_{val}^{(i)} \cap \mathcal{G}_{val}^{(j)} = \emptyset \quad \forall i \neq j$$

**Audit Finding:** Group overlap is identically zero ($\emptyset$) across all folds in both tracks. No session-level contamination occurred.

---

## 4. Fold Assignments and Sample Counts Alignment

We verified the integrity of fold assignments between the flow datasets and fold split files:
- [output/phase5/track_a_flows.csv](file:///c:/Users/saipr/OneDrive/Desktop/EncryptedTrafficResearch/output/phase5/track_a_flows.csv) vs [output/phase5/track_a_folds.csv](file:///c:/Users/saipr/OneDrive/Desktop/EncryptedTrafficResearch/output/phase5/track_a_folds.csv)
- [output/phase5/track_b_flows.csv](file:///c:/Users/saipr/OneDrive/Desktop/EncryptedTrafficResearch/output/phase5/track_b_flows.csv) vs [output/phase5/track_b_folds.csv](file:///c:/Users/saipr/OneDrive/Desktop/EncryptedTrafficResearch/output/phase5/track_b_folds.csv)

### Alignment Verification
1. **Row Count Match:**
   - Track A: Exactly 17,617 rows in both `track_a_flows.csv` and `track_a_folds.csv`. Matches original `output/flow_features.csv` (17,617 rows).
   - Track B: Exactly 10,770 rows in both `track_b_flows.csv` and `track_b_folds.csv`. (10,770 flows = 17,617 total minus 6,847 low-confidence Browsing flows).
2. **1-to-1 Index Alignment:**
   - Verified that $(flows["flow\_id"] == folds["flow\_id"]).all()$ evaluates to `True` across all 17,617 rows (Track A) and 10,770 rows (Track B).

### Fold Sample Breakdown & Class Coverage

#### Track A (4-Class Exploratory)

| Fold | Train Samples | Validation Samples | Total | Browsing | Bulk Data | Real-Time | Streaming |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Fold 0** | 15,992 | 1,625 | 17,617 | 305 (18.8%) | 224 (13.8%) | 966 (59.4%) | 130 (8.0%) |
| **Fold 1** | 5,940 | 11,677 | 17,617 | 5,870 (50.3%) | 11 (0.1%) | 5,321 (45.6%) | 475 (4.1%) |
| **Fold 2** | 13,302 | 4,315 | 17,617 | 672 (15.6%) | 1,113 (25.8%) | 1,845 (42.8%) | 685 (15.9%) |
| **Total** | — | **17,617** | **17,617** | **6,847** | **1,348** | **8,132** | **1,290** |

#### Track B (3-Class Benchmark)

| Fold | Train Samples | Validation Samples | Total | Bulk Data | Real-Time | Streaming |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Fold 0** | 9,047 | 1,723 | 10,770 | 210 (12.2%) | 1,101 (63.9%) | 412 (23.9%) |
| **Fold 1** | 9,373 | 1,397 | 10,770 | 74 (5.3%) | 907 (64.9%) | 416 (29.8%) |
| **Fold 2** | 6,513 | 4,257 | 10,770 | 63 (1.5%) | 4,078 (95.8%) | 116 (2.7%) |
| **Fold 3** | 8,638 | 2,132 | 10,770 | 30 (1.4%) | 1,943 (91.1%) | 159 (7.5%) |
| **Fold 4** | 9,509 | 1,261 | 10,770 | 971 (77.0%) | 103 (8.2%) | 187 (14.8%) |
| **Total** | — | **10,770** | **10,770** | **1,348** | **8,132** | **1,290** |

**Audit Finding:** All target classes are present in every single validation fold ($count > 0$), satisfying the stratification prerequisite.

---

## 5. Dynamic Preprocessing & Data Snooping Audit

A critical leak in many published studies is calculating global feature statistics (such as variance thresholds, correlation matrices, or scaler means and variances) across the entire dataset prior to cross-validation.

In [run_phase6a_experiments.py](file:///c:/Users/saipr/OneDrive/Desktop/EncryptedTrafficResearch/scripts/run_phase6a_experiments.py), function `fit_transform_fold_features()` (lines 142–165) operates strictly on `X_train_df`:
- Constant features are identified via `X_train_df.std(axis=0) == 0.0`.
- Collinear features are identified via `X_train_clean.corr().abs() >= 0.999`.
- Identical dropped column lists are subsequently removed from `X_val_df`.
- For P-MLP, `StandardScaler().fit_transform(X_train_clean)` fits means and scales solely on the training partition, followed by `scaler.transform(X_val_clean)`.

### Independent Re-Simulation Verification

We independently re-implemented the exact preprocessing steps from raw features and compared our outputs with the saved records in `feature_list.csv`:

| Track | Fold | Re-Simulated Dropped Constants | Re-Simulated Dropped Collinear ($|r| \ge 0.999$) | Active Features | Matches `feature_list.csv`? |
|---|:---:|---|---|:---:|:---:|
| **Track A** | Fold 0 | `tcp_urg_count` | `bwd_burst_count` | 77 | **100% Match** |
| **Track A** | Fold 1 | `tcp_urg_count` | `bwd_burst_count` | 77 | **100% Match** |
| **Track A** | Fold 2 | `tcp_urg_count` | `fwd_iat_max`, `bwd_burst_count`, `fwd_burst_pkts_mean` | 75 | **100% Match** |
| **Track B** | Fold 0 | `tcp_urg_count` | `bwd_bytes`, `bwd_burst_count` | 76 | **100% Match** |
| **Track B** | Fold 1 | `tcp_urg_count` | `bwd_burst_count` | 77 | **100% Match** |
| **Track B** | Fold 2 | `tcp_urg_count` | `bwd_burst_count` | 77 | **100% Match** |
| **Track B** | Fold 3 | `tcp_urg_count` | `bwd_burst_count` | 77 | **100% Match** |
| **Track B** | Fold 4 | `tcp_urg_count` | `fwd_iat_max`, `bwd_burst_count` | 76 | **100% Match** |

### Proof of Dynamic Adaptation (Absence of Global Snooping)
- In Track A Fold 2, training split variations caused `fwd_iat_max` and `fwd_burst_pkts_mean` to exceed the 0.999 collinearity threshold, resulting in **75 active features**.
- In Track B Fold 0, `bwd_bytes` became collinear ($r \ge 0.999$) with `total_bytes` on the training fold, resulting in **76 active features**.
- In Track B Fold 4, `fwd_iat_max` became collinear ($r \ge 0.999$) with `bidir_iat_max` on the training fold, resulting in **76 active features**.

If the pipeline had snooped on the entire dataset, the dropped columns and active feature counts would have been identical across all folds. This variation provides conclusive mathematical proof of dynamic, leak-free training fold adaptation.

---

## 6. Class Weights & Sample Weights Derivation Audit

Both tracks exhibit substantial class imbalance (Track A: Real-Time 8,132 vs Streaming 1,290, 6.3:1; Track B: Real-Time 8,132 vs Streaming 1,290, 6.3:1).

To avoid majority-class bias without corrupting flow distributions via synthetic oversampling (SMOTE), Phase 6A specified inverse frequency weighting:
$$w_c = \frac{N_{train}}{K \cdot N_{train, c}}$$
where $N_{train}$ is the number of samples in the training fold, $K$ is the number of classes, and $N_{train, c}$ is the count of class $c$ in the training partition.

### Verification of Implementation
In [run_phase6a_experiments.py](file:///c:/Users/saipr/OneDrive/Desktop/EncryptedTrafficResearch/scripts/run_phase6a_experiments.py) (lines 274–279):
```python
classes_in_train, counts_in_train = np.unique(y_train, return_counts=True)
total_train_samples = len(y_train)
class_weights_dict = {c: total_train_samples / (num_classes * count) for c, count in zip(classes_in_train, counts_in_train)}
class_weights_arr = np.array([class_weights_dict[i] for i in range(num_classes)], dtype=np.float32)
sample_weights_train = np.array([class_weights_dict[y] for y in y_train], dtype=np.float32)
```

### Verification Against Global Snooping

We calculated the hypothetical class weights derived from the whole dataset versus the actual weights computed per fold:

#### Track A Class Weights

| Fold | Browsing ($w_0$) | Bulk Data ($w_1$) | Real-Time ($w_2$) | Streaming ($w_3$) | Max $|\Delta|$ vs Global |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **Global (Snooped)** | *0.6432* | *3.2672* | *0.5416* | *3.4141* | — |
| **Fold 0** | 0.6111 | 3.5569 | 0.5579 | 3.4466 | 0.2897 |
| **Fold 1** | 1.5200 | 1.1107 | 0.5283 | 1.8221 | 2.1566 |
| **Fold 2** | 0.5385 | 14.1511 | 0.5289 | 5.4967 | 10.8838 |

#### Track B Class Weights

| Fold | Bulk Data ($w_0$) | Real-Time ($w_1$) | Streaming ($w_2$) | Max $|\Delta|$ vs Global |
|:---:|:---:|:---:|:---:|:---:|
| **Global (Snooped)** | *2.6632* | *0.4415* | *2.7829* | — |
| **Fold 0** | 2.6500 | 0.4289 | 3.4347 | 0.6518 |
| **Fold 1** | 2.4524 | 0.4324 | 3.5748 | 0.7918 |
| **Fold 2** | 1.6895 | 0.5355 | 1.8492 | 0.9737 |
| **Fold 3** | 2.1846 | 0.4652 | 2.5458 | 0.4786 |
| **Fold 4** | 8.4076 | 0.3948 | 2.8737 | 5.7444 |

**Audit Finding:** Class weights and per-sample weights were derived strictly from the training split of each fold, never incorporating validation fold sample counts.

---

## 7. Model Hyperparameters Audit Against Approved Specifications

We compared the exact instantiation parameters in [scripts/run_phase6a_experiments.py](file:///c:/Users/saipr/OneDrive/Desktop/EncryptedTrafficResearch/scripts/run_phase6a_experiments.py) with the approved specification file [output/phase5/model_pipeline_specs.json](file:///c:/Users/saipr/OneDrive/Desktop/EncryptedTrafficResearch/output/phase5/model_pipeline_specs.json).

### Detailed Hyperparameter Comparison Matrix

| Model | Parameter | Approved Spec ([model_pipeline_specs.json](file:///c:/Users/saipr/OneDrive/Desktop/EncryptedTrafficResearch/output/phase5/model_pipeline_specs.json)) | Implemented in Script ([run_phase6a_experiments.py](file:///c:/Users/saipr/OneDrive/Desktop/EncryptedTrafficResearch/scripts/run_phase6a_experiments.py)) | Verification Status |
|---|---|---|---|:---:|
| **Random Forest** | `n_estimators` | `100` | `100` (line 290) | **EXACT MATCH** |
| | `criterion` | `"gini"` | `"gini"` (line 291) | **EXACT MATCH** |
| | `class_weight` | `"balanced"` | `"balanced"` (line 292) | **EXACT MATCH** |
| | `random_state` | `42` | `GLOBAL_SEED = 42` (line 293) | **EXACT MATCH** |
| | `n_jobs` | `-1` | `-1` (line 294) | **EXACT MATCH** |
| **XGBoost** | `n_estimators` | `100` | `100` (line 306) | **EXACT MATCH** |
| | `learning_rate` | `0.1` | `0.1` (line 307) | **EXACT MATCH** |
| | `max_depth` | `6` | `6` (line 308) | **EXACT MATCH** |
| | `subsample` | `0.8` | `0.8` (line 309) | **EXACT MATCH** |
| | `colsample_bytree`| `0.8` | `0.8` (line 310) | **EXACT MATCH** |
| | `objective` | `"multi:softprob"` | `"multi:softprob"` (line 311) | **EXACT MATCH** |
| | `eval_metric` | `"mlogloss"` | `"mlogloss"` (line 312) | **EXACT MATCH** |
| | `random_state` | `42` | `GLOBAL_SEED = 42` (line 313) | **EXACT MATCH** |
| | `n_jobs` | `-1` | `-1` (line 314) | **EXACT MATCH** |
| | `sample_weight` | Inverse training frequency | `sample_weights_train` (line 316) | **EXACT MATCH** |
| **P-MLP** | Layer 1 | `Linear(in_features, 128)` | `nn.Linear(in_features, 128)` (line 63) | **EXACT MATCH** |
| | Block 1 | `BatchNorm1d(128)`, `ReLU()`, `Dropout(0.3)` | Lines 64–66 | **EXACT MATCH** |
| | Layer 2 | `Linear(128, 64)` | `nn.Linear(128, 64)` (line 67) | **EXACT MATCH** |
| | Block 2 | `BatchNorm1d(64)`, `ReLU()`, `Dropout(0.3)` | Lines 68–70 | **EXACT MATCH** |
| | Layer 3 | `Linear(64, 32)` | `nn.Linear(64, 32)` (line 71) | **EXACT MATCH** |
| | Block 3 | `BatchNorm1d(32)`, `ReLU()`, `Dropout(0.2)` | Lines 72–74 | **EXACT MATCH** |
| | Output Layer | `Linear(32, num_classes)` | `nn.Linear(32, num_classes)` (line 75) | **EXACT MATCH** |
| | Loss Function | `CrossEntropyLoss(weight=class_weights)` | `nn.CrossEntropyLoss(weight=...)` (line 84) | **EXACT MATCH** |
| | Optimizer | `Adam(lr=0.001, weight_decay=1e-4)` | `optim.Adam(..., lr=0.001, weight_decay=1e-4)` (line 85) | **EXACT MATCH** |
| | Scheduler | `ReduceLROnPlateau(mode='min', factor=0.5, patience=5)` | `ReduceLROnPlateau(..., factor=0.5, patience=5)` (line 86) | **EXACT MATCH** |
| | Batch Size | `64` | `batch_size=64` (line 81, 91) | **EXACT MATCH** |
| | Max Epochs | `100` | `max_epochs=100` (line 81, 99) | **EXACT MATCH** |
| | Early Stopping | `patience=15` on `val_loss` | `patience=15` (line 81, 131) | **EXACT MATCH** |

**Audit Finding:** 100% adherence to approved model pipeline specifications. No unauthorized hyperparameter tuning or deviations occurred.

---

## 8. Artifact Integrity, Training Logs, and Metric Recalculation

All saved artifacts in [output/phase6/track_a/](file:///c:/Users/saipr/OneDrive/Desktop/EncryptedTrafficResearch/output/phase6/track_a/) and [output/phase6/track_b/](file:///c:/Users/saipr/OneDrive/Desktop/EncryptedTrafficResearch/output/phase6/track_b/) were inspected and independently recalculated.

### 8.1 Out-Of-Fold (OOF) Prediction Matrix Audit
File: `out_of_fold_predictions.csv`
1. **Completeness:**
   - Track A: Exactly 17,617 rows. Every flow has non-null predictions for `pred_Random_Forest`, `pred_XGBoost`, and `pred_P_MLP`. Zero unpredicted flows.
   - Track B: Exactly 10,770 rows. Every flow has non-null predictions for `pred_Random_Forest`, `pred_XGBoost`, and `pred_P_MLP`. Zero unpredicted flows.
2. **Probability Simplex Verification:**
   - For all models and all rows, the predicted class probabilities sum to $1.0 \pm 10^{-4}$:
     $$\sum_{c} P(y = c \mid \mathbf{x}) = 1.0000$$

### 8.2 Independent Metric Recalculation Verification

We recalculated all primary and secondary evaluation metrics directly from the raw ground-truth labels and predicted labels in `out_of_fold_predictions.csv` using `scikit-learn` independent functions. We then performed a bitwise comparison against `fold_metrics.csv` and `aggregate_metrics.csv`.

#### Track A Metric Recalculation Comparison

| Model | Evaluation Scope | Reported Macro F1 | Recalculated Macro F1 | Absolute Difference ($|\Delta|$) | Reported Bal Acc | Recalculated Bal Acc | Absolute Difference ($|\Delta|$) | Verification |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Random Forest** | Fold Mean | 0.3428 | 0.3428 | $< 10^{-4}$ | 0.4734 | 0.4734 | $< 10^{-4}$ | **EXACT MATCH** |
| | Pooled OOF | 0.2864 | 0.2864 | $< 10^{-4}$ | 0.3700 | 0.3700 | $< 10^{-4}$ | **EXACT MATCH** |
| **XGBoost** | Fold Mean | 0.3809 | 0.3809 | $< 10^{-4}$ | 0.5476 | 0.5476 | $< 10^{-4}$ | **EXACT MATCH** |
| | Pooled OOF | 0.3163 | 0.3163 | $< 10^{-4}$ | 0.4380 | 0.4380 | $< 10^{-4}$ | **EXACT MATCH** |
| **P-MLP** | Fold Mean | 0.3135 | 0.3135 | $< 10^{-4}$ | 0.4244 | 0.4244 | $< 10^{-4}$ | **EXACT MATCH** |
| | Pooled OOF | 0.3707 | 0.3707 | $< 10^{-4}$ | 0.4929 | 0.4929 | $< 10^{-4}$ | **EXACT MATCH** |

#### Track B Metric Recalculation Comparison

| Model | Evaluation Scope | Reported Macro F1 | Recalculated Macro F1 | Absolute Difference ($|\Delta|$) | Reported Bal Acc | Recalculated Bal Acc | Absolute Difference ($|\Delta|$) | Verification |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Random Forest** | Fold Mean | 0.3970 | 0.3970 | $< 10^{-4}$ | 0.5425 | 0.5425 | $< 10^{-4}$ | **EXACT MATCH** |
| | Pooled OOF | 0.4664 | 0.4664 | $< 10^{-4}$ | 0.5051 | 0.5051 | $< 10^{-4}$ | **EXACT MATCH** |
| **XGBoost** | Fold Mean | 0.4564 | 0.4564 | $< 10^{-4}$ | 0.6072 | 0.6072 | $< 10^{-4}$ | **EXACT MATCH** |
| | Pooled OOF | 0.5452 | 0.5452 | $< 10^{-4}$ | 0.6349 | 0.6349 | $< 10^{-4}$ | **EXACT MATCH** |
| **P-MLP** | Fold Mean | 0.4069 | 0.4069 | $< 10^{-4}$ | 0.5454 | 0.5454 | $< 10^{-4}$ | **EXACT MATCH** |
| | Pooled OOF | 0.5567 | 0.5567 | $< 10^{-4}$ | 0.6823 | 0.6823 | $< 10^{-4}$ | **EXACT MATCH** |

### 8.3 Training Log Analysis (P-MLP Convergence & Early Stopping)
File: `training_log.csv`

Audit of the epoch-by-epoch loss trajectories confirms expected deep learning convergence behavior:
- **Track A (58 Total Epochs Across 3 Folds):**
  - Fold 0: Reached lowest validation loss at epoch 10 (`val_loss = 0.9784`), early-stopped at epoch 25 after 15 consecutive epochs without improvement.
  - Fold 1: Minimum validation loss at epoch 1 (`val_loss = 1.7090`), early-stopped at epoch 16.
  - Fold 2: Minimum validation loss at epoch 2 (`val_loss = 1.7173`), early-stopped at epoch 17.
- **Track B (113 Total Epochs Across 5 Folds):**
  - Fold 0: Minimum validation loss at epoch 1 (`val_loss = 1.0945`), early-stopped at epoch 16.
  - Fold 1: Minimum validation loss at epoch 5 (`val_loss = 0.5762`), early-stopped at epoch 20.
  - Fold 2: Minimum validation loss at epoch 22 (`val_loss = 0.4937`), early-stopped at epoch 37.
  - Fold 3: Minimum validation loss at epoch 1 (`val_loss = 1.0308`), early-stopped at epoch 16.
  - Fold 4: Minimum validation loss at epoch 9 (`val_loss = 0.7816`), early-stopped at epoch 24.

In every case, the script restored `best_model_state` prior to computing test fold predictions, preventing overfitted weights from contaminating evaluation.

### 8.4 Runtime & Latency Audit
File: `runtime_summary.csv`

| Track | Model | Average Training Time (s) | Average Prediction Time (s) | Per-Sample Inference Latency ($\mu$s) | Line-Rate Deployment Suitability |
|---|---|:---:|:---:|:---:|:---:|
| **Track A** | Random Forest | 0.38 s | 0.0429 s | 11.83 $\mu$s | High-throughput software switch (84,500 flows/sec) |
| | XGBoost | 1.44 s | 0.0114 s | 2.13 $\mu$s | Edge gateway line-rate (469,000 flows/sec) |
| | P-MLP | 44.37 s | 0.0091 s | 2.46 $\mu$s | Edge gateway line-rate (406,000 flows/sec) |
| **Track B** | Random Forest | 0.49 s | 0.0707 s | 39.09 $\mu$s | High-throughput software switch (25,500 flows/sec) |
| | XGBoost | 1.70 s | 0.0068 s | 3.38 $\mu$s | Edge gateway line-rate (295,000 flows/sec) |
| | P-MLP | 26.32 s | 0.0034 s | 1.78 $\mu$s | Edge gateway line-rate (561,000 flows/sec) |

---

## 9. Independent Audit Conclusions & Research Sign-Off

The comprehensive independent audit confirms that Phase 6A was conducted under flawless methodological hygiene:
1. **Zero Information Leakage:** Neither metadata identifiers, session groups, validation statistical moments, nor validation class distributions leaked into the training process.
2. **Honest Baseline Performance:** Performance figures (Macro F1: Track A XGBoost 0.3809, Track B XGBoost 0.4564 / P-MLP 0.4069) represent genuine, uninflated generalization capabilities on completely unseen network capture sessions.
3. **Artifact Completeness & Reproducibility:** Every generated table, metric, log, and prediction matches the underlying code and input data with zero discrepancies.

The experimental setup is formally validated, and the project is fully cleared to proceed to **Phase 6B (Early-Packet Machine Learning Experiments)**.

---
*Audit Completed on: November 21, 2026*  
*Independent Audit Verification Script:* [scripts/audit_phase6a_independent.py](file:///c:/Users/saipr/OneDrive/Desktop/EncryptedTrafficResearch/scripts/audit_phase6a_independent.py)  
*Audited Execution Script:* [scripts/run_phase6a_experiments.py](file:///c:/Users/saipr/OneDrive/Desktop/EncryptedTrafficResearch/scripts/run_phase6a_experiments.py)
