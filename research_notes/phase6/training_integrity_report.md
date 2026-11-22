# Phase 6A Training Integrity & Leakage Verification Audit

**Project:** QoS-Aware Encrypted Network Traffic Classification for Next-Generation Networks  
**Phase:** 6A — Comprehensive Full-Flow Machine Learning Baseline Experiment  
**Auditor:** Experimental Methodology Auditor & Lead Research Coordinator  
**Date:** November 2026

---

## 1. Experimental Integrity Checkpoints

This audit independently verifies compliance with the approved research directives, confirming that no methodological compromises or data leakages occurred during Phase 6A baseline execution.

| Integrity Checkpoint | Audit Verification Requirement | Observed Experimental Reality | Audit Status |
|---|---|---|:---:|
| **Original Dataset Immutability** | `output/flow_features.csv` must be completely untouched. | File modified date, size (17,617 rows, 117 cols), and SHA256 match Phase 2/4. | **PASSED** |
| **PCAP & Raw Data Integrity** | Raw PCAP files in `data/raw/` must not be altered, moved, or deleted. | All 37 PCAPs intact and read-only. | **PASSED** |
| **ISCX Dataset Isolation** | No ISCX files may be used for model training in this phase. | ISCX directory was completely segregated; 0 ISCX flows used. | **PASSED** |
| **No Synthetic Data Generation** | SMOTE, ADASYN, random oversampling strictly forbidden. | Exact sample counts trained: Track A = 17,617, Track B = 10,770. | **PASSED** |
| **No Hyperparameter Tuning** | No GridSearchCV, random search, or parameter adjustments. | Exact Phase 5 configurations applied across all models and folds. | **PASSED** |
| **No Early-Packet Leakage** | All 24 early-packet features excluded from Phase 6A inputs. | Exactly 79 candidate full-flow features used; 0 early features. | **PASSED** |
| **Metadata Exclusion** | 14 metadata/identity columns strictly excluded from feature matrix. | `flow_id`, `src_ip`, `dst_ip`, `src_port`, `dst_port`, `capture_group`, etc. excluded. | **PASSED** |
| **Group-Aware Disjointness** | Train groups $\cap$ Validation groups must be empty ($\emptyset$). | Overlap $= 0$ mathematically verified across all 8 folds. | **PASSED** |
| **Training-Only Feature Preprocessing** | Constant/collinear detection and scaling fitted on train fold only. | `std == 0.0`, correlation $|r| \ge 0.999$, and `StandardScaler` fitted on train fold only. | **PASSED** |
| **Class Weight Derivation** | Class weights / sample weights derived from train fold counts only. | $w_c = \frac{N_{train}}{K \cdot N_{train, c}}$ computed on training split only. | **PASSED** |

---

## 2. Mathematical Group Disjointness Proof

For every fold $k$, the set of capture groups assigned to the validation fold $\mathcal{G}_{val}^{(k)}$ was compared with the set of capture groups in the training fold $\mathcal{G}_{train}^{(k)}$:

$$\mathcal{G}_{train}^{(k)} \cap \mathcal{G}_{val}^{(k)} = \emptyset \quad \forall k \in \{0, \dots, K-1\}$$

### Track A (3 Folds, 32 Groups)
- **Fold 0:** 25 Train Groups $\cap$ 7 Val Groups = $\emptyset$ (0 overlap).
- **Fold 1:** 23 Train Groups $\cap$ 9 Val Groups = $\emptyset$ (0 overlap).
- **Fold 2:** 16 Train Groups $\cap$ 16 Val Groups = $\emptyset$ (0 overlap).

### Track B (5 Folds, 29 Groups)
- **Fold 0:** 21 Train Groups $\cap$ 8 Val Groups = $\emptyset$ (0 overlap).
- **Fold 1:** 24 Train Groups $\cap$ 5 Val Groups = $\emptyset$ (0 overlap).
- **Fold 2:** 23 Train Groups $\cap$ 6 Val Groups = $\emptyset$ (0 overlap).
- **Fold 3:** 24 Train Groups $\cap$ 5 Val Groups = $\emptyset$ (0 overlap).
- **Fold 4:** 24 Train Groups $\cap$ 5 Val Groups = $\emptyset$ (0 overlap).

**Conclusion:** Zero cross-fold session leakage occurred. Models were evaluated purely on unseen network sessions.

---

## 3. Dynamic Fold-Level Feature Selection Audit

The feature audit reported two anomalies in the candidate feature space:
1. `tcp_urg_count` is a zero-variance constant.
2. `bwd_burst_count` is perfectly correlated with `fwd_burst_count` ($r=1.0000$).

Our training pipeline strictly identified and eliminated these features **on each training partition independently**, never peeking at the validation fold:
- In all folds of Track A and Track B, `tcp_urg_count` was dynamically identified as having variance $\sigma^2 = 0.0$ on the training split and dropped.
- In all folds of Track A and Track B, `bwd_burst_count` was dynamically identified as having Pearson correlation $|r| \ge 0.999$ with `fwd_burst_count` on the training split and dropped.
- Exactly 77 active behavioral features were fed into the classifiers for every fold.

---

## 4. Failure Handling & Anomaly Documentation

No training crashes, CUDA out-of-memory errors, NaN loss divergences, or unhandled exceptions occurred. All 24 model-fold training runs executed to completion and converged successfully.
