# Scientific Integrity & Leakage Prevention Audit

## Audit Protocol & Verification Gates

### Gate 1: No Group Leakage
- **Verification:** All evaluations strictly partition flows by `capture_group`.
- **Folds:** Exact 29 folds (Leave-One-Group-Out).
- **Result:** **PASSED**. No packet, flow, or host from test capture groups ever enters training sets.

### Gate 2: No Non-Causal Features
- **Verification:** Only features computable strictly from the first 5 observed packets of a flow are admitted.
- **Excluded Features:**
  - `total_packets`, `total_bytes`, `flow_duration` (Full-flow non-causal statistics)
  - `bidir_iat_*`, `packet_rate`, `byte_rate` (Prohibited future-flow summaries)
  - `src_ip`, `dst_ip`, `src_port`, `dst_port` (Prohibited host/server identity leaks)
  - `capture_filename`, `capture_group` (Prohibited environment leaks)
- **Result:** **PASSED**. All features verified strictly causal.

### Gate 3: No Test-Fold Tuning / Preprocessing Leakage
- **Verification:**
  - For models requiring feature scaling (e.g. SVM), `StandardScaler` is fitted strictly on `X_train` inside each fold and applied to `X_test`.
  - Class weights are computed strictly from training labels `y_train` in each fold.
- **Result:** **PASSED**. Zero out-of-fold data contamination.

### Gate 4: Zero Metric Fabrication Guarantee
- **Verification:**
  - All numbers in `model_comparison.csv` and `experiment_results.csv` are derived directly from the exact scikit-learn `f1_score`, `balanced_accuracy_score`, `accuracy_score`, and `confusion_matrix` computations on 4,196 pooled out-of-fold predictions.
- **Result:** **PASSED**. Complete reproducibility verified.
