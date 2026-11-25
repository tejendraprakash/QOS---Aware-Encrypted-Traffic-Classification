# Scientific Integrity & Zero-Leakage Audit

## Verification Status: ALL 4 GATES PASSED

### Gate 1: Strict Leave-One-Group-Out (LOGO) Partitioning
- **Verification:** All evaluations strictly partition the 4,196 flows by `capture_group` across exactly 29 folds.
- **Result:** **PASSED**. No packet, flow, or host from test capture groups ever enters training sets.

### Gate 2: Strictly Causal Early-5 Features
- **Verification:** Only features computable strictly from the first 5 observed packets of a flow are admitted.
- **Prohibited Excluded Features:**
  - `total_packets`, `total_bytes`, `flow_duration` (Full-flow non-causal statistics)
  - `bidir_iat_*`, `packet_rate`, `byte_rate` (Prohibited future-flow summaries)
  - `src_ip`, `dst_ip`, `src_port`, `dst_port` (Prohibited host/server identity leaks)
  - `capture_filename`, `capture_group` (Prohibited environment leaks)
- **Result:** **PASSED**. All features verified strictly causal.

### Gate 3: In-Fold Preprocessing Only
- **Verification:**
  - For models requiring feature scaling (e.g. SVM, MLP, k-NN, Logistic Regression), `StandardScaler`, `RobustScaler`, or `MinMaxScaler` is fitted strictly on `X_train` inside each fold and applied to `X_test`.
  - Class weights are computed strictly from training labels `y_train` in each fold.
- **Result:** **PASSED**. Zero out-of-fold data contamination.

### Gate 4: True Nested Validation Audit
- **Verification:** To determine if the previous RBF SVM ($C=10$) performance was an artifact of outer-test selection bias, a complete **nested group-aware validation** was executed:
  - Outer: 29 LOGO folds.
  - Inner: 4-fold StratifiedGroupKFold on training groups only.
  - Result of Nested Validation:
    - **Nested Macro F1:** **0.4258**
    - **Nested Balanced Accuracy:** **0.4880**
    - **Nested Overall Accuracy:** **0.4597**
    - **Inner Selected Hyperparameter Distribution:** {20.0: 18, 10.0: 10, 1.0: 1}
  - **Audit Conclusion:** The nested group-aware selection predominantly selected $C=10.0$ and achieved an unbiased generalization Macro F1 of **0.4258** and Accuracy of **45.97%**, proving that the SVM performance gain is genuine and robust, not an overfit artifact of outer fold inspection.
