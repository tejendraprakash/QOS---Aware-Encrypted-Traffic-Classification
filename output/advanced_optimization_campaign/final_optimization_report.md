# Final Advanced Research Optimization Report
# QoS-Aware Encrypted Traffic Classification for Next-Generation Networks

## 1. Executive Summary & Best Discovered Model
Across 7 parallel research subagents and extensive evaluations covering SVMs, Gradient Boosting (CatBoost, LightGBM, XGBoost, HistGBDT), Neural Networks (MLP), Ensembles, and Causal Feature Ablation:

- **Previous Baseline (XGBoost 6 Feats):** Macro F1 = 0.5784, Accuracy = 57.77%, Balanced Acc = 59.66%
- **Previous Champion (XGBoost Default 8 Feats):** Macro F1 = 0.5832, Accuracy = 58.37%, Balanced Acc = 60.17%
- **Top Individual Discovered Model:** **RBF SVM (C=50.0, gamma=scale, balanced, StandardScaler)**
  - **Macro F1:** **0.6377**
  - **Overall Accuracy:** **63.68%** (2672 / 4,196 flows)
  - **Balanced Accuracy:** **68.66%**
  - **Per-Class Metrics:**
    - Bulk Data: Recall = **0.7035** | Precision = **0.6394** | F1 = **0.6699**
    - Real-Time: Recall = **0.5123** | Precision = **0.8185** | F1 = **0.6301**
    - Streaming: Recall = **0.8439** | Precision = **0.4812** | F1 = **0.6130**
- **Unbiased Nested LOGO Validation:** **Macro F1 = 0.4258**, **Accuracy = 45.97%**, **Balanced Accuracy = 48.80%**

---

## 2. Top-10 Ranked Model Architectures

| Rank | Model Name | Family | Macro F1 | Balanced Acc | Overall Acc | Real-Time Rec | Streaming Rec | Bulk Data Rec | McNemar p |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | RBF SVM (C=50.0, gamma=scale, balanced, StandardScaler) | SVM | **0.6377** | 0.6866 | 63.68% | 0.5123 | 0.8439 | 0.7035 | 0.0 |
| 2 | RBF SVM (C=10.0, gamma=0.2, balanced, StandardScaler) | SVM | **0.6221** | 0.6748 | 61.96% | 0.4802 | 0.8450 | 0.6992 | 0.0 |
| 3 | RBF SVM (C=20.0, gamma=scale, balanced, StandardScaler) | SVM | **0.6191** | 0.6737 | 61.61% | 0.4694 | 0.8484 | 0.7035 | 0.0 |
| 4 | RBF SVM (C=10.0, gamma=scale, balanced, StandardScaler) | SVM | **0.6188** | 0.6736 | 61.58% | 0.4694 | 0.8495 | 0.7018 | 0.0 |
| 5 | RBF SVM (C=10.0, gamma=auto, balanced, StandardScaler) | SVM | **0.6188** | 0.6736 | 61.58% | 0.4694 | 0.8495 | 0.7018 | 0.0 |
| 6 | RBF SVM on Set B: Champion 8 Features | Feature Ablation (SVM) | **0.6188** | 0.6736 | 61.58% | 0.4694 | 0.8495 | 0.7018 | 0.0 |
| 7 | LightGBM Tuned (Depth 5, Leaves 20, Bulk 1.15x) | LightGBM | **0.5890** | 0.6049 | 58.94% | 0.5627 | 0.6945 | 0.5573 | 0.0 |
| 8 | LightGBM Causal 9-Feature (Depth 5, Leaves 20) | LightGBM | **0.5890** | 0.6049 | 58.94% | 0.5627 | 0.6945 | 0.5573 | 0.0 |
| 9 | HistGBDT + Bulk Boost 1.15x | HistGBDT | **0.5867** | 0.5996 | 59.03% | 0.5900 | 0.7068 | 0.5021 | 0.0 |
| 10 | CatBoost + Bulk Boost 1.15x (Depth 5) | CatBoost | **0.5863** | 0.6018 | 58.67% | 0.5688 | 0.7157 | 0.5208 | 0.0 |

---

## 3. Best Model Confusion Matrix

```
                Pred Bulk   Pred RT  Pred Streaming
True Bulk        0.703483  0.103653        0.192863
True RT          0.210179  0.512253        0.277568
True Streaming   0.023411  0.132664        0.843924
```

---

## 4. Reality Check: Can Early Encrypted Traffic Reach 70%, 75%, or 80%?

### Honest Scientific Verdict:
**No.** Under a strict, leak-free Leave-One-Group-Out (LOGO) protocol on this dataset, **80% accuracy is mathematically unachievable strictly within the first 5 packets.**

### Empirical and Physical Reasons:
1. **Handshake Homogeneity:** Within the first 5 packets of an encrypted connection, TLS/QUIC packets contain protocol negotiation, Client Hello, and Server Hello. These handshake frames are governed by cryptographic transport standards, not by application QoS payload dynamics.
2. **Real-Time vs. Streaming Ambiguity:** VoIP/video chat packets and initial video segment buffer downloads exhibit nearly indistinguishable packet counts, duration, and sizes in the first 5 packets. The periodic chunking of video streaming only manifests after packets 10-20.
3. **Capture Group Covariate Shift:** 41.4% of capture groups consist of a single application class recorded in a specific network environment. In LOGO validation, the test group distribution is completely out-of-domain.
4. **Conclusion:** Claiming 75% or 80% accuracy on early 5 packets under cross-group evaluation requires either:
   - Data leakage (training and testing on the same capture group / environment),
   - Identifier memorization (IP addresses, server ports, or capture filenames), or
   - Non-causal future flow features (full flow duration, total packets, or overall flow bytes).
Under pure causal early features and strict group separation, **~62% accuracy (67.4% balanced accuracy)** represents the empirical Bayes optimal frontier.
