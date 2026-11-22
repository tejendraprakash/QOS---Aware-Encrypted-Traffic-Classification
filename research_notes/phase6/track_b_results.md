# Track B Full-Flow Baseline Results: Primary 3-Class Benchmark

**Project:** QoS-Aware Encrypted Network Traffic Classification for Next-Generation Networks  
**Phase:** 6A — Comprehensive Full-Flow Machine Learning Baseline Experiment  
**Date:** November 2026  
**Artifact Directory:** [output/phase6/track_b/](file:///c:/Users/saipr/OneDrive/Desktop/EncryptedTrafficResearch/output/phase6/track_b/)  
**Input Dataset:** [output/phase5/track_b_flows.csv](file:///c:/Users/saipr/OneDrive/Desktop/EncryptedTrafficResearch/output/phase5/track_b_flows.csv) (10,770 flows, 117 columns)  
**Validation Scheme:** 5-Fold `StratifiedGroupKFold` (random_state = 2)

---

## 1. Executive Summary

Track B represents our **Primary Benchmark Experiment**, evaluating the three scientifically verified QoS classes:
1. **Bulk Data:** High-throughput, non-interactive file transfers (FTPS, SCP) — 1,348 flows (12.52%), 10 capture groups.
2. **Streaming:** Adaptive video and audio streaming (Netflix, Spotify, Vimeo, YouTube) — 1,290 flows (11.98%), 9 capture groups.
3. **Real-Time:** Low-latency conversational audio and video conferencing (Hangouts, Skype, VoIPBuster, Facebook Audio/Video) — 8,132 flows (75.51%), 10 capture groups.

With the contaminated Browsing class removed, all three classes have balanced capture group diversity (9–10 groups per class), enabling a standard, leak-free **5-Fold Stratified Group Cross-Validation**.

---

## 2. Quantitative Results Across Folds

### 2.1 Fold-Level Metric Summary

| Model | Fold | Val Samples | Val Groups | Macro F1 | Balanced Accuracy | Accuracy | Weighted F1 | Train Time (s) | Pred Time (s) | Latency ($\mu s$/sample) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Random Forest** | 0 | 1,723 | 8 | **0.6832** | 0.6664 | 0.7400 | 0.7490 | 0.44 | 0.050 | 28.9 |
| **Random Forest** | 1 | 1,397 | 5 | 0.3298 | 0.4143 | 0.3099 | 0.2714 | 0.48 | 0.062 | 44.3 |
| **Random Forest** | 2 | 4,257 | 6 | 0.4510 | 0.6402 | 0.8172 | 0.8385 | 0.50 | 0.076 | 17.9 |
| **Random Forest** | 3 | 2,132 | 5 | 0.3215 | 0.7174 | 0.4625 | 0.5482 | 0.49 | 0.083 | 39.0 |
| **Random Forest** | 4 | 1,261 | 5 | 0.1994 | 0.2742 | 0.2427 | 0.3414 | 0.54 | 0.083 | 65.4 |
| **XGBoost** | 0 | 1,723 | 8 | **0.6571** | 0.7225 | 0.6808 | 0.6836 | 1.83 | 0.005 | 3.2 |
| **XGBoost** | 1 | 1,397 | 5 | 0.4863 | 0.5829 | 0.3916 | 0.3262 | 1.75 | 0.006 | 4.2 |
| **XGBoost** | 2 | 4,257 | 6 | 0.4170 | 0.5925 | 0.7717 | 0.8038 | 1.50 | 0.011 | 2.6 |
| **XGBoost** | 3 | 2,132 | 5 | 0.3925 | 0.7821 | 0.5797 | 0.6536 | 1.76 | 0.007 | 3.2 |
| **XGBoost** | 4 | 1,261 | 5 | 0.3291 | 0.3563 | 0.4560 | 0.6158 | 1.64 | 0.005 | 3.8 |
| **P-MLP** | 0 | 1,723 | 8 | 0.3831 | 0.5605 | 0.3685 | 0.3424 | 21.01 | 0.003 | 1.7 |
| **P-MLP** | 1 | 1,397 | 5 | 0.3544 | 0.4518 | 0.3243 | 0.2694 | 27.73 | 0.003 | 2.4 |
| **P-MLP** | 2 | 4,257 | 6 | **0.4514** | 0.6628 | 0.7940 | 0.8258 | 36.78 | 0.005 | 1.1 |
| **P-MLP** | 3 | 2,132 | 5 | 0.4162 | 0.5801 | 0.6454 | 0.7161 | 17.67 | 0.003 | 1.6 |
| **P-MLP** | 4 | 1,261 | 5 | 0.4294 | 0.4720 | 0.7098 | 0.7523 | 28.41 | 0.003 | 2.1 |

### 2.2 Aggregate vs. Pooled Out-Of-Fold (OOF) Metrics

| Model | Mean Macro F1 (±Std) | Mean Bal Acc (±Std) | Mean Acc (±Std) | Mean Weighted F1 (±Std) | Pooled Macro F1 | Pooled Bal Acc | Pooled Acc | Pooled Weighted F1 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Random Forest** | 0.3970 ± 0.1831 | 0.5425 ± 0.1897 | 0.5145 ± 0.2554 | 0.5517 ± 0.2565 | 0.4664 | 0.5051 | 0.6016 | 0.6260 |
| **XGBoost** | **0.4564 ± 0.1255** | **0.6072 ± 0.1640** | **0.5759 ± 0.1563** | **0.6166 ± 0.1968** | **0.5452** | **0.6349** | **0.6329** | **0.6608** |
| **P-MLP** | 0.4069 ± **0.0384** | 0.5454 ± 0.0857 | 0.5684 ± 0.2100 | 0.5812 ± 0.2676 | **0.5567** | **0.6823** | 0.6257 | 0.6576 |

---

## 3. Detailed Pooled Per-Class Performance

| Model | Class Name | Support | Precision | Recall | F1-Score | Primary Confusion Analysis |
|---|---|:---:|:---:|:---:|:---:|---|
| **Random Forest** | Bulk Data | 1,348 | 0.3304 | **0.3665** | 0.3475 | 60.76% misclassified as Real-Time |
| | Real-Time | 8,132 | 0.7916 | **0.6581** | 0.7187 | 22.70% misclassified as Streaming |
| | Streaming | 1,290 | 0.2518 | **0.4907** | 0.3328 | 45.74% misclassified as Real-Time |
| **XGBoost** | Bulk Data | 1,348 | 0.3956 | **0.5905** | 0.4738 | 35.31% misclassified as Real-Time |
| | Real-Time | 8,132 | 0.8577 | **0.6321** | 0.7279 | 22.25% misclassified as Streaming |
| | Streaming | 1,290 | 0.3183 | **0.6822** | 0.4340 | 29.22% misclassified as Real-Time |
| **P-MLP** | Bulk Data | 1,348 | 0.4135 | **0.8034** | 0.5461 | 14.69% misclassified as Real-Time |
| | Real-Time | 8,132 | 0.9042 | **0.5922** | 0.7156 | 23.59% misclassified as Streaming |
| | Streaming | 1,290 | 0.2974 | **0.6512** | 0.4086 | 24.19% misclassified as Real-Time |

---

## 4. In-Depth Computer Networks Interpretation

### 4.1 Why Does Real-Time Act as a "Gravity Well"?
In Track B, Real-Time represents 75.51% of all flows. When an encrypted flow exhibits ambiguous statistical markers (e.g., short session duration, modest packet count, moderate inter-arrival time), decision boundaries inherently bias predictions toward Real-Time. 
- In Random Forest, 60.76% of Bulk Data and 45.74% of Streaming are pulled into Real-Time.
- XGBoost mitigates this via training-fold inverse frequency sample weighting, reducing Bulk Data false assignments to 35.31% and Streaming false assignments to 29.22%.
- P-MLP performs best in resisting this attractor: only 14.69% of Bulk Data is misclassified as Real-Time, achieving **80.34% Bulk Data recall**.

### 4.2 The Streaming vs. Real-Time Symmetrical Cross-Talk
A striking symmetrical error occurs between Streaming and Real-Time:
- Approximately **22% to 24% of Real-Time flows are classified as Streaming** across all three models.
- Approximately **24% to 45% of Streaming flows are classified as Real-Time**.
- *Underlying Network Cause:* Both applications utilize adaptive video streaming codecs (H.264/AVC, VP9) encapsulated in UDP or HTTPS. When a video call (Skype/Hangouts) experiences one-way speaking or high-bandwidth camera feed, its packet size distribution and burst dynamics become nearly indistinguishable from adaptive video streaming chunks (YouTube/Netflix). Without deep packet inspection (DPI) of video container headers, purely statistical flow classifiers face a fundamental physical ambiguity boundary.

---

## 5. Architectural Consistency & Stability

A major scientific takeaway of Track B is the **cross-fold stability** of the models:
- **Random Forest:** Exhibits extreme instability across folds ($\text{std} = 0.1831$). In Fold 0, RF achieves 0.6832 Macro F1; but in Fold 4, it drops to 0.1994. Tree ensembles split on greedy, local orthogonal thresholds that overfit session-specific MTUs.
- **XGBoost:** Displays moderate stability ($\text{std} = 0.1255$), maintaining Macro F1 between 0.3291 and 0.6571.
- **P-MLP:** Displays **remarkable cross-fold consistency ($\text{std} = 0.0384$)**. Across all 5 folds, its Macro F1 remains tightly bounded between 0.3544 and 0.4514. Its continuous latent representations and batch normalization regularize against session-specific shift.
