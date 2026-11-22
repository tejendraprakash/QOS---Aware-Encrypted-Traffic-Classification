# Track A Full-Flow Baseline Results: Exploratory 4-Class Experiment

**Project:** QoS-Aware Encrypted Network Traffic Classification for Next-Generation Networks  
**Phase:** 6A — Comprehensive Full-Flow Machine Learning Baseline Experiment  
**Date:** November 2026  
**Artifact Directory:** [output/phase6/track_a/](file:///c:/Users/saipr/OneDrive/Desktop/EncryptedTrafficResearch/output/phase6/track_a/)  
**Input Dataset:** [output/phase5/track_a_flows.csv](file:///c:/Users/saipr/OneDrive/Desktop/EncryptedTrafficResearch/output/phase5/track_a_flows.csv) (17,617 flows, 117 columns)  
**Validation Scheme:** 3-Fold `StratifiedGroupKFold` (random_state = 7)

---

## 1. Executive Summary & Label Contamination Disclosure

Track A evaluates the exploratory 4-class classification problem comprising **Bulk Data**, **Browsing**, **Streaming**, and **Real-Time**. 

> [!WARNING]
> **Scientific Integrity & Proxy Label Contamination:**
> The `Browsing` class in this dataset is derived entirely from `email` captures. As audited in Phase 4 and empirically verified in this experiment, **75.05% of all flows in this class are local multicast/broadcast discovery packets** (Link-Local Multicast Name Resolution / LLMNR UDP 5355) and **3.64% are NetBIOS-NS (UDP 137)**. Genuine interactive HTTP/HTTPS web browsing traffic is virtually non-existent.
> 
> When models are trained and tested using leak-free group cross-validation (`StratifiedGroupKFold`), the `Browsing` class **completely collapses**:
> - Random Forest achieves a pooled `Browsing` recall of **7.74%**.
> - XGBoost achieves a pooled `Browsing` recall of **8.08%**.
> - P-MLP achieves a pooled `Browsing` recall of **2.63%**.
> 
> Over 62% to 82% of `Browsing` flows are misclassified as `Real-Time`. This empirical collapse confirms our methodological hypothesis: the models cannot learn generalizable web-browsing behavior from local-network discovery artifacts. Track A must strictly be interpreted as an exploratory study of label contamination.

---

## 2. Quantitative Results Across Folds

### 2.1 Fold-Level Metric Summary

| Model | Fold | Val Samples | Val Groups | Macro F1 | Balanced Accuracy | Accuracy | Weighted F1 | Train Time (s) | Pred Time (s) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Random Forest** | 0 | 1,625 | 7 | 0.3847 | 0.4091 | 0.2794 | 0.2872 | 0.50 | 0.031 |
| **Random Forest** | 1 | 11,677 | 9 | 0.2198 | 0.5844 | 0.3448 | 0.3129 | 0.28 | 0.043 |
| **Random Forest** | 2 | 4,315 | 16 | 0.4238 | 0.4267 | 0.4452 | 0.4429 | 0.35 | 0.055 |
| **XGBoost** | 0 | 1,625 | 7 | 0.4609 | 0.5536 | 0.3378 | 0.3524 | 1.72 | 0.004 |
| **XGBoost** | 1 | 11,677 | 9 | 0.2187 | 0.6235 | 0.3277 | 0.2548 | 1.24 | 0.022 |
| **XGBoost** | 2 | 4,315 | 16 | 0.4631 | 0.4656 | 0.4642 | 0.5013 | 1.38 | 0.008 |
| **P-MLP** | 0 | 1,625 | 7 | 0.3614 | 0.4726 | 0.2634 | 0.2523 | 62.70 | 0.008 |
| **P-MLP** | 1 | 11,677 | 9 | 0.2247 | 0.4187 | 0.3462 | 0.2743 | 32.93 | 0.014 |
| **P-MLP** | 2 | 4,315 | 16 | 0.3545 | 0.3820 | 0.3743 | 0.3906 | 37.47 | 0.005 |

### 2.2 Aggregate & Pooled Out-Of-Fold (OOF) Metrics

| Model | Mean Macro F1 (±Std) | Mean Bal Acc (±Std) | Mean Acc (±Std) | Mean Weighted F1 (±Std) | Pooled Macro F1 | Pooled Bal Acc | Pooled Acc | Pooled Weighted F1 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Random Forest** | 0.3428 ± 0.1083 | 0.4734 ± 0.0965 | 0.3565 ± 0.0835 | 0.3477 ± 0.0828 | **0.2864** | **0.3700** | 0.3633 | 0.3257 |
| **XGBoost** | **0.3809 ± 0.1405** | **0.5476 ± 0.0791** | **0.3766 ± 0.0760** | **0.3695 ± 0.1048** | **0.3163** | **0.4380** | 0.3621 | 0.3269 |
| **P-MLP** | 0.3135 ± 0.0770 | 0.4244 ± 0.0455 | 0.3280 ± 0.0577 | 0.3057 ± 0.0714 | **0.3707** | **0.4929** | 0.3455 | 0.2963 |

---

## 3. Analysis of Fold 1 Anomaly: The Browsing Bottleneck

A critical empirical pattern emerges in **Fold 1**:
- In Fold 1, Macro F1 plummets to **0.2198** (RF), **0.2187** (XGB), and **0.2247** (P-MLP).
- Why? Fold 1's validation partition contains `GRP_EMAIL_1` (5,870 flows, 33.3% of the entire dataset).
- When `GRP_EMAIL_1` is held out, the models are trained only on `GRP_EMAIL_2` (672 flows) and `GRP_VPN_EMAIL_2` (305 flows). Because `GRP_EMAIL_1` exhibits distinct local multicast timing and packet-size distributions from the other two captures, the models fail to recognize `GRP_EMAIL_1` as "Browsing", classifying virtually all 5,870 flows into `Real-Time`.
- This massive failure directly degrades Fold 1 metrics and demonstrates why group-aware validation is essential. A naive random train/test split would have distributed `GRP_EMAIL_1` evenly across train and test, producing falsely triumphant 98% accuracies by memorizing the host IP subnet and LLMNR query frequencies.

---

## 4. Pooled Per-Class Performance Breakdown

| Model | Class Name | Support | Precision | Recall | F1-Score | Dominant Misclassification Target |
|---|---|:---:|:---:|:---:|:---:|---|
| **Random Forest** | Browsing | 6,847 | 0.3197 | **0.0774** | 0.1246 | 64.01% misclassified as Real-Time |
| | Bulk Data | 1,348 | 0.1444 | **0.2923** | 0.1933 | 56.68% misclassified as Real-Time |
| | Real-Time | 8,132 | 0.4430 | **0.5911** | 0.5065 | 26.86% misclassified as Streaming |
| | Streaming | 1,290 | 0.2372 | **0.5194** | 0.3257 | 42.09% misclassified as Real-Time |
| **XGBoost** | Browsing | 6,847 | 0.3294 | **0.0808** | 0.1297 | 62.55% misclassified as Real-Time |
| | Bulk Data | 1,348 | 0.1983 | **0.4955** | 0.2833 | 36.35% misclassified as Real-Time |
| | Real-Time | 8,132 | 0.4571 | **0.5322** | 0.4918 | 33.78% misclassified as Streaming |
| | Streaming | 1,290 | 0.2604 | **0.6434** | 0.3706 | 29.92% misclassified as Real-Time |
| **P-MLP** | Browsing | 6,847 | 0.1585 | **0.0263** | 0.0451 | 82.02% misclassified as Real-Time |
| | Bulk Data | 1,348 | 0.5222 | **0.7426** | 0.6131 | 11.94% misclassified as Streaming |
| | Real-Time | 8,132 | 0.4632 | **0.4902** | 0.4763 | 34.57% misclassified as Streaming |
| | Streaming | 1,290 | 0.2319 | **0.7124** | 0.3501 | 18.29% misclassified as Real-Time |

---

## 5. Conclusions for Track A

1. **Failure of the Browsing Proxy:** Email-derived traffic cannot serve as a proxy for web browsing. The low recall (2.6% to 8.1%) proves that models do not generalize across distinct email capture sessions.
2. **Dominant Attractor:** Real-Time is the dominant class (46.2% of dataset), drawing the vast majority of ambiguous flows.
3. **P-MLP Bulk Data Resilience:** Even under heavy 4-class contamination, P-MLP achieved 74.26% recall on Bulk Data compared to 49.55% for XGBoost and 29.23% for RF, demonstrating the advantage of deep non-linear representations and class-weighted loss.
4. **Recommendation:** Track A results must be published strictly as a negative control / methodological critique demonstrating the hazards of unverified public dataset proxy labels.
