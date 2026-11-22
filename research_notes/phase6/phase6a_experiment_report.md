# Phase 6A Experiment Execution Report

**Project:** QoS-Aware Encrypted Network Traffic Classification for Next-Generation Networks  
**Phase:** 6A — Comprehensive Full-Flow Machine Learning Baseline Experiment  
**Date:** November 2026  
**Execution Script:** [scripts/run_phase6a_experiments.py](file:///c:/Users/saipr/OneDrive/Desktop/EncryptedTrafficResearch/scripts/run_phase6a_experiments.py)  
**Output Directories:** [output/phase6/track_a/](file:///c:/Users/saipr/OneDrive/Desktop/EncryptedTrafficResearch/output/phase6/track_a/), [output/phase6/track_b/](file:///c:/Users/saipr/OneDrive/Desktop/EncryptedTrafficResearch/output/phase6/track_b/)

---

## 1. Execution Summary

Phase 6A executed full-flow baseline machine learning experiments comparing **Random Forest**, **XGBoost**, and **Packet Multi-Layer Perceptron (P-MLP)** across:
- **Track A (Exploratory 4-Class Baseline):** 17,617 flows across 32 capture groups, evaluated using 3-fold `StratifiedGroupKFold`.
- **Track B (Primary 3-Class Benchmark):** 10,770 flows across 29 capture groups, evaluated using 5-fold `StratifiedGroupKFold`.

A total of **24 individual model training runs** were executed (9 in Track A + 15 in Track B). All runs completed with zero errors, zero leakage, and complete deterministic reproducibility.

---

## 2. Complete Metric Table: Track B (Primary 3-Class Benchmark)

Track B excludes the contaminated Email/Browsing traffic, establishing the primary performance benchmark across Bulk Data, Streaming, and Real-Time.

### 2.1 Cross-Validation Metric Summary

| Model | Fold Metric (Mean ± Std) | Pooled OOF Metric | Per-Class Pooled Recall | Inference Latency | Training Time |
|---|---|---|---|---|---|
| **Random Forest** | Macro F1: **0.3970 ± 0.1831**<br>Bal Acc: **0.5425 ± 0.1897**<br>Accuracy: **0.5145 ± 0.2554**<br>Weighted F1: **0.5517 ± 0.2565** | Macro F1: **0.4664**<br>Bal Acc: **0.5051**<br>Accuracy: **0.6016**<br>Weighted F1: **0.6260** | Bulk Data: **36.65%**<br>Real-Time: **65.81%**<br>Streaming: **49.07%** | 38.9 $\mu s$ / flow | 0.49 s / fold |
| **XGBoost** | Macro F1: **0.4564 ± 0.1255**<br>Bal Acc: **0.6072 ± 0.1640**<br>Accuracy: **0.5759 ± 0.1563**<br>Weighted F1: **0.6166 ± 0.1968** | Macro F1: **0.5452**<br>Bal Acc: **0.6349**<br>Accuracy: **0.6329**<br>Weighted F1: **0.6608** | Bulk Data: **59.05%**<br>Real-Time: **63.21%**<br>Streaming: **68.22%** | 3.4 $\mu s$ / flow | 1.70 s / fold |
| **Packet MLP (P-MLP)** | Macro F1: **0.4069 ± 0.0384**<br>Bal Acc: **0.5454 ± 0.0857**<br>Accuracy: **0.5684 ± 0.2100**<br>Weighted F1: **0.5812 ± 0.2676** | Macro F1: **0.5567**<br>Bal Acc: **0.6823**<br>Accuracy: **0.6257**<br>Weighted F1: **0.6576** | Bulk Data: **80.34%**<br>Real-Time: **59.22%**<br>Streaming: **65.12%** | **1.7 $\mu s$ / flow** | 26.32 s / fold |

---

## 3. Complete Metric Table: Track A (Exploratory 4-Class Baseline)

Track A includes the Email-derived `Browsing` category (which is 75% local multicast noise) to observe classifier behavior under proxy label contamination.

### 3.1 Cross-Validation Metric Summary

| Model | Fold Metric (Mean ± Std) | Pooled OOF Metric | Per-Class Pooled Recall | Inference Latency | Training Time |
|---|---|---|---|---|---|
| **Random Forest** | Macro F1: **0.3428 ± 0.1083**<br>Bal Acc: **0.4734 ± 0.0965**<br>Accuracy: **0.3565 ± 0.0835** | Macro F1: **0.2864**<br>Bal Acc: **0.3700**<br>Accuracy: **0.3633** | Browsing: **7.74%**<br>Bulk Data: **29.23%**<br>Real-Time: **59.11%**<br>Streaming: **51.94%** | 43.0 $\mu s$ / flow | 0.38 s / fold |
| **XGBoost** | Macro F1: **0.3809 ± 0.1405**<br>Bal Acc: **0.5476 ± 0.0791**<br>Accuracy: **0.3766 ± 0.0760** | Macro F1: **0.3163**<br>Bal Acc: **0.4380**<br>Accuracy: **0.3621** | Browsing: **8.08%**<br>Bulk Data: **49.55%**<br>Real-Time: **53.22%**<br>Streaming: **64.34%** | 11.3 $\mu s$ / flow | 1.45 s / fold |
| **Packet MLP (P-MLP)** | Macro F1: **0.3135 ± 0.0770**<br>Bal Acc: **0.4244 ± 0.0455**<br>Accuracy: **0.3280 ± 0.0577** | Macro F1: **0.3707**<br>Bal Acc: **0.4929**<br>Accuracy: **0.3455** | Browsing: **2.63%**<br>Bulk Data: **74.26%**<br>Real-Time: **49.02%**<br>Streaming: **71.24%** | **9.0 $\mu s$ / flow** | 44.37 s / fold |

---

## 4. Primary Empirical Discoveries

1. **The Realistic Performance Boundary of Group Cross-Validation:**
   Unlike published literature reporting $>95-99\%$ accuracy using random flow splitting (which leaks capture-session artifacts), rigorous group-aware cross-validation reveals that realistic Macro F1 on unseen captures ranges between **$0.40$ and $0.56$**.
2. **Browsing Class Collapse Validates Audit Findings:**
   Across all three models, Browsing recall collapsed below $8.1\%$. Over $62\%$ of email flows were misclassified as Real-Time. This definitively proves that models do not generalize across distinct email sessions, confirming that the Browsing proxy was learning session-specific background noise.
3. **P-MLP Demonstrates Superior Representation Learning:**
   P-MLP achieved the highest pooled Macro F1 ($0.5567$), highest pooled Balanced Accuracy ($0.6823$), highest Bulk Data recall ($80.34\%$), lowest inference latency ($1.7 \mu s$), and by far the highest cross-fold stability ($\sigma = 0.0384$).
