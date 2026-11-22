# Independent Metrics Verification & Out-Of-Fold Audit Report

**Audit Date:** November 21, 2026  
**Scope:** Phase 6 Baseline Models (Random Forest, XGBoost, P_MLP) across Track A (4-Class QoS) and Track B (3-Class VPN)  
**Audit Target:** `output/phase6/track_a/out_of_fold_predictions.csv` & `output/phase6/track_b/out_of_fold_predictions.csv`  
**Auditor:** Independent Subagent Verification Harness (`scripts/verify_phase6_metrics.py`)  
**Status:** **PASSED — 100% EXACT REPRODUCIBILITY VERIFIED**  

---

## 1. Executive Summary & Verification Verdict

An exhaustive, independent mathematical audit was conducted on all saved Out-Of-Fold (OOF) predictions generated during Phase 6 experiments. The verification pipeline recalculated every metric directly from raw flow-level predictions (`true_label` and `pred_<model>`) without relying on any cached intermediate states.

### Key Verification Findings
1. **Exact Confusion Matrix Match (0 Discrepancy):** Recalculating the raw confusion matrix counts across all 17,617 flows in Track A and 10,770 flows in Track B produced an exact bitwise match (`sum(|recalculated - reported|) = 0`) across all three models (Random Forest, XGBoost, P_MLP).
2. **Exact Fold-Level Precision (Machine Epsilon):** Recalculating fold-level Accuracy, Balanced Accuracy, Macro F1, and Weighted F1 across all 8 individual validation folds (3 folds in Track A, 5 folds in Track B) matched `fold_metrics.csv` to within floating-point epsilon ($< 8.4 \times 10^{-17}$), confirming identical algorithmic execution.
3. **Aggregate Metrics Verification:** Recalculated unweighted fold means, fold standard deviations, and pooled OOF metrics match the published values in `aggregate_metrics.csv` to 4 decimal places without exception. The maximum observed difference across all 72 reported aggregate metric fields was $0.000050$, which is entirely attributable to standard round-to-even formatting (`round(val, 4)`).
4. **Resolution of Fold Mean vs. Pooled OOF Divergence:** A rigorous mathematical and empirical analysis was conducted explaining why unweighted fold means diverge significantly from pooled OOF metrics. Under GroupKFold partitioning, cluster sizes are highly unbalanced (e.g., Track A Fold 1 contains 66.3% of all flows, while Fold 0 contains only 9.2%). This structural heterogeneity causes unweighted fold means to severely over-represent small folds and misrepresent global population classification performance.

| Track | Model | Reported Pooled Macro F1 | Recalculated Pooled Macro F1 | Reported Mean Macro F1 | Recalculated Mean Macro F1 | Audit Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Track A** | Random Forest | 0.2864 | 0.2864 | 0.3428 | 0.3428 | **EXACT MATCH** |
| **Track A** | XGBoost | 0.3163 | 0.3163 | 0.3809 | 0.3809 | **EXACT MATCH** |
| **Track A** | P_MLP | 0.3707 | 0.3707 | 0.3135 | 0.3135 | **EXACT MATCH** |
| **Track B** | Random Forest | 0.4664 | 0.4664 | 0.3970 | 0.3970 | **EXACT MATCH** |
| **Track B** | XGBoost | 0.5452 | 0.5452 | 0.4564 | 0.4564 | **EXACT MATCH** |
| **Track B** | P_MLP | 0.5567 | 0.5567 | 0.4069 | 0.4069 | **EXACT MATCH** |

---

## 2. Mathematical & Methodological Framework

### Metric Formulations
Let $C$ denote the set of classes, $N$ the total number of samples, and $K$ the number of validation folds.
For each class $c \in C$:
- $TP_c$: True Positives
- $FP_c$: False Positives
- $FN_c$: False Negatives
- $S_c$: Class support (true class count, $S_c = TP_c + FN_c$)

1. **Per-Class Precision, Recall, and F1:**
   $$\text{Precision}_c = \frac{TP_c}{TP_c + FP_c}, \quad \text{Recall}_c = \frac{TP_c}{TP_c + FN_c}, \quad F1_c = \frac{2 \cdot \text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c} = \frac{2 \cdot TP_c}{2 \cdot TP_c + FP_c + FN_c}$$

2. **Overall Accuracy:**
   $$\text{Accuracy} = \frac{\sum_{c \in C} TP_c}{N}$$

3. **Balanced Accuracy (Macro-averaged Recall):**
   $$\text{Balanced Accuracy} = \frac{1}{|C|} \sum_{c \in C} \text{Recall}_c = \frac{1}{|C|} \sum_{c \in C} \frac{TP_c}{S_c}$$

4. **Macro F1:**
   $$\text{Macro } F1 = \frac{1}{|C|} \sum_{c \in C} F1_c$$

5. **Weighted F1:**
   $$\text{Weighted } F1 = \sum_{c \in C} \left( \frac{S_c}{N} \right) F1_c$$

### Aggregation Paradigms: Unweighted Fold Means vs. Pooled OOF Metrics

In cross-validation literature, two primary paradigms exist for summarizing out-of-fold performance:

1. **Unweighted Fold Means (Fold-Level Macro Average):**
   Each validation fold $k \in \{1, \dots, K\}$ is treated as an independent experiment. The metric $M_k$ is calculated on fold $k$, and the summary is the unweighted arithmetic mean and sample standard deviation across the $K$ folds:
   $$\bar{M}_{\text{fold}} = \frac{1}{K} \sum_{k=1}^K M_k, \quad s_M = \sqrt{\frac{1}{K-1} \sum_{k=1}^K (M_k - \bar{M}_{\text{fold}})^2}$$

2. **Pooled Out-Of-Fold (OOF) Metrics (Global Population Evaluation):**
   Because every flow in the dataset appears in the validation set of exactly one fold, the predictions across all $K$ folds form a complete, mutually exclusive, and collectively exhaustive partition of the entire dataset. The pooled metric concatenates all out-of-fold predictions into a single global vector $\hat{y} \in C^N$ and evaluates the metric globally:
   $$M_{\text{pooled}} = \mathcal{M}(y_{\text{true}}, \hat{y}_{\text{OOF}})$$

---

## 3. Deep-Dive: Theoretical & Empirical Distinction Between Fold Means and Pooled OOF

A central question in machine learning benchmarking is why $\bar{M}_{\text{fold}}$ and $M_{\text{pooled}}$ diverge, and which metric is appropriate for decision-making. In Phase 6, this divergence is pronounced. For instance, in Track A for XGBoost, the unweighted fold mean Macro F1 is **0.3809**, whereas the pooled OOF Macro F1 is **0.3163** (a gap of $-0.0646$). Conversely, in Track B for P_MLP, the unweighted fold mean Macro F1 is **0.4069**, while the pooled OOF Macro F1 is **0.5567** (a jump of $+0.1498$).

This divergence is governed by three mathematical and structural principles:

### Principle 1: Severe Cluster Heterogeneity Under GroupKFold (Simpson's Paradox)
In standard stratified cross-validation, folds are approximately equal in size ($N_k \approx N/K$) and preserve class marginals. However, Phase 6 enforces strict group-based splitting (`capture_group`) to guarantee **zero capture group leakage** between training and validation.

Because capture groups correspond to discrete network capture sessions, group sizes vary dramatically. As verified from `out_of_fold_predictions.csv`:

- **Track A (3 Folds, 17,617 total flows):**
  - Fold 0: 1,625 flows (**9.2%** of dataset)
  - Fold 1: 11,677 flows (**66.3%** of dataset)
  - Fold 2: 4,315 flows (**24.5%** of dataset)
- **Track B (5 Folds, 10,770 total flows):**
  - Fold 0: 1,723 flows (**16.0%**)
  - Fold 1: 1,397 flows (**13.0%**)
  - Fold 2: 4,257 flows (**39.5%**)
  - Fold 3: 2,132 flows (**19.8%**)
  - Fold 4: 1,261 flows (**11.7%**)

When calculating the unweighted fold mean $\bar{M}_{\text{fold}}$, each fold receives an identical weight $w_k = 1/K = 33.3\%$ (Track A) or $20.0\%$ (Track B). Consequently:
- In Track A, Fold 0 (9.2% of samples) is **over-weighted by 3.6x**, while Fold 1 (66.3% of samples) is **under-weighted by 0.5x**.
- If a model performs very well on a small fold (e.g. Track A Fold 0 Macro F1 = 0.4609 for XGBoost) but struggles on the dominant fold (Fold 1 Macro F1 = 0.2187), the unweighted average artificially inflates the reported score (0.3809) compared to the actual experience across 66.3% of the network traffic (pooled Macro F1 = 0.3163).
- Conversely, in Track B, Fold 2 contains 39.5% of the data and exhibits high accuracy (e.g., 81.7% for RF). In the unweighted fold mean, Fold 4 (only 11.7% of flows, accuracy 24.3%) exerts the same weight as Fold 2, dragging the unweighted accuracy down to 51.45%, whereas the pooled OOF accuracy across all flows is **60.16%**.

### Principle 2: Non-Linearity of Harmonic Means (Jensen's Inequality)
Classification metrics like F1 and Balanced Accuracy are non-linear rational functions of confusion matrix entries. Even if fold sizes were identical, the arithmetic mean of ratios is not mathematically equal to the ratio of sums:
$$\frac{1}{K} \sum_{k=1}^K \frac{2 \cdot TP_{c,k}}{2 \cdot TP_{c,k} + FP_{c,k} + FN_{c,k}} \neq \frac{2 \sum_{k=1}^K TP_{c,k}}{2 \sum_{k=1}^K TP_{c,k} + \sum_{k=1}^K FP_{c,k} + \sum_{k=1}^K FN_{c,k}}$$
Due to Jensen's inequality and the convexity properties of fractional harmonic terms, aggregating non-linear metrics before summing counts introduces systemic bias.

### Principle 3: Degenerate Local Class Support from Group Segregation
When entire applications/capture groups are assigned to single folds, class distributions within individual folds become severely distorted. For example, in Track A:
- In Fold 1, **Bulk Data** has only **11 ground-truth samples** (out of 11,677 flows!).
- In Fold 1, Random Forest predicted Bulk Data for 2,184 flows, resulting in a precision of $11 / 2184 = 0.0050$, recall of $1.0000$, and an F1 score of **0.0100**.
- This single degraded fold F1 score of 0.0100 directly pulls down Fold 1's Macro F1 to 0.2198.
- When calculating the unweighted 3-fold mean, that 0.0100 is factored in with 33.3% weight. In contrast, in pooled OOF evaluation, Bulk Data has 1,348 total samples globally, and the 11 samples from Fold 1 are placed in proper context alongside the 224 samples from Fold 0 and 1,113 samples from Fold 2, avoiding artificial distortion.

### Metric Selection Recommendation
- **Use Pooled OOF Metrics ($M_{\text{pooled}}$) for:** Global model selection, executive reporting, deployment performance estimates, and confusion matrix auditing. Pooled OOF represents the true empirical performance across the entire 100% census of flows.
- **Use Fold-Level Metrics ($\bar{M}_{\text{fold}} \pm s_M$) for:** Assessing model stability across distinct capture environments and detecting domain generalization vulnerability.

---

## 4. Track A (4-Class QoS) Comprehensive Verification

### Dataset Partition Structure
- **Total Out-Of-Fold Flows:** 17,617
- **Classes:** Browsing (6,847 flows, 38.87%), Bulk Data (1,348 flows, 7.65%), Real-Time (8,132 flows, 46.16%), Streaming (1,290 flows, 7.32%)
- **Folds:** Fold 0 (1,625 flows), Fold 1 (11,677 flows), Fold 2 (4,315 flows)

### Table 1: Track A Aggregate Metrics Verification
Comparison between independently recalculated metrics and reported values in `output/phase6/track_a/aggregate_metrics.csv`.

| Model | Metric Type | Metric Name | Recalculated (Exact) | Reported (CSV) | Absolute Difference | Verification Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| Random_Forest | Fold Mean | Mean Macro F1 | 0.342774 | 0.3428 | 0.000026 | MATCH (exact to 4 dec) |
| Random_Forest | Fold Std | Std Macro F1 | 0.108256 | 0.1083 | 0.000044 | MATCH (exact to 4 dec) |
| Random_Forest | Fold Mean | Mean Balanced Acc | 0.473410 | 0.4734 | 0.000010 | MATCH (exact to 4 dec) |
| Random_Forest | Fold Std | Std Balanced Acc | 0.096543 | 0.0965 | 0.000043 | MATCH (exact to 4 dec) |
| Random_Forest | Fold Mean | Mean Accuracy | 0.356452 | 0.3565 | 0.000048 | MATCH (exact to 4 dec) |
| Random_Forest | Fold Std | Std Accuracy | 0.083517 | 0.0835 | 0.000017 | MATCH (exact to 4 dec) |
| Random_Forest | Fold Mean | Mean Weighted F1 | 0.347742 | 0.3477 | 0.000042 | MATCH (exact to 4 dec) |
| Random_Forest | Fold Std | Std Weighted F1 | 0.082836 | 0.0828 | 0.000036 | MATCH (exact to 4 dec) |
| Random_Forest | Pooled OOF | Pooled Macro F1 | 0.286353 | 0.2864 | 0.000047 | MATCH (exact to 4 dec) |
| Random_Forest | Pooled OOF | Pooled Balanced Acc | 0.370048 | 0.3700 | 0.000048 | MATCH (exact to 4 dec) |
| Random_Forest | Pooled OOF | Pooled Accuracy | 0.363342 | 0.3633 | 0.000042 | MATCH (exact to 4 dec) |
| Random_Forest | Pooled OOF | Pooled Weighted F1 | 0.325734 | 0.3257 | 0.000034 | MATCH (exact to 4 dec) |
| XGBoost | Fold Mean | Mean Macro F1 | 0.380910 | 0.3809 | 0.000010 | MATCH (exact to 4 dec) |
| XGBoost | Fold Std | Std Macro F1 | 0.140464 | 0.1405 | 0.000036 | MATCH (exact to 4 dec) |
| XGBoost | Fold Mean | Mean Balanced Acc | 0.547557 | 0.5476 | 0.000043 | MATCH (exact to 4 dec) |
| XGBoost | Fold Std | Std Balanced Acc | 0.079101 | 0.0791 | 0.000001 | MATCH (exact to 4 dec) |
| XGBoost | Fold Mean | Mean Accuracy | 0.376593 | 0.3766 | 0.000007 | MATCH (exact to 4 dec) |
| XGBoost | Fold Std | Std Accuracy | 0.076033 | 0.0760 | 0.000033 | MATCH (exact to 4 dec) |
| XGBoost | Fold Mean | Mean Weighted F1 | 0.369451 | 0.3695 | 0.000049 | MATCH (exact to 4 dec) |
| XGBoost | Fold Std | Std Weighted F1 | 0.104792 | 0.1048 | 0.000008 | MATCH (exact to 4 dec) |
| XGBoost | Pooled OOF | Pooled Macro F1 | 0.316251 | 0.3163 | 0.000049 | MATCH (exact to 4 dec) |
| XGBoost | Pooled OOF | Pooled Balanced Acc | 0.437986 | 0.4380 | 0.000014 | MATCH (exact to 4 dec) |
| XGBoost | Pooled OOF | Pooled Accuracy | 0.362093 | 0.3621 | 0.000007 | MATCH (exact to 4 dec) |
| XGBoost | Pooled OOF | Pooled Weighted F1 | 0.326934 | 0.3269 | 0.000034 | MATCH (exact to 4 dec) |
| P_MLP | Fold Mean | Mean Macro F1 | 0.313533 | 0.3135 | 0.000033 | MATCH (exact to 4 dec) |
| P_MLP | Fold Std | Std Macro F1 | 0.077015 | 0.0770 | 0.000015 | MATCH (exact to 4 dec) |
| P_MLP | Fold Mean | Mean Balanced Acc | 0.424437 | 0.4244 | 0.000037 | MATCH (exact to 4 dec) |
| P_MLP | Fold Std | Std Balanced Acc | 0.045536 | 0.0455 | 0.000036 | MATCH (exact to 4 dec) |
| P_MLP | Fold Mean | Mean Accuracy | 0.327966 | 0.3280 | 0.000034 | MATCH (exact to 4 dec) |
| P_MLP | Fold Std | Std Accuracy | 0.057659 | 0.0577 | 0.000041 | MATCH (exact to 4 dec) |
| P_MLP | Fold Mean | Mean Weighted F1 | 0.305680 | 0.3057 | 0.000020 | MATCH (exact to 4 dec) |
| P_MLP | Fold Std | Std Weighted F1 | 0.071415 | 0.0714 | 0.000015 | MATCH (exact to 4 dec) |
| P_MLP | Pooled OOF | Pooled Macro F1 | 0.370728 | 0.3707 | 0.000028 | MATCH (exact to 4 dec) |
| P_MLP | Pooled OOF | Pooled Balanced Acc | 0.492859 | 0.4929 | 0.000041 | MATCH (exact to 4 dec) |
| P_MLP | Pooled OOF | Pooled Accuracy | 0.345462 | 0.3455 | 0.000038 | MATCH (exact to 4 dec) |
| P_MLP | Pooled OOF | Pooled Weighted F1 | 0.296278 | 0.2963 | 0.000022 | MATCH (exact to 4 dec) |

### Table 2: Track A Fold-Level Metrics Verification
Verification of individual fold values against `output/phase6/track_a/fold_metrics.csv`.

| Model | Fold | Metric Name | Recalculated (Exact) | Reported in CSV | Absolute Difference | Status |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: |
| Random_Forest | 0 | Macro F1 | 0.3847212999594010 | 0.3847212999594009 | 5.55e-17 | EXACT MATCH |
| Random_Forest | 0 | Balanced Acc | 0.4091278435273446 | 0.4091278435273446 | 0.00e+00 | EXACT MATCH |
| Random_Forest | 0 | Accuracy | 0.2793846153846154 | 0.2793846153846154 | 0.00e+00 | EXACT MATCH |
| Random_Forest | 0 | Weighted F1 | 0.3227620388321287 | 0.3227620388321287 | 0.00e+00 | EXACT MATCH |
| Random_Forest | 1 | Macro F1 | 0.2198211926113270 | 0.2198211926113270 | 2.78e-17 | EXACT MATCH |
| Random_Forest | 1 | Balanced Acc | 0.5844269944350767 | 0.5844269944350767 | 0.00e+00 | EXACT MATCH |
| Random_Forest | 1 | Accuracy | 0.3447803374154321 | 0.3447803374154320 | 5.55e-17 | EXACT MATCH |
| Random_Forest | 1 | Weighted F1 | 0.2802706428264615 | 0.2802706428264614 | 5.55e-17 | EXACT MATCH |
| Random_Forest | 2 | Macro F1 | 0.4237786299868133 | 0.4237786299868133 | 0.00e+00 | EXACT MATCH |
| Random_Forest | 2 | Balanced Acc | 0.4266765957492611 | 0.4266765957492611 | 5.55e-17 | EXACT MATCH |
| Random_Forest | 2 | Accuracy | 0.4451911935110081 | 0.4451911935110081 | 0.00e+00 | EXACT MATCH |
| Random_Forest | 2 | Weighted F1 | 0.4401926078393116 | 0.4401926078393116 | 0.00e+00 | EXACT MATCH |
| XGBoost | 0 | Macro F1 | 0.4609066774266059 | 0.4609066774266059 | 0.00e+00 | EXACT MATCH |
| XGBoost | 0 | Balanced Acc | 0.5535917563815371 | 0.5535917563815371 | 0.00e+00 | EXACT MATCH |
| XGBoost | 0 | Accuracy | 0.3378461538461539 | 0.3378461538461538 | 5.55e-17 | EXACT MATCH |
| XGBoost | 0 | Weighted F1 | 0.3624844683130439 | 0.3624844683130439 | 5.55e-17 | EXACT MATCH |
| XGBoost | 1 | Macro F1 | 0.2187211859703525 | 0.2187211859703525 | 0.00e+00 | EXACT MATCH |
| XGBoost | 1 | Balanced Acc | 0.6234678232055940 | 0.6234678232055940 | 0.00e+00 | EXACT MATCH |
| XGBoost | 1 | Accuracy | 0.3277382889440781 | 0.3277382889440781 | 0.00e+00 | EXACT MATCH |
| XGBoost | 1 | Weighted F1 | 0.2683166537428128 | 0.2683166537428128 | 0.00e+00 | EXACT MATCH |
| XGBoost | 2 | Macro F1 | 0.4631032000497792 | 0.4631032000497792 | 0.00e+00 | EXACT MATCH |
| XGBoost | 2 | Balanced Acc | 0.4656113036047728 | 0.4656113036047727 | 5.55e-17 | EXACT MATCH |
| XGBoost | 2 | Accuracy | 0.4641946697566628 | 0.4641946697566628 | 0.00e+00 | EXACT MATCH |
| XGBoost | 2 | Weighted F1 | 0.4775524576403593 | 0.4775524576403593 | 0.00e+00 | EXACT MATCH |
| P_MLP | 0 | Macro F1 | 0.3614178284192641 | 0.3614178284192640 | 5.55e-17 | EXACT MATCH |
| P_MLP | 0 | Balanced Acc | 0.4725642975609565 | 0.4725642975609565 | 0.00e+00 | EXACT MATCH |
| P_MLP | 0 | Accuracy | 0.2633846153846154 | 0.2633846153846154 | 0.00e+00 | EXACT MATCH |
| P_MLP | 0 | Weighted F1 | 0.2736438861252392 | 0.2736438861252392 | 0.00e+00 | EXACT MATCH |
| P_MLP | 1 | Macro F1 | 0.2246936162732477 | 0.2246936162732477 | 0.00e+00 | EXACT MATCH |
| P_MLP | 1 | Balanced Acc | 0.4187132142318411 | 0.4187132142318411 | 0.00e+00 | EXACT MATCH |
| P_MLP | 1 | Accuracy | 0.3462361908024321 | 0.3462361908024321 | 0.00e+00 | EXACT MATCH |
| P_MLP | 1 | Weighted F1 | 0.2558912955345365 | 0.2558912955345365 | 5.55e-17 | EXACT MATCH |
| P_MLP | 2 | Macro F1 | 0.3544885555361323 | 0.3544885555361323 | 0.00e+00 | EXACT MATCH |
| P_MLP | 2 | Balanced Acc | 0.3820334622658305 | 0.3820334622658304 | 5.55e-17 | EXACT MATCH |
| P_MLP | 2 | Accuracy | 0.3742757821552723 | 0.3742757821552723 | 0.00e+00 | EXACT MATCH |
| P_MLP | 2 | Weighted F1 | 0.3875035823385102 | 0.3875035823385102 | 5.55e-17 | EXACT MATCH |

### Table 3: Track A Recalculated Pooled Per-Class Metrics
Evaluated across all 17,617 out-of-fold predictions.

| Model | Class | Precision | Recall | F1-Score | Support | Support % |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Random_Forest | Browsing | 0.3914 | 0.0774 | 0.1293 | 6,847 | 38.87% |
| Random_Forest | Bulk Data | 0.1524 | 0.2923 | 0.2003 | 1,348 | 7.65% |
| Random_Forest | Real-Time | 0.4579 | 0.5911 | 0.5161 | 8,132 | 46.16% |
| Random_Forest | Streaming | 0.2107 | 0.5194 | 0.2998 | 1,290 | 7.32% |
| XGBoost | Browsing | 0.4022 | 0.0808 | 0.1345 | 6,847 | 38.87% |
| XGBoost | Bulk Data | 0.2415 | 0.4955 | 0.3247 | 1,348 | 7.65% |
| XGBoost | Real-Time | 0.4562 | 0.5322 | 0.4913 | 8,132 | 46.16% |
| XGBoost | Streaming | 0.2081 | 0.6434 | 0.3145 | 1,290 | 7.32% |
| P_MLP | Browsing | 0.1427 | 0.0263 | 0.0444 | 6,847 | 38.87% |
| P_MLP | Bulk Data | 0.6638 | 0.7426 | 0.7010 | 1,348 | 7.65% |
| P_MLP | Real-Time | 0.4012 | 0.4902 | 0.4413 | 8,132 | 46.16% |
| P_MLP | Streaming | 0.1870 | 0.7124 | 0.2963 | 1,290 | 7.32% |

### Table 4: Track A Confusion Matrices (Counts & Row-Normalized %)

#### Random_Forest Confusion Matrix (Track A)

**Raw Prediction Counts:**

| True \ Pred | Browsing | Bulk Data | Real-Time | Streaming | Total True |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Browsing** | 530 | 1,695 | 4,383 | 239 | 6,847 |
| **Bulk Data** | 103 | 394 | 764 | 87 | 1,348 |
| **Real-Time** | 700 | 441 | 4,807 | 2,184 | 8,132 |
| **Streaming** | 21 | 56 | 543 | 670 | 1,290 |

**Row-Normalized Percentages (Recall per Cell):**

| True \ Pred | Browsing | Bulk Data | Real-Time | Streaming | Row Sum |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Browsing** | 7.74% | 24.76% | 64.01% | 3.49% | 100.0% |
| **Bulk Data** | 7.64% | 29.23% | 56.68% | 6.45% | 100.0% |
| **Real-Time** | 8.61% | 5.42% | 59.11% | 26.86% | 100.0% |
| **Streaming** | 1.63% | 4.34% | 42.09% | 51.94% | 100.0% |

#### XGBoost Confusion Matrix (Track A)

**Raw Prediction Counts:**

| True \ Pred | Browsing | Bulk Data | Real-Time | Streaming | Total True |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Browsing** | 553 | 1,684 | 4,283 | 327 | 6,847 |
| **Bulk Data** | 105 | 668 | 490 | 85 | 1,348 |
| **Real-Time** | 686 | 371 | 4,328 | 2,747 | 8,132 |
| **Streaming** | 31 | 43 | 386 | 830 | 1,290 |

**Row-Normalized Percentages (Recall per Cell):**

| True \ Pred | Browsing | Bulk Data | Real-Time | Streaming | Row Sum |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Browsing** | 8.08% | 24.59% | 62.55% | 4.78% | 100.0% |
| **Bulk Data** | 7.79% | 49.55% | 36.35% | 6.31% | 100.0% |
| **Real-Time** | 8.44% | 4.56% | 53.22% | 33.78% | 100.0% |
| **Streaming** | 2.40% | 3.33% | 29.92% | 64.34% | 100.0% |

#### P_MLP Confusion Matrix (Track A)

**Raw Prediction Counts:**

| True \ Pred | Browsing | Bulk Data | Real-Time | Streaming | Total True |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Browsing** | 180 | 28 | 5,616 | 1,023 | 6,847 |
| **Bulk Data** | 90 | 1,001 | 96 | 161 | 1,348 |
| **Real-Time** | 957 | 378 | 3,986 | 2,811 | 8,132 |
| **Streaming** | 34 | 101 | 236 | 919 | 1,290 |

**Row-Normalized Percentages (Recall per Cell):**

| True \ Pred | Browsing | Bulk Data | Real-Time | Streaming | Row Sum |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Browsing** | 2.63% | 0.41% | 82.02% | 14.94% | 100.0% |
| **Bulk Data** | 6.68% | 74.26% | 7.12% | 11.94% | 100.0% |
| **Real-Time** | 11.77% | 4.65% | 49.02% | 34.57% | 100.0% |
| **Streaming** | 2.64% | 7.83% | 18.29% | 71.24% | 100.0% |

---

## 5. Track B (3-Class VPN) Comprehensive Verification

### Dataset Partition Structure
- **Total Out-Of-Fold Flows:** 10,770
- **Classes:** Bulk Data (1,348 flows, 12.52%), Real-Time (8,132 flows, 75.51%), Streaming (1,290 flows, 11.98%)
- **Folds:** Fold 0 (1,723 flows), Fold 1 (1,397 flows), Fold 2 (4,257 flows), Fold 3 (2,132 flows), Fold 4 (1,261 flows)

### Table 5: Track B Aggregate Metrics Verification
Comparison between independently recalculated metrics and reported values in `output/phase6/track_b/aggregate_metrics.csv`.

| Model | Metric Type | Metric Name | Recalculated (Exact) | Reported (CSV) | Absolute Difference | Verification Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| Random_Forest | Fold Mean | Mean Macro F1 | 0.396956 | 0.3970 | 0.000044 | MATCH (exact to 4 dec) |
| Random_Forest | Fold Std | Std Macro F1 | 0.183107 | 0.1831 | 0.000007 | MATCH (exact to 4 dec) |
| Random_Forest | Fold Mean | Mean Balanced Acc | 0.542490 | 0.5425 | 0.000010 | MATCH (exact to 4 dec) |
| Random_Forest | Fold Std | Std Balanced Acc | 0.189681 | 0.1897 | 0.000019 | MATCH (exact to 4 dec) |
| Random_Forest | Fold Mean | Mean Accuracy | 0.514464 | 0.5145 | 0.000036 | MATCH (exact to 4 dec) |
| Random_Forest | Fold Std | Std Accuracy | 0.255411 | 0.2554 | 0.000011 | MATCH (exact to 4 dec) |
| Random_Forest | Fold Mean | Mean Weighted F1 | 0.551661 | 0.5517 | 0.000039 | MATCH (exact to 4 dec) |
| Random_Forest | Fold Std | Std Weighted F1 | 0.256527 | 0.2565 | 0.000027 | MATCH (exact to 4 dec) |
| Random_Forest | Pooled OOF | Pooled Macro F1 | 0.466352 | 0.4664 | 0.000048 | MATCH (exact to 4 dec) |
| Random_Forest | Pooled OOF | Pooled Balanced Acc | 0.505102 | 0.5051 | 0.000002 | MATCH (exact to 4 dec) |
| Random_Forest | Pooled OOF | Pooled Accuracy | 0.601578 | 0.6016 | 0.000022 | MATCH (exact to 4 dec) |
| Random_Forest | Pooled OOF | Pooled Weighted F1 | 0.626041 | 0.6260 | 0.000041 | MATCH (exact to 4 dec) |
| XGBoost | Fold Mean | Mean Macro F1 | 0.456393 | 0.4564 | 0.000007 | MATCH (exact to 4 dec) |
| XGBoost | Fold Std | Std Macro F1 | 0.125502 | 0.1255 | 0.000002 | MATCH (exact to 4 dec) |
| XGBoost | Fold Mean | Mean Balanced Acc | 0.607250 | 0.6072 | 0.000050 | MATCH (exact to 4 dec) |
| XGBoost | Fold Std | Std Balanced Acc | 0.164025 | 0.1640 | 0.000025 | MATCH (exact to 4 dec) |
| XGBoost | Fold Mean | Mean Accuracy | 0.575947 | 0.5759 | 0.000047 | MATCH (exact to 4 dec) |
| XGBoost | Fold Std | Std Accuracy | 0.156289 | 0.1563 | 0.000011 | MATCH (exact to 4 dec) |
| XGBoost | Fold Mean | Mean Weighted F1 | 0.616647 | 0.6166 | 0.000047 | MATCH (exact to 4 dec) |
| XGBoost | Fold Std | Std Weighted F1 | 0.196820 | 0.1968 | 0.000020 | MATCH (exact to 4 dec) |
| XGBoost | Pooled OOF | Pooled Macro F1 | 0.545210 | 0.5452 | 0.000010 | MATCH (exact to 4 dec) |
| XGBoost | Pooled OOF | Pooled Balanced Acc | 0.634915 | 0.6349 | 0.000015 | MATCH (exact to 4 dec) |
| XGBoost | Pooled OOF | Pooled Accuracy | 0.632869 | 0.6329 | 0.000031 | MATCH (exact to 4 dec) |
| XGBoost | Pooled OOF | Pooled Weighted F1 | 0.660814 | 0.6608 | 0.000014 | MATCH (exact to 4 dec) |
| P_MLP | Fold Mean | Mean Macro F1 | 0.406920 | 0.4069 | 0.000020 | MATCH (exact to 4 dec) |
| P_MLP | Fold Std | Std Macro F1 | 0.038398 | 0.0384 | 0.000002 | MATCH (exact to 4 dec) |
| P_MLP | Fold Mean | Mean Balanced Acc | 0.545433 | 0.5454 | 0.000033 | MATCH (exact to 4 dec) |
| P_MLP | Fold Std | Std Balanced Acc | 0.085670 | 0.0857 | 0.000030 | MATCH (exact to 4 dec) |
| P_MLP | Fold Mean | Mean Accuracy | 0.568391 | 0.5684 | 0.000009 | MATCH (exact to 4 dec) |
| P_MLP | Fold Std | Std Accuracy | 0.209967 | 0.2100 | 0.000033 | MATCH (exact to 4 dec) |
| P_MLP | Fold Mean | Mean Weighted F1 | 0.581244 | 0.5812 | 0.000044 | MATCH (exact to 4 dec) |
| P_MLP | Fold Std | Std Weighted F1 | 0.267620 | 0.2676 | 0.000020 | MATCH (exact to 4 dec) |
| P_MLP | Pooled OOF | Pooled Macro F1 | 0.556658 | 0.5567 | 0.000042 | MATCH (exact to 4 dec) |
| P_MLP | Pooled OOF | Pooled Balanced Acc | 0.682268 | 0.6823 | 0.000032 | MATCH (exact to 4 dec) |
| P_MLP | Pooled OOF | Pooled Accuracy | 0.625720 | 0.6257 | 0.000020 | MATCH (exact to 4 dec) |
| P_MLP | Pooled OOF | Pooled Weighted F1 | 0.657643 | 0.6576 | 0.000043 | MATCH (exact to 4 dec) |

### Table 6: Track B Fold-Level Metrics Verification
Verification of individual fold values against `output/phase6/track_b/fold_metrics.csv`.

| Model | Fold | Metric Name | Recalculated (Exact) | Reported in CSV | Absolute Difference | Status |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: |
| Random_Forest | 0 | Macro F1 | 0.6831897999588491 | 0.6831897999588491 | 0.00e+00 | EXACT MATCH |
| Random_Forest | 0 | Balanced Acc | 0.6664359471467749 | 0.6664359471467749 | 0.00e+00 | EXACT MATCH |
| Random_Forest | 0 | Accuracy | 0.7399883923389438 | 0.7399883923389438 | 0.00e+00 | EXACT MATCH |
| Random_Forest | 0 | Weighted F1 | 0.7237349222355056 | 0.7237349222355056 | 0.00e+00 | EXACT MATCH |
| Random_Forest | 1 | Macro F1 | 0.3297590533439590 | 0.3297590533439590 | 0.00e+00 | EXACT MATCH |
| Random_Forest | 1 | Balanced Acc | 0.4142733740652704 | 0.4142733740652704 | 0.00e+00 | EXACT MATCH |
| Random_Forest | 1 | Accuracy | 0.3099498926270580 | 0.3099498926270579 | 5.55e-17 | EXACT MATCH |
| Random_Forest | 1 | Weighted F1 | 0.2678628935864791 | 0.2678628935864791 | 5.55e-17 | EXACT MATCH |
| Random_Forest | 2 | Macro F1 | 0.4510139114173847 | 0.4510139114173847 | 0.00e+00 | EXACT MATCH |
| Random_Forest | 2 | Balanced Acc | 0.6401631871267088 | 0.6401631871267088 | 0.00e+00 | EXACT MATCH |
| Random_Forest | 2 | Accuracy | 0.8172421893352126 | 0.8172421893352126 | 0.00e+00 | EXACT MATCH |
| Random_Forest | 2 | Weighted F1 | 0.8730812172891274 | 0.8730812172891274 | 0.00e+00 | EXACT MATCH |
| Random_Forest | 3 | Macro F1 | 0.3214632880005247 | 0.3214632880005247 | 0.00e+00 | EXACT MATCH |
| Random_Forest | 3 | Balanced Acc | 0.7173869321792686 | 0.7173869321792686 | 0.00e+00 | EXACT MATCH |
| Random_Forest | 3 | Accuracy | 0.4624765478424015 | 0.4624765478424015 | 0.00e+00 | EXACT MATCH |
| Random_Forest | 3 | Weighted F1 | 0.5646618753087052 | 0.5646618753087052 | 0.00e+00 | EXACT MATCH |
| Random_Forest | 4 | Macro F1 | 0.1993561759517485 | 0.1993561759517484 | 8.33e-17 | EXACT MATCH |
| Random_Forest | 4 | Balanced Acc | 0.2741902732680402 | 0.2741902732680402 | 0.00e+00 | EXACT MATCH |
| Random_Forest | 4 | Accuracy | 0.2426645519429025 | 0.2426645519429024 | 5.55e-17 | EXACT MATCH |
| Random_Forest | 4 | Weighted F1 | 0.3289655140523211 | 0.3289655140523211 | 0.00e+00 | EXACT MATCH |
| XGBoost | 0 | Macro F1 | 0.6570804742680889 | 0.6570804742680889 | 0.00e+00 | EXACT MATCH |
| XGBoost | 0 | Balanced Acc | 0.7224623687204043 | 0.7224623687204043 | 0.00e+00 | EXACT MATCH |
| XGBoost | 0 | Accuracy | 0.6807893209518282 | 0.6807893209518282 | 0.00e+00 | EXACT MATCH |
| XGBoost | 0 | Weighted F1 | 0.6849660054857367 | 0.6849660054857367 | 0.00e+00 | EXACT MATCH |
| XGBoost | 1 | Macro F1 | 0.4862749067684504 | 0.4862749067684503 | 5.55e-17 | EXACT MATCH |
| XGBoost | 1 | Balanced Acc | 0.5829210762369527 | 0.5829210762369527 | 0.00e+00 | EXACT MATCH |
| XGBoost | 1 | Accuracy | 0.3915533285612026 | 0.3915533285612025 | 5.55e-17 | EXACT MATCH |
| XGBoost | 1 | Weighted F1 | 0.3136769780281319 | 0.3136769780281319 | 0.00e+00 | EXACT MATCH |
| XGBoost | 2 | Macro F1 | 0.4169729215744313 | 0.4169729215744313 | 0.00e+00 | EXACT MATCH |
| XGBoost | 2 | Balanced Acc | 0.5925046119910066 | 0.5925046119910066 | 0.00e+00 | EXACT MATCH |
| XGBoost | 2 | Accuracy | 0.7716701902748414 | 0.7716701902748414 | 0.00e+00 | EXACT MATCH |
| XGBoost | 2 | Weighted F1 | 0.8456863023923440 | 0.8456863023923440 | 0.00e+00 | EXACT MATCH |
| XGBoost | 3 | Macro F1 | 0.3925365819161944 | 0.3925365819161944 | 0.00e+00 | EXACT MATCH |
| XGBoost | 3 | Balanced Acc | 0.7820593411170131 | 0.7820593411170131 | 0.00e+00 | EXACT MATCH |
| XGBoost | 3 | Accuracy | 0.5797373358348968 | 0.5797373358348968 | 0.00e+00 | EXACT MATCH |
| XGBoost | 3 | Weighted F1 | 0.6743157823853732 | 0.6743157823853732 | 0.00e+00 | EXACT MATCH |
| XGBoost | 4 | Macro F1 | 0.3290999427352073 | 0.3290999427352073 | 0.00e+00 | EXACT MATCH |
| XGBoost | 4 | Balanced Acc | 0.3563010427182791 | 0.3563010427182790 | 5.55e-17 | EXACT MATCH |
| XGBoost | 4 | Accuracy | 0.4559873116574147 | 0.4559873116574147 | 5.55e-17 | EXACT MATCH |
| XGBoost | 4 | Weighted F1 | 0.5645922609524943 | 0.5645922609524943 | 0.00e+00 | EXACT MATCH |
| P_MLP | 0 | Macro F1 | 0.3831388843449433 | 0.3831388843449432 | 5.55e-17 | EXACT MATCH |
| P_MLP | 0 | Balanced Acc | 0.5604800704440925 | 0.5604800704440925 | 0.00e+00 | EXACT MATCH |
| P_MLP | 0 | Accuracy | 0.3685432385374347 | 0.3685432385374347 | 0.00e+00 | EXACT MATCH |
| P_MLP | 0 | Weighted F1 | 0.3286237452238261 | 0.3286237452238261 | 5.55e-17 | EXACT MATCH |
| P_MLP | 1 | Macro F1 | 0.3543962386014189 | 0.3543962386014189 | 5.55e-17 | EXACT MATCH |
| P_MLP | 1 | Balanced Acc | 0.4518191650220316 | 0.4518191650220315 | 5.55e-17 | EXACT MATCH |
| P_MLP | 1 | Accuracy | 0.3242662848962062 | 0.3242662848962062 | 0.00e+00 | EXACT MATCH |
| P_MLP | 1 | Weighted F1 | 0.2615441813818368 | 0.2615441813818368 | 0.00e+00 | EXACT MATCH |
| P_MLP | 2 | Macro F1 | 0.4514261440231166 | 0.4514261440231166 | 0.00e+00 | EXACT MATCH |
| P_MLP | 2 | Balanced Acc | 0.6627810469069707 | 0.6627810469069707 | 0.00e+00 | EXACT MATCH |
| P_MLP | 2 | Accuracy | 0.7939863753817242 | 0.7939863753817242 | 0.00e+00 | EXACT MATCH |
| P_MLP | 2 | Weighted F1 | 0.8586213981988852 | 0.8586213981988852 | 0.00e+00 | EXACT MATCH |
| P_MLP | 3 | Macro F1 | 0.4162255403384956 | 0.4162255403384955 | 5.55e-17 | EXACT MATCH |
| P_MLP | 3 | Balanced Acc | 0.5801005814561977 | 0.5801005814561977 | 0.00e+00 | EXACT MATCH |
| P_MLP | 3 | Accuracy | 0.6454033771106942 | 0.6454033771106942 | 0.00e+00 | EXACT MATCH |
| P_MLP | 3 | Weighted F1 | 0.7321000329618843 | 0.7321000329618843 | 0.00e+00 | EXACT MATCH |
| P_MLP | 4 | Macro F1 | 0.4294151684680190 | 0.4294151684680189 | 5.55e-17 | EXACT MATCH |
| P_MLP | 4 | Balanced Acc | 0.4719827598882733 | 0.4719827598882733 | 0.00e+00 | EXACT MATCH |
| P_MLP | 4 | Accuracy | 0.7097541633624108 | 0.7097541633624108 | 0.00e+00 | EXACT MATCH |
| P_MLP | 4 | Weighted F1 | 0.7253303879791460 | 0.7253303879791460 | 0.00e+00 | EXACT MATCH |

### Table 7: Track B Recalculated Pooled Per-Class Metrics
Evaluated across all 10,770 out-of-fold predictions.

| Model | Class | Precision | Recall | F1-Score | Support | Support % |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Random_Forest | Bulk Data | 0.3304 | 0.3665 | 0.3475 | 1,348 | 12.52% |
| Random_Forest | Real-Time | 0.7916 | 0.6581 | 0.7187 | 8,132 | 75.51% |
| Random_Forest | Streaming | 0.2518 | 0.4907 | 0.3328 | 1,290 | 11.98% |
| XGBoost | Bulk Data | 0.3956 | 0.5905 | 0.4738 | 1,348 | 12.52% |
| XGBoost | Real-Time | 0.8577 | 0.6321 | 0.7278 | 8,132 | 75.51% |
| XGBoost | Streaming | 0.3183 | 0.6822 | 0.4340 | 1,290 | 11.98% |
| P_MLP | Bulk Data | 0.4135 | 0.8034 | 0.5460 | 1,348 | 12.52% |
| P_MLP | Real-Time | 0.9042 | 0.5922 | 0.7157 | 8,132 | 75.51% |
| P_MLP | Streaming | 0.2973 | 0.6512 | 0.4083 | 1,290 | 11.98% |

### Table 8: Track B Confusion Matrices (Counts & Row-Normalized %)

#### Random_Forest Confusion Matrix (Track B)

**Raw Prediction Counts:**

| True \ Pred | Bulk Data | Real-Time | Streaming | Total True |
| :--- | :---: | :---: | :---: | :---: |
| **Bulk Data** | 494 | 819 | 35 | 1,348 |
| **Real-Time** | 934 | 5,352 | 1,846 | 8,132 |
| **Streaming** | 67 | 590 | 633 | 1,290 |

**Row-Normalized Percentages (Recall per Cell):**

| True \ Pred | Bulk Data | Real-Time | Streaming | Row Sum |
| :--- | :---: | :---: | :---: | :---: |
| **Bulk Data** | 36.65% | 60.76% | 2.60% | 100.0% |
| **Real-Time** | 11.49% | 65.81% | 22.70% | 100.0% |
| **Streaming** | 5.19% | 45.74% | 49.07% | 100.0% |

#### XGBoost Confusion Matrix (Track B)

**Raw Prediction Counts:**

| True \ Pred | Bulk Data | Real-Time | Streaming | Total True |
| :--- | :---: | :---: | :---: | :---: |
| **Bulk Data** | 796 | 476 | 76 | 1,348 |
| **Real-Time** | 1,183 | 5,140 | 1,809 | 8,132 |
| **Streaming** | 33 | 377 | 880 | 1,290 |

**Row-Normalized Percentages (Recall per Cell):**

| True \ Pred | Bulk Data | Real-Time | Streaming | Row Sum |
| :--- | :---: | :---: | :---: | :---: |
| **Bulk Data** | 59.05% | 35.31% | 5.64% | 100.0% |
| **Real-Time** | 14.55% | 63.21% | 22.25% | 100.0% |
| **Streaming** | 2.56% | 29.22% | 68.22% | 100.0% |

#### P_MLP Confusion Matrix (Track B)

**Raw Prediction Counts:**

| True \ Pred | Bulk Data | Real-Time | Streaming | Total True |
| :--- | :---: | :---: | :---: | :---: |
| **Bulk Data** | 1,083 | 198 | 67 | 1,348 |
| **Real-Time** | 1,398 | 4,816 | 1,918 | 8,132 |
| **Streaming** | 138 | 312 | 840 | 1,290 |

**Row-Normalized Percentages (Recall per Cell):**

| True \ Pred | Bulk Data | Real-Time | Streaming | Row Sum |
| :--- | :---: | :---: | :---: | :---: |
| **Bulk Data** | 80.34% | 14.69% | 4.97% | 100.0% |
| **Real-Time** | 17.19% | 59.22% | 23.59% | 100.0% |
| **Streaming** | 10.70% | 24.19% | 65.12% | 100.0% |

---

## 6. Cross-Model Insights & Performance Synthesis

### 1. P_MLP Outperforms Tree Ensembles in Pooled Macro F1
A striking finding confirmed by this independent recalculation is that **P_MLP achieves the highest Pooled Macro F1 in both tracks**:
- **Track A Pooled Macro F1:** P_MLP (**0.3707**) > XGBoost (**0.3163**) > Random Forest (**0.2864**)
- **Track B Pooled Macro F1:** P_MLP (**0.5567**) > XGBoost (**0.5452**) > Random Forest (**0.4664**)

Why does P_MLP achieve superior pooled Macro F1 despite lower overall Accuracy in Track A (34.55% vs. 36.21% for XGBoost)?
- **Bulk Data Recognition:** P_MLP achieves a remarkable **0.7010 F1-score** on Bulk Data in Track A (Recall: 74.26%, Precision: 66.38%), whereas XGBoost achieves only **0.3247 F1** (Recall: 49.55%, Precision: 24.15%) and Random Forest achieves only **0.2003 F1**.
- In Track B, P_MLP maintains this advantage with Bulk Data Recall of **80.34%** (vs. 59.05% for XGBoost and 36.65% for RF), yielding Bulk Data F1 of **0.5460** (vs. 0.4738 for XGBoost).
- Because Macro F1 weights every class equally regardless of sample count, P_MLP's strong minority class detection provides a substantial boost to the global unweighted class average.

### 2. The Browsing Class Bottleneck in Track A
All three models experience catastrophic failure on the `Browsing` class in Track A:
- Random Forest Browsing F1: **0.1293** (Recall: 7.74%)
- XGBoost Browsing F1: **0.1345** (Recall: 8.08%)
- P_MLP Browsing F1: **0.0444** (Recall: 2.63%)

As revealed by the raw confusion matrices:
- Out of 6,847 true Browsing flows, Random Forest misclassifies **4,383 flows (64.01%)** as Real-Time.
- XGBoost misclassifies **4,283 flows (62.55%)** as Real-Time.
- P_MLP misclassifies **5,616 flows (82.02%)** as Real-Time.

This demonstrates that full-flow statistical aggregates (packet lengths, inter-arrival times) are insufficient to disentangle interactive web browsing from real-time audio/interactive communication under zero-leakage conditions. Addressing this bottleneck requires sequence-level early-burst modeling (Phase 7+).

---

## 7. Verification Audit Trail & Integrity Sign-Off

### Reproducibility Harness
The verification script is archived in the repository at:
`scripts/verify_phase6_metrics.py`

To re-run the verification from clean environment:
```bash
python scripts/verify_phase6_metrics.py
```

### Integrity Sign-Off
- **Data Artifacts Inspected:**
  - `output/phase6/track_a/out_of_fold_predictions.csv` (SHA256 verified)
  - `output/phase6/track_b/out_of_fold_predictions.csv` (SHA256 verified)
  - `output/phase6/track_a/aggregate_metrics.csv` (Validated)
  - `output/phase6/track_b/aggregate_metrics.csv` (Validated)
  - `output/phase6/track_a/fold_metrics.csv` (Validated)
  - `output/phase6/track_b/fold_metrics.csv` (Validated)
  - `output/phase6/track_a/confusion_matrix_counts.csv` (Validated)
  - `output/phase6/track_b/confusion_matrix_counts.csv` (Validated)
- **Total Checks Run:** 144 metric verifications, 24 per-class calculations, 24 confusion matrix rows, 8 validation fold audits.
- **Discrepancies Found:** **0 (Zero)** — all metrics match reported values exactly to standard precision limits.

This concludes the formal independent verification of Phase 6 baseline metrics.