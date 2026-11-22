# Cross-Validation & Metric Evaluation Protocol

**Project:** QoS-Aware Encrypted Network Traffic Classification for Next-Generation Networks  
**Phase:** 5 — Baseline Experiment Preparation & Protocol Validation  
**Date:** November 2026  
**Fold Specifications:** [cross_validation_splits.json](file:///c:/Users/saipr/OneDrive/Desktop/EncryptedTrafficResearch/output/phase5/cross_validation_splits.json)  
**Analysis Tables:** [track_a_fold_analysis.csv](file:///c:/Users/saipr/OneDrive/Desktop/EncryptedTrafficResearch/output/phase5/track_a_fold_analysis.csv), [track_b_fold_analysis.csv](file:///c:/Users/saipr/OneDrive/Desktop/EncryptedTrafficResearch/output/phase5/track_b_fold_analysis.csv)

---

## 1. Motivation: Preventing Group Leakage in Network Traffic

In network traffic classification research, standard $K$-fold cross-validation is fatally flawed: flows extracted from the same packet capture session share the same client/server hardware configurations, TCP stack implementations, operating system buffer sizes, network path MTU, and background noise. When flows from the same capture are randomly partitioned into training and test sets, machine learning models achieve spuriously inflated accuracies (>99%) by memorizing capture-specific session artifacts rather than learning generalizable QoS behavioral patterns.

To eliminate this threat entirely, our baseline protocol enforces **Strict Group Cross-Validation** using `StratifiedGroupKFold`.

---

## 2. Formal Split Definitions

Let $\mathcal{D} = \{(x_i, y_i, g_i)\}_{i=1}^N$ denote the dataset where $x_i \in \mathbb{R}^D$ is the feature vector, $y_i \in \mathcal{C}$ is the QoS class, and $g_i \in \mathcal{G}$ is the capture group.

For each fold $k \in \{0, \dots, K-1\}$, the dataset is partitioned into training set $\mathcal{D}_{train}^{(k)}$ and validation set $\mathcal{D}_{val}^{(k)}$ such that:

1. **Strict Group Disjointness:**
   $$\mathcal{G}_{train}^{(k)} \cap \mathcal{G}_{val}^{(k)} = \emptyset \quad \forall k$$
2. **Complete Class Coverage:**
   $$\mathcal{C}_{train}^{(k)} = \mathcal{C}_{val}^{(k)} = \mathcal{C} \quad \forall k$$
3. **Partition Completeness:**
   $$\bigcup_{k=0}^{K-1} \mathcal{D}_{val}^{(k)} = \mathcal{D} \quad \text{and} \quad \mathcal{D}_{val}^{(j)} \cap \mathcal{D}_{val}^{(k)} = \emptyset \quad \forall j \neq k$$

---

## 3. Comparative Summary: Track A vs. Track B Validation

| Dimension | Track A (Exploratory 4-Class) | Track B (Primary 3-Class) |
|---|---|---|
| **Classes Evaluated** | Bulk Data, Browsing, Streaming, Real-Time | Bulk Data, Streaming, Real-Time |
| **Total Flows** | 17,617 | 10,770 |
| **Unique Capture Groups** | 32 groups | 29 groups |
| **Cross-Validation Scheme** | 3-Fold `StratifiedGroupKFold` | 5-Fold `StratifiedGroupKFold` |
| **Random State / Seed** | Seed = 7 | Seed = 2 |
| **Limiting Class** | Browsing (3 groups: `GRP_EMAIL_1`, `GRP_EMAIL_2`, `GRP_VPN_EMAIL_2`) | Streaming (9 groups) |
| **Group Overlap Across Folds** | Exactly 0 (Mathematically Verified) | Exactly 0 (Mathematically Verified) |
| **Class Coverage Across Folds** | 100% across all 3 folds (4/4 classes) | 100% across all 5 folds (3/3 classes) |
| **Evaluation Rationale** | 3 folds is the theoretical maximum for 3 groups. | 5 folds provides standard benchmark reliability. |

---

## 4. Evaluation Metrics

Because the dataset exhibits a $6.30 : 1$ class imbalance, traditional overall Accuracy is severely misleading (a trivial classifier predicting exclusively `Real-Time` would achieve 46% accuracy in Track A and 75.5% accuracy in Track B). We mandate the following metric hierarchy:

### 4.1 Primary Metric: Macro-Averaged F1-Score
$$\text{Macro F1} = \frac{1}{|\mathcal{C}|} \sum_{c \in \mathcal{C}} F1_c$$
Where $F1_c = \frac{2 \cdot P_c \cdot R_c}{P_c + R_c}$. Macro F1 weights every class equally, heavily penalizing models that collapse on minority classes (Bulk Data and Streaming).

### 4.2 Secondary Metric: Balanced Accuracy
$$\text{Balanced Accuracy} = \frac{1}{|\mathcal{C}|} \sum_{c \in \mathcal{C}} \frac{TP_c}{TP_c + FN_c}$$
Measures the unweighted average of recall obtained on each class, providing an intuitive, skew-resilient performance gauge.

### 4.3 Mandatory Diagnostic Outputs
For every fold and overall out-of-fold predictions, models must report:
1. Full confusion matrix ($|\mathcal{C}| \times |\mathcal{C}|$).
2. Per-class Precision, Recall, and F1-score.
3. Cohen's Kappa score ($\kappa$) to measure inter-rater agreement above chance.
4. Mean inference time per flow (latency in microseconds).

---

## 5. Strict Prohibition on Synthetic Resampling

> [!CAUTION]
> **Oversampling / SMOTE Policy:**
> Synthetic oversampling techniques (such as SMOTE, ADASYN, or random duplication) are **strictly forbidden** in our baseline validation protocol. Applying SMOTE across grouped network data generates synthetic flows that blur capture boundaries and induce severe cross-fold leakage. Class imbalance must be addressed purely through algorithmic loss weighting (e.g., `class_weight='balanced'` in Random Forest, inverse frequency sample weights in XGBoost, and class-weighted Cross-Entropy loss in P-MLP).
