# Model Improvement Campaign: Best Model Report

## 1. Executive Summary
The comprehensive model improvement campaign evaluated candidate model families (CatBoost, LightGBM, Random Forest, Extra Trees, Support Vector Machines, HistGradientBoosting, and Tuned XGBoost) across strict 29-fold Leave-One-Group-Out (LOGO) cross-validation on the 4,196-flow Track B clean dataset.

- **Baseline Anchor (6 Feats, XGBoost):** Macro F1 = 0.5784, Accuracy = 0.5777, Balanced Accuracy = 0.5966
- **Previous Champion (8 Feats, XGBoost Default):** Macro F1 = 0.5832, Accuracy = 0.5837, Balanced Accuracy = 0.6017
- **Best Validated Model:** **RBF SVM (C=10.0, balanced, in-fold scaled)**
  - **Macro F1:** **0.6188** (+0.0356 vs Champion, +0.0404 vs Baseline)
  - **Overall Accuracy:** **0.6158** (61.58%)
  - **Balanced Accuracy:** **0.6736** (67.36%)
  - **Bulk Data Recall:** **0.7018** | F1: **0.6688**
  - **Real-Time Recall:** **0.4694** | F1: **0.5929**
  - **Streaming Recall:** **0.8495** | F1: **0.5948**
  - **Runtime:** 50.2 seconds (29 folds)

---

## 2. Top-5 Performing Architectures

| Rank | Model Name | Family | Macro F1 | Balanced Acc | Accuracy | Real-Time Rec | Streaming Rec | Bulk Data Rec | Runtime |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | RBF SVM (C=10.0, balanced, in-fold scaled) | SVM | **0.6188** | 0.6736 | 0.6158 | 0.4694 | 0.8495 | 0.7018 | 50.2s |
| 2 | CatBoost Default | CatBoost | **0.6060** | 0.6273 | 0.6072 | 0.5778 | 0.7603 | 0.5438 | 79.06s |
| 3 | HistGBDT Default (max_iter 100) | HistGBDT | **0.6005** | 0.6109 | 0.6051 | 0.6140 | 0.7090 | 0.5098 | 200.7s |
| 4 | XGBoost CFG-D: Bulk Boost Fine-Tuning (Bulk 1.15x) | XGBoost | **0.5987** | 0.6139 | 0.6001 | 0.5848 | 0.7224 | 0.5344 | 29.18s |
| 5 | CatBoost + Bulk Boost 1.20x | CatBoost | **0.5928** | 0.6146 | 0.5887 | 0.5339 | 0.7313 | 0.5786 | 65.69s |

---

## 3. Confusion Matrix (Best Model)

```
                Pred Bulk   Pred RT  Pred Streaming
True Bulk        0.701784  0.107052        0.191164
True RT          0.211122  0.469369        0.319510
True Streaming   0.021182  0.129320        0.849498
```

---

## 4. Statistical Significance
- **McNemar Binomial Test (Best vs Baseline):** p = 0.0
- **McNemar Binomial Test (Best vs Champion):** p = 0.0003
- **Wilcoxon Signed-Rank Test (Fold-wise):** p = 0.0981

---

## 5. Hyperparameter Specification
```json
{
  "C": 10.0,
  "kernel": "rbf",
  "gamma": "scale",
  "scale": true,
  "class_weight": "balanced"
}
```

---

## 6. Recommendations
1. The evaluated results demonstrate that Early-5 causal classification is bounded by the intrinsic overlap between conversational real-time flows and bursty downloads in the initial 5 packets.
2. The current champion architecture remains highly resilient across 29 capture groups without overfitting.
