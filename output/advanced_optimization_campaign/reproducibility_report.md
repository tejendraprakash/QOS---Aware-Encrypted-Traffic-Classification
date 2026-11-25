# Advanced Optimization Campaign: Reproducibility Report

## Environment & Execution Manifest
- **Platform:** Windows x86_64, Python 3.12+
- **Core Dependencies:** scikit-learn, xgboost, lightgbm, catboost, pandas, numpy, scipy
- **Execution Script:** `scripts/adv_campaign_engine.py`, `scripts/run_adv_agent*.py`
- **Dataset Hash Verification:** Clean Track B dataset (`output/phase6d/clean_track_b_flows.csv`), 4,196 flows, 29 groups.
- **Random Seeds:** Fixed at 42 across all probabilistic splits and estimators.

## Exact Metric Calculation Standard:
- All metrics are computed strictly from the pooled $N=4,196$ out-of-fold prediction vector:
  - Overall Accuracy: `sklearn.metrics.accuracy_score(y, oof_preds)`
  - Macro F1: `sklearn.metrics.f1_score(y, oof_preds, average='macro')`
  - Balanced Accuracy: `sklearn.metrics.balanced_accuracy_score(y, oof_preds)`
  - Confusion Matrix: `sklearn.metrics.confusion_matrix(y, oof_preds, normalize='true')`
