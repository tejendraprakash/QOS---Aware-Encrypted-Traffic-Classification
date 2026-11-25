#!/usr/bin/env python3
"""
scripts/run_agent5_xgboost_optimization.py
AGENT 5: Systematic XGBoost Hyperparameter Optimization & Class Weight Tuning under strict 29-fold LOGO.
"""

import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from xgboost import XGBClassifier
from campaign_evaluation_engine import (
    load_clean_dataset,
    evaluate_model_logo,
    standard_weights,
    bulk_boost_weights,
    FEATS_ROBUST8,
    OUTPUT_DIR
)

def main():
    print("=" * 70)
    print("AGENT 5: XGBOOST HYPERPARAMETER OPTIMIZATION")
    print("=" * 70)
    
    df, y, groups, classes, cls2idx, idx2cls = load_clean_dataset()
    X = df[FEATS_ROBUST8].astype(float).values
    
    configurations = [
        {
            "name": "XGBoost Champion Anchor (depth 5, lr 0.10, Bulk 1.20x)",
            "model_fn": lambda: XGBClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.10,
                subsample=0.8, colsample_bytree=0.8,
                objective="multi:softprob", eval_metric="mlogloss",
                random_state=42, n_jobs=1
            ),
            "weight_fn": lambda y_tr, nc: bulk_boost_weights(y_tr, nc, boost=1.20),
            "params": {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.10, "subsample": 0.8, "colsample_bytree": 0.8, "boost": 1.20}
        },
        {
            "name": "XGBoost CFG-A: Deeper + L1/L2 Regularization (depth 6, lr 0.07)",
            "model_fn": lambda: XGBClassifier(
                n_estimators=120, max_depth=6, learning_rate=0.07,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=2,
                reg_alpha=0.1, reg_lambda=1.5,
                objective="multi:softprob", eval_metric="mlogloss",
                random_state=42, n_jobs=1
            ),
            "weight_fn": lambda y_tr, nc: bulk_boost_weights(y_tr, nc, boost=1.20),
            "params": {"n_estimators": 120, "max_depth": 6, "learning_rate": 0.07, "reg_alpha": 0.1, "reg_lambda": 1.5, "boost": 1.20}
        },
        {
            "name": "XGBoost CFG-B: Shallow & Robust (depth 4, lr 0.08, min_child 2)",
            "model_fn": lambda: XGBClassifier(
                n_estimators=120, max_depth=4, learning_rate=0.08,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=2,
                reg_alpha=0.2, reg_lambda=1.2,
                objective="multi:softprob", eval_metric="mlogloss",
                random_state=42, n_jobs=1
            ),
            "weight_fn": lambda y_tr, nc: bulk_boost_weights(y_tr, nc, boost=1.20),
            "params": {"n_estimators": 120, "max_depth": 4, "learning_rate": 0.08, "reg_alpha": 0.2, "reg_lambda": 1.2, "boost": 1.20}
        },
        {
            "name": "XGBoost CFG-C: Low Learning Rate Ensemble (depth 5, lr 0.05, n_est 180)",
            "model_fn": lambda: XGBClassifier(
                n_estimators=180, max_depth=5, learning_rate=0.05,
                subsample=0.85, colsample_bytree=0.85, min_child_weight=2,
                reg_lambda=1.2,
                objective="multi:softprob", eval_metric="mlogloss",
                random_state=42, n_jobs=1
            ),
            "weight_fn": lambda y_tr, nc: bulk_boost_weights(y_tr, nc, boost=1.20),
            "params": {"n_estimators": 180, "max_depth": 5, "learning_rate": 0.05, "subsample": 0.85, "colsample_bytree": 0.85, "boost": 1.20}
        },
        {
            "name": "XGBoost CFG-D: Bulk Boost Fine-Tuning (Bulk 1.15x)",
            "model_fn": lambda: XGBClassifier(
                n_estimators=120, max_depth=5, learning_rate=0.08,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=2,
                reg_alpha=0.1, reg_lambda=1.5,
                objective="multi:softprob", eval_metric="mlogloss",
                random_state=42, n_jobs=1
            ),
            "weight_fn": lambda y_tr, nc: bulk_boost_weights(y_tr, nc, boost=1.15),
            "params": {"n_estimators": 120, "max_depth": 5, "learning_rate": 0.08, "boost": 1.15}
        },
        {
            "name": "XGBoost CFG-E: Bulk Boost Aggressive (Bulk 1.30x)",
            "model_fn": lambda: XGBClassifier(
                n_estimators=120, max_depth=5, learning_rate=0.08,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=2,
                reg_alpha=0.1, reg_lambda=1.5,
                objective="multi:softprob", eval_metric="mlogloss",
                random_state=42, n_jobs=1
            ),
            "weight_fn": lambda y_tr, nc: bulk_boost_weights(y_tr, nc, boost=1.30),
            "params": {"n_estimators": 120, "max_depth": 5, "learning_rate": 0.08, "boost": 1.30}
        },
        {
            "name": "XGBoost CFG-F: Conservative Shallow (depth 3, lr 0.10)",
            "model_fn": lambda: XGBClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.10,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                objective="multi:softprob", eval_metric="mlogloss",
                random_state=42, n_jobs=1
            ),
            "weight_fn": lambda y_tr, nc: bulk_boost_weights(y_tr, nc, boost=1.20),
            "params": {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.10, "min_child_weight": 3, "boost": 1.20}
        },
        {
            "name": "XGBoost CFG-G: Gamma Regularized (gamma 0.2, subsample 0.75)",
            "model_fn": lambda: XGBClassifier(
                n_estimators=120, max_depth=5, learning_rate=0.08,
                gamma=0.2, subsample=0.75, colsample_bytree=0.75,
                reg_alpha=0.1, reg_lambda=1.5,
                objective="multi:softprob", eval_metric="mlogloss",
                random_state=42, n_jobs=1
            ),
            "weight_fn": lambda y_tr, nc: bulk_boost_weights(y_tr, nc, boost=1.20),
            "params": {"n_estimators": 120, "max_depth": 5, "learning_rate": 0.08, "gamma": 0.2, "subsample": 0.75, "boost": 1.20}
        },
    ]
    
    results = []
    for cand in configurations:
        print(f"Evaluating {cand['name']}...", flush=True)
        res = evaluate_model_logo(
            X, y, groups, cand["model_fn"], weight_fn=cand["weight_fn"],
            scale_features=False, num_classes=len(classes)
        )
        print(f"  -> Macro F1: {res['macro_f1']:.4f} | Bal Acc: {res['balanced_acc']:.4f} | Acc: {res['overall_acc']:.4f} | Time: {res['runtime_sec']}s", flush=True)
        
        oof_path = OUTPUT_DIR / f"oof_agent5_{cand['name'][:25].replace(' ', '_').replace(':', '').replace('-', '_')}.npy"
        np.save(oof_path, res["oof_preds"])
        
        results.append({
            "name": cand["name"],
            "family": "XGBoost",
            "features": "Robust 8-Feature",
            "feature_count": 8,
            "params": cand["params"],
            "macro_f1": res["macro_f1"],
            "balanced_acc": res["balanced_acc"],
            "overall_acc": res["overall_acc"],
            "per_class_f1": res["per_class_f1"],
            "per_class_rec": res["per_class_rec"],
            "per_class_prec": res["per_class_prec"],
            "confusion_matrix": res["confusion_matrix"],
            "per_group_f1s": res["per_group_f1s"],
            "runtime_sec": res["runtime_sec"],
            "oof_file": str(oof_path.name)
        })
        
    out_file = OUTPUT_DIR / "agent5_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[DONE] Agent 5 results saved to {out_file}")

if __name__ == "__main__":
    main()
