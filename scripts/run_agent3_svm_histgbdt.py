#!/usr/bin/env python3
"""
scripts/run_agent3_svm_histgbdt.py
AGENT 3: Support Vector Machine (RBF) and HistGradientBoosting under strict 29-fold LOGO.
Features strict in-fold feature scaling for SVM to prevent any test-fold data leakage.
"""

import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.svm import SVC
from sklearn.ensemble import HistGradientBoostingClassifier
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
    print("AGENT 3: SVM (RBF) AND HIST-GBDT BENCHMARK")
    print("=" * 70)
    
    df, y, groups, classes, cls2idx, idx2cls = load_clean_dataset()
    X = df[FEATS_ROBUST8].astype(float).values
    
    candidates = [
        # RBF SVM Models (with strict in-fold StandardScaler)
        {
            "name": "RBF SVM (C=1.0, balanced, in-fold scaled)",
            "family": "SVM",
            "model_fn": lambda: SVC(
                C=1.0, kernel="rbf", gamma="scale", class_weight="balanced",
                random_state=42
            ),
            "weight_fn": None,
            "scale": True,
            "params": {"C": 1.0, "kernel": "rbf", "gamma": "scale", "scale": True, "class_weight": "balanced"}
        },
        {
            "name": "RBF SVM (C=10.0, balanced, in-fold scaled)",
            "family": "SVM",
            "model_fn": lambda: SVC(
                C=10.0, kernel="rbf", gamma="scale", class_weight="balanced",
                random_state=42
            ),
            "weight_fn": None,
            "scale": True,
            "params": {"C": 10.0, "kernel": "rbf", "gamma": "scale", "scale": True, "class_weight": "balanced"}
        },
        {
            "name": "RBF SVM (C=0.5, balanced, in-fold scaled)",
            "family": "SVM",
            "model_fn": lambda: SVC(
                C=0.5, kernel="rbf", gamma="scale", class_weight="balanced",
                random_state=42
            ),
            "weight_fn": None,
            "scale": True,
            "params": {"C": 0.5, "kernel": "rbf", "gamma": "scale", "scale": True, "class_weight": "balanced"}
        },
        
        # HistGradientBoosting Models
        {
            "name": "HistGBDT Default (max_iter 100)",
            "family": "HistGBDT",
            "model_fn": lambda: HistGradientBoostingClassifier(
                max_iter=100, max_depth=5, learning_rate=0.1,
                random_state=42
            ),
            "weight_fn": standard_weights,
            "scale": False,
            "params": {"max_iter": 100, "max_depth": 5, "learning_rate": 0.1, "weights": "standard"}
        },
        {
            "name": "HistGBDT Tuned (max_iter 120, depth 5, l2=1.0)",
            "family": "HistGBDT",
            "model_fn": lambda: HistGradientBoostingClassifier(
                max_iter=120, max_depth=5, learning_rate=0.08, min_samples_leaf=15,
                l2_regularization=1.0, random_state=42
            ),
            "weight_fn": standard_weights,
            "scale": False,
            "params": {"max_iter": 120, "max_depth": 5, "learning_rate": 0.08, "min_samples_leaf": 15, "l2": 1.0, "weights": "standard"}
        },
        {
            "name": "HistGBDT + Bulk Boost 1.20x",
            "family": "HistGBDT",
            "model_fn": lambda: HistGradientBoostingClassifier(
                max_iter=120, max_depth=5, learning_rate=0.08, min_samples_leaf=15,
                l2_regularization=1.0, random_state=42
            ),
            "weight_fn": lambda y_tr, nc: bulk_boost_weights(y_tr, nc, boost=1.20),
            "scale": False,
            "params": {"max_iter": 120, "max_depth": 5, "learning_rate": 0.08, "min_samples_leaf": 15, "l2": 1.0, "weights": "bulk_boost_1.20"}
        },
    ]
    
    results = []
    for cand in candidates:
        print(f"Evaluating {cand['name']}...", flush=True)
        res = evaluate_model_logo(
            X, y, groups, cand["model_fn"], weight_fn=cand["weight_fn"],
            scale_features=cand["scale"], num_classes=len(classes)
        )
        print(f"  -> Macro F1: {res['macro_f1']:.4f} | Bal Acc: {res['balanced_acc']:.4f} | Acc: {res['overall_acc']:.4f} | Time: {res['runtime_sec']}s", flush=True)
        
        oof_path = OUTPUT_DIR / f"oof_agent3_{cand['name'].replace(' ', '_').replace('+', '_').replace('(', '').replace(')', '').replace('=', '').replace(',', '')}.npy"
        np.save(oof_path, res["oof_preds"])
        
        results.append({
            "name": cand["name"],
            "family": cand["family"],
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
        
    out_file = OUTPUT_DIR / "agent3_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[DONE] Agent 3 results saved to {out_file}")

if __name__ == "__main__":
    main()
