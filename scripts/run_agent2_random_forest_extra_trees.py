#!/usr/bin/env python3
"""
scripts/run_agent2_random_forest_extra_trees.py
AGENT 2: Random Forest and Extra Trees Benchmarking under strict 29-fold LOGO.
"""

import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
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
    print("AGENT 2: RANDOM FOREST AND EXTRA TREES BENCHMARK")
    print("=" * 70)
    
    df, y, groups, classes, cls2idx, idx2cls = load_clean_dataset()
    X = df[FEATS_ROBUST8].astype(float).values
    
    candidates = [
        # Random Forest Models
        {
            "name": "Random Forest Default (100 trees, balanced)",
            "family": "RandomForest",
            "model_fn": lambda: RandomForestClassifier(
                n_estimators=100, max_depth=None, min_samples_leaf=1,
                class_weight="balanced", random_state=42, n_jobs=1
            ),
            "weight_fn": None,
            "params": {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1, "class_weight": "balanced"}
        },
        {
            "name": "Random Forest Regularized (150 trees, depth 10, leaf 3)",
            "family": "RandomForest",
            "model_fn": lambda: RandomForestClassifier(
                n_estimators=150, max_depth=10, min_samples_leaf=3,
                max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=1
            ),
            "weight_fn": None,
            "params": {"n_estimators": 150, "max_depth": 10, "min_samples_leaf": 3, "max_features": "sqrt", "class_weight": "balanced"}
        },
        {
            "name": "Random Forest Conservative (100 trees, depth 7, leaf 5)",
            "family": "RandomForest",
            "model_fn": lambda: RandomForestClassifier(
                n_estimators=100, max_depth=7, min_samples_leaf=5,
                max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=1
            ),
            "weight_fn": None,
            "params": {"n_estimators": 100, "max_depth": 7, "min_samples_leaf": 5, "max_features": "sqrt", "class_weight": "balanced"}
        },
        {
            "name": "Random Forest + Bulk Boost 1.20x",
            "family": "RandomForest",
            "model_fn": lambda: RandomForestClassifier(
                n_estimators=150, max_depth=10, min_samples_leaf=3,
                max_features="sqrt", random_state=42, n_jobs=1
            ),
            "weight_fn": lambda y_tr, nc: bulk_boost_weights(y_tr, nc, boost=1.20),
            "params": {"n_estimators": 150, "max_depth": 10, "min_samples_leaf": 3, "weights": "bulk_boost_1.20"}
        },
        
        # Extra Trees Models
        {
            "name": "Extra Trees Default (100 trees, balanced)",
            "family": "ExtraTrees",
            "model_fn": lambda: ExtraTreesClassifier(
                n_estimators=100, max_depth=None, min_samples_leaf=1,
                class_weight="balanced", random_state=42, n_jobs=1
            ),
            "weight_fn": None,
            "params": {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1, "class_weight": "balanced"}
        },
        {
            "name": "Extra Trees Regularized (150 trees, depth 10, leaf 3)",
            "family": "ExtraTrees",
            "model_fn": lambda: ExtraTreesClassifier(
                n_estimators=150, max_depth=10, min_samples_leaf=3,
                max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=1
            ),
            "weight_fn": None,
            "params": {"n_estimators": 150, "max_depth": 10, "min_samples_leaf": 3, "max_features": "sqrt", "class_weight": "balanced"}
        },
        {
            "name": "Extra Trees + Bulk Boost 1.20x",
            "family": "ExtraTrees",
            "model_fn": lambda: ExtraTreesClassifier(
                n_estimators=150, max_depth=10, min_samples_leaf=3,
                max_features="sqrt", random_state=42, n_jobs=1
            ),
            "weight_fn": lambda y_tr, nc: bulk_boost_weights(y_tr, nc, boost=1.20),
            "params": {"n_estimators": 150, "max_depth": 10, "min_samples_leaf": 3, "weights": "bulk_boost_1.20"}
        },
    ]
    
    results = []
    for cand in candidates:
        print(f"Evaluating {cand['name']}...", flush=True)
        res = evaluate_model_logo(
            X, y, groups, cand["model_fn"], weight_fn=cand["weight_fn"],
            scale_features=False, num_classes=len(classes)
        )
        print(f"  -> Macro F1: {res['macro_f1']:.4f} | Bal Acc: {res['balanced_acc']:.4f} | Acc: {res['overall_acc']:.4f} | Time: {res['runtime_sec']}s", flush=True)
        
        oof_path = OUTPUT_DIR / f"oof_agent2_{cand['name'].replace(' ', '_').replace('+', '_').replace('(', '').replace(')', '').replace(',', '')}.npy"
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
        
    out_file = OUTPUT_DIR / "agent2_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[DONE] Agent 2 results saved to {out_file}")

if __name__ == "__main__":
    main()
