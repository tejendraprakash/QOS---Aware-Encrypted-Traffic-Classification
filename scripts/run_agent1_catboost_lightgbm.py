#!/usr/bin/env python3
"""
scripts/run_agent1_catboost_lightgbm.py
AGENT 1: CatBoost and LightGBM Benchmarking under strict 29-fold LOGO.
"""

import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
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
    print("AGENT 1: CATBOOST AND LIGHTGBM BENCHMARK")
    print("=" * 70)
    
    df, y, groups, classes, cls2idx, idx2cls = load_clean_dataset()
    X = df[FEATS_ROBUST8].astype(float).values
    
    candidates = [
        # LightGBM Models
        {
            "name": "LightGBM Default",
            "family": "LightGBM",
            "model_fn": lambda: LGBMClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=-1, num_leaves=31,
                random_state=42, verbose=-1, n_jobs=1
            ),
            "weight_fn": standard_weights,
            "params": {"n_estimators": 100, "learning_rate": 0.1, "num_leaves": 31, "weights": "standard"}
        },
        {
            "name": "LightGBM Tuned (Depth 5, Leaves 20)",
            "family": "LightGBM",
            "model_fn": lambda: LGBMClassifier(
                n_estimators=120, learning_rate=0.08, max_depth=5, num_leaves=20,
                min_child_samples=15, colsample_bytree=0.8, subsample=0.8,
                random_state=42, verbose=-1, n_jobs=1
            ),
            "weight_fn": standard_weights,
            "params": {"n_estimators": 120, "learning_rate": 0.08, "max_depth": 5, "num_leaves": 20, "weights": "standard"}
        },
        {
            "name": "LightGBM + Bulk Boost 1.20x",
            "family": "LightGBM",
            "model_fn": lambda: LGBMClassifier(
                n_estimators=120, learning_rate=0.08, max_depth=5, num_leaves=20,
                min_child_samples=15, colsample_bytree=0.8, subsample=0.8,
                random_state=42, verbose=-1, n_jobs=1
            ),
            "weight_fn": lambda y_tr, nc: bulk_boost_weights(y_tr, nc, boost=1.20),
            "params": {"n_estimators": 120, "learning_rate": 0.08, "max_depth": 5, "num_leaves": 20, "weights": "bulk_boost_1.20"}
        },
        {
            "name": "LightGBM Shallow (Depth 3, Leaves 8)",
            "family": "LightGBM",
            "model_fn": lambda: LGBMClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=3, num_leaves=8,
                min_child_samples=20, random_state=42, verbose=-1, n_jobs=1
            ),
            "weight_fn": standard_weights,
            "params": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3, "num_leaves": 8, "weights": "standard"}
        },
        
        # CatBoost Models
        {
            "name": "CatBoost Default",
            "family": "CatBoost",
            "model_fn": lambda: CatBoostClassifier(
                iterations=100, learning_rate=0.1, depth=6,
                random_seed=42, verbose=0, thread_count=1
            ),
            "weight_fn": standard_weights,
            "params": {"iterations": 100, "learning_rate": 0.1, "depth": 6, "weights": "standard"}
        },
        {
            "name": "CatBoost Tuned (Depth 5, Iter 150)",
            "family": "CatBoost",
            "model_fn": lambda: CatBoostClassifier(
                iterations=150, learning_rate=0.08, depth=5, l2_leaf_reg=3.0,
                random_seed=42, verbose=0, thread_count=1
            ),
            "weight_fn": standard_weights,
            "params": {"iterations": 150, "learning_rate": 0.08, "depth": 5, "l2_leaf_reg": 3.0, "weights": "standard"}
        },
        {
            "name": "CatBoost + Bulk Boost 1.20x",
            "family": "CatBoost",
            "model_fn": lambda: CatBoostClassifier(
                iterations=150, learning_rate=0.08, depth=5, l2_leaf_reg=3.0,
                random_seed=42, verbose=0, thread_count=1
            ),
            "weight_fn": lambda y_tr, nc: bulk_boost_weights(y_tr, nc, boost=1.20),
            "params": {"iterations": 150, "learning_rate": 0.08, "depth": 5, "l2_leaf_reg": 3.0, "weights": "bulk_boost_1.20"}
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
        
        # Save OOF predictions to disk
        oof_path = OUTPUT_DIR / f"oof_agent1_{cand['name'].replace(' ', '_').replace('+', '_').replace('(', '').replace(')', '')}.npy"
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
        
    out_file = OUTPUT_DIR / "agent1_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[DONE] Agent 1 results saved to {out_file}")

if __name__ == "__main__":
    main()
