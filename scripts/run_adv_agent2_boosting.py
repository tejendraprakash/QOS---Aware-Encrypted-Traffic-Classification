#!/usr/bin/env python3
"""
scripts/run_adv_agent2_boosting.py
AGENT 2: Advanced Boosting Optimization (CatBoost, LightGBM, XGBoost, HistGBDT).
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from adv_campaign_engine import (
    load_advanced_dataset,
    evaluate_adv_model_logo,
    standard_weights,
    bulk_boost_weights,
    FEATS_CHAMPION8,
    FEATS_CAUSAL9,
    OUTPUT_DIR
)

def main():
    print("=" * 70)
    print("AGENT 2: ADVANCED BOOSTING OPTIMIZATION")
    print("=" * 70)
    
    df, y, groups, classes, cls2idx, idx2cls = load_advanced_dataset()
    X8 = df[FEATS_CHAMPION8].values.astype(float)
    X9 = df[FEATS_CAUSAL9].values.astype(float)
    
    results = []
    
    # 1. CatBoost Variations
    catboost_cfgs = [
        ("CatBoost Default (Depth 6, Iter 1000)", {"depth": 6, "iterations": 500, "verbose": 0, "random_seed": 42}, X8, FEATS_CHAMPION8, None, "catboost_default"),
        ("CatBoost Compact (Depth 4, Iter 250, l2=5)", {"depth": 4, "iterations": 250, "l2_leaf_reg": 5.0, "verbose": 0, "random_seed": 42}, X8, FEATS_CHAMPION8, None, "catboost_depth4"),
        ("CatBoost Causal 9-Feature (Depth 5, Iter 300)", {"depth": 5, "iterations": 300, "l2_leaf_reg": 3.0, "verbose": 0, "random_seed": 42}, X9, FEATS_CAUSAL9, None, "catboost_causal9"),
        ("CatBoost + Bulk Boost 1.15x (Depth 5)", {"depth": 5, "iterations": 300, "verbose": 0, "random_seed": 42}, X8, FEATS_CHAMPION8, lambda y_tr, num_classes=3: bulk_boost_weights(y_tr, num_classes, boost=1.15), "catboost_bulk115"),
    ]
    for name, p, feat_mat, feat_names, w_fn, prefix in catboost_cfgs:
        print(f"Evaluating {name}...")
        res = evaluate_adv_model_logo(
            feat_mat, y, groups,
            model_fn=lambda p_dict=p: CatBoostClassifier(**p_dict),
            weight_fn=w_fn,
            save_prefix=prefix
        )
        res.update({
            "name": name,
            "family": "CatBoost",
            "features": "Champion 8" if len(feat_names) == 8 else "Causal 9",
            "feature_count": len(feat_names),
            "params": p
        })
        results.append(res)
        print(f"  -> Macro F1: {res['macro_f1']:.4f} | Bal Acc: {res['balanced_acc']:.4f} | Acc: {res['overall_acc']:.4f} | Time: {res['runtime_sec']}s")

    # 2. HistGradientBoosting Variations
    hist_cfgs = [
        ("HistGBDT Baseline (max_iter 100)", {"max_iter": 100, "random_state": 42}, X8, FEATS_CHAMPION8, None, "histgbdt_default"),
        ("HistGBDT Regularized (iter 150, depth 4, l2=2.0)", {"max_iter": 150, "max_depth": 4, "l2_regularization": 2.0, "random_state": 42}, X8, FEATS_CHAMPION8, None, "histgbdt_reg"),
        ("HistGBDT Causal 9-Feature (iter 120, depth 5)", {"max_iter": 120, "max_depth": 5, "random_state": 42}, X9, FEATS_CAUSAL9, None, "histgbdt_causal9"),
        ("HistGBDT + Bulk Boost 1.15x", {"max_iter": 120, "max_depth": 5, "random_state": 42}, X8, FEATS_CHAMPION8, lambda y_tr, num_classes=3: bulk_boost_weights(y_tr, num_classes, boost=1.15), "histgbdt_bulk115")
    ]
    for name, p, feat_mat, feat_names, w_fn, prefix in hist_cfgs:
        print(f"Evaluating {name}...")
        res = evaluate_adv_model_logo(
            feat_mat, y, groups,
            model_fn=lambda p_dict=p: HistGradientBoostingClassifier(**p_dict),
            weight_fn=w_fn,
            save_prefix=prefix
        )
        res.update({
            "name": name,
            "family": "HistGBDT",
            "features": "Champion 8" if len(feat_names) == 8 else "Causal 9",
            "feature_count": len(feat_names),
            "params": p
        })
        results.append(res)
        print(f"  -> Macro F1: {res['macro_f1']:.4f} | Bal Acc: {res['balanced_acc']:.4f} | Acc: {res['overall_acc']:.4f} | Time: {res['runtime_sec']}s")

    # 3. XGBoost Fine-Tuned Variations
    xgb_cfgs = [
        ("XGBoost CFG-D Optimal (depth 5, lr 0.08, Bulk 1.15x)", {"max_depth": 5, "learning_rate": 0.08, "n_estimators": 100, "min_child_weight": 2, "reg_alpha": 0.1, "reg_lambda": 1.5, "random_state": 42, "n_jobs": 2, "eval_metric": "mlogloss"}, X8, FEATS_CHAMPION8, lambda y_tr, num_classes=3: bulk_boost_weights(y_tr, num_classes, boost=1.15), "xgb_cfg_d"),
        ("XGBoost Causal 9-Feature (depth 5, lr 0.08, Bulk 1.15x)", {"max_depth": 5, "learning_rate": 0.08, "n_estimators": 100, "min_child_weight": 2, "reg_alpha": 0.1, "reg_lambda": 1.5, "random_state": 42, "n_jobs": 2, "eval_metric": "mlogloss"}, X9, FEATS_CAUSAL9, lambda y_tr, num_classes=3: bulk_boost_weights(y_tr, num_classes, boost=1.15), "xgb_causal9"),
        ("XGBoost Subsampled (subsample 0.8, colsample 0.85, depth 4)", {"max_depth": 4, "learning_rate": 0.07, "n_estimators": 120, "subsample": 0.8, "colsample_bytree": 0.85, "reg_lambda": 2.0, "random_state": 42, "n_jobs": 2, "eval_metric": "mlogloss"}, X8, FEATS_CHAMPION8, lambda y_tr, num_classes=3: bulk_boost_weights(y_tr, num_classes, boost=1.15), "xgb_subsample"),
    ]
    for name, p, feat_mat, feat_names, w_fn, prefix in xgb_cfgs:
        print(f"Evaluating {name}...")
        res = evaluate_adv_model_logo(
            feat_mat, y, groups,
            model_fn=lambda p_dict=p: XGBClassifier(**p_dict),
            weight_fn=w_fn,
            save_prefix=prefix
        )
        res.update({
            "name": name,
            "family": "XGBoost",
            "features": "Champion 8" if len(feat_names) == 8 else "Causal 9",
            "feature_count": len(feat_names),
            "params": p
        })
        results.append(res)
        print(f"  -> Macro F1: {res['macro_f1']:.4f} | Bal Acc: {res['balanced_acc']:.4f} | Acc: {res['overall_acc']:.4f} | Time: {res['runtime_sec']}s")

    # 4. LightGBM Variations
    lgb_cfgs = [
        ("LightGBM Tuned (Depth 5, Leaves 20, Bulk 1.15x)", {"max_depth": 5, "num_leaves": 20, "n_estimators": 100, "learning_rate": 0.08, "random_state": 42, "n_jobs": 2, "verbose": -1}, X8, FEATS_CHAMPION8, lambda y_tr, num_classes=3: bulk_boost_weights(y_tr, num_classes, boost=1.15), "lgbm_tuned"),
        ("LightGBM Causal 9-Feature (Depth 5, Leaves 20)", {"max_depth": 5, "num_leaves": 20, "n_estimators": 100, "learning_rate": 0.08, "random_state": 42, "n_jobs": 2, "verbose": -1}, X9, FEATS_CAUSAL9, lambda y_tr, num_classes=3: bulk_boost_weights(y_tr, num_classes, boost=1.15), "lgbm_causal9"),
    ]
    for name, p, feat_mat, feat_names, w_fn, prefix in lgb_cfgs:
        print(f"Evaluating {name}...")
        res = evaluate_adv_model_logo(
            feat_mat, y, groups,
            model_fn=lambda p_dict=p: LGBMClassifier(**p_dict),
            weight_fn=w_fn,
            save_prefix=prefix
        )
        res.update({
            "name": name,
            "family": "LightGBM",
            "features": "Champion 8" if len(feat_names) == 8 else "Causal 9",
            "feature_count": len(feat_names),
            "params": p
        })
        results.append(res)
        print(f"  -> Macro F1: {res['macro_f1']:.4f} | Bal Acc: {res['balanced_acc']:.4f} | Acc: {res['overall_acc']:.4f} | Time: {res['runtime_sec']}s")

    out_file = OUTPUT_DIR / "agent2_boosting_results.json"
    with open(out_file, "w") as fp:
        json.dump(results, fp, indent=2)
    print(f"\n[DONE] Agent 2 completed {len(results)} experiments -> {out_file}")

if __name__ == "__main__":
    main()
