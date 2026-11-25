#!/usr/bin/env python3
"""
scripts/run_adv_agent1_svm.py
AGENT 1: Advanced Support Vector Machine Optimization & Nested Model Selection.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from adv_campaign_engine import (
    load_advanced_dataset,
    evaluate_adv_model_logo,
    inner_group_cv_score,
    standard_weights,
    bulk_boost_weights,
    FEATS_CHAMPION8,
    FEATS_CAUSAL9,
    OUTPUT_DIR
)

def main():
    print("=" * 70)
    print("AGENT 1: ADVANCED SVM OPTIMIZATION & NESTED SELECTION")
    print("=" * 70)
    
    df, y, groups, classes, cls2idx, idx2cls = load_advanced_dataset()
    X8 = df[FEATS_CHAMPION8].values.astype(float)
    X9 = df[FEATS_CAUSAL9].values.astype(float)
    
    results = []
    
    # 1. C-parameter search on RBF SVM (StandardScaler, balanced)
    c_candidates = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    for c in c_candidates:
        name = f"RBF SVM (C={c}, gamma=scale, balanced, StandardScaler)"
        print(f"Evaluating {name}...")
        res = evaluate_adv_model_logo(
            X8, y, groups,
            model_fn=lambda c_val=c: SVC(C=c_val, kernel="rbf", gamma="scale", class_weight="balanced", probability=True),
            scaler_cls=StandardScaler,
            save_prefix=f"svm_rbf_c_{str(c).replace('.', '_')}"
        )
        res.update({
            "name": name,
            "family": "SVM",
            "features": "Champion 8-Feature",
            "feature_count": 8,
            "params": {"C": c, "kernel": "rbf", "gamma": "scale", "class_weight": "balanced", "scaler": "StandardScaler"}
        })
        results.append(res)
        print(f"  -> Macro F1: {res['macro_f1']:.4f} | Bal Acc: {res['balanced_acc']:.4f} | Acc: {res['overall_acc']:.4f} | Time: {res['runtime_sec']}s")

    # 2. Gamma search at C=10.0
    gamma_candidates = [0.05, 0.1, 0.2, "auto"]
    for g in gamma_candidates:
        name = f"RBF SVM (C=10.0, gamma={g}, balanced, StandardScaler)"
        print(f"Evaluating {name}...")
        res = evaluate_adv_model_logo(
            X8, y, groups,
            model_fn=lambda g_val=g: SVC(C=10.0, kernel="rbf", gamma=g_val, class_weight="balanced", probability=True),
            scaler_cls=StandardScaler,
            save_prefix=f"svm_rbf_gamma_{str(g)}"
        )
        res.update({
            "name": name,
            "family": "SVM",
            "features": "Champion 8-Feature",
            "feature_count": 8,
            "params": {"C": 10.0, "kernel": "rbf", "gamma": g, "class_weight": "balanced", "scaler": "StandardScaler"}
        })
        results.append(res)
        print(f"  -> Macro F1: {res['macro_f1']:.4f} | Bal Acc: {res['balanced_acc']:.4f} | Acc: {res['overall_acc']:.4f} | Time: {res['runtime_sec']}s")

    # 3. Kernel Exploration: Linear & Polynomial
    kernel_configs = [
        ("Linear SVM (C=1.0, balanced)", lambda: SVC(C=1.0, kernel="linear", class_weight="balanced", probability=True), "linear", 1.0, StandardScaler),
        ("Linear SVM (C=10.0, balanced)", lambda: SVC(C=10.0, kernel="linear", class_weight="balanced", probability=True), "linear", 10.0, StandardScaler),
        ("Poly SVM (degree=2, C=5.0, balanced)", lambda: SVC(C=5.0, kernel="poly", degree=2, class_weight="balanced", probability=True), "poly", 5.0, StandardScaler),
        ("Poly SVM (degree=3, C=5.0, balanced)", lambda: SVC(C=5.0, kernel="poly", degree=3, class_weight="balanced", probability=True), "poly", 5.0, StandardScaler),
    ]
    for name, fn, kern, c_val, sc in kernel_configs:
        print(f"Evaluating {name}...")
        res = evaluate_adv_model_logo(
            X8, y, groups,
            model_fn=fn,
            scaler_cls=sc,
            save_prefix=f"svm_{kern}_c_{str(c_val).replace('.', '_')}"
        )
        res.update({
            "name": name,
            "family": "SVM",
            "features": "Champion 8-Feature",
            "feature_count": 8,
            "params": {"C": c_val, "kernel": kern, "class_weight": "balanced", "scaler": "StandardScaler"}
        })
        results.append(res)
        print(f"  -> Macro F1: {res['macro_f1']:.4f} | Bal Acc: {res['balanced_acc']:.4f} | Acc: {res['overall_acc']:.4f} | Time: {res['runtime_sec']}s")

    # 4. Alternative Scaling: RobustScaler & MinMaxScaler (at C=10.0 RBF)
    scaler_configs = [
        ("RBF SVM (C=10.0, RobustScaler)", RobustScaler, "RobustScaler"),
        ("RBF SVM (C=10.0, MinMaxScaler)", MinMaxScaler, "MinMaxScaler"),
    ]
    for name, sc_cls, sc_name in scaler_configs:
        print(f"Evaluating {name}...")
        res = evaluate_adv_model_logo(
            X8, y, groups,
            model_fn=lambda: SVC(C=10.0, kernel="rbf", gamma="scale", class_weight="balanced", probability=True),
            scaler_cls=sc_cls,
            save_prefix=f"svm_rbf_{sc_name.lower()}"
        )
        res.update({
            "name": name,
            "family": "SVM",
            "features": "Champion 8-Feature",
            "feature_count": 8,
            "params": {"C": 10.0, "kernel": "rbf", "gamma": "scale", "class_weight": "balanced", "scaler": sc_name}
        })
        results.append(res)
        print(f"  -> Macro F1: {res['macro_f1']:.4f} | Bal Acc: {res['balanced_acc']:.4f} | Acc: {res['overall_acc']:.4f} | Time: {res['runtime_sec']}s")

    # 5. RBF SVM with Causal 9 Features (+ early_5_is_bidir)
    print("Evaluating RBF SVM (C=10.0, Causal 9-Feature)...")
    res9 = evaluate_adv_model_logo(
        X9, y, groups,
        model_fn=lambda: SVC(C=10.0, kernel="rbf", gamma="scale", class_weight="balanced", probability=True),
        scaler_cls=StandardScaler,
        save_prefix="svm_rbf_c10_causal9"
    )
    res9.update({
        "name": "RBF SVM (C=10.0, Causal 9-Feature, StandardScaler)",
        "family": "SVM",
        "features": "Causal 9-Feature (+is_bidir)",
        "feature_count": 9,
        "params": {"C": 10.0, "kernel": "rbf", "gamma": "scale", "class_weight": "balanced", "scaler": "StandardScaler"}
    })
    results.append(res9)
    print(f"  -> Macro F1: {res9['macro_f1']:.4f} | Bal Acc: {res9['balanced_acc']:.4f} | Acc: {res9['overall_acc']:.4f} | Time: {res9['runtime_sec']}s")

    # Save all Agent 1 outputs
    out_file = OUTPUT_DIR / "agent1_svm_results.json"
    with open(out_file, "w") as fp:
        json.dump(results, fp, indent=2)
    print(f"\n[DONE] Agent 1 completed {len(results)} experiments -> {out_file}")

if __name__ == "__main__":
    main()
