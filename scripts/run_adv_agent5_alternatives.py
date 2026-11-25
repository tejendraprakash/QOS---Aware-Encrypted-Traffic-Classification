#!/usr/bin/env python3
"""
scripts/run_adv_agent5_alternatives.py
AGENT 5: Alternative Model Approaches (MLP Neural Net, Regularized Logistic Regression, GaussianNB, k-NN).
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from adv_campaign_engine import (
    load_advanced_dataset,
    evaluate_adv_model_logo,
    FEATS_CHAMPION8,
    FEATS_CAUSAL9,
    OUTPUT_DIR
)

def main():
    print("=" * 70)
    print("AGENT 5: ALTERNATIVE MODEL FAMILIES BENCHMARK")
    print("=" * 70)
    
    df, y, groups, classes, cls2idx, idx2cls = load_advanced_dataset()
    X8 = df[FEATS_CHAMPION8].values.astype(float)
    X9 = df[FEATS_CAUSAL9].values.astype(float)
    
    results = []
    
    # 1. Multi-Layer Perceptron (MLP) Neural Network
    mlp_configs = [
        ("MLP (64-32, alpha=0.01, early_stopping)", {"hidden_layer_sizes": (64, 32), "alpha": 0.01, "max_iter": 300, "early_stopping": True, "random_state": 42}, X8, FEATS_CHAMPION8, "mlp_64_32"),
        ("MLP (32-16, alpha=0.1, early_stopping)", {"hidden_layer_sizes": (32, 16), "alpha": 0.1, "max_iter": 300, "early_stopping": True, "random_state": 42}, X8, FEATS_CHAMPION8, "mlp_32_16"),
        ("MLP Causal 9-Feature (64-32)", {"hidden_layer_sizes": (64, 32), "alpha": 0.05, "max_iter": 300, "early_stopping": True, "random_state": 42}, X9, FEATS_CAUSAL9, "mlp_causal9"),
    ]
    for name, p, feat_mat, feat_names, prefix in mlp_configs:
        print(f"Evaluating {name}...")
        res = evaluate_adv_model_logo(
            feat_mat, y, groups,
            model_fn=lambda p_dict=p: MLPClassifier(**p_dict),
            scaler_cls=StandardScaler,
            save_prefix=prefix
        )
        res.update({
            "name": name,
            "family": "Neural Network (MLP)",
            "features": "Champion 8" if len(feat_names) == 8 else "Causal 9",
            "feature_count": len(feat_names),
            "params": p
        })
        results.append(res)
        print(f"  -> Macro F1: {res['macro_f1']:.4f} | Bal Acc: {res['balanced_acc']:.4f} | Acc: {res['overall_acc']:.4f} | Time: {res['runtime_sec']}s")

    # 2. Regularized Logistic Regression
    lr_configs = [
        ("Logistic Regression (C=1.0, L2, balanced)", {"C": 1.0, "penalty": "l2", "class_weight": "balanced", "max_iter": 500, "random_state": 42}, X8, FEATS_CHAMPION8, "lr_c1"),
        ("Logistic Regression (C=0.1, L2, balanced)", {"C": 0.1, "penalty": "l2", "class_weight": "balanced", "max_iter": 500, "random_state": 42}, X8, FEATS_CHAMPION8, "lr_c01"),
        ("Logistic Regression (C=10.0, L2, balanced)", {"C": 10.0, "penalty": "l2", "class_weight": "balanced", "max_iter": 500, "random_state": 42}, X8, FEATS_CHAMPION8, "lr_c10"),
    ]
    for name, p, feat_mat, feat_names, prefix in lr_configs:
        print(f"Evaluating {name}...")
        res = evaluate_adv_model_logo(
            feat_mat, y, groups,
            model_fn=lambda p_dict=p: LogisticRegression(**p_dict),
            scaler_cls=StandardScaler,
            save_prefix=prefix
        )
        res.update({
            "name": name,
            "family": "Logistic Regression",
            "features": "Champion 8" if len(feat_names) == 8 else "Causal 9",
            "feature_count": len(feat_names),
            "params": p
        })
        results.append(res)
        print(f"  -> Macro F1: {res['macro_f1']:.4f} | Bal Acc: {res['balanced_acc']:.4f} | Acc: {res['overall_acc']:.4f} | Time: {res['runtime_sec']}s")

    # 3. Gaussian Naive Bayes
    print("Evaluating Gaussian Naive Bayes...")
    res_gnb = evaluate_adv_model_logo(
        X8, y, groups,
        model_fn=lambda: GaussianNB(),
        scaler_cls=StandardScaler,
        save_prefix="gaussian_nb"
    )
    res_gnb.update({
        "name": "Gaussian Naive Bayes (StandardScaler)",
        "family": "Naive Bayes",
        "features": "Champion 8",
        "feature_count": 8,
        "params": {}
    })
    results.append(res_gnb)
    print(f"  -> Macro F1: {res_gnb['macro_f1']:.4f} | Bal Acc: {res_gnb['balanced_acc']:.4f} | Acc: {res_gnb['overall_acc']:.4f} | Time: {res_gnb['runtime_sec']}s")

    # 4. k-Nearest Neighbors
    knn_configs = [
        ("k-NN (k=11, distance weights)", {"n_neighbors": 11, "weights": "distance"}, "knn_k11"),
        ("k-NN (k=21, distance weights)", {"n_neighbors": 21, "weights": "distance"}, "knn_k21"),
    ]
    for name, p, prefix in knn_configs:
        print(f"Evaluating {name}...")
        res = evaluate_adv_model_logo(
            X8, y, groups,
            model_fn=lambda p_dict=p: KNeighborsClassifier(**p_dict),
            scaler_cls=StandardScaler,
            save_prefix=prefix
        )
        res.update({
            "name": name,
            "family": "k-Nearest Neighbors",
            "features": "Champion 8",
            "feature_count": 8,
            "params": p
        })
        results.append(res)
        print(f"  -> Macro F1: {res['macro_f1']:.4f} | Bal Acc: {res['balanced_acc']:.4f} | Acc: {res['overall_acc']:.4f} | Time: {res['runtime_sec']}s")

    out_file = OUTPUT_DIR / "agent5_alternatives_results.json"
    with open(out_file, "w") as fp:
        json.dump(results, fp, indent=2)
    print(f"\n[DONE] Agent 5 completed {len(results)} experiments -> {out_file}")

if __name__ == "__main__":
    main()
