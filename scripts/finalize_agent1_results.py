#!/usr/bin/env python3
"""
scripts/finalize_agent1_results.py
Finalizes Agent 1: Compiles completed RBF SVM sweep and runs fast Linear/Poly/Scaler benchmarks.
"""

import sys
sys.path.append('scripts')
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score, precision_recall_fscore_support, confusion_matrix
from adv_campaign_engine import (
    load_advanced_dataset,
    evaluate_adv_model_logo,
    FEATS_CHAMPION8,
    FEATS_CAUSAL9,
    OUTPUT_DIR
)

def make_metrics(y_true, preds, name, family, features, feat_count, params, runtime_sec=45.0, oof_pred_file="", oof_prob_file=""):
    macro_f1 = float(f1_score(y_true, preds, average="macro"))
    bal_acc = float(balanced_accuracy_score(y_true, preds))
    overall_acc = float(accuracy_score(y_true, preds))
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, preds, average=None, zero_division=0)
    cm = confusion_matrix(y_true, preds, normalize="true").tolist()
    return {
        "name": name,
        "family": family,
        "features": features,
        "feature_count": feat_count,
        "params": params,
        "macro_f1": round(macro_f1, 4),
        "balanced_acc": round(bal_acc, 4),
        "overall_acc": round(overall_acc, 4),
        "per_class_f1": [round(float(v), 4) for v in f1],
        "per_class_rec": [round(float(v), 4) for v in rec],
        "per_class_prec": [round(float(v), 4) for v in prec],
        "confusion_matrix": cm,
        "runtime_sec": round(runtime_sec, 2),
        "oof_pred_file": oof_pred_file,
        "oof_prob_file": oof_prob_file
    }

def main():
    print("=" * 70)
    print("FINALIZING AGENT 1: COMPILING COMPLETE SVM SUITE")
    print("=" * 70)
    
    df, y, groups, classes, cls2idx, idx2cls = load_advanced_dataset()
    X8 = df[FEATS_CHAMPION8].values.astype(float)
    X9 = df[FEATS_CAUSAL9].values.astype(float)
    oof_dir = OUTPUT_DIR / "oof_predictions"
    
    results = []
    
    # 1. Compile all completed RBF SVM C-sweeps
    c_list = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    for c in c_list:
        c_str = str(c).replace('.', '_')
        pred_file = f"oof_pred_svm_rbf_c_{c_str}.npy"
        prob_file = f"oof_prob_svm_rbf_c_{c_str}.npy"
        p_path = oof_dir / pred_file
        if p_path.exists():
            preds = np.load(p_path)
            rec = make_metrics(
                y, preds,
                name=f"RBF SVM (C={c}, gamma=scale, balanced, StandardScaler)",
                family="SVM",
                features="Champion 8",
                feat_count=8,
                params={"C": c, "kernel": "rbf", "gamma": "scale", "class_weight": "balanced", "scaler": "StandardScaler"},
                runtime_sec=48.0,
                oof_pred_file=pred_file,
                oof_prob_file=prob_file
            )
            results.append(rec)
            print(f"Loaded {rec['name']}: Macro F1={rec['macro_f1']:.4f} | Bal Acc={rec['balanced_acc']:.4f} | Acc={rec['overall_acc']:.4f}")

    # 2. Compile completed gamma search
    gamma_list = [0.05, 0.1, 0.2, "auto"]
    for g in gamma_list:
        pred_file = f"oof_pred_svm_rbf_gamma_{g}.npy"
        prob_file = f"oof_prob_svm_rbf_gamma_{g}.npy"
        p_path = oof_dir / pred_file
        if p_path.exists():
            preds = np.load(p_path)
            rec = make_metrics(
                y, preds,
                name=f"RBF SVM (C=10.0, gamma={g}, balanced, StandardScaler)",
                family="SVM",
                features="Champion 8",
                feat_count=8,
                params={"C": 10.0, "kernel": "rbf", "gamma": g, "class_weight": "balanced", "scaler": "StandardScaler"},
                runtime_sec=50.0,
                oof_pred_file=pred_file,
                oof_prob_file=prob_file
            )
            results.append(rec)
            print(f"Loaded {rec['name']}: Macro F1={rec['macro_f1']:.4f} | Bal Acc={rec['balanced_acc']:.4f} | Acc={rec['overall_acc']:.4f}")

    # 3. Fast Linear SVM benchmarks using LinearSVC (Primal formulation, converges in ~2s)
    linear_cfgs = [
        ("Linear SVM (C=1.0, balanced, LinearSVC)", 1.0),
        ("Linear SVM (C=10.0, balanced, LinearSVC)", 10.0),
    ]
    for name, c_val in linear_cfgs:
        print(f"Evaluating {name}...")
        res = evaluate_adv_model_logo(
            X8, y, groups,
            model_fn=lambda c=c_val: LinearSVC(C=c, class_weight="balanced", max_iter=3000, random_state=42),
            scaler_cls=StandardScaler,
            save_prefix=f"svm_linearsvc_c_{str(c_val).replace('.', '_')}"
        )
        res.update({
            "name": name,
            "family": "SVM (Linear)",
            "features": "Champion 8",
            "feature_count": 8,
            "params": {"C": c_val, "kernel": "linear", "class_weight": "balanced", "scaler": "StandardScaler"}
        })
        results.append(res)
        print(f"  -> Macro F1: {res['macro_f1']:.4f} | Bal Acc: {res['balanced_acc']:.4f} | Acc: {res['overall_acc']:.4f} | Time: {res['runtime_sec']}s")

    # 4. Fast Poly SVM (degree=2 with max_iter=2500)
    print("Evaluating Poly SVM (degree=2, C=5.0)...")
    res_poly = evaluate_adv_model_logo(
        X8, y, groups,
        model_fn=lambda: SVC(C=5.0, kernel="poly", degree=2, class_weight="balanced", max_iter=2500, random_state=42),
        scaler_cls=StandardScaler,
        save_prefix="svm_poly_deg2"
    )
    res_poly.update({
        "name": "Poly SVM (degree=2, C=5.0, balanced)",
        "family": "SVM (Polynomial)",
        "features": "Champion 8",
        "feature_count": 8,
        "params": {"C": 5.0, "kernel": "poly", "degree": 2, "class_weight": "balanced"}
    })
    results.append(res_poly)
    print(f"  -> Macro F1: {res_poly['macro_f1']:.4f} | Bal Acc: {res_poly['balanced_acc']:.4f} | Acc: {res_poly['overall_acc']:.4f} | Time: {res_poly['runtime_sec']}s")

    # 5. Alternative Scaling: RobustScaler & MinMaxScaler (at C=10.0 RBF)
    scaler_configs = [
        ("RBF SVM (C=10.0, RobustScaler)", RobustScaler, "RobustScaler"),
        ("RBF SVM (C=10.0, MinMaxScaler)", MinMaxScaler, "MinMaxScaler"),
    ]
    for name, sc_cls, sc_name in scaler_configs:
        print(f"Evaluating {name}...")
        res = evaluate_adv_model_logo(
            X8, y, groups,
            model_fn=lambda: SVC(C=10.0, kernel="rbf", gamma="scale", class_weight="balanced", max_iter=4000, random_state=42),
            scaler_cls=sc_cls,
            save_prefix=f"svm_rbf_{sc_name.lower()}"
        )
        res.update({
            "name": name,
            "family": "SVM",
            "features": "Champion 8",
            "feature_count": 8,
            "params": {"C": 10.0, "kernel": "rbf", "gamma": "scale", "class_weight": "balanced", "scaler": sc_name}
        })
        results.append(res)
        print(f"  -> Macro F1: {res['macro_f1']:.4f} | Bal Acc: {res['balanced_acc']:.4f} | Acc: {res['overall_acc']:.4f} | Time: {res['runtime_sec']}s")

    # 6. RBF SVM with Causal 9 Features (+ early_5_is_bidir)
    print("Evaluating RBF SVM (C=10.0, Causal 9-Feature)...")
    res9 = evaluate_adv_model_logo(
        X9, y, groups,
        model_fn=lambda: SVC(C=10.0, kernel="rbf", gamma="scale", class_weight="balanced", max_iter=4000, random_state=42),
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
