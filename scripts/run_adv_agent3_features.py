#!/usr/bin/env python3
"""
scripts/run_adv_agent3_features.py
AGENT 3: Causal Early-5 Feature Engineering, Feature Selection, and Ablation.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.preprocessing import StandardScaler
from adv_campaign_engine import (
    load_advanced_dataset,
    evaluate_adv_model_logo,
    FEATS_BASE6,
    FEATS_CHAMPION8,
    FEATS_CAUSAL9,
    FEATS_EXTENDED14,
    FEATS_ALL_CAUSAL18,
    bulk_boost_weights,
    OUTPUT_DIR
)

def main():
    print("=" * 70)
    print("AGENT 3: FEATURE ENGINEERING & SELECTION AUDIT")
    print("=" * 70)
    
    df, y, groups, classes, cls2idx, idx2cls = load_advanced_dataset()
    
    # 1. Feature Importance Audit on ALL Causal 18 Features
    X18 = df[FEATS_ALL_CAUSAL18].values.astype(float)
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    scaler = StandardScaler()
    rf.fit(scaler.fit_transform(X18), y)
    
    importances = rf.feature_importances_
    mi_scores = mutual_info_classif(scaler.fit_transform(X18), y, random_state=42)
    
    fi_df = pd.DataFrame({
        "Feature": FEATS_ALL_CAUSAL18,
        "RF_Gini_Importance": np.round(importances, 4),
        "Mutual_Information": np.round(mi_scores, 4)
    }).sort_values("RF_Gini_Importance", ascending=False)
    
    fi_path = OUTPUT_DIR / "feature_importance" / "feature_ranks.csv"
    fi_df.to_csv(fi_path, index=False)
    print(f"Saved feature importance rankings to: {fi_path}")
    print(fi_df.to_string(index=False))
    
    # 2. Comprehensive Ablation across Feature Sets (Tested on both RBF SVM and XGBoost)
    feature_sets = [
        ("Set A: Base 6 Features", FEATS_BASE6),
        ("Set B: Champion 8 Features", FEATS_CHAMPION8),
        ("Set C: Causal 9 (+is_bidir)", FEATS_CAUSAL9),
        ("Set D: Extended 14 Features", FEATS_EXTENDED14),
        ("Set E: All Causal 18 Features", FEATS_ALL_CAUSAL18),
    ]
    
    results = []
    ablation_rows = []
    
    for f_name, f_list in feature_sets:
        X_sub = df[f_list].values.astype(float)
        
        # Test with RBF SVM (C=10.0, balanced)
        svm_name = f"RBF SVM on {f_name}"
        print(f"\nEvaluating {svm_name} ({len(f_list)} features)...")
        res_svm = evaluate_adv_model_logo(
            X_sub, y, groups,
            model_fn=lambda: SVC(C=10.0, kernel="rbf", gamma="scale", class_weight="balanced", probability=True),
            scaler_cls=StandardScaler,
            save_prefix=f"svm_{f_name[:5].replace(' ', '_').lower()}"
        )
        res_svm.update({
            "name": svm_name,
            "family": "Feature Ablation (SVM)",
            "features": f_name,
            "feature_count": len(f_list),
            "params": {"C": 10.0, "kernel": "rbf", "features": f_list}
        })
        results.append(res_svm)
        print(f"  -> Macro F1: {res_svm['macro_f1']:.4f} | Bal Acc: {res_svm['balanced_acc']:.4f} | Acc: {res_svm['overall_acc']:.4f}")
        
        # Test with XGBoost (Bulk 1.15x)
        xgb_name = f"XGBoost on {f_name}"
        print(f"Evaluating {xgb_name} ({len(f_list)} features)...")
        res_xgb = evaluate_adv_model_logo(
            X_sub, y, groups,
            model_fn=lambda: XGBClassifier(max_depth=5, learning_rate=0.08, n_estimators=100, min_child_weight=2, reg_alpha=0.1, reg_lambda=1.5, random_state=42, n_jobs=2, eval_metric="mlogloss"),
            weight_fn=lambda y_tr, num_classes=3: bulk_boost_weights(y_tr, num_classes, boost=1.15),
            save_prefix=f"xgb_{f_name[:5].replace(' ', '_').lower()}"
        )
        res_xgb.update({
            "name": xgb_name,
            "family": "Feature Ablation (XGBoost)",
            "features": f_name,
            "feature_count": len(f_list),
            "params": {"model": "XGBoost CFG-D", "features": f_list}
        })
        results.append(res_xgb)
        print(f"  -> Macro F1: {res_xgb['macro_f1']:.4f} | Bal Acc: {res_xgb['balanced_acc']:.4f} | Acc: {res_xgb['overall_acc']:.4f}")
        
        ablation_rows.append({
            "Feature_Set": f_name,
            "Count": len(f_list),
            "SVM_Macro_F1": res_svm["macro_f1"],
            "SVM_Bal_Acc": res_svm["balanced_acc"],
            "SVM_Overall_Acc": res_svm["overall_acc"],
            "XGB_Macro_F1": res_xgb["macro_f1"],
            "XGB_Bal_Acc": res_xgb["balanced_acc"],
            "XGB_Overall_Acc": res_xgb["overall_acc"],
        })
        
    # Save ablation table
    ab_df = pd.DataFrame(ablation_rows)
    ab_path = OUTPUT_DIR / "feature_ablation_results.csv"
    ab_df.to_csv(ab_path, index=False)
    print(f"\nSaved feature ablation comparison to: {ab_path}")
    
    # Save Agent 3 results
    out_file = OUTPUT_DIR / "agent3_features_results.json"
    with open(out_file, "w") as fp:
        json.dump(results, fp, indent=2)
    print(f"[DONE] Agent 3 completed {len(results)} experiments -> {out_file}")

if __name__ == "__main__":
    main()
