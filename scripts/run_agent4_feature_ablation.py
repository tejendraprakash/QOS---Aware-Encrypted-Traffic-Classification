#!/usr/bin/env python3
"""
scripts/run_agent4_feature_ablation.py
AGENT 4: Feature Engineering Audit & Systematic Feature Ablation under strict 29-fold LOGO.
Evaluates:
- Baseline 6-feature vector
- Champion 8-feature vector (+ is_udp, early_5_fwd_ratio)
- Candidate 10-feature vector (+ early_5_is_bidir, early_5_bwd_fwd_diff)
- Candidate 12-feature vector (+ log_bytes, log_duration)
- Candidate 14-feature vector (+ bwd_ratio, pkt_count)
- Single-feature additions to test exact marginal contributions.
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
    bulk_boost_weights,
    FEATS_BASE,
    FEATS_ROBUST8,
    FEATS_EXTENDED12,
    FEATS_ALL_CAUSAL14,
    OUTPUT_DIR
)

def default_xgb_fn():
    return XGBClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.10,
        subsample=0.8, colsample_bytree=0.8,
        objective="multi:softprob", eval_metric="mlogloss",
        random_state=42, n_jobs=1
    )

def main():
    print("=" * 70)
    print("AGENT 4: FEATURE ENGINEERING AUDIT & ABLATION STUDY")
    print("=" * 70)
    
    df, y, groups, classes, cls2idx, idx2cls = load_clean_dataset()
    weight_fn = lambda y_tr, nc: bulk_boost_weights(y_tr, nc, boost=1.20)
    
    feature_experiments = [
        {
            "name": "Set A: Baseline 6 Features",
            "features": FEATS_BASE,
            "description": "Baseline causal features (duration, bytes, fwd/bwd pkts, avg size, complete flag)"
        },
        {
            "name": "Set B: Champion 8 Features",
            "features": FEATS_ROBUST8,
            "description": "Set A + is_udp (L4 header) + early_5_fwd_ratio (directional symmetry)"
        },
        {
            "name": "Set B + early_5_is_bidir (9 Feats)",
            "features": FEATS_ROBUST8 + ["early_5_is_bidir"],
            "description": "Set B + binary indicator of observed bidirectional traffic in early 5"
        },
        {
            "name": "Set B + early_5_bwd_fwd_diff (9 Feats)",
            "features": FEATS_ROBUST8 + ["early_5_bwd_fwd_diff"],
            "description": "Set B + forward minus backward packet count difference"
        },
        {
            "name": "Set B + Log Transforms (10 Feats)",
            "features": FEATS_ROBUST8 + ["early_5_log_bytes", "early_5_log_duration"],
            "description": "Set B + log1p(bytes) + log1p(duration) to reduce skew"
        },
        {
            "name": "Set C: Extended Causal 12 Features",
            "features": FEATS_EXTENDED12,
            "description": "Set B + is_bidir + bwd_fwd_diff + log_bytes + log_duration"
        },
        {
            "name": "Set D: All Causal 14 Features",
            "features": FEATS_ALL_CAUSAL14,
            "description": "Set C + bwd_ratio + pkt_count"
        },
    ]
    
    ablation_records = []
    agent4_json_results = []
    
    for exp in feature_experiments:
        feat_cols = exp["features"]
        X = df[feat_cols].astype(float).values
        print(f"Evaluating {exp['name']} ({len(feat_cols)} features)...", flush=True)
        
        res = evaluate_model_logo(
            X, y, groups, default_xgb_fn, weight_fn=weight_fn,
            scale_features=False, num_classes=len(classes)
        )
        print(f"  -> Macro F1: {res['macro_f1']:.4f} | Bal Acc: {res['balanced_acc']:.4f} | Acc: {res['overall_acc']:.4f} | Time: {res['runtime_sec']}s", flush=True)
        
        oof_path = OUTPUT_DIR / f"oof_agent4_{exp['name'].replace(' ', '_').replace('+', '_').replace('(', '').replace(')', '').replace(':', '')}.npy"
        np.save(oof_path, res["oof_preds"])
        
        rec = {
            "Feature_Set": exp["name"],
            "Feature_Count": len(feat_cols),
            "Features_List": ", ".join(feat_cols),
            "Description": exp["description"],
            "Macro_F1": res["macro_f1"],
            "Delta_vs_Baseline": round(res["macro_f1"] - 0.5734, 4),
            "Delta_vs_Champion": round(res["macro_f1"] - 0.5895, 4),
            "Balanced_Accuracy": res["balanced_acc"],
            "Overall_Accuracy": res["overall_acc"],
            "Bulk_Data_F1": res["per_class_f1"][0],
            "Real_Time_F1": res["per_class_f1"][1],
            "Streaming_F1": res["per_class_f1"][2],
            "Bulk_Data_Recall": res["per_class_rec"][0],
            "Real_Time_Recall": res["per_class_rec"][1],
            "Streaming_Recall": res["per_class_rec"][2],
            "Runtime_Sec": res["runtime_sec"]
        }
        ablation_records.append(rec)
        
        agent4_json_results.append({
            **rec,
            "confusion_matrix": res["confusion_matrix"],
            "per_group_f1s": res["per_group_f1s"],
            "oof_file": str(oof_path.name)
        })
        
    df_ablation = pd.DataFrame(ablation_records)
    csv_path = OUTPUT_DIR / "feature_ablation_results.csv"
    df_ablation.to_csv(csv_path, index=False)
    print(f"\nSaved feature ablation CSV to: {csv_path}")
    
    json_path = OUTPUT_DIR / "agent4_results.json"
    with open(json_path, "w") as f:
        json.dump(agent4_json_results, f, indent=2)
    print(f"Saved feature ablation JSON to: {json_path}")

if __name__ == "__main__":
    main()
