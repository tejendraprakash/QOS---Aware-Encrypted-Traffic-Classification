#!/usr/bin/env python3
"""
Comprehensive Independent Verification Script for Phase 6 Out-Of-Fold Predictions
================================================================================
This script independently recalculates all metrics directly from the raw out-of-fold
predictions files:
  - output/phase6/track_a/out_of_fold_predictions.csv
  - output/phase6/track_b/out_of_fold_predictions.csv

It computes:
  1. Fold-level metrics (Accuracy, Balanced Accuracy, Macro F1, Weighted F1, Per-class PRF1)
  2. Unweighted fold means and standard deviations
  3. Pooled Out-Of-Fold (OOF) metrics
  4. Raw and row-normalized confusion matrices
  5. Per-class pooled precision, recall, and F1
  6. Exact discrepancy analysis against reported files:
     - aggregate_metrics.csv
     - fold_metrics.csv
     - confusion_matrix_counts.csv
     - confusion_matrix_normalized.csv
     - per_class_metrics.csv
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "phase6")

TRACKS = {
    "track_a": {
        "name": "Track A (4-Class QoS)",
        "classes": ["Browsing", "Bulk Data", "Real-Time", "Streaming"],
        "models": ["Random_Forest", "XGBoost", "P_MLP"],
        "num_folds": 3
    },
    "track_b": {
        "name": "Track B (3-Class VPN)",
        "classes": ["Bulk Data", "Real-Time", "Streaming"],
        "models": ["Random_Forest", "XGBoost", "P_MLP"],
        "num_folds": 5
    }
}

def verify_track(track_key):
    cfg = TRACKS[track_key]
    track_dir = os.path.join(OUTPUT_DIR, track_key)
    oof_path = os.path.join(track_dir, "out_of_fold_predictions.csv")
    agg_path = os.path.join(track_dir, "aggregate_metrics.csv")
    fold_path = os.path.join(track_dir, "fold_metrics.csv")
    cm_cnt_path = os.path.join(track_dir, "confusion_matrix_counts.csv")
    cm_norm_path = os.path.join(track_dir, "confusion_matrix_normalized.csv")
    per_cls_path = os.path.join(track_dir, "per_class_metrics.csv")
    
    print(f"\n{'='*80}\nVerifying {cfg['name']} ({track_key})\n{'='*80}")
    
    df_oof = pd.read_csv(oof_path)
    df_agg_rep = pd.read_csv(agg_path)
    df_fold_rep = pd.read_csv(fold_path)
    df_cm_cnt_rep = pd.read_csv(cm_cnt_path)
    df_cm_norm_rep = pd.read_csv(cm_norm_path)
    df_per_cls_rep = pd.read_csv(per_cls_path)
    
    classes = cfg["classes"]
    models = cfg["models"]
    num_folds = cfg["num_folds"]
    
    print(f"Loaded OOF predictions: {len(df_oof)} rows, classes: {classes}, models: {models}")
    
    results = {
        "track": track_key,
        "name": cfg["name"],
        "num_samples": len(df_oof),
        "fold_sizes": df_oof["fold"].value_counts().to_dict(),
        "class_distribution": df_oof["true_label"].value_counts().to_dict(),
        "models": {}
    }
    
    for model in models:
        pred_col = f"pred_{model}"
        print(f"\n--- Model: {model} ---")
        
        # 1. Fold-level recomputations
        fold_metrics_recalc = []
        per_class_fold_recalc = []
        
        for fold in range(num_folds):
            mask = df_oof["fold"] == fold
            sub = df_oof[mask]
            y_true = sub["true_label"]
            y_pred = sub[pred_col]
            
            acc = accuracy_score(y_true, y_pred)
            bal_acc = balanced_accuracy_score(y_true, y_pred)
            mac_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
            wt_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
            
            p_per, r_per, f_per, s_per = precision_recall_fscore_support(
                y_true, y_pred, labels=classes, zero_division=0
            )
            
            fold_metrics_recalc.append({
                "fold": fold,
                "n_samples": len(sub),
                "accuracy": acc,
                "balanced_accuracy": bal_acc,
                "macro_f1": mac_f1,
                "weighted_f1": wt_f1
            })
            
            for c_idx, cls_name in enumerate(classes):
                per_class_fold_recalc.append({
                    "fold": fold,
                    "class_name": cls_name,
                    "precision": p_per[c_idx],
                    "recall": r_per[c_idx],
                    "f1_score": f_per[c_idx],
                    "support": s_per[c_idx]
                })
        
        df_f_recalc = pd.DataFrame(fold_metrics_recalc)
        
        # Fold statistics (mean & sample std, ddof=1)
        mean_acc = df_f_recalc["accuracy"].mean()
        std_acc = df_f_recalc["accuracy"].std()
        mean_bal_acc = df_f_recalc["balanced_accuracy"].mean()
        std_bal_acc = df_f_recalc["balanced_accuracy"].std()
        mean_mac_f1 = df_f_recalc["macro_f1"].mean()
        std_mac_f1 = df_f_recalc["macro_f1"].std()
        mean_wt_f1 = df_f_recalc["weighted_f1"].mean()
        std_wt_f1 = df_f_recalc["weighted_f1"].std()
        
        # 2. Pooled OOF metrics
        y_true_all = df_oof["true_label"]
        y_pred_all = df_oof[pred_col]
        
        pooled_acc = accuracy_score(y_true_all, y_pred_all)
        pooled_bal_acc = balanced_accuracy_score(y_true_all, y_pred_all)
        pooled_mac_f1 = f1_score(y_true_all, y_pred_all, average="macro", zero_division=0)
        pooled_wt_f1 = f1_score(y_true_all, y_pred_all, average="weighted", zero_division=0)
        
        p_pool, r_pool, f_pool, s_pool = precision_recall_fscore_support(
            y_true_all, y_pred_all, labels=classes, zero_division=0
        )
        
        per_class_pooled = {}
        for c_idx, cls_name in enumerate(classes):
            per_class_pooled[cls_name] = {
                "precision": p_pool[c_idx],
                "recall": r_pool[c_idx],
                "f1_score": f_pool[c_idx],
                "support": int(s_pool[c_idx])
            }
        
        # 3. Confusion Matrix
        cm_counts = confusion_matrix(y_true_all, y_pred_all, labels=classes)
        cm_norm = cm_counts.astype(float) / cm_counts.sum(axis=1)[:, np.newaxis]
        
        # 4. Compare with reported values
        # Reported aggregate
        rep_agg_row = df_agg_rep[df_agg_rep["model"] == model].iloc[0]
        
        agg_comparison = {
            "mean_macro_f1": {"recalculated": mean_mac_f1, "reported": rep_agg_row["mean_macro_f1"], "diff": abs(mean_mac_f1 - rep_agg_row["mean_macro_f1"])},
            "std_macro_f1": {"recalculated": std_mac_f1, "reported": rep_agg_row["std_macro_f1"], "diff": abs(std_mac_f1 - rep_agg_row["std_macro_f1"])},
            "mean_balanced_accuracy": {"recalculated": mean_bal_acc, "reported": rep_agg_row["mean_balanced_accuracy"], "diff": abs(mean_bal_acc - rep_agg_row["mean_balanced_accuracy"])},
            "std_balanced_accuracy": {"recalculated": std_bal_acc, "reported": rep_agg_row["std_balanced_accuracy"], "diff": abs(std_bal_acc - rep_agg_row["std_balanced_accuracy"])},
            "mean_accuracy": {"recalculated": mean_acc, "reported": rep_agg_row["mean_accuracy"], "diff": abs(mean_acc - rep_agg_row["mean_accuracy"])},
            "std_accuracy": {"recalculated": std_acc, "reported": rep_agg_row["std_accuracy"], "diff": abs(std_acc - rep_agg_row["std_accuracy"])},
            "mean_weighted_f1": {"recalculated": mean_wt_f1, "reported": rep_agg_row["mean_weighted_f1"], "diff": abs(mean_wt_f1 - rep_agg_row["mean_weighted_f1"])},
            "std_weighted_f1": {"recalculated": std_wt_f1, "reported": rep_agg_row["std_weighted_f1"], "diff": abs(std_wt_f1 - rep_agg_row["std_weighted_f1"])},
            "pooled_macro_f1": {"recalculated": pooled_mac_f1, "reported": rep_agg_row["pooled_macro_f1"], "diff": abs(pooled_mac_f1 - rep_agg_row["pooled_macro_f1"])},
            "pooled_balanced_accuracy": {"recalculated": pooled_bal_acc, "reported": rep_agg_row["pooled_balanced_accuracy"], "diff": abs(pooled_bal_acc - rep_agg_row["pooled_balanced_accuracy"])},
            "pooled_accuracy": {"recalculated": pooled_acc, "reported": rep_agg_row["pooled_accuracy"], "diff": abs(pooled_acc - rep_agg_row["pooled_accuracy"])},
            "pooled_weighted_f1": {"recalculated": pooled_wt_f1, "reported": rep_agg_row["pooled_weighted_f1"], "diff": abs(pooled_wt_f1 - rep_agg_row["pooled_weighted_f1"])},
        }
        
        # Compare fold metrics
        fold_comparisons = []
        df_fold_rep_m = df_fold_rep[df_fold_rep["model"] == model]
        for fold in range(num_folds):
            rep_f_row = df_fold_rep_m[df_fold_rep_m["fold"] == fold].iloc[0]
            rec_f_row = df_f_recalc[df_f_recalc["fold"] == fold].iloc[0]
            
            f_comp = {
                "fold": fold,
                "accuracy": {"rec": rec_f_row["accuracy"], "rep": rep_f_row["accuracy"], "diff": abs(rec_f_row["accuracy"] - rep_f_row["accuracy"])},
                "balanced_accuracy": {"rec": rec_f_row["balanced_accuracy"], "rep": rep_f_row["balanced_accuracy"], "diff": abs(rec_f_row["balanced_accuracy"] - rep_f_row["balanced_accuracy"])},
                "macro_f1": {"rec": rec_f_row["macro_f1"], "rep": rep_f_row["macro_f1"], "diff": abs(rec_f_row["macro_f1"] - rep_f_row["macro_f1"])},
                "weighted_f1": {"rec": rec_f_row["weighted_f1"], "rep": rep_f_row["weighted_f1"], "diff": abs(rec_f_row["weighted_f1"] - rep_f_row["weighted_f1"])},
            }
            fold_comparisons.append(f_comp)
            
        # Compare confusion matrix counts
        cm_counts_diff = 0
        cm_norm_diff = 0.0
        df_cm_m = df_cm_cnt_rep[df_cm_cnt_rep["model"] == model]
        df_cmn_m = df_cm_norm_rep[df_cm_norm_rep["model"] == model]
        
        for i, true_cls in enumerate(classes):
            for j, pred_cls in enumerate(classes):
                cnt_rep = df_cm_m[(df_cm_m["true_class"] == true_cls) & (df_cm_m["pred_class"] == pred_cls)]["count"].iloc[0]
                norm_rep = df_cmn_m[(df_cmn_m["true_class"] == true_cls) & (df_cmn_m["pred_class"] == pred_cls)]["normalized_rate"].iloc[0]
                
                cnt_rec = cm_counts[i, j]
                norm_rec = cm_norm[i, j]
                
                cm_counts_diff += abs(cnt_rec - cnt_rep)
                cm_norm_diff += abs(norm_rec - norm_rep)
                
        # Compare per-class fold metrics
        per_class_diff_sum = 0.0
        df_pc_m = df_per_cls_rep[df_per_cls_rep["model"] == model]
        for pcf in per_class_fold_recalc:
            f = pcf["fold"]
            c = pcf["class_name"]
            row_rep = df_pc_m[(df_pc_m["fold"] == f) & (df_pc_m["class_name"] == c)].iloc[0]
            per_class_diff_sum += abs(pcf["precision"] - row_rep["precision"])
            per_class_diff_sum += abs(pcf["recall"] - row_rep["recall"])
            per_class_diff_sum += abs(pcf["f1_score"] - row_rep["f1_score"])
            assert pcf["support"] == row_rep["support"], f"Support mismatch in {model} fold {f} class {c}"
        
        results["models"][model] = {
            "unweighted_fold_summary": {
                "mean_macro_f1": mean_mac_f1,
                "std_macro_f1": std_mac_f1,
                "mean_balanced_accuracy": mean_bal_acc,
                "std_balanced_accuracy": std_bal_acc,
                "mean_accuracy": mean_acc,
                "std_accuracy": std_acc,
                "mean_weighted_f1": mean_wt_f1,
                "std_weighted_f1": std_wt_f1,
            },
            "pooled_oof_metrics": {
                "macro_f1": pooled_mac_f1,
                "balanced_accuracy": pooled_bal_acc,
                "accuracy": pooled_acc,
                "weighted_f1": pooled_wt_f1,
            },
            "per_class_pooled": per_class_pooled,
            "confusion_matrix_counts": cm_counts.tolist(),
            "confusion_matrix_normalized": cm_norm.tolist(),
            "aggregate_comparison": agg_comparison,
            "fold_comparisons": fold_comparisons,
            "cm_counts_total_diff": int(cm_counts_diff),
            "cm_norm_total_diff": float(cm_norm_diff),
            "per_class_fold_diff_sum": float(per_class_diff_sum)
        }
        
        print(f"  [Recalculated Pooled] Macro F1: {pooled_mac_f1:.4f} | Bal Acc: {pooled_bal_acc:.4f} | Acc: {pooled_acc:.4f} | Wt F1: {pooled_wt_f1:.4f}")
        print(f"  [Reported Pooled]     Macro F1: {rep_agg_row['pooled_macro_f1']:.4f} | Bal Acc: {rep_agg_row['pooled_balanced_accuracy']:.4f} | Acc: {rep_agg_row['pooled_accuracy']:.4f} | Wt F1: {rep_agg_row['pooled_weighted_f1']:.4f}")
        print(f"  [Recalculated Fold Mean] Macro F1: {mean_mac_f1:.4f} (std: {std_mac_f1:.4f}) | Bal Acc: {mean_bal_acc:.4f} | Acc: {mean_acc:.4f}")
        print(f"  [Reported Fold Mean]     Macro F1: {rep_agg_row['mean_macro_f1']:.4f} (std: {rep_agg_row['std_macro_f1']:.4f}) | Bal Acc: {rep_agg_row['mean_balanced_accuracy']:.4f} | Acc: {rep_agg_row['mean_accuracy']:.4f}")
        print(f"  Max Diff in Aggregate Table: {max(v['diff'] for v in agg_comparison.values()):.6f}")
        print(f"  CM Counts Diff: {cm_counts_diff}, CM Norm Diff: {cm_norm_diff:.6f}, Per-Class Fold Diff Sum: {per_class_diff_sum:.6e}")
        
    return results

def main():
    all_results = {}
    for track_key in ["track_a", "track_b"]:
        all_results[track_key] = verify_track(track_key)
        
    # Save results to a json file for easy report generation
    out_json = os.path.join(OUTPUT_DIR, "independent_verification_results.json")
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved verification JSON to {out_json}")

if __name__ == "__main__":
    main()
