#!/usr/bin/env python3
"""
scripts/run_phase6d_experiments.py

Phase 6D: Experimental Benchmarking on Clean Reconstructed Dataset (4,196 flows)
Evaluates:
1. Fast Baselines (Logistic Regression, Linear SVM, Random Forest, Extra Trees, HistGradientBoosting, XGBoost)
2. Class Weighting & Tuning on Clean Data
3. Early Horizons: Early-5, Early-10, Early-15, Early-20
4. Generates Normalized Confusion Matrix and Publication Plots
"""

import os
import sys
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

os.environ['LOKY_MAX_CPU_COUNT'] = '4'

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score,
    balanced_accuracy_score,
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)

GLOBAL_SEED = 42

def main():
    print("=" * 80)
    print("PHASE 6D: CLEAN RECONSTRUCTED DATASET BENCHMARKING (4,196 FLOWS)")
    print("=" * 80)
    
    out_dir = Path("output/phase6d")
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    df_clean = pd.read_csv(out_dir / "clean_track_b_flows.csv")
    df_folds = pd.read_csv(out_dir / "clean_track_b_folds.csv")
    
    classes = sorted(df_clean["qos_class"].unique())
    num_classes = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class = {i: c for i, c in enumerate(classes)}
    total_samples = len(df_clean)
    
    y_str = df_clean["qos_class"].values
    y_int = np.array([class_to_idx[c] for c in y_str], dtype=int)
    
    early_5_feats = [
        "early_5_complete", "early_5_duration", "early_5_bytes",
        "early_5_fwd_pkts", "early_5_bwd_pkts", "early_5_avg_size"
    ]
    X_base = df_clean[early_5_feats].astype(float).values
    
    # -------------------------------------------------------------------------
    # 1. BASELINE BENCHMARK ON CLEAN EARLY-5
    # -------------------------------------------------------------------------
    print("\n>>> STAGE 1: Fast Baseline Benchmark on Clean Early-5...")
    baseline_models = {
        "Logistic_Regression": lambda: LogisticRegression(class_weight="balanced", max_iter=1000, random_state=GLOBAL_SEED),
        "Linear_SVM": lambda: SGDClassifier(loss="hinge", class_weight="balanced", max_iter=1000, random_state=GLOBAL_SEED),
        "Random_Forest": lambda: RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=GLOBAL_SEED, n_jobs=2),
        "Extra_Trees": lambda: ExtraTreesClassifier(n_estimators=100, class_weight="balanced", random_state=GLOBAL_SEED, n_jobs=2),
        "HistGradientBoosting": lambda: HistGradientBoostingClassifier(class_weight="balanced", random_state=GLOBAL_SEED, max_iter=100),
        "XGBoost_Baseline": lambda: XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, subsample=0.8, colsample_bytree=0.8, objective="multi:softprob", eval_metric="mlogloss", random_state=GLOBAL_SEED, n_jobs=2)
    }
    
    baseline_rows = []
    for m_name, m_fn in baseline_models.items():
        fold_f1s = []
        oof_preds = np.full(total_samples, -1, dtype=int)
        total_pred_time = 0.0
        t0 = time.time()
        needs_scaling = m_name in ["Logistic_Regression", "Linear_SVM"]
        
        for fold in range(5):
            tr_idx = np.where(df_folds["fold"] != fold)[0]
            va_idx = np.where(df_folds["fold"] == fold)[0]
            X_tr, y_tr = X_base[tr_idx], y_int[tr_idx]
            X_va, y_va = X_base[va_idx], y_int[va_idx]
            
            if needs_scaling:
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_tr)
                X_va = scaler.transform(X_va)
                
            clf = m_fn()
            
            c_counts = np.bincount(y_tr)
            sw = len(y_tr) / (num_classes * c_counts[y_tr])
            
            if "XGB" in m_name:
                clf.fit(X_tr, y_tr, sample_weight=sw)
            else:
                clf.fit(X_tr, y_tr)
                
            tp0 = time.time()
            preds = clf.predict(X_va)
            total_pred_time += (time.time() - tp0)
            
            oof_preds[va_idx] = preds
            fold_f1s.append(f1_score(y_va, preds, average="macro", zero_division=0))
            
        train_time = time.time() - t0 - total_pred_time
        lat_us = (total_pred_time / total_samples) * 1e6
        
        p_macro = f1_score(y_int, oof_preds, average="macro", zero_division=0)
        p_bal = balanced_accuracy_score(y_int, oof_preds)
        p_acc = accuracy_score(y_int, oof_preds)
        prec, rec, f1s, _ = precision_recall_fscore_support(y_int, oof_preds, labels=range(num_classes), zero_division=0)
        
        baseline_rows.append({
            "model": m_name,
            "mean_macro_f1": round(np.mean(fold_f1s), 4),
            "std_macro_f1": round(np.std(fold_f1s), 4),
            "pooled_macro_f1": round(p_macro, 4),
            "pooled_balanced_acc": round(p_bal, 4),
            "pooled_accuracy": round(p_acc, 4),
            "bulk_recall": round(rec[0], 4),
            "realtime_recall": round(rec[1], 4),
            "streaming_recall": round(rec[2], 4),
            "bulk_precision": round(prec[0], 4),
            "realtime_precision": round(prec[1], 4),
            "streaming_precision": round(prec[2], 4),
            "train_time_sec": round(train_time, 2),
            "latency_us_per_sample": round(lat_us, 2)
        })
        print(f"  {m_name:<22}: Mean CV F1={np.mean(fold_f1s):.4f} +/- {np.std(fold_f1s):.4f} | Pooled F1={p_macro:.4f} | Bal Acc={p_bal:.4f} | RT Rec={rec[1]:.4f} | Bulk Rec={rec[0]:.4f} | Str Rec={rec[2]:.4f}")
        
    df_baselines = pd.DataFrame(baseline_rows)
    df_baselines.to_csv(out_dir / "clean_baseline_comparison.csv", index=False)
    
    # -------------------------------------------------------------------------
    # 2. TUNED MODELS & REAL-TIME SHIELDING (Clean Data)
    # -------------------------------------------------------------------------
    print("\n>>> STAGE 2: Targeted Tuning & Real-Time Weighting on Clean Data...")
    tuned_variants = [
        ("XGBoost_Clean_Balanced", lambda: XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, objective="multi:softprob", eval_metric="mlogloss", random_state=GLOBAL_SEED, n_jobs=2), [1.0, 1.0, 1.0]),
        ("XGBoost_Clean_RT_Shield (RT=1.5x)", lambda: XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, objective="multi:softprob", eval_metric="mlogloss", random_state=GLOBAL_SEED, n_jobs=2), [1.0, 1.5, 1.0]),
        ("Extra_Trees_Clean (n=100, balanced)", lambda: ExtraTreesClassifier(n_estimators=100, class_weight="balanced", random_state=GLOBAL_SEED, n_jobs=2), None),
        ("Extra_Trees_Clean_RT (RT=1.5x)", lambda: ExtraTreesClassifier(n_estimators=100, class_weight={0: 1.0, 1: 1.5, 2: 1.0}, random_state=GLOBAL_SEED, n_jobs=2), None),
        ("HistGradientBoosting_Clean", lambda: HistGradientBoostingClassifier(class_weight="balanced", random_state=GLOBAL_SEED, max_iter=100), None)
    ]
    
    tuned_rows = []
    best_model_name = None
    best_macro_f1 = -1.0
    best_oof_preds = None
    best_oof_probs = None
    
    for c_name, clf_fn, mult in tuned_variants:
        fold_f1s = []
        oof_preds = np.full(total_samples, -1, dtype=int)
        oof_probs = np.zeros((total_samples, num_classes), dtype=float)
        t0 = time.time()
        pred_time_total = 0.0
        
        for fold in range(5):
            tr_idx = np.where(df_folds["fold"] != fold)[0]
            va_idx = np.where(df_folds["fold"] == fold)[0]
            X_tr, y_tr = X_base[tr_idx], y_int[tr_idx]
            X_va, y_va = X_base[va_idx], y_int[va_idx]
            
            clf = clf_fn()
            c_counts = np.bincount(y_tr)
            base_sw = len(y_tr) / (num_classes * c_counts[y_tr])
            
            if mult is not None:
                sw = base_sw * np.array([mult[y] for y in y_tr])
            else:
                sw = base_sw
                
            if "XGB" in c_name:
                clf.fit(X_tr, y_tr, sample_weight=sw)
            else:
                clf.fit(X_tr, y_tr)
                
            tp0 = time.time()
            probs = clf.predict_proba(X_va)
            preds = np.argmax(probs, axis=1)
            pred_time_total += (time.time() - tp0)
            
            oof_preds[va_idx] = preds
            oof_probs[va_idx] = probs
            fold_f1s.append(f1_score(y_va, preds, average="macro", zero_division=0))
            
        train_time = time.time() - t0 - pred_time_total
        lat_us = (pred_time_total / total_samples) * 1e6
        
        p_macro = f1_score(y_int, oof_preds, average="macro", zero_division=0)
        p_bal = balanced_accuracy_score(y_int, oof_preds)
        p_acc = accuracy_score(y_int, oof_preds)
        prec, rec, f1s, _ = precision_recall_fscore_support(y_int, oof_preds, labels=range(num_classes), zero_division=0)
        
        tuned_rows.append({
            "model_variant": c_name,
            "mean_macro_f1": round(np.mean(fold_f1s), 4),
            "std_macro_f1": round(np.std(fold_f1s), 4),
            "pooled_macro_f1": round(p_macro, 4),
            "pooled_balanced_acc": round(p_bal, 4),
            "pooled_accuracy": round(p_acc, 4),
            "bulk_recall": round(rec[0], 4),
            "realtime_recall": round(rec[1], 4),
            "streaming_recall": round(rec[2], 4),
            "bulk_precision": round(prec[0], 4),
            "realtime_precision": round(prec[1], 4),
            "streaming_precision": round(prec[2], 4),
            "train_time_sec": round(train_time, 2),
            "latency_us_per_sample": round(lat_us, 2)
        })
        print(f"  {c_name:<36}: Mean F1={np.mean(fold_f1s):.4f} | Pooled F1={p_macro:.4f} | Bal Acc={p_bal:.4f} | RT Rec={rec[1]:.4f} | RT Prec={prec[1]:.4f}")
        
        if p_macro > best_macro_f1:
            best_macro_f1 = p_macro
            best_model_name = c_name
            best_oof_preds = oof_preds
            best_oof_probs = oof_probs
            
    df_tuned = pd.DataFrame(tuned_rows)
    df_tuned.to_csv(out_dir / "clean_tuned_comparison.csv", index=False)
    
    # -------------------------------------------------------------------------
    # 3. HORIZON PROGRESSION BENCHMARK ON CLEAN DATA
    # -------------------------------------------------------------------------
    print("\n>>> STAGE 3: Horizon Progression on Clean Data (Early-5 to Early-20)...")
    horizon_features = {
        "early_5": early_5_feats,
        "early_10": early_5_feats + [
            "early_10_complete", "early_10_duration", "early_10_bytes",
            "early_10_fwd_pkts", "early_10_bwd_pkts", "early_10_avg_size"
        ],
        "early_15": early_5_feats + [
            "early_10_complete", "early_10_duration", "early_10_bytes",
            "early_10_fwd_pkts", "early_10_bwd_pkts", "early_10_avg_size",
            "early_15_complete", "early_15_duration", "early_15_bytes",
            "early_15_fwd_pkts", "early_15_bwd_pkts", "early_15_avg_size"
        ],
        "early_20": early_5_feats + [
            "early_10_complete", "early_10_duration", "early_10_bytes",
            "early_10_fwd_pkts", "early_10_bwd_pkts", "early_10_avg_size",
            "early_15_complete", "early_15_duration", "early_15_bytes",
            "early_15_fwd_pkts", "early_15_bwd_pkts", "early_15_avg_size",
            "early_20_complete", "early_20_duration", "early_20_bytes",
            "early_20_fwd_pkts", "early_20_bwd_pkts", "early_20_avg_size"
        ]
    }
    
    horizon_rows = []
    for h_name, f_list in horizon_features.items():
        X_h = df_clean[f_list].astype(float).values
        oof_preds = np.full(total_samples, -1, dtype=int)
        fold_f1s = []
        t0 = time.time()
        
        for fold in range(5):
            tr_idx = np.where(df_folds["fold"] != fold)[0]
            va_idx = np.where(df_folds["fold"] == fold)[0]
            X_tr, y_tr = X_h[tr_idx], y_int[tr_idx]
            X_va, y_va = X_h[va_idx], y_int[va_idx]
            
            c_counts = np.bincount(y_tr)
            sw = len(y_tr) / (num_classes * c_counts[y_tr])
            
            clf = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, objective="multi:softprob", eval_metric="mlogloss", random_state=GLOBAL_SEED, n_jobs=2)
            clf.fit(X_tr, y_tr, sample_weight=sw)
            preds = clf.predict(X_va)
            oof_preds[va_idx] = preds
            fold_f1s.append(f1_score(y_va, preds, average="macro", zero_division=0))
            
        elapsed = time.time() - t0
        p_macro = f1_score(y_int, oof_preds, average="macro", zero_division=0)
        p_bal = balanced_accuracy_score(y_int, oof_preds)
        p_acc = accuracy_score(y_int, oof_preds)
        prec, rec, f1s, _ = precision_recall_fscore_support(y_int, oof_preds, labels=range(num_classes), zero_division=0)
        
        horizon_rows.append({
            "horizon": h_name,
            "feature_count": len(f_list),
            "mean_macro_f1": round(np.mean(fold_f1s), 4),
            "std_macro_f1": round(np.std(fold_f1s), 4),
            "pooled_macro_f1": round(p_macro, 4),
            "pooled_balanced_acc": round(p_bal, 4),
            "pooled_accuracy": round(p_acc, 4),
            "bulk_recall": round(rec[0], 4),
            "realtime_recall": round(rec[1], 4),
            "streaming_recall": round(rec[2], 4),
            "train_time_sec": round(elapsed, 2)
        })
        print(f"  {h_name.upper():<10} ({len(f_list)} feats): Pooled F1={p_macro:.4f} | Bal Acc={p_bal:.4f} | RT Rec={rec[1]:.4f} | Bulk Rec={rec[0]:.4f} | Str Rec={rec[2]:.4f}")
        
    df_horizons = pd.DataFrame(horizon_rows)
    df_horizons.to_csv(out_dir / "clean_horizon_comparison.csv", index=False)
    
    # -------------------------------------------------------------------------
    # 4. EXPORT BEST MODEL ARTIFACTS & NORMALIZED CONFUSION MATRIX
    # -------------------------------------------------------------------------
    print(f"\n>>> STAGE 4: Exporting Best Clean Model ({best_model_name})...")
    df_oof = df_clean[["flow_id", "capture_group", "qos_class", "application", "is_vpn"]].copy()
    df_oof["fold"] = df_folds["fold"].values
    df_oof["pred_qos_class"] = [idx_to_class[p] for p in best_oof_preds]
    for c_idx, c_name in idx_to_class.items():
        df_oof[f"prob_{c_name}"] = best_oof_probs[:, c_idx]
    df_oof.to_csv(out_dir / "clean_best_oof_predictions.csv", index=False)
    
    cm = confusion_matrix(y_int, best_oof_preds, labels=range(num_classes))
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    cm_rows = []
    for i, t_cls in enumerate(classes):
        for j, p_cls in enumerate(classes):
            cm_rows.append({
                "true_class": t_cls,
                "pred_class": p_cls,
                "count": int(cm[i, j]),
                "normalized_rate": round(float(cm_norm[i, j]), 4)
            })
    df_cm = pd.DataFrame(cm_rows)
    df_cm.to_csv(out_dir / "clean_best_confusion_matrix.csv", index=False)
    
    print("\n=== CLEAN OPTIMIZED BEST MODEL CONFUSION MATRIX (Normalized) ===")
    cm_display = pd.DataFrame(cm_norm, index=classes, columns=classes)
    print(cm_display.round(4))
    
    # -------------------------------------------------------------------------
    # 5. GENERATE PUBLICATION VISUALIZATIONS
    # -------------------------------------------------------------------------
    print("\n>>> STAGE 5: Generating Visualizations...")
    sns.set_theme(style="whitegrid", font="sans-serif")
    
    # 1. Confusion Matrix
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm_display, annot=True, fmt=".3f", cmap="Greens", cbar=True, vmin=0, vmax=1.0)
    plt.title(f"Phase 6D: Clean Reconstructed Early-5 Confusion Matrix\n({best_model_name})", fontsize=12, fontweight="bold")
    plt.xlabel("Predicted QoS Class", fontsize=11)
    plt.ylabel("True QoS Class", fontsize=11)
    plt.tight_layout()
    plt.savefig(plots_dir / "clean_confusion_matrix.png", dpi=300)
    plt.close()
    
    # 2. Pareto Plot
    plt.figure(figsize=(9, 5))
    for _, r in df_baselines.iterrows():
        plt.scatter(r["latency_us_per_sample"], r["pooled_macro_f1"], s=100, label=r["model"])
        plt.annotate(r["model"], (r["latency_us_per_sample"] * 1.1, r["pooled_macro_f1"]), fontsize=9)
    best_tuned = df_tuned[df_tuned["model_variant"] == best_model_name].iloc[0]
    plt.scatter(best_tuned["latency_us_per_sample"], best_tuned["pooled_macro_f1"], s=160, color="green", marker="D", label=f"BEST: {best_model_name}")
    plt.annotate(f"★ {best_model_name}", (best_tuned["latency_us_per_sample"] * 1.1, best_tuned["pooled_macro_f1"]), fontsize=10, fontweight="bold", color="green")
    
    plt.xscale("log")
    plt.title("Phase 6D: Clean Accuracy vs. Latency Pareto Frontier (Early-5)", fontsize=13, fontweight="bold")
    plt.xlabel("Inference Latency per Flow (microseconds, log scale)", fontsize=11)
    plt.ylabel("Pooled Macro F1 Score (Clean Dataset)", fontsize=11)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(plots_dir / "clean_accuracy_vs_latency_pareto.png", dpi=300)
    plt.close()
    
    # 3. Horizon Progression
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=df_horizons, x="horizon", y="pooled_macro_f1", marker="o", linewidth=2.5, color="darkgreen", label="Pooled Macro F1")
    sns.lineplot(data=df_horizons, x="horizon", y="pooled_balanced_acc", marker="s", linewidth=2.5, color="teal", label="Pooled Balanced Accuracy")
    sns.lineplot(data=df_horizons, x="horizon", y="realtime_recall", marker="^", linewidth=2.5, color="coral", label="Real-Time Recall")
    plt.title("Clean Dataset Performance Across Early Horizons (Early-5 to Early-20)", fontsize=13, fontweight="bold")
    plt.xlabel("Observation Horizon (Early Packet Count)", fontsize=11)
    plt.ylabel("Metric Score", fontsize=11)
    plt.ylim(0.40, 0.75)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(plots_dir / "clean_horizon_comparison.png", dpi=300)
    plt.close()
    
    print(f"\n[SUCCESS] Phase 6D experiments completed successfully. All artifacts in {out_dir}")

if __name__ == "__main__":
    main()
