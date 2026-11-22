#!/usr/bin/env python3
"""
scripts/test_phase6c_baselines.py

Stage 1: Fast Baseline Benchmarking on Phase 6B Early-5 Feature Set (6 features).
Evaluates available models:
1. Logistic Regression (class_weight='balanced')
2. Linear SVM (SGDClassifier, loss='hinge', class_weight='balanced')
3. Random Forest (class_weight='balanced')
4. Extra Trees (class_weight='balanced')
5. HistGradientBoosting (class_weight='balanced')
6. XGBoost (sample_weight='balanced')
"""

import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score, precision_recall_fscore_support, confusion_matrix

GLOBAL_SEED = 42

def run_baselines():
    df_flows = pd.read_csv("output/phase5/track_b_flows.csv")
    df_folds = pd.read_csv("output/phase5/track_b_folds.csv")
    
    classes = sorted(df_flows["qos_class"].unique())
    num_classes = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class = {i: c for i, c in enumerate(classes)}
    
    y_str = df_flows["qos_class"].values
    y_int = np.array([class_to_idx[c] for c in y_str], dtype=int)
    
    feature_list = [
        "early_5_complete", "early_5_duration", "early_5_bytes",
        "early_5_fwd_pkts", "early_5_bwd_pkts", "early_5_avg_size"
    ]
    X_df = df_flows[feature_list].copy()
    for col in X_df.columns:
        if X_df[col].dtype == bool:
            X_df[col] = X_df[col].astype(float)
            
    num_folds = 5
    total_samples = len(df_flows)
    
    models = {
        "Logistic_Regression": lambda: LogisticRegression(class_weight="balanced", max_iter=1000, random_state=GLOBAL_SEED),
        "Linear_SVM": lambda: SGDClassifier(loss="hinge", class_weight="balanced", max_iter=1000, random_state=GLOBAL_SEED),
        "Random_Forest": lambda: RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=GLOBAL_SEED, n_jobs=-1),
        "Extra_Trees": lambda: ExtraTreesClassifier(n_estimators=100, class_weight="balanced", random_state=GLOBAL_SEED, n_jobs=-1),
        "HistGradientBoosting": lambda: HistGradientBoostingClassifier(class_weight="balanced", random_state=GLOBAL_SEED),
        "XGBoost": lambda: XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, subsample=0.8, colsample_bytree=0.8, objective="multi:softprob", eval_metric="mlogloss", random_state=GLOBAL_SEED, n_jobs=-1)
    }
    
    results = []
    
    for model_name, model_fn in models.items():
        print(f"--- Running {model_name} ---")
        fold_f1s = []
        oof_preds = np.full(total_samples, -1, dtype=int)
        total_train_time = 0.0
        total_pred_time = 0.0
        
        needs_scaling = model_name in ["Logistic_Regression", "Linear_SVM"]
        
        for fold in range(num_folds):
            train_idx = np.where(df_folds["fold"] != fold)[0]
            val_idx = np.where(df_folds["fold"] == fold)[0]
            
            X_train = X_df.iloc[train_idx].values
            X_val = X_df.iloc[val_idx].values
            y_train = y_int[train_idx]
            y_val = y_int[val_idx]
            
            if needs_scaling:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_val = scaler.transform(X_val)
                
            clf = model_fn()
            
            # Train weights
            classes_in_train, counts_in_train = np.unique(y_train, return_counts=True)
            class_weights_dict = {c: len(y_train) / (num_classes * count) for c, count in zip(classes_in_train, counts_in_train)}
            sample_weights_train = np.array([class_weights_dict[y] for y in y_train], dtype=np.float32)
            
            t0 = time.time()
            if model_name == "XGBoost":
                clf.fit(X_train, y_train, sample_weight=sample_weights_train)
            else:
                clf.fit(X_train, y_train)
            train_t = time.time() - t0
            total_train_time += train_t
            
            t1 = time.time()
            val_preds = clf.predict(X_val)
            pred_t = time.time() - t1
            total_pred_time += pred_t
            
            oof_preds[val_idx] = val_preds
            f1 = f1_score(y_val, val_preds, average="macro", zero_division=0)
            fold_f1s.append(f1)
            
        pooled_macro_f1 = f1_score(y_int, oof_preds, average="macro", zero_division=0)
        pooled_bal_acc = balanced_accuracy_score(y_int, oof_preds)
        pooled_acc = accuracy_score(y_int, oof_preds)
        p, r, f, _ = precision_recall_fscore_support(y_int, oof_preds, labels=range(num_classes), zero_division=0)
        
        avg_latency_us = (total_pred_time / total_samples) * 1e6
        
        print(f"  Mean Macro F1: {np.mean(fold_f1s):.4f} +/- {np.std(fold_f1s):.4f} | Pooled Macro F1: {pooled_macro_f1:.4f} | Bal Acc: {pooled_bal_acc:.4f} | Train Time: {total_train_time:.2f}s | Latency: {avg_latency_us:.2f}us")
        print(f"  Per-class Recall: Bulk={r[0]:.4f}, Real-Time={r[1]:.4f}, Streaming={r[2]:.4f}")
        print(f"  Per-class Precision: Bulk={p[0]:.4f}, Real-Time={p[1]:.4f}, Streaming={p[2]:.4f}")
        
        results.append({
            "model": model_name,
            "mean_macro_f1": np.mean(fold_f1s),
            "std_macro_f1": np.std(fold_f1s),
            "pooled_macro_f1": pooled_macro_f1,
            "pooled_balanced_accuracy": pooled_bal_acc,
            "pooled_accuracy": pooled_acc,
            "bulk_recall": r[0],
            "realtime_recall": r[1],
            "streaming_recall": r[2],
            "bulk_precision": p[0],
            "realtime_precision": p[1],
            "streaming_precision": p[2],
            "total_train_time_sec": total_train_time,
            "latency_us_per_sample": avg_latency_us
        })
        
    df_res = pd.DataFrame(results)
    print("\n=== SUMMARY BASELINE TABLE ===")
    print(df_res[["model", "mean_macro_f1", "pooled_macro_f1", "pooled_balanced_accuracy", "realtime_recall", "total_train_time_sec", "latency_us_per_sample"]])

if __name__ == "__main__":
    run_baselines()
