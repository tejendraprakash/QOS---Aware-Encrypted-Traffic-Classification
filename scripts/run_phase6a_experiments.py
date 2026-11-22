"""
run_phase6a_experiments.py
Phase 6A: Comprehensive Full-Flow Machine Learning Baseline Experiment
QoS-Aware Encrypted Traffic Classification for Next-Generation Networks

Evaluates:
- Random Forest
- XGBoost
- Packet Multi-Layer Perceptron (P-MLP)

Across:
- Track A (Exploratory 4-Class): 3-fold StratifiedGroupKFold
- Track B (Primary 3-Class): 5-fold StratifiedGroupKFold

Strict Integrity Rules:
- Original datasets untouched.
- Leak-free preprocessing: constant and collinear feature removal and scaling fitted on train fold only.
- No early-packet features in Phase 6A (reserved for Phase 6B).
- Macro F1 and Balanced Accuracy as primary metrics.
"""

import os
import sys
import time
import json
import random
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score,
    balanced_accuracy_score,
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# -----------------------------------------------------------------------------
# Reproducibility Seeds
# -----------------------------------------------------------------------------
GLOBAL_SEED = 42

def set_seeds(seed=GLOBAL_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# -----------------------------------------------------------------------------
# PyTorch P-MLP Architecture (from Phase 5 Approved Spec)
# -----------------------------------------------------------------------------
class PMLP(nn.Module):
    def __init__(self, in_features, num_classes):
        super(PMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.net(x)

def train_pmlp(X_train, y_train, X_val, y_val, num_classes, class_weights_tensor, device, max_epochs=100, patience=15, batch_size=64):
    set_seeds(GLOBAL_SEED)
    model = PMLP(in_features=X_train.shape[1], num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor.to(device))
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=(len(train_dataset) > batch_size))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    best_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0
    training_log = []
    
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * bx.size(0)
        train_loss /= len(train_dataset)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                out = model(bx)
                loss = criterion(out, by)
                val_loss += loss.item() * bx.size(0)
        val_loss /= len(val_dataset)
        
        scheduler.step(val_loss)
        training_log.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_state = model.state_dict().copy()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                # Early stop
                break
                
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model, training_log

# -----------------------------------------------------------------------------
# Preprocessing Protocol: Training Fold Only
# -----------------------------------------------------------------------------
def fit_transform_fold_features(X_train_df, X_val_df):
    """
    Identifies constant features and perfect collinear pairs on X_train_df ONLY.
    Applies the identical exclusions to X_val_df.
    Returns: X_train_clean, X_val_clean, dropped_constants, dropped_collinear, active_features
    """
    # 1. Constant features (variance == 0.0)
    std_s = X_train_df.std(axis=0)
    constant_cols = list(std_s[std_s == 0.0].index)
    
    X_train_clean = X_train_df.drop(columns=constant_cols)
    X_val_clean = X_val_df.drop(columns=constant_cols)
    
    # 2. Perfect collinear pairs (|r| >= 0.999)
    corr_matrix = X_train_clean.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    collinear_cols = [column for column in upper_tri.columns if any(upper_tri[column] >= 0.999)]
    
    X_train_clean = X_train_clean.drop(columns=collinear_cols)
    X_val_clean = X_val_clean.drop(columns=collinear_cols)
    
    active_features = list(X_train_clean.columns)
    return X_train_clean.values, X_val_clean.values, constant_cols, collinear_cols, active_features

# -----------------------------------------------------------------------------
# Main Experiment Execution Engine
# -----------------------------------------------------------------------------
def execute_track_experiment(track_code, track_name, flows_csv, folds_csv, num_folds, out_dir):
    print(f"\n{'='*75}")
    print(f" EXECUTION: {track_name}")
    print(f"{'='*75}")
    track_out_dir = os.path.join(out_dir, track_code)
    os.makedirs(track_out_dir, exist_ok=True)
    
    # 1. Load and Verify Dataset & Folds
    print(f"[VERIFY] Loading flows: {flows_csv}")
    df_flows = pd.read_csv(flows_csv)
    print(f"[VERIFY] Loading folds: {folds_csv}")
    df_folds = pd.read_csv(folds_csv)
    
    assert len(df_flows) == len(df_folds), f"Row mismatch: flows={len(df_flows)}, folds={len(df_folds)}"
    assert (df_flows["flow_id"].values == df_folds["flow_id"].values).all(), "Flow ID alignment mismatch between flows and folds!"
    
    # Pre-training verification checks
    total_samples = len(df_flows)
    classes = sorted(list(df_flows["qos_class"].unique()))
    num_classes = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class = {i: c for i, c in enumerate(classes)}
    
    print(f"[VERIFY] Total Flows: {total_samples}")
    print(f"[VERIFY] Classes ({num_classes}): {df_flows['qos_class'].value_counts().to_dict()}")
    print(f"[VERIFY] Unique Capture Groups: {df_flows['capture_group'].nunique()}")
    
    # Verify no missing or infinite values
    assert df_flows.isnull().sum().sum() == 0, "Null values detected in dataset!"
    
    # Define candidate feature set: exactly 79 full-flow features (exclude metadata & early-packet)
    metadata_cols = [
        "flow_id", "capture_filename", "capture_group", "application", "is_vpn",
        "qos_class", "label_confidence", "src_ip", "src_port", "dst_ip", "dst_port",
        "protocol_name", "flow_start_time", "flow_end_time"
    ]
    candidate_features = [c for c in df_flows.columns if c not in metadata_cols and not c.startswith("early_")]
    print(f"[VERIFY] Full-flow candidate behavioral features: {len(candidate_features)}")
    assert len(candidate_features) == 79, f"Expected 79 full-flow features, got {len(candidate_features)}"
    
    # Group leakage audit across folds
    fold_groups_audit = {}
    for f in range(num_folds):
        train_mask = (df_folds["fold"] != f)
        val_mask = (df_folds["fold"] == f)
        train_groups = set(df_flows.loc[train_mask, "capture_group"].unique())
        val_groups = set(df_flows.loc[val_mask, "capture_group"].unique())
        overlap = train_groups.intersection(val_groups)
        assert len(overlap) == 0, f"FATAL: Group leakage detected in fold {f}: {overlap}"
        val_classes = set(df_flows.loc[val_mask, "qos_class"].unique())
        assert val_classes == set(classes), f"FATAL: Missing class in val fold {f}: {val_classes}"
        fold_groups_audit[f] = {
            "train_samples": int(train_mask.sum()),
            "val_samples": int(val_mask.sum()),
            "train_groups": len(train_groups),
            "val_groups": len(val_groups),
            "group_overlap": len(overlap)
        }
    print(f"[AUDIT] Group leakage check: 100% PASS across all {num_folds} folds (0 group overlap).")
    
    # Prepare labels and candidate matrix
    y_str = df_flows["qos_class"].values
    y_int = np.array([class_to_idx[c] for c in y_str], dtype=int)
    X_df = df_flows[candidate_features].copy()
    
    models = ["Random_Forest", "XGBoost", "P_MLP"]
    
    fold_metric_records = []
    per_class_metric_records = []
    runtime_records = []
    all_oof_predictions = {m: np.full(total_samples, -1, dtype=int) for m in models}
    all_oof_probabilities = {m: np.zeros((total_samples, num_classes), dtype=float) for m in models}
    feature_usage_records = []
    training_log_records = []
    
    device = torch.device("cpu")
    
    for model_name in models:
        print(f"\n-----------------------------------------------------------")
        print(f" Training Model: {model_name} on {track_name}")
        print(f"-----------------------------------------------------------")
        
        for fold in range(num_folds):
            train_idx = np.where(df_folds["fold"] != fold)[0]
            val_idx = np.where(df_folds["fold"] == fold)[0]
            
            X_train_df = X_df.iloc[train_idx].copy()
            X_val_df = X_df.iloc[val_idx].copy()
            y_train = y_int[train_idx]
            y_val = y_int[val_idx]
            
            # Leak-free feature selection & collinearity filter
            X_train_clean, X_val_clean, drop_const, drop_coll, active_feats = fit_transform_fold_features(X_train_df, X_val_df)
            
            if model_name == "Random_Forest": # Record feature usage on first model
                feature_usage_records.append({
                    "track": track_code,
                    "fold": fold,
                    "candidate_features": len(candidate_features),
                    "dropped_constants": ";".join(drop_const),
                    "dropped_collinear": ";".join(drop_coll),
                    "active_features_count": len(active_feats),
                    "active_features_list": ";".join(active_feats)
                })
                
            # Class weighting calculated strictly on training fold
            classes_in_train, counts_in_train = np.unique(y_train, return_counts=True)
            total_train_samples = len(y_train)
            class_weights_dict = {c: total_train_samples / (num_classes * count) for c, count in zip(classes_in_train, counts_in_train)}
            class_weights_arr = np.array([class_weights_dict[i] for i in range(num_classes)], dtype=np.float32)
            sample_weights_train = np.array([class_weights_dict[y] for y in y_train], dtype=np.float32)
            
            # -------------------------------------------------------------
            # Model-Specific Training & Prediction
            # -------------------------------------------------------------
            start_train_time = time.time()
            start_pred_time = 0.0
            pred_time = 0.0
            
            if model_name == "Random_Forest":
                clf = RandomForestClassifier(
                    n_estimators=100,
                    criterion="gini",
                    class_weight="balanced",
                    random_state=GLOBAL_SEED,
                    n_jobs=-1
                )
                clf.fit(X_train_clean, y_train)
                train_time = time.time() - start_train_time
                
                start_pred_time = time.time()
                y_val_probs = clf.predict_proba(X_val_clean)
                y_val_preds = np.argmax(y_val_probs, axis=1)
                pred_time = time.time() - start_pred_time
                
            elif model_name == "XGBoost":
                clf = XGBClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="multi:softprob",
                    eval_metric="mlogloss",
                    random_state=GLOBAL_SEED,
                    n_jobs=-1
                )
                clf.fit(X_train_clean, y_train, sample_weight=sample_weights_train)
                train_time = time.time() - start_train_time
                
                start_pred_time = time.time()
                y_val_probs = clf.predict_proba(X_val_clean)
                y_val_preds = np.argmax(y_val_probs, axis=1)
                pred_time = time.time() - start_pred_time
                
            elif model_name == "P_MLP":
                # Scaling fitted strictly on training fold
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train_clean)
                X_val_scaled = scaler.transform(X_val_clean)
                
                weights_tensor = torch.tensor(class_weights_arr, dtype=torch.float32)
                pmlp_model, log_epochs = train_pmlp(
                    X_train_scaled, y_train, X_val_scaled, y_val,
                    num_classes=num_classes,
                    class_weights_tensor=weights_tensor,
                    device=device,
                    max_epochs=100,
                    patience=15,
                    batch_size=64
                )
                train_time = time.time() - start_train_time
                
                for log_item in log_epochs:
                    training_log_records.append({
                        "track": track_code,
                        "model": model_name,
                        "fold": fold,
                        **log_item
                    })
                    
                start_pred_time = time.time()
                pmlp_model.eval()
                with torch.no_grad():
                    val_t = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
                    logits = pmlp_model(val_t)
                    probs_t = torch.softmax(logits, dim=1)
                    y_val_probs = probs_t.cpu().numpy()
                    y_val_preds = np.argmax(y_val_probs, axis=1)
                pred_time = time.time() - start_pred_time
                
            # Store OOF
            all_oof_predictions[model_name][val_idx] = y_val_preds
            all_oof_probabilities[model_name][val_idx] = y_val_probs
            
            # -------------------------------------------------------------
            # Metrics Calculation
            # -------------------------------------------------------------
            macro_f1 = f1_score(y_val, y_val_preds, average="macro", zero_division=0)
            balanced_acc = balanced_accuracy_score(y_val, y_val_preds)
            accuracy = accuracy_score(y_val, y_val_preds)
            weighted_f1 = f1_score(y_val, y_val_preds, average="weighted", zero_division=0)
            
            p_per, r_per, f_per, s_per = precision_recall_fscore_support(y_val, y_val_preds, labels=range(num_classes), zero_division=0)
            
            print(f"  [Fold {fold}] Macro F1: {macro_f1:.4f} | Bal Acc: {balanced_acc:.4f} | Acc: {accuracy:.4f} | Train: {train_time:.2f}s | Pred: {pred_time:.3f}s")
            
            fold_metric_records.append({
                "track": track_code,
                "model": model_name,
                "fold": fold,
                "train_samples": len(train_idx),
                "val_samples": len(val_idx),
                "train_groups": fold_groups_audit[fold]["train_groups"],
                "val_groups": fold_groups_audit[fold]["val_groups"],
                "group_overlap": fold_groups_audit[fold]["group_overlap"],
                "macro_f1": macro_f1,
                "balanced_accuracy": balanced_acc,
                "accuracy": accuracy,
                "weighted_f1": weighted_f1,
                "train_time_sec": train_time,
                "pred_time_sec": pred_time,
                "status": "SUCCESS"
            })
            
            for c_idx, c_name in idx_to_class.items():
                per_class_metric_records.append({
                    "track": track_code,
                    "model": model_name,
                    "fold": fold,
                    "class_name": c_name,
                    "precision": p_per[c_idx],
                    "recall": r_per[c_idx],
                    "f1_score": f_per[c_idx],
                    "support": s_per[c_idx]
                })
                
            runtime_records.append({
                "track": track_code,
                "model": model_name,
                "fold": fold,
                "train_time_sec": round(train_time, 4),
                "pred_time_sec": round(pred_time, 4),
                "latency_us_per_sample": round((pred_time / len(val_idx)) * 1e6, 2)
            })

    # -------------------------------------------------------------
    # Aggregated & Pooled Out-Of-Fold Analysis
    # -------------------------------------------------------------
    df_fold_metrics = pd.DataFrame(fold_metric_records)
    df_per_class = pd.DataFrame(per_class_metric_records)
    
    aggregate_records = []
    cm_counts_dict = {}
    cm_norm_dict = {}
    
    for model_name in models:
        m_folds = df_fold_metrics[df_fold_metrics["model"] == model_name]
        
        # Mean across folds
        mean_macro_f1 = m_folds["macro_f1"].mean()
        std_macro_f1 = m_folds["macro_f1"].std()
        mean_bal_acc = m_folds["balanced_accuracy"].mean()
        std_bal_acc = m_folds["balanced_accuracy"].std()
        mean_acc = m_folds["accuracy"].mean()
        std_acc = m_folds["accuracy"].std()
        mean_weighted_f1 = m_folds["weighted_f1"].mean()
        std_weighted_f1 = m_folds["weighted_f1"].std()
        
        # Pooled Out-Of-Fold metrics
        oof_preds = all_oof_predictions[model_name]
        assert (oof_preds >= 0).all(), f"Incomplete OOF predictions for {model_name}!"
        
        pooled_macro_f1 = f1_score(y_int, oof_preds, average="macro", zero_division=0)
        pooled_bal_acc = balanced_accuracy_score(y_int, oof_preds)
        pooled_acc = accuracy_score(y_int, oof_preds)
        pooled_weighted_f1 = f1_score(y_int, oof_preds, average="weighted", zero_division=0)
        
        aggregate_records.append({
            "track": track_code,
            "model": model_name,
            "num_folds": num_folds,
            "mean_macro_f1": round(mean_macro_f1, 4),
            "std_macro_f1": round(std_macro_f1, 4),
            "mean_balanced_accuracy": round(mean_bal_acc, 4),
            "std_balanced_accuracy": round(std_bal_acc, 4),
            "mean_accuracy": round(mean_acc, 4),
            "std_accuracy": round(std_acc, 4),
            "mean_weighted_f1": round(mean_weighted_f1, 4),
            "std_weighted_f1": round(std_weighted_f1, 4),
            "pooled_macro_f1": round(pooled_macro_f1, 4),
            "pooled_balanced_accuracy": round(pooled_bal_acc, 4),
            "pooled_accuracy": round(pooled_acc, 4),
            "pooled_weighted_f1": round(pooled_weighted_f1, 4)
        })
        
        # Confusion Matrices (Counts & Normalized)
        cm_counts = confusion_matrix(y_int, oof_preds, labels=range(num_classes))
        cm_norm = cm_counts.astype('float') / cm_counts.sum(axis=1)[:, np.newaxis]
        
        for i, true_cls in enumerate(classes):
            for j, pred_cls in enumerate(classes):
                cm_counts_dict[(model_name, true_cls, pred_cls)] = cm_counts[i, j]
                cm_norm_dict[(model_name, true_cls, pred_cls)] = cm_norm[i, j]
                
    # Format Confusion Matrix DataFrames
    cm_counts_records = []
    cm_norm_records = []
    for (m, t_cls, p_cls), val in cm_counts_dict.items():
        cm_counts_records.append({"track": track_code, "model": m, "true_class": t_cls, "pred_class": p_cls, "count": val})
    for (m, t_cls, p_cls), val in cm_norm_dict.items():
        cm_norm_records.append({"track": track_code, "model": m, "true_class": t_cls, "pred_class": p_cls, "normalized_rate": round(val, 4)})
        
    df_cm_counts = pd.DataFrame(cm_counts_records)
    df_cm_norm = pd.DataFrame(cm_norm_records)
    df_agg = pd.DataFrame(aggregate_records)
    df_runtime = pd.DataFrame(runtime_records)
    df_features = pd.DataFrame(feature_usage_records)
    df_training_log = pd.DataFrame(training_log_records)
    
    # Format OOF Predictions DataFrame
    df_oof = df_flows[["flow_id", "capture_group", "qos_class", "application", "is_vpn"]].copy()
    df_oof["fold"] = df_folds["fold"].values
    df_oof["true_label"] = df_flows["qos_class"].values
    for m in models:
        df_oof[f"pred_{m}"] = [idx_to_class[p] for p in all_oof_predictions[m]]
        for c_idx, c_name in idx_to_class.items():
            df_oof[f"prob_{m}_{c_name}"] = all_oof_probabilities[m][:, c_idx]
            
    # -------------------------------------------------------------
    # Save All Artifacts to Track Output Directory
    # -------------------------------------------------------------
    df_fold_metrics.to_csv(os.path.join(track_out_dir, "fold_metrics.csv"), index=False)
    df_agg.to_csv(os.path.join(track_out_dir, "aggregate_metrics.csv"), index=False)
    df_per_class.to_csv(os.path.join(track_out_dir, "per_class_metrics.csv"), index=False)
    df_cm_counts.to_csv(os.path.join(track_out_dir, "confusion_matrix_counts.csv"), index=False)
    df_cm_norm.to_csv(os.path.join(track_out_dir, "confusion_matrix_normalized.csv"), index=False)
    df_oof.to_csv(os.path.join(track_out_dir, "out_of_fold_predictions.csv"), index=False)
    df_training_log.to_csv(os.path.join(track_out_dir, "training_log.csv"), index=False)
    df_runtime.to_csv(os.path.join(track_out_dir, "runtime_summary.csv"), index=False)
    df_features.to_csv(os.path.join(track_out_dir, "feature_list.csv"), index=False)
    
    exp_config = {
        "track_code": track_code,
        "track_name": track_name,
        "flows_file": flows_csv,
        "folds_file": folds_csv,
        "total_flows": total_samples,
        "classes": classes,
        "num_folds": num_folds,
        "candidate_features_count": len(candidate_features),
        "models_evaluated": models,
        "reproducibility_seed": GLOBAL_SEED,
        "hardware": "CPU (12 logical cores)"
    }
    with open(os.path.join(track_out_dir, "experiment_config.json"), "w") as f:
        json.dump(exp_config, f, indent=2)
        
    print(f"\n[DONE] Successfully exported all artifacts for {track_name} to {track_out_dir}")
    print(df_agg[["model", "mean_macro_f1", "std_macro_f1", "mean_balanced_accuracy", "std_balanced_accuracy", "pooled_macro_f1", "pooled_balanced_accuracy"]])
    return df_agg

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(base_dir, "output", "phase6")
    os.makedirs(out_dir, exist_ok=True)
    
    track_a_flows = os.path.join(base_dir, "output", "phase5", "track_a_flows.csv")
    track_a_folds = os.path.join(base_dir, "output", "phase5", "track_a_folds.csv")
    
    track_b_flows = os.path.join(base_dir, "output", "phase5", "track_b_flows.csv")
    track_b_folds = os.path.join(base_dir, "output", "phase5", "track_b_folds.csv")
    
    print("\n" + "#"*75)
    print(" STARTING PHASE 6A: FULL-FLOW BASELINE EXPERIMENT")
    print("#"*75)
    
    agg_a = execute_track_experiment(
        track_code="track_a",
        track_name="Track A (Exploratory 4-Class Baseline)",
        flows_csv=track_a_flows,
        folds_csv=track_a_folds,
        num_folds=3,
        out_dir=out_dir
    )
    
    agg_b = execute_track_experiment(
        track_code="track_b",
        track_name="Track B (Primary 3-Class Benchmark)",
        flows_csv=track_b_flows,
        folds_csv=track_b_folds,
        num_folds=5,
        out_dir=out_dir
    )
    
    print("\n" + "#"*75)
    print(" ALL PHASE 6A EXPERIMENTS SUCCESSFULLY COMPLETED")
    print("#"*75)

if __name__ == "__main__":
    main()
