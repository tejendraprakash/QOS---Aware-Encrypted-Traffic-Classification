#!/usr/bin/env python3
"""
scripts/adv_campaign_engine.py
Core Leak-Free Infrastructure for the Advanced Optimization Campaign.
Enforces:
1. Strict 29-fold Leave-One-Group-Out (LOGO) outer evaluation.
2. Inner group-aware cross-validation (StratifiedGroupKFold on training groups) for hyperparameter selection.
3. In-fold scaling (StandardScaler, RobustScaler, MinMaxScaler) with zero outer-test leakage.
4. Exact metric tracking: Macro F1, Balanced Accuracy, Overall Accuracy, Per-Class Precision/Recall/F1.
5. Out-of-fold probability caching for stacking and soft-voting ensembles.
"""

import sys
import os
os.environ['LOKY_MAX_CPU_COUNT'] = '4'
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold
from sklearn.metrics import (
    f1_score,
    balanced_accuracy_score,
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

DATASET_PATH = Path("output/phase6d/clean_track_b_flows.csv")
OUTPUT_DIR = Path("output/advanced_optimization_campaign")

def load_advanced_dataset():
    """Load and verify clean Track B dataset (4,196 flows, 29 groups)."""
    assert DATASET_PATH.exists(), f"Dataset not found at {DATASET_PATH}"
    df = pd.read_csv(DATASET_PATH)
    assert len(df) == 4196, f"Expected 4196 flows, found {len(df)}"
    assert df["capture_group"].nunique() == 29, f"Expected 29 groups, found {df['capture_group'].nunique()}"
    
    classes = sorted(df["qos_class"].unique()) # ['Bulk Data', 'Real-Time', 'Streaming']
    cls2idx = {c: i for i, c in enumerate(classes)}
    idx2cls = {i: c for i, c in enumerate(classes)}
    y = df["qos_class"].map(cls2idx).values
    groups = df["capture_group"].values
    
    # Base causal features
    df["is_udp"] = (df["protocol"] == 17).astype(float)
    total_early = df["early_5_fwd_pkts"] + df["early_5_bwd_pkts"]
    df["early_5_fwd_ratio"] = df["early_5_fwd_pkts"] / np.maximum(total_early, 1)
    
    # Causal extensions (first 5 packets strictly)
    df["early_5_is_bidir"] = ((df["early_5_fwd_pkts"] > 0) & (df["early_5_bwd_pkts"] > 0)).astype(float)
    df["early_5_bwd_fwd_diff"] = (df["early_5_fwd_pkts"] - df["early_5_bwd_pkts"]).astype(float)
    df["early_5_bwd_ratio"] = df["early_5_bwd_pkts"] / np.maximum(total_early, 1)
    df["early_5_pkt_count"] = np.minimum(total_early, 5).astype(float)
    df["early_5_log_bytes"] = np.log1p(df["early_5_bytes"].clip(lower=0)).astype(float)
    df["early_5_log_duration"] = np.log1p(df["early_5_duration"].clip(lower=0)).astype(float)
    df["early_5_byte_rate"] = df["early_5_bytes"] / (df["early_5_duration"] + 1e-4)
    df["early_5_pkt_rate"] = total_early / (df["early_5_duration"] + 1e-4)
    df["early_5_is_pure_fwd"] = (df["early_5_bwd_pkts"] == 0).astype(float)
    df["early_5_is_pure_bwd"] = (df["early_5_fwd_pkts"] == 0).astype(float)
    df["early_5_burst_indicator"] = (df["early_5_duration"] < 0.005).astype(float)
    df["early_5_fwd_to_bwd_ratio"] = df["early_5_fwd_pkts"] / np.maximum(df["early_5_bwd_pkts"], 1)
    df["early_5_bytes_per_fwd"] = df["early_5_bytes"] / np.maximum(df["early_5_fwd_pkts"], 1)
    
    return df, y, groups, classes, cls2idx, idx2cls

# Canonical feature sets
FEATS_BASE6 = [
    "early_5_complete", "early_5_duration", "early_5_bytes",
    "early_5_fwd_pkts", "early_5_bwd_pkts", "early_5_avg_size"
]
FEATS_CHAMPION8 = FEATS_BASE6 + ["is_udp", "early_5_fwd_ratio"]
FEATS_CAUSAL9 = FEATS_CHAMPION8 + ["early_5_is_bidir"]
FEATS_EXTENDED14 = FEATS_CAUSAL9 + [
    "early_5_bwd_fwd_diff", "early_5_log_bytes", "early_5_log_duration",
    "early_5_bwd_ratio", "early_5_pkt_count"
]
FEATS_ALL_CAUSAL18 = FEATS_EXTENDED14 + [
    "early_5_byte_rate", "early_5_is_pure_fwd", "early_5_burst_indicator", "early_5_bytes_per_fwd"
]

def standard_weights(y_tr, num_classes=3):
    counts = np.bincount(y_tr, minlength=num_classes)
    return len(y_tr) / (num_classes * np.maximum(counts[y_tr], 1))

def bulk_boost_weights(y_tr, num_classes=3, boost=1.15):
    counts = np.bincount(y_tr, minlength=num_classes)
    w = len(y_tr) / (num_classes * np.maximum(counts, 1))
    w[0] *= boost
    return w[y_tr]

def compute_mcnemar_test(y_true, y_base, y_cand):
    c_base = (y_base == y_true)
    c_cand = (y_cand == y_true)
    n01 = int(np.sum(~c_base & c_cand))
    n10 = int(np.sum(c_base & ~c_cand))
    n_disc = n01 + n10
    if n_disc == 0:
        return 1.0, 0.0
    chi2 = (abs(n01 - n10) - 1.0)**2 / n_disc
    p_exact = float(stats.binomtest(min(n01, n10), n_disc, 0.5, alternative="two-sided").pvalue)
    return p_exact, float(chi2)

def inner_group_cv_score(X_tr, y_tr, groups_tr, model_factory, n_splits=4, scale_fn=None):
    """
    Evaluates candidate model strictly on training data using StratifiedGroupKFold.
    Zero leakage to outer test group.
    """
    sgkf = StratifiedGroupKFold(n_splits=n_splits)
    f1_scores = []
    for in_tr, in_val in sgkf.split(X_tr, y_tr, groups_tr):
        X_in_tr, y_in_tr = X_tr[in_tr], y_tr[in_tr]
        X_in_val, y_in_val = X_tr[in_val], y_tr[in_val]
        
        if scale_fn is not None:
            scaler = scale_fn()
            X_in_tr = scaler.fit_transform(X_in_tr)
            X_in_val = scaler.transform(X_in_val)
            
        clf = model_factory()
        clf.fit(X_in_tr, y_in_tr)
        preds = np.asarray(clf.predict(X_in_val)).ravel()
        f1_scores.append(f1_score(y_in_val, preds, average="macro"))
    return float(np.mean(f1_scores))

def evaluate_adv_model_logo(
    X, y, groups, model_fn,
    weight_fn=None,
    scaler_cls=None,
    num_classes=3,
    save_prefix=None
):
    """
    Strict 29-fold Leave-One-Group-Out cross-validation.
    Ensures in-fold scaling and zero leakage.
    Returns comprehensive metrics dictionary.
    """
    logo = LeaveOneGroupOut()
    oof_preds = np.zeros(len(y), dtype=int)
    oof_probs = np.zeros((len(y), num_classes), dtype=float)
    per_group_f1s = []
    per_group_accs = []
    
    t0 = time.time()
    for tr, te in logo.split(X, y, groups):
        X_tr, y_tr = X[tr].copy(), y[tr]
        X_te, y_te = X[te].copy(), y[te]
        
        if scaler_cls is not None:
            scaler = scaler_cls()
            X_tr = scaler.fit_transform(X_tr)
            X_te = scaler.transform(X_te)
            
        clf = model_fn()
        
        # Fit model
        if weight_fn is not None:
            sample_w = weight_fn(y_tr, num_classes=num_classes)
            clf.fit(X_tr, y_tr, sample_weight=sample_w)
        else:
            clf.fit(X_tr, y_tr)
            
        # Predict labels
        preds = np.asarray(clf.predict(X_te)).ravel()
        oof_preds[te] = preds
        
        # Predict probabilities if supported
        if hasattr(clf, "predict_proba"):
            try:
                probs = clf.predict_proba(X_te)
                if probs.shape[1] == num_classes:
                    oof_probs[te] = probs
            except Exception:
                pass
        elif hasattr(clf, "decision_function"):
            try:
                dec = clf.decision_function(X_te)
                # Softmax of decision function
                exp_d = np.exp(dec - np.max(dec, axis=1, keepdims=True))
                oof_probs[te] = exp_d / np.sum(exp_d, axis=1, keepdims=True)
            except Exception:
                pass
                
        # Group-level metrics
        if len(np.unique(y_te)) > 1:
            grp_f1 = f1_score(y_te, preds, average="macro", zero_division=0)
        else:
            grp_f1 = float(np.mean(preds == y_te))
        per_group_f1s.append(grp_f1)
        per_group_accs.append(float(np.mean(preds == y_te)))
        
    elapsed = round(time.time() - t0, 2)
    
    # Calculate pooled out-of-fold metrics
    macro_f1 = float(f1_score(y, oof_preds, average="macro"))
    bal_acc = float(balanced_accuracy_score(y, oof_preds))
    overall_acc = float(accuracy_score(y, oof_preds))
    
    prec, rec, f1, _ = precision_recall_fscore_support(y, oof_preds, average=None, zero_division=0)
    cm = confusion_matrix(y, oof_preds, normalize="true").tolist()
    
    # Save OOF files if requested
    oof_pred_file = ""
    oof_prob_file = ""
    if save_prefix:
        pred_path = OUTPUT_DIR / "oof_predictions" / f"oof_pred_{save_prefix}.npy"
        prob_path = OUTPUT_DIR / "oof_predictions" / f"oof_prob_{save_prefix}.npy"
        np.save(pred_path, oof_preds)
        np.save(prob_path, oof_probs)
        oof_pred_file = str(pred_path.name)
        oof_prob_file = str(prob_path.name)
        
    return {
        "macro_f1": round(macro_f1, 4),
        "balanced_acc": round(bal_acc, 4),
        "overall_acc": round(overall_acc, 4),
        "per_class_f1": [round(float(v), 4) for v in f1],
        "per_class_rec": [round(float(v), 4) for v in rec],
        "per_class_prec": [round(float(v), 4) for v in prec],
        "confusion_matrix": cm,
        "per_group_f1s": [round(float(v), 4) for v in per_group_f1s],
        "per_group_accs": [round(float(v), 4) for v in per_group_accs],
        "runtime_sec": elapsed,
        "oof_pred_file": oof_pred_file,
        "oof_prob_file": oof_prob_file
    }
