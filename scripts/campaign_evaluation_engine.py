#!/usr/bin/env python3
"""
scripts/campaign_evaluation_engine.py

Core Leak-Free Evaluation & Benchmarking Infrastructure for the Model Improvement Campaign.
Enforces strict Leave-One-Group-Out (29 folds) evaluation, zero test leakage,
consistent causal early-5 features, and rigorous statistical tests.
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
from sklearn.preprocessing import StandardScaler

DATASET_PATH = Path("output/phase6d/clean_track_b_flows.csv")
OUTPUT_DIR = Path("output/model_improvement_campaign")

def load_clean_dataset():
    """Load and verify clean Track B dataset (4,196 flows, 29 groups)."""
    assert DATASET_PATH.exists(), f"Dataset not found at {DATASET_PATH}"
    df = pd.read_csv(DATASET_PATH)
    assert len(df) == 4196, f"Expected 4196 flows, found {len(df)}"
    assert df["capture_group"].nunique() == 29, f"Expected 29 groups, found {df['capture_group'].nunique()}"
    
    # Standard class encoding
    classes = sorted(df["qos_class"].unique()) # ['Bulk Data', 'Real-Time', 'Streaming']
    cls2idx = {c: i for i, c in enumerate(classes)}
    idx2cls = {i: c for i, c in enumerate(classes)}
    y = df["qos_class"].map(cls2idx).values
    groups = df["capture_group"].values
    
    # Build causal features
    df["is_udp"] = (df["protocol"] == 17).astype(float)
    total_early = df["early_5_fwd_pkts"] + df["early_5_bwd_pkts"]
    df["early_5_fwd_ratio"] = df["early_5_fwd_pkts"] / np.maximum(total_early, 1)
    
    # Additional strictly causal early-5 candidates (engineered without future info)
    df["early_5_is_bidir"] = ((df["early_5_fwd_pkts"] > 0) & (df["early_5_bwd_pkts"] > 0)).astype(float)
    df["early_5_bwd_fwd_diff"] = (df["early_5_fwd_pkts"] - df["early_5_bwd_pkts"]).astype(float)
    df["early_5_log_bytes"] = np.log1p(df["early_5_bytes"].clip(lower=0)).astype(float)
    df["early_5_log_duration"] = np.log1p(df["early_5_duration"].clip(lower=0)).astype(float)
    df["early_5_bwd_ratio"] = df["early_5_bwd_pkts"] / np.maximum(total_early, 1)
    df["early_5_pkt_count"] = np.minimum(total_early, 5).astype(float)
    
    return df, y, groups, classes, cls2idx, idx2cls

# Feature subsets
FEATS_BASE = [
    "early_5_complete", "early_5_duration", "early_5_bytes",
    "early_5_fwd_pkts", "early_5_bwd_pkts", "early_5_avg_size"
]

FEATS_ROBUST8 = FEATS_BASE + ["is_udp", "early_5_fwd_ratio"]

FEATS_EXTENDED12 = FEATS_ROBUST8 + [
    "early_5_is_bidir", "early_5_bwd_fwd_diff", "early_5_log_bytes", "early_5_log_duration"
]

FEATS_ALL_CAUSAL14 = FEATS_EXTENDED12 + [
    "early_5_bwd_ratio", "early_5_pkt_count"
]

def standard_weights(y_tr, num_classes=3):
    counts = np.bincount(y_tr, minlength=num_classes)
    return len(y_tr) / (num_classes * np.maximum(counts[y_tr], 1))

def bulk_boost_weights(y_tr, num_classes=3, boost=1.20):
    counts = np.bincount(y_tr, minlength=num_classes)
    w = len(y_tr) / (num_classes * np.maximum(counts, 1))
    w[0] *= boost # Bulk Data is index 0
    return w[y_tr]

def compute_mcnemar_test(y_true, y_base, y_cand):
    """Computes exact McNemar test p-value."""
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

def evaluate_model_logo(
    X, y, groups, model_fn,
    weight_fn=None,
    scale_features=False,
    num_classes=3
):
    """
    Executes full 29-fold Leave-One-Group-Out cross validation.
    Guarantees no test group data is used in scaling or model fitting.
    """
    logo = LeaveOneGroupOut()
    oof_preds = np.zeros(len(y), dtype=int)
    oof_probs = np.zeros((len(y), num_classes), dtype=float)
    per_group_f1s = []
    
    t0 = time.time()
    for tr, te in logo.split(X, y, groups):
        X_tr, y_tr = X[tr].copy(), y[tr]
        X_te, y_te = X[te].copy(), y[te]
        
        if scale_features:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_te = scaler.transform(X_te)
            
        clf = model_fn()
        sw = weight_fn(y_tr, num_classes) if weight_fn else None
        
        if sw is not None:
            try:
                clf.fit(X_tr, y_tr, sample_weight=sw)
            except TypeError:
                clf.fit(X_tr, y_tr)
        else:
            clf.fit(X_tr, y_tr)
            
        preds = np.asarray(clf.predict(X_te)).ravel()
        oof_preds[te] = preds
        if hasattr(clf, "predict_proba"):
            try:
                oof_probs[te] = clf.predict_proba(X_te)
            except Exception:
                pass
                
        g_f1 = f1_score(y_te, preds, average="macro", zero_division=0)
        per_group_f1s.append(g_f1)
        
    t_elapsed = time.time() - t0
    
    macro_f1 = float(f1_score(y, oof_preds, average="macro", zero_division=0))
    bal_acc = float(balanced_accuracy_score(y, oof_preds))
    acc = float(accuracy_score(y, oof_preds))
    prec, rec, f1s, supp = precision_recall_fscore_support(y, oof_preds, labels=range(num_classes), zero_division=0)
    cm = confusion_matrix(y, oof_preds, normalize="true")
    cm_counts = confusion_matrix(y, oof_preds)
    
    return {
        "macro_f1": round(macro_f1, 4),
        "balanced_acc": round(bal_acc, 4),
        "overall_acc": round(acc, 4),
        "per_class_f1": [round(float(f), 4) for f in f1s],
        "per_class_rec": [round(float(r), 4) for r in rec],
        "per_class_prec": [round(float(p), 4) for p in prec],
        "per_class_supp": [int(s) for s in supp],
        "confusion_matrix": cm.tolist(),
        "confusion_counts": cm_counts.tolist(),
        "per_group_f1s": [round(float(f), 4) for f in per_group_f1s],
        "oof_preds": oof_preds,
        "oof_probs": oof_probs,
        "runtime_sec": round(t_elapsed, 2)
    }
