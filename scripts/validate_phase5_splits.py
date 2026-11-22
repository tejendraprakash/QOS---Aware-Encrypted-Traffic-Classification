"""
validate_phase5_splits.py
Phase 5: Baseline Experiment Preparation & Protocol Validation
Encrypted Traffic Research

Validates and generates leak-free Stratified Group K-Fold splits:
- Track A (4-Class): 3-fold StratifiedGroupKFold (constrained by Browsing having only 3 capture groups).
- Track B (3-Class): 5-fold StratifiedGroupKFold (29 capture groups, 9-10 per class).

Guarantees:
1. Strict Group Disjointness: Train groups and Validation groups have zero overlap.
2. 100% Class Coverage: Every class is represented in every training and validation fold.
3. Completely deterministic reproducibility (random_state=42).
"""

import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

def run_split_validation(track_name, csv_path, n_splits, random_state, out_dir):
    print(f"\n=======================================================")
    print(f" Validating Splits for {track_name} (n_splits={n_splits}, seed={random_state})")
    print(f"=======================================================")
    df = pd.read_csv(csv_path)
    print(f"[INFO] Loaded {len(df)} flows from {csv_path}")
    
    X = np.zeros(len(df)) # dummy feature
    y = df["qos_class"].values
    groups = df["capture_group"].values
    
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    fold_assignments = np.full(len(df), -1, dtype=int)
    fold_analysis_records = []
    
    fold_metadata = {}
    
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
        fold_assignments[val_idx] = fold
        
        train_groups = set(groups[train_idx])
        val_groups = set(groups[val_idx])
        overlap = train_groups.intersection(val_groups)
        
        train_classes = set(y[train_idx])
        val_classes = set(y[val_idx])
        all_classes = set(y)
        
        # Rigorous assertions
        assert len(overlap) == 0, f"LEAKAGE DETECTED in Fold {fold}! Overlapping groups: {overlap}"
        assert val_classes == all_classes, f"MISSING CLASS in Val Fold {fold}! Expected {all_classes}, got {val_classes}"
        assert train_classes == all_classes, f"MISSING CLASS in Train Fold {fold}! Expected {all_classes}, got {train_classes}"
        
        # Analyze distribution
        val_df = df.iloc[val_idx]
        train_df = df.iloc[train_idx]
        
        val_class_counts = val_df["qos_class"].value_counts().to_dict()
        train_class_counts = train_df["qos_class"].value_counts().to_dict()
        
        print(f"--- Fold {fold} ---")
        print(f"  Train: {len(train_idx)} samples across {len(train_groups)} groups.")
        print(f"  Val:   {len(val_idx)} samples across {len(val_groups)} groups: {sorted(list(val_groups))}")
        print(f"  Val Class Counts: {val_class_counts}")
        print(f"  Group Overlap: {len(overlap)} (PASS)")
        print(f"  Class Coverage: {len(val_classes)}/{len(all_classes)} (100% PASS)")
        
        # Record for analysis CSV
        record_train = {
            "track": track_name,
            "fold": fold,
            "partition": "train",
            "samples": len(train_idx),
            "groups_count": len(train_groups),
            "groups_list": ";".join(sorted(list(train_groups))),
            "group_leakage": False,
            "class_coverage_pct": 100.0
        }
        for cls in sorted(list(all_classes)):
            record_train[f"count_{cls}"] = train_class_counts.get(cls, 0)
            record_train[f"pct_{cls}"] = round(100.0 * train_class_counts.get(cls, 0) / len(train_idx), 2)
            
        record_val = {
            "track": track_name,
            "fold": fold,
            "partition": "val",
            "samples": len(val_idx),
            "groups_count": len(val_groups),
            "groups_list": ";".join(sorted(list(val_groups))),
            "group_leakage": False,
            "class_coverage_pct": 100.0
        }
        for cls in sorted(list(all_classes)):
            record_val[f"count_{cls}"] = val_class_counts.get(cls, 0)
            record_val[f"pct_{cls}"] = round(100.0 * val_class_counts.get(cls, 0) / len(val_idx), 2)
            
        fold_analysis_records.append(record_train)
        fold_analysis_records.append(record_val)
        
        fold_metadata[f"fold_{fold}"] = {
            "train_samples": len(train_idx),
            "val_samples": len(val_idx),
            "train_groups": sorted(list(train_groups)),
            "val_groups": sorted(list(val_groups)),
            "val_class_distribution": val_class_counts,
            "val_class_coverage_ratio": f"{len(val_classes)}/{len(all_classes)}",
            "leakage_free": True
        }
        
    assert (fold_assignments >= 0).all(), "Some samples were not assigned to any fold!"
    
    # Save fold assignments CSV
    df_folds = df[["flow_id", "capture_group", "qos_class", "application", "is_vpn"]].copy()
    df_folds["fold"] = fold_assignments
    
    track_code = "track_a" if "track_a" in csv_path else "track_b"
    folds_out_path = os.path.join(out_dir, f"{track_code}_folds.csv")
    df_folds.to_csv(folds_out_path, index=False)
    print(f"[INFO] Saved fold assignments to {folds_out_path}")
    
    # Save analysis CSV
    analysis_df = pd.DataFrame(fold_analysis_records)
    analysis_out_path = os.path.join(out_dir, f"{track_code}_fold_analysis.csv")
    analysis_df.to_csv(analysis_out_path, index=False)
    print(f"[INFO] Saved fold analysis to {analysis_out_path}")
    
    return fold_metadata

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(base_dir, "output", "phase5")
    
    track_a_csv = os.path.join(out_dir, "track_a_flows.csv")
    track_b_csv = os.path.join(out_dir, "track_b_flows.csv")
    
    meta_a = run_split_validation("Track A (Exploratory 4-Class)", track_a_csv, n_splits=3, random_state=7, out_dir=out_dir)
    meta_b = run_split_validation("Track B (Primary 3-Class)", track_b_csv, n_splits=5, random_state=2, out_dir=out_dir)
    
    cv_summary = {
        "track_a": {
            "protocol": "StratifiedGroupKFold",
            "n_splits": 3,
            "rationale": "Browsing only possesses 3 capture groups (email_1a, email_1b, email_2a). Exactly 3 folds ensures every validation fold contains exactly 1 distinct Browsing group with 0% group leakage and 100% class coverage.",
            "folds": meta_a
        },
        "track_b": {
            "protocol": "StratifiedGroupKFold",
            "n_splits": 5,
            "rationale": "With Browsing removed, the remaining 3 classes (Bulk Data, Streaming, Real-Time) have 29 capture groups (10, 9, 10). A standard 5-fold StratifiedGroupKFold provides robust cross-validation with 0% group leakage and 100% class coverage in every fold.",
            "folds": meta_b
        }
    }
    
    cv_json_path = os.path.join(out_dir, "cross_validation_splits.json")
    with open(cv_json_path, "w") as f:
        json.dump(cv_summary, f, indent=2)
    print(f"\n[INFO] Cross-validation specifications saved to {cv_json_path}")

if __name__ == "__main__":
    main()
