"""
prepare_phase5_datasets.py
Phase 5: Baseline Experiment Preparation & Protocol Validation
Encrypted Traffic Research

This script prepares the validated datasets for:
- Track A (Exploratory 4-Class): Bulk Data, Browsing, Streaming, Real-Time (17,617 flows)
- Track B (Primary 3-Class): Bulk Data, Streaming, Real-Time (10,770 flows)

Original datasets are strictly preserved without modification.
"""

import os
import json
import pandas as pd
import numpy as np

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(base_dir, "output", "flow_features.csv")
    out_dir = os.path.join(base_dir, "output", "phase5")
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"[INFO] Reading base dataset: {input_path}")
    df = pd.read_csv(input_path)
    print(f"[INFO] Full dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
    assert df.shape == (17617, 117), f"Expected (17617, 117), got {df.shape}"
    
    # -------------------------------------------------------------
    # 1. Track A: Exploratory 4-Class Dataset
    # -------------------------------------------------------------
    track_a_path = os.path.join(out_dir, "track_a_flows.csv")
    print(f"[INFO] Exporting Track A dataset to {track_a_path}...")
    df.to_csv(track_a_path, index=False)
    
    track_a_class_dist = df["qos_class"].value_counts().to_dict()
    track_a_groups = df["capture_group"].nunique()
    print(f"[INFO] Track A Classes: {track_a_class_dist}")
    print(f"[INFO] Track A Capture Groups: {track_a_groups}")
    
    # -------------------------------------------------------------
    # 2. Track B: Primary 3-Class Dataset (Exclude Browsing / Email)
    # -------------------------------------------------------------
    track_b_df = df[df["qos_class"] != "Browsing"].copy()
    track_b_path = os.path.join(out_dir, "track_b_flows.csv")
    print(f"[INFO] Exporting Track B dataset to {track_b_path}...")
    track_b_df.to_csv(track_b_path, index=False)
    
    track_b_class_dist = track_b_df["qos_class"].value_counts().to_dict()
    track_b_groups = track_b_df["capture_group"].nunique()
    print(f"[INFO] Track B Classes: {track_b_class_dist}")
    print(f"[INFO] Track B Total Flows: {len(track_b_df)}")
    print(f"[INFO] Track B Capture Groups: {track_b_groups}")
    assert len(track_b_df) == 10770, f"Expected 10770 flows, got {len(track_b_df)}"
    assert track_b_groups == 29, f"Expected 29 capture groups, got {track_b_groups}"
    
    # -------------------------------------------------------------
    # 3. Export Summary Metadata
    # -------------------------------------------------------------
    summary = {
        "source_file": "output/flow_features.csv",
        "total_source_flows": len(df),
        "total_columns": df.shape[1],
        "track_a": {
            "name": "Track A - Exploratory 4-Class",
            "description": "Exploratory baseline containing Bulk Data, Browsing, Streaming, Real-Time. Browsing is derived from Email captures and contains background noise (LLMNR/NetBIOS).",
            "file": "output/phase5/track_a_flows.csv",
            "total_flows": len(df),
            "class_distribution": track_a_class_dist,
            "unique_capture_groups": track_a_groups,
            "imbalance_ratio": float(max(track_a_class_dist.values()) / min(track_a_class_dist.values())),
            "vpn_distribution": df["is_vpn"].value_counts().to_dict()
        },
        "track_b": {
            "name": "Track B - Primary 3-Class",
            "description": "Primary benchmark baseline excluding contaminated Browsing/Email. Classes: Bulk Data, Streaming, Real-Time.",
            "file": "output/phase5/track_b_flows.csv",
            "total_flows": len(track_b_df),
            "class_distribution": track_b_class_dist,
            "unique_capture_groups": track_b_groups,
            "imbalance_ratio": float(max(track_b_class_dist.values()) / min(track_b_class_dist.values())),
            "vpn_distribution": track_b_df["is_vpn"].value_counts().to_dict()
        }
    }
    
    summary_path = os.path.join(out_dir, "dataset_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[INFO] Dataset summary written to {summary_path}")

if __name__ == "__main__":
    main()
