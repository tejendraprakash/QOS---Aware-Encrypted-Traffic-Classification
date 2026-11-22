#!/usr/bin/env python3
"""
Phase 5: Feature Audit and Statistical Verification Script
Project: QoS-Aware Encrypted Traffic Classification for Next-Generation Networks

Author: Lead ML Research Coordinator
Purpose: Inspects output/flow_features.csv, identifies column types, audits missing/infinite values,
         flags zero-variance and collinear features, and outputs a structured audit CSV.
"""

import os
import sys
import csv
import json
import numpy as np
import pandas as pd

def run_feature_audit():
    input_csv = 'output/flow_features.csv'
    output_dir = 'output/phase5'
    os.makedirs(output_dir, exist_ok=True)

    print(f"[INFO] Reading {input_csv}...")
    df = pd.read_csv(input_csv)
    n_rows, n_cols = df.shape
    print(f"[INFO] Dataset shape: {n_rows:,} rows, {n_cols} columns")

    # Define metadata columns
    metadata_cols = [
        'flow_id', 'capture_filename', 'capture_group', 'application',
        'is_vpn', 'qos_class', 'label_confidence', 'src_ip', 'src_port',
        'dst_ip', 'dst_port', 'protocol_name', 'flow_start_time', 'flow_end_time'
    ]

    all_cols = list(df.columns)
    actual_meta_cols = [c for c in all_cols if c in metadata_cols]
    behavioral_cols = [c for c in all_cols if c not in metadata_cols]

    print(f"[INFO] Metadata/Audit columns: {len(actual_meta_cols)}")
    print(f"[INFO] Behavioral ML-safe candidate columns: {len(behavioral_cols)}")

    # Early-packet features
    early_cols = [c for c in behavioral_cols if c.startswith('early_')]
    full_flow_cols = [c for c in behavioral_cols if not c.startswith('early_')]

    print(f"  - Full-flow behavioral features : {len(full_flow_cols)}")
    print(f"  - Early-packet profiles (5,10,15,20): {len(early_cols)}")

    # Feature audit metrics
    audit_records = []
    num_df = df[behavioral_cols].select_dtypes(include=[np.number])

    # Compute correlation matrix for numeric behavioral features
    corr_matrix = num_df.corr().abs()

    for col in all_cols:
        col_type = 'Metadata' if col in metadata_cols else ('Early-Packet' if col.startswith('early_') else 'Full-Flow Behavioral')
        dtype_str = str(df[col].dtype)
        n_missing = int(df[col].isna().sum())
        n_unique = int(df[col].nunique())

        if col in num_df.columns:
            n_inf = int(np.isinf(df[col]).sum())
            col_min = float(df[col].min())
            col_max = float(df[col].max())
            col_mean = round(float(df[col].mean()), 4)
            col_std = round(float(df[col].std()), 4)
            is_constant = bool(col_std == 0.0)

            # Find highest correlated partner (excluding self)
            corrs = corr_matrix[col].drop(col)
            max_corr_val = float(corrs.max()) if len(corrs) > 0 else 0.0
            max_corr_partner = str(corrs.idxmax()) if len(corrs) > 0 else 'None'
        else:
            n_inf = 0
            col_min = 'N/A'
            col_max = 'N/A'
            col_mean = 'N/A'
            col_std = 'N/A'
            is_constant = False
            max_corr_val = 0.0
            max_corr_partner = 'N/A'

        status = 'Valid'
        if col in metadata_cols:
            status = 'Audit/Metadata (Non-ML)'
        elif is_constant:
            status = 'Constant Feature (Zero Variance)'
        elif max_corr_val >= 0.999:
            status = f'Collinear with {max_corr_partner} (r={max_corr_val:.4f})'

        audit_records.append({
            'Column_Name': col,
            'Category': col_type,
            'Data_Type': dtype_str,
            'Missing_Count': n_missing,
            'Infinite_Count': n_inf,
            'Unique_Values': n_unique,
            'Min': col_min,
            'Max': col_max,
            'Mean': col_mean,
            'Std': col_std,
            'Is_Constant': is_constant,
            'Max_Correlation': round(max_corr_val, 4) if isinstance(max_corr_val, float) else 0.0,
            'Top_Correlated_Feature': max_corr_partner,
            'Audit_Status': status
        })

    audit_csv_path = os.path.join(output_dir, 'feature_audit_summary.csv')
    pd.DataFrame(audit_records).to_csv(audit_csv_path, index=False)
    print(f"[INFO] Feature audit summary written to {audit_csv_path}")

    # Specific investigations
    urg_std = df['tcp_urg_count'].std() if 'tcp_urg_count' in df else None
    burst_corr = df['fwd_burst_count'].corr(df['bwd_burst_count']) if ('fwd_burst_count' in df and 'bwd_burst_count' in df) else None

    print("\n=== Specific Feature Checks ===")
    print(f"tcp_urg_count std: {urg_std} (Constant = {urg_std == 0.0})")
    print(f"fwd_burst_count <-> bwd_burst_count correlation: {burst_corr:.4f}")

    # Export dictionary of feature sets for model pipelines
    feature_sets = {
        'metadata_columns': actual_meta_cols,
        'all_behavioral_features': behavioral_cols,
        'full_flow_features_raw': full_flow_cols,
        'full_flow_features_clean': [c for c in full_flow_cols if c not in ('tcp_urg_count', 'bwd_burst_count', 'bwd_bytes')],
        'early_5_features': [c for c in early_cols if 'early_5_' in c],
        'early_10_features': [c for c in early_cols if 'early_10_' in c],
        'early_15_features': [c for c in early_cols if 'early_15_' in c],
        'early_20_features': [c for c in early_cols if 'early_20_' in c],
        'dropped_due_to_collinearity_or_constant': ['tcp_urg_count', 'bwd_burst_count', 'bwd_bytes']
    }

    feature_json_path = os.path.join(output_dir, 'feature_sets.json')
    with open(feature_json_path, 'w', encoding='utf-8') as f:
        json.dump(feature_sets, f, indent=2)
    print(f"[INFO] Feature sets exported to {feature_json_path}")

if __name__ == '__main__':
    run_feature_audit()
