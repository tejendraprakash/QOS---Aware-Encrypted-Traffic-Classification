#!/usr/bin/env python3
"""
scripts/reconstruct_clean_dataset.py

Phase 6D: Dataset Forensic Audit & Clean Reconstruction Pipeline
Filters background infrastructure noise, multicast leaks, and ephemeral resolution traffic
from the UNB ISCX VPN-nonVPN raw captures to produce an authentic, scientifically sound dataset.

Filtering Rules (Standard Computer Networking Hygiene):
1. Exclude DNS (Port 53), mDNS (Port 5353), LLMNR (Port 5355), NetBIOS (Ports 137, 138), SSDP (Port 1900)
2. Exclude non-transport protocols (ICMP proto 1, IGMP proto 2)
3. Exclude Multicast and Broadcast addresses (IP >= 224.0.0.0)
4. Exclude zero/single-packet artifacts (total_packets < 2)
5. Group-aware fold assignment preserving 100% capture-group isolation (Max folds per group = 1)
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

def is_noise_flow(row):
    # Rule 1: Infrastructure noise ports
    noise_ports = {53, 5353, 5355, 137, 138, 1900}
    if row['src_port'] in noise_ports or row['dst_port'] in noise_ports:
        return True, "Infrastructure Port (DNS/LLMNR/NetBIOS/SSDP)"
        
    # Rule 2: Non-transport protocols
    if row['protocol'] not in (6, 17):
        return True, f"Non-TCP/UDP Protocol ({row['protocol_name']})"
        
    # Rule 3: Multicast / Broadcast IPs
    for ip_str in [row['src_ip'], row['dst_ip']]:
        if isinstance(ip_str, str):
            octets = ip_str.split('.')
            if len(octets) == 4 and octets[0].isdigit():
                if int(octets[0]) >= 224:
                    return True, "Multicast/Broadcast IP"
                    
    # Rule 4: Trivial packet fragments
    if row['total_packets'] < 2:
        return True, "Single Packet Fragment"
        
    return False, "Authentic Application Flow"

def main():
    print("=" * 80)
    print("PHASE 6D: DATASET FORENSIC AUDIT & CLEAN RECONSTRUCTION")
    print("=" * 80)
    
    out_dir = Path("output/phase6d")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    in_flows = Path("output/phase5/track_b_flows.csv")
    in_folds = Path("output/phase5/track_b_folds.csv")
    
    df_raw = pd.read_csv(in_flows)
    df_raw_folds = pd.read_csv(in_folds)
    print(f"Loaded raw Track B dataset: {len(df_raw)} flows across {df_raw['qos_class'].nunique()} classes.")
    
    # Audit noise
    audit_results = [is_noise_flow(r) for _, r in df_raw.iterrows()]
    df_raw['is_noise'] = [a[0] for a in audit_results]
    df_raw['noise_reason'] = [a[1] for a in audit_results]
    
    print("\n=== FORENSIC CONTAMINATION BREAKDOWN ===")
    print(df_raw['noise_reason'].value_counts())
    
    print("\nContamination by QoS Class:")
    print(pd.crosstab(df_raw['qos_class'], df_raw['noise_reason']))
    
    # Filter authentic flows
    df_clean = df_raw[~df_raw['is_noise']].copy().reset_index(drop=True)
    print(f"\nFiltered clean authentic dataset: {len(df_clean)} flows ({len(df_clean)/len(df_raw)*100:.1f}% retained).")
    print("Clean Class Distribution:")
    for cls, cnt in df_clean['qos_class'].value_counts().items():
        print(f"  {cls:<12}: {cnt} flows ({cnt/len(df_clean)*100:.2f}%)")
        
    # Group preservation
    print(f"\nCapture Groups present: {df_clean['capture_group'].nunique()}")
    
    # 5-fold StratifiedGroupKFold
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    df_clean['fold'] = -1
    for fold, (tr, va) in enumerate(sgkf.split(df_clean, df_clean['qos_class'], df_clean['capture_group'])):
        df_clean.loc[va, 'fold'] = fold
        
    print("\n=== CLEAN DATASET 5-FOLD DISTRIBUTION ===")
    print(pd.crosstab(df_clean['fold'], df_clean['qos_class']))
    
    # Verify zero capture group leakage
    max_folds = df_clean.groupby('capture_group')['fold'].nunique().max()
    print(f"\nCapture group leakage audit (Max folds per capture group): {max_folds}")
    assert max_folds == 1, "CRITICAL ERROR: Capture group leakage detected across folds!"
    print("LEAKAGE AUDIT PASSED: Exactly 0 capture groups leak between train and validation splits.")
    
    # Save clean dataset & folds
    clean_flows_path = out_dir / "clean_track_b_flows.csv"
    clean_folds_path = out_dir / "clean_track_b_folds.csv"
    
    df_clean.to_csv(clean_flows_path, index=False)
    df_clean[['flow_id', 'capture_group', 'qos_class', 'application', 'is_vpn', 'fold']].to_csv(clean_folds_path, index=False)
    
    print(f"\n[DONE] Saved clean flows to: {clean_flows_path}")
    print(f"[DONE] Saved clean folds to: {clean_folds_path}")

if __name__ == "__main__":
    main()
