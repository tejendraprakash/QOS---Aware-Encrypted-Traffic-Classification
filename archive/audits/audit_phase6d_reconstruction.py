#!/usr/bin/env python3
"""
scripts/audit_phase6d_reconstruction.py

Comprehensive independent verification of Phase 6D dataset reconstruction,
filtering rules, flow counts, capture groups, and labeling.
"""

import pandas as pd
import numpy as np

def run_audit():
    print("=" * 80)
    print("PHASE 6D.1: INDEPENDENT DATASET RECONSTRUCTION AUDIT")
    print("=" * 80)
    
    df_raw = pd.read_csv("output/phase5/track_b_flows.csv")
    df_clean = pd.read_csv("output/phase6d/clean_track_b_flows.csv")
    
    print(f"Raw Track B flows: {len(df_raw)}")
    print(f"Clean Track B flows: {len(df_clean)}")
    print(f"Difference (Removed): {len(df_raw) - len(df_clean)}")
    
    # Define exact individual criteria
    c_dns = (df_raw['src_port'] == 53) | (df_raw['dst_port'] == 53)
    c_llmnr = (df_raw['src_port'] == 5355) | (df_raw['dst_port'] == 5355)
    c_netbios = (df_raw['src_port'].isin([137, 138])) | (df_raw['dst_port'].isin([137, 138]))
    c_ssdp = (df_raw['src_port'] == 1900) | (df_raw['dst_port'] == 1900)
    c_non_transport = ~df_raw['protocol'].isin([6, 17])
    
    def is_mcast(r):
        for ip in [r['src_ip'], r['dst_ip']]:
            if isinstance(ip, str):
                parts = ip.split('.')
                if len(parts) == 4 and parts[0].isdigit() and int(parts[0]) >= 224:
                    return True
        return False
        
    c_mcast = df_raw.apply(is_mcast, axis=1)
    c_single = df_raw['total_packets'] < 2
    
    criteria = {
        'DNS (Port 53)': c_dns,
        'LLMNR (Port 5355)': c_llmnr,
        'NetBIOS (Ports 137, 138)': c_netbios,
        'SSDP (Port 1900)': c_ssdp,
        'Non-TCP/UDP (ICMP/IGMP)': c_non_transport,
        'Multicast/Broadcast IP (>=224.0.0.0)': c_mcast,
        'Single-Packet Fragments (<2)': c_single
    }
    
    print("\n--- INDIVIDUAL RULE BREAKDOWN ---")
    rule_summary = []
    for name, mask in criteria.items():
        sub = df_raw[mask]
        rule_summary.append({
            'Rule': name,
            'Flows_Removed': len(sub),
            'Real_Time': int((sub['qos_class'] == 'Real-Time').sum()),
            'Bulk_Data': int((sub['qos_class'] == 'Bulk Data').sum()),
            'Streaming': int((sub['qos_class'] == 'Streaming').sum()),
            'Groups_Affected': int(sub['capture_group'].nunique()),
            'Mean_Pkts': round(sub['total_packets'].mean(), 2) if len(sub) > 0 else 0,
            'Median_Pkts': int(sub['total_packets'].median()) if len(sub) > 0 else 0,
            'Median_Duration_s': round(sub['flow_duration'].median(), 6) if len(sub) > 0 else 0
        })
    df_rules = pd.DataFrame(rule_summary)
    print(df_rules.to_string(index=False))
    
    total_rule_firings = sum(r['Flows_Removed'] for r in rule_summary)
    print(f"\nSum of individual rule hits: {total_rule_firings}")
    
    # Combined union mask
    union_mask = c_dns | c_llmnr | c_netbios | c_ssdp | c_non_transport | c_mcast | c_single
    print(f"Union of all noise rules (Unique flows removed): {union_mask.sum()}")
    print(f"Overlapping rule firings: {total_rule_firings - union_mask.sum()}")
    print(f"Verified Remaining Flows: {len(df_raw) - union_mask.sum()} (Matches clean dataset: {(len(df_raw) - union_mask.sum()) == len(df_clean)})")
    
    # Inspect overlap details
    print("\n--- OVERLAP ANALYSIS ---")
    overlap_mcast_dns = (c_mcast & c_dns).sum()
    overlap_mcast_llmnr = (c_mcast & c_llmnr).sum()
    overlap_mcast_ssdp = (c_mcast & c_ssdp).sum()
    overlap_mcast_netbios = (c_mcast & c_netbios).sum()
    overlap_mcast_nontransport = (c_mcast & c_non_transport).sum()
    print(f"Multicast & DNS: {overlap_mcast_dns}")
    print(f"Multicast & LLMNR (224.0.0.252): {overlap_mcast_llmnr}")
    print(f"Multicast & SSDP (239.255.255.250): {overlap_mcast_ssdp}")
    print(f"Multicast & NetBIOS (Broadcast): {overlap_mcast_netbios}")
    print(f"Multicast & Non-Transport (IGMP): {overlap_mcast_nontransport}")
    
    # Audit remaining 4196 flows
    print("\n--- AUDIT OF REMAINING 4,196 CLEAN FLOWS ---")
    print(f"Protocol distribution in clean data:\n{df_clean['protocol_name'].value_counts()}")
    print(f"\nClass distribution in clean data:\n{df_clean['qos_class'].value_counts()}")
    print(f"\nPacket count percentiles in clean data:\n{df_clean['total_packets'].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])}")
    print(f"\nDuration percentiles in clean data:\n{df_clean['flow_duration'].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])}")
    
    # Check minimum packet threshold
    pkts_under_5 = (df_clean['total_packets'] < 5).sum()
    print(f"\nFlows with < 5 packets in clean dataset: {pkts_under_5}")
    if pkts_under_5 > 0:
        print("Note on Early-5 horizon: Flows with < 5 packets cannot form a complete 5-packet window without padding or truncation.")
        print(f"Breakdown by class for <5 pkts:\n{df_clean[df_clean['total_packets'] < 5]['qos_class'].value_counts()}")

if __name__ == "__main__":
    run_audit()
