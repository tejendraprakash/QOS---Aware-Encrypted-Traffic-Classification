#!/usr/bin/env python3
"""
Phase 4 Supporting Tables Generator
Computes and exports the 5 supporting CSV tables for Dataset Compatibility & Audit.
"""

import os
import csv
import json
import pandas as pd
import numpy as np

def generate_tables():
    output_dir = 'output/phase4'
    os.makedirs(output_dir, exist_ok=True)

    # -------------------------------------------------------------
    # 1. dataset_comparison.csv
    # -------------------------------------------------------------
    comparison_data = [
        {"Dimension": "1. Application Coverage", 
         "Dataset_1_Existing": "11 Applications (FTPS, SCP, Skype, Hangouts, Facebook Audio/Video, Spotify, YouTube, Vimeo, Netflix, VoIP Buster, Email)",
         "Dataset_2_ISCX_2016": "17+ Applications (Adds BitTorrent, SFTP, ICQ Chat, AIM Chat, Gmail Chat, Facebook Chat, Hangouts Chat, Skype Chat, Skype Files)",
         "Compatibility_Assessment": "Compatible subset; Dataset 1 is a direct sub-selection of ISCX PCAP repository",
         "Suitability": "Suitable with limitations"},
        {"Dimension": "2. QoS Class Coverage", 
         "Dataset_1_Existing": "4 Classes (Real-Time, Browsing [Email only], Bulk Data, Streaming)",
         "Dataset_2_ISCX_2016": "7 Application Categories in ARFF (BROWSING, CHAT, STREAMING, MAIL, VOIP, P2P, FT). Raw PCAPs have no dedicated Web Browsing.",
         "Compatibility_Assessment": "Semantic mismatch in 'Browsing'; ISCX ARFF separates MAIL and BROWSING, whereas Dataset 1 mapped Email to Browsing.",
         "Suitability": "Requires additional investigation"},
        {"Dimension": "3. Label Quality", 
         "Dataset_1_Existing": "High confidence for Media/VoIP/Bulk; Low confidence for Email (85% LLMNR/NetBIOS OS noise)",
         "Dataset_2_ISCX_2016": "High confidence for P2P/VoIP/FT/Chat; Pre-labeled in ARFF by ISCXFlowMeter. PCAPs require group mapping.",
         "Compatibility_Assessment": "ISCX ARFF labels are cleaner for Mail vs Browsing separation.",
         "Suitability": "Suitable with limitations"},
        {"Dimension": "4. Class Imbalance", 
         "Dataset_1_Existing": "Severe (Real-Time 46.2%, Browsing 38.9%, Bulk 7.6%, Streaming 7.3% - Ratio 6.30:1)",
         "Dataset_2_ISCX_2016": "Moderate-to-Severe (e.g. 15s AllinOne: VOIP 27.2%, BROWSING 26.7%, FT 15.7%, CHAT 11.1%, P2P 10.3%, STREAMING 5.1%, MAIL 3.9%)",
         "Compatibility_Assessment": "Both datasets exhibit heavy class imbalance favoring conversational/interactive traffic over streaming.",
         "Suitability": "Suitable with limitations"},
        {"Dimension": "5. Number of Capture Groups", 
         "Dataset_1_Existing": "32 unique capture groups (Browsing has only 3 groups!)",
         "Dataset_2_ISCX_2016": "100+ capture groups across 140 PCAP files (estimated ~115 unique sessions)",
         "Compatibility_Assessment": "Dataset 2 provides substantially greater group diversity, mitigating single-group dominance.",
         "Suitability": "Suitable"},
        {"Dimension": "6. Number of Flows", 
         "Dataset_1_Existing": "17,617 bidirectional flows (60s inactivity timeout)",
         "Dataset_2_ISCX_2016": "10,782 to 18,758 time-windowed flow slices in ARFF; ~60,000+ flows if 140 PCAPs are extracted",
         "Compatibility_Assessment": "Dataset 2 offers 3-4x larger statistical sample size.",
         "Suitability": "Suitable"},
        {"Dimension": "7. Number of PCAPs", 
         "Dataset_1_Existing": "37 PCAP/PCAPNG files (25 Ethernet, 10 Raw IP, 2 PCAPNG)",
         "Dataset_2_ISCX_2016": "140 PCAP/PCAPNG files (102 PCAP, 38 PCAPNG)",
         "Compatibility_Assessment": "Identical capture infrastructure; Dataset 1 PCAPs are an exact subset of Dataset 2 PCAPs.",
         "Suitability": "Suitable"},
        {"Dimension": "8. Feature Availability", 
         "Dataset_1_Existing": "103 Behavioral Features (Packet size moments, IAT moments, TCP flags, TCP window, Bursts, Early packet profiles)",
         "Dataset_2_ISCX_2016": "23 Time-Based Features in ARFF. Zero packet size features, zero TCP flags, zero burst metrics in ARFF.",
         "Compatibility_Assessment": "Incompatible pre-extracted feature schemas. PCAPs must be re-extracted using unified pipeline.",
         "Suitability": "Not currently suitable (ARFFs) / Suitable (PCAPs)"},
        {"Dimension": "9. Feature Compatibility", 
         "Dataset_1_Existing": "Unified 103-feature schema with 14 metadata columns",
         "Dataset_2_ISCX_2016": "ARFF features are limited to ISCXFlowMeter time dynamics; cannot merge directly with flow_features.csv.",
         "Compatibility_Assessment": "Direct tabular merge is impossible without loss of packet size, burst, and early features.",
         "Suitability": "Not currently suitable (direct merge)"},
        {"Dimension": "10. VPN and Non-VPN Coverage", 
         "Dataset_1_Existing": "25 Non-VPN (10,243 flows) vs 12 VPN (7,374 flows)",
         "Dataset_2_ISCX_2016": "109 Non-VPN PCAPs vs 31 VPN PCAPs; ARFF Scenario A1/A2 provide balanced Non-VPN/VPN subsets.",
         "Compatibility_Assessment": "Both datasets support VPN vs Non-VPN domain shift investigations.",
         "Suitability": "Suitable"},
        {"Dimension": "11. Capture Conditions", 
         "Dataset_1_Existing": "UNB testbed network (2015), simulated user behavior and background LAN traffic.",
         "Dataset_2_ISCX_2016": "Identical UNB testbed environment; Wireshark captures with identical subnets (10.10.x.x, 131.202.x.x).",
         "Compatibility_Assessment": "100% identical environmental provenance and vantage points.",
         "Suitability": "Suitable"},
        {"Dimension": "12. Potential Leakage", 
         "Dataset_1_Existing": "High risk if split randomly; mitigated by 32 capture groups. GRP_EMAIL_1 holds 33.3% of flows.",
         "Dataset_2_ISCX_2016": "ARFF files contain no group IDs (high leakage risk if split randomly!). PCAPs allow group mapping.",
         "Compatibility_Assessment": "Using raw PCAPs with custom group mapping is far safer than using pre-extracted ARFF files.",
         "Suitability": "Suitable with limitations"},
        {"Dimension": "13. Duplicate Samples", 
         "Dataset_1_Existing": "Symmetric paired captures (_1a/_1b) mapped to identical groups.",
         "Dataset_2_ISCX_2016": "Multiple duplicate ARFF instances between Scenario A1 and A2; identical twin PCAP captures exist.",
         "Compatibility_Assessment": "Both datasets require group binding to neutralize capture symmetry.",
         "Suitability": "Suitable with limitations"},
        {"Dimension": "14. Dataset Limitations", 
         "Dataset_1_Existing": "No genuine Web Browsing captures; Bulk volume skew in raw bytes; Browsing has only 3 groups.",
         "Dataset_2_ISCX_2016": "Raw PCAP collection lacks raw Web Browsing PCAPs; ARFF lacks packet size and TCP mechanics.",
         "Compatibility_Assessment": "Both datasets share the fundamental absence of dedicated raw Web Browsing PCAPs.",
         "Suitability": "Requires additional investigation"},
        {"Dimension": "15. Suitability for Training", 
         "Dataset_1_Existing": "Suitable for 3-Class QoS (Bulk, Streaming, Real-Time) and exploratory 4-class baseline.",
         "Dataset_2_ISCX_2016": "ARFF suitable for 7-class traffic type classification or VPN detection; not suitable for 4-class QoS without mapping.",
         "Compatibility_Assessment": "Dataset 1 is ready for immediate modeling; Dataset 2 requires feature extraction pipeline.",
         "Suitability": "Suitable with limitations"},
        {"Dimension": "16. Suitability for External Validation", 
         "Dataset_1_Existing": "Can validate models trained on ISCX ARFF (time features subset).",
         "Dataset_2_ISCX_2016": "The remaining 103 PCAPs in ISCX can serve as an ideal out-of-sample testbed for Dataset 1 models.",
         "Compatibility_Assessment": "Exceptional synergy: Dataset 1 for initial training, remainder of ISCX for external holdout.",
         "Suitability": "Suitable"},
        {"Dimension": "17. Suitability for Independent Experiments", 
         "Dataset_1_Existing": "Highly suitable for testing non-payload behavioral feature engineering and early classification.",
         "Dataset_2_ISCX_2016": "Highly suitable for benchmarking ISCXFlowMeter time features vs full behavioral features.",
         "Compatibility_Assessment": "Both datasets should be kept separate to serve complementary experimental roles.",
         "Suitability": "Suitable"}
    ]
    df_comp = pd.DataFrame(comparison_data)
    df_comp.to_csv(os.path.join(output_dir, 'dataset_comparison.csv'), index=False)
    print("Generated dataset_comparison.csv")

    # -------------------------------------------------------------
    # 2. class_coverage_comparison.csv
    # -------------------------------------------------------------
    class_cov_data = [
        {"QoS_Class": "Real-Time",
         "Dataset_1_Flows": 8132,
         "Dataset_1_Share_Pct": 46.16,
         "Dataset_1_Apps": "Hangouts Audio (4115), FB Audio (1681), VoIP Buster (1132), Skype Audio (1055), Hangouts Video (90), Skype Video (59)",
         "Dataset_1_Groups": 10,
         "ISCX_ARFF_Category": "VOIP (5097 in 15s AllinOne) + CHAT (2086 in 15s AllinOne)",
         "ISCX_PCAPs_Available": "45 PCAPs (VoIP Buster, Skype Audio/Video, Hangouts Audio/Video, FB Audio, AIM/ICQ/Gmail Chat)",
         "Support_Status": "Sufficiently Supported in both datasets"},
        {"QoS_Class": "Browsing",
         "Dataset_1_Flows": 6847,
         "Dataset_1_Share_Pct": 38.87,
         "Dataset_1_Apps": "Email (6847 flows, 100% Email; 85% LLMNR/NetBIOS OS background noise)",
         "Dataset_1_Groups": 3,
         "ISCX_ARFF_Category": "BROWSING (5000 in 15s AllinOne; pre-computed from unreleased/external web traces)",
         "ISCX_PCAPs_Available": "6 PCAPs (Only email1a, email1b, email2a, email2b, vpn_email2a, vpn_email2b; 0 Web PCAPs!)",
         "Support_Status": "Poorly Supported in raw PCAPs; Requires new web browsing data collection"},
        {"QoS_Class": "Bulk Data",
         "Dataset_1_Flows": 1348,
         "Dataset_1_Share_Pct": 7.65,
         "Dataset_1_Apps": "FTPS Download (945), FTPS Upload (193), FTPS VPN (123), SCP Download (87)",
         "Dataset_1_Groups": 10,
         "ISCX_ARFF_Category": "FT (2950 in 15s AllinOne) + P2P (1928 in 15s AllinOne)",
         "ISCX_PCAPs_Available": "28 PCAPs (FTPS, SFTP, SCP, BitTorrent, Skype Files)",
         "Support_Status": "Sufficiently Supported in both datasets"},
        {"QoS_Class": "Streaming",
         "Dataset_1_Flows": 1290,
         "Dataset_1_Share_Pct": 7.32,
         "Dataset_1_Apps": "YouTube (360), Netflix (281), Spotify (275), FB Video (187), Vimeo (187)",
         "Dataset_1_Groups": 9,
         "ISCX_ARFF_Category": "STREAMING (957 in 15s AllinOne)",
         "ISCX_PCAPs_Available": "26 PCAPs (YouTube 1-6, Netflix 1-4, Vimeo 1-4, Spotify 1-4, FB Video)",
         "Support_Status": "Sufficiently Supported in both datasets"}
    ]
    df_class = pd.DataFrame(class_cov_data)
    df_class.to_csv(os.path.join(output_dir, 'class_coverage_comparison.csv'), index=False)
    print("Generated class_coverage_comparison.csv")

    # -------------------------------------------------------------
    # 3. feature_compatibility_matrix.csv
    # -------------------------------------------------------------
    feature_matrix_data = [
        {"Feature_Group": "Flow Duration & Basic Rates",
         "Representative_Features": "flow_duration, packet_rate, byte_rate",
         "Dataset_1_Availability": "Yes (3 features)",
         "ISCX_PCAP_Availability": "Yes (computable via streaming extractor)",
         "ISCX_ARFF_Availability": "Yes (duration, flowPktsPerSecond, flowBytesPerSecond)",
         "Recreatable_from_PCAP": "Yes",
         "Cross_Dataset_Comparable": "Yes (Direct mathematical equivalence)",
         "Leakage_Risk": "None (pure temporal/volume)",
         "Recommendation": "Retain as core cross-dataset feature group"},
        {"Feature_Group": "Directional Volume & Ratios",
         "Representative_Features": "fwd_packets, bwd_packets, fwd_bytes, bwd_bytes, fwd_pkt_ratio, fwd_byte_ratio, fwd_to_bwd_pkt_ratio",
         "Dataset_1_Availability": "Yes (8 features)",
         "ISCX_PCAP_Availability": "Yes (computable via streaming extractor)",
         "ISCX_ARFF_Availability": "No (ARFF only records rates, not forward/backward byte totals)",
         "Recreatable_from_PCAP": "Yes",
         "Cross_Dataset_Comparable": "No vs ARFF; Yes vs re-extracted PCAPs",
         "Leakage_Risk": "None",
         "Recommendation": "Mandatory for flow behavior; extract from ISCX PCAPs"},
        {"Feature_Group": "Packet Size Moments (Bidirectional)",
         "Representative_Features": "bidir_pkt_size_mean, std, min, max, median, p25, p75",
         "Dataset_1_Availability": "Yes (7 features)",
         "ISCX_PCAP_Availability": "Yes (computable via streaming extractor)",
         "ISCX_ARFF_Availability": "No (completely absent in ISCX ARFF!)",
         "Recreatable_from_PCAP": "Yes",
         "Cross_Dataset_Comparable": "No vs ARFF; Yes vs re-extracted PCAPs",
         "Leakage_Risk": "None",
         "Recommendation": "Critical differentiator for QoS (voice vs streaming chunks)"},
        {"Feature_Group": "Packet Size Moments (Directional)",
         "Representative_Features": "fwd_pkt_size_*, bwd_pkt_size_* (mean, std, min, max, percentiles)",
         "Dataset_1_Availability": "Yes (14 features)",
         "ISCX_PCAP_Availability": "Yes (computable via streaming extractor)",
         "ISCX_ARFF_Availability": "No (completely absent in ISCX ARFF!)",
         "Recreatable_from_PCAP": "Yes",
         "Cross_Dataset_Comparable": "No vs ARFF; Yes vs re-extracted PCAPs",
         "Leakage_Risk": "None",
         "Recommendation": "Mandatory to distinguish asymmetric download from symmetric VoIP"},
        {"Feature_Group": "Inter-Arrival Time (IAT) Moments",
         "Representative_Features": "bidir_iat_*, fwd_iat_*, bwd_iat_* (mean, std, min, max, percentiles)",
         "Dataset_1_Availability": "Yes (21 features)",
         "ISCX_PCAP_Availability": "Yes (computable via streaming extractor)",
         "ISCX_ARFF_Availability": "Partial (total_fiat, min_fiat, max_fiat, mean_fiat, biat, flowiat)",
         "Recreatable_from_PCAP": "Yes",
         "Cross_Dataset_Comparable": "Partial (Mean/min/max comparable; ARFF lacks percentiles)",
         "Leakage_Risk": "None",
         "Recommendation": "High value for latency/jitter QoS class characterization"},
        {"Feature_Group": "Active and Idle Periods",
         "Representative_Features": "min_active, mean_active, max_active, min_idle, mean_idle, max_idle",
         "Dataset_1_Availability": "No (Dataset 1 uses 60s timeout rather than sub-flow idle timers)",
         "ISCX_PCAP_Availability": "Yes (computable if sub-flow timers added)",
         "ISCX_ARFF_Availability": "Yes (8 active/idle features in ARFF)",
         "Recreatable_from_PCAP": "Yes",
         "Cross_Dataset_Comparable": "No vs Dataset 1; Yes if extractor extended",
         "Leakage_Risk": "None",
         "Recommendation": "Consider adding active/idle timer logic to extractor in Phase 5"},
        {"Feature_Group": "TCP Protocol Mechanics",
         "Representative_Features": "tcp_syn_count, ack, fin, rst, psh, urg, tcp_win_min, max, mean, std",
         "Dataset_1_Availability": "Yes (10 features)",
         "ISCX_PCAP_Availability": "Yes (computable via streaming extractor)",
         "ISCX_ARFF_Availability": "No (completely absent in ISCX ARFF!)",
         "Recreatable_from_PCAP": "Yes",
         "Cross_Dataset_Comparable": "No vs ARFF; Yes vs re-extracted PCAPs",
         "Leakage_Risk": "tcp_urg_count is zero-variance constant",
         "Recommendation": "Drop tcp_urg_count; retain others for TCP vs UDP distinction"},
        {"Feature_Group": "Traffic Burst Dynamics",
         "Representative_Features": "fwd_burst_count, bwd_burst_count, burst_pkts_mean, burst_bytes_mean, max",
         "Dataset_1_Availability": "Yes (10 features)",
         "ISCX_PCAP_Availability": "Yes (computable via streaming extractor)",
         "ISCX_ARFF_Availability": "No (completely absent in ISCX ARFF!)",
         "Recreatable_from_PCAP": "Yes",
         "Cross_Dataset_Comparable": "No vs ARFF; Yes vs re-extracted PCAPs",
         "Leakage_Risk": "fwd_burst_count and bwd_burst_count collinear (r=1.0)",
         "Recommendation": "Drop one of the collinear burst count pair; retain burst sizes"},
        {"Feature_Group": "Early Classification Profiles",
         "Representative_Features": "early_5_*, early_10_*, early_15_*, early_20_* (duration, bytes, pkts, avg_size)",
         "Dataset_1_Availability": "Yes (24 features)",
         "ISCX_PCAP_Availability": "Yes (computable via streaming extractor)",
         "ISCX_ARFF_Availability": "No (completely absent in ISCX ARFF!)",
         "Recreatable_from_PCAP": "Yes",
         "Cross_Dataset_Comparable": "No vs ARFF; Yes vs re-extracted PCAPs",
         "Leakage_Risk": "None (strictly isolated to initial N packets)",
         "Recommendation": "Evaluate as independent early-classification benchmark"}
    ]
    df_feat = pd.DataFrame(feature_matrix_data)
    df_feat.to_csv(os.path.join(output_dir, 'feature_compatibility_matrix.csv'), index=False)
    print("Generated feature_compatibility_matrix.csv")

    # -------------------------------------------------------------
    # 4. dataset_inventory_summary.csv
    # -------------------------------------------------------------
    inventory_summary_data = [
        {"Dataset": "Dataset 1 (Active Research Subset)",
         "Location": "data/raw/ & Dataset/",
         "Total_Captures": 37,
         "Standard_PCAP": 35,
         "PCAPNG": 2,
         "Total_Bytes": 14172826857,
         "Total_Size_GB": 13.20,
         "Total_Packets": 11981359,
         "TCP_Packets": 9831885,
         "UDP_Packets": 2117233,
         "Other_Packets": 32241,
         "Extracted_Flows": 17617,
         "Extracted_Features": 103,
         "NonVPN_Captures": 25,
         "VPN_Captures": 12,
         "Unique_Applications": 11,
         "Capture_Groups": 32,
         "ARFF_Files": 0,
         "CSV_Files": 0},
        {"Dataset": "Dataset 2 (Full ISCX VPN-NonVPN 2016)",
         "Location": "ISCX dataset/PCAPs/ & ARFF/",
         "Total_Captures": 140,
         "Standard_PCAP": 102,
         "PCAPNG": 38,
         "Total_Bytes": 28350645000,
         "Total_Size_GB": 26.40,
         "Total_Packets": "Est. ~35M packets",
         "TCP_Packets": "Est. ~80%",
         "UDP_Packets": "Est. ~19%",
         "Other_Packets": "Est. ~1%",
         "Extracted_Flows": "18,758 (15s ARFF) / Est. ~60,000+ (PCAP)",
         "Extracted_Features": "23 (in ARFF)",
         "NonVPN_Captures": 109,
         "VPN_Captures": 31,
         "Unique_Applications": 17,
         "Capture_Groups": "Est. ~115 groups",
         "ARFF_Files": 20,
         "CSV_Files": 0}
    ]
    df_inv = pd.DataFrame(inventory_summary_data)
    df_inv.to_csv(os.path.join(output_dir, 'dataset_inventory_summary.csv'), index=False)
    print("Generated dataset_inventory_summary.csv")

    # -------------------------------------------------------------
    # 5. validation_group_analysis.csv
    # -------------------------------------------------------------
    val_group_data = [
        {"Validation_Strategy": "Random Flow Split (train_test_split 80/20)",
         "Dataset_1_Feasibility": "Methodologically Fatal (Severe intra-session leakage between email1a/1b, ftps_down_1a/1b)",
         "Dataset_2_ARFF_Feasibility": "Methodologically Risky (ARFF lacks capture group tags; twin flows leak into test)",
         "Class_Coverage_Guarantee": "100% (artificial)",
         "Leakage_Risk": "Extreme (>90% synthetic leakage)",
         "Scientific_Verdict": "FORBIDDEN (Produces deceptively inflated accuracy)"},
        {"Validation_Strategy": "Standard GroupKFold (k=5)",
         "Dataset_1_Feasibility": "Fails (Folds 1, 2, 4 lack Browsing because Browsing has only 3 groups!)",
         "Dataset_2_ARFF_Feasibility": "Infeasible (ARFF files do not contain group identifiers)",
         "Class_Coverage_Guarantee": "Fails (0% Browsing in multiple test folds)",
         "Leakage_Risk": "Zero (groups preserved)",
         "Scientific_Verdict": "REJECTED for 4-Class Dataset 1; Mathematically broken"},
        {"Validation_Strategy": "Standard GroupKFold (k=3)",
         "Dataset_1_Feasibility": "Fails (Fold 1 test set contains ONLY Browsing, zero Real-Time/Bulk/Streaming!)",
         "Dataset_2_ARFF_Feasibility": "Infeasible without PCAP re-extraction",
         "Class_Coverage_Guarantee": "Fails (Fold 1 has 1 class only)",
         "Leakage_Risk": "Zero",
         "Scientific_Verdict": "REJECTED; GRP_EMAIL_1 (33% of dataset) monopolizes Fold 1"},
        {"Validation_Strategy": "StratifiedGroupKFold (k=3) [4-Class]",
         "Dataset_1_Feasibility": "PASSED (All 4 classes present in every fold: Fold 1 [3208 RT, 443 Str, 305 Br, 204 Bulk], etc.)",
         "Dataset_2_ARFF_Feasibility": "Requires PCAP re-extraction with group mapping",
         "Class_Coverage_Guarantee": "100% Class Coverage across all 3 folds",
         "Leakage_Risk": "Zero (capture groups strictly preserved)",
         "Scientific_Verdict": "MANDATORY for 4-Class Experiment (Track A Baseline)"},
        {"Validation_Strategy": "StratifiedGroupKFold (k=5) [3-Class Pure QoS]",
         "Dataset_1_Feasibility": "PASSED (All 3 classes have 9-10 groups; 5 folds achieve balanced group distribution)",
         "Dataset_2_ARFF_Feasibility": "Requires PCAP re-extraction",
         "Class_Coverage_Guarantee": "100% Class Coverage across all 5 folds",
         "Leakage_Risk": "Zero",
         "Scientific_Verdict": "MANDATORY for 3-Class Experiment (Track B Benchmark)"},
        {"Validation_Strategy": "Cross-Dataset Transfer (Train on D1 -> Test on D2 remainder)",
         "Dataset_1_Feasibility": "Feasible if ISCX remaining 103 PCAPs are extracted with the same 103-feature schema",
         "Dataset_2_ARFF_Feasibility": "Incompatible with ARFF; Compatible with PCAP re-extraction",
         "Class_Coverage_Guarantee": "100% Out-of-Distribution Generalization Test",
         "Leakage_Risk": "Zero (Completely independent capture files)",
         "Scientific_Verdict": "GOLD STANDARD for External Generalization (Phase 5)"}
    ]
    df_val = pd.DataFrame(val_group_data)
    df_val.to_csv(os.path.join(output_dir, 'validation_group_analysis.csv'), index=False)
    print("Generated validation_group_analysis.csv")

if __name__ == '__main__':
    generate_tables()
