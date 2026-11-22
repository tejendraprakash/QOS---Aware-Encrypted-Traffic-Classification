# Phase 5: Feature Audit and Selection Specification

**Project:** QoS-Aware Encrypted Network Traffic Classification for Next-Generation Networks  
**Phase:** 5 — Baseline Experiment Preparation & Protocol Validation  
**Date:** November 2026  
**Audited Artifact:** [flow_features.csv](file:///c:/Users/saipr/OneDrive/Desktop/EncryptedTrafficResearch/output/flow_features.csv)  
**Output Tables:** [feature_audit_summary.csv](file:///c:/Users/saipr/OneDrive/Desktop/EncryptedTrafficResearch/output/phase5/feature_audit_summary.csv), [feature_sets.json](file:///c:/Users/saipr/OneDrive/Desktop/EncryptedTrafficResearch/output/phase5/feature_sets.json)

---

## 1. Executive Summary

This feature audit independently inspects the 117 columns of the 17,617 extracted network flows in [flow_features.csv](file:///c:/Users/saipr/OneDrive/Desktop/EncryptedTrafficResearch/output/flow_features.csv). We systematically categorize each column, audit data types, evaluate missingness and infinite values, identify zero-variance constants, inspect collinearity, and formally partition features into ML-safe operational sets and prohibited metadata columns.

---

## 2. Global Column Partitioning

The 117 columns are partitioned into two mutually exclusive sets:
1. **Metadata and Audit Columns (14 columns):** Strictly excluded from machine learning inputs to prevent identity leakage and spurious correlation.
2. **Behavioral ML-Safe Candidate Columns (103 columns):** Divided into Full-Flow Behavioral features (79) and Early-Packet Behavioral profiles (24).

```
Total Columns: 117
├── Metadata / Audit Identifiers (14 columns) [PROHIBITED FROM MODEL INPUT]
└── Behavioral ML-Safe Candidates (103 columns)
    ├── Full-Flow Behavioral Features (79 columns)
    │   ├── 1 Zero-Variance Constant (tcp_urg_count, std = 0.0) -> DROPPED IN TRAIN
    │   ├── 1 Collinear Duplicate (bwd_burst_count, r = 1.0000 with fwd_burst_count) -> DROPPED IN TRAIN
    │   └── 77 Active Full-Flow Statistical/Temporal Features
    └── Early-Packet Behavioral Profiles (24 columns)
        ├── Window N = 5  (6 features: complete, duration, bytes, fwd_pkts, bwd_pkts, avg_size)
        ├── Window N = 10 (6 features: complete, duration, bytes, fwd_pkts, bwd_pkts, avg_size)
        ├── Window N = 15 (6 features: complete, duration, bytes, fwd_pkts, bwd_pkts, avg_size)
        └── Window N = 20 (6 features: complete, duration, bytes, fwd_pkts, bwd_pkts, avg_size)
```

---

## 3. Metadata and Leakage-Prone Columns (Excluded)

The following 14 columns are designated as **Ground Truth / Metadata / Audit Identifiers** and must NEVER be exposed to any feature matrix $X$:

| Column Name | Data Type | Scientific Reason for Exclusion |
|---|---|---|
| `flow_id` | string | Arbitrary hash/identifier; models would memorize unique strings. |
| `capture_filename` | string | Reveals the origin PCAP and application name (direct target leakage). |
| `capture_group` | string | Grouping identifier used exclusively for `StratifiedGroupKFold` fold splitting. |
| `application` | string | Sub-label identifier (e.g., Hangouts, Netflix); leaks ground truth class. |
| `is_vpn` | integer | Domain indicator; learning VPN status directly biases cross-domain generalizability. |
| `qos_class` | string | Target variable $y$. |
| `label_confidence` | string | Audit metric evaluating whether flow is definitive, probable, or background. |
| `src_ip` | string | IP address; results in trivial host/subnet memorization rather than traffic behavior. |
| `src_port` | integer | Ephemeral/well-known port; trivial port-based lookup rather than payload-agnostic QoS. |
| `dst_ip` | string | Destination server IP (e.g., Google/Netflix CDN prefixes); severe target leakage. |
| `dst_port` | integer | Destination port (e.g., 443, 80, 5355); leaks service type trivially. |
| `protocol_name` | string | Text string ('TCP', 'UDP'); redundant with numeric `protocol`. |
| `flow_start_time` | float | Wall-clock epoch timestamp; leaks capture date, time, and collection batch. |
| `flow_end_time` | float | Wall-clock epoch timestamp; leaks capture batch. |

---

## 4. Missingness, Infs, and Data Integrity

An automated audit across all 17,617 rows confirms:
- **Null / NaN Count:** Exactly 0 across all 117 columns (100% complete).
- **Infinite Values (+Inf / -Inf):** Exactly 0 across all numeric columns.
- **Data Types:** All 103 behavioral features are numeric (`float64`, `int64`, or `bool`).

---

## 5. Constant and Collinear Feature Anomalies

### 5.1 Zero-Variance Constant: `tcp_urg_count`
- **Mean:** 0.0
- **Standard Deviation:** 0.0
- **Min / Max:** 0 / 0
- **Finding:** In modern encrypted TCP traffic (TLS 1.2/1.3, HTTPS, QUIC), the TCP Urgent pointer (`URG`) flag is obsolete and never set.
- **Handling Protocol:** Must be eliminated by a `VarianceThreshold(threshold=0.0)` step fitted strictly on each training fold.

### 5.2 Perfect Collinear Pair: `fwd_burst_count` vs `bwd_burst_count`
- **Pearson Correlation:** $r = 1.0000$
- **Explanation:** In the current flow extraction logic, forward bursts and backward bursts are triggered symmetrically across conversational turns.
- **Handling Protocol:** Must be detected and dropped on the training fold correlation matrix ($|r| \ge 0.999$) before feeding models.

---

## 6. Full-Flow Feature Taxonomy (79 Features)

The full-flow feature set captures statistical, distributional, and temporal traffic behavior without inspecting payload bytes:

1. **Protocol & Duration:** `protocol`, `flow_duration`, `packet_rate`, `byte_rate`.
2. **Volume & Ratios:** `total_packets`, `fwd_packets`, `bwd_packets`, `total_bytes`, `fwd_bytes`, `bwd_bytes`, `fwd_pkt_ratio`, `fwd_byte_ratio`, `fwd_to_bwd_pkt_ratio`, `fwd_to_bwd_byte_ratio`.
3. **Packet Size Statistics (Bidirectional, Forward, Backward):** Mean, standard deviation, min, max, median, 25th percentile, 75th percentile (21 features).
4. **Inter-Arrival Time (IAT) Statistics (Bidirectional, Forward, Backward):** Mean, standard deviation, min, max, median, 25th percentile, 75th percentile (21 features).
5. **TCP Header Metrics:** `tcp_syn_count`, `tcp_ack_count`, `tcp_fin_count`, `tcp_rst_count`, `tcp_psh_count`, `tcp_urg_count`, `tcp_win_min`, `tcp_win_max`, `tcp_win_mean`, `tcp_win_std` (10 features).
6. **UDP Metrics:** `is_udp`, `udp_packet_count`, `udp_byte_count` (3 features).
7. **Burst Dynamics:** `fwd_burst_count`, `bwd_burst_count`, `fwd_burst_pkts_mean`, `bwd_burst_pkts_mean`, `fwd_burst_bytes_mean`, `bwd_burst_bytes_mean`, `fwd_burst_pkts_max`, `bwd_burst_pkts_max`, `fwd_burst_bytes_max`, `bwd_burst_bytes_max` (10 features).

---

## 7. Leak-Free Preprocessing Pipeline Architecture

To ensure zero information leakage across folds, the preprocessing sequence is defined as an atomic pipeline:

$$\text{Raw Feature Vector } X \xrightarrow{\text{Train-Fitted DropConstant}} X_{clean} \xrightarrow{\text{Train-Fitted DropCollinear}} X_{uncorr} \xrightarrow{\text{Train-Fitted StandardScaler}} X_{scaled}$$

No test-fold or validation-fold statistics are ever computed during variance calculation, correlation filtering, or standardization.
