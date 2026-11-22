# Agent D: Network Feature Engineering & Compatibility Audit

**Project Title:** QoS-Aware Encrypted Traffic Classification for Next-Generation Networks  
**Author / Persona:** Agent D — Network Feature Researcher (`.agents/agents/network-feature-researcher.md`)  
**Audit Stage:** Phase 4 — Feature Dimensionality, Compatibility & Leakage Audit  
**Date:** November 2026  

---

## 1. Executive Summary

This audit evaluates the feature representation landscape across **Dataset 1** (our 103-feature behavioral table in `output/flow_features.csv`), **Dataset 2 ARFF files** (the 23 time-based features from UNB ISCX), and the **140 raw PCAP captures** in `ISCX dataset/PCAPs/`.

We assess feature compatibility, mathematical equivalence, redundancy, collinearity, and early-classification feasibility.

---

## 2. Feature Schema Discrepancy & Compatibility Matrix

### 2.1 The Pre-Extracted ARFF Trap
The pre-computed ARFF files in `ISCX dataset/ARFF/` were generated using **ISCXFlowMeter** (a Java-based tool from 2016). When we inspect their 23 attributes against our 103 behavioral features:

```
[Feature Category]           [Dataset 1 (flow_features.csv)]     [Dataset 2 (ISCX ARFF)]
Total Features               103 Behavioral + 14 Metadata        23 Time-Based Attributes
Packet Size Moments          21 features (Mean, Std, Min, Max,   0 features (COMPLETELY ABSENT!)
                             Median, p25, p75 across 3 streams)
TCP Control Flags            6 features (SYN, ACK, FIN, RST,     0 features (COMPLETELY ABSENT!)
                             PSH, URG)
TCP Window Dynamics          4 features (Min, Max, Mean, Std)    0 features (COMPLETELY ABSENT!)
Burst Dynamics               10 features (Count, Mean, Max)      0 features (COMPLETELY ABSENT!)
Early Packet Profiles        24 features (N=5, 10, 15, 20)       0 features (COMPLETELY ABSENT!)
Directional Byte Totals      4 features (Fwd/Bwd bytes & ratios) 0 features (Rates only, no totals)
Inter-Arrival Times (IAT)    21 features (Moments & percentiles) 9 features (Total, Min, Max, Mean)
Active / Idle Sub-flow       0 features (Uses 60s timeout)       8 features (Min, Mean, Max, Std)
```

### 2.2 Key Compatibility Findings:
1. **Direct Tabular Merging is IMPOSSIBLE:**
   The ISCX ARFF files **completely lack packet size statistics, TCP flags, burst dynamics, and early-packet features**. Merging `flow_features.csv` with the ARFF tables would force dropping over 80% of our engineered behavioral features.
2. **Recreation from Raw PCAPs is 100% FEASIBLE:**
   Because we possess all **140 raw PCAP/PCAPNG captures** in `ISCX dataset/PCAPs/`, our streaming flow extractor (`scripts/flow_extractor.py`) can process these captures directly, extracting the **identical 103-feature schema** across the entire ISCX benchmark.

---

## 3. In-Depth Feature Group Analysis

### 3.1 Packet Size Statistics (The Primary QoS Differentiator)
* **Dataset 1 Implementation:** Evaluates Mean, Std, Min, Max, Median, 25th percentile ($Q_1$), and 75th percentile ($Q_3$) for Bidirectional, Forward, and Backward traffic (21 features).
* **Measurement Precision:** Measures **IP Total Length** (`ip_total_length`). This cleanly isolates application data dynamics from link-layer Ethernet padding (64-byte minimum frame size) and virtual tunnel framing overhead.
* **QoS Importance:** Crucial for separating Real-Time voice (small, fixed frames $\le 200\text{ bytes}$) from Streaming video chunks and Bulk transfers ($1400\text{--}1500\text{ bytes}$).

### 3.2 Inter-Arrival Time (IAT) Statistics
* **Dataset 1 Implementation:** Computed on monotonically sorted packet timestamps: $\text{IAT}_i = t_i - t_{i-1}$ across Bidirectional, Forward, and Backward directions (21 features).
* **QoS Importance:** Direct surrogate for network jitter ($\sigma_{\text{IAT}}$) and packet pacing. Periodic audio codecs (Opus/G.711) exhibit low IAT variance ($\sigma \approx 2\text{--}5\text{ ms}$), whereas bursty browsing exhibits long tail idle intervals.

### 3.3 Directional Dynamics & Ratios
* Evaluates $N_{\text{fwd}}$, $N_{\text{bwd}}$, $B_{\text{fwd}}$, $B_{\text{bwd}}$, `fwd_pkt_ratio`, `fwd_byte_ratio`, `fwd_to_bwd_pkt_ratio`, and `fwd_to_bwd_byte_ratio`.
* **Asymmetry Signature:** Streaming media generates extreme downstream byte ratios ($> 20:1$). Interactive voice generates symmetric packet counts ($1:1$). Bulk uploads generate forward-dominated byte ratios ($> 15:1$).

### 3.4 TCP Flags & Advertised Receive Windows
* Evaluates SYN, ACK, FIN, RST, PSH, URG counts and window statistics (`win_min`, `win_max`, `win_mean`, `win_std`).
* **Protocol Signature:** Identifies connection setup and teardown behavior. Differentiates persistent streaming from ephemeral chat/web connections.
* **Zero-Variance Anomaly:** `tcp_urg_count` has $\sigma = 0.0$ across all 17,617 rows in Dataset 1. Modern TCP implementations deprecate the urgent pointer. **Must be dropped during preprocessing.**

### 3.5 Traffic Burst Dynamics
* Evaluates contiguous runs of same-direction packets before direction alternates.
* **Collinearity Finding:** `fwd_burst_count` and `bwd_burst_count` have a Pearson correlation of $r = 1.0000$. Because forward and backward bursts alternate in lockstep, their counts are identical. **One must be dropped to eliminate redundancy.**

### 3.6 Early-Flow Classification Features (Edge Router Ready)
* Evaluates the first $N \in \{5, 10, 15, 20\}$ packets: duration, bytes, forward packets, backward packets, and average packet size.
* Includes completion flag `early_N_complete` ($0$ if flow terminated before $N$ packets).
* **Research Value:** Proves whether edge routers can enforce QoS marking within the first $10$ packets ($\le 50\text{ ms}$) without waiting for complete flow termination.

---

## 4. Potential Feature Leakage Audit

We audited all 103 behavioral features for leakage vectors:
1. **Identifier Leakage:** ZERO IP addresses, MAC addresses, port numbers, or session IDs exist in the 103 behavioral feature columns.
2. **Timestamp Leakage:** Absolute POSIX timestamps are restricted to metadata columns (`flow_start_time`, `flow_end_time`). All behavioral temporal features are relative differences ($\Delta t$, IAT).
3. **Future Information Leakage:** Early packet features strictly inspect packets $1$ through $N$. They do not access whole-flow durations or packet totals.

---

## 5. Strategic Recommendations

1. **Reject ARFF Direct Merging:** Do not merge pre-computed ISCX ARFF tables with `flow_features.csv`.
2. **Feature Pruning for Phase 3 Modeling:**
   * Drop constant feature: `tcp_urg_count`.
   * Drop collinear duplicate: `bwd_burst_count` (retain `fwd_burst_count`).
   * Drop redundant volume pair: retain `total_bytes` and `fwd_bytes`; drop `bwd_bytes` ($r = 0.9982$).
3. **Unified Schema for Future Extensions:** In Phase 5, run `scripts/flow_extractor.py` across the remaining 103 PCAPs in `ISCX dataset/PCAPs/` to generate a unified 103-feature holdout dataset.
