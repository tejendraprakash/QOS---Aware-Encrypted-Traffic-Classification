# Phase 3 Pre-Modeling Audit & Strategic Decisions

**Project Title:** QoS-Aware Encrypted Network Traffic Classification for Next-Generation Networks  
**Document Purpose:** Independent verification of Phase 2 dataset claims, audit of data integrity, and cross-validation design before Phase 3 model training.  
**Auditor:** Consolidated Subagent Research Team (Lead Research Engineer, Dataset Auditor, Network Feature Researcher, Label Mapping Reviewer, Code Reviewer, Validation Strategist)  
**Date:** November 2026  
**Status:** Comprehensive Pre-Flight Review (No models trained, no raw data altered)

---

## 1. Executive Summary & Verification Matrix

Before commencing any machine learning modeling in Phase 3, we conducted an exhaustive, independent audit of `output/flow_features.csv` and verified every claim made in earlier reports.

| Audit Dimension | Claim in Prior Report | Verified Ground Truth | Status / Risk Level |
| :--- | :--- | :--- | :--- |
| **Row Count** | 17,617 flows | **17,617 rows** | **Verified** |
| **Column Count** | 101 columns (14 metadata + 87 ML) | **117 columns (14 metadata + 103 ML)** | **Inconsistent in Report (Under-reported)** |
| **Class Balance** | "Clean, balanced dataset" | **Severe Imbalance (6.30 : 1 ratio)** | **FALSE CLAIM (Identified & Corrected)** |
| **Capture Groups** | 24 capture groups | **32 unique capture groups** | **Refined & Verified** |
| **Groups per Class** | Implied sufficient for 5-fold CV | **Browsing has ONLY 3 groups!** | **HIGH RISK for Cross-Validation** |
| **GroupKFold (5-fold)** | "Use standard GroupKFold" | **Fails completely (0 Browsing in test folds)** | **CRITICAL FLAW (Identified & Resolved)** |
| **Email as Browsing** | "Browsing category" | **85% local OS multicast noise (LLMNR/NetBIOS)**| **SEVERE SEMANTIC RISK** |
| **Browsing Diversity** | Assumed multi-app browsing | **0 non-Email Browsing captures exist** | **STRUCTURAL DEFECT in Raw Dataset** |
| **Feature Redundancy** | Assumed all 87 features active | **1 constant feature (`tcp_urg_count`), 5 $r \ge 0.99$** | **Minor / Normal (Addressed)** |
| **Data Leakage in Features**| Clean behavioral features | **Zero IP/port/MAC/timestamp leakage in ML cols** | **Verified ML-Safe** |

---

## 2. Detailed Subagent Investigations

### 2.1 Dataset Auditor: Dimensions, Integrity, and Grouping
* **Exact Dimensions:**
  * Rows: **17,617**
  * Columns: **117**
  * Total data cells: **2,061,189**
  * Null / NaN values: **0** across all 117 columns.
  * Infinite values: **0** across all numerical columns.
* **Capture Groups Audit:**
  * Across the 37 raw PCAP/PCAPNG capture files, exactly **32 distinct capture groups** are established in `research_notes/capture_groups.csv`.
  * Paired vantage-point captures were successfully merged into single groups to eliminate intra-session leakage:
    * `email1a.pcap` + `email1b.pcap` $\to$ `GRP_EMAIL_1` (5,870 flows)
    * `email2a.pcap` + `email2b.pcap` $\to$ `GRP_EMAIL_2` (672 flows)
    * `vpn_email2a.pcap` + `vpn_email2b.pcap` $\to$ `GRP_VPN_EMAIL_2` (305 flows)
    * `ftps_down_1a.pcap` + `ftps_down_1b.pcap` $\to$ `GRP_FTPS_DOWN_1` (945 flows)
    * `ftps_up_2a.pcap` + `ftps_up_2b.pcap` $\to$ `GRP_FTPS_UP_2` (193 flows)

---

### 2.2 Network Feature Researcher: Feature Counts, Redundancy & Correlation
* **Feature Count Discrepancy Resolved:**
  * The Phase 2 summary text cited "101 columns (14 metadata + 87 behavioral)".
  * The actual dataset contains **117 columns**:
    * **14 Metadata/Audit columns:** `flow_id`, `capture_filename`, `capture_group`, `application`, `is_vpn`, `qos_class`, `label_confidence`, `src_ip`, `src_port`, `dst_ip`, `dst_port`, `protocol_name`, `flow_start_time`, `flow_end_time`.
    * **103 ML-Safe Behavioral Features:**
      * Basic volume/rate features (14)
      * Packet size distribution moments (21)
      * Inter-arrival time distribution moments (21)
      * TCP flags and window statistics (10)
      * UDP indicators and volume (3)
      * Burst dynamics (10)
      * Early classification profiles (4 intervals $\times$ 6 features = 24)
* **Constant & Redundant Features:**
  * **Zero Variance Feature:** `tcp_urg_count` has $\sigma = 0.0$ (value is 0 across all 17,617 flows). Urgent pointer signaling is virtually non-existent in modern traffic. Must be dropped during preprocessing.
  * **Collinear Feature Pairs ($r \ge 0.99$):**
    1. `fwd_burst_count` $\leftrightarrow$ `bwd_burst_count` ($r = 1.0000$): In bidirectional flows, bursts strictly alternate directions, making the counts identical.
    2. `total_bytes` $\leftrightarrow$ `bwd_bytes` ($r = 0.9982$): In client-server traffic, download payload dominates total bytes.
    3. `fwd_to_bwd_pkt_ratio` $\leftrightarrow$ `fwd_burst_pkts_mean` ($r = 0.9983$).
    4. `bidir_iat_max` $\leftrightarrow$ `fwd_iat_max` ($r = 0.9985$).
    5. `bidir_iat_min` $\leftrightarrow$ `bidir_iat_p25` ($r = 0.9902$).
* **Early-Stage Feature Separation:**
  * `early_5_*`, `early_10_*`, `early_15_*`, and `early_20_*` are strictly derived from the first $N$ packets.
  * They do not leak whole-flow metrics (`flow_duration` or `total_packets`).
  * However, they should be modeled as an **independent early-classification benchmark**, not blended indiscriminately with full-flow summary features.

---

### 2.3 Label Mapping Reviewer: The Email & Browsing Dilemma
Our deep inspection of the flows assigned to the **Browsing** class revealed an alarming structural defect:
1. **Zero Web Browsing Captures in Dataset:**
   * The dataset contains captures for FTPS, SCP, Skype, Hangouts, Facebook Audio/Video, Spotify, YouTube, Vimeo, Netflix, and VoIP Buster.
   * **There are NO HTTP or HTTPS Web Browsing captures (Chrome, Firefox, Safari) in the raw dataset!**
   * The entire "Browsing" class (6,847 flows, 38.87% of the dataset) is composed exclusively of **Email** captures.
2. **Email Flows are Dominated by Local OS Broadcast Noise:**
   * When examining the destination ports of the 6,847 flows in the Email captures:
     * **Port 5355 (LLMNR - Link-Local Multicast Name Resolution): 5,135 flows (75.0%)**
     * **Port 53 (DNS): 368 flows (5.4%)**
     * **Port 137 (NetBIOS Name Service): 245 flows (3.6%)**
     * **Port 547 (DHCPv6): 95 flows (1.4%)**
     * **Port 138 (NetBIOS Datagram): 78 flows (1.1%)**
     * **Port 5353 (mDNS): 69 flows (1.0%)**
     * **Genuine Email Ports (SMTPS 465, IMAPS 993, IMAP 143): Only 302 flows (~4.4%)!**
   * Over **86% of the flows in Email captures are background UDP broadcast/multicast queries** sent by the host OS while the mail client was idling or attempting local name lookups.
   * In fact, 6,304 flows in Email captures are **UDP**, even though email is strictly a **TCP** protocol!
3. **Scientific Implication:**
   * If we train a classifier to predict "Browsing" using these flows, the model will not learn Web Browsing behavior; it will learn to identify **Windows LLMNR multicast packet bursts**!
4. **Effect of Excluding Email:**
   * If Email is excluded from the dataset, the **Browsing class drops to exactly 0 flows**.
   * The classification problem fundamentally shifts from a **4-class QoS problem** to a **3-class QoS problem** (`Real-Time`, `Streaming`, `Bulk Data`).

---

### 2.4 Validation Strategist: Class Imbalance & GroupKFold Breakdown

#### A. Empirical Class Imbalance
The previous report’s claim that the dataset is "balanced" is mathematically incorrect:

$$\text{Class Distribution:}$$
* **Real-Time:** 8,132 flows (**46.16%**)
* **Browsing:** 6,847 flows (**38.87%**)
* **Bulk Data:** 1,348 flows (**7.65%**)
* **Streaming:** 1,290 flows (**7.32%**)

$$\text{Imbalance Ratio} = \frac{8,132}{1,290} \approx \mathbf{6.30 : 1}$$

* **Major Finding:** Bulk Data and Streaming combined account for only **14.97%** of the dataset, while Real-Time and Browsing make up **85.03%**.
* **Remedy:** Evaluation must strictly use **Macro-Averaged F1-Score**, **Balanced Accuracy**, and **Class-Weighted Loss functions**. Standard accuracy will be deceptively dominated by the majority classes.

#### B. The GroupKFold Failure
* Number of capture groups per class:
  * Bulk Data: 10 groups
  * Real-Time: 10 groups
  * Streaming: 9 groups
  * **Browsing: ONLY 3 GROUPS (`GRP_EMAIL_1`, `GRP_EMAIL_2`, `GRP_VPN_EMAIL_2`)**
* Furthermore, `GRP_EMAIL_1` contains **5,870 flows (33.3% of the entire dataset)**.
* **Why standard 5-fold `GroupKFold` fails:**
  * You cannot distribute 3 groups across 5 folds without at least 2 folds having **zero Browsing flows** in their test set.
  * Our empirical simulation confirmed: in 5-fold `GroupKFold`, Folds 1, 2, and 4 completely lack class coverage.
* **The Solution:**
  * If a 4-class experiment is conducted, we must use **`StratifiedGroupKFold(n_splits=3)`**, which is the **only configuration that achieves 100% class coverage across all folds**.

#### C. Domain Shift (VPN vs Non-VPN)
* Non-VPN flows: 10,243 (58.14%)
* VPN flows: 7,374 (41.86%)
* Empirical comparison across matching features:
  * Mean packet size: Non-VPN = $119.18\text{ B}$ vs VPN = $177.40\text{ B}$ (**+48.8% increase** due to VPN tunnel encapsulation and padding).
  * Flow duration: Non-VPN = $15.36\text{ s}$ vs VPN = $23.83\text{ s}$ (**+55.1% longer**).
* **Research Warning:** A model trained exclusively on non-VPN traffic will experience significant performance degradation when evaluated on VPN traffic due to transport-layer encapsulation shift.

---

## 3. What is Verified vs. Unverified vs. Risky

### What is Verified
1. **Data Integrity:** All 17,617 rows in `output/flow_features.csv` are fully populated with zero NaNs, zero Infs, and zero corrupt cells.
2. **ML Safety of Features:** The 103 behavioral features contain no IP addresses, MAC addresses, port numbers, or timestamps.
3. **Capture Group Traceability:** All 37 capture files are mapped to 32 mutually exclusive capture groups.
4. **Link-Type Robustness:** Both Ethernet (`linktype 1`) and Raw IP (`linktype 101`) captures were correctly parsed at the binary level.

### What is Unverified
1. **Whether the 4 Cisco QoS classes are linearly separable:** Feature space clusterability (UMAP / t-SNE) has not yet been computed.
2. **Early-classification accuracy trade-off:** Whether 10 packets provide sufficient discriminative power compared to full flows remains to be empirically tested in Phase 3.

### What is Wrong or Risky in the Current Dataset
1. **Browsing Class Validity (Critical Risk):** The Browsing class is 100% Email, of which 85% is OS multicast background noise. Claiming this represents "Web Browsing" is scientifically indefensible.
2. **Extreme Volume Skew in Capture Groups:** `GRP_EMAIL_1` alone contains 5,870 flows (1/3 of the entire dataset). Any random split will cause massive data leakage; standard 5-fold GroupKFold breaks.
3. **Severe Class Imbalance:** A 6.3 : 1 imbalance ratio between Real-Time and Streaming.

---

## 4. What We Need to Fix Before/During Phase 3 Modeling

1. **Feature Preprocessing Pipeline:**
   * Drop the constant feature `tcp_urg_count` ($\sigma = 0$).
   * Handle the collinear pair `fwd_burst_count` / `bwd_burst_count` ($r = 1.0$).
2. **Validation Strategy:**
   * Abandon standard `KFold` and 5-fold `GroupKFold`.
   * For 4-class evaluation, enforce **`StratifiedGroupKFold(n_splits=3)`**.
3. **Email Filtration Strategy:**
   * Implement a two-track experimental design:
     * **Track A (4-Class Baseline):** Includes Email filtered to remove non-application multicast noise (ports 5355, 5353, 137).
     * **Track B (3-Class Pure QoS):** Excludes Email entirely, evaluating the unambiguous QoS classes: `Bulk Data`, `Streaming`, `Real-Time`.

---

## 5. The Exact First Machine Learning Experiment to Run

To establish a scientifically defensible benchmark that directly improves upon the paper by Huang et al. (2021), our first ML experiment must be structured as follows:

### Experiment 3.1: Dual-Track Baseline QoS Classification

```
                           Raw Feature Dataset (17,617 flows)
                                           │
                    ┌──────────────────────┴──────────────────────┐
                    ▼                                             ▼
          [Track A: 4-Class QoS]                        [Track B: 3-Class Pure QoS]
      (Bulk, Streaming, Real-Time, Email)           (Bulk, Streaming, Real-Time)
           Total Flows: 17,617                           Total Flows: 10,770
                    │                                             │
                    ▼                                             ▼
     StratifiedGroupKFold (k=3)                     StratifiedGroupKFold (k=5)
      Group: capture_group (32 grps)                 Group: capture_group (29 grps)
                    │                                             │
                    ├──────────────────────┬──────────────────────┤
                    ▼                                             ▼
           [Random Forest Classifier]                    [XGBoost Classifier]
           - Class-weighted loss                         - Multi-softmax
           - 100 estimators                              - Subsample 0.8
                    │                                             │
                    └──────────────────────┬──────────────────────┘
                                           ▼
                                [Comparative Evaluation]
                         - Macro Precision, Recall, F1-Score
                         - Balanced Accuracy (immune to imbalance)
                         - Confusion Matrix & Misclassification Analysis
```

### Protocol Specifications:
1. **Input Features:** 79 full-flow behavioral features (excluding the 14 metadata fields and 24 early-packet features).
2. **Pre-filtering:** Remove `tcp_urg_count` and collinear duplicates.
3. **Evaluation Metrics:** **Macro F1-Score** and **Balanced Accuracy** (never raw accuracy).
4. **Baselines:** Random Forest and XGBoost with class-weighted loss to counter the 6.3:1 imbalance.
5. **Ablation Comparison:** Compare Track A vs. Track B to rigorously prove the exact impact of the Email/Browsing ambiguity on next-generation network traffic classification.
