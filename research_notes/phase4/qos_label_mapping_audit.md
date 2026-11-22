# Agent C: QoS Label Mapping Audit & Traffic Semantics Analysis

**Project Title:** QoS-Aware Encrypted Traffic Classification for Next-Generation Networks  
**Author / Persona:** Agent C — QoS Label Mapping Researcher (`.agents/agents/qos-label-mapping-researcher.md`)  
**Audit Stage:** Phase 4 — Label Integrity, Multi-Modal Ambiguity & Semantics  
**Date:** November 2026  

---

## 1. Executive Summary & Label Mapping Overview

The objective of QoS-aware traffic classification is to map network flows into service categories that dictate edge router queue management. In our research, the target classes are:
1. **Bulk Data**
2. **Browsing**
3. **Streaming**
4. **Real-Time**

This audit examines whether current application-to-QoS label assignments accurately represent physical network traffic behavior, uncovers the critical flaw in the "Browsing" category, analyzes background broadcast contamination, and establishes rules for multi-modal applications.

---

## 2. Deep Dive: Is "Email" a Scientifically Defensible "Browsing" Label?

### 2.1 The Core Structural Discovery
In Dataset 1 (`output/flow_features.csv`), the **Browsing** class contains **6,847 flows** (38.87% of the dataset). 
* **100% of these flows originate from Email capture files** (`email1a`, `email1b`, `email2a`, `email2b`, `vpn_email2a`, `vpn_email2b`).
* **There are ZERO Web Browsing captures (HTTP/HTTPS, Chrome, Firefox) in the dataset.**

### 2.2 Empirical Dissection of Email Flows
When we inspect the actual layer-4 protocols and destination ports of these 6,847 flows, an alarming reality emerges:

```
Total Flows in Email Captures: 6,847
  ├─ UDP Transport Protocol: 6,304 flows (92.07%)  <── Anomalous for Email!
  ├─ TCP Transport Protocol:   532 flows ( 7.77%)  <── True Transport for Email
  └─ ICMP / IGMP Telemetry :    11 flows ( 0.16%)

Top Destination Ports across the 6,847 Email Flows:
  1. Port 5355 (LLMNR - Link-Local Multicast Name Resolution) : 5,135 flows (75.00%)
  2. Port 53   (DNS - Standard Domain Name Resolution)       :   368 flows ( 5.37%)
  3. Port 137  (NetBIOS Name Service)                        :   245 flows ( 3.58%)
  4. Port 465  (SMTPS - Secure SMTP Mail Submission)         :   250 flows ( 3.65%)
  5. Port 547  (DHCPv6 - Local Address Configuration)        :    95 flows ( 1.39%)
  6. Port 138  (NetBIOS Datagram Service)                    :    78 flows ( 1.14%)
  7. Port 5353 (mDNS - Multicast DNS)                        :    69 flows ( 1.01%)
  8. Port 443  (HTTPS - Webmail API / CDN)                   :    60 flows ( 0.88%)
  9. Port 993  (IMAPS - Secure IMAP Mail Retrieval)          :    50 flows ( 0.73%)
 10. Port 17500 (Dropbox LAN Sync)                           :    42 flows ( 0.61%)
```

### 2.3 Scientific Assessment:
* **Background LAN Noise Dominance:** Over **86% of the flows** in the Email captures are local-link broadcast/multicast queries generated automatically by the operating system (Windows/Linux) while the mail client ran.
* **True Email Volume:** Genuine email transactions (SMTPS port 465, IMAPS port 993, IMAP port 143) account for **only 302 flows (~4.4%)**.
* **Semantic Incoherence:** Calling this class "Browsing" is indefensible. If an ML model is trained on this data, it will not learn the request-response behavior of Web Browsing; it will learn to detect **Windows LLMNR multicast packet bursts**!
* **The UNB Benchmark Distinction:** In the original UNB ISCX benchmark (Scenario B), the dataset creators explicitly maintained separate classes: `BROWSING` (5,000 instances) and `MAIL` (740 instances). Huang et al. (2021) collapsed Email into Browsing, introducing severe label noise.

---

## 3. Multi-Modal Application Decomposition

Modern applications are **multi-modal platforms**. A single application executes multiple distinct activities that possess completely different QoS contracts:

| Application Platform | Sub-Activity / Capture | Observed Physical Behavior | True QoS Contract | Current Dataset Status |
| :--- | :--- | :--- | :--- | :--- |
| **Facebook** | `facebook_audio1a.pcap` | Periodic UDP datagrams ($20\text{ ms}$ IAT, small frames) | **Real-Time (EF)** | Correctly mapped |
| **Facebook** | `facebook_video1a.pcap` | High UDP throughput ($155\text{K pkts}$, large frames) | **Streaming (AF3x)** | Correctly mapped |
| **Facebook** | `facebookchat*.pcapng` (ISCX D2)| Sporadic, small text bursts, long idle gaps | **Browsing / Interactive (AF2x)** | Available in D2 only |
| **Skype** | `skype_audio1a.pcap` | Symmetric bidirectional UDP pacing | **Real-Time (EF)** | Correctly mapped |
| **Skype** | `skype_video1a.pcap` | Heavy bidirectional UDP video | **Real-Time (EF)** | Correctly mapped |
| **Skype** | `skype_files*.pcap` (ISCX D2) | Unidirectional sustained TCP max window | **Bulk Data (AF1x)** | Available in D2 only |
| **Hangouts** | `hangouts_audio1a.pcap` | WebRTC audio UDP stream | **Real-Time (EF)** | Correctly mapped |
| **Hangouts** | `hangouts_video1b.pcapng` | WebRTC video conferencing (PCAPNG) | **Real-Time (EF)** | Correctly mapped |

**Key Finding:** Sub-activities of the same brand **must be treated as distinct traffic types**. Mappping "Skype" as a single entity would mix Real-Time voice with Bulk file transfers.

---

## 4. Evaluation of 3-Class vs. 4-Class Experimental Viability

### Option 1: 4-Class Experiment (Bulk Data, Browsing, Streaming, Real-Time)
* **Status:** **Scientifically Compromised** in the raw dataset.
* **Why:** The Browsing class is 100% Email, 86% of which is local LLMNR/NetBIOS noise, and is confined to only 3 capture groups (`GRP_EMAIL_1`, `GRP_EMAIL_2`, `GRP_VPN_EMAIL_2`).
* **Justification to Keep:** Only justifiable as a **reproducibility baseline (Track A)** to replicate the exact conditions of Huang et al. (2021) and demonstrate its flaws.

### Option 2: 3-Class Pure QoS Experiment (Bulk Data, Streaming, Real-Time)
* **Status:** **Highly Defensible, Clean, and Methodologically Sound (Track B)**.
* **Why:**
  * Eliminates the contaminated Email class entirely.
  * Preserves 10,770 flows across 29 independent capture groups (Bulk: 10, Streaming: 9, Real-Time: 10).
  * Every class has at least 9 capture groups, enabling standard 5-fold `StratifiedGroupKFold`.
  * Reflects clean physical traffic boundaries: conversational voice/video (Real-Time) vs. buffered media (Streaming) vs. high-throughput file transfers (Bulk Data).

---

## 5. Strategic Recommendations

1. **Do Not Relabel Raw Data:** Keep all original files and mappings documented in `research_notes/label_mapping.csv`.
2. **Execute Dual-Track Evaluation in Phase 3:**
   * **Track A (4-Class Baseline):** Includes Email to evaluate baseline performance and illustrate the impact of LLMNR noise.
   * **Track B (3-Class Benchmark):** Excludes Email to establish the true, uncorrupted QoS classification benchmark.
3. **Phase 5 Data Collection:** To evaluate genuine 4-class QoS, collect clean, dedicated HTTP/2 and HTTP/3 HTTPS Web Browsing PCAPs (Chrome, Firefox, Edge).
