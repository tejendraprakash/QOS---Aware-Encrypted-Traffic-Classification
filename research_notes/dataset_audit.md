# Academic Dataset Audit & Research Paper Analysis

**Project Title:** QoS-Aware Encrypted Network Traffic Classification for Next-Generation Networks  
**Target Paper:** *Research on QoS Classification of Network Encrypted Traffic Behavior Based on Machine Learning* (Huang et al., *Electronics* 2021, 10, 1376)  
**Lead Research Engineer:** Agent Coordination Team  
**Date:** November 2026  

---

## 1. Executive Summary & Research Context

Encrypted traffic classification is an indispensable capability in next-generation networks (5G/6G, software-defined networks, and DiffServ IP backbones). Modern transport encryption (TLS 1.3, QUIC/HTTP3, SSH, WireGuard, and IPsec) prevents legacy Deep Packet Inspection (DPI) engines from reading application-layer payloads. Consequently, Quality of Service (QoS) enforcement, admission control, and dynamic resource allocation must rely exclusively on **statistical and behavioral metadata** observed at the packet and flow levels.

This research environment is established to conduct a rigorous, academic-grade audit of the **UNB ISCX VPN-nonVPN Traffic Dataset**, evaluate the baseline methodology proposed by Huang et al. (2021), identify scientific vulnerabilities in existing literature, and lay the groundwork for a flow-level feature extraction pipeline.

---

## 2. In-Depth Analysis of the Research Paper

### 2.1 Paper Metadata & Core Thesis
* **Title:** *Research on QoS Classification of Network Encrypted Traffic Behavior Based on Machine Learning*
* **Authors:** Yung-Fa Huang, Chuan-Bi Lin, Chien-Min Chung, and Ching-Mu Chen (Chaoyang University of Technology & Chung Chou University of Science and Technology)
* **Publication:** *MDPI Electronics*, 2021, Vol. 10, No. 12, Article 1376. DOI: [10.3390/electronics10121376](https://doi.org/10.3390/electronics10121376)
* **Central Proposal:** A **Hybrid Traffic Classification (HTC)** framework combining:
  1. Deep Packet Inspection (DPI) for unencrypted legacy traffic.
  2. IP reverse DNS / Autonomous System Number (ASN) querying to supplement identification.
  3. Machine learning models (P-MLP, CNN, SAE) trained on truncated raw packet byte representations.
  4. A **Majority Voting strategy** across $K$ consecutive packets ($K \in \{1, 5, 9, 13, 17\}$) within a connection, discarding the first 3 TCP handshake packets to improve prediction stability.
  5. Dynamic Differentiated Services Code Point (DSCP) marking on Edge Routers according to Cisco QoS baseline recommendations.

### 2.2 Four Target QoS Classes in the Paper
The paper aggregates application traffic into four primary DiffServ QoS behavior classes:
1. **Bulk Data (Class 1 / AF1x / CS1):** High-volume file transfers where throughput matters more than latency (e.g., FTPS, SCP downloads).
2. **Browsing / Interactive (Class 2 / AF2x / CS2):** Web browsing, email (SMTP/IMAP), and instant text messaging (Discord, Line, Messenger text).
3. **Streaming Media (Class 3 / AF3x / CS3):** Unidirectional buffered multimedia (YouTube, Netflix, Vimeo, Spotify, Facebook video). Tolerates initial buffering delays but requires steady bandwidth.
4. **Real-Time Interactive (Class 4 / EF / CS4/CS5):** Low-latency, jitter-sensitive voice and video conferencing (VoIP Buster, Skype audio/video, Hangouts audio/video, Facebook audio). Packet delays $> 150\text{ ms}$ or jitter degrades user experience severely.

---

### 2.3 Critical Scientific Evaluation & Gaps in Huang et al. (2021)

While the paper presents an interesting operational system architecture, a rigorous academic audit reveals several **major methodological flaws, dataset vulnerabilities, and evaluation limitations**:

```
+--------------------------------------------------------------------------------------------------+
|                              CRITICAL WEAKNESSES IN HUANG ET AL. (2021)                         |
+--------------------------------------------------------------------------------------------------+
| 1. Artificial Byte-Level Truncation vs. Real Statistical Traffic Dynamics                        |
|    - The paper feeds raw truncated packet headers/bytes (first N bytes) into P-MLP and CNN.      |
|    - Risk: The neural network easily overfits to static IP addresses, MACs, or ephemeral port     |
|      artifacts rather than genuine transport dynamics (packet sizes, IAT, burstiness).          |
+--------------------------------------------------------------------------------------------------+
| 2. Private IP / DNS Query Collapse in Bulk Data                                                  |
|    - As admitted in Table 10 of the paper, FTPS and SCP servers used private IPs (10.10.39.8),  |
|      causing DNS reverse and ASN lookups to return "N/A". In an enterprise or VPN setting,       |
|      IP/ASN querying is completely ineffective for internal or tunnel endpoints.                 |
+--------------------------------------------------------------------------------------------------+
| 3. Packet-Level Evaluation vs. Flow-Level Coherence                                              |
|    - The paper evaluates accuracy primarily on sliced individual packets with a K-packet         |
|      heuristic vote, rather than complete bidirectional flow statistics.                         |
|    - This distorts real classification performance, as packets from the same flow are split      |
|      across train/test sets, leading to severe synthetic data leakage.                           |
+--------------------------------------------------------------------------------------------------+
| 4. Extreme Dataset Byte Skew & Lack of Class Balance Controls                                    |
|    - The authors subsampled fixed packet quotas (e.g., 6000 FTPS, 1750 Email) without accounting  |
|      for underlying flow durations or burst structures. The raw captures are wildly imbalanced.  |
+--------------------------------------------------------------------------------------------------+
| 5. Tunnel Blindness Under VPN                                                                    |
|    - When VPN encapsulation is enabled (e.g., OpenVPN UDP/TCP), outer packet headers only show   |
|      VPN server IP and port. Payload inspection and IP/ASN lookup fail completely.               |
+--------------------------------------------------------------------------------------------------+
```

### 2.4 Opportunities for Improving Classification Accuracy in Our Research
To outperform and improve upon the existing research paper, our next phase will introduce:
1. **True Bidirectional Flow Representation:** Extract comprehensive statistical features across both forward and backward directions (packet size distributions, inter-arrival time moments, active/idle periods).
2. **Robust Multi-Feature Sets:** In place of raw packet bytes, compute payload-agnostic statistical metrics (mean, std, skewness, min, max, percentiles) and burst profiles that remain valid across TLS 1.3 and VPN tunnels.
3. **Leakage-Free Validation:** Split training and test sets strictly at the **session / capture file level** (group k-fold cross-validation), ensuring packets from the same flow never contaminate test folds.
4. **Early-Flow vs. Full-Flow Trade-off Analysis:** Measure classification accuracy using only the first $N$ packets ($N \in \{5, 10, 20\}$) to achieve ultra-fast QoS classification without waiting for flow completion.

---

## 3. Comprehensive Dataset Audit

### 3.1 Dataset Overview & Provenance
* **Dataset:** UNB ISCX VPN-nonVPN Traffic Dataset (University of New Brunswick, Canadian Institute for Cybersecurity).
* **Storage Location:** `data/raw/` (mirrored from original `Dataset/` directory).
* **Original Captures Kept Untouched:** The raw files in `Dataset/` remain preserved and unmodified.
* **Total Capture Files:** 37 files.
* **Total Volume:** ~13.20 GB (14,172,125,123 bytes).

### 3.2 Link-Layer Diversity & Parser Traps
A critical discovery during our binary header audit:
* **Standard Captures (25 files):** Encapsulated in standard Ethernet frames (`linktype = 1`, `DLT_EN10MB`). The IP header starts at byte offset 14.
* **VPN Captures (10 files):** Captured on virtual `tun`/`tap` adapters, resulting in `linktype = 101` (`DLT_RAW`). Packets start **directly with the IPv4 header** at byte 0. A naive PCAP analyzer assuming Ethernet offset 14 fails or produces garbage headers for all VPN files!
* **PCAPNG Format Captures (2 files):** `hangouts_video1b.pcapng` (388.5 MB) and `voipbuster1b.pcapng` (5.8 MB) use the Next Generation PCAP format (`pcapng`), requiring block-level section parsing (`RawPcapNgReader`) rather than legacy libpcap 24-byte global headers.

### 3.3 Extreme Imbalance & Volume Skew
* **Massive Bulk Dominance:** Two capture files (`ftps_down_1a.pcap` at 5.43 GB and `ftps_down_1b.pcap` at 4.23 GB) account for **9.66 GB (over 73%)** of the entire dataset volume!
* **Micro-Captures / Anomalies:** `scpDown4.pcap` is merely 96 KB (~620 packets), indicating an aborted transfer or connection drop.
* **Implication for Training:** Training models on raw packet distributions without flow normalization will cause the classifier to predict "Bulk Data" almost unconditionally.

### 3.4 Suffix Symmetry & Capture Duplicates
Captures feature paired naming conventions:
* `email1a.pcap` & `email1b.pcap`
* `email2a.pcap` & `email2b.pcap`
* `ftps_down_1a.pcap` & `ftps_down_1b.pcap`
* `ftps_up_2a.pcap` & `ftps_up_2b.pcap`
* `vpn_ftps_A.pcap` & `vpn_ftps_B.pcap`
These paired captures represent either synchronized dual-ended vantage points (client and server side) or repeated runs of identical transfer scripts. They must not be split naively across train and test sets.

---

## 4. QoS Label Mapping & Ambiguity Analysis

The configurable mapping file is maintained in `research_notes/label_mapping.csv`.

| Application | Captures | Proposed QoS Class | Confidence | Key Ambiguity / Justification |
| :--- | :---: | :--- | :---: | :--- |
| **FTPS (Up/Down)** | 6 | Bulk Data | High | Delay-tolerant, sustained high-volume bulk transfer over TLS control/data channels. |
| **SCP Download** | 6 | Bulk Data | High | SSH file transfer. `scpDown4.pcap` requires verification due to miniature 96 KB size. |
| **Email** | 6 | Browsing | **Low (Flagged)** | **High Ambiguity:** Text reading/headers behaves like Browsing; large attachments behave like Bulk Data. |
| **Facebook Video**| 1 | Streaming | Medium | Buffered video playback, but if live video conferencing, crosses into Real-Time. |
| **YouTube** | 2 | Streaming | High | Adaptive bitrate DASH/HTTP streaming. High bandwidth, delay-tolerant initial buffering. |
| **Netflix** | 2 | Streaming | High | Adaptive bitrate buffered media streaming over TLS. |
| **Vimeo** | 2 | Streaming | High | Video on demand streaming over HTTPS. |
| **Spotify** | 2 | Streaming | High | Audio streaming chunks over HTTP/HTTPS. |
| **VoIP Buster** | 2 | Real-Time | High | SIP/RTP interactive voice. High jitter sensitivity ($< 150\text{ ms}$). |
| **Skype Audio/Video**| 3 | Real-Time | High | Interactive two-way voice/video communication. Strict real-time QoS. |
| **Hangouts Audio/Video**| 3 | Real-Time | High | WebRTC/interactive conferencing. Real-time QoS class. |
| **Facebook Audio** | 2 | Real-Time | High | Voice call over Messenger. Interactive real-time communication. |

---

## 5. Flow Definition & Theoretical Constraints

### 5.1 Flow Abstraction
In `scripts/pcap_analyzer.py`, a flow is canonically defined as a **bidirectional 5-tuple**:
$$\text{Flow Key} = \left( \min(IP_A:Port_A, IP_B:Port_B), \max(IP_A:Port_A, IP_B:Port_B), \text{Protocol} \right)$$

### 5.2 Scientific Limitations: 5-Tuples vs. Application Sessions
In academic literature, it is an elementary mistake to equate 5-tuple network flows with user application sessions:
1. **Multi-Flow Modern Applications:** A single application session (e.g., playing a YouTube video or opening an email) initiates dozens of distinct 5-tuples: DNS lookups, TLS control handshakes, CDN content delivery connections, authentication tokens, and analytical telemetry.
2. **Ephemeral Port Churn & Re-use:** Without active session timeout slicing (e.g., 60-second inactivity thresholds), multiple independent application requests spaced minutes apart may share the same 5-tuple if ephemeral ports are re-used.
3. **Tunnel Multiplexing:** In VPN captures, all traffic from multiple internal applications is multiplexed through a single outer tunnel 5-tuple (e.g., OpenVPN UDP on port 1194). Outer 5-tuples represent aggregate tunnel flows, not individual application streams.

---

## 6. Next Steps for Flow-Level Feature Extraction

1. **Implement Inactivity Flow Slicing:** Segment packet streams into discrete bidirectional flows using a standard 60-second or 120-second inter-packet silence threshold.
2. **Statistical Feature Engineering (Non-Payload):**
   - Packet Length Metrics: $\mu, \sigma, \text{skew}, \min, \max, Q_{25}, Q_{50}, Q_{75}$ (Forward, Backward, Bidirectional).
   - Inter-Arrival Times (IAT): $\mu_{IAT}, \sigma_{IAT}, \max_{IAT}, \min_{IAT}$.
   - Volume & Ratio Dynamics: Forward/Backward packet ratio, Forward/Backward byte ratio, bytes per second, packets per second.
   - Initial Burst Signatures: Size and direction of the first $N$ packets (representing handshake and initial application negotiation).
3. **Stratified Group Cross-Validation:** Prevent data leakage by isolating whole capture sessions into distinct folds.
4. **Machine Learning Model Benchmark:** Compare lightweight models (Random Forest, XGBoost) against deep learning architectures (1D-CNN, MLP) to evaluate classification accuracy, inference latency, and computational feasibility on next-generation edge routers.
