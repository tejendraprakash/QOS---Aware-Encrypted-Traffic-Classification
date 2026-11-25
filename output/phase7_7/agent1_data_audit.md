# Phase 7.7 Data and Label Audit Report
**Dataset Investigated**: `output/phase6d/clean_track_b_flows.csv`  
**Auditor**: SUBAGENT 1 — Data and Label Auditor  
**Date**: November 24, 2026  

---

## 1. Executive Summary

This audit provides a comprehensive investigation into the data quality, label integrity, class overlap, and architectural biases present in `output/phase6d/clean_track_b_flows.csv`. The dataset comprises **4,196 network flows** extracted across **29 capture groups**, mapped into three Quality of Service (QoS) classes: **Real-Time**, **Bulk Data**, and **Streaming**.

### Key Findings
1. **Severe Background Traffic Contamination in Bulk Data**:
   - The capture group `GRP_FTPS_DOWN_1` accounts for **871 out of 1,177 Bulk Data flows (74.00%)**.
   - Within `GRP_FTPS_DOWN_1`, **776 flows (89.1%) are UDP flows**, of which **664 flows have exactly 2 packets** with a median payload of 448 bytes, communicating with BitTorrent/DHT ports (6881, 51413).
   - Actual FTPS file transfers operate over TCP (represented by only 95 flows). The remaining 776 UDP flows are ephemeral background peer-to-peer/DHT probes mistakenly swept into the capture file and labeled as Bulk Data.
   - As a direct consequence, **72.9% of the Bulk Data class (858/1,177 flows) is UDP**, severely confounding Bulk Data with Real-Time VoIP/signaling.
2. **Dominance of Degenerate Short Flows**:
   - **42.37% (1,778 flows)** contain **$\le 2$ packets**, and **56.55% (2,373 flows)** contain **$\le 5$ packets**.
   - For a 2-packet flow, higher-order statistical features (inter-arrival time variance, skewness, kurtosis, burstiness, Fourier coefficients) are degenerate or undefined.
   - Empirical Out-Of-Fold (OOF) classification accuracy on $\le 2$-packet flows is **51.63%** (with 420 Bulk Data flows misclassified as Real-Time and 344 Real-Time flows misclassified as Bulk Data). Conversely, on flows with $> 10$ packets, accuracy rises to **61.53%** with high Bulk Data separation.
3. **100% Homogeneous Capture Groups & Extreme Leakage Risk**:
   - All **29 capture groups are 100% homogeneous** (0 mixed groups). Every capture group represents a single recording session of a single application.
   - Any standard random cross-validation will cause catastrophic data leakage by allowing models to memorize capture-specific host artifacts (IP subnets, ephemeral port ranges, OS TCP window parameters, host MTU).
   - Strict `GroupKFold` evaluation is mandatory, but GroupKFold currently produces severe fold imbalances: **Fold 0 contains 80.1% of all Bulk Data flows**, while Fold 4 contains **0% Bulk Data flows**.
4. **Cross-Class Ambiguity & Labeling Nuances**:
   - Interactive communications (Skype Video, Hangouts Video) are categorized under `Real-Time`, whereas stored video on-demand (Facebook Video, YouTube, Netflix, Vimeo) is categorized under `Streaming`. While conceptually sound from latency SLA perspectives, short signaling/handshake flows across both video categories are statistically indistinguishable.
   - `GRP_FB_VIDEO_1` (100 flows) and `GRP_SCP_4` (1 flow) carry `label_confidence = Medium` (101 flows total, 2.41%).

---

## 2. Dataset Verification and Overall Distributions

### 2.1 Flow & Group Totals
- **Total Flows**: 4,196
- **Total Capture Groups**: 29
- **Total Applications**: 15
- **Feature Columns**: 105 statistical flow features + 15 metadata/identifier columns.
- **Missing Values**: 0 across all 4,196 rows and 120 columns.
- **Zero Variance Columns**: 2 (`tcp_urg_count` = 0 across all flows; `is_noise` = False across all flows).

### 2.2 QoS Class Breakdown
| QoS Class | Flow Count | Percentage (%) | Primary Applications |
| :--- | :---: | :---: | :--- |
| **Real-Time** | 2,122 | 50.57% | Skype Audio (976), Facebook Audio (445), VoIP Buster (436), Hangouts Audio (172), Hangouts Video (67), Skype Video (26) |
| **Bulk Data** | 1,177 | 28.05% | FTPS Download (871), FTPS Upload (181), FTPS (113), SCP Download (12) |
| **Streaming** | 897 | 21.38% | Netflix (235), YouTube (230), Spotify (200), Vimeo (132), Facebook Video (100) |
| **Total** | **4,196** | **100.00%** | |

### 2.3 Protocol & VPN Bias
| QoS Class | TCP Flows | UDP Flows | % UDP | VPN: No | VPN: Yes | % VPN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Real-Time** | 846 | 1,276 | 60.13% | 670 | 1,452 | 68.43% |
| **Bulk Data** | 319 | 858 | 72.90% | 1,064 | 113 | 9.60% |
| **Streaming** | 761 | 136 | 15.16% | 408 | 489 | 54.52% |
| **Total** | **1,926** | **2,270** | **54.10%** | **2,142** | **2,054** | **48.95%** |

> **Critical Bias Observation**: Bulk Data is **90.4% Non-VPN** and **72.9% UDP** (due to background noise), whereas Real-Time is **68.4% VPN** and Streaming is **84.8% TCP**. Unregularized models can exploit the `is_vpn` status or protocol type as spurious shortcuts rather than learning true QoS flow dynamics.

---

## 3. Capture Group Architecture & Homogeneity Analysis

### 3.1 Complete Capture Group Inventory
Every single one of the 29 capture groups is **100% homogeneous** with respect to its QoS class and application:

| Capture Group | Flows | QoS Class | Application | VPN | TCP / UDP | $\le 2$ pkts (%) | $\le 5$ pkts (%) | Med Bytes | Med Dur (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `GRP_VPN_SKYPE_AUDIO_1` | 890 | Real-Time | Skype Audio | Yes | 171 / 719 | 70.4% | 84.0% | 220 | 0.12 |
| `GRP_FTPS_DOWN_1` | 871 | Bulk Data | FTPS Download | No | 95 / 776 | 76.7% | 87.1% | 448 | 0.15 |
| `GRP_VOIPBUSTER_1` | 370 | Real-Time | VoIP Buster | No | 30 / 340 | 72.7% | 83.5% | 448 | 0.10 |
| `GRP_VPN_FB_AUDIO_2` | 363 | Real-Time | Facebook Audio | Yes | 259 / 104 | 18.7% | 60.9% | 352 | 0.08 |
| `GRP_FTPS_UP_2` | 181 | Bulk Data | FTPS Upload | No | 104 / 77 | 33.1% | 38.7% | 1,607 | 0.01 |
| `GRP_VPN_YOUTUBE_A` | 174 | Streaming | YouTube | Yes | 116 / 58 | 0.0% | 9.8% | 4,326 | 32.49 |
| `GRP_VPN_SPOTIFY_A` | 134 | Streaming | Spotify | Yes | 109 / 25 | 13.4% | 21.6% | 3,628 | 29.25 |
| `GRP_VPN_NETFLIX_A` | 133 | Streaming | Netflix | Yes | 122 / 11 | 0.0% | 1.5% | 780 | 81.58 |
| `GRP_VPN_HANGOUTS_AUDIO_1`| 133 | Real-Time | Hangouts Audio | Yes | 129 / 4 | 0.0% | 0.8% | 8,769 | 59.11 |
| `GRP_NETFLIX_1` | 102 | Streaming | Netflix | No | 102 / 0 | 0.0% | 2.0% | 520 | 180.26 |
| `GRP_FB_VIDEO_1` | 100 | Streaming | Facebook Video | No | 65 / 35 | 12.0% | 44.0% | 868 | 0.41 |
| `GRP_SKYPE_AUDIO_1` | 86 | Real-Time | Skype Audio | No | 64 / 22 | 23.3% | 32.6% | 468 | 28.91 |
| `GRP_VIMEO_1` | 84 | Streaming | Vimeo | No | 84 / 0 | 0.0% | 6.0% | 3,865 | 18.83 |
| `GRP_FB_AUDIO_1` | 82 | Real-Time | Facebook Audio | No | 56 / 26 | 14.6% | 58.5% | 488 | 0.16 |
| `GRP_VPN_FTPS_A` | 70 | Bulk Data | FTPS | Yes | 66 / 4 | 0.0% | 2.9% | 249,638 | 2.84 |
| `GRP_HANGOUTS_VIDEO_1` | 67 | Real-Time | Hangouts Video | No | 65 / 2 | 0.0% | 37.3% | 1,358 | 0.12 |
| `GRP_SPOTIFY_1` | 66 | Streaming | Spotify | No | 65 / 1 | 4.5% | 4.5% | 809 | 132.80 |
| `GRP_VPN_VOIPBUSTER_1` | 66 | Real-Time | VoIP Buster | Yes | 24 / 42 | 13.6% | 54.5% | 323 | 0.36 |
| `GRP_YOUTUBE_1` | 56 | Streaming | YouTube | No | 56 / 0 | 0.0% | 1.8% | 505 | 59.71 |
| `GRP_VPN_VIMEO_A` | 48 | Streaming | Vimeo | Yes | 42 / 6 | 0.0% | 2.1% | 697 | 18.87 |
| `GRP_VPN_FTPS_B` | 43 | Bulk Data | FTPS | Yes | 42 / 1 | 0.0% | 2.3% | 292,817 | 4.81 |
| `GRP_HANGOUTS_AUDIO_1` | 39 | Real-Time | Hangouts Audio | No | 33 / 6 | 0.0% | 0.0% | 5,606 | 58.95 |
| `GRP_SKYPE_VIDEO_1` | 26 | Real-Time | Skype Video | No | 15 / 11 | 38.5% | 69.2% | 553 | 0.24 |
| `GRP_SCP_3` | 3 | Bulk Data | SCP Download | No | 3 / 0 | 0.0% | 0.0% | 5,906,826 | 10.45 |
| `GRP_SCP_1` | 3 | Bulk Data | SCP Download | No | 3 / 0 | 0.0% | 0.0% | 6,366,419 | 11.08 |
| `GRP_SCP_5` | 2 | Bulk Data | SCP Download | No | 2 / 0 | 0.0% | 0.0% | 5,475,340 | 9.94 |
| `GRP_SCP_2` | 2 | Bulk Data | SCP Download | No | 2 / 0 | 0.0% | 0.0% | 5,892,168 | 10.29 |
| `GRP_SCP_6` | 1 | Bulk Data | SCP Download | No | 1 / 0 | 0.0% | 0.0% | 4,964,285 | 8.89 |
| `GRP_SCP_4` | 1 | Bulk Data | SCP Download | No | 1 / 0 | 0.0% | 0.0% | 5,753,042 | 9.87 |

### 3.2 Concentration Analysis
- **Top 2 Groups**: `GRP_VPN_SKYPE_AUDIO_1` (890) and `GRP_FTPS_DOWN_1` (871) comprise **1,761 flows (41.97%)**.
- **Top 4 Groups**: Comprise **2,494 flows (59.44%)**.
- **Bottom 6 Groups** (`GRP_SCP_1` through `GRP_SCP_6`): Contain only **12 flows (0.29%)** combined.
- **Fold-Level Imbalance in Phase 6d GroupKFold**:
  - Fold 0: 943 Bulk Data flows (80.1% of entire class).
  - Fold 1: 9 Bulk Data flows.
  - Fold 2: 182 Bulk Data flows.
  - Fold 3: 43 Bulk Data flows.
  - Fold 4: **0 Bulk Data flows**.

---

## 4. Deep Dive: FTPS Download Dominance & Noise Contamination

### 4.1 Statistical Profiling of `GRP_FTPS_DOWN_1`
`GRP_FTPS_DOWN_1` contains 871 flows, representing **74.00% of all Bulk Data**:
- **UDP Flows**: 776 (89.09%)
- **TCP Flows**: 95 (10.91%)

Comparing the two protocols inside `GRP_FTPS_DOWN_1`:
| Metric | TCP Flows ($N=95$) | UDP Flows ($N=776$) |
| :--- | :---: | :---: |
| **Total Packets (Median)** | 65.0 | 2.0 |
| **Total Packets (Mean)** | 80,870.5 | 5.3 |
| **Total Packets (Max)** | 2,429,639 | 960 |
| **Total Bytes (Median)** | 23,461 bytes | 448 bytes |
| **Total Bytes (Mean)** | 60,419,430 bytes (~60.4 MB) | 727 bytes |
| **Total Bytes (Max)** | 2,396,607,000 bytes (~2.4 GB) | 128,155 bytes |
| **Duration (Median)** | 0.55 s | 0.08 s |
| **$\le 2$ Packets** | 4 flows (4.2%) | **664 flows (85.6%)** |
| **Primary Destination Ports** | 21 (FTP Control), 443, dynamic TCP | **6881 (BitTorrent: 111)**, 51413 (Transmission: 33), ephemeral UDP |

### 4.2 Impact on Model Classification
Because 776 UDP background flows were labeled as "Bulk Data":
1. **Misclassification of FTPS Out-of-Fold**:
   When `GRP_FTPS_DOWN_1` is evaluated out-of-fold in Fold 0:
   - **60.70% of its UDP flows are predicted as Real-Time**.
   - Only 39.18% are predicted as Bulk Data.
   - Its TCP flows: 43.16% predicted as Streaming, 35.79% as Real-Time, and only 21.05% as Bulk Data.
2. **Catastrophic Misclassification of VoIP Buster**:
   In Fold 3, `GRP_VOIPBUSTER_1` (N=370, Real-Time) was evaluated after training on folds containing `GRP_FTPS_DOWN_1`. Because both VoIP Buster and the FTPS noise consist of 2-packet UDP exchanges with 448 bytes:
   - **322 out of 370 VoIP Buster flows (87.03%) were predicted as Bulk Data!**
   - Out-of-fold accuracy for `GRP_VOIPBUSTER_1` collapsed to **6.76%**.

---

## 5. Short-Flow Analysis & Feature Degeneration

### 5.1 Cumulative Packet and Duration Regimes
Across all 4,196 flows:
- **$\le 2$ packets**: 1,778 flows (42.37%)
- **$\le 3$ packets**: 2,052 flows (48.90%)
- **$\le 5$ packets**: 2,373 flows (56.55%)
- **$\le 10$ packets**: 2,904 flows (69.21%)
- **Duration $\le 0.1$ s**: 1,366 flows (32.55%)
- **Duration $\le 1.0$ s**: 2,478 flows (59.06%)

### 5.2 Flow Size vs. Classification Performance
Analyzing the empirical out-of-fold predictions from Phase 6d demonstrates that model error is concentrated almost entirely in short flows:

| Flow Size Regime | Total Flows | Overall Accuracy | Bulk Data Acc | Real-Time Acc | Streaming Acc |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **$\le 2$ Packets** | 1,778 (42.4%) | **51.63%** | 42.47% (310/730) | 59.90% (608/1015) | 0.00% (0/33) |
| **$\le 5$ Packets** | 2,373 (56.6%) | **50.61%** | 40.72% (340/835) | 58.16% (834/1434) | 25.96% (27/104) |
| **$\le 10$ Packets**| 2,904 (69.2%) | **50.03%** | 40.77% (358/878) | 52.70% (869/1649) | 59.95% (226/377) |
| **$> 10$ Packets** | 1,292 (30.8%) | **61.53%** | **73.24%** (219/299) | 48.20% (228/473) | **66.92%** (348/520) |

> **Diagnostic Insight**: On flows with $\le 2$ packets, 420 Bulk Data flows are misclassified as Real-Time and 344 Real-Time flows are misclassified as Bulk Data. Statistical features computed on 2 packets lack the discriminative capacity to separate transfer categories.

---

## 6. Class Ambiguity & Labeling Nuances

### 6.1 Interactive Video (Real-Time) vs. Video Streaming (Streaming)
- **Skype Video** ($N=26$) and **Hangouts Video** ($N=67$) are assigned to `Real-Time`.
  - Skype Video: 38.5% $\le 2$ packets, median duration 0.24s.
  - Hangouts Video: 0% $\le 2$ packets, 37.3% $\le 5$ packets, median duration 0.12s, 65 TCP / 2 UDP.
- **Facebook Video** ($N=100$) is assigned to `Streaming` with `label_confidence = Medium`.
  - 12.0% $\le 2$ packets, 44.0% $\le 5$ packets, median duration 0.41s.
- **Underlying Mechanics**: Interactive video demands ultra-low latency ($<150\text{ ms}$) without playback buffering, whereas video streaming tolerates multi-second buffers. However, during initial flow negotiation (TLS ClientHello, STUN/TURN binding, HTTP GET), both interactive and streaming video exhibit identical transport signatures.

### 6.2 Medium Confidence Label Review
Only two capture groups contain flows with `label_confidence = Medium`:
1. `GRP_FB_VIDEO_1` (100 flows, Streaming): High proportion of short-lived TCP flows; potential overlap with web browsing or interactive feed scrolling.
2. `GRP_SCP_4` (1 flow, Bulk Data): A single isolated SCP transfer flow.

---

## 7. Dataset Leakage & Feature Correlation Risks

Univariate ANOVA F-statistic analysis between features, `capture_group`, and `qos_class` reveals features with high capture-group specificity relative to QoS class separation:

| Feature | F-Statistic (Capture Group) | F-Statistic (QoS Class) | Group / QoS Ratio | Leakage / Bias Mechanism |
| :--- | :---: | :---: | :---: | :--- |
| `tcp_rst_count` | 61.45 | 2.45 | **25.06** | Specific host/server tear-down behavior during recording. |
| `fwd_pkt_ratio` | 20.39 | 3.69 | **5.52** | Asymmetric routing/capture artifacts specific to network setup. |
| `fwd_bytes` | 15.07 | 5.37 | **2.81** | Total payload size determined by file chosen in specific test. |
| `udp_byte_count`| 2.50 | 0.94 | **2.65** | Driven by FTPS background UDP noise. |
| `tcp_win_max` | 51.10 | 26.20 | **1.95** | Host OS TCP socket buffer configuration. |
| `tcp_win_std` | 70.64 | 40.15 | **1.76** | Host OS TCP window scaling algorithms. |

---

## 8. Recommendations for Feature & Model Development

To address the empirical realities uncovered in this audit while strictly respecting the constraint that labels must not be changed and flows must not be deleted:

1. **Stratified Group Partitioning**:
   - Rather than arbitrary GroupKFold that places 80.1% of Bulk Data in Fold 0, future CV strategies must partition capture groups such that large groups (`GRP_FTPS_DOWN_1`, `GRP_VPN_SKYPE_AUDIO_1`, `GRP_VOIPBUSTER_1`) are balanced across folds.
2. **Length-Regime or Hierarchical Modeling**:
   - Short flows ($\le 5$ packets) and long flows ($> 10$ packets) have fundamentally different statistical properties.
   - Long flows achieve $>73\%$ Bulk Data and $>67\%$ Streaming accuracy with standard temporal features.
   - Short flows require specialized early-packet features (e.g., initial packet lengths, directional bytes in first 3 packets) or dedicated sub-models rather than relying on degenerate whole-flow variance/skewness.
3. **Pruning of Host-Stack and Capture Artifacts**:
   - Features exhibiting high capture-group association without physical QoS rationale (`tcp_rst_count`, raw `tcp_win_max`, raw byte sums) should be regularized, normalized, or excluded to prevent shortcut memorization.
4. **Protocol & Directional Disentanglement**:
   - Explicitly engineer features that capture bidirectional transfer asymmetry (e.g. forward-to-backward byte ratio and sustained packet rate) rather than relying on transport protocol flags (`is_udp`), preventing the model from confusing UDP background noise with VoIP audio.
5. **Robust Loss & Cost-Sensitive Weighting**:
   - Given the substantial noise in short Bulk Data flows, models should incorporate label-smoothing or loss penalties that prevent overconfident predictions on degenerate 2-packet flows.
