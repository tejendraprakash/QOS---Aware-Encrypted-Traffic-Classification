# Track B Dataset Specification: Primary 3-Class Benchmark

**Project:** QoS-Aware Encrypted Network Traffic Classification for Next-Generation Networks  
**Phase:** 5 — Baseline Experiment Preparation & Protocol Validation  
**Date:** November 2026  
**Dataset Artifact:** [track_b_flows.csv](file:///c:/Users/saipr/OneDrive/Desktop/EncryptedTrafficResearch/output/phase5/track_b_flows.csv)  
**Fold Artifact:** [track_b_folds.csv](file:///c:/Users/saipr/OneDrive/Desktop/EncryptedTrafficResearch/output/phase5/track_b_folds.csv)  
**Analysis Table:** [track_b_fold_analysis.csv](file:///c:/Users/saipr/OneDrive/Desktop/EncryptedTrafficResearch/output/phase5/track_b_fold_analysis.csv)

---

## 1. Track Overview & Purpose

Track B serves as the **Primary 3-Class Benchmark** for our research project. It excludes the contaminated `Browsing` (Email) category, focusing exclusively on the three rigorously validated QoS traffic categories:
1. **Bulk Data:** High-throughput, non-interactive, loss-sensitive file transfers (FTPS, SCP).
2. **Streaming:** High-throughput, buffer-tolerant video/audio streaming (Netflix, Spotify, Vimeo, YouTube).
3. **Real-Time:** Low-latency, jitter-sensitive conversational voice and video conferencing (Hangouts, Skype, VoIPBuster, Facebook Audio/Video).

By removing the 6,847 Email flows (which were 75% local broadcast/multicast LLMNR noise), Track B establishes a scientifically defensible, noise-free foundation for QoS policy mapping.

---

## 2. Dataset Dimensions & Class Distribution

- **Total Flows:** 10,770
- **Total Columns:** 117 (14 metadata/audit, 103 behavioral ML-safe candidates)
- **Total Capture Groups:** 29 unique capture sessions
- **Imbalance Ratio:** $6.30 : 1$ (Real-Time vs. Streaming)

### Class Distribution Summary

| Class Name | Flow Count | Percentage (%) | Capture Groups | Major Applications |
|---|---|---|---|---|
| **Real-Time** | 8,132 | 75.51% | 10 groups | Hangouts Audio/Video, Skype Audio/Video, VoIPBuster, Facebook Audio/Video |
| **Bulk Data** | 1,348 | 12.52% | 10 groups | FTPS Upload/Download, SCP (1 through 6) |
| **Streaming** | 1,290 | 11.98% | 9 groups | Netflix, Spotify, Vimeo, YouTube |
| **Total** | **10,770** | **100.0%** | **29 groups** | — |

---

## 3. Capture Diversity Analysis

Unlike Track A where Browsing was bottlenecked to 3 groups, Track B displays balanced capture diversity across all three classes:
- **Bulk Data:** 10 capture groups (e.g., FTPS, SCP across multiple independent runs)
- **Streaming:** 9 capture groups (e.g., Netflix, Spotify, Vimeo, YouTube across standard and VPN captures)
- **Real-Time:** 10 capture groups (e.g., Hangouts, Skype, VoIPBuster, Facebook across standard and VPN captures)

This 9–10 group distribution enables a standard **5-Fold Cross-Validation** scheme where each fold receives approximately 2 capture groups per class.

---

## 4. Validated 5-Fold Stratified Group K-Fold Protocol

Track B is evaluated using:
$$\text{StratifiedGroupKFold}(n\_splits=5, shuffle=True, random\_state=2)$$

### Partition Statistics

| Fold | Partition | Total Samples | Bulk Data | Real-Time | Streaming | Groups Count | Group Overlap | Class Coverage |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **0** | **Train** | 9,047 | 1,138 (12.6%) | 7,031 (77.7%) | 878 (9.7%) | 21 groups | **0** | **100% (3/3)** |
| **0** | **Val** | 1,723 | 210 (12.2%) | 1,101 (63.9%) | 412 (23.9%) | 8 groups | **0** | **100% (3/3)** |
| **1** | **Train** | 9,373 | 1,274 (13.6%) | 7,225 (77.1%) | 874 (9.3%) | 24 groups | **0** | **100% (3/3)** |
| **1** | **Val** | 1,397 | 74 (5.3%) | 907 (64.9%) | 416 (29.8%) | 5 groups | **0** | **100% (3/3)** |
| **2** | **Train** | 6,513 | 1,285 (19.7%) | 4,054 (62.2%) | 1,174 (18.0%) | 23 groups | **0** | **100% (3/3)** |
| **2** | **Val** | 4,257 | 63 (1.5%) | 4,078 (95.8%) | 116 (2.7%) | 6 groups | **0** | **100% (3/3)** |
| **3** | **Train** | 8,638 | 1,318 (15.3%) | 6,189 (71.6%) | 1,131 (13.1%) | 24 groups | **0** | **100% (3/3)** |
| **3** | **Val** | 2,132 | 30 (1.4%) | 1,943 (91.1%) | 159 (7.5%) | 5 groups | **0** | **100% (3/3)** |
| **4** | **Train** | 9,509 | 377 (4.0%) | 8,029 (84.4%) | 1,103 (11.6%) | 24 groups | **0** | **100% (3/3)** |
| **4** | **Val** | 1,261 | 971 (77.0%) | 103 (8.2%) | 187 (14.8%) | 5 groups | **0** | **100% (3/3)** |

### Validation Group Assignments
- **Fold 0 Validation Groups (8):** `GRP_FTPS_UP_2`, `GRP_HANGOUTS_VIDEO_1`, `GRP_NETFLIX_1`, `GRP_SCP_6`, `GRP_SKYPE_VIDEO_1`, `GRP_VIMEO_1`, `GRP_VPN_SKYPE_AUDIO_1`, `GRP_YOUTUBE_1`
- **Fold 1 Validation Groups (5):** `GRP_HANGOUTS_AUDIO_1`, `GRP_VPN_FTPS_A`, `GRP_VPN_NETFLIX_A`, `GRP_VPN_VIMEO_A`, `GRP_VPN_YOUTUBE_A`
- **Fold 2 Validation Groups (6):** `GRP_FB_AUDIO_1`, `GRP_SCP_3`, `GRP_SPOTIFY_1`, `GRP_VPN_FTPS_B`, `GRP_VPN_HANGOUTS_AUDIO_1`, `GRP_VPN_VOIPBUSTER_1`
- **Fold 3 Validation Groups (5):** `GRP_SCP_1`, `GRP_SCP_5`, `GRP_VOIPBUSTER_1`, `GRP_VPN_FB_AUDIO_2`, `GRP_VPN_SPOTIFY_A`
- **Fold 4 Validation Groups (5):** `GRP_FB_VIDEO_1`, `GRP_FTPS_DOWN_1`, `GRP_SCP_2`, `GRP_SCP_4`, `GRP_SKYPE_AUDIO_1`

### Key Methodological Strengths
1. **Zero Capture Leakage:** No single capture session appears in both train and validation splits.
2. **Comprehensive Class Representation:** 100% class coverage is mathematically achieved in all 5 folds.
3. **Sufficient Group Redundancy:** Every class has between 9 and 10 capture groups, ensuring models cannot memorize specific session dynamics.
