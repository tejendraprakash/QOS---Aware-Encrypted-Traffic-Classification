# Track A Dataset Specification: Exploratory 4-Class Baseline

**Project:** QoS-Aware Encrypted Network Traffic Classification for Next-Generation Networks  
**Phase:** 5 — Baseline Experiment Preparation & Protocol Validation  
**Date:** November 2026  
**Dataset Artifact:** [track_a_flows.csv](file:///c:/Users/saipr/OneDrive/Desktop/EncryptedTrafficResearch/output/phase5/track_a_flows.csv)  
**Fold Artifact:** [track_a_folds.csv](file:///c:/Users/saipr/OneDrive/Desktop/EncryptedTrafficResearch/output/phase5/track_a_folds.csv)  
**Analysis Table:** [track_a_fold_analysis.csv](file:///c:/Users/saipr/OneDrive/Desktop/EncryptedTrafficResearch/output/phase5/track_a_fold_analysis.csv)

---

## 1. Track Overview & Purpose

Track A serves as the **Exploratory 4-Class Baseline**, representing the original four intended QoS classes:
1. **Bulk Data**
2. **Browsing**
3. **Streaming**
4. **Real-Time**

> [!WARNING]
> **Scientific Integrity Disclosure — Label Contamination:**
> The `Browsing` class in this dataset is derived entirely from `email` captures. As audited in Phase 4, **75.05% of all flows in this class are local multicast discovery packets** (Link-Local Multicast Name Resolution / LLMNR UDP port 5355) and **3.64% are NetBIOS-NS** (UDP port 137). Genuine interactive HTTP/HTTPS web browsing traffic is virtually absent. Track A must therefore be treated strictly as an *exploratory baseline* to evaluate classifier behavior under real-world label noise, and must not be reported in academic publications as representing true web browsing.

---

## 2. Dataset Dimensions & Class Distribution

- **Total Flows:** 17,617
- **Total Columns:** 117 (14 metadata/audit, 103 behavioral ML-safe candidates)
- **Total Capture Groups:** 32 unique capture sessions
- **Imbalance Ratio:** $6.30 : 1$ (Real-Time vs. Streaming)

### Class Distribution Summary

| Class Name | Flow Count | Percentage (%) | Capture Groups | Major Contributing Applications |
|---|---|---|---|---|
| **Real-Time** | 8,132 | 46.16% | 10 groups | Hangouts Audio/Video, Skype Audio/Video, VoIPBuster, Facebook Audio/Video |
| **Browsing** | 6,847 | 38.87% | 3 groups | Email 1, Email 2, VPN Email 2 (75% LLMNR UDP 5355 background) |
| **Bulk Data** | 1,348 | 7.65% | 10 groups | FTPS Upload/Download, SCP (1 through 6) |
| **Streaming** | 1,290 | 7.32% | 9 groups | Netflix, Spotify, Vimeo, YouTube |
| **Total** | **17,617** | **100.0%** | **32 groups** | — |

---

## 3. Capture Group Bottleneck in Browsing

While Bulk Data (10), Streaming (9), and Real-Time (10) possess adequate capture diversity, **Browsing possesses only 3 capture groups**:
1. `GRP_EMAIL_1`: 5,870 flows (33.32% of entire dataset!)
2. `GRP_EMAIL_2`: 672 flows
3. `GRP_VPN_EMAIL_2`: 305 flows

Because group cross-validation requires complete isolation of capture groups between training and validation sets, a standard 5-fold `GroupKFold` mathematically fails: with 3 groups across 5 folds, at least 2 folds would have zero Browsing samples in validation.

---

## 4. Validated 3-Fold Stratified Group K-Fold Protocol

To achieve 100% class coverage across all validation folds while guaranteeing zero group leakage, Track A is evaluated using:
$$\text{StratifiedGroupKFold}(n\_splits=3, shuffle=True, random\_state=7)$$

### Partition Statistics

| Fold | Partition | Total Samples | Browsing | Bulk Data | Real-Time | Streaming | Groups Count | Group Overlap | Class Coverage |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **0** | **Train** | 15,992 | 6,542 (40.9%) | 1,124 (7.0%) | 7,166 (44.8%) | 1,160 (7.3%) | 25 groups | **0** | **100% (4/4)** |
| **0** | **Val** | 1,625 | 305 (18.8%) | 224 (13.8%) | 966 (59.4%) | 130 (8.0%) | 7 groups | **0** | **100% (4/4)** |
| **1** | **Train** | 5,940 | 977 (16.4%) | 1,337 (22.5%) | 2,811 (47.3%) | 815 (13.7%) | 23 groups | **0** | **100% (4/4)** |
| **1** | **Val** | 11,677 | 5,870 (50.3%) | 11 (0.1%) | 5,321 (45.6%) | 475 (4.1%) | 9 groups | **0** | **100% (4/4)** |
| **2** | **Train** | 13,302 | 6,175 (46.4%) | 235 (1.8%) | 6,287 (47.3%) | 605 (4.5%) | 16 groups | **0** | **100% (4/4)** |
| **2** | **Val** | 4,315 | 672 (15.6%) | 1,113 (25.8%) | 1,845 (42.8%) | 685 (15.9%) | 16 groups | **0** | **100% (4/4)** |

### Validation Group Assignments
- **Fold 0 Validation Groups (7):** `GRP_FTPS_UP_2`, `GRP_HANGOUTS_AUDIO_1`, `GRP_SCP_1`, `GRP_SCP_6`, `GRP_SKYPE_VIDEO_1`, `GRP_VIMEO_1`, `GRP_VPN_EMAIL_2`
- **Fold 1 Validation Groups (9):** `GRP_EMAIL_1`, `GRP_FB_AUDIO_1`, `GRP_NETFLIX_1`, `GRP_SCP_2`, `GRP_VOIPBUSTER_1`, `GRP_VPN_FB_AUDIO_2`, `GRP_VPN_HANGOUTS_AUDIO_1`, `GRP_VPN_SPOTIFY_A`, `GRP_VPN_YOUTUBE_A`
- **Fold 2 Validation Groups (16):** `GRP_EMAIL_2`, `GRP_FB_VIDEO_1`, `GRP_FTPS_DOWN_1`, `GRP_HANGOUTS_VIDEO_1`, `GRP_SCP_3`, `GRP_SCP_4`, `GRP_SCP_5`, `GRP_SKYPE_AUDIO_1`, `GRP_SPOTIFY_1`, `GRP_VPN_FTPS_A`, `GRP_VPN_FTPS_B`, `GRP_VPN_NETFLIX_A`, `GRP_VPN_SKYPE_AUDIO_1`, `GRP_VPN_VIMEO_A`, `GRP_VPN_VOIPBUSTER_1`, `GRP_YOUTUBE_1`

Every fold contains exactly one of the three email groups, ensuring all 4 classes are present with zero group leakage.
