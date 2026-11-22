# Agent B: Comprehensive Dataset Audit & Multi-Dataset Inventory

**Project Title:** QoS-Aware Encrypted Traffic Classification for Next-Generation Networks  
**Author / Persona:** Agent B — Dataset Auditor (`.agents/agents/dataset-auditor.md`)  
**Audit Stage:** Phase 4 — Multi-Dataset Provenance & Compatibility Audit  
**Date:** November 2026  

---

## 1. Executive Audit Overview

This audit independently inspects, catalogs, and compares the two datasets residing within the project repository:
1. **Dataset 1 (Active Research Subset):** Stored in `data/raw/` (with pristine source in `Dataset/`), analyzed in Phase 1 & Phase 2.
2. **Dataset 2 (Full UNB ISCX VPN-NonVPN-2016 Collection):** Stored in `ISCX dataset/` (comprising `PCAPs/`, `ARFF/`, `CSVs/`, and `original_archives/`).

Every file, byte count, packet total, and attribute schema was verified against actual filesystem contents.

---

## 2. Independent Audit of Dataset 1 (Active Research Subset)

### 2.1 Verified Physical Metrics
* **Total Capture Files:** **37 files** (verified 100% readable, 0 failed files).
* **Storage Footprint:** **14,172,826,857 bytes (~13.20 GB)**.
* **Total Packets Processed:** **11,981,359 packets**.
  * TCP Packets: $9,831,885$ ($82.06\%$)
  * UDP Packets: $2,117,233$ ($17.67\%$)
  * Other Protocols (ICMP, IGMP, ARP): $32,241$ ($0.27\%$)
* **Extracted Flow Records:** **17,617 bidirectional flows** in `output/flow_features.csv`.
* **Total Features Extracted:** **117 columns** (14 metadata/audit fields + 103 ML-safe behavioral features).

### 2.2 Link-Layer Encapsulation Breakdown
* **Ethernet (`DLT_EN10MB` / linktype 1):** 25 captures. Standard 14-byte frame header.
* **Raw IP (`DLT_RAW` / linktype 101):** 10 captures (all `vpn_*.pcap` files except `vpn_voipbuster1a.pcap`). Packets start directly at byte 0 with the IPv4 header.
* **Next Generation PCAP (`pcapng`):** 2 captures (`hangouts_video1b.pcapng` and `voipbuster1b.pcapng`).

### 2.3 Application & Volume Skew in Dataset 1
* **Massive Byte Dominance:** Two single bulk downloads (`ftps_down_1a.pcap` at 5.43 GB and `ftps_down_1b.pcap` at 4.23 GB) account for **73.4% of total dataset byte volume** and 7.68 million packets.
* **Flow Count Inversion:** At the flow level, **Real-Time (8,132 flows) and Browsing/Email (6,847 flows) account for 85.03%** of all records, while Bulk Data (1,348 flows) represents only 7.65%.
* **Capture Pair Symmetry:** Captures ending in `_1a`/`_1b` and `_2a`/`_2b` are synchronized twin vantage points (client and server interfaces) and must remain bound to the same capture group.

---

## 3. Independent Audit of Dataset 2 (UNB ISCX VPN-NonVPN-2016)

### 3.1 Verified Physical Metrics in `ISCX dataset/`
* **Total Packet Captures:** **140 files** across 5 subdirectories.
  * Standard `.pcap`: **102 files** (~24.3 GB)
  * Next-Generation `.pcapng`: **38 files** (~2.1 GB)
  * Total Volume: **~26.40 GB** (28,350,645,000 bytes).
* **Pre-Computed ARFF Files:** **20 files** (43.03 MB) across 3 scenario folders.
* **Pre-Computed CSV Files:** **0 files** currently in `ISCX dataset/CSVs/`.
* **Master Inventory Catalog:** Generated and verified in `ISCX dataset/iscx_dataset_inventory.csv` (160 indexed files).

### 3.2 Capture Distribution by Folder in `ISCX dataset/PCAPs/`
1. `NonVPN-PCAPs-01/` (23 files, 975.56 MB): AIM chat, Facebook chat/audio/video, Email 1 & 2.
2. `NonVPN-PCAPs-02/` (26 files, 12.73 GB): FTPS upload/download, ICQ chat, Gmail chat, Netflix 1–4.
3. `NonVPN-PCAPs-03/` (60 files, 10.26 GB): SCP, SFTP, Skype, Spotify 1–4, Vimeo 1–4, VoIP Buster, YouTube 1–6.
4. `VPN-PCAPS-01/` (14 files, 728.42 MB): VPN-encapsulated AIM, BitTorrent, Email 2, FB Audio, FTPS, Hangouts.
5. `VPN-PCAPs-02/` (17 files, 1.71 GB): VPN-encapsulated Netflix, Spotify, YouTube, SFTP, ICQ, Skype, Vimeo, VoIP Buster.

### 3.3 Verification of ISCX ARFF Files
All 20 ARFF files were verified. They originate from the Canadian Institute for Cybersecurity (UNB CIC):
* **Scenario A1 (VPN Verification):** 4 files ($15\text{s}, 30\text{s}, 60\text{s}, 120\text{s}$ windows), 2 classes (`Non-VPN` vs. `VPN`).
* **Scenario A2 (VPN Traffic Characterization):** 8 files, 14 classes (`VPN-BROWSING`, `BROWSING`, `VPN-CHAT`, `CHAT`, `VPN-FT`, `FT`, etc.).
* **Scenario B (Traffic Type Classification):** 8 files, 7 application categories (`BROWSING`, `CHAT`, `STREAMING`, `MAIL`, `VOIP`, `P2P`, `FT`).

---

## 4. Key Differences & Dataset Compatibility Findings

| Comparison Parameter | Dataset 1 (Active Research Set) | Dataset 2 (Full ISCX 2016 Collection) | Compatibility Verdict |
| :--- | :--- | :--- | :--- |
| **Origin & Provenance** | Subset of UNB ISCX 2016 | Complete UNB ISCX 2016 benchmark | 100% Identical Provenance |
| **Capture Count** | 37 PCAPs | 140 PCAPs | Dataset 1 is a 26.4% subset of Dataset 2 |
| **Application Breadth** | 11 applications | 17+ applications | Dataset 2 adds BitTorrent, SFTP, Chat clients |
| **Feature Set** | **103 Features** (Sizes, IAT, Flags, Bursts, Early) | **23 Features** in ARFF (Time IAT/active/idle only) | **Incompatible Pre-Extracted Schemas** |
| **Flow Slicing Method** | **Inactivity Timeout (60.0 seconds)** | **Fixed Time Windows (15s, 30s, 60s, 120s)** | **Methodological Difference** |
| **Browsing PCAPs** | **0 Web PCAPs** (Email only) | **0 Web PCAPs** (Email only; 0 HTTP/HTTPS PCAPs) | **Shared Structural Gap** |

---

## 5. Audit Recommendation on Dataset Utilization

1. **DO NOT Merge Dataset 1 and ISCX ARFF Tables Directly:**
   The pre-extracted ARFF files are sliced on fixed time windows (e.g. 15s) and lack all packet-size, burst, and TCP flag dynamics. Merging them directly would corrupt the feature space.
2. **DO NOT Merge Dataset 1 PCAPs with Dataset 2 PCAPs in Phase 3:**
   Dataset 1 represents an established, audited benchmark with 17,617 flows and verified 32 capture groups.
3. **STRATEGIC RECOMMENDATION — Staged Independent Roles:**
   * **Phase 3:** Train and benchmark models exclusively on **Dataset 1** (17,617 flows, 103 behavioral features).
   * **Phase 5 (External Generalization):** Run `scripts/flow_extractor.py` on the remaining **103 unextracted PCAPs** in `ISCX dataset/PCAPs/` to create a true out-of-distribution, external holdout testbed.