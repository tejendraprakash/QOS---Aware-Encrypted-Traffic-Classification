# ISCX VPN-NonVPN-2016 Dataset Organization Report

**Project Title:** QoS-Aware Encrypted Network Traffic Classification for Next-Generation Networks  
**Target Dataset:** UNB ISCX VPN-NonVPN 2016 Benchmark Dataset  
**Storage Root:** `ISCX dataset/`  
**Execution Date:** November 2026  
**Auditor:** Lead Research Engineer & Dataset Auditor  

---

## 1. Executive Summary

In accordance with project data governance rules, the **ISCX VPN-NonVPN-2016 dataset** has been organized into a standardized, academic-grade hierarchy within `ISCX dataset/`. 

### Key Protections & Integrity Constraints:
* **Zero Modification to Active Research Data:** All existing project artifacts—including `data/`, `Dataset/`, `output/`, `research_notes/` (existing notes), and `scripts/`—were kept strictly untouched and unmerged.
* **Original Filenames Preserved:** No PCAP, PCAPNG, or ARFF files were renamed or altered.
* **Provenance Maintained:** Original subfolder distributions (`NonVPN-PCAPs-01`, `NonVPN-PCAPs-02`, `NonVPN-PCAPs-03`, `VPN-PCAPS-01`, `VPN-PCAPs-02`, `Scenario A1`, `Scenario A2`, `Scenario B`) were preserved exactly.
* **No Unauthorized Downloads or Deletions:** Zero files were deleted, overwritten, or downloaded.

---

## 2. Directory Structure of `ISCX dataset/`

```
ISCX dataset/
├── ARFF/
│   ├── Scenario A1/               # 4 ARFF files (Time-based features 15s, 30s, 60s, 120s - VPN)
│   ├── Scenario A2/               # 8 ARFF files (Time-based features 15s, 30s, 60s, 120s - VPN & Non-VPN)
│   └── Scenario B/                # 8 ARFF files (Time-based features 15s, 30s, 60s, 120s - AllinOne & Standard)
├── CSVs/                          # Prepared directory for ISCX CSV feature tables
├── original_archives/             # Dedicated repository for raw ZIP downloads
│   ├── ARFF/                      # Archive landing for ARFF ZIPs
│   ├── CSVs/                      # Archive landing for CSV ZIPs
│   └── PCAPs/                     # Archive landing for PCAP ZIPs
├── PCAPs/
│   ├── NonVPN-PCAPs-01/           # 23 files (12 .pcap, 11 .pcapng) - 975.56 MB
│   ├── NonVPN-PCAPs-02/           # 26 files (15 .pcap, 11 .pcapng) - 12,731.79 MB
│   ├── NonVPN-PCAPs-03/           # 60 files (44 .pcap, 16 .pcapng) - 10,255.11 MB
│   ├── VPN-PCAPS-01/              # 14 files (14 .pcap) - 728.42 MB
│   └── VPN-PCAPs-02/              # 17 files (17 .pcap) - 1,709.12 MB
└── iscx_dataset_inventory.csv     # Complete inventory metadata (160 indexed files)
```

---

## 3. Comprehensive File Counts & Inventory Summary

| Category | File Extension / Set | Count | Total Size | Notes |
| :--- | :---: | :---: | :---: | :--- |
| **PCAP Files** | `.pcap` | **102** | ~24.3 GB | Standard libpcap format |
| **PCAPNG Files** | `.pcapng` | **38** | ~2.1 GB | Next-Generation PCAP format |
| **Total Packet Captures**| `.pcap` + `.pcapng` | **140** | **~26.40 GB** | All 5 archive sets extracted |
| **ARFF Files** | `.arff` | **20** | **43.03 MB** | Scenarios A1 (4), A2 (8), B (8) |
| **CSV Files** | `.csv` | **0** | 0 B | Directory prepared; files currently absent |
| **ZIP Archives** | `.zip` | **0** | 0 B | `original_archives/` prepared; ZIPs not retained on disk |
| **Inventory Records** | `.csv` | **1** | 21 KB | `iscx_dataset_inventory.csv` |

---

## 4. Extraction & Organization Status

1. **PCAP Capture Sets (All 5 Present & Extracted):**
   * `NonVPN-PCAPs-01`: 23 captures (browsing, chat, email, audio).
   * `NonVPN-PCAPs-02`: 26 captures (large file transfers, FTPS, P2P, torrents).
   * `NonVPN-PCAPs-03`: 60 captures (streaming media, YouTube, Vimeo, Skype, VoIP).
   * `VPN-PCAPS-01`: 14 captures (VPN-encapsulated email, audio, chat, FTPS).
   * `VPN-PCAPs-02`: 17 captures (VPN-encapsulated Netflix, Spotify, YouTube, SFTP, ICQ, Skype).
   * *Status:* **100% Present, fully organized, 0 failed moves.**

2. **ARFF Feature Sets (All 3 Present & Extracted):**
   * `Scenario A1` (VPN Service Classification): 4 files ($15\text{s}, 30\text{s}, 60\text{s}, 120\text{s}$ windows).
   * `Scenario A2` (VPN vs. Non-VPN Characterization): 8 files ($15\text{s}, 30\text{s}, 60\text{s}, 120\text{s}$ windows for both VPN and Non-VPN).
   * `Scenario B` (Traffic Type Classification): 8 files ($15\text{s}, 30\text{s}, 60\text{s}, 120\text{s}$ windows, including All-in-One sets).
   * *Status:* **100% Present, flat scenario structure organized, nested redundant paths cleaned.**

3. **CSV Files & Original ZIP Archives:**
   * `ISCX dataset/CSVs/` was created. Currently, no `.csv` feature files from UNB ISCX exist in the folder (our active project CSVs in `output/` remain strictly isolated).
   * `ISCX dataset/original_archives/` with subfolders `PCAPs/`, `CSVs/`, and `ARFF/` is initialized and ready to receive ZIP files if downloaded or transferred.

---

## 5. Duplicate & Conflict Status

* **PCAP / PCAPNG Files:** **Zero duplicate filenames** across the 5 PCAP subdirectories. Every capture filename is distinct.
* **ARFF Files:** 4 identical filenames exist across scenarios:
  * `TimeBasedFeatures-Dataset-15s-VPN.arff`
  * `TimeBasedFeatures-Dataset-30s-VPN.arff`
  * `TimeBasedFeatures-Dataset-60s-VPN.arff`
  * `TimeBasedFeatures-Dataset-120s-VPN.arff`
  * *Resolution:* These files belong to different research scenarios (Scenario A1 vs. Scenario A2) in the official benchmark. Because they are segregated into `ISCX dataset/ARFF/Scenario A1/` and `ISCX dataset/ARFF/Scenario A2/`, **there is zero file conflict or risk of overwriting**.
* **Relationship to Active Research Dataset (`data/raw/`):**
  * Our project's active research subset in `data/raw/` contains 37 PCAP files (~13.20 GB).
  * These 37 files represent a targeted sub-selection of the 140 total captures in `ISCX dataset/PCAPs/`.
  * **Crucial Rule Maintained:** `data/raw/` and `Dataset/` were kept strictly independent. No cross-directory file moves or merges occurred.

---

## 6. Warnings & Recommendations for Next Steps

> [!NOTE]
> **CSV Status:** If you download the pre-computed ISCX CSV flow tables from the UNB CIC website, place them directly into `ISCX dataset/CSVs/`. If you retain the original `.zip` archive downloads, store them in `ISCX dataset/original_archives/CSVs/`.

> [!IMPORTANT]
> **Dataset Separation:** The active modeling pipeline (Phase 1 & Phase 2) operates exclusively on `data/raw/` and `output/flow_features.csv`. Do **not** run automated training scripts over `ISCX dataset/PCAPs/` without re-running flow extraction and group mapping across the full 140 captures.

---

## 7. Master Inventory Reference

The complete machine-readable catalog is saved at:
* [ISCX dataset/iscx_dataset_inventory.csv](file:///c:/Users/saipr/OneDrive/Desktop/EncryptedTrafficResearch/ISCX%20dataset/iscx_dataset_inventory.csv)
