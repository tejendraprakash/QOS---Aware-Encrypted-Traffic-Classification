# QoS-Aware Encrypted Network Traffic Classification for Next-Generation Networks

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status: Research Prototype](https://img.shields.io/badge/Status-Research%20Verified-brightgreen.svg)]()

A comprehensive, reproducible research framework for **early-stage encrypted network traffic classification** and **QoS scheduling evaluation** in next-generation networks.

Grounded in empirical PCAP data from the University of New Brunswick (UNB) ISCX VPN-nonVPN 2016 benchmark and validated using an RFC-compliant discrete-event queueing network simulation engine.

---

## 📌 Research Overview

In modern networks, end-to-end encryption (TLS 1.3, QUIC, HTTPS, WireGuard) blinds intermediate network middleboxes from inspecting packet payloads. Consequently, traditional Deep Packet Inspection (DPI) fails to assign differentiated Quality of Service (QoS) priorities.

This project investigates and validates an end-to-end autonomous pipeline:
1. **Early-Horizon Flow Feature Extraction:** Observes strictly the first $N$ packets ($N=5$) of a connection without payload decryption.
2. **Leak-Free Machine Learning Classification:** Classifies flows into 3 core 3GPP QoS traffic classes (**Real-Time**, **Streaming**, and **Bulk Data**) with strict `StratifiedGroupKFold` isolation across 29 independent capture sessions.
3. **Discrete-Event Network Simulation:** Feeds empirical confusion matrices directly into a closed-loop discrete-event network simulator (100 Mbps bottleneck link, 125% High Congestion load) to quantify real-world QoS impacts on packet delay, jitter, loss, and goodput.

---

## 🚀 Key Results & Breakthroughs

- **Forensic Dataset Contamination Audit:** Identified and removed 6,574 ephemeral infrastructure noise flows (DNS, LLMNR, NetBIOS, SSDP, multicast) that previously distorted voice traffic proportions.
- **Clean Application Benchmark:** Constructed an authentic, leak-free dataset of **4,196 flows** (50.57% Real-Time, 28.05% Bulk Data, 21.38% Streaming).
- **Early-5 Sufficiency:** Proved that $N=5$ packets achieve **Pooled Macro F1 = 0.5348** with an inference latency of **5.79 μs per sample** (>170,000 classifications/sec/core), matching or exceeding 10-, 15-, and 20-packet horizons.
- **Empirical QoS Performance Gain (125% High Congestion):**
  - **Real-Time Mean Delay:** Reduced by **53.5%** ($65.82\text{ ms} \to 30.61\text{ ms}$).
  - **Real-Time P95 Delay:** Maintained at **$69.62\text{ ms}$**, strictly complying with the 100 ms 3GPP Conversational Voice SLA (5QI 1).
  - **Real-Time Loss:** Reduced by **57.7%** ($8.10\% \to 3.43\%$).
  - **Bulk Data Starvation Eliminated:** Goodput boosted by **11.9x** ($1.66\text{ Mbps} \to 19.73\text{ Mbps}$) via Deficit Round Robin (DRR) queuing.

---

## 📂 Repository Structure

```
EncryptedTrafficResearch/
├── .agents/                      # Multi-agent role specifications & research guides
├── output/                       # Evaluation CSVs, metrics, and figures
│   ├── phase5/                   # Feature extraction datasets & audits
│   ├── phase6a/                  # Full-flow classification baselines
│   ├── phase6b/                  # Early-5 baseline benchmarks
│   ├── phase6c/                  # Optimization & baseline exploration
│   ├── phase6d/                  # Clean reconstructed datasets, models, & confusion matrices
│   └── phase7/                   # Discrete-event QoS simulation results & figures
├── research_notes/               # Publication-grade technical reports & literature audits
│   ├── dataset_audit.md          # Forensic dataset audit & provenance documentation
│   ├── feature_dictionary.md     # Feature mathematical formulas & definitions
│   ├── flow_formation_guide.md   # Flow definition & timeout guidelines
│   ├── phase6d/                  # Forensic contamination & clean reconstruction logs
│   └── phase7/                   # Traffic generation blueprints & audit reports
├── scripts/                      # Production Python scripts & pipelines
│   ├── flow_extractor.py         # Packet-to-flow feature extraction engine
│   ├── pcap_analyzer.py          # PCAP packet parser & telemetry extractor
│   ├── reconstruct_clean_dataset.py # Infrastructure noise filter & clean dataset builder
│   ├── run_phase6d_experiments.py   # Multi-model benchmarking on clean folds
│   ├── simulate_phase7_qos.py    # Production discrete-event queue & QoS simulator
│   └── test_clean_qos_simulation.py # 5-seed clean empirical QoS simulation evaluator
├── .gitignore                    # Excludes raw PCAPs (>40 GB) and local artifacts
├── extracted_paper_text.txt      # Text extraction of the foundational research paper
├── Research Paper.pdf            # Huang et al. (Electronics 2021) reference paper
└── README.md                     # Project documentation and reproduction guide
```

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.10+
- Recommended: Windows 10/11 or Linux (Ubuntu 20.04+)

### Dependencies
```bash
pip install numpy pandas scikit-learn xgboost matplotlib seaborn scipy
```

---

## 🔬 Reproducing the Experiments

### 1. Reconstruct Clean Authentic Dataset
Filter out background infrastructure noise (DNS, LLMNR, SSDP, NetBIOS, multicast):
```bash
python scripts/reconstruct_clean_dataset.py
```
*Outputs: `output/phase6d/clean_track_b_flows.csv` and `output/phase6d/clean_track_b_folds.csv`.*

### 2. Run Clean Machine Learning Benchmarks
Train and evaluate 6 baseline and tuned classifiers using 5-fold StratifiedGroupKFold:
```bash
python scripts/run_phase6d_experiments.py
```
*Generates comparison tables, Out-of-Fold predictions, confusion matrices, and Pareto frontier plots.*

### 3. Run Closed-Loop Discrete-Event QoS Simulation
Simulate the 100 Mbps bottleneck link under 125% High Congestion using the clean empirical Early-5 confusion matrix:
```bash
python scripts/test_clean_qos_simulation.py
```
*Evaluates 5 independent random seeds and computes per-class delay, loss, jitter, and goodput.*

---

## 📖 Citation & References

This research builds upon and extends:
- **Foundational Paper:** Huang, Y.-F., Lin, C.-B., Chung, C.-M., & Chen, C.-M. (2021). *Research on QoS Classification of Network Encrypted Traffic Behavior Based on Machine Learning*. **Electronics**, 10(12), 1376. [https://doi.org/10.3390/electronics10121376](https://doi.org/10.3390/electronics10121376)
- **Dataset:** Draper-Gil, G., Lashkari, A. H., Mamun, M. S. I., & Ghorbani, A. A. (2016). *Characterization of Encrypted Traffic Through Feature Selection*. In **ICISSP 2016** (pp. 248–255). UNB ISCX VPN-nonVPN 2016.
- **Traffic Standards:** 3GPP TS 23.501 (5G QoS Identifier mapping), RFC 8312 (TCP CUBIC), ITU-T P.59 (Conversational Voice).

---

## 📄 License
This project is released under the [MIT License](LICENSE).
