# Research Dashboard Data Inventory & Technical Specification

**Project Title**: QoS-Aware Encrypted Traffic Classification for Next-Generation Networks  
**Document**: Data Discovery, Schema Catalog, and Dashboard Page Integration Matrix  
**Status**: Production Verified  
**Date**: November 2026  
**Audit Scope**: `output/phase7_7/`, `output/runtime_verification/`, `output/phase6d/`, `output/phase7/`, and `data/raw/`

---

## 1. Executive Summary & Audit Overview

This document provides a comprehensive technical inventory of all empirical datasets, trained model configurations, network simulation logs, and validation artifacts supporting the interactive Research Dashboard.

The research framework addresses early-flow traffic classification over encrypted, encapsulated (VPN/non-VPN) network streams to enable automated Quality of Service (QoS) provisioning at the network edge. Using early packet inspection (the first 5 packets of each flow), an engineered 8-feature representation, and strict group-aware evaluation (29-Fold Leave-One-Group-Out cross-validation), the system guarantees zero data leakage across dual-ended network captures.

### Global Data Volume & Metrics Summary

| Directory | File Count | Primary Formats | Key Contents | Primary Purpose |
|:---|:---:|:---:|:---|:---|
| `data/raw/` | 37 | `.pcap`, `.pcapng` (13.20 GB) | Dual-ended raw network captures across 15 apps | Source ground-truth packet traces |
| `output/phase6d/` | 13 | `.csv`, `.png` (4.4 MB) | 4,196 canonical flows, baseline evaluations, OOF predictions | Ground truth dataset, early horizon benchmarks |
| `output/phase7/` | 8 | `.csv`, `.png` (0.8 MB) | 135 simulation runs, aggregate metrics, SLA compliance | Multi-regime QoS queueing and SLA verification |
| `output/phase7_7/` | 14 | `.csv`, `.json`, `.md` (0.9 MB) | Champion XGBoost model, 8-feature vector, candidate matrix | Scientific audit & production model artifacts |
| `output/runtime_verification/` | 13 | `.csv`, `.json`, `.pcap`, `.md` | E2E runtime inference, flow extraction, discrete simulation | End-to-end verification pipeline artifacts |

---

## 2. Directory-by-Directory File Inventory & Schema Catalog

### 2.1. `output/phase7_7/` (Production Champion Model & Ablation Matrix)

This directory contains the final production champion model, rigorous cross-validation results, ablation candidate matrices, and statistical significance tests.

#### 1. `best_model_config.json` (1,508 bytes)
* **Description**: Canonical machine-readable metadata specification for the Phase 7.7 Champion model.
* **Format**: JSON object.
* **Key Fields**:
  * `model_name`: `"Phase 7.7 Champion Early-5 Model"`
  * `configuration_id`: `"CFG-03: Robust 8-Feat Vector"`
  * `feature_count`: 8
  * `features`: `["early_5_complete", "early_5_duration", "early_5_bytes", "early_5_fwd_pkts", "early_5_bwd_pkts", "early_5_avg_size", "is_udp", "early_5_fwd_ratio"]`
  * `outer_evaluation`: `"29-Fold Leave-One-Group-Out (LOGO)"`
  * `nested_evaluation_macro_f1`: 0.5913
  * `outer_macro_f1`: 0.5895 (vs. Baseline 0.5734, $\Delta = +0.0161$)
  * `balanced_accuracy`: 0.6054, `overall_accuracy`: 0.5922
  * `per_class_metrics`: Bulk Data ($F_1 = 0.5447$), Real-Time ($F_1 = 0.6039$), Streaming ($F_1 = 0.6198$)
  * `hyperparameters`: `max_depth: 5`, `learning_rate: 0.1`, `n_estimators: 100`, `subsample: 0.8`, `colsample_bytree: 0.8`, `objective: "multi:softprob"`.

#### 2. `candidate_matrix_results.csv` (10 rows $\times$ 19 columns, 2,897 bytes)
* **Description**: Hyperparameter optimization and calibration matrix comparing 10 model variants (CFG-01 through CFG-10) across XGBoost, HistGBDT, Random Forest, and Platt-calibrated models.
* **Schema**:
  * `Config_ID` (string, non-null): Identifier (CFG-01 to CFG-10).
  * `Config_Name` (string, non-null): Human-readable configuration title.
  * `Model_Family` (string, non-null): Algorithm class (`XGBoost`, `HistGBDT`, `RandomForest`, `CalibratedXGB`, `IsotonicXGB`).
  * `Description` (string, non-null): Parametric formulation.
  * `Macro_F1`, `CV_Mean_F1`, `CV_Std_F1` (float64): F1 metrics.
  * `Balanced_Accuracy`, `Overall_Accuracy` (float64): Accuracy metrics.
  * `Bulk_Data_F1`, `Real_Time_F1`, `Streaming_F1` (float64): Per-class F1 scores.
  * `Bulk_Data_Recall`, `Real_Time_Recall`, `Streaming_Recall` (float64): Per-class recall.
  * `Bulk_Data_Precision`, `Real_Time_Precision`, `Streaming_Precision` (float64): Per-class precision.
  * `Runtime_Sec` (float64): Wall-clock training latency.
* **Null Count**: 0 across all columns.

#### 3. `experiment_comparison.csv` (8 rows $\times$ 21 columns, 2,935 bytes)
* **Description**: Head-to-head empirical comparison of 8 distinct architectural paradigms under strict LOGO validation, including McNemar and Wilcoxon significance tests against baseline.
* **Schema**:
  * `Configuration` (string, non-null): CFG-01 through CFG-08.
  * `Description` (string, non-null): Architecture summary.
  * `Feature_Count` (int64, non-null): 6 or 8 features.
  * `Features` (string, non-null): Comma-delimited list of active feature names.
  * `Macro_F1`, `Delta_Macro_F1`, `Balanced_Accuracy`, `Overall_Accuracy` (float64).
  * Class-specific metrics (`Bulk_Data_*`, `Real_Time_*`, `Streaming_*`).
  * `McNemar_p_value` (float64), `Wilcoxon_p_value` (float64), `Stat_Significant_McNemar` (bool).
  * `Runtime_Sec` (float64): Execution latency (ranging from 6.84s to 22.41s).
* **Null Count**: 0 across all columns.

#### 4. `per_class_metrics.csv` (3 rows $\times$ 7 columns, 224 bytes)
* **Description**: Detailed classification performance breakdown for the Champion 8-feature model.
* **Schema**:
  * `QoS_Class` (string, non-null): Bulk Data, Real-Time, Streaming.
  * `Precision`, `Recall`, `F1_Score` (float64): Final performance scores.
  * `Support` (int64, non-null): Flow count (Bulk Data: 1,177, Real-Time: 2,122, Streaming: 897; Total: 4,196).
  * `Baseline_Recall` (float64): Prior baseline recall rates (0.5021, 0.5655, 0.6968).
  * `Delta_Recall` (float64): Absolute recall improvements (+0.0102, +0.0160, +0.0256).
* **Null Count**: 0 across all columns.

#### 5. `confusion_matrix.csv` (3 rows $\times$ 4 columns, 238 bytes)
* **Description**: Normalized $3 \times 3$ confusion matrix under 29-Fold LOGO evaluation.
* **Values**:
  * True Bulk Data $\rightarrow$ Pred Bulk: 0.5123, Real-Time: 0.4367, Streaming: 0.0510.
  * True Real-Time $\rightarrow$ Pred Bulk: 0.1894, Real-Time: 0.5815, Streaming: 0.2290.
  * True Streaming $\rightarrow$ Pred Bulk: 0.0357, Real-Time: 0.2419, Streaming: 0.7224.

#### 6. `phase7_7_champion_model.json` (819,299 bytes)
* **Description**: Production serialized XGBoost booster model JSON containing decision tree structures, split feature IDs, threshold values, and leaf weights for 8-feature early classification.

---

### 2.2. `output/runtime_verification/` (End-to-End Operational Pipeline)

This directory houses the operational verification results validating the entire pipeline from raw PCAP capture to discrete-event simulation.

#### 1. `inferred_flows.csv` (15 rows $\times$ 123 columns, 12,124 bytes)
* **Description**: Complete end-to-end inference verification output for 15 flows extracted from `data/raw/scpDown4.pcap`.
* **Schema Highlights**:
  * Flow identifiers & 5-tuple: `flow_id`, `capture_filename`, `capture_group`, `src_ip`, `src_port`, `dst_ip`, `dst_port`, `protocol_name`, `protocol`.
  * Traffic labels: `application` (`SCP Download`), `is_vpn` (`No`), `qos_class` (`Bulk Data`), `label_confidence` (1.0).
  * Flow statistics: `flow_duration`, `total_packets`, `fwd_packets`, `bwd_packets`, `total_bytes`, `byte_rate`.
  * Early features: `early_5_complete`, `early_5_duration`, `early_5_bytes`, `early_5_fwd_pkts`, `early_5_bwd_pkts`, `early_5_avg_size`, `is_udp`, `early_5_fwd_ratio`.
  * Predictions & probabilities: `pred_class` (`Bulk Data`), `confidence` (0.835 to 0.887), `prob_Bulk Data`, `prob_Real-Time`, `prob_Streaming`.
* **Null Count**: 0 across all 123 columns.

#### 2. `extracted_flows/flow_features.csv` (15 rows $\times$ 117 columns, 11,411 bytes)
* **Description**: Pre-inference extracted feature table generated directly by the flow extractor module.
* **Null Count**: 0 across all 117 columns.

#### 3. `extracted_flows/flow_extraction_errors.csv` (0 rows $\times$ 3 columns, 37 bytes)
* **Description**: Error log for flow extraction failures.
* **Schema**: `capture_filename`, `error_type`, `error_message`.
* **Status**: 0 errors recorded across the verification run.

#### 4. `e2e_simulation_metrics.csv` (3 rows $\times$ 15 columns, 483 bytes)
* **Description**: End-to-end discrete event queue simulation results under runtime conditions.
* **Schema**: `Scenario`, `Congestion_Regime`, `Seed`, `QoS_Class`, `Total_Tx_Pkts`, `Total_Rx_Pkts`, `Total_Dropped_Pkts`, `Packet_Loss_Rate_Pct`, `Mean_Delay_ms`, `P95_Delay_ms`, `P99_Delay_ms`, `Jitter_ms`, `Goodput_Mbps`, `Mean_Queue_Occupancy_Bytes`, `Conservation_Error`.
* **Verification Outcome**: 0% packet loss across all classes; P95 delay < 20.2 ms; Conservation Error = 0.

#### 5. `early5_xgboost_model.json` (806,768 bytes)
* **Description**: Operational XGBoost runtime model deployed in the live inference pipeline.

---

### 2.3. `output/phase6d/` (Baseline Models, Horizons & Clean Dataset)

This directory contains the cleaned, standardized dataset and comprehensive early packet horizon benchmarks.

#### 1. `clean_track_b_flows.csv` (4,196 rows $\times$ 120 columns, 3,060,110 bytes)
* **Description**: Master ground-truth dataset containing 4,196 fully characterized bidirectional flows.
* **Class Distribution**:
  * Real-Time: 2,122 flows (50.57%)
  * Bulk Data: 1,177 flows (28.05%)
  * Streaming: 897 flows (21.38%)
* **Encapsulation Distribution**:
  * Non-VPN (`No`): 2,142 flows (51.05%)
  * Encapsulated VPN (`Yes`): 2,054 flows (48.95%)
* **Transport Protocols**: UDP (2,270 flows), TCP (1,926 flows).
* **Application Diversity**: 15 distinct applications (Skype Audio: 976, FTPS Download: 871, Facebook Audio: 445, VoIP Buster: 436, Netflix: 235, YouTube: 230, Spotify: 200, FTPS Upload: 181, Hangouts Audio: 172, Vimeo: 132, FTPS: 113, Facebook Video: 100, Hangouts Video: 67, Skype Video: 26, SCP Download: 12).
* **Capture Groups**: 29 unique capture groups (`GRP_SKYPE_AUDIO_1`, `GRP_FTPS_DOWN_1`, etc.).
* **Data Integrity**: Exactly 0 nulls across all 4,196 rows and 120 columns.

#### 2. `clean_track_b_folds.csv` (4,196 rows $\times$ 6 columns, 306,864 bytes)
* **Description**: Fold allocation table enforcing group-aware validation partitions across all 4,196 flows.
* **Schema**: `flow_id`, `capture_group`, `qos_class`, `application`, `is_vpn`, `fold`.

#### 3. `clean_best_oof_predictions.csv` (4,196 rows $\times$ 10 columns, 598,771 bytes)
* **Description**: Out-of-fold probability vectors and class assignments for all 4,196 flows from baseline cross-validation.
* **Schema**: `flow_id`, `capture_group`, `qos_class`, `application`, `is_vpn`, `fold`, `pred_qos_class`, `prob_Bulk Data`, `prob_Real-Time`, `prob_Streaming`.

#### 4. `clean_horizon_comparison.csv` (4 rows $\times$ 11 columns, 450 bytes)
* **Description**: Performance scaling across early packet inspection horizons: `early_5` (6 feats), `early_10` (12 feats), `early_15` (18 feats), and `early_20` (24 feats).
* **Key Insight**: Macro F1 plateaus early (`early_5`: 0.5348 pooled, `early_10`: 0.5311, `early_15`: 0.5369, `early_20`: 0.5408), proving that inspecting just 5 packets captures 98.9% of achievable predictive power while slashing classification latency.

#### 5. `clean_baseline_comparison.csv` (6 rows $\times$ 14 columns, 826 bytes)
* **Description**: Initial benchmark comparing Logistic Regression, Linear SVM, Decision Tree, Random Forest, HistGBDT, and XGBoost on early-5 features.

#### 6. `clean_simulation_statistical_audit.csv` (45 rows $\times$ 13 columns, 4,425 bytes)
* **Description**: Rigorous statistical tests (paired t-test, Wilcoxon signed-rank, Cohen's d) for delay, packet loss, and goodput across Low, Moderate, and High congestion regimes.

---

### 2.4. `output/phase7/` (Discrete-Event Network Simulation & SLA Compliance)

This directory contains the full evaluation of network QoS policies under synthetic congestion regimes.

#### 1. `run_metrics.csv` (135 rows $\times$ 15 columns, 14,597 bytes)
* **Description**: Detailed results from 135 simulation runs (3 Scenarios: `Baseline_FIFO`, `Predicted_QoS`, `Oracle_QoS` $\times$ 3 Regimes: `Low`, `Moderate`, `High` $\times$ 5 Replications $\times$ 3 QoS Classes).
* **Schema**:
  * `Scenario`, `Congestion_Regime`, `Seed`, `QoS_Class`
  * Packet counts: `Total_Tx_Pkts`, `Total_Rx_Pkts`, `Total_Dropped_Pkts`
  * Quality metrics: `Packet_Loss_Rate_Pct`, `Mean_Delay_ms`, `P95_Delay_ms`, `P99_Delay_ms`, `Jitter_ms`, `Goodput_Mbps`
  * Queue statistics: `Mean_Queue_Occupancy_Bytes`, `Conservation_Error`.
* **Data Integrity**: Conservation error is identically 0.0 across all 135 runs.

#### 2. `aggregate_scenario_metrics.csv` (27 rows $\times$ 39 columns, 8,229 bytes)
* **Description**: Statistical aggregates across all seeds with means, standard deviations, and 95% confidence intervals for every scenario and regime.

#### 3. `sla_compliance_summary.csv` (9 rows $\times$ 7 columns, 522 bytes)
* **Description**: Contractual SLA compliance audit across scenarios and congestion regimes.
* **SLA Threshold Standards**:
  * **Real-Time**: Max P95 Delay $\le 100\text{ ms}$, Packet Loss $\le 1.0\%$.
  * **Streaming**: Max P95 Delay $\le 300\text{ ms}$, Packet Loss $\le 0.01\%$.
  * **Bulk Data**: Max P95 Delay $\le 300\text{ ms}$, Packet Loss $\le 1.0\%$.
* **Compliance Matrix Summary**:

| Scenario | Congestion Regime | Real-Time SLA (%) | Streaming SLA (%) | Bulk Data SLA (%) | Composite SLA (%) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Baseline FIFO** | Low | 0.0% | 0.0% | 0.0% | **0.0%** |
| **Baseline FIFO** | Moderate | 0.0% | 0.0% | 0.0% | **0.0%** |
| **Baseline FIFO** | High | 0.0% | 0.0% | 0.0% | **0.0%** |
| **Predicted QoS (ML)** | Low | 100.0% | 0.0% | 40.0% | **58.0%** |
| **Predicted QoS (ML)** | Moderate | 80.0% | 0.0% | 0.0% | **40.0%** |
| **Predicted QoS (ML)** | High | 40.0% | 0.0% | 0.0% | **20.0%** |
| **Oracle QoS** | Low | 100.0% | 0.0% | 100.0% | **70.0%** |
| **Oracle QoS** | Moderate | 100.0% | 0.0% | 100.0% | **70.0%** |
| **Oracle QoS** | High | 100.0% | 0.0% | 100.0% | **70.0%** |

#### 4. `per_class_performance.csv` (11 rows $\times$ 8 columns, 929 bytes)
* **Description**: Head-to-head comparison between Baseline FIFO, Predicted QoS, and Oracle QoS with Hedges' g effect sizes and Wilcoxon p-values.
* **Key Achievements**:
  * Real-Time P95 Latency reduced from $70.06\text{ ms}$ to $51.76\text{ ms}$ (**$26.11\%$ reduction**, Hedges' $g = 8.28$, $p = 0.0625$).
  * Real-Time Packet Loss reduced from $8.03\%$ to $1.05\%$ (**$86.99\%$ reduction**, Hedges' $g = 8.07$).
  * Bulk Data Packet Loss reduced from $57.26\%$ to $10.52\%$ (**$81.64\%$ reduction**, Hedges' $g = 7.41$).
  * Bulk Data Goodput increased from $1.66\text{ Mbps}$ to $8.56\text{ Mbps}$ (**$+416.89\%$ boost**, Hedges' $g = -5.24$).

---

### 2.5. `data/raw/` (Raw Network PCAP Captures)

* **Total Capture Files**: 37 files (35 `.pcap`, 2 `.pcapng`).
* **Total Storage**: 13.20 GB (14,172,118,502 bytes).
* **Capture Diversity**:
  * Large bulk captures: `ftps_down_1a.pcap` (5.56 GB), `ftps_down_1b.pcap` (4.33 GB), `vpn_netflix_A.pcap` (769 MB), `hangouts_video1b.pcapng` (370 MB).
  * Interactive/Audio captures: `facebook_audio1a.pcap` (8.15 MB), `hangouts_audio1a.pcap` (15.67 MB), `skype_audio1a.pcap` (9.12 MB), `voipbuster1b.pcapng` (5.57 MB).
  * Verification capture: `scpDown4.pcap` (96.29 KB) used in automated runtime verification tests.

---

## 3. Dashboard Page Integration & Data Mapping Matrix

The interactive research dashboard comprises 8 distinct pages. The table below specifies the required data sources, schemas, and visualization widgets for each page.

| Page # | Page Name | Primary Data Sources | Secondary / Fallback Sources | Core Visualizations & Widgets | Key Metrics Displayed |
|:---:|:---|:---|:---|:---|:---|
| **1** | **Overview** | `dashboard/data_inventory.json`<br>`output/phase7_7/best_model_config.json`<br>`output/phase7/sla_compliance_summary.csv` | `output/phase6d/clean_track_b_flows.csv`<br>`output/phase7/per_class_performance.csv` | - KPI summary cards<br>- System pipeline flowchart<br>- High-level SLA improvement bar chart<br>- Class & encapsulation donut charts | Macro F1 ($0.5895$), Flows ($4,196$), Raw Data ($13.20\text{ GB}$), SLA Gain ($0\% \rightarrow 58\%$), RT Loss Drop ($-86.99\%$) |
| **2** | **Traffic Classification** | `output/runtime_verification/inferred_flows.csv`<br>`output/phase7_7/phase7_7_champion_model.json`<br>`output/phase7_7/feature_definitions.md` | `output/runtime_verification/extracted_flows/flow_features.csv` | - Live Flow Classifier simulator<br>- Confidence probability gauges<br>- Early-5 Feature Radar / Bar chart<br>- Inferred flows data table with CSV export | Class probability vector ($P_{\text{Bulk}}, P_{\text{RT}}, P_{\text{Stream}}$), Feature values, Classification confidence |
| **3** | **Flow Explorer** | `output/phase6d/clean_track_b_flows.csv`<br>`output/phase6d/clean_track_b_folds.csv` | `output/phase6d/clean_best_oof_predictions.csv` | - Multi-filter control panel (App, VPN, Protocol, QoS)<br>- Bivariate scatter plot (Duration vs Bytes)<br>- Packet size distribution histogram<br>- Correlation heatmap of early features | Flow count ($4,196$), Filtered subsets, Duration, Byte volume, Packet count, Mean packet size |
| **4** | **Model Performance** | `output/phase7_7/confusion_matrix.csv`<br>`output/phase7_7/per_class_metrics.csv`<br>`output/phase6d/clean_horizon_comparison.csv` | `output/phase7_7/candidate_matrix_results.csv`<br>`output/phase6d/clean_best_oof_predictions.csv` | - Normalized $3 \times 3$ Confusion Matrix<br>- Per-class Precision/Recall/F1 grouped bar chart<br>- Early Packet Horizon curve (5, 10, 15, 20)<br>- ROC / PR curves by class | Macro F1 ($0.5895$), Balanced Acc ($0.6054$), Streaming Recall ($72.24\%$), RT Precision ($62.80\%$) |
| **5** | **QoS Performance** | `output/phase7/per_class_performance.csv`<br>`output/phase7/aggregate_scenario_metrics.csv`<br>`output/phase7/run_metrics.csv` | `output/phase6d/clean_simulation_statistical_audit.csv` | - Scenario comparison bar charts (FIFO vs Predicted vs Oracle)<br>- Latency CDF / Boxplots across regimes<br>- Packet Loss vs Congestion line graph<br>- Goodput distribution by class | Real-Time P95 Latency ($51.76\text{ ms}$), Loss Rate ($1.05\%$), Bulk Goodput ($8.56\text{ Mbps}$), Hedges' g effect sizes |
| **6** | **Scheduler** | `output/phase7/sla_compliance_summary.csv`<br>`output/phase7/aggregate_scenario_metrics.csv`<br>`dashboard/data_inventory.json` | `output/phase7/run_metrics.csv` | - Queue Architecture Diagram (Strict Priority + DRR)<br>- SLA Compliance Matrix (Heatmap / Bar chart)<br>- Mean Queue Occupancy over time<br>- Packet drop breakdown by priority queue | Composite SLA compliance ($58.0\%$), Queue weights (Strict Priority RT, DRR 60% Stream, DRR 40% Bulk) |
| **7** | **Experiment Comparison** | `output/phase7_7/experiment_comparison.csv`<br>`output/phase7_7/candidate_matrix_results.csv`<br>`output/phase6d/clean_baseline_comparison.csv` | `output/phase6d/clean_tuned_comparison.csv`<br>`output/phase7_6_fast/fast_experiments_comparison.csv` | - Interactive model comparison matrix<br>- Pareto Frontier (Macro F1 vs Inference Latency)<br>- Delta F1 waterfall chart vs Baseline<br>- Statistical significance indicators ($p$-values) | Best Macro F1 ($0.5895$), Training Runtime ($8.10\text{ s}$), McNemar $p$-value ($0.2661$), Model Families |
| **8** | **Run Pipeline** | `output/runtime_verification/flow_extraction_summary.txt`<br>`output/runtime_verification/e2e_simulation_metrics.csv`<br>`output/runtime_verification/inferred_flows.csv` | `output/runtime_verification/RUNTIME_VERIFICATION_REPORT.md` | - Interactive Pipeline Stage Stepper (Pcap $\rightarrow$ Extract $\rightarrow$ Classify $\rightarrow$ Simulate)<br>- Real-time log terminal / streaming view<br>- Sample PCAP selector (`scpDown4.pcap`)<br>- Execution verification health badge | Execution time ($0.03\text{ s}$), Processed pkts ($166$), Extracted flows ($15$), Zero drop verification ($0.0\%$) |

---

## 4. Key Performance Indicators & Verified Metrics Reference

All dashboard components must reference the validated empirical metrics documented below to preserve absolute scientific consistency across UI widgets:

### 4.1. Champion Classification Metrics (Phase 7.7)
* **Model**: XGBoost Default on 8-Feature Vector (`CFG-03`)
* **Validation Strategy**: 29-Fold Leave-One-Group-Out (LOGO)
* **Macro F1**: **$0.5895$** (Baseline: $0.5734$, Absolute Gain: $+0.0161$)
* **Balanced Accuracy**: **$0.6054$** (Baseline: $0.5881$)
* **Overall Accuracy**: **$0.5922$**
* **Per-Class Breakdown**:
  * **Bulk Data**: Precision = $0.5815$, Recall = $0.5123$, F1 = $0.5447$ (Support: $1,177$)
  * **Real-Time**: Precision = $0.6280$, Recall = $0.5815$, F1 = $0.6039$ (Support: $2,122$)
  * **Streaming**: Precision = $0.5427$, Recall = $0.7224$, F1 = $0.6198$ (Support: $897$)

### 4.2. Network Quality of Service (QoS) Gains
* **Real-Time P95 Latency**: Dropped from $70.06\text{ ms}$ to $51.76\text{ ms}$ (**$-26.11\%$ improvement**, Hedges' $g = 8.28$)
* **Real-Time Packet Loss**: Dropped from $8.03\%$ to $1.05\%$ (**$-86.99\%$ reduction**, Hedges' $g = 8.07$)
* **Bulk Data Packet Loss**: Dropped from $57.26\%$ to $10.52\%$ (**$-81.64\%$ reduction**, Hedges' $g = 7.41$)
* **Bulk Data Goodput**: Increased from $1.66\text{ Mbps}$ to $8.56\text{ Mbps}$ (**$+416.89\%$ boost**, Hedges' $g = -5.24$)
* **Composite SLA Compliance**: Improved from **$0.0\%$** under default FIFO to **$58.0\%$** under ML-predicted scheduling (Low congestion regime: $100\%$ Real-Time compliance, $40\%$ Bulk compliance).

### 4.3. Early Packet Inspection Feature Vector
The champion model utilizes an 8-feature representation derived exclusively from the first 5 packets:
1. `early_5_complete`: Binary flag indicating whether 5 packets were observed before flow termination.
2. `early_5_duration`: Elapsed duration (seconds) across the first 5 packets.
3. `early_5_bytes`: Sum of payload and header bytes across the first 5 packets.
4. `early_5_fwd_pkts`: Count of forward packets in the early window.
5. `early_5_bwd_pkts`: Count of backward packets in the early window.
6. `early_5_avg_size`: Mean packet size across the first 5 packets (`early_5_bytes / 5`).
7. `is_udp`: Binary flag indicating UDP transport encapsulation ($1.0 = \text{UDP}, 0.0 = \text{TCP}$).
8. `early_5_fwd_ratio`: Ratio of forward packets to total early packets (`early_5_fwd_pkts / 5`).

---

## 5. Verification & Data Integrity Status

- **Zero Missing Values**: Master dataset `output/phase6d/clean_track_b_flows.csv` contains 0 null values across all 4,196 flows and 120 attributes.
- **Simulation Conservation Guarantee**: All 135 simulation runs in `output/phase7/run_metrics.csv` exhibit identical packet conservation (`Conservation_Error == 0`).
- **Runtime Integrity**: `output/runtime_verification/inferred_flows.csv` and `e2e_simulation_metrics.csv` confirm seamless operational stability with zero packet drops and zero extraction errors.
- **Data File Availability**: All 109 audited project files are active, accessible, and structured for real-time dashboard loading.
