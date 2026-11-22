# Advanced Research Lab (ARL) — Artifact Registry

**Project:** QoS-Aware Encrypted Network Traffic Classification for Next-Generation Networks  
**Lead Research Orchestrator:** AI Research Lab Manager  
**Last Updated:** November 2026  

---

## 1. Primary Dataset Artifacts

| Artifact Name | Path | Size | Description / Schema | Provenance |
| :--- | :--- | :---: | :--- | :--- |
| `track_b_flows.csv` | `output/phase5/track_b_flows.csv` | 7.26 MB | 10,770 flows $\times$ 117 features. Ground-truth benchmark for Real-Time, Streaming, Bulk Data. | Extracted in Phase 5 from ISCX VPN-nonVPN raw PCAPs. |
| `track_b_folds.csv` | `output/phase5/track_b_folds.csv` | 853 KB | 5-fold non-overlapping cross-validation assignments grouped by `capture_group`. | Phase 5 leak-free split generator. |
| `early_feature_definitions.json`| `output/phase5/early_feature_definitions.json` | 4.2 KB | Formal specification of early packet classification horizons ($N \in \{5, 10, 15, 20\}$). | Phase 5 baseline specs. |
| `model_pipeline_specs.json` | `output/phase5/model_pipeline_specs.json` | 5.3 KB | Preprocessing rules, collinearity filters, and model hyperparameters. | Phase 5 pipeline definitions. |

---

## 2. Machine Learning Model Artifacts (Phase 6 / 6B)

| Artifact Name | Path | Horizon | Model Type | Verified Metric (Track B) |
| :--- | :--- | :---: | :---: | :---: |
| `early_5` Experiment Config | `output/phase6b/early_5/experiment_config.json` | Early-5 (5 pkts) | Config | Macro F1 $= 0.8524$ (XGBoost candidate model) |
| `early_5` Normalized CM | `output/phase6b/early_5/confusion_matrix_normalized.csv` | Early-5 (5 pkts) | Confusion Matrix | Empirical $3 \times 3$ class transition probabilities. |
| `early_5` Out-of-Fold Preds | `output/phase6b/early_5/out_of_fold_predictions.csv` | Early-5 (5 pkts) | CSV Predictions | 10,770 cross-validated predictions and probabilities. |
| `early_10` Benchmarks | `output/phase6b/early_10/` | Early-10 (10 pkts) | Multi-Model | Macro F1 $= 0.8812$ |
| `early_15` Benchmarks | `output/phase6b/early_15/` | Early-15 (15 pkts) | Multi-Model | Macro F1 $= 0.8945$ |
| `early_20` Benchmarks | `output/phase6b/early_20/` | Early-20 (20 pkts) | Multi-Model | Macro F1 $= 0.9018$ |

---

## 3. Phase 7 Simulation & Research Reports

| Artifact Name | Path | Size | Description | Approval Status |
| :--- | :--- | :---: | :--- | :---: |
| `traffic_generation_design.md` | `research_notes/phase7/traffic_generation_design.md` | 45.0 KB | Initial baseline traffic generation report. | Superceded by Final Spec |
| `traffic_generation_audit.md` | `research_notes/phase7/traffic_generation_audit.md` | 38.5 KB | 7-agent comprehensive technical audit report. | **APPROVED** |
| `traffic_generation_corrections.md`| `research_notes/phase7/traffic_generation_corrections.md`| 28.2 KB | Itemized mathematical and code corrections. | **APPROVED** |
| `traffic_generation_final_specification.md`| `research_notes/phase7/traffic_generation_final_specification.md`| 44.8 KB | Authoritative production specification with verified SimPy code. | **APPROVED** |
| `traffic_generation_validation_audit.md` | `research_notes/phase7/traffic_generation_validation_audit.md` | 34.0 KB | 4-Gate Statistical Validation Protocol. | **APPROVED** |

---

## 4. Production Scripts & Simulation Engines

| Script Name | Path | Size | Role in Pipeline | Status |
| :--- | :--- | :---: | :--- | :---: |
| `flow_extractor.py` | `scripts/flow_extractor.py` | 34.8 KB | PCAP flow parsing and early feature extraction. | Production / Baseline |
| `run_phase6b_experiments.py`| `scripts/run_phase6b_experiments.py` | 21.0 KB | Early classification training and cross-validation engine. | Production / Baseline |
| `simulate_phase7_qos.py` | `scripts/simulate_phase7_qos.py` | 23.2 KB | Phase 7 discrete-event network queue simulator. | Target for Refactoring |
| `evaluate_phase7_results.py` | `scripts/evaluate_phase7_results.py` | 7.4 KB | Statistical evaluation and SLA compliance reporting. | Target for Refactoring |
