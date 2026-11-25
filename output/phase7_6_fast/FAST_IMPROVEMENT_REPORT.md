# PHASE 7.6: FAST TARGETED EARLY-5 CLASSIFIER IMPROVEMENT REPORT

**Objective:** Rapid, targeted investigation and optimization of the Early-5 encrypted traffic classifier under strict 29-fold Leave-One-Group-Out (LOGO) cross-validation.  
**Integrity Rule:** Zero future leakage, zero IP/port/timestamp usage, strictly preserved 29 capture groups.  

---

## 1. Pipeline Inspection & Confusion Matrix Analysis

1. **Feature Leakage & Availability Audit:**
   - All 6 Early-5 features (`early_5_complete`, `early_5_duration`, `early_5_bytes`, `early_5_fwd_pkts`, `early_5_bwd_pkts`, `early_5_avg_size`) were re-verified at the source code level (`scripts/flow_extractor.py`).
   - All features strictly use packets $1 \le k \le \min(\text{total}, 5)$ and arrival timestamps $t_1 \dots t_k$. Zero complete-flow duration or future packet metadata is accessed.
2. **Primary Classification Bottleneck Identified:**
   - Inspection of the normalized LOGO confusion matrix revealed that **Real-Time is a strong attractor**:
     * $44.86\%$ of true Bulk Data flows are misclassified as Real-Time.
     * $28.21\%$ of true Streaming flows are misclassified as Real-Time.
   - Root Cause: Real-Time comprises $50.6\%$ of dataset flows ($2,122 / 4,196$). Furthermore, $55.6\%$ of dataset flows have fewer than 5 packets ($2,335$ flows, primarily brief handshakes, audio bursts, or aborted transfers), where small packet count and low byte count mimic VoIP frames.

---

## 2. Experiments Performed (Strict 29-Fold LOGO)

We evaluated exactly 5 targeted, fast experiments against the verified baseline without altering group definitions:

| Experiment | Description | Feature Count | Macro $F_1$ | $\Delta F_1$ | Balanced Acc | Overall Acc | Bulk Recall | RT Recall | Str Recall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | 6 features, XGBoost depth=5, Inverse Freq Weights | 6 | **0.5734** | $0.0000$ | 0.5881 | 0.5758 | 0.5021 | 0.5655 | 0.6968 |
| **Exp 1: Enhanced Features** | + `early_5_fwd_ratio`, `early_5_byte_rate`, `is_udp` | 9 | **0.5635** | $-0.0099$ | 0.5767 | 0.5660 | 0.4945 | 0.5589 | 0.6767 |
| **Exp 2: Regularized XGBoost** | Shallower depth=3, $\lambda=1.5$, min_child_weight=3 | 6 | **0.5062** | $-0.0672$ | 0.5360 | 0.5086 | 0.3798 | 0.4779 | 0.7503 |
| **Exp 3: Targeted Weighting** | Boost Bulk Data weight $1.35\times$ to counter RT attractor | 6 | **0.5802** | **$+0.0068$** | **0.5976** | **0.5786** | **0.5506** | 0.5434 | 0.6990 |
| **Exp 4: Alternative Model** | Random Forest (150 trees, max_depth=8, min_samples=4) | 6 | **0.5513** | $-0.0221$ | 0.5735 | 0.5539 | 0.4545 | 0.5335 | 0.7324 |
| **Exp 5: Combined Champion** | Enhanced 9 feats + Regularized XGB + Targeted Weights | 9 | **0.5608** | $-0.0126$ | 0.5869 | 0.5591 | 0.4826 | 0.5156 | 0.7625 |

---

## 3. Best Performance & Scientific Analysis

- **Champion Setting:** **Experiment 3 (Targeted Class Weighting: Bulk Boost $1.35\times$ on 6 Early-5 Features)**.
- **Performance Gains:**
  - Macro $F_1$: **$0.5734 \longrightarrow 0.5802$** ($+0.0068$ / $+0.68\%$)
  - Balanced Accuracy: **$0.5881 \longrightarrow 0.5976$** ($+0.0095$ / $+0.95\%$)
  - Overall Accuracy: **$0.5758 \longrightarrow 0.5786$** ($+0.0028$)
  - **Bulk Data Recall:** **$50.21\% \longrightarrow 55.06\%$** ($+4.85$ percentage points)
  - Bulk Data misclassified as Real-Time decreased from $44.86\% \longrightarrow 40.53\%$ ($-4.33\%$).
- **Is the Improvement Statistically / Practically Meaningful?**
  - **Practically:** **Yes.** Boosting Bulk Data recall by $+4.85\%$ directly protects high-volume TCP bulk elephant flows from being misclassified into the Strict Priority Real-Time queue, mitigating downstream queue starvation in the scheduler.
  - **Scientifically / Statistically:** **Incremental.** A $+0.68\%$ Macro $F_1$ gain is a modest refinement, not a structural breakthrough. The physical limit of Early-5 classification across 29 unseen capture environments is bounded by early packet indistinguishability: at packet 5, short connection handshakes without sustained payload exhibit substantial structural overlap across application classes.

---

## 4. Generated Artifacts in `output/phase7_6_fast/`

1. `fast_experiments_comparison.csv`: Full metric scorecard across all 5 experiments.
2. `improved_early5_model.json`: Serialized XGBoost model trained on clean data with optimal weighting.
3. `improved_model_config.json`: Hyperparameters, feature schema, and LOGO metadata.
4. `improved_confusion_matrix.csv`: Normalized LOGO confusion matrix for the improved model.

---

## 5. Exact Execution Command

To execute the end-to-end pipeline with the improved Early-5 model:

```bash
python scripts/run_pipeline_end_to_end.py --pcap data/raw/scpDown4.pcap --model output/phase7_6_fast/improved_early5_model.json --regime Moderate --sim-time 60.0
```
