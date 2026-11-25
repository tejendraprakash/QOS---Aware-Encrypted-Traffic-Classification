# PHASE 7.7: INDEPENDENT SCIENTIFIC AUDIT OF RESEARCH METHODOLOGY, DATA LEAKAGE RISKS, AND EVALUATION DESIGN

**Role:** Subagent 4 — Independent Scientific Reviewer & Research Auditor  
**Date:** November 2026  
**Target Repository:** [EncryptedTrafficResearch](file:///c:/Users/saipr/OneDrive/Desktop/EncryptedTrafficResearch)  
**Scope of Audit:** 
- Baseline Evaluation (`scripts/deep_validation_experiments.py`, `output/phase6d3/audit1_logo_horizon_comparison.csv`)
- Phase 7.6 Targeted Experiments (`scripts/test_phase7_6_fast_experiments.py`, `output/phase7_6_fast/`)
- 29-Fold Leave-One-Group-Out (LOGO) Protocol & Generalization Mechanics
- Data Leakage Vectors, Statistical Significance, and Downstream QoS Validity

---

## EXECUTIVE SUMMARY & AUDIT VERDICT

An exhaustive independent scientific audit was conducted across the dataset architecture ($N = 4,196$ flows across 29 capture groups), feature extraction pipelines, cross-validation protocols, and statistical claims.

### Summary of Audit Findings:
1. **Feature Extraction Causality:** The core Early-5 snapshot features (`early_5_complete`, `early_5_duration`, `early_5_bytes`, `early_5_fwd_pkts`, `early_5_bwd_pkts`, `early_5_avg_size`) and packet-1 protocol indicators (`is_udp`) are **strictly causal**. Tclusively on packets $k \le 5$ and layer-4 headers. Zero full-flow future packet leakage is present in the extracted features.
2. **Identification of Five Potential Data Leakage Vectors:** While the feature extractors are clean, critical risks exist regarding identifier memorization (`src_ip`, `dst_ip`, `src_port`, `dst_port`), target leakage (`capture_group`, `application`), global dataset preprocessing (fitting scalers across folds), and global inactivity timeout assumptions.
3. **Phase 7.6 Statistical Reality Check:** The reported Macro $F_1$ gain of $+0.0068$ ($0.5734 \to 0.5802$) in Phase 7.6 Experiment 3 (Targeted Weighting $1.35\times$) is **NOT statistically significant**:
   - **Flow-Level McNemar Test:** $n_{01} = 146$ (improved), $n_{10} = 134$ (degraded). Exact Binomial test $p = 0.5110$; Continuity-corrected $\chi^2 = 0.4321, p = 0.5109$.
   - **Group-Level Paired Wilcoxon Test:** Across 29 capture groups, $W = 71.0, p = 0.5277$.
   - **Mechanism:** The model simply swapped 134 errors in Real-Time/Streaming for 146 corrections in Bulk Data. Due to group support differences, unweighted Macro $F_1$ moved slightly upward while overall accuracy changed by only $+0.28\%$ (+12 net flows).
4. **Outer-Fold Test Dredging Warning:** Directly tuning hyperparameter multipliers (such as $1.35\times$ Bulk weight) against the 29-fold outer LOGO test score violates strict generalization principles. Any candidate champion must be discovered via **Nested Group-Aware Cross-Validation** or fixed *a priori*.
5. **Group Imbalance Pathology:** All 29 capture groups are **100% monotypic** (each group contains flows from exactly one QoS class). Two dominant groups (`GRP_VPN_SKYPE_AUDIO_1` with 890 flows and `GRP_FTPS_DOWN_1` with 871 flows) account for **$41.97\%$** of the entire dataset. In contrast, 6 groups have fewer than 10 flows (12 flows total). This necessitates dual sensitivity reporting.
6. **Six Mandatory Verification Gates:** Formulated below to govern the adoption of any Phase 7.7 champion model.

---

## 1. POTENTIAL DATA LEAKAGE VECTORS & PROTOCOL AUDIT

In high-stakes network traffic analysis, data leakage often produces artificially inflated evaluation metrics ($F_1 > 0.95$) that collapse completely in live deployments. We audited five potential leakage vectors:

### 1.1 Identifier Leakage (IP Addresses, Ports, MACs)
- **Vector Description:** IP addresses and port numbers reflect specific server hosting clusters, CDN IP ranges, and local testbed configurations (e.g., Netflix/YouTube server subnets). A tree-based model trained on raw IP addresses or ephemeral port numbers will memorize endpoint identities rather than statistical flow dynamics.
- **Audit Findings:** 
  - `clean_track_b_flows.csv` stores metadata columns: `src_ip`, `dst_ip`, `src_port`, `dst_port`.
  - In `scripts/deep_validation_experiments.py` and `scripts/test_phase7_6_fast_experiments.py`, features are explicitly selected from feature lists (`feats_base`, `feats_enhanced`), ensuring IP/port columns are excluded.
  - **Audit Rule:** No candidate model may ingest raw IPs, ports, MAC addresses, or autonomous system numbers (ASNs).

### 1.2 Target & Capture-Group Leakage (`capture_group`, `application`)
- **Vector Description:** Because capture groups correspond to dedicated capture sessions (e.g., `GRP_FTPS_DOWN_1`, `GRP_NETFLIX_1`), knowledge of the capture group identifier is equivalent to knowing the ground-truth application and QoS class.
- **Audit Findings:** Every single capture group in Track B has **100% class purity** ($1.000$ entropy). Including `capture_group`, `capture_filename`, or `application` in training would result in instantaneous target leakage.
- **Audit Rule:** These columns must be restricted strictly to splitting logic and post-hoc subpopulation reporting.

### 1.3 Temporal Lookahead & Future Flow Information
- **Vector Description:** Using complete-flow statistics (e.g., total flow duration, total packet count, total bytes, backward-to-forward packet ratio across the entire flow, or TCP teardown FIN/RST flags) in an early classification system violates causality.
- **Audit Findings:**
  - Audited `scripts/flow_extractor.py` lines 375–401:
    ```python
    sub_seq = self.direction_sequence[:N]
    k = len(sub_seq)
    complete_flag = 1 if total_pkts >= N else 0
    sub_bytes = sum(sz for _, sz in sub_seq)
    sub_dur = self.all_timestamps[k - 1] - self.all_timestamps[0]
    ```
  - For $N = 5$, metrics are computed strictly over packets $1 \dots \min(\text{total}, 5)$.
  - `early_5_complete`: $44.35\%$ ($1,861$ flows) reach 5 packets; $55.65\%$ ($2,335$ flows) terminate early. In an online system, decision triggers at packet 5 for persistent flows, or upon inactivity timeout/FIN for sub-5 flows. It does not disclose future packets.
  - In Phase 7.6: `is_udp` is extracted from `protocol == 17` (IP header protocol field in packet 1), which is strictly causal.
  - **Audit Rule:** Any newly engineered feature must be mathematically provable to be computable at or before $t_5$.

### 1.4 Global Preprocessing Leakage (Across Cross-Validation Folds)
- **Vector Description:** Fitting feature scalers (`StandardScaler`, `MinMaxScaler`), power transformers, quantile mappers, or imputation schemes across the entire dataset prior to folding leaks test-fold distribution parameters (mean, standard deviation, quantiles) into the training pipeline.
- **Audit Findings:** In Phase 7.6, XGBoost and Random Forest were used without continuous scaling, avoiding continuous distribution leakage. However, in `scripts/deep_validation_experiments.py`, if standard scaling or calibration is performed, it must be contained strictly within the training fold.
- **Audit Rule:** All stateful feature transformations must be instantiated inside an `sklearn.pipeline.Pipeline` fit strictly on $X_{\text{train}}$.

### 1.5 Global Inactivity Timeout Assumptions
- **Vector Description:** Flows containing $< 5$ packets are classified at termination. If a model assumes an instantaneous "flow ended" signal without accounting for the standard inter-packet timeout ($T_{\text{idle}} \in [15\text{s}, 60\text{s}]$), packet delays during simulation will be underestimated.
- **Audit Rule:** Downstream queueing simulations must enforce realistic timeout delays for short-lived flows before applying post-termination classification decisions.

---

## 2. AUDIT OF 29-FOLD LEAVE-ONE-GROUP-OUT (LOGO) PROTOCOL & PHASE 7.6 EXPERIMENTS

### 2.1 Monotypic Group Structure & Evaluation Inconsistencies
Cross-tabulation of all 29 capture groups against QoS classes reveals an essential structural property:

```
Group Breakdown:
- 10 Groups = Bulk Data  (1,177 flows: FTPS Down, FTPS Up, VPN FTPS, SCP 1-6)
- 10 Groups = Real-Time  (2,122 flows: FB Audio, Hangouts Audio/Video, Skype Audio/Video, VoipBuster, VPN counterparts)
-  9 Groups = Streaming  (  897 flows: FB Video, Netflix, Spotify, Vimeo, YouTube, VPN counterparts)
Total: 29 Groups, 4,196 Flows.
```

**Critical Mathematical Consequence:**
Because every group contains **only one QoS class**:
- In any single LOGO test fold, $y_{\text{test}}$ has cardinality $|C| = 1$.
- Computing standard unweighted Macro $F_1$ per fold is mathematically ill-posed: precision and recall for absent classes are undefined ($0/0$).
- **Methodological Standard:** LOGO predictions must be accumulated across all 29 out-of-fold partitions into a complete Out-Of-Fold (OOF) prediction vector $\hat{y}_{\text{OOF}} \in \{0, 1, 2\}^N$, on which the global Pooled Macro $F_1$ and Pooled Balanced Accuracy are evaluated.
- Comparing an average of fold-level $F_1$ scores against pooled $F_1$ is an **invalid comparison**. All historical benchmarks must adhere to the pooled formulation.

### 2.2 Deep Statistical Audit of Phase 7.6 "Targeted Improvement"
Phase 7.6 reported an improvement from Macro $F_1 = 0.5734$ to $0.5802$ ($+0.0068$) using Experiment 3 (Targeted Class Weighting: Bulk Data boosted $1.35\times$).

We conducted an independent re-execution and paired hypothesis test:

#### A. Flow-Level McNemar Paired Test ($N = 4,196$)
We constructed the contingency matrix between Baseline ($C_1$) and Targeted Weighting ($C_2$):
- Both Correct ($n_{11}$): **$2,282$ flows**
- Both Incorrect ($n_{00}$): **$1,634$ flows**
- Baseline Correct, Target Incorrect ($n_{10}$): **$134$ flows**
- Baseline Incorrect, Target Correct ($n_{01}$): **$146$ flows**

```
Contingency Matrix:
                  Targeted Correct    Targeted Incorrect
Baseline Correct        2,282                134
Baseline Incorrect        146              1,634
```

- **Exact Two-Sided Binomial Test:**
  $$p = \sum_{k=146}^{280} \binom{280}{k} (0.5)^k (0.5)^{280-k} = \mathbf{0.51101}$$
- **Edwards Continuity-Corrected McNemar $\chi^2$:**
  $$\chi^2 = \frac{(|146 - 134| - 1)^2}{146 + 134} = \frac{121}{280} = 0.4321 \quad (p = \mathbf{0.51094})$$

#### B. Group-Level Paired Wilcoxon Signed-Rank Test ($N = 29$)
Across the 29 capture groups:
- 11 groups exhibited zero change in accuracy.
- 7 groups exhibited increased accuracy (e.g., `GRP_FTPS_DOWN_1`: $+6.77\%$, `GRP_VPN_HANGOUTS_AUDIO_1`: $+12.78\%$).
- 11 groups exhibited decreased accuracy (e.g., `GRP_VPN_SKYPE_AUDIO_1`: $-7.53\%$, `GRP_HANGOUTS_VIDEO_1`: $-8.96\%$, `GRP_SCP_3`: $-33.33\%$, `GRP_SCP_5`: $-50.00\%$).
- **Wilcoxon Test Statistic:** $W = 71.0, \quad p = \mathbf{0.52773}$.

#### C. Scientific Verdict on Phase 7.6 Gain
The $+0.0068$ Macro $F_1$ gain is **statistically indistinguishable from random noise** ($p > 0.50$). It reflects an arbitrary trade-off: boosting Bulk Data weight forced the classifier to predict Bulk more frequently, which naturally corrected 146 Bulk flows but simultaneously degraded 134 Real-Time and Streaming flows. Claiming this as an architectural breakthrough in an IEEE/ACM paper would fail peer review.

---

## 3. RISKS OF TEST-DATA DREDGING & OUTER-FOLD OVERFITTING

### 3.1 The Outer LOGO Feedback Loop
In Phase 7.6, five hand-crafted model variants were evaluated directly against the 29 outer LOGO test folds, and the configuration scoring highest ($0.5802$) was selected as "champion."

**Methodological Risk:**
- The 29 outer folds represent the final generalization test benchmark.
- Evaluating multiple candidate hyperparameters (depths $\{3, 5, 8\}$, multipliers $\{1.0, 1.35\}$, feature sets $\{6, 9\}$) directly on the outer folds converts the test folds into an informal validation set.
- This creates **optimistic selection bias**: picking the peak of the empirical noise distribution across the 29 folds.

```
FLAWED WORKFLOW (Test-Data Dredging):
[Data: 29 Groups] 
      │
      ▼
Outer 29-Fold LOGO Test Splits ──► Tune Multiplier (1.35x), Depth (3 vs 5), Features ──► Pick Best Outer Score (0.5802)
                                                                                                  ▲
                                                                                                  │
                                                                   LEAKAGE: Outer Test Informs Selection!
```

### 3.2 Correct Nested Group-Aware Evaluation Design
To maintain pristine IEEE/ACM transactions scientific rigor, model selection (hyperparameter tuning, feature subset selection, threshold tuning) must be executed strictly within an **Inner Cross-Validation Loop**:

```
RIGOROUS NESTED WORKFLOW:
Outer Loop: 29 LOGO Folds (Fold i: Hold out Group i as unseen test set)
  │
  ├── Training Data: 28 Groups
  │     │
  │     ├── Inner Loop: Stratified Group K-Fold (K=5) or Inner LOGO on the 28 Groups
  │     │     ├── Evaluate Candidate Hyperparameters θ ∈ Θ
  │     │     └── Select Optimal Hyperparameter θ* purely on Inner Validation Score
  │     │
  │     └── Fit Final Outer-Fold Model using θ* on all 28 Training Groups
  │
  └── Evaluate ONCE on Unseen Outer Group i (Test Set)
```

**Alternative Acceptable Path:**
If full nested cross-validation is computationally expensive, the coordinator may define a **Frozen Hypothesis Protocol**:
1. Fix all model hyperparameters, feature sets, and loss weights *a priori* based on established theory or domain principles.
2. Run the 29-fold LOGO evaluation exactly **once**.
3. Report the result transparently without post-hoc tuning on outer scores.

---

## 4. CAPTURE-GROUP IMBALANCE EFFECTS & SENSITIVITY DESIGN

### 4.1 Quantitative Imbalance Profile
The dataset exhibits extreme subpopulation skew:

| Category | Group Name | Flow Count | % of Dataset | % of Class | Cumulative Impact |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Dominant Group 1** | `GRP_VPN_SKYPE_AUDIO_1` | 890 | $21.21\%$ | $41.94\%$ (RT) | 890 flows |
| **Dominant Group 2** | `GRP_FTPS_DOWN_1` | 871 | $20.76\%$ | $74.00\%$ (Bulk) | 1,761 flows ($41.97\%$) |
| **Top 4 Groups** | Top 2 + VoipBuster + VPN FB Audio | 2,494 | $59.44\%$ | — | 2,494 flows |
| **Micro Groups ($N < 10$)** | 6 SCP Groups (`GRP_SCP_1`..`6`) | 12 | $0.28\%$ | $1.02\%$ (Bulk) | 12 flows total |

### 4.2 Impact on Cross-Validation & Generalization
1. **The FTPS Down Anomaly:** When `GRP_FTPS_DOWN_1` is held out as the test group:
   - The training set contains only $1,177 - 871 = 306$ Bulk Data flows, alongside $2,122$ Real-Time flows ($7:1$ imbalance).
   - In the test fold, $871$ Bulk flows must be predicted. Any drop in Bulk recall on this single fold severely depresses global pooled metrics.
   - Removing `GRP_FTPS_DOWN_1` drops baseline Macro $F_1$ from $0.5734 \to 0.5110$.
2. **Micro-Group Volatility:** In `GRP_SCP_4` ($N=1$) and `GRP_SCP_6` ($N=1$), a single error yields $0\%$ accuracy. Averaging per-group accuracies gives equal weight ($1/29 \approx 3.45\%$) to a 1-flow group and an 890-flow group.
3. **Mandatory Reporting Strategy:** The manuscript must present **Dual-Track Evaluation**:
   - Primary: Full 29-Fold LOGO ($N = 4,196$).
   - Sensitivity: 28-Fold LOGO excluding `GRP_FTPS_DOWN_1` ($N = 3,325$).

---

## 5. DEFECTIVE VS. DEFENSIBLE SCIENTIFIC CLAIMS (IEEE/ACM TRANSACTIONS STYLE)

| Research Dimension | Defective Claim (Vulnerable to Rejection) | Defensible Scientific Claim (IEEE/ACM Standard) |
| :--- | :--- | :--- |
| **Performance Gain** | *"Our targeted weighting breaks the early classification barrier, achieving superior state-of-the-art Macro $F_1$ of 0.5802."* | *"Targeted class weighting yielded an incremental shift in pooled Macro $F_1$ ($+0.0068$, $0.5734 \to 0.5802$), primarily improving Bulk Data recall ($+4.85\%$) at the cost of Real-Time recall ($-2.21\%$). A paired McNemar test ($p = 0.511$) and group-level Wilcoxon test ($p = 0.528$) confirm this difference is not statistically significant across capture environments, demonstrating that early packet representations face a structural indistinguishability floor at packet 5."* |
| **Early Feature Availability** | *"Our classifier detects application types immediately using early packet features."* | *"Features are extracted strictly from packets $1 \dots \min(\text{total}, 5)$. Classification triggers at packet 5 for persisting connections ($44.4\%$ of flows). For short transactions under 5 packets ($55.6\%$), decisions occur upon flow termination (FIN/RST or inactivity timeout), where `early_5_complete = 0` causally encodes the truncated window."* |
| **Sample Size & Significance** | *"Our QoS improvements are statistically verified at $p < 0.05$ across 5 simulation seeds."* | *"Across $N = 15$ independent seeds, paired Wilcoxon signed-rank tests confirmed delay and goodput improvements at exact combinatorial significance ($p = 6.1035 \times 10^{-5}$). We note that for $N = 5$, the minimum achievable two-sided Wilcoxon $p$-value is $0.0625$, necessitating $N \ge 6$ (and here $N = 15$) for formal rejection of the null hypothesis at $\alpha = 0.05$."* |
| **Generalization Claim** | *"The model achieves high accuracy across random splits of the dataset."* | *"Standard random $K$-fold cross-validation yields inflated accuracy ($> 95\%$) due to capture session memorization. Under rigorous Leave-One-Group-Out cross-validation across 29 disjoint network sessions, generalization Macro $F_1$ is $0.5734$, reflecting true zero-day environment transferability."* |

---

## 6. SIX MANDATORY VERIFICATION GATES FOR PHASE 7.7 CHAMPION ADOPTION

Before the coordinator may adopt any Phase 7.7 model as the project champion, the candidate model must satisfy all six non-negotiable verification gates:

```
┌────────────────────────────────────────────────────────────────────────┐
│               PHASE 7.7 SIX SCIENTIFIC VERIFICATION GATES             │
├────────────────────────────────────────────────────────────────────────┤
│ GATE 1: Zero-Leakage & Protocol Causality Verification                 │
│         - No IPs, ports, capture group IDs, timestamps, or full-flow   │
│           features in input matrix. Code audit of feature pipeline.    │
├────────────────────────────────────────────────────────────────────────┤
│ GATE 2: Leak-Free Nested Tuning or A-Priori Hyperparameter Freeze      │
│         - Model selection conducted strictly via inner CV on training  │
│           groups, or fixed a priori without outer LOGO optimization.   │
├────────────────────────────────────────────────────────────────────────┤
│ GATE 3: Paired Statistical Significance & Effect Size Threshold        │
│         - Minimum Macro F1 gain of +0.0150 over baseline (0.5734).     │
│         - McNemar flow-level test p < 0.05 AND group-level Wilcoxon    │
│           p < 0.05. Both tests must independently reject H0.           │
├────────────────────────────────────────────────────────────────────────┤
│ GATE 4: Subpopulation Stability & Class Floor Protection               │
│         - Real-Time recall must not fall below 54.0%.                  │
│         - Streaming recall must not fall below 65.0%.                  │
│         - Sensitivity validation: Macro F1 without FTPS Down >= 0.510. │
├────────────────────────────────────────────────────────────────────────┤
│ GATE 5: Closed-Loop Downstream QoS Simulation Verification             │
│         - Candidate must be evaluated in the discrete-event queueing   │
│           simulator across Low (50%), Moderate (85%), and High (125%). │
│         - Must maintain strict Real-Time P95 delay <= 45 ms and packet │
│           loss <= 4.0% under 125% overload without bulk starvation.   │
├────────────────────────────────────────────────────────────────────────┤
│ GATE 6: Cryptographic Reproducibility & Artifact Manifestation         │
│         - Serialized JSON model, configuration schema, confusion matrix│
│           CSV, and SHA-256 manifest recorded in output/phase7_7/.      │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 7. RECOMMENDATIONS FOR THE PHASE 7.7 COORDINATOR

1. **Retain Baseline Early-5 ($F_1 = 0.5734$) as Solid Scientific Benchmark:** Do not rush to claim minor $+0.0068$ fluctuations as victories. In top-tier networking venues (IEEE TNSM, IEEE INFOCOM, ACM SIGCOMM), demonstrating that early classification hits a well-characterized performance ceiling under zero-day LOGO conditions is considered a strong, honest, and publishable scientific contribution.
2. **Focus Innovation on Confidence-Aware Fallbacks & Scheduling:** The true breakthrough in this project is not pushing noisy Early-5 classification from $57\%$ to $58\%$, but rather the **Confidence-Aware Closed-Loop Scheduler ($\tau = 0.55$)**, which gracefully handles classification uncertainty and prevents high-load queue starvation.
3. **Formalize Nested LOGO in Code:** If feature engineering (e.g., packet 1–5 inter-arrival time statistics, packet size skewness) is pursued, implement an inner `GroupKFold(n_splits=5)` inside each outer LOGO fold to tune regularization parameters.
4. **Publish Dual-Track Metrics in All Tables:** Include full 29-group results and 28-group results (without `GRP_FTPS_DOWN_1`) side-by-side to preempt any reviewer concerns regarding capture imbalance.

---
*Report certified by Subagent 4 (Independent Scientific Auditor) for Phase 7.7.*
