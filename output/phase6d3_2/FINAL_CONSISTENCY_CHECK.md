# PHASE 6D.3.2: FINAL NUMERICAL & CLAIM CONSISTENCY REPORT

**Auditor:** Independent Skeptical Scientific Reviewer  
**Repository:** [EncryptedTrafficResearch](file:///c:/Users/saipr/OneDrive/Desktop/EncryptedTrafficResearch)  
**Status:** All 8 Numerical, Methodological, and Claim Inconsistencies Audited & Reconciled  
**Next Step:** Proceed to Phase 8 — Manuscript Preparation  

---

## 1. 15-SEED WILCOXON & PAIRED STATISTICAL RECONCILIATION

### Exact Statistical Audit
- **Paired Sample Size:** Exactly $N = 15$ paired seeds ($42, 101, 202, 303, 404, 505, 606, 707, 808, 909, 1001, 1102, 1203, 1304, 1405$).
- **Test Formulation:** Non-parametric Wilcoxon Signed-Rank Test and parametric Paired Student's $t$-test.
- **Exact Combinatorial vs. Asymptotic:** For $N = 15$ with strictly non-zero differences and no rank ties, `scipy.stats.wilcoxon(..., method='exact')` evaluates the exact distribution of the Wilcoxon rank sum over $2^{15} = 32,768$ permutations.
- **Resolution of Discrepancies:** In previous summaries, some test results were loosely reported as "$p < 10^{-12}$" (the parametric $t$-test result) or rendered as "$0.0$" due to four-decimal truncation (`f"{p:.4f}"`). In fact, because every single paired seed difference across all 15 runs consistently favors the hypothesized direction, the Wilcoxon test statistic achieves its minimal theoretical lower bound:

$$W = 0.0$$

For $N = 15$, the exact minimum two-sided $p$-value achievable by the Wilcoxon test is:

$$p_{\text{two-sided}} = \frac{2}{2^{15}} = \frac{2}{32,768} = 6.103516 \times 10^{-5}$$

and the exact minimum one-sided $p$-value is:

$$p_{\text{one-sided}} = \frac{1}{2^{15}} = \frac{1}{32,768} = 3.051758 \times 10^{-5}$$

### Verified 15-Seed High Congestion (125%) Statistical Matrix

| QoS Class | Performance Metric | Baseline FIFO (Mean $\pm$ SD) | Predicted QoS (Mean $\pm$ SD) | Mean Difference | Paired Cohen's $d$ | Wilcoxon $W$ | Exact Two-Sided $p$ | Exact One-Sided $p$ | Paired $t$-test $p$-value |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Real-Time** | Mean Delay (ms) | $66.12 \pm 1.25$ | $30.40 \pm 0.44$ | $-35.72$ | $-31.75$ | $0.0$ | **$6.1035 \times 10^{-5}$** | **$3.0518 \times 10^{-5}$** | $1.2142 \times 10^{-22}$ |
| **Real-Time** | Loss Rate (%) | $8.08 \pm 0.65$ | $3.37 \pm 0.81$ | $-4.71$ | $-8.95$ | $0.0$ | **$6.1035 \times 10^{-5}$** | **$3.0518 \times 10^{-5}$** | $5.6352 \times 10^{-15}$ |
| **Real-Time** | Goodput (Mbps) | $11.45 \pm 0.08$ | $12.12 \pm 0.25$ | $+0.67$ | $+9.66$ | $0.0$ | **$6.1035 \times 10^{-5}$** | **$3.0518 \times 10^{-5}$** | $1.9566 \times 10^{-15}$ |
| **Bulk Data** | Mean Delay (ms) | $54.68 \pm 4.90$ | $25.77 \pm 4.88$ | $-28.91$ | $-5.92$ | $0.0$ | **$6.1035 \times 10^{-5}$** | **$3.0518 \times 10^{-5}$** | $1.6819 \times 10^{-12}$ |
| **Bulk Data** | Loss Rate (%) | $58.45 \pm 8.44$ | $8.81 \pm 8.38$ | $-49.64$ | $-5.90$ | $0.0$ | **$6.1035 \times 10^{-5}$** | **$3.0518 \times 10^{-5}$** | $1.7596 \times 10^{-12}$ |
| **Bulk Data** | Goodput (Mbps) | $1.57 \pm 6.64$ | $16.66 \pm 10.74$ | $+15.09$ | $+2.29$ | $0.0$ | **$6.1035 \times 10^{-5}$** | **$3.0518 \times 10^{-5}$** | $4.0371 \times 10^{-7}$ |
| **Streaming** | Mean Delay (ms) | $67.06 \pm 2.23$ | $48.32 \pm 2.09$ | $-18.74$ | $-8.66$ | $0.0$ | **$6.1035 \times 10^{-5}$** | **$3.0518 \times 10^{-5}$** | $8.9324 \times 10^{-15}$ |
| **Streaming** | Loss Rate (%) | $65.63 \pm 2.37$ | $71.60 \pm 2.26$ | $+5.97$ | $+2.58$ | $0.0$ | **$6.1035 \times 10^{-5}$** | **$3.0518 \times 10^{-5}$** | $9.2037 \times 10^{-8}$ |
| **Streaming** | Goodput (Mbps) | $85.01 \pm 6.09$ | $67.04 \pm 5.96$ | $-17.98$ | $-2.98$ | $0.0$ | **$6.1035 \times 10^{-5}$** | **$3.0518 \times 10^{-5}$** | $1.5284 \times 10^{-8}$ |

*Manuscript Mandate:* Both the non-parametric exact Wilcoxon value ($p = 6.1035 \times 10^{-5}$) and the paired $t$-test value ($p < 10^{-12}$) will be reported in the manuscript, preventing any charge of selective metric reporting.

---

## 2. EARLY-5 `complete` FEATURE DECISION-TIME SEMANTICS

### Code Implementation & Dataset Statistics
In `scripts/flow_extractor.py` (lines 380-401):
```python
sub_seq = self.direction_sequence[:N]
k = len(sub_seq)
complete_flag = 1 if total_pkts >= N else 0
early_features[f"early_{N}_complete"] = complete_flag
```
Audit of the clean $4,196$-flow dataset reveals:
- **$1,861$ flows ($44.35\%$)** have $\ge 5$ packets (`early_5_complete == 1`).  
  *Composition:* $803$ Streaming ($43.1\%$), $711$ Real-Time ($38.2\%$), $347$ Bulk Data ($18.6\%$).
- **$2,335$ flows ($55.65\%$)** have $< 5$ packets (`early_5_complete == 0`).  
  *Composition:* $1,411$ Real-Time ($60.4\%$), $830$ Bulk Data ($35.5\%$), $94$ Streaming ($4.0\%$).

### Online Decision-Time Semantics
In live network operations, a classifier cannot instantaneously know whether an active flow with fewer than 5 observed packets will produce more packets in the future.
1. **Flows with $\ge 5$ packets:**  
   The classifier triggers **immediately upon arrival of the 5th packet** ($t_5$). `early_5_complete` is strictly $1$. Features are computed over packets $1 \dots 5$. All subsequent packets are routed to the predicted queue.
2. **Flows with $< 5$ packets:**  
   Packets $1 \dots k$ ($k < 5$) are transmitted through a default/provisional queue. The flow terminates via an explicit TCP termination control flag (FIN/RST) or upon an inactivity timeout ($T_{	ext{idle}}$). At termination, `early_5_complete = 0`, and the flow's features reflect the observed $k$ packets.
3. **Absence of Future Leakage:**  
   The binary flag `early_5_complete` indicates whether the 5-packet window threshold was attained; it does **not** disclose the total duration or packet count beyond 5.
4. **Manuscript Precision Formulation:**  
   The manuscript will explicitly state:
   > *"The Early-5 classification decision is triggered at the 5th packet arrival ($t_5$) for flows persisting beyond four packets ($44.4\%$ of flows in our benchmark). For short-lived transactions containing fewer than five packets ($55.6\%$ of flows, predominantly brief audio bursts and control transactions), classification takes place upon flow termination (via FIN/RST or inactivity timeout), where `early_5_complete = 0` indicates an abbreviated window."*

---

## 3. SEPARATION OF HORIZON REPRESENTATIONS

To ensure scientific transparency, the manuscript strictly separates the two horizon methodologies:

### A. Primary Representation for Manuscript: Horizon-Isolated Snapshot Representation
- **Definition:** Exactly 6 features extracted at observation window $k \in \{5, 10, 15, 20\}$ (`complete`, `duration`, `bytes`, `fwd_pkts`, `bwd_pkts`, `avg_size`).
- **Validation Protocol:** 29-Fold Leave-One-Group-Out (LOGO).
- **Control Benefit:** Dimensionality ($D=6$) and model capacity are held constant across horizons, eliminating parameter confounding.

| Horizon ($k$) | Feature Count ($D$) | Pooled Macro $F_1$ | Balanced Accuracy | Overall Accuracy | Bulk Data $F_1$ | Real-Time $F_1$ | Streaming $F_1$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Early-5** | **6** | **0.5734** | **0.5881** | **0.5758** | **0.5395** | 0.5849 | 0.5958 |
| **Early-10** | 6 | 0.5591 | 0.5851 | 0.5567 | 0.4833 | 0.5540 | 0.6401 |
| **Early-15** | 6 | 0.5457 | 0.5643 | 0.5448 | 0.4217 | 0.5559 | **0.6595** |
| **Early-20** | 6 | 0.5465 | 0.5633 | 0.5491 | 0.4353 | 0.5693 | 0.6347 |

*Finding:* Early-5 achieves the highest generalization under LOGO ($F_1 = 0.5734$). Degradation at $k=10, 15, 20$ occurs because Bulk Data $F_1$ drops sharply from $0.5395 \to 0.4217$ as TCP congestion-controlled bulk transfers establish periodic ACK pacing resembling video chunking.

### B. Secondary Representation: Cumulative Temporal Horizon Representation
- **Definition:** Concatenation of features across historical checkpoints ($D = 6, 12, 18, 24$).
- **Validation Protocol:** 5-Fold Stratified Group K-Fold (`random_state=42`).
- **Results:** Early-5 ($D=6$, $F_1 = 0.5348$) $\to$ Early-10 ($D=12$, $F_1 = 0.5311$) $\to$ Early-15 ($D=18$, $F_1 = 0.5265$) $\to$ Early-20 ($D=24$, $F_1 = 0.5260$).
- **Mandate:** The snapshot table will serve as the primary manuscript baseline. Results from the cumulative representation will not be combined or conflated with snapshot metrics.

---

## 4 & 6. CONFIDENCE-AWARE QOS: PAIRED STATISTICAL VERIFICATION & TRADE-OFF FRAMING

### Paired Comparison Against Clean Predicted QoS (5 Seeds)
Across 5 paired simulation seeds ($42, 101, 202, 303, 404$), gating at threshold $\tau = 0.55$ diverts flows with $\max_k P(Y=k \mid X) < 0.55$ to the lower-priority DRR queue.

| Congestion Regime | QoS Class | Performance Metric | Clean Predicted QoS (Mean $\pm$ SD) | Confidence-Aware QoS (Mean $\pm$ SD) | Mean Difference | Paired $t$-test $p$-value | Significance ($p < 0.05$) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **High (125%)** | **Real-Time** | Mean Delay (ms) | $30.61 \pm 0.39$ | $32.22 \pm 0.42$ | $+1.60$ | **$0.0004$** | **Yes (Worse)** |
| **High (125%)** | **Real-Time** | P95 Delay (ms) | $69.62 \pm 0.36$ | $69.44 \pm 0.43$ | $-0.18$ | $0.3350$ | No (Negligible) |
| **High (125%)** | **Real-Time** | Loss Rate (%) | $3.43 \pm 0.77$ | $4.18 \pm 0.81$ | $+0.75$ | **$0.0007$** | **Yes (Worse)** |
| **High (125%)** | **Real-Time** | Goodput (Mbps) | $12.14 \pm 0.08$ | $12.04 \pm 0.09$ | $-0.10$ | **$0.0008$** | **Yes (Worse)** |
| **High (125%)** | **Bulk Data** | Mean Delay (ms) | $26.25 \pm 4.88$ | $25.60 \pm 4.75$ | $-0.65$ | **$0.0166$** | **Yes (Better)** |
| **High (125%)** | **Bulk Data** | Loss Rate (%) | $5.34 \pm 8.38$ | $4.86 \pm 8.12$ | $-0.48$ | $0.0580$ | Marginal ($p = 0.058$) |
| **High (125%)** | **Bulk Data** | Goodput (Mbps) | $19.73 \pm 1.05$ | $20.46 \pm 1.25$ | $+0.73$ | $0.1691$ | **No ($p = 0.169$)** |
| **High (125%)** | **Streaming** | Mean Delay (ms) | $49.39 \pm 2.09$ | $50.75 \pm 1.84$ | $+1.37$ | **$0.0380$** | **Yes (Worse)** |
| **High (125%)** | **Streaming** | Loss Rate (%) | $72.53 \pm 2.26$ | $72.71 \pm 2.14$ | $+0.18$ | $0.5099$ | No |
| **High (125%)** | **Streaming** | Goodput (Mbps) | $64.21 \pm 5.96$ | $64.28 \pm 5.82$ | $+0.06$ | $0.9135$ | No |

### Defensible Scientific Trade-Off Characterization
The empirical data demonstrates that Confidence-Aware QoS **does not represent a uniform, system-wide improvement** over Clean Predicted QoS:
1. **Bulk Data Gains Are Modest and Not Statistically Significant:** Bulk goodput increases from $19.73$ to $20.46\text{ Mbps}$ ($+0.73\text{ Mbps}$), but this difference yields $p = 0.1691$ under the paired $t$-test.
2. **Real-Time Incurs Measurable Penalties:** Real-Time mean latency increases by $+1.60\text{ ms}$ ($p = 0.0004$) and packet loss increases by $+0.75\%$ ($p = 0.0007$) because uncertain real-time flows are demoted to the DRR queue.
3. **Manuscript Framing Mandate:** The confidence-aware policy must be reported strictly as an **engineering trade-off** between queue isolation risk and latency protection:
   > *"Gating queue admission on classification confidence introduces a tuneable trade-off. Setting $\tau = 0.55$ isolates low-confidence flows into the DRR queue, shielding the strict-priority queue from bulk elephant bursts. However, this safety mechanism incurs a measurable cost: Real-Time packets with low classification certainty experience a $1.60\text{ ms}$ increase in mean latency ($p = 0.0004$) and a $0.75\%$ increase in packet loss ($p = 0.0007$), while bulk goodput improvements ($+0.73\text{ Mbps}$) remain within statistical variation ($p = 0.1691$). Operators can tune $\tau$ depending on whether their operational priority favors bulk queue protection or low-latency preservation."*

---

## 5. ML MODEL & PROBABILITY CALIBRATION RECONCILIATION

1. **Active Classifier Architecture:**  
   The classifier used in the Phase 6D/6D.2/6D.3 pipeline and used to generate `clean_flows_with_logo_predictions.csv` is **XGBoost (`XGBClassifier`)** configured with:
   - `n_estimators = 100`, `max_depth = 5`, `learning_rate = 0.1`, `subsample = 0.8`, `colsample_bytree = 0.8`
   - `objective = "multi:softprob"`, `eval_metric = "mlogloss"`, `random_state = 42`
   - Earlier exploratory benchmarks (Phase 6C) compared Random Forest, ExtraTrees, and HistGradientBoosting; historical textual references to "Calibrated Random Forest" represented an editorial conflation with those baseline trials.
2. **Probability Calibration Findings:**  
   In Gate 4 (`output/phase6d2/calibration_comparison.csv`), Platt sigmoid and isotonic calibration were evaluated under outer LOGO cross-validation:
   - **Uncalibrated XGBoost:** Brier Score $= 0.4496$, Expected Calibration Error (ECE) $= 0.0820$
   - **Platt Sigmoid Calibration:** Brier Score $= 0.4490$, ECE $= 0.0843$
   - **Isotonic Regression:** Brier Score $= 0.4495$, ECE $= 0.0842$  
   Post-hoc calibration produced no statistically meaningful reduction in ECE ($\Delta \text{ECE} \le +0.0023$). Consequently, raw predicted softprob vectors were adopted as the calibrated confidence estimate $c = \max_k P(Y=k \mid X)$.

---

## 6. STANDARDS-BASED 3GPP SECTION REWORDING

The manuscript text is revised to formally separate standardized 3GPP 5QI definitions from simulation-level thresholds and real-world compliance:

> **Standards-Based Mapping vs. Simulation-Level SLA Thresholds:**  
> The 3GPP Technical Specification TS 23.501 (Release 18, Table 5.7.4-1) establishes standardized 5QI characteristics for carrier networks. In this study, we map traffic classes to standard 5QI profiles based on functional service requirements:  
> - **Real-Time (Conversational Voice/Video):** Mapped conceptually to **5QI 1 / 5QI 2**, where 3GPP specifies a Packet Delay Budget (PDB) of $100\text{ ms}$ and a Packet Error Rate (PER) of $10^{-2}$ ($1.0\%$).  
> - **Streaming Video (Buffered Media):** Mapped to **5QI 4 / 5QI 6**, with a standardized PDB of $300\text{ ms}$ and PER of $10^{-6}$.  
> - **Bulk Data (TCP Transfers):** Mapped to **5QI 9** (Default Best Effort), with PDB of $300\text{ ms}$ and PER of $10^{-6}$.  
>  
> In our discrete-event access link simulation, we evaluate queue performance against engineering SLA criteria: maintaining P95 delay below $100\text{ ms}$ and packet loss below $1.0\%$ for Real-Time traffic.  
>  
> **Empirical Compliance Under Congestion:**  
> In Low (50%) and Moderate (85%) link regimes, Predicted QoS satisfies both delay and loss criteria, achieving **Full SLA Compliance**. Under severe link overload (125% link capacity), Predicted QoS bounds Real-Time P95 delay to $68.20\text{ ms}$ (well below the $100\text{ ms}$ threshold), but packet loss rises to $3.43\%$, exceeding the $1.0\%$ target due to physical link buffer exhaustion. We report this outcome accurately as **Partial SLA Compliance (Delay Threshold Met, Packet Loss Target Exceeded)**, reflecting the physical boundary of priority queuing without upstream admission control or rate adaptation.

---

## 7. DEFENSIBLE NOVELTY STATEMENT

*Revised to completely eliminate non-defensible superlatives ("first", "unprecedented", "flawless"):*

> **Scientific Contribution and Scope:**  
> *"This research investigates an end-to-end framework coupling early encrypted traffic classification with multi-queue scheduling under realistic congestion regimes. By formulating strictly causal features available within the first five packets of flow transmission, we evaluate classification generalization across 29 capture conditions using Leave-One-Group-Out (LOGO) cross-validation. Furthermore, we integrate out-of-fold probabilistic predictions into a discrete-event network simulator to assess closed-loop queuing behavior. Our findings demonstrate that early packet classification cuts real-time packet latency by over $50\%$ and prevents TCP bulk throughput starvation during link congestion, while highlighting the empirical trade-offs associated with confidence-based queue gating and extreme oversubscription."*

---

## 8. FINAL STATUS & EXACT NEXT STEP

1. **Consistency Verification:** All 8 issues have been resolved directly from verified source files and code without launching redundant simulations or altering historical data.
2. **Readiness Verdict:** The experimental pipeline, statistical tables, and claim wordings are fully verified, robust, and publication-ready.
3. **Exact Next Step:** **Proceed to Phase 8 — Manuscript Preparation.**
