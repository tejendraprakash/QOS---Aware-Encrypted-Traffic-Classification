# PHASE 7.5 — INDEPENDENT SCIENTIFIC AUDIT, STATISTICAL REVALIDATION & CLAIMS VERIFICATION

**Project:** QoS-Aware Encrypted Network Traffic Classification for Next-Generation Networks  
**Milestone:** Phase 7.5 Scientific Audit & Integrity Verification  
**Status:** COMPLETE & APPROVED FOR PHASE 8 WITH MANDATORY CONDITIONS  
**Date:** November 2026  
**Lead Auditor:** Principal Computer Networks Research Scientist & Independent Scientific Review Board  

---

## 1. Executive Summary & Audit Verdict

An exhaustive 8-perspective independent scientific audit of the Phase 7 closed-loop network simulation was conducted across all 45 simulation runs ($3\text{ Policies} \times 3\text{ Congestion Regimes} \times 5\text{ Seeds}$).

### High-Level Verdict:
1. **Raw Simulation & Packet Conservation:** **PASSED WITH ZERO DEFECTS.** 45/45 runs accounted for, with exact flow and packet conservation ($\Delta = |N_{\text{tx}} - (N_{\text{rx}} + N_{\text{dropped}})| \equiv 0$) across all 135 evaluated class-level records.
2. **Metric Recalculation:** **PASSED.** Recomputed sample means, sample standard deviations, P95/P99 latency quantiles, and Goodput values match published summary tables within $10^{-4}$ precision.
3. **Statistical Significance Claims:** **CORRECTED.** Claims of statistical significance at $\alpha = 0.05$ via two-sided Wilcoxon signed-rank test are mathematically impossible with $N=5$ (where $p_{\min} = 2 \times (1/2)^5 = 0.0625$). The findings must be strictly reported as **100% consistent directional dominance (5/5 paired seeds strictly favor Predicted QoS over FIFO, exact $p = 0.0625$)**.
4. **Classifier Profile Qualification:** **MANDATORY CORRECTION.** The simulation evaluated an **optimistic early-classification sensitivity benchmark ($\approx 85\%$ accuracy)** rather than direct live inference from the empirical Phase 6B Early-5 XGBoost model (which achieved pooled Macro F1 of $0.5323$ and Real-Time recall of $58.85\%$). Phase 8 manuscript writing must transparently frame Phase 7 as an upper-bound sensitivity study under an 85% accurate classifier profile.
5. **Trade-Off Transparency:** The paper must prominently report the **transient quarantine jitter increase ($4.60\text{ ms}$ vs. $0.04\text{ ms}$)** and **streaming loss increase ($68.8\%$ vs. $65.7\%$)** as necessary systems costs of early quarantine and DRR bandwidth reservation.

---

## 2. Itemized Audit Findings & Required Corrections

### Audit Finding 1: The $N=5$ Wilcoxon Mathematical Boundary
- **Issue:** Several Phase 7 notes described Wilcoxon $p = 0.0625$ as "statistically significant at $\alpha = 0.05$".
- **Mathematical Law:** For $N=5$ non-zero paired differences, the two-sided Wilcoxon signed-rank test has:
  $$p_{\min} = 2 \times \left(\frac{1}{2}\right)^5 = \frac{2}{32} = 0.0625$$
- **Mandatory Correction for Phase 8:** Do not claim $p < 0.05$ significance. Formulate as:
  > *"Predicted QoS achieved 100% directional dominance over Baseline FIFO across all five random seed replications (5/5 paired seeds, exact Wilcoxon $p = 0.0625$, the theoretical minimum attainable for $N=5$)."*

### Audit Finding 2: Classifier Performance Discrepancy
- **Empirical Reality (Phase 6B Early-5 XGBoost on Track B):**
  - Fold Mean Macro F1: $0.4213 \pm 0.0610$
  - Pooled Out-of-Fold Macro F1: **$0.5323$**
  - Balanced Accuracy: **$0.6424$**
  - Real-Time Recall: $58.85\%$ (Precision: $88.04\%$)
  - Bulk Data Recall: $65.58\%$
  - Streaming Recall: $68.29\%$
- **Simulation Parameterization:** The simulation used a diagonal matrix rate of $89.34\%$ Real-Time, $81.23\%$ Bulk, and $84.74\%$ Streaming (average accuracy $\approx 85.1\%$).
- **Mandatory Correction for Phase 8:** Clearly state:
  > *"To evaluate the potential of early QoS scheduling under near-ideal operational conditions, the network simulation employed an optimistic early-classifier profile ($\approx 85\%$ accuracy), demonstrating the performance ceiling achievable as early feature representation techniques mature towards 85% accuracy."*

### Audit Finding 3: Low Simulator Variance & Large Effect Sizes
- **Observation:** Hedges' $g = +8.28$ (Real-Time P95 latency) and $g = +8.07$ (Real-Time packet loss) are unusually large compared to empirical human or wireless studies.
- **Root Cause:** In discrete-event simulation of DropTail FIFO queues, deterministic packet arrival rates produce near-zero latency variance across seeds ($\sigma_{\text{FIFO}} = 0.0039\text{ ms}$). Dividing a large mean difference ($18.3\text{ ms}$) by a tiny pooled standard deviation ($1.997\text{ ms}$) naturally yields $g > 8$.
- **Mandatory Correction for Phase 8:** Report Hedges' $g$ with an explicit methodological note that discrete-event simulation exhibits lower variance than physical wireless testbeds, which inflates standardized mean difference metrics.

### Audit Finding 4: Trade-Off Disclosures (Jitter and Secondary Queue Loss)
- **Quarantine Jitter:** Real-Time jitter rises from $0.042\text{ ms}$ (FIFO) to $4.599\text{ ms}$ (Predicted QoS) because packets 1–4 experience exploratory queueing delay in Queue 1 ($\approx 50\text{ ms}$) before packet 5 promotes the flow to Queue 0 ($\approx 20\text{ ms}$). This $30\text{ ms}$ delay shift injects packet-to-packet delay variation.
- **Streaming Loss:** Deficit Round Robin limits video traffic to its allocated $60\%$ link share, preventing video bursts from starving Bulk Data. As a consequence, Streaming packet loss rises from $65.7\%$ (FIFO) to $68.8\%$ (Predicted QoS), while Bulk Data throughput surges from $1.66\text{ Mbps}$ to $8.56\text{ Mbps}$ ($+416.9\%$).
- **Mandatory Correction for Phase 8:** Present these trade-offs transparently in the results section as proof of rigorous, unbiased evaluation.

---

## 3. Scientific Claims Verification Matrix

| Claim ID | Claim in Phase 7 Documentation | Audit Decision | Phase 8 Action / Approved Phrasing |
| :---: | :--- | :---: | :--- |
| **CLM-01** | Real-Time mean latency reduced by $65.3\%$ ($22.87\text{ ms}$ vs $65.82\text{ ms}$). | **VERIFIED** | Report with qualification: achieved under simulated 85% accuracy classifier profile. |
| **CLM-02** | Real-Time packet loss slashed by $87.0\%$ ($1.045\%$ vs $8.031\%$). | **VERIFIED** | Report with qualification: achieved under simulated 85% accuracy classifier profile. |
| **CLM-03** | Bulk Data goodput expanded by $+416.9\%$ ($8.56\text{ Mbps}$ vs $1.66\text{ Mbps}$). | **VERIFIED** | Formally proved as a direct consequence of DRR deficit counter starvation protection. |
| **CLM-04** | Early-5 XGBoost Macro F1 is $0.8524$. | **REVERTED** | State exact Phase 6B empirical Pooled Macro F1 = **$0.5323$** ($0.4213$ fold mean). |
| **CLM-05** | Statistical significance at $p < 0.05$ via Wilcoxon test. | **REVERTED** | Reframe as **100% directional dominance (5/5 paired seeds, exact Wilcoxon $p = 0.0625$)**. |
| **CLM-06** | Hedges' $g = +8.28$ indicates massive practical effect. | **QUALIFIED** | Report $g = +8.28$ with variance-reduction caveat. |
| **CLM-07** | Real-Time 5QI 1 SLA compliance is $40\%$ under High Congestion. | **VERIFIED** | Exactly $2/5$ seeds passed both $\le 100\text{ ms}$ delay and $\le 1.0\%$ packet loss. |
| **CLM-08** | Zero packet leakage ($\Delta = 0$). | **VERIFIED** | Verified across all 135 class-level telemetry records. |

---

## 4. Phase 8 Manuscript Preparation Guidelines

With this audit completed and committed, the research pipeline transitions to **Phase 8: Academic Manuscript Preparation & Dissertation Synthesis**:

1. **Target Venues:** IEEE Transactions on Network and Service Management (TNSM) / IEEE INFOCOM / ACM SIGCOMM CCR.
2. **Key Sections:**
   - **Section I: Introduction & Motivation:** Encrypted traffic explosion (TLS 1.3, ECH, QUIC), demise of DPI, requirement for in-network early QoS enforcement in 5G/6G.
   - **Section II: Related Work:** DiffServ, 5G 5QI, first-few-packet classification, P4 programmable data planes.
   - **Section III: Dataset Auditing & Flow Feature Extraction:** Multi-agent dataset audit, flow formation, leakage elimination, Track B specification (10,770 flows).
   - **Section IV: Early-Packet Machine Learning Benchmarks:** Full-flow vs. early packet horizons (N=5, 10, 15, 20), Early-5 plateau, XGBoost efficiency ($3.4\ \mu\text{s}$, Macro F1 = 0.5323).
   - **Section V: QoS Architecture & Discrete-Event Simulation:** Topology, Byte-bounded buffer conservation ($625\text{ KB}$), Strict Priority with Ingress Token Bucket Policing, Deficit Round Robin.
   - **Section VI: Performance Evaluation & Trade-off Analysis:** 45-run evaluation, latency suppression ($-65.3\%$), voice loss reduction ($-87.0\%$), bulk throughput resuscitation ($+416.9\%$), quarantine jitter analysis, and streaming trade-offs.
   - **Section VII: Discussion, Hardware Realizability & Conclusion:** P4/eBPF implementation feasibility, 11-byte flow state memory scalability, limitations, and future work.
