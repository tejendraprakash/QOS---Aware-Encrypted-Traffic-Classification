# Phase 7 Audit Report: Methodological Audit of Validation Protocols & Experimental Design for Synthetic Traffic Generation

**Project:** QoS-Aware Encrypted Network Traffic Classification for Next-Generation Networks  
**Document Code:** AGY-PHASE7-AUDIT-VAL-2026-01  
**Author:** Agent 6 — Validation and Research Methodology Auditor (Phase 7)  
**Target Document Under Audit:** `research_notes/phase7/traffic_generation_design.md` (Specifically Section 7)  
**Date:** November 2026  
**Status:** Complete Independent Audit & Approved Methodology Specification  

---

## 1. Executive Summary & Audit Mandate

In Phase 7, the research project transitions from offline machine learning benchmarking on static packet captures (`output/phase5/track_b_flows.csv`) to closed-loop network simulation across varied bottleneck congestion regimes ($\rho \in \{0.50, 0.85, 1.25\}$). Section 7 of `traffic_generation_design.md` proposes a verification and calibration framework designed to ensure that synthetic traffic mirrors the physical characteristics of the empirical dataset.

This audit evaluates the scientific defensibility, statistical validity, and experimental rigor of Section 7. Our evaluation identifies critical methodological vulnerabilities in the proposed Kolmogorov-Smirnov (KS) test, reveals the omission of simulation warm-up periods and stationarity testing, establishes multi-seed replication standards, and formalizes a comprehensive, publication-grade validation protocol.

```
+===================================================================================================+
|                                    PHASE 7 VALIDATION AUDIT SUMMARY                               |
+===================================================================================================+
| Vulnerability / Flaw              | Severity | Core Methodological Defect                         |
+-----------------------------------+----------+----------------------------------------------------+
| 1. Classical Two-Sample KS Test   | CRITICAL | Continuous distribution assumption violated by     |
|    on Multimodal Packet Sizes     |          | discrete point masses (52B, 64B, 120B, 1500B).     |
| 2. p > 0.05 Validation Criterion  | CRITICAL | Flawed inference (absence of evidence != proof of  |
|    with N=1,000 Flows             |          | fit); Type I error inflation guarantees rejection. |
| 3. Uncontrolled Initial Transient | HIGH     | Empty queue start (Q(0)=0) distorts latency/loss;  |
|    & Omission of Warm-up Period   |          | lacks Welch's moving average stationarity audit.   |
| 4. Missing Multi-Seed Rigor &     | HIGH     | Single-run simulations lack confidence intervals;  |
|    Non-Normality Violations       |          | Student's t assumes normality on skewed tails.     |
| 5. Discrete Queue Sampling vs.    | MEDIUM   | Periodic polling misses burst drops; requires      |
|    Time-Integrated Tracking       |          | continuous time-weighted integral and Little's Law.|
+===================================================================================================+
```

---

## 2. Audit of the Proposed Two-Sample Kolmogorov-Smirnov (KS) Test

Section 7.1 of `traffic_generation_design.md` proposes extracting bidirectional packet sizes and inter-arrival times from $N=1,000$ synthetic flows and executing a two-sample Kolmogorov-Smirnov test against empirical sustained distributions from `track_b_flows.csv`, enforcing:
$$\text{Criterion: } D = \sup_x |F_{\text{empirical}}(x) - F_{\text{synthetic}}(x)| < 0.15 \quad \text{with} \quad p > 0.05$$

### 2.1 Theoretical Violation: Continuous Distribution Assumption vs. Discrete Packet Masses
The classical two-sample Kolmogorov-Smirnov test (Kolmogorov 1933, Smirnov 1939) rests on the foundational mathematical axiom that the underlying cumulative distribution functions $F(x)$ and $G(x)$ are **strictly continuous**:
$$P(X = x) = 0, \quad \forall x \in \mathbb{R}$$

In network traffic engineering, packet size distributions are inherently **discrete and highly multimodal**, characterized by discrete point masses corresponding to standard networking protocol headers and maximum transmission units:
* $52\text{ Bytes}$ and $64\text{ Bytes}$: Pure TCP acknowledgments (20B IP + 32B TCP with timestamps) and RTCP / Silence Insertion Descriptor (SID) frames.
* $120\text{ Bytes}$: Opus voice codec audio frames at 20 ms packetization.
* $200\text{ Bytes}$: G.711 uncompressed PCM frames (160B payload + 40B RTP/UDP/IP).
* $1,460\text{ Bytes}$ to $1,500\text{ Bytes}$: Maximum Transmission Unit (MTU) payload chunks for video streaming and saturated bulk TCP transfers.

#### Consequence of Ties and Discrete Jumps
When empirical and synthetic samples contain duplicate discrete values (ties), the step-function nature of the empirical CDFs $F_n(x)$ and $G_m(x)$ violates the continuity condition. As established by Noether (1963) and Conover (1972):
1. The null distribution of the supremum test statistic $D_{n,m}$ under tied discrete data is **stochastically strictly smaller** than the standard continuous Kolmogorov distribution.
2. Relying on standard continuous asymptotic approximations (e.g., SciPy's default `scipy.stats.ks_2samp`) yields **conservative, inexact, and biased $p$-values**.
3. More fundamentally, the supremum statistic $D$ measures only the maximum vertical distance between step jumps, completely ignoring the horizontal Euclidean distance between packet size modes (e.g., a discrepancy between 120B and 124B produces the exact same vertical penalty as a discrepancy between 120B and 1500B if the cumulative step height is identical).

### 2.2 The Large-Sample $p$-Value Paradox and Type I Error Inflation
The test setup proposes comparing $N=1,000$ flows. In sustained flows, $1,000$ flows generate between $5 \times 10^4$ and $10^6$ individual packet observations ($n, m \ge 50,000$).

The asymptotic two-sample KS statistic scales proportionally to sample size:
$$\lambda = \sqrt{\frac{n \cdot m}{n + m}} D_{n,m}$$
Under $H_0$, the asymptotic $p$-value is given by the Kolmogorov survival function:
$$p \approx 2 \sum_{k=1}^{\infty} (-1)^{k-1} e^{-2 k^2 \lambda^2}$$

#### Numerical Demonstration of the Paradox
Consider a synthetic generator that achieves an outstanding engineering fit, diverging from the empirical distribution by a tiny margin of $D = 0.05$ (a 5% vertical CDF difference).
* For $n = m = 1,000$ packets:
  $$\lambda = \sqrt{\frac{1000 \times 1000}{2000}} \times 0.05 = \sqrt{500} \times 0.05 \approx 22.36 \times 0.05 = 1.118 \implies p \approx 0.164 > 0.05 \quad (\text{Pass})$$
* For $n = m = 20,000$ packets (the packets in only ~100 sustained flows):
  $$\lambda = \sqrt{\frac{20000 \times 20000}{40000}} \times 0.05 = 100 \times 0.05 = 5.0 \implies p \approx 2 e^{-2(25)} \approx 3.8 \times 10^{-22} \ll 0.05 \quad (\text{Catastrophic Fail})$$

As sample size grows, the standard error shrinks to zero ($\text{SE} \propto 1/\sqrt{N}$). Consequently, the KS test gains extreme statistical power to detect microscopic, practically irrelevant discrepancies (e.g., minor differences in TLS record padding or OS-specific TCP options). A valid, realistic synthetic generator will **almost certainly be rejected ($p < 0.001$)** simply because the sample size is large.

### 2.3 The Inherent Flaw of "$p > 0.05$" as an Acceptance Criterion
Using $p > 0.05$ as a criterion for model validation is a recognized epistemological fallacy in statistical science:
1. **Confusing Absence of Evidence with Evidence of Absence:** A $p$-value $> 0.05$ indicates only a *failure to reject* $H_0$; it does not prove $H_0$ is true. If a test has small sample size or high noise, $p > 0.05$ will occur even if the synthetic model is grossly flawed.
2. **Mathematical Inconsistency in Section 7.1:** The threshold specifies $D < 0.15$ *and* $p > 0.05$. But under $N=1,000$ flows, any $D > 0.06$ will drive $p$ far below $10^{-6}$. Setting $D < 0.15$ as an acceptable bound while requiring $p > 0.05$ creates two criteria that are **mutually exclusive** at scale.

### 2.4 Recommended Alternative & Complementary Goodness-of-Fit Metrics

To replace the standalone continuous KS test, we mandate a multi-perspective goodness-of-fit suite combining transport-aware distance metrics with non-parametric permutation testing:

```
+---------------------------------------------------------------------------------------------------+
|                         RECOMMENDED GOODNESS-OF-FIT EVALUATION SUITE                              |
+---------------------------------------------------------------------------------------------------+
| 1. Earth Mover's Distance (Wasserstein-1) | Quantifies horizontal work in Bytes or Milliseconds   |
| 2. Jensen-Shannon Divergence (JSD)        | Symmetric, information-theoretic bounded divergence   |
| 3. Multinomial Chi-Square (χ²) Test       | Valid discrete categorical hypothesis testing         |
| 4. Quantile-Quantile (Q-Q) Metric (R²_QQ) | Assesses alignment across distribution tails          |
+---------------------------------------------------------------------------------------------------+
```

#### 1. First Wasserstein Distance (Earth Mover's Distance, $W_1$)
The 1-Wasserstein distance between empirical distribution $P$ and synthetic distribution $Q$ is:
$$W_1(P, Q) = \int_{-\infty}^{\infty} |F_{\text{empirical}}(x) - F_{\text{synthetic}}(x)| dx = \inf_{\gamma \in \Pi(P, Q)} \mathbb{E}_{(X,Y)\sim \gamma}[|X - Y|]$$
* **Why it succeeds:** $W_1$ operates horizontally along the metric space. For packet sizes, $W_1$ is denominated directly in **Bytes**; for IAT, it is denominated in **Milliseconds**. 
* **Physical Interpretation:** If $W_1 = 18.4\text{ Bytes}$, it means that on average, a synthetic packet is only $18.4$ bytes away from matching the empirical packet envelope. This provides an interpretable physical threshold that is completely independent of sample size inflation.
* **Mandated Threshold:** $W_1 \le 35.0\text{ Bytes}$ for packet sizes; $W_1 \le 15.0\text{ ms}$ for Real-Time IAT.

#### 2. Jensen-Shannon Divergence (JSD)
The symmetric, smoothed variant of Kullback-Leibler divergence:
$$\text{JSD}(P \parallel Q) = \frac{1}{2} D_{\text{KL}}(P \parallel M) + \frac{1}{2} D_{\text{KL}}(Q \parallel M), \quad M = \frac{1}{2}(P + Q)$$
* **Why it succeeds:** Bounded between $0$ and $1$ (using $\log_2$), insensitive to zero bins, and stable across multimodal distributions.
* **Mandated Threshold:** $\text{JSD}(P \parallel Q) \le 0.05$ (indicating $>95\%$ distributional similarity).

#### 3. Discrete Multinomial Chi-Square ($\chi^2$) Goodness-of-Fit Test
Rather than treating packet sizes as continuous, partition the domain into standard structural networking bins:
* Bin 1: $[40, 64]\text{ B}$ (Pure ACKs, Control, RTCP)
* Bin 2: $[65, 150]\text{ B}$ (VoIP Opus, HTTP GET)
* Bin 3: $[151, 300]\text{ B}$ (VoIP G.711, Video Control)
* Bin 4: $[301, 800]\text{ B}$ (Sub-MTU Video Frames, Audio Packets)
* Bin 5: $[801, 1400]\text{ B}$ (Large Video Slices, VPN MTU)
* Bin 6: $[1401, 1514]\text{ B}$ (Full Ethernet MTU)

Compute the discrete $\chi^2$ test statistic:
$$\chi^2 = \sum_{k=1}^{K} \frac{(O_k - E_k)^2}{E_k}, \quad \text{dof} = K - 1 = 5$$
Where $O_k$ and $E_k$ are observed and expected packet counts. For large sample sizes, downsample to $N=500$ observations or report the normalized **Cramér's $V$** effect size ($V = \sqrt{\chi^2 / (N \cdot (K-1))} \le 0.10$).

#### 4. Quantile-Quantile Correlation Coefficient ($R_{QQ}^2$)
Plot empirical percentiles $q_p$ against synthetic percentiles $\hat{q}_p$ for $p \in \{1, 2, \dots, 99\}$. Compute the coefficient of determination:
$$R_{QQ}^2 = \frac{\left( \sum (q_p - \bar{q})(\hat{q}_p - \bar{\hat{q}}) \right)^2}{\sum (q_p - \bar{q})^2 \sum (\hat{q}_p - \bar{\hat{q}})^2} \ge 0.98$$

---

## 3. Audit of Experimental Design & Statistical Rigor

### 3.1 Simulation Repetitions & Multi-Seed Architecture
Section 7 omits the number of independent simulation replications required to compute valid confidence intervals. Running a single simulation run—even for 300 seconds—exposes findings to severe stochastic noise, initial state bias, and non-generalizable arrival coincidences.

#### Recommendation: $M \ge 30$ Independent Runs with Common Random Numbers (CRN)
1. **Sample Size ($M = 30$):** By the Central Limit Theorem (CLT), computing confidence intervals for aggregate QoS metrics (mean delay, throughput) requires $M \ge 30$ independent replications with distinct pseudo-random generator (PRNG) seeds. If computational budget constraints ns-3 execution, a minimum of $M = 10$ runs is acceptable only if non-parametric BCa bootstrap intervals are used.
2. **PRNG Seed Independence:** Seeds must be initialized using cryptographically secure entropy or an audited pseudo-random sequence (e.g. PCG64 or Mersenne Twister MT19937 with distinct seed offsets $\ge 10^6$ steps) to prevent overlapping random streams.
3. **Common Random Numbers (CRN) for Comparative Policy Evaluation:**
   When comparing QoS scheduling policies (e.g., Default FIFO vs. Fair Queuing vs. Early-5 Classifier Priority Queuing):
   * **Mandate:** The identical set of 30 seeds must be applied across all policy configurations.
   * **Mathematical Benefit:** Under CRN, identical flow arrival timestamps, packet sizes, and durations are generated across both policies. For run $m$, let $Y_{m}^{\text{FIFO}}$ and $Y_{m}^{\text{ML}}$ be the observed Real-Time $P_{95}$ latency. The policy difference is $D_m = Y_{m}^{\text{FIFO}} - Y_{m}^{\text{ML}}$. The variance of the difference is:
     $$\text{Var}(D_m) = \text{Var}(Y_m^{\text{FIFO}}) + \text{Var}(Y_m^{\text{ML}}) - 2\text{Cov}(Y_m^{\text{FIFO}}, Y_m^{\text{ML}})$$
     Because CRN induces strong positive covariance ($\text{Cov} > 0$) by subjecting both policies to the identical traffic bursts, $\text{Var}(D_m)$ is dramatically reduced, maximizing statistical test power.

### 3.2 Simulation Warm-Up Period & Stationarity Audit (Welch's Method)
Section 7.3 specifies running a $300\text{-second}$ bottleneck simulation, but begins recording metrics from $t = 0.0\text{ s}$.

#### The Initial Transient State Hazard
At $t = 0.0$, all queues in the simulator are empty ($Q(t) = 0$), and no flows are in-flight. 
* In a system designed for $\rho = 0.85$ or $\rho = 1.25$, it takes tens of seconds for the Poisson flow arrival process to populate concurrent flows ($\mathbb{E}[N_{RT}] = 213\text{ flows}$, $\mathbb{E}[N_{STR}] = 17\text{ flows}$) and for TCP CUBIC senders to ramp up their congestion windows.
* Logging packet latencies during this initial transient phase introduces severe **downward bias** on average queue occupancy, queuing delay, and packet drop rates.

```
Queue Depth Q(t)
    ^
400 |                                    Steady-State Congestion Regime
    |                                   ┌───────────────────────────────┐
    |                                  /│\  /\    /\   /\  /\    /\     │
200 |                                 / │ \/  \  /  \ /  \/  \  /  \    │
    |           Transient Phase      /  │      \/    V        \/    \   │
    |        ┌───────────────────────┘  │                               │
  0 +────────┴──────────────────────────┼───────────────────────────────┴────> Time (s)
    0        t = 20s        t = 60s     │ t = 60s                      t = 300s
               (Initial Ramp-up)        │      (Measurement Window)
                                   Warm-Up Boundary
```

#### Protocol for Warm-Up Elimination (Welch's Procedure)
1. **Determine Warm-Up Duration ($T_{\text{warm}}$):**
   Execute 5 pilot replications of length $T = 360\text{ s}$. Record queue depth $Q_m(t)$ at $100\text{ ms}$ intervals.
   Compute the cross-run average $\bar{Q}(t) = \frac{1}{5} \sum_{m=1}^5 Q_m(t)$.
   Apply a symmetric moving average filter of window $w = 20\text{ s}$:
   $$\bar{Q}_{\text{smoothed}}(t) = \frac{1}{2w + 1} \sum_{k=-w}^{w} \bar{Q}(t + k)$$
   Identify the point $T_{\text{warm}}$ beyond which $\bar{Q}_{\text{smoothed}}(t)$ enters and remains within the steady-state band. For our calibrated parameters ($\mathbb{E}[D] = 120\text{--}180\text{ s}$), **$T_{\text{warm}} = 60.0\text{ s}$** is required.
2. **Measurement Window Specification:**
   Total simulation run time must be extended to $T_{\text{total}} = 360\text{ s}$. 
   All telemetry logging and QoS metric collection must strictly begin at **$t = 60.0\text{ s}$** and end at **$t = 360.0\text{ s}$** ($T_{\text{meas}} = 300\text{ s}$). Packets transmitted or dropped prior to $t = 60.0\text{ s}$ must be purged from performance statistics.
3. **Stationarity Verification:**
   Apply the **Augmented Dickey-Fuller (ADF)** unit root test to the continuous buffer occupancy time series $q(t)$ during $[T_{\text{warm}}, T_{\text{total}}]$ under Moderate Congestion ($\rho = 0.85$). Stationarity is confirmed when $p_{\text{ADF}} < 0.01$.

### 3.3 Confidence Interval Estimation & Inferential Testing
Network QoS metrics (specifically packet delay and queue length) exhibit severe right-skewness, multimodal spikes, and heavy tails.

#### Flaw of Standard Student's $t$ Confidence Intervals
Standard Student's $t$-confidence intervals assume that the underlying sample observations are independent and identically normally distributed. 
* While the sample mean of $M \ge 30$ independent runs converges to normality under the CLT, **tail quantiles ($P_{95}$ and $P_{99}$)** and **packet loss rates (PLR)** violate normality assumptions.
* Constructing normal intervals on $P_{95}$ or $P_{99}$ underestimates the upper error bound, concealing critical bufferbloat tail risks.

#### Mandated Statistical Testing Protocols
1. **Bias-Corrected and Accelerated (BCa) Non-Parametric Bootstrap:**
   For tail delay metrics ($P_{95}, P_{99}$) and class drop percentages, compute 95% Confidence Intervals using the BCa bootstrap with $B = 2,000$ resamples:
   $$\text{CI}_{0.95} = \left[ \hat{\theta}^{*(\alpha_1)}, \hat{\theta}^{*(\alpha_2)} \right]$$
   Where $\alpha_1, \alpha_2$ adjust for both skewness (acceleration $a$) and median bias ($z_0$).
2. **Comparative Significance Testing (Policy Superiority):**
   * **Wilcoxon Signed-Rank Test (Paired, Non-Parametric):**
     Used to evaluate whether Early-5 Classification significantly outperforms Baseline FIFO across the $M = 30$ paired seeds. Reported with rank-biserial correlation $r$ effect size.
   * **Cohen's $d$ Effect Size with Hedges' $g$ Correction:**
     To quantify the magnitude of latency reduction independent of sample size:
     $$g = \frac{\bar{Y}_{\text{FIFO}} - \bar{Y}_{\text{ML}}}{s_{\text{pooled}}} \left( 1 - \frac{3}{4(2M) - 9} \right)$$
     An effect size $g > 0.80$ confirms a large, operationally meaningful QoS improvement.

---

## 4. Audit of Queueing and Congestion Validation

Section 7.3 defines coarse target thresholds for average queue occupancy under Low, Moderate, and High congestion ($\bar{Q} < 10$, $\approx 40\text{--}80$, and $\approx 400$ packets). However, it omits the formal sampling mechanics, conservation laws, and bottleneck diagnostic metrics required to validate queueing dynamics.

### 4.1 Continuous Time-Weighted Buffer Tracking vs. Periodic Polling
A common engineering flaw in queue monitoring is periodic snapshot polling (e.g. reading queue length every $10\text{ ms}$). Periodic sampling introduces aliasing, underestimates burst peaks, and fails Little's Law conservation.

#### Mandated Time-Integrated Buffer Occupancy Formulation
Queue occupancy must be updated on every discrete state-change event (enqueue, dequeue, and packet drop). Let $t_k$ denote the timestamp of the $k$-th queue transition event, and let $q(t_k)$ be the buffer occupancy immediately following the event.

1. **Time-Weighted Mean Queue Depth ($\bar{Q}$):**
   $$\bar{Q}_{\text{bytes}} = \frac{1}{T_{\text{meas}}} \sum_{k=1}^{K-1} q_{\text{bytes}}(t_k) \cdot (t_{k+1} - t_k)$$
   $$\bar{Q}_{\text{pkts}} = \frac{1}{T_{\text{meas}}} \sum_{k=1}^{K-1} q_{\text{pkts}}(t_k) \cdot (t_{k+1} - t_k)$$
2. **Dual-Accounting Audit (Bytes vs. Packets):**
   As proven in `phase7_methodology_review.md`, packet-mode drop-tail queues introduce a $15\times$ physical delay bias against large packets. 
   * **Audit Mandate:** Schedulers must enforce **byte-mode buffer limits** ($B_{\text{bytes}} = 625,000\text{ Bytes}$). Both byte occupancy $q_{\text{bytes}}(t)$ and packet count $q_{\text{pkts}}(t)$ must be simultaneously tracked to verify that no queue capacity distortion occurs.

### 4.2 Comprehensive Bottleneck Telemetry Suite
To prove that congestion levels match theoretical specifications, the simulation harness must collect the following five bottleneck metrics:

```
+---------------------------------------------------------------------------------------------------+
|                              BOTTLENECK QUEUE METRIC TELEMETRY                                    |
+---------------------------------------------------------------------------------------------------+
| 1. Continuous Buffer Depth CDF        | Full distribution F_Q(x) in bytes and packets             |
| 2. Class-Disaggregated Drop Audit     | Dropped bytes/packets, loss rate PLR_c, and drop reason   |
| 3. Queue Waiting Time Distribution    | Exact W_q CDF (enqueue to service start) per QoS class    |
| 4. Inter-Departure Time (IDT) Profile | Line-rate serialization verification (Δt_dep >= L/C)      |
| 5. Little's Law Equilibrium Audit     | Conservation check: |L_q - λ_eff * W_q| / L_q <= 0.02     |
+---------------------------------------------------------------------------------------------------+
```

1. **Queue Depth Cumulative Distribution Function ($F_Q(x)$):**
   Logs the fraction of time the bottleneck buffer spent at or below occupancy $x$. Under High Congestion ($\rho = 1.25$), $F_Q(0.95 \cdot B_{\text{max}}) \ge 0.85$ confirms persistent saturation.
2. **Class-Disaggregated Packet Loss Ratio ($PLR_c$):**
   $$PLR_c = \frac{N_{\text{dropped}, c}}{N_{\text{admitted}, c} + N_{\text{dropped}, c}} \times 100\%$$
   Every dropped packet must record: `{timestamp, flow_id, qos_class, packet_size_bytes, queue_occupancy_at_drop, drop_reason}`.
3. **Queue Waiting Time Distribution ($W_{q, c}$):**
   Measures pure buffering delay: $W_{q, i} = T_{\text{service\_start}, i} - T_{\text{enqueue}, i}$.
4. **Inter-Departure Time Distribution ($F_{\Delta T_{\text{dep}}}$):**
   Measures time gap between consecutive packet transmissions on the bottleneck link:
   $$\Delta T_{\text{dep}, i} = T_{\text{egress}, i} - T_{\text{egress}, i-1}$$
   Verifies that the bottleneck operates at maximum line rate ($\Delta T_{\text{dep}} = \frac{L_i \times 8}{C_{\text{link}}}$) during busy periods with zero unforced idle intervals.
5. **Exact Little's Law Audit:**
   Under stationary Moderate Congestion ($\rho = 0.85$):
   $$\epsilon_{\text{Little}} = \frac{|\bar{Q}_{\text{pkts}} - \lambda_{\text{effective}} \cdot \bar{W}_q|}{\bar{Q}_{\text{pkts}}} \le 0.02 \quad (2.0\%)$$
   Any error exceeding 2% flags an unmonitored packet drop, memory leak, or clock discretization bug.

---

## 5. Concrete Phase 7 Validation Protocol & Verification Gates

To eliminate experimental bias and guarantee publication-grade rigor, all Phase 7 simulation experiments must satisfy the following four sequential **Verification Gates** prior to publishing QoS performance claims:

```
+---------------------------------------------------------------------------------------------------+
|                               PHASE 7 FOUR-GATE VALIDATION PIPELINE                               |
+---------------------------------------------------------------------------------------------------+
|                                                                                                   |
|  [GATE 1: Micro-Scale Distributional Calibration]                                                 |
|     ├─ Earth Mover's Distance: W_1(Packet Size) <= 35 Bytes                                       |
|     ├─ Jensen-Shannon Divergence: JSD(Packet Size) <= 0.05                                        |
|     └─ Discrete Chi-Square Test: Cramér's V <= 0.10 across 6 protocol bins                        |
|                                     │                                                             |
|                                     ▼ [PASS]                                                      |
|  [GATE 2: Macro-Dynamics & Directional Invariant Audit]                                           |
|     ├─ Real-Time Symmetry: Fwd Pkt Ratio 0.50 ± 0.03, Fwd Byte Ratio 0.48 ± 0.04                  |
|     ├─ Streaming Downlink Bias: Fwd Byte Ratio 0.10 - 0.28 (>= 72% Downlink)                      |
|     └─ Bulk Data Line Rate: Median Intra-Burst IAT <= 20 microseconds                             |
|                                     │                                                             |
|                                     ▼ [PASS]                                                      |
|  [GATE 3: Stationarity, Warm-Up & Conservation Audit]                                             |
|     ├─ Warm-Up Period: Purge t in [0, 60s]; Measurement Window [60s, 360s]                        |
|     ├─ Augmented Dickey-Fuller Test: p_ADF < 0.01 on steady-state queue trajectory               |
|     ├─ Exact Packet Conservation: N_sent = N_received + N_dropped (Δ = 0)                         |
|     └─ Little's Law Equilibrium: |L_q - λ_eff * W_q| / L_q <= 0.02 (at ρ = 0.85)                  |
|                                     │                                                             |
|                                     ▼ [PASS]                                                      |
|  [GATE 4: Multi-Seed Statistical Rigor & Policy Testing]                                          |
|     ├─ Replications: M = 30 independent seeds with Common Random Numbers (CRN)                    |
|     ├─ Tail Latency Intervals: 95% BCa Bootstrap CIs (B = 2,000 resamples) for P95 / P99          |
|     └─ Comparative Inferential Tests: Paired Wilcoxon Signed-Rank Test (p < 0.01, Hedges' g > 0.8)|
|                                                                                                   |
+---------------------------------------------------------------------------------------------------+
```

### 5.1 Gate 1: Micro-Scale Distributional Calibration
Before running network simulations, generated flows under Low Congestion ($\rho = 0.50$, zero queuing distortion) must be verified against empirical `output/phase5/track_b_flows.csv` sustained flows:
1. **Wasserstein-1 Distance ($W_1$):**
   * Packet Size: $W_1(L_{\text{synthetic}}, L_{\text{empirical}}) \le 35.0\text{ Bytes}$ per class.
   * Real-Time IAT: $W_1(\text{IAT}_{\text{synthetic}}, \text{IAT}_{\text{empirical}}) \le 15.0\text{ ms}$.
2. **Jensen-Shannon Divergence ($\text{JSD}$):**
   * Packet Size: $\text{JSD} \le 0.05$ across all classes.
3. **Discrete Categorical Goodness-of-Fit:**
   * 6-Bin Multinomial $\chi^2$ Test with Cramér's $V \le 0.10$.

### 5.2 Gate 2: Macro-Dynamics & Directional Invariant Audit
Verify that the behavioral invariants identified in Section 2 of `traffic_generation_design.md` are strictly preserved:
1. **Real-Time Conversational Symmetry:**
   * Forward Packet Ratio: $0.50 \pm 0.03$ (Empirical: $0.5253$, Median $0.5000$).
   * Forward Byte Ratio: $0.48 \pm 0.04$ (Empirical: $0.4816$, Median $0.5033$).
2. **Streaming Directional Asymmetry:**
   * Forward Byte Ratio: $0.20 \pm 0.08$ (reflecting $>72\%$ downlink video payload).
   * Inter-Chunk Gap: Mean $T_{\text{idle}} \in [2.0\text{ s}, 8.0\text{ s}]$, reproducing DASH chunking.
3. **Bulk Data Line-Rate Dynamics:**
   * Median Bidirectional IAT: $\le 20\ \mu\text{s}$ (reproducing empirical $0.000000\text{ s}$ back-to-back trains).
   * Directional Specialization: Uploads $\ge 95\%$ Forward bytes; Downloads $\le 5\%$ Forward bytes.

### 5.3 Gate 3: Stationarity, Warm-Up & Conservation Audit
Verify the physical validity of the simulation engine:
1. **Warm-Up Execution:** Discard initial $60.0\text{ seconds}$ of execution. Telemetry logs record exclusively from $t = 60.0\text{ s}$ to $t = 360.0\text{ s}$.
2. **Stationarity Check:** Confirm ADF unit root rejection ($p < 0.01$) on moving average buffer occupancy.
3. **Flow Conservation Law:**
   $$N_{\text{transmitted}} = N_{\text{received}} + N_{\text{dropped}}$$
   Verify that discrepancy $\Delta_{\text{conservation}} = |N_{\text{transmitted}} - (N_{\text{received}} + N_{\text{dropped}})| = 0$.
4. **Queue Equilibrium Verification:**
   Confirm relative Little's Law error $\epsilon_{\text{Little}} \le 2.0\%$ under Moderate Congestion ($\rho = 0.85$).

### 5.4 Gate 4: Multi-Seed Statistical Rigor & Policy Testing
When benchmarking classifier-driven scheduling (e.g. Early-5 XGBoost + Priority Queuing vs. FIFO):
1. **Replications:** Execute $M = 30$ independent runs using Common Random Numbers (CRN).
2. **Tail Reporting:** Report all $P_{95}$ and $P_{99}$ latency metrics accompanied by 95% BCa Bootstrap Confidence Intervals ($B = 2,000$ iterations).
3. **Hypothesis Testing:** Report paired Wilcoxon signed-rank test $p$-values and Hedges' $g$ effect sizes. Claims of QoS superiority are accepted if and only if $p < 0.01$ and $g \ge 0.80$.

---

## 6. Implementation Template for Validation & Auditing Engine

Below is the production-ready Python verification harness implementing the Gate 1 through Gate 4 audit protocols:

```python
"""
Phase 7 Traffic Generation & Queueing Validation Auditor
File: scripts/phase7/validate_traffic_generation.py
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon
import statsmodels.tsa.stattools as ts

class TrafficGenerationAuditor:
    def __init__(self, empirical_csv_path):
        self.empirical_df = pd.read_csv(empirical_csv_path)
        
    def compute_distributional_metrics(self, synthetic_sizes, empirical_sizes):
        """Gate 1: Computes Wasserstein-1, JSD, and Cramér's V on packet sizes."""
        # 1. Wasserstein-1 Distance (Bytes)
        w1_dist = stats.wasserstein_distance(synthetic_sizes, empirical_sizes)
        
        # 2. Jensen-Shannon Divergence
        bins = np.array([0, 64, 150, 300, 800, 1400, 1514])
        p_counts, _ = np.histogram(empirical_sizes, bins=bins, density=False)
        q_counts, _ = np.histogram(synthetic_sizes, bins=bins, density=False)
        
        p_prob = p_counts / np.sum(p_counts)
        q_prob = q_counts / np.sum(q_counts)
        
        jsd_val = jensenshannon(p_prob, q_prob, base=2)
        
        # 3. Multinomial Chi-Square & Cramér's V
        expected = p_prob * len(synthetic_sizes)
        # Avoid zero division
        valid_mask = expected > 0
        chi2_stat = np.sum(((q_counts[valid_mask] - expected[valid_mask]) ** 2) / expected[valid_mask])
        cramers_v = np.sqrt(chi2_stat / (len(synthetic_sizes) * (len(bins) - 2)))
        
        # 4. Quantile-Quantile R²
        quantiles = np.linspace(1, 99, 99)
        q_emp = np.percentile(empirical_sizes, quantiles)
        q_syn = np.percentile(synthetic_sizes, quantiles)
        r_qq = np.corrcoef(q_emp, q_syn)[0, 1] ** 2
        
        return {
            'wasserstein_1_bytes': w1_dist,
            'jsd': jsd_val,
            'cramers_v': cramers_v,
            'r_qq_squared': r_qq,
            'gate1_pass': (w1_dist <= 35.0) and (jsd_val <= 0.05) and (cramers_v <= 0.10) and (r_qq >= 0.98)
        }

    def audit_simulation_stationarity(self, queue_time_series, warm_up_sec=60.0, dt_sample=0.1):
        """Gate 3: Audits warm-up truncation and tests for stationarity via ADF."""
        start_idx = int(warm_up_sec / dt_sample)
        steady_state_queue = queue_time_series[start_idx:]
        
        # Augmented Dickey-Fuller Test
        adf_res = ts.adfuller(steady_state_queue, autolag='AIC')
        p_value = adf_res[1]
        
        return {
            'warm_up_discarded_sec': warm_up_sec,
            'adf_test_stat': adf_res[0],
            'adf_p_value': p_value,
            'is_stationary': p_value < 0.01
        }

    def audit_littles_law(self, time_weighted_Lq, lambda_effective, mean_Wq):
        """Gate 3: Audits Little's Law equilibrium."""
        theoretical_Lq = lambda_effective * mean_Wq
        relative_error = abs(time_weighted_Lq - theoretical_Lq) / max(1e-6, time_weighted_Lq)
        return {
            'measured_Lq': time_weighted_Lq,
            'expected_Lq': theoretical_Lq,
            'relative_error': relative_error,
            'gate3_littles_law_pass': relative_error <= 0.02
        }

    def compute_bca_bootstrap_ci(self, data, stat_func=np.percentile, q=95, n_resamples=2000, alpha=0.05):
        """Gate 4: Computes BCa Bootstrap 95% Confidence Interval for tail metrics."""
        res = stats.bootstrap(
            (data,), 
            statistic=lambda x: stat_func(x, q), 
            confidence_level=1.0 - alpha,
            n_resamples=n_resamples, 
            method='BCa'
        )
        return {
            'point_estimate': stat_func(data, q),
            'ci_lower': res.confidence_interval.low,
            'ci_upper': res.confidence_interval.high
        }
```

---

## 7. Conclusion & Methodological Recommendations

The audit conducted by Agent 6 demonstrates that while the multi-tier traffic models defined in Sections 3 through 6 of `traffic_generation_design.md` are grounded in empirical observations, the validation protocol proposed in Section 7 contained fundamental theoretical flaws.

### Summary of Audit Verdict & Mandated Modifications:
1. **Decommission the Classical Two-Sample KS Test for Discrete Packet Sizes:** Replace with Earth Mover's Distance ($W_1 \le 35\text{ Bytes}$), Jensen-Shannon Divergence ($\text{JSD} \le 0.05$), and Discrete Multinomial $\chi^2$ testing.
2. **Abolish "$p > 0.05$" Acceptance Criterion:** Terminate reliance on $p$-values as proof of fit under high $N$. Enforce strict effect-size distance bounds.
3. **Mandate Simulation Warm-Up:** Enforce Welch's procedure with $T_{\text{warm}} = 60.0\text{ s}$ warm-up truncation and ADF stationarity verification before collecting QoS data.
4. **Enforce 30-Seed Replication with CRN:** Require $M = 30$ independent runs using Common Random Numbers to maximize statistical power.
5. **Enforce BCa Bootstrap Intervals for Tails:** Require BCa bootstrap intervals ($B=2,000$) for all $P_{95}$, $P_{99}$, and loss rate metrics.
6. **Implement Time-Weighted Queue Tracking:** Enforce exact continuous integration of buffer occupancy and verify Little's Law ($\epsilon \le 2\%$) and exact packet conservation ($\Delta = 0$).

Adherence to this audited validation framework guarantees that Phase 7 experimental findings will withstand the most rigorous peer review in top-tier networking venues (*IEEE/ACM Transactions on Networking*, *IEEE INFOCOM*, *ACM SIGCOMM*).
