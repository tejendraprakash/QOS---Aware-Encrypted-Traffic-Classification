# QoS Measurement Framework & Auditing Protocol (Phase 7)

**Project:** QoS-Aware Encrypted Network Traffic Classification for Next-Generation Networks  
**Phase:** 7 — Network Simulation, QoS Differentiation & Policy Enforcement  
**Role:** QoS Metrics & Measurement Auditor  
**Document Status:** Approved Specification & Auditing Standard  
**Date:** November 2026  
**Artifact Path:** [measurement_protocol.md](file:///c:/Users/saipr/OneDrive/Desktop/EncryptedTrafficResearch/research_notes/phase7/measurement_protocol.md)

---

## 1. Executive Summary & Auditing Mandate

The primary objective of Phase 7 is to evaluate the operational network utility of the machine-learning-driven traffic classification system established in Phase 6 and Phase 6b. While Phases 5 and 6 demonstrated high classification accuracy, Macro F1, and early-horizon discrimination ($N \in \{5, 10, 15, 20\}$ packets) on the Track B benchmark dataset (Bulk Data, Streaming, Real-Time), machine learning metrics alone do not quantify end-user Quality of Experience (QoE) or network Quality of Service (QoS).

This document establishes the **formal quantitative measurement framework**, **mathematical formulations**, **per-class Service Level Agreement (SLA) thresholds**, and **statistical auditing protocols** governing Phase 7. All simulation engines (e.g., discrete-event simulation testbeds, ns-3 / Mininet / Linux `tc` emulators), traffic schedulers (Priority Queuing [PQ], Deficit Weighted Round Robin [DWRR], Weighted Fair Queuing [WFQ]), and baseline comparative runs (First-In, First-Out [FIFO] Best-Effort) must adhere strictly to these auditing standards.

```
+---------------------------------------------------------------------------------------------------+
|                                  PHASE 7 MEASUREMENT ARCHITECTURE                                 |
+---------------------------------------------------------------------------------------------------+
|  [Ingress Traffic Generator]                                                                      |
|     Track B PCAP / Flow Traces (Bulk Data, Streaming, Real-Time)                                  |
|         │                                                                                         |
|         ▼                                                                                         |
|  [Classifier & Tagging Engine]  ─── Early Decision (N ∈ {5, 10, 15, 20}) or Full-Flow Oracle      |
|         │                                                                                         |
|         ▼                                                                                         |
|  [Bottleneck Link & Schedulers] ─── Schedulers: PQ, DWRR, WFQ vs. Baseline FIFO (Utilization ρ)   |
|         │                                                                                         |
|         ▼                                                                                         |
|  [Auditing & Measurement Framework]                                                               |
|     ├─ Packet-Level: OWD, P95/P99 Tail, IPDV (RFC 3393), Smoothed Jitter (RFC 3550)               |
|     ├─ Flow-Level: Throughput, Goodput, Flow Completion Time (FCT), Disaggregated PLR             |
|     ├─ SLA Compliance: Real-Time (150ms/30ms/1%), Streaming (1s/100ms/2%), Bulk Data (5%/Max G)   |
|     └─ Conservation Audits: Flow Balance (Sent = Recv + Drop), Little's Law (L = λW)              |
+---------------------------------------------------------------------------------------------------+
```

---

## 2. Standard Metric Definitions & Mathematical Formulations

To ensure mathematical precision and comparability across experimental runs, all QoS metrics are defined below at the per-packet, per-flow, and aggregate class levels.

### 2.1 One-Way Delay (OWD) and Tail Latency Formulation

Let $p_i$ denote the $i$-th packet belonging to flow $f$ transmitted across the network path. Let $T_{\text{send}, i}$ be the exact timestamp recorded upon egress from the source network interface, and let $T_{\text{recv}, i}$ be the ingress timestamp recorded upon arrival at the destination network interface.

#### 2.1.1 Per-Packet One-Way Delay
The One-Way Delay $D_i$ of packet $p_i$ is defined as:
$$D_i = T_{\text{recv}, i} - T_{\text{send}, i}$$

*Clock Synchronization Assumption:* In discrete-event simulation (e.g., ns-3), timestamps are derived from a unified global simulation clock ($T_{\text{drift}} = 0$). In hardware or containerized testbed environments, high-precision Precision Time Protocol (IEEE 1588 PTP) or synchronized hardware timestamping (NIC hardware clocks) is mandated, ensuring synchronization error $|\epsilon_{\text{clock}}| \le 1.0\ \mu\text{s} \ll D_i$.

#### 2.1.2 Mean and Median Delay
For a flow or aggregate traffic class comprising $N_{\text{recv}}$ successfully received packets:
- **Sample Mean OWD:**
  $$\bar{D} = \frac{1}{N_{\text{recv}}} \sum_{i=1}^{N_{\text{recv}}} D_i$$
- **Sample Median OWD ($\tilde{D}$):**
  Given the ordered sequence of delays $D_{(1)} \le D_{(2)} \le \dots \le D_{(N_{\text{recv}})}$:
  $$\tilde{D} = D_{\left(\left\lfloor \frac{N_{\text{recv}}+1}{2} \right\rfloor\right)}$$

#### 2.1.3 Tail Latency Quantiles ($P_{95}$ and $P_{99}$)
Tail latency reflects bufferbloat and transient queuing shock, critical for interactive conversational traffic. Let $F_D(x) = \frac{1}{N_{\text{recv}}} \sum_{i=1}^{N_{\text{recv}}} \mathbb{I}(D_i \le x)$ denote the empirical cumulative distribution function (ECDF) of packet delays.

The $q$-th quantile (percentile $P_{100q}$) is formally defined as:
$$P_{100q} = F_D^{-1}(q) = \inf \{ x \in \mathbb{R} : F_D(x) \ge q \}$$
Specifically:
- **95th Percentile Delay ($P_{95}$):**
  $$P_{95} = \inf \{ x : F_D(x) \ge 0.95 \}$$
- **99th Percentile Delay ($P_{99}$):**
  $$P_{99} = \inf \{ x : F_D(x) \ge 0.99 \}$$

For discrete data with sample size $N_{\text{recv}}$, linear interpolation between adjacent ranks is calculated according to NIST / Hyndman-Fan Type 7 formulation:
$$k = 1 + q(N_{\text{recv}} - 1), \quad j = \lfloor k \rfloor, \quad \gamma = k - j$$
$$P_{100q} = (1 - \gamma) D_{(j)} + \gamma D_{(j+1)}$$

---

### 2.2 Delay Variation & Jitter Formulations

Jitter characterizes the stochastic variation in packet transit times. Phase 7 implements both instantaneous and smoothed international standard metrics.

#### 2.2.1 RFC 3393: Instantaneous Packet Delay Variation (IPDV)
According to IETF RFC 3393, IPDV evaluates the delay difference between consecutively received packets:
$$\text{IPDV}_i = D_i - D_{i-1} = (T_{\text{recv}, i} - T_{\text{send}, i}) - (T_{\text{recv}, i-1} - T_{\text{send}, i-1})$$

Rearranging into interarrival and interdeparture intervals:
$$\text{IPDV}_i = (T_{\text{recv}, i} - T_{\text{recv}, i-1}) - (T_{\text{send}, i} - T_{\text{send}, i-1}) = \Delta T_{\text{recv}, i} - \Delta T_{\text{send}, i}$$

Statistical summary metrics for IPDV include:
- **Mean Absolute IPDV:**
  $$\overline{|\text{IPDV}|} = \frac{1}{N_{\text{recv}} - 1} \sum_{i=2}^{N_{\text{recv}}} |\text{IPDV}_i|$$
- **IPDV Sample Standard Deviation:**
  $$\sigma_{\text{IPDV}} = \sqrt{\frac{1}{N_{\text{recv}} - 2} \sum_{i=2}^{N_{\text{recv}}} \left( \text{IPDV}_i - \overline{\text{IPDV}} \right)^2}$$

#### 2.2.2 RFC 3550 / RFC 1889: RTP Smoothed Interarrival Jitter
For real-time multimedia flows (RTP VoIP and videoconferencing), IETF RFC 3550 defines a low-pass, first-order autoregressive filter that continuously estimates smoothed interarrival jitter:

Let $D(i, i-1)$ be the difference in relative transit time between packets $i-1$ and $i$:
$$D(i, i-1) = |D_i - D_{i-1}| = |(T_{\text{recv}, i} - T_{\text{recv}, i-1}) - (T_{\text{send}, i} - T_{\text{send}, i-1})|$$

The smoothed jitter $J_i$ is recursively updated for each received packet $i \ge 2$:
$$J_i = J_{i-1} + \frac{D(i, i-1) - J_{i-1}}{16}$$

This corresponds to an Exponentially Weighted Moving Average (EWMA) with filter smoothing parameter $\alpha = \frac{1}{16} = 0.0625$:
$$J_i = (1 - \alpha) J_{i-1} + \alpha |D_i - D_{i-1}|$$
with initial boundary condition $J_1 = 0$.

*Operational Significance:* While IPDV (RFC 3393) captures instantaneous differential spikes (sensitive to isolated queue spikes), RTP Jitter (RFC 3550) provides the psychoacoustic smoothing metric that directly governs playout buffer sizing in VoIP decoders.

---

### 2.3 Packet Loss Rate (PLR) & Disaggregated Drop Taxonomy

Packet loss severely impairs both transport throughput (TCP congestion window collapse) and interactive voice/video intelligibility.

#### 2.3.1 Global Packet Loss Rate
For a given evaluation interval or flow $f$:
$$\text{PLR} = \frac{N_{\text{dropped}}}{N_{\text{transmitted}}} \times 100\% = \frac{N_{\text{transmitted}} - N_{\text{received}}}{N_{\text{transmitted}}} \times 100\%$$
where $N_{\text{transmitted}} = N_{\text{received}} + N_{\text{dropped}}$.

#### 2.3.2 Disaggregated Drop Taxonomy
To audit whether packet loss is caused by physical buffer exhaustion or dynamic queue management, Phase 7 partitions $N_{\text{dropped}}$ into three disjoint causal categories:
$$N_{\text{dropped}} = N_{\text{drop, overflow}} + N_{\text{drop, timeout}} + N_{\text{drop, AQM}}$$

1. **Queue-Overflow Drops ($N_{\text{drop, overflow}}$):**
   Occurs when packet $p_i$ arrives at a link buffer of capacity $B_{\max}$ (in packets or bytes) and current queue occupancy $Q(t) + \text{size}(p_i) > B_{\max}$ (Tail-Drop or Head-Drop under buffer saturation).
   $$\text{PLR}_{\text{overflow}} = \frac{N_{\text{drop, overflow}}}{N_{\text{transmitted}}} \times 100\%$$
2. **Deadline / Timeout Drops ($N_{\text{drop, timeout}}$):**
   Occurs when an in-flight or buffered packet exceeds its maximum allowable age $T_{\text{deadline}}$ before service or decoding:
   $$T_{\text{current}} - T_{\text{send}, i} > T_{\text{deadline}}$$
   (Crucial for Real-Time voice where packets delayed beyond 250 ms are discarded as unusable by the receiver).
   $$\text{PLR}_{\text{timeout}} = \frac{N_{\text{drop, timeout}}}{N_{\text{transmitted}}} \times 100\%$$
3. **Active Queue Management (AQM) Discards ($N_{\text{drop, AQM}}$):**
   Early stochastic drops initiated by RED, CoDel, or PIE algorithms to signal congestion before full buffer exhaustion.

---

### 2.4 Throughput vs. Goodput Formulation

A clear distinction between raw link occupancy and application utility is required to avoid misrepresenting network efficiency.

#### 2.4.1 Raw Network Throughput ($Th$)
The aggregate rate of physical or link-layer bits transmitted across the interface over duration $\Delta t = T_{\text{end}} - T_{\text{start}}$:
$$Th = \frac{8 \cdot \sum_{i=1}^{N_{\text{serv}}} S_{\text{wire}}(p_i)}{\Delta t} \quad [\text{bps}]$$
where $S_{\text{wire}}(p_i) = S_{\text{payload}}(p_i) + S_{\text{headers}}(p_i)$ includes L2, L3, L4 headers and retransmitted packets.

#### 2.4.2 Application Goodput ($G$)
The rate of useful, non-duplicate, successfully acknowledged application-layer payload delivered to the destination socket over duration $\Delta t$:
$$G = \frac{8 \cdot \sum_{j \in \mathcal{P}_{\text{valid}}} S_{\text{payload}}(p_j)}{\Delta t} \quad [\text{bps}]$$
where $\mathcal{P}_{\text{valid}}$ excludes:
- Protocol header overhead (IP, TCP/UDP headers).
- Retransmitted segments (spurious or loss-recovery retransmissions).
- Packets dropped before socket delivery or arrived out-of-order and dropped.

#### 2.4.3 Transport Protocol Efficiency Ratio ($\eta$)
$$\eta = \frac{G}{Th} \le 1.0$$
A drop in $\eta$ indicates excessive retransmission overhead, header bloat, or badput under congestion.

---

### 2.5 Flow Completion Time (FCT)

For finite transactional and bulk data flows (e.g., FTPS, SCP, file downloads), Flow Completion Time is the definitive performance metric.

#### 2.5.1 Absolute Flow Completion Time ($\text{FCT}_f$)
Let $T_{\text{first\_tx}, f}$ be the timestamp when the source transmits the initial SYN or first data packet of flow $f$, and let $T_{\text{last\_ack}, f}$ be the timestamp when the receiver delivers the final application byte (or sends the final ACK):
$$\text{FCT}_f = T_{\text{last\_ack}, f} - T_{\text{first\_tx}, f} \quad [\text{seconds}]$$

#### 2.5.2 Normalized Flow Completion Time ($\text{nFCT}_f$)
Because raw FCT depends directly on flow byte volume $V_f = \sum S_{\text{payload}}$, comparing flows of different sizes requires normalization against the theoretical ideal completion time under an unloaded link of capacity $C$ and minimum propagation round-trip time $\text{RTT}_{\min}$:
$$\text{FCT}_{\text{ideal}, f} = \frac{8 \cdot V_f}{C} + \text{RTT}_{\min}$$
$$\text{nFCT}_f = \frac{\text{FCT}_f}{\text{FCT}_{\text{ideal}, f}} \ge 1.0$$
An $\text{nFCT}_f = 1.0$ represents optimal wire-speed delivery with zero queuing delay.

---

## 3. Per-Class Service Level Agreement (SLA) & QoS Thresholds

In alignment with Track B's validated 3-class architecture (derived from ITU-T recommendations G.1010, G.114, and 3GPP TS 23.203 QCI classes), we establish explicit, quantifiable SLA pass/fail thresholds.

### 3.1 Formal SLA Criteria Matrix

| Traffic Class | Represented Applications (Track B) | One-Way Delay ($D$) | Jitter ($J$, RFC 3550) | Packet Loss Rate ($\text{PLR}$) | Goodput / Bandwidth Requirement | Primary Optimization Objective |
|:---|:---|:---:|:---:|:---:|:---|:---|
| **Real-Time** | VoIP, Skype, Hangouts Audio/Video, VoIPBuster, Facebook Voice | $\le 150\ \text{ms}$ (Strict)<br>$\le 200\ \text{ms}$ (Upper) | $\le 30\ \text{ms}$ | $\le 1.0\%$ | Guaranteed minimum $\ge 64\ \text{kbps}$ (Voice), $\ge 512\ \text{kbps}$ (Video) | Minimize Tail Latency ($P_{99}$) & Jitter; Absolute priority |
| **Streaming** | Netflix, YouTube, Vimeo, Spotify | $\le 1000\ \text{ms}$ | $\le 100\ \text{ms}$ | $\le 2.0\%$ | Sustained $G \ge R_{\text{target}}$<br>(Audio: 160 kbps; Video: 5 Mbps HD) | Prevent buffer starvation; Smooth rate allocation |
| **Bulk Data** | FTPS, SCP Downloads/Uploads | Delay-Tolerant (Unbounded) | Unbounded | $\le 5.0\%$ | Maximize Goodput ($G \to C_{\text{available}}$) | Minimize Flow Completion Time ($\text{nFCT} \to 1.0$) |

> [!NOTE]
> **International Standards Alignment:**
> - The Real-Time threshold ($D \le 150\ \text{ms}$, $\text{PLR} \le 1\%$) directly corresponds to ITU-T Recommendation G.114 ("One-way transmission time") for highly acceptable conversational voice quality. Beyond 150 ms, conversational interactivity degrades rapidly.
> - The Streaming threshold ($D \le 1000\ \text{ms}$) reflects client media playout buffer dynamics (DASH/HLS chunk duration typically 2–6 seconds). As long as average Goodput exceeds playback bitrate $R_{\text{target}}$ and PLR $\le 2\%$, media players avoid rebuffering stalls.
> - Bulk Data transfers use TCP (CUBIC/BBR) with selective acknowledgments (SACK). They tolerate queuing delay but suffer exponential throughput degradation if PLR exceeds 5%.

---

### 3.2 Quantitative SLA Compliance Verification Formulas

For any experimental run $s$ (scenario, scheduler, traffic mix, seed):

#### 3.2.1 Real-Time Class Compliance Function ($\Phi_{\text{RT}}$)
A Real-Time flow $f$ satisfies its SLA if and only if all three constraint functions evaluate to 1:
$$\Phi_{\text{RT}}(f) = \mathbb{I}\left( \bar{D}_f \le 150\ \text{ms} \right) \times \mathbb{I}\left( P_{95, f} \le 200\ \text{ms} \right) \times \mathbb{I}\left( J_{\text{RTP}, f} \le 30\ \text{ms} \right) \times \mathbb{I}\left( \text{PLR}_f \le 1.0\% \right)$$

The Real-Time Class SLA Compliance Rate is:
$$\text{CR}_{\text{RT}} = \frac{1}{|\mathcal{F}_{\text{RT}}|} \sum_{f \in \mathcal{F}_{\text{RT}}} \Phi_{\text{RT}}(f) \times 100\%$$

#### 3.2.2 Streaming Class Compliance Function ($\Phi_{\text{Stream}}$)
$$\Phi_{\text{Stream}}(f) = \mathbb{I}\left( \bar{D}_f \le 1000\ \text{ms} \right) \times \mathbb{I}\left( J_{\text{RTP}, f} \le 100\ \text{ms} \right) \times \mathbb{I}\left( \text{PLR}_f \le 2.0\% \right) \times \mathbb{I}\left( G_f \ge R_{\text{target}, f} \right)$$
$$\text{CR}_{\text{Stream}} = \frac{1}{|\mathcal{F}_{\text{Stream}}|} \sum_{f \in \mathcal{F}_{\text{Stream}}} \Phi_{\text{Stream}}(f) \times 100\%$$

#### 3.2.3 Bulk Data Class Compliance Function ($\Phi_{\text{Bulk}}$)
$$\Phi_{\text{Bulk}}(f) = \mathbb{I}\left( \text{PLR}_f \le 5.0\% \right) \times \mathbb{I}\left( \text{nFCT}_f \le \gamma_{\text{threshold}} \right)$$
where $\gamma_{\text{threshold}} = 2.50$ (i.e., FCT under congestion does not exceed $2.5\times$ ideal wire speed).
$$\text{CR}_{\text{Bulk}} = \frac{1}{|\mathcal{F}_{\text{Bulk}}|} \sum_{f \in \mathcal{F}_{\text{Bulk}}} \Phi_{\text{Bulk}}(f) \times 100\%$$

#### 3.2.4 System Composite QoS Satisfaction Index ($\Psi$)
To compare scheduling architectures (e.g., Machine-Learning-Guided PQ vs. Default Best-Effort FIFO) across all classes simultaneously, we define the weighted composite satisfaction index:
$$\Psi = w_{\text{RT}} \cdot \text{CR}_{\text{RT}} + w_{\text{Stream}} \cdot \text{CR}_{\text{Stream}} + w_{\text{Bulk}} \cdot \text{CR}_{\text{Bulk}}$$
where default regulatory weights are normalized to:
$$w_{\text{RT}} = 0.50, \quad w_{\text{Stream}} = 0.30, \quad w_{\text{Bulk}} = 0.20$$
reflecting the catastrophic subjective cost of degrading conversational voice relative to elastic file transfers.

---

## 4. Statistical Aggregation, Confidence Intervals & Monte Carlo Protocol

Network traffic exhibits severe non-linear queuing behavior and phase effects (e.g., TCP synchronization under drop-tail queues). Single-run evaluations are scientifically invalid.

### 4.1 Multi-Seed Monte Carlo Replication Design

For every experimental configuration, the simulation harness must execute an ensemble of independent replications:
- **Replication Count:** $M = 5$ independent seeds per scenario-congestion pair:
  $$\mathcal{S} = \{ \text{Seed}_1, \text{Seed}_2, \text{Seed}_3, \text{Seed}_4, \text{Seed}_5 \} = \{ 42, 137, 2024, 777, 999 \}$$
- **Congestion Factor Sweeps ($\rho$):**
  Link offered load $\rho = \frac{\lambda_{\text{offered}}}{C_{\text{bottleneck}}}$ evaluated across six discrete operational regimes:
  $$\rho \in \{ 0.50\ (\text{Uncongested}),\ 0.70\ (\text{Moderate}),\ 0.85\ (\text{High}),\ 0.90\ (\text{Critical}),\ 0.95\ (\text{Near-Saturation}),\ 1.05\ (\text{Overload}) \}$$
- **Classification Policies Evaluated:**
  1. *Unclassified Baseline:* FIFO (Best-Effort single queue).
  2. *ML-Driven Early Classification:* Packets classified at horizon $N \in \{5, 10, 15, 20\}$ using the frozen Random Forest / XGBoost / LightGBM models trained in Phase 6b.
  3. *Ground-Truth Oracle:* 100% accurate classification from packet 1 (theoretical upper bound).
- **Queuing Disciplines Evaluated:**
  - Strict Priority Queuing (PQ): Real-Time in High-Priority Queue; Streaming in Medium; Bulk Data in Low.
  - Deficit Weighted Round Robin (DWRR): Quantum weights $W_{\text{RT}} = 50\%$, $W_{\text{Stream}} = 35\%$, $W_{\text{Bulk}} = 15\%$.
  - Weighted Fair Queuing (WFQ): Equivalent GPS weight allocations.

Total required Monte Carlo runs per scheduler:
$$N_{\text{runs}} = 6\ (\text{congestion levels}) \times 6\ (\text{classification variants}) \times 5\ (\text{seeds}) = 180\ \text{runs per queuing architecture}$$

---

### 4.2 Confidence Interval Estimation

For any evaluated metric $\theta$ (e.g., Mean Delay, $P_{95}$, Loss Rate, Goodput):

#### 4.2.1 Small-Sample Student's $t$ Confidence Intervals
Given $M = 5$ independent replication estimates $X_1, X_2, \dots, X_M$, the sample mean $\bar{X}$ and sample variance $S^2$ are:
$$\bar{X} = \frac{1}{M} \sum_{m=1}^{M} X_m, \quad S^2 = \frac{1}{M - 1} \sum_{m=1}^{M} (X_m - \bar{X})^2$$

The two-sided $(1 - \alpha) \times 100\%$ confidence interval is given by:
$$\text{CI}_{1-\alpha} = \left[ \bar{X} - t_{\alpha/2, M-1} \cdot \frac{S}{\sqrt{M}}, \quad \bar{X} + t_{\alpha/2, M-1} \cdot \frac{S}{\sqrt{M}} \right]$$

For a 95% Confidence Interval with $M = 5$ ($df = 4$), the critical value is:
$$t_{0.025, 4} = 2.776$$
$$\text{CI}_{95\%} = \bar{X} \pm 2.776 \cdot \frac{S}{\sqrt{5}} = \bar{X} \pm 1.2415 \cdot S$$

#### 4.2.2 Non-Parametric Bootstrap for Tail Percentiles ($P_{95}, P_{99}$)
Because sample tail percentiles from queuing delays exhibit heavy-tailed (Pareto/Weibull-like) non-Gaussian distributions, calculating confidence intervals of $P_{95}$ and $P_{99}$ via Student's $t$ introduces distributional bias.

For tail metrics, Phase 7 mandates **Bias-Corrected and Accelerated (BCa) Non-Parametric Bootstrapping**:
1. From the empirical delay distribution $\mathcal{D} = \{D_1, \dots, D_N\}$, draw $B = 2,000$ independent bootstrap resamples with replacement: $\mathcal{D}^{*b}$, each of size $N$.
2. For each resample, calculate the bootstrap statistic $\hat{\theta}^{*b} = P_{95}(\mathcal{D}^{*b})$.
3. Sort the bootstrap estimates $\hat{\theta}^{*(1)} \le \hat{\theta}^{*(2)} \le \dots \le \hat{\theta}^{*(B)}$.
4. Compute the bias-correction constant $z_0$ and acceleration parameter $a$:
   $$z_0 = \Phi^{-1} \left( \frac{1}{B} \sum_{b=1}^{B} \mathbb{I}(\hat{\theta}^{*b} < \hat{\theta}) \right)$$
5. The 95% BCa confidence interval $[\theta_L, \theta_U]$ is extracted from percentiles $\alpha_1, \alpha_2$:
   $$\alpha_1 = \Phi \left( z_0 + \frac{z_0 + z_{\alpha/2}}{1 - a(z_0 + z_{\alpha/2})} \right), \quad \alpha_2 = \Phi \left( z_0 + \frac{z_0 + z_{1-\alpha/2}}{1 - a(z_0 + z_{1-\alpha/2})} \right)$$

---

## 5. Rigorous Auditing & Conservation Verification Protocols

To detect simulation anomalies, queue leakage, numerical drift, or timestamp inaccuracies, the test framework must run automated integrity audits after every simulation execution. A run failing any of these conservation checks is declared **INVALID** and aborted immediately.

### 5.1 Conservation of Flow Audit
Network nodes cannot synthesize or silently destroy packets without logging an event.

#### Audit Formulation
For every network node $u \in \mathcal{V}$ and every traffic class $c \in \mathcal{C}$ over the simulation duration $[0, T_{\text{sim}}]$:
$$N_{\text{transmitted}, c}^{\text{source}} = N_{\text{received}, c}^{\text{dest}} + N_{\text{dropped}, c}^{\text{bottleneck}} + N_{\text{inflight}, c}(T_{\text{sim}})$$

Where $N_{\text{inflight}, c}(T_{\text{sim}})$ represents packets currently stored in queues or propagation links at the final simulation tick.

#### Audit Assertion
$$\Delta_{\text{conservation}} = \left| N_{\text{transmitted}, c} - \left( N_{\text{received}, c} + N_{\text{dropped}, c} + N_{\text{inflight}, c} \right) \right|$$
$$\text{Audit Rule: } \Delta_{\text{conservation}} = 0 \quad \text{(Exact integer balance)}$$

Any non-zero discrepancy indicates a memory leak, unmonitored queue drop, or packet replication bug in the simulation harness.

---

### 5.2 Queueing Stability & Little's Law Audit

In any stationary queuing system (or during stable, non-transient measurement intervals $[T_1, T_2]$ where arrival rate $\lambda$ is strictly below link capacity $C$), the fundamental relationship proven by Little (1961) must hold:

$$L_q = \lambda_{\text{effective}} \cdot W_q$$

#### Variables Defined
1. **Time-Averaged Queue Length ($L_q$):**
   Calculated from continuous buffer occupancy trajectory $q(t)$:
   $$L_q = \frac{1}{T_2 - T_1} \int_{T_1}^{T_2} q(t) dt$$
2. **Effective Arrival Rate ($\lambda_{\text{effective}}$):**
   Excludes dropped packets:
   $$\lambda_{\text{effective}} = \frac{N_{\text{admitted}}}{T_2 - T_1} = \frac{N_{\text{transmitted}} - N_{\text{dropped}}}{T_2 - T_1}$$
3. **Mean Waiting Time in Queue ($W_q$):**
   Sample mean of time spent waiting in buffer before transmission begins:
   $$W_q = \frac{1}{N_{\text{admitted}}} \sum_{i=1}^{N_{\text{admitted}}} (T_{\text{service\_start}, i} - T_{\text{enqueued}, i})$$

#### Audit Assertion
The relative Little's Law error $\epsilon_{\text{Little}}$ must satisfy:
$$\epsilon_{\text{Little}} = \frac{|L_q - \lambda_{\text{effective}} W_q|}{L_q} \le 0.02\ (2.0\%)$$
A divergence exceeding 2.0% indicates that the measurement interval was non-stationary, queue sampling frequency was inadequate, or timer arithmetic suffered precision truncation.

---

### 5.3 Work Conservation & Capacity Utilization Audit

For non-idling work-conserving schedulers (e.g., WFQ, DWRR, standard PQ):

1. **Work Conservation Condition:**
   The bottleneck link must never remain idle whenever any queue in the system contains at least one packet:
   $$Q_{\text{total}}(t) > 0 \implies \text{Link State}(t) = \text{BUSY}$$
2. **Capacity Boundedness:**
   Instantaneous aggregate throughput summed across all traffic classes cannot exceed physical link capacity $C$:
   $$\sum_{c \in \mathcal{C}} Th_c(t) \le C_{\text{bottleneck}} + \delta_{\text{jitter}}$$
3. **Starvation Prevention Audit:**
   Under DWRR and WFQ, class Goodput must respect configured minimum guarantee weights even under maximum congestion ($\rho \ge 1.05$):
   $$G_c \ge W_c \cdot C_{\text{bottleneck}} \cdot (1 - \text{HeaderOverhead})$$

---

### 5.4 Early Classification Policy Decision Auditing

In Phase 6b, early classification was evaluated at horizons $N \in \{5, 10, 15, 20\}$ packets. In a live network environment, a packet scheduling system must assign queue priority *before* the horizon $N$ is reached.

#### Audit Protocol for Early Horizon Inference
1. **Cold-Start Buffer Rule ($i < N$):**
   For the first $N-1$ packets of an unclassified flow:
   - *Default Fallback Class:* All unclassified packets default to **Best-Effort (Medium Priority / Streaming equivalent)** queue.
   - *Prohibition:* Unclassified flows are **strictly forbidden** from entering the Highest Priority Real-Time queue prior to model prediction to prevent malicious or accidental starvation of legitimate voice flows.
2. **Inference Latency Constraint ($T_{\text{inference}}$):**
   Let $T_{\text{arrival}, N}$ be the arrival time of the $N$-th packet. The classifier prediction $\hat{y}_f$ must be returned within budget:
   $$T_{\text{decision}} = T_{\text{arrival}, N} + T_{\text{inference}}$$
   $$T_{\text{inference}} \le 2.0\ \text{ms}$$
   (Well within the sub-millisecond inference latencies verified in Phase 6: Random Forest 2.6 $\mu\text{s}$, XGBoost 11.4 $\mu\text{s}$, P-MLP 8.1 $\mu\text{s}$).
3. **Reclassification & Queue Migration Audit:**
   Upon classification at packet $N$:
   - Packet $N$ and all subsequent packets $i \ge N$ are enqueued in queue $Q_{\hat{y}_f}$.
   - Packets already enqueued in the fallback queue ($i < N$) must complete transmission without reordering to maintain TCP sequence integrity.

---

## 6. Implementation Blueprint & Data Structures

All Phase 7 simulation engines must output structured data matching the schemas below to facilitate automated aggregation and audit script verification.

### 6.1 Telemetry Event Schema (`packet_trace.csv`)
Logs every packet crossing the bottleneck:
```csv
packet_id,flow_id,class_ground_truth,class_predicted,send_time_sec,recv_time_sec,owd_ms,ipdv_ms,smoothed_jitter_ms,packet_size_bytes,queue_id,queue_occupancy_bytes,dropped_flag,drop_reason
10001,flow_42,Real-Time,Real-Time,10.001245000,10.015420000,14.175,0.120,2.340,160,0,1280,0,NONE
10002,flow_89,Bulk Data,Bulk Data,10.001250000,10.082100000,80.850,5.210,12.110,1460,2,65536,0,NONE
10003,flow_89,Bulk Data,Bulk Data,10.001255000,0.000000000,0.000,0.000,0.000,1460,2,65536,1,QUEUE_OVERFLOW
```

### 6.2 Flow Summary Schema (`flow_summary.csv`)
Aggregates end-to-end metrics per flow:
```csv
flow_id,ground_truth,predicted,horizon_N,packets_sent,packets_recv,packets_dropped,mean_owd_ms,p95_owd_ms,p99_owd_ms,rfc3550_jitter_ms,plr_pct,throughput_bps,goodput_bps,fct_sec,nfct,sla_compliant
flow_42,Real-Time,Real-Time,10,1500,1498,2,14.22,18.45,24.10,2.15,0.13,64200,60150,30.01,1.02,1
flow_89,Bulk Data,Bulk Data,10,25000,24100,900,65.40,95.20,112.50,14.80,3.60,4850000,4520000,7.82,1.35,1
```

### 6.3 Verification Audit Checklist for Phase 7 Execution

Before accepting experimental results from any simulation scenario, the following checklist must be systematically verified by the auditing harness:

- [ ] **Exact Flow Conservation:** $\Delta_{\text{conservation}} = 0$ for all classes across all runs.
- [ ] **Queue Stability Check:** Little's Law error $\epsilon_{\text{Little}} \le 2.0\%$ under stationary intervals ($\rho \le 0.85$).
- [ ] **SLA Compliance Computation:** $\text{CR}_{\text{RT}}$, $\text{CR}_{\text{Stream}}$, $\text{CR}_{\text{Bulk}}$, and composite $\Psi$ calculated.
- [ ] **Tail Quantile Reliability:** BCa bootstrap confidence intervals computed for $P_{95}$ and $P_{99}$ with $B \ge 2,000$.
- [ ] **Multi-Seed Replication:** 5 distinct seeds verified with reported mean $\pm$ 95% Student's $t$ CI.
- [ ] **Work-Conservation Verified:** Zero unforced idle link ticks when aggregate queue length $> 0$.
- [ ] **Early Horizon Fallback Checked:** No unclassified packets ($i < N$) injected into High-Priority Queue.

---

*This specification serves as the formal QoS auditing protocol for Phase 7 network simulation experiments.*
