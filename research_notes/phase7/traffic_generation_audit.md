# Phase 7 Independent Technical Audit Report: Traffic Generation Blueprint

**Document Reference:** `research_notes/phase7/traffic_generation_audit.md`  
**Audited Document:** `research_notes/phase7/traffic_generation_design.md` (Document Code: AGY-PHASE7-TRFGEN-2026-01)  
**Lead Auditor & Synthesis Chair:** Lead Research Engineer & Multi-Agent Audit Board  
**Contributing Independent Auditors:**
1. Agent 1 — TCP CUBIC Auditor (`b721e74a-f2f9-4a6f-bc6d-bf435e5cc382`)
2. Agent 2 — Streaming Generator Auditor (`b09eb552-7d2d-4c73-bb3b-3d27772b11e3`)
3. Agent 3 — Real-Time Traffic Auditor (`97cdeb57-9e09-48ea-9187-80d7df259563`)
4. Agent 4 — Statistical Calibration Auditor (`b5061fee-bedd-473b-b643-efe251233bee`)
5. Agent 5 — Feature Compatibility Auditor (`19a29998-4b3a-4784-8ab4-ef1bc1e0814c`)
6. Agent 6 — Validation & Methodology Auditor (`40611ab8-aaf2-4864-afea-ccb4f68053e1`)
7. Agent 7 — Standards & Literature Auditor (`8e526cf5-1214-4554-bd22-1846191388c4`)  
**Date:** November 2026  
**Final Audit Verdict:** **APPROVED WITH CONDITIONS**

---

## 1. Executive Summary & Audit Synthesis

Before advancing to Phase 7 network simulation and QoS classifier integration, an independent, multi-perspective technical audit of the traffic generation blueprint (`research_notes/phase7/traffic_generation_design.md`) was conducted. Seven specialized audit agents evaluated the document independently across transport mechanics, video streaming, real-time audio/video, empirical calibration against `output/phase5/track_b_flows.csv`, early-feature compatibility with Phase 6B models, experimental validation methodology, and international networking standards (3GPP, IETF RFCs, ITU-T, ISO/IEC).

### Overall Audit Assessment: **APPROVED WITH CONDITIONS**
The conceptual architecture of the traffic generation blueprint—specifically its hierarchical design (Poisson flow scheduling $\to$ macro-burst state transitions $\to$ packet-level morphological distributions), its grounding in the Track B dataset, and its focus on early-packet QoS evaluation—is scientifically sound.

However, the audit revealed **five critical mathematical errors, seven major pseudo-code bugs, and two systemic network modeling conflicts** that would cause simulation collapse, buffer overflow, and invalid experimental conclusions if implemented without correction:
1. **Unbounded TCP CUBIC Window Explosion:** The pseudo-code implements an open-loop packet injector that never detects packet loss or updates loss timestamps, causing $cwnd$ to explode cubically to $13,984\text{ packets}$ ($20.9\text{ MB}$ per burst) within 34 seconds, obliterating the 417-packet bottleneck buffer.
2. **Mathematical Parameter Errors in RFC 8312 CUBIC:** The $K$ formula omitted the $(1 - \beta_{cubic})$ term, conflated the multiplicative decrease factor ($\beta = 0.7$) with the reduction fraction ($0.2$), and omitted the TCP-friendly region ($W_{est}$).
3. **Jensen's Inequality Bias in Lognormal Video Sizing:** Naive lognormal parameterization $\mu = \ln(M)$ introduces a $+3.17\%$ upward bias in video segment volume ($\mathbb{E}[X] = M e^{\sigma^2/2}$), distorting planned bottleneck utilization.
4. **Catastrophic Playback Buffer Collapse & Erratic Delayed ACKs:** The streaming generator lacked non-negative buffer clamping (plunging into negative infinity under congestion and trapping the generator in an infinite 5 ms loop) and implemented delayed ACKs via `(bytes_sent // pkt_size) % 2 == 0`, which malfunctions on variable packet sizes ($6.5\%$ error rate).
5. **Real-Time Random Walk Clock Drift & False DTX/SID Assumption:** Recursive packet pacing ($t_{i+1} = t_i + \tau + \epsilon_i$) causes cumulative Brownian clock drift ($>80\text{ ms}$ over a call), and modeling 160 ms cellular SID comfort noise contradicts modern encrypted WebRTC (which disables DTX by default and runs continuous 50 Hz).
6. **Open-Loop vs. Closed-Loop Little's Law Conflict:** Assuming an open-loop bit rate of $8.0\text{ Mbps}$ for greedy TCP CUBIC bulk flows is impossible on an unpoliced 100 Mbps link; 3 bulk flows will greedily consume all remaining capacity (~70 Mbps) and fill the queue, destroying the target Low Congestion ($\rho = 0.50$) regime.
7. **Classical Kolmogorov-Smirnov Test Invalidation:** The proposed two-sample KS test with $N=1,000$ flows is theoretically invalid on discrete multimodal packet sizes and guarantees false rejection ($p \ll 0.001$) due to the large-sample $p$-value paradox.

---

## 2. Master Itemized List of Mathematical Errors

| Error ID | Section / Location in Blueprint | Original Mathematical Form | Correct Mathematical Form | Standards & Theoretical Derivation |
| :---: | :--- | :--- | :--- | :--- |
| **M-01** | Section 3.3.1, Eq. 2 (line 212) | $K = \sqrt[3]{\frac{W_{max} \cdot \beta_{cubic}}{C}}$ | $K = \sqrt[3]{\frac{W_{max} \cdot (1 - \beta_{cubic})}{C}}$ | **RFC 8312 Section 4.1, Eq. 2:** The time $K$ required to recover to $W_{max}$ depends on the deficit to be recovered: $(1 - \beta_{cubic})W_{max}$. The $(1 - \cdot)$ operation was completely omitted. |
| **M-02** | Section 3.3.1, Bullet 2 (line 214) | $\beta_{cubic} = 0.2$ ("reducing window by 20% to $0.8 W_{max}$") | $\beta_{cubic} = 0.7$ (standard) or $\beta_{cubic} = 0.8$ with reduction fraction $1 - \beta = 0.2$ | **RFC 8312 Section 4.6:** $\beta_{cubic}$ is defined as the *retained* fraction of window after loss. Default is $0.7$ (a $30\%$ drop). Setting $\beta = 0.2$ represents an $80\%$ drop. |
| **M-03** | Section 3.3.1 (omission) | Missing $W_{est}(t)$ | $W_{est}(t) = W_{max} \beta_{cubic} + \left[\frac{3(1-\beta_{cubic})}{1+\beta_{cubic}}\right] \frac{t}{RTT}$<br>$cwnd = \max(W_{cubic}(t), W_{est}(t))$ | **RFC 8312 Section 4.2:** TCP-friendly AIMD lower bound is mandatory to prevent window starvation on small BDP / low RTT connections. |
| **M-04** | Section 3.2.2, Eq. (line 166) | $\mu = \ln\left(\frac{R_{video} T_{seg}}{8}\right)$ | $\mu^* = \ln\left(\frac{R_{video} T_{seg}}{8}\right) - \frac{\sigma^2}{2}$ | **Jensen's Inequality:** For lognormal $X \sim \text{Lognormal}(\mu, \sigma^2)$, $\mathbb{E}[X] = \exp(\mu + \sigma^2/2)$. Setting $\mu = \ln(M)$ causes $+3.17\%$ upward bias for $\sigma = 0.25$ ($+13.3\%$ for $\sigma = 0.50$). |
| **M-05** | Section 3.1.2, Eq. (line 123) | $t_{i+1} = t_i + \tau + \epsilon_i$<br>$\epsilon_i \sim \mathcal{N}(0, \sigma^2)$ | $t_k = t_{start} + k \cdot \tau + \delta_k$<br>$\delta_k \sim \text{Uniform}(-\delta_{max}, \delta_{max})$ | **Brownian Random Walk:** Recursive formulation has variance $\text{Var}(t_N) = N \sigma^2$. After $10,000$ packets, clock drifts by $>80\text{ ms}$. Pacing must be anchored to an absolute hardware sample clock ($t_k = t_0 + k\tau$). |

---

## 3. Master Itemized List of Pseudo-Code Bugs

| Bug ID | Component | Location in Code | Bug Mechanism & Flaw | Concrete Impact & Failure Mode | Required Correction |
| :---: | :--- | :--- | :--- | :--- | :--- |
| **B-01** | `BulkDataFlow` | Section 6, lines 540–585 | `self.last_loss_time` and `self.w_max` are never updated upon packet drop. | Zero loss detection; $cwnd$ grows cubically without bound ($W \to \infty$), reaching 13,984 packets in 34 seconds. | Implement closed-loop ACK listening and queue drop feedback to trigger multiplicative decrease and $W_{max}$ reset. |
| **B-02** | `BulkDataFlow` | Section 6, lines 580–585 | Emits all reverse ACKs in a single zero-delay loop at `self.env.now`. | Dumps hundreds/thousands of ACKs at identical timestamps with zero IAT (ACK clumping anomaly). | Space ACKs across reverse link transmission and propagation delay. |
| **B-03** | `StreamingFlow` | Section 6, line 493 | `if (bytes_sent // pkt_size) % 2 == 0:` | Division by fluctuating packet size ($1460, 850, 120\text{ B}$) produces consecutive ACKs ($6.5\%$ of pkts) and skips ACKs. | Implement explicit packet sequence counter: `pkt_count += 1; if pkt_count % 2 == 0: emit_ack()`. |
| **B-04** | `StreamingFlow` | Section 6, line 464 | `req_size = np.random.choice(STREAMING_UP_SIZES, p=STREAMING_UP_PROBS)` | Samples HTTP GET requests from aggregate uplink distribution where $85\%$ of sizes are $52\text{ B}$ bare TCP ACKs. | Hardcode authentic HTTP GET request header envelope ($450\text{ B}$) for chunk requests. |
| **B-05** | `StreamingFlow` | Section 6, lines 510–522 | `current_buffer += self.segment_duration - t_download` lacks non-negative clamp; `buffer_target = 30.0` is dead code. | Under congestion ($t_{download} > T_{seg}$), buffer plunges to negative infinity; flow is trapped in infinite 5 ms fast-buffering loop. | Implement 3-state buffer machine (`STARTUP`, `STEADY`, `REBUFFERING`) with $B(t) = \max(0, B(t))$ and active target regulation. |
| **B-06** | `StreamingFlow` | Section 6, lines 463–478 | Zero timeout between HTTP GET emission and first downlink chunk packet. | HTTP GET request and first chunk packet arrive at identical timestamps; zero RTT network turnaround. | Inject $RTT / 2$ network and server turnaround delay before chunk transmission begins. |
| **B-07** | `RealTimeFlow` | Section 6, lines 404, 423 | `self.packet_sink.put({...})` called synchronously without `yield`. | In SimPy, `simpy.Store.put()` returns an event; unyielded calls fail to insert packets into the simulation event queue. | Prepend `yield` to all `packet_sink.put(...)` calls or use safe dispatch wrapper. |

---

## 4. Comprehensive Parameter Classification Matrix

Every parameter in `traffic_generation_design.md` has been audited and classified into one of five rigorous scientific categories:

```
[VC] Verified Correct        : Empirically verified or derived from standards with zero errors.
[IC] Incorrect               : Factually, mathematically, or empirically false.
[SM] Simplified Model        : Stylized engineering approximation lacking physical dynamics.
[UC] Unsupported Claim       : Statement contradicted or unevidenced by empirical dataset.
[RC] Requires Calibration    : Parameter requiring empirical tuning before execution.
```

| Parameter Name | Document Value | Audited Truth / Empirical Source | Category | Detailed Audit Findings |
| :--- | :---: | :---: | :---: | :--- |
| **Total Flow Population** | 8132 RT, 1290 STR, 1348 BLK | `output/phase5/track_b_flows.csv` | **[VC]** | Exact match with Track B dataset. |
| **Sustained Flow Population** | 377 RT, 344 STR, 231 BLK | `total_packets >= 20` filter | **[VC]** | Exact match with sustained flow filtering. |
| **Sustained Flow Durations** | RT: 246.2s, STR: 250.8s, BLK: 34.4s | Track B sustained flow means | **[VC]** | Exact numerical match (4 decimal places). |
| **Sustained Flow Volumes** | RT: 1.72MB, STR: 5.17MB, BLK: 30.2MB | Track B sustained byte means | **[VC]** | Exact numerical match. |
| **Sustained Byte Rates** | RT: 27.5 KB/s, STR: 114.2 KB/s, BLK: 4.44 MB/s | Track B sustained throughput | **[VC]** | Exact numerical match. |
| **Sustained Packet Sizes** | RT Med 200B, STR Bwd 75th 1316B, BLK 1500B | Track B sustained packet sizes | **[VC]** | Exact numerical match. |
| **Directional Byte Ratios** | RT: 0.48, STR: 0.33, BLK: 0.57 | Track B sustained byte ratios | **[VC]** | Exact numerical match. |
| **Burst Byte Sizes** | RT: 674/496B, STR: 667/1331B, BLK: 1790/895B | Track B burst feature means | **[VC]** | Exact numerical match. |
| **Sustained Dominant Transport** | RT UDP 88%, STR TCP 59%, BLK UDP 76% | **RT TCP 87%, STR TCP 85%, BLK TCP 96%** | **[IC]** | **Transcription Error:** Values transcribed from raw dataset (including 9,818 ephemeral flows). Sustained Bulk Data is 96.1% TCP! |
| **RT "50.34 pkts/s Mode"** | Claimed mode at 50.34 pkts/s | Sample mean $= 50.34$, Median $= 0.66$ | **[IC]** | **Statistical Fallacy:** Confused skewed arithmetic mean with distribution mode. Only 1 flow out of 377 is in $[40, 60]\text{ pkts/s}$. |
| **Real-Time Mode 20 ms IAT** | Claimed mode at 20 ms in Track B | Track B median IAT $= 1.55\text{ s}$ | **[UC]** | Track B records flow summary statistics; 20 ms is an imported standard, not an empirical dataset mode. |
| **CUBIC $K$ Time Deficit** | $K = (W_{max}\beta/C)^{1/3}$ | $K = [W_{max}(1-\beta)/C]^{1/3}$ | **[IC]** | Dropped $(1 - \beta)$ term from RFC 8312 Equation 2. |
| **CUBIC Decrease Factor** | $\beta_{cubic} = 0.2$ (stated as 20% drop) | $\beta_{cubic} = 0.7$ (RFC 8312) | **[IC]** | Conflated retained fraction with reduction fraction; $\beta = 0.2$ means an $80\%$ drop. |
| **Lognormal Chunk Parameter** | $\mu = \ln(M)$ | $\mu^* = \ln(M) - \sigma^2/2$ | **[IC]** | Violates Jensen's inequality; induces $+3.17\%$ upward volume bias. |
| **3GPP 5QI 9 Delay Budget** | "Unbounded" | PDB $= 300\text{ ms}$ (3GPP TS 23.501) | **[IC]** | Misquoted 3GPP TS 23.501 Table 5.7.4-1. |
| **Queueing Flow Durations** | $\mathbb{E}[D] = 120\text{s}, 180\text{s}, 15\text{s}$ | Heuristic assumptions | **[SM]** | Deviates from Track B means (246s, 251s, 34s) to simplify simulation runtimes. |
| **Queueing Flow Bitrates** | $\bar{r} = 80\text{k}, 2.0\text{M}, 8.0\text{M}$ | Heuristic assumptions | **[SM]** | Round-number engineering heuristics; conflicts with closed-loop TCP dynamics. |
| **Open-Loop Bulk Load $\rho=0.50$**| Assumed unpoliced bulk rate $= 8.0\text{ Mbps}$ | Closed-loop TCP saturates link | **[SM]** | Greedy CUBIC consumes all remaining bandwidth (~70 Mbps), destroying $\rho = 0.50$ baseline. |
| **Streaming Bitrate Mismatch** | `StreamingFlow` $= 3.5\text{M}$, Orch $= 2.0\text{M}$ | Parameter divergence | **[RC]** | $+75\%$ bitrate mismatch between generator and orchestrator. Harmonize to $2.0\text{ Mbps}$. |
| **Real-Time 80–100 B Mode** | Omitted from discrete CDF | Mode $= 90\text{ B}$ in $>45\%$ of flows | **[RC]** | Recalibrate discrete packet size CDF to include dominant 90 B Opus mode. |
| **Two-Sample KS Test** | $D < 0.15$ with $p > 0.05$ | Flawed on discrete packet sizes | **[IC]** | Continuous axiom violated; large-sample $p$-value paradox guarantees rejection ($p \ll 0.001$). |

---

## 5. Unverified Assumptions from Dataset 1 (Track B)

The following assumptions in the blueprint cannot be directly validated from the UNB ISCX VPN-nonVPN dataset (`track_b_flows.csv`) and must be explicitly documented as synthetic modeling priors:

1. **Active Talkspurt / Silence Alternation (Brady Model):**  
   `track_b_flows.csv` contains aggregate summary features per flow (means, standard deviations, percentiles), not raw packet arrival timestamps. The parameters $\mathbb{E}[T_{on}] = 1.20\text{ s}$ and $\mathbb{E}[T_{off}] = 1.80\text{ s}$ are derived from ITU-T P.59 and Brady (1969), not fitted directly from Track B flow traces.
2. **Video Segment Duration ($T_{seg} = 3.0\text{ s}$):**  
   The $3.0\text{ s}$ chunk duration is imported from ISO/IEC 23009-1 (DASH-IF IOP v4.3) and commercial CDN literature. The dataset reflects encrypted byte volumes and durations, but segment boundaries were obscured by TLS encryption.
3. **Queueing Target Flow Bitrates ($\bar{r}_k$):**  
   The target flow rates used in Little's Law ($\bar{r}_{RT} = 80\text{ kbps}$, $\bar{r}_{STR} = 2.0\text{ Mbps}$, $\bar{r}_{BLK} = 8.0\text{ Mbps}$) are stylized operational heuristics. In Track B, sustained flows had empirical mean bitrates of $219.8\text{ kbps}$ (RT), $913.8\text{ kbps}$ (STR), and $35.5\text{ Mbps}$ (BLK).
4. **Flow Holding Times ($\mathbb{E}[D_k]$):**  
   The simulation holding times ($120\text{ s}, 180\text{ s}, 15\text{ s}$) are compressed engineering heuristics selected to allow multi-flow turnover within a 360-second simulation window. Track B sustained flows had empirical means of $246.2\text{ s}, 250.8\text{ s}, 34.4\text{ s}$.
5. **Traffic Mix Shares ($\phi_{RT} = 0.20, \phi_{STR} = 0.40, \phi_{BLK} = 0.40$):**  
   These ratios represent a standardized 5G network slicing benchmark (20% voice, 40% video, 40% data). In Track B, flow counts were heavily skewed toward Real-Time (75.5% RT, 12.0% STR, 12.5% BLK).
6. **Network Bottleneck Parameters ($C_{link} = 100\text{ Mbps}, RTT = 40\text{--}50\text{ ms}, BDP = 625\text{ KB}$):**  
   These values define the target next-generation network testbed. They are external simulation constraints, not properties of the captured host environment where Dataset 1 was collected.

---

## 6. Concrete Recommendations & Architectural Remediation

### 6.1 TCP CUBIC Remediation Strategy
1. **Mathematical Correction:** Update the CUBIC module to use RFC 8312 Equation 2:
   $$K = \sqrt[3]{\frac{W_{max} \cdot (1 - \beta_{cubic})}{C}}$$
   with standard parameter $\beta_{cubic} = 0.7$ and RFC 8312 Section 4.2 TCP-friendly lower bound $W_{est}(t)$.
2. **Closed-Loop Feedback Implementation:** Replace open-loop packet injection with a closed-loop SimPy architecture featuring sequence number tracking, bottleneck drop feedback, and slow-start initialization.
3. **Paced Packet Transmission:** Space packet transmissions across the smoothed RTT ($dt = RTT / cwnd$) to prevent artificial square-wave queue monopolization.
4. **Resolution of Little's Law Contradiction:** For Low/Moderate congestion benchmarks, enforce rate shaping via a Token Bucket Filter (TBF) at $8.0\text{ Mbps}$, or model Bulk Data as Poisson arrivals of finite file transfers ($S \sim \text{Bounded-Pareto}$) under processor sharing.

### 6.2 Streaming Generator Remediation Strategy
1. **Jensen's Inequality Offset:** Apply the second-order correction to lognormal chunk sizing:
   $$\mu^* = \ln\left(\frac{R_{video} T_{seg}}{8}\right) - \frac{\sigma^2}{2}$$
2. **Delayed ACK Packet Counter:** Replace integer division with an explicit packet counter (`pkt_count % 2 == 0`) with a trailing flush for odd-sized segments.
3. **Three-State Buffer Engine:** Implement non-negative buffer clamping ($B(t) = \max(0, B(t))$), startup priming threshold ($B_{startup} = 10\text{ s}$), and active steady-state regulation around $B_{target} = 30\text{ s}$.
4. **HTTP Header & RTT Turnaround:** Hardcode HTTP GET request envelopes to $450\text{ B}$ (isolated from $52\text{ B}$ ACKs) and inject $RTT / 2$ network turnaround delay.
5. **Bitrate Harmonization:** Align `StreamingFlow` default bitrate to $2.0\text{ Mbps}$ to match `TrafficGenerationOrchestrator`.

### 6.3 Real-Time Generator Remediation Strategy
1. **Absolute Clock Scheduling:** Replace recursive random walks with absolute sampling clock pacing ($t_k = t_{start} + k \cdot \tau + \delta_k$) to eliminate cumulative clock drift.
2. **Coupled 4-State Dialogue Model:** Replace independent talker processes with a coupled 4-state Markov dialogue model (Mutual Silence, Speaker A, Speaker B, Double-Talk) to enforce turn-taking and limit double-talk to empirical levels ($3\%\text{--}7\%$).
3. **Encrypted WebRTC Pacing (No 160 ms SID):** Replace cellular AMR DTX with modern encrypted WebRTC continuous 50 Hz low-bitrate comfort noise framing ($64\text{ B}$) to mirror timing obfuscation.
4. **Empirical Opus Mode Calibration:** Include the dominant $90\text{ B}$ packet size mode (representing $>45\%$ of Track B sustained flows) in the discrete packet size mixture.

### 6.4 Classifier Integration & Feature Pipeline
1. **Zero-Scaling Rule:** Confirm that features fed to `xgb_model.predict()` are **strictly unscaled**, matching Phase 6B training protocol.
2. **Standard 6-Feature Schema:** Guarantee that `early_5_complete`, `early_5_duration`, `early_5_bytes`, `early_5_fwd_pkts`, `early_5_bwd_pkts`, `early_5_avg_size` are extracted using the exact logic from `scripts/flow_extractor.py` (forward direction established by the first observed packet).
3. **Label & Queue Mapping:** Invariant mapping: $0 \to \text{Bulk Data}$ (Queue 2), $1 \to \text{Real-Time}$ (Queue 0), $2 \to \text{Streaming}$ (Queue 1).

### 6.5 Statistical Validation Methodology (The 4-Gate Protocol)
1. **Replace Standalone KS Test:** Discard the continuous KS test $p > 0.05$ criterion. Deploy **Wasserstein-1 Distance** ($W_1 \le 35.0\text{ B}$ for packet size, $W_1 \le 15.0\text{ ms}$ for IAT), **Jensen-Shannon Divergence** ($\text{JSD} \le 0.05$), and **6-Bin Chi-Square** (Cramér's $V \le 0.10$).
2. **Welch Warm-Up Period:** Discard initial transient data over $t \in [0, 60\text{ s}]$; record steady-state performance over $[60\text{ s}, 360\text{ s}]$.
3. **Replication & CRN:** Mandate $M = 30$ independent runs using Common Random Numbers across comparative queueing policies (FIFO vs Priority Queuing).
4. **Non-Parametric Tail Inference:** Report 95% BCa Bootstrap Confidence Intervals for $P_{95}$, $P_{99}$, and packet loss rates; evaluate policy superiority via paired Wilcoxon signed-rank tests ($p < 0.01$) and Hedges' $g \ge 0.80$.

---

## 7. Final Audit Decision

### Status: **APPROVED WITH CONDITIONS**

Phase 7 network simulation implementation may proceed **only after** the following mandatory conditions are fully satisfied:

1. **Condition 1 (Mathematical & Equation Updates):** Update `research_notes/phase7/traffic_generation_design.md` with corrected formulas for RFC 8312 CUBIC $K$, $\beta_{cubic} = 0.7$, Jensen's inequality offset for lognormal video chunks, and absolute sampling clock scheduling for Real-Time audio.
2. **Condition 2 (Pseudo-Code Replacement):** Replace the pseudo-code implementations of `BulkDataFlow`, `StreamingFlow`, and `RealTimeFlow` in Section 6 with the bug-free, closed-loop reference implementations provided in `traffic_generation_corrections.md`.
3. **Condition 3 (Harmonization of Parameters):** Reconcile bitrate mismatches ($2.0\text{ Mbps}$ for Streaming) and clarify the Dominant Transport row in Table 2.1 to reflect sustained flow reality ($96.1\%$ TCP for Bulk Data).
4. **Condition 4 (Validation Protocol Adoption):** Incorporate the 4-Gate Statistical Validation Protocol (Wasserstein-1, JSD, 6-bin Chi-Square, Welch warm-up $T_{warm}=60\text{ s}$, $M=30$ runs with Common Random Numbers) into the experimental harness.
5. **Condition 5 (Verification of Classifier Compatibility):** Implement the `Early5FeatureExtractor` module to guarantee zero-leakage, unscaled feature extraction matching Phase 6B XGBoost input requirements.

*The corrected blueprints and final specification documents are authored in companion files: `traffic_generation_corrections.md` and `traffic_generation_final_specification.md`.*
