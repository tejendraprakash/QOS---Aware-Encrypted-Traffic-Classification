# Advanced Research Lab (ARL) — Research Memory

**Project:** QoS-Aware Encrypted Network Traffic Classification for Next-Generation Networks  
**Lead Research Orchestrator:** AI Research Lab Manager  
**Last Updated:** November 2026  

---

## 1. Domain Discoveries & Empirical Insights

### 1.1 Dataset Properties & Survivorship Dynamics (`output/phase5/track_b_flows.csv`)
* **Total Population vs. Sustained Sessions:**
  The dataset contains 10,770 flows (8,132 Real-Time, 1,290 Streaming, 1,348 Bulk Data).
  Crucially, **91.16% of all flows are transient/ephemeral** (< 20 packets), consisting of DNS lookups, TCP handshakes, STUN/ICE NAT keepalives, and aborted sessions.
  Filtering for sustained flows ($\ge 20$ packets) yields 952 flows (377 Real-Time, 344 Streaming, 231 Bulk Data).
* **Real-Time Traffic Reality:**
  - Sustained VoIP exhibits a sharp packet rate mode at $50.34\text{ pkts/s}$ ($20\text{ ms}$ isochronous framing; ITU-T G.711 / RFC 6716 Opus).
  - Packet sizes are heavily concentrated at $80\text{--}100\text{ B}$ (Opus mode at $90\text{ B}$) and $200\text{ B}$ (G.711 PCM + RTP/UDP/IP).
  - Strictly symmetric: forward packet ratio is $0.50$, forward byte ratio is $0.48$.
* **Streaming Video Reality:**
  - Characterized by periodic chunk downloads ($T_{seg} = 3.0\text{ s}$) following adaptive bitrate (ABR) buffer dynamics.
  - Severe directional asymmetry: forward byte ratio median is $0.24$, 25th percentile is $0.05$ (meaning $\mathbf{76\%\text{--}95\%}$ of traffic volume is downlink).
  - Packet sizes are bimodal: MTU packets ($1,360\text{--}1,500\text{ B}$) on downlink, pure TCP ACKs ($52\text{ B}$) on uplink.
* **Bulk Data Reality:**
  - Sustained flows are **96.1% TCP** (contrasting with raw dataset where transient UDP probes skewed the count).
  - Transmits back-to-back MTU trains limited only by line rate and congestion window. Median bidirectional IAT is $0.000000\text{ s}$ ($0.0\text{ ms}$).

---

## 2. Mathematical Formulations & Physical Laws

### 2.1 RFC 8312 TCP CUBIC Congestion Control
* Deficit recovery time constant:
  $$K = \sqrt[3]{\frac{W_{max} \cdot (1 - \beta_{cubic})}{C}}$$
  where $C = 0.4\text{ packets/s}^3$ and $\beta_{cubic} = 0.7$ (multiplicative decrease factor).
* Window growth:
  $$W_{cubic}(t) = C(t - K)^3 + W_{max}$$
  Bounded below by standard TCP-friendly AIMD:
  $$W_{est}(t) = W_{max} \beta_{cubic} + \left[\frac{3(1 - \beta_{cubic})}{1 + \beta_{cubic}}\right] \frac{t}{RTT}$$
  $$cwnd = \max(W_{cubic}(t), W_{est}(t))$$

### 2.2 Unbiased Lognormal Video Sizing (Jensen's Inequality)
To sample video chunk sizes $S_k \sim \text{Lognormal}(\mu^*, \sigma^2)$ with exact desired mean $M = \frac{R_{video} T_{seg}}{8}$:
$$\mu^* = \ln(M) - \frac{\sigma^2}{2}$$
Prevents the $+3.17\%$ upward bias that occurs when setting $\mu = \ln(M)$.

### 2.3 Isochronous Sampling Clock Pacing (Zero Brownian Drift)
Anchoring packet generation to an absolute hardware audio clock:
$$t_k = t_{start} + k \cdot \tau + \delta_k$$
where $\tau = 0.020\text{ s}$ ($50\text{ Hz}$) and $\delta_k \sim \text{Uniform}(-0.3\text{ ms}, +0.3\text{ ms})$, maintaining zero cumulative drift ($\text{Var}(t_k) = \text{constant}$).

---

## 3. Methodological & Experimental Discoveries

### 3.1 The Large-Sample $p$-Value Paradox on Discrete Distributions
The classical continuous Two-Sample Kolmogorov-Smirnov (KS) test is invalid for discrete, multimodal packet distributions (e.g. 52B, 90B, 200B, 1500B). Because standard error scales as $1/\sqrt{N}$, sample sizes $N \ge 20,000$ packets guarantee rejection ($p \ll 0.001$) under minor protocol variations.
* *Solution:* Deploy **Wasserstein-1 Distance (Earth Mover's Distance)** ($W_1 \le 35\text{ B}$), **Jensen-Shannon Divergence** ($\text{JSD} \le 0.05$), and **6-Bin Chi-Square** (Cramér's $V \le 0.10$).

### 3.2 Transient Handling in Queueing Simulation (Welch's Method)
Simulating queueing systems from empty state ($Q(0) = 0$) causes severe downward bias in delay and drop rates during the initial buffer filling transient.
* *Solution:* Discard initial $60.0\text{ s}$ warm-up period ($T_{warm} = 60\text{ s}$); measure steady-state metrics over $[60\text{ s}, 360\text{ s}]$.

---

## 4. Machine Learning & Classifier Interface

* **Candidate Model:** Phase 6B Early-5 XGBoost classifier (`output/phase6b/early_5/`).
* **Input Schema:** 6 unscaled physical features (`early_5_complete`, `early_5_duration`, `early_5_bytes`, `early_5_fwd_pkts`, `early_5_bwd_pkts`, `early_5_avg_size`).
* **Forward Direction Invariant:** Strictly established by the source IP and port of the very first packet observed ($t_0$).
* **Model Class Mapping:** $0 \to \text{Bulk Data}$, $1 \to \text{Real-Time}$, $2 \to \text{Streaming}$.
* **Queue Mapping:** Real-Time $\to$ Queue 0 (Strict Priority), Streaming $\to$ Queue 1 (DRR 60%), Bulk Data $\to$ Queue 2 (DRR 40%). Exploratory packets 1–4 are held in Quarantined Queue 1.
