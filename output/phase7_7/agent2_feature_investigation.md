# Phase 7.7: Early-5 Feature Representation Investigation
**Agent Role:** SUBAGENT 2 — EARLY-5 FEATURE INVESTIGATOR  
**Date:** November 24, 2026  
**Status:** Complete  
**Scope:** Rigorous investigation of candidate causal features available strictly within the first five packets ($k \le 5$) for QoS-aware encrypted network traffic classification.  

---

## 1. Executive Summary & Core Verdict

The objective of Phase 7.7 Subagent 2 is to evaluate feature representation improvements for the Early-5 classification regime. Under strict operational constraints, early QoS routing decisions must be executed **either upon the arrival of the 5th packet ($t_5$) or upon flow termination if the flow contains fewer than five packets ($k < 5$)**.

### Key Findings:
1. **Current Benchmark State:**
   - The primary dataset (`output/phase6d/clean_track_b_flows.csv`, $N=4,196$ flows across 29 capture groups) contains **only 6 features** explicitly computed for the 5-packet horizon:
     $$\mathcal{F}_{\text{base}} = \{\text{early\_5\_complete}, \text{early\_5\_duration}, \text{early\_5\_bytes}, \text{early\_5\_fwd\_pkts}, \text{early\_5\_bwd\_pkts}, \text{early\_5\_avg\_size}\}$$
   - Baseline performance under strict 29-Fold Leave-One-Group-Out (LOGO) cross-validation is **Macro $F_1 = 0.5734$**, Balanced Accuracy = $0.5881$.
   - Under 5-Fold StratifiedGroupKFold (Seed 157), Baseline performance is **Macro $F_1 = 0.6591$**, Balanced Accuracy = $0.6883$.

2. **Feasibility Stratification:**
   - **Immediately Derivable Features (Category A):** Features that can be computed directly from `clean_track_b_flows.csv` without touching raw PCAPs or breaking dataset integrity. These include:
     * Layer 4 protocol flag: `is_udp` ($\mathbb{I}(\text{protocol} == 17)$), available from the first packet's IP header.
     * Early forward packet ratio: `early_5_fwd_pkt_ratio` ($k_{\text{fwd}} / (k_{\text{fwd}} + k_{\text{bwd}})$).
     * Early mean inter-arrival time: `early_5_iat_mean` ($\text{duration} / (k - 1)$, by telescoping sum property).
     * Early packet and byte rates (`pkt_rate`, `byte_rate`, `log_byte_rate`).
   - **Re-extraction Required Features (Category B):** Fine-grained sequence and dispersion features that require re-parsing the 37 raw PCAP captures ($14.17\text{ GB}$). These include individual packet sizes $(p_1 \dots p_5)$, direction sequence $(d_1 \dots d_5)$, directional bytes ($B_{\text{fwd}}, B_{\text{bwd}}$), packet-size dispersion ($\sigma_p, p_{\min}, p_{\max}$), and IAT dispersion ($\sigma_{\text{IAT}}, \text{IAT}_{\max}$).
   - **Critical Leakage Pitfalls (Category C):** Full-flow columns in `clean_track_b_flows.csv` (e.g., `fwd_bytes`, `bidir_pkt_size_min`, `bidir_iat_mean`, `tcp_ack_count`) are aggregated over the **entire session** ($10$ to $10,000+$ packets). Using these columns for the Early-5 classifier represents catastrophic temporal leakage.

3. **Empirical Breakthrough on Immediately Derivable Features:**
   - Systematically testing each candidate addition under 29-fold LOGO cross-validation revealed that **adding `is_udp` and `early_5_fwd_pkt_ratio` to the baseline improves Macro $F_1$ from $0.5734 \longrightarrow 0.5899$ ($+0.0165$) and Balanced Accuracy from $0.5881 \longrightarrow 0.6060$ ($+0.0179$)**.
   - Crucially, **all three QoS classes improve simultaneously**:
     * Bulk Data Recall: $50.21\% \longrightarrow 50.89\%$ ($F_1: 0.5395 \longrightarrow 0.5445$)
     * Real-Time Recall: $56.55\% \longrightarrow 0.5834\%$ ($F_1: 0.5849 \longrightarrow 0.6049$)
     * Streaming Recall: $69.68\% \longrightarrow 72.58\%$ ($F_1: 0.5958 \longrightarrow 0.6203$)
   - Under 5-fold StratifiedGroupKFold, Macro $F_1$ increases from $0.6591 \longrightarrow 0.6632$.
   - **Why Earlier Experiments Failed:** In Phase 7.6, Experiment 1 bundled `is_udp` and `fwd_ratio` together with `early_5_byte_rate`, causing Macro $F_1$ to drop ($0.5635$). Our feature-by-feature ablation isolates the root cause: **temporal rate features (`byte_rate`, `pkt_rate`, `iat_mean`) overfit to the capture environment bandwidth and link RTT**, causing severe negative transfer across capture groups. Omission of rate features unleashes the true predictive power of structural protocol features.

---

## 2. Anatomy of Early-5 Traffic in the Benchmark

Understanding the physical structure of the flows is essential before defining candidate features.

### 2.1 Packet Distribution Across the Horizon
Analyzing `output/phase6d/clean_track_b_flows.csv` reveals the exact distribution of early observed packet count $k = \min(\text{total\_packets}, 5)$:

| Observed Early Packets ($k$) | Flow Count ($N$) | Percentage | Flow State at Decision |
| :---: | :---: | :---: | :--- |
| **$k = 2$** | $1,778$ | $42.37\%$ | Terminated (TCP FIN/RST or Inactivity Timeout) |
| **$k = 3$** | $274$ | $6.53\%$ | Terminated (TCP FIN/RST or Inactivity Timeout) |
| **$k = 4$** | $283$ | $6.74\%$ | Terminated (TCP FIN/RST or Inactivity Timeout) |
| **$k = 5$** | $1,861$ | $44.35\%$ | Active flow reaching threshold trigger ($t_5$) |
| **Total** | **$4,196$** | **$100.0\%$** | |

### 2.2 Decision-Time Realities
- **Short Flows ($55.65\%$ of dataset):** $2,335$ flows terminate before the 5th packet arrives. For these flows, `early_5_complete = 0`, and the classifier triggers upon flow termination. All packets of the flow are already observed; $k$ represents the complete flow size.
- **Persistent Flows ($44.35\%$ of dataset):** $1,861$ flows reach $\ge 5$ packets. For these flows, `early_5_complete = 1`, and the classification decision triggers at the exact arrival timestamp $t_5$. All subsequent packets ($k > 5$) are strictly in the future.
- **Zero-Packet or Single-Packet Flows:** There are **zero** 1-packet flows in `clean_track_b_flows.csv` (the minimum packet threshold in `flow_extractor.py` was set to `min_packets=2`). Hence, $k \ge 2$ for all flows.
- **Duration Distribution:** $\text{early\_5\_duration} > 0$ for **100% of flows** (minimum observed duration is $1.9 \times 10^{-5}\text{ s} = 19\ \mu\text{s}$; median is $0.1356\text{ s}$; maximum is $128.44\text{ s}$).

---

## 3. Comprehensive Evaluation of Candidate Features

We evaluate all 9 candidate feature families requested in the specification against the strict criteria of causal availability, mathematical precision, numerical safety, feasibility, and empirical predictive validity.

```
+-----------------------------------------------------------------------------------------------------------------------------+
| Candidate Feature Family              | Mathematical Definition                     | Availability | Leakage Risk | Feasibility |
+-----------------------------------------------------------------------------------------------------------------------------+
| 1. Packet Size Sequence               | [p1, p2, p3, p4, p5], zero-padded           | At pkt 1..5  | Zero         | Raw PCAP    |
| 2. Direction Sequence / Signed Sizes  | [d1, d2, d3, d4, d5], s_i = d_i * p_i       | At pkt 1..5  | Zero         | Raw PCAP    |
| 3. Directional Counts & Bytes         | k_fwd, k_bwd, B_fwd, B_bwd                  | At pkt 1..5  | High if CSV* | Mixed       |
| 4. Directional Ratios                 | r_pkt = k_fwd / k, r_byte = B_fwd / B       | At pkt 1..5  | Zero         | Mixed       |
| 5. Packet Size Dispersion             | min(p), max(p), range(p), std(p)            | At pkt 1..5  | High if CSV* | Raw PCAP    |
| 6. Inter-Arrival Times (IAT)          | mean(IAT), min(IAT), max(IAT), std(IAT)     | At pkt 1..5  | High if CSV* | Mixed       |
| 7. Directional Size Statistics        | mean_fwd_size, mean_bwd_size                | At pkt 1..5  | High if CSV* | Raw PCAP    |
| 8. Early Flow Rates                   | byte_rate, pkt_rate, log_byte_rate          | At pkt 1..5  | Env Overfit  | Immediate   |
| 9. Layer 4 Protocol Flag              | is_udp = I(proto == 17)                     | Packet 1     | Zero         | Immediate   |
+-----------------------------------------------------------------------------------------------------------------------------+
*Note: High leakage risk occurs if naive users pull full-flow columns (fwd_bytes, bidir_pkt_size_min, bidir_iat_mean) from the existing CSV.
```

---

## 4. Deep-Dive Specification for Each Candidate

### Candidate 1: First-Five Packet Size Sequence $(p_1, p_2, p_3, p_4, p_5)$
- **Mathematical Formula:**
  Let $\mathbf{p} = [p_1, p_2, p_3, p_4, p_5]^T \in \mathbb{R}^5_{\ge 0}$, where $p_i$ is the IP total length (in bytes) of the $i$-th observed packet.
  For flows terminating at $k < 5$:
  $$p_i = \begin{cases} \text{TotalLength}(i\text{-th packet}), & 1 \le i \le k \\ 0, & k < i \le 5 \end{cases}$$
- **Decision-Time Availability:** Strictly available. $p_i$ is read directly from the IPv4/IPv6 header as each packet arrives.
- **Leakage & Edge Cases:**
  * Zero future leakage.
  * Minimum valid IP packet size is 20 bytes (IPv4) or 40 bytes (IPv6). Therefore, $p_i = 0$ is an unambiguous padding indicator that cannot collide with a valid packet.
- **Feasibility:** **Category B (Requires PCAP Re-extraction).** `clean_track_b_flows.csv` stores only $\sum_{i=1}^k p_i$ (`early_5_bytes`) and the average (`early_5_avg_size`). Recovering individual packet sizes from the scalar sum is underdetermined.
- **Domain Value:** **Extremely High.** In encrypted protocols, $p_1$ is typically the TCP SYN (52-60 B) or TLS Client Hello (~200-500 B), $p_2$ is TCP SYN-ACK or Server Hello, and $p_4/p_5$ represent initial key exchange or certificate lengths. This distinguishes TLS handshakes (Streaming/Web) from RTP/UDP audio frames (Real-Time).

---

### Candidate 2: Packet Direction Sequence $(d_1, \dots, d_5)$ & Signed Sizes $(s_1, \dots, s_5)$
- **Mathematical Formula:**
  Let $d_i \in \{+1, -1\}$ indicate packet direction relative to the flow initiator ($d_1 = +1$ by definition).
  $$d_i = \begin{cases} +1, & \text{Forward (Client } \to \text{ Server)} \\ -1, & \text{Backward (Server } \to \text{ Client)} \\ 0, & i > k \text{ (Padding)} \end{cases}$$
  Signed packet size representation:
  $$s_i = d_i \cdot p_i \quad \text{for } i \in \{1, \dots, 5\}$$
- **Decision-Time Availability:** Strictly available at packet arrival.
- **Leakage & Edge Cases:**
  * Zero leakage. $d_1$ has zero variance (always $+1$), so $d_1$ can be omitted in fixed models, while $d_2 \dots d_5$ provide the client-server interaction topology.
- **Feasibility:** **Category B (Requires PCAP Re-extraction).** The CSV records total counts (`early_5_fwd_pkts`, `early_5_bwd_pkts`) but does not record the sequence order (e.g., F-F-B vs F-B-F).
- **Domain Value:** **High.** TCP interactive sessions follow a strict F-B-F-B handshake, whereas bulk uploads follow F-F-F bursts and unidirectional UDP audio streaming exhibits long runs of same-direction packets.

---

### Candidate 3: Forward & Backward Packet and Byte Counts
- **Mathematical Formula:**
  - Forward Packet Count: $k_{\text{fwd}} = \sum_{i=1}^k \mathbb{I}(d_i = \text{Forward})$
  - Backward Packet Count: $k_{\text{bwd}} = \sum_{i=1}^k \mathbb{I}(d_i = \text{Backward})$
  - Forward Byte Count: $B_{\text{fwd}} = \sum_{i=1}^k p_i \cdot \mathbb{I}(d_i = \text{Forward})$
  - Backward Byte Count: $B_{\text{bwd}} = \sum_{i=1}^k p_i \cdot \mathbb{I}(d_i = \text{Backward})$
- **Decision-Time Availability:** Strictly available at packet $k$.
- **Leakage & Edge Cases:**
  * **CRITICAL LEAKAGE WARNING:** `clean_track_b_flows.csv` contains columns named `fwd_bytes` and `bwd_bytes`. These represent the **complete flow byte totals** across the entire capture (up to tens of megabytes). Using them for Early-5 is a severe leakage bug.
- **Feasibility:**
  - $k_{\text{fwd}}$ and $k_{\text{bwd}}$ are **ALREADY IN CSV** (`early_5_fwd_pkts`, `early_5_bwd_pkts`).
  - $B_{\text{fwd}}$ and $B_{\text{bwd}}$ are **Category B (Requires Re-extraction)**.

---

### Candidate 4: Directional Ratios within Early-5
- **Mathematical Formula:**
  - Early Forward Packet Ratio:
    $$r_{\text{fwd\_pkt}} = \frac{k_{\text{fwd}}}{\max(k_{\text{fwd}} + k_{\text{bwd}}, 1)}$$
  - Early Forward Byte Ratio:
    $$r_{\text{fwd\_byte}} = \frac{B_{\text{fwd}}}{\max(B_{\text{fwd}} + B_{\text{bwd}}, 1)}$$
- **Decision-Time Availability:** Strictly available at decision time.
- **Leakage & Edge Cases:**
  * Guarding against division by zero via $\max(k, 1)$. In our dataset, $k \ge 2$, so $k$ is always strictly positive.
  * Bounded strictly to $[0.0, 1.0]$. Bounded ratios avoid the numerical explosion of odds ratios ($k_{\text{fwd}} / k_{\text{bwd}}$ when $k_{\text{bwd}} = 0$).
- **Feasibility:**
  - $r_{\text{fwd\_pkt}}$ is **Category A (100% Immediately Derivable)**:
    `df['early_5_fwd_ratio'] = df['early_5_fwd_pkts'] / (df['early_5_fwd_pkts'] + df['early_5_bwd_pkts'])`
  - $r_{\text{fwd\_byte}}$ is **Category B** (requires $B_{\text{fwd}}$ from re-extraction).
- **Domain Value:** **Very High.** Real-Time conversational audio tends to be symmetric ($r_{\text{fwd\_pkt}} \approx 0.5$), whereas Bulk Data initiation exhibits client request push ($r_{\text{fwd\_pkt}} \ge 0.60$).

---

### Candidate 5: Packet-Size Distribution Statistics within Early-5
- **Mathematical Formula:**
  For $p_1, \dots, p_k$:
  - Mean: $\mu_p = \frac{1}{k} \sum_{i=1}^k p_i$ (`early_5_avg_size`)
  - Variance & Std: $\sigma_p = \sqrt{\frac{1}{k} \sum_{i=1}^k (p_i - \mu_p)^2}$
  - Minimum Size: $p_{\min} = \min_{1 \le i \le k} p_i$
  - Maximum Size: $p_{\max} = \max_{1 \le i \le k} p_i$
  - Size Range: $\Delta p = p_{\max} - p_{\min}$
- **Decision-Time Availability:** Strictly available at packet $k$.
- **Leakage & Edge Cases:**
  * Full-flow columns `bidir_pkt_size_min`, `bidir_pkt_size_max`, `bidir_pkt_size_std` exist in the CSV and must NOT be used.
  * For $k < 2$, $\sigma_p = 0$. For identical packet sizes (e.g., pure ACK flood), $\sigma_p = 0, \Delta p = 0$.
- **Feasibility:** **Category B (Requires Re-extraction).** Only $\mu_p$ is currently stored in `clean_track_b_flows.csv`.

---

### Candidate 6: Inter-Arrival Times (IAT) within Early-5
- **Mathematical Formula:**
  For timestamps $t_1, \dots, t_k$:
  $$\text{IAT}_i = t_{i+1} - t_i \ge 0 \quad \text{for } i \in \{1, \dots, k-1\}$$
  Number of intervals: $M = k - 1 \in \{1, 2, 3, 4\}$.
  - **Telescoping Mean Proof:**
    $$\sum_{i=1}^{k-1} \text{IAT}_i = (t_2 - t_1) + (t_3 - t_2) + \dots + (t_k - t_{k-1}) = t_k - t_1 \equiv \text{early\_5\_duration}$$
    Therefore, the mean inter-arrival time is analytically:
    $$\mu_{\text{IAT}} = \frac{\text{early\_5\_duration}}{k - 1}$$
- **Decision-Time Availability:** Strictly available at $t_k$.
- **Leakage & Edge Cases:**
  * Division by zero: guarded as $\max(k - 1, 1)$. Because all flows have $k \ge 2$, $k - 1 \ge 1$ everywhere.
- **Feasibility:**
  - $\mu_{\text{IAT}}$ is **Category A (100% Derivable)** from existing columns.
  - $\text{IAT}_{\min}$, $\text{IAT}_{\max}$, and $\sigma_{\text{IAT}}$ are **Category B (Requires Re-extraction)**.
- **Empirical Warning:** Adding $\mu_{\text{IAT}}$ directly reduced 29-fold LOGO Macro $F_1$ from $0.5734 \longrightarrow 0.5572$ ($-0.0162$). Because $\mu_{\text{IAT}}$ is linearly proportional to duration, it acts as an environmental proxy rather than a causal application signature.

---

### Candidate 7: Directional Packet-Size Statistics
- **Mathematical Formula:**
  - Forward Mean Size: $\mu_{\text{fwd\_size}} = \frac{B_{\text{fwd}}}{\max(k_{\text{fwd}}, 1)}$
  - Backward Mean Size: $\mu_{\text{bwd\_size}} = \frac{B_{\text{bwd}}}{\max(k_{\text{bwd}}, 1)}$
  - Directional Size Delta: $\Delta \mu_{\text{dir}} = \mu_{\text{fwd\_size}} - \mu_{\text{bwd\_size}}$
- **Decision-Time Availability:** Strictly available at packet $k$.
- **Leakage & Edge Cases:**
  * When $k_{\text{bwd}} = 0$, backward size is undefined; should be imputed to $0$ with a binary indicator `has_bwd_packets`.
- **Feasibility:** **Category B (Requires Re-extraction).**

---

### Candidate 8: Early Flow Rate Calculations
- **Mathematical Formula:**
  - Early Byte Rate:
    $$\text{early\_5\_byte\_rate} = \frac{\text{early\_5\_bytes}}{\max(\text{early\_5\_duration}, \epsilon)}$$
  - Early Packet Rate:
    $$\text{early\_5\_pkt\_rate} = \frac{k}{\max(\text{early\_5\_duration}, \epsilon)}$$
  - Log Transformations:
    $$\text{log\_early\_5\_byte\_rate} = \ln(1 + \text{early\_5\_byte\_rate})$$
- **Decision-Time Availability:** Strictly available at $t_k$.
- **Leakage & Edge Cases:**
  * Safe with $\epsilon = 10^{-6}\text{ s}$. Duration is non-zero across all flows.
  * Byte rate spans 7 orders of magnitude ($2.2\text{ B/s}$ to $143.9\text{ MB/s}$), creating extreme outliers in linear space.
- **Feasibility:** **Category A (100% Derivable).**
- **Empirical Warning:** Adding `early_5_byte_rate` degraded 29-fold LOGO Macro $F_1$ from $0.5734 \longrightarrow 0.5616$ ($-0.0118$); `log_byte_rate` gave $0.5625$ ($-0.0109$). Rates reflect local capture environment bottleneck bandwidth and cross-host RTT, failing to generalize to unseen networks.

---

### Candidate 9: Layer 4 Protocol Flag (`is_udp` from IP Header)
- **Mathematical Formula:**
  $$\text{is\_udp} = \mathbb{I}(\text{protocol} == 17)$$
  Alternatively, one-hot encoding for $\{\text{TCP: } 6, \text{UDP: } 17\}$.
- **Decision-Time Availability:** **Available at Packet 1!** Read from octet 9 of the IPv4 header or Next Header field of IPv6.
- **Leakage & Edge Cases:** Zero leakage. Completely payload-agnostic header metadata. No division, no missing values.
- **Feasibility:** **Category A (Already Present in CSV).** Present as `is_udp` and `protocol` in `clean_track_b_flows.csv`.
- **Domain Value:** **Decisive.** Real-Time traffic (audio/video calls via RTP/WebRTC) frequently operates over UDP, whereas Bulk Data (FTPS, SCP, HTTP download) overwhelmingly uses TCP. In our dataset, Real-Time contains substantial UDP traffic, whereas Bulk Data is $0\%$ UDP. Adding `is_udp` alone boosts Macro $F_1$ by $+0.0120$ under LOGO.

---

## 5. Empirical Verification: Feature Addition Matrix

To provide concrete empirical validation, we tested all Category A candidates on `clean_track_b_flows.csv` across the **full 29-Fold Leave-One-Group-Out (LOGO)** cross-validation protocol using standard class weighting on the verified XGBoost architecture:

| Configuration | Features Used | Dim | LOGO Macro $F_1$ | $\Delta F_1$ | Balanced Acc | Bulk Rec | RT Rec | Streaming Rec | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Baseline Anchor** | Baseline 6 features | 6 | **0.5734** | $0.0000$ | 0.5881 | 0.5021 | 0.5655 | 0.6968 | Benchmark Anchor |
| **+ fwd_pkt_ratio** | Base + `early_5_fwd_ratio` | 7 | **0.5743** | $+0.0009$ | 0.5891 | 0.5042 | 0.5670 | 0.6960 | Positive |
| **+ iat_mean** | Base + `early_5_iat_mean` | 7 | **0.5572** | $-0.0162$ | 0.5752 | 0.4852 | 0.5523 | 0.6881 | Negative (Overfits) |
| **+ pkt_rate** | Base + `early_5_pkt_rate` | 7 | **0.5673** | $-0.0061$ | 0.5829 | 0.4903 | 0.5601 | 0.6983 | Negative (Overfits) |
| **+ byte_rate** | Base + `early_5_byte_rate` | 7 | **0.5616** | $-0.0118$ | 0.5752 | 0.4877 | 0.5580 | 0.6800 | Negative (Overfits) |
| **+ log_byte_rate** | Base + `log_early_5_byte_rate` | 7 | **0.5625** | $-0.0109$ | 0.5771 | 0.4920 | 0.5589 | 0.6804 | Negative (Overfits) |
| **+ is_udp** | Base + `is_udp` | 7 | **0.5854** | $+0.0120$ | 0.6027 | 0.5089 | 0.5768 | 0.7224 | **Strong Positive** |
| **Champion (8 Feats)** | Base + `is_udp` + `fwd_pkt_ratio` | **8** | **0.5899** | **$+0.0165$** | **0.6060** | **0.5089** | **0.5834** | **0.7258** | **WINNER (All Classes $\uparrow$)** |
| **Champion + Rates** | Base + `is_udp` + `fwd_ratio` + `byte_rate` | 9 | **0.5635** | $-0.0099$ | 0.5767 | 0.4945 | 0.5589 | 0.6767 | Degraded (Phase 7.6 Exp 1) |

### Performance under 5-Fold StratifiedGroupKFold (Seed 157):
- Baseline (6 features): Macro $F_1 = 0.6591$ | Balanced Accuracy = $0.6883$
- **Champion 8 Features (`+ is_udp, fwd_pkt_ratio`): Macro $F_1 = 0.6632$ ($+0.0042$) | Balanced Accuracy = $0.6918$**

---

## 6. Scientific Insight: Why Rates Fail and Protocol Features Succeed

1. **The Invariance Principle in Cross-Group Generalization:**
   In Leave-One-Group-Out validation, the test group was recorded in a completely different network environment (different router, Wi-Fi vs Ethernet, RTT latency, ISP bottleneck).
   - Packet and byte rates ($\frac{\text{bytes}}{\Delta t}$) are functions of the **physical link capacity and RTT**, which change drastically between groups. When a tree model splits on `early_5_byte_rate > 50,000 B/s`, it memorizes that capture group GRP_A was on high-speed LAN, failing completely when evaluating capture group GRP_B recorded over DSL or a throttled VPN tunnel.
   - Conversely, **`is_udp`** is an invariant protocol rule: DNS and Real-Time RTP voice frames use UDP regardless of link speed.
   - Similarly, **`fwd_pkt_ratio`** captures the dialogic balance (request-response vs bulk transmission), which is governed by application protocol semantics rather than clock speed.

---

## 7. Concrete Ranked Recommendations

We provide a three-tiered recommendation matrix for Phase 7.7 and subsequent phases:

### Tier 1: Immediately Deployable Candidate (No Re-extraction Required)
- **Recommended Feature Set:** **The Robust 8-Feature Early-5 Vector**
  $$\mathcal{F}_{\text{early5\_v2}} = \{\text{early\_5\_complete}, \text{early\_5\_duration}, \text{early\_5\_bytes}, \text{early\_5\_fwd\_pkts}, \text{early\_5\_bwd\_pkts}, \text{early\_5\_avg\_size}, \mathbf{is\_udp}, \mathbf{early\_5\_fwd\_ratio}\}$$
- **Formulation:**
  ```python
  total_early_pkts = df["early_5_fwd_pkts"] + df["early_5_bwd_pkts"]
  df["early_5_fwd_ratio"] = df["early_5_fwd_pkts"] / np.maximum(total_early_pkts, 1)
  df["is_udp"] = (df["protocol"] == 17).astype(float)
  ```
- **Expected Impact:**
  - 29-Fold LOGO Macro $F_1$: $+0.0165$ ($0.5734 \longrightarrow 0.5899$)
  - 29-Fold LOGO Balanced Acc: $+0.0179$ ($0.5881 \longrightarrow 0.6060$)
  - 5-Fold SGKF Macro $F_1$: $+0.0042$ ($0.6591 \longrightarrow 0.6632$)
  - Zero modifications required to underlying PCAP extraction files.

### Tier 2: Priority Feature Additions for Extended PCAP Re-extraction (Phase 8)
If the project schedules a full PCAP re-extraction pass ($14.17\text{ GB}$), the following 4 features should be prioritized in `flow_extractor.py`:
1. **$p_1, p_2, p_3, p_4, p_5$ (First 5 Packet Sizes, zero-padded):** Directly separates TLS handshake records (Streaming) from small fixed RTP frames (Real-Time).
2. **$s_1, s_2, s_3, s_4, s_5$ (Signed Packet Sizes $d_i \cdot p_i$):** Encodes both size and direction in a compact 5-dimensional vector.
3. **$\text{early\_5\_fwd\_byte\_ratio}$ ($B_{\text{fwd}} / \text{early\_5\_bytes}$):** Distinguishes bulk data uploads from downloads.
4. **$\text{early\_5\_pkt\_size\_std}$ ($\sigma_p$ across first 5 packets):** High variance indicates mixed control/data exchanges; near-zero variance indicates fixed-frame audio streaming.

### Tier 3: Deprecated / Prohibited Features (Do Not Use)
1. **Full-Flow Columns:** `fwd_bytes`, `bidir_pkt_size_*`, `bidir_iat_*`, `tcp_*_count`. Severe temporal leakage if applied at Early-5.
2. **Raw Rate Features:** `early_5_byte_rate`, `early_5_pkt_rate`, `early_5_iat_mean`. Proven to cause cross-group environmental overfitting and negative transfer under LOGO.
3. **Odds Ratios:** `fwd_to_bwd_pkt_ratio`. Susceptible to division-by-zero instability when flows have zero backward packets.

---

## 8. Guidance for Peer Agents & Coordinator

- **For Agent 3 (Model Family & Hyperparameter Tuner):**
  Evaluate model families (XGBoost, HistGradientBoosting, Random Forest, Calibrated Classifier) on the **Robust 8-Feature Vector** ($\mathcal{F}_{\text{early5\_v2}}$). Testing hyperparameter regularizations on the 8-feature representation will provide higher gains than tuning on the constrained 6-feature baseline.
- **For Agent 4 (Scientific Auditor):**
  Confirm that `is_udp` and `early_5_fwd_ratio` strictly adhere to causal decision-time availability (packet 1 and packets $1 \dots k$ respectively). Verify that no full-flow features are accessed.
- **For Coordinator:**
  Adopt the Robust 8-Feature Vector as the primary feature representation candidate for Phase 7.7.
