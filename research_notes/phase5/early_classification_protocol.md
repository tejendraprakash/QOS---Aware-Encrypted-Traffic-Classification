# Early-Stage Encrypted Traffic Classification Protocol

**Project:** QoS-Aware Encrypted Network Traffic Classification for Next-Generation Networks  
**Phase:** 5 — Baseline Experiment Preparation & Protocol Validation  
**Date:** November 2026  
**Protocol Artifact:** [early_feature_definitions.json](file:///c:/Users/saipr/OneDrive/Desktop/EncryptedTrafficResearch/output/phase5/early_feature_definitions.json)  
**Base Features:** [flow_features.csv](file:///c:/Users/saipr/OneDrive/Desktop/EncryptedTrafficResearch/output/flow_features.csv)

---

## 1. Network Engineering Motivation

In Next-Generation Networks (5G/6G, SDN, programmable data planes with P4), dynamic QoS management requires **sub-second policy enforcement**. Switches, routers, and edge firewalls cannot wait for an encrypted flow to terminate before assigning priority queues, reserving bandwidth slices, or shaping traffic:

```
[Flow Initiation: Packet 1] 
       │ 
       ▼
 [Window N=5 packets]  ──────► [Early Classifier] ───► Assign QoS Queue (e.g., Strict Priority for Real-Time)
       │
       ▼
[Window N=10 packets] ──────► [Confidence Refinement / Slice Reallocation]
       │
       ▼
[Window N=20 packets] ──────► [Final Steady-State QoS Policy]
       │
       ▼
[Flow Completion: 10,000+ packets] (Full-flow offline audit only; useless for real-time QoS)
```

Waiting for full-flow completion defeats the fundamental purpose of QoS-aware networking. Therefore, evaluating classifier accuracy at early packet horizons is a central contribution of this research.

---

## 2. Packet Windows & Feature Taxonomy

We evaluate early classification performance across four distinct early packet horizons: $N \in \{5, 10, 15, 20\}$ packets.

For each window $N$, the flow extractor computes exactly 6 behavioral metrics derived exclusively from packet timestamps and IP/TCP header lengths without payload decryption:

| Feature Name | Type | Unit | Description |
|---|---|---|---|
| `early_{N}_complete` | Boolean | $\{0, 1\}$ | Flag indicating whether the flow contained $\ge N$ total packets. |
| `early_{N}_duration` | Float | seconds | Elapsed time from the 1st packet to the $N$-th packet (or flow end if $< N$). |
| `early_{N}_bytes` | Integer | bytes | Total cumulative bytes transmitted in the first $N$ packets (bidirectional). |
| `early_{N}_fwd_pkts` | Integer | packets | Number of forward packets observed within the first $N$ packets. |
| `early_{N}_bwd_pkts` | Integer | packets | Number of backward packets observed within the first $N$ packets. |
| `early_{N}_avg_size` | Float | bytes | Average packet size across observed early packets ($\text{bytes} / \text{pkts}$). |

---

## 3. Cumulative Experimental Subsets

To assess classification accuracy as more packets become available, we formulate five standardized feature sets:

### Set 1: Early-5 (6 Features)
- **Features:** `early_5_complete`, `early_5_duration`, `early_5_bytes`, `early_5_fwd_pkts`, `early_5_bwd_pkts`, `early_5_avg_size`.
- **Target Operational Latency:** $< 5$ milliseconds.
- **Network Purpose:** Immediate packet queue steering (e.g., separating interactive audio bursts from bulk uploads).

### Set 2: Early-10 (12 Features)
- **Features:** Cumulative features of Window 5 + Window 10.
- **Target Operational Latency:** $10 - 25$ milliseconds.
- **Network Purpose:** Captures the full TCP 3-way handshake and the initial TLS ClientHello / ServerHello exchange.

### Set 3: Early-15 (18 Features)
- **Features:** Cumulative features of Windows 5, 10, and 15.
- **Target Operational Latency:** $25 - 50$ milliseconds.
- **Network Purpose:** Captures TLS cryptographic negotiation completion and initial application data framing.

### Set 4: Early-20 (24 Features)
- **Features:** Cumulative features of Windows 5, 10, 15, and 20.
- **Target Operational Latency:** $50 - 100$ milliseconds.
- **Network Purpose:** Upper boundary for early QoS classification before steady-state transport is established.

### Set 5: Full-Flow Baseline (78 Features)
- **Features:** Complete set of 78 non-constant full-flow statistical and temporal metrics.
- **Operational Latency:** Post-flow (offline).
- **Network Purpose:** Serves as the theoretical performance ceiling / benchmark reference.

---

## 4. Evaluation Protocol for Early Classification

When Phase 6 benchmarks run:
1. All three candidate models (Random Forest, XGBoost, P-MLP) will be trained and evaluated separately on each of the 5 feature sets: Early-5, Early-10, Early-15, Early-20, and Full-Flow.
2. Performance will be cross-validated across identical `StratifiedGroupKFold` splits.
3. Results will be plotted on an **Accuracy vs. Packets Curve** to determine the minimal packet threshold required to achieve $\ge 90\%$ Macro F1-score.
