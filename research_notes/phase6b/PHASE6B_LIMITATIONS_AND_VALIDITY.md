# Phase 6B Scientific Limitations & Threats to Validity

**Project:** QoS-Aware Encrypted Network Traffic Classification for Next-Generation Networks  
**Phase:** 6B — Parallel Multi-Agent Early Traffic Classification  
**Date:** November 2026

---

## 1. Threats to Internal Validity

### 1.1 Decoupling of Packet Horizon from Protocol State
A core finding from our independent audit is that **packet count does not strictly correspond to cryptographic handshake state**:
- In TLS 1.2, handshakes require 10–12 packets; in TLS 1.3, they require 7–8 packets (or fewer with 0-RTT PSK).
- When a server certificate chain is large, it fragments into 3–6 TCP segments, consuming packets without advancing application-layer exchange.
- TCP retransmissions, duplicate ACKs, or window probes advance packet counts without advancing session state.
- Therefore, at $N=5$, some flows are still in the TCP 3-way handshake, while other flows (e.g. UDP VoIP) have already exchanged application frames.

### 1.2 Decoupling of Packet Horizon from Wall-Clock Latency
- Packet arrival time is governed by Round-Trip Time (RTT) and network jitter. In our dataset, the duration to observe the first 5 packets ranges from **41 microseconds to 156.15 seconds** (median 0.428s, 95th percentile 90.18s).
- For short interactive audio bursts, 5 packets arrive in milliseconds; for idle TCP connections, 5 packets can take tens of seconds.
- Consequently, an $N=5$ packet classifier does not guarantee sub-second classification across all network scenarios.

---

## 2. Threats to External Validity

### 2.1 Single-Testbed Homogeneity
- All captures originate from the UNB ISCX testbed collected in 2015. While multi-capture grouping isolates individual recording sessions, the underlying client/server machines, operating systems, and network topology share common MTU (1500 or 1420 bytes) and TCP stack parameters.
- Evaluating performance on live carrier-grade ISP networks with thousands of diverse mobile and desktop operating systems requires multi-dataset cross-validation.

### 2.2 Absence of Modern Encrypted Protocols
- Modern encrypted traffic is increasingly dominated by **QUIC (HTTP/3)**, **Encrypted Client Hello (ECH)**, and **WireGuard**.
- QUIC mandates client initial packet padding to $\ge 1200$ bytes and encrypts packet numbers, which blurs early packet size heuristics. Future work must benchmark these architectures on modern QUIC/ECH datasets.

---

## 3. Threats to Construct Validity

### 3.1 Classification Performance vs. Actual QoS Enforcement
- Achieving a high Macro F1 score on an offline dataset **does not directly prove QoS improvement in a live network**.
- Actual QoS benefits (reduced latency, bounded jitter, prioritized throughput, zero queue drops) depend on:
  1. Queuing discipline (e.g., Strict Priority vs. Deficit Round Robin).
  2. Buffer sizing and head-of-line blocking while waiting for $N$ packets.
  3. Cost of misclassification (e.g., misclassifying Bulk Data as Real-Time causes bandwidth starvation for genuine voice calls).
- Phase 6B is strictly a classification feasibility experiment; dynamic scheduling evaluation is a separate future phase.

---

## 4. Threats to Statistical Conclusion Validity

### 4.1 Sample Size Imbalance Across Folds (Simpson's Paradox)
- In Track B 5-fold cross-validation, Fold 2 contains **4,257 flows (39.5% of total dataset)** and is 95.8% Real-Time. In contrast, Fold 4 contains **1,261 flows (11.7%)** and is 77.0% Bulk Data.
- Unweighted fold averaging weights each fold as $1/5 = 20\%$, artificially penalizing models that perform poorly on small, difficult folds. Pooled Out-Of-Fold (OOF) evaluation evaluates all flows on an equal footing and provides a more faithful measure of population performance.
