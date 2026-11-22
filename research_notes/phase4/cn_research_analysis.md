# Agent A: Computer Networks Research Analysis & Technical Audit

**Project Title:** QoS-Aware Encrypted Traffic Classification for Next-Generation Networks  
**Author / Persona:** Agent A — Computer Networks Research Analyzer (`.agents/agents/cn-research-analyzer.md`)  
**Audit Stage:** Phase 4 — Dataset Compatibility & Technical Soundness  
**Date:** November 2026  

---

## 1. Technical Evaluation of the Four Target QoS Classes

Our research establishes four primary Quality of Service (QoS) classes:
1. **Bulk Data**
2. **Browsing**
3. **Streaming**
4. **Real-Time**

### 1.1 Alignment with IETF, ITU-T, and 3GPP QoS Specifications
To evaluate whether these four classes are technically sound, we benchmark them against standardized telecommunication traffic contracts:

| Target QoS Class | Standard Mappings | One-Way Delay ($\Delta t$) | Jitter ($\sigma_{\text{IAT}}$) | Packet Loss Rate ($P_L$) | Router Queuing & Scheduling Discipline |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Real-Time** | 3GPP 5QI 1/2/7; ITU-T Y.1541 Class 0/1; IETF RFC 3246 (Expedited Forwarding - EF) | $\le 100\text{--}150\text{ ms}$ | $\le 30\text{ ms}$ | $\le 10^{-3}$ ($0.1\%$) | **Strict Priority Queueing (PQ)**; immediate dispatch; dedicated low-latency bandwidth reservation. |
| **Streaming** | 3GPP 5QI 4/6/8; ITU-T Y.1541 Class 2; IETF RFC 2597 (Assured Forwarding 3 - AF3x) | $\le 2\text{--}4\text{ s}$ (Initial buffer) | Delay-tolerant via playback buffer | $\le 10^{-5}$ (TCP) / $\le 1\%$ (UDP) | **Class-Based Weighted Fair Queueing (CB-WFQ)**; guaranteed sustained throughput allocation. |
| **Browsing** | 3GPP 5QI 9; ITU-T Y.1541 Class 3; IETF RFC 2597 (Assured Forwarding 2 - AF2x) | $\le 1\text{--}2\text{ s}$ (Interactive delay) | Burst-tolerant | $\le 10^{-6}$ (Strict TCP reliability) | **Fair Queueing (FQ)** with Deficit Round Robin; handles bursty request-response transactions. |
| **Bulk Data** | IETF RFC 2597 (AF1x) / RFC 3662 (Lower Effort / Best Effort - BE) | Delay-insensitive ($> 10\text{ s}$) | Irrelevant | $\approx 0\%$ (TCP recovery) | **Best-Effort FIFO / Scavenger Queueing**; preemptible by Real-Time and Streaming flows. |

### Technical Verdict on Class Feasibility:
The 4-class taxonomy is **fundamentally sound and directly reflects real router queuing requirements**. However, the current dataset implementation suffers from an operational disconnect: **no true HTTP/HTTPS Web Browsing traces exist in the raw PCAP captures**. What is currently labeled as "Browsing" is 100% Email traffic (and overwhelmingly local OS broadcast queries).

---

## 2. Distinguishing Application Identity, Traffic Behavior, and QoS Classes

In academic literature, researchers frequently confuse three distinct networking concepts:

```
[Application Identity]            [Traffic Behavior]                  [QoS Service Class]
 "What software sent it?"        "How does it physically move?"       "What network contract does it need?"
       e.g. Skype                         e.g.                              e.g.
  ┌───────────────────┐             ┌─────────────────────┐            ┌────────────────────┐
  │ - Skype Text Chat │ ──────────> │ Low byte rate, burst │ ─────────> │ Browsing (AF2x)    │
  │ - Skype Voice     │ ──────────> │ Periodic, low IAT    │ ─────────> │ Real-Time (EF)     │
  │ - Skype Video     │ ──────────> │ Heavy UDP, high rate │ ─────────> │ Real-Time (EF)     │
  │ - Skype File Xfer │ ──────────> │ Sustained max TCP win│ ─────────> │ Bulk Data (AF1x)   │
  └───────────────────┘             └─────────────────────┘            └────────────────────┘
```

* **Application Identity:** Knowing that traffic is from "Skype" or "Facebook" tells a router almost nothing about its immediate queueing priority. If a user is transferring a 2 GB file over Skype, treating it as Expedited Forwarding (EF) Real-Time traffic would starve voice calls.
* **Traffic Behavior:** Physical transport-layer dynamics: packet size histograms, inter-arrival time distributions, burst lengths, and forward/backward byte ratios.
* **QoS Class:** The operational forwarding priority (DSCP code) assigned to a flow based on its delay and loss tolerances.

**Key Finding:** Our classifier must classify **QoS traffic behavior**, NOT application brands. A packet scheduler operates on packets and queues, not software brand names.

---

## 3. Observable Transport Characteristics Under Encryption

Modern transport security (TLS 1.3 [RFC 8446], QUIC [RFC 9000], WireGuard, OpenVPN) completely encrypts application payloads, HTTP headers, URLs, and server names (via Encrypted Client Hello - ECH).

### What Remains Physically Observable (Without Decryption):
1. **Packet Size Moments:** TLS records and QUIC frames are encapsulated into IP datagrams. Packet size distributions reflect application chunking:
   * Real-Time VoIP generates small, uniform packets ($60\text{--}200\text{ bytes}$).
   * Bulk file transfers generate full Maximum Segment Size (MSS) packets ($1400\text{--}1500\text{ bytes}$).
   * Adaptive video streaming (DASH/HLS) generates cyclical packet-size bursts corresponding to video keyframe intervals.
2. **Inter-Arrival Times (IAT):**
   * Real-Time audio codecs (G.711, Opus) generate packets at fixed periodic intervals ($20\text{ ms}$ pacing $\pm 2\text{ ms}$ jitter).
   * Bulk transfers flood the bottleneck link at the line rate determined by TCP Cubic or BBR congestion windows.
3. **Directionality & Burstiness:**
   * Browsing exhibits asymmetric request-response turn-taking (client sends small request $\to$ server responds with payload burst $\to$ idle period).
   * Streaming exhibits massive downstream byte ratios ($B_{\text{bwd}} / B_{\text{fwd}} > 20:1$).
   * Interactive voice exhibits near-symmetric packet ratios ($N_{\text{fwd}} \approx N_{\text{bwd}}$).

---

## 4. Line-Rate Edge Router Realities & Early-Flow Classification

A fundamental technical flaw in existing literature (including Huang et al., 2021) is **post-mortem flow classification**:
* In Huang et al., classification is performed after observing complete connections or accumulating large packet windows.
* **The Network Engineer's Dilemma:** An edge router cannot wait 60 seconds for a voice call or file transfer to terminate before deciding which DSCP queue to assign. By the time a flow is finished, the packets have already traversed the network and suffered delay or jitter!
* **Mandatory Requirement:** Traffic classification must be feasible **within the first 5 to 10 packets** ($\le 50\text{ ms}$) of connection initiation.
* **Our Implementation Advantage:** In Phase 2, our feature extractor specifically computed early-flow feature profiles (`early_5_*`, `early_10_*`, `early_15_*`, `early_20_*`). Evaluating early classification accuracy is a primary research contribution of our project.

---

## 5. Network Hardware Offloading Artifacts

Our packet inspection identified critical hardware offloading signatures in the PCAP files:
* **Large Receive Offload (LRO) / Generic Receive Offload (GRO):** In `scpDown1.pcap` and `ftps_down_1b.pcap`, individual packets measure up to $30,474\text{ bytes}$ and $64,294\text{ bytes}$.
* **Physical Reality:** Ethernet frames cannot exceed $1514\text{ bytes}$ on standard links. These mega-packets are synthesized by the capture card's NIC driver reassembling TCP segments before Wireshark captures them.
* **Mitigation:** Machine learning models must not treat $64\text{ KB}$ frames as "anomalous application packets"; they are standard TCP reassembly artifacts on modern server hardware.

---

## 6. Summary Technical Verdict

1. The 4-class QoS paradigm is technically sound and aligns with RFC 4594 and 3GPP 5QI standards.
2. The current dataset lacks true Web Browsing PCAPs; the Email captures are confounded by local multicast noise.
3. Classifying behavioral transport features (packet sizes, IAT, bursts) is resilient to TLS 1.3 and VPN encryption.
4. Early classification ($N \le 10$ packets) is technically mandatory for realistic edge router QoS admission control.
