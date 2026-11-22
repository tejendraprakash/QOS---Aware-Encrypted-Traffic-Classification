# Phase 7 Technical Specification: Discrete-Event Packet-Level Network Emulation Engine

**Project:** QoS-Aware Encrypted Network Traffic Classification for Next-Generation Networks  
**Component:** Phase 7 Network Emulation Engine  
**Author:** Network Emulation Implementation Engineer  
**Date:** November 2026  
**Status:** Approved Architecture & Implementation Specification  
**Target Environment:** Windows 11 Pro 64-bit, Python 3.13.5 (Zero Host Disruption, No Root/Privilege Elevation Required)  
**Deliverable File:** `research_notes/phase7/network_emulation_implementation.md`  

---

## 1. Executive Summary & Design Rationale

### 1.1 Objective and Research Context
In Phase 6B, our research demonstrated that an **Early-5 XGBoost classifier** (observing only the first 5 packets with 6 features) retains **97.6% of full-flow classification capability** (Pooled Macro F1 = 0.5323, Pooled Balanced Accuracy = 0.6424, Inference Latency = 3.4 $\mu\text{s}$/flow). While Phase 6B proved machine learning accuracy and inference speed, ML metrics alone cannot quantify the operational impact on end-to-end network performance.

The primary objective of **Phase 7** is to close the loop between **machine learning classification** and **actionable Quality of Service (QoS)** in Next-Generation Networks (NGN / 5G / SDN). We evaluate whether routing traffic based on Early-5 XGBoost predictions delivers tangible QoS gains—reducing latency and jitter for mission-critical Real-Time voice/video, buffering Streaming video without stalls, and preventing Bulk Data transfers from starving interactive flows.

### 1.2 The Emulation Paradigm: Discrete-Event Simulation vs. OS-Level NetEm
Modern network research on Windows workstations often encounters severe roadblocks when attempting to use kernel-level emulation frameworks such as Mininet (requiring Linux kernel network namespaces and Open vSwitch) or `iproute2` / `tc-netem` (requiring Linux kernel queuing disciplines).

| Evaluation Dimension | OS-Level Emulation (`tc-netem` / Mininet in WSL2/VM) | Discrete-Event Packet-Level Simulator (Pure Python DES) |
|---|---|---|
| **Host System Safety** | High risk of host networking disruption, driver conflicts, or proxy breakage. | **100% Zero Host Disruption:** Runs as standard user-space application without socket or admin privileges. |
| **Operating System** | Requires virtualized Linux / WSL2 kernel modules. | **Native Windows 11 Pro 64-bit with Python 3.13.5.** |
| **Measurement Precision** | Subject to OS scheduler quantum, thread preemption, and clock jitter ($\pm 1\text{ ms}$). | **Nanosecond-exact deterministic timestamps** ($10^{-9}\text{ s}$ mathematical precision). |
| **Reproducibility** | Non-deterministic; run-to-run variance caused by CPU load and OS interrupts. | **100% Bitwise Deterministic across seeds and machines.** |
| **Event Time Travel** | Real-time bound (simulating 1 hour takes 1 hour). | **Decoupled from wall-clock:** Can simulate 100,000 packets in 2–5 seconds. |
| **Inspection & Tracing** | Opaque kernel counters, coarse packet sampling. | **Complete per-packet lifecycle telemetry** from generation to drop/reception. |

Therefore, we implement a **High-Fidelity Discrete-Event Packet-Level Simulator** in Python. It models packet injection, queue ingress buffering, classifier invocation, active queue management, transmission serialization, propagation delay, and receiver reception with absolute mathematical determinism.

---

## 2. Theoretical Formulation & Network Mechanics

### 2.1 Packet Delay Decomposition
For any packet $p_i$ belonging to flow $f$, the total end-to-end latency $D(p_i)$ through the network topology is given by:

$$D(p_i) = t_{\text{arrival}}(p_i) - t_{\text{creation}}(p_i) = d_{q}(p_i) + d_{tx}(p_i) + d_{prop}(p_i)$$

Where:
1. **Queuing Delay ($d_q$):** The duration packet $p_i$ waits in the router buffer before transmission:
   $$d_q(p_i) = t_{\text{dequeue}}(p_i) - t_{\text{enqueue}}(p_i)$$
   If the buffer is full upon arrival, the packet experiences a tail-drop event ($d_q = \infty, \text{dropped} = \text{True}$).
2. **Transmission Serialization Delay ($d_{tx}$):** The time required to inject the packet bits onto the physical link of capacity $C$ (in bits per second):
   $$d_{tx}(p_i) = \frac{L(p_i) \times 8}{C}$$
   Where $L(p_i)$ is the packet wire length in bytes (including headers).
3. **Propagation Delay ($d_{prop}$):** The physical transit time across the physical medium over distance $d$ with propagation velocity $v$:
   $$d_{prop}(p_i) = \tau = \frac{d}{v}$$

### 2.2 Packet Delay Variation (Jitter) Formulation
Following RFC 3393 and RFC 4689, packet jitter $J_i$ for consecutive received packets $p_{i-1}$ and $p_i$ of the same flow is defined as:

$$J_i = \left| D(p_i) - D(p_{i-1}) \right| = \left| (t_{\text{arrival}}(p_i) - t_{\text{creation}}(p_i)) - (t_{\text{arrival}}(p_{i-1}) - t_{\text{creation}}(p_{i-1})) \right|$$

For real-time multimedia flows (e.g., VoIP, WebRTC audio/video), the smoothed Mean Absolute Jitter $\bar{J}$ is computed via the standard exponential moving average (RFC 3550):

$$\bar{J}_i = \bar{J}_{i-1} + \frac{|D(p_i) - D(p_{i-1})| - \bar{J}_{i-1}}{16}$$

### 2.3 Link Utilization and Congestion Formulation
Given a bottleneck link of capacity $C$ (bps) and an evaluation window $T_{\text{sim}}$ (seconds), the offered traffic load $\lambda_{\text{offered}}$ is:

$$\lambda_{\text{offered}} = \frac{\sum_{i=1}^{N_{\text{total}}} L(p_i) \times 8}{T_{\text{sim}}} \quad (\text{bps})$$

The normalized link utilization (load factor $\rho$) is:

$$\rho = \frac{\lambda_{\text{offered}}}{C}$$

The emulation engine evaluates three specific operational regimes:
* **Low Congestion ($\rho \approx 0.50$):** Link is under-utilized. Queuing delay is negligible; transmission and propagation delays dominate.
* **Moderate Congestion ($\rho \approx 0.85$):** Operating near the knee of the delay-throughput curve. Queues begin to form intermittently; bursts induce transient queuing delay.
* **Severe Bottleneck Congestion ($\rho \approx 1.20 - 1.40$):** Link is heavily over-subscribed. Buffers saturate; unmanaged FIFO drops up to 25–35% of packets, causing catastrophic degradation for delay-sensitive applications.

---

## 3. Emulation Engine Architecture

```
+--------------------------------------------------------------------------------------------------+
|                                    DISCRETE EVENT SCHEDULER                                      |
|                             Min-Heap Event Queue (heapq: timestamp, event_id)                     |
+--------------------------------------------------------------------------------------------------+
                                                 |
         +---------------------------------------+---------------------------------------+
         |                                       |                                       |
         v                                       v                                       v
[1. Packet Arrival]                   [2. Link Dequeue/Tx Start]             [3. Packet Reception]
  - Trace / PCAP Replay                 - Query Queue Discipline               - Propagate delay tau
  - Packet 1-5 Ingress Buffer           - Strict Priority / DRR / FIFO         - Record t_arrival
  - Classifier trigger (N=5)            - Serialize at link rate C             - Telemetry & Metrics
         |                                       |
         v                                       v
+------------------------+             +-------------------+
| Ingress Flow Demux     |             | Bottleneck Link   |
| - Precomputed / Online |             | - Rate C (bps)    |
| - Assign QoS Queue     |             | - Delay tau (sec) |
+------------------------+             +-------------------+
```

### 3.1 Event Engine & Event Hierarchy
The engine operates on a central Priority Queue using Python's standard `heapq` module. Every event is represented as a strictly ordered tuple:

$$\text{Event} = (t_{\text{event}}, \text{event\_id}, \text{event\_type}, \text{handler}, \text{payload})$$

The `event_id` is an auto-incrementing 64-bit integer ensuring stable sorting when multiple events coincide at the exact same floating-point timestamp, preventing arbitrary Python object comparison failures.

#### Supported Event Types:
1. `EVENT_PACKET_ARRIVAL`: A packet arrives at the router ingress interface from a traffic source (PCAP trace or synthetic flow generator).
2. `EVENT_CLASSIFICATION_COMPLETE`: Triggered when flow packet count reaches horizon $N=5$. The Early-5 classifier returns the predicted QoS class, triggering a policy transition for subsequent packets.
3. `EVENT_TRANSMISSION_START`: The bottleneck link finishes transmitting the previous packet and requests the next packet from the active queuing discipline.
4. `EVENT_TRANSMISSION_COMPLETE`: The link finishes pushing the packet bits onto the wire. The link returns to `IDLE` state (or immediately services the next queued packet).
5. `EVENT_PACKET_RECEPTION`: The packet arrives at the destination sink after the propagation delay $\tau$, where end-to-end metrics are recorded.

---

## 4. Packet and Flow Data Models

### 4.1 Packet Data Model
```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

class DropReason(Enum):
    NOT_DROPPED = "not_dropped"
    BUFFER_OVERFLOW = "buffer_overflow"
    CLASSIFIER_OVERHEAD = "classifier_overhead"

class QoSClass(Enum):
    BULK_DATA = "Bulk Data"
    STREAMING = "Streaming"
    REAL_TIME = "Real-Time"
    UNCLASSIFIED = "Unclassified"

@dataclass(slots=True)
class Packet:
    packet_id: int                  # Globally unique 64-bit packet sequence number
    flow_id: str                    # Unique 5-tuple flow identifier
    class_label: QoSClass           # Ground-truth QoS class
    predicted_label: QoSClass       # Predicted class (assigned at N=5 or precomputed)
    size_bytes: int                 # Layer 3/4 frame size on wire (e.g., 64 - 1514 bytes)
    creation_time: float            # Timestamp packet generated at source (seconds)
    
    # State tracking across router components
    enqueue_time: float = -1.0      # Timestamp packet entered router buffer
    dequeue_time: float = -1.0      # Timestamp packet dequeued for transmission
    tx_start_time: float = -1.0     # Timestamp serialization began on physical link
    tx_end_time: float = -1.0       # Timestamp last bit transmitted onto link
    arrival_time: float = -1.0      # Timestamp packet arrived at receiver sink
    
    # Drop telemetry
    dropped: bool = False           # Flag indicating whether packet was dropped
    drop_reason: DropReason = DropReason.NOT_DROPPED
    queue_id_at_drop: int = -1      # Which internal queue dropped the packet
```

### 4.2 Flow State and Ingress Demux Model
```python
@dataclass
class FlowState:
    flow_id: str
    ground_truth_class: QoSClass
    predicted_class: QoSClass = QoSClass.UNCLASSIFIED
    packet_count: int = 0
    byte_count: int = 0
    is_classified: bool = False
    ingress_buffer: list[Packet] = field(default_factory=list)
```

---

## 5. Queue Disciplines Specification & Algorithms

The engine implements three distinct queuing disciplines conforming to a unified abstract interface:

```python
from abc import ABC, abstractmethod

class QueueDiscipline(ABC):
    @abstractmethod
    def enqueue(self, packet: Packet, current_time: float) -> bool:
        """Attempts to enqueue a packet. Returns True if enqueued, False if dropped."""
        pass

    @abstractmethod
    def dequeue(self, current_time: float) -> Optional[Packet]:
        """Dequeues the next packet to transmit according to the scheduling policy."""
        pass

    @abstractmethod
    def is_empty(self) -> bool:
        """Returns True if all internal buffers are empty."""
        pass

    @abstractmethod
    def total_bytes(self) -> int:
        """Returns total bytes currently queued across all buffers."""
        pass
```

### 5.1 Discipline 1: Single-Buffer Drop-Tail FIFO (Baseline)
The baseline representative of current Best-Effort Internet routers. All packets share a single finite buffer of capacity $B_{\text{max}}$ (configurable in bytes or packets). Packets arriving when the buffer is saturated are dropped unconditionally.

```
       Incoming Packets
             |
             v
   +--------------------+
   | [P_k] [P_2] [P_1]  | ---> Link Interface (C bps)
   +--------------------+
    Capacity: B_max bytes
```

#### Enqueue Algorithm (FIFO):
```python
class DropTailFIFO(QueueDiscipline):
    def __init__(self, max_bytes: int = 256 * 1024):
        self.max_bytes = max_bytes
        self.current_bytes = 0
        self.queue: collections.deque[Packet] = collections.deque()

    def enqueue(self, packet: Packet, current_time: float) -> bool:
        if self.current_bytes + packet.size_bytes > self.max_bytes:
            packet.dropped = True
            packet.drop_reason = DropReason.BUFFER_OVERFLOW
            return False
        packet.enqueue_time = current_time
        self.queue.append(packet)
        self.current_bytes += packet.size_bytes
        return True

    def dequeue(self, current_time: float) -> Optional[Packet]:
        if not self.queue:
            return None
        packet = self.queue.popleft()
        self.current_bytes -= packet.size_bytes
        packet.dequeue_time = current_time
        return packet
```
* **Complexity:** Enqueue $O(1)$, Dequeue $O(1)$.
* **Memory Footprint:** Finite, strictly bounded by `max_bytes`.

---

### 5.2 Discipline 2: Strict Priority (SP) Queuing
Strict Priority implements three dedicated queues ordered by strict hierarchy:
1. **Priority 0 (High Priority / Expedited Forwarding - EF):** Assigned to **Real-Time** traffic (interactive voice/video).
2. **Priority 1 (Medium Priority / Assured Forwarding - AF):** Assigned to **Streaming** media (buffered video chunks).
3. **Priority 2 (Low Priority / Best Effort - BE):** Assigned to **Bulk Data** (file downloads, background sync) and unclassified initial packets.

```
Incoming
Packets
   |
   +---> [ Classifier / Mapping ]
               |
               +---> Queue 0: Real-Time  (High)   [P_rt]  \
               |                                           |---> Non-Preemptive
               +---> Queue 1: Streaming  (Medium) [P_str]  |---> Selector
               |                                           |     (Checks 0, then 1, then 2)
               +---> Queue 2: Bulk Data  (Low)    [P_blk] /
```

#### Operational Mechanics:
* **Non-Preemptive Service:** If a low-priority Bulk packet is actively transmitting on the physical link, an arriving High-Priority Real-Time packet cannot preempt it. It must wait until the current packet finishes transmission ($t_{\text{tx\_end}}$).
* **Selection Rule:** When the link becomes idle, Queue 0 is serviced until empty; Queue 1 is serviced only when Queue 0 is empty; Queue 2 is serviced only when Queues 0 and 1 are empty.
* **Buffer Partitioning:** Each queue maintains its own finite capacity ($B_0, B_1, B_2$) to prevent buffer starvation from one class crashing another.

#### Implementation Pseudo-Code:
```python
class StrictPriorityQueue(QueueDiscipline):
    def __init__(self, max_bytes_per_queue: dict[int, int]):
        self.queues = {0: collections.deque(), 1: collections.deque(), 2: collections.deque()}
        self.max_bytes = max_bytes_per_queue
        self.current_bytes = {0: 0, 1: 0, 2: 0}

    def _map_class_to_priority(self, qos_class: QoSClass) -> int:
        if qos_class == QoSClass.REAL_TIME:
            return 0
        elif qos_class == QoSClass.STREAMING:
            return 1
        else: # BULK_DATA or UNCLASSIFIED
            return 2

    def enqueue(self, packet: Packet, current_time: float) -> bool:
        priority = self._map_class_to_priority(packet.predicted_label)
        if self.current_bytes[priority] + packet.size_bytes > self.max_bytes[priority]:
            packet.dropped = True
            packet.drop_reason = DropReason.BUFFER_OVERFLOW
            packet.queue_id_at_drop = priority
            return False
        packet.enqueue_time = current_time
        self.queues[priority].append(packet)
        self.current_bytes[priority] += packet.size_bytes
        return True

    def dequeue(self, current_time: float) -> Optional[Packet]:
        for prio in (0, 1, 2):
            if self.queues[prio]:
                packet = self.queues[prio].popleft()
                self.current_bytes[prio] -= packet.size_bytes
                packet.dequeue_time = current_time
                return packet
        return None
```
* **Complexity:** Enqueue $O(1)$, Dequeue $O(K) = O(3) \equiv O(1)$.

---

### 5.3 Discipline 3: Deficit Round Robin (DRR)
While Strict Priority guarantees minimal latency for Real-Time traffic, it risks starving Bulk Data during prolonged high-volume periods. Deficit Round Robin (Shreedhar & Varghese, 1996) provides fair queuing with configurable bandwidth guarantees without starvation, achieving $O(1)$ packet scheduling complexity.

#### Mechanics and State Variables:
* Three active queues: $i \in \{0 (\text{Real-Time}), 1 (\text{Streaming}), 2 (\text{Bulk Data})\}$.
* **Weights ($w_i$):** Normalized bandwidth allocations:
  $$w_{\text{Real-Time}} = 0.50, \quad w_{\text{Streaming}} = 0.30, \quad w_{\text{Bulk Data}} = 0.20$$
* **Quantum ($Q_i$):** Configured proportional to weights:
  $$Q_i = w_i \times \text{Base Quantum} \quad (\text{e.g., Base Quantum} = 3000\text{ bytes})$$
* **Deficit Counter ($DC_i$):** Accumulates unused quantum across rounds.
* **Active List:** Tracks non-empty queues to skip idle queues in $O(1)$ time.

```
Round Robin Pointer
      |
      v
+-------------+      +-------------+      +-------------+
| Queue 0: RT |      | Queue 1: ST |      | Queue 2: BK |
| DC_0 += Q_0 |      | DC_1 += Q_1 |      | DC_2 += Q_2 |
| While size  |      | While size  |      | While size  |
| <= DC_0:    |      | <= DC_1:    |      | <= DC_2:    |
|   transmit  |      |   transmit  |      |   transmit  |
+-------------+      +-------------+      +-------------+
```

#### DRR Dequeue Algorithm:
```python
class DeficitRoundRobinQueue(QueueDiscipline):
    def __init__(self, quantums: dict[int, int], max_bytes: dict[int, int]):
        self.quantums = quantums            # e.g., {0: 3000, 1: 1800, 2: 1200} bytes
        self.deficit_counter = {0: 0, 1: 0, 2: 0}
        self.queues = {0: collections.deque(), 1: collections.deque(), 2: collections.deque()}
        self.max_bytes = max_bytes
        self.current_bytes = {0: 0, 1: 0, 2: 0}
        self.active_list = collections.deque()
        self.in_active_list = {0: False, 1: False, 2: False}

    def enqueue(self, packet: Packet, current_time: float) -> bool:
        qid = self._map_class(packet.predicted_label)
        if self.current_bytes[qid] + packet.size_bytes > self.max_bytes[qid]:
            packet.dropped = True
            packet.drop_reason = DropReason.BUFFER_OVERFLOW
            return False
        
        packet.enqueue_time = current_time
        self.queues[qid].append(packet)
        self.current_bytes[qid] += packet.size_bytes
        
        if not self.in_active_list[qid]:
            self.active_list.append(qid)
            self.in_active_list[qid] = True
            self.deficit_counter[qid] = 0
        return True

    def dequeue(self, current_time: float) -> Optional[Packet]:
        while self.active_list:
            qid = self.active_list[0]
            if not self.queues[qid]:
                # Queue emptied; remove from active list and reset deficit
                self.in_active_list[qid] = False
                self.deficit_counter[qid] = 0
                self.active_list.popleft()
                continue

            # Replenish deficit counter if starting new round for this queue
            if self.deficit_counter[qid] == 0:
                self.deficit_counter[qid] += self.quantums[qid]

            head_pkt = self.queues[qid][0]
            if head_pkt.size_bytes <= self.deficit_counter[qid]:
                # Packet can be served
                self.deficit_counter[qid] -= head_pkt.size_bytes
                packet = self.queues[qid].popleft()
                self.current_bytes[qid] -= packet.size_bytes
                packet.dequeue_time = current_time
                return packet
            else:
                # Cannot serve; move to next queue in round-robin order
                self.active_list.rotate(-1)
                if self.active_list:
                    next_qid = self.active_list[0]
                    self.deficit_counter[next_qid] += self.quantums[next_qid]
        return None
```
* **Complexity:** $O(1)$ amortized per packet. Work-conserving.

---

## 6. Integration with Early-5 XGBoost Classifier

### 6.1 Early Packet Ingress Buffer Pipeline (Packets 1 to 5)
A critical requirement in Next-Generation Networks is handling traffic *before* classification occurs. A flow cannot be classified at packet 1 without payload inspection (which fails under TLS 1.3 encryption). 

#### Ingress Flow Demux Policy:
1. **Packets 1 through 4 (Accumulation Window):**
   * Arriving packets update the flow's early feature tracking window (`early_5_avg_size`, `early_5_bwd_pkts`, `early_5_iat_mean`, etc.).
   * These packets enter the **Default Best-Effort Queue** (Priority 2 / Low Priority) to ensure immediate forward progress without stalling the TCP handshake.
2. **Packet 5 Arrival (Classification Event):**
   * The 5th packet arrives, completing the Early-5 feature vector.
   * The flow manager assigns the `predicted_label` from the Early-5 XGBoost model.
   * If the flow's ingress buffer still holds un-transmitted packets, they are immediately tagged with the new classification.
3. **Packets 6 to $\infty$ (Classified QoS Forwarding):**
   * Every subsequent packet of the flow is directly stamped with the predicted label upon arrival and routed to the corresponding high-priority or fair-share queue.

```
Packet 1 ---> [ Ingress Buffer / BE Queue ] ---> Link
Packet 2 ---> [ Ingress Buffer / BE Queue ] ---> Link
Packet 3 ---> [ Ingress Buffer / BE Queue ] ---> Link
Packet 4 ---> [ Ingress Buffer / BE Queue ] ---> Link
Packet 5 ---> [ EARLY-5 XGBOOST CLASSIFIER ]
                     |
                     +---> Prediction: REAL-TIME (High Priority EF)
                     |
Packet 6+ --> [ Queue 0 (Real-Time EF) ] ---------> Fast-Track Transmission
```

### 6.2 The Three Experimental Scenarios
To isolate the exact quantitative benefit of the classifier, the emulation engine evaluates three comparative scenarios across identical packet traces:

| Scenario | Classifier Mode | Ingress Queue Discipline | Description |
|---|---|---|---|
| **Scenario 1: Best-Effort Baseline** | None | Drop-Tail FIFO | Traditional unmanaged network. All flows share one buffer regardless of application. Represents status quo Internet edge. |
| **Scenario 2: Deployable Early-5 QoS** | Phase 6B Early-5 XGBoost | Strict Priority / DRR | Packets 1–4 default to BE; Packet 5+ steered according to Early-5 XGBoost predictions (`pred_XGBoost` from `out_of_fold_predictions.csv`). Reflects real-world deployable performance including ML misclassifications. |
| **Scenario 3: Oracle Upper-Bound** | Ground-Truth Oracle | Strict Priority / DRR | Packets steered according to true labels (`true_label`). Represents theoretical maximum achievable QoS gain with zero classification error. |

### 6.3 Seamless Integration with Phase 6B Artifacts
Phase 6B generated exact out-of-fold predictions for all 10,770 Track B flows in `output/phase6b/early_5/out_of_fold_predictions.csv`:
```csv
flow_id,capture_group,qos_class,application,is_vpn,fold,true_label,pred_Random_Forest,...,pred_XGBoost,...
flow_facebook_audio1a_000001,GRP_FB_AUDIO_1,Real-Time,Facebook Audio,No,2,Real-Time,...,Bulk Data,...
flow_facebook_audio1a_000002,GRP_FB_AUDIO_1,Real-Time,Facebook Audio,No,2,Real-Time,...,Streaming,...
```

The emulation engine supports dual classification modes:
1. **Trace-Driven Lookup Mode (Fast & Exact):** Loads `out_of_fold_predictions.csv` into an in-memory hash table:
   $$\text{PredictionMap}: \text{flow\_id} \mapsto (\text{true\_label}, \text{pred\_XGBoost})$$
   Enables sub-millisecond execution of large multi-flow traces with verified out-of-fold predictions.
2. **Online Feature Extractor Mode (Live Pipeline):** Dynamically computes the 6 Early-5 features from incoming packet timestamps and sizes, executing the scikit-learn/XGBoost `predict()` method at packet 5.

---

## 7. Bottleneck Link & Congestion Regimes

### 7.1 Bottleneck Link Model
The bottleneck link models the constrained physical egress interface of an edge router (e.g., DSL/Fiber uplink, 5G gNodeB radio bearer, or satellite terminal).

```python
class BottleneckLink:
    def __init__(self, capacity_bps: float, propagation_delay_s: float):
        self.capacity_bps = capacity_bps
        self.propagation_delay_s = propagation_delay_s
        self.is_busy = False
        self.current_packet: Optional[Packet] = None
        self.tx_end_time: float = 0.0

    def transmission_duration(self, packet_bytes: int) -> float:
        return (packet_bytes * 8.0) / self.capacity_bps
```

### 7.2 Experimental Congestion Regimes
To rigorously test the queuing disciplines under stress, we specify three link configurations:

| Parameter | Low Congestion | Moderate Congestion | Severe Congestion |
|---|---|---|---|
| **Offered Load Factor ($\rho$)** | $\approx 0.50$ (50%) | $\approx 0.85$ (85%) | $\approx 1.20 - 1.40$ (120–140%) |
| **Bottleneck Capacity ($C$)** | 50.0 Mbps | 15.0 Mbps | 8.0 Mbps |
| **Propagation Delay ($\tau$)** | 15.0 ms | 15.0 ms | 25.0 ms |
| **Buffer Sizing Rule** | Bandwidth-Delay Product (BDP): $B = C \times \tau$ | $1.0 \times \text{BDP}$ | $0.5 \times \text{BDP}$ (Constrained) |
| **Target Packet Loss (FIFO)** | $0.0\%$ | $< 2.0\%$ | $15.0\% - 30.0\%$ |

---

## 8. Telemetry, Performance Metrics & Analytical Outputs

At simulation completion, the `MetricsCollector` aggregates packet logs and computes class-specific and global performance metrics:

### 8.1 Evaluated Metrics:
1. **Packet Delivery Ratio (PDR) & Packet Loss Ratio (PLR):**
   $$\text{PDR}_c = \frac{N_{\text{received}, c}}{N_{\text{total}, c}} \times 100\%, \quad \text{PLR}_c = \frac{N_{\text{dropped}, c}}{N_{\text{total}, c}} \times 100\%$$
2. **End-to-End Latency Statistics:**
   * Mean Latency: $\bar{D}_c$
   * Median Latency: $D_{\text{median}, c}$
   * 95th Percentile Latency: $D_{p95, c}$
   * 99th Percentile Latency: $D_{p99, c}$
3. **Packet Delay Variation (Jitter):**
   * RFC 3393 Inter-Packet Delay Variation (IPDV) Mean and $p95$.
4. **Effective Goodput (Mbps):**
   $$\text{Goodput}_c = \frac{\sum_{p \in \text{Received}_c} L_{\text{payload}}(p) \times 8}{T_{\text{sim}} \times 10^6} \quad (\text{Mbps})$$
5. **Jain's Fairness Index:**
   For bandwidth allocation across flows in DRR:
   $$\mathcal{J}(x_1, x_2, \dots, x_n) = \frac{\left( \sum_{i=1}^n x_i \right)^2}{n \sum_{i=1}^n x_i^2}$$

### 8.2 Misclassification Impact Analysis
The engine explicitly measures the penalty of classification errors:
* **False Bulk (Real-Time misclassified as Bulk Data):** Latency increase $\Delta D = D_{\text{pred}} - D_{\text{oracle}}$ and loss increase $\Delta \text{PLR}$.
* **False Real-Time (Bulk Data misclassified as Real-Time):** Impact on Real-Time queue delay (queue pollution / buffer bloat caused by bulk packet insertion).

---

## 9. Comprehensive Implementation Architecture

### 9.1 File Structure for Phase 7
```
c:\Users\saipr\OneDrive\Desktop\EncryptedTrafficResearch\
├── src\
│   └── emulation\
│       ├── __init__.py
│       ├── engine.py           # Discrete-event engine (heapq scheduler, event loops)
│       ├── models.py           # Packet, FlowState, QoSClass, DropReason dataclasses
│       ├── queues.py           # DropTailFIFO, StrictPriorityQueue, DeficitRoundRobinQueue
│       ├── link.py             # BottleneckLink serialization and propagation model
│       ├── traffic_generator.py# PCAP trace player and synthetic burst generator
│       ├── classifier_shim.py  # Early-5 XGBoost prediction integration
│       └── metrics.py          # Telemetry, jitter calculation, export to CSV/JSON
├── scripts\
│   ├── run_phase7_emulation.py # Master execution script across Scenarios & Regimes
│   └── test_emulation_engine.py# Comprehensive unit and validation test suite
└── output\
    └── phase7\
        ├── scenario1_baseline\
        ├── scenario2_predicted_qos\
        └── scenario3_oracle_qos\
```

### 9.2 Event Scheduler Core Implementation (`engine.py`)
```python
import heapq
from typing import Callable, Any

class SimulationEngine:
    def __init__(self):
        self.current_time: float = 0.0
        self.event_queue: list[tuple[float, int, str, Callable, Any]] = []
        self._event_counter: int = 0

    def schedule(self, delay: float, event_type: str, handler: Callable, payload: Any = None):
        """Schedules an event to occur after 'delay' seconds."""
        event_time = self.current_time + delay
        self._event_counter += 1
        heapq.heappush(self.event_queue, (event_time, self._event_counter, event_type, handler, payload))

    def run_until(self, stop_time: float):
        """Executes events in strict timestamp order until event queue is empty or stop_time is reached."""
        while self.event_queue:
            event_time, _, event_type, handler, payload = self.event_queue[0]
            if event_time > stop_time:
                break
            heapq.heappop(self.event_queue)
            self.current_time = event_time
            handler(payload)
```

### 9.3 Main Event Processing Handlers
```python
class NetworkEmulator:
    def __init__(self, queue_discipline: QueueDiscipline, link: BottleneckLink, engine: SimulationEngine):
        self.queue = queue_discipline
        self.link = link
        self.engine = engine
        self.metrics = MetricsCollector()

    def handle_packet_arrival(self, packet: Packet):
        """Triggered when a packet reaches the router ingress interface."""
        enqueued = self.queue.enqueue(packet, self.engine.current_time)
        if not enqueued:
            self.metrics.record_drop(packet)
            return

        # If link is idle, trigger immediate transmission
        if not self.link.is_busy:
            self.schedule_next_transmission()

    def schedule_next_transmission(self):
        """Extracts next packet from queue discipline and begins link transmission."""
        if self.queue.is_empty():
            self.link.is_busy = False
            return

        packet = self.queue.dequeue(self.engine.current_time)
        if packet is None:
            self.link.is_busy = False
            return

        self.link.is_busy = True
        self.link.current_packet = packet
        packet.tx_start_time = self.engine.current_time
        
        # Calculate serialization time: (size_bytes * 8) / C
        tx_duration = self.link.transmission_duration(packet.size_bytes)
        packet.tx_end_time = self.engine.current_time + tx_duration

        # Schedule transmission complete event
        self.engine.schedule(
            delay=tx_duration,
            event_type="TX_COMPLETE",
            handler=self.handle_transmission_complete,
            payload=packet
        )

    def handle_transmission_complete(self, packet: Packet):
        """Triggered when the last bit of the packet is pushed onto the physical link."""
        # Immediately schedule next transmission on link to maintain work conservation
        self.schedule_next_transmission()

        # Schedule packet arrival at destination after propagation delay tau
        self.engine.schedule(
            delay=self.link.propagation_delay_s,
            event_type="PACKET_RECEPTION",
            handler=self.handle_packet_reception,
            payload=packet
        )

    def handle_packet_reception(self, packet: Packet):
        """Triggered when packet reaches the receiving end-host."""
        packet.arrival_time = self.engine.current_time
        self.metrics.record_reception(packet)
```

---

## 10. Verification and Validation Checklist

Before executing full-scale Phase 7 experiments, the implementation will be validated against this verification protocol:

- [ ] **Check 1: Single Packet Latency Verification**
  - Inject 1 packet of 1000 bytes onto a 10 Mbps link ($C = 10^7\text{ bps}$) with $\tau = 10\text{ ms}$.
  - Expected: $d_{tx} = \frac{8000}{10^7} = 0.8\text{ ms}$.
  - Expected Total Latency: $D = 0.8\text{ ms} + 10.0\text{ ms} = 10.8\text{ ms}$.
  - Verification: Simulator recorded $D$ must match $0.010800000\text{ s}$ within $10^{-9}\text{ s}$.

- [ ] **Check 2: Buffer Overflow & Packet Conservation**
  - Buffer set to 5000 bytes. Inject 10 packets of 1000 bytes at $t=0$ while link is halted.
  - Expected: Exactly 5 packets enqueued, exactly 5 packets dropped with `DropReason.BUFFER_OVERFLOW`.
  - Invariant Check: $N_{\text{injected}} = N_{\text{received}} + N_{\text{dropped}} + N_{\text{in\_buffer}} + N_{\text{in\_flight}}$.

- [ ] **Check 3: Strict Priority Non-Preemptive Precedence**
  - Enqueue 1 Bulk packet (Priority 2) followed by 1 Real-Time packet (Priority 0).
  - Dequeue order must strictly return Real-Time first, then Bulk.
  - If Bulk is already transmitting, arriving Real-Time packet must wait for Bulk transmission completion (non-preemptive verification).

- [ ] **Check 4: Deficit Round Robin Bandwidth Proportionality**
  - Configure weights $w_0 = 3, w_1 = 1$. Continuously inject equal-sized packets into both queues.
  - Dequeue stream under heavy load must yield a transmitted byte ratio of exactly $3.0 \pm 0.01$ to $1.0$.

- [ ] **Check 5: Determinism and Seed Repeatability**
  - Execute a 10,000-packet simulation twice with seed `42`.
  - SHA-256 hash of output packet logs must be bitwise identical.

- [ ] **Check 6: Zero Host Disruption Safety Audit**
  - Verify zero socket bindings (`socket.bind()`), zero raw network sniffers, zero administrative privileges, and zero routing table modifications.

---

## 11. Conclusion & Implementation Roadmap

The discrete-event emulation architecture specified above establishes a rigorous, scientifically defensible platform for Phase 7. By coupling the verified Phase 6B Early-5 XGBoost predictions with work-conserving queuing disciplines in a deterministic Python simulation, the research directly quantifies the real-world QoS value of encrypted traffic classification for next-generation networks.

### Execution Phases:
1. **Milestone 7.1:** Implement core DES classes in `src/emulation/` and verify unit test checklist.
2. **Milestone 7.2:** Load Phase 6B `out_of_fold_predictions.csv` and construct trace player for ISCX Track B flows.
3. **Milestone 7.3:** Execute the 9 matrix experiments (3 Scenarios $\times$ 3 Congestion Regimes) across FIFO, Strict Priority, and DRR.
4. **Milestone 7.4:** Generate master QoS publication tables, latency distribution graphs ($D_{p95}, D_{p99}$), and jitter plots for the final thesis report.
