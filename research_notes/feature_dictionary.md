# Feature Dictionary: Encrypted Traffic Flow Classification

**Project:** QoS-Aware Encrypted Network Traffic Classification for Next-Generation Networks  
**Dataset Phase:** Phase 2 (Flow Formation & Feature Extraction)  
**Lead Research Engineer:** Agent Coordination Team  
**Date:** November 2026  

---

## 1. Overview & Methodological Principles

In modern encrypted networks (TLS 1.3, QUIC/HTTP3, SSH, WireGuard, OpenVPN), deep packet inspection of application payloads is impossible. Furthermore, extracting or learning from static transport identifiers (IP addresses, MAC addresses, port numbers, or raw packet headers) creates **severe data leakage and memorization vulnerabilities**, because machine learning models simply memorize specific IP subnets or server ports rather than learning genuine traffic behaviors.

To achieve scientific validity, generalizability, and compliance with next-generation network standards (3GPP / ITU-T DiffServ QoS), our feature set is divided into two strict categories:
1. **Audit & Metadata Fields (Non-ML Columns):** Retained exclusively for debugging, dataset traceability, group cross-validation, and ground-truth verification.
2. **ML-Safe Behavioral Features:** Strictly payload-agnostic statistical, temporal, directional, burst, and early-packet dynamics.

---

## 2. Inactivity Timeout & Flow Definition

### 2.1 The Bidirectional 5-Tuple Abstraction
A bidirectional flow is identified by the canonical 5-tuple:
$$\text{Canonical Key} = \left( \min(IP_A:Port_A, IP_B:Port_B), \max(IP_A:Port_A, IP_B:Port_B), \text{Protocol} \right)$$
* **Forward Direction ($F$):** Defined by the source IP and source port of the **very first packet** observed for that flow.
* **Backward Direction ($B$):** Packets traveling in the reverse direction.

### 2.2 Inactivity Timeout Slicing ($T_{timeout} = 60\text{ seconds}$)
If no packet is observed for a given 5-tuple for more than $60.0$ seconds:
$$\Delta t = t_{current} - t_{last} > 60.0\text{ s}$$
The active flow is finalized and exported as a completed flow record. Any subsequent packet on that 5-tuple initiates a **new flow instance**.

### 2.3 Scientific Limitations
* A 5-tuple flow is **not** identical to an application-layer user session. A single browsing session or streaming video playback initiates multiple concurrent and sequential 5-tuples (DNS, CDN media chunks, telemetry, TLS handshakes).
* In VPN tunnels, multiple user applications may be multiplexed through a single outer tunnel 5-tuple (e.g. OpenVPN over UDP 1194).

---

## 3. Comprehensive Feature Dictionary

### Category 1: Audit Metadata (Strictly Excluded from ML Input Features)

| Column Name | Type | Description | Scientific Purpose |
| :--- | :---: | :--- | :--- |
| `flow_id` | String | Unique flow identifier (`flow_<capture>_<seq>`) | Primary key for dataset tracking |
| `capture_filename` | String | Originating capture filename (e.g., `email2a.pcap`) | Traceability to raw capture |
| `capture_group` | String | Cross-validation group (e.g., `GRP_EMAIL_2`) | **Critical:** Used for GroupKFold to prevent data leakage |
| `application` | String | Application name (e.g., `Email`, `Netflix`) | Ground-truth service label |
| `is_vpn` | String | Whether traffic was captured inside a VPN tunnel (`Yes`/`No`) | Controlled experiment stratification |
| `qos_class` | String | Target QoS class (`Bulk Data`, `Browsing`, `Streaming`, `Real-Time`) | Supervised target label $y$ |
| `label_confidence`| String | Confidence of label assignment (`High`, `Medium`, `Low`) | Ambiguity filtering |
| `src_ip` | String | Forward source IP address | Audit only (Excluded from ML) |
| `src_port` | Integer| Forward source port number | Audit only (Excluded from ML) |
| `dst_ip` | String | Forward destination IP address | Audit only (Excluded from ML) |
| `dst_port` | Integer| Forward destination port number | Audit only (Excluded from ML) |
| `protocol_name` | String | Human-readable transport name (`TCP`, `UDP`) | Verification |
| `flow_start_time` | String | ISO 8601 UTC timestamp of flow initiation | Temporal audit |
| `flow_end_time` | String | ISO 8601 UTC timestamp of flow termination | Temporal audit |

---

### Category 2: Basic Volume & Rate Features (ML-Safe)

| Column Name | Type | Description & Formula |
| :--- | :---: | :--- |
| `protocol` | Integer | Numerical IP protocol number (6 for TCP, 17 for UDP). |
| `flow_duration` | Float | Total flow duration in seconds: $\Delta t = t_{last} - t_{first}$. |
| `total_packets` | Integer | Total packets in flow: $N_{total} = N_{fwd} + N_{bwd}$. |
| `fwd_packets` | Integer | Number of packets sent by the flow initiator. |
| `bwd_packets` | Integer | Number of packets sent in the reverse direction. |
| `total_bytes` | Integer | Total IP datagram bytes transferred: $B_{total} = B_{fwd} + B_{bwd}$. |
| `fwd_bytes` | Integer | Total IP datagram bytes sent forward. |
| `bwd_bytes` | Integer | Total IP datagram bytes sent backward. |
| `packet_rate` | Float | Flow throughput in packets/sec: $N_{total} / \Delta t$ (or $N_{total}$ if $\Delta t = 0$). |
| `byte_rate` | Float | Flow throughput in bytes/sec: $B_{total} / \Delta t$ (or $B_{total}$ if $\Delta t = 0$). |
| `fwd_pkt_ratio` | Float | Ratio of forward packets to total packets: $N_{fwd} / N_{total}$. |
| `fwd_byte_ratio` | Float | Ratio of forward bytes to total bytes: $B_{fwd} / B_{total}$. |
| `fwd_to_bwd_pkt_ratio` | Float | Ratio of forward to backward packets: $N_{fwd} / (N_{bwd} + 10^{-6})$. |
| `fwd_to_bwd_byte_ratio` | Float | Ratio of forward to backward bytes: $B_{fwd} / (B_{bwd} + 10^{-6})$. |

*Note on Packet Size Measurement:* Packet size is measured as the **IP Total Length** (`ip_len` from IPv4 header or `payload_len + 40` from IPv6 header). This isolates genuine traffic volume from link-layer framing, Ethernet 64-byte padding, and virtual TUN/TAP encapsulation differences.

---

### Category 3: Packet Size Distribution Statistics (ML-Safe)
Computed independently for:
1. **Bidirectional traffic:** `bidir_pkt_size_*`
2. **Forward traffic:** `fwd_pkt_size_*`
3. **Backward traffic:** `bwd_pkt_size_*`

| Metric Suffix | Type | Description & Derivation |
| :--- | :---: | :--- |
| `*_mean` | Float | Arithmetic mean: $\mu = \frac{1}{n}\sum_{i=1}^{n} s_i$. |
| `*_std` | Float | Standard deviation: $\sigma = \sqrt{\frac{1}{n}\sum_{i=1}^{n} (s_i - \mu)^2}$. |
| `*_min` | Float | Minimum observed IP packet size in bytes. |
| `*_max` | Float | Maximum observed IP packet size in bytes. |
| `*_median` | Float | 50th percentile (median) packet size. |
| `*_p25` | Float | 25th percentile (first quartile $Q_1$) packet size. |
| `*_p75` | Float | 75th percentile (third quartile $Q_3$) packet size. |

---

### Category 4: Inter-Arrival Time (IAT) Statistics (ML-Safe)
Computed independently for:
1. **Bidirectional inter-arrival times:** `bidir_iat_*`
2. **Forward inter-arrival times:** `fwd_iat_*`
3. **Backward inter-arrival times:** `bwd_iat_*`

IAT is calculated on monotonically sorted packet timestamps: $\text{IAT}_i = t_i - t_{i-1}$.
* If $n < 2$, all IAT statistics default to $0.0$.

| Metric Suffix | Type | Description & Derivation |
| :--- | :---: | :--- |
| `*_mean` | Float | Mean inter-arrival time in seconds. |
| `*_std` | Float | Jitter / standard deviation of inter-arrival times. |
| `*_min` | Float | Minimum inter-arrival time in seconds. |
| `*_max` | Float | Maximum silence interval between consecutive packets. |
| `*_median` | Float | Median inter-arrival time. |
| `*_p25` | Float | 25th percentile inter-arrival time. |
| `*_p75` | Float | 75th percentile inter-arrival time. |

---

### Category 5: Protocol-Specific Behavior (TCP / UDP) (ML-Safe)

| Column Name | Type | Description & Networking Significance |
| :--- | :---: | :--- |
| `tcp_syn_count` | Integer | Total SYN packets. Indicates connection setup attempts. |
| `tcp_ack_count` | Integer | Total ACK packets. Reflects acknowledgment frequency. |
| `tcp_fin_count` | Integer | Total FIN packets. Indicates orderly connection termination. |
| `tcp_rst_count` | Integer | Total RST packets. Indicates connection aborts or rejections. |
| `tcp_psh_count` | Integer | Total PSH packets. Indicates immediate data push to application layer. |
| `tcp_urg_count` | Integer | Total URG packets. Indicates out-of-band urgent data. |
| `tcp_win_min` | Float | Minimum TCP advertised receive window size observed. |
| `tcp_win_max` | Float | Maximum TCP advertised receive window size observed. |
| `tcp_win_mean` | Float | Average TCP advertised receive window size. |
| `tcp_win_std` | Float | Standard deviation of TCP window fluctuations (reflects congestion control). |
| `is_udp` | Integer | Binary indicator (1 if UDP transport, 0 otherwise). |
| `udp_packet_count` | Integer | Total UDP packets in flow. |
| `udp_byte_count` | Integer | Total UDP bytes transferred. |

---

### Category 6: Traffic Burst Dynamics (ML-Safe)
A **burst** is defined as an unbroken sequence of consecutive packets traveling in the same direction before the direction alternates (e.g., $F, F, F \to$ forward burst of 3 packets; $B, B \to$ backward burst of 2 packets). This directly models application-layer request/response turn-taking (ping-pong) and streaming chunks.

| Column Name | Type | Description |
| :--- | :---: | :--- |
| `fwd_burst_count` | Integer | Number of distinct forward packet bursts. |
| `bwd_burst_count` | Integer | Number of distinct backward packet bursts. |
| `fwd_burst_pkts_mean` | Float | Average number of packets per forward burst. |
| `bwd_burst_pkts_mean` | Float | Average number of packets per backward burst. |
| `fwd_burst_bytes_mean` | Float | Average bytes transferred per forward burst. |
| `bwd_burst_bytes_mean` | Float | Average bytes transferred per backward burst. |
| `fwd_burst_pkts_max` | Integer | Maximum packets observed in a single forward burst. |
| `bwd_burst_pkts_max` | Integer | Maximum packets observed in a single backward burst. |
| `fwd_burst_bytes_max` | Integer | Maximum bytes observed in a single forward burst. |
| `bwd_burst_bytes_max` | Integer | Maximum bytes observed in a single backward burst. |

---

### Category 7: Early Classification Features (Edge Router Ready) (ML-Safe)
To support ultra-fast QoS classification at edge routers without waiting for entire flows to finish, features are computed on the initial $N \in \{5, 10, 15, 20\}$ packets of each flow.

For each $N \in \{5, 10, 15, 20\}$:
* `early_N_complete`: Binary flag ($1$ if flow reached $\ge N$ packets; $0$ if flow finished earlier).
* `early_N_duration`: Elapsed time from packet $1$ to packet $\min(N, \text{total\_packets})$.
* `early_N_bytes`: Total IP bytes across the first $\min(N, \text{total\_packets})$ packets.
* `early_N_fwd_pkts`: Forward packets among the first $\min(N, \text{total\_packets})$ packets.
* `early_N_bwd_pkts`: Backward packets among the first $\min(N, \text{total\_packets})$ packets.
* `early_N_avg_size`: Average packet size across the first $\min(N, \text{total\_packets})$ packets.

---

## 4. Summary of Feature Counts

* **Metadata / Audit Columns:** 14 columns
* **ML-Safe Behavioral Feature Columns:** 87 columns
* **Total Columns per Flow Record:** 101 columns
