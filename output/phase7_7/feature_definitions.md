# Phase 7.7 Feature Definitions

## Overview
Phase 7.7 introduces the **Robust 8-Feature Vector** for Early-5 encrypted network traffic classification. All features are strictly causal, payload-agnostic, and available within the first five observed packets of a flow.

---

## 1. Feature Specifications

| Feature Name | Type | Unit / Range | Mathematical Definition | Availability Time | Causal Verification & Security |
| :--- | :---: | :---: | :--- | :---: | :--- |
| `early_5_complete` | Integer (Binary) | {0, 1} | $\mathbb{I}(\text{total\_pkts} \ge 5)$ | Packet 5 (or flow termination) | Indicates whether flow window reached 5 packets. Does not leak total length beyond 5. |
| `early_5_duration` | Float | Seconds ($s$) | $t_{\min(N, 5)} - t_1$ | Packet 5 | Elapsed duration across observed packets $1 \dots \min(N, 5)$. Zero future packet usage. |
| `early_5_bytes` | Integer | Bytes | $\sum_{k=1}^{\min(N, 5)} \text{len}(p_k)$ | Packet 5 | Sum of wire lengths of packets $1 \dots \min(N, 5)$. Strictly causal. |
| `early_5_fwd_pkts` | Integer | Count ($[0, 5]$) | $\sum_{k=1}^{\min(N, 5)} \mathbb{I}(\text{dir}(p_k) = \text{FWD})$ | Packet 5 | Count of client-to-server packets. Strictly causal. |
| `early_5_bwd_pkts` | Integer | Count ($[0, 5]$) | $\sum_{k=1}^{\min(N, 5)} \mathbb{I}(\text{dir}(p_k) = \text{BWD})$ | Packet 5 | Count of server-to-client packets. Strictly causal. |
| `early_5_avg_size` | Float | Bytes | $\frac{\text{early\_5\_bytes}}{\min(N, 5)}$ | Packet 5 | Mean packet size across observed early packets. Derived exclusively from observed bytes. |
| `is_udp` | Float (Binary) | {0.0, 1.0} | $\mathbb{I}(\text{protocol} = 17)$ | Packet 1 | Layer 4 transport protocol header present in the very first IP packet. Zero payload inspection. |
| `early_5_fwd_ratio`| Float | $[0.0, 1.0]$ | $\frac{\text{early\_5\_fwd\_pkts}}{\max(\text{early\_5\_fwd\_pkts} + \text{early\_5\_bwd\_pkts}, 1)}$ | Packet 5 | Directional symmetry ratio in early window. Distinguishes symmetric conversational voice from asymmetric downloads. |

---

## 2. Explicitly Excluded Features (Leakage Prevention)
1. **IP Addresses & Port Numbers:** Excluded to prevent host/server identity memorization.
2. **Capture Group & Application Metadata:** Excluded to prevent environment target leakage.
3. **Full-Flow Statistics:** `total_packets`, `total_bytes`, `bidir_iat_*`, `packet_rate`, and `byte_rate` are full-flow aggregations and strictly prohibited at Early-5.
4. **Temporal Rate Calculations:** Early byte rates ($rac{B_5}{\Delta t}$) were evaluated in Phase 7.6 Exp 1 and Subagent 2, but rejected because they overfit to cross-group link bandwidth and RTT variations under Leave-One-Group-Out.
