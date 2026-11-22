# Advanced Research Lab (ARL) — Research Claims Ledger

**Project:** QoS-Aware Encrypted Network Traffic Classification for Next-Generation Networks  
**Lead Research Orchestrator:** AI Research Lab Manager  
**Last Updated:** November 2026 (Phase 7 Experimental Verification Complete)  

---

## Formal Classification of Research Claims

Each claim is formally classified as:
* **EV** — Experimentally Verified (backed by reproducible experimental data)
* **CV** — Code-Verified (independently validated through algorithmic execution)
* **SV** — Statistically Verified (backed by rigorous statistical hypothesis testing / CIs)
* **LS** — Literature Supported (backed by peer-reviewed literature or formal standards)
* **MD** — Mathematical Derivation (analytically proven from physical/queueing laws)
* **AS** — Explicit Modeling Assumption (documented prior / heuristic)
* **UV** — Unverified / Speculative (provisional hypothesis requiring testing)

---

## Master Ledger of Scientific Claims

| Claim ID | Formal Scientific Claim | Evidence Type | Source & Verification Evidence | Verification Status | Limitations & Boundary Conditions |
| :---: | :--- | :---: | :--- | :---: | :--- |
| **CLM-01** | Transport layer encryption blinds deep packet inspection, requiring statistical metadata classification. | **LS, EV** | RFC 8446 (TLS 1.3), RFC 9000 (QUIC), ISCX dataset audit (`dataset_audit.md`). | **VERIFIED** | Applicable only when payloads are encrypted; unencrypted headers remain visible. |
| **CLM-02** | Sustained encrypted VoIP flows exhibit an isochronous packetization mode at $50\text{ Hz}$ ($20\text{ ms}$). | **EV, LS** | `output/phase5/track_b_flows.csv`, ITU-T G.711, RFC 6716 Opus codec. | **VERIFIED** | Skewed by short keepalive flows in raw traces; holds strictly for active speech sessions. |
| **CLM-03** | Encrypted streaming video exhibits severe directional byte asymmetry ($\ge 75\%$ downlink bytes). | **EV, LS** | Track B sustained median fwd byte ratio $= 0.24$ (25th $= 0.05$); ISO/IEC 23009-1 DASH. | **VERIFIED** | Uplink is dominated by HTTP requests and pure ACKs; holds for client-server streaming. |
| **CLM-04** | Early flow classification within the first 5 packets enables QoS enforcement before head-of-line blocking. | **EV, CV** | Phase 6B Early-5 XGBoost benchmark (Macro F1 $= 0.8524$, duration $<5\text{ ms}$). | **VERIFIED** | Evaluated on static traces; confirmed inside closed-loop network simulation (Phase 7). |
| **CLM-05** | Classical Two-Sample KS test with $N > 1000$ packets produces false rejection ($p \ll 0.001$) on discrete distributions. | **MD, SV** | Mathematical derivation of large-$N$ paradox; `traffic_generation_validation_audit.md`. | **VERIFIED** | Requires continuous underlying CDFs; inapplicable to multimodal network packet point masses. |
| **CLM-06** | Open-loop TCP CUBIC simulation without drop feedback causes unbounded $cwnd \to 13,984$ explosion. | **MD, CV** | Mathematical analysis of RFC 8312 cubic equation without $W_{max}$ updates; Agent 1 report. | **VERIFIED** | Resolved by implementing closed-loop sequence tracking and queue drop feedback. |
| **CLM-07** | Classification-based priority queueing protects Real-Time voice from bufferbloat under high congestion ($\rho=1.25$). | **EV, SV** | Phase 7 simulation benchmark (`output/phase7/run_metrics.csv`): Real-Time loss drops from $8.03\%$ to $1.04\%$ ($-87.0\%$, $g = +8.07$); P95 delay drops from $70.06\text{ ms}$ to $51.76\text{ ms}$ ($-26.1\%$, $g = +8.28$). | **VERIFIED** | Evaluated on 100 Mbps bottleneck link with BDP buffer; holds across 5 random seeds. |
| **CLM-08** | Lognormal chunk sizing with $\mu = \ln(M)$ introduces a $+3.17\%$ upward volume bias via Jensen's inequality. | **MD, CV** | Mathematical derivation $\mathbb{E}[X] = M e^{\sigma^2/2}$; Monte Carlo verification ($N=100,000$). | **VERIFIED** | Offset by $\mu^* = \ln(M) - \sigma^2/2$. |
| **CLM-09** | 3GPP TS 23.501 specifies standardized Packet Delay Budget of $300\text{ ms}$ for 5QI 9 (Default Best Effort). | **LS** | 3GPP TS 23.501 Table 5.7.4-1. | **VERIFIED** | Corrects blueprint's misquote of "Unbounded" delay. |
| **CLM-10** | Sustained Bulk Data in Track B is $96.1\%$ TCP, not $76.1\%$ UDP. | **EV** | Empirical cross-tabulation of `output/phase5/track_b_flows.csv` on `total_packets >= 20`. | **VERIFIED** | Corrects transcription error in baseline Table 2.1. |
| **CLM-11** | Deficit Round Robin (DRR) scheduling prevents Bulk Data starvation under high video load, increasing goodput by $+416.9\%$. | **EV, SV** | Phase 7 simulation benchmark: Bulk Data goodput increases from $1.66\text{ Mbps}$ (FIFO) to $8.56\text{ Mbps}$ (Predicted QoS) under $\rho = 1.25$ ($g = -5.24$). | **VERIFIED** | Evaluated with DRR quantum allocation (60% Streaming, 40% Bulk Data). |
