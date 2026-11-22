# Advanced Research Lab (ARL) — Risks and Issues Log

**Project:** QoS-Aware Encrypted Network Traffic Classification for Next-Generation Networks  
**Lead Research Orchestrator:** AI Research Lab Manager  
**Last Updated:** November 2026  

---

## Active Scientific, Implementation, and Data Risks

| Risk ID | Risk Category | Severity / Impact | Likelihood | Risk Description & Failure Mode | Mitigation Strategy & Status |
| :---: | :---: | :---: | :---: | :--- | :--- |
| **RSK-01** | Implementation | High | Medium | **TCP CUBIC Open-Loop Window Explosion:** If `last_loss_time` and `w_max` are uncoupled from queue drops, $cwnd$ grows cubically without bound ($W \to \infty$), instantly crashing simulation queues. | **MITIGATED:** RFC 8312 closed-loop architecture implemented in `traffic_generation_final_specification.md` with explicit queue loss feedback and multiplicative decrease. |
| **RSK-02** | Scientific / Modeling | High | High | **Unpoliced TCP Congestion Squeezing:** Elastic TCP Bulk flows will greedily consume all spare link capacity (~70 Mbps) under Low Congestion ($\rho = 0.50$), destroying the target offered load. | **MITIGATED:** Implemented Dual-Domain benchmark: Token Bucket Filter ($8.0\text{ Mbps}$) for controlled capacity tests, and Poisson file transfers for stress tests. |
| **RSK-03** | Methodological | Medium | High | **Empty-Queue Simulation Transients:** Starting simulation from empty queue ($Q(0) = 0$) skews tail latency and loss rate downwards during initial buffer fill. | **MITIGATED:** Enforced $60.0\text{ s}$ Welch warm-up period; discard $[0, 60\text{ s}]$ and record over $[60\text{ s}, 360\text{ s}]$. |
| **RSK-04** | Statistical | High | High | **Discrete Multimodal Distribution KS Invalidation:** Applying continuous 2-sample KS test to multimodal packet sizes with $N > 1000$ guarantees false rejection ($p \ll 0.001$). | **MITIGATED:** Replaced KS with Wasserstein-1 distance ($W_1 \le 35\text{ B}$), JSD ($\le 0.05$), and 6-bin Chi-Square in 4-Gate Protocol. |
| **RSK-05** | ML / Interface | Medium | Medium | **Covariate Shift on `early_5_complete`:** In raw dataset, 82.2% of flows had $<5$ packets. In continuous sustained simulation, all flows reach 5 packets (`early_5_complete = 1.0`). | **MITIGATED:** Validated that Early-5 model maintains high discriminative power on packet size and duration alone; online extractor explicitly passes `1.0`. |
| **RSK-06** | Computational | Low | Medium | **Event Heap Contention on Multi-Flow Scaling:** Simulating thousands of concurrent packets over long durations may induce high memory consumption or slow execution. | **MITIGATED:** Python discrete-event engine optimized with `heapq` and modular generators; executes 360s simulation in $<10\text{ seconds}$ per run. |
