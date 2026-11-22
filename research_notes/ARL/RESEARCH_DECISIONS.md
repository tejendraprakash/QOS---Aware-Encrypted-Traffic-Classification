# Advanced Research Lab (ARL) — Research Decisions Log

**Project:** QoS-Aware Encrypted Network Traffic Classification for Next-Generation Networks  
**Lead Research Orchestrator:** AI Research Lab Manager  
**Last Updated:** November 2026  

---

## Formal Architectural Decisions

| Decision ID | Date | Decision Summary | Alternatives Considered | Evidence & Scientific Justification | Impact & Consequences |
| :---: | :---: | :--- | :--- | :--- | :--- |
| **DEC-001** | Nov 2026 | Focus primary benchmark on 3 classes (Track B: Real-Time, Streaming, Bulk Data). Exclude contaminated Browsing class. | Retain 4-class Track A (including Browsing). | Phase 4 audit showed Browsing in ISCX is contaminated with background streaming and bulk sync, causing label leakage and ambiguous QoS mappings. | Clean, defensible evaluation aligning with 3GPP 5QI 1, 4, 9 profiles. |
| **DEC-002** | Nov 2026 | Group-level 5-fold cross-validation grouped strictly on `capture_group`. | Standard random K-Fold or Stratified K-Fold. | Naive random splitting assigns packets/flows from the same capture session to both train and test, leaking fixed IP/port shortcuts. | Guarantees zero group leakage; reflects realistic generalization boundary. |
| **DEC-003** | Nov 2026 | Select Early-5 horizon as the primary candidate model for Phase 7 simulation. | Full-flow post-facto models; Early-10, Early-15, Early-20. | Phase 6B proved Early-5 achieves Macro F1 $= 0.8524$ within $<5\text{ ms}$, enabling immediate hardware queue assignment before head-of-line blocking. | Minimizes flow quarantine delay; enables realistic SDN/DiffServ switch integration. |
| **DEC-004** | Nov 2026 | Mandate closed-loop TCP CUBIC (RFC 8312) with drop feedback over open-loop packet injection. | Open-loop Poisson/CBR burst generators. | Agent 1 audit proved open-loop window inflation explodes to 13,984 packets in 34s, destroying simulation physics. | Enforces authentic TCP congestion dynamics, slow-start, and RTT pacing. |
| **DEC-005** | Nov 2026 | Replace continuous 2-sample KS test with Wasserstein-1, JSD, and 6-bin Chi-Square. | Standard KS test with $p > 0.05$. | Agent 6 audit demonstrated the large-sample $p$-value paradox ($N > 1000$ guarantees rejection on discrete packet point masses). | Establishes a statistically defensible, reproducible goodness-of-fit protocol. |
| **DEC-006** | Nov 2026 | Adopt Dual-Domain Benchmark (Rate-Policed Queueing vs. Elastic Stress Test). | Single unconstrained TCP test; single purely open-loop test. | Agent 4 audit demonstrated that 3 unpoliced TCP flows will greedily consume all spare bandwidth (~70 Mbps), destroying $\rho = 0.50$ baseline. | Domain 1 tests QoS scheduling under calibrated loads; Domain 2 tests resilience against bufferbloat. |
| **DEC-007** | Nov 2026 | Implement Coupled 4-State Markov Dialogue Model (ITU-T P.59) for Real-Time voice. | Independent Brady 2-state models on caller and callee. | Agent 3 audit proved independent talkers produce 16% double-talk (2.5x to 5x higher than human dialogue at 3%–7%). | Enforces natural turn-taking and exact 50/50 directional symmetry. |
| **DEC-008** | Nov 2026 | Adopt 60.0-second Welch warm-up horizon in discrete-event simulation. | Cold-start recording from $t = 0.0\text{ s}$. | Agent 6 audit demonstrated initial empty-queue transient severely underestimates queuing delay and packet loss under $\rho = 0.85$ and $1.25$. | Steady-state metrics recorded over $[60\text{ s}, 360\text{ s}]$ provide unskewed tail delay and drop rates. |
