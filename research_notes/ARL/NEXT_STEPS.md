# Advanced Research Lab (ARL) — Next Steps & Execution Roadmap

**Project:** QoS-Aware Encrypted Network Traffic Classification for Next-Generation Networks  
**Lead Research Orchestrator:** AI Research Lab Manager  
**Last Updated:** November 2026  

---

## 1. Immediate Highest-Priority Objective (P0 / P1)

**Execute Milestone 7.1:** Implement the approved, corrected, production-grade Phase 7 Network Simulation & QoS Evaluation Engine in `scripts/simulate_phase7_qos.py`, execute the 45 comparative simulation runs (3 Scenarios $\times$ 3 Congestion Regimes $\times$ 5 Multi-Seed Replications), and validate network performance metrics against the 4-Gate Statistical Protocol.

---

## 2. Dependency-Aware Execution Sequence

```
Step 1: Simulator Engine Refactoring (Squad C — Systems Engineering)
        - Update scripts/simulate_phase7_qos.py with corrected RealTimeFlowAgent (4-state Markov dialogue),
          StreamingFlowAgent (unbiased lognormal, 3-state buffer), and BulkDataFlowAgent (closed-loop RFC 8312 CUBIC).
        - Enforce 100 Mbps bottleneck capacity, BDP buffer = 625 KB, and Token-Bucket rate policing.
        - Incorporate Welch warm-up horizon (Twarm = 60s, measurement over [60s, 360s]).

Step 2: Classifier Interface Verification (Squad B — Machine Learning)
        - Confirm Early5OnlineFeatureExtractor schema and unscaled feature pass-through.
        - Verify empirical confusion matrix transition probabilities or live XGBoost inference hook.

Step 3: Multi-Scenario Experiment Execution (Squad C & Squad A)
        - Run 3 Scenarios: Baseline FIFO, Predicted QoS, Oracle Upper Bound.
        - Run 3 Congestion Regimes: Low (50%), Moderate (85%), High (125%).
        - Execute across 5 independent seeds (e.g. 42, 101, 202, 303, 404). Total = 45 runs.
        - Save raw packet logs and aggregate scenario metrics to output/phase7/.

Step 4: Statistical Validation & SLA Analysis (Squad D — Scientific Validation)
        - Update scripts/evaluate_phase7_results.py to compute 4-Gate goodness-of-fit metrics (W1, JSD).
        - Run Welch-ANOVA / paired Wilcoxon signed-rank tests and calculate Hedges' g effect sizes.
        - Generate 3GPP 5QI SLA compliance tables and delay/jitter CDFs.

Step 5: Master Synthesis & Research Publication Preparation (Squad E & Lab Manager)
        - Author research_notes/phase7/PHASE7_MASTER_REPORT.md.
        - Synthesize core empirical discoveries into LaTeX publication draft tables.
```

---

## 3. Recommended Research Squad Allocations

* **Squad C (Systems Engineering):** Assign Lead Simulation Engineer to execute `scripts/simulate_phase7_qos.py` refactoring.
* **Squad A (Networking Research):** Assign QoS Architect to verify bottleneck buffer occupancy and token-bucket shaper calibration.
* **Squad D (Scientific Validation):** Assign Statistical Scientist to execute `scripts/evaluate_phase7_results.py` and run Gates 1–4.
* **Squad E (Research Intelligence):** Monitor citation alignment with 3GPP TS 23.501 and RFC 8312.

---

## 4. Completion & Quality Gates

* **Gate A (Code Correctness):** Syntax compilation zero-error guarantee; test execution of 1 complete run completes in $<15\text{ seconds}$.
* **Gate B (Flow Conservation):** Transmitted packets $=$ Received packets $+$ Dropped packets across all 45 runs.
* **Gate C (Statistical Rigor):** $p < 0.01$ and Hedges' $g \ge 0.80$ for Real-Time delay reduction under High Congestion ($\rho = 1.25$).
