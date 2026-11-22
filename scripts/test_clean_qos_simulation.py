#!/usr/bin/env python3
"""
scripts/test_clean_qos_simulation.py

Evaluates High Congestion (125% load) under the clean empirical confusion matrix
from Phase 6D (clean_best_confusion_matrix.csv).
Runs all 5 seeds (42, 101, 202, 303, 404) for Clean Predicted QoS.
"""

import sys
sys.path.append("scripts")
import numpy as np
import pandas as pd
import simulate_phase7_qos as sim_mod

# Clean Empirical Confusion Matrix (Phase 6D)
CONFUSION_MATRIX_CLEAN = {
    "Bulk Data": {"Bulk Data": 0.4902, "Real-Time": 0.4588, "Streaming": 0.0510},
    "Real-Time": {"Bulk Data": 0.2050, "Real-Time": 0.5170, "Streaming": 0.2780},
    "Streaming": {"Bulk Data": 0.0167, "Real-Time": 0.3434, "Streaming": 0.6399}
}

sim_mod.CONFUSION_MATRIX_EARLY5_XGB = CONFUSION_MATRIX_CLEAN

def main():
    print("=" * 80)
    print("PHASE 8: EVALUATING CLEAN EMPIRICAL CLASSIFIER IN NETWORK SIMULATION")
    print("Regime: High Congestion (rho = 1.25), Link: 100 Mbps, 5 Seeds")
    print("=" * 80)
    
    seeds = [42, 101, 202, 303, 404]
    all_metrics = []
    
    for s in seeds:
        print(f"\n--- Running Seed {s} ---")
        sim = sim_mod.NetworkSimulator(
            scenario="Predicted_QoS",
            congestion_regime="High",
            seed=s
        )
        sim.run()
        res = sim_mod.compute_run_metrics(sim)
        for r in res:
            r["Scenario"] = "Clean_Predicted_QoS"
            all_metrics.append(r)
            print(f"  {r['QoS_Class']:<12}: Mean Delay={r['Mean_Delay_ms']:.2f}ms | P95={r['P95_Delay_ms']:.2f}ms | Loss={r['Packet_Loss_Rate_Pct']:.3f}% | Goodput={r['Goodput_Mbps']:.2f}Mbps")
            
    df_clean_qos = pd.DataFrame(all_metrics)
    df_clean_qos.to_csv("output/phase6d/clean_qos_simulation_run_metrics.csv", index=False)
    
    print("\n" + "=" * 80)
    print("AGGREGATE CLEAN PREDICTED QOS METRICS (HIGH CONGESTION, N=5 SEEDS)")
    print("=" * 80)
    grouped = df_clean_qos.groupby("QoS_Class")
    for cls, grp in grouped:
        print(f"\nClass: {cls}")
        print(f"  Mean Delay:  {grp['Mean_Delay_ms'].mean():.2f} +/- {grp['Mean_Delay_ms'].std():.2f} ms")
        print(f"  P95 Delay:   {grp['P95_Delay_ms'].mean():.2f} +/- {grp['P95_Delay_ms'].std():.2f} ms")
        print(f"  Packet Loss: {grp['Packet_Loss_Rate_Pct'].mean():.3f} +/- {grp['Packet_Loss_Rate_Pct'].std():.3f} %")
        print(f"  Goodput:     {grp['Goodput_Mbps'].mean():.2f} +/- {grp['Goodput_Mbps'].std():.2f} Mbps")
        print(f"  Jitter:      {grp['Jitter_ms'].mean():.2f} +/- {grp['Jitter_ms'].std():.2f} ms")

if __name__ == "__main__":
    main()
