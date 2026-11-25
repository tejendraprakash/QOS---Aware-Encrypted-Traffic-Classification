#!/usr/bin/env python3
"""
scripts/run_clean_qos_all_regimes.py

Runs Clean Predicted QoS for Low (50%) and Moderate (85%) congestion regimes
across the 5 standard seeds (42, 101, 202, 303, 404) to complete the full
3-regime x 3-policy paired evaluation matrix.
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
    print("RUNNING CLEAN PREDICTED QOS FOR LOW & MODERATE REGIMES")
    print("=" * 80)
    
    seeds = [42, 101, 202, 303, 404]
    regimes = ["Low", "Moderate"]
    all_metrics = []
    
    # First, load existing High congestion results if available
    high_file = "output/phase6d/clean_qos_simulation_run_metrics.csv"
    try:
        df_high = pd.read_csv(high_file)
        print(f"Loaded {len(df_high)} existing High Congestion records from {high_file}")
        for _, row in df_high.iterrows():
            all_metrics.append(dict(row))
    except Exception as e:
        print(f"Could not load High file: {e}")
        regimes.append("High")
        
    for regime in regimes:
        print(f"\n>>> REGIME: {regime} Congestion <<<")
        for s in seeds:
            print(f"  Running {regime} - Seed {s}...", flush=True)
            sim = sim_mod.NetworkSimulator(
                scenario="Predicted_QoS",
                congestion_regime=regime,
                seed=s
            )
            sim.run()
            res = sim_mod.compute_run_metrics(sim)
            for r in res:
                r["Scenario"] = "Clean_Predicted_QoS"
                all_metrics.append(r)
                print(f"    {r['QoS_Class']:<12}: Mean Delay={r['Mean_Delay_ms']:.2f}ms | Loss={r['Packet_Loss_Rate_Pct']:.3f}% | Goodput={r['Goodput_Mbps']:.2f}Mbps", flush=True)
                
    df_all = pd.DataFrame(all_metrics)
    out_path = "output/phase6d/clean_predicted_qos_all_regimes.csv"
    df_all.to_csv(out_path, index=False)
    print(f"\n[DONE] Saved complete Clean Predicted QoS run metrics to {out_path}")

if __name__ == "__main__":
    main()
