#!/usr/bin/env python3
"""
scripts/audit3_and_4_qos_statistical_expansion.py

Executes:
1. AUDIT 4: Confidence-Aware QoS at Low (50%) and Moderate (85%) congestion across 5 seeds,
   combining with High (125%) to produce the complete 3-congestion comparison table.
   Documents tau=0.55 selection rationale.
2. AUDIT 3: QoS Statistical Robustness with 15 Seeds (parallelized across 4 workers).
   Computes Shapiro-Wilk normality, 95% CIs, Cohen's d with 95% CIs, and 5-seed sufficiency.
"""

import sys
sys.path.append("scripts")
import os
import time
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from multiprocessing import Pool
import simulate_phase7_qos as sim_mod
from test_confidence_aware_qos_simulation import ConfidenceAwareSimulator, cond_probs

# Clean Empirical Confusion Matrix (Phase 6D)
CONFUSION_MATRIX_CLEAN = {
    "Bulk Data": {"Bulk Data": 0.4902, "Real-Time": 0.4588, "Streaming": 0.0510},
    "Real-Time": {"Bulk Data": 0.2050, "Real-Time": 0.5170, "Streaming": 0.2780},
    "Streaming": {"Bulk Data": 0.0167, "Real-Time": 0.3434, "Streaming": 0.6399}
}
sim_mod.CONFUSION_MATRIX_EARLY5_XGB = CONFUSION_MATRIX_CLEAN

def run_single_sim(args):
    """Worker function for parallel simulation."""
    sim_type, scenario, regime, seed = args
    if sim_type == "confidence":
        sim = ConfidenceAwareSimulator(scenario=scenario, congestion_regime=regime, seed=seed)
    else:
        sim = sim_mod.NetworkSimulator(scenario=scenario, congestion_regime=regime, seed=seed)
    sim.run()
    res = sim_mod.compute_run_metrics(sim)
    for r in res:
        r["Sim_Type"] = sim_type
        r["Scenario"] = scenario
        r["Congestion_Regime"] = regime
        r["Seed"] = seed
    return res

def compute_cohens_d_with_ci(x, y, n_boot=2000):
    """Computes paired Cohen's d with 95% bootstrap confidence interval."""
    diff = np.array(x) - np.array(y)
    n = len(diff)
    if n < 2 or np.std(diff, ddof=1) == 0:
        return 0.0, (0.0, 0.0)
    d = float(np.mean(diff) / np.std(diff, ddof=1))
    
    # Bootstrap CI
    rng = np.random.default_rng(42)
    boot_ds = []
    for _ in range(n_boot):
        sample = rng.choice(diff, size=n, replace=True)
        s_std = np.std(sample, ddof=1)
        if s_std > 0:
            boot_ds.append(float(np.mean(sample) / s_std))
        else:
            boot_ds.append(0.0)
    ci_low = float(np.percentile(boot_ds, 2.5))
    ci_high = float(np.percentile(boot_ds, 97.5))
    return round(d, 3), (round(ci_low, 3), round(ci_high, 3))

def main():
    print("=" * 80)
    print("AUDIT 3 & 4: CONFIDENCE-AWARE QOS ACROSS CONGESTIONS & 15-SEED ROBUSTNESS")
    print("=" * 80)
    
    out_dir = Path("output/phase6d3")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # -------------------------------------------------------------
    # PART A: AUDIT 4 - Confidence-Aware QoS at Low and Moderate Regimes
    # -------------------------------------------------------------
    print("\n>>> PART A (AUDIT 4): Running Confidence-Aware QoS at Low & Moderate Regimes...")
    seeds_5 = [42, 101, 202, 303, 404]
    tasks_conf = []
    for reg in ["Low", "Moderate"]:
        for s in seeds_5:
            tasks_conf.append(("confidence", "Confidence_Aware_QoS", reg, s))
            
    with Pool(processes=4) as pool:
        results_conf = pool.map(run_single_sim, tasks_conf)
        
    flat_conf = [item for sublist in results_conf for item in sublist]
    
    # Load High congestion confidence runs
    df_conf_high = pd.read_csv("output/phase6d2/confidence_aware_simulation_run_metrics.csv")
    for _, row in df_conf_high.iterrows():
        d = dict(row)
        d["Sim_Type"] = "confidence"
        flat_conf.append(d)
        
    df_all_conf = pd.DataFrame(flat_conf)
    df_all_conf.to_csv(out_dir / "audit4_confidence_aware_all_regimes.csv", index=False)
    
    print("\n=== AUDIT 4: CONFIDENCE-AWARE QOS ACROSS 3 CONGESTION REGIMES ===")
    for reg in ["Low", "Moderate", "High"]:
        print(f"\n--- Regime: {reg} Congestion ---")
        sub_reg = df_all_conf[df_all_conf["Congestion_Regime"] == reg]
        for cls_name in ["Real-Time", "Streaming", "Bulk Data"]:
            sub_cls = sub_reg[sub_reg["QoS_Class"] == cls_name]
            print(f"  {cls_name:<12}: Mean Delay={sub_cls['Mean_Delay_ms'].mean():.2f}ms | P95={sub_cls['P95_Delay_ms'].mean():.2f}ms | Loss={sub_cls['Packet_Loss_Rate_Pct'].mean():.3f}% | Goodput={sub_cls['Goodput_Mbps'].mean():.2f}Mbps")
            
    # -------------------------------------------------------------
    # PART B: AUDIT 3 - 15-Seed Expansion for High & Moderate Congestion
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PART B (AUDIT 3): 15-SEED EXPANSION & STATISTICAL SUFFICIENCY AUDIT")
    print("=" * 80)
    
    seeds_15 = [42, 101, 202, 303, 404, 505, 606, 707, 808, 909, 1001, 1102, 1203, 1304, 1405]
    print(f"Executing 15 paired seeds across Baseline FIFO and Clean Predicted QoS (High Congestion)...")
    
    # We already have seeds 1..5 for High; generate seeds 6..15
    new_seeds = seeds_15[5:]
    tasks_15 = []
    for s in new_seeds:
        tasks_15.append(("standard", "Baseline_FIFO", "High", s))
        tasks_15.append(("standard", "Predicted_QoS", "High", s))
        
    with Pool(processes=4) as pool:
        results_15 = pool.map(run_single_sim, tasks_15)
        
    flat_15 = [item for sublist in results_15 for item in sublist]
    df_new_15 = pd.DataFrame(flat_15)
    
    # Load original 5 seeds for High
    df_p7 = pd.read_csv("output/phase7/run_metrics.csv")
    df_p7_fifo_high = df_p7[(df_p7["Scenario"] == "Baseline_FIFO") & (df_p7["Congestion_Regime"] == "High")].copy()
    df_p7_fifo_high["Sim_Type"] = "standard"
    
    df_clean_pred = pd.read_csv("output/phase6d/clean_qos_simulation_run_metrics.csv")
    df_clean_pred_high = df_clean_pred[df_clean_pred["Congestion_Regime"] == "High"].copy()
    df_clean_pred_high["Sim_Type"] = "standard"
    df_clean_pred_high["Scenario"] = "Predicted_QoS"
    
    df_high_15 = pd.concat([df_p7_fifo_high, df_clean_pred_high, df_new_15], ignore_index=True)
    df_high_15.to_csv(out_dir / "audit3_15seed_high_congestion_runs.csv", index=False)
    
    print(f"Total 15-Seed High Congestion Records: {len(df_high_15)}")
    
    # Statistical Analysis: 5 seeds vs 15 seeds comparison
    stats_comparison = []
    print("\n--- STATISTICAL SUFFICIENCY & EFFECT SIZE AUDIT (5 vs 15 SEEDS) ---")
    
    for cls in ["Real-Time", "Bulk Data", "Streaming"]:
        for metric in ["Mean_Delay_ms", "Packet_Loss_Rate_Pct", "Goodput_Mbps"]:
            # 5 seeds
            f5_vals = df_high_15[(df_high_15["Scenario"] == "Baseline_FIFO") & (df_high_15["QoS_Class"] == cls) & (df_high_15["Seed"].isin(seeds_5))].sort_values("Seed")[metric].values
            p5_vals = df_high_15[(df_high_15["Scenario"] == "Predicted_QoS") & (df_high_15["QoS_Class"] == cls) & (df_high_15["Seed"].isin(seeds_5))].sort_values("Seed")[metric].values
            
            # 15 seeds
            f15_vals = df_high_15[(df_high_15["Scenario"] == "Baseline_FIFO") & (df_high_15["QoS_Class"] == cls)].sort_values("Seed")[metric].values
            p15_vals = df_high_15[(df_high_15["Scenario"] == "Predicted_QoS") & (df_high_15["QoS_Class"] == cls)].sort_values("Seed")[metric].values
            
            # Normality tests (Shapiro-Wilk)
            shapiro_f15 = stats.shapiro(f15_vals)[1] if len(f15_vals) >= 3 else 1.0
            shapiro_p15 = stats.shapiro(p15_vals)[1] if len(p15_vals) >= 3 else 1.0
            
            # 5-seed metrics
            m_f5, sem_f5 = np.mean(f5_vals), stats.sem(f5_vals)
            ci_f5 = sem_f5 * stats.t.ppf(0.975, len(f5_vals)-1)
            m_p5, sem_p5 = np.mean(p5_vals), stats.sem(p5_vals)
            ci_p5 = sem_p5 * stats.t.ppf(0.975, len(p5_vals)-1)
            d5, ci_d5 = compute_cohens_d_with_ci(p5_vals, f5_vals)
            p_val5 = stats.ttest_rel(p5_vals, f5_vals)[1]
            
            # 15-seed metrics
            m_f15, sem_f15 = np.mean(f15_vals), stats.sem(f15_vals)
            ci_f15 = sem_f15 * stats.t.ppf(0.975, len(f15_vals)-1)
            m_p15, sem_p15 = np.mean(p15_vals), stats.sem(p15_vals)
            ci_p15 = sem_p15 * stats.t.ppf(0.975, len(p15_vals)-1)
            d15, ci_d15 = compute_cohens_d_with_ci(p15_vals, f15_vals)
            p_val15 = stats.ttest_rel(p15_vals, f15_vals)[1]
            
            stats_comparison.append({
                "Class": cls,
                "Metric": metric,
                "5Seed_FIFO": f"{m_f5:.2f} +/- {ci_f5:.2f}",
                "5Seed_Pred": f"{m_p5:.2f} +/- {ci_p5:.2f}",
                "5Seed_d": f"{d5:+.2f} {ci_d5}",
                "5Seed_p": f"{p_val5:.4f}",
                "15Seed_FIFO": f"{m_f15:.2f} +/- {ci_f15:.2f}",
                "15Seed_Pred": f"{m_p15:.2f} +/- {ci_p15:.2f}",
                "15Seed_d": f"{d15:+.2f} {ci_d15}",
                "15Seed_p": f"{p_val15:.4f}",
                "Shapiro_Normal_p": f"{min(shapiro_f15, shapiro_p15):.3f}",
                "Conclusion_Consistent": (p_val5 < 0.05 and p_val15 < 0.05) or (p_val5 >= 0.05 and p_val15 >= 0.05)
            })
            print(f"  {cls:<10} {metric:<20}: 5-Seed d={d5:+.2f} (p={p_val5:.4f}) | 15-Seed d={d15:+.2f} CI={ci_d15} (p={p_val15:.4f}) | Consistent: {p_val5 < 0.05 == p_val15 < 0.05}")
            
    df_stats_comp = pd.DataFrame(stats_comparison)
    df_stats_comp.to_csv(out_dir / "audit3_5_vs_15_seed_statistical_comparison.csv", index=False)
    print("\n[DONE] Saved 15-seed robustness and 3-congestion confidence tables to output/phase6d3/")

if __name__ == "__main__":
    main()
