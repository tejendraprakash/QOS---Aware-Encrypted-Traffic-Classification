#!/usr/bin/env python3
"""
scripts/audit_simulation_statistics.py

Audits the clean simulation metrics across all 3 regimes (Low, Moderate, High)
and 3 policies (Baseline FIFO, Clean Predicted QoS, Oracle QoS) with paired
statistical tests, confidence intervals, Cohen's d effect sizes, and SLA compliance.
"""

import sys
import numpy as np
import pandas as pd
from scipy import stats

# 3GPP TS 23.501 SLA Thresholds
SLA_DELAY_MS = {"Real-Time": 100.0, "Streaming": 300.0, "Bulk Data": 300.0}
SLA_LOSS_PCT = {"Real-Time": 1.0, "Streaming": 0.01, "Bulk Data": 1.0}

def compute_cohens_d(x, y):
    """Computes paired Cohen's d effect size."""
    diff = np.array(x) - np.array(y)
    n = len(diff)
    if n < 2 or np.std(diff, ddof=1) == 0:
        return 0.0
    return float(np.mean(diff) / np.std(diff, ddof=1))

def compute_ci(data, confidence=0.95):
    """Computes mean and 95% confidence interval half-width."""
    arr = np.array(data, dtype=float)
    n = len(arr)
    mean = float(np.mean(arr))
    if n < 2:
        return mean, 0.0
    sem = stats.sem(arr)
    h = sem * stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return mean, float(h)

def run_statistical_audit():
    print("=" * 80)
    print("PHASE 6D.1: INDEPENDENT STATISTICAL AUDIT OF CLEAN QOS SIMULATION")
    print("=" * 80)
    
    # 1. Load Phase 7 raw run metrics (FIFO and Oracle)
    df_p7 = pd.read_csv("output/phase7/run_metrics.csv")
    df_p7_fifo = df_p7[df_p7["Scenario"] == "Baseline_FIFO"].copy()
    df_p7_oracle = df_p7[df_p7["Scenario"] == "Oracle_QoS"].copy()
    
    # 2. Load Clean Predicted QoS metrics
    clean_file = "output/phase6d/clean_predicted_qos_all_regimes.csv"
    try:
        df_clean = pd.read_csv(clean_file)
    except FileNotFoundError:
        # Fall back to High congestion only if full file not ready
        clean_file = "output/phase6d/clean_qos_simulation_run_metrics.csv"
        df_clean = pd.read_csv(clean_file)
        
    print(f"Loaded Clean Predicted QoS runs from: {clean_file} ({len(df_clean)} records)")
    print(f"Regimes present in clean dataset: {df_clean['Congestion_Regime'].unique()}")
    
    # Combine into unified dataframe
    df_unified = pd.concat([df_p7_fifo, df_clean, df_p7_oracle], ignore_index=True)
    
    # Audit across all regimes
    regimes = [r for r in ["Low", "Moderate", "High"] if r in df_clean["Congestion_Regime"].unique()]
    classes = ["Real-Time", "Streaming", "Bulk Data"]
    
    summary_rows = []
    
    for regime in regimes:
        print(f"\n" + "#" * 80)
        print(f"CONGESTION REGIME: {regime.upper()}")
        print("#" * 80)
        
        for cls in classes:
            print(f"\n--- QoS Class: {cls} ({regime} Congestion) ---")
            
            sub_fifo = df_p7_fifo[(df_p7_fifo["Congestion_Regime"] == regime) & (df_p7_fifo["QoS_Class"] == cls)].sort_values("Seed")
            sub_pred = df_clean[(df_clean["Congestion_Regime"] == regime) & (df_clean["QoS_Class"] == cls)].sort_values("Seed")
            sub_oracle = df_p7_oracle[(df_p7_oracle["Congestion_Regime"] == regime) & (df_p7_oracle["QoS_Class"] == cls)].sort_values("Seed")
            
            # Metrics to compare
            for metric in ["Mean_Delay_ms", "P95_Delay_ms", "Packet_Loss_Rate_Pct", "Goodput_Mbps", "Jitter_ms"]:
                fifo_vals = sub_fifo[metric].values
                pred_vals = sub_pred[metric].values
                oracle_vals = sub_oracle[metric].values
                
                m_fifo, ci_fifo = compute_ci(fifo_vals)
                m_pred, ci_pred = compute_ci(pred_vals)
                m_oracle, ci_oracle = compute_ci(oracle_vals)
                
                # Paired tests (Pred vs FIFO)
                t_stat, p_val_t = stats.ttest_rel(pred_vals, fifo_vals)
                try:
                    w_stat, p_val_w = stats.wilcoxon(pred_vals, fifo_vals)
                except ValueError:
                    p_val_w = 1.0 # zero diffs
                    
                d = compute_cohens_d(pred_vals, fifo_vals)
                pct_change = ((m_pred - m_fifo) / (m_fifo + 1e-9)) * 100.0
                
                print(f"  {metric:<22}: FIFO = {m_fifo:6.2f} +/- {ci_fifo:4.2f} | CleanPred = {m_pred:6.2f} +/- {ci_pred:4.2f} | Delta = {pct_change:+6.1f}% | p(t)={p_val_t:.4f} | d={d:+5.2f}")
                
                summary_rows.append({
                    "Regime": regime,
                    "Class": cls,
                    "Metric": metric,
                    "FIFO_Mean": round(m_fifo, 3),
                    "FIFO_CI95": round(ci_fifo, 3),
                    "Pred_Mean": round(m_pred, 3),
                    "Pred_CI95": round(ci_pred, 3),
                    "Oracle_Mean": round(m_oracle, 3),
                    "Oracle_CI95": round(ci_oracle, 3),
                    "Delta_Pct": round(pct_change, 2),
                    "p_value_paired_t": round(p_val_t, 4),
                    "p_value_wilcoxon": round(p_val_w, 4),
                    "Cohens_d": round(d, 3)
                })
                
            # SLA Compliance Audit
            print(f"  [SLA CHECK: {cls}]")
            delay_threshold = SLA_DELAY_MS[cls]
            loss_threshold = SLA_LOSS_PCT[cls]
            
            # Check for Pred
            pred_p95 = sub_pred["P95_Delay_ms"].values
            pred_loss = sub_pred["Packet_Loss_Rate_Pct"].values
            delay_pass_pred = bool(np.all(pred_p95 <= delay_threshold))
            loss_pass_pred = bool(np.all(pred_loss <= loss_threshold))
            full_sla_pred = delay_pass_pred and loss_pass_pred
            
            # Check for FIFO
            fifo_p95 = sub_fifo["P95_Delay_ms"].values
            fifo_loss = sub_fifo["Packet_Loss_Rate_Pct"].values
            delay_pass_fifo = bool(np.all(fifo_p95 <= delay_threshold))
            loss_pass_fifo = bool(np.all(fifo_loss <= loss_threshold))
            full_sla_fifo = delay_pass_fifo and loss_pass_fifo
            
            print(f"    FIFO:      Delay Compliance = {delay_pass_fifo} (Max P95={np.max(fifo_p95):.1f}ms <= {delay_threshold}ms) | Loss Compliance = {loss_pass_fifo} (Max Loss={np.max(fifo_loss):.2f}% <= {loss_threshold}%) | Overall SLA = {full_sla_fifo}")
            print(f"    CleanPred: Delay Compliance = {delay_pass_pred} (Max P95={np.max(pred_p95):.1f}ms <= {delay_threshold}ms) | Loss Compliance = {loss_pass_pred} (Max Loss={np.max(pred_loss):.2f}% <= {loss_threshold}%) | Overall SLA = {full_sla_pred}")

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv("output/phase6d/clean_simulation_statistical_audit.csv", index=False)
    print(f"\n[DONE] Saved statistical audit summary to output/phase6d/clean_simulation_statistical_audit.csv")

if __name__ == "__main__":
    run_statistical_audit()
