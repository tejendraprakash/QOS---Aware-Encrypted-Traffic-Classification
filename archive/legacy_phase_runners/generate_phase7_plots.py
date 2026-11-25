#!/usr/bin/env python3
"""
scripts/generate_phase7_plots.py

Generates publication-quality figures for Phase 7 results:
1. P95 Latency by Class, Scenario, and Regime (with 95% Confidence Intervals)
2. Packet Loss Rate (%) Comparison
3. Goodput (Mbps) Allocation & Starvation Relief
4. SLA Compliance Matrix Heatmap
"""

from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def main():
    plots_dir = Path("output/phase7/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    df_runs = pd.read_csv("output/phase7/run_metrics.csv")
    df_agg = pd.read_csv("output/phase7/aggregate_scenario_metrics.csv")
    df_sla = pd.read_csv("output/phase7/sla_compliance_summary.csv")

    sns.set_theme(style="whitegrid", font="sans-serif")
    palette = {"Baseline_FIFO": "#e74c3c", "Predicted_QoS": "#3498db", "Oracle_QoS": "#2ecc71"}
    policy_labels = {"Baseline_FIFO": "Baseline FIFO", "Predicted_QoS": "Early-5 Predicted QoS", "Oracle_QoS": "Oracle QoS (Upper Bound)"}

    # 1. P95 Latency Comparison
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    classes = ["Real-Time", "Streaming", "Bulk Data"]
    regime_order = ["Low", "Moderate", "High"]

    for i, qos_cls in enumerate(classes):
        ax = axes[i]
        subset = df_runs[df_runs["QoS_Class"] == qos_cls]
        sns.barplot(
            data=subset,
            x="Congestion_Regime",
            y="P95_Delay_ms",
            hue="Scenario",
            order=regime_order,
            palette=palette,
            ax=ax,
            errorbar=("ci", 95),
            capsize=0.1,
            err_kws={'linewidth': 1.5}
        )
        ax.set_title(f"Class: {qos_cls}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Congestion Regime", fontsize=11)
        ax.set_ylabel("P95 Latency (ms)", fontsize=11)
        if qos_cls == "Real-Time":
            ax.axhline(100.0, color="darkred", linestyle="--", linewidth=1.2, label="3GPP 5QI 1 SLA (100 ms)")
        if i == 0:
            ax.legend(title="Policy", fontsize=9, title_fontsize=10)
        else:
            if ax.get_legend():
                ax.get_legend().remove()

    plt.suptitle("Phase 7: P95 Delay (ms) Across Congestion Regimes (95% CI)", fontsize=15, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(plots_dir / "p95_delay_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved p95_delay_comparison.png")

    # 2. Packet Loss Rate Comparison
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    for i, qos_cls in enumerate(classes):
        ax = axes[i]
        subset = df_runs[df_runs["QoS_Class"] == qos_cls]
        sns.barplot(
            data=subset,
            x="Congestion_Regime",
            y="Packet_Loss_Rate_Pct",
            hue="Scenario",
            order=regime_order,
            palette=palette,
            ax=ax,
            errorbar=("ci", 95),
            capsize=0.1,
            err_kws={'linewidth': 1.5}
        )
        ax.set_title(f"Class: {qos_cls}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Congestion Regime", fontsize=11)
        ax.set_ylabel("Packet Loss Rate (%)", fontsize=11)
        if qos_cls == "Real-Time":
            ax.axhline(1.0, color="darkred", linestyle="--", linewidth=1.2, label="3GPP 5QI 1 SLA (1.0%)")
        if i == 0:
            ax.legend(title="Policy", fontsize=9, title_fontsize=10)
        else:
            if ax.get_legend():
                ax.get_legend().remove()

    plt.suptitle("Phase 7: Packet Loss Rate (%) Across Congestion Regimes (95% CI)", fontsize=15, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(plots_dir / "packet_loss_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved packet_loss_comparison.png")

    # 3. Goodput Comparison
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    for i, qos_cls in enumerate(classes):
        ax = axes[i]
        subset = df_runs[df_runs["QoS_Class"] == qos_cls]
        sns.barplot(
            data=subset,
            x="Congestion_Regime",
            y="Goodput_Mbps",
            hue="Scenario",
            order=regime_order,
            palette=palette,
            ax=ax,
            errorbar=("ci", 95),
            capsize=0.1,
            err_kws={'linewidth': 1.5}
        )
        ax.set_title(f"Class: {qos_cls}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Congestion Regime", fontsize=11)
        ax.set_ylabel("Goodput (Mbps)", fontsize=11)
        if i == 0:
            ax.legend(title="Policy", fontsize=9, title_fontsize=10)
        else:
            if ax.get_legend():
                ax.get_legend().remove()

    plt.suptitle("Phase 7: Goodput (Mbps) Across Congestion Regimes (Bulk Starvation Relief)", fontsize=15, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(plots_dir / "goodput_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved goodput_comparison.png")

    # 4. SLA Compliance Heatmap
    plt.figure(figsize=(9, 6))
    pivot_sla = df_sla.pivot(index=["Scenario", "Congestion_Regime"], columns=[], values=[]).copy() if False else None
    
    # Restructure df_sla for heatmap
    df_sla["Config"] = df_sla["Scenario"] + " (" + df_sla["Congestion_Regime"] + ")"
    order = [
        "Baseline_FIFO (Low)", "Baseline_FIFO (Moderate)", "Baseline_FIFO (High)",
        "Predicted_QoS (Low)", "Predicted_QoS (Moderate)", "Predicted_QoS (High)",
        "Oracle_QoS (Low)", "Oracle_QoS (Moderate)", "Oracle_QoS (High)"
    ]
    df_sla_ordered = df_sla.set_index("Config").reindex(order)
    heatmap_data = df_sla_ordered[["Real_Time_SLA_Compliance_Pct", "Bulk_Data_SLA_Compliance_Pct", "Composite_SLA_Compliance_Pct"]]
    heatmap_data.columns = ["Real-Time SLA (Delay<=100ms, Loss<=1%)", "Bulk Data SLA (Loss<=1%)", "Composite SLA Compliance"]

    sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="YlGnBu", cbar_kws={'label': 'Compliance (%)'}, vmin=0, vmax=100)
    plt.title("Phase 7: 3GPP TS 23.501 SLA Compliance Matrix (%)", fontsize=13, fontweight="bold", pad=15)
    plt.ylabel("Policy Configuration", fontsize=11)
    plt.tight_layout()
    plt.savefig(plots_dir / "sla_compliance_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved sla_compliance_matrix.png")

if __name__ == "__main__":
    main()
