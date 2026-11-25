"""
Plotly Chart Factory Functions for QoS Research Dashboard.

Modular, reusable, and styled with an academic aesthetic ('plotly_white' template,
clean publication-ready palettes, responsive layouts, explicit units and labels).
All factory functions return `plotly.graph_objects.Figure` and gracefully handle
empty or missing datasets.
"""

from typing import Optional, List, Dict, Any, Union
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Design System & Aesthetic Constants
# ---------------------------------------------------------------------------

THEME_TEMPLATE = "plotly_white"

# Standard QoS class colors (Professional Enterprise Theme)
CLASS_COLORS = {
    "Real-Time": "#2563EB",   # Primary Blue
    "Streaming": "#0D9488",   # Teal
    "Bulk Data": "#D97706",   # Warning / Amber
}

SCENARIO_COLORS = {
    "Baseline_FIFO": "#64748B",  # Neutral Slate Gray
    "Predicted_QoS": "#2563EB",  # Primary Blue
    "Oracle_QoS": "#16A34A",     # Success Green
    "Clean_Predicted_QoS": "#2563EB",
}

REGIME_ORDER = ["Low", "Moderate", "High"]
CLASS_ORDER = ["Real-Time", "Streaming", "Bulk Data"]

# Standard 3GPP SLA threshold definitions
SLA_THRESHOLDS = {
    "Real-Time": {
        "p95_delay_max_ms": 100.0,
        "loss_max_pct": 1.0,
        "queue": "Queue 0 (Strict Priority)",
        "cir_mbps": 25.0
    },
    "Streaming": {
        "p95_delay_max_ms": 300.0,
        "loss_max_pct": 0.01,
        "queue": "Queue 1 (DRR 60% Weight)",
        "weight_pct": 60.0
    },
    "Bulk Data": {
        "p95_delay_max_ms": 300.0,
        "loss_max_pct": 1.0,
        "queue": "Queue 2 (DRR 40% Weight)",
        "weight_pct": 40.0
    }
}


def _empty_figure(message: str = "No data available") -> go.Figure:
    """Return an empty figure with a clean centered annotation."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=14, color="#64748B", family="Inter, sans-serif")
    )
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=40, r=40, t=40, b=40),
        height=350,
    )
    return fig


def _apply_theme(
    fig: go.Figure,
    title: Optional[str] = None,
    xaxis_title: Optional[str] = None,
    yaxis_title: Optional[str] = None,
    legend_title: Optional[str] = None,
    height: Optional[int] = None,
    show_legend: bool = True
) -> go.Figure:
    """Apply consistent publication layout styling across all figures."""
    layout_update: Dict[str, Any] = dict(
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(family="Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif", size=12, color="#0F172A"),
        title=dict(
            text=f"<b>{title}</b>" if title else None,
            font=dict(size=15, color="#0F172A"),
            x=0.02,
            xanchor="left",
            y=0.96,
        ) if title else None,
        margin=dict(l=55, r=35, t=55 if title else 35, b=45),
        showlegend=show_legend,
        legend=dict(
            title=dict(text=legend_title or "", font=dict(size=11, color="#475569")),
            bgcolor="#FFFFFF",
            bordercolor="#E2E8F0",
            borderwidth=1,
            orientation="h" if not legend_title else "v",
            yanchor="bottom" if not legend_title else "top",
            y=1.02 if not legend_title else 0.98,
            xanchor="right",
            x=1.0
        ),
        hoverlabel=dict(
            bgcolor="#0F172A",
            font_size=12,
            font_family="Inter, sans-serif",
            font_color="#FFFFFF",
            bordercolor="#E2E8F0"
        )
    )
    if height:
        layout_update["height"] = height

    fig.update_layout(**layout_update)
    if xaxis_title:
        fig.update_xaxes(title_text=xaxis_title, title_font=dict(size=12, color="#475569"), tickfont=dict(color="#475569"))
    if yaxis_title:
        fig.update_yaxes(title_text=yaxis_title, title_font=dict(size=12, color="#475569"), tickfont=dict(color="#475569"))
    
    fig.update_xaxes(showgrid=True, gridcolor="#E2E8F0", zeroline=False, tickfont=dict(color="#475569"))
    fig.update_yaxes(showgrid=True, gridcolor="#E2E8F0", zeroline=False, tickfont=dict(color="#475569"))
    return fig


# ---------------------------------------------------------------------------
# 1. Class Distribution
# ---------------------------------------------------------------------------

def plot_class_distribution(
    df: Optional[pd.DataFrame],
    class_col: str = "qos_class",
    chart_type: str = "donut",
    title: Optional[str] = "Traffic Class Distribution"
) -> go.Figure:
    """
    Generate a publication-ready Donut or Bar chart of QoS traffic classes.

    Parameters
    ----------
    df : Optional[pd.DataFrame]
        Flow dataframe containing class labels.
    class_col : str
        Column name for traffic class (e.g. 'qos_class', 'pred_class').
    chart_type : str
        'donut' or 'bar'.
    title : Optional[str]
        Figure title.
    """
    if df is None or df.empty:
        return _empty_figure("No flow data available for class distribution")
    
    # Fallback column detection if requested column not found
    if class_col not in df.columns:
        for candidate in ["qos_class", "pred_class", "QoS_Class", "Class", "label"]:
            if candidate in df.columns:
                class_col = candidate
                break
        else:
            return _empty_figure(f"Class column '{class_col}' not found in dataframe")

    counts = df[class_col].value_counts().reset_index()
    counts.columns = ["Class", "Count"]
    counts["Percentage"] = counts["Count"] / counts["Count"].sum() * 100.0

    colors = [CLASS_COLORS.get(cls, "#64748B") for cls in counts["Class"]]

    if chart_type.lower() == "donut":
        fig = go.Figure(
            data=[
                go.Pie(
                    labels=counts["Class"],
                    values=counts["Count"],
                    hole=0.55,
                    marker=dict(colors=colors, line=dict(color="#FFFFFF", width=2)),
                    textinfo="label+percent",
                    texttemplate="<b>%{label}</b><br>%{percent:.1%}",
                    hovertemplate="<b>%{label}</b><br>Flows: %{value:,}<br>Share: %{percent:.2%}<extra></extra>",
                    sort=False,
                    direction="clockwise"
                )
            ]
        )
        total_flows = counts["Count"].sum()
        fig.add_annotation(
            text=f"<b>{total_flows:,}</b><br><span style='font-size:10px;color:#64748B'>Total Flows</span>",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color="#1E293B", family="Arial")
        )
        fig = _apply_theme(fig, title=title, show_legend=True)
    else:
        fig = go.Figure(
            data=[
                go.Bar(
                    x=counts["Class"],
                    y=counts["Count"],
                    marker=dict(color=colors, line=dict(color="#FFFFFF", width=1.5)),
                    text=[f"{cnt:,} ({pct:.1f}%)" for cnt, pct in zip(counts["Count"], counts["Percentage"])],
                    textposition="outside",
                    hovertemplate="<b>%{x}</b><br>Count: %{y:,}<br>Share: %{customdata:.2f}%<extra></extra>",
                    customdata=counts["Percentage"],
                    width=0.45
                )
            ]
        )
        fig = _apply_theme(fig, title=title, xaxis_title="QoS Class", yaxis_title="Flow Count", show_legend=False)

    return fig


# ---------------------------------------------------------------------------
# 2. Prediction Confidence Distribution
# ---------------------------------------------------------------------------

def plot_confidence_distribution(
    df: Optional[pd.DataFrame],
    confidence_col: str = "confidence",
    class_col: Optional[str] = "pred_class",
    title: Optional[str] = "Model Prediction Confidence Distribution"
) -> go.Figure:
    """
    Generate histogram and KDE / box distribution of prediction confidence.

    Parameters
    ----------
    df : Optional[pd.DataFrame]
        Flow dataframe with model confidence values.
    confidence_col : str
        Column name holding confidence values [0.0, 1.0].
    class_col : Optional[str]
        Optional class column to split histograms/box plots by class.
    title : Optional[str]
        Figure title.
    """
    if df is None or df.empty:
        return _empty_figure("No confidence data available")

    if confidence_col not in df.columns:
        for candidate in ["confidence", "label_confidence", "prob_max", "score"]:
            if candidate in df.columns:
                confidence_col = candidate
                break
        else:
            return _empty_figure(f"Confidence column '{confidence_col}' not found")

    clean_df = df.dropna(subset=[confidence_col]).copy()
    if clean_df.empty:
        return _empty_figure("No valid numeric confidence values")

    fig = go.Figure()

    has_classes = class_col and class_col in clean_df.columns
    if has_classes:
        unique_classes = [c for c in CLASS_ORDER if c in clean_df[class_col].unique()]
        remaining = [c for c in clean_df[class_col].unique() if c not in unique_classes]
        all_classes = unique_classes + remaining

        for cls in all_classes:
            sub = clean_df[clean_df[class_col] == cls]
            color = CLASS_COLORS.get(cls, "#4B5563")
            fig.add_trace(
                go.Histogram(
                    x=sub[confidence_col],
                    name=str(cls),
                    xbins=dict(start=0.0, end=1.0, size=0.025),
                    opacity=0.65,
                    marker=dict(color=color, line=dict(color="#FFFFFF", width=0.5)),
                    hovertemplate=f"<b>{cls}</b><br>Confidence: %{{x:.2f}}<br>Count: %{{y:,}}<extra></extra>"
                )
            )
    else:
        fig.add_trace(
            go.Histogram(
                x=clean_df[confidence_col],
                name="Confidence",
                xbins=dict(start=0.0, end=1.0, size=0.025),
                marker=dict(color="#1D3557", line=dict(color="#FFFFFF", width=0.5)),
                hovertemplate="Confidence: %{x:.2f}<br>Count: %{y:,}<extra></extra>"
            )
        )

    median_conf = float(clean_df[confidence_col].median())
    mean_conf = float(clean_df[confidence_col].mean())

    fig.add_vline(
        x=median_conf,
        line_dash="dash",
        line_color="#E63946",
        line_width=2,
        annotation_text=f"Median ({median_conf:.3f})",
        annotation_position="top left",
        annotation_font=dict(size=11, color="#E63946")
    )

    fig.update_layout(barmode="overlay")
    fig = _apply_theme(
        fig,
        title=title,
        xaxis_title="Prediction Confidence Score",
        yaxis_title="Flow Frequency",
        legend_title="Predicted Class" if has_classes else None
    )
    fig.update_xaxes(range=[0.0, 1.02])
    return fig


# ---------------------------------------------------------------------------
# 3. Confusion Matrix Heatmap
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    cm_df: Optional[pd.DataFrame],
    title: Optional[str] = "Normalized Confusion Matrix (Champion Model)"
) -> go.Figure:
    """
    Generate normalized Confusion Matrix heatmap with explicit percentages.

    Parameters
    ----------
    cm_df : Optional[pd.DataFrame]
        Square dataframe of confusion matrix or with index = true classes.
    title : Optional[str]
        Figure title.
    """
    if cm_df is None or cm_df.empty:
        return _empty_figure("No confusion matrix data provided")

    df = cm_df.copy()
    # Normalize index/columns if first column is class name
    if "Unnamed: 0" in df.columns:
        df = df.set_index("Unnamed: 0")
    elif df.columns[0] in ["True_Class", "Actual", "index", "class"]:
        df = df.set_index(df.columns[0])

    true_classes = list(df.index)
    pred_classes = list(df.columns)

    # Convert to numeric float matrix
    matrix_values = df.to_numpy(dtype=float)

    # Check if matrix is in [0, 1] (normalized) or counts
    is_normalized = (matrix_values.max() <= 1.0)
    if not is_normalized:
        row_sums = matrix_values.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        norm_matrix = matrix_values / row_sums
    else:
        norm_matrix = matrix_values

    # Formatted text annotations
    annotations_text = []
    for r in range(len(true_classes)):
        row_text = []
        for c in range(len(pred_classes)):
            val = norm_matrix[r, c]
            row_text.append(f"<b>{val:.1%}</b>")
        annotations_text.append(row_text)

    fig = go.Figure(
        data=go.Heatmap(
            z=norm_matrix,
            x=pred_classes,
            y=true_classes,
            colorscale=[[0, "#EFF6FF"], [0.25, "#DBEAFE"], [0.65, "#3B82F6"], [1.0, "#1D4ED8"]],
            zmin=0.0,
            zmax=1.0,
            text=annotations_text,
            texttemplate="%{text}",
            textfont=dict(size=14, color="#0F172A", family="Inter, sans-serif"),
            hoverongaps=False,
            colorbar=dict(
                title=dict(text="Recall", font=dict(size=11, color="#475569")),
                tickformat=".0%",
                tickfont=dict(color="#475569"),
                len=0.9
            ),
            hovertemplate="<b>True:</b> %{y}<br><b>Predicted:</b> %{x}<br><b>Normalized Rate:</b> %{z:.2%}<extra></extra>"
        )
    )

    fig = _apply_theme(
        fig,
        title=title,
        xaxis_title="Predicted QoS Class",
        yaxis_title="True QoS Class",
        show_legend=False
    )
    fig.update_xaxes(side="bottom")
    fig.update_yaxes(autorange="reversed")  # Standard top-to-bottom matrix orientation
    return fig


# ---------------------------------------------------------------------------
# 4. Per-Class Metrics Bar Chart
# ---------------------------------------------------------------------------

def plot_per_class_metrics(
    metrics_df: Optional[pd.DataFrame],
    title: Optional[str] = "Per-Class Performance Metrics (Precision, Recall, F1)"
) -> go.Figure:
    """
    Generate grouped bar chart comparing Precision, Recall, and F1 per class.

    Parameters
    ----------
    metrics_df : Optional[pd.DataFrame]
        Dataframe containing QoS_Class, Precision, Recall, F1_Score.
    title : Optional[str]
        Figure title.
    """
    if metrics_df is None or metrics_df.empty:
        return _empty_figure("No per-class metrics data available")

    df = metrics_df.copy()
    class_col = next((c for c in ["QoS_Class", "Class", "qos_class", "class"] if c in df.columns), df.columns[0])

    metric_cols = {
        "Precision": next((c for c in ["Precision", "precision"] if c in df.columns), None),
        "Recall": next((c for c in ["Recall", "recall"] if c in df.columns), None),
        "F1-Score": next((c for c in ["F1_Score", "F1", "f1_score", "f1"] if c in df.columns), None),
    }

    active_metrics = {k: v for k, v in metric_cols.items() if v is not None}
    if not active_metrics:
        return _empty_figure("Required metric columns (Precision, Recall, F1) missing")

    palette = {
        "Precision": "#2563EB",
        "Recall": "#0D9488",
        "F1-Score": "#64748B"
    }

    fig = go.Figure()

    classes = df[class_col].tolist()
    for metric_name, col_name in active_metrics.items():
        vals = df[col_name].tolist()
        fig.add_trace(
            go.Bar(
                name=metric_name,
                x=classes,
                y=vals,
                marker=dict(color=palette.get(metric_name, "#64748B"), line=dict(color="#FFFFFF", width=1.5)),
                text=[f"{v:.3f}" if isinstance(v, (int, float)) else "" for v in vals],
                textposition="outside",
                textfont=dict(size=11),
                hovertemplate=f"<b>%{{x}}</b><br>{metric_name}: %{{y:.4f}}<extra></extra>"
            )
        )

    # Check for Baseline Recall comparison
    if "Baseline_Recall" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=classes,
                y=df["Baseline_Recall"],
                mode="markers",
                name="Baseline Recall",
                marker=dict(symbol="line-ew-open", size=24, color="#DC2626", line=dict(width=3, color="#DC2626")),
                hovertemplate="<b>%{x}</b><br>Baseline Recall: %{y:.4f}<extra></extra>"
            )
        )

    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08)
    fig = _apply_theme(
        fig,
        title=title,
        xaxis_title="QoS Class",
        yaxis_title="Score [0.0 - 1.0]",
        legend_title="Metric"
    )
    fig.update_yaxes(range=[0.0, 1.08])
    return fig


# ---------------------------------------------------------------------------
# 5. Experiment Comparison Chart
# ---------------------------------------------------------------------------

def plot_experiment_comparison(
    exp_df: Optional[pd.DataFrame],
    metric: str = "Macro_F1",
    baseline_val: Optional[float] = 0.5734,
    title: Optional[str] = None
) -> go.Figure:
    """
    Generate horizontal/vertical bar chart comparing candidate model configurations.

    Parameters
    ----------
    exp_df : Optional[pd.DataFrame]
        Dataframe containing Configuration, Macro_F1, Balanced_Accuracy, etc.
    metric : str
        Target metric column name.
    baseline_val : Optional[float]
        Baseline anchor reference value to draw dashed line.
    title : Optional[str]
        Figure title.
    """
    if exp_df is None or exp_df.empty:
        return _empty_figure("No experiment comparison data available")

    df = exp_df.copy()
    cfg_col = next((c for c in ["Configuration", "config", "Model", "Name"] if c in df.columns), df.columns[0])

    if metric not in df.columns:
        for candidate in ["Macro_F1", "Balanced_Accuracy", "Overall_Accuracy", "macro_f1"]:
            if candidate in df.columns:
                metric = candidate
                break
        else:
            return _empty_figure(f"Metric '{metric}' not found in experiment data")

    chart_title = title or f"Candidate Architecture Comparison: {metric.replace('_', ' ')}"

    # Sort configurations if beneficial or preserve order
    scores = df[metric].tolist()
    configs = df[cfg_col].tolist()
    max_score = max(scores) if scores else 0.0

    # Color code champion vs candidates
    colors = [
        "#16A34A" if s == max_score else ("#2563EB" if "Champion" in str(cfg) or "7.7" in str(cfg) else "#64748B")
        for s, cfg in zip(scores, configs)
    ]

    hover_texts = []
    for _, row in df.iterrows():
        p_val = row.get("McNemar_p_value", "N/A")
        feat_cnt = row.get("Feature_Count", "N/A")
        hover_texts.append(
            f"<b>{row[cfg_col]}</b><br>{metric}: {row[metric]:.4f}<br>Features: {feat_cnt}<br>McNemar p-val: {p_val}"
        )

    fig = go.Figure(
        data=[
            go.Bar(
                y=configs,
                x=scores,
                orientation="h",
                marker=dict(color=colors, line=dict(color="#FFFFFF", width=1.2)),
                text=[f"{s:.4f}" for s in scores],
                textposition="outside",
                textfont=dict(size=11, color="#0F172A"),
                hovertext=hover_texts,
                hoverinfo="text",
                width=0.55
            )
        ]
    )

    if baseline_val is not None:
        fig.add_vline(
            x=baseline_val,
            line_dash="dash",
            line_color="#DC2626",
            line_width=2,
            annotation_text=f"Baseline ({baseline_val:.4f})",
            annotation_position="bottom right",
            annotation_font=dict(size=11, color="#DC2626")
        )

    min_x = max(0.0, min(scores) - 0.05) if scores else 0.5
    max_x = max(scores) + 0.05 if scores else 0.65

    fig = _apply_theme(
        fig,
        title=chart_title,
        xaxis_title=metric.replace("_", " "),
        yaxis_title="Architecture Configuration",
        show_legend=False
    )
    fig.update_xaxes(range=[min_x, max_x])
    fig.update_yaxes(autorange="reversed")
    return fig


# ---------------------------------------------------------------------------
# 6. QoS Metric by Congestion Regime
# ---------------------------------------------------------------------------

def plot_qos_metric_by_regime(
    qos_df: Optional[pd.DataFrame],
    metric: str = "P95_Delay_ms",
    qos_class: Optional[str] = "Real-Time",
    title: Optional[str] = None
) -> go.Figure:
    """
    Generate grouped bar chart across Low, Moderate, High congestion regimes
    comparing Baseline_FIFO vs Predicted_QoS vs Oracle_QoS.

    Parameters
    ----------
    qos_df : Optional[pd.DataFrame]
        Simulation runs dataframe.
    metric : str
        Metric to plot (e.g. 'P95_Delay_ms', 'Packet_Loss_Rate_Pct', 'Mean_Delay_ms', 'Goodput_Mbps').
    qos_class : Optional[str]
        Specific class to filter (e.g. 'Real-Time', 'Streaming', 'Bulk Data'). If None, facets by class.
    title : Optional[str]
        Figure title.
    """
    if qos_df is None or qos_df.empty:
        return _empty_figure("No QoS simulation metrics data available")

    df = qos_df.copy()

    # Column name validation
    scenario_col = next((c for c in ["Scenario", "scenario"] if c in df.columns), None)
    regime_col = next((c for c in ["Congestion_Regime", "regime", "Regime"] if c in df.columns), None)
    class_col = next((c for c in ["QoS_Class", "Class", "qos_class"] if c in df.columns), None)

    if not scenario_col or not regime_col:
        return _empty_figure("Missing required Scenario or Congestion_Regime columns")

    if metric not in df.columns:
        for candidate in ["P95_Delay_ms", "Packet_Loss_Rate_Pct", "Mean_Delay_ms", "Goodput_Mbps"]:
            if candidate in df.columns:
                metric = candidate
                break
        else:
            return _empty_figure(f"QoS metric '{metric}' not found in simulation data")

    # Filter by class if requested and available
    if qos_class and class_col and class_col in df.columns:
        df = df[df[class_col] == qos_class]
        if df.empty:
            return _empty_figure(f"No simulation records found for class '{qos_class}'")

    display_title = title or f"{metric.replace('_', ' ')} across Congestion Regimes"
    if qos_class:
        display_title += f" ({qos_class})"

    # Group by Regime and Scenario
    grouped = df.groupby([regime_col, scenario_col])[metric].mean().reset_index()

    # Ensure orderly regimes
    present_regimes = [r for r in REGIME_ORDER if r in grouped[regime_col].unique()]
    if not present_regimes:
        present_regimes = list(grouped[regime_col].unique())

    fig = go.Figure()
    scenarios = grouped[scenario_col].unique()

    for scen in scenarios:
        sub = grouped[grouped[scenario_col] == scen]
        # Reorder by present_regimes
        sub = sub.set_index(regime_col).reindex(present_regimes).reset_index()
        color = SCENARIO_COLORS.get(scen, "#64748B")

        fig.add_trace(
            go.Bar(
                name=scen.replace("_", " "),
                x=sub[regime_col],
                y=sub[metric],
                marker=dict(color=color, line=dict(color="#FFFFFF", width=1.5)),
                text=[f"{v:.1f}" if pd.notnull(v) else "" for v in sub[metric]],
                textposition="outside",
                textfont=dict(size=11),
                hovertemplate=f"<b>{scen}</b><br>Regime: %{{x}}<br>{metric}: %{{y:.2f}}<extra></extra>"
            )
        )

    # Add 3GPP SLA Reference Line if applicable
    if qos_class and qos_class in SLA_THRESHOLDS:
        sla = SLA_THRESHOLDS[qos_class]
        if "delay" in metric.lower() and "p95_delay_max_ms" in sla:
            threshold = sla["p95_delay_max_ms"]
            fig.add_hline(
                y=threshold,
                line_dash="dot",
                line_color="#DC2626",
                line_width=2,
                annotation_text=f"3GPP SLA Limit ({threshold:.0f} ms)",
                annotation_position="top left",
                annotation_font=dict(size=11, color="#DC2626")
            )
        elif "loss" in metric.lower() and "loss_max_pct" in sla:
            threshold = sla["loss_max_pct"]
            fig.add_hline(
                y=threshold,
                line_dash="dot",
                line_color="#DC2626",
                line_width=2,
                annotation_text=f"3GPP SLA Limit ({threshold:.2f}%)",
                annotation_position="top left",
                annotation_font=dict(size=11, color="#DC2626")
            )

    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.1)
    unit_label = "ms" if "delay" in metric.lower() or "jitter" in metric.lower() else ("%" if "pct" in metric.lower() or "rate" in metric.lower() else "Mbps")
    fig = _apply_theme(
        fig,
        title=display_title,
        xaxis_title="Congestion Regime",
        yaxis_title=f"{metric.replace('_', ' ')} ({unit_label})",
        legend_title="Scenario"
    )
    return fig


# ---------------------------------------------------------------------------
# 7. SLA Compliance Margin Visualization
# ---------------------------------------------------------------------------

def plot_sla_compliance(
    qos_df: Optional[pd.DataFrame],
    regime: str = "High",
    title: Optional[str] = "3GPP SLA Compliance Margin (High Congestion Regime)"
) -> go.Figure:
    """
    Generate SLA compliance pass/fail margin visualization against 3GPP standards.
    3GPP Thresholds:
    - Real-Time: P95 Delay <= 100 ms, Packet Loss <= 1.0%
    - Streaming: P95 Delay <= 300 ms, Packet Loss <= 0.01%
    - Bulk Data: P95 Delay <= 300 ms, Packet Loss <= 1.0%

    Parameters
    ----------
    qos_df : Optional[pd.DataFrame]
        Simulation metrics dataframe.
    regime : str
        Congestion regime to evaluate ('High', 'Moderate', 'Low').
    title : Optional[str]
        Figure title.
    """
    if qos_df is None or qos_df.empty:
        return _empty_figure("No simulation data available for SLA verification")

    df = qos_df.copy()
    scenario_col = next((c for c in ["Scenario", "scenario"] if c in df.columns), None)
    regime_col = next((c for c in ["Congestion_Regime", "regime"] if c in df.columns), None)
    class_col = next((c for c in ["QoS_Class", "Class", "qos_class"] if c in df.columns), None)

    if not (scenario_col and regime_col and class_col):
        return _empty_figure("Required columns (Scenario, Regime, QoS_Class) missing")

    # Filter for desired regime
    regime_df = df[df[regime_col].str.lower() == regime.lower()]
    if regime_df.empty:
        regime_df = df  # fallback to whole dataset if specific regime not filtered

    # Aggregate by Scenario and Class
    agg = regime_df.groupby([scenario_col, class_col]).agg({
        "P95_Delay_ms": "mean",
        "Packet_Loss_Rate_Pct": "mean"
    }).reset_index()

    records = []
    for _, row in agg.iterrows():
        scen = row[scenario_col]
        cls = row[class_col]
        if cls not in SLA_THRESHOLDS:
            continue
        sla = SLA_THRESHOLDS[cls]

        actual_delay = row["P95_Delay_ms"]
        max_delay = sla["p95_delay_max_ms"]
        delay_pass = actual_delay <= max_delay
        delay_margin_pct = ((max_delay - actual_delay) / max_delay) * 100.0

        actual_loss = row["Packet_Loss_Rate_Pct"]
        max_loss = sla["loss_max_pct"]
        loss_pass = actual_loss <= max_loss

        overall_pass = delay_pass and loss_pass

        records.append({
            "Scenario": scen,
            "Class": cls,
            "P95_Delay_ms": actual_delay,
            "Delay_Limit_ms": max_delay,
            "Delay_Margin_Pct": delay_margin_pct,
            "Packet_Loss_Pct": actual_loss,
            "Loss_Limit_Pct": max_loss,
            "Status": "COMPLIANT" if overall_pass else "VIOLATED",
            "Delay_Status": "Pass" if delay_pass else "Fail",
            "Loss_Status": "Pass" if loss_pass else "Fail"
        })

    comp_df = pd.DataFrame(records)
    if comp_df.empty:
        return _empty_figure("Unable to match classes with 3GPP SLA parameters")

    # Create Grouped Bar of Delay with SLA threshold indicators
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("P95 Delay vs 3GPP Limit (ms)", "Packet Loss Rate (%)"),
        horizontal_spacing=0.15
    )

    scenarios = comp_df["Scenario"].unique()

    for scen in scenarios:
        sub = comp_df[comp_df["Scenario"] == scen]
        color = SCENARIO_COLORS.get(scen, "#457B9D")

        # Delay subplot
        fig.add_trace(
            go.Bar(
                name=scen.replace("_", " "),
                x=sub["Class"],
                y=sub["P95_Delay_ms"],
                marker=dict(color=color, line=dict(color="#FFFFFF", width=1.2)),
                text=[f"{v:.1f}ms" for v in sub["P95_Delay_ms"]],
                textposition="outside",
                legendgroup=scen,
                hovertemplate="<b>%{x}</b> (%{fullData.name})<br>P95 Delay: %{y:.2f} ms<br>Limit: %{customdata[0]:.0f} ms<br>Status: %{customdata[1]}<extra></extra>",
                customdata=np.stack((sub["Delay_Limit_ms"], sub["Delay_Status"]), axis=-1)
            ),
            row=1, col=1
        )

        # Loss subplot
        fig.add_trace(
            go.Bar(
                name=scen.replace("_", " "),
                x=sub["Class"],
                y=sub["Packet_Loss_Pct"],
                marker=dict(color=color, line=dict(color="#FFFFFF", width=1.2)),
                text=[f"{v:.2f}%" for v in sub["Packet_Loss_Pct"]],
                textposition="outside",
                legendgroup=scen,
                showlegend=False,
                hovertemplate="<b>%{x}</b> (%{fullData.name})<br>Loss Rate: %{y:.4f}%<br>Limit: %{customdata[0]:.2f}%<br>Status: %{customdata[1]}<extra></extra>",
                customdata=np.stack((sub["Loss_Limit_Pct"], sub["Loss_Status"]), axis=-1)
            ),
            row=1, col=2
        )

    # Add SLA threshold markers
    for cls in ["Real-Time", "Streaming", "Bulk Data"]:
        if cls in SLA_THRESHOLDS:
            d_lim = SLA_THRESHOLDS[cls]["p95_delay_max_ms"]
            l_lim = SLA_THRESHOLDS[cls]["loss_max_pct"]
            fig.add_shape(
                type="line",
                x0=-0.4 if cls == "Real-Time" else (0.6 if cls == "Streaming" else 1.6),
                x1=0.4 if cls == "Real-Time" else (1.4 if cls == "Streaming" else 2.4),
                y0=d_lim, y1=d_lim,
                line=dict(color="#DC2626", width=2.5, dash="dot"),
                row=1, col=1
            )

    fig.update_layout(barmode="group", bargap=0.22, bargroupgap=0.08)
    fig = _apply_theme(
        fig,
        title=f"{title} - [{regime.upper()} Congestion]",
        legend_title="Architecture"
    )
    fig.update_yaxes(title_text="P95 Delay (ms)", row=1, col=1)
    fig.update_yaxes(title_text="Packet Loss (%)", row=1, col=2)
    return fig


# ---------------------------------------------------------------------------
# 8. Conceptual Queue Dynamics Visualization
# ---------------------------------------------------------------------------

def plot_queue_dynamics() -> go.Figure:
    """
    Generate conceptual architectural visualization of the 3-tier scheduling hierarchy:
    - Queue 0: Strict Priority (Real-Time traffic, CIR 25 Mbps, zero buffer delay)
    - Queue 1: Deficit Round Robin 60% Weight (Streaming multimedia)
    - Queue 2: Deficit Round Robin 40% Weight (Elastic bulk data)
    """
    fig = go.Figure()

    # Define queue tier blocks
    queues = [
        {
            "name": "Queue 0: Strict Priority",
            "traffic": "Real-Time (VoIP / Video Conf / Telemetry)",
            "policy": "SP + 25 Mbps CIR Guarantee",
            "color": CLASS_COLORS["Real-Time"],
            "sla": "P95 Delay <= 100ms | Loss <= 1.0%",
            "share": "Guaranteed 1st Departure"
        },
        {
            "name": "Queue 1: Deficit Round Robin (60%)",
            "traffic": "Streaming (Adaptive Video / Audio)",
            "policy": "DRR Quantum Weight: 60%",
            "color": CLASS_COLORS["Streaming"],
            "sla": "P95 Delay <= 300ms | Loss <= 0.01%",
            "share": "60% of Remaining Bandwidth"
        },
        {
            "name": "Queue 2: Deficit Round Robin (40%)",
            "traffic": "Bulk Data (File Transfer / Backup / Mail)",
            "policy": "DRR Quantum Weight: 40%",
            "color": CLASS_COLORS["Bulk Data"],
            "sla": "P95 Delay <= 300ms | Loss <= 1.0%",
            "share": "40% of Remaining Bandwidth"
        }
    ]

    # Horizontal stacked / segmented representation of queue scheduling allocation
    y_positions = [2.0, 1.0, 0.0]

    for q, y in zip(queues, y_positions):
        # Background card
        fig.add_shape(
            type="rect",
            x0=0.0, x1=10.0,
            y0=y - 0.38, y1=y + 0.38,
            fillcolor="#F8FAFC",
            line=dict(color=q["color"], width=2.5),
        )
        # Left color badge
        fig.add_shape(
            type="rect",
            x0=0.0, x1=0.35,
            y0=y - 0.38, y1=y + 0.38,
            fillcolor=q["color"],
            line=dict(color=q["color"], width=0)
        )
        # Header text
        fig.add_annotation(
            x=0.6, y=y + 0.18,
            text=f"<b>{q['name']}</b>",
            font=dict(size=14, color="#0F172A"),
            showarrow=False,
            xanchor="left"
        )
        # Subtitle & properties
        fig.add_annotation(
            x=0.6, y=y - 0.12,
            text=f"<b>Class:</b> {q['traffic']}<br><b>Discipline:</b> {q['policy']} &nbsp;|&nbsp; <b>Allocation:</b> {q['share']}<br><b>3GPP SLA:</b> <span style='color:{q['color']}'>{q['sla']}</span>",
            font=dict(size=11, color="#334155"),
            showarrow=False,
            xanchor="left",
            align="left"
        )

    # Egress Link representation
    fig.add_shape(
        type="rect",
        x0=11.2, x1=13.0,
        y0=-0.38, y1=2.38,
        fillcolor="#F8FAFC",
        line=dict(color="#CBD5E1", width=1.5),
    )
    fig.add_annotation(
        x=12.1, y=1.0,
        text="<b>EGRESS<br>PORT<br>DISPATCH</b><br><span style='font-size:10px;color:#64748B'>1 Gbps Core Link</span>",
        font=dict(size=12, color="#0F172A"),
        showarrow=False,
        align="center"
    )

    # Flow arrows from queues to egress
    for y in y_positions:
        fig.add_annotation(
            x=11.1, y=y,
            ax=10.1, ay=y,
            xref="x", yref="y",
            axref="x", ayref="y",
            showarrow=True,
            arrowhead=2,
            arrowsize=1.2,
            arrowwidth=2.5,
            arrowcolor="#475569"
        )

    # Ensure at least one trace for Plotly rendering engines
    fig.add_trace(go.Scatter(
        x=[0, 13], y=[-0.5, 2.5],
        mode="markers",
        marker=dict(opacity=0, size=1),
        showlegend=False,
        hoverinfo="none"
    ))

    fig.update_layout(
        template=THEME_TEMPLATE,
        height=420,
        xaxis=dict(range=[-0.2, 13.5], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[-0.6, 2.6], showgrid=False, zeroline=False, showticklabels=False),
        title=dict(
            text="<b>QoS-Aware Multi-Queue Scheduling Architecture</b><br><span style='font-size:12px;color:#64748B'>Hierarchical Strict Priority (SP) with Deficit Round Robin (DRR)</span>",
            font=dict(size=15, color="#1E293B"),
            x=0.02,
            y=0.96
        ),
        margin=dict(l=20, r=20, t=65, b=20)
    )
    return fig
