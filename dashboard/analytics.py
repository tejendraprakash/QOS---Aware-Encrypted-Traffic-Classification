"""
Analytics and calculation utilities for the Encrypted Traffic Research Dashboard.
Includes flow filtering, summary statistics, scenario/regime aggregations, and SLA compliance evaluation.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default SLA thresholds defined in project benchmark
DEFAULT_SLA_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "Real-Time": {
        "p95_delay_max_ms": 100.0,
        "loss_max_pct": 1.0,
        "weight_pct": 40.0,
    },
    "Streaming": {
        "p95_delay_max_ms": 300.0,
        "loss_max_pct": 0.01,
        "weight_pct": 30.0,
    },
    "Bulk Data": {
        "p95_delay_max_ms": 300.0,
        "loss_max_pct": 1.0,
        "weight_pct": 30.0,
    },
}


# -----------------------------------------------------------------------------
# 1. Flow Filtering Utility
# -----------------------------------------------------------------------------
def filter_flows(
    df: pd.DataFrame,
    qos_classes: Optional[Union[str, List[str]]] = None,
    min_confidence: float = 0.0,
    protocols: Optional[Union[str, int, List[Union[str, int]]]] = None,
    min_duration: Optional[float] = None,
    max_duration: Optional[float] = None,
    is_vpn: Optional[Union[str, bool]] = None,
    capture_groups: Optional[Union[str, List[str]]] = None,
    use_predicted_class: bool = False,
) -> pd.DataFrame:
    """
    Filter a flows DataFrame across multiple dimensions.

    Parameters
    ----------
    df : pd.DataFrame
        Input flows DataFrame.
    qos_classes : Optional[Union[str, List[str]]]
        Traffic class or list of classes to include.
    min_confidence : float
        Minimum prediction confidence threshold (0.0 to 1.0).
    protocols : Optional[Union[str, int, List[Union[str, int]]]]
        Protocols to filter by (e.g. 'TCP', 'UDP', 6, 17).
    min_duration : Optional[float]
        Minimum flow duration in seconds.
    max_duration : Optional[float]
        Maximum flow duration in seconds.
    is_vpn : Optional[Union[str, bool]]
        Filter by VPN encapsulation ('Yes'/'No' or True/False).
    capture_groups : Optional[Union[str, List[str]]]
        Capture group name or list of group names.
    use_predicted_class : bool
        If True, filter against 'pred_class'; otherwise against 'qos_class'.

    Returns
    -------
    pd.DataFrame
        Filtered flows DataFrame copy.
    """
    if df.empty:
        return df.copy()

    filtered = df.copy()

    # 1. QoS Class Filter
    if qos_classes:
        target_classes = [qos_classes] if isinstance(qos_classes, str) else list(qos_classes)
        class_col = "pred_class" if (use_predicted_class and "pred_class" in filtered.columns) else "qos_class"
        if class_col in filtered.columns:
            filtered = filtered[filtered[class_col].isin(target_classes)]

    # 2. Confidence Threshold Filter
    if min_confidence > 0.0 and "confidence" in filtered.columns:
        filtered = filtered[pd.to_numeric(filtered["confidence"], errors="coerce").fillna(0.0) >= min_confidence]

    # 3. Protocol Filter
    if protocols is not None:
        proto_list = [protocols] if isinstance(protocols, (str, int)) else list(protocols)
        proto_str_set = {str(p).upper() for p in proto_list}

        mask = pd.Series(False, index=filtered.index)
        if "protocol_name" in filtered.columns:
            mask |= filtered["protocol_name"].astype(str).str.upper().isin(proto_str_set)
        if "protocol" in filtered.columns:
            mask |= filtered["protocol"].astype(str).isin(proto_str_set)
            # Map numeric TCP(6) and UDP(17)
            if "TCP" in proto_str_set:
                mask |= filtered["protocol"].astype(str).isin(["6", "6.0"])
            if "UDP" in proto_str_set:
                mask |= filtered["protocol"].astype(str).isin(["17", "17.0"])
        filtered = filtered[mask]

    # 4. Duration Range Filter
    if "flow_duration" in filtered.columns:
        dur_col = pd.to_numeric(filtered["flow_duration"], errors="coerce")
        if min_duration is not None:
            filtered = filtered[dur_col >= min_duration]
            dur_col = dur_col[dur_col >= min_duration]
        if max_duration is not None:
            filtered = filtered[dur_col <= max_duration]

    # 5. VPN Filter
    if is_vpn is not None and "is_vpn" in filtered.columns:
        if isinstance(is_vpn, bool):
            vpn_str = "Yes" if is_vpn else "No"
            filtered = filtered[
                (filtered["is_vpn"] == vpn_str) | (filtered["is_vpn"] == is_vpn)
            ]
        else:
            filtered = filtered[filtered["is_vpn"].astype(str).str.lower() == str(is_vpn).lower()]

    # 6. Capture Groups Filter
    if capture_groups is not None and "capture_group" in filtered.columns:
        grp_list = [capture_groups] if isinstance(capture_groups, str) else list(capture_groups)
        filtered = filtered[filtered["capture_group"].isin(grp_list)]

    return filtered


# -----------------------------------------------------------------------------
# 2. Summary Statistics Utilities
# -----------------------------------------------------------------------------
def calculate_percentiles(
    series: pd.Series,
    percentiles: Optional[List[int]] = None,
) -> Dict[str, float]:
    """
    Calculate percentiles for a numeric series.
    """
    if percentiles is None:
        percentiles = [25, 50, 75, 90, 95, 99]

    clean_series = pd.to_numeric(series, errors="coerce").dropna()
    if clean_series.empty:
        return {f"p{p}": float("nan") for p in percentiles}

    res = {}
    for p in percentiles:
        res[f"p{p}"] = float(np.percentile(clean_series, p))
    return res


def calculate_summary_statistics(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    group_by: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compute comprehensive statistical summary (count, mean, std, min, median, p95, max)
    for numeric columns, optionally grouped by a category column.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    columns : Optional[List[str]]
        Specific numeric columns to summarize. If None, summarizes all numeric columns.
    group_by : Optional[str]
        Column name to group by (e.g. 'qos_class' or 'Scenario').

    Returns
    -------
    pd.DataFrame
        Summary statistics table.
    """
    if df.empty:
        return pd.DataFrame()

    if columns is None:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        num_cols = [c for c in columns if c in df.columns]

    if not num_cols:
        return pd.DataFrame()

    def _stats_for_subset(sub_df: pd.DataFrame) -> pd.DataFrame:
        records = []
        for col in num_cols:
            s = pd.to_numeric(sub_df[col], errors="coerce").dropna()
            if s.empty:
                continue
            p = np.percentile(s, [25, 50, 75, 95, 99])
            records.append({
                "metric": col,
                "count": int(s.count()),
                "mean": float(s.mean()),
                "std": float(s.std()) if s.count() > 1 else 0.0,
                "min": float(s.min()),
                "p25": float(p[0]),
                "median": float(p[1]),
                "p75": float(p[2]),
                "p95": float(p[3]),
                "p99": float(p[4]),
                "max": float(s.max()),
            })
        return pd.DataFrame(records)

    if group_by and group_by in df.columns:
        grouped_records = []
        for grp_val, sub_df in df.groupby(group_by):
            stats = _stats_for_subset(sub_df)
            if not stats.empty:
                stats[group_by] = grp_val
                grouped_records.append(stats)
        if not grouped_records:
            return pd.DataFrame()
        res = pd.concat(grouped_records, ignore_index=True)
        # Reorder so group_by is first
        cols = [group_by] + [c for c in res.columns if c != group_by]
        return res[cols]

    return _stats_for_subset(df)


# -----------------------------------------------------------------------------
# 3. Scenario x Regime x Class QoS Comparison Aggregations
# -----------------------------------------------------------------------------
def aggregate_scenario_regime_class(
    metrics_df: pd.DataFrame,
    baseline_scenario: str = "Baseline_FIFO",
) -> pd.DataFrame:
    """
    Aggregate QoS simulation metrics across Scenario x Regime x QoS_Class.
    Computes mean and std across replications/seeds, and calculates relative
    delay and packet loss reduction percentages relative to the baseline scenario.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Simulation run metrics table (from Phase 7 or Runtime Verification).
    baseline_scenario : str
        Scenario name to treat as comparative baseline.

    Returns
    -------
    pd.DataFrame
        Aggregated comparison metrics table.
    """
    if metrics_df.empty:
        return pd.DataFrame()

    required_keys = ["Scenario", "Congestion_Regime", "QoS_Class"]
    if not all(col in metrics_df.columns for col in required_keys):
        logger.warning(f"Missing required grouping keys {required_keys} in metrics DataFrame.")
        return pd.DataFrame()

    metric_cols = [
        "Packet_Loss_Rate_Pct",
        "Mean_Delay_ms",
        "P95_Delay_ms",
        "P99_Delay_ms",
        "Jitter_ms",
        "Goodput_Mbps",
        "Mean_Queue_Occupancy_Bytes",
    ]
    avail_metrics = [c for c in metric_cols if c in metrics_df.columns]

    # Groupby Scenario, Congestion_Regime, QoS_Class
    agg_funcs = {m: ["mean", "std", "median"] for m in avail_metrics}
    agg_df = metrics_df.groupby(required_keys).agg(agg_funcs)

    # Flatten multi-level columns
    flat_cols = []
    for col, stat in agg_df.columns:
        flat_cols.append(f"{col}_{stat}")
    agg_df.columns = flat_cols
    agg_df = agg_df.reset_index()

    # Compute improvements relative to baseline if available
    if baseline_scenario in agg_df["Scenario"].values:
        baseline_slice = agg_df[agg_df["Scenario"] == baseline_scenario].set_index(
            ["Congestion_Regime", "QoS_Class"]
        )

        def _calc_reduction(row, metric_name):
            key = (row["Congestion_Regime"], row["QoS_Class"])
            base_col = f"{metric_name}_mean"
            if key in baseline_slice.index and base_col in baseline_slice.columns:
                base_val = baseline_slice.loc[key, base_col]
                curr_val = row.get(base_col, np.nan)
                if pd.notnull(base_val) and base_val > 0:
                    # Reduction %: (Baseline - Current) / Baseline * 100
                    return round(((base_val - curr_val) / base_val) * 100.0, 2)
            return np.nan

        if "P95_Delay_ms_mean" in agg_df.columns:
            agg_df["P95_Delay_Reduction_Pct"] = agg_df.apply(
                lambda r: _calc_reduction(r, "P95_Delay_ms"), axis=1
            )
        if "Packet_Loss_Rate_Pct_mean" in agg_df.columns:
            agg_df["Loss_Reduction_Pct"] = agg_df.apply(
                lambda r: _calc_reduction(r, "Packet_Loss_Rate_Pct"), axis=1
            )

    return agg_df


# -----------------------------------------------------------------------------
# 4. SLA Compliance Evaluation Helper
# -----------------------------------------------------------------------------
def evaluate_sla_compliance(
    metrics_df: pd.DataFrame,
    sla_thresholds: Optional[Dict[str, Dict[str, float]]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate per-run SLA compliance for network traffic flows or simulation runs.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Simulation metrics containing QoS_Class, P95_Delay_ms, Packet_Loss_Rate_Pct.
    sla_thresholds : Optional[Dict[str, Dict[str, float]]]
        Threshold dict per class with 'p95_delay_max_ms' and 'loss_max_pct'.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        1. Evaluated row-level DataFrame with compliance flags.
        2. Aggregated summary compliance table across Scenario and Congestion_Regime.
    """
    if metrics_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    thresholds = sla_thresholds or DEFAULT_SLA_THRESHOLDS
    eval_df = metrics_df.copy()

    # Match QoS Class column name
    class_col = "QoS_Class" if "QoS_Class" in eval_df.columns else "qos_class"
    if class_col not in eval_df.columns:
        logger.warning("No QoS class column found in metrics DataFrame for SLA evaluation.")
        return eval_df, pd.DataFrame()

    delay_compliant = []
    loss_compliant = []
    overall_compliant = []
    reasons = []

    for _, row in eval_df.iterrows():
        q_class = str(row[class_col])
        thresh = thresholds.get(q_class, {"p95_delay_max_ms": 300.0, "loss_max_pct": 1.0})

        p95_delay = row.get("P95_Delay_ms", row.get("p95_delay_ms", np.nan))
        loss_pct = row.get("Packet_Loss_Rate_Pct", row.get("packet_loss_rate_pct", row.get("loss_pct", np.nan)))

        d_ok = True
        l_ok = True
        violation_parts = []

        if pd.notnull(p95_delay):
            if p95_delay > thresh["p95_delay_max_ms"]:
                d_ok = False
                violation_parts.append(f"Delay {p95_delay:.1f}ms > {thresh['p95_delay_max_ms']:.1f}ms")

        if pd.notnull(loss_pct):
            if loss_pct > thresh["loss_max_pct"]:
                l_ok = False
                violation_parts.append(f"Loss {loss_pct:.2f}% > {thresh['loss_max_pct']:.2f}%")

        is_ok = d_ok and l_ok
        delay_compliant.append(d_ok)
        loss_compliant.append(l_ok)
        overall_compliant.append(is_ok)
        reasons.append("; ".join(violation_parts) if violation_parts else "Compliant")

    eval_df["Delay_SLA_Met"] = delay_compliant
    eval_df["Loss_SLA_Met"] = loss_compliant
    eval_df["SLA_Compliant"] = overall_compliant
    eval_df["Violation_Reason"] = reasons

    # Build Aggregated Compliance Table
    group_cols = [c for c in ["Scenario", "Congestion_Regime"] if c in eval_df.columns]
    if group_cols:
        summary_records = []
        for grp_keys, sub in eval_df.groupby(group_cols):
            if isinstance(grp_keys, str):
                grp_keys = (grp_keys,)

            rec: Dict[str, Any] = dict(zip(group_cols, grp_keys))
            rec["Total_Replications"] = len(sub[class_col].unique()) if "Seed" in sub.columns else len(sub)

            # Per-class compliance rates
            comp_weighted_sum = 0.0
            total_weight = 0.0
            for cls_name, cls_cfg in thresholds.items():
                cls_sub = sub[sub[class_col] == cls_name]
                if not cls_sub.empty:
                    rate = float(cls_sub["SLA_Compliant"].mean() * 100.0)
                else:
                    rate = 0.0
                clean_cls_key = cls_name.replace(" ", "_").replace("-", "_")
                rec[f"{clean_cls_key}_SLA_Pct"] = round(rate, 1)

                wt = cls_cfg.get("weight_pct", 33.33)
                comp_weighted_sum += rate * wt
                total_weight += wt

            rec["Composite_SLA_Pct"] = round(comp_weighted_sum / total_weight, 1) if total_weight > 0 else 0.0
            summary_records.append(rec)

        summary_df = pd.DataFrame(summary_records)
    else:
        summary_df = pd.DataFrame([{
            "Overall_SLA_Compliance_Pct": round(float(eval_df["SLA_Compliant"].mean() * 100.0), 2)
        }])

    return eval_df, summary_df


# -----------------------------------------------------------------------------
# 5. Classification Performance Helper
# -----------------------------------------------------------------------------
def calculate_classification_metrics(
    df: pd.DataFrame,
    true_col: str = "qos_class",
    pred_col: str = "pred_class",
) -> Dict[str, Any]:
    """
    Compute multi-class classification metrics (Accuracy, Balanced Accuracy,
    Macro F1, per-class Precision/Recall/F1/Support, and confusion matrix).
    """
    if df.empty or true_col not in df.columns or pred_col not in df.columns:
        return {
            "overall_accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "macro_f1": 0.0,
            "per_class": {},
            "confusion_matrix": pd.DataFrame(),
        }

    valid_mask = df[true_col].notnull() & df[pred_col].notnull()
    valid_df = df[valid_mask]
    if valid_df.empty:
        return {
            "overall_accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "macro_f1": 0.0,
            "per_class": {},
            "confusion_matrix": pd.DataFrame(),
        }

    classes = sorted(valid_df[true_col].astype(str).unique())
    cm = pd.crosstab(
        valid_df[true_col].astype(str),
        valid_df[pred_col].astype(str),
        rownames=["Actual"],
        colnames=["Predicted"],
        dropna=False
    ).reindex(index=classes, columns=classes, fill_value=0)

    per_class = {}
    recalls = []
    f1s = []

    for c in classes:
        tp = int(cm.loc[c, c]) if c in cm.index and c in cm.columns else 0
        total_actual = int(cm.loc[c].sum()) if c in cm.index else 0
        total_pred = int(cm[c].sum()) if c in cm.columns else 0

        prec = tp / total_pred if total_pred > 0 else 0.0
        rec = tp / total_actual if total_actual > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        per_class[c] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": total_actual,
        }
        recalls.append(rec)
        f1s.append(f1)

    overall_acc = float((valid_df[true_col].astype(str) == valid_df[pred_col].astype(str)).mean())
    balanced_acc = float(np.mean(recalls)) if recalls else 0.0
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0

    return {
        "overall_accuracy": round(overall_acc, 4),
        "balanced_accuracy": round(balanced_acc, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
        "confusion_matrix": cm,
    }
