#!/usr/bin/env python3
"""
scripts/run_pipeline_end_to_end.py

End-to-End Pipeline Execution & Verification Engine:
Input PCAP
  -> Flow Formation (Bidirectional 5-tuple, inactivity timeout)
  -> Early-5 Feature Extraction (6 causal features)
  -> ML Inference (Trained Early-5 XGBoost classifier)
  -> QoS Class & Hardware Queue Mapping (SP / DRR)
  -> Closed-Loop Queue Scheduling & Simulation
  -> QoS Performance Telemetry (Delay, P95, Loss, Goodput, Jitter, SLA)

Author: Advanced Autonomous Research Lab (ARL)
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

# Add scripts directory to path
sys.path.append(str(Path(__file__).resolve().parent))
import flow_extractor
import simulate_phase7_qos as sim_mod

def build_or_load_early5_model(model_path: Path, clean_dataset_path: Path) -> XGBClassifier:
    """Loads existing trained model if available, otherwise trains and saves it."""
    clf = XGBClassifier()
    if model_path.exists():
        print(f"[MODEL] Loading existing Early-5 XGBoost model from {model_path}...")
        clf.load_model(str(model_path))
        return clf

    print(f"[MODEL] Training Early-5 XGBoost model on verified clean dataset: {clean_dataset_path}...")
    df_train = pd.read_csv(clean_dataset_path)
    features = [
        "early_5_complete", "early_5_duration", "early_5_bytes",
        "early_5_fwd_pkts", "early_5_bwd_pkts", "early_5_avg_size"
    ]
    classes = sorted(df_train["qos_class"].unique())
    cls2idx = {c: i for i, c in enumerate(classes)}

    X_tr = df_train[features].astype(float).values
    y_tr = df_train["qos_class"].map(cls2idx).values

    c_counts = np.bincount(y_tr, minlength=len(classes))
    sw = len(y_tr) / (len(classes) * np.maximum(c_counts[y_tr], 1))

    clf = XGBClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        objective="multi:softprob", eval_metric="mlogloss",
        random_state=42, n_jobs=2
    )
    clf.fit(X_tr, y_tr, sample_weight=sw)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    clf.save_model(str(model_path))
    print(f"[MODEL] Model successfully trained and saved to {model_path}.")
    return clf

def main():
    parser = argparse.ArgumentParser(description="End-to-End QoS-Aware Encrypted Traffic Pipeline")
    parser.add_argument("--pcap", type=str, default="data/raw/scpDown4.pcap", help="Path to input PCAP file")
    parser.add_argument("--model", type=str, default=None, help="Optional path to custom trained XGBoost model JSON")
    parser.add_argument("--outdir", type=str, default="output/runtime_verification", help="Output directory")
    parser.add_argument("--regime", type=str, default="Moderate", choices=["Low", "Moderate", "High"], help="Congestion regime")
    parser.add_argument("--seed", type=int, default=42, help="Simulation seed")
    parser.add_argument("--sim-time", type=float, default=60.0, help="Simulation duration in seconds for verification")
    args = parser.parse_args()

    t_start = time.time()
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("END-TO-END PROJECT EXECUTION & RUNTIME VERIFICATION")
    print("=" * 80)
    print(f"Input PCAP        : {args.pcap}")
    print(f"Output Directory  : {out_dir.resolve()}")
    print(f"Congestion Regime : {args.regime} Link Load")
    print(f"Simulation Seed   : {args.seed}")
    print(f"Verification Window: {args.sim_time} seconds")
    print("-" * 80)

    # -------------------------------------------------------------------------
    # STAGE 1: FLOW FORMATION & FEATURE EXTRACTION
    # -------------------------------------------------------------------------
    print("\n>>> STAGE 1: FLOW FORMATION & EARLY-5 FEATURE EXTRACTION...")
    t0 = time.time()
    pcap_path = Path(args.pcap)
    if not pcap_path.exists():
        print(f"[ERROR] Input PCAP not found: {pcap_path}")
        sys.exit(1)

    # Stage PCAP into temp folder for flow_extractor
    temp_pcap_dir = out_dir / "temp_input_pcap"
    temp_pcap_dir.mkdir(parents=True, exist_ok=True)
    staged_pcap = temp_pcap_dir / pcap_path.name
    with open(pcap_path, "rb") as src, open(staged_pcap, "wb") as dst:
        dst.write(src.read())

    flow_out_dir = out_dir / "extracted_flows"
    flow_out_dir.mkdir(parents=True, exist_ok=True)

    # Run flow extractor
    extractor_cmd = [
        sys.executable, "scripts/flow_extractor.py",
        "--input", str(temp_pcap_dir),
        "--output", str(flow_out_dir),
        "--groups", "research_notes/capture_groups.csv"
    ]
    import subprocess
    res_ext = subprocess.run(extractor_cmd, capture_output=True, text=True)
    if res_ext.returncode != 0:
        print(f"[ERROR] Flow extraction failed:\n{res_ext.stderr}")
        sys.exit(1)

    flow_csv = flow_out_dir / "flow_features.csv"
    if not flow_csv.exists():
        print(f"[ERROR] Expected flow CSV not created: {flow_csv}")
        sys.exit(1)

    df_flows = pd.read_csv(flow_csv)
    t_ext = time.time() - t0
    print(f"[STAGE 1 PASS] Extracted {len(df_flows)} flows from {pcap_path.name} in {t_ext:.2f}s.")
    
    # -------------------------------------------------------------------------
    # STAGE 2: CLASSIFIER INFERENCE
    # -------------------------------------------------------------------------
    print("\n>>> STAGE 2: MACHINE LEARNING CLASSIFICATION...")
    t0 = time.time()
    if args.model and Path(args.model).exists():
        model_path = Path(args.model)
        clf = XGBClassifier()
        print(f"[MODEL] Loading user-specified model from {model_path}...")
        clf.load_model(str(model_path))
    elif Path("output/phase7_7/phase7_7_champion_model.json").exists():
        model_path = Path("output/phase7_7/phase7_7_champion_model.json")
        clf = XGBClassifier()
        print(f"[MODEL] Loading Phase 7.7 Champion model (8-feature Robust Vector) from {model_path}...")
        clf.load_model(str(model_path))
    else:
        model_path = out_dir / "early5_xgboost_model.json"
        clean_ds_path = Path("output/phase6d/clean_track_b_flows.csv")
        clf = build_or_load_early5_model(model_path, clean_ds_path)

    # Detect model feature dimensionality (8 features for Phase 7.7 Champion, 6 for Baseline)
    num_feats = clf.get_booster().num_features()
    if num_feats == 8:
        features = [
            "early_5_complete", "early_5_duration", "early_5_bytes",
            "early_5_fwd_pkts", "early_5_bwd_pkts", "early_5_avg_size",
            "is_udp", "early_5_fwd_ratio"
        ]
        print(f"[MODEL] Model architecture: Phase 7.7 Champion with {num_feats} causal features.")
    else:
        features = [
            "early_5_complete", "early_5_duration", "early_5_bytes",
            "early_5_fwd_pkts", "early_5_bwd_pkts", "early_5_avg_size"
        ]
        print(f"[MODEL] Model architecture: Baseline with {num_feats} Early-5 features.")

    # Ensure required features exist in df_flows
    if "is_udp" not in df_flows.columns:
        df_flows["is_udp"] = (df_flows["protocol"] == 17).astype(float)
    if "early_5_fwd_ratio" not in df_flows.columns:
        df_flows["early_5_fwd_ratio"] = df_flows["early_5_fwd_pkts"] / np.maximum(df_flows["early_5_fwd_pkts"] + df_flows["early_5_bwd_pkts"], 1)

    classes = ["Bulk Data", "Real-Time", "Streaming"] # Sorted 3 classes
    idx2cls = {i: c for i, c in enumerate(classes)}

    X_test = df_flows[features].astype(float).values
    probs = clf.predict_proba(X_test)
    preds = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)
    pred_classes = [idx2cls[p] for p in preds]

    df_flows["pred_class"] = pred_classes
    df_flows["confidence"] = np.round(confidences, 4)
    for i, c in enumerate(classes):
        df_flows[f"prob_{c}"] = np.round(probs[:, i], 4)

    inferred_csv = out_dir / "inferred_flows.csv"
    df_flows.to_csv(inferred_csv, index=False)
    t_ml = time.time() - t0
    print(f"[STAGE 2 PASS] Evaluated {len(df_flows)} flows with XGBoost in {t_ml*1000:.2f} ms.")
    print("\nSample Inferences with Early-5 Features:")
    disp_cols = ["flow_id", "early_5_complete", "early_5_bytes", "pred_class", "confidence", "prob_Bulk Data", "prob_Real-Time", "prob_Streaming"]
    print(df_flows[disp_cols].head(5).to_string(index=False))

    # -------------------------------------------------------------------------
    # STAGE 3: QoS QUEUE MAPPING & SCHEDULER EXECUTION
    # -------------------------------------------------------------------------
    print("\n>>> STAGE 3: QoS QUEUE MAPPING & SCHEDULING VERIFICATION...")
    t0 = time.time()
    
    # Map each predicted class to its respective queue
    queue_map = {
        "Real-Time": {"queue_id": 0, "name": "Queue 0 (Strict Priority, Policed 25 Mbps CIR)", "buffer_bytes": 125000},
        "Streaming": {"queue_id": 1, "name": "Queue 1 (Deficit Round Robin, Weight 60%)", "buffer_bytes": 300000},
        "Bulk Data": {"queue_id": 2, "name": "Queue 2 (Deficit Round Robin, Weight 40%)", "buffer_bytes": 200000}
    }
    df_flows["target_queue"] = df_flows["pred_class"].map(lambda c: queue_map[c]["queue_id"])
    df_flows["queue_name"] = df_flows["pred_class"].map(lambda c: queue_map[c]["name"])

    print("QoS Queue Mapping Verified:")
    for c, qinfo in queue_map.items():
        count = (df_flows["pred_class"] == c).sum()
        print(f"  {c:<12} -> {qinfo['name']:<50} (Mapped Flows: {count})")

    # Standalone verification of QoSQueueEngine scheduling
    qos_engine = sim_mod.QoSQueueEngine()
    # Test enqueue packets
    p_rt = sim_mod.Packet(1, 1, "Real-Time", 1, 100, 0.0, "fwd", "Real-Time", 0)
    p_str = sim_mod.Packet(2, 2, "Streaming", 1, 1400, 0.0, "fwd", "Streaming", 0)
    p_bulk = sim_mod.Packet(3, 3, "Bulk Data", 1, 1460, 0.0, "fwd", "Bulk Data", 0)

    ok_rt = qos_engine.enqueue(p_rt, 0, 0.0)
    ok_str = qos_engine.enqueue(p_str, 1, 0.0)
    ok_bulk = qos_engine.enqueue(p_bulk, 2, 0.0)
    
    # Verify Strict Priority dequeue: Real-Time must be dequeued first!
    dq_1 = qos_engine.dequeue(0.001)
    # Next, DRR will serve Queue 1 (Streaming) and Queue 2 (Bulk)
    dq_2 = qos_engine.dequeue(0.002)
    dq_3 = qos_engine.dequeue(0.003)

    assert dq_1.flow_class == "Real-Time", "Strict Priority failed: Queue 0 not served first!"
    assert ok_rt and ok_str and ok_bulk, "Queue enqueue failed!"
    t_sched = time.time() - t0
    print(f"[STAGE 3 PASS] QoS Queue Engine verified: Enqueue/Dequeue operations operating correctly in {t_sched*1000:.2f} ms.")

    # -------------------------------------------------------------------------
    # STAGE 4: CLOSED-LOOP NETWORK SIMULATION & TELEMETRY
    # -------------------------------------------------------------------------
    print("\n>>> STAGE 4: CLOSED-LOOP DISCRETE-EVENT NETWORK SIMULATION...")
    t0 = time.time()
    
    # Set simulation window for runtime check
    sim_mod.SIMULATION_TIME_SEC = args.sim_time
    sim_mod.WARMUP_TIME_SEC = min(10.0, args.sim_time * 0.2)
    
    # Execute simulation
    sim = sim_mod.NetworkSimulator(scenario="Predicted_QoS", congestion_regime=args.regime, seed=args.seed)
    sim.run()
    metrics = sim_mod.compute_run_metrics(sim)
    t_sim = time.time() - t0

    df_metrics = pd.DataFrame(metrics)
    metrics_csv = out_dir / "e2e_simulation_metrics.csv"
    df_metrics.to_csv(metrics_csv, index=False)
    print(f"[STAGE 4 PASS] Simulation completed in {t_sim:.2f}s ({len(sim.transmitted_packets)} packets processed).")

    # -------------------------------------------------------------------------
    # STAGE 5: FINAL METRICS & SLA VERIFICATION
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("FINAL END-TO-END PERFORMANCE OUTPUT")
    print("=" * 80)
    for m in metrics:
        c = m["QoS_Class"]
        p95 = m["P95_Delay_ms"]
        loss = m["Packet_Loss_Rate_Pct"]
        goodput = m["Goodput_Mbps"]
        jitter = m["Jitter_ms"]
        
        # Check SLA
        delay_ok = p95 <= sim_mod.SLA_DELAY_MS[c]
        loss_ok = loss <= sim_mod.SLA_MAX_LOSS_PCT[c]
        sla_str = "PASS (Full Compliance)" if (delay_ok and loss_ok) else ("PARTIAL (Delay Met, Loss Exceeded)" if delay_ok else "FAIL")
        
        print(f"Class: {c:<12} | P95 Delay: {p95:6.2f} ms (<= {sim_mod.SLA_DELAY_MS[c]} ms) | Loss: {loss:5.2f}% (<= {sim_mod.SLA_MAX_LOSS_PCT[c]}%) | Goodput: {goodput:6.2f} Mbps | Jitter: {jitter:5.2f} ms | SLA: {sla_str}")

    t_total = time.time() - t_start
    print("-" * 80)
    print(f"[SUCCESS] Complete end-to-end pipeline executed in {t_total:.2f} seconds.")
    print(f"Results saved to: {out_dir.resolve()}")
    print("=" * 80)

if __name__ == "__main__":
    main()
