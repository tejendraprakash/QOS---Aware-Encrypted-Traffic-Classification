#!/usr/bin/env python3
"""
scripts/test_confidence_aware_qos_simulation.py

Evaluates Confidence-Aware QoS Policy in Discrete-Event Network Simulation
under High Congestion (125% load) across 5 seeds (42, 101, 202, 303, 404).

Policy Rule:
- Flows with max(prob) >= tau (0.55): Assigned to predicted queue (Q0 Strict Priority for RT, Q1 DRR 60% for Stream, Q2 DRR 40% for Bulk).
- Flows with max(prob) < tau: Quarantined to Queue 1 (DRR) to protect Queue 0 from low-confidence misclassified bulk bursts.
"""

import sys
sys.path.append("scripts")
import numpy as np
import pandas as pd
import simulate_phase7_qos as sim_mod

# Derive empirical joint distribution: (true_class, pred_class, is_confident)
df_flows = pd.read_csv("output/phase6d2/clean_flows_with_logo_predictions.csv")
TAU = 0.55
df_flows["is_confident"] = df_flows["logo_confidence"] >= TAU

# Compute conditional probabilities per true class
cond_probs = {}
for true_c in ["Bulk Data", "Real-Time", "Streaming"]:
    sub = df_flows[df_flows["qos_class"] == true_c]
    total = len(sub)
    cond_probs[true_c] = {
        "pred_bulk_conf": np.sum((sub["logo_pred_class"] == "Bulk Data") & sub["is_confident"]) / total,
        "pred_bulk_unconf": np.sum((sub["logo_pred_class"] == "Bulk Data") & ~sub["is_confident"]) / total,
        "pred_rt_conf": np.sum((sub["logo_pred_class"] == "Real-Time") & sub["is_confident"]) / total,
        "pred_rt_unconf": np.sum((sub["logo_pred_class"] == "Real-Time") & ~sub["is_confident"]) / total,
        "pred_stream_conf": np.sum((sub["logo_pred_class"] == "Streaming") & sub["is_confident"]) / total,
        "pred_stream_unconf": np.sum((sub["logo_pred_class"] == "Streaming") & ~sub["is_confident"]) / total,
    }

class ConfidenceAwareSimulator(sim_mod.NetworkSimulator):
    def predict_flow_class_with_confidence(self, true_cls: str):
        """Samples prediction and confidence flag from empirical LOGO joint distribution."""
        probs = cond_probs[true_cls]
        r = self.rng.random()
        cum = 0.0
        for outcome, p in probs.items():
            cum += p
            if r <= cum:
                if outcome == "pred_bulk_conf": return "Bulk Data", True
                if outcome == "pred_bulk_unconf": return "Bulk Data", False
                if outcome == "pred_rt_conf": return "Real-Time", True
                if outcome == "pred_rt_unconf": return "Real-Time", False
                if outcome == "pred_stream_conf": return "Streaming", True
                if outcome == "pred_stream_unconf": return "Streaming", False
        return true_cls, True

    def handle_packet_generation(self, flow_id: int):
        fstate = self.flow_states[flow_id]
        if self.current_time > fstate["end_t"]:
            return

        fstate["packet_seq"] += 1
        seq = fstate["packet_seq"]
        true_cls = fstate["class"]

        # Classification decision occurs at packet seq = 5
        if seq == 5 and fstate["pred_class"] is None:
            pred_cls, is_conf = self.predict_flow_class_with_confidence(true_cls)
            fstate["pred_class"] = pred_cls
            fstate["is_confident"] = is_conf

        pred_cls = fstate["pred_class"]
        is_conf = fstate.get("is_confident", True)

        # Packet sizing (identical to base simulator)
        direction = 'fwd'
        if true_cls == "Real-Time":
            if self.current_time >= fstate["state_end_t"]:
                trans_matrix = {
                    0: [0.00, 0.48, 0.48, 0.04],
                    1: [0.35, 0.00, 0.58, 0.07],
                    2: [0.35, 0.58, 0.00, 0.07],
                    3: [0.20, 0.40, 0.40, 0.00]
                }
                d_state = fstate["dialogue_state"]
                r = self.rng.random()
                cum = 0.0
                for nxt, p in enumerate(trans_matrix[d_state]):
                    cum += p
                    if r <= cum:
                        fstate["dialogue_state"] = nxt
                        break
                mean_durs = {0: 1.20, 1: 1.40, 2: 1.40, 3: 0.25}
                dur = self.rng.expovariate(1.0 / mean_durs[fstate["dialogue_state"]])
                fstate["state_end_t"] = self.current_time + dur

            d_state = fstate["dialogue_state"]
            if d_state in (1, 3):
                pkt_size = self.rng.choices([64, 90, 140, 200], weights=[0.15, 0.45, 0.25, 0.15])[0]
                direction = 'fwd'
            elif d_state == 2:
                pkt_size = self.rng.choices([64, 90, 140, 200], weights=[0.15, 0.45, 0.25, 0.15])[0]
                direction = 'bwd'
            else:
                pkt_size = 64
                direction = 'fwd' if self.rng.random() < 0.50 else 'bwd'

        elif true_cls == "Streaming":
            if fstate["chunk_remaining_bytes"] <= 0:
                pkt_size = 450
                direction = 'fwd'
                target_bytes = (2_000_000.0 * 3.0) / 8.0
                mu_corr = np.log(target_bytes) - (0.25 ** 2 / 2.0)
                fstate["chunk_remaining_bytes"] = max(1500, int(round(self.rng.lognormvariate(mu_corr, 0.25))))
                fstate["pkt_in_chunk"] = 0
            else:
                pkt_size = self.rng.choices([1460, 850, 120], weights=[0.85, 0.10, 0.05])[0]
                fstate["chunk_remaining_bytes"] -= pkt_size
                fstate["pkt_in_chunk"] += 1
                direction = 'bwd'
        else: # Bulk Data
            pkt_size = 1500
            direction = 'fwd'

        self.packet_counter += 1
        pkt = sim_mod.Packet(
            packet_id=self.packet_counter,
            flow_id=flow_id,
            flow_class=true_cls,
            packet_seq=seq,
            size_bytes=pkt_size,
            send_time=self.current_time,
            direction=direction,
            predicted_class=pred_cls,
            seq_num=fstate.get("snd_nxt", 0)
        )
        if true_cls == "Bulk Data":
            fstate["snd_nxt"] += pkt_size

        self.transmitted_packets.append(pkt)

        # CONFIDENCE-AWARE QUEUE TARGETING:
        # Packets 1-4 enter Quarantined Queue 1
        if seq < 5 or pred_cls is None:
            target_q = 1
        else:
            if not is_conf:
                # Low confidence flow fallback to Queue 1 (DRR)
                target_q = 1
            else:
                # High confidence flow maps to intended queue
                target_q = self.map_class_to_queue(pred_cls)

        success = self.queue_engine.enqueue(pkt, target_q, self.current_time)

        if not success:
            self.dropped_packets.append(pkt)
            if true_cls == "Bulk Data":
                self.schedule(self.current_time + self.prop_delay_sec, "TCP_ACK", (flow_id, fstate["snd_nxt"], True))
        else:
            self.try_start_transmission()

        # Schedule next packet
        if true_cls == "Real-Time":
            fstate["frame_k"] += 1
            target_t = fstate["base_t"] + (fstate["frame_k"] * fstate["frame_interval"])
            delta_jitter = self.rng.uniform(-0.0003, 0.0003)
            next_t = max(self.current_time + 0.001, target_t + delta_jitter)
            self.schedule(next_t, "GENERATE_PACKET", flow_id)
        elif true_cls == "Streaming":
            if fstate["chunk_remaining_bytes"] > 0:
                dt_pace = (pkt_size * 8.0) / self.link_capacity_bps
                self.schedule(self.current_time + max(1e-6, dt_pace), "GENERATE_PACKET", flow_id)
            else:
                idle_gap = max(0.05, 3.0 - (fstate["pkt_in_chunk"] * 0.005) + self.rng.gauss(0, 0.05))
                self.schedule(self.current_time + idle_gap, "GENERATE_PACKET", flow_id)
        else:
            effective_rtt = max(self.prop_delay_sec * 2.0, sim_mod.RTT_NOMINAL_SEC)
            dt_pace = effective_rtt / max(1.0, fstate["cwnd"])
            self.schedule(self.current_time + max(1e-5, dt_pace), "GENERATE_PACKET", flow_id)

def main():
    print("=" * 80)
    print("GATE 6: SIMULATING CONFIDENCE-AWARE QOS SCHEDULING (TAU = 0.55)")
    print("Regime: High Congestion (125% load), Link: 100 Mbps, 5 Seeds")
    print("=" * 80)

    seeds = [42, 101, 202, 303, 404]
    all_metrics = []

    for s in seeds:
        print(f"\n--- Running Seed {s} (Confidence-Aware QoS) ---")
        sim = ConfidenceAwareSimulator(
            scenario="Confidence_Aware_QoS",
            congestion_regime="High",
            seed=s
        )
        sim.run()
        res = sim_mod.compute_run_metrics(sim)
        for r in res:
            r["Scenario"] = "Confidence_Aware_QoS"
            all_metrics.append(r)
            print(f"  {r['QoS_Class']:<12}: Mean Delay={r['Mean_Delay_ms']:.2f}ms | P95={r['P95_Delay_ms']:.2f}ms | Loss={r['Packet_Loss_Rate_Pct']:.3f}% | Goodput={r['Goodput_Mbps']:.2f}Mbps")

    df_res = pd.DataFrame(all_metrics)
    out_csv = "output/phase6d2/confidence_aware_simulation_run_metrics.csv"
    df_res.to_csv(out_csv, index=False)

    print("\n" + "=" * 80)
    print("CONFIDENCE-AWARE QOS AGGREGATE METRICS (HIGH CONGESTION, N=5 SEEDS)")
    print("=" * 80)
    grouped = df_res.groupby("QoS_Class")
    for cls, grp in grouped:
        print(f"\nClass: {cls}")
        print(f"  Mean Delay:  {grp['Mean_Delay_ms'].mean():.2f} +/- {grp['Mean_Delay_ms'].std():.2f} ms")
        print(f"  P95 Delay:   {grp['P95_Delay_ms'].mean():.2f} +/- {grp['P95_Delay_ms'].std():.2f} ms")
        print(f"  Packet Loss: {grp['Packet_Loss_Rate_Pct'].mean():.3f} +/- {grp['Packet_Loss_Rate_Pct'].std():.3f} %")
        print(f"  Goodput:     {grp['Goodput_Mbps'].mean():.2f} +/- {grp['Goodput_Mbps'].std():.2f} Mbps")
        print(f"  Jitter:      {grp['Jitter_ms'].mean():.2f} +/- {grp['Jitter_ms'].std():.2f} ms")

if __name__ == "__main__":
    main()
