#!/usr/bin/env python3
"""
Bidirectional Encrypted Network Traffic Flow Extractor
Project: QoS-Aware Encrypted Network Traffic Classification for Next-Generation Networks

Author: Lead Research Engineer
Purpose: Converts raw PCAP / PCAPNG files into a standardized, payload-agnostic,
         flow-level feature dataset using bidirectional 5-tuple aggregation with
         configurable inactivity timeout slicing.
"""

import os
import sys
import argparse
import struct
import socket
import csv
import json
import time
from datetime import datetime, timezone
from collections import defaultdict
import numpy as np

# Verify Scapy
try:
    from scapy.utils import RawPcapReader, RawPcapNgReader
except ImportError:
    print("[ERROR] Scapy is required. Install via: pip install scapy")
    sys.exit(1)


def format_iso_timestamp(ts):
    """Format POSIX float timestamp to ISO 8601 UTC string."""
    if ts is None:
        return ""
    try:
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    except (ValueError, OverflowError, OSError):
        return str(ts)


def compute_distribution_stats(values):
    """
    Computes summary distribution metrics:
    Mean, Std, Min, Max, Median, 25th percentile, 75th percentile.
    """
    if not values or len(values) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    arr = np.array(values, dtype=np.float64)
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr))
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))
    p25, p50, p75 = np.percentile(arr, [25, 50, 75])

    return (
        round(mean_val, 4),
        round(std_val, 4),
        round(min_val, 4),
        round(max_val, 4),
        round(float(p50), 4),
        round(float(p25), 4),
        round(float(p75), 4),
    )


def compute_iats(timestamps):
    """Compute inter-arrival times between consecutive sorted timestamps."""
    if len(timestamps) < 2:
        return []
    sorted_ts = np.sort(np.array(timestamps, dtype=np.float64))
    diffs = np.diff(sorted_ts)
    # Clip negative diffs resulting from sub-microsecond out-of-order multi-core capture cards
    return np.maximum(diffs, 0.0).tolist()


def parse_packet_data(pkt_data, linktype):
    """
    Fast binary parsing of packet headers.
    Returns:
      (status_ok, is_ipv4, is_ipv6, src_ip, dst_ip, proto_num, proto_name,
       sport, dport, ip_total_length, tcp_flags, tcp_window)
    """
    offset = 0
    eth_type = None

    if linktype == 1:  # DLT_EN10MB (Standard Ethernet)
        if len(pkt_data) < 14:
            return (False, False, False, "", "", 0, "Malformed Link", 0, 0, 0, 0, 0)
        eth_type = struct.unpack("!H", pkt_data[12:14])[0]
        offset = 14
        if eth_type == 0x8100:  # 802.1Q VLAN Tagged
            if len(pkt_data) < 18:
                return (False, False, False, "", "", 0, "Malformed VLAN", 0, 0, 0, 0, 0)
            eth_type = struct.unpack("!H", pkt_data[16:18])[0]
            offset = 18

    elif linktype in (12, 101):  # DLT_RAW / Raw IP (VPN tunnel captures)
        if len(pkt_data) < 1:
            return (False, False, False, "", "", 0, "Malformed Raw", 0, 0, 0, 0, 0)
        version = (pkt_data[0] >> 4) & 0x0F
        if version == 4:
            eth_type = 0x0800
        elif version == 6:
            eth_type = 0x86DD
        else:
            return (False, False, False, "", "", 0, f"Unknown Raw IP v{version}", 0, 0, 0, 0, 0)
        offset = 0

    elif linktype == 113:  # DLT_LINUX_SLL (Linux Cooked)
        if len(pkt_data) < 16:
            return (False, False, False, "", "", 0, "Malformed SLL", 0, 0, 0, 0, 0)
        eth_type = struct.unpack("!H", pkt_data[14:16])[0]
        offset = 16

    elif linktype == 0:  # DLT_NULL / BSD Loopback
        if len(pkt_data) < 4:
            return (False, False, False, "", "", 0, "Malformed Loopback", 0, 0, 0, 0, 0)
        family = struct.unpack("!I", pkt_data[0:4])[0]
        eth_type = 0x0800 if family == 2 else (0x86DD if family in (24, 28, 30) else None)
        offset = 4
        if eth_type is None:
            return (False, False, False, "", "", 0, f"Unknown BSD Family {family}", 0, 0, 0, 0, 0)

    else:
        # Fallback inspection of first nibble
        if len(pkt_data) >= 1:
            ver = (pkt_data[0] >> 4) & 0x0F
            if ver == 4:
                eth_type = 0x0800
                offset = 0
            elif ver == 6:
                eth_type = 0x86DD
                offset = 0
            else:
                return (False, False, False, "", "", 0, f"Unsupported LinkType {linktype}", 0, 0, 0, 0, 0)
        else:
            return (False, False, False, "", "", 0, f"Unsupported LinkType {linktype}", 0, 0, 0, 0, 0)

    # --- IPv4 Header ---
    if eth_type == 0x0800:
        if len(pkt_data) < offset + 20:
            return (False, True, False, "", "", 0, "Truncated IPv4", 0, 0, 0, 0, 0)

        v_ihl = pkt_data[offset]
        ihl = (v_ihl & 0x0F) * 4
        if ihl < 20 or len(pkt_data) < offset + ihl:
            return (False, True, False, "", "", 0, "Invalid IPv4 IHL", 0, 0, 0, 0, 0)

        # IP Total Length (Layer 3 payload + header length)
        ip_total_length = struct.unpack("!H", pkt_data[offset + 2 : offset + 4])[0]
        proto_num = pkt_data[offset + 9]
        src_ip = socket.inet_ntoa(pkt_data[offset + 12 : offset + 16])
        dst_ip = socket.inet_ntoa(pkt_data[offset + 16 : offset + 20])
        l4_offset = offset + ihl

        if proto_num == 6:  # TCP
            if len(pkt_data) >= l4_offset + 14:
                sport, dport = struct.unpack("!HH", pkt_data[l4_offset : l4_offset + 4])
                tcp_flags = pkt_data[l4_offset + 13]
                tcp_window = 0
                if len(pkt_data) >= l4_offset + 16:
                    tcp_window = struct.unpack("!H", pkt_data[l4_offset + 14 : l4_offset + 16])[0]
                return (True, True, False, src_ip, dst_ip, 6, "TCP", sport, dport, ip_total_length, tcp_flags, tcp_window)
            return (False, True, False, src_ip, dst_ip, 6, "Truncated TCP", 0, 0, ip_total_length, 0, 0)

        elif proto_num == 17:  # UDP
            if len(pkt_data) >= l4_offset + 4:
                sport, dport = struct.unpack("!HH", pkt_data[l4_offset : l4_offset + 4])
                return (True, True, False, src_ip, dst_ip, 17, "UDP", sport, dport, ip_total_length, 0, 0)
            return (False, True, False, src_ip, dst_ip, 17, "Truncated UDP", 0, 0, ip_total_length, 0, 0)

        else:
            return (True, True, False, src_ip, dst_ip, proto_num, f"Proto-{proto_num}", 0, 0, ip_total_length, 0, 0)

    # --- IPv6 Header ---
    elif eth_type == 0x86DD:
        if len(pkt_data) < offset + 40:
            return (False, False, True, "", "", 0, "Truncated IPv6", 0, 0, 0, 0, 0)

        payload_len = struct.unpack("!H", pkt_data[offset + 4 : offset + 6])[0]
        ip_total_length = payload_len + 40
        next_header = pkt_data[offset + 6]
        src_ip = socket.inet_ntop(socket.AF_INET6, pkt_data[offset + 8 : offset + 24])
        dst_ip = socket.inet_ntop(socket.AF_INET6, pkt_data[offset + 24 : offset + 40])
        l4_offset = offset + 40

        if next_header == 6:  # TCP
            if len(pkt_data) >= l4_offset + 14:
                sport, dport = struct.unpack("!HH", pkt_data[l4_offset : l4_offset + 4])
                tcp_flags = pkt_data[l4_offset + 13]
                tcp_window = 0
                if len(pkt_data) >= l4_offset + 16:
                    tcp_window = struct.unpack("!H", pkt_data[l4_offset + 14 : l4_offset + 16])[0]
                return (True, False, True, src_ip, dst_ip, 6, "TCP", sport, dport, ip_total_length, tcp_flags, tcp_window)
            return (False, False, True, src_ip, dst_ip, 6, "Truncated TCPv6", 0, 0, ip_total_length, 0, 0)

        elif next_header == 17:  # UDP
            if len(pkt_data) >= l4_offset + 4:
                sport, dport = struct.unpack("!HH", pkt_data[l4_offset : l4_offset + 4])
                return (True, False, True, src_ip, dst_ip, 17, "UDP", sport, dport, ip_total_length, 0, 0)
            return (False, False, True, src_ip, dst_ip, 17, "Truncated UDPv6", 0, 0, ip_total_length, 0, 0)

        else:
            return (True, False, True, src_ip, dst_ip, next_header, f"Proto-{next_header}", 0, 0, ip_total_length, 0, 0)

    return (False, False, False, "", "", 0, "Non-IP", 0, 0, 0, 0, 0)


class FlowTracker:
    """
    Maintains packet sequence, directions, timestamps, packet sizes,
    and TCP behavior for a single bidirectional 5-tuple flow instance.
    """

    def __init__(self, canonical_key, fwd_src_ip, fwd_sport, fwd_dst_ip, fwd_dport, proto_num, proto_name, first_ts):
        self.canonical_key = canonical_key
        self.fwd_src_ip = fwd_src_ip
        self.fwd_sport = fwd_sport
        self.fwd_dst_ip = fwd_dst_ip
        self.fwd_dport = fwd_dport
        self.proto_num = proto_num
        self.proto_name = proto_name

        self.first_ts = first_ts
        self.last_ts = first_ts

        # Packet sizes
        self.fwd_pkt_sizes = []
        self.bwd_pkt_sizes = []

        # Timestamps
        self.fwd_timestamps = []
        self.bwd_timestamps = []
        self.all_timestamps = []

        # Directions sequence for bursts: list of ('F'|'B', size)
        self.direction_sequence = []

        # TCP metrics
        self.syn_count = 0
        self.ack_count = 0
        self.fin_count = 0
        self.rst_count = 0
        self.psh_count = 0
        self.urg_count = 0
        self.tcp_windows = []

    def add_packet(self, ts, src_ip, sport, size, tcp_flags=0, tcp_window=0):
        """Add an observed packet to the flow."""
        if ts < self.first_ts:
            self.first_ts = ts
        if ts > self.last_ts:
            self.last_ts = ts

        self.all_timestamps.append(ts)

        is_forward = (src_ip == self.fwd_src_ip and sport == self.fwd_sport)

        if is_forward:
            self.fwd_pkt_sizes.append(size)
            self.fwd_timestamps.append(ts)
            self.direction_sequence.append(("F", size))
        else:
            self.bwd_pkt_sizes.append(size)
            self.bwd_timestamps.append(ts)
            self.direction_sequence.append(("B", size))

        if self.proto_num == 6:  # TCP flags
            if tcp_flags & 0x02:
                self.syn_count += 1
            if tcp_flags & 0x10:
                self.ack_count += 1
            if tcp_flags & 0x01:
                self.fin_count += 1
            if tcp_flags & 0x04:
                self.rst_count += 1
            if tcp_flags & 0x08:
                self.psh_count += 1
            if tcp_flags & 0x20:
                self.urg_count += 1
            if tcp_window > 0:
                self.tcp_windows.append(tcp_window)

    def extract_features(self, flow_id, filename, group_id, app_name, is_vpn, qos_class, confidence):
        """
        Calculates all flow-level statistical metrics and returns a flat dict.
        """
        fwd_pkts = len(self.fwd_pkt_sizes)
        bwd_pkts = len(self.bwd_pkt_sizes)
        total_pkts = fwd_pkts + bwd_pkts

        fwd_bytes = sum(self.fwd_pkt_sizes)
        bwd_bytes = sum(self.bwd_pkt_sizes)
        total_bytes = fwd_bytes + bwd_bytes

        duration = max(0.0, round(self.last_ts - self.first_ts, 6))
        pkt_rate = round(total_pkts / duration, 4) if duration > 0 else float(total_pkts)
        byte_rate = round(total_bytes / duration, 4) if duration > 0 else float(total_bytes)

        # Direction ratios
        fwd_pkt_ratio = round(fwd_pkts / total_pkts, 4) if total_pkts > 0 else 0.0
        fwd_byte_ratio = round(fwd_bytes / total_bytes, 4) if total_bytes > 0 else 0.0
        fwd_to_bwd_pkt_ratio = round(fwd_pkts / (bwd_pkts + 1e-6), 4)
        fwd_to_bwd_byte_ratio = round(fwd_bytes / (bwd_bytes + 1e-6), 4)

        # Packet size statistics
        all_sizes = self.fwd_pkt_sizes + self.bwd_pkt_sizes
        b_mean, b_std, b_min, b_max, b_med, b_p25, b_p75 = compute_distribution_stats(all_sizes)
        f_mean, f_std, f_min, f_max, f_med, f_p25, f_p75 = compute_distribution_stats(self.fwd_pkt_sizes)
        bw_mean, bw_std, bw_min, bw_max, bw_med, bw_p25, bw_p75 = compute_distribution_stats(self.bwd_pkt_sizes)

        # Inter-arrival time statistics
        bidir_iats = compute_iats(self.all_timestamps)
        fwd_iats = compute_iats(self.fwd_timestamps)
        bwd_iats = compute_iats(self.bwd_timestamps)

        biat_mean, biat_std, biat_min, biat_max, biat_med, biat_p25, biat_p75 = compute_distribution_stats(bidir_iats)
        fiat_mean, fiat_std, fiat_min, fiat_max, fiat_med, fiat_p25, fiat_p75 = compute_distribution_stats(fwd_iats)
        biat_bw_mean, biat_bw_std, biat_bw_min, biat_bw_max, biat_bw_med, biat_bw_p25, biat_bw_p75 = compute_distribution_stats(bwd_iats)

        # TCP Window statistics
        tcp_win_mean, tcp_win_std, tcp_win_min, tcp_win_max, _, _, _ = compute_distribution_stats(self.tcp_windows)

        # UDP Metrics
        is_udp = 1 if self.proto_num == 17 else 0
        udp_pkts = total_pkts if is_udp else 0
        udp_bytes = total_bytes if is_udp else 0

        # Burst features (consecutive packet runs in same direction)
        fwd_burst_lengths = []
        bwd_burst_lengths = []
        fwd_burst_bytes = []
        bwd_burst_bytes = []

        cur_dir = None
        cur_len = 0
        cur_bytes = 0

        for direction, sz in self.direction_sequence:
            if direction == cur_dir:
                cur_len += 1
                cur_bytes += sz
            else:
                if cur_dir == "F":
                    fwd_burst_lengths.append(cur_len)
                    fwd_burst_bytes.append(cur_bytes)
                elif cur_dir == "B":
                    bwd_burst_lengths.append(cur_len)
                    bwd_burst_bytes.append(cur_bytes)
                cur_dir = direction
                cur_len = 1
                cur_bytes = sz

        if cur_dir == "F":
            fwd_burst_lengths.append(cur_len)
            fwd_burst_bytes.append(cur_bytes)
        elif cur_dir == "B":
            bwd_burst_lengths.append(cur_len)
            bwd_burst_bytes.append(cur_bytes)

        f_burst_cnt = len(fwd_burst_lengths)
        b_burst_cnt = len(bwd_burst_lengths)
        f_burst_pkts_m = round(float(np.mean(fwd_burst_lengths)), 4) if f_burst_cnt > 0 else 0.0
        b_burst_pkts_m = round(float(np.mean(bwd_burst_lengths)), 4) if b_burst_cnt > 0 else 0.0
        f_burst_bytes_m = round(float(np.mean(fwd_burst_bytes)), 4) if f_burst_cnt > 0 else 0.0
        b_burst_bytes_m = round(float(np.mean(bwd_burst_bytes)), 4) if b_burst_cnt > 0 else 0.0
        f_burst_pkts_max = max(fwd_burst_lengths) if f_burst_cnt > 0 else 0
        b_burst_pkts_max = max(bwd_burst_lengths) if b_burst_cnt > 0 else 0
        f_burst_bytes_max = max(fwd_burst_bytes) if f_burst_cnt > 0 else 0
        b_burst_bytes_max = max(bwd_burst_bytes) if b_burst_cnt > 0 else 0

        # Early classification features (First 5, 10, 15, 20 packets)
        early_features = {}
        for N in (5, 10, 15, 20):
            sub_seq = self.direction_sequence[:N]
            k = len(sub_seq)
            complete_flag = 1 if total_pkts >= N else 0

            if k > 0:
                sub_bytes = sum(sz for _, sz in sub_seq)
                sub_fwd = sum(1 for d, _ in sub_seq if d == "F")
                sub_bwd = sum(1 for d, _ in sub_seq if d == "B")
                sub_dur = round(self.all_timestamps[k - 1] - self.all_timestamps[0], 6) if k > 1 else 0.0
                sub_avg_sz = round(sub_bytes / k, 4)
            else:
                sub_bytes = 0
                sub_fwd = 0
                sub_bwd = 0
                sub_dur = 0.0
                sub_avg_sz = 0.0

            early_features[f"early_{N}_complete"] = complete_flag
            early_features[f"early_{N}_duration"] = sub_dur
            early_features[f"early_{N}_bytes"] = sub_bytes
            early_features[f"early_{N}_fwd_pkts"] = sub_fwd
            early_features[f"early_{N}_bwd_pkts"] = sub_bwd
            early_features[f"early_{N}_avg_size"] = sub_avg_sz

        record = {
            # --- Metadata / Audit Info (NOT for behavioral ML input) ---
            "flow_id": flow_id,
            "capture_filename": filename,
            "capture_group": group_id,
            "application": app_name,
            "is_vpn": is_vpn,
            "qos_class": qos_class,
            "label_confidence": confidence,
            "src_ip": self.fwd_src_ip,
            "src_port": self.fwd_sport,
            "dst_ip": self.fwd_dst_ip,
            "dst_port": self.fwd_dport,
            "protocol_name": self.proto_name,
            "flow_start_time": format_iso_timestamp(self.first_ts),
            "flow_end_time": format_iso_timestamp(self.last_ts),
            # --- ML-Safe Behavioral Features ---
            "protocol": self.proto_num,
            "flow_duration": duration,
            "total_packets": total_pkts,
            "fwd_packets": fwd_pkts,
            "bwd_packets": bwd_pkts,
            "total_bytes": total_bytes,
            "fwd_bytes": fwd_bytes,
            "bwd_bytes": bwd_bytes,
            "packet_rate": pkt_rate,
            "byte_rate": byte_rate,
            "fwd_pkt_ratio": fwd_pkt_ratio,
            "fwd_byte_ratio": fwd_byte_ratio,
            "fwd_to_bwd_pkt_ratio": fwd_to_bwd_pkt_ratio,
            "fwd_to_bwd_byte_ratio": fwd_to_bwd_byte_ratio,
            # Packet Size Distribution
            "bidir_pkt_size_mean": b_mean,
            "bidir_pkt_size_std": b_std,
            "bidir_pkt_size_min": b_min,
            "bidir_pkt_size_max": b_max,
            "bidir_pkt_size_median": b_med,
            "bidir_pkt_size_p25": b_p25,
            "bidir_pkt_size_p75": b_p75,
            "fwd_pkt_size_mean": f_mean,
            "fwd_pkt_size_std": f_std,
            "fwd_pkt_size_min": f_min,
            "fwd_pkt_size_max": f_max,
            "fwd_pkt_size_median": f_med,
            "fwd_pkt_size_p25": f_p25,
            "fwd_pkt_size_p75": f_p75,
            "bwd_pkt_size_mean": bw_mean,
            "bwd_pkt_size_std": bw_std,
            "bwd_pkt_size_min": bw_min,
            "bwd_pkt_size_max": bw_max,
            "bwd_pkt_size_median": bw_med,
            "bwd_pkt_size_p25": bw_p25,
            "bwd_pkt_size_p75": bw_p75,
            # Inter-Arrival Times (IAT)
            "bidir_iat_mean": biat_mean,
            "bidir_iat_std": biat_std,
            "bidir_iat_min": biat_min,
            "bidir_iat_max": biat_max,
            "bidir_iat_median": biat_med,
            "bidir_iat_p25": biat_p25,
            "bidir_iat_p75": biat_p75,
            "fwd_iat_mean": fiat_mean,
            "fwd_iat_std": fiat_std,
            "fwd_iat_min": fiat_min,
            "fwd_iat_max": fiat_max,
            "fwd_iat_median": fiat_med,
            "fwd_iat_p25": fiat_p25,
            "fwd_iat_p75": fiat_p75,
            "bwd_iat_mean": biat_bw_mean,
            "bwd_iat_std": biat_bw_std,
            "bwd_iat_min": biat_bw_min,
            "bwd_iat_max": biat_bw_max,
            "bwd_iat_median": biat_bw_med,
            "bwd_iat_p25": biat_bw_p25,
            "bwd_iat_p75": biat_bw_p75,
            # TCP Flags & Windows
            "tcp_syn_count": self.syn_count,
            "tcp_ack_count": self.ack_count,
            "tcp_fin_count": self.fin_count,
            "tcp_rst_count": self.rst_count,
            "tcp_psh_count": self.psh_count,
            "tcp_urg_count": self.urg_count,
            "tcp_win_min": tcp_win_min,
            "tcp_win_max": tcp_win_max,
            "tcp_win_mean": tcp_win_mean,
            "tcp_win_std": tcp_win_std,
            # UDP
            "is_udp": is_udp,
            "udp_packet_count": udp_pkts,
            "udp_byte_count": udp_bytes,
            # Bursts
            "fwd_burst_count": f_burst_cnt,
            "bwd_burst_count": b_burst_cnt,
            "fwd_burst_pkts_mean": f_burst_pkts_m,
            "bwd_burst_pkts_mean": b_burst_pkts_m,
            "fwd_burst_bytes_mean": f_burst_bytes_m,
            "bwd_burst_bytes_mean": b_burst_bytes_m,
            "fwd_burst_pkts_max": f_burst_pkts_max,
            "bwd_burst_pkts_max": b_burst_pkts_max,
            "fwd_burst_bytes_max": f_burst_bytes_max,
            "bwd_burst_bytes_max": b_burst_bytes_max,
        }

        record.update(early_features)
        return record


def load_mappings(groups_csv_path):
    """Load capture metadata mapping from CSV."""
    mapping = {}
    if os.path.exists(groups_csv_path):
        with open(groups_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fn = row["Capture_Filename"].strip()
                mapping[fn] = {
                    "application": row.get("Application", "Unknown"),
                    "qos_class": row.get("QoS_Class", "Unclassified"),
                    "is_vpn": row.get("Is_VPN", "No"),
                    "capture_group": row.get("Capture_Group", "GRP_DEFAULT"),
                    "confidence": row.get("Confidence", "Medium"),
                }
    return mapping


def extract_flows_from_pcap(filepath, meta_info, inactivity_timeout=60.0, min_packets=2):
    """
    Streams a single PCAP/PCAPNG capture, aggregating packets into bidirectional
    5-tuple flows sliced by inactivity timeout.
    Returns: (list_of_flow_records, total_packets_processed, total_flows_created)
    """
    filename = os.path.basename(filepath)
    is_pcapng = filename.lower().endswith(".pcapng")

    active_flows = {}
    completed_flows = []
    flow_sequence_counter = 0

    reader = None
    total_packets = 0

    try:
        if is_pcapng:
            reader = RawPcapNgReader(filepath)
        else:
            reader = RawPcapReader(filepath)

        linktype = getattr(reader, "linktype", 1)

        for pkt_data, meta in reader:
            total_packets += 1

            if is_pcapng:
                ts_resol = float(getattr(meta, "tsresol", 1000000))
                ts = (meta.tshigh << 32 | meta.tslow) / ts_resol
                if hasattr(meta, "linktype") and meta.linktype is not None:
                    linktype = meta.linktype
            else:
                sec = getattr(meta, "sec", 0)
                usec = getattr(meta, "usec", 0)
                ts = sec + (usec * 1e-6)

            (
                status_ok,
                is_ipv4,
                is_ipv6,
                src_ip,
                dst_ip,
                proto_num,
                proto_name,
                sport,
                dport,
                ip_len,
                tcp_flags,
                tcp_win,
            ) = parse_packet_data(pkt_data, linktype)

            if not status_ok or not (is_ipv4 or is_ipv6):
                continue

            # Canonical bidirectional key
            endpoint_a = (src_ip, sport)
            endpoint_b = (dst_ip, dport)
            canonical_key = (min(endpoint_a, endpoint_b), max(endpoint_a, endpoint_b), proto_num)

            # Check for active flow instance
            tracker = active_flows.get(canonical_key)
            if tracker is not None:
                # Inactivity timeout evaluation
                if (ts - tracker.last_ts) > inactivity_timeout:
                    # Finalize timed-out flow
                    completed_flows.append(tracker)
                    # Start fresh flow instance
                    tracker = FlowTracker(
                        canonical_key, src_ip, sport, dst_ip, dport, proto_num, proto_name, ts
                    )
                    active_flows[canonical_key] = tracker
            else:
                # New flow starting
                tracker = FlowTracker(
                    canonical_key, src_ip, sport, dst_ip, dport, proto_num, proto_name, ts
                )
                active_flows[canonical_key] = tracker

            tracker.add_packet(ts, src_ip, sport, ip_len, tcp_flags, tcp_win)

    finally:
        if reader is not None:
            try:
                reader.close()
            except Exception:
                pass

    # Flush remaining active flows at EOF
    for tracker in active_flows.values():
        completed_flows.append(tracker)

    # Filter and serialize
    extracted_records = []
    base_prefix = os.path.splitext(filename)[0]

    for tracker in completed_flows:
        # Require at least min_packets (default 2: allows bidirectional/roundtrip or multiple packet behavior)
        if (len(tracker.fwd_pkt_sizes) + len(tracker.bwd_pkt_sizes)) < min_packets:
            continue

        flow_sequence_counter += 1
        flow_id = f"flow_{base_prefix}_{flow_sequence_counter:06d}"

        row = tracker.extract_features(
            flow_id=flow_id,
            filename=filename,
            group_id=meta_info.get("capture_group", "GRP_DEFAULT"),
            app_name=meta_info.get("application", "Unknown"),
            is_vpn=meta_info.get("is_vpn", "No"),
            qos_class=meta_info.get("qos_class", "Unclassified"),
            confidence=meta_info.get("confidence", "Medium"),
        )
        extracted_records.append(row)

    return extracted_records, total_packets, flow_sequence_counter


def main():
    parser = argparse.ArgumentParser(
        description="Bidirectional Encrypted Network Traffic Flow Extractor"
    )
    parser.add_argument("--input", "-i", default="data/raw", help="Path to input raw PCAP directory")
    parser.add_argument("--output", "-o", default="output", help="Path to output directory")
    parser.add_argument(
        "--groups",
        "-g",
        default="research_notes/capture_groups.csv",
        help="Path to capture groups mapping CSV",
    )
    parser.add_argument(
        "--timeout",
        "-t",
        type=float,
        default=60.0,
        help="Flow inactivity timeout threshold in seconds (default: 60.0)",
    )
    parser.add_argument(
        "--min-packets",
        "-m",
        type=int,
        default=2,
        help="Minimum packets required to retain a flow (default: 2)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional file limit for rapid testing",
    )

    args = parser.parse_args()

    input_dir = os.path.abspath(args.input)
    output_dir = os.path.abspath(args.output)
    groups_csv = os.path.abspath(args.groups)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("      BIDIRECTIONAL ENCRYPTED TRAFFIC FLOW EXTRACTOR (PHASE 2)")
    print(f"Input Directory    : {input_dir}")
    print(f"Output Directory   : {output_dir}")
    print(f"Inactivity Timeout : {args.timeout} seconds")
    print(f"Min Packets/Flow   : {args.min_packets}")
    print("=" * 80)

    mappings = load_mappings(groups_csv)

    # Discover captures
    valid_exts = (".pcap", ".pcapng", ".cap")
    files = []
    for root, _, filenames in os.walk(input_dir):
        for f in filenames:
            if any(f.lower().endswith(ext) for ext in valid_exts):
                files.append(os.path.join(root, f))
    files.sort()

    if args.max_files:
        print(f"[TEST MODE] Limiting extraction to first {args.max_files} capture file(s).")
        files = files[: args.max_files]

    print(f"[INFO] Discovered {len(files)} capture file(s) for flow formation.\n")

    csv_output_path = os.path.join(output_dir, "flow_features.csv")
    json_output_path = os.path.join(output_dir, "flow_features.json")
    summary_path = os.path.join(output_dir, "flow_extraction_summary.txt")
    errors_csv_path = os.path.join(output_dir, "flow_extraction_errors.csv")

    error_records = []
    fieldnames = None
    csv_file = None
    csv_writer = None

    total_flows_all = 0
    total_packets_all = 0
    qos_counter = defaultdict(int)
    app_counter = defaultdict(int)
    vpn_counter = defaultdict(int)

    sample_flows_for_json = []

    start_wall = time.time()

    try:
        csv_file = open(csv_output_path, "w", newline="", encoding="utf-8")

        for idx, filepath in enumerate(files, 1):
            filename = os.path.basename(filepath)
            size_mb = round(os.path.getsize(filepath) / (1024 * 1024), 2)
            meta = mappings.get(filename, {})

            print(f"[{idx}/{len(files)}] Processing: {filename} ({size_mb} MB) [App: {meta.get('application', '?')}, QoS: {meta.get('qos_class', '?')}]...")
            t0 = time.time()

            try:
                records, pkts, f_count = extract_flows_from_pcap(
                    filepath, meta, inactivity_timeout=args.timeout, min_packets=args.min_packets
                )
                elapsed = time.time() - t0
                total_packets_all += pkts
                total_flows_all += len(records)

                # Write out incrementally
                if records:
                    if csv_writer is None:
                        fieldnames = list(records[0].keys())
                        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                        csv_writer.writeheader()

                    for r in records:
                        csv_writer.writerow(r)
                        qos_counter[r["qos_class"]] += 1
                        app_counter[r["application"]] += 1
                        vpn_counter[r["is_vpn"]] += 1

                    # Keep a small sample of flows for JSON
                    if len(sample_flows_for_json) < 200:
                        sample_flows_for_json.extend(records[: max(1, 200 - len(sample_flows_for_json))])

                print(f"    -> Done in {elapsed:.2f}s | Pkts: {pkts:,} | Formed Flows: {len(records):,}")

            except Exception as e:
                elapsed = time.time() - t0
                err_msg = str(e)
                print(f"    -> [ERROR] ({elapsed:.2f}s): {err_msg}")
                error_records.append({
                    "filename": filename,
                    "file_size_mb": size_mb,
                    "error_message": err_msg,
                })

    finally:
        if csv_file is not None:
            csv_file.close()

    total_wall = time.time() - start_wall

    # Write Errors CSV
    with open(errors_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "file_size_mb", "error_message"])
        writer.writeheader()
        for err in error_records:
            writer.writerow(err)

    # Write JSON (Sample + Metadata)
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": {
                    "extraction_time": datetime.now(timezone.utc).isoformat(),
                    "total_captures": len(files),
                    "failed_captures": len(error_records),
                    "total_packets_parsed": total_packets_all,
                    "total_flows_extracted": total_flows_all,
                    "inactivity_timeout_seconds": args.timeout,
                    "min_packets_threshold": args.min_packets,
                    "elapsed_time_seconds": round(total_wall, 2),
                    "qos_class_distribution": dict(qos_counter),
                    "application_distribution": dict(app_counter),
                    "vpn_distribution": dict(vpn_counter),
                },
                "sample_flows": sample_flows_for_json,
            },
            f,
            indent=2,
        )

    # Write Flow Extraction Summary Report
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("         FLOW FORMATION & FEATURE EXTRACTION SUMMARY REPORT\n")
        f.write("Project: QoS-Aware Encrypted Network Traffic Classification for Next-Gen Networks\n")
        f.write(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n")
        f.write("=" * 80 + "\n\n")

        f.write("1. EXTRACTION EXECUTION OVERVIEW\n")
        f.write("-" * 50 + "\n")
        f.write(f"Total Captures Processed   : {len(files)}\n")
        f.write(f"Failed Captures            : {len(error_records)}\n")
        f.write(f"Total Packets Processed    : {total_packets_all:,}\n")
        f.write(f"Total Bidirectional Flows  : {total_flows_all:,}\n")
        f.write(f"Inactivity Timeout         : {args.timeout} seconds\n")
        f.write(f"Min Packets per Flow       : {args.min_packets}\n")
        f.write(f"Total Processing Time      : {total_wall:.2f} seconds\n\n")

        f.write("2. FLOW DISTRIBUTION BY QoS CLASS\n")
        f.write("-" * 50 + "\n")
        for qos, count in sorted(qos_counter.items(), key=lambda x: x[1], reverse=True):
            pct = (count / total_flows_all * 100) if total_flows_all else 0
            f.write(f"  - {qos:<20}: {count:>6,} flows ({pct:>5.2f}%)\n")
        f.write("\n")

        f.write("3. FLOW DISTRIBUTION BY APPLICATION\n")
        f.write("-" * 50 + "\n")
        for app, count in sorted(app_counter.items(), key=lambda x: x[1], reverse=True):
            pct = (count / total_flows_all * 100) if total_flows_all else 0
            f.write(f"  - {app:<20}: {count:>6,} flows ({pct:>5.2f}%)\n")
        f.write("\n")

        f.write("4. FLOW DISTRIBUTION BY ENCAPSULATION\n")
        f.write("-" * 50 + "\n")
        for vpn, count in sorted(vpn_counter.items(), key=lambda x: x[1], reverse=True):
            pct = (count / total_flows_all * 100) if total_flows_all else 0
            f.write(f"  - VPN = {vpn:<14}: {count:>6,} flows ({pct:>5.2f}%)\n")
        f.write("\n")

        f.write("5. KEY METHODOLOGICAL NOTES & SAFETY\n")
        f.write("-" * 50 + "\n")
        f.write("  - Flow Definition: Canonical 5-tuple (min(IP:port), max(IP:port), proto).\n")
        f.write("  - Inactivity Timeout: Flows inactive > 60s are split to reflect independent sessions.\n")
        f.write("  - Feature Safety: IP addresses, MACs, and port numbers are strictly isolated as\n")
        f.write("    audit metadata and MUST NOT be used as ML training inputs.\n")
        f.write("  - Group-Based Validation: Model evaluation must group captures by 'capture_group'\n")
        f.write("    to prevent synthetic data leakage across dual-ended captures.\n")
        f.write("=" * 80 + "\n")

    print("\n" + "=" * 80)
    print(f"[COMPLETED] Extracted {total_flows_all:,} flows across {len(files)} captures in {total_wall:.2f}s.")
    print(f"Features Dataset CSV : {csv_output_path}")
    print(f"Features JSON Sample : {json_output_path}")
    print(f"Summary Report       : {summary_path}")
    print(f"Errors Log           : {errors_csv_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
