# Agent E: Validation Feasibility & Leakage Audit

**Project Title:** QoS-Aware Encrypted Traffic Classification for Next-Generation Networks  
**Author / Persona:** Agent E — Validation Strategist (`.agents/agents/validation-strategy.md`)  
**Audit Stage:** Phase 4 — Data Leakage, Cross-Validation & Domain Shift Audit  
**Date:** November 2026  

---

## 1. Executive Summary

In machine learning for network traffic classification, improper validation protocols are the leading cause of irreproducible claims. Naive random splitting causes severe synthetic data leakage, creating the illusion of $>99\%$ accuracy on benchmarks that collapses when deployed on real networks.

This audit evaluates data leakage vectors, tests group-based cross-validation feasibility across both datasets, analyzes class distribution within capture groups, quantifies VPN domain shift, and establishes mathematical validation protocols for Phase 3.

---

## 2. Leakage Vectors & Capture Symmetry

### 2.1 The Twin Vantage Point Trap
In Dataset 1 and the broader ISCX benchmark, captures feature paired suffix conventions:
* `email1a.pcap` & `email1b.pcap` (both start at 2015-05-25 ~16:52 UTC)
* `email2a.pcap` & `email2b.pcap` (both start at 2015-06-05 ~18:14 UTC)
* `ftps_down_1a.pcap` & `ftps_down_1b.pcap` (both start at 2015-05-01 ~14:37 UTC)
* `ftps_up_2a.pcap` & `ftps_up_2b.pcap` (both start at 2015-05-01 ~16:24 UTC)

**Physical Reality:** These files represent **simultaneous captures of the exact same network conversation from two different network vantage points** (e.g., client NIC vs router interface).
* If a researcher performs random `train_test_split(df, test_size=0.2)`:
  * Packets from `email1a` go to the training set.
  * Packets from `email1b` go to the testing set.
  * The model memorizes exact packet lengths, TLS cipher suites, and server response times, achieving near-perfect accuracy without learning generalizable behavior.
* **Our Solution:** In `research_notes/capture_groups.csv`, twin captures are bound to the identical group identifier (e.g., `GRP_EMAIL_1`, `GRP_FTPS_DOWN_1`), ensuring they remain strictly on the same side of any train/test split.

---

## 3. Mathematical Evaluation of Cross-Validation Partitioning

### 3.1 Group Cardinality per QoS Class in Dataset 1
Our audit revealed an extreme asymmetry in capture group availability across the 4 classes:

```
Total Capture Groups in Dataset 1: 32 Groups
  ├─ Real-Time : 10 Capture Groups (8,132 flows)
  ├─ Bulk Data : 10 Capture Groups (1,348 flows)
  ├─ Streaming :  9 Capture Groups (1,290 flows)
  └─ Browsing  :  3 Capture Groups (6,847 flows)  <── CRITICAL BOTTLENECK!
                  ├── GRP_EMAIL_1     : 5,870 flows (33.32% of total dataset!)
                  ├── GRP_EMAIL_2     :   672 flows ( 3.81%)
                  └── GRP_VPN_EMAIL_2 :   305 flows ( 1.73%)
```

### 3.2 The Failure of Standard GroupKFold
We simulated scikit-learn's standard `GroupKFold` partitioning across multiple fold counts:
* **Standard `GroupKFold(n_splits=5)`:** **FAILS COMPLETELY**.
  * Fold 1: Test set contains ONLY Browsing (0 Real-Time, 0 Bulk, 0 Streaming).
  * Fold 2: Test set contains 0 Bulk, 0 Browsing, 0 Streaming.
  * Fold 4: Test set contains 0 Browsing.
  * *Verdict:* Mathematically broken. You cannot divide 3 groups across 5 folds without at least 2 folds missing the class entirely.
* **Standard `GroupKFold(n_splits=3)`:** **FAILS**.
  * Fold 1 test set is monopolized by `GRP_EMAIL_1` (5,870 flows), containing zero flows from the other 3 classes.

### 3.3 The StratifiedGroupKFold Solution
To overcome group imbalance, scikit-learn provides `StratifiedGroupKFold`, which attempts to preserve class distribution while respecting group boundaries. Our empirical simulations confirmed:
* **`StratifiedGroupKFold(n_splits=3)` [4-Class]:** **PASSED 100% CLASS COVERAGE**.
  * Fold 1: Real-Time (3,208), Streaming (443), Browsing (305), Bulk Data (204). All 4 present.
  * Fold 2: Real-Time (2,471), Bulk Data (945), Browsing (672), Streaming (403). All 4 present.
  * Fold 3: Browsing (5,870), Real-Time (2,453), Streaming (444), Bulk Data (199). All 4 present.
* **`StratifiedGroupKFold(n_splits=5)` [3-Class Pure QoS]:** **PASSED 100% CLASS COVERAGE**.
  * When Email is excluded, all 3 remaining classes have 9 to 10 groups each. 5-fold cross-validation achieves smooth, balanced class and group distributions across every fold.

---

## 4. Domain Shift: Non-VPN vs. VPN Encapsulation

We quantified feature distribution divergence between direct (Non-VPN) and encapsulated (VPN) traffic across matching applications:

| Feature Dimension | Non-VPN (10,243 flows) | VPN (7,374 flows) | Distributional Shift ($\Delta$) | Physical Networking Cause |
| :--- | :---: | :---: | :---: | :--- |
| **Mean Packet Size** | $119.18\text{ B} \pm 177.98$ | $177.40\text{ B} \pm 137.37$ | **+48.8% increase** | OpenVPN/IPsec tunnel header overhead and crypto padding. |
| **Mean Flow Duration**| $15.36\text{ s} \pm 73.29$ | $23.83\text{ s} \pm 197.64$ | **+55.1% increase** | Increased latency, handshake negotiation, and tunnel keep-alives. |
| **Mean Packet Rate** | $1,231.93\text{ pkts/s}$ | $1,484.32\text{ pkts/s}$ | **+20.5% increase** | Periodic tunnel control frames multiplexed onto data channel. |
| **Fwd/Bwd Packet Ratio**| $2.40 \times 10^6$ (high variance)| $3.61 \times 10^5$ | Substantial compression | Tunnel encapsulation smooths extreme unidirectional bursts. |

### Domain Shift Recommendation:
Models trained exclusively on direct Non-VPN traffic will experience significant out-of-distribution performance drops if evaluated naively on VPN traffic. Phase 3 must include an **in-domain vs. cross-domain evaluation track** (Train on Non-VPN $\to$ Test on VPN).

---

## 5. Resampling Integrity & Evaluation Metrics

1. **Strict Prohibition of SMOTE / Synthetic Oversampling:**
   * Synthetic Minority Over-sampling (SMOTE) generates synthetic samples by taking linear combinations of feature vectors: $\vec{x}_{\text{new}} = \vec{x}_i + \lambda (\vec{x}_j - \vec{x}_i)$.
   * *The Networking Violation:* In network flow data, features represent discrete physical states (integer packet counts, valid TCP flag masks, discrete MTU bounds). Interpolating between a UDP VoIP flow and a TCP transfer creates non-physical Frankenstein feature vectors.
   * *The Valid Solution:* **Cost-Sensitive Learning / Class-Weighted Loss Functions** ($w_c = \frac{N}{K \cdot N_c}$).
2. **Mandatory Evaluation Metrics:**
   * **Macro F1-Score:** Treats all classes equally, preventing majority class dominance.
   * **Balanced Accuracy:** Average recall across all classes, immune to the 6.30:1 class skew.
   * **Confusion Matrix:** Required to inspect exact cross-class confusion.
   * **Raw Overall Accuracy:** FORBIDDEN as primary metric.

---

## 6. Strategic Validation Protocol for Phase 3

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        PHASE 3 DUAL-TRACK VALIDATION PROTOCOL                          │
├────────────────────────────────────────┬───────────────────────────────────────────────┤
│ Track A: 4-Class Baseline              │ Track B: 3-Class Pure QoS Benchmark           │
├────────────────────────────────────────┼───────────────────────────────────────────────┤
│ Target Classes: Bulk, Browsing,        │ Target Classes: Bulk, Streaming, Real-Time    │
│                 Streaming, Real-Time   │                 (Email completely excluded)   │
│ Flow Count    : 17,617 flows           │ Flow Count    : 10,770 flows                  │
│ Capture Groups: 32 Groups              │ Capture Groups: 29 Groups (9-10 per class)    │
│ Validation    : StratifiedGroupKFold   │ Validation    : StratifiedGroupKFold          │
│ Fold Count    : n_splits = 3           │ Fold Count    : n_splits = 5                  │
│ Primary Metric: Macro F1 & Bal Acc     │ Primary Metric: Macro F1 & Bal Acc            │
│ Purpose       : Benchmark vs Huang 2021│ Purpose       : Clean, uncorrupted QoS test   │
└────────────────────────────────────────┴───────────────────────────────────────────────┘
```
