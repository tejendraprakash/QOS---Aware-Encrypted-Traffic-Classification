# Comprehensive Confusion Matrix & Error Pattern Analysis

**Project:** QoS-Aware Encrypted Network Traffic Classification for Next-Generation Networks  
**Phase:** 6A — Comprehensive Full-Flow Machine Learning Baseline Experiment  
**Date:** November 2026  
**Artifact Sources:** [confusion_matrix_counts.csv](file:///c:/Users/saipr/OneDrive/Desktop/EncryptedTrafficResearch/output/phase6/track_b/confusion_matrix_counts.csv), [confusion_matrix_normalized.csv](file:///c:/Users/saipr/OneDrive/Desktop/EncryptedTrafficResearch/output/phase6/track_b/confusion_matrix_normalized.csv)

---

## 1. Overview of Pooled Out-of-Fold Confusion Matrices (Track B Primary 3-Class)

The pooled confusion matrix aggregates the out-of-fold predictions across all 10,770 flows in Track B. Because each flow is predicted exactly once when its capture group is in the validation fold, this pooled matrix provides an unbiased snapshot of classifier generalizability.

### 1.1 Random Forest (Track B)
**Raw Counts (Total: 10,770 flows):**
```
                Predicted Bulk Data    Predicted Real-Time    Predicted Streaming    Total Actual
Actual Bulk Data        494                    819                    35                 1,348
Actual Real-Time        934                  5,352                 1,846                 8,132
Actual Streaming         67                    590                   633                 1,290
Total Predicted       1,495                  6,761                 2,514                10,770
```
**Row-Normalized Confusion Rates:**
- **Bulk Data:** **36.65% correct**, 60.76% $\rightarrow$ Real-Time, 2.60% $\rightarrow$ Streaming.
- **Real-Time:** **65.81% correct**, 11.49% $\rightarrow$ Bulk Data, 22.70% $\rightarrow$ Streaming.
- **Streaming:** **49.07% correct**, 5.19% $\rightarrow$ Bulk Data, 45.74% $\rightarrow$ Real-Time.

---

### 1.2 XGBoost (Track B)
**Raw Counts (Total: 10,770 flows):**
```
                Predicted Bulk Data    Predicted Real-Time    Predicted Streaming    Total Actual
Actual Bulk Data        796                    476                    76                 1,348
Actual Real-Time      1,183                  5,140                 1,809                 8,132
Actual Streaming         33                    377                   880                 1,290
Total Predicted       2,012                  5,993                 2,765                10,770
```
**Row-Normalized Confusion Rates:**
- **Bulk Data:** **59.05% correct**, 35.31% $\rightarrow$ Real-Time, 5.64% $\rightarrow$ Streaming.
- **Real-Time:** **63.21% correct**, 14.55% $\rightarrow$ Bulk Data, 22.25% $\rightarrow$ Streaming.
- **Streaming:** **68.22% correct**, 2.56% $\rightarrow$ Bulk Data, 29.22% $\rightarrow$ Real-Time.

---

### 1.3 Packet Multi-Layer Perceptron (P-MLP) (Track B)
**Raw Counts (Total: 10,770 flows):**
```
                Predicted Bulk Data    Predicted Real-Time    Predicted Streaming    Total Actual
Actual Bulk Data      1,083                    198                    67                 1,348
Actual Real-Time      1,398                  4,816                 1,918                 8,132
Actual Streaming        138                    312                   840                 1,290
Total Predicted       2,619                  5,326                 2,825                10,770
```
**Row-Normalized Confusion Rates:**
- **Bulk Data:** **80.34% correct**, 14.69% $\rightarrow$ Real-Time, 4.97% $\rightarrow$ Streaming.
- **Real-Time:** **59.22% correct**, 17.19% $\rightarrow$ Bulk Data, 23.59% $\rightarrow$ Streaming.
- **Streaming:** **65.12% correct**, 10.70% $\rightarrow$ Bulk Data, 24.19% $\rightarrow$ Real-Time.

---

## 2. In-Depth Error Pattern Analysis

### Pattern 1: Bulk Data Absorption by Real-Time in Shallow Trees
- In Random Forest, **60.76% of Bulk Data flows** are misclassified as Real-Time.
- *Root Cause:* Many FTPS and SCP data transfers include short control connections (e.g. FTP PORT/PASV command channels or SSH key negotiation flows) that have small packet counts ($< 20$ packets) and short durations. A greedy tree splits on `packet_count` or `flow_duration`, grouping these control flows with conversational audio bursts.
- *Why P-MLP Solves This:* P-MLP maps the combination of bidirectional packet size variance, TCP window scale parameters, and forward-to-backward byte ratios into continuous latent space, successfully classifying **80.34% of Bulk Data correctly**.

### Pattern 2: The Streaming vs. Real-Time Symmetrical Confusion
- In all three models, a substantial fraction of Streaming flows are misclassified as Real-Time ($45.7\%$ in RF, $29.2\%$ in XGB, $24.2\%$ in P-MLP).
- Symmetrically, **$22.3\%$ to $23.6\%$ of Real-Time flows are misclassified as Streaming** across all models.
- *Network Mechanism:* Both traffic classes deliver compressed video streams using dynamic bitrates (ABR). When a video conference participant is speaking and streaming 720p/1080p camera frames over UDP/HTTPS, the statistical packet signature (mean packet size $\sim 900-1100$ bytes, high burstiness, rapid inter-arrival times) is physically identical to a YouTube or Netflix adaptive video chunk. Without reading unencrypted application headers, separating one-way adaptive video from two-way interactive video represents the hardest classification boundary in encrypted network traffic.

### Pattern 3: Bulk Data vs. Streaming Separation is Highly Accurate
- Misclassification between Bulk Data and Streaming is remarkably low:
  - In XGBoost, only $2.56\%$ of Streaming is predicted as Bulk Data, and $5.64\%$ of Bulk Data is predicted as Streaming.
  - In RF, only $5.19\%$ of Streaming is predicted as Bulk Data, and $2.60\%$ of Bulk Data is predicted as Streaming.
- *Network Explanation:* Bulk Data (FTPS/SCP) saturates TCP window buffers, producing continuous maximum transmission unit packets (1420–1500 bytes) with near-zero inter-arrival jitter in one direction. Streaming traffic operates in episodic chunk bursts (downloading 2–4 seconds of video buffer, then idling for several seconds). The burst dynamics features (`fwd_burst_bytes_mean`, `fwd_burst_pkts_max`) cleanly separate file transfers from media streaming.

---

## 3. Confusion Patterns in Track A (Exploratory 4-Class)

In Track A, the addition of the Email-derived `Browsing` class causes severe network confusion:
- **Browsing Recall Collapses:** RF achieves $7.74\%$, XGB achieves $8.08\%$, and P-MLP achieves $2.63\%$ recall on Browsing.
- **Where Does Browsing Go?** 
  - Over $62\%$ to $82\%$ of all Browsing flows are misclassified as `Real-Time`.
  - Over $24\%$ are misclassified as `Bulk Data`.
- *Empirical Cause:* As verified in the dataset audit, 75.05% of the "Browsing" captures consist of background LLMNR (port 5355) multicast packets sent every few seconds. These periodic, small UDP packets mimic background keep-alive probes of VoIP/Real-Time protocols (e.g. SIP/RTP keep-alives in VoIPBuster and Hangouts). When evaluated on an unseen capture session, the classifier correctly classifies them as low-latency UDP keep-alives (Real-Time), demonstrating the complete failure of the Email proxy label.
