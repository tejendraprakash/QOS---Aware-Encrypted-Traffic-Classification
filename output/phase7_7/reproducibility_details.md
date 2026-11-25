# Phase 7.7 Reproducibility Details

**Audit Timestamp:** 2026-11-24  
**Operating System:** Windows 11 (64-bit)  
**Python Runtime:** Python 3.13.5 (tags/v3.13.5:6cb20a2)  
**Compiler / Toolchain:** MSC v.1943 64 bit (AMD64)  

---

## 1. Verified Software Dependencies
- `scapy`: 2.6.1
- `numpy`: 2.1.3
- `pandas`: 2.2.3
- `scipy`: 1.17.1
- `scikit-learn`: 1.5.2
- `xgboost`: 3.2.0
- `torch`: 2.6.0+cpu

---

## 2. Dataset Checksums & Specifications
- **Clean Flow Dataset:** `output/phase6d/clean_track_b_flows.csv`
  - Total Flows: 4,196
  - Total Unique Capture Groups: 29
  - Application Classes: Bulk Data ($1,177$), Real-Time ($2,122$), Streaming ($897$)
- **Capture Groups Metadata:** `research_notes/capture_groups.csv`

---

## 3. Exact Reproduction Command
To reproduce the full Phase 7.7 29-fold LOGO benchmarks and nested model selection:
```bash
python scripts/run_phase7_7_nested_experiments.py
```

To run end-to-end PCAP classification and queue simulation with the new champion model:
```bash
python scripts/run_pipeline_end_to_end.py --pcap data/raw/scpDown4.pcap --model output/phase7_7/phase7_7_champion_model.json --regime Moderate --sim-time 60.0
```
