---
title: "OpenCity: Open Spatio-Temporal Foundation Models for Traffic Prediction"
type: source-summary
tags:
  - spatio-temporal
  - foundation-model
  - traffic-prediction
  - zero-shot
  - transformer
  - gnn
created: 2026-05-31
last_updated: 2026-05-31
source_count: 1
confidence: medium
status: active
---

# OpenCity: Open Spatio-Temporal Foundation Models for Traffic Prediction

**Authors**: Zhonghang Li, Long Xia, Lei Shi, Yong Xu, Dawei Yin, Chao Huang (HKU, SCUT, Baidu Inc.)
**Venue**: arXiv:2408.10269 (August 2024)
**Code**: https://github.com/HKUDS/OpenCity

## Core Contribution

OpenCity is a spatio-temporal foundation model for traffic prediction that enables zero-shot generalization across unseen regions and cities without any fine-tuning[^src-opencity]. It addresses three key generalization challenges: spatial generalization (cross-region/cross-city), temporal generalization (long-term forecasting beyond one hour), and cross-task generalization (different traffic metrics)[^src-opencity].

## Key Innovations

### Zero-Shot Spatio-Temporal Embedding
Instance normalization replaces Z-score normalization — each input sample is normalized by its own mean and standard deviation ($\bar{X}_{r,t} = (X_{r,t} - \mu_r) / \sigma_r$), eliminating dependence on training set statistics[^src-opencity]. Patch embedding compresses time series into non-overlapping hourly patches ($P=12$ steps, $S=12$), reducing Transformer complexity from $O(T^2)$ to $O(N^2)$ while improving robustness to temporal distribution shifts[^src-opencity].

### TimeShift Transformer
A two-stage attention mechanism decouples periodic and dynamic traffic patterns[^src-opencity]. PTTM (Periodic Traffic Transition Modeling) uses "future→past" cross-attention to capture daily/weekly rhythms (e.g., Monday 8AM ↔ next Monday 8AM). DTP (Dynamic Traffic Pattern learning) applies self-attention to PTTM outputs, capturing non-recurring anomalies like accidents. Both stages use RMSNorm for stability and SwiGLU activation in feed-forward layers[^src-opencity].

### Spatio-Temporal Context Encoding
Temporal context encodes hour-of-day and day-of-week as learned embeddings. Spatial context uses Laplacian eigenvector decomposition of the road network graph ($\Phi = U_{[:, :k]}$), providing zero-external-data spatial embeddings that work for any city with a road graph[^src-opencity].

### GCN Spatial Aggregation
Uses a mixed spatial aggregation $G_t = \delta[\alpha H_t + (1-\alpha)(W_g \bar{A} H_t)]$ with $\alpha=0.05$, balancing raw region information with neighborhood aggregation — critical for zero-shot scenarios where target graph topology differs from training[^src-opencity].

## Experimental Results

Pre-trained on 21 datasets (10,110 regions, 352,796 time points, 151M observations) across traffic flow, speed, taxi demand, and bike trajectories[^src-opencity]. Three model scales: mini (2M), base (5M), plus (26M)[^src-opencity].

**Zero-shot results** (6 datasets, 4 categories)[^src-opencity]:
- CAD3 (flow): MAE 15.88 vs GWN 16.94 (full-shot) — wins
- TrafficSH (speed): MAE 0.55 vs ASTGNN 0.69 — wins
- CHI-TAXI (demand): MAE 1.91 vs STGCN 3.09 — wins
- NYC-BIKE (bike, unseen in pre-training): MAE 6.32 vs ASTGNN 6.44 — wins
- CAD5/PEMS07M: within 8% of best full-shot baselines

**vs foundation models** on CHI-TAXI[^src-opencity]: OpenCity_mini (2M) MAE=1.74 vs UniST 2.94 vs UrbanGPT 3.26; inference 1.5s vs UrbanGPT 45,000s (~30,000× faster). Fast adaptation with 3 epochs of prediction-head fine-tuning further improves performance while requiring only 2-12s training time[^src-opencity].

## Limitations

- Requires a predefined adjacency matrix (road network graph) for Laplacian spatial encoding — cannot handle cities without graph data[^src-opencity]
- Single-modal: operates on numerical time series only, cannot leverage POI/text/satellite imagery[^src-opencity]
- Pre-training did not include bike trajectory data; cross-category zero-shot accuracy remains limited for unseen data types[^src-opencity]
- Scaling law exhibits diminishing returns beyond 10M parameters[^src-opencity]

[^src-opencity]: [[source-opencity]]
