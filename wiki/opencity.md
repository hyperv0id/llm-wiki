---
title: "OpenCity"
type: technique
tags:
  - spatio-temporal
  - foundation-model
  - traffic-prediction
  - zero-shot
  - transformer
  - gnn
created: 2026-05-03
last_updated: 2026-06-01
source_count: 1
confidence: medium
status: active
---

# OpenCity

**OpenCity** is a spatio-temporal foundation model for traffic prediction that enables **zero-shot generalization across unseen cities** without any fine-tuning on target data[^src-opencity]. Proposed by Li et al. (HKU / SCUT / Baidu, 2024), it integrates Transformer and GNN architectures with instance normalization and patch embedding to handle the extreme distribution shifts inherent in cross-city traffic data[^src-opencity].

## Design Philosophy

OpenCity pursues a **"one-for-all" paradigm**: a single pre-trained model that works out-of-the-box for any city, any traffic metric (flow, speed, demand), without requiring training samples from the target[^src-opencity]. This contrasts with:
- **[[unist|UniST]]** (KDD 2024): one-for-all but requires prompt fine-tuning on target data
- **UrbanGPT** (KDD 2024): LLM-based, slow inference (~45,000s), needs POI text
- **MoST** (KDD 2026): multi-modal but requires satellite/POI data

OpenCity's key insight: **generalization comes from eliminating dependencies on training data statistics, not from learning alignment mechanisms**[^src-opencity].

## Architecture

### 1. Spatio-Temporal Embedding Layer

**Instance Normalization** replaces Z-score normalization[^src-opencity]:
$$\bar{X}_{r,t} = \frac{X_{r,t} - \mu_r}{\sigma_r}$$
Each region $r$ is normalized by its own input mean $\mu_r$ and std $\sigma_r$. After prediction, inverse normalization restores the original scale: $\hat{Y}_r = \hat{\bar{Y}}_r \cdot \sigma_r + \mu_r$. This eliminates all dependence on training-set statistics — the model sees uniformly-scaled values regardless of source city or metric type[^src-opencity].

**Patch Embedding** segments temporal sequences into non-overlapping patches[^src-opencity]:
$$E_r = W_e \cdot \text{Patchify}(\bar{X}_r) + \text{PE}, \quad P=12, \; S=12$$
288 time steps (1 day @ 5min) → 24 patches. Benefits: (1) $O(24^2)$ attention complexity instead of $O(288^2)$; (2) robustness to temporal distribution shift; (3) handles different sampling frequencies by adjusting patch length[^src-opencity].

### 2. Spatio-Temporal Context Encoding

**Temporal context**: hour-of-day and day-of-week are discretized, embedded into $d/2$-dim vectors via linear layers, and concatenated into a $d$-dim temporal encoding $D$[^src-opencity]. This tells the model: "it's Monday, 8 AM."

**Spatial context**: Laplacian eigenvector decomposition of the normalized graph Laplacian $\triangle = I - D^{-1/2} A D^{-1/2}$[^src-opencity]:
$$C = W_c \Phi, \quad \Phi = U_{[:, :k]}, \; k=8$$
The $k$ smallest non-trivial eigenvectors encode topological proximity — nearby, structurally-similar nodes yield similar eigenvectors. This requires only the road graph, no external POI/demographic data[^src-opencity].

### 3. TimeShift Transformer

A two-stage attention mechanism decouples periodic and dynamic traffic signals[^src-opencity]:

**PTTM (Periodic Traffic Transition Modeling)** — captures daily/weekly rhythms:
$$M^h_r = \text{softmax}\left(\frac{\delta_a(Q^h_r {K^h_r}^T)}{\sqrt{d_h}}\right) V^h_r$$
where $Q$ comes from future time embedding + spatial context, $K$ from historical time embedding + spatial context, and $V$ from data embedding. This cross-attention explicitly links "yesterday 8 AM" to "tomorrow 8 AM"[^src-opencity].

**DTP (Dynamic Traffic Pattern learning)** — captures anomalies:
A second self-attention pass over PTTM outputs $M$ (Q=K=V all from $M$) to detect non-recurring patterns like accidents or rain-induced surges[^src-opencity].

Both stages use RMSNorm for training stability; feed-forward uses SwiGLU activation[^src-opencity].

### 4. GCN Spatial Aggregation

Mixed self/neighbor aggregation with a balancing parameter $\alpha=0.05$[^src-opencity]:
$$G_t = \delta [\alpha H_t + (1-\alpha)(W_g \bar{A} H_t)]$$
The small $\alpha$ strongly prioritizes the region's own signal — crucial for zero-shot deployment where the target city's graph topology may differ substantially from training graphs[^src-opencity].

### 5. Output

Features from $L$ stacked layers are flattened and projected via a linear layer to produce predictions $\hat{Y} \in \mathbb{R}^{R \times F}$. Training loss is MAE[^src-opencity].

## Pre-Training

**Data**: 21 datasets across 10,110 regions, 352,796 time points, ~151M observations. Covers traffic flow (CAD-X, PEMS-X), traffic speed (METR-LA, PEMS-BAY, TrafficX), taxi demand (NYC-TAXI, CHI-TAXI)[^src-opencity]. Per-epoch: one batch from each dataset, random order. Hardware: 8× NVIDIA A100-SXM4-40GB[^src-opencity].

## Key Results

| Setting | Dataset | OpenCity | Best Full-Shot | Win? |
|---------|---------|----------|----------------|------|
| Zero-shot | CAD3 (flow) | MAE 15.88 | GWN 16.94 | ✅ |
| Zero-shot | TrafficSH (speed) | MAE 0.55 | ASTGNN 0.69 | ✅ |
| Zero-shot | CHI-TAXI (demand) | MAE 1.91 | STGCN 3.09 | ✅ |
| Zero-shot | NYC-BIKE (bike)* | MAE 6.32 | ASTGNN 6.44 | ✅ |
| Zero-shot | CAD5 (flow) | MAE 11.09 | GWN 10.69 | ~ (3.7% gap) |
| Zero-shot | PEMS07M (speed) | MAE 4.50 | GWN 4.17 | ~ (7.9% gap) |
| Supervised | CAD8-1 (flow) | MAE 17.95 | PDFormer 23.43 | ✅ (23.4%↓) |

*NYC-BIKE was completely absent from pre-training data[^src-opencity].

**vs foundation models** on CHI-TAXI: OpenCity_mini (2M params) MAE=1.74 vs [[unist|UniST]] 2.94, UrbanGPT 3.26; inference 1.5s vs 45,000s[^src-opencity].

**Fast adaptation**: with 3 epochs of prediction-head fine-tuning, OpenCity on SZ-DIDI achieves MAE 2.42 vs best full-shot baseline 2.87, with training time 2.8s vs 46.8s (1.7% of baseline time)[^src-opencity].

**Scaling**: performance improves from 2M→5M→26M parameters, but with diminishing returns — similar to UniST's observation of a scalability ceiling in spatio-temporal data[^src-opencity].

## Ablation Insights

Removing components, ranked by severity of degradation[^src-opencity]:
1. **STC (spatio-temporal context encoding)** — most critical: model loses awareness of spatial and temporal identity
2. **SDM (spatial dependency modeling)** — GCN neighborhood aggregation provides essential auxiliary signal even in zero-shot
3. **PTTM (periodic modeling)** — explicit "history→future" cross-attention is irreplaceable
4. **DTP (dynamic modeling)** — anomaly detection contributes meaningfully to zero-shot performance

## Deployment Efficiency

Single-prediction latency on one NVIDIA A100: <3 seconds regardless of city size (77–896 regions, 3,936–138,240 data points)[^src-opencity]. This makes OpenCity viable for real-time deployment with a 5-minute sampling interval[^src-opencity].

## Limitations

- Requires predefined adjacency matrix for Laplacian spatial encoding; cannot handle cities without road graph data[^src-opencity]
- Single-modal: numerical time series only, no text/image modalities
- Pre-training did not cover bike trajectory data; cross-category zero-shot accuracy remains limited for truly novel data types[^src-opencity]
- Scaling benefits diminish beyond ~10M parameters[^src-opencity]

## Lineage

OpenCity is the culmination of the HKU DAO Lab's (Chao Huang group) spatio-temporal pre-training roadmap[^src-opencity]:
- **GPT-ST** (NeurIPS 2023): MAE pre-training for ST graphs, same first author (Zhonghang Li)
- **FlashST** (ICML 2024): prompt-tuning framework for lightweight ST adaptation
- **OpenCity** (2024): the endpoint — "pre-training + fine-tuning" → "pre-training + zero-shot"

## Related Pages

- [[gpt-st]] — GPT-ST, pre-training framework for ST graphs (same lab, preceding work)
- [[urbangpt]] — UrbanGPT (KDD 2024), first spatio-temporal LLM, zero-shot via instruction-tuning (same first author)
- [[spatio-temporal-foundation-model]] — overview of ST foundation model paradigm
- [[most]] — MoST, multi-modal ST foundation model (KDD 2026)
- [[urbandit]] — UrbanDiT, diffusion transformer for open-world ST prediction (NeurIPS 2025)
- [[unist]] — UniST, one-for-all ST foundation model with MAE pre-training + prompt learning (KDD 2024, predecessor)
- [[urbanfm]] — UrbanFM, scaling-centric ST foundation model, 100+ cities, 39-70% zero-shot gains (arXiv 2026)
- [[uniflow]] — UniFlow, unified grid+graph ST foundation model (same FIB Lab, arXiv 2024)
- [[traffic-forecasting]] — general traffic prediction
- [[urbandit-paper-river]] — UrbanDiT citation lineage analysis
- [[source-gpt-st]] — GPT-ST source summary
- [[patchtst]] — PatchTST, origin of patch-based time series tokenization
- [[bigcity]] — BIGCity, first MTMD ST model, extends foundation model scope from OpenCity's traffic-only to trajectory+traffic state (arXiv 2024)
- [[urbanpg]] — UrbanPG, prompt-backbone decoupled ST framework with O(N·d²) linear attention, unifies large-scale + few-shot + continual learning (AAAI 2026)

[^src-opencity]: [[source-opencity]]
