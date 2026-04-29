---
title: "Traffic Forecasting"
type: concept
tags:
  - time-series
  - spatial-temporal
  - intelligent-transportation
created: 2026-04-27
last_updated: 2026-04-28
source_count: 8
confidence: high
status: active
---

# Traffic Forecasting

Traffic forecasting is the task of predicting future traffic states (speed, flow, occupancy) based on historical observations from sensor networks[^src-hyperd-hybrid-periodicity-decoupling]. It is a core component of Intelligent Transportation Systems (ITS), enabling route planning, traffic control, and congestion management.

## Core Challenges

Two main dependencies must be modeled simultaneously[^src-hyperd-hybrid-periodicity-decoupling]:

- **Spatial correlations** — dependencies among sensors, including road topology, geographic proximity, and similarity of usage patterns
- **Temporal dynamics** — trends, seasonality, daily/weekly rhythms, and abrupt changes over time

## Methods

### Classical
ARIMA, VAR, and SVR fail to capture non-linear spatial correlations[^src-hyperd-hybrid-periodicity-decoupling].

### Deep Graph-Based
The dominant paradigm since DCRNN (2018): Graph Neural Networks (GNNs) combined with temporal models (TCN, RNN, attention). Key milestones include STGCN (2018), ASTGCN (2019), GWNet (2019), STSGCN (2020), STFGNN (2020), D2STGNN (2022), and STGODE (2021)[^src-hyperd-hybrid-periodicity-decoupling].

### Transformer-Based
STTN (2020), GMAN (2020), PDFormer (2023), and STAEformer (2024) use attention mechanisms to model global spatial-temporal dependencies[^src-hyperd-hybrid-periodicity-decoupling].

### Frequency-Domain
FEDformer (2022), FreTS (2023), and StemGNN (2020) apply Fourier transforms but treat components uniformly without separating periodic from residual signals[^src-hyperd-hybrid-periodicity-decoupling].

### Periodicity-Decoupled
[[hyperd|HyperD]] (2025) explicitly decouples short-term and long-term periodicity via hybrid frequency-domain decomposition[^src-hyperd-hybrid-periodicity-decoupling].

### Accident-Aware
Traditional models assume stationary traffic patterns but fail during accidents which create non-stationary perturbations with directional shockwaves. ConFormer (KDD 2026) addresses this through accident-aware graph propagation and Guided Layer Normalization (GLN), achieving up to 10.7% improvement in accident scenarios[^src-conformer].

### Large-Scale Long-Horizon
FaST (KDD 2026) addresses computational bottlenecks in large-scale graphs (8,600+ nodes) with long-horizon predictions (672 steps = 1 week) using [[adaptive-graph-agent-attention|AGA-Att]] for O(N·a) spatial complexity and [[mixture-of-experts|Dense MoE]] for efficient feature extraction. Achieves 4.4%-18.4% MAE improvement over SOTA with 1.3x-2.2x faster inference[^src-fast-long-horizon-forecasting].

## Key Models

Several influential models span the development of traffic and spatial-temporal forecasting:

- **[[source-st-resnet|ST-ResNet]]** (AAAI 2017) — one of the first deep learning approaches for citywide crowd flow prediction, using residual convolutional units to model spatial-temporal dependencies[^src-st-resnet].
- **[[source-astgcn|ASTGCN]]** (AAAI 2019) — combines attention mechanisms with graph convolution to jointly capture spatial and temporal patterns in traffic flow[^src-astgcn].
- **[[source-prnet|PRNet]]** — introduces periodic residual learning to explicitly model recurring temporal patterns in crowd flow forecasting[^src-prnet].
- **[[source-penguin|PENGUIN]]** (AISTATS 2026) — proposes periodic-nested group attention for long-sequence time-series forecasting, with applicability to traffic domains[^src-penguin].

For a comprehensive overview of deep learning methods for time series, including traffic forecasting, the [[source-deep-time-series-survey|TSLib survey]] provides systematic benchmarking across multiple domains[^src-deep-time-series-survey].

## Benchmarks

The standard benchmarks are the PeMS (Caltrans Performance Measurement System) datasets from California highways: PEMS03, PEMS04, PEMS07, PEMS08. Standard setup: 12 input steps (1 hour) → 12 output steps (1 hour)[^src-hyperd-hybrid-periodicity-decoupling].

[^src-hyperd-hybrid-periodicity-decoupling]: [[source-hyperd-hybrid-periodicity-decoupling]]
[^src-st-resnet]: [[source-st-resnet]]
[^src-astgcn]: [[source-astgcn]]
[^src-prnet]: [[source-prnet]]
[^src-penguin]: [[source-penguin]]
[^src-deep-time-series-survey]: [[source-deep-time-series-survey]]
[^src-conformer]: [[source-conformer]]
[^src-fast-long-horizon-forecasting]: [[source-fast-long-horizon-forecasting]]
