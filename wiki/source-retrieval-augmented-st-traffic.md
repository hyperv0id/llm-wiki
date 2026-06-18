---
title: "RAST: Retrieval-Augmented Spatio-Temporal Traffic Forecasting"
type: source-summary
tags:
  - spatio-temporal
  - traffic-forecasting
  - retrieval-augmented
  - deep-learning
  - time-series
created: 2026-06-18
last_updated: 2026-06-18
source_count: 1
confidence: medium
status: active
---

# RAST: Retrieval-Augmented Spatio-Temporal Traffic Forecasting

RAST introduces a retrieval-augmented mechanism for spatio-temporal forecasting, addressing the limited contextual capacity of traditional STGNNs by maintaining an external memory bank of historical patterns [^src-retrieval-augmented-st-traffic].

## Core Architecture

The framework decouples spatial and temporal encoding. **Dual-dimension feature disentanglement** uses separate encoders: 2D convolutions for temporal features (capturing cyclicity) and graph-based spatial transformations (capturing regionality). A **context-aware query generator** concatenates and projects the decoupled embeddings through residual encoder layers to produce query representations [^src-retrieval-augmented-st-traffic].

The **spatio-temporal retrieval store** uses FAISS for efficient similarity-based indexing, maintaining separate memory banks for spatial and temporal patterns with associated metadata. An information-theoretic retriever selects top-k most relevant patterns using L2 distance. A **momentum-based memory management** system updates the store periodically with exponential moving averages, balancing pattern freshness and stability, with diversity-similarity coefficients for confidence calibration [^src-retrieval-augmented-st-traffic].

**Cross-attention knowledge fusion** uses multi-head attention to combine query embeddings with retrieved spatial and temporal patterns, concatenating the fused retrieval with the original query. A universal backbone predictor (MLP by default, compatible with frozen pre-trained STGNNs) generates final predictions through a residual enhancement pipeline [^src-retrieval-augmented-st-traffic].

## Empirical Results

RAST is evaluated on six traffic datasets (PEMS03/04/07/08, SD, GBA) against 21 baselines including ARIMA, LSTM, Transformer, DCRNN, STGCN, GWNet, and DSTAGNN. It achieves state-of-the-art performance on most datasets, with 8.87% MAE improvement over DSTAGNN on PEMS07 and consistent gains on large-scale datasets (SD, GBA) across all prediction horizons. Ablation studies reveal the query generator as the most critical component (25.6% MAE degradation when removed), with spatial/temporal encoders and the ST-retriever all contributing significantly. RAST demonstrates exceptional computational efficiency, with training times of 154s/epoch on SD and 45s/epoch on GBA, outperforming even lightweight models like STGCN [^src-retrieval-augmented-st-traffic].

Theoretical foundations rest on extending mutual information capacity beyond fixed parameters: the retrieval store adds H(M) to the information bound, enabling capture of more complex dependencies without increasing model size [^src-retrieval-augmented-st-traffic].

[^src-retrieval-augmented-st-traffic]: [[source-retrieval-augmented-st-traffic]]