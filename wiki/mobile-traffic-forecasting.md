---
title: "Mobile Traffic Forecasting"
type: concept
tags:
  - mobile-networks
  - wireless-communication
  - traffic-analysis
  - network-planning
created: 2026-06-08
last_updated: 2026-06-09
source_count: 2
confidence: high
status: active
---

# Mobile Traffic Forecasting

Mobile traffic forecasting is the task of predicting or generating mobile network traffic patterns — the volume of data transmitted over wireless channels between mobile devices and base stations (BSs) — to enable proactive network planning and optimization[^src-uomo].

## Distinction from Traffic Forecasting

While [[traffic-forecasting|general traffic forecasting]] typically addresses vehicle traffic (speed, flow, occupancy from road sensors), mobile traffic forecasting focuses on **wireless network traffic**: aggregated data volumes at base stations, cell-level user counts, and quality-of-service metrics. The data sources are fundamentally different: cellular network measurements (MR data at millisecond granularity, PM data at 15-minute intervals) vs. road sensors[^src-uomo].

## Tasks

[[uomo|UoMo]] (KDD 2025) formalizes three core tasks[^src-uomo]:

1. **Short-term prediction**: Long history (e.g., 48 steps) to predict near-future dynamics (e.g., 16 steps). Used for real-time resource allocation, user access control, and improving live network user experience.

2. **Long-term prediction**: Limited history (e.g., 16 steps) to forecast extended future patterns (e.g., 48 steps). Captures inherent periodical patterns. Used for BS deployment planning, cell dormancy, and network capacity expansion.

3. **Generation**: No historical data; generates traffic distribution purely from contextual features (POI distribution, urban layout). Used for greenfield deployment in new regions lacking historical measurements.

## Key Challenges

Unlike general time series, mobile traffic has unique characteristics[^src-uomo]:

- **Heterogeneous collection**: Varying time granularities (millisecond MR vs. 15-min PM) and spatial scopes (cell-level vs. regional)
- **Urban context dependence**: Strongly shaped by population distribution, human mobility patterns, and geographical layout (POIs)
- **Multi-city variability**: Different cities have different geographic environments, lifestyle habits, and urban layouts, requiring strong generalization
- **Multiple optimization scenarios**: Different network optimization tasks require fundamentally different forecasting types

## Historical Approaches

Early methods used statistical approaches (ARIMA, HA) and simulation techniques[^src-uomo]. Machine learning brought LSTM-based models for temporal dependencies, GCN+Transformer hybrids for spatio-temporal correlations, and GAN-based methods for traffic generation. Recent work includes diffusion models (Open-Diff, CSDI) and reprogramming LLMs (Time-LLM, Tempo) for time series forecasting.

## Universal Models

[[uomo|UoMo]] (KDD 2025) is the first universal model unifying all three tasks under one framework, deployed at China Mobile. Prior to UoMo, models were task-specific (one model per forecasting type), increasing deployment complexity and computational overhead[^src-uomo].

[[st-vision-llm|ST-Vision-LLM]] (Yang et al., arXiv 2025) addresses grid-based mobile traffic forecasting on the Telecom Italia dataset (Milan, Trentino) by reframing it as a vision-language task — rendering global traffic matrices as images for a Vision-LLM and generating per-cell forecasts as single numerical tokens, with strong few-shot, cross-domain, and zero-shot transfer[^src-st-vision-llm].

## Related Pages

- [[uomo]] — UoMo, first universal mobile traffic forecasting model
- [[traffic-forecasting]] — general traffic forecasting (vehicle traffic)
- [[spatio-temporal-foundation-model]] — ST foundation model landscape
- [[jiutian-platform]] — China Mobile's AI deployment platform

[^src-uomo]: [[source-uomo]]
[^src-st-vision-llm]: [[source-st-vision-llm]]
