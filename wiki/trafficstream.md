---
title: "TrafficStream"
type: entity
tags:
  - continual-learning
  - spatio-temporal
  - traffic-forecasting
  - replay-based
  - ijcai-2021
created: 2026-07-22
last_updated: 2026-07-22
source_count: 1
confidence: medium
status: active
---

# TrafficStream

**TrafficStream** (Chen et al., IJCAI 2021) is the first framework to integrate spatio-temporal modeling with continual learning for streaming traffic flow prediction[^src-stbp]. It pioneered the field of [[continual-spatio-temporal-forecasting|continual spatio-temporal forecasting (CSTF)]].

## Core Mechanisms

TrafficStream employs two key strategies to handle long-term streaming traffic data[^src-stbp]:

1. **Historical Data Replay**: Periodically retrains on a buffer of past samples to prevent catastrophic forgetting of previously learned spatio-temporal patterns
2. **Parameter Smoothing**: Regularizes model weights to avoid abrupt changes when adapting to new periods

## Limitations

- **Storage cost**: Maintaining a replay buffer of historical data requires non-trivial storage, especially as the graph expands
- **Privacy concerns**: Storing raw historical sensor data may conflict with data privacy requirements
- **Performance**: Later CSTF methods (STKEC, PECPM, EAC, STBP) significantly outperform TrafficStream on standard benchmarks[^src-stbp]

## Comparison to Later Methods

| Method | Anti-Forgetting Strategy | Backbone Status |
|--------|-------------------------|-----------------|
| TrafficStream | Replay + smoothing | Full training |
| [[pecpm|PECPM]] | Pattern bank (no replay) | Full training |
| [[eac|EAC]] | Frozen backbone + prompt pool | Frozen |
| [[stbp|STBP]] | Frozen backbone + pattern bank | Frozen |

TrafficStream represents the **replay-based** CSTF paradigm, while later methods moved toward parameter-isolation approaches that avoid storing raw data[^src-stbp].

## Related Pages

- [[continual-spatio-temporal-forecasting]] — The CSTF paradigm
- [[stbp]] — STBP, the SOTA CSTF method
- [[eac]] — EAC, prompt-based CSTF

[^src-stbp]: [[source-stbp]]
