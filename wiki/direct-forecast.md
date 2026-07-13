---
title: "Direct Forecast"
type: concept
tags:
  - time-series-forecasting
  - multi-step-forecast
  - learning-objective
created: 2026-07-13
last_updated: 2026-07-13
source_count: 1
confidence: high
status: active
---

## Definition

**Direct forecast (DF)** is a multi-step forecasting paradigm that maps a history window $L \in \mathbb{R}^{L \times D}$ to the entire future window $Y \in \mathbb{R}^{T \times D}$ in one forward pass via a multi-output head $\hat Y = g(L)$, then optimizes a step-wise loss (typically MSE) over all horizon steps jointly.[^src-fredf]

## Contrast with Iterative Forecast (IF)

| | Direct forecast (DF) | Iterative forecast (IF) |
|--|----------------------|-------------------------|
| Generation | All $T$ steps in parallel | One step at a time, feeding predictions back |
| Label structure | Implicitly assumes step independence under MSE | Naturally respects sequential dependence |
| Failure mode | Ignores [[label-autocorrelation]] → biased likelihood training | Error propagation / high variance on long horizons |
| Modern use | Dominant since Informer; used by TimesNet, PatchTST, iTransformer | Early RNNs / DeepAR-style models |

DF became dominant for long-horizon tasks because of faster inference, simpler implementation, and better empirical accuracy than IF under error accumulation.[^src-fredf]

## Likelihood Gap

FreDF's Theorem 3.1: the DF MSE objective equals the conditional negative log-likelihood only if $Y_t \perp Y_{t'} \mid L$. When labels are autocorrelated, DF training diverges from maximum-likelihood principles. [[fredf|FreDF]] mitigates this by supervising in an orthogonal frequency domain; [[source-distdf|DistDF]] later attacks residual [[autocorrelation-bias]] via distributional alignment.[^src-fredf]

## Related

- [[label-autocorrelation]]
- [[frequency-enhanced-direct-forecast]]
- [[fredf]]
- [[source-fredf]]

---

[^src-fredf]: [[source-fredf]]
