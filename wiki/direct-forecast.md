---
title: "Direct Forecast"
type: concept
tags:
  - time-series-forecasting
  - multi-step-forecast
  - learning-objective
created: 2026-07-13
last_updated: 2026-11-09
source_count: 5
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
| Modern use | Dominant since Informer; used by TimesNet, PatchTST, iTransformer, [[tide|TiDE]][^src-tide]; early multi-horizon quantile exemplar [[tft|TFT]] | Early RNNs / DeepAR-style models |

DF became dominant for long-horizon tasks because of faster inference, simpler implementation, and better empirical accuracy than IF under error accumulation. Earlier multi-horizon direct models such as [[tft|TFT]] and MQRNN already forecast all horizons jointly with quantile heads before the LTSF Transformer wave.[^src-fredf][^src-tft]

## Likelihood Gap

FreDF's Theorem 3.1: the DF MSE objective equals the conditional negative log-likelihood only if $Y_t \perp Y_{t'} \mid L$. When labels are autocorrelated, DF training diverges from maximum-likelihood principles. [[fredf|FreDF]] mitigates this by supervising in an orthogonal frequency domain; [[qdf|QDF]] learns a quadratic $L_\Sigma$ with adaptive $\Sigma$ (autocorrelation + heterogeneous step weights).[^src-fredf][^src-qdf]

[[distdf|DistDF]] ([[source-distdf]], arXiv:2510.24574, ICLR 2026) sharpens the statement: the bias is $\|Y-\hat Y\|^2_{\Sigma_{|x}^{-1}}-\|Y-\hat Y\|^2_2$ under the *conditional* covariance $\Sigma_{|x}$, so it disappears only when future steps are conditionally independent given history, and Fourier/PCA transforms that diagonalize only the marginal covariance leave it in place. DistDF therefore keeps the DF multi-output head but replaces the likelihood objective with [[joint-distribution-wasserstein-alignment|joint-distribution Wasserstein alignment]] between $\mathbb{P}_{Y\mid X}$ and $\mathbb{P}_{\hat Y\mid X}$, using a [[bures-wasserstein|Bures–Wasserstein]] term on joint mean and covariance plus an MSE anchor; inference stays a single forward pass.[^src-distdf] See [[autocorrelation-bias]] for the formal bias and the Traffic case study behind this argument.

## Related

- [[label-autocorrelation]], [[heterogeneous-task-weights]]
- [[frequency-enhanced-direct-forecast]], [[quadratic-form-weighted-objective]], [[joint-distribution-wasserstein-alignment]], [[bures-wasserstein]]
- [[fredf]], [[qdf]], [[distdf]], [[tide]], [[tft]]
- [[source-fredf]], [[source-qdf]], [[source-distdf]], [[source-tide]], [[source-tft]]

---

[^src-fredf]: [[source-fredf]]
[^src-qdf]: [[source-qdf]]
[^src-tide]: [[source-tide]]
[^src-tft]: [[source-tft]]
[^src-distdf]: [[source-distdf]]
