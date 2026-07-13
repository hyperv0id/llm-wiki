---
title: "Temporal Decoder"
type: technique
tags:
  - time-series
  - forecasting
  - covariates
  - mlp
  - residual-block
  - TiDE
created: 2026-07-13
last_updated: 2026-07-13
source_count: 1
confidence: high
status: active
---

# Temporal Decoder

**Temporal decoder** is the per-horizon-step residual fusion head introduced in [[tide|TiDE]]. After a dense decoder emits a decoded vector $d_t$ for each future step $t$, the temporal decoder maps the concatenation of $d_t$ and the projected future covariates $\tilde x_{L+t}$ to a scalar prediction $\hat y_{L+t}$.[^src-tide]

## Motivation

Flattening covariates only inside a global encoder can bury step-local signals (promotions, holidays, event indicators). TiDE therefore adds an explicit **highway** from known future covariates at time $L+t$ to the prediction at the same step, so strong contemporaneous effects need not be recovered only through deep residual paths.[^src-tide]

## Formulation

For each horizon index $t \in [H]$:

$$
\hat y^{(i)}_{L+t} = \mathrm{TemporalDecoder}\big(d^{(i)}_t ; \tilde x^{(i)}_{L+t}\big)
$$

where TemporalDecoder is a residual MLP block with output size 1, and $\tilde x$ comes from the shared feature-projection residual block (dimensionality reduction of raw dynamic covariates). Hyperparameter `temporalDecoderHidden` controls the residual hidden width.[^src-tide]

## Empirical Evidence

On a semi-synthetic Electricity dataset with Type A/B event covariates that multiplicatively spike/drop series for 24-hour blocks, TiDE **with** the temporal decoder adapts after one training epoch both during the event and immediately after; without it, post-event predictions remain disrupted because the model has not yet re-normalized its use of the contaminated look-back.[^src-tide]

On M5, TiDE's ability to consume static attributes plus rich dynamic covariates (promotions/events), of which the temporal decoder is a key interface for future-known dynamics, yields large WRMSSE gains over PatchTST (no covariates) and DeepAR.[^src-tide]

## Relation to Broader Designs

- Complements TiDE's **global linear residual** (look-back→horizon), which preserves [[ltsf-linear|DLinear]]-style pure temporal linear maps.[^src-tide]
- Related in spirit to later exogenous fusion modules that keep a direct path from future exogenous variables into the forecast (e.g., TimeXer / ExoTST-style future-exo pathways), though TiDE implements the path as a residual MLP rather than cross-attention.[^src-tide]
- Operates under [[channel-independence|channel independence]]: fusion is per series, not across series.

## Related

- [[tide]], [[source-tide]]
- [[channel-independence]], [[direct-forecast]], [[lstf]]
- [[ltsf-linear]], [[patchtst]]

---

[^src-tide]: [[source-tide]]
