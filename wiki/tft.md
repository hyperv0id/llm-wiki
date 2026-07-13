---
title: "Temporal Fusion Transformer (TFT)"
type: entity
tags:
  - time-series
  - forecasting
  - transformer
  - multi-horizon
  - interpretability
  - exogenous
  - quantile
  - google
  - 2020
created: 2026-07-13
last_updated: 2026-07-13
source_count: 1
confidence: high
status: active
---

# Temporal Fusion Transformer (TFT)

**TFT** is an attention-based multi-horizon time-series forecaster by Lim, Arık, Loeff & Pfister (Google Cloud AI / Oxford; arXiv:1912.09363, 2020). It is designed for practical forecasting with **static covariates**, **past-observed** exogenous inputs, and **a priori known future** inputs, producing simultaneous multi-horizon **quantile** forecasts with built-in interpretability.[^src-tft]

## Problem Setting

For each entity \(i\), TFT maps a look-back window of targets and observed inputs, known inputs over past and future, and static metadata \(s_i\) to \(\tau \in \{1,\ldots,\tau_{\max}\}\) quantile forecasts:

\[
\hat y_i(q,t,\tau)=f_q\bigl(\tau,\, y_{i,t-k:t},\, z_{i,t-k:t},\, x_{i,t-k:t+\tau},\, s_i\bigr).
\]

This is a **[[direct-forecast|direct]]** multi-horizon model (all horizons in one forward pass), not an iterative one-step recursive forecaster.[^src-tft]

## Architecture

Major constituents (paper Fig. 2):[^src-tft]

1. **[[gated-residual-network|Gated Residual Network (GRN)]]** + **[[glu-gated-linear-unit|GLU]]** — adaptive depth; skip unused nonlinear paths.
2. **[[variable-selection-network|Variable selection networks]]** — instance-wise Softmax weights for static / past / future inputs after entity embeddings (categorical) or linear maps (continuous).
3. **Static covariate encoders** — four context vectors \(c_s,c_e,c_c,c_h\) for selection, static enrichment, and LSTM cell/hidden initialization.
4. **Temporal fusion decoder**:
   - LSTM sequence-to-sequence **local processing** over unequal past vs future lengths (positional-encoding substitute);
   - static enrichment GRN;
   - decoder-masked **[[interpretable-multi-head-attention|interpretable multi-head attention]]** (shared values, averaged heads);
   - position-wise GRN + gated residual over the whole transformer block.
5. **Quantile heads** — linear maps per \(q\) (experiments use \(\{0.1,0.5,0.9\}\)); train with multi-horizon quantile loss; report P50/P90 q-Risk.

```
static / past / future inputs
  → variable selection (GRN + Softmax)
  → LSTM seq2seq local features φ
  → static enrichment → interpretable self-attention → GRN
  → quantile outputs ŷ(q, t, τ)
```

## Empirical Position

On Electricity, Traffic, Retail (Favorita), and Volatility, TFT outperforms DeepAR, DSSM, ConvTrans, Seq2Seq, and MQRNN; median forecasts average ~7% lower P50 and ~9% lower P90 than the next-best model in the paper’s tables. Ablations attribute large gains to local processing and self-attention, with static encoding, variable selection, and gating contributing dataset-dependently (gating especially on small/noisy Volatility).[^src-tft]

## Interpretability Use Cases

TFT aggregates selection and attention weights **across the dataset** (not only single instances):[^src-tft]

| Use case | Mechanism | Example |
|----------|-----------|---------|
| Variable importance | Distribution of selection weights \(v_{\chi t}\) | Retail: item/store IDs; past log-sales; future promotions/holidays |
| Persistent temporal patterns | Lag attention \(\alpha(t,n,\tau)\) | Daily spikes (Electricity/Traffic); weekly + recency (Retail) |
| Regime / event detection | Distance of attention maps to average pattern | High-volatility regimes around 2008 S&P 500 |

## Historical Position

TFT is an early **covariate-complete** deep multi-horizon baseline: it codifies static / known / observed inputs and quantile intervals years before the LTSF Transformer boom. Later residual exogenous MLPs ([[nbeatsx|NBEATSx]], [[tide|TiDE]]) emphasize efficient dense maps; later Transformer exogenous models ([[source-timexer|TimeXer]], [[source-exotst|ExoTST]]) reframe endo/exo token fusion; ST select-then-balance ([[source-exost|ExoST]]) and linear plug-ins ([[source-crosslinear|CrossLinear]]) push many-to-one exogenous efficiency. Heterogeneous **image/text** covariates and TSFM adapters ([[heterogeneous-covariates|UniCA]]-class) remain outside TFT’s original scope. TFT remains a standard reference for **interpretable multi-horizon** architecture design and a frequent baseline in exogenous forecasting tables.[^src-tft]

## Limitations

- Tabular numerical + categorical covariates only (not multimodal exogenous).
- Quantile bands, not full generative densities.
- Fixed horizon set; single interpretable attention layer in the paper’s explainability setup.
- Pre-LTSF benchmark suite; not optimized for ultra-long pure multivariate LTSF without rich covariates.[^src-tft]

## Connections

- Paper: [[source-tft]]
- Building blocks: [[gated-residual-network]], [[variable-selection-network]], [[interpretable-multi-head-attention]], [[glu-gated-linear-unit]]
- Paradigm: [[direct-forecast]]
- Related exogenous models: [[nbeatsx]], [[tide]], [[source-timexer]], [[source-exotst]], [[source-exost]], [[source-crosslinear]]
- Covariate taxonomy: [[heterogeneous-covariates]]

---

[^src-tft]: [[source-tft]]
