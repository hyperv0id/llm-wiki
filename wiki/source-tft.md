---
title: "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting"
type: source-summary
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

# Temporal Fusion Transformers (TFT)

**Authors**: Bryan Lim (University of Oxford; Google Cloud AI internship), Sercan Ö. Arık, Nicolas Loeff, Tomas Pfister (Google Cloud AI).[^src-tft]

**Venue / preprint**: arXiv:1912.09363v3 [stat.ML] (27 Sep 2020). **Code**: [google-research/tft](https://github.com/google-research/google-research/tree/master/tft).

## Summary

The paper introduces the **Temporal Fusion Transformer ([[tft|TFT]])** — an attention-based architecture for **interpretable multi-horizon forecasting** with static covariates, past-observed inputs, and a priori known future inputs. TFT is a **[[direct-forecast|direct]]** multi-horizon quantile model built from [[gated-residual-network|GRNs]], instance-wise [[variable-selection-network|variable selection]], static covariate encoders, LSTM local processing, and [[interpretable-multi-head-attention|interpretable multi-head attention]]. On Electricity, Traffic, Retail, and Volatility it improves P50/P90 q-Risk over DeepAR, DSSM, ConvTrans, Seq2Seq, and MQRNN (~7% / ~9% average vs next-best), and supports global analyses of variable importance, seasonality/lags, and regimes.[^src-tft]

## Core Arguments

**1. Heterogeneous multi-horizon inputs.** Practical forecasting mixes static \(s_i\), observed \(z_{i,t}\) (unknown in the future), and known future \(x_{i,t}\). Iterative baselines often assume full future exogenous availability or weak static handling; TFT aligns specialized pathways to this taxonomy.[^src-tft]

**2. Adaptive blocks.** [[gated-residual-network|GRN]] + [[glu-gated-linear-unit|GLU]] skip unused nonlinearity; separate selection networks weight static/past/future features; four static contexts \((c_s,c_e,c_c,c_h)\) condition selection, LSTM init, and enrichment.[^src-tft]

**3. Local + long-range fusion.** LSTM encoder–decoder handles unequal past/future lengths; decoder-masked [[interpretable-multi-head-attention|interpretable multi-head attention]] (shared values, averaged heads) captures long-range structure with interpretable weights.[^src-tft]

**4. Quantiles and global interpretability.** Joint \(q\in\{0.1,0.5,0.9\}\) heads train under quantile loss; aggregated selection/attention yields importance scores, lag patterns, and Bhattacharyya-distance regime detection (e.g. 2008 S&P 500 volatility).[^src-tft]

## Experiments (highlights)

| Dataset | \(k\) | \(\tau_{\max}\) | Notes |
|---------|-------|-----------------|-------|
| Electricity | 168 h | 24 h | Univariate + calendar |
| Traffic | 168 h | 24 h | Skewed occupancy |
| Retail (Favorita) | 90 d | 30 d | Full static/observed/known |
| Volatility | 252 d | 5 d | Small/noisy finance |

Ablations: local processing and self-attention often dominate; static encoding and variable selection matter especially on Electricity; gating helps Volatility. Single-GPU training; one interpretable attention layer for explainability.[^src-tft]

## Limitations / Scope

Tabular numerical/categorical covariates only (not image/text multimodal). Quantile bands, not full generative densities. Fixed \(\tau_{\max}\). Pre-LTSF benchmarks; later work often cites TFT as a strong **covariate-aware** multi-horizon baseline rather than pure LTSF SOTA.[^src-tft]

## Related Pages

- Entity: [[tft]]
- Techniques: [[gated-residual-network]], [[variable-selection-network]], [[interpretable-multi-head-attention]]
- Concepts: [[direct-forecast]], [[heterogeneous-covariates]], [[glu-gated-linear-unit]]
- Later exogenous / covariate models: [[source-nbeatsx]], [[source-tide]], [[source-timexer]], [[source-exotst]], [[source-exost]], [[source-crosslinear]]

---

[^src-tft]: [[source-tft]]
