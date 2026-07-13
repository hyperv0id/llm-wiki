---
title: "NBEATSx"
type: entity
tags:
  - time-series
  - forecasting
  - exogenous
  - nbeats
  - electricity-price-forecasting
  - interpretable
  - residual-mlp
  - 2022
created: 2026-07-13
last_updated: 2026-07-13
source_count: 1
confidence: high
status: active
---

# NBEATSx (Neural Basis Expansion Analysis with Exogenous Variables)

**NBEATSx** extends NBEATS (Oreshkin et al., 2020) by admitting static and time-dependent exogenous variables into the neural basis-expansion architecture. Proposed by Olivares, Challu, Marcjasz, Weron & Dubrawski (IJF 2022; arXiv:2104.05522), it is a foundational *univariate + exogenous* deep forecaster, validated primarily on day-ahead electricity price forecasting (EPF).[^src-nbeatsx]

## Problem Setting

Given look-back window \(y^{\mathrm{back}}\in\mathbb{R}^{L}\) of the target series and covariate matrix \(X\) (static attributes plus temporal covariates known over backcast/forecast spans), predict horizon \(y^{\mathrm{for}}\in\mathbb{R}^{H}\). In the paper’s EPF setup, \(L=168\) (one week of hourly lags) and \(H=24\) (day-ahead prices); covariates include day-ahead load and renewable generation forecasts (market-specific).[^src-nbeatsx]

## Architecture

Stacks of residual blocks implement **doubly residual stacking**: each block’s FCNN maps residual inputs to backcast and forecast expansion coefficients \(\theta^{\mathrm{back}},\theta^{\mathrm{for}}\); coefficients project onto stack-specific bases \(V\); backcast residuals clean the input for the next block, while partial forecasts sum within and across stacks to form \(\hat y^{\mathrm{for}}\).[^src-nbeatsx]

```
Input: (y_back, X)
  → Stack 1 … Stack S
      Block b: FCNN → θ_for, θ_back
               basis expansion → ŷ_for, ŷ_back
               residual: y ← y − ŷ_back
  → Global forecast = Σ stack forecasts
```

### Interpretable configuration (NBEATSx-I)

Three specialized stacks (typically):[^src-nbeatsx]

1. **Trend** — polynomial basis \(T=[1,t,\ldots,t^{N_{\mathrm{pol}}}]\)
2. **Seasonality** — Fourier harmonics with period control \(N_{\mathrm{hr}}\)
3. **Exogenous** — direct basis expansion \(\hat y^{\mathrm{exog}}=X\theta^{\mathrm{exog}}\) (time-varying local regression on covariates)

### Generic configuration (NBEATSx-G)

Free (identity) forecast bases behave like classic multi-horizon FCNN heads; an **exogenous encoder stack** learns context \(C=\mathrm{TCN}(X)\) or WaveNet over covariates, then \(\hat y^{\mathrm{exog}}=C\theta^{\mathrm{for}}\). Stack order (identity vs TCN/WaveNet first) is hyperparameter-tuned.[^src-nbeatsx]

### Exogenous variable types

- **Static** — region/product IDs enabling shared multi-series parameters
- **Seasonal/calendar** — harmonic and calendar indicators for periods beyond the look-back window
- **Domain temporal** — EPF load and renewable generation day-ahead forecasts[^src-nbeatsx]

## Empirical Position (EPF)

Five open EPFtoolbox markets (NP, PJM, EPEX-BE/FR/DE), two-year tests, daily recalibration, Hyperopt + 4-model mean ensembling. Ensembled NBEATSx is ~20% better than NBEATS/ESRNN without time-dependent covariates and competitive-to-better than LEARx and DNN EPF specialists (up to ~5% average metric gains; Giacomini–White: no market where those baselines significantly dominate NBEATSx). Inference remains millisecond-scale.[^src-nbeatsx]

## Historical Position

NBEATSx is an early, widely cited bridge from pure univariate neural basis expansion (NBEATS/ESRNN, M4 era) to **covariate-aware** residual deep forecasting. Later long-horizon MLP work such as [[tide|TiDE]] reuses residual dense blocks with stronger future-covariate highways; Transformer exogenous models ([[source-timexer|TimeXer]], [[source-exotst|ExoTST]]) and Linear plug-ins ([[source-crosslinear|CrossLinear]]) reframe many-to-one endo/exo fusion; spatiotemporal select-then-balance ([[source-exost|ExoST]]) and LLM exogenous prompts ([[source-exollm|ExoLLM]]) move beyond numerical univariate EPF. NBEATSx remains a strong interpretable residual-MLP reference for short-horizon exogenous point forecasts.[^src-nbeatsx]

## Limitations

- Univariate target; no graph/spatial coupling
- Numerical/static covariates only (not image/text multimodal exogenous)
- Point forecast focus (MAE); not a probabilistic generative model
- Ensemble + large Hyperopt budget for reported SOTA tables[^src-nbeatsx]

## Connections

- Paper: [[source-nbeatsx]]
- Related exogenous models: [[tide]], [[source-timexer]], [[source-exotst]], [[source-exost]], [[source-crosslinear]]
- Covariate concepts: [[heterogeneous-covariates]]

---

[^src-nbeatsx]: [[source-nbeatsx]]
