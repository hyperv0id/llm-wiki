---
title: "NBEATSx: Neural Basis Expansion Analysis with Exogenous Variables"
type: source-summary
tags:
  - time-series
  - forecasting
  - exogenous
  - nbeats
  - electricity-price-forecasting
  - interpretable
  - 2022
created: 2026-07-13
last_updated: 2026-07-13
source_count: 1
confidence: high
status: active
---

# NBEATSx: Neural Basis Expansion Analysis with Exogenous Variables

**Authors**: Kin G. Olivares, Cristian Challu, Grzegorz Marcjasz, Rafał Weron, Artur Dubrawski (CMU Auton Lab; Wrocław University of Science and Technology).[^src-nbeatsx]

**Venue**: *International Journal of Forecasting* (2022) | **arXiv**: 2104.05522v6 (4 Apr 2022) | **Domain showcase**: day-ahead electricity price forecasting (EPF)

## Summary

NBEATSx ([[nbeatsx]]) extends NBEATS so that temporal and static exogenous covariates can enter the neural basis-expansion stack. Fully connected residual blocks still learn backcast/forecast expansion coefficients, but new exogenous stacks either use the covariates as an explicit linear basis (interpretable configuration) or encode them with a temporal convolutional / WaveNet-style subnetwork into a context basis (generic configuration). On five open day-ahead EPF markets with two-year hold-outs, ensembled NBEATSx improves ~20% over original NBEATS/ESRNN without time-dependent covariates and up to ~5% over specialized LEAR and DNN EPF baselines, while the interpretable configuration decomposes forecasts into trend, seasonality, and exogenous effects.[^src-nbeatsx]

## Core Arguments

**1. Exogenous stacks close a practical gap in pure AR neural basis expansion.** Original NBEATS and ESRNN excelled on M4-style univariate series but omit time-dependent covariates required by EPF (day-ahead load, wind/solar generation). NBEATSx keeps doubly residual stack aggregation while admitting static IDs and calendar/domain temporal covariates through dedicated stacks.[^src-nbeatsx]

**2. Interpretable vs generic exogenous bases.** Interpretable NBEATSx-I uses polynomial trend, Fourier seasonality, and direct exogenous basis expansion \(\hat y^{\mathrm{exog}}_{s,b}=X\theta^{\mathrm{exog}}_{s,b}\). Generic NBEATSx-G learns free basis vectors (identity / FCNN-like) plus an exogenous encoder \(C_{s,b}=\mathrm{TCN}(X)\) (or WaveNet) whose context multiplies forecast coefficients—convolutions act as weighted moving-average filters while the final linear projection remains a decoder.[^src-nbeatsx]

**3. Signal decomposition without sacrificing accuracy.** Additive stack forecasts yield classical level/trend/seasonality plus covariate partial forecasts. On a high-load NP day, NBEATS-I residual bias shrinks once load/generation enter the exogenous stack; quantitative tables show no significant accuracy–interpretability trade-off between NBEATSx-G and NBEATSx-I ensembles.[^src-nbeatsx]

**4. Rigorous EPF protocol.** Five markets (NP, PJM, EPEX-BE/FR/DE), six years of hourly data, two-year out-of-sample tests, daily recalibration with early stopping, Hyperopt over architecture/regularization, four-member mean ensembles (data-augmentation sampling × early-stop choice), MAE/rMAE/sMAPE/RMSE plus Giacomini–White tests against AR1/ARx1, LEARx, DNN, ESRNN, and NBEATS.[^src-nbeatsx]

## Experiments (highlights)

- **Ensemble vs no-exo neural nets:** average ~18.8% better than NBEATS and ~20.6% better than ESRNN across metrics/markets.[^src-nbeatsx]
- **Vs specialized EPF ML:** ensemble RMSE/MAE/rMAE/sMAPE improve ~4.7/2.5/2.0/1.3% on average vs DNN; market-level gains up to ~5.4% (NP); GW tests show no market where LEAR/DNN significantly beat NBEATSx-G/I.[^src-nbeatsx]
- **Cost:** day-ahead inference on the order of milliseconds; recalibration ~50% slower than the parsimonious DNN but still practical for daily retrain.[^src-nbeatsx]

## Limitations / Scope

Univariate target series with numerical (and static categorical) covariates—not multimodal image/text exogenous ST settings later studied by TimeXer/ExoST-class work. Point forecasts only (MAE training); probabilistic heads are out of scope. Hyperparameter search is heavy (1500 Hyperopt trials in the reported protocol). Suggested extensions include wavelet bases, spline covariate encoders, and smoothness regularization on generic stacks.[^src-nbeatsx]

## Related Pages

- Entity: [[nbeatsx]]
- Later exogenous forecasting: [[source-tft]], [[source-tide]], [[source-timexer]], [[source-exotst]], [[source-exost]], [[source-crosslinear]], [[source-exollm]]
- Related covariate concepts: [[heterogeneous-covariates]], [[texts-as-auxiliary-variables]]

---

[^src-nbeatsx]: [[source-nbeatsx]]
