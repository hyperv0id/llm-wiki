---
title: "Flow Matching for Time Series Forecasting"
type: concept
tags:
  - flow-matching
  - time-series
  - generative-model
  - probabilistic-forecasting
  - rectified-flow
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: medium
status: active
---

# Flow Matching for Time Series Forecasting

**Flow Matching for Time Series Forecasting** refers to the application of flow matching (specifically rectified flow) as the generative backbone for probabilistic time series prediction, as introduced by DiTS[^src-dits]. Unlike standard DDPM-based diffusion forecasters, flow matching provides analytical tractability, straighter transport paths, and more efficient sampling[^src-dits].

## Why Flow Matching for Time Series?

Traditional diffusion-based time series models (TimeGrad, CSDI, DiffSTG) use DDPM-style denoising with hundreds of inference steps[^src-dits]. Flow matching offers several advantages for time series[^src-dits]:

1. **Fewer inference steps**: DiTS achieves optimal performance at just **5 sampling steps** — far fewer than DDPM's typical 50–1000[^src-dits].
2. **Straight transport paths**: Rectified flow defines linear interpolation paths $x_t = (1-t)x_0 + t\epsilon$, producing nearly straight ODE trajectories that are easier to solve[^src-dits].
3. **Analytical tractability**: The velocity field $v_\theta$ directly predicts the drift $(\epsilon - x_0)$ rather than the noise, providing clearer gradient signal[^src-dits].
4. **Stability**: No need for the noise schedule engineering required by DDPM[^src-dits].

## DiTS Formulation

DiTS uses **rectified flow** with the following specifics[^src-dits]:

**Forward process** (linear interpolation):
$$x_t = (1-t)x_0 + t\epsilon, \quad t \in [0,1], \quad \epsilon \sim \mathcal{N}(0, I)$$

**Training objective** (conditional flow matching):
$$\mathcal{L}_{\text{velocity}}(\theta) := \mathbb{E}_{x_0, \epsilon, t}\left[\|v_\theta(x_t, t, c) - (\epsilon - x_0)\|^2\right]$$

where $c$ represents exogenous covariates and $v_\theta$ is the MM-DiT-based velocity network[^src-dits].

**Inference** (Euler ODE solver):
$$x_{t-\Delta t} = x_t - v_\theta(x_t, t, c) \Delta t$$

Starting from $x_1 \sim \mathcal{N}(0, I)$ and integrating backward to $t=0$[^src-dits].

## Key Findings from DiTS

### Metric Misalignment with Inference Steps

A surprising finding: as the number of inference steps increases beyond 5[^src-dits]:
- **MSE increases** (worsening deterministic accuracy).
- **CRPS remains stable** (probabilistic quality unchanged).

This reveals a **metric misalignment** in existing evaluation frameworks: traditional deterministic metrics like MSE may penalize the increased stochasticity of more sampling steps, while probabilistic metrics (CRPS) correctly recognize that the distributional quality remains consistent[^src-dits].

This phenomenon is attributed to the **characteristically low information density** of time series data — the generative model reaches its performance ceiling quickly, and additional sampling introduces noise rather than refinement[^src-dits].

### Noise Schedule Selection

DiTS experiments compare three noise schedules for covariate-aware forecasting[^src-dits]:
- **Log-Normal** > Cosine > Linear

The Log-Normal strategy better characterizes the denoising trajectory during critical intermediate stages of the flow[^src-dits].

### v-Prediction Parameterization

DiTS uses **v-prediction** parameterization and the conditional flow matching v-loss, following best practices from image generation literature[^src-dits].

## Comparison to Other Generative TS Methods

| Method | Generative Framework | Inference Steps | Probabilistic Output |
|--------|---------------------|-----------------|---------------------|
| TimeGrad | DDPM (autoregressive) | ~100 | Yes |
| CSDI | Score-based diffusion | ~100 | Yes |
| SimDiff | DDPM (end-to-end) | ~50 | No (point + std) |
| TEDM | EDM-style diffusion | $O(H)$ | Yes |
| StaTS | DDPM (adaptive schedule) | ~10 | Yes |
| **DiTS** | **Rectified flow** | **~5** | **Yes** |

DiTS achieves the fewest inference steps among probabilistic forecasters while maintaining SOTA uncertainty quantification[^src-dits].

## Related Concepts

- [[flow-matching|Flow Matching]] — the general generative framework
- [[dits|DiTS]] — the model that applies flow matching to time series
- [[diffusion-models|Diffusion Models]] — the broader class of generative models
- [[generative-time-series-forecasting|Generative Time Series Forecasting]] — the forecasting paradigm

[^src-dits]: [[source-dits]]
