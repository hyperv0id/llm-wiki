---
title: "OmniCast: A Masked Latent Diffusion Model for Weather Forecasting Across Time Scales"
type: source-summary
tags:
  - weather-forecasting
  - diffusion
  - latent-space
  - s2s
  - masked-generative
  - neurips-2025
  - vae
created: 2026-07-21
last_updated: 2026-07-21
source_count: 1
confidence: medium
status: active
---

# OmniCast: A Masked Latent Diffusion Model for Weather Forecasting Across Time Scales

Nguyen et al. (UCLA, UCI, Argonne National Laboratory, AI2; NeurIPS 2025) propose **OmniCast**, a scalable probabilistic model unifying weather forecasting across medium-range and subseasonal-to-seasonal (S2S) timescales. OmniCast addresses the fundamental limitation of autoregressive weather models — error accumulation over long horizons — by generating entire sequences of future weather states jointly[^src-omnicast].

## Two-Stage Architecture

**Stage 1 — Continuous VAE**: A VAE compresses each weather state (69 variables × H × W) into a lower-dimensional continuous latent map (e.g., 1024 × 8 × 16 for 1.40625° data, 256 × 45 × 90 for 0.25° data). Unlike VQ-VAE approaches, the continuous latent space achieves a ~100:1 compression ratio (vs. ~3938:1 for discrete), preserving reconstruction fidelity critical for downstream forecasting[^src-omnicast].

**Stage 2 — Masked Generative Transformer**: An MAE-style encoder-decoder transformer models the conditional distribution of future latent tokens given initial conditions. During training, 50–100% of future tokens are randomly masked, and a per-token diffusion MLP head estimates the continuous distribution of masked tokens. An auxiliary weighted MSE loss on the first 10 frames (with exponential decay) improves near-term accuracy[^src-omnicast].

## Key Innovations

- **Joint space-time sampling**: Tokens are unmasked randomly across both spatial and temporal dimensions in an iterative decoding process, avoiding compounding errors of autoregressive rollouts. Random unmasking order produces more diverse ensembles than autoregressive or random-framewise strategies[^src-omnicast].
- **Full-sequence training**: Training on 44 future frames at 24hr intervals enables the model to learn both initial-condition and boundary-condition dynamics critical for S2S[^src-omnicast].
- **Efficiency**: Only one forward pass through the transformer backbone; subsequent diffusion steps use a lightweight MLP head. At 0.25°, OmniCast generates a 15-day forecast in 29s (A100) vs. 480s for GenCast (TPUv5)[^src-omnicast].

## Results

On ChaosBench, OmniCast achieves SOTA S2S performance across RMSE, bias, SSIM, spectral divergence, residual, CRPS, and SSR metrics, matching or exceeding ECMWF-ENS beyond day 10–15. On WeatherBench2 medium-range, it performs competitively with GenCast and IFS-ENS while being 10–20× faster. The model generates stable rollouts up to 100 years[^src-omnicast].

## Limitations

- VAE reconstruction quality imposes an upper bound on forecasting skill; the continuous VAE still smooths fine details[^src-omnicast].
- Does not yet incorporate multimodal inputs (e.g., ocean/land surface boundary conditions) that could further improve S2S skill.
- 32 A100 GPUs × 4 days training is substantial, though still more efficient than comparable methods.

[^src-omnicast]: [[source-omnicast]]
