---
title: "TimeDiT"
type: entity
tags:
  - time-series
  - foundation-model
  - diffusion
  - transformer
  - kdd-2025
  - usc
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: medium
status: active
---

**TimeDiT** (Time Series Diffusion Transformer) is a proto-foundation model for multivariate time series proposed by USC researchers (Cao, Ye, Zhang & Liu), accepted at **KDD 2025**[^src-timedit]. It is the first model to unify DiT-style transformer backbone with diffusion probabilistic sampling for time series, establishing a single model that handles forecasting, imputation, anomaly detection, and synthetic data generation.

## What Makes TimeDiT Different

TimeDiT addresses three fundamental gaps in existing time series foundation models[^src-timedit]:

1. **From deterministic to probabilistic**: Existing forecasting FMs (TimesFM, Chronos, Moirai, Lag-Llama) use autoregressive transformers that learn deterministic mappings — TimeDiT uses diffusion sampling to capture inherent uncertainty and stochasticity.

2. **Beyond forecasting-only**: Most time series FMs focus exclusively on forecasting. TimeDiT unifies four downstream tasks through a masking mechanism: forecasting, imputation, anomaly detection, and data generation.

3. **Physics integration without retraining**: Physical knowledge (PDEs) is incorporated at inference time via energy-based Langevin dynamics sampling — a finetuning-free model editing strategy with theoretical guarantees (Boltzmann distribution closed form).

## Architecture

Built on [[dit|DiT]]'s transformer backbone (Peebles & Xie, 2022) adapted for time series[^src-timedit]:

- **Input**: Multivariate time series X ∈ R^(K×L) with padding for missing values and multi-resolution data
- **Tokenization**: Direct linear projection (WYSIWYG), no vector quantization
- **Conditioning**: AdaLN from conditional observations x_con controls scale/shift of noised target
- **Diffusion**: Standard DDPM-style forward/reverse process with T=1000 steps
- **Sizes**: S (33M), B (130M), L (460M), XL (680M) — matching DiT configurations

### Key Architectural Differences from Standard DiT

| Aspect | DiT (Image) | TimeDiT |
|--------|------------|---------|
| Input domain | VAE latent (32×32×4) | Raw time series (K×L) |
| Conditioning | timestep + class label | conditional obs x_con via AdaLN |
| Timestep injection | Separate conditioning pathway | Injected into target noise directly |
| Token structure | Patchify (16×16 patches) | Direct array projection |
| Output | Epsilon-prediction | Mean-prediction μ_θ |

## Training & Data

- **Pre-training data**: Chronos dataset (Ansari et al., 2024), ~5B time points, covering diverse domains
- **Zero-shot**: Single pre-trained checkpoint evaluated across unseen datasets without fine-tuning
- **Fine-tuning**: Available when needed for domain-specific tasks
- **Training setup**: Adam (lr=0.0001), batch 256-512, A100 40G GPUs, >100 epochs

## Performance Profile

### Strengths

- **Zero-shot forecasting**: Avg MSE 0.356 on ETTh1 — outperforms all Moirai variants and competitive with full-shot supervised models[^src-timedit]
- **Task breadth**: Single model for forecasting + imputation + anomaly detection + data generation — widest coverage among time series FMs[^src-timedit]
- **Physics-informed**: Zero-shot physics sampling surpasses fully trained baselines on PDE benchmarks; one model replaces 18 task-specific models[^src-timedit]
- **Inference speed**: 1 second per sample (vs CSDI 2s, Diffusion-TS 6s)[^src-timedit]
- **Uncertainty quantification**: SOTA CRPS with missing values or multi-resolution[^src-timedit]

### Weaknesses

- Sequence length limited to ~198 in experiments; long-sequence performance unverified
- Max channels K_max=20-40; high-dimensional MTS (>100 channels) not tested
- Code not public; checkpoint promised but unreleased as of paper date
- Physics-informed mechanism requires known PDE form; cannot infer physical laws from data

## Comparison with Other Time Series Foundation Models

| Model | Architecture | Tasks | Probabilistic | Zero-shot | Physics |
|-------|-------------|-------|---------------|-----------|---------|
| **TimeDiT** | DiT + Diffusion | F, I, A, G | Yes (diffusion) | Yes | Yes (inference) |
| [[chronos|Chronos]] | T5 encoder-decoder | F only | No | Yes | No |
| [[timesfm|TimesFM]] | Decoder-only | F only | No | Yes | No |
| Moirai | Transformer | F only | No | Yes | No |
| Lag-Llama | Transformer | F only | Yes (AR) | Yes | No |

F=Forecasting, I=Imputation, A=Anomaly Detection, G=Data Generation

## Related Pages

- [[source-timedit]] — Paper summary
- [[timedit-masking]] — Unified masking mechanism
- [[timedit-physics-informed]] — Physics-informed sampling
- [[dit]] — DiT architecture TimeDiT is built upon
- [[diffusion-models]] — Diffusion model fundamentals
- [[chronos]] — Chronos (competing time series FM)
- [[timesfm]] — TimesFM (competing time series FM)
- [[csdi]] — CSDI, prior diffusion work for time series imputation
- [[energy-based-model]] — EBM, physics-informed sampling foundation

[^src-timedit]: [[source-timedit]]
