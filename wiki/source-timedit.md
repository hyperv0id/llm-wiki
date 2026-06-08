---
title: "TimeDiT — General-purpose Diffusion Transformers for Time Series Foundation Model"
type: source-summary
tags:
  - diffusion
  - transformer
  - time-series
  - foundation-model
  - kdd-2025
  - physics-informed
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: high
status: active
---

**TimeDiT** is a proto-foundation model for multivariate time series proposed by Defu Cao, Wen Ye, Yizhou Zhang, and Yan Liu (University of Southern California), accepted at **KDD 2025** (arXiv:2409.02322)[^src-timedit]. TimeDiT integrates DiT-style transformer backbone with diffusion-based probabilistic sampling, establishing the first diffusion transformer framework for comprehensive time series tasks beyond forecasting alone.

## Core Contributions

1. **DiT + Diffusion for Time Series**: Replaces autoregressive transformer forecasting with diffusion transformer sampling, capturing inherent uncertainty and stochasticity in time series rather than learning deterministic mappings[^src-timedit].

2. **Unified masking mechanism**: Four mask types (random, block, stride, reconstruction) harmonize pre-training and inference across diverse tasks — forecasting, imputation, anomaly detection, and synthetic data generation — in a single model without task-specific architecture changes[^src-timedit].

3. **Physics-informed sampling (finetuning-free model editing)**: Incorporates PDE-based physics knowledge as an energy-based prior during inference via Langevin dynamics, with a closed-form Boltzmann distribution solution (Theorem 3.1). No model retraining or parameter updates required[^src-timedit].

4. **Comprehensive evaluation**: Zero-shot and fine-tuned results on 8 forecasting benchmarks (ETTh, Solar, Electricity, Traffic, Taxi, Exchange, ETTm, Weather), multi-resolution forecasting (MIMIC-III, PhysioNet, NASDAQ), anomaly detection (MSL, SMAP, SWaT, SMD, PSM), and data generation (Stock, Air Quality, Energy). SOTA uncertainty quantification with missing values or multi-resolution[^src-timedit].

## Architecture

TimeDiT uses a DiT-style transformer backbone operating on padded multivariate time series X ∈ R^(K×L). Key design choices[^src-timedit]:

- **WYSIWYG tokenization**: Direct linear projection of input arrays into continuous token space, no vector quantization.
- **AdaLN condition injection**: Conditional observations x_con control scale/shift of target noise x_tar through adaptive layer normalization — empirically outperforms concatenation-based conditioning.
- **Diffusion time embedding**: Injected directly into target noise embeddings rather than through separate conditioning pathways, unlike original DiT.
- **Four model sizes**: S (33M), B (130M), L (460M), XL (680M), matching DiT configurations.

## Masking Strategy (Key Innovation)

The Time Series Mask Unit generates four mask types[^src-timedit]:

| Mask | Formula | Pre-training Use | Inference Use |
|------|---------|-----------------|---------------|
| Random M^R | Uniform random with ratio r | General SSL | Missing value handling |
| Block M^B | Mask last l time steps | Variable forecast horizon training | Forecasting |
| Stride M^S | Alternating masked/unmasked blocks | Temporal correlation modeling | — |
| Reconstruction M^Rec | M^Rec = 0 (whole sequence) | — | Generation, anomaly detection |

Stride masking is the most critical component — its removal causes MSE to surge from 0.424→0.862 on Solar[^src-timedit].

## Training

- Pre-trained on Chronos dataset (~5B time points) with zero exposure to downstream evaluation data.
- Adam optimizer, lr=0.0001, batch size 256-512, max channel=20-40, max sequence length=198.
- NVIDIA A100 40G GPUs. Over 100 epochs to convergence.
- Zero-shot results use a single pre-trained checkpoint evaluated with or without fine-tuning[^src-timedit].

## Key Results

- **Forecasting**: TimeDiT (zero-shot) achieves avg MSE 0.356 on ETTh1, outperforming all Moirai variants (0.400-0.510) and competitive with or surpassing full-shot supervised models (iTransformer 0.454, PatchTST 0.469)[^src-timedit].
- **Physics-informed zero-shot**: On PDE benchmarks (Burgers, Advection, Diffusion-Reaction, Vorticity), zero-shot TimeDiT with physics-informed sampling outperforms fully trained baselines (DLinear, PatchTST, NeuralCDE) that were trained on 5,000 PDE-generated samples — e.g., Burgers MSE 0.011 vs DLinear 0.031[^src-timedit].
- **Inference speed**: Only 1 second for single-sample generation — superior to Diffusion-TS (6s) and CSDI (2s)[^src-timedit].
- **Uncertainty quantification**: SOTA CRPS across real-world datasets with missing values or multi-resolution[^src-timedit].

## Limitations

- Primarily explored common sequence lengths (max 198), not tested on very long sequences.
- Maximum channel handling (K_max=20-40) limits ultra-high-dimensional MTS.
- Understanding of how different domain information types contribute to performance is still under investigation.
- Code not yet public; pre-trained checkpoint promised for release[^src-timedit].

[^src-timedit]: [[source-timedit]]
