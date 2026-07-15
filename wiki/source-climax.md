---
title: "ClimaX: A Foundation Model for Weather and Climate"
type: source-summary
tags:
  - foundation-model
  - weather
  - climate
  - pretraining
  - 2023
  - icml
created: 2026-07-07
last_updated: 2026-07-21
source_count: 2
confidence: medium
status: active
---

# ClimaX: A Foundation Model for Weather and Climate

Nguyen, Brandstetter, Kapoor, Gupta & Grover (UCLA, Microsoft Research, ICML 2023) propose **ClimaX**, the first foundation model designed for weather and climate science. ClimaX addresses the challenge that existing data-driven models are trained for specific tasks and cannot generalize across different variables, spatio-temporal resolutions, and geographical regions[^src-climax].

## Key Challenges and Solutions

ClimaX tackles three core challenges:

**1. Heterogeneous pretraining data.** Unlike vision or language, weather/climate data grows at a fixed rate from sensors. ClimaX instead pretrains on the **CMIP6** collection of physics-informed climate simulations from ~100 distinct models across 49 modeling groups. This provides rich, diverse, large-scale data[^src-climax].

**2. Variable heterogeneity.** Climate variables (pressure, temperature, humidity, etc.) have different datatypes and spatio-temporal coverage. ClimaX repurposes the Vision Transformer (ViT) with **variable tokenization**: each variable is embedded separately with a learned variable-specific token and lead-time embedding, rather than treating all variables as image channels. A **cross-attention variable aggregation** block compresses the expanded token sequence before self-attention, enabling flexible handling of irregular variable subsets[^src-climax].

**3. General pretraining objective.** A **randomized forecasting objective** trains the model to predict arbitrary subsets of variables at arbitrary future lead times. This enables fine-tuning to tasks beyond the pretraining window, including sub-seasonal to seasonal (S2S) prediction, climate projections, and downscaling[^src-climax].

## Architecture

ClimaX extends the standard ViT with:
- **Variable tokenization**: Each (variable, patch) pair is encoded with its variable-specific embedding and lead-time embedding, preserving modality identity
- **Variable aggregation**: Cross-attention over variable tokens produces a fixed-length representation per patch, mitigating sequence length explosion
- **Standard Transformer encoder** with self-attention and feed-forward layers

## Empirical Results

- **WeatherBench**: Competitive with operational IFS (ECMWF) despite training at moderate resolution (5.625° → 1.40625°) with only 80 V100 GPUs
- **ClimateBench**: SOTA on climate projection tasks for temperature, precipitation, and extreme indices
- **Downscaling**: Effective super-resolution from low-res to high-res spatial fields
- **Scaling laws**: Performance improves with larger pretraining datasets, larger models, and higher resolution
- Pretraining at 5.625° transfers effectively to 1.40625° fine-tuning (unseen resolution)[^src-climax]

## Significance

ClimaX pioneered the foundation model paradigm for Earth systems science, demonstrating that pretraining on heterogeneous physics-based simulation data (CMIP6) enables generalization across tasks, variables, and resolutions unseen during training. It established the variable tokenization + aggregation pattern adopted by later weather foundation models, and provided empirical evidence for scaling laws in climate ML[^src-climax].

## 后续工作

同一 UCLA 组（Nguyen, Grover）后续提出 [[omnicast|OmniCast]]（NeurIPS 2025），将掩码潜扩散模型应用于 S2S 预测，在 ChaosBench 上取得 SOTA。OmniCast 与 ClimaX 的设计哲学形成对比：ClimaX 为每个 lead time 单独微调模型，OmniCast 则用一个模型同时生成完整 44 天序列，且避免了自回归累积误差[^src-omnicast]。

## 相关页面

- [[omnicast]] — 同组后继 S2S 预测模型（NeurIPS 2025）
- [[source-aurora]] — 多模态时序基础模型
- [[source-cirt]] — CirT S2S 预测模型（ClimaX 为其基线之一）
- [[source-climatear]] — ClimateAR，VAR 生成式概率气候预测（ClimaX 为其确定性 baseline 之一）

[^src-climax]: [[source-climax]] — ClimaX: A Foundation Model for Weather and Climate (Nguyen et al., ICML 2023)

[^src-omnicast]: [[source-omnicast]]
