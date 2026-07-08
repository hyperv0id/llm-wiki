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
last_updated: 2026-07-07
source_count: 1
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

## 相关页面

- [[source-aurora]] — 多模态时序基础模型

[^src-climax]: [[source-climax]] — ClimaX: A Foundation Model for Weather and Climate (Nguyen et al., ICML 2023)
