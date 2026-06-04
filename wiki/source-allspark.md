---
title: "source-allspark"
type: source-summary
tags:
  - multimodal
  - spatio-temporal
  - geospatial
  - foundation-model
  - llm
  - remote-sensing
created: 2026-06-04
last_updated: 2026-06-04
source_count: 0
confidence: high
status: active
---

# AllSpark: A Multimodal Spatio-Temporal General Intelligence Model

Shao et al. (Central South University, arXiv 2024, revised Jan 2025) propose **AllSpark**, a unified multimodal spatio-temporal general intelligence model that integrates **ten heterogeneous modalities** into a single framework[^src-allspark]. Source: `raw/allspark-shao-2024.pdf`.

## Core Principle: Language as Reference Framework (LaRF)

The key insight is inspired by human cognition: perceptual signals from multiple senses converge into language for reasoning. LaRF defines language as the alignment anchor for all modalities, achieving both **cohesion** (shared information across modalities mapped to language space) and **autonomy** (modality-specific encoders preserving unique information)[^src-allspark].

## Architecture

AllSpark consists of five modules[^src-allspark]:

1. **Modality-specific encoders**: Independent encoders for each modality — ResNet/EVA for RGB, ViT-adapted PatchEmbed for MSI, 12-layer Transformer for HSI, TabFormer for tables, Lynx tokenizer for text/code, PointBERT-based encoder for point clouds, TUTR-based for trajectories, CNN for SAR, STAEformer-based for graphs.
2. **Modal bridge**: Based on Perceiver, uses learnable query vectors and cross-attention to project modality tokens into the unified 4096-dim language feature space.
3. **Text tokenizer**: From Lynx, converts text prompts into token sequences.
4. **Multimodal LLM**: Lynx-based, with lightweight multimodal adapter layers trained (not frozen) for cross-modal adaptability.
5. **Task heads**: Lightweight single-layer linear layers for classification/regression, native decoder for text/code.

## Ten Modalities

| Dimension | Modalities |
|-----------|-----------|
| 1D | Language, Code, Table |
| 2D | RGB, SAR, Multispectral (MSI), Hyperspectral (HSI), Graph, Trajectory |
| 3D | Point Cloud |

## Key Results

- **Few-shot learning**: Without any additional training (training-free), AllSpark achieves 95.58% 5-way 1-shot on UC-Merced (RGB) and 97.64% 5-way 5-shot — up to 41.82% improvement over baseline models[^src-allspark].
- **Cross-modality adaptability**: Competes with or approaches SOTA models across all 10 modalities despite no modality-specific expert knowledge in most cases. RGB: 94.85% (0.84 from SOTA); Trajectory: ADE 0.43 (0.07 from SOTA); SAR: 97.24% (1.89 from SOTA)[^src-allspark].
- **Training cost**: ~8B total parameters (270M-1.1B trainable depending on modality), trained on 2× NVIDIA A6000 GPUs[^src-allspark].

## Limitations

- No direct interaction between non-language modalities (e.g., RGB ↔ point cloud)
- Poor performance on dense prediction tasks (video: 27.5%, oblique photography: 6.4% PAG₆)
- High training cost (~87 hours for SAR modality)
- Spatial information degradation in deep layers of large models[^src-allspark]

## Significance

AllSpark represents the first model to unify 10 spatio-temporal modalities under the LaRF principle. It demonstrates that language can serve as a universal alignment anchor for highly heterogeneous modalities, enabling training-free few-shot learning and cross-modal generalization. Theoretically extensible to arbitrary modalities. Source code: https://github.com/GeoX-Lab/AllSpark[^src-allspark].

[^src-allspark]: [[source-allspark]]
