---
title: "AllSpark"
type: entity
tags:
  - multimodal
  - spatio-temporal
  - geospatial
  - foundation-model
  - remote-sensing
  - llm
created: 2026-06-04
last_updated: 2026-06-04
source_count: 1
confidence: high
status: active
---

# AllSpark

**AllSpark** is a multimodal spatio-temporal general intelligence model proposed by Shao et al. (Central South University) that integrates **ten heterogeneous modalities** into a unified framework, using language as the universal alignment reference[^src-allspark]. It covers 1D (language, code, table), 2D (RGB, SAR, multispectral, hyperspectral, graph, trajectory), and 3D (point cloud) modalities[^src-allspark].

## Key Innovation: LaRF

AllSpark is built on the **[[language-as-reference-framework|Language as Reference Framework (LaRF)]]** principle: diverse modality features are mapped into the language feature space via a **modal bridge** (based on Perceiver cross-attention), enabling a unified multimodal LLM to jointly interpret all modalities while preserving modality-specific autonomy through dedicated encoders[^src-allspark].

## Architecture Overview

```
Modality Data → Modal Encoder (per-modality) → Modal Bridge (Perceiver) → Multimodal LLM (Lynx) → Task Head
                                                                 ↑
                                                          Text Prompt (tokenizer)
```

- **Modal encoders**: EVA (RGB), ViT-PatchEmbed (MSI), 12-layer Transformer (HSI), PointBERT encoder (point cloud), TUTR-based (trajectory), CNN (SAR), TabFormer (table), Lynx tokenizer (text/code), STAEformer-based (graph)[^src-allspark]
- **Modal bridge**: Learnable query vectors Q ∈ ℝ^(N×4096) attend to modality tokens via cross-attention, projecting all modalities to the language dimension[^src-allspark]
- **Multimodal LLM**: Lynx with trainable adapter layers (not frozen)[^src-allspark]
- **Task heads**: Lightweight single-layer linear layers for most tasks[^src-allspark]

## Few-Shot Capability

Without any meta-learning or additional training steps, AllSpark achieves training-free few-shot classification[^src-allspark]:

| Modality | Dataset | 5-way 1-shot | 5-way 5-shot |
|----------|---------|-------------|-------------|
| RGB | UC-Merced | 95.58% | 97.64% |
| RGB | WHU-RS19 | 97.16% | 98.94% |
| Point Cloud | ShapeNet | 67.20% | 82.12% |

This represents up to 41.82% improvement over traditional few-shot methods that require explicit training[^src-allspark].

## Full-Training Performance

AllSpark, despite no expert knowledge in most modalities and a unified architecture, achieves competitive results[^src-allspark]:

| Modality | Dataset | Task | AllSpark | SOTA | Gap |
|----------|---------|------|----------|------|-----|
| RGB | NWPU-RESISC45 | Classification | 94.85% | 95.69% | 0.84 |
| Trajectory | ETH | Prediction | ADE 0.43 | 0.36 | 0.07 |
| SAR | MSTAR | Classification | 97.24% | 99.13% | 1.89 |
| Point Cloud | ModelNet40 | Classification | 91.2% | 94.9% | 3.7 |
| Language | IMDB | Sentiment | 96.78% | 97.1% | 0.32 |

## Comparison with Other Multimodal Models

| Model | Modalities | Domain | Approach |
|-------|-----------|--------|----------|
| **AllSpark** | 10 | Geospatial/General | LaRF + modal bridge |
| [[most|MoST]] | 4 | Traffic | SNR-based modality selection |
| [[aurora|Aurora]] | 3+ | General TS | Modality-Guided Self-Attention |
| Meta-Transformer | 12 | General Vision | Frozen ViT + shared encoder |
| OneLLM | 8 | General | Multimodal LLM backbone |

AllSpark is uniquely focused on **geospatial spatio-temporal modalities** (SAR, MSI, HSI, trajectory, point cloud) that are absent from most other multimodal models[^src-allspark].

## Limitations

- No cross-modal interaction between non-language modalities[^src-allspark]
- Poor dense prediction performance (video, oblique photography)[^src-allspark]
- High training cost: 30-87 hours per modality on 2× A6000[^src-allspark]
- Spatial information degradation in deep large model layers[^src-allspark]

## Related Pages

- [[source-allspark]] — source summary
- [[language-as-reference-framework]] — LaRF concept
- [[spatio-temporal-foundation-model]] — ST foundation model concept
- [[multimodal-time-series-forecasting]] — multimodal TS forecasting
- [[most]] — MoST multimodal ST model
- [[aurora]] — Aurora multimodal foundation model

[^src-allspark]: [[source-allspark]]
