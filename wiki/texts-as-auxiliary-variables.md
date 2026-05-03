---
title: "Texts as Auxiliary Variables"
type: concept
tags:
  - multimodal-time-series
  - text-alignment
  - variable-augmentation
  - plug-and-play
  - iclr-2026
created: 2026-05-03
last_updated: 2026-05-03
source_count: 1
confidence: medium
status: active
---

# Texts as Auxiliary Variables

**Texts as Auxiliary Variables** is the core design concept behind the [[tats|TaTS]] framework (ICLR 2026), which treats time-series-paired texts as additional variables that augment the original multivariate time series[^src-language-in-the-flow-of-time].

## Core Idea

The key insight is that time-series-paired texts behave similarly to variables in a multivariate time series: they are influenced by shared external drivers, interact dynamically with the time series, and — as demonstrated by [[chronological-textual-resonance|CTR]] — exhibit aligned periodic properties[^src-language-in-the-flow-of-time].

By encoding texts into numerical vectors and concatenating them as new "channels" alongside the original time series variables, any existing time series model can process both modalities through its standard architecture without modification.

## Mechanism

1. **Text Encoding**: Each timestamp's paired text $s_t$ is encoded via a pre-trained LLM to produce $\mathbf{e}_t \in \mathbb{R}^{d_{\text{text}}}$

2. **Dimensionality Reduction**: An MLP maps the high-dimensional text embedding to a lower-dimensional space: $\mathbf{z}_t = \text{MLP}(\mathbf{e}_t) \in \mathbb{R}^{d_{\text{mapped}}}$

3. **Variable Augmentation**: The mapped embeddings are concatenated as new variables:
   $$\mathbf{U} = [\mathbf{X}; \mathbf{Z}^{\intercal}] \in \mathbb{R}^{T \times (N + d_{\text{mapped}})}$$

4. **Standard Processing**: The augmented sequence is fed into any existing time series model, which treats the text-derived variables identically to numerical variables[^src-language-in-the-flow-of-time]

## Design Rationale

| Principle | Justification |
|-----------|---------------|
| **No architecture change** | Maximizes compatibility with existing models |
| **Concatenation over attention fusion** | Simpler, avoids modality-specific fusion modules |
| **Joint training** | MLP and TS model co-adapt for optimal text utilization |
| **Dimensionality reduction** | Prevents text from dominating due to high embedding dimension |
| **Channel-independence compatible** | Text variables can be processed independently like numerical channels |

## Comparison with Other Multimodal Fusion Strategies

| Strategy | Method | Complexity | Architecture Change |
|----------|--------|------------|---------------------|
| **Auxiliary Variables** | [[tats|TaTS]] | Low (MLP + concat) | None |
| Covariate Homogenization | [[unica|UniCA]] | Medium (encoder + projection) | Pre/Post-Fusion modules |
| Cross-modal Attention | [[mindts|MindTS]] | Medium (cross-attention) | Fusion module |
| LLM Reasoning | [[vot|VoT]] | High (LLM generation) | Dual-branch architecture |
| Modality-Guided Attention | [[aurora|Aurora]] | High (tokenization + distillation) | Custom attention mechanism |

The auxiliary variables approach is the **simplest** and most **broadly compatible**, trading off deeper semantic integration for universal applicability[^src-language-in-the-flow-of-time].

## Advantages

- **Universal compatibility**: Works with Transformer-based, linear, frequency-based, and convolution-based models
- **Minimal overhead**: Only adds an MLP and concatenation operation
- **Task-agnostic**: Applicable to forecasting, imputation, and potentially other TS tasks
- **Encoder-agnostic**: Compatible with various text encoders (GPT-2, BERT, etc.)[^src-language-in-the-flow-of-time]

## Limitations

- **Shallow integration**: Concatenation provides less semantic interaction than attention-based or reasoning-based fusion
- **CTR-dependent**: Effectiveness relies on the paired texts exhibiting meaningful periodic alignment
- **Dimensionality sensitivity**: The mapped dimension $d_{\text{mapped}}$ is a hyperparameter requiring tuning[^src-language-in-the-flow-of-time]

## Related Pages

- [[tats]] — TaTS framework
- [[chronological-textual-resonance]] — CTR phenomenon
- [[tt-wasserstein]] — TT-Wasserstein metric
- [[source-language-in-the-flow-of-time]] — source summary
- [[multimodal-time-series-forecasting]] — task concept
- [[channel-independence]] — related TS technique
- [[covariate-homogenization]] — UniCA's alternative approach

[^src-language-in-the-flow-of-time]: [[source-language-in-the-flow-of-time]]