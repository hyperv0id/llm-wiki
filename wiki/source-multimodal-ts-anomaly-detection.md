---
title: "Towards Multimodal Time Series Anomaly Detection with Semantic Alignment and Condensed Interaction"
type: source-summary
tags:
  - multimodal-time-series
  - anomaly-detection
  - text-encoding
  - mutual-information
  - iclr-2026
created: 2026-05-03
last_updated: 2026-05-03
source_count: 0
confidence: medium
status: active
---

# Source Summary: MindTS (ICLR 2026)

**Title**: Towards Multimodal Time Series Anomaly Detection with Semantic Alignment and Condensed Interaction  
**Authors**: Shiyan Hu, Jianxin Jin, Yang Shu, Peng Chen, Bin Yang, Chenjuan Guo (East China Normal University)  
**Conference**: ICLR 2026  
**arXiv**: 2603.21612  

## Core Contribution

This paper proposes **MindTS** (Multimodal Time Series Anomaly Detection with Semantic Alignment and Condensed Interaction), the first dedicated multimodal anomaly detection model that jointly leverages time series and text modalities.

Unlike previous unimodal anomaly detection methods (DADA, GDN, DCdetector) which only use numerical data, MindTS integrates exogenous text (background knowledge) and endogenous text (statistical summaries of time patches) to improve anomaly detection.

## Key Innovations

### 1. Fine-grained Time-text Semantic Alignment

The model decomposes text into two complementary views:
- **Exogenous text**: External background information (e.g., news, weather reports, policies) that is shared across time steps
- **Endogenous text**: Statistical descriptions generated per patch (mean, extrema, trend) that capture time-specific characteristics

Cross-view fusion uses endogenous text as query and exogenous text as key/value, producing semantically aligned text representations. Contrastive learning explicitly aligns time and text representations in a shared embedding space.

### 2. Content Condenser Reconstruction

Inspired by the Information Bottleneck principle, a content condenser filters redundant textual information by minimizing mutual information between aligned text and condensed text. A binary mask (sampled via Bernoulli with straight-through estimator) removes irrelevant tokens. A smoothness loss ensures stable compression across adjacent patches.

Condensed text then reconstructs **masked** time series, forcing cross-modal interaction. Random masking of time series (≈50% mask ratio) makes reconstruction challenging, encouraging the condensed representation to preserve time-relevant information.

## Technical Details

- **Time encoder**: Patch-based Transformer (patch size p=6 typical)
- **Text encoder**: Open-source LLM (DeepSeek default, GPT2/BERT/LLAMA as ablations)
- **Loss**: L_total = L_MA (multimodal alignment) + L_CL (condenser KL + smoothness) + L_Rec (reconstruction MSE)
- **Hyperparameters**: μ (compression strength, robust 0.1–0.9), mask ratio ≈0.5

## Results

Evaluated on 6 real-world multimodal datasets (Weather, Energy, Environment, KR, EWJ, MDT):

| Dataset | MindTS Aff-F | Best baseline | Improvement |
|---------|--------------|---------------|-------------|
| Weather | 82.66 | 81.06 | +1.6 |
| Energy | 74.37 | 70.81 | +3.6 |
| Environment | 85.29 | 84.36 | +0.9 |
| KR | 90.28 | 88.58 | +1.7 |
| EWJ | 83.89 | 81.82 | +2.1 |
| MDT | 89.19 | 80.81 | +8.4 |

MindTS outperforms both unimodal baselines (DCdetector, Anomaly Transformer, PatchTST) and LLM-based methods (GPT4TS, LLMMixer) across Aff-F, VUS-PR, and VUS-ROC metrics. When compared to multimodal extensions via MM-TSFLib, MindTS still achieves best or competitive results.

## Limitations

1. Only handles text modality (not images/video)
2. Requires both endogenous and exogenous text to be available
3. Patch-based endogenous text generation depends on LLM quality and prompt design
4. Experiments focus on reconstruction-based anomaly detection; applicability to other TS tasks not validated

## Relation to Existing Wiki Content

- Extends [[multimodal-time-series-forecasting]] to anomaly detection task
- Complements [[most]] (multi-modality ST prediction) with different task focus
- Similar cross-modal alignment concerns as [[channelmts]] but with text instead of environmental features
- Content condenser shares Information Bottleneck philosophy with [[conditional-attention-pooling]] (UniCA) but applies to redundancy filtering
