---
title: "ExoTST: Exogenous-Aware Temporal Sequence Transformer for Time Series Prediction"
type: source-summary
tags:
  - exogenous
  - transformer
  - time-series
  - 2024
created: 2026-07-07
last_updated: 2026-07-07
source_count: 1
confidence: medium
status: active
---

# ExoTST: Exogenous-Aware Temporal Sequence Transformer for Time Series Prediction

**Authors:** Kshitij Tayal, Arvind Renganathan, Xiaowei Jia, Vipin Kumar, Dan Lu  
**Year:** 2024  
**arXiv:** 2410.12184  
**Affiliations:** Oak Ridge National Laboratory, University of Minnesota, University of Pittsburgh

## Summary

ExoTST proposes a novel transformer-based framework that effectively incorporates both past context and current/projected exogenous variables for improved time series prediction. It addresses a critical gap in existing methods: autoregressive models (e.g., PatchTST, iTransformer) cannot leverage future exogenous drivers, while forward models (e.g., LSTM) cannot leverage past context.[^src-exotst]

### Problem Setting (Class 1d)

ExoTST targets the problem class where both past endogenous values $y_{1:L}$, past exogenous $X_{1:L}$, and future/projected exogenous $X_{L+1:L+f}$ are available for predicting future endogenous $\hat{y}_{L+1:L+f}$. This is common in scientific applications like carbon flux (GPP) prediction where climate projections provide future weather drivers.[^src-exotst]

### Key Innovations

1. **Separate exogenous encoders** — Past and current/projected exogenous series are processed by two distinct attention-based encoders, treating them as different modalities to handle distribution shifts between historical and future exogenous data.[^src-exotst]

2. **Cross-temporal modality fusion module** — Uses an aggregation token ($e_{agg}$, similar to BERT/ViT's [CLS] token) as queries in cross-attention between the two exogenous encoders, enabling efficient information exchange with linear (rather than quadratic) attention complexity. The fusion repeats over multiple layers at different abstraction levels.[^src-exotst]

3. **Patch-wise encoding** — All input series (endogenous and exogenous) are divided into patches (overlapping segments), reducing token count from $L$ to $L/S$ and enabling effective local semantic extraction.[^src-exotst]

4. **Endogenous decoder with cross-attention** — The fused exogenous embedding serves as keys/values in cross-attention with the endogenous decoder's self-attention output, enabling information transfer from exogenous to endogenous modalities.[^src-exotst]

### Experimental Results

- Evaluated on real-world carbon flux (GPP) datasets and time series benchmarks.
- Outperforms SOTA baselines (TiDE, PatchTST, iTransformer, Crossformer, LSTM-based forward models) by 8–12% in prediction accuracy.[^src-exotst]
- Demonstrates strong robustness against missing values and noise in exogenous drivers — maintains consistent performance under realistic imperfections.
- Particularly effective in scenarios where future exogenous projections are available (e.g., climate scenarios from Earth system models).[^src-exotst]

### Significance

ExoTST systematically bridges the gap between autoregressive and forward modeling by treating past and future exogenous series as distinct modalities with explicit cross-temporal fusion. This provides a principled framework for incorporating exogenous information into time series prediction, with direct applicability to climate science and other domains where future driver projections are available.[^src-exotst]

### Related exogenous pages

- [[source-timexer|TimeXer]] — historical endo patch + exo variate cross-attention (no future-exo modality split)
- [[source-crosslinear|CrossLinear]] — plug-and-play 1D-conv cross-correlation embedding for historical exo (KDD 2025)
- [[source-exost|ExoST]] — select-then-balance past/future exo for ST backbones
- [[source-exollm|ExoLLM]] — LLM multi-grained prompts for exogenous forecasting

[^src-exotst]: [[source-exotst]]