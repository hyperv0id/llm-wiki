---
title: "Cross-Correlation Embedding"
type: technique
tags:
  - time-series-forecasting
  - exogenous
  - convolution
  - plug-and-play
  - channel-dependency
created: 2026-07-13
last_updated: 2026-07-16
source_count: 1
confidence: high
status: active
---

# Cross-Correlation Embedding

**Cross-correlation embedding** is the plug-and-play module at the core of [[crosslinear|CrossLinear]] (KDD 2025) for injecting exogenous information into an endogenous target series without full channel-dependent attention.[^src-crosslinear]

## Motivation

Explicit CD models (cross-attention, GNNs) try to capture all inter-variable dependencies, including time-varying and indirect ones, and often overfit under limited data. CrossLinear instead models only **time-invariant, direct** endogenous–exogenous dependencies, analogous to how positional embeddings inject fixed structure into Transformers.[^src-crosslinear]

## Formulation

After RevIN instance normalization of endogenous and exogenous series:[^src-crosslinear]

\[
X^{\mathrm{cross}}_{1:T,1} = \mathrm{Conv1D}\big(\mathrm{Stack}(X^{\mathrm{exo}*}_{1:T,N-1},\, X^{\mathrm{endo}*}_{1:T,1})\big)
\]

\[
X^{\mathrm{emb}}_{1:T,1} = \alpha \cdot X^{\mathrm{endo}*}_{1:T,1} + (1-\alpha)\cdot X^{\mathrm{cross}}_{1:T,1}
\]

- `Stack` concatenates variables into an \(N\times T\) matrix.
- `Conv1D` uses fixed kernel size 3 and stride 1 in the paper’s default setting, which can absorb short lead–lag effects.
- \(\alpha\in[0,1]\) is learnable; initialization near 1 stabilizes early training by prioritizing the endogenous series.

The embedded series keeps the **same length as the endogenous input**, so it drops into CI patch/linear pipelines without reshaping and remains usable under missing exogenous values (mask experiments).[^src-crosslinear]

## Why Not Cross-Only / Concat?

Ablations compare: (i) endo-only, (ii) cross-only (\(\alpha=0\)), (iii) concat, (iv) weighted sum (default). Sum wins on long- and short-term many-to-one averages. Appendix proves sum and cross-only can be **mathematically equivalent** under a reparameterized kernel \(K'=(1-\alpha)K+\alpha S\), yet equal treatment of endo/exo fails empirically—separating endogenous focus improves learning under limited data.[^src-crosslinear]

## Generality and Cost

Complexity \(O(T)\). As a drop-in front-end it improves SparseTSF, RLinear, PatchTST, DLinear, and Autoformer, with larger gains when exogenous count is high (ECL 320, Traffic 861; RLinear Traffic MSE −27.8%).[^src-crosslinear]

## Relation to Other Exogenous Techniques

| Method | Exogenous fusion style |
|--------|------------------------|
| Cross-correlation embedding | 1D conv residual mix into endo series (CI-friendly) |
| [[source-timexer\|TimeXer]] | Patch endo + variate exo tokens; cross-attention via global endo token |
| [[source-exotst\|ExoTST]] | Separate past/future exo encoders + aggregation-token fusion |
| [[source-exost\|ExoST]] | Select (gated experts) then balance past/future exo for ST backbones |
| [[source-exollm\|ExoLLM]] | Multi-grained text prompts + dual TS–text attention |
| [[source-gcgnet\|GCGNet]] | Joint temporal–channel graphs under generative alignment |

## Links

- Model: [[crosslinear]]
- Source: [[source-crosslinear]]
- Related concepts: [[channel-independence]], [[cross-dimension-dependency]], [[patch-based-tokenization]]

---

[^src-crosslinear]: [[source-crosslinear]]
