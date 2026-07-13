---
title: "CrossLinear: Plug-and-Play Cross-Correlation Embedding for Time Series Forecasting with Exogenous Variables"
type: source-summary
tags:
  - time-series-forecasting
  - exogenous
  - linear-models
  - plug-and-play
  - kdd-2025
created: 2026-07-13
last_updated: 2026-07-13
source_count: 1
confidence: high
status: active
---

# CrossLinear: Plug-and-Play Cross-Correlation Embedding for Time Series Forecasting with Exogenous Variables

**Authors**: Pengfei Zhou, Yunlong Liu, Junli Liang, Qi Song, Xiangyang Li (University of Science and Technology of China; Deqing Alpha Innovation Institute).[^src-crosslinear]

**Venue**: KDD 2025 (Toronto) | **arXiv**: 2505.23116v1 (29 May 2025) | **Code**: https://github.com/mumiao2000/CrossLinear | DOI: 10.1145/3711896.3736899

## Core Arguments

**1. Many-to-one exogenous forecasting is a distinct paradigm.** Univariate (one-to-one) and multivariate (many-to-many) forecasting do not match real systems where only endogenous targets matter (e.g., traffic volume driven by holidays/weather). Pure univariate models ignore exogenous drivers; many-to-many models waste capacity forecasting exogenous series that are not targets.[^src-crosslinear]

**2. Explicit CD dependency modeling often overfits.** Channel-dependent (CD) Transformers/GNNs (cross-attention, graph convolution) theoretically capture inter-variable relations, but time-varying and indirect correlations plus sparse data make full CD prone to spurious features. Empirically, channel-independent (CI) models often beat CD—yet CI cannot *use* exogenous series for the target.[^src-crosslinear]

**3. Lightweight time-invariant cross-correlation is enough.** Inspired by positional embeddings, CrossLinear injects **only time-invariant, direct** endogenous–exogenous dependencies via a single-layer 1D convolution over the stacked normalized series, then residual-mixes with the endogenous series through a learnable \(\alpha\). This is a nearly free plug-in that upgrades CI backbones without full CD complexity.[^src-crosslinear]

**4. Architecture of [[crosslinear|CrossLinear]].** (i) RevIN instance norm/de-norm; (ii) [[cross-correlation-embedding|cross-correlation embedding]] \(X^{\mathrm{cross}}=\mathrm{Conv1D}(\mathrm{Stack}(X^{\mathrm{exo}*},X^{\mathrm{endo}*}))\), \(X^{\mathrm{emb}}=\alpha X^{\mathrm{endo}*}+(1-\alpha)X^{\mathrm{cross}}\); (iii) patchify + linear projection + learnable \(\beta\)-weighted positional embedding; (iv) global linear forecasting head. Complexity remains \(O(T)\). Weight-sharing extends the design to multivariate many-to-many forecasting.[^src-crosslinear]

**5. Ablation insight.** Weighted sum of endo + cross embedding beats endo-only, cross-only, and concatenation; though sum and “cross-only” can be shown mathematically equivalent under a reparameterized kernel, treating endo/exo equally fails in practice under limited data—distinguishing endogenous focus matters.[^src-crosslinear]

## Experiments

**12 datasets** aligned with [[source-timexer|TimeXer]]: long-term ECL / Weather / ETTh1–2 / ETTm1–2 / Traffic (lookback 96, horizons 96–720); short-term EPF markets NP/PJM/BE/FR/DE (lookback 168, horizon 24). Exogenous counts range from 2 (EPF) to 861 (Traffic).[^src-crosslinear]

**Baselines (10):** TimeXer, iTransformer, MSGNet, SparseTSF, RLinear, PatchTST, TiDE, TimesNet, DLinear, Autoformer. Metrics: MSE / MAE.[^src-crosslinear]

**Results.** CrossLinear ranks top on most long/short many-to-one settings (paper: 30 first-place MSE and 29 MAE among reported cells), outperforming TimeXer on the majority of datasets while training faster (O(T) vs \(O((T/p)^2)\)). Plug-in generality: adding the embedding improves SparseTSF / RLinear / PatchTST / DLinear / Autoformer (e.g., RLinear Traffic MSE −27.8%). Missing-value masks show exogenous information is critical; even with full endo mask the model can still exploit exo via the embedding.[^src-crosslinear]

## Limitations / Scope

Numerical exogenous series only (no image/text multimodal ST). Models time-invariant direct correlations, deliberately ignoring time-varying/indirect graphs. Focus is many-to-one (with weight-sharing many-to-many extension), not future-exogenous Class-1d settings emphasized by [[source-exotst|ExoTST]]. Future work mentioned: anomaly detection/classification and diffusion integration.[^src-crosslinear]

## Related Pages

- Entity: [[crosslinear]]
- Technique: [[cross-correlation-embedding]]
- Related exogenous work: [[source-timexer]], [[source-exost]], [[source-exollm]], [[source-exotst]], [[source-gcgnet]]
- Related CI/CD context: [[channel-independence]], [[patch-based-tokenization]], [[source-sparsetsf]]

---

[^src-crosslinear]: [[source-crosslinear]]
