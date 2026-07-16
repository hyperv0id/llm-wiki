---
title: "Interpretable Multi-Head Attention"
type: technique
tags:
  - attention
  - transformer
  - interpretability
  - tft
  - time-series
created: 2026-07-13
last_updated: 2026-07-24
source_count: 1
confidence: medium
status: active
---

# Interpretable Multi-Head Attention

**Interpretable multi-head attention** is TFT’s modification of standard Transformer multi-head attention so that attention weights remain meaningful for temporal feature importance. Heads keep separate query/key projections but **share a single value projection**, then **average** head attention maps before applying values.[^src-tft]

## Standard vs Interpretable Form

Standard multi-head attention concatenates head-specific value projections, so a large attention weight in one head does not correspond to a unique importance of an input feature. TFT instead uses:

\[
\begin{aligned}
\tilde H &= \tilde A(Q,K)\,V W_V, \\
\tilde A(Q,K) &= \frac{1}{m_H}\sum_{h=1}^{m_H} A\bigl(Q W_Q^{(h)}, K W_K^{(h)}\bigr), \\
\mathrm{InterpretableMultiHead}(Q,K,V) &= \tilde H\, W_H,
\end{aligned}
\]

with \(A\) scaled dot-product Softmax. Heads still specialize in different temporal patterns while attending a **common** value subspace; \(\tilde A\) acts as an ensemble of attention maps.[^src-tft]

## Use in TFT

Inside the temporal fusion decoder, static-enriched features \(\Theta(t)\) self-attend with decoder masking (causal over past+future positions). Attention weights \(\alpha(t,n,\tau)\) support:

- **Persistent patterns** — lag/seasonality peaks (daily Electricity/Traffic; weekly Retail).
- **Regime detection** — deviation of \(\alpha(t,\cdot)\) from entity-average maps via Bhattacharyya-style distance (volatility regimes).[^src-tft]

Ablating instance-wise attention (trainable fixed attention matrix) hurts P90 substantially on average, confirming that dynamic long-range attention is not interchangeable with a static lag template.[^src-tft]

## Related Pages

- [[tft]], [[source-tft]]
- [[gated-residual-network]], [[variable-selection-network]]
- Contrast: standard multi-head attention in [[informer|Informer]] / general Transformers (per-head values)

---

[^src-tft]: [[source-tft]]
