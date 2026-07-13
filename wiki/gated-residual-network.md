---
title: "Gated Residual Network (GRN)"
type: technique
tags:
  - time-series
  - gating
  - residual
  - tft
  - glu
  - architecture
created: 2026-07-13
last_updated: 2026-07-13
source_count: 1
confidence: high
status: active
---

# Gated Residual Network (GRN)

The **Gated Residual Network** is the primary building block of the [[tft|Temporal Fusion Transformer]]. It applies optional context-conditioned nonlinear processing with a residual skip and a [[glu-gated-linear-unit|GLU]] gate so the model can suppress unused capacity on small or noisy datasets.[^src-tft]

## Definition

Given primary input \(a\) and optional context \(c\):

\[
\begin{aligned}
\eta_2 &= \mathrm{ELU}(W_{2,\omega}a + W_{3,\omega}c + b_{2,\omega}), \\
\eta_1 &= W_{1,\omega}\eta_2 + b_{1,\omega}, \\
\mathrm{GRN}_\omega(a,c) &= \mathrm{LayerNorm}\bigl(a + \mathrm{GLU}_\omega(\eta_1)\bigr).
\end{aligned}
\]

When no context is provided, \(c=0\). Dropout is applied to \(\eta_1\) before gating during training.[^src-tft]

## Role of GLU

\[
\mathrm{GLU}_\omega(\gamma)=\sigma(W_{4,\omega}\gamma+b_{4,\omega})\odot (W_{5,\omega}\gamma+b_{5,\omega})
\]

allows the nonlinear branch to vanish (gate near 0), recovering a near-identity residual path. Combined with ELU’s identity-like behavior for large positive pre-activations, GRN can act nearly linear when nonlinear processing is harmful.[^src-tft]

## Uses inside TFT

| Location | Role |
|----------|------|
| Variable selection | Softmax weights from \(\mathrm{GRN}(\Xi_t,c_s)\); per-variable feature GRNs |
| Static encoders | Map selected static features to contexts \(c_s,c_e,c_c,c_h\) |
| Static enrichment | \(\theta(t,n)=\mathrm{GRN}(\tilde\phi(t,n),c_e)\) |
| Position-wise FFN | Post-attention GRN with gated residual over the transformer block |

Ablating GLU gates (replace by linear+ELU) raises P90 loss ~1.9% on average and ~4.1% on Volatility in the TFT paper — largest relative gating benefit on the smallest/noisiest set.[^src-tft]

## Related Pages

- [[tft]], [[source-tft]]
- [[glu-gated-linear-unit]], [[gated-linear-units]]
- [[variable-selection-network]]

---

[^src-tft]: [[source-tft]]
