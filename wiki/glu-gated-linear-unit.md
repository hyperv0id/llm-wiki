---
title: "Gated Linear Unit (GLU)"
type: technique
tags:
  - attention
  - feed-forward
  - activation-function
created: 2026-05-04
last_updated: 2026-05-04
source_count: 0
confidence: medium
status: active
---

# Gated Linear Unit (GLU)

**Gated Linear Units (GLU)** are activation-modulated linear transformations where the output is a component-wise product of two linear projections, one of which is gated by a non-linear activation:

$$\text{GLU}(\mathbf{x}) = (\mathbf{x}\mathbf{W}_1 + \mathbf{b}_1) \odot \sigma(\mathbf{x}\mathbf{W}_2 + \mathbf{b}_2)$$

GLU variants (SwiGLU, GEGLU, etc.) replace the sigmoid $\sigma$ with other activations like Swish or GELU. GLUs have become the dominant choice for feed-forward layers in modern Transformer architectures (LLaMA, PaLM, etc.) and are also used as efficient MoE experts in models like [[mixture-of-experts|FaST]].

## Related Pages
- [[mixture-of-experts]]
- [[gated-linear-units]]