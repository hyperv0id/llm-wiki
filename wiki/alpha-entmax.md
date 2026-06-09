---
title: "α-Entmax"
type: technique
tags:
  - attention
  - sparse-activation
  - graph-learning
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: medium
status: active
---

# α-Entmax

**α-Entmax** is a parametric, sparsity-controllable generalization of the softmax activation, governed by a tunable hyperparameter α[^src-fstllm]. It generalizes the family of normalizing transforms: α=1.0 recovers **softmax**, and α=2.0 recovers **sparsemax**, with intermediate values trading off density and sparsity[^src-fstllm].

It is defined as[^src-fstllm]:

$$\alpha\text{-Entmax}(z) = [(\alpha-1)z - \tau\mathbf{1}]_{+}^{1/(\alpha-1)}$$

where $[x]_+ := \max(x,0)$ and the threshold $\tau(z)$ is set so the outputs sum to one[^src-fstllm].

## Use in spatio-temporal graph learning

When spatial correlations among series are normalized with softmax, the result contains many low-weight entries (near-zero similarities). Applying graph convolution over these noisy entries causes inaccurate message passing and dilutes the focus on the node of interest[^src-fstllm]. α-Entmax mitigates this by zeroing out low-similarity edges — suppressing information flow from distant/noise nodes while amplifying closer ones — giving finer control over the normalized spatial-correlation scores[^src-fstllm]. [[fstllm|FSTLLM]] (ICML 2025) uses α=2.0 in its LLM-enhanced graph-construction module to produce a sparse adjacency matrix[^src-fstllm].

## Related

- [[fstllm]] — uses α-Entmax for sparse adjacency construction
- [[traffic-forecasting]] — adaptive graph learning in spatio-temporal models

[^src-fstllm]: [[source-fstllm]]

