---
title: "AMS-MoE"
type: technique
tags:
  - mixture-of-experts
  - multi-scale
  - dynamic-routing
  - time-series
  - traffic-forecasting
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: medium
status: active
---

# AMS-MoE (Adaptive Multi-Scale Mixture of Experts)

AMS-MoE is the core routing architecture of [[hephestus|HEPHAESTUS]], enabling input-adaptive multi-scale temporal modeling for traffic forecasting. It replaces the fixed-scale decomposition strategies used in prior [[timemixer|TimeMixer]] and [[pathformer|PathFormer]] with a dynamic router that selects which temporal scales to activate per input[^src-hephestus].

## How It Works

### Moving-Patch

An enhanced patching mechanism with three phases[^src-hephestus]:

1. **Left-Padding with Boundary Replication**: Input extended by prepending P−1 copies of the initial element for temporal continuity at boundaries.
2. **Sliding Patch Extraction**: Overlapping patches extracted with stride 1 across the extended sequence, producing H patches with dense temporal coverage.
3. **Linear Projection**: Patches flattened and projected via learnable weight W ∈ R^(P·d × C).

### Temporal-Aware Expert Routing

Each expert is a Transformer encoder operating at a fixed patch size, representing one temporal resolution (small patch = fine-grained; large patch = long-term trend)[^src-hephestus]:

1. **Multi-scale feature extraction**: Raw input processed through Moving-Patch at each candidate patch size pi, fused via learned Softmax-gated combination.
2. **Compact temporal representation**: Multi-scale features + original input → linear projection → Xh ∈ R^d.
3. **Noisy Top-K gating**:
   - Soft assignment: R(Xh) = Softmax(Xh·Wr + ϵ·Softplus(Xh·Wnoise)), ϵ ~ N(0,1)
   - Top-K sparsification: Only K experts receive non-zero weights
   - Gaussian noise encourages exploration during training, preventing premature convergence to suboptimal scales[^src-hephestus].

### Multi-Scale Output Aggregation

Final output = weighted sum of selected expert outputs: Xout = Σ S(R̄i > 0)·R̄i·Ei(Xit), where only activated (Top-K) experts contribute[^src-hephestus].

### Load Balancing

Auxiliary loss Laux = M·Σ fi·ri (fi = expert selection frequency; ri = mean router probability) prevents routing collapse where the router consistently selects only a small subset of experts[^src-hephestus].

## Key Properties

- **Input-adaptive**: Different inputs activate different scale experts. Peak hours → small patches; off-peak → large patches[^src-hephestus].
- **Optimal config**: M=4 experts, K=2 sparsity (ablation-tuned on METR-LA and PEMS08)[^src-hephestus].
- **Most critical component**: Removing AMS-MoE causes the largest performance degradation in ablation studies, more than removing PTA or HSA[^src-hephestus].

## Comparison to Related Approaches

| Method | Scale selection | Routing |
|--------|----------------|---------|
| [[timemixer|TimeMixer]] | Fixed down-sampling pyramid | None (linear mixing) |
| [[pathformer|PathFormer]] | Fixed patch sizes + learnable pathways | Soft routing via pathways |
| [[hephestus|AMS-MoE]] | Input-adaptive via dynamic router | Noisy Top-K sparse MoE |

[^src-hephestus]: [[source-hephestus]]
