---
title: "DiffusionBlocks"
type: entity
tags:
  - training-method
  - diffusion-models
  - memory-optimization
  - sakana-ai
created: 2026-06-16
last_updated: 2026-08-08
source_count: 1
confidence: high
status: active
---

# DiffusionBlocks

**DiffusionBlocks** is a principled framework for transforming transformer-based neural networks into independently trainable blocks by interpreting residual connections as discretized steps of continuous-time diffusion processes (Shing, Koyama & Akiba, ICLR 2026)[^src-diffusionblocks].

## Key Innovation

Unlike prior block-wise training methods that rely on ad-hoc local objectives, DiffusionBlocks derives each block's training objective from score matching theory, enabling genuinely independent block training while maintaining competitive performance with end-to-end backpropagation[^src-diffusionblocks].

## Core Mechanism

The framework leverages the fact that residual connections naturally correspond to Euler discretization of the probability flow ODE in diffusion models[^src-diffusionblocks]:

$$z_\ell = z_{\ell-1} + f_{\theta_\ell}(z_{\ell-1}) \quad \Leftrightarrow \quad z_{\sigma_\ell} = z_{\sigma_{\ell-1}} + \frac{\Delta\sigma_\ell}{\sigma_{\ell-1}}(z_{\sigma_{\ell-1}} - D_\theta(z_{\sigma_{\ell-1}}, \sigma_{\ell-1}))$$

This correspondence allows partitioning networks into blocks that each handle specific noise-level ranges and train completely independently[^src-diffusionblocks].

## Performance Highlights

- **Vision Transformers** (CIFAR-100): 59.30% accuracy vs 60.25% baseline, with 3× memory reduction; Forward-Forward achieves only 7.85%[^src-diffusionblocks]
- **Diffusion Models** (ImageNet-256): FID 9.00/10.63 vs 9.01/12.09 baseline, 3× training and inference memory reduction[^src-diffusionblocks]
- **Masked Diffusion** (text8): 1.45 BPC vs 1.56 BPC baseline — **better than end-to-end**[^src-diffusionblocks]
- **Autoregressive LMs**: Comparable MAUVE and perplexity with 4× memory reduction[^src-diffusionblocks]
- **Recurrent-Depth Models** (Huginn): Better MAUVE (0.70 vs 0.49) while eliminating 32 training iterations[^src-diffusionblocks]

## Technical Components

### 1. Equi-Probability Partitioning

Partition noise levels based on equal cumulative probability mass rather than uniform spacing[^src-diffusionblocks]:

$$\int_{\sigma_b}^{\sigma_{b-1}} p_{\text{noise}}(\sigma) d\sigma = \frac{1}{B}$$

This ensures balanced parameter utilization across blocks[^src-diffusionblocks].

### 2. Independent Block Training

Each block $b$ trains independently via score matching[^src-diffusionblocks]:

$$\mathcal{L}_b(\theta_b) = \mathbb{E}_{(x,y), \sigma \sim p_{\text{noise}}^{(b)}, \epsilon} \left[ w(\sigma) \cdot \text{Loss}(\bar{f}_{\theta_b|\sigma}(x, y+\sigma\epsilon), y) \right]$$

Only L/B layers require gradients at a time, achieving B× memory reduction[^src-diffusionblocks].

### 3. Noise-Level Conditioning

Blocks are augmented with noise-level conditioning (e.g., AdaLN) and extended input $(x, z_\sigma)$ where $z_\sigma = y + \sigma\epsilon$[^src-diffusionblocks].

## Efficiency Characteristics

### Memory

- **Training**: $(4P + A) \times (L/B)$ — all components reduced by factor B[^src-diffusionblocks]
- **Inference** (diffusion models): B× reduction per denoising step[^src-diffusionblocks]

### Computation

- **Training**: Same as standard training (L×K layer evaluations)[^src-diffusionblocks]
- **Wall-time overhead**: ~7% from noise conditioning[^src-diffusionblocks]
- **Parallelization**: Embarrassingly parallel across blocks with zero communication[^src-diffusionblocks]

### Composability

- Combines with activation checkpointing for maximum memory reduction[^src-diffusionblocks]
- Compatible with mixed-precision training[^src-diffusionblocks]

## Surprising Finding

Moderate block partitioning (B=2 or B=3) sometimes **outperforms end-to-end training** on ImageNet and CIFAR-10[^src-diffusionblocks]. Hypothesized reasons:
1. Direct denoising objectives create different optimization structure than standard backpropagation[^src-diffusionblocks]
2. Noise-range specialization induces beneficial curriculum learning effects[^src-diffusionblocks]

## Comparison with Competing Methods

### vs Forward-Forward Algorithm

| Aspect | Forward-Forward | DiffusionBlocks |
|--------|-----------------|-----------------|
| Theoretical foundation | Heuristic contrastive learning | Diffusion theory + score matching |
| Task applicability | Classification only | Classification + generation |
| CIFAR-100 ViT accuracy | 7.85% | 59.30% (baseline: 60.25%) |

### vs NoProp

On CIFAR-100 with NoProp's architecture, DiffusionBlocks (46.88%) is the only method achieving both continuous-time formulation and block-wise training[^src-diffusionblocks]:

- NoProp-DT: 46.06% (discrete, block-wise)
- NoProp-CT: 21.31% (continuous, not block-wise)
- NoProp-FM: 37.57% (continuous, not block-wise)

## Applicable Architectures

Successfully demonstrated on[^src-diffusionblocks]:
- Vision Transformers (ViT)
- Diffusion Transformers (DiT)
- Autoregressive transformers (Llama-style)
- Masked diffusion models (MD4-based)
- Recurrent-depth models (Huginn)

**Requirement**: Residual connections with matching input-output dimensions[^src-diffusionblocks].  
**Limitation**: U-Net-style architectures with dimension changes currently unsupported[^src-diffusionblocks].

## Implementation

Open-source code available at: https://github.com/SakanaAI/DiffusionBlocks[^src-diffusionblocks]

## Future Directions

1. **Architecture extension**: Enabling U-Net and other dimension-changing architectures[^src-diffusionblocks]
2. **Sampler exploration**: Using non-Euler discretization schemes with modified inter-block connections[^src-diffusionblocks]
3. **Optimal partitioning**: Principled method for determining ideal block granularity[^src-diffusionblocks]
4. **Pre-trained model conversion**: Fine-tuning large pre-trained models into block-wise form[^src-diffusionblocks]
5. **Theoretical analysis**: Formal understanding of why moderate partitioning sometimes improves performance[^src-diffusionblocks]

## Significance

DiffusionBlocks represents the first block-wise training method with:
- Theoretical grounding in diffusion models rather than heuristics[^src-diffusionblocks]
- True block independence without shared parameters or joint fine-tuning[^src-diffusionblocks]
- Systematic conversion procedure for modern transformer architectures[^src-diffusionblocks]
- Broad task applicability beyond classification to generative tasks[^src-diffusionblocks]
- Competitive or superior performance to end-to-end training[^src-diffusionblocks]

By dramatically reducing memory requirements without performance degradation, DiffusionBlocks helps democratize large-scale model training[^src-diffusionblocks].

## Related Concepts

- [[block-wise-training]] — general paradigm
- [[residual-connections-as-diffusion]] — theoretical foundation
- [[equi-probability-noise-partitioning]] — key technical innovation
- [[memory-efficient-training]] — broader context

[^src-diffusionblocks]: [[source-diffusionblocks]]
