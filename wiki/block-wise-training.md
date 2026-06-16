---
title: "Block-wise Training"
type: concept
tags:
  - training-methods
  - memory-optimization
  - neural-networks
created: 2026-06-16
last_updated: 2026-06-16
source_count: 1
confidence: high
status: active
---

# Block-wise Training

Block-wise training (also called layer-wise training when each block is a single layer) refers to methods that partition neural networks into smaller independently trainable components, promising dramatic memory savings by avoiding the need to store activations across all layers simultaneously[^src-diffusionblocks].

## Motivation: Memory Bottleneck

End-to-end backpropagation requires storing intermediate activations throughout all network layers during training[^src-diffusionblocks]. This causes memory consumption to grow linearly with network depth, creating computational bottlenecks that limit both research flexibility and practical deployment[^src-diffusionblocks].

For an L-layer network where each layer has parameter size P and activation size A, standard training with Adam optimizer requires $(4P + A)L$ total memory, where 4P accounts for parameters, gradients, and optimizer states (momentum + variance)[^src-diffusionblocks].

## Historical Approaches and Limitations

Prior block-wise training methods consistently underperform end-to-end training due to two fundamental challenges[^src-diffusionblocks]:

### 1. Lack of Theoretical Grounding

Existing methods rely on ad-hoc local objectives without principled coordination between blocks[^src-diffusionblocks]:

- **Forward-Forward** (Hinton, 2022): Uses contrastive objectives that fundamentally limit application to classification tasks[^src-diffusionblocks]
- **Greedy Layer-wise** (Bengio et al., 2006): Layer-by-layer pretraining without global coherence guarantees[^src-diffusionblocks]
- **Local Error Signals** (Nøkland & Eidnes, 2019; Belilovsky et al., 2019): Heuristic auxiliary losses at intermediate layers[^src-diffusionblocks]

### 2. Limited Applicability

Prior methods require paradigm-specific designs and task-specific objectives that do not naturally extend beyond classification[^src-diffusionblocks]. Results are typically demonstrated only on custom architectures without systematic procedures for modern transformers[^src-diffusionblocks].

## DiffusionBlocks: Principled Solution

DiffusionBlocks (Shing et al., ICLR 2026) provides the first theoretically grounded framework by interpreting residual networks as discretized diffusion processes[^src-diffusionblocks]. Key innovations:

### Theoretical Foundation

Block training objectives derive from score matching theory rather than ad-hoc heuristics[^src-diffusionblocks]. Each block learns to denoise within an assigned noise-level range, and consistent local optimization collectively yields a faithful approximation of the global reverse diffusion process[^src-diffusionblocks].

### True Independence

Unlike prior methods with shared parameters or joint fine-tuning, DiffusionBlocks achieves complete block isolation[^src-diffusionblocks]. Each block trains independently with gradients for only L/B layers at a time, where B is the number of blocks[^src-diffusionblocks].

### Broad Applicability

DiffusionBlocks provides a systematic conversion procedure for any residual network, particularly modern transformers, with minimal modifications[^src-diffusionblocks]. Successfully demonstrated on:

- Vision transformers (classification)
- Diffusion models (image generation)
- Autoregressive transformers (text generation)
- Masked diffusion models (text generation)
- Recurrent-depth models (text generation)

All applications maintain competitive performance with end-to-end training[^src-diffusionblocks].

## Memory-Computation Tradeoffs

### Block-wise Training

- **Memory**: Reduced by factor B (only L/B layers active)[^src-diffusionblocks]
- **Computation**: Same as standard training (L×K layer evaluations for K iterations)[^src-diffusionblocks]
- **Training time**: Minimal overhead (~7% from noise conditioning)[^src-diffusionblocks]

### Activation Checkpointing

- **Memory**: Reduces only activation memory, not parameters/gradients/optimizer states[^src-diffusionblocks]
- **Computation**: Increases by ~33% due to recomputation[^src-diffusionblocks]
- **Training time**: Approximately 4/3× slower[^src-diffusionblocks]

### Composition

DiffusionBlocks and activation checkpointing compose multiplicatively for maximum memory reduction[^src-diffusionblocks]. DiffusionBlocks also enables embarrassingly parallel training across blocks with zero communication overhead[^src-diffusionblocks].

## Comparison: Forward-Forward vs DiffusionBlocks

| Aspect | Forward-Forward | DiffusionBlocks |
|--------|-----------------|-----------------|
| Objective | Contrastive (positive/negative samples) | Score matching (denoising) |
| Task scope | Classification only | Classification + generation |
| Theory | Heuristic | Diffusion theory |
| Architecture | Custom designs | Systematic transformer conversion |
| Performance | 7.85% (CIFAR-100 ViT) | 59.30% (vs 60.25% baseline) |

Source: DiffusionBlocks experiments[^src-diffusionblocks]

## Open Questions

1. **Optimal partitioning**: No principled method yet for determining ideal block granularity[^src-diffusionblocks]
2. **Performance gain mechanism**: Why moderate partitioning (B=2-3) sometimes outperforms end-to-end training needs theoretical analysis[^src-diffusionblocks]
3. **Architecture generalization**: Extension to non-residual architectures like U-Net remains challenging[^src-diffusionblocks]

## Related Concepts

- [[memory-efficient-training]] — broader strategies
- [[diffusion-model]] — theoretical foundation for DiffusionBlocks
- [[residual-connections-as-diffusion]] — key insight enabling principled block-wise training
- [[activation-checkpointing]] — complementary memory optimization

[^src-diffusionblocks]: [[source-diffusionblocks]]
