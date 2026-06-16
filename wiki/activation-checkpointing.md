---
title: "Activation Checkpointing"
type: technique
tags:
  - memory-optimization
  - training-efficiency
  - gradient-computation
created: 2026-06-16
last_updated: 2026-06-16
source_count: 1
confidence: high
status: active
---

# Activation Checkpointing

Activation checkpointing (also known as gradient checkpointing, activation recomputation, or rematerialization) is a memory optimization technique that trades computation for memory by selectively storing only a subset of activations during the forward pass and recomputing others as needed during backpropagation[^src-diffusionblocks].

## Memory-Computation Trade-off

### Standard Backpropagation

Stores all intermediate activations from the forward pass to compute gradients during backward pass[^src-diffusionblocks]:

- **Memory**: $L \times A$ (where L = number of layers, A = activation size per layer)
- **Computation**: $3F$ (F = forward pass cost: 1F forward + 2F backward)

### With Checkpointing

Stores only strategic checkpoints; recomputes intermediate activations on-demand during backward pass[^src-diffusionblocks]:

- **Memory**: Approximately $\sqrt{L} \times A$ with optimal checkpointing strategy
- **Computation**: Approximately $4F$ (1F forward + recomputation + 2F backward)
- **Overhead**: ~33% increase in training time[^src-diffusionblocks]

## What It Reduces (and Doesn't)

### Reduces
- **Activation memory only**[^src-diffusionblocks]

### Does NOT Reduce
- Parameters (P)[^src-diffusionblocks]
- Gradients (P)[^src-diffusionblocks]
- Optimizer states (2P for Adam)[^src-diffusionblocks]

For modern large models, the $4P \times L$ component often dominates over activations, limiting the effectiveness of checkpointing alone[^src-diffusionblocks].

## Comparison with DiffusionBlocks

| Aspect | Activation Checkpointing | DiffusionBlocks |
|--------|-------------------------|-----------------|
| **Activations** | Reduced to ~√L×A | Reduced by B× to (L/B)×A |
| **Parameters** | Unchanged: L×P | Reduced by B× to (L/B)×P |
| **Gradients** | Unchanged: L×P | Reduced by B× to (L/B)×P |
| **Optimizer** | Unchanged: L×2P | Reduced by B× to (L/B)×2P |
| **Computation** | +33% overhead | No overhead (~7% from noise conditioning) |
| **Parallelizable** | No | Yes (embarrassingly parallel across blocks) |

Source: DiffusionBlocks analysis[^src-diffusionblocks]

## Composition: Checkpointing + DiffusionBlocks

The two techniques compose multiplicatively for maximum memory reduction[^src-diffusionblocks]:

$$\text{Memory} = \frac{4P \times L}{B} + \frac{\sqrt{L/B} \times A}{\text{recompute}}$$

DiffusionBlocks reduces all components by B, then checkpointing further reduces activations to $\sqrt{L/B}$[^src-diffusionblocks].

## When to Use

### Use Activation Checkpointing When:
- Activation memory dominates over parameters/optimizer[^src-diffusionblocks]
- Cannot modify training paradigm (need exact end-to-end gradients)
- 33% compute overhead is acceptable
- Working with very deep networks (large L)

### Use DiffusionBlocks Instead When:
- All memory components need reduction (parameters + gradients + optimizer + activations)[^src-diffusionblocks]
- Architecture has residual connections with matching dimensions
- Can tolerate minimal accuracy variation
- Want embarrassingly parallel training

### Use Both When:
- Facing extreme memory constraints
- Need maximum possible memory reduction
- Willing to pay both 33% compute cost and minimal accuracy variation

## Implementation Strategies

### Uniform Checkpointing
Store every k-th layer activation; recompute k-1 intermediate layers during backward pass.

### Optimal Checkpointing
Use dynamic programming to minimize recomputation while meeting memory budget. Achieves $\sqrt{L}$ memory with optimal checkpoint placement.

### Selective Checkpointing
Store activations at computationally expensive operations (e.g., attention layers); recompute cheap operations (e.g., LayerNorm, residual additions).

## Practical Considerations

### Framework Support
- **PyTorch**: `torch.utils.checkpoint.checkpoint()`
- **TensorFlow**: `tf.recompute_grad()`
- **JAX**: `jax.checkpoint()` or `jax.remat()`

### Determinism
Requires deterministic operations during recomputation. Stochastic operations (dropout with different seeds) can cause gradient mismatch.

### Memory-Compute Profile
Training time increases by ~33% (from 3F to 4F)[^src-diffusionblocks]. Actual overhead depends on:
- Ratio of forward to backward compute
- Memory access patterns
- Hardware characteristics

## Historical Context

Activation checkpointing has been a standard technique for training very deep networks since early deep learning. DiffusionBlocks (2026) represents the first method to reduce **all** memory components rather than just activations[^src-diffusionblocks].

## Related Techniques

- [[gradient-accumulation]] — trading memory for compute in batch dimension
- [[mixed-precision-training]] — reducing memory via FP16/BF16
- [[block-wise-training]] — DiffusionBlocks approach reducing all components

## Related Concepts

- [[memory-efficient-training]] — broader context
- [[diffusionblocks]] — complementary technique reducing all memory components

[^src-diffusionblocks]: [[source-diffusionblocks]]
