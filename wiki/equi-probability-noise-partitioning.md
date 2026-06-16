---
title: "Equi-Probability Noise Partitioning"
type: technique
tags:
  - diffusion-models
  - training-strategy
  - noise-schedule
  - block-wise-training
created: 2026-06-16
last_updated: 2026-06-16
source_count: 1
confidence: high
status: active
---

# Equi-Probability Noise Partitioning

Equi-probability partitioning is a principled strategy for dividing the noise-level range in diffusion models based on equal cumulative probability mass rather than uniform spacing, ensuring balanced parameter utilization and learning difficulty across blocks[^src-diffusionblocks].

## Motivation: Non-Uniform Denoising Difficulty

In diffusion models trained with log-normal noise distributions (following EDM framework), different noise levels contribute unequally to generation quality[^src-diffusionblocks]. Intermediate noise levels are perceptually most important—where image structure emerges—and receive higher probability mass during training[^src-diffusionblocks].

Naive uniform partitioning ($\sigma_b = \sigma_{\min} + b \cdot (\sigma_{\max} - \sigma_{\min})/B$) fails to account for this varying difficulty, leading to imbalanced block utilization[^src-diffusionblocks].

## Mathematical Formulation

### Goal

For B blocks, choose boundaries $\{\sigma_b\}_{b=0}^B$ such that each block handles exactly $1/B$ of the total probability mass[^src-diffusionblocks]:

$$\int_{\sigma_b}^{\sigma_{b-1}} p_{\text{noise}}(\sigma) d\sigma = \frac{1}{B}$$

### Log-Normal Noise Distribution

Following Karras et al. (2022), the noise distribution is log-normal[^src-diffusionblocks]:

$$\log \sigma \sim \mathcal{N}(P_{\text{mean}}, P_{\text{std}}^2)$$

Typical values: $P_{\text{mean}} = -1.2$, $P_{\text{std}} = 1.2$[^src-diffusionblocks].

### Boundary Computation

Boundaries are computed via inverse CDF[^src-diffusionblocks]:

$$\sigma_b = \exp\left(P_{\text{mean}} + P_{\text{std}} \cdot \Phi^{-1}(q_b)\right)$$

where:
- $\Phi^{-1}$ is the inverse standard normal CDF
- $q_b = q_{\min} + \frac{b}{B}(q_{\max} - q_{\min})$ linearly spaces quantiles
- $q_{\min/\max} = \Phi\left(\frac{\log \sigma_{\min/\max} - P_{\text{mean}}}{P_{\text{std}}}\right)$

## Effect on Block Capacity Allocation

Equi-probability partitioning automatically adjusts block width based on denoising difficulty[^src-diffusionblocks]:

| Noise Level | Denoising Difficulty | Probability Mass | Block Width |
|-------------|----------------------|------------------|-------------|
| Very high ($\sigma \to \sigma_{\max}$) | Easy (pure noise → coarse structure) | Low | Wide |
| Intermediate | Hard (critical structure formation) | High | **Narrow** |
| Very low ($\sigma \to \sigma_{\min}$) | Easy (fine detail refinement) | Low | Wide |

Blocks assigned to intermediate noise levels receive narrower intervals, concentrating capacity where learning is most challenging[^src-diffusionblocks].

## Experimental Validation

### Ablation: Uniform vs Equi-Probability (CIFAR-10, DiT-S/2, B=3)

| Partitioning | Layer Distribution | FID ↓ |
|--------------|-------------------|-------|
| Uniform | [4,4,4] | 43.53 |
| Uniform | [3,6,3] | 43.59 |
| Uniform | [6,4,2] | 47.49 |
| Uniform | [2,4,6] | 42.37 |
| **Equi-Prob** | **[4,4,4]** | **38.03** |
| Equi-Prob | [3,6,3] | 41.64 |
| Equi-Prob | [6,4,2] | 45.42 |
| Equi-Prob | [2,4,6] | 40.40 |

Key findings[^src-diffusionblocks]:
1. Equi-probability significantly outperforms uniform across all layer distributions (best: 38.03 vs 42.37)
2. Within equi-probability, uniform layer distribution [4,4,4] works best
3. Practitioners can simply divide layers equally; noise-based partitioning handles difficulty balancing automatically

### Comparison with NoProp

On CIFAR-100 classification with NoProp's architecture[^src-diffusionblocks]:

| Method | Continuous-Time | Block-wise | Accuracy |
|--------|----------------|-----------|----------|
| Backprop (BPTT) | — | ✗ | 47.80% |
| NoProp-DT | ✗ | ✓ | 46.06% |
| NoProp-CT | ✓ | ✗ | 21.31% |
| NoProp-FM | ✓ | ✗ | 37.57% |
| **DiffusionBlocks** | **✓** | **✓** | **46.88%** |

DiffusionBlocks is the only method achieving both continuous-time formulation and block-wise training, demonstrating that equi-probability partitioning with independent denoisers per block is crucial for success[^src-diffusionblocks].

## Relationship to Loss Weighting

Equi-probability partitioning works in conjunction with EDM's loss weighting function[^src-diffusionblocks]:

$$w(\sigma) = \frac{\sigma^2 + \sigma_{\text{data}}^2}{(\sigma \cdot \sigma_{\text{data}})^2}$$

The weighting counteracts sampling bias from the log-normal distribution, ensuring balanced gradient magnitudes across all noise levels[^src-diffusionblocks]. Without this weighting, equi-probability partitioning would not be effective[^src-diffusionblocks].

## Extension to Masked Diffusion

For discrete masked diffusion language models, equi-probability partitioning applies to the **masking schedule** rather than continuous noise levels[^src-diffusionblocks].

Given masking schedule $\alpha(t): [0,1] \to [1,0]$ (probability of remaining unmasked), the effective density is $-\alpha'(t)$[^src-diffusionblocks]. Equal probability mass in $\alpha$ gives[^src-diffusionblocks]:

$$\alpha_b = 1 - \frac{b}{B}, \quad b = 0, \ldots, B$$

For linear schedule $\alpha(t) = 1-t$, this simplifies to $t_b = b/B$[^src-diffusionblocks].

## Theoretical Interpretation

Equi-probability partitioning induces a form of curriculum learning[^src-diffusionblocks]:
- All blocks receive equal training signal (equal probability mass)
- Blocks naturally specialize to their assigned difficulty level
- Balanced difficulty allocation prevents under-utilized or overtaxed blocks

This may partially explain why moderate block partitioning (B=2-3) sometimes outperforms end-to-end training—the specialization introduces beneficial structure without the heuristics of manual curriculum design[^src-diffusionblocks].

## Implementation Details

### Block Overlap

To smooth transitions, blocks' noise intervals are slightly extended in log-$\sigma$ space[^src-diffusionblocks]:

$$\text{Training range for block } b: \left[\frac{\sigma_b}{\alpha_b}, \alpha_b \sigma_{b-1}\right]$$

where $\alpha_b = (\sigma_{b-1}/\sigma_b)^\gamma$ and $\gamma \in [0, 0.1]$ controls overlap (default: 0.05 for images, 0.1 for text)[^src-diffusionblocks].

### Sampling During Training

Each iteration:
1. Sample a block $b \in [1, B]$ uniformly[^src-diffusionblocks]
2. Sample $\sigma$ from $p_{\text{noise}}^{(b)}$—the renormalized log-normal restricted to block $b$'s range[^src-diffusionblocks]
3. Train only block $b$ with the sampled $\sigma$[^src-diffusionblocks]

## Related Concepts

- [[edm-design-space]] — framework providing log-normal distribution and loss weighting
- [[block-wise-training]] — application context
- [[curriculum-learning]] — related learning paradigm
- [[score-matching]] — training objective used within each partition

[^src-diffusionblocks]: [[source-diffusionblocks]]
