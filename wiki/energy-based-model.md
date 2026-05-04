---
title: "Energy-Based Model (EBM)"
type: concept
tags:
  - generative-model
  - energy-function
  - score-function
created: 2026-05-04
last_updated: 2026-05-04
source_count: 0
confidence: medium
status: active
---

# Energy-Based Model (EBM)

An **Energy-Based Model (EBM)** is a generative modeling framework where probability distributions are defined through an unnormalized **energy function** $E_\theta(\mathbf{x})$:

$$p_\theta(\mathbf{x}) = \frac{\exp(-E_\theta(\mathbf{x}))}{Z_\theta}$$

where $Z_\theta = \int \exp(-E_\theta(\mathbf{x})) d\mathbf{x}$ is the intractable normalizing constant (partition function). This is the same form as a Boltzmann distribution from statistical mechanics. EBMs are foundational to many modern generative approaches, including [[score-based-generative-modeling|score-based models]] where the score function $\nabla_\mathbf{x} \log p(\mathbf{x}) = -\nabla_\mathbf{x} E(\mathbf{x})$ provides a way to generate samples without computing $Z_\theta$.

## Related Pages
- [[score-function]]
- [[score-based-generative-modeling]]
- [[diffusion-model]]