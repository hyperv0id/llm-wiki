---
title: "PI-MFM: Physics-informed multimodal foundation model for solving PDEs"
type: source-summary
tags:
  - physics-informed
  - multimodal
  - pde
  - foundation-model
  - 2025
created: 2026-07-07
last_updated: 2026-07-07
source_count: 1
confidence: medium
status: active
---

# PI-MFM: Physics-informed multimodal foundation model for solving PDEs

**Authors:** Min Zhu, Jingmin Sun, Zecheng Zhang, Hayden Schaeffer, Lu Lu  
**Year:** 2025  
**arXiv:** 2512.23056  
**Affiliations:** Yale University, Johns Hopkins University, University of Notre Dame, UCLA

## Summary

PI-MFM proposes a physics-informed multimodal foundation model (MFM) framework that directly enforces governing partial differential equations (PDEs) as loss constraints during both pretraining and adaptation, extending the paradigm of multi-operator learning (MOL) beyond purely data-driven approaches.[^src-pi-mfm]

### Key Innovations

1. **Physics-informed multi-operator learning** — Unlike existing MOL methods that treat PDEs as side information, PI-MFM embeds PDE residuals directly into the training objective via four loss terms: PDE residual loss $L_{PDE}$, initial condition loss $L_{IC}$, second-order initial condition loss $L_{IC'}$ (for second-order-in-time PDEs), and data prediction loss $L_{data}$.[^src-pi-mfm]

2. **Automatic symbolic PDE loss assembly** — Given a symbolic PDE expression encoded in Polish (prefix) notation, the framework automatically parses it into an expression tree, computes all required derivatives at collocation points in a single vectorized batch, and assembles the corresponding physics loss. This eliminates manual per-PDE loss derivation.[^src-pi-mfm]

3. **Architecture-flexible design** — PI-MFM separates the pipeline into data encoder, symbol encoder, fusion module, and query-coordinate-conditioned data decoder. Any MFM that accepts symbolic PDE descriptions can be trained with this framework. The paper uses PROSE as its backbone.[^src-pi-mfm]

4. **Zero-shot physics-informed fine-tuning** — A pretrained PI-MFM can adapt to unseen PDE families using only PDE residuals and initial/boundary conditions (no labeled solution pairs), rapidly achieving ~1% relative error and outperforming physics-only training from scratch.[^src-pi-mfm]

### Experimental Results

- Evaluated on 13 parametric 1D time-dependent PDE families (10 training + 3 held-out for generalization).
- Consistent improvements over purely data-driven counterparts, especially under sparse labeled points and partially observed time domains.[^src-pi-mfm]
- Physics losses improve robustness against label noise; resampling collocation points substantially improves accuracy.
- Comprehensive comparison of automatic differentiation (AD) vs. finite difference method (FDM) for derivative computation, providing guidance on backend selection.[^src-pi-mfm]

### Significance

PI-MFM provides a practical and scalable path toward data-efficient, transferable PDE solvers that combine the flexibility of multimodal foundation models with the physical fidelity of physics-informed neural networks (PINNs). It bridges the gap between single-operator PINNs and data-driven MOL by unifying symbolic PDE encoding with automatic physics loss construction.[^src-pi-mfm]

[^src-pi-mfm]: [[source-pi-mfm]]
