---
title: "Variational Generator (Exogenous Forecasting)"
type: technique
tags:
  - variational-autoencoder
  - exogenous
  - time-series-forecasting
  - generative-model
created: 2026-07-13
last_updated: 2026-07-13
source_count: 1
confidence: high
status: active
---

# Variational Generator (Exogenous Forecasting)

The **Variational Generator** in [[gcgnet|GCGNet]] produces **coarse** future series that seed joint graph construction for exogenous-aware forecasting.[^src-gcgnet]

## Steps

1. **Instance Normalization** unifies train/test distribution scale.[^src-gcgnet]
2. Separate VAEs map historical endogenous and exogenous inputs to coarse futures:
   \[
   \tilde Y^{\mathrm{endo}}=\mathrm{VAE}(X^{\mathrm{endo}}),\quad
   \tilde Y^{\mathrm{exo}}=\mathrm{VAE}(X^{\mathrm{exo}}).
   \]
3. If **true future exogenous** \(Y^{\mathrm{exo}}\) is available, it **replaces** \(\tilde Y^{\mathrm{exo}}\); otherwise the generated \(\tilde Y^{\mathrm{exo}}\) fills the future exo slots so the pipeline still runs without future covariates.[^src-gcgnet]
4. Concatenate history and (coarse) future channels into full sequence \(\tilde S\) for the [[graph-structure-aligner|Graph Structure Aligner]].[^src-gcgnet]

## Role

The generator is not the final forecaster: its job is to provide a full-horizon multi-channel canvas so that [[joint-temporal-channel-correlation|joint correlations]] can be evaluated as graphs. KL regularization \(L_{\mathrm{KL}}^{V}\) keeps the latent space well-behaved; replacing the VAE with an MLP hurts accuracy, especially under missing exogenous inputs.[^src-gcgnet]

## Links

- Entity: [[gcgnet]]
- Source: [[source-gcgnet]]
- Related: [[graph-structure-aligner]], [[graph-refiner]], [[variational-autoencoder]]

---

[^src-gcgnet]: [[source-gcgnet]]
