---
title: "Graph Structure Aligner"
type: technique
tags:
  - graph-neural-network
  - variational-autoencoder
  - exogenous
  - time-series-forecasting
  - structure-alignment
created: 2026-07-13
last_updated: 2026-07-13
source_count: 1
confidence: high
status: active
---

# Graph Structure Aligner

The **Graph Structure Aligner** is a core module of [[gcgnet|GCGNet]] that constrains a generative forecaster by **aligning graph-structured joint correlations** between the generated full sequence and the ground-truth full sequence, instead of relying only on pointwise reconstruction.[^src-gcgnet]

## Pipeline

Given ground-truth full sequence \(S\) (history+future, endo+exo) and generated \(\tilde S\):[^src-gcgnet]

1. **Patchify + embed** along time → \(S^p,\tilde S^p\in\mathbb{R}^{(N+D)\times L\times d}\).
2. **Graph Learner**: \(A'=\mathrm{GELU}((W_1 X^p)(W_2 X^p)^\top)\), then symmetrize \(\tilde A=\frac12(A'+A'^\top)\).
3. **Graph VAE**: \(A=\mathrm{VAE}(\tilde A)\) denoises pairwise similarities and models uncertainty in adjacency.
4. **Alignment loss**: \(L_{\mathrm{align}}=\|A-\hat A\|_1\), where \(A\) is from ground-truth patches and \(\hat A\) from generated patches.

Optimizing \(L_{\mathrm{align}}\) pushes the [[variational-generator-exogenous|Variational Generator]] to produce sequences whose **patch-wise joint temporal–channel relations** match the true structure, improving robustness under noisy observations.[^src-gcgnet]

## Design notes

- Shared Graph VAE for \(A\) and \(\hat A\) can **degenerate** (identical outputs for all inputs) if the adjacency is never used downstream; [[graph-refiner]] closes this loop by consuming \(\hat A\) in the prediction path.[^src-gcgnet]
- Ablation: removing \(L_{\mathrm{align}}\) or replacing Graph VAE with a deterministic Graph Learner degrades MSE/MAE on NP/PJM/DE/Energy.[^src-gcgnet]

## Links

- Entity: [[gcgnet]]
- Source: [[source-gcgnet]]
- Related: [[joint-temporal-channel-correlation]], [[graph-refiner]], [[variational-autoencoder]]

---

[^src-gcgnet]: [[source-gcgnet]]
