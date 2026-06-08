---
title: "PRDIM — Missing Pattern Recognized Diffusion Imputation Model for Missing Not At Random"
type: source-summary
tags:
  - time-series
  - data-imputation
  - diffusion-models
  - mnar
  - expectation-maximization
  - pattern-recognizer
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: high
status: active
---

**PRDIM** (Missing Pattern Recognized Diffusion Imputation Model) is a diffusion-based imputation framework for the **Missing Not At Random (MNAR)** setting, by Gyuwon Sim, Sumin Lee, Heesun Bae, Byeonghu Na, Doyun Kwon (KAIST), Ju-Hee Hwang, Jae-Young Lim (Seoul National University), and Il-Chul Moon (KAIST), arXiv:2605.25439v1 (May 2026, preprint)[^src-prdim].

## Problem: MNAR

In the real world, missingness often depends on the *unobserved values themselves* (e.g., a sensor fails or a patient's measurement is skipped *because* of the underlying value) — the [[missing-not-at-random|MNAR]] mechanism. Under MNAR the missing process is **non-ignorable**: one must model the mask distribution $p_\phi(M \mid X^{obs}, X^{mis})$, unlike MCAR/MAR where it can be ignored[^src-prdim]. Most diffusion imputers ([[csdi|CSDI]] and successors) instead train/evaluate under MCAR-style *artificial* masking $p_\theta(X \mid X^{obs,A}, A)$; if the original mask $M$ differs from the artificial mask $A$, such models transfer poorly to real missingness. PRDIM shows imputing **original** missing entries is markedly harder than artificial ones, and closes that gap[^src-prdim].

## Method

PRDIM jointly models the data $p_\theta(X)$ and the mask $p_\phi(M\mid X)$, maximizing the joint likelihood of observed data and mask $p_{\theta,\phi}(X^{obs}, M)$ via **Expectation-Maximization** (the missing values $X^{mis}$ are the latent variable)[^src-prdim]:

- **Pattern recognizer** $D_\phi$ — a discriminator (lineage: GAIN, not-MIWAE) trained with binary cross-entropy to approximate the per-entry missing probability $p(M\mid X)$. See [[pattern-recognizer-guidance]].
- **ELBO under diffusion** (Proposition 3.1): the joint log-likelihood lower-bounds to the diffusion VLB + $\mathbb{E}[\log p_\phi(M\mid X_0)]$ + an entropy term, explicitly folding the pattern recognizer into the diffusion objective.
- **Hard EM** (vs DiffPuter's soft EM) for stronger exploration of the $X^{mis}$ distribution; **EM monotonicity** of the joint log-likelihood is guaranteed (Corollary 3.2).

Two phases[^src-prdim]:
1. **Phase 1 — diffusion pre-training / pre-imputation**: a conditional diffusion backbone (Observed Reconstruction Task) trained with **adjacent target masking** — artificial missing entries are placed *near original missing values* (temporal axis for time series; neighboring pixels for images) rather than CSDI's MCAR masking, making the backbone robust to any missing pattern.
2. **Phase 2 — EM iteration**: *M-step* trains $\theta$ (diffusion) and $\phi$ (pattern recognizer) independently; *E-step* generates $X^{mis}$ via the reverse process with **pattern-recognizer guidance** — a gradient $\nabla_{X_t}\mathcal{L}_{PR}$ steering denoising toward mask-consistent imputations (a [[classifier-guidance|classifier-guidance]] analogue; Proposition 3.3), using a [[tweedies-formula|Tweedie]] posterior mean $\hat{X}_0$. A randomly-initialized recognizer gives near-zero (neutral) guidance early on.

## Results

Evaluated under MNAR across **three modalities**[^src-prdim]: time series (ETT, STOCK, PEMS-Bay), images (FMNIST, CelebA-HQ), and 5 UCI tabular datasets, vs 10 imputation baselines (Mean; discriminative TimesNet/TimeMixer++/BRITS/SAITS; generative GP-VAE/not-MIWAE; diffusion CSDI/MTSCI/cDiffPuter).
- On original (real) missing entries, PRDIM improves over the strongest diffusion baseline cDiffPuter — e.g. RMSE 1.209→1.057, MAE 0.782→0.663, MRE 46.19→39.16 — with gains most pronounced on **out-of-sample** (unseen) missing values[^src-prdim].
- Beats DiffPuter and classical imputers (MissForest, MICE, HyperImpute) on tabular data; recovers semantic structure (eyes/nose/mouth) on CelebA-HQ where vanilla diffusion fills with averaged color[^src-prdim].

## Key Findings (Ablations)

Removing the pattern recognizer or replacing hard EM with soft EM (= DiffPuter) both degrade performance significantly — explicit missing modeling and iterative EM are indispensable[^src-prdim]. The artificial missing rate of $M{-}A$ (10/50/90%) has limited influence. Under **MCAR** the advantage diminishes (the recognizer just learns randomness), confirming the benefit is MNAR-specific[^src-prdim].

## Limitations

Gradient-based pattern guidance adds moderate inference cost (the recognizer itself is lightweight)[^src-prdim].

[^src-prdim]: [[source-prdim]]
