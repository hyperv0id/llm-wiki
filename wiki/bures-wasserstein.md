---
title: "Bures-Wasserstein Discrepancy"
type: technique
tags:
  - optimal-transport
  - wasserstein
  - distribution-alignment
  - gaussian
created: 2026-07-17
last_updated: 2026-07-17
source_count: 1
confidence: medium
status: active
---

## Definition

The **Bures-Wasserstein (BW) discrepancy** is the closed-form squared 2-Wasserstein distance between two multivariate Gaussian distributions. For $\mathcal{N}(\mu_1, \Sigma_1)$ and $\mathcal{N}(\mu_2, \Sigma_2)$:

$$\text{BW} = \|\mu_1 - \mu_2\|^2_2 + \text{Tr}\left(\Sigma_1 + \Sigma_2 - 2(\Sigma_1^{1/2} \Sigma_2 \Sigma_1^{1/2})^{1/2}\right)$$

where the first term is **mean alignment** and the second (the Bures metric $\mathcal{B}(\Sigma_1, \Sigma_2)$) is **covariance alignment**.[^src-distdf]

## Role in DistDF

DistDF approximates the joint-distribution Wasserstein discrepancy under a Gaussian assumption, yielding the BW metric as a tractable, differentiable loss component. Given a batch, DistDF concatenates history $X$ with labels $Y$ (or forecasts $\hat{Y}$) into joint vectors $Z = [X, Y]$ and $\hat{Z} = [X, \hat{Y}]$, then computes BW between their empirical Gaussian fits. The overall loss is:

$$\mathcal{L}_{\text{DistDF}} = \gamma \cdot \text{BW}(\mu_Z, \mu_{\hat{Z}}, \Sigma_Z, \Sigma_{\hat{Z}}) + (1-\gamma) \cdot \text{MSE}$$

This adds negligible training cost (<1ms per batch at $T=1024$) and zero inference cost.[^src-distdf]

## Limitations

BW captures only first- and second-order moments (mean and covariance). Non-Gaussian real-world data may require higher-order statistics for full distributional characterization. Both mean and covariance alignment contribute synergistically—ablation in DistDF confirms that removing either component degrades performance.[^src-distdf]

## Related

- [[joint-distribution-wasserstein-alignment]] — the full DistDF framework using BW as its core discrepancy
- [[optimal-transport]] — theoretical foundation of Wasserstein distances
- [[gaussian-schrodinger-bridge]] — uses BW geometry for closed-form Gaussian SB solutions
- [[source-distdf]] — primary source paper

---
[^src-distdf]: [[source-distdf]]
