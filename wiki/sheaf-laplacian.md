---
title: "Sheaf Laplacian"
type: technique
tags:
  - algebraic-topology
  - spectral-methods
  - graph-neural-network
  - linear-algebra
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: medium
status: active
---

# Sheaf Laplacian

The **sheaf Laplacian** $L_F$ is a generalization of the combinatorial graph Laplacian that arises from a [[cellular-sheaf|cellular sheaf]] structure. It encodes both the graph topology and the sheaf's restriction maps, enabling richer modeling of signal propagation than standard Laplacian-based methods[^src-ssf].

## Definition

Given a cellular sheaf $\mathcal{F}$ over a graph $G = (V, E)$, the sheaf Laplacian is defined as[^src-ssf]:

$$L_F = D - P - P^\top$$

where $D_v = \sum_{e; v \in e} \mathcal{F}_{v \triangleleft e} \mathcal{F}_{v \triangleleft e}^\top$ and $D = \text{diag}(D_1, D_2, \ldots, D_n)$ is block-diagonal. The matrix $P$ encodes the off-diagonal sheaf adjacency, built from restriction map pairs across edges.

The sheaf Laplacian models the **difference in sheaf-projected features** across each edge, aggregated node-by-node. It captures both structural (topological) and semantic (restriction-map-encoded) discrepancies between connected nodes[^src-ssf].

## Properties

- **Generalizes the graph Laplacian**: When stalk dimension $d=1$ and all restriction maps are identity ($\mathcal{F}_{v \triangleleft e} = 1$), $L_F$ reduces to the combinatorial graph Laplacian $L = D - A$[^src-ssf].
- **Positive semidefinite**: $L_F$ is symmetric PSD when the base graph is undirected, edge stalks have inner-product structures, and restriction maps satisfy $\mathcal{F}_{v \triangleleft e}^\top \mathcal{F}_{v \triangleleft e} \in S_d^+$. This guarantees real, non-negative eigenvalues[^src-ssf].
- **Mitigates oversmoothing**: Unlike the normalized graph Laplacian, whose diffusion leads all node representations to converge to the same value, sheaf Laplacian diffusion preserves local discriminative information through edge-specific restriction maps[^src-ssf].

## Eigendecomposition

The sheaf Laplacian admits eigendecomposition $L_F = U\Lambda U^T$ (Theorem 1 in [[ssf|SSF]]), where[^src-ssf]:

- $\Lambda = \text{diag}(\lambda_1, \lambda_2, \ldots, \lambda_{Nd})$ — eigenvalues sorted ascending
- $U = [u_1, u_2, \ldots, u_{Nd}]$ — orthonormal eigenvectors

Small eigenvalues correspond to **low-frequency components** (globally smooth, slowly varying across connected nodes). Large eigenvalues correspond to **high-frequency components** (localized, oscillatory, capturing sharp or irregular changes). The eigenvectors form an orthonormal basis for the space of sheaf signals, generalizing Fourier modes to the sheaf-theoretic setting[^src-ssf].

## Spectral Filtering

[[ssf|SSF]] applies a **heat kernel spectral filter** $g_\text{heat}(\lambda) = e^{-\alpha\lambda}$ over the sheaf Laplacian's eigenspectrum to selectively modulate frequency components. The filtering is performed as:

$$\hat{X}^{(l)} = U^T X^{(l)}, \quad \hat{X}^{(l+1)}_\text{filtered} = \hat{g}_\text{heat} \hat{X}^{(l)} W^{(l)}, \quad X^{(l+1)} = \sigma(U \hat{X}^{(l+1)}_\text{filtered})$$

This suppresses high-frequency noise while emphasizing the low-frequency components encoding the most coherent structural information across the graph[^src-ssf].

## Complexity

Constructing restriction maps and assembling $L_F$ costs $O(|E|d^2) = O(Nd^2)$ for sparse graphs. Computing $k$ truncated eigenpairs costs $O((Nd)^2 k)$. Each layer applies dense transforms $U_k^T X$ and $U_k \hat{X}$, costing $O(kNd^2)$. The eigendecomposition is the bottleneck, but remains tractable at $N \sim 10^2\text{–}10^3$ for traffic networks[^src-ssf].

## See Also

- [[cellular-sheaf]] — the structure that defines the sheaf Laplacian
- [[ssf|SSF]] — the framework applying sheaf Laplacian spectral filtering to traffic forecasting
- [[over-smoothing-in-gnns]] — the problem addressed by sheaf Laplacian's richer diffusion

[^src-ssf]: [[source-ssf]]
