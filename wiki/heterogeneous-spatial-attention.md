---
title: "Heterogeneous Spatial Attention (HSA)"
type: technique
tags:
  - attention
  - spatial-modeling
  - tensor-decomposition
  - gated-fusion
  - traffic-forecasting
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: medium
status: active
---

# Heterogeneous Spatial Attention (HSA)

Heterogeneous Spatial Attention (HSA) is the spatial modeling component of [[hephestus|HEPHAESTUS]], designed to balance shared global structural patterns with node-specific behaviors in traffic networks at low parametric cost[^src-hephestus].

## Mechanism

### Query Construction

Queries combine learned node embeddings Se ∈ R^(N×r) with input features[^src-hephestus]:

1. Q' = H·Wqs1 (linear projection of input)
2. Qs = [S̃e ∥ Q']·Wqs2 (concat with broadcasted node embeddings, then project)

This injects node identity into the query, enabling the attention mechanism to condition on which sensor is being attended to.

### Dual Value Representations

HSA produces two complementary value projections[^src-hephestus]:

**Common Linear (Global Pattern)**
```
Vsc = H·Wvc
```
A single shared weight matrix Wvc ∈ R^(C×D) applied uniformly to all nodes, capturing shared global structure[^src-hephestus].

**Specific Linear (Node-Specific via Low-Rank Decomposition)**
A learned pattern library PL ∈ R^(r×C×D) combined with node embeddings Se via tensor contraction[^src-hephestus]:
```
Wvs^(i) = Σ_k Se[i,k]·PL[k,:,:]   (for node i)
Vss^(i) = H[i,:,:]·Wvs^(i)
```
This decomposes O(N×C×D) parameters (direct per-node weight assignment) into O(N×r + r×C×D), achieving efficiency while preserving expressiveness. The pattern library PL acts as a learned basis, and node embeddings Se select how each node combines these basis patterns[^src-hephestus].

### Gated Fusion

Dynamic fusion of common and specific values[^src-hephestus]:
```
λ = σ([Vsc ∥ Vss]·Wg)                Wg ∈ R^(2D×1)
Vs = Λ ⊙ Vsc + (1−Λ) ⊙ Vss
```
The sigmoid gate λ ∈ [0,1] adaptively balances global vs local contributions per node and timestep.

### Key and Attention

Key Ks = H·Wks (standard projection). Spatial cross-attention computed along the spatial axis: As = Softmax(Qs·Ks^T / √D) with As ∈ R^(H×N×N), then Zs = As·Vs. Multi-head, residual, and layer norm follow standard practice[^src-hephestus].

## Design Rationale

HSA addresses a fundamental problem in traffic forecasting: different road segments exhibit qualitatively different traffic patterns (e.g., highways vs. residential streets), but most spatial attention methods either impose a uniform transformation (ignoring heterogeneity) or incur prohibitive per-node parameter costs. HSA's low-rank decomposition (pattern library PL with rank r=8, optimal from parameter sensitivity analysis) strikes a principled balance[^src-hephestus].

## Ablation Impact

Removing HSA degrades performance across all metrics[^src-hephestus]:
- METR-LA: MAE 3.36→3.48 (+3.6%), MAPE 9.76%→10.15%
- PEMS08: MAE 13.56→14.12 (+4.1%)

The gated fusion between common and specific values is essential — removing it (using only shared projections, w/o HSA baseline) causes the largest spatial-related degradation[^src-hephestus].

## Relationship to Other Approaches

| Method | Spatial design |
|--------|---------------|
| [[gwnet|GWNet]] | Self-adaptive adjacency (E1·E2^T) — uniform transformation, no per-node specificity |
| [[dgcrn|DGCRN]] | Dynamic adjacency per timestep — high parameter cost |
| [[astgcn|ASTGCN]] | Spatial attention + temporal attention, but uniform spatial transformation |
| [[hephestus|HSA]] | Low-rank pattern library + gated fusion — explicit per-node with low cost |

[^src-hephestus]: [[source-hephestus]]
