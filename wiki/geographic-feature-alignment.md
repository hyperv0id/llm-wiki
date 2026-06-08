---
title: "Geographic Feature Alignment"
type: technique
tags:
  - domain-adaptation
  - optimal-transport
  - geographic-representation
  - cross-city-transfer
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: high
status: active
---

# Geographic Feature Alignment (GFA)

**Geographic Feature Alignment (GFA)** is a lightweight plug-in module proposed in [[craft|CRAFT]] (NeurIPS 2025) for enabling cross-city transfer in traffic flow generation. It addresses the **domain shift** problem: regions with similar geographic features (POIs, roads, population) exhibit very different traffic flow patterns across different cities[^src-craft].

## Motivation

Within a single city, geographic features and traffic flow patterns are well-correlated — dense commercial areas consistently generate high traffic. However, in cross-city settings, this correspondence breaks down[^src-craft]. The same TF-IDF POI vector may indicate a busy downtown in one city but a quiet suburb in another, due to differences in urban planning, culture, and transportation infrastructure.

GFA solves this by learning **cross-city transferable geographic representations** that align with actual flow patterns and are invariant to city-specific distribution shifts[^src-craft].

## Components

### Basic Geographic Representation

Three publicly available features are extracted for each urban region[^src-craft]:

- **POIs**: TF-IDF weighted category counts, treating POI categories as "words" and regions as "documents" to normalize across imbalanced categories[^src-craft]
- **Roads**: Total length of road segments across all categories — captures transportation capacity[^src-craft]
- **Population**: UN-adjusted 100m resolution data from WorldPop — reflects traffic potential[^src-craft]

These are concatenated and projected through an MLP: $z_i = \text{MLP}(f_i^{(poi)} \| f_i^{(road)} \| f_i^{(pop)})$

### Spatial Encoder

A Graph Transformer takes region graph $G$ and basic representations $\{z_i\}$ to produce higher-level representations $\{h_i\}$ that model inter-region spatial correlations[^src-craft].

### Traffic Flow Alignment (TFA)

Enforces that the **distance between region representations reflects the distance between their actual traffic flow patterns**[^src-craft]:

$$L_{FA} = \frac{1}{N_s^2} \sum_{i,j} \left( \hat{d}_{ij} - d_{ij} \right)^2$$

where $\hat{d}_{ij}$ is the normalized representation distance and $d_{ij}$ is the normalized average flow distance. Only source city regions are used (target cities have no flow data)[^src-craft].

### Cross-City Alignment (CCA)

Formulated as an **optimal transport (OT) problem**: find the minimum-cost mass transport between source city region representations and target city region representations[^src-craft]:

$$L_{CA} = \sum_{ij} T_{ij} \cdot D_{ij}$$

where $D_{ij}$ is the Euclidean distance between source region $i$ and target region $j$, and $T = \text{OTSolver}(D)$ is the OT transport matrix. This aligns regions across cities without requiring explicit correspondence labels[^src-craft].

### Total Loss

$L_A = \lambda_1 L_{FA} + \lambda_2 L_{CA}$ — TFA provides direction (flow pattern alignment), CCA enables cross-city transfer (distribution alignment)[^src-craft].

## Key Findings

- **GFA is the most critical component** in CRAFT — removing it causes the largest performance drop, confirming domain shift as the primary cross-city bottleneck[^src-craft]
- **TFA alone**: groups regions with similar flow patterns together (t-SNE analysis shows geographic features align with flow values)[^src-craft]
- **CCA alone**: reduces but does not eliminate distribution shift between source and target city representations[^src-craft]
- **TFA + CCA together**: achieves both flow-aligned and cross-city-aligned representations[^src-craft]

## Relationship to Other Techniques

- Unlike entity-embedding-based geographic representation learning (e.g., POI-Enhancer, GTG), GFA aligns representations through flow supervision and cross-city OT[^src-craft]
- Unlike standard domain adaptation (e.g., adversarial training), GFA uses OT-based alignment which provides a principled geometric solution[^src-craft]
- GFA is a **pre-training module**: trained once on both source and target geographic features, then frozen during diffusion backbone training[^src-craft]

## See Also

- [[craft]] — CRAFT model overview
- [[retrieval-based-condition-augmentation]] — RCA, complement to GFA
- [[cross-city-traffic-flow-generation]] — problem domain
- [[optimal-transport]] — optimal transport theory

[^src-craft]: [[source-craft]]
