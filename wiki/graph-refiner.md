---
title: "Graph Refiner"
type: technique
tags:
  - graph-neural-network
  - gcn
  - exogenous
  - time-series-forecasting
  - sparsification
created: 2026-07-13
last_updated: 2026-07-13
source_count: 1
confidence: high
status: active
---

# Graph Refiner

The **Graph Refiner** is the prediction head stage of [[gcgnet|GCGNet]]. It uses the generated adjacency \(\hat A\) from the [[graph-structure-aligner|Graph Structure Aligner]] to refine coarse patch features and produce the final endogenous forecast, while **preventing Graph VAE degeneration**.[^src-gcgnet]

## Mechanism

Inputs: generated patch embeddings \(\tilde S^p\) (nodes) and adjacency \(\hat A\) (edges).[^src-gcgnet]

1. **Sparsify**: top-\(k\) edge selection per node (following TimeFilter-style practice) → sparse \(A_s\), dropping weak noisy links.
2. **GCN stack**: \(H=\mathrm{GCN}(\tilde S^p,A_s)\), multi-hop aggregation over joint temporal–channel neighborhoods.
3. **Head**: \(\hat Y^{\mathrm{endo}}=\mathrm{Linear}(\mathrm{Flatten}(H))\).

Because \(\hat A\) enters the supervised forecasting path, a degenerate Graph VAE that ignores inputs would increase \(L_f\); the refiner therefore forces informative graphs *and* improves accuracy via message passing.[^src-gcgnet]

## Ablation evidence

Removing the Graph Refiner (output only from Variational Generator under aligner guidance) causes the largest average degradation among GCGNet ablations (e.g., average MSE rising substantially on NP/PJM/DE/Energy in Table 2).[^src-gcgnet]

## Links

- Entity: [[gcgnet]]
- Source: [[source-gcgnet]]
- Related: [[graph-structure-aligner]], [[joint-temporal-channel-correlation]], [[variational-generator-exogenous]]

---

[^src-gcgnet]: [[source-gcgnet]]
