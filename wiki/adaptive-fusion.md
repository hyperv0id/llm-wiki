---
title: "Adaptive Fusion (SRS)"
type: technique
tags:
  - time-series
  - forecasting
  - patch
  - embedding-fusion
  - neurips-2025
created: 2026-07-13
last_updated: 2026-07-21
source_count: 1
confidence: medium
status: active
---

# Adaptive Fusion (SRS)

**Adaptive Fusion** is the embedding-stage operator of the [[selective-representation-space|SRS]] module (Wu et al., NeurIPS 2025). It **convex-combines** patch embeddings from conventional adjacent patching and from the selectively reassembled patches so the two spaces complement each other.[^src-srsnet]

## Method

Let $P$ be adjacent patches and $\tilde{P}$ the patches after [[selective-patching|Selective Patching]] + [[dynamic-reassembly|Dynamic Reassembly]]. Two linear patch projections produce $E^{c}$ and $E^{s}$; fusion is[^src-srsnet]

$$
\tilde{E}=\alpha\odot E^{c}+(1-\alpha)\odot E^{s},\quad \alpha\in[0,1]^{n\times d}
$$

(broadcast over channels). Positional embeddings (e.g., sinusoidal) are then added before the patch backbone. Parameterizing $\alpha$ in patch×feature space (not channel×…) keeps the design compatible with settings where channel count is ambiguous (e.g., pretraining).[^src-srsnet]

## Initialization note

The paper reports $\alpha$ initialization matters: higher $\alpha$ (favor conventional patches) for periodic/stationary data, lower for non-stationary/shifting series, improves convergence/stability even though random init can still work after a few epochs.[^src-srsnet]

## Ablations

Removing Adaptive Fusion degrades [[srsnet|SRSNet]] relative to the full module, though less than removing Selective Patching; fusion provides complementary enhancement rather than the primary space construction.[^src-srsnet]

## Connections

- Parent: [[selective-representation-space]]
- Inputs: [[selective-patching]], [[dynamic-reassembly]], [[patch-based-tokenization]]
- Used by: [[srsnet]]
- Source: [[source-srsnet]]

---

[^src-srsnet]: [[source-srsnet]]
