---
title: "SNR-based Modality Selection"
type: concept
tags:
  - multi-modality
  - modality-selection
  - signal-to-noise-ratio
created: 2026-05-03
last_updated: 2026-05-03
source_count: 1
confidence: medium
status: active
---

# SNR-based Modality Selection

SNR-based modality selection is a strategy for multi-modal spatio-temporal prediction that uses the signal-to-noise ratio (SNR) of each modality to determine its contribution weight, rather than treating all modalities equally or relying on manual selection[^src-most].

## Core Idea

Different modalities contribute different amounts of useful signal vs. noise to a prediction task. Rather than fusing all modalities uniformly, SNR-based selection adaptively weights modalities based on their estimated information quality for the current input.

## Related

- [[multi-modality-refinement]] — MoST's implementation of SNR-based modality selection
- [[multi-modality-guided-spatial-expert]] — alternative modality handling via MoE routing
- [[most]] — MoST foundation model that introduced this approach

[^src-most]: [[source-most]]
