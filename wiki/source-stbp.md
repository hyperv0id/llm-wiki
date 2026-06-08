---
title: "STBP — A General Spatio-Temporal Backbone with Scalable Contextual Pattern Bank for Urban Continual Forecasting"
type: source-summary
tags:
  - continual-learning
  - spatio-temporal
  - traffic-forecasting
  - pattern-bank
  - linear-attention
  - streaming-data
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: high
status: active
---

# STBP — A General Spatio-Temporal Backbone with Scalable Contextual Pattern Bank

**Authors:** Aoyu Liu, Yaying Zhang (Tongji University)  
**Venue:** ICLR 2026 (Poster)  
**Code:** <https://github.com/Aoyu-Liu/STBP>

## Summary

STBP proposes a novel framework for **continual spatio-temporal forecasting (CSTF)**, where a fixed, general-purpose spatio-temporal backbone is paired with a dynamically expanding **contextual pattern bank**. When new data arrives (e.g., new sensors added to a traffic network), the backbone is frozen—preventing catastrophic forgetting—while only the pattern bank expands and fine-tunes, enabling efficient adaptation to evolving distributions. This decoupling achieves SOTA on PEMS-Stream, CA-Stream, and AIR-Stream with 21.44%, 21.93%, and 2.35% MAE reduction over the best baseline (EAC), while keeping computational overhead minimal via linear attention and frequency-domain processing.

## Core Contributions

1. **Contextual Pattern Bank**: A purely parametric, incrementally expandable bank of trainable parameters ($\mathbf{P}_\tau \in \mathbb{R}^{N_\tau \times d}$) that autonomously distinguishes node heterogeneity and relevance through data-driven learning, without explicit clustering constraints. Three sub-components ($\mathbf{P}_\tau^{(0)}, \mathbf{P}_\tau^{(1)}, \mathbf{P}_\tau^{(2)}$) interact with the backbone via gating and attention to provide prompt-based guidance.

2. **General Spatio-Temporal Backbone**: Two core modules:
   - **FreNet (Frequency-Domain Network)**: Uses FFT to extract stable low-frequency components (periodicity, trends) while suppressing high-frequency noise, providing robustness to distributional drift.
   - **DLGA (Dual-Stream Linear Graph Attention)**: Random feature mapping-based linear attention reducing $O(N^2)$ to $O(N)$, with a dual-stream design that incorporates pattern bank parameters as additional keys for knowledge-aware spatial correlation modeling.

3. **Fixed Backbone + Expandable Bank Paradigm**: After initial joint training, the backbone is frozen permanently. The pattern bank expands via $\mathbf{P}'_\tau = \mathbf{P}_{\tau-1} \| \Delta\mathbf{P}_\tau$ and is the only component fine-tuned, enabling knowledge retention without historical data replay.

## Key Results

- PEMS-Stream: MAE 12.31 (EAC: 15.67, ↓21.44%)
- CA-Stream: MAE 15.77 (EAC: 20.20, ↓21.93%)  
- AIR-Stream: MAE 23.64 (EAC: 24.21, ↓2.35%)
- Few-shot (10% data): MAE 13.58 (PEMS) / 17.11 (CA) — both SOTA
- Efficiency: Near-EAC compute with substantially better accuracy; linear attention key to scalability
- Ablation: Removing backbone (replacing with CNN+GCN) causes significant degradation; removing DLGA equally harmful

## Limitations

- Single-task continual learning only; cross-domain generalization not yet validated
- Pattern bank is node-specific (transductive); entirely new sensor types may require different expansion strategies

## Related Work Referenced

TrafficStream, STKEC, PECPM, STRAP, EAC (CSTF baselines); GWNet, STID, iTransformer (conventional ST baselines); STID, STAEformer, HimNet (node-specific parameter learning); Katharopoulos et al. (linear attention); Peebles & Xie / Zhang et al. (prompt-based guidance).
