---
title: "Dual-View Alignment Loss"
type: technique
tags:
  - loss-function
  - representation-learning
  - temporal-alignment
  - traffic-forecasting
created: 2026-04-27
last_updated: 2026-04-27
source_count: 1
confidence: high
status: active
---

# Dual-View Alignment (DVA) Loss

The Dual-View Alignment (DVA) loss is a regularization component in [[hyperd|HyperD]] that aligns representations across the two periodicity views (short-term and long-term) to ensure they produce **consistent yet complementary** representations[^src-hyperd-hybrid-periodicity-decoupling].

## Formulation

The total training loss combines prediction error with DVA[^src-hyperd-hybrid-periodicity-decoupling]:

**L = L_mae(Ŷ, Y) + λ · L_DVA**

Where L_DVA has two terms:

### Temporal Alignment (L_temporal)
Aligns the short-term representation Z_s and long-term representation Z_l using **Soft-DTW (Differentiable Dynamic Time Warping)**. This promotes consistency in how the two encoders represent the same underlying traffic state at different time scales[^src-hyperd-hybrid-periodicity-decoupling].

### Periodic Alignment (L_periodic)
Aligns the short-term and long-term periodicity embeddings H_s and H_l in the frequency domain using **cosine similarity**. This ensures the embeddings capture complementary rather than redundant periodic information[^src-hyperd-hybrid-periodicity-decoupling].

## Why Alignment is Necessary

Without explicit alignment, the dual-pathway [[spatial-temporal-attentive-encoder|STAE]] encoder may learn representations that contradict each other for the same time step, degrading overall prediction accuracy. The DVA loss acts as a regularizer that encourages the two pathways to produce representations that are **consistent** (capturing the same underlying ground truth) while **complementary** (emphasizing different temporal scales)[^src-hyperd-hybrid-periodicity-decoupling].

## Impact

Ablation studies show removing DVA loss increases MAE by +1.5%[^src-hyperd-hybrid-periodicity-decoupling].

[^src-hyperd-hybrid-periodicity-decoupling]: [[source-hyperd-hybrid-periodicity-decoupling]]
