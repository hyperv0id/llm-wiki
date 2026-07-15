---
title: "EAC (Expand and Compress)"
type: entity
tags:
  - continual-learning
  - spatio-temporal
  - prompt-learning
  - traffic-forecasting
  - stgnn
  - iclr-2025
created: 2026-07-18
last_updated: 2026-07-18
source_count: 1
confidence: medium
status: active
---

# EAC (Expand and Compress)

**EAC** (Expand and Compress) is a prompt-based continual spatio-temporal graph forecasting framework proposed by Wei Chen and Yuxuan Liang (HKUST-GZ), published at ICLR 2025[^src-eac]. It addresses the dual challenges of **catastrophic forgetting** and **retraining inefficiency** in streaming spatio-temporal scenarios where the underlying graph continuously expands with new sensors.

## Core Idea

EAC freezes a base STGNN backbone after initial joint training with a **prompt parameter pool** — node-level learnable parameters added element-wise to input features. In subsequent periods, only the prompt pool is tuned; the backbone never changes, completely avoiding catastrophic forgetting. The prompt pool expands for newly added nodes and is compressed via low-rank approximation to control parameter inflation[^src-eac].

## Two Tuning Principles

### Principle I: Expand (Heterogeneity-Guided)

**Insight**: Node-customized prompt parameters increase the feature space's ability to express heterogeneity. The paper formalizes this via the Average Node Deviation metric D(X), proving that D(X^θ) − D(X) ≥ 0 for any prompt parameter set — the prompt pool strictly expands representational capacity[^src-eac].

**Implementation**: For each new period τ, a prompt parameter matrix A^(τ) is created for newly added nodes and appended to the pool P = [P^(1), ..., P^(τ)]. The backbone is frozen, and only P is optimized on current-period data[^src-eac].

### Principle II: Compress (Low-Rank-Guided)

**Insight**: Spectral analysis reveals the prompt pool exhibits a strong low-rank property: >75% of singular value mass is concentrated in the first few components across all periods. Proposition 2 proves that with high probability, P can be approximated as AB with k = O(log(min(n, d)))[^src-eac].

**Implementation**: Instead of maintaining full prompt matrices P^(τ) ∈ R^(n×d), EAC decomposes them as A^(τ)B where A^(τ) ∈ R^(n×k) and B ∈ R^(k×d) with k ≪ d. B is shared across all periods; only A^(τ) grows per node. With k=6, EAC uses only ~59% of tuning parameters vs full prompt, with minimal performance loss[^src-eac].

## Workflow

1. **Period 1**: Construct initial prompt pool P via A^(1)B (compress), element-wise add to X₁, jointly train backbone f_θ and P[^src-eac].
2. **Period τ > 1**: Reload frozen backbone f_θ* and pool P. Detect new nodes, construct A^(τ), append A^(τ)B to pool (expand). Train only P on X_τ[^src-eac].
3. **Prediction**: Use f_θ* with current prompt pool for forecasting[^src-eac].

## Performance

On PEMS-Stream (7 periods, 655→871 nodes), EAC achieves avg MAE 13.53 (±0.06), RMSE 21.77 — a 3.90% MAE reduction over the second-best method (Online-ST-AN). On Air-Stream (4 periods, 1087→1202 nodes): avg MAE 20.75 (−1.75%). On Energy-Stream (4 periods, 103→134 nodes): avg MAE 5.10 (−4.85%)[^src-eac].

**Efficiency**: Freezing the backbone accelerates training 1.26–3.02× on Energy-Stream. With k=2 (EAC-Efficient), only ~33% of baseline parameters while maintaining competitive performance[^src-eac].

**Few-shot**: With 20% training data per period, EAC shows the mildest performance decline among all methods, particularly strong at 12-step horizons[^src-eac].

## Universality

EAC consistently improves performance across all six STGNN architecture combinations (spectral/spatial graph convolutions × recurrent/convolution/attention sequence operators). Recurrent-based sequence operators benefit most; attention-based methods show the smallest but still positive gains[^src-eac].

## Comparison to Related Methods

| Method | Core Mechanism | Backbone | Tuning Scope |
|--------|---------------|----------|-------------|
| TrafficStream | Replay + regularization | Full STGNN | All parameters |
| STKEC | Pattern bank + expansion | Full STGNN | All parameters |
| PECPM | Pattern bank + consolidation | Full STGNN | All parameters |
| TFMoE | Mixture of experts | Full STGNN | All parameters |
| **EAC** | **Prompt pool + expand/compress** | **Frozen STGNN** | **Prompt pool only** |
| [[stbp|STBP]] | Contextual pattern bank (pure expansion) | Frozen backbone | Pattern bank only |

EAC is the first CSTF method to freeze the entire backbone, a design later adopted and refined by [[stbp|STBP]][^src-eac].

## Significance

EAC is authored by the same HKUST-GZ group (Yuxuan Liang as corresponding author) that later produced [[urbanfm|UrbanFM]] and [[factost|FactoST]], establishing a research lineage from prompt-based continual adaptation toward spatio-temporal foundation models. The paper explicitly frames its expand-and-compress principles as a first step toward large-scale spatio-temporal pre-training paradigms[^src-eac].

[^src-eac]: [[source-eac]]
