---
title: "Learning to Rotate: Temporal and Semantic Rotary Encoding for Sequential Modeling"
authors: Hailing Cheng, Daqi Sun, Xinyu Lu
institution: LinkedIn Inc.
arXiv: 2604.24717v1
date: 2026-04-27
categories:
  - positional-encoding
  - rotary-position-embedding
  - temporal-modeling
  - sequential-recommendation
  - transformer
---

# Learning to Rotate: Temporal and Semantic Rotary Encoding for Sequential Modeling

## Abstract

Every Transformer architecture dedicates enormous capacity to learning rich representations in semantic embedding space—yet the rotation manifold acted upon by Rotary Positional Embeddings (RoPE) has been treated as a fixed, hand-crafted structure, populated only by discrete ordinal indices. We argue that this rotation space is a largely overlooked second dimension of expressivity in the attention mechanism, one whose systematic exploration may open a new door for attention-based architectures. The analogy to complex numbers is instructive: just as introducing the imaginary axis—orthogonal to and independent of the real line—unlocked new algebraic structure once believed impossible, treating the rotation manifold as a learnable, signal-conditioned space opens an orthogonal degree of freedom in attention. In this framing, the token embedding encodes the semantic (real) component of a representation—what a token means—while the rotation encodes its dynamic (imaginary) component—how it relates to every other token across time, position, and context.

We introduce SIREN-RoPE, a concrete instantiation of this idea, which populates the rotation dimension with heterogeneous signals—continuous timestamps, cyclical temporal patterns, and categorical metadata—via a dual-branch Sinusoidal Representation Network (SIREN). As a proof of concept, we evaluate on a production-scale news feed dataset from a major social network using a generative recommender as the ranking model, demonstrating that activating this hidden dimension yields consistent improvements across calibration and ranking objectives with negligible computational overhead. We invite the community to view the rotation space not as a solved positional-encoding detail, but as an untapped axis whose rich structure may prove as consequential for attention as the imaginary unit proved for algebra. Demo code is available at https://github.com/hailingc/siren_rope.

## 1 Introduction

Rotary Position Embeddings (RoPE) in Large Language Models (LLMs) encode sequence order as planar rotations of query/key subspaces, yielding strong relative-position modeling. Yet the rotation manifold itself has always been a fixed, hand-crafted structure driven by discrete ordinal indices. In sequential recommendation and event-stream modeling, this is a poor proxy: an interaction seven days ago is qualitatively different from one seven minutes ago, even at the same ordinal offset.

We argue this points to a deeper opportunity. Just as extending the real line to the complex plane—by adding an orthogonal imaginary axis—unlocked entirely new algebraic structure, treating the RoPE rotation manifold as a learnable, signal-conditioned space opens an orthogonal expressive dimension in attention. Token embeddings encode the semantic (real) component—what a token means; the rotation encodes the dynamic (imaginary) component—how it relates to every other token across time and context. This dimension has been almost entirely unexplored.

### 1.1 Contributions

We introduce SIREN-RoPE, which treats the rotation manifold as a first-class learnable space:

- **Temporal rotation**: A dual-branch SIREN–DNN network maps continuous timestamps into per-dimension rotation angles, capturing both periodic (daily/weekly) and aperiodic (recency decay) temporal structure directly in the geometry of attention.
- **Adaptive frequency learning**: Per-dimension frequency scales and an ordinal-gate scalar λ are jointly learned, replacing hand-crafted inverse-frequency constants.
- **Empirical validation**: On a production-scale social feed dataset, SIREN-RoPE consistently improves calibration (NE) and ranking (AUC) across three engagement tasks with ∼0.2% extra parameters. Ablations confirm the rotation dimension carries signal complementary to and independent of the embedding space.

## 2 Related Work

**Positional Encodings and RoPE**: Early Transformers inject order via fixed sinusoidal or learned absolute embeddings; subsequent work shifted to relative formulations for better length generalization. RoPE encodes position as planar rotations in query-key subspaces, expressing relative offsets as phase differences while preserving vector norms. Extensions such as YaRN and LongRoPE focus on length extrapolation via interpolation. All of these treat the rotation manifold as a fixed positional scaffold; we instead treat it as a learnable, signal-conditioned space.

**Temporal Dynamics in Sequential Models**: Methods like Time2Vec and TiSASRec incorporate pairwise time intervals and periodic embeddings to capture non-linear temporal dynamics. DeBERTa and TUPE disentangle content from positional context in language modeling. Most closely related to our work, TO-RoPE applies the same inverse-frequency schedule as standard RoPE but substitutes the raw timestamp Ti for the ordinal index. SIREN-RoPE generalizes this idea by replacing the fixed inverse-frequency mapping with a dual-branch SIREN-DNN network.

**Implicit Neural Representations and Continuous-Time Models**: SIRENs use periodic activations with principled initialization to overcome spectral bias, enabling multi-scale periodic function learning. Neural ODEs/CDEs and hybrid models like ContiFormer handle continuous-time sequences at significant inference cost. SIREN-RoPE injects continuous-time inductive bias directly into the rotary angle, achieving comparable expressivity at the cost of a small MLP.

## 3 SIREN-RoPE: Unified Rotary Encoding

### 3.1 Problem Formulation

Let S = {(ei, Ti)}i=1^C denote a user interaction sequence of length C, where ei ∈ R^d is the item embedding and Ti ∈ R represents the interaction timestamp. Conventional RoPE modulates the inner product between query and key representations using rotations derived from discrete ordinal indices pi. While the attention weights are primarily driven by the semantic content of ei, this formulation constrains the relative positional dependency to be a strictly periodic function of the ordinal displacement |pi - pj|. Consequently, it neglects the irregular temporal intervals ΔT = Ti - Tj that often characterize real-world user behavior.

In social-feed recommendation, treating ordinal position as a proxy for time is fundamentally misaligned with empirical user behavior. An interaction at 08:00 on a Monday is qualitatively distinct from one at 22:00 on a Saturday, even when they share the same ordinal displacement. Human behavior is governed by strong periodicities—24-hour diurnal cycles and 7-day weekly rhythms—so user intent and content relevance are tied to absolute temporal coordinates, a structure that ordinal indices alone cannot capture.

We therefore seek a rotation angle Θj(Ti, pi) satisfying three criteria:
1. Temporal richness: captures multi-scale cyclical patterns from Ti
2. Ordinal preservation: retains the monotone recency-decay and translational equivariance properties of standard RoPE
3. Adaptive balance: allows end-to-end gradient descent to determine how much each component contributes

### 3.2 Ordinal-Temporal Fusion

SIREN-RoPE replaces the fixed angle pi·θj with:

Θj(Ti, pi) = fϕ(Ti)j · ωjs + pi · θj · λ

where:
- i ∈ {0, ..., C - 1} is the item's ordinal index in the sequence
- pi = i is the ordinal position of item i
- fϕ: Rdt → Rdk/2 is the dual-branch SIREN network, mapping dt-dimensional timestamp features to rotation angles
- ωjs ∈ Rdk/2 are learnable per-dimension frequency scalings
- θj = base^(-2j/dk) are the fixed inverse frequencies used in standard RoPE
- λ ∈ R is a learnable scalar gate (initialized to 1.0) controlling the ordinal contribution

The rotation is then applied identically to standard RoPE with Θj substituted for pθj.

**Initialization and convergence**: At initialization (ωjs = π, λ = 1.0), the model begins close to a scaled form of standard RoPE augmented by a SIREN offset. As training progresses, ωjs and λ learn to balance temporal and ordinal contributions. In practice, we observe: without temporal signals, λ remains near 1.0 throughout training. Upon introducing SIREN-RoPE, λ drops to 0.044 at convergence, indicating that the model has learned to rely predominantly on temporal modulation over ordinal position.

### 3.3 Dual-Branch SIREN Architecture

fϕ additively combines a periodic SIREN branch and an aperiodic DNN branch:

fϕ(T) = fsin(T) + fDNN(T)

- **Periodic branch** fsin: Uses SIREN architecture where each hidden layer computes sin(ω0Wx + b). The sine activations enable the branch to autonomously discover temporal periodicities beyond manually specified daily/weekly cycles.
- **Aperiodic branch** fDNN: Uses standard ReLU activations to capture monotone trends such as content-recency decay.

### 3.4 Temporal Input Features

The input to fϕ is a 5-dimensional decomposition of the raw Unix timestamp T:

t(T) = [cos(2πT/τd), sin(2πT/τd), cos(2πT/τw), sin(2πT/τw), T̃]

where τd = 86,400 s (daily periodicity), τw = 604,800 s (weekly periodicity), and T̃ is a normalized long-range offset.

## 4 Experiments

### 4.1 Experimental Setup

**Dataset**: Production-scale dataset from the news feed of a major social network, comprising user-item interaction events from one year of production logs. Each user's history is represented as an interleaved sequence [e1, a1, e2, a2, ..., eC, aC]. Tasks: Contribution (like, comment or share), Like, LongDwell (sustained dwell-time).

**Baselines**:
- Ordinal RoPE: Standard RoPE with discrete ordinal position indices
- Timestamp-as-Feature: Standard ordinal RoPE, but timestamp features appended to item sequence features
- TO-RoPE: Applies same inverse-frequency schedule but uses normalized timestamp as index

**Implementation**: 12 Transformer layers, d=512 hidden dimensions, 4 attention heads, max sequence length C=1024. SIREN-RoPE uses 2-layer SIREN branch and 2-layer DNN branch (each 64 hidden units), adding ~0.2% extra parameters.

### 4.2 Main Results

| Model | Contribution NE | Contribution AUC | Like NE | Like AUC | LongDwell NE | LongDwell AUC |
|-------|-----------------|------------------|---------|----------|--------------|---------------|
| Ordinal RoPE | 0.6206 | 0.9102 | 0.5985 | 0.9238 | 0.8362 | 0.7597 |
| Timestamp-as-Feature | 0.6218 | 0.9098 | 0.5997 | 0.9233 | 0.8350 | 0.7603 |
| TO-RoPE | 0.6218 | 0.9095 | 0.5999 | 0.9231 | 0.8349 | 0.7613 |
| **SIREN-RoPE** | **0.6182** | **0.9115** | **0.5963** | **0.9249** | **0.8334** | **0.7633** |

SIREN-RoPE improves across all six metric-task combinations.

### 4.3 Ablation Study

Key findings:
1. Cyclical features (cos, sin) improve calibration vs scalar time_in_year
2. DNN-only with full temporal achieves best NE, but SIREN branch needed for discovering hidden periodicities
3. Semantic rotation (static signals like in/out-network) performs no better than ordinal RoPE

## 5 Discussion

The rotation manifold is a structured expressive channel rather than a mere positional scaffold. Table 3 confirms: routing temporal features through the rotation manifold yields consistent gains, while injecting them into embeddings is ineffective or detrimental. This identifies rotation as a qualitatively distinct representation mode.

## 6 Conclusion

We presented SIREN-RoPE, which populates the RoPE rotation manifold with temporal signals via a dual-branch SIREN-DNN network fused with ordinal position through a learnable gate. The broader argument is that the rotation space is an underexplored second dimension of attention expressivity—the dynamic (imaginary) complement to the semantic (real) embedding space. Experiments confirm consistent gains at negligible computational cost.

**Future work**: (1) theoretical characterization of expressivity; (2) conditioning fϕ on token-type or categorical features; (3) adapting to cross-attention; (4) unifying positional encoding across modalities.