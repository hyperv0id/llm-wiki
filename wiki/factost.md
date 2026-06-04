---
title: "FactoST"
type: technique
tags:
  - foundation-model
  - spatio-temporal
  - transformer
  - factorization
  - pretraining
  - zero-shot
  - quantile-regression
  - scaling-laws
  - arxiv-2026
created: 2026-06-01
last_updated: 2026-06-01
source_count: 1
confidence: high
status: active
---

# FactoST

**FactoST** (Factorized Spatio-Temporal Foundation Model) is a two-stage factorized framework for universal spatio-temporal prediction, proposed by Siru Zhong, Junjie Qiu et al. at HKUST(GZ), with journal version (FactoST-v2) appearing as arXiv:2601.12083 (Jan 2026), extending the NeurIPS 2025 conference version[^src-factost]. Corresponding author Yuxuan Liang — the same HKUST(GZ) group behind [[urbanfm|UrbanFM]].

## Core Hypothesis

FactoST is built on the **Pattern Factorization Hypothesis**: effective spatio-temporal generalization requires decoupling **domain-invariant temporal dynamics** (trends, seasonality, periodicity — universal across domains) from **domain-specific spatial contexts** (road topology, power grid structure, sensor layout — unique per dataset)[^src-factost]. This contrasts sharply with existing ST Foundation Models ([[opencity|OpenCity]], [[unist|UniST]]) that jointly pretrain spatial and temporal patterns, incurring quadratic complexity $O(N^2T^2)$ and risking negative transfer from conflicting spatial topologies[^src-factost].

## Architecture: Two-Stage Factorization

### Stage I: Universal Temporal Pretraining (UTP)

A **graph-agnostic, encoder-only** Transformer backbone pretrained on 11B+ univariate time points across 8 domains (traffic, energy, weather, transport, economics, web, healthcare, + KernelSynth synthetic)[^src-factost]. Four key components:

1. **Random Sequence Masking + [REG] Token**: Dynamically samples variable context lengths from $L_{min}$ to $L_{max}$ during pretraining. A learnable `[REG]` token semantically separates historical context from future placeholders. This forces the backbone to learn a length-agnostic mapping — downstream shorter horizons use truncation, longer horizons use rolling prediction[^src-factost].

2. **Partial Rotary Position Embedding (p-RoPE)**: Decomposes embedding space into high-frequency (rotary-encoded for sequential order) and low-frequency (invariant, preserving trend semantics)[^src-factost]. Avoids the signal degradation of full RoPE while maintaining extrapolation.

3. **Gated Attention**: A gating score $G$ modulates attention output: $O = \text{Attention}(Q', K', V) \odot \sigma(G)$. Filters noise and eliminates attention sink problems[^src-factost].

4. **Multi-Quantile Prediction Head**: Predicts the full conditional distribution at multiple quantile levels (e.g., {0.1, 0.5, 0.9}) via Pinball Loss, enabling probabilistic forecasting with adaptive confidence intervals[^src-factost].

### Stage II: Spatio-Temporal Adaptation (STA)

A lightweight adapter (much fewer parameters than the backbone) injects spatial awareness. Four modules[^src-factost]:

| Module | Role | Mechanism |
|--------|------|-----------|
| **STMF** (ST Metadata Fusion) | Provides base ST context | Node-specific embeddings $E_n$ + calendar-aware cyclic embeddings (minute-of-hour, day-of-week, etc.) projected to hidden dim |
| **STF** (ST Filtering) | Dynamic context reweighting | Three learned affinities: spatial ($S_s$, node compatibility), temporal ($S_t$, pattern alignment), time-lagged ($S_d$, delayed causal effects via learnable prototypes). Softmax-gated fusion with Sigmoid |
| **DSPA** (Domain-Specific Prompt Alignment) | Distribution shift alignment | Low-rank ($r \ll d$) learnable prompt tokens $P = UV^T$ prepended to input; aligns pretraining corpora → target domain without fine-tuning backbone |
| **CMR** (Continual Memory Replay) | Catastrophic forgetting prevention | Memory buffer (20% of dataset) mixes historical samples with current stream during few-shot adaptation |

STA is architecturally agnostic — the authors demonstrate it can be plugged into any temporal backbone (e.g., PatchTST+STA yields consistent gains)[^src-factost].

## v1 → v2 Evolution

| Aspect | Conference (v1, NeurIPS 2025) | Journal (v2) |
|--------|-------------------------------|-------------|
| Backbone | Encoder-Decoder (fixed horizon) | Encoder-Only (arbitrary horizon) |
| Pretraining Objective | Hybrid (Reconstruction + Prediction) | Pure (Quantile Prediction) |
| Adaptation | Complex Hierarchical Alignment (HDA) | Streamlined DSPA (Prompt Alignment) |
| Uncertainty | Deterministic point estimate | Probabilistic quantile |
| Weight Transfer | Partial (Decoder Gap) | Full (100%) |
| Pretraining Scale | ~13M (Monash, 6 domains) | ~11B (8 domains + KernelSynth) |

The shift to encoder-only removes the decoder gap that prevented full parameter reuse in v1, while quantile prediction replaces the auxiliary reconstruction task that introduced optimization redundancy[^src-factost].

## Complexity: Linear vs. Quadratic

FactoST's factorized design achieves **linear** complexity $O(NP^2D + NMPD)$ per layer, compared to **quadratic** $O(N^2P^2D)$ for joint STFMs (where $N$ = nodes, $P$ = patches, $M \ll N$ = latent prototypes)[^src-factost]. This is theoretically grounded: the UTP processes node-wise series independently ($O(NP^2D)$), and the STA adapter adds only linear overhead. Joint models like GWNet and D2STGNN OOM on long-horizon full-shot training (>883 nodes × 96 steps) — FactoST-v2 does not[^src-factost].

## Theoretical Generalization Bound

FactoST-v2 decomposes the hypothesis $h = h_{adapt} \circ h_{time}$, yielding:

$$\sqrt{\frac{C(H_{adapt})}{|D_{tgt}|}} \ll \sqrt{\frac{C(H_{joint})}{|D_{tgt}|}}$$

The adapter's restricted hypothesis space $C(H_{adapt})$ is far smaller than the monolithic joint model's $C(H_{joint})$ — explaining why factorization generalizes better across all data regimes[^src-factost].

## Performance

### Few-Shot (10% labeled data)

SOTA on all 9 benchmarks (short 12→12 and long 96→96 horizons). v2 consistently outperforms v1 — e.g., PEMS-04 short MAE 22.61 vs 23.93 (v1), vs 23.27 (best expert GWNet)[^src-factost].

### Full-Shot (100% data)

SOTA across all datasets. GWNet/D2STGNN OOM on long-horizon; FactoST-v2 remains stable. On PEMS-04 long-horizon: v2 MAE=26.68 vs v1=42.04 (↓36.5%), vs PatchTST=53.14[^src-factost].

### Zero-Shot (no fine-tuning)

FactoST-v2 dominates all baselines (TimesFM, Moirai, Rose, OpenCity, UniST) on all datasets. Key insight: **coupled STFMs (OpenCity) frequently underperform purely temporal baselines (TimesFM) in zero-shot** due to negative transfer from rigid source-domain topologies — FactoST's decoupled design avoids this[^src-factost].

### Scaling Properties

- **Data scaling**: 10% labeled data already achieves near-full-shot MAE (gap <1 on short-term); the pretrained UTP backbone's temporal manifold enables rapid alignment[^src-factost]
- **Model scaling**: Zero-shot improves with model depth (Minuscule→Base: MAE 0.368→0.362 on ETTh2); few-shot is relatively insensitive to scale (adapter compensates)[^src-factost]
- **Ablation ranking** (zero-shot PEMS08, by impact): Random Sequence Mask (+17.7% MAE) >> Quantile Loss (+7.0%) > Gated Attn (+3.97%) > p-RoPE (+0.62%)[^src-factost]

### Efficiency

Tiny variant (4.4M params, 11.0s inference) in Pareto-optimal region — lower latency than D2STGNN (54.5s), OpenCity (25.3s), Moirai (22.2s)[^src-factost].

## Model Variants

| Variant | Parameters | $d_{model}$ | Layers | Heads |
|---------|-----------|-------------|--------|-------|
| Minuscule | 2.5M | 192 | 3 | 3 |
| **Tiny (default)** | **4.4M** | **256** | **3** | **4** |
| Small | 12.2M | 384 | 4 | 6 |
| Base | 30.4M | 512 | 6 | 8 |

## Place in the STFM Landscape

| Model | Paradigm | Complexity | Spatial Decoupling | Weight Transfer |
|-------|----------|-----------|-------------------|-----------------|
| **FactoST-v2** | **Factorized (UTP+STA)** | **$O(N)$ Linear** | **Full** | **100%** |
| [[opencity|OpenCity]] | Joint (Transformer+GNN) | $O(N^2)$ Quadratic | None | None |
| [[unist|UniST]] | Joint (MAE+Prompt) | $O(N^2)$ Quadratic | None | None |
| [[urbandit|UrbanDiT]] | Joint (DiT+Prompt) | $O(N^2)$ Quadratic | None | None |
| [[uniflow|UniFlow]] | Joint (Transformer+ST-MRA) | $O(N^2)$ Quadratic | None | None |
| [[urbanfm|UrbanFM]] | Joint (Factorized Attn+ST-RoPE) | $O(N)$ Factorized Attn | Partial (decomposed attn) | Full |

FactoST is the only STFM that achieves linear complexity through complete **stage-level** factorization — pretraining is entirely graph-free, and spatial modeling enters only through the lightweight adapter[^src-factost].

## Limitations

- **Transductive node embeddings**: STA uses learned $E_n$, requiring retraining for new nodes; not fully inductive for open-world topologies[^src-factost]
- **No exogenous modalities**: Current framework models only endogenous historical patterns; future work aims to integrate LLM-powered covariate encoders for event/text/weather signals[^src-factost]
- **Temporal granularity sensitivity**: Coarse features (week/month) degrade performance by ~18% vs. fine-grained (minute/hour/day-of-week); architecture inherently supports plug-and-play temporal feature swapping[^src-factost]

## Connections

- [[spatio-temporal-foundation-model]] — STFM concept; FactoST defines the factorized paradigm within this space
- [[traffic-forecasting]] — primary evaluation domain (PEMS family, METR-LA, PEMS-BAY)
- [[urbanfm]] — same HKUST(GZ) lab; contrasts with UrbanFM's scaling-centric factorized attention approach
- [[opencity]] — joint STFM baseline that FactoST surpasses in zero-shot
- [[unist]] — first one-for-all grid STFM, FactoST's key comparison target
- [[uniflow]] — unified grid+graph STFM; FactoST's factorization is complementary
- [[urbandit]] — DiT-based STFM; FactoST offers a non-diffusion alternative
- [[urbangpt]] — LLM-based approach; FactoST demonstrates pure-parameter approach
- [[bigcity]] — MTMD model; FactoST could serve as an alternative UTP backbone
- [[timesfm]] — pure TSFM, FactoST's temporal-only pretraining shares philosophy but adds STA
- [[spatio-temporal-foundation-model-landscape]] — 时空基础模型全景分析：解耦派代表的深层设计哲学
- [[source-factost]] — source summary page

[^src-factost]: [[source-factost]]
