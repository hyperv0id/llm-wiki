---
title: "FactoST"
type: source-summary
tags:
  - foundation-model
  - spatio-temporal
  - transformer
  - factorization
  - pretraining
  - quantile-regression
  - arxiv-2026
created: 2026-06-01
last_updated: 2026-06-01
source_count: 0
confidence: high
status: active
---

# FactoST: Learning to Factorize and Adapt

**Source**: Siru Zhong, Junjie Qiu, Yangyu Wu, Yiqiu Liu, Yuanpeng He, Zhongwen Rao, Bin Yang, Chenjuan Guo, Hao Xu, Yuxuan Liang. "Learning to Factorize and Adapt: A Versatile Approach Toward Universal Spatio-Temporal Foundation Models." arXiv:2601.12083, Jan 2026. Journal extension of NeurIPS 2025 conference version.

**Affiliations**: HKUST(GZ), Peking University, Huawei 2012 Laboratories, East China Normal University. Corresponding author: Yuxuan Liang.

## Core Argument

FactoST-v2 proposes the **Pattern Factorization Hypothesis**: effective spatio-temporal (ST) generalization requires decoupling domain-invariant temporal dynamics from domain-specific spatial contexts. Unlike existing ST Foundation Models (STFMs) that jointly pretrain spatial and temporal patterns — incurring $O(N^2)$ complexity and risking negative transfer from conflicting topologies — FactoST-v2 factorizes pretraining into two lightweight stages.

## Method

**Stage I — Universal Temporal Pretraining (UTP)**: A minimalist encoder-only Transformer backbone learns general temporal patterns from 11B+ time points across 8 domains (traffic, energy, weather, transport, economics, web, healthcare, plus KernelSynth synthetic data). Key innovations: (1) Random Sequence Masking with a learnable `[REG]` token allows arbitrary-length input-output mapping without architectural changes; (2) Partial Rotary Positional Embedding (p-RoPE) applies rotary operations only to high-frequency components, preserving trend semantics; (3) Gated Attention filters noise and eliminates attention sinks; (4) Multi-Quantile Prediction Head (pinball loss) models full conditional distributions instead of point estimates.

**Stage II — Spatio-Temporal Adaptation (STA)**: A lightweight adapter (parameter count ≪ backbone) injects spatial awareness into the frozen UTP backbone via four modules: (1) ST Metadata Fusion (STMF) — injects node-specific and calendar-aware embeddings; (2) ST Filtering (STF) — dynamically reweights spatial/temporal/time-lagged affinities via learned scalar gates; (3) Domain-Specific Prompt Alignment (DSPA) — low-rank learnable prompt tokens align pretraining-to-target distribution shifts; (4) Continual Memory Replay (CMR) — mixes current data with historical buffer samples to prevent catastrophic forgetting.

**v2 Upgrades from NeurIPS 2025 v1**: Encoder-only (was encoder-decoder, enabling 100% weight transfer), probabilistic quantile (was deterministic point prediction), streamlined DSPA (replaces complex hierarchical alignment), arbitrary-length generalization (was fixed horizon).

## Key Findings

- **Few-shot**: SOTA on all 9 benchmarks (PEMS03/04/07/08, PEMS-BAY, METR-LA, ETTh2, ECL, Weather), both short (12→12) and long (96→96) horizons
- **Full-shot**: Only model that avoids OOM on long-horizon full-shot training (GWNet/D2STGNN all OOM); SOTA across domains
- **Zero-shot**: Surpasses TimesFM, Moirai, Rose, OpenCity, and UniST on all datasets
- **Scaling**: 10% labeled data achieves near-full-shot performance (MAE gap <1 on short-term); model depth improves zero-shot but not few-shot
- **Ablation**: Random Sequence Masking most critical (+17.7% MAE if removed), followed by Quantile Loss (+7.0%), Gated Attention (+4.0%), p-RoPE (+0.6%)
- **Efficiency**: Tiny variant (4.4M params, 11.0s inference) — Pareto-optimal accuracy vs. latency vs. parameter trade-off

## Significance

FactoST-v2 redefines the "STGNN in the era of foundation models" by proving that architectural disentanglement trumps brute-force joint modeling. It establishes a theoretical generalization bound argument: factorization yields $C(H_{adapt}) \ll C(H_{joint})$, implying tighter guarantees across all data regimes. The UTP backbone is graph-agnostic and fully reusable — STA is architecturally agnostic and can be plugged into any temporal backbone (demonstrated with PatchTST+STA).

## Limitations

Current STA relies on learned node embeddings (transductive), not fully inductive for open-world topologies. Multi-modal exogenous factors (events, text) not integrated. Coarse temporal granularity (week/month) degrades performance by ~18% vs. fine-grained (minute/hour/day-of-week).
