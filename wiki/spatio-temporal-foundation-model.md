---
title: "Spatio-Temporal Foundation Model"
type: concept
tags:
  - foundation-model
  - spatial-temporal
  - zero-shot
  - generalization
created: 2026-05-03
last_updated: 2026-06-04
source_count: 13
confidence: high
status: active
---

# Spatio-Temporal Foundation Model

A **spatio-temporal foundation model** is a large-scale pre-trained model designed for cross-city/cross-domain spatio-temporal prediction without requiring per-dataset training or fine-tuning[^src-most]. Unlike task-specific spatio-temporal models (e.g., [[stgcn|STGCN]], GWN) that are trained and evaluated on the same dataset, foundation models aim to capture universal spatio-temporal patterns transferable to unseen cities[^src-most].

## Motivation

Traditional spatio-temporal models face two deployment barriers[^src-most]:
1. **High cost**: Each new city requires collecting data, training a model from scratch, and tuning hyperparameters
2. **Poor generalization**: Models overfit to specific sensor topologies and fail on cities with different spatial configurations

Foundation models address both by pre-training once on diverse multi-city data and enabling zero-shot prediction on unseen cities[^src-most].

## Existing Models

### Pre-Foundation ST Pre-Training

Before the emergence of true foundation models, **[[gpt-st|GPT-ST]]** (NeurIPS 2023) pioneered the MAE pre-training paradigm for spatio-temporal graphs. It demonstrated that pre-trained representations can universally enhance diverse downstream STGNNs without architecture modification, but remained task-specific (per-dataset pre-training required)[^src-gpt-st]. [[std-mae|STD-MAE]] (IJCAI 2024) further developed this with spatial-temporal-decoupled masking[^src-2312-00516-std-mae].

### Single-Modal
- **[[opencity|OpenCity]]** (2024): Transformer + GNN architecture with instance normalization for zero-shot traffic prediction. Uses TimeShift Transformer (PTTM + DTP dual attention) to decouple periodic and dynamic patterns, and Laplacian eigenvectors for spatial context encoding. Pre-trained on 21 datasets (151M observations), achieves zero-shot performance surpassing full-shot baselines on 4/6 test datasets. Three scales: 2M (mini), 5M (base), 26M (plus)[^src-opencity].
- **[[urbanpg|UrbanPG]]** (AAAI 2026): Prompt-backbone decoupled architecture with STCA linear attention (O(N·d²)), unifying large-scale, few-shot, and continual learning under one framework. Freeze general backbone, fine-tune/expand only personalized context prompts. SOTA on CA 8600 nodes with 48-72% efficiency gains over PatchSTG[^src-urbanpg].
- **[[urbandit|UrbanDiT]]** (NeurIPS 2025): Diffusion Transformer (DiT) backbone with unified prompt learning (time/frequency/spatial memory pools). Unifies grid-based and graph-based spatio-temporal data, supports 5 tasks (forward/backward prediction, temporal interpolation, spatial extrapolation, spatio-temporal imputation). Uses rectified flow for 25× inference acceleration. Zero-shot outperforms nearly all trained baselines[^src-urbandit].
- **[[unist|UniST]]** (KDD 2024): First one-for-all grid-based spatio-temporal foundation model using MAE pre-training + knowledge-guided memory-based prompt learning. Single model covers 20+ datasets across 5 domains with zero-shot capability surpassing few-shot baselines[^src-unist]. Same Tsinghua FIB Lab as [[urbandit|UrbanDiT]].
- **[[uniflow|UniFlow]]** (arXiv 2024): First unified grid+graph foundation model using pure Transformer + ST-MRA (Spatio-Temporal Memory Retrieval Augmentation). Four structured learnable memory pools store shared spatio-temporal patterns for cross-learning. 9 datasets (6 grid + 3 graph), 9.1% avg RMSE improvement. Same Tsinghua FIB Lab as UrbanDiT and UniST[^src-uniflow].
- **[[urbanfm|UrbanFM]]** (arXiv 2026): First scaling-centric ST foundation model with WorldST (100+ cities, 1B+ data points), MiniST (KD-Tree clustering tokenization), and minimalist factorized attention architecture. Achieves 39-70.2% zero-shot improvement over existing foundation models, surpasses full-shot experts. Imputation capability without any imputation training[^src-urbanfm].
- **[[factost|FactoST]]** (NeurIPS 2025 / arXiv 2026): Factorized two-stage STFM — Universal Temporal Pretraining (UTP, encoder-only, 11B+ time points) + Spatio-Temporal Adaptation (STA, lightweight adapter). First STFM to achieve linear O(N) complexity through complete stage-level factorization. 4 model scales (2.5M-30.4M), SOTA on few-shot/full-shot/zero-shot across 9 benchmarks. Same HKUST-GZ group (Yuxuan Liang) as UrbanFM[^src-factost].
- **[[urbangpt|UrbanGPT]]** (KDD 2024): First spatio-temporal LLM using Vicuna-7b + instruction-tuning paradigm. Encoder uses multi-level gated dilated convolution (no graph), spatial reasoning delegated to LLM via textual POI descriptions. Processes one sensor at a time, making it computationally expensive (7B parameters, 174s inference)[^src-urbangpt].
- **Pangu-Weather / Fengwu**: Weather-specific foundation models on Euclidean grids[^src-most].

### Multi-Modal
- **[[most|MoST]]** (KDD 2026): First multi-modality spatio-temporal foundation model. Supports satellite imagery, POI text, location, and time series as input modalities with adaptive SNR-based selection[^src-most].
- **[[allspark|AllSpark]]** (Shao et al., 2024): Unifies **10 spatio-temporal modalities** (1D: language/code/table, 2D: RGB/SAR/MSI/HSI/graph/trajectory, 3D: point cloud) via the [[language-as-reference-framework|LaRF]] principle, using language as the universal alignment anchor[^src-allspark]. Extends beyond traffic to remote sensing and general geospatial intelligence.
- **E²-CSTP** (NeurIPS 2025): Causal multi-modal ST prediction with cross-modal attention (text+image+ST), dual-branch causal inference via backdoor adjustment, and GCN+Mamba hybrid encoder achieving 17-56% efficiency gains over Transformers[^src-e2-cstp].

## Key Challenges

1. **Modality variability**: Different cities have different available modalities with varying quality[^src-most]
2. **Spatial heterogeneity**: Local spatial patterns are highly region-specific and resist uniform modeling[^src-most]
3. **Scalability**: Must handle thousands of sensors efficiently[^src-most]

## Comparison with Time Series Foundation Models

| Dimension | ST Foundation Models | TS Foundation Models ([[timesfm|TimesFM]], [[chronos|Chronos]]) |
|-----------|---------------------|------------------------------------------------------------------|
| Input structure | Graph (sensors + topology) | Independent time series |
| Spatial modeling | Explicit (GNN, attention, experts) | None (channel-independent) |
| Modalities | Multi (image, text, location) | Single (numerical TS) |
| Primary task | Traffic/weather prediction | General forecasting |

## Related Pages

- [[most]] — MoST, first multi-modality ST foundation model
- [[urbandit]] — UrbanDiT, diffusion transformer for open-world spatiotemporal prediction
- [[gpt-st]] — GPT-ST, MAE pre-training for ST graphs, precursor to foundation models
- [[std-mae]] — STD-MAE, spatial-temporal-decoupled masked pre-training
- [[urbangpt]] — UrbanGPT, first spatio-temporal LLM with instruction-tuning (KDD 2024)
- [[uniflow]] — UniFlow, unified grid+graph ST foundation model with ST-MRA (arXiv 2024)
- [[traffic-forecasting]] — general traffic prediction
- [[multimodal-time-series-forecasting]] — multimodal TS forecasting
- [[large-scale-spatial-temporal-graph]] — large-scale ST graph challenges
- [[urbanfm]] — UrbanFM, scaling-centric ST foundation model with WorldST+MiniST+minimalist Transformer (arXiv 2026)
- [[factost]] — FactoST, factorized ST foundation model with UTP+STA, linear complexity (NeurIPS 2025 / arXiv 2026)
- [[urbanverse]] — UrbanVerse, complementary model for cross-city/cross-task urban region attribute prediction (crime/population/carbon/nightlight), non-temporal (arXiv 2026)
- [[spatio-temporal-foundation-model-landscape]] — 时空基础模型全景分析：设计哲学、架构范式、预训练策略、泛化机制深度比较
- [[urbanpg]] — UrbanPG, prompt-backbone decoupled ST framework unifying large-scale + few-shot + continual learning (AAAI 2026)
- [[ustd]] — USTD, task-unified ST diffusion (SIGSPATIAL 2024), complementary to foundation model paradigm
- [[allspark]] — AllSpark, 10-modality geospatial intelligence via language as reference framework
- [[language-as-reference-framework]] — LaRF principle

[^src-most]: [[source-most]]
[^src-urbandit]: [[source-urbandit]]
[^src-gpt-st]: [[source-gpt-st]]
[^src-2312-00516-std-mae]: [[source-2312-00516-std-mae]]
[^src-opencity]: [[source-opencity]]
[^src-unist]: [[source-unist]]
[^src-uniflow]: [[source-uniflow]]
[^src-urbanfm]: [[source-urbanfm]]
[^src-urbanpg]: [[source-urbanpg]]
[^src-factost]: [[source-factost]]
[^src-allspark]: [[source-allspark]]
[^src-e2-cstp]: [[source-e2-cstp]]