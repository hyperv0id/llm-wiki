---
title: "UniST"
type: technique
tags:
  - spatio-temporal
  - foundation-model
  - prompt-learning
  - masked-autoencoder
  - one-for-all
  - zero-shot
  - traffic-forecasting
created: 2026-05-31
last_updated: 2026-05-31
source_count: 1
confidence: high
status: active
---

# UniST

**UniST** (Universal Spatio-Temporal model) is the first one-for-all spatio-temporal foundation model for urban prediction, enabling a single pre-trained model to handle 20+ datasets across diverse cities and domains without architectural modification[^src-unist]. Published at KDD 2024 by Yuan Yuan, Jingtao Ding, Jie Feng, Depeng Jin, and Yong Li (Tsinghua University, Department of Electronic Engineering), UniST establishes a non-LLM route to universal spatio-temporal generalization — using pure numerical data with MAE-style self-supervised pre-training and knowledge-guided prompt learning[^src-unist].

## Core Insight

While different cities and domains produce spatio-temporal data with wildly different formats and distributions, the underlying dynamics are all driven by **human activity patterns** — morning/evening peaks, spatial proximity effects, functional zone hierarchies. UniST's thesis: if a model can learn these shared patterns from abundant multi-source data during pre-training, and then /adapt to each dataset's specific distribution via input-space prompts/ rather than parameter fine-tuning, a single model can serve all scenarios[^src-unist].

## Architecture

### Stage 1: Spatio-Temporal Pre-Training (MAE-Style)

**Data Unification via Spatio-Temporal Patching.** A 3D convolution with kernel = stride = $(l, h, w)$ transforms any 3D tensor $X \in \mathbb{R}^{L \times H \times W}$ (channel-independent) into patch tokens $\mathbf{E}_x \in \mathbb{R}^{L' \times H' \times W'}$, then flattened into a 1D sequence of length $L' \times H' \times W'$. Position encoding uses sine-cosine (not learnable), experimentally proven more robust to noise and grid-size generalization[^src-unist].

**Encoder-Decoder with Full Decoder.** Unlike MAE's lightweight decoder, UniST uses a /full-capacity decoder/ because spatio-temporal prediction requires both high-level semantics and precise numerical reconstruction. Mask tokens are shared learnable embeddings[^src-unist].

**Four Complementary Masking Strategies** (randomly selected per mini-batch)[^src-unist]:
| Strategy | Mechanism | Primary Capability |
|----------|-----------|-------------------|
| Random masking | Uniform random patch masking | Fine-grained spatio-temporal relations |
| Tube masking | Full temporal masking of a spatial unit | Spatial extrapolation (sensor failure) |
| Block masking | Contiguous spatial block masked across all time | Spatial transfer (no spatial context) |
| Temporal masking | Future time steps entirely masked | Causal temporal modeling |

**Pre-training loss**: MSE on masked patch reconstruction. Each epoch samples one dataset + one masking strategy randomly[^src-unist].

### Stage 2: Knowledge-Guided Prompt Learning

After pre-training, the encoder-decoder backbone (attention + FFN layers) is **frozen**. Only the prompt network is trained[^src-unist].

**Domain Knowledge Extraction.** Four features are extracted from input $X$[^src-unist]:
- $E_{sc}$ — Spatial Closeness: 3×3 2D CNN on temporally-compressed spatial representation
- $E_{sh}$ — Spatial Hierarchy: Multi-scale 2D CNNs (5×5, 7×7, 9×9) stacked for hierarchical spatial patterns
- $E_{tc}$ — Temporal Closeness: Attention-weighted aggregation of recent $M$ time steps
- $E_{tp}$ — Temporal Periodicity: Attention-weighted aggregation of corresponding time slots from $N$ previous days

**Memory Pools.** Two learnable key-value memory pools[^src-unist]:
$$KM_s = \{(k_{s,0}, m_{s,0}), \dots, (k_{s,N-1}, m_{s,N-1})\}$$
$$KM_t = \{(k_{t,0}, m_{t,0}), \dots, (k_{t,N-1}, m_{t,N-1})\}$$

Each domain knowledge feature queries its respective memory pool via attention:
$$\alpha = [k_0; \dots; k_{N-1}] E^T, \quad P = \sum_i \alpha_i m_i$$

**Prompt Injection.** The four generated prompts $P_{sc}, P_{sh}, P_{tc}, P_{tp}$ are added to input tokens at each Transformer layer. This means different input samples generate /different/ prompts — adaptive rather than dataset-fixed[^src-unist].

## Key Results

### Short-Term Prediction (6→6 steps)
| Dataset | Domain | UniST RMSE | Best Baseline RMSE | Improvement |
|---------|--------|-----------|-------------------|-------------|
| TaxiBJ | Taxi flow (Beijing) | 26.84 | STID 27.36 | ↓1.9% |
| Crowd | Crowd flow (Nanjing) | 3.00 | STID 3.85 | ↓22.1% |
| Cellular | Cellular (Nanjing) | 14.29 | PromptST 15.74 | ↓9.2% |
| BikeNYC | Bike demand (NYC) | 3.50 | STGSP 5.00 | ↓30.0% |
| TrafficSH | Traffic speed (Shanghai) | 0.665 | STID 0.742 | ↓10.4% |

### Long-Term Prediction (64→64 steps)
TaxiNYC RMSE 19.83 vs SimVP 20.18; Crowd RMSE 4.25 vs STID 4.91. Average improvement ~10.1%[^src-unist].

### Few-Shot & Zero-Shot
- **1% training data** on Crowd: UniST RMSE 13.95 vs PatchTST 14.49[^src-unist]
- **Zero-shot** on Crowd: UniST RMSE 14.67, surpassing ACFM 1%-shot (21.17) and PredRNN 1%-shot (24.90)[^src-unist]

## Key Design Choices

| Choice | Rationale | Evidence |
|--------|-----------|----------|
| Sine-cosine PE > learnable PE | Better generalization to unseen grid sizes + noise robustness | Table 7 vs Table 8 |
| Full decoder > lightweight decoder | Spatio-temporal reconstruction needs numerical precision | Architecture ablation |
| 512 memory pool entries (optimal) | 128→512 improves; 512→1024 saturates | Figure 5b |
| Freeze backbone, train only prompt | Prevents catastrophic forgetting of general patterns | Design rationale |
| Channel-independent processing | Enables model to handle varying channel counts across datasets | Data unification |

## Comparison with Contemporaries

| Dimension | UniST (KDD 2024) | [[urbangpt|UrbanGPT]] (KDD 2024) | [[opencity|OpenCity]] (2024) | [[urbandit|UrbanDiT]] (NeurIPS 2025) |
|-----------|-----------------|------------------------------|---------------------------|----------------------------------|
| **Core paradigm** | MAE pre-training + prompt learning | LLM instruction-tuning | Instance norm + TimeShift Transformer | Diffusion Transformer + prompt |
| **Data format** | Grid-based only | Grid-based (per-sensor) | Graph-based (adjacency matrix) | Grid + Graph |
| **Zero-shot** | ✅ Cross-city, ST scenarios | ✅ Cross-region + cross-city | ✅ Cross-city (no fine-tuning) | ✅ Cross-data-type (5 tasks) |
| **Fine-tuning** | Prompt tuning (lightweight) | Full LLM fine-tuning | Prediction head only | Rectified flow inference |
| **Model size** | 6.71M | ~7B (Vicuna-7b) | 2M/5M/26M | Not disclosed |
| **Inference** | 0.034 min | 174s per sensor (~45,000s) | <3s | Fast (rectified flow) |

UniST and [[urbandit|UrbanDiT]] share the same Tsinghua FIB Lab lineage — Yuan Yuan is first author on UniST and a co-author on UrbanDiT, representing a consistent research program from prompt-learning ST foundation models (UniST, KDD 2024) to diffusion-based universal ST modeling (UrbanDiT, NeurIPS 2025)[^src-unist].

## Lineage

UniST sits at the intersection of three research threads[^src-unist]:
1. **Self-supervised pre-training**: BERT → [[mae|MAE]] → UniST (extending MAE to 4D spatio-temporal data)
2. **Prompt learning**: Visual Prompt Tuning → L2P (Learning to Prompt) → UniST (memory-based adaptive prompts)
3. **Spatio-temporal modeling**: STResNet → STID → UniST (unifying multiple formats under one model)

**Predecessors**: [[gpt-st|GPT-ST]] (per-dataset MAE pre-training) showed pre-training works for ST data but remained task-specific; UniST is the first to achieve true /cross-dataset/ one-for-all pre-training[^src-unist].
**Successors**: [[urbandit|UrbanDiT]] (NeurIPS 2025, same lab) extends UniST's prompt-learning paradigm from grid-only to grid+graph, from MAE to diffusion, from prediction-only to 5-task coverage[^src-urbandit].

## Related Pages

- [[source-unist]] — source summary page
- [[spatio-temporal-foundation-model]] — broader ST foundation model paradigm
- [[opencity]] — OpenCity, zero-shot ST foundation model with instance normalization (2024)
- [[urbangpt]] — UrbanGPT, first spatio-temporal LLM (KDD 2024)
- [[urbandit]] — UrbanDiT, diffusion-based ST foundation model (NeurIPS 2025, same lab successor)
- [[urbanfm]] — UrbanFM, scaling-centric ST foundation model (arXiv 2026), minimalist transformer, 100+ cities
- [[gpt-st]] — GPT-ST, MAE pre-training for ST graphs (NeurIPS 2023, predecessor)
- [[traffic-forecasting]] — traffic prediction task overview
- [[mae]] — Masked Autoencoders (CVPR 2022), foundational paradigm
- [[std-mae]] — STD-MAE, spatial-temporal-decoupled masked pre-training (IJCAI 2024)
- [[patchtst]] — PatchTST, origin of patch-based time series tokenization (ICLR 2023)
- [[bigcity]] — BIGCity, first MTMD ST model, extends beyond UniST's traffic-only scope to include trajectories (arXiv 2024)

[^src-unist]: [[source-unist]]
[^src-urbandit]: [[source-urbandit]]
