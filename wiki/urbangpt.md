---
title: "UrbanGPT"
type: technique
tags:
  - spatial-temporal
  - large-language-model
  - instruction-tuning
  - zero-shot
  - foundation-model
  - traffic-forecasting
created: 2026-05-31
last_updated: 2026-06-09
source_count: 4
confidence: high
status: active
---

# UrbanGPT

**UrbanGPT** is the first spatio-temporal large language model that enables zero-shot prediction across diverse urban phenomena — traffic flow, bike demand, and crime rates — by integrating a spatio-temporal dependency encoder with the instruction-tuning paradigm of LLMs[^src-urbangpt]. Published at KDD 2024 by Li et al. (HKU / SCUT / Baidu), it uses Vicuna-7b as backbone and achieves consistent superiority over 10 traditional spatio-temporal baselines in both cross-region and cross-city zero-shot scenarios[^src-urbangpt].

## Core Problem

Traditional spatio-temporal models ([[stgcn|STGCN]], [[gwnet|GWNet]], [[mtgnn|MTGNN]], etc.) require abundant labeled data from the target city and fail catastrophically in zero-shot settings — predicting traffic in an unseen Chicago region using a model trained only on New York data[^src-urbangpt]. UrbanGPT addresses this by leveraging LLMs' world knowledge (urban patterns, human activity rhythms, functional zone semantics) to complement the limited numerical signals available in zero-shot scenarios[^src-urbangpt].

## Architecture: Four-Component Pipeline

### 1. Spatio-Temporal Dependency Encoder

A multi-level gated dilated convolutional network that processes temporal data without relying on graph structures[^src-urbangpt]:

- **Gated dilated convolution** (Eq. 3): Ψ_r^{(l)} = (W̄_k * E'_r + b̄_k) ⊙ δ(W̄_g * E'_r + b̄_g) + E'_r — gate controls information flow, dilated kernel captures multi-scale temporal dependencies
- **Multi-level correlation injection** (Eq. 4): S_r^{(l)} = (W_s * Ψ_r^{(l)} + b_s) + S_r^{(l-1)} — residual accumulation preserves fine-to-coarse temporal patterns
- **No graph dependency**: Explicit spatial propagation (GCN, GAT) is deliberately avoided. The encoder operates per-region independently, treating each sensor as a separate instance. Spatial reasoning is delegated to the LLM via textual POI descriptions — the LLM infers that "two commercial districts share similar traffic patterns" without requiring an adjacency matrix[^src-urbangpt]
- Output: Ψ̃ ∈ R^{R×F×d} (d=64), representing R regions × F features in a 64-dim space

### 2. Spatio-Temporal-Text Alignment Module

A lightweight projection that bridges the modality gap[^src-urbangpt]:

H = Ψ̃ · W_p + b_p (W_p ∈ R^{64×4096}, b_p ∈ R^{4096})

The aligned representations are embedded as special tokens `<ST_start>, <ST_HIS>, ..., <ST_HIS>, <ST_end>` in the LLM's vocabulary. Each `<ST_HIS>` carries a 4096-dim vector (matching Vicuna-7b's hidden dim), allowing the LLM to process spatio-temporal signals through its native self-attention — treating them as a "special kind of word"[^src-urbangpt].

### 3. Spatio-Temporal Prompt Instructions

Natural language descriptions encode three semantic dimensions[^src-urbangpt]:

| Dimension | Example | What LLM Infers |
|-----------|---------|-----------------|
| **Temporal** | "January 7, 2020, 08:30 Tuesday...30-min intervals" | Rush hour patterns, weekday vs weekend |
| **Spatial** | "Staten Island, POIs: Public Safety, Education, Residential" | Functional zone type, expected activity rhythms |
| **Task** | "Predict taxi inflow/outflow for next 12 steps" | Prediction task type, output horizon |

This is **UrbanGPT's deepest philosophical departure from [[gpt-st|GPT-ST]]**. GPT-ST believes pre-training on numerical data alone can learn universal urban patterns. UrbanGPT believes LLMs *already have* urban common sense from reading millions of text documents — the task is to "show" them the spatio-temporal data in a language they understand[^src-urbangpt].

### 4. Regression Prediction Layer

The critical design that resolves the LLM output mismatch[^src-urbangpt]:

Ŷ_{r,f} = W_3 [σ(W_1 · H_{r,f}), σ(W_2 · Γ_{r,f})]

- H_{r,f} ∈ R^{64}: raw spatio-temporal encoding (kept outside LLM)
- Γ_{r,f} ∈ R^{4096}: LLM's hidden representation of the prediction token — a vector encoding the LLM's *reasoning* about the spatio-temporal context (not a scalar prediction)
- Concatenation + 2-layer MLP maps the fused understanding to P=12 future steps

LLM never outputs numbers directly. It outputs a semantically rich hidden vector (Γ) that captures high-level reasoning — "flow should decrease in this residential area as the workday ends." The regression layer translates this reasoning + raw ST features → precise values[^src-urbangpt].

## Training

Multi-task joint optimization on NYC data[^src-urbangpt]:

- **Regression loss** (traffic/bike): L_r = (1/N) Σ|y_i - ŷ_i| (MAE)
- **Classification loss** (crime): L_c = binary cross-entropy
- **Language loss**: L_LLMs on prediction token generation
- **Total**: L = L_LLMs + L_r + L_c
- Training: 80 regions each from NYC-taxi (2017Q1), NYC-bike (2017Q2), NYC-crime (2016-2018)
- Backbone: Vicuna-7b, H=12 input steps, P=12 prediction steps, max 100 epochs

## Experimental Results

### Zero-Shot Cross-Region (Table 1)

| Dataset | Task | UrbanGPT | Best Baseline | Improvement |
|---------|------|----------|---------------|-------------|
| NYC-taxi | Inflow MAE | 6.16 | ASTGCN 9.75 | ↓36.8% |
| NYC-taxi | Outflow MAE | 6.83 | GWN 9.67 | ↓29.4% |
| NYC-bike | Inflow MAE | 2.02 | TGCN 2.88 | ↓29.9% |
| NYC-bike | Outflow MAE | 2.01 | GWN 3.07 | ↓34.5% |
| NYC-crime | Burglary Macro-F1 | 0.67 | MTGNN 0.64 | ↑4.7% |
| NYC-crime | Robbery Macro-F1 | 0.69 | MTGNN 0.65 | ↑6.2% |

Traditional models achieve Recall≈0 on crime prediction — they completely fail on unseen sparse data. UrbanGPT maintains Recall=0.34-0.42, demonstrating LLM world knowledge bridges extreme data sparsity[^src-urbangpt].

### Zero-Shot Cross-City (CHI-taxi)

Consistent superiority across all 12 prediction steps with no decay at long horizons — LLM-learned patterns are generic urban knowledge, not New York-specific surface features[^src-urbangpt].

### Supervised Prediction (Table 2)

UrbanGPT also matches or surpasses baselines in traditional supervised settings (NYC-taxi 2017 train → 2021 test). LLM text knowledge does NOT introduce noise; it improves long-term generalization by providing semantic understanding of urban dynamics[^src-urbangpt].

### Ablation Study (Figure 5)

| Variant | What's Removed | Impact |
|---------|---------------|--------|
| **-T2P** | Regression layer (LLM outputs text numbers directly) | **Severe degradation** — worst variant |
| **-STE** | ST dependency encoder | **Severe degradation** — LLM can't extract temporal patterns from raw numbers |
| **-STC** | Spatial + temporal prompt text | Significant degradation — world knowledge injection cut off |
| **-Multi** | Multi-dataset training (only NYC-taxi) | Moderate degradation — less cross-scenario common sense accumulated |

The encoder and the regression layer are equally critical — one provides "numerical understanding," the other provides "numerical output" — both are non-negotiable[^src-urbangpt].

### Robustness Analysis (Figure 6)

Partitioning regions by flow variance into 4 quartiles: all models perform well in low-variance (stable residential) areas. In high-variance quartile (dynamic commercial hubs), baselines collapse while UrbanGPT maintains low error — LLM common sense is most valuable for complex, highly dynamic patterns[^src-urbangpt].

## Comparison with Other ST Foundation Models

| Dimension | UrbanGPT | [[unist|UniST]] | [[urbandit|UrbanDiT]] | [[urbanpg|UrbanPG]] | [[gpt-st|GPT-ST]] | [[opencity|OpenCity]] | [[urbanverse|UrbanVerse]] |
|-----------|----------|-----------------|----------------------|-------------------|-------------------|----------------------|----------------------|
| **Core paradigm** | LLM instruction-tuning | MAE pre-training + prompt learning | Diffusion Transformer | Prompt-backbone decoupled | MAE pre-training | Transformer + GNN | Random walk + mask-reconstruct |
| **Spatial model** | Textual semantics (no graph) | Memory-pool adaptive prompts | Learnable prompt + diffusion | Es embedding + STCA linear attention | Hypergraph capsule | GNN with graph topology | Hexagon grid + region-centric local walks |
| **Zero-shot** | Cross-region + cross-city | Cross-city + cross-domain | Cross-data-type (5 tasks) | Cross-city (prompt fine-tune) | ❌ Dataset-specific | Cross-city | Cross-city + cross-task |
| **Inference speed** | 174s (7B params, per-sensor) | 0.034 min (~2s) | Fast (rectified flow 20 steps) | Fast (72% faster than PatchSTG) | Normal (per-dataset) | Normal | Fast (<1s, 10 diffusion steps) |
| **Modality** | Numerical + text prompts | Numerical only | Numerical only | Numerical only | Numerical only | Numerical only | POI only (15-dim) |
| **Task type** | Traffic + crime | Traffic flow | Traffic flow (5 tasks) | Traffic flow | Traffic flow | Traffic flow | Region attributes (6 tasks) |

[[urbanmind|UrbanMind]] (KDD 2025) extends the LLM-based ST line further with Muffin-MAE multifaceted pre-training (3 masking types for inter-correlated dynamics) and test-time adaptation (masked reconstruction of LLM embeddings), achieving SOTA zero-shot across 9 urban dynamics datasets with cross-city generalization 8.5% MAE better than UrbanGPT[^src-urbanmind].

UrbanGPT's key limitation is computational cost: 7B parameters and per-sensor processing make it impractical for large-scale sensor networks (hundreds or thousands of sensors requiring real-time updates)[^src-urbangpt]. [[urbandit|UrbanDiT]] addresses this by training from scratch with parallel processing and 25× inference acceleration via rectified flow. [[urbanmind|UrbanMind]] (KDD 2025) extends the LLM-based ST paradigm with two critical improvements: (1) Muffin-MAE — a multifaceted masked autoencoder with temporal/spatial/global masking that jointly models inter-correlated urban dynamics (speed, inflow, demand); (2) test-time adaptation — a masked reconstruction mechanism that mitigates distributional shift during inference by adapting shared predictor-reconstructor layers to unseen test regions[^src-urbanmind]. UrbanMind achieves 8.5% lower MAE than UrbanGPT in cross-city zero-shot transfer and ~33-47% MAE reduction on several urban dynamics tasks[^src-urbanmind].

[[fstllm|FSTLLM]] (ICML 2025) shares UrbanGPT's goal of injecting LLM contextual knowledge into spatio-temporal forecasting but differs in two ways: it keeps a swappable STGNN backbone (UrbanGPT discards graphs for POI text) and targets few-shot calibration rather than zero-shot cross-city transfer, QLoRA-fine-tuning LLaMA-2-7B to refine the backbone's numerical predictions[^src-fstllm].

[[st-vision-llm|ST-Vision-LLM]] (arXiv 2025) takes a different route from UrbanGPT's separate-encoder strategy: instead of a temporal-conv encoder plus textual POI prompts feeding a node-based LLM, it renders the whole grid history as images for a Vision-LLM's visual encoder, arguing that node-centric LLM approaches like UrbanGPT and ST-LLM become computationally prohibitive on large dense grids[^src-st-vision-llm].

## Related Pages

- [[source-urbangpt]] — source summary page
- [[spatio-temporal-foundation-model]] — broader ST foundation model paradigm
- [[gpt-st]] — GPT-ST, MAE pre-training for ST graphs (NeurIPS 2023), non-LLM contrast
- [[urbandit]] — UrbanDiT, diffusion-based ST foundation model (NeurIPS 2025), successor
- [[uniflow]] — UniFlow, unified grid+graph transformer-based ST foundation model (same FIB Lab, arXiv 2024)
- [[urbanfm]] — UrbanFM, scaling-centric ST foundation model (arXiv 2026), minimalist transformer, 100+ cities
- [[opencity]] — OpenCity, open-source ST foundation model for traffic
- [[traffic-forecasting]] — traffic prediction task overview
- [[stgcn]] — STGCN, baseline used in UrbanGPT's zero-shot experiments
- [[gwnet]] — GWNet, baseline used in UrbanGPT's zero-shot experiments
- [[mtgnn]] — MTGNN, baseline used in UrbanGPT's zero-shot experiments
- [[bigcity]] — BIGCity, first MTMD ST model, extends the LLM route from UrbanGPT's traffic-only to simultaneous trajectory+traffic via GPT-2+LoRA+prompt (arXiv 2024)
- [[urbanpg]] — UrbanPG, prompt-backbone decoupled ST framework with linear attention, unifies large-scale + few-shot + continual learning (AAAI 2026)
- [[urbanverse]] — UrbanVerse, foundation model for cross-city/cross-task urban region attribute prediction (crime/population/carbon/nightlight), complementary to traffic-focused ST models (arXiv 2026)
- [[spatio-temporal-foundation-model-landscape]] — 时空基础模型全景分析：LLM-Based 路线代表
- [[urbanmind]] — UrbanMind, multifaceted ST-LLM with Muffin-MAE and test-time adaptation (KDD 2025)
- [[muffin-mae]] — Muffin-MAE, multifaceted masked autoencoder for inter-correlated urban dynamics
- [[fstllm]] — FSTLLM, LLM-augmented few-shot ST forecasting keeping a swappable STGNN backbone (ICML 2025)
- [[st-vision-llm]] — ST-Vision-LLM, grid-rendered Vision-LLM approach contrasting UrbanGPT's node-based separate encoder (arXiv 2025)

[^src-urbangpt]: [[source-urbangpt]]
[^src-urbanmind]: [[source-urbanmind]]
[^src-fstllm]: [[source-fstllm]]
[^src-st-vision-llm]: [[source-st-vision-llm]]
