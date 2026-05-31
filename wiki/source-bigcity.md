---
title: "source-bigcity"
type: source-summary
tags:
  - spatiotemporal
  - trajectory
  - traffic-state
  - llm
  - foundation-model
  - universal-model
  - gpt-2
  - lora
created: 2026-06-01
last_updated: 2026-06-01
source_count: 1
confidence: high
status: active
---

# BIGCity: A Universal Spatiotemporal Model for Unified Trajectory and Traffic State Data Analysis

**Authors**: Xie Yu, Jingyuan Wang (corresponding), Yifan Yang (Beihang University), Qian Huang, Ke Qu (Huawei). **Venue**: arXiv:2412.00953v1, December 2024. **Code**: [github.com/bigscity/BIGCity](https://github.com/bigscity/BIGCity)[^src-bigcity].

## Core Contribution

BIGCity is the **first MTMD (Multi-Task, Multi-Data modality) spatio-temporal model** — a single model that simultaneously handles both individual-level trajectory data and population-level traffic state data across 8 heterogeneous tasks with one set of parameters[^src-bigcity]. Prior universal ST models ([[unist|UniST]], [[opencity|OpenCity]], [[urbangpt|UrbanGPT]], [[uniflow|UniFlow]], [[urbandit|UrbanDiT]]) were MTSD (Multi-Task, Solo-Data modality) — handling multiple tasks but only within a single data modality (either trajectories OR traffic states)[^src-bigcity].

## Key Innovations

1. **ST-unit**: A novel unified representation $U_{i,\tau} = (\mathbf{e}_i^{(s)}, \mathbf{e}_{i,t_\tau}^{(d)}, \iota_\tau)$ — a triplet of static road segment features, dynamic traffic states, and timestamps. Both trajectories (variable-length individual-level sequences) and traffic states (fixed-length population-level sequences) are expressed as ST-unit sequences in identical format, eliminating the data modality gap[^src-bigcity].

2. **ST Tokenizer**: A four-module neural pipeline converting ST-units into token sequences for LLM consumption: static GAT encoder (spatial topology), dynamic GAT encoder (instant traffic state), fusion cross-attention (global road-to-road dependencies), and temporal integration MLP with $\delta_\tau$ time intervals for handling non-uniform sampling[^src-bigcity].

3. **VMTP (Versatile Model with Task-oriented Prompt)**: GPT-2 backbone (1.5B) with LoRA (r=8, all attention blocks) processes task-oriented prompts composed of text instructions $X^{(txt)}$, ST-token sequences $X^{(st)}$, and task placeholders $X^{(tsk)}$ ([CLAS] for classification/generation, [REG] for regression). Three shared MLP heads ($\text{MLP}_c$, $\text{MLP}_t$, $\text{MLP}_r$) cover all 8 tasks — the prompt determines which heads are activated[^src-bigcity].

4. **Two-stage training**: Stage 1 (Masked Reconstruction) trains ST Tokenizer + LoRA to reconstruct masked ST-units without task labels. Stage 2 (Prompt Tuning) freezes the tokenizer and trains LoRA with all 8 tasks jointly, where cross-modal multi-task training exhibits stronger positive transfer than same-modal multi-task training[^src-bigcity].

## Results

Evaluated on 3 real-world datasets (Beijing 101M trajectories/40,306 road segments; Xi'an 385K/5,269; Chengdu 560K/6,195). Surpasses 18 independently trained baselines across all 8 tasks (trajectory: TTE, Next Hop, Classification, Similarity Search, Recovery; traffic state: One-Step, Multi-Step, Imputation). Cross-city generalization (BJ→XA/CD) shows <7% average performance loss and still outperforms from-scratch baselines. Ablation: removal of Task-oriented Prompt causes the largest degradation (10.5% avg)[^src-bigcity].

## Limitations

GPT-2 backbone is a 2019-era LLM; LoRA parameter sensitivity not fully explored; road network dependency makes the model vulnerable to incomplete OSM data; cross-cultural generalization (China→non-Chinese cities) unverified; no interpretability analysis[^src-bigcity].

[^src-bigcity]: [[source-bigcity]]
