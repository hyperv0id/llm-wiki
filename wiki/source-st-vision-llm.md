---
title: "Source: Vision-LLMs for Spatiotemporal Traffic Forecasting (ST-Vision-LLM)"
type: source-summary
tags:
  - mobile-traffic-forecasting
  - vision-language-model
  - large-language-model
  - reinforcement-learning
  - few-shot
  - spatio-temporal
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: medium
status: active
---

# Source: Vision-LLMs for Spatiotemporal Traffic Forecasting

**Authors**: Ning Yang (Institute of Automation, CAS), Hengyu Zhong (Southwest University), Haijun Zhang (USTB), Randall Berry (Northwestern). **Venue**: arXiv:2510.11282 (v1 Oct 2025, v2 May 2026)[^src-st-vision-llm].

## Core Idea

The paper proposes **[[st-vision-llm|ST-Vision-LLM]]**, which reframes 2D grid-based spatiotemporal traffic forecasting as a **[[vision-language-traffic-forecasting|vision-language fusion]]** problem rather than a 1D sequence-modeling problem[^src-st-vision-llm]. The motivation: LLMs adapted for time series (e.g. [[time-llm|Time-LLM]]) are fundamentally 1D-sequence models and lack mechanisms to capture the topological/spatial dependencies of dense grid data, while node-centric LLM approaches ([[urbangpt|UrbanGPT]], ST-LLM, STG-LLM) become computationally prohibitive when grids have thousands of cells[^src-st-vision-llm]. Instead, ST-Vision-LLM renders each historical global traffic matrix as an image and lets the native visual encoder of a Vision-LLM (Qwen2.5-VL-7B-Instruct) perceive the whole scene at once[^src-st-vision-llm].

## Method

1. **Multimodal input**: historical traffic matrices are Power-Law normalized into [0,1], replicated single-channel→grayscale pseudo-RGB images, patchified, and encoded by the Vision-LLM's built-in visual tower; the resulting visual embeddings are concatenated with a textual prompt carrying the target cell's coordinates, normalization params, and scalar history[^src-st-vision-llm].
2. **Conditional-independence factorization** (inspired by Conditional Neural Processes): the H×W grid forecast is decomposed into per-cell predictions, each conditioned on the shared global history and its own coordinates[^src-st-vision-llm].
3. **[[direct-numerical-encoding|Direct Numerical Encoding]]**: floats are mapped to single tokens `⟨|FP m/b|⟩` (3-digit mantissa m, exponent b∈{−4..5}), then a **two-stage numerical alignment fine-tuning** (semantic alignment of embeddings, then LoRA arithmetic alignment on vector add/sub/Hadamard) teaches the LLM to read/write these tokens[^src-st-vision-llm].
4. **Two-stage optimization**: SFT (causal LM with loss masking) followed by **[[grpo-for-forecasting|GRPO]]** reinforcement learning (critic-free PPO variant from DeepSeekMath) using an NRMSE-based reward with length-mismatch and decode-failure penalties[^src-st-vision-llm].

## Contributions

- First to cast spatially-correlated traffic forecasting as a vision-language generation task, embedding the global history through a visual encoder without modifying the LLM backbone[^src-st-vision-llm].
- An efficient single-token numerical encoding + aligned fine-tuning that conserves context length while keeping outputs interpretable[^src-st-vision-llm].
- Validation across long/short-term, cross-domain, few-shot, cross-domain few-shot, and zero-shot settings on real mobile-traffic data[^src-st-vision-llm].

## Results

On the Telecom Italia dataset (Milan 100×100, Trentino 117×98, 10-min interval), ST-Vision-LLM outperforms 12 baselines by ~15.6% in long-term accuracy and exceeds the best baseline by ~30% on average in cross-domain few-shot[^src-st-vision-llm]. It wins all four zero-shot subsets and most few-shot subsets; numerical encoding cuts the output from 465→39 tokens (91.6%) and full context by 54.8%, dropping single-cell latency from 2.13s→0.41s[^src-st-vision-llm].

## Limitations

The conditional-independence approximation does not model residual synchronous coupling among future cells after conditioning on history[^src-st-vision-llm]. Due to compute cost, training/evaluation is confined to a central 10×10 sub-region (though the model perceives the full grid), and at single-step horizon GWNet still outperforms it[^src-st-vision-llm].

[^src-st-vision-llm]: [[source-st-vision-llm]]
