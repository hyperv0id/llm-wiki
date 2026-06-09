---
title: "ST-Vision-LLM"
type: entity
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

# ST-Vision-LLM

**ST-Vision-LLM** (Spatiotemporal Vision Large Language Model) is a framework that reframes 2D grid-based spatiotemporal traffic forecasting as a **[[vision-language-traffic-forecasting|vision-language fusion]]** task, proposed by Ning Yang et al. (CAS / Southwest University / USTB / Northwestern) in arXiv:2510.11282[^src-st-vision-llm]. Rather than feeding spatial locations to an LLM as a 1D token list, it renders historical global traffic matrices as images, encodes them with the native visual tower of a Vision-LLM (Qwen2.5-VL-7B-Instruct), and generates per-cell forecasts as numerical tokens[^src-st-vision-llm].

## Motivation

LLM-for-time-series methods such as [[time-llm|Time-LLM]] and LLM4TS are inherently 1D-sequence models and lack mechanisms for the topological/spatial structure of grid data[^src-st-vision-llm]. Existing spatial-LLM approaches fall into two camps, both with drawbacks: (a) **separate encoders** before the LLM ([[urbangpt|UrbanGPT]]'s temporal conv encoder, TPLLM's GCN), and (b) **making the LLM spatially aware** via positional encodings (ST-LLM), graph tokenizers (STG-LLM), or modified attention (ST-LINK)[^src-st-vision-llm]. These either append spatial info as inefficient linear sequences or intrusively alter the backbone — and they assume discrete node-based structures (sensor networks), becoming computationally prohibitive on large, dense grids[^src-st-vision-llm]. ST-Vision-LLM's thesis: a visual encoder naturally models patch-based 2D grid structure, local neighborhoods, and long-range spatial dependencies as a unified scene, unleashing the LLM's reasoning without architectural surgery[^src-st-vision-llm].

This addresses mobile traffic at the **grid** level, distinct from sensor-graph traffic — see [[mobile-traffic-forecasting]][^src-st-vision-llm].

## Problem Formulation

The urban area is modeled as an H×W grid; the task forecasts traffic matrices for the next K steps given S past matrices[^src-st-vision-llm]. To scale to large grids under LLM output-length limits, the paper adopts a **conditional-independence approximation** (inspired by Conditional Neural Processes), factorizing the future grid distribution so each cell-level prediction is conditioned on the **shared global history** `D_{t−S+1:t}` and its **own coordinates** (x,y)[^src-st-vision-llm]:

$$ p(D_{t+1:t+K} \mid D_{t-S+1:t}) \approx \prod_{x=1}^{H}\prod_{y=1}^{W} p\big(d^{(x,y)}_{t+1:t+K} \mid D_{t-S+1:t}, (x,y)\big) $$

This preserves the major spatial dependencies carried by the shared global history but does **not** model the residual synchronous coupling among future cells after conditioning[^src-st-vision-llm].

## Architecture

### 1. Multimodal Input Construction

- **Power-Law Normalization** (Box-Cox style): a power transform with exponent p∈(0,1] suppresses the long-tailed traffic distribution, then division by the powered max maps values strictly into [0,1] to satisfy the image-encoder input range[^src-st-vision-llm].
- **Grayscale pseudo-RGB**: the single-channel scalar field is replicated across all three channels (`I_t = [D_t^{norm}, D_t^{norm}, D_t^{norm}]`). This is purely to satisfy Qwen2.5-VL's 3-channel interface and to avoid spurious cross-channel correlations — not to inject color semantics; replicating single-channel data into 3 channels is a common adaptation (e.g. medical imaging)[^src-st-vision-llm].
- Each image is split into L×L patches and run through `ImageEncoder(·)`; per-frame embeddings are concatenated in temporal order into `E_visual ∈ R^{(S·N)×d}`, then concatenated with the text-prompt embeddings to form the LLM context[^src-st-vision-llm].
- The **text prompt** carries: task instruction, target coordinates (x,y), data type, input time range, normalization parameters (max, p), and the target cell's own historical scalar sequence (as numerical tokens)[^src-st-vision-llm].

### 2. Direct Numerical Encoding

A core innovation (see [[direct-numerical-encoding]]): floating-point values are encoded as **single tokens** `⟨|FP m/b|⟩ ↦ Norm(m)×10^b`, where m is a 3-digit mantissa in {−999..−1}∪{1..999} and b an exponent in {−4..5}; zero gets a dedicated `⟨|FP0/0|⟩`[^src-st-vision-llm]. Inspired by Charton (2022)'s linear-algebra transformers, but applied to *fine-tuning* a pretrained LLM rather than training from scratch[^src-st-vision-llm]. Decoding back to readable values is done by a numerical token decoder.

A **two-stage numerical alignment fine-tuning** imbues the LLM with these tokens[^src-st-vision-llm]:
- **Stage 1 — Semantic alignment**: extend the input/output embedding matrices with the FP vocabulary; freeze the backbone and train only the new embeddings on transcription tasks (string↔token), so `⟨|FP114/0|⟩` lands near "1.14" in embedding space.
- **Stage 2 — Basic arithmetic alignment**: unfreeze backbone + token/output layers via LoRA and train on three linear-algebra ops (vector addition, subtraction, Hadamard product), covering token→string, string→token, and token→token formats; the final forecasting task always outputs numerical tokens.

### 3. SFT + Reinforcement Learning

- **SFT**: standard causal-LM objective with **loss masking** — cross-entropy computed only over target-output tokens, ignoring the input context[^src-st-vision-llm].
- **GRPO** (see [[grpo-for-forecasting]]): a memory-efficient, critic-free variant of PPO (from DeepSeekMath). For each prompt the model samples G candidate sequences; the baseline is the group-average reward, advantages are group-relative, and a KL penalty against the frozen SFT reference model maintains stability[^src-st-vision-llm]. The **reward** combines an accuracy term `exp(−(log2/x_h)·E)` on NRMSE `E`, a length-mismatch penalty `−(|L_out−L_gt|/L_gt)·0.5`, and a decode-failure penalty `δ_dec=−0.5`[^src-st-vision-llm].

## Experiments

- **Data**: Telecom Italia Big Data Challenge — Milan (100×100 grid) and Trentino (117×98), Nov 2013–Jan 2014, 10-min interval, channels {Internet, SMS, Call} → 6 subsets[^src-st-vision-llm]. Linear interpolation imputes gaps; error is computed only on originally-observed points. Due to compute cost, training/evaluation is confined to the central region (x,y)∈[45,55)×[45,55), though the model perceives the full grid[^src-st-vision-llm].
- **Baselines** (12): ARIMA; STN, ST-ResNet, ResLSTM, ACFM; [[stgcn|STGCN]], MCSTGCN, [[gwnet|GWNet]]; ST-LLM, GCNGPT, GATGPT, [[time-llm|Time-LLM]] (Qwen2.5-7B backbone)[^src-st-vision-llm].
- **Long/short-term** (Milan-Internet, S=12, K∈{1,10,30,60}): SOTA at multi-step and long horizons; at 60-step it cuts NRMSE ~45% vs ST-LLM and degrades far more gracefully than baselines[^src-st-vision-llm]. **GWNet is the strongest at single-step (1-step)**, but ST-Vision-LLM overtakes it from 10-step onward[^src-st-vision-llm].
- **Cross-domain** (K=36): best on 3 of 4 subsets; on Milan-Internet, RMSE ↓>42% vs second-best MCSTGCN[^src-st-vision-llm].
- **Few-shot** (10% / 5% data): best on 3 of 4 subsets in both settings[^src-st-vision-llm].
- **Cross-domain few-shot** (pretrain Trentino-Call, fine-tune on 2% of target): best on **all four** subsets[^src-st-vision-llm].
- **Zero-shot** (pretrain Trentino-Call, evaluate directly): best on **all four** subsets[^src-st-vision-llm].

### Ablations & Efficiency

Removing the image encoder, the LLM backbone, or the GRPO stage each degrades accuracy; removing the LLM backbone is by far the most damaging[^src-st-vision-llm]. The numerical-encoding trade-off is striking: Decimal-String / Integer-Approximation variants reach slightly lower error but need 112 / 42 output tokens vs **13** for numerical encoding, which dominates decoding latency[^src-st-vision-llm]. Reported efficiency: output length 465→39 tokens (91.6% reduction), full context 1034→468 (54.8%); single-cell latency 0.41s (vs 2.13s for decimal strings), 10×10 region 1.95s via a shared-prefix KV cache for the reused visual prefix[^src-st-vision-llm].

## Relation to Other Work

| Model | Spatial handling | Modality | RL | Backbone |
|-------|------------------|----------|----|----------|
| **ST-Vision-LLM** | grid → **image** via visual encoder | TS-as-image + text | **GRPO** | Qwen2.5-VL-7B |
| [[urbangpt|UrbanGPT]] | temporal conv encoder + text POI | numerical + text | ✗ | Vicuna-7B |
| [[time-llm|Time-LLM]] | none (1D reprogramming) | numerical + text proto | ✗ | frozen LLM |
| [[most|MoST]] | MoE spatial experts + graph | image + text + loc + TS | ✗ | from scratch |
| [[streasoner|STReasoner]] | graph reasoning (CoT) | TS + text | **S-GRPO** | TS-LM |

ST-Vision-LLM and [[streasoner|STReasoner]] are the two members of this run that pair an LLM with **GRPO reinforcement learning** for spatio-temporal tasks, but differ in objective: ST-Vision-LLM optimizes numerical forecasting accuracy, while STReasoner optimizes multi-step natural-language reasoning[^src-st-vision-llm]. Unlike [[most|MoST]] (which uses real satellite imagery as a genuine modality), ST-Vision-LLM's "images" are a *rendering of the numerical traffic field itself* — a visual representation chosen for its 2D inductive bias, placing it within [[multimodal-time-series-forecasting|multimodal time-series forecasting]] but with a unique time-series-as-image route[^src-st-vision-llm].

## Related Pages

- [[source-st-vision-llm]] — source summary
- [[vision-language-traffic-forecasting]] — the new paradigm this paper establishes
- [[direct-numerical-encoding]] — single-token float encoding + alignment fine-tuning
- [[grpo-for-forecasting]] — GRPO RL for forecasting accuracy
- [[mobile-traffic-forecasting]] — task setting (grid-level wireless traffic)
- [[time-llm]] — 1D LLM-for-TS baseline and contrast
- [[urbangpt]] — node-based ST-LLM, separate-encoder route
- [[most]] — multimodal ST foundation model (genuine image modality contrast)
- [[streasoner]] — sibling LLM+GRPO ST model (reasoning vs forecasting)
- [[multimodal-time-series-forecasting]] — broader multimodal TS landscape
- [[traffic-forecasting]] — general traffic prediction
- [[flow-grpo]] — GRPO applied to flow-matching generation (RL analogue in vision)

[^src-st-vision-llm]: [[source-st-vision-llm]]
