---
title: "FSTLLM"
type: entity
tags:
  - llm
  - spatio-temporal
  - few-shot-learning
  - traffic-forecasting
  - plug-and-play
  - icml-2025
created: 2026-06-08
last_updated: 2026-07-18
source_count: 1
confidence: medium
status: active
---

# FSTLLM (Few-shot Spatio-Temporal Large Language Models)

**FSTLLM** is an LLM-augmented framework for **few-shot** multivariate time series forecasting, proposed by Jiang, Chen, Li, Chao, Liu & Cong (NTU / Alibaba DAMO / HIT-Shenzhen) at ICML 2025[^src-fstllm]. It injects the contextual, common-sense reasoning of large language models into spatio-temporal forecasting and is designed to **plug into existing forecasting models** to boost their accuracy in data-scarce settings[^src-fstllm].

## Motivation

[[dcrnn|STGNNs]] and TSFMs (GPT4TS, [[time-llm|Time-LLM]]) need large training data and fail when only days of history exist[^src-fstllm]. FSTLLM argues two things are missing in prior work: (1) numerical-only pipelines ignore real-world context (geography, urban patterns, human behavior); (2) existing LLM-for-TS methods either fine-tune on raw numbers or prepend generic prompts, underusing the LLM's reasoning. FSTLLM addresses both with context-rich graph construction and prompt-based knowledge injection[^src-fstllm].

## Architecture

FSTLLM has three modules[^src-fstllm]:

### 1. LLM-Enhanced Graph Construction
For each node (time-series channel), real-world textual documents (e.g., parking rates, location, capacity, user reviews) are encoded by a frozen **LLaMA-2-7B**; final-layer hidden states $H_D \in \mathbb{R}^{N\times D}$ are projected by an FFN to $E \in \mathbb{R}^{N\times d}$[^src-fstllm]. A graph-attention layer computes pairwise node scores, normalized by the sparse **[[alpha-entmax|α-Entmax]]** activation (generalizing softmax/sparsemax; α=2.0) to suppress noisy/distant nodes, yielding a semantically meaningful adjacency matrix $A \in \mathbb{R}^{N\times N}$[^src-fstllm]. This context-aware graph is especially valuable when scarce data makes purely data-driven node embeddings unstable[^src-fstllm].

### 2. STGNN Backbone
$A$ feeds an STGNN that outputs numerical prediction tokens $C$. The default backbone replaces standard matrix multiplication in a GRU with **graph diffusion convolution** (GTS-style; diffusion depth $S=3$, GRU hidden size 64), trained with MAE loss[^src-fstllm]. The backbone is **swappable** — any standard STGNN works[^src-fstllm].

### 3. Domain Knowledge Injection
A LLaMA-2-7B is supervised-fine-tuned (SFT) with **QLoRA** (4-bit quantization, LoRA rank 64, lr 2e-4, 2 epochs) to calibrate the STGNN's numerical tokens[^src-fstllm]. Each prompt has **six components**[^src-fstllm]:

| Component | Content |
|-----------|---------|
| Task Instruction | domain, frequency, historical/forecasting steps |
| Node Description | LLM-summarized node documents + reviews (generated with ChatGPT-4o) |
| Node Pattern | LLM-summarized daily/weekly trends, peak/off-peak periods |
| Historical Input | input series $X_{ij} \in \mathbb{R}^T$ |
| Numerical Prediction Token | STGNN prediction $C_{ij} \in \mathbb{R}^T$ |
| Future Token | ground truth (training only; omitted at inference) |

At inference the fine-tuned LLM jointly reasons over domain knowledge, temporal dynamics and spatial correlations to produce context-aware predictions[^src-fstllm].

## Plug-and-Play Integration

FSTLLM can augment external forecasters: removing the LLM-graph + STGNN backbone and substituting the numerical prediction tokens with those from a transformer-based model (no retraining), the Domain Knowledge Injection module still improves them[^src-fstllm]. On Nottingham, GPT4TS MAE 27.6→21.8 and iTransformer MAE 28.4→22.3 when wrapped by FSTLLM[^src-fstllm].

## Results

- **Nottingham** (19 car parks, 15-min): best in 22/24 evaluations, ~30% MAPE reduction vs baselines incl. GTS/VAR[^src-fstllm].
- **ECL** (19 clients, hourly): best in 25/36 evaluations, >50% relative MAPE reduction vs GPT4TS and [[itransformer|iTransformer]][^src-fstllm].
- **Data efficiency**: FSTLLM with 3 days of data outperforms all baselines trained with 30 days (10× more)[^src-fstllm].
- **Ablation**: FSTLLM-NoInjection degrades most (MAE 21.0→25.1), confirming Domain Knowledge Injection is the key component; FSTLLM-NoLLM (cosine-similarity graph, no injection) is worse still (MAE 27.1)[^src-fstllm].
- **Interpretability**: the LLM gives textual rationales and respects real-world constraints (e.g., caps parking availability at the 512-space capacity)[^src-fstllm].

## Limitations

- LLaMA-2-7B inference is slow on a single GPU (no data parallelism), so only 19 of the 320 ECL clients were used[^src-fstllm].
- Evaluated only on small graphs (≤19 nodes); the Domain Knowledge Injection fine-tunes node-by-node, so the authors argue node count does not raise fine-tuning cost, but large-scale validation is future work[^src-fstllm].

## Connections

- Task: [[few-shot-traffic-forecasting]] — few-shot / data-scarce spatio-temporal forecasting (proposed)
- Domain: [[traffic-forecasting]] — spatio-temporal forecasting on sensor networks
- Backbone: graph diffusion convolution / GTS-style STGNN; related to [[dcrnn|DCRNN]], [[mtgnn|MTGNN]]
- Contrast: [[time-llm|Time-LLM]] — reprograms a frozen LLM with generic Prompt-as-Prefix; FSTLLM instead fine-tunes with node-specific descriptions, patterns and STGNN predictions
- Contrast: [[urbangpt|UrbanGPT]] — instruction-tuned ST-LLM that replaces graphs with POI text for zero-shot; FSTLLM keeps an STGNN backbone and targets few-shot calibration
- Contrast: [[gpd|GPD]] — few-shot spatio-temporal transfer via parameter-space diffusion hypernetwork (no LLM)
- Baselines compared: [[itransformer|iTransformer]], [[patchtst|PatchTST]], [[ltsf-linear|DLinear]], GPT4TS, GMAN
- Technique: [[alpha-entmax]] — sparse softmax generalization for graph attention (adopted by FSTLLM with α=2.0)
- Technique: [[llm-enhanced-graph-construction]] — LLM encoding of node-specific text to build semantic adjacency matrices
- Technique: [[domain-knowledge-injection]] — six-component prompt + QLoRA SFT for contextual prediction calibration

[^src-fstllm]: [[source-fstllm]]
