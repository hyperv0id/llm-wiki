---
title: "BIGCity"
type: technique
tags:
  - spatiotemporal
  - foundation-model
  - trajectory
  - traffic-state
  - llm
  - gpt-2
  - lora
  - prompt-learning
  - universal-model
  - mtmd
created: 2026-06-01
last_updated: 2026-06-01
source_count: 1
confidence: high
status: active
---

# BIGCity

**BIGCity** (Bi-modality unIfied General model for ST data analysis in road network-based City scenarios) is the first MTMD (Multi-Task, Multi-Data modality) spatio-temporal model, proposed by Xie Yu, Jingyuan Wang et al. (Beihang University / Huawei, arXiv Dec 2024)[^src-bigcity]. It is the first model to simultaneously handle individual-level trajectory data and population-level traffic state data in a single framework with one set of parameters, covering 8 heterogeneous tasks across 3 cities and surpassing 18 independently trained baselines[^src-bigcity].

## Problem: Three-Level Silos

Existing ST models operate at three levels of generality[^src-bigcity]:

| Level | Description | Examples |
|-------|-------------|----------|
| **STSD** (Sole Task, Sole Data) | One model, one task, one data type | [[dcrnn|DCRNN]], [[gwnet|GWNet]], [[stgcn|STGCN]], individual trajectory models |
| **MTSD** (Multiple Tasks, Sole Data) | One model, multiple tasks, but only within one data modality | [[unist|UniST]], [[opencity|OpenCity]], [[urbandit|UrbanDiT]] (traffic state only); START, UniTraj (trajectory only); [[urbangpt|UrbanGPT]] (traffic state via LLM) |
| **MTMD** (Multiple Tasks, Multiple Data) | One model, multiple tasks, BOTH trajectory and traffic state data | **BIGCity (first and only)** |

Prior to BIGCity, no model could process a GPS trajectory from a taxi AND predict road-segment-level traffic speed with the same parameters — these were treated as entirely independent problems[^src-bigcity].

## Core Insight: ST-Unit

The foundational innovation is the **ST-unit**: a triple that unifies trajectory and traffic state data into the same atomic format[^src-bigcity]:

$$U_{i,\tau} = \left( \mathbf{e}_i^{(s)},\; \mathbf{e}_{i,t_\tau}^{(d)},\; \iota_\tau \right)$$

- $\mathbf{e}_i^{(s)} \in \mathbb{R}^{D_r}$ — static road segment features (ID, type, length, lanes, degree, speed limit)
- $\mathbf{e}_{i,t_\tau}^{(d)} \in \mathbb{R}^{D_d}$ — dynamic traffic state (average speed, inflow/outflow at time $t_\tau$), set to NULL when unavailable
- $\iota_\tau$ — timestamp features (absolute time, time-slice index)

Under this definition[^src-bigcity]:
- **Trajectory**: $U_{tr} = (U_{tr_1}, \dots, U_{tr_L})$ — a variable-length sequence of road segments traversed by an individual
- **Traffic state**: $U_i = (U_{i,\tau_1}, \dots, U_{i,\tau_T})$ — a fixed-length sequence of traffic states on road segment $r_i$ across time slices

Both are ST-unit sequences. The model sees no modality distinction — only sequences.

This is analogous to how ViT turned images into patch sequences and GPT turned text into token sequences for unified processing. ST-unit does the same for spatiotemporal data[^src-bigcity].

## Architecture: Four Components, One Pipeline

### 1. ST Tokenizer

Converts ST-unit sequences into ST-token sequences for LLM consumption[^src-bigcity]:

**Static Feature Encoder**: GAT on the road network graph extracts spatial topology representations.
$$H^{(s)} = \text{FFN}\left(\text{GAT}_s\left(E^{(s)}, G\right)\right), \quad H^{(s)} \in \mathbb{R}^{I \times D_h}$$

**Dynamic Feature Encoder**: GAT on time-windowed dynamic features captures instant traffic conditions.
$$H_t^{(d)} = \text{FFN}\left(\text{GAT}_d\left(\tilde{E}_t^{(d)}, \tilde{G}_t\right)\right), \quad H_t^{(d)} \in \mathbb{R}^{I \times D_h}$$

**Fusion Cross-Attention**: Global attention across ALL road segments (unlike GAT's neighbor-only).
$$s_{i,t} = \sum_{j=1}^{I} \text{softmax}\left(\frac{q_i^\top h_{j,t}}{\sqrt{2D_h}}\right)_j \cdot h_{j,t}$$

Critical for zero-shot/cross-city generalization — cross-attention captures functional similarity ("two commercial districts behave similarly") without relying on identical topology[^src-bigcity].

**ST Token Generation**:
$$x_{i,\tau} = \text{MLP}\left([s_{i,t_\tau} \| \iota_\tau \| \delta_\tau]\right)$$

Where $\delta_{\tau_l} = \tau_l - \tau_{l-1}$ encodes inter-sample time intervals — handling real-world non-uniform GPS sampling[^src-bigcity].

### 2. VMTP: Versatile Model with Task-oriented Prompt

GPT-2 (1.5B) serves as the LLM backbone, with LoRA (r=8) attached to all Q/K/V and FFN matrices. Original GPT-2 parameters are frozen[^src-bigcity].

**Prompt Structure**:
$$X = \left(X^{(txt)},\; X^{(st)},\; X^{(tsk)}\right)$$

| Component | Content | Purpose |
|-----------|---------|---------|
| $X^{(txt)}$ | Text instruction (e.g., "Predict the road segment on [CLAS] based on 'input'") | Tells model what task to perform |
| $X^{(st)}$ | ST-token sequence from ST Tokenizer | Input data |
| $X^{(tsk)}$ | [CLAS] (classification) and [REG] (regression) placeholders | Specifies output format |

Four prompt templates cover all tasks[^src-bigcity]:
- **Classification**: ST-tokens + [CLAS] → class prediction (next hop, trajectory class, user ID)
- **Regression (TTE)**: time-masked ST-tokens + [REG]×L → timestamp regression
- **Regression (traffic)**: history ST-tokens + [REG]×P → future P-step traffic states
- **Generation (recovery)**: masked ST-tokens + [CLAS]×K → road segment IDs for missing positions

### 3. LLM Backbone + LoRA

$$Z, V = \text{LLM}\left(X,\; \Phi_{\text{LoRA}}\right)$$

Only the task placeholder output tokens $Z$ feed into the task heads; $V$ (the rest) is discarded[^src-bigcity].

### 4. General-Task Heads

Three shared MLP heads serve all 8 tasks — which heads activate is determined by the prompt's placeholder arrangement[^src-bigcity]:

$$\hat{\mathbf{y}}_k^{(clas)} = \text{MLP}_c(z_k^{(clas)}), \quad \hat{y}_k^{(tim)} = \text{MLP}_t(z_k^{(reg)}), \quad \hat{\mathbf{y}}_k^{(reg)} = \text{MLP}_r(z_k^{(reg)})$$

## Training: Two Stages

**Stage 1 — Masked Reconstruction Training**[^src-bigcity]:
- Randomly mask K positions in ST-unit sequences
- Model reconstructs road segment ID (CE loss), dynamic features (MSE), and timestamp (MSE)
- Trains ST Tokenizer + LoRA simultaneously
- Analogous to MAE pre-training but on ST-units
- No task labels required

**Stage 2 — Task-oriented Prompt Tuning**[^src-bigcity]:
- All 8 tasks' training data mixed into one unified dataset
- ST Tokenizer frozen; only LoRA updated
- Joint training across all tasks with combined loss $\mathcal{L}_{PT} = \mathcal{L}_{CLAS} + \lambda_2 \mathcal{L}_{REG} + \lambda_3 \mathcal{L}_{GEN}$
- Cross-modal task pairs (e.g., Next Hop + Multi-Step traffic prediction) show stronger mutual benefit than same-modal pairs

## Experimental Evidence

**Datasets** (all with OSM road networks)[^src-bigcity]:

| City | Trajectories | Users | Road Segments | Source |
|------|-------------|-------|---------------|--------|
| Beijing (BJ) | 1,018,312 | 1,677 | 40,306 | Taxi GPS (Nov 2015) |
| Xi'an (XA) | 384,618 | 26,787 | 5,269 | DiDi GAIA (Nov 2018) |
| Chengdu (CD) | 559,729 | 48,295 | 6,195 | DiDi GAIA (Nov 2018) |

**Trajectory Tasks** (7 baselines): TTE, Next Hop Prediction, Trajectory Classification, Most Similar Trajectory Search, Trajectory Recovery (85%-95% mask rates)[^src-bigcity].

**Traffic State Tasks** (7 baselines): One-Step Prediction, Multi-Step Prediction (6 steps = 3h), Traffic State Imputation (25% masked)[^src-bigcity].

**Key Results**[^src-bigcity]:
- BJ-TTE: MAE 8.87 vs START 9.16 (↓3.1%), MAPE 30.34 vs JGRM 39.51 (↓23.2%)
- BJ-Next: ACC@1 0.751 vs JGRM 0.746
- XA-TTE: MAE 1.72 vs START 1.83 (↓6.0%), RMSE ↓12.3%
- XA-Multi-Step: MAE 1.16 vs MTGNN 1.22 (↓5.0%)
- XA-Imputation: MAPE 6.67 vs SSTBAN 11.23 (↓40.6%) — masked reconstruction training transfers directly
- Trajectory Recovery (95% mask): ACC 0.368 vs RNTrajRec 0.338

**Cross-City Generalization** (BJ trained → XA/CD, only last MLP + head fine-tuned)[^src-bigcity]:
- XA: avg performance loss <7%
- CD: avg performance loss <6%
- BIGCity-BJ still outperforms from-scratch baselines on XA and CD

**Ablation Ranking** (by impact)[^src-bigcity]:
1. w/o-Pro (remove Task-oriented Prompt): **10.5%** avg degradation — by far the largest
2. w/o-Dyn+Fus (remove dynamic encoder + fusion): 7.3%
3. w/o-Dyn (remove dynamic encoder): 6.9%
4. w/o-Sta+Fus (remove static encoder + fusion): 5.8%
5. w/o-Sta (remove static encoder): 4.4%

The prompt mechanism is both a task router AND a performance enabler — it enables cross-task semantic sharing during LLM attention computation[^src-bigcity].

## Design Rationale: Why This Works

1. **Unified representation eliminates modality gap**: The model doesn't "know" it's processing trajectory vs. traffic state — both are ST-unit sequences. Ablation confirms information complementarity: removing dynamic features hurts trajectory tasks (missing environmental context), removing static features hurts traffic tasks (missing topology)[^src-bigcity].

2. **Prompt mechanism solves task heterogeneity**: Instead of per-task output heads (MTSD route), the same parameters produce different behaviors via different prompts. Cross-modal joint training yields stronger mutual benefit than same-modal — individual behavioral patterns and population-level flow patterns are two views of the same system[^src-bigcity].

3. **Masked reconstruction provides universal pre-training**: Reconstructing masked ST-units teaches general spatio-temporal dependency patterns without task-specific labels. This directly transfers to imputation tasks (40.6% MAPE improvement on XA-Imputation)[^src-bigcity].

## Limitations

- **GPT-2 (2019) backbone** limits ceiling — no experiments with newer LLMs (LLaMA, GPT-Neo)[^src-bigcity]
- **Road network dependency** — requires complete OSM data; no degradation strategy for incomplete road networks[^src-bigcity]
- **Dynamic feature availability** — BJ experiments used NULL dynamic features due to sparse trajectories; w/o-Dyn ablation shows this costs ~12% on trajectory tasks[^src-bigcity]
- **LoRA rank sensitivity** — r≥16 causes performance degradation (overfitting); optimal r=8 may be insufficient for larger/diverse datasets[^src-bigcity]
- **Cross-cultural generalization unverified** — only China→China migration tested; performance on Paris (radial road network) or New Delhi (irregular) unknown[^src-bigcity]
- **No interpretability** — which parts of the prompt or ST-token drive specific task performance remains unknown[^src-bigcity]
- **Task coverage incompleteness** — trajectory generation, OD demand prediction, and anomaly detection not covered[^src-bigcity]

## Place in the ST Foundation Model Landscape

| Model | Year | Paradigm | Data Modality | Cross-City | LLM |
|-------|------|----------|--------------|------------|-----|
| [[unist|UniST]] | 2024 | MAE + Prompt | Traffic state (grid only) | Yes | No |
| [[opencity|OpenCity]] | 2024 | Transformer + GNN | Traffic state | Zero-shot | No |
| [[urbangpt|UrbanGPT]] | 2024 | LLM + Instruction-Tuning | Traffic state | Zero-shot | Vicuna-7b |
| [[urbandit|UrbanDiT]] | 2025 | DiT + Rectified Flow | Traffic state (grid+graph) | Zero-shot | No |
| [[uniflow|UniFlow]] | 2024 | Transformer + ST-MRA | Traffic state (grid+graph) | Yes | No |
| [[factost|FactoST]] | 2026 | Factorized (UTP+STA) | Traffic state + general TS | Zero/few/full-shot | No |
| [[urbanpg|UrbanPG]] | 2026 | Prompt-Backbone decoupled | Traffic state | Few-shot (prompt tune) | No |
| [[gpt-st|GPT-ST]] | 2023 | MAE Pre-training Plugin | Traffic state | No | No |
| **BIGCity** | **2024** | **GPT-2 + LoRA + Prompt** | **Trajectory + Traffic state** | **<7% loss** | **GPT-2** |

BIGCity is the only model in the MTMD column — all others are MTSD[^src-bigcity].

## Epistemological Significance

BIGCity's core contribution is not a specific SOTA number (though it achieves SOTA on all 8 tasks), but proving three "it can be done" claims[^src-bigcity]:

1. **Trajectory and traffic state can share an atomic representation** — ST-unit shows that individual mobility and population-level flow are two projections of the same physical system at different observation scales.

2. **Task heterogeneity can be solved by LLM prompt mechanisms, and the prompt itself IS a performance source** — not just a task router. Removing prompts forces independent MLP heads, destroying cross-task semantic sharing.

3. **Cross-modal multi-task training yields greater benefit than same-modal** — individual behavior + population patterns are complementary views; seeing both surfaces deeper spatiotemporal principles.

ST-unit may become the standard representation for spatiotemporal data, analogous to ViT's patch embedding for images and GPT's BPE tokenization for text[^src-bigcity].

## Related Analysis

- [[spatio-temporal-foundation-model-landscape]] — 时空基础模型全景分析：LLM-Based + 多数据类型路线代表

[^src-bigcity]: [[source-bigcity]]
