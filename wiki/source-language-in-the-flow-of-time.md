---
title: "Language in the Flow of Time: Time-Series-Paired Texts Weaved into a Unified Temporal Narrative"
type: source-summary
tags:
  - multimodal-time-series
  - text-alignment
  - forecasting
  - imputation
  - iclr-2026
  - plug-and-play
created: 2026-05-03
last_updated: 2026-05-03
source_count: 1
confidence: medium
status: active
---

# Language in the Flow of Time

**Source**: Zihao Li, Xiao Lin, Zhining Liu, Jiaru Zou, Ziwei Wu, Lecheng Zheng, Dongqi Fu, Yada Zhu, Hendrik Hamann, Hanghang Tong, Jingrui He. "Language in the Flow of Time: Time-Series-Paired Texts Weaved into a Unified Temporal Narrative." ICLR 2026. arXiv:2502.08942.

## Core Thesis

The paper revisits multimodal time series through the **Platonic Representation Hypothesis (PRH)** (Huh et al., 2024), which posits that representations of different modalities describing the same object converge to shared latent spaces. Extending PRH, if time series and paired text both describe the same changing event, their representations are dynamic projections from a common underlying source and should exhibit similar periodic properties. The authors identify that time-series-paired texts naturally exhibit periodic properties mirroring those of the original time series — a phenomenon they term **Chronological Textual Resonance (CTR)**[^src-language-in-the-flow-of-time].

## Key Contributions

### 1. Chronological Textual Resonance (CTR)

Through Fourier analysis of three real-world datasets (Economy, Social Good, Traffic from Time-MMD), the authors demonstrate that the dominant frequencies of paired texts closely match those of the corresponding time series. Specifically, both modalities showed periodicity of 12 (frequency 0.083) for monthly-sampled data[^src-language-in-the-flow-of-time]. Three reasons are proposed: (i) shared external drivers inducing periodicity in both modalities, (ii) influence of time series dynamics on text content, and (iii) texts containing additional variables with aligned periodicity[^src-language-in-the-flow-of-time].

### 2. TT-Wasserstein Metric

A new metric based on Wasserstein distance between normalized spectra of time series and texts, designed to quantify CTR level and alignment quality[^src-language-in-the-flow-of-time]. Lower values indicate higher alignment. Validated on 9 Time-MMD datasets across monthly, weekly, and daily sampling frequencies. Shuffled datasets consistently yield much larger distances (e.g., Economy: 0.022 → 0.098/0.099 for TS-shuffled/text-shuffled), confirming the metric's sensitivity to alignment disruption[^src-language-in-the-flow-of-time]. The metric also predicts TaTS effectiveness: higher CTR (lower TT-Wasserstein) correlates with greater performance gains.

### 3. Texts as Time Series (TaTS) Framework

A plug-and-play framework that:
- Encodes paired texts using a pre-trained LLM (GPT-2 by default; also validated with BERT, RoBERTa, etc.)
- Reduces dimensionality via MLP from $d_{\text{text}}$ to $d_{\text{mapped}}$
- Concatenates text representations as auxiliary variables with the original time series: $\mathbf{U} = [\mathbf{X}; \mathbf{Z}^{\intercal}] \in \mathbb{R}^{T \times (N + d_{\text{mapped}})}$
- Feeds the augmented sequence into any existing time series model

No architecture modification is required. Jointly trains the mapping MLP and the time series model using MSE loss[^src-language-in-the-flow-of-time].

## Experimental Results

Evaluated on 18 datasets from Time-MMD, FNSPID, and FNF, integrated with 9 time series models (iTransformer, PatchTST, Crossformer, DLinear, FEDformer, FiLM, Autoformer, Informer, Transformer)[^src-language-in-the-flow-of-time]. TaTS achieves:
- Average >5% improvement on 6/9 datasets for forecasting
- >30% improvement on the largest dataset (Environment)
- Up to 30% improvement on imputation tasks (Climate, Economy, Traffic)
- Consistent gains across both short-term ({6,8,10,12}) and long-term ({48,96,192,336}) forecasting
- Outperforms covariate-based methods (N-BEATS, N-HiTS), TCN, ChatTime, and GPT4MTS[^src-language-in-the-flow-of-time]

Higher CTR levels (lower TT-Wasserstein) correlate with greater improvements. Monthly-sampled datasets (higher CTR) show larger gains than daily-sampled datasets (lower CTR)[^src-language-in-the-flow-of-time].

## Limitations

- Relies on pre-trained text encoders; performance depends on encoder quality
- TT-Wasserstein is an empirical statistical metric with estimation sensitivity
- Text modality introduces additional computational overhead
- Not all time-series-paired texts exhibit meaningful CTR (e.g., daily lottery numbers)

## Related Pages

- [[tats]] — TaTS framework entity
- [[chronological-textual-resonance]] — CTR phenomenon
- [[tt-wasserstein]] — TT-Wasserstein metric
- [[texts-as-auxiliary-variables]] — core design concept
- [[multimodal-time-series-forecasting]] — task concept
- [[vot]] — VoT (LLM reasoning approach)
- [[mindts]] — MindTS (anomaly detection)
- [[chronos]] — Chronos (tokenization approach)
- [[unica]] — UniCA (covariate adaptation)
- [[aurora]] — Aurora (generative foundation model)

[^src-language-in-the-flow-of-time]: [[source-language-in-the-flow-of-time]]