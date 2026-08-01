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
last_updated: 2026-08-01
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

A new metric based on Wasserstein distance between normalized spectra of time series and texts, designed to quantify CTR level and alignment quality[^src-language-in-the-flow-of-time]. Lower values indicate higher alignment. Validated on 9 Time-MMD datasets across monthly, weekly, and daily sampling frequencies. Shuffled datasets generally yield much larger distances (e.g., Economy: 0.022 → 0.098/0.099 for TS-shuffled/text-shuffled), confirming the metric's sensitivity to alignment disruption[^src-language-in-the-flow-of-time]. The metric also predicts TaTS effectiveness: higher CTR (lower TT-Wasserstein) correlates with greater performance gains.

### 3. Texts as Time Series (TaTS) Framework

A plug-and-play framework that:
- Encodes paired texts using a pre-trained LLM (GPT-2 1.5B by default; also validated with BERT and LLaMA2)
- Reduces dimensionality via a three-layer MLP from $d_{\text{text}}$ to $d_{\text{mapped}}$
- Concatenates text representations as auxiliary variables with the original time series: $\mathbf{U} = [\mathbf{X}; \mathbf{Z}^{\intercal}] \in \mathbb{R}^{T \times (N + d_{\text{mapped}})}$
- Feeds the augmented sequence into any existing time series model; for forecasting, only the first N variables are extracted as predictions ($\hat{\mathbf{X}}[:N]$, Eq. 7)

No architecture modification is required. Jointly trains the mapping MLP and the time series model using MSE loss[^src-language-in-the-flow-of-time]. Code: https://github.com/iDEA-iSAIL-Lab-UIUC/TaTS.

## Experimental Results

Evaluated on 18 datasets from Time-MMD, FNSPID, and FNF, integrated with 9 time series models (iTransformer, PatchTST, Crossformer, DLinear, FEDformer, FiLM, Autoformer, [[informer|Informer]] (AAAI 2021 Best Paper), and Transformer)[^src-language-in-the-flow-of-time]. TaTS achieves:
- Average >5% improvement on 6/9 datasets for forecasting; over 30% on the largest dataset (Environment)
- Average ~14% reduction in forecasting MSE relative to uni-modal modeling (reported in the efficiency analysis, Figure 4(d)/5 and Appendix E.8)
- Up to 30% improvement on imputation tasks (Climate, Economy, Traffic)
- Consistent gains across both short-term ({6,8,10,12}) and long-term ({48,96,192,336}) forecasting
- Outperforms covariate-based methods (N-BEATS, N-HiTS), TCN, ChatTime, and GPT4MTS[^src-language-in-the-flow-of-time]

Higher CTR levels (lower TT-Wasserstein) correlate with greater improvements. Within the same sampling frequency, a lower original/shuffled TT-Wasserstein ratio correlates with larger TaTS gains (except Climate); e.g., monthly Economy (ratio 22.3%) achieves 64.80% improvement vs Security (91.5%) with 4.05%, while daily Environment (83.6%) still reaches 36.00% (Table 5)[^src-language-in-the-flow-of-time].

### Ablation Insights

- **Text encoders**: TaTS remains robust across BERT (110M), GPT-2 (1.5B), and LLaMA2 (7B), consistently outperforming uni-modal and MM-TSFLib baselines; larger encoders give slight improvement[^src-language-in-the-flow-of-time].
- **Timestamp shuffling**: Randomly shuffling texts across timestamps drops performance to matching or even worse than the uni-modal baseline (Table 7)[^src-language-in-the-flow-of-time].
- **Text dropping**: With 25% of texts randomly dropped and replaced by "no information available", TaTS stays comparable to MM-TSFLib (Table 8)[^src-language-in-the-flow-of-time].
- **Extremely noisy texts**: Dropping 40% or 80% of corrupted texts recovers performance close to the uni-modal baseline (Table 9)[^src-language-in-the-flow-of-time].
- **Efficiency**: TaTS adds only ~1% learnable parameters and ~8% training time per epoch (a three-layer MLP projection) while delivering ~14% forecasting improvement[^src-language-in-the-flow-of-time].
- **Alternative fusion**: Gated residual and cross-attention fusion modules perform comparably to the MLP projection (Table 6); linear projections already competitive when paired with strong backbones[^src-language-in-the-flow-of-time].

## Limitations

- Relies on pre-trained text encoders; performance depends on encoder quality
- TT-Wasserstein is an empirical statistical metric with estimation sensitivity
- Text modality introduces additional computational overhead
- Not all time-series-paired texts exhibit meaningful CTR (e.g., daily lottery numbers)
- Does not deeply explore more fine-grained multimodal fusion architectures; the relation between text encoder size and TaTS effectiveness is left as an open direction[^src-language-in-the-flow-of-time]

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