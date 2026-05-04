---
title: "FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting"
type: source-summary
tags:
  - time-series
  - forecasting
  - frequency-domain
  - transformer
  - fourier-transform
  - wavelet-transform
  - seasonal-trend-decomposition
  - ICML-2022
created: 2026-04-28
last_updated: 2026-05-04
source_count: 1
confidence: high
status: active
---

# FEDformer: Frequency Enhanced Decomposed Transformer

**FEDformer** (Frequency Enhanced Decomposed Transformer) is a Transformer architecture for long-term series forecasting (LSTF), proposed by Tian Zhou, Ziqing Ma, Qingsong Wen, Xue Wang, Liang Sun, and Rong Jin from Alibaba DAMO Academy, published at **ICML 2022**[^src-fedformer]. It combines seasonal-trend decomposition with frequency-domain attention mechanisms, achieving state-of-the-art forecasting accuracy with $O(L)$ linear complexity—a significant improvement over standard Transformer's $O(L^2)$.

## Problem & Motivation

Transformer-based forecasting methods make point-wise predictions independently per timestep, failing to maintain the **global properties** and statistics of time series (overall trend, distribution). The paper demonstrates this via Figure 1: on the ETTm1 dataset, vanilla Transformer's predicted time series has a visibly different distribution from the ground truth[^src-fedformer].

Two key ideas address this[^src-fedformer]:
1. **Seasonal-trend decomposition** — captures the global profile; Transformers handle detailed structures
2. **Frequency-domain operation** — applying attention in the Fourier/Wavelet domain helps capture global properties

## Core Theoretical Insight: Random Fourier Mode Selection

A critical question: which frequency components to keep? Conventional wisdom says "keep low-frequency only." FEDformer challenges this[^src-fedformer]:

- **Keeping all components** → overfitting to noisy high-frequency changes
- **Keeping only low-frequency** → misses trend changes tied to important events
- **Random selection of a fixed subset** → preserves most information with fewer components

**Theorem 1** formally justifies this: for a time series Fourier matrix $A \in \mathbb{R}^{m \times d}$, with coherence measure $\mu(A) = \Omega(k/n)$, randomly selecting $s = O(k^2/\epsilon^2)$ Fourier components gives:

$$|A - P_{A'}(A)| \leq (1 + \epsilon)|A - A_k|$$

with high probability, where $A_k$ is the best rank-$k$ approximation. For real-world multivariate time series, the Fourier matrix $A$ exhibits low-rank property (variables share similar frequency components), making random selection a theoretically sound strategy[^src-fedformer].

## Architecture

FEDformer follows an encoder-decoder structure with $N$ encoders and $M$ decoders. It introduces three novel components[^src-fedformer]:

### 1. Frequency Enhanced Block (FEB) — substitutes self-attention

**FEB-f (Fourier variant):** Input $x \in \mathbb{R}^{N \times D}$ is linearly projected $q = x \cdot w$, transformed via DFT to $Q \in \mathbb{C}^{N \times D}$. Randomly selects $M$ modes ($M \ll N$), applies a learnable parameterized kernel $R \in \mathbb{C}^{D \times D \times M}$, then inverts via inverse DFT:

$$\text{FEB-f}(q) = \mathcal{F}^{-1}(\text{Padding}(\tilde{Q} \odot R)), \quad \tilde{Q} = \text{Select}(\mathcal{F}(q))$$

where $\odot$ performs channel-wise multiplication: $Y_{m,d_o} = \sum_{d_i=0}^{D} Q_{m,d_i} \cdot R_{d_i,d_o,m}$[^src-fedformer].

**FEB-w (Wavelet variant):** Uses multiwavelet decomposition with fixed Legendre polynomial bases. The input is recursively decomposed into high-frequency, low-frequency, and remaining parts. Three shared FEB-f modules process each part across $L$ decomposition cycles ($L=3$ default), with a ladder-down/ladder-up approach that decimates/enlarges by factor 2 each cycle[^src-fedformer].

### 2. Frequency Enhanced Attention (FEA) — substitutes cross-attention

**FEA-f (Fourier variant):** Queries, keys, and values are individually transformed via DFT, randomly selecting $M$ modes each. Frequency-domain attention is performed:

$$\text{FEA-f}(q, k, v) = \mathcal{F}^{-1}(\text{Padding}(\sigma(\tilde{Q} \cdot \tilde{K}^\top) \cdot \tilde{V}))$$

where $\sigma$ is softmax or tanh (performance varies per dataset), and $\tilde{Q}, \tilde{K}, \tilde{V} \in \mathbb{C}^{M \times D}$ are the selected frequency representations[^src-fedformer].

**FEA-w (Wavelet variant):** Same decomposition/reconstruction structure as FEB-w, but $q, k, v$ signals are decomposed separately using shared matrices. Each FEB-f module in FEB-w is replaced with a FEA-f module, and an additional FEA-f processes the coarsest remaining signals[^src-fedformer].

### 3. Mixture of Experts Decomposition (MOEDecomp) — learnable trend extraction

Fixed-window average pooling struggles with complex periodic patterns. MOEDecomp uses a **set of average filters of different sizes** (multiple experts) plus **data-dependent softmax weights** to combine them:

$$X_{\text{trend}} = \text{Softmax}(L(x)) \cdot (\mathcal{F}(x))$$

where $\mathcal{F}(\cdot)$ is the set of average pooling filters and $\text{Softmax}(L(x))$ produces the mixing weights[^src-fedformer]. Experiments show MOEDecomp brings **2.96% average improvement** over a single decomposition scheme[^src-fedformer].

### Encoder-Decoder Flow

The encoder progressively decomposes: `FEB → MOEDecomp → FeedForward → MOEDecomp`. The decoder additionally includes a cross-attention step: `FEB → MOEDecomp → FEA(encoder output) → MOEDecomp → FeedForward → MOEDecomp`. Trend components $T_{de}^l$ are accumulated: $T_{de}^l = T_{de}^{l-1} + \sum_{i=1}^{3} W_{l,i} \cdot T_{de}^{l,i}$. Final prediction: $\text{output} = W_S \cdot X_{de}^M + T_{de}^M$[^src-fedformer].

## Complexity

| Model | Training Time | Training Memory | Testing Steps |
|:------|:--------------|:----------------|:--------------|
| **FEDformer** | **$O(L)$** | **$O(L)$** | **1** |
| Autoformer | $O(L \log L)$ | $O(L \log L)$ | 1 |
| Informer | $O(L \log L)$ | $O(L \log L)$ | 1 |
| Transformer | $O(L^2)$ | $O(L^2)$ | $L$ |

FEDformer achieves linear complexity by[^src-fedformer]:
1. Pre-selecting $M=64$ Fourier modes before DFT/inverse DFT (avoids $O(L \log L)$ FFT cost)
2. Fixed recursive depth $L=3$ for wavelet variant
3. The frequency-domain operation **decouples** sequence length from attention matrix dimension

## Experiments

Six benchmark datasets covering energy, economics, traffic, weather, and disease[^src-fedformer]:

| Dataset | Length | Dim | Frequency |
|:--------|:------|:----|:----------|
| ETTm2 | 69,680 | 8 | 15 min |
| Electricity | 26,304 | 322 | 1H |
| Exchange | 7,588 | 9 | 1 Day |
| Traffic | 17,544 | 863 | 1H |
| Weather | 52,696 | 22 | 10 min |
| ILI | 966 | 8 | 7 Days |

Settings: input length $I=96$ ($I=36$ for ILI), prediction horizons $O \in \{96, 192, 336, 720\}$ ($\{24, 36, 48, 60\}$ for ILI). Baselines: Autoformer, Informer, LogTrans, Reformer[^src-fedformer].

### Multivariate Results

FEDformer achieves **best performance on all 6 benchmarks at all horizons**[^src-fedformer]:
- **14.8% overall MSE reduction** vs. Autoformer
- On Exchange and ILI (low periodicity): improvement >20%
- Consistent improvement across varying horizons

Notable: Exchange dataset lacks clear periodicity, yet FEDformer still excels—frequency-domain attention captures structures invisible to time-domain methods[^src-fedformer].

### Univariate Results

- **22.6% overall MSE reduction** vs. Autoformer
- Traffic and Weather: improvement >30%
- FEDformer-f and FEDformer-w perform differently across datasets, making them **complementary choices**[^src-fedformer]

### Ablation Studies

Three FEDformer variants tested against Autoformer baseline on ETTm1/ETTm2[^src-fedformer]:

| Variant | Self-Att | Cross-Att | Improvement |
|:--------|:---------|:----------|:------------|
| V1 (FEB only) | FEB-f | AutoCorr | 10/16 cases |
| V2 (FEA only) | AutoCorr | FEA-f | 12/16 cases |
| V3 (FEA both) | FEA-f | FEA-f | — |
| **Full FEDformer** | **FEB-f** | **FEA-f** | **16/16 cases** |

### Random vs. Fixed Mode Selection (Figure 6)

Across $M \in \{2, 4, 8, \dots, 256\}$ on ETT full-benchmark, random selection **consistently outperforms** fixed low-frequency selection. Random policy also exhibits **mode saturation**—an appropriate number of modes (not all) yields best performance, consistent with theoretical analysis[^src-fedformer].

### Kolmogorov-Smirnov (KS) Distribution Test

Applied to ETTm1 and ETTm2: testing whether predicted and input sequences come from the same distribution[^src-fedformer]:| Model | All P-value > 0.01? | Interpretation |
|:------|:-------------------|:---------------|
| Transformer | No | Rejected — different distribution |
| Informer | No | Rejected |
| Autoformer | Partial (ETTm2 only) | Close |
| **FEDformer** | **Yes (all cases)** | **Cannot reject** — same distribution |

FEDformer is the **only model** where the null hypothesis cannot be rejected ($P > 0.01$) in all cases, confirming that its outputs share the same distribution as inputs—directly validating the design motivation[^src-fedformer].

### MOEDecomp vs. Single Decomposition

The mixture-of-experts decomposition scheme brings an average of **2.96% improvement** over single moving-average decomposition on ETT and Weather datasets[^src-fedformer].

## Key Differences from Autoformer

Since both share a decomposed encoder-decoder architecture, the authors explicitly contrast them[^src-fedformer]:

| Aspect | Autoformer | FEDformer |
|:-------|:-----------|:----------|
| Core mechanism | Auto-correlation (sub-sequence similarity) | Frequency-domain attention |
| Domain | Time domain (sub-sequences) | Frequency domain (Fourier/Wavelet modes) |
| Feature extraction | Top-k sub-sequence correlation | Global frequency features from whole sequence |
| Fourier usage | Accelerate auto-correlation computation | Primary representation learning |
| Complexity | $O(L \log L)$ | $O(L)$ |

## Critique & Legacy

### Strengths
FEDformer was a **seminal work** that proved frequency-domain operations can be both more accurate and more efficient for time series Transformers[^src-fedformer]. Key contributions that stood the test of time:
1. **Random Fourier mode selection** — theoretically (Theorem 1) and empirically (Figure 6) validated, challenging the "low-pass = better" conventional wisdom
2. **$O(L)$ linear complexity** — the first Transformer-based LSTF model to achieve this, later matched by FreTS and SparseTSF
3. **KS distribution test** — rigorous quantitative evidence that decomposition preserves input distribution
4. **Fourier + Wavelet dual variants** — complementary representations for different data characteristics
5. **MOEDecomp** — learnable decomposition that inspired later adaptive decomposition approaches

### Limitations
1. **Fixed random mode selection is input-agnostic** — not adaptive to data characteristics. Later addressed by [[dualsformer|Dualformer]]'s periodicity-aware hierarchical frequency sampling and [[hyperd|HyperD]]'s explicit periodicity decoupling[^src-fedformer]
2. **Fourier vs. Wavelet choice requires dataset-specific tuning** — no automated variant selection mechanism
3. **Baselines limited to 2021 models** — did not include PatchTST, DLinear, or iTransformer (published later)
4. **ILI dataset shows high absolute errors** despite relative improvement (small dataset, 966 samples)
5. **MOEDecomp adds architectural complexity** — the benefit (2.96%) is modest relative to the added parameters

### Influence
FEDformer inspired an entire line of frequency-domain forecasting models: [[source-frets|FreTS]] (frequency-domain MLPs, NeurIPS 2023), [[dualsformer|Dualformer]] (adaptive frequency sampling), [[source-afe-tfnet|AFE-TFNet]], FiLM, and FITS. It also established the **decomposition + efficient attention** paradigm that Informer pioneered and Autoformer refined, pushing it to the limit of linear complexity[^src-fedformer].

[^src-fedformer]: [[source-fedformer]]