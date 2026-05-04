---
title: "Frequency Enhanced Attention (FEA)"
type: technique
tags:
  - frequency-domain
  - fourier-transform
  - wavelet-transform
  - cross-attention
  - transformer
  - time-series
  - ICML-2022
created: 2026-05-04
last_updated: 2026-05-04
source_count: 1
confidence: high
status: active
---

# Frequency Enhanced Attention (FEA)

**Frequency Enhanced Attention (FEA)** is a core component of [[fedformer|FEDformer]] that substitutes the cross-attention block between encoder and decoder in Transformer architectures for time series forecasting. It performs attention computation in the **frequency domain** rather than the time domain—transforming queries, keys, and values via Fourier or Wavelet transforms, computing attention there, and projecting back[^src-fedformer].

## FEA-f: Fourier Variant

For queries $q \in \mathbb{R}^{L \times D}$, keys $k \in \mathbb{R}^{L \times D}$, values $v \in \mathbb{R}^{L \times D}$ (where $q$ comes from decoder, $k, v$ from encoder outputs)[^src-fedformer]:

1. **Individual DFT** on each input:
   $$\tilde{Q} = \text{Select}(\mathcal{F}(q)), \quad \tilde{K} = \text{Select}(\mathcal{F}(k)), \quad \tilde{V} = \text{Select}(\mathcal{F}(v))$$
   All reduced to $\mathbb{C}^{M \times D}$ ($M=64$ default)

2. **Frequency-domain attention**:
   $$\text{FEA-f}(q, k, v) = \mathcal{F}^{-1}(\text{Padding}(\sigma(\tilde{Q} \cdot \tilde{K}^\top) \cdot \tilde{V}))$$
   where $\sigma$ is softmax or tanh activation (choice depends on dataset)

3. **Zero-padded inverse DFT** back to $\mathbb{R}^{L \times D}$

**Key properties**[^src-fedformer]:
- Random mode selection per input enables **diverse frequency representations** across $q, k, v$
- Complexity: $O(L)$ (pre-selected modes)
- The activation $\sigma$ choice matters: softmax and tanh have different convergence behavior per dataset

## FEA-w: Wavelet Variant

Uses the same recursive multiwavelet decomposition/reconstruction structure as [[frequency-enhanced-block|FEB-w]], but adapted for cross-attention[^src-fedformer]:

### Decomposition Stage
- $q, k, v$ signals are decomposed **separately** using the same fixed Legendre wavelet matrices
- Each FEB-f module in the FEB-w structure is replaced with a **FEA-f module**
- An additional FEA-f module processes the **coarsest remaining signals** $q^{(L)}, k^{(L)}, v^{(L)}$

### Reconstruction Stage
- Identical to FEB-w reconstruction: ladder-up building output tensor from coarsest to finest scale

### Why Separate Decomposition?
Unlike FEB-w (single input), FEA-w processes three distinct signals ($q, k, v$). The decomposition matrices are shared across them for parameter efficiency, but the FEA-f modules operate independently on each signal's frequency representation[^src-fedformer].

## Role in FEDformer Architecture

FEA serves as the **cross-attention bridge** between encoder and decoder[^src-fedformer]:

$$\text{Decoder}: S_{de}^{l,2}, T_{de}^{l,2} = \text{MOEDecomp}(\text{FEA}(S_{de}^{l,1}, X_{en}^N) + S_{de}^{l,1})$$

Where $S_{de}^{l,1}$ comes from the decoder's self-attention (FEB) output, and $X_{en}^N$ is the final encoder output. The [[moe-decomposition|MOEDecomp]] block then decomposes the attention output into seasonal and trend components.

## Ablation Evidence

Ablation studies on ETTm1/ETTm2 show[^src-fedformer]:

| Configuration | Self-Attention | Cross-Attention | Cases Improved |
|:--------------|:--------------|:----------------|:---------------|
| FEDformer V1 | FEB-f | AutoCorr | 10/16 |
| FEDformer V2 | AutoCorr | FEA-f | 12/16 |
| FEDformer V3 | FEA-f | FEA-f | — |
| **Full FEDformer** | **FEB-f** | **FEA-f** | **16/16** |

FEA alone (V2) already improves on 12/16 cases vs. Autoformer baseline. Combining FEB (self-attention) + FEA (cross-attention) achieves improvement in all 16 cases, validating that **frequency-domain attention benefits both self and cross attention**[^src-fedformer].

## Comparison with Standard Cross-Attention

| Aspect | Standard Cross-Attention | FEA-f | FEA-w |
|:-------|:--------------------------|:------|:------|
| Domain | Time domain | Frequency (Fourier) | Time-frequency (Wavelet) |
| $q$ source | Decoder hidden states | Same (then DFT) | Same (then wavelet) |
| $k, v$ source | Encoder outputs | Same (then DFT) | Same (then wavelet) |
| Attention computation | $\text{Softmax}(qk^\top/\sqrt{d}) \cdot v$ | $\sigma(\tilde{Q}\tilde{K}^\top) \cdot \tilde{V}$ | Multi-scale FEA-f |
| Complexity | $O(L^2)$ | $O(L)$ | $O(L)$ |
| Global alignment | Token-level | Spectral-level | Multi-scale spectral |

## Connections

- **[[fedformer|FEDformer]]** — the model that introduced FEA (ICML 2022)
- **[[frequency-enhanced-block|FEB]]** — the self-attention counterpart, substituting self-attention with frequency-domain processing
- **[[moe-decomposition|MOEDecomp]]** — the decomposition block that follows FEA in the decoder pipeline
- **[[autoformer|Autoformer]]** — uses Auto-Correlation in cross-attention; FEA replaces this with frequency-domain attention
- **[[informer|Informer]]** — uses standard cross-attention; FEA replaces this entirely
- **[[dualsformer|Dualformer]]** — improves upon FEA's fixed random frequency selection with adaptive sampling

[^src-fedformer]: [[source-fedformer]]