---
title: "Frequency Enhanced Block (FEB)"
type: technique
tags:
  - frequency-domain
  - fourier-transform
  - wavelet-transform
  - attention-replacement
  - transformer
  - time-series
  - ICML-2022
created: 2026-05-04
last_updated: 2026-05-04
source_count: 1
confidence: high
status: active
---

# Frequency Enhanced Block (FEB)

**Frequency Enhanced Block (FEB)** is a core component of [[fedformer|FEDformer]] that substitutes the standard self-attention block in Transformer architectures for time series forecasting. Instead of computing attention in the time domain, FEB projects the input into the frequency domain (Fourier or Wavelet), processes it there with learnable kernels, and projects back[^src-fedformer]. This achieves **linear $O(L)$ complexity** while better capturing global time series properties.

## FEB-f: Fourier Variant

For input $x \in \mathbb{R}^{N \times D}$[^src-fedformer]:

1. **Linear projection**: $q = x \cdot w$, where $w \in \mathbb{R}^{D \times D}$
2. **DFT**: $Q = \mathcal{F}(q) \in \mathbb{C}^{N \times D}$
3. **Random mode selection**: $\tilde{Q} = \text{Select}(Q) \in \mathbb{C}^{M \times D}$, keeping $M \ll N$ randomly chosen frequency modes (default $M=64$)
4. **Frequency-domain processing**: $\tilde{Q} \odot R$, where $R \in \mathbb{C}^{D \times D \times M}$ is a learnable kernel. The operation is: $Y_{m,d_o} = \sum_{d_i=0}^D Q_{m,d_i} \cdot R_{d_i,d_o,m}$
5. **Zero-padding** back to $\mathbb{C}^{N \times D}$
6. **Inverse DFT**: $\text{FEB-f}(q) = \mathcal{F}^{-1}(\text{Padding}(\tilde{Q} \odot R))$

**Key properties**[^src-fedformer]:
- Random mode selection is both **theoretically justified** (Theorem 1: bound on reconstruction error) and **empirically superior** to fixed low-frequency selection
- Complexity: $O(L)$ because mode indices are pre-selected, avoiding $O(L \log L)$ FFT
- The learnable kernel $R$ is randomly initialized and trained end-to-end

## FEB-w: Wavelet Variant

Uses **multiwavelet decomposition** with fixed Legendre polynomial bases, processing signals across $L$ recursive decomposition cycles (default $L=3$)[^src-fedformer]:

### Decomposition Stage (Ladder-Down)
Each cycle decimates the signal by factor 2:
1. Apply fixed Legendre wavelet decomposition matrix to decompose input into: high-frequency part $Ud$, low-frequency part $Us$, and coarsest remaining $X^{(L+1)}$
2. Three **shared FEB-f modules** independently process each part:
   - $Ud^{(l)} = A_n(d_l^n)$ — high-frequency processing
   - $Us^{(l)} = C_n(d_l^n)$ — low-frequency processing
   - $X^{(l+1)}$ passes to next level
3. At the coarsest scale $L$: $Us^L = \bar{F}(s_l^L)$ — single perceptron layer

### Reconstruction Stage (Ladder-Up)
Builds output tensor from coarsest to finest, combining $X^{(L+1)}$, $Us^{(L)}$, and $Ud^{(L)}$ at each cycle, doubling length each step.

**Advantages over FEB-f**[^src-fedformer]:
- Captures **localized structures** in time series via the time-frequency representation of wavelets
- Multi-scale processing handles patterns at different temporal resolutions
- Complementary to Fourier — performs better on some datasets (e.g., Electricity, Exchange in certain horizons)

## Role in FEDformer Architecture

In the FEDformer encoder-decoder, FEB replaces the self-attention block in both encoder and decoder layers[^src-fedformer]:

$$\text{Encoder}: S_{en}^{l,1}, \_ = \text{MOEDecomp}(\text{FEB}(X_{en}^{l-1}) + X_{en}^{l-1})$$

$$\text{Decoder}: S_{de}^{l,1}, T_{de}^{l,1} = \text{MOEDecomp}(\text{FEB}(X_{de}^{l-1}) + X_{de}^{l-1})$$

FEB processes the signal **before** the [[moe-decomposition|MOEDecomp]] block extracts seasonal and trend components.

## Comparison with Standard Self-Attention

| Aspect | Standard Self-Attention | FEB-f | FEB-w |
|:-------|:------------------------|:------|:------|
| Domain | Time domain | Frequency domain (Fourier) | Time-frequency (Wavelet) |
| Complexity | $O(L^2)$ | $O(L)$ | $O(L)$ |
| Global view | Point-wise | Spectral (global frequencies) | Multi-scale spectral |
| Local structures | Via attention weights | Limited (global DFT) | Captured (wavelet localization) |
| Parameter count | $O(D^2)$ | $O(D^2 \cdot M)$ per kernel | $3 \times$ FEB-f + wavelet matrices |

## Connections

- **[[fedformer|FEDformer]]** — the model that introduced FEB (ICML 2022)
- **[[frequency-enhanced-attention|FEA]]** — the cross-attention counterpart, also operating in frequency domain
- **[[moe-decomposition|MOEDecomp]]** — the decomposition block that follows FEB in the FEDformer pipeline
- **[[informer|Informer]]** — uses ProbSparse self-attention ($O(L \log L)$); FEB replaces this entirely
- **[[autoformer|Autoformer]]** — uses Auto-Correlation ($O(L \log L)$); FEB replaces this with frequency-domain processing
- **[[dualsformer|Dualformer]]** — improves upon FEB's fixed random selection with input-adaptive hierarchical frequency sampling

[^src-fedformer]: [[source-fedformer]]