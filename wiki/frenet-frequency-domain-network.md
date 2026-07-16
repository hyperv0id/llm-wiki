---
title: "FreNet (Frequency-Domain Network)"
type: technique
tags:
  - frequency-domain
  - fft
  - spatio-temporal
  - continual-learning
  - distribution-shift
created: 2026-07-22
last_updated: 2026-07-22
source_count: 1
confidence: medium
status: active
---

# FreNet (Frequency-Domain Network)

**FreNet** (Frequency-Domain Network) is a temporal processing module introduced by [[stbp|STBP]] (ICLR 2026) that operates in the frequency domain to extract stable temporal components resilient to distributional drift[^src-stbp].

## Design

STBP employs two FreNets—one at the backbone entry and one at the exit[^src-stbp].

### Forward FreNet (Entry)

1. Map input $\mathbf{X}_\tau \in \mathbb{R}^{N_\tau \times T_h}$ through a linear layer to $\mathbf{H}_\tau \in \mathbb{R}^{N_\tau \times d}$
2. Transform to frequency domain via **Fast Fourier Transform (FFT)**
3. Apply a learnable **frequency-domain embedding** $\mathbf{F}_\tau \in \mathbb{C}^{(\frac{d}{2}+1)}$ via element-wise multiplication:

$$\mathbf{H}^f_\tau = \text{IFFT}\left(\text{FFT}(\mathbf{H}_\tau) \odot \mathbf{F}_\tau\right)$$

4. Pass through another linear layer
5. Output interacts with the contextual pattern bank's gating component $\mathbf{P}^{(0)}_\tau$ before entering the DLGA module[^src-stbp]

### Reverse FreNet (Exit)

After DLGA and feedforward processing, the second FreNet reverses the operation: restores the feature shape to $\mathbb{R}^{N_\tau \times T_h}$ for final prediction[^src-stbp].

## Why Frequency Domain?

The key insight: in evolving spatio-temporal environments, high-frequency noise varies significantly across periods, but **low-frequency components**—periodicity (e.g., daily/weekly cycles) and long-term trends—remain relatively stable[^src-stbp]. By emphasizing these stable components and suppressing high-frequency noise, FreNet provides[^src-stbp]:

1. **Robustness to distributional drift**: Stable frequency components are more resilient to distribution changes across incremental periods
2. **Computational efficiency**: FFT-based processing is computationally cheaper than RNNs (Li et al., 2018) or TCNs (Zheng et al., 2023)
3. **Stronger temporal representations**: Learned frequency embedding adaptively highlights informative frequency bands for the prediction task

## Relationship to Other Methods

Unlike traditional temporal modules that operate in the time domain:
- **RNNs** (DCRNN, MegaCRN): Sequential processing, limited parallelization
- **TCNs** (GWNet, STGCN): Local receptive field, requires deep stacking for long-range dependencies
- **Transformers**: Global context but $O(T^2)$ complexity

FreNet captures global temporal patterns in a single FFT pass with $O(T \log T)$ complexity, while providing natural robustness to distribution shifts through frequency-domain filtering[^src-stbp].

## Related Pages

- [[stbp]] — The STBP framework
- [[dlga-dual-stream-linear-graph-attention]] — DLGA, the spatial counterpart in STBP's backbone
- [[contextual-pattern-bank]] — The pattern bank that receives FreNet's output through gating
- [[continual-spatio-temporal-forecasting]] — The CSTF paradigm where distributional drift is a key challenge

[^src-stbp]: [[source-stbp]]
