---
title: "DiT (Diffusion Transformer)"
type: technique
tags:
  - diffusion
  - transformer
  - image-generation
  - scaling-law
  - iccv-2023
created: 2026-05-31
last_updated: 2026-06-08
source_count: 3
confidence: high
status: active
---

**DiT (Diffusion Transformer)** is the architecture proposed by Peebles & Xie (2022, ICCV 2023) that replaces the convolutional U-Net backbone in diffusion models with a Vision Transformer (ViT)[^src-dit]. DiT operates in VAE latent space and achieves SOTA ImageNet class-conditional generation via adaLN-Zero conditioning and Gflops-based scaling analysis[^src-dit].

## Overall Pipeline

DiT is a conditional diffusion model running in VAE latent space[^src-dit]:

1. **VAE Encode/Decode**: Uses Stable Diffusion's pretrained VAE (f8 downsampling, 84M params). 256x256x3 image -> 32x32x4 latent z. VAE is frozen[^src-dit].
2. **DiT Denoising Network**: Operates in latent space using DDPM-style forward/reverse diffusion, with Transformer as the backbone for epsilon_theta(z_t, t, c)[^src-dit].

## Patchify

The noisy latent z_t (32x32x4) is split into p x p patches, each linearly projected to d dimensions[^src-dit]:

```
T = (32/p)^2
```

p=2 -> T=256 tokens, p=4 -> T=64 tokens, p=8 -> T=16 tokens. Standard ViT sine-cosine positional encodings are added[^src-dit]. Patch size changes barely affect parameter count but heavily impact Gflops (self-attention is O(T^2*d))[^src-dit].

## adaLN-Zero Block (Core Innovation)

Each DiT block contains[^src-dit]:

1. Standard Layer Norm
2. Multi-Head Self-Attention (standard ViT)
3. Scale (alpha_1): per-dimension multiplication, regressed from adaLN module
4. Residual connection
5. Layer Norm + Pointwise Feedforward (2-layer MLP + GELU)
6. Scale (alpha_2) + residual connection

The adaLN module workflow[^src-dit]:
- Input: timestep t (256-dim sinusoidal -> MLP -> d-dim) + class label c (d-dim embedding)
- t + c -> SiLU -> linear layer (output d*6) -> split into beta_1, gamma_1, alpha_1, beta_2, gamma_2, alpha_2
- beta/gamma scale and shift the two Layer Norm outputs
- alpha_1 and alpha_2 multiply block output before residuals

**Zero-initialization**: The adaLN module's final layer weights and biases are zero-initialized (except gamma bias = 1), making alpha_1 = alpha_2 = 0, gamma = 1, beta = 0. At training start, each DiT block is an identity function -- a technique from Goyal et al. (2017)[^src-dit].

### Four Conditioning Schemes Compared

On DiT-XL/2 after 400K training steps[^src-dit]:

| Scheme | Gflops | FID-50K |
|--------|--------|---------|
| In-context (t/c embeddings as extra tokens) | 119.4 | 35.24 |
| Cross-attention (extra cross-attn layers) | 137.6 | 26.14 |
| adaLN (regress gamma, beta for LN) | 118.6 | 25.21 |
| **adaLN-Zero (regress gamma, beta, alpha; zero-init alpha)** | **118.6** | **19.47** |

adaLN-Zero is both the most compute-efficient and highest-quality option. Zero-initialization avoids the typical Transformer loss spike by starting from "small noise perturbation"[^src-dit].

## Model Configurations

| Config | Layers N | Hidden d | Heads | Gflops (p=4) | Params |
|--------|----------|----------|-------|-------------|--------|
| DiT-S | 12 | 384 | 6 | 1.4 | 33M |
| DiT-B | 12 | 768 | 12 | 5.6 | 130M |
| DiT-L | 24 | 1024 | 16 | 19.7 | 458M |
| DiT-XL | 28 | 1152 | 16 | 29.1 | 675M |

Gflops and params are at p=4. p=2 gives ~4x Gflops, p=8 gives ~1/4 Gflops[^src-dit].

## Gflops Scaling Analysis

Core finding from 12 DiT variants (S/B/L/XL x p=8/4/2) after 400K steps[^src-dit]:

- **FID strongly anti-correlated with Gflops**: Pearson r = -0.93, NOT correlated with parameter count
- **DiT-S/2 (6.06 Gflops) and DiT-B/4 (5.56 Gflops) have nearly identical FID** (68.40 vs 68.38), despite B/4 having 4x the parameters
- **Doubling Gflops reduces FID by ~0.3-0.35x** -- predictable, plannable scaling behavior

Engineering implication: to reduce FID under parameter constraints, decrease patch size. Under latency constraints, increase depth or width. FID is predicted by total Gflops -- both paths are equivalent[^src-dit].

### Model Compute vs Sampling Compute

Small model + more sampling steps cannot compensate: DiT-XL/2 at 128 steps (15.2 Tflops/image) achieves FID=23.7, while DiT-L/2 at 1000 steps (80.7 Tflops/image, 5x inference compute) achieves only FID=25.9[^src-dit]. Under latency constraints, prioritize larger model parameters over more sampling steps[^src-dit].

## SOTA Results

**ImageNet 256x256**[^src-dit]:

| Model | FID | sFID | IS |
|-------|-----|------|-----|
| BigGAN-deep | 6.95 | 7.36 | 171.4 |
| ADM-G, ADM-U | 3.94 | 6.14 | 215.84 |
| LDM-4-G (cfg=1.50) | 3.60 | -- | 247.67 |
| DiT-XL/2 (no CFG) | 9.62 | 6.85 | 121.50 |
| **DiT-XL/2-G (cfg=1.50)** | **2.27** | **4.60** | **278.24** |

**ImageNet 512x512**[^src-dit]:

| Model | FID | sFID | IS |
|-------|-----|------|-----|
| ADM-G, ADM-U | 3.85 | 5.86 | 221.72 |
| **DiT-XL/2-G (cfg=1.50)** | **3.04** | **5.02** | **240.82** |

DiT-XL/2 needs only 118.6 Gflops to beat LDM-4-G (103.6 Gflops, FID 3.60), while ADM requires 1120 Gflops -- DiT uses roughly 1/10 the compute of pixel-space ADM[^src-dit].

## Training Details

All diffusion hyperparameters inherited directly from ADM with zero tuning[^src-dit]:
- T=1000 steps, linear noise schedule beta=[1e-4, 2e-2]
- Epsilon-prediction L_simple + IDDPM covariance learning
- AdamW, lr=1e-4, batch size=256, EMA decay=0.9999
- No lr warmup, weight decay, dropout, or gradient clipping; only horizontal flip augmentation
- Classifier-free guidance, optimal cfg scale=1.50

Training is extremely stable: all 12 variants show no loss spikes, attributed to adaLN-Zero's identity initialization[^src-dit].

## Subsequent Impact (2023-2026)

- **Sora (OpenAI, 2024)**: Uses DiT as core architecture for video generation with 3D patchify[^src-dit]
- **Stable Diffusion 3 / Flux (2024)**: Text-to-image shifts from U-Net to DiT + text cross-attention[^src-dit]
- **PixArt-alpha (Huawei, 2023)**: First large-scale text-to-image DiT, T5 encoding + cross-attention[^src-dit]
- **[[urbandit|UrbanDiT]] (NeurIPS 2025)**: Brings DiT to urban spatiotemporal prediction[^src-dit]
- **[[timedit|TimeDiT]] (KDD 2025)**: Adapts DiT backbone to time series foundation model, replacing autoregressive forecasting with diffusion probabilistic sampling across forecasting/imputation/anomaly detection/generation[^src-timedit]
- **[[dits|DiTS]] (arXiv 2026)**: Adapts MM-DiT dual-stream architecture to time series, treating exogenous/endogenous variates as distinct modalities with joint variate attention[^src-dits]
- **Rectified Flow + DiT**: Orthogonal combination -- Rectified Flow compresses sampling steps, DiT improves per-step precision -- becoming the standard recipe for SD3/Flux[^src-dit]

## Limitations

- Only class-conditional generation; no text conditioning explored[^src-dit]
- Depends on pretrained VAE; not an end-to-end solution[^src-dit]
- Self-attention O(T^2) complexity remains a bottleneck for high resolution/video[^src-dit]

## Related Pages

- [[source-dit]] -- Paper summary
- [[diffusion-model]] -- Diffusion model fundamentals
- [[ddpm]] -- DDPM, DiT directly inherits its diffusion framework
- [[latent-diffusion-models]] -- LDM, DiT reuses its pretrained VAE
- [[classifier-free-guidance]] -- CFG, DiT's conditional generation strategy
- [[urbandit]] -- UrbanDiT, DiT extended to urban spatiotemporal prediction
- [[mae]] -- MAE, DiT's patchify + Transformer design inspired by ViT/MAE
- [[timedit]] -- TimeDiT, DiT backbone adapted for time series foundation model (KDD 2025)

- [[dits]] -- DiTS, MM-DiT dual-stream adaptation for time series forecasting (arXiv 2026)

[^src-dit]: [[source-dit]]
[^src-timedit]: [[source-timedit]]
[^src-dits]: [[source-dits]]
