---
title: "UoMo"
type: entity
tags:
  - mobile-traffic
  - foundation-model
  - diffusion-model
  - network-optimization
  - china-mobile
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: high
status: active
---

# UoMo

UoMo (Universal model for Mobile traffic forecasting) is the **first universal foundation model for mobile traffic forecasting** in wireless networks, proposed by Tsinghua University and China Mobile, published at KDD 2025 ADS Track[^src-uomo].

## Overview

UoMo unifies three distinct mobile traffic forecasting tasks under a single framework[^src-uomo]:

- **Short-term prediction**: 48 steps of history to forecast 16 future steps, used for real-time resource allocation and access control
- **Long-term prediction**: 16 steps of history to forecast 48 future steps, used for network planning, BS deployment, and capacity expansion
- **Generation**: zero-history traffic distribution generation from contextual features only, used for greenfield network planning in regions without historical data

## Architecture

UoMo combines a **transformer-based diffusion model** backbone (inspired by DiT/Sora) with task-oriented masking and context-aware contrastive fine-tuning[^src-uomo]:

1. **Data tokenization**: Unifies heterogeneous mobile traffic data (varying time granularity, diverse spatial scope) into standardized mobile tokens
2. **Masked diffusion pre-training**: 4 task-oriented masks (short-term, long-term, generation, random) enable self-supervised learning of diverse forecasting capabilities
3. **Urban context-aware fine-tuning**: Contrastive learning integrates mobile user dynamics and dynamic POI embeddings, with FiLM-style adaptive conditioning

See [[masked-diffusion-pre-training]] and [[contrastive-diffusion-alignment]] for detailed technique pages.

## Performance

Evaluated on **9 real-world datasets** covering 4G/5G mobile traffic from 7 Chinese cities plus Munich (Germany), against 13 baselines including statistical models (HA, ARIMA), NLP-based (Time-LLM, Tempo), spatio-temporal (CSDI, TimeGPT, Lagllama, PatchTST, UniST), and mobile-specific models (SpectraGAN, KEGAN, ADAPTIVE, Open-Diff)[^src-uomo]:

| Task | Avg Improvement |
|------|----------------|
| Long-term prediction | 27.85% RMSE |
| Short-term prediction | 18.57% RMSE |
| Generation | 15.6% JSD/MAE |

## Zero/Few-Shot Learning

UoMo exhibits strong generalization to unseen cities. On Munich (Germany) and Hangzhou datasets never seen during training, zero-shot performance surpasses Open-Diff after small-scale training. With only 5-10% few-shot data, UoMo achieves competitive forecasting on Fuyang, Nanning, Hangzhou, and Munich[^src-uomo].

## Scaling Properties

UoMo was evaluated at 4 scales (5M, 35M, 100M, 200M parameters). Key findings[^src-uomo]:
- Smaller models (5M-35M) improve rapidly with more parameters
- Larger models (100M-200M) show diminishing returns at fixed data sizes due to overfitting
- Larger models excel as data volume increases, leveraging extensive parameters
- Optimal model size depends on available data volume, not a universal "bigger is better" law

## Production Deployment

Deployed on China Mobile's [[jiutian-platform|Jiutian platform]], currently live in Nanning, Guangxi Province[^src-uomo]:

| Optimization Scenario | Improvement |
|----------------------|-------------|
| BS deployment: served users | +25.3% |
| BS deployment: operation cost | -18.03% |
| BS deployment: capacity deficits | -9.00% |
| BS sleep control: QoS | +21.9% |
| BS sleep control: equipment depreciation | **-40.7%** |

Trained on 4x NVIDIA A100 GPUs (80GB each). 35M-parameter variant (16 Transformer layers, hidden dim 256) used for deployment. Training time: 0.32 min/sample; inference: 0.054 min/sample.

## Related Models

UoMo is from the same **Tsinghua FIB Lab** as [[urbandit|UrbanDiT]], [[unist|UniST]], and [[uniflow|UniFlow]]. It represents the lab's extension of spatio-temporal foundation models from urban computing into mobile network domains.

## Comparison with Other Foundation Models

UoMo is the first universal model specifically for **mobile network traffic**, as opposed to general urban spatio-temporal prediction (UniST, UrbanDiT) or general time series forecasting (TimesFM, Chronos). Its key differentiator: explicit modeling of mobile network contextual factors (user count, POI distribution) via contrastive learning, and direct deployment in live network optimization.

[^src-uomo]: [[source-uomo]]
