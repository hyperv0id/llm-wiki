---
title: "UrbanMind — Urban Dynamics Prediction with Multifaceted Spatial-Temporal Large Language Models"
type: source-summary
tags:
  - spatial-temporal
  - large-language-model
  - masked-autoencoder
  - test-time-adaptation
  - urban-dynamics
  - KDD
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: high
status: active
---

# UrbanMind (KDD 2025)

**Authors**: Anonymous (published at KDD 2025, August 3–7, Toronto). Yanhua Li supported by NSF grants IIS-1942680, CNS-1952085, DGE-2021871. Jun Luo supported by Innovation and Technology Fund ITP/012/25LP.

**Code**: [github.com/Yliu1111/UrbanMind](https://github.com/Yliu1111/UrbanMind.git)

## Core Contribution

UrbanMind is a novel spatio-temporal large language model for urban dynamics prediction that integrates three key innovations: (1) **Muffin-MAE** — a multifaceted masked autoencoder with temporal, spatial, and global masking strategies that capture intercorrelated dependencies across multiple urban dynamics; (2) **Semantic-Aware Prompting + LLM Fine-Tuning** — LLaMA3 backbone with frozen early layers and query-only fine-tuning in later layers; (3) **Test-Time Adaptation** — a masked reconstruction mechanism (reconstructor G shares layers with predictor) that adapts to distributional shifts between training and unseen testing regions at inference time.

## Architecture

**Stage 1 — Muffin-MAE**: Dual encoders E_φ₁ and E_φ₂ separately encode multifaceted dynamics X (multiple auxiliary urban dynamics) and target dynamics Xᵏ. Three masking strategies (temporal with ratio p_t=0.33, spatial with p_s=0.25, global) are applied for self-supervised reconstruction. Dual decoders D_ψ₁, D_ψ₂ reconstruct from masked embeddings. Final tokens U = concat(V, Vᵏ) fuse multifaceted and target embeddings.

**Stage 2 — LLM Fine-Tuning**: Tokens U are combined with natural language descriptions (spatial POI, temporal, task instructions) as prompts. LLaMA3 layers are split into frozen TFM_fr (layers 1..l) and trainable TFM_tr (layers l+1..L). Only query matrices W_q in self-attention of trainable layers are updated. A spatial-temporal predictor module P (self-attention + FC) maps LLM output to numerical predictions. MSE loss.

**Stage 3 — Test-Time Adaptation**: During testing, LLM output embeddings E are randomly masked with ratio p. A reconstructor G (sharing self-attention layers with the predictor) recovers masked elements via reconstruction loss L_recon = (1/n)Σ‖G(e_i^masked) − e_i‖². After few-epoch updates, adapted shared layers improve predictor accuracy for unseen regions.

## Experiments

Evaluated on 9 datasets: 3 cities (Shenzhen, Xi'an, Chengdu) × 3 urban dynamics (traffic speed, taxi inflow, travel demand). Grid-based representation (10×10 regions). Shenzhen: 162 days, 63 regions. Xi'an/Chengdu: 30 days, 4 regions. 11 baselines: DYffusion, TGC-LSTM, GCRN, GAGCN, GATGPT, GCNGPT, ST-LLM, TPLLM, LLaMA3 (raw, frozen), STG-LLM, UrbanGPT.

**Zero-shot**: UrbanMind consistently best across all 9 scenario combinations. Cross-city generalization: trained on Shenzhen → tested on Xi'an, UrbanMind achieves 8.5% lower MAE (0.194 vs 0.212) and 9.9% lower RMSE (0.236 vs 0.262) vs UrbanGPT.

**Standard prediction**: UrbanMind SOTA across all settings (Table 2).

**Ablation**: Removing Muffin-MAE causes largest degradation; removing any masking type (temporal/spatial/global) or embedding component (target/multifaceted) also degrades; removing LLM fine-tuning or test-time adaptation each causes significant drops.

## Key Findings

- Muffin-MAE's multifaceted masking is essential — temporal p_t=0.33, spatial p_s=0.25 optimal
- More trainable LLM layers and more multifaceted dynamics both improve performance
- Test-time adaptation effectively mitigates distribution shifts absent in competing LLM-based ST models
- 70.9s/epoch training, 16.5s/epoch test-time adaptation (vs UrbanGPT 80.1s/epoch)
