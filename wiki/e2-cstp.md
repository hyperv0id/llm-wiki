---
title: "E²-CSTP"
type: entity
tags:
  - spatio-temporal
  - multimodal
  - causal-inference
  - mamba
  - gcn
  - neurips
created: 2026-06-04
last_updated: 2026-07-05
source_count: 2
confidence: high
status: active
---

# E²-CSTP

**E²-CSTP** (Effective and Efficient Causal multi-modal Spatio-Temporal Prediction) is a framework proposed by Huang et al. (Zhejiang University, NeurIPS 2025) that integrates cross-modal fusion, causal inference, and a hybrid GCN-Mamba architecture for spatio-temporal prediction[^src-e2-cstp].

## Three Core Modules

### 1. Cross-Modal Feature Fusion

```
ST Sequence ──→ Normalize → Fst ──→ Q ──┐
                                          ├→ Cross-Modal Attention → Fusion Gate → Ffused
Text ──→ BERT ──→ Ftext ──→ K,V ────────┘
Image ──→ CNN ──→ Fimg ──→ K,V ─────────┘
```

- **BERT** encodes event text → contextual embeddings
- **CNN** extracts visual features from environmental images
- **Cross-modal attention**: ST features as queries, text/image features as keys and values
- **Fusion gate**: Learns dynamic weights per modality for adaptive fusion

All modalities are aligned temporally (text: regex timestamp matching; image: spatial+timestamp matching)[^src-e2-cstp].

### 2. Dual-Branch Causal Inference

E²-CSTP formalizes confounding via a Structural Causal Model (SCM)[^src-e2-cstp]:

```
S (confounder)
├──→ Xst (ST data)
├──→ Yst (target)
└──→ E (image), C (text) —— auxiliary modalities
```

The **backdoor path** Xst ← S → Yst is blocked through intervention. The backdoor-adjusted target distribution is estimated by integrating over the confounder S[^src-e2-cstp]:

```
P(Yst | do(Xst=x), E, C) = ∫_S P(Yst | Xst=x, S=sᵢ, E, C) · P(S=sᵢ | E, C) dS
```

Realized via the intervention adjustment[^src-e2-cstp]:

```
x̂ = x + x ⊙ W[α₁·h(S) + α₂·p(E) + α₃·q(C)]
∂x̂/∂S → 0  (adversarial training objective, ensures x̂ ⊥ S | E,C)
```

- **Causal matrix**: DeepSHAP estimates node-wise influence → hybrid adjacency `A = λ·A^(0) + (1-λ)·A_SHAP`, updated via EMA every 5 epochs[^src-e2-cstp]
- **Main branch**: Pure ST → minimizes spurious correlations
- **Auxiliary branch**: Multi-modal fused → captures external context
- **Final**: `ŷ_final = MLP(f(x_st, A); f(F_fused, A))`
- **Loss**: `L_all = L_pred + β·L_st + (1-β)·L_mm`, where `L_st` supervises the raw ST branch and `L_mm` the causally-adjusted multi-modal branch[^src-e2-cstp]

### 3. STED: GCN + Mamba

```
X → GCN (spatial) → ──→ LayerNorm
  → Mamba (temporal) → ─┘         } × 3 stacked layers → MLP Decoder
```

- **GCN**: O(B·T·N²·d), captures spatial neighborhood dependencies
- **Mamba**: O(B·T·N·d), linear-complexity selective SSM for temporal dynamics
- Overall complexity **O(B·T·N²·d)** vs Transformer's **O(B·T²·N²·d)**[^src-e2-cstp]

## Main Results (NeurIPS 2025)

| Dataset | Modalities | E²-CSTP MAE | Best Baseline | Improvement |
|---------|-----------|:--:|:--:|:--:|
| Terra | ST + image + text | **2.43** | 2.47 (UniST) | 1.61% |
| BjTT | Traffic + events | **3.56** | 3.62 (UniST) | 1.66% |
| GreenEarthNet | Satellite | **0.13** | 0.13 (HimNet) | tied |
| BikeNYC | Bike flow only | **2.99** | 3.31 (UniST) | **9.66%** |

Efficiency: **17.37%–56.11%** faster than Transformer baselines[^src-e2-cstp].

## Ablation: Component Contributions

| Removed Component | Impact |
|-------------------|--------|
| Text Feature | Significant on event-driven BjTT |
| Image Feature | Critical on Terra (visual context) |
| DeepSHAP | Less prior knowledge → worse causal structure |
| Causal Inference | **Largest degradation** on BjTT |
| GCN | Spatial modeling collapses |
| Mamba | Temporal modeling degrades |

All six components measurable; causal inference and spatial encoding most critical[^src-e2-cstp].

## Parameter Sensitivity

- **λ (graph fusion factor)**: 0.25 on remote-sensing tasks (leans on the SHAP-derived causal graph), 0.5 on urban traffic (balances prior structure and causal graph)[^src-e2-cstp].
- **β (loss balancing factor)**: 0.5–0.75 depending on the strength of exogenous influence in the dataset[^src-e2-cstp].

## Comparison with Related Models

| Model | Causal? | Multi-Modal? | Encoder | Complexity |
|-------|:--:|:--:|------|------|
| **E²-CSTP** | ✓ dual-branch | ✓ text+image+ST | GCN+Mamba | O(B·T·N²·d) |
| [[most|MoST]] | ✗ | ✓ image+text+location+TS | CNN+GNN+Transformer | Quadratic |
| [[conformer|ConFormer]] | ✗ (accident informed) | ✗ | GCN+Attention | Quadratic |
| UniST | ✗ | ✗ | MAE+Transformer | Quadratic |
| CaST/CauSTG | ✓ | ✗ | Single-modal causal | ~Quadratic |
| NuwaDynamics | ✓ | ✗ | Single-modal causal | ~Quadratic |

E²-CSTP is the first to combine **multi-modal fusion**, **dual-branch causal inference**, and **linear-complexity GCN+Mamba** in a unified ST framework[^src-e2-cstp].

与 E²-CSTP 通过后门路径阻断处理混杂不同，[[doflow|DoFlow]]（ICLR 2026）在已知因果 DAG 上用连续归一化流统一观测、干预与反事实时间序列预测，并假设因果充分性（无隐藏混杂）[^src-doflow]。

## Related Pages

- [[source-e2-cstp]] — source summary
- [[spatio-temporal-foundation-model]] — ST foundation model concept
- [[multimodal-time-series-forecasting]] — multimodal TS forecasting
- [[most]] — MoST multimodal ST model
- [[mamba]] — Mamba architecture
- [[conformer]] — ConFormer causality-informed

[^src-e2-cstp]: [[source-e2-cstp]]
[^src-doflow]: [[source-doflow]]
