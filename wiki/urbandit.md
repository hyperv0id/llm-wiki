---
title: "UrbanDiT"
type: entity
tags:
  - spatiotemporal
  - foundation-model
  - diffusion-transformer
  - traffic-forecasting
  - rectified-flow
created: 2026-05-12
last_updated: 2026-05-30
source_count: 1
confidence: medium
status: active
---

**UrbanDiT**（Urban Diffusion Transformer）是清华大学电子工程系 FIB Lab（Yuan Yuan 等）提出的开放世界城市时空基础模型，发表于 **NeurIPS 2025**[^src-urbandit]。它首次将 Diffusion Transformer（DiT）架构成功扩展到城市时空领域，通过统一提示学习（Unified Prompt Learning）实现多数据类型（grid-based + graph-based）和多任务（5 种）的统一建模[^src-urbandit]。

## 核心特色

- **多数据类型统一** — 同时处理 grid-based 数据（crowd flow, taxi demand, cellular traffic 等）和 graph-based 数据（traffic speed, road networks），通过 3D CNN / GCN 统一为序列化格式[^src-urbandit]
- **五任务覆盖** — 单模型支持 Forward Prediction、Backward Prediction、Temporal Interpolation、Spatial Extrapolation、Spatio-Temporal Imputation[^src-urbandit]
- **零样本泛化** — 在未见过的城市/场景上，零样本性能超越几乎所有有训练数据的基线模型[^src-urbandit]

## 架构概览

```
输入数据 → 时空 Patching → Embedding ─┬─→ Noisy Data ─┐
                                      │                │
Data Memory Pool → Data-driven Prompt ─┤                ├→ Spatiotemporal Transformer Blocks
Task Memory Pool  → Task-specific Prompt ─┘            │  (Temporal Attn + Spatial Attn)
                                                        │  adaLN + timestep injection
                                                        └→ 去噪预测输出
```

### 关键机制

1. **数据统一化**：grid-based 数据通过 3D CNN（kernel=pt×ps×ps）patched + reshaped；graph-based 数据通过 1D CNN（时间）+ GCN（空间）处理为统一序列 $X^{N \times T}$[^src-urbandit]
2. **掩码策略**：通过掩码矩阵 $M$ 将不同任务统一为"重建缺失数据"——Forward 掩码未来、Backward 掩码过去、Interpolation 掩码中间时间点、Extrapolation 掩码未知空间区域、Imputation 随机掩码[^src-urbandit]
3. **[[unified-prompt-learning|统一提示学习]]**：三个 learnable key-value memory pools 分别捕捉时域、频域、空域模式，通过 cosine similarity 检索最相关 patterns 作为 data-driven prompts；task mask 通过 attention 生成 task-specific prompt[^src-urbandit]
4. **Rectified Flow 训练**：采用 InstaFlow 的 straightened ODE trajectory，比传统 DDPM 弯曲路径更高效，扩散步数 500、推理步数 20，实现 25 倍加速[^src-urbandit]

## 模型变体

| 变体 | Layer 数 | Hidden Size | Attention Heads |
|------|---------|-------------|-----------------|
| UrbanDiT-S | 4 | 256 | 4 |
| UrbanDiT-M | 6 | 384 | 6 |
| UrbanDiT-L | 12 | 384 | 12 |

Memory pool 容量：每池 512 个 embedding，维度 = hidden size。学习率 1e-4，最大 500 epochs（early stopping）[^src-urbandit]。

UrbanDiT-L 展现出最强的扩展行为——数据量增加时性能提升斜率（0.011）远高于 M（0.0015）和 S（0.0019），预示大模型的规模潜力[^src-urbandit]。

## 评估数据集

**Grid-based（6 个城市/领域）**[^src-urbandit]：

| 数据集 | 城市 | 类型 | 时空分辨率 | 时间段 |
|--------|------|------|-----------|--------|
| FlowSH | 上海 | Mobility flow | 20×20 / 15min | 2016/04 |
| PopBJ | 北京 | Crowd flow | 28×24 / 1h | 2021/10-11 |
| TaxiBJ | 北京 | Taxi flow | 32×32 / 30min | 2013/06-10 |
| CrowdNJ | 南京 | Crowd flow | 20×28 / 1h | 2021/02-03 |
| TaxiNYC | 纽 | Taxi flow | 10×20 / 30min | 2015/01-03 |
| PopSH | 上海 | Dynamic population | 32×28 / 1h | 2014/08 |

**Graph-based（3 个城市）**[^src-urbandit]：SpeedSH（21099 节点）、SpeedBJ（13675 节点）、SpeedNJ（13419 节点），均为 15min traffic speed。

## 关键性能

| 任务 | 性能 |
|------|------|
| Forward Prediction（grid） | 最佳，相对提升 **11.3%** |
| Backward Prediction | 超越专门训练的 CSDI **30.4%** |
| 零样本推理 | 超越几乎所有有训练数据的基线 |
| 小样本（5%/1%） | 持续超越基线 |
| 推理加速 | 25× vs DDPM（20 步 vs 500 步） |

## 与相关模型的区别

| 方面 | UrbanDiT vs |
|------|------------|
| **vs UniST** | 同实验室前身。UniST 仅支持 grid 数据 + 预测任务；UrbanDiT 扩展到 graph 数据 + 5 种任务 + rectified flow 训练[^src-urbandit] |
| **vs UrbanGPT** | UrbanGPT 基于 LLM 逐一处理传感器（7B params，174s inference）；UrbanDiT 从零训练，多传感器并行[^src-urbandit] |
| **vs MoST** | MoST 专注多模态（图像+文本+时序），UrbanDiT 单模态但多数据类型+多任务[^src-urbandit] |
| **vs CSDI** | CSDI 是扩散插补模型，UrbanDiT 在插补任务上超越 CSDI，且覆盖预测任务，25 倍更快[^src-urbandit] |

## 相关资源

- 论文：[[source-urbandit]] — NeurIPS 2025 (arXiv:2411.12164v2)
- 代码仓库：https://github.com/tsinghua-fib-lab/UrbanDiT
- [[unified-prompt-learning]] — UrbanDiT 的核心提示学习机制
- [[spatio-temporal-foundation-model]] — 时空基础模型的通用框架
- [[rectified-flow]] — 训练使用的 rectified flow 加速方法

[^src-urbandit]: [[source-urbandit]]