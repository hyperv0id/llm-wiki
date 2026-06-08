---
title: "S-Mamba"
type: entity
tags:
  - time-series
  - mamba
  - state-space-model
  - forecasting
  - multivariate
  - mtsf
created: 2026-05-31
last_updated: 2026-06-08
source_count: 2
confidence: medium
status: active
---

# S-Mamba

**S-Mamba (Simple-Mamba)** 是首个将 [[mamba|Mamba]] 选择性状态空间模型引入多变量时间序列预测（MTSF）的 baseline 框架，由 Zihan Wang 等人在 Neurocomputing 2024 提出。它将变量间相关性编码从 Transformer self-attention 迁移到双向 Mamba block，同时用 FFN 保留时间依赖提取，在低计算开销下取得领先性能[^src-s-mamba]。

## 核心架构

S-Mamba 采用两阶段编码设计[^src-s-mamba]：

```
多变量输入 (L × M)
       │
       ▼
┌─────────────────────────┐
│  Mamba VC Encoding Layer │  ← 双向 Mamba 编码变量间相关性
│  各变量 = 独立 token      │    代替 Transformer attention
│  双向 Mamba 扫描全变量     │
└─────────────────────────┘
       │
       ▼
┌─────────────────────────┐
│   FFN TD Encoding Layer  │  ← FFN 提取每个变量的时间依赖
│  各变量内部独立处理        │    类似 iTransformer 的 FFN-on-time
└─────────────────────────┘
       │
       ▼
      预测输出 (L × T)
```

- **VC (Variable Correlation) 编码**：双向 Mamba block 扫描所有变量，像 RNN 一样依次处理各变量 token，捕获全局跨变量关系。双向设计确保每个变量"看到"所有其他变量方向的信息。
- **TD (Temporal Dependency) 编码**：FFN 在每个变量内部独立处理时间维度，挖掘时间序列的顺序模式。
- 默认各变量视为独立 channel（类似 [[channel-independence]]），跨变量交互仅发生在 Mamba VC 层。

## 实验结果

在 13 个公开数据集（PEMS03/04/07/08、Traffic、Electricity、Weather、Solar-Energy、ETTm1/m2/h1/h2、Exchange）上对比 9 个 SOTA 模型[^src-s-mamba]：

- **性能领先**：在 Traffic 相关、Electricity、Solar-Energy 等多数数据集上取得最优 MSE
- **计算高效**：GPU 内存占用 0.05-2.04 GB，训练时间低于 [[itransformer|iTransformer]]、Crossformer 等 Transformer 基线
- **变量数影响**：多变量 + 周期性强的数据集（Traffic, Electricity）优势明显；少变量 + 非周期性数据集（Exchange, ETT 系列）优势有限

## 关键发现

### Mamba vs FFN 的分工

消融实验揭示 VC 和 TD 编码的最优选择[^src-s-mamba]：

- **VC 编码**：Mamba >> Attention >> 移除。双向 Mamba 以更低开销获得更好的变量间相关性理解。
- **TD 编码**：FFN ≈ Attention > Mamba >> 移除。FFN 在时间序列内部信息提取上保持统治地位。
- 结论：S-Mamba 的 Mamba-VC + FFN-TD 组合是最优架构选择。

### 变量顺序不敏感

尽管 Mamba 的 Hippo 矩阵初始化天然倾向"邻近变量优先"，通过 Fourier 变换分离周期/非周期变量并重排序的实验证明：S-Mamba 经充分训练后能有效获取**全局**跨变量相关性，不受变量排列顺序影响[^src-s-mamba]。

### 泛化能力

仅用 40% 变量训练、预测全部 100% 变量的实验表明 Mamba 具备与 Transformer 相当的跨变量泛化能力[^src-s-mamba]。

### 可用 Mamba 提升现有模型

- 在 Reformer/Informer/Transformer 的 Encoder-Decoder 间插入 Mamba block 即获性能增益[^src-s-mamba]
- 可将 Autoformer、Flashformer、Flowformer 的 Encoder 直接替换为 uni-Mamba，GPU 内存和训练时间均降低[^src-s-mamba]

## 适用场景

- **高维多变量预测**：变量数多、周期性强的场景（如交通流量、电力负荷）
- **计算资源受限**：Mamba 的线性复杂度使其适合低 GPU 内存环境
- **需全局跨变量理解**：多变量间存在复杂的交互周期模式

## 局限

- 变量数少且非周期性强时（Exchange, ETT 系列），VC Encoding 可能引入噪声[^src-s-mamba]
- 未探索 Mamba-based 预训练 backbone（列为 Future Work）

## 相关页面

- [[mamba|Mamba]] — 选择性状态空间模型，S-Mamba 的基础构建块
- [[mila|MILA]] — Mamba 启发的线性注意力视觉模型
- [[channel-independence|Channel Independence]] — 各通道独立处理的策略
- [[patchtst|PatchTST]] — ICLR 2023，首个 CI + patching 的 Transformers 时序模型
- [[itransformer|iTransformer]] — ICLR 2024，反转 attention/FFN 维度，S-Mamba 的主要对比基线
- [[lstf|LSTF]] — 长序列时间序列预测问题设定
- [[dst-mamba|DST-Mamba]] — AAAI 2025，将 Mamba 扩展至时空交通预测，引入分解 + 空间视角的双向 Mamba
- [[gamma-net|GAMMA-Net]] — arXiv 2026，交错式 GAT + 多轴 Mamba 时空交通预测，以闭环信息流突破三难困境

## Mamba 家族演进

截至 2026 年 6 月，Mamba 在时序预测领域的演进路径：

| 模型 | 年份 | 核心创新 | Mamba 应用 |
|------|------|---------|-----------|
| S-Mamba | 2024 | 首个 Mamba MTSF baseline | 双向 Mamba 编码变量间相关性 |
| DST-Mamba | 2025 | 趋势-季节分解 + 双向 Mamba | 空间视角的季节成分 Mamba 编码 |
| GAMMA-Net | 2026 | **交错式 GAT-Mamba 闭环** | 时间轴 + 空间轴双轴 Mamba，与 GAT 交替堆叠 |

GAMMA-Net 是首次将 Mamba 以**交错闭环**方式与 GAT 结合的时空交通预测模型，通过 (GAT → Mamba_Temporal) → (GAT → Mamba_Spatial) 的交替设计，使时间理解与空间理解相互增强[^src-gamma]。

[^src-gamma]: [[source-gamma-net]]

[^src-s-mamba]: [[source-s-mamba]]
