---
title: "Temporal Query Network (TQNet)"
type: entity
tags:
  - time-series
  - forecasting
  - model
  - icml-2025
  - multivariate
created: 2026-04-28
last_updated: 2026-08-30
source_count: 2
confidence: medium
status: active
---

# Temporal Query Network (TQNet)

TQNet 是 ICML 2025 接收的多变量时序预测模型，由华南理工大学林伟伟团队提出。其核心创新是 **Temporal Query (TQ) 技术**——用周期性偏移的可学习向量作为注意力机制中的 Query，从而在单层注意力 + 浅层 MLP 的极简架构下，实现对变量间相关性的有效建模[^src-tqn]。

## 核心创新

### Temporal Query (TQ) 技术

传统自注意力（如 iTransformer）将 Q、K、V 全部从输入序列生成，完全依赖样本级相关性，容易受到噪声、极端值、缺失值等非平稳干扰的影响。

TQ 技术将这一范式颠倒：
- **Query (Q)**：来自可学习参数 θ_TQ ∈ R^{C×W}，周期性偏移提取，捕捉全局变量相关性
- **Key (K) & Value (V)**：来自原始输入 X_t，保留样本级局部信息

这种设计使注意力同时融合了**全局先验**和**局部样本信息**，从而对噪声更加鲁棒。

### 周期偏移机制

对于周期长度 W，每隔 W 个时间步提取相同的 TQ 向量片段：
```
θ_TQ^(t,L) = θ_TQ^(t+i·W,L),  i ∈ N
```

这实现了参数复用，使模型能够捕捉数据中的周期性结构，同时通过平均化降低噪声干扰。

## 架构

TQNet 极其简洁：
1. **TQ-MHA**：单层多头注意力，Query 来自 TQ 向量
2. **MLP**：两层全连接 + GeLU 激活
3. **Instance Normalization**（可选）：缓解分布漂移

## 性能

- 在 12 个真实数据集上取得 Top 2 性能（24/24 个预测指标）
- 在高维数据集（Electricity 321 通道、PEMS 170+ 通道）上优势明显
- 推理效率与 DLinear 等线性模型相当

## 与其他模型的关系

| 模型 | 变量相关性建模方式 |
|------|-------------------|
| TQNet | TQ 技术：可学习向量作为 Query |
| iTransformer | 样本级自注意力 |
| CycleNet | 可学习周期参数（无注意力） |
| PENGUIN | 周期分组注意力偏置 |
| PatchTST | 通道独立（无变量相关性） |
| [[unica|UniCA]] | 统一的协变量适应框架，可为 TQNet 添加异构协变量支持 |

## 与可学习 Query 设计家族的关系

可学习向量作为 Query 的设计在计算机视觉中有一个更早的实例：[[detr|DETR]] 将 N 个 learned positional embeddings（[[object-queries|object queries]]）输入 Transformer decoder，逐 query 并行解码目标集合（Sec 3.2）[^src-detr]。TQ 技术与 object queries 同属「Query 不来自输入、而来自可学习参数」的设计家族，但用途不同：TQ 在时序注意力中注入全局变量相关性先验[^src-tqn]，object queries 用于集合预测的非自回归并行解码[^src-detr]。（此对照为 wiki 的组织口径，非 TQNet 论文自身的定位。）

## 相关页面

- [[unified-covariate-adaptation]] — 统一协变量适应概念
- [[instance-normalization|RevIN]] — TQNet 使用 RevIN 处理分布漂移
- [[detr]] / [[object-queries]] — 可学习 Query 并行解码在目标检测中的出处实例

## 引用

[^src-tqn]: [[source-tqn]]
[^src-detr]: [[source-detr]]