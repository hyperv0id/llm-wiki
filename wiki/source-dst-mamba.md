---
title: "DST-Mamba — Decomposed Spatio-Temporal Mamba for Long-Term Traffic Prediction"
type: source-summary
tags:
  - traffic-prediction
  - mamba
  - state-space-model
  - spatio-temporal
  - decomposition
  - time-series
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: high
status: active
---

# Source: DST-Mamba

Sicheng He, Junzhong Ji, Minglong Lei, "Decomposed Spatio-Temporal Mamba for Long-Term Traffic Prediction," AAAI 2025.

## 核心贡献

DST-Mamba 提出了一种基于时间序列分解与 [[mamba|Mamba]] 状态空间模型的时空交通预测框架。其核心洞察：长程时空交通数据中存在复杂的**时空纠缠**（spatio-temporal entanglement）——传统分解方法（如 [[autoformer|Autoformer]]、[[fedformer|FEDformer]]）仅关注时序维度的纠缠，忽略了路网空间约束的影响。DST-Mamba 同时应对时序纠缠和空间依赖[^src-dst-mamba]。

## 架构设计

DST-Mamba 将输入交通序列 X 通过移动平均分解为趋势部分 X_TR 和季节部分 X_SE[^src-dst-mamba]：

```
X  →  AvgPool(Padding(X))  →  X_TR (趋势部分)
  →  X - X_TR              →  X_SE (季节部分)
```

- **趋势部分 → 多尺度线性预测模块**：下采样生成 m 个尺度的趋势序列，通过自上而下混合（top-down mixing）去除噪声后，用逐点线性映射聚合多尺度趋势预测 λ · Ŷ_TR[^src-dst-mamba]。
- **季节部分 → 时空 Mamba 编码器**：视图从时间视角**切换至空间视角**。节点 token 通过图聚合（EI = S · X_SE）获得，结合可学习的自适应空间嵌入（EA ∈ R^(N×DA)），输入双向 Mamba block 捕获前向/后向跨节点相关性。FFN 在节点 token 上隐式编码时序依赖[^src-dst-mamba]。

最终预测：Ŷ = Ŷ_SE + λ · Ŷ_TR。

## 关键结果

在 5 个公开数据集（Traffic、PEMS03/04/07/08）上对比 8 个 baseline（iTransformer、PatchTST、Crossformer、FEDformer、Autoformer、DLinear、S-Mamba、SOR-Mamba）。DST-Mamba 在多数数据集和预测 horizon 上取得 SOTA[^src-dst-mamba]：

- **主要优势**：空间视角的 Mamba 编码在长程预测中显著优于跨时间 Transformer（Crossformer）和纯时序方法（PatchTST, DLinear）
- **计算效率**：Mamba block 复杂度近 O(N)（N 为节点数），在长程场景下远低于 Transformer 的二次复杂度
- **消融实验**（PEMS08）：分解步骤至关重要——移除后性能显著下降；去除季节组件比去除趋势组件损失更大；双向 Mamba 优于单向（丢失一半方向信息）[^src-dst-mamba]

## 与 [[s-mamba|S-Mamba]] 的关系

S-Mamba 是首个将 Mamba 引入 MTSF 的框架（Neurocomputing 2024），重点在**变量间相关性**（VC）编码。DST-Mamba 将此扩展至**时空交通预测**场景，核心差异[^src-dst-mamba]：

| 维度 | S-Mamba | DST-Mamba |
|------|---------|-----------|
| 应用场景 | 通用 MTSF | 时空交通预测 |
| 空间建模 | 无（仅 VC） | 图聚合 + 节点 token + 自适应嵌入 |
| 时序处理 | FFN TD 编码 | 分解 + 多尺度线性 + Mamba+FFN |
| Mamba 方向 | 双向（变量维度） | 双向（空间节点维度）|
| 分解 | 无 | 显式趋势/季节分解 |

实验也印证了这种差异：DST-Mamba 在交通数据集上的优势源于其专门的空间建模能力，而 S-Mamba 在通用 MTSF 上保持竞争力。两者共同构成了 Mamba-based 时序预测从通用到专用的完整技术谱系。

## 实验设置

在单张 NVIDIA RTX 4090（24 GB）上进行实验。输入长度为 96，预测 horizon 对 Traffic 为 96/192/336/720，对 PEMS 系列为 12/24/48/96。使用 MSE loss 和 ADAM 优化器（lr=10⁻³），batch size 32，early stopping patience 10 epochs。DST-Mamba 为 encoder-only 架构，双向 Mamba block 层数从 {1,2,3,4} 中选择。趋势预测权重 λ 从 0.5–1.0 中选择。Z-Score 归一化应用于每个时间序列[^src-dst-mamba]。

## 参数敏感性

在 PEMS08 上对 4 个超参数进行敏感性分析[^src-dst-mamba]：
- **趋势权重 λ**：模型性能对 λ 相对稳定，交通动态更依赖局部波动（季节组件）——符合复杂 Mamba 建模季节 + 简单线性处理趋势的架构设计
- **下采样窗口**：过大或过小均导致信息损失，需根据数据周期特性选择
- **空间嵌入维度**：32 维最佳，平衡节点独立信息与节点依赖信息
- **Mamba block 数量**：反映模型容量，过少或过多均影响精度

## 可视化分析

DST-Mamba 的季节输出能捕获频繁波动并保留细节变化，趋势输出补充稳定的全局模式以调整最终预测范围。这验证了交通序列的季节模式与空间交互密切相关的设计假设[^src-dst-mamba]。

## 局限性

- 分解权重 λ 需人工选择（0.5–1.0），未提供自适应方案
- 自适应空间嵌入维度对节点数敏感
- 仅验证于交通领域，未测试通用 MTSF 场景
- 未探索 Mamba 作为预训练 backbone 的潜力

[^src-dst-mamba]: [[dst-mamba]]
