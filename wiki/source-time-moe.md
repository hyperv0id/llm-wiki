---
title: "Time-MoE: Billion-Scale Time Series Foundation Models with Mixture of Experts"
type: source-summary
tags:
  - time-series
  - foundation-model
  - mixture-of-experts
  - forecasting
  - pretraining
  - scaling-laws
created: 2026-07-25
last_updated: 2026-07-25
source_count: 0
confidence: low
status: active
---

# Time-MoE: Billion-Scale Time Series Foundation Models with Mixture of Experts

**Authors**: Xiaoming Shi, Shiyu Wang, Yuqi Nie, Dianqi Li, Zhou Ye, Qingsong Wen, Ming Jin (Princeton, Xiaohongshu, Squirrel Ai Learning, Griffith). ICLR 2025.

## 核心问题

时间序列基础模型预训练面临三个挑战：(1) 现有预训练模型规模有限、通用性不足；(2) 密集模型在高容量下推理成本线性增长；(3) 缩放定律在时序领域未经验证。本文首次将稀疏 MoE 引入时序基础模型，将参数量推到 2.4B。

## 核心方法：Time-MoE

**架构**：Decoder-only Transformer + Sparse MoE。三大组件：

1. **逐点 Token 嵌入**：SwiGLU 将每个时间点嵌入为 D 维向量，保证时序信息完整性，支持任意长度输入。
2. **MoE Transformer Block**：RMSNorm + RoPE + Causal Attention + 稀疏 MoE（替换 FFN）。MoE 层包含 N 个独立专家（FFN）+ 1 个共享专家，每个 token 通过 Top-K 路由激活 K 个专家。使用辅助负载均衡损失防止路由坍塌。
3. **多分辨率预测头**：4 个输出投影层对应 horizon {1, 8, 32, 64}，训练时多任务联合优化，推理时贪心调度组合实现任意长度预测。

**Time-300B 数据集**：300B+ 时间点、9+ 领域（能源/金融/医疗/自然/零售/交通/网络等），采样频率从秒到年。设计了细粒度数据清洗管线：缺失值按断点切分子序列、无效观测通过差分比阈值过滤。

**模型配置**：三档规模 — Time-MoEbase (50M 激活/113M 总参)、Time-MoElarge (200M/453M)、Time-MoEultra (1.1B/2.4B)。训练：100K 步、batch 1024、seq len 4096、Huber loss + 辅助均衡 loss、BF16 + FlashAttention、128×A100。

## 实验结果

在 6 个基准（ETT/Weather/Global Temp）上评估：
- **零样本**：vs Moirai/TimesFM/Moment/Chronos 平均 MSE 降低 20%+。Time-MoEultra 相对 Chronoslarge/Moment/Moirailarge MSE 分别降低 23%/30%/11%。
- **全样本微调**（仅 1 epoch）：vs iTransformer/TimeMixer/PatchTST 等平均 MSE 降低 24%。
- **效率**：相比等激活参数量的 Dense 模型，训练成本降 78%、推理成本降 39%。

## 消融与缩放分析

- 移除 MoE（改为密集 FFN）：MSE 从 0.262 升到 0.272
- 移除 Huber loss → MSE loss：0.262→0.267
- 移除辅助均衡 loss → 路由坍塌，0.262→0.275
- Top-K 从 1→8 变化：Top2 最优，更多 expert 性能持平但推理变慢
- 模型和数据规模持续增长均带来性能提升，验证了时序领域的缩放定律

## 局限与意义

第一次将时序基础模型推到 2.4B 参数规模，验证了稀疏 MoE 在时序预训练中的可行性和效率优势。但采样频率从秒到年级别的极端跨度可能带来 token 语义不一致问题；多分辨率头的贪心调度在超长 horizon 下可能累积误差。
