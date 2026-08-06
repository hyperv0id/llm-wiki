---
title: "Time-MoE"
type: entity
tags:
  - time-series
  - foundation-model
  - mixture-of-experts
  - forecasting
created: 2026-07-25
last_updated: 2026-08-06
source_count: 3
confidence: high
status: active
---

# Time-MoE

Time-MoE 是首个将稀疏[[mixture-of-experts|混合专家]]（Sparse MoE）架构引入大规模时间序列预训练的基础模型系列，由 Xiaoming Shi、Shiyu Wang、Yuqi Nie 等提出，发表于 ICLR 2025[^src-time-moe]。《Moirai-MoE》论文将其描述为同期工作（concurrent work），两者在 [[token-level-specialization|token 级专业化]] 机制上走向了不同路线[^src-moirai-moe]。

## 架构设计

Time-MoE 采用 **decoder-only Transformer** 架构，核心创新在于将标准 FFN 层替换为稀疏 MoE 层[^src-time-moe]：

| 组件 | 设计选择 |
|------|---------|
| Token 化 | 逐点 (point-wise)，SwiGLU 嵌入 |
| 位置编码 | RoPE（旋转位置嵌入） |
| 归一化 | RMSNorm |
| 注意力 | Causal multi-head self-attention |
| 前馈网络 | 稀疏 MoE（N 个独立专家 + 1 个共享专家，Top-K 激活） |
| 预测头 | 多分辨率（horizon {1, 8, 32, 64}），贪心调度 |

与 [[moirai-moe|Moirai-MoE]] 的关键差异：Time-MoE 使用标准线性门控 + 辅助负载均衡损失，而 Moirai-MoE 使用簇基门控（k-means 聚类中心）[^src-moirai-moe]。Time-MoE 使用逐点 token 化（保留全部时序精度），Moirai-MoE 使用 patch token 化（更大感受野、更快推理）[^src-moirai-moe]。

## 模型规模

| 模型 | 激活参数 | 总参数 | 层数 | d_model |
|------|---------|--------|------|---------|
| Time-MoEbase | 50M | 113M | 12 | 384 |
| Time-MoElarge | 200M | 453M | 12 | 768 |
| Time-MoEultra | 1.1B | 2.4B | 36 | 1024 |

所有模型共享 4096 的最大上下文长度、16 个专家（含 1 个共享）、Top-2 激活[^src-time-moe]。

## 训练

- **数据**：[[time-300b|Time-300B]]，309B 时间点，9+ 领域[^src-time-moe]
- **目标**：自回归 Huber loss + 多分辨率联合优化 + 辅助负载均衡 loss[^src-time-moe]
- **配置**：AdamW、lr=1e-3、warmup 10K→cosine、BF16、FlashAttention、128×A100-80G[^src-time-moe]

## 性能

零样本预测在 6 基准上平均超越 Moirai/TimesFM/Moment/Chronos 20%+ MSE 降低。全样本微调（仅 1 epoch）平均超越 iTransformer/TimeMixer/PatchTST 等 24%。相比等激活参数量的 Dense 模型，训练成本降低 78%、推理成本降低 39%[^src-time-moe]。

## 与其他时序基础模型的对比

Time-MoE 是首个 2.4B 参数的时序基础模型。随后 [[moirai-moe|Moirai-MoE]]（ICML 2025）进一步改进了门控函数设计。两者共同确立了稀疏 MoE 作为时序基础模型缩放的核心范式[^src-moirai-moe]。

### 与 Zeus 的实测对比

[[zeus|Zeus]]（ICML 2026）论文报告了与 Time-MoEbase（约 113M 参数，同为 point tokenization）的实测效率对比：L=4096、各 1000 次运行平均、均开启 FlashAttention 时，Zeus 推理 2.1× 更快、GPU 显存节省 3.1×（实证结果，论文图 8）[^src-2607-01918]。

[^src-time-moe]: [[source-time-moe]]
[^src-moirai-moe]: [[source-moirai-moe]]
[^src-2607-01918]: [[source-2607-01918]]
