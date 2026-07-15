---
title: "Moirai-MoE"
type: entity
tags:
  - time-series
  - foundation-model
  - mixture-of-experts
  - forecasting
  - pretraining
created: 2026-07-20
last_updated: 2026-07-21
source_count: 1
confidence: medium
status: active
---

# Moirai-MoE

Moirai-MoE 是 Salesforce AI Research 提出的首个稀疏混合专家（Sparse Mixture of Experts）时间序列基础模型，发表于 ICML 2025[^src-moirai-moe]。它构建于 Moirai (Woo et al., 2024) 之上，核心改进是移除人为定义的频率级投影层，改为在 Transformer 内部使用 MoE 实现数据驱动的 **token 级专业化**。

## 与 Moirai 的关键差异

| 维度 | Moirai | Moirai-MoE |
|------|--------|------------|
| 专业化策略 | 频率级（多组输入/输出投影层） | Token 级（MoE 自动学习） |
| 输入投影 | 按频率分离（Monthly/Daily/Hourly 各一） | 单层统一投影 |
| FFN | 标准 FFN | MoE 层（32 experts, Top-2 激活） |
| 门控函数 | 无 | Token 簇门控（基于 dense 预训练模型聚类中心） |
| 预训练目标 | 掩码填充（masked filling） | Next-token prediction (NLL) |
| 推理 | 并行生成全部 token | 自回归逐 token 生成 |

## 模型配置

| 变体 | 层数 | d_model | 激活参数 | 总参数 |
|------|------|---------|---------|--------|
| Moirai-MoE-S | 6 | 384 | 11M | 117M |
| Moirai-MoE-B | 12 | 768 | 86M | 935M |

## 关键性能

- **In-distribution (Monash 29 datasets)**：Moirai-MoE-S 聚合 MAE 0.65 vs Moirai-S 0.78（17% 提升），超越 Moirai-B (0.71) 和 Moirai-L (0.70)[^src-moirai-moe]。
- **Zero-shot (10 datasets)**：Moirai-MoE-B 在 CRPS 和 MASE 上总体最优。Moirai-MoE-S 以 28× 更少激活参数 (11M vs 310M) 优于 Moirai-L。以 65× 更少激活参数超越 Chronos-L[^src-moirai-moe]。
- **效率**：Moirai-MoE-S 推理 273s vs Chronos-S 551s（patch size 16 vs 1 的巨大差异）[^src-moirai-moe]。

## 与 Time-MoE 的关系

Time-MoE (Shi et al., 2024) 是同期工作，也将 MoE 引入时间序列基础模型，但使用逐点 tokenization 和标准线性门控。Moirai-MoE 在零样本预测上显著优于 Time-MoE-B 和 Time-MoE-L，归因于基于 patch 的 tokenization 和簇基门控的优越性[^src-moirai-moe]。

## 局限

Moirai-MoE-L 因计算资源需求未实现。部分 expert 在推理时低利用率，剪枝留待未来[^src-moirai-moe]。MoE 推理的内存开销（需加载全部 32 experts）仍是实际部署的挑战。

[^src-moirai-moe]: [[source-moirai-moe]]
