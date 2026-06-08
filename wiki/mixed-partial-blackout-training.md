---
title: "Mixed Partial Blackout Training"
type: technique
tags:
  - time-series
  - data-imputation
  - diffusion-models
  - training-strategy
  - self-supervised-learning
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: high
status: active
---

# Mixed Partial Blackout Training (MPB)

**MPB** (Mixed Partial Blackout) 是 [[sadi|SADI]] 提出的两阶段训练策略，旨在增强扩散模型对 [[partial-blackout|partial blackout]] 缺失模式的鲁棒性[^src-sadi]。

## 训练流程

### 第一阶段：随机缺失 (RM) 预训练

从头训练 SADI，使用标准的随机缺失策略：每 epoch 从观测值中随机选取一定比例作为插补目标[^src-sadi]。此阶段使模型收敛到一个良好的局部最优，获得基础的插补能力。

### 第二阶段：混合微调

在每个训练迭代中，随机选择两种缺失模式之一[^src-sadi]：

1. **随机缺失 (RM)**：继续沿用第一阶段策略，维持基础能力
2. **Partial blackout**：随机生成缺失块——
   - 特征数 $\sim \text{Uniform}(1, \frac{K}{2})$
   - 连续步数 $\sim \text{Uniform}(1, \frac{L}{2})$
   - 随机选择具体缺失特征和起始时间步

交替暴露于两种模式确保模型既保持对随机缺失的泛化能力，又适应结构化的 partial blackout 场景[^src-sadi]。

## 为何有效

- **RM 预训练提供强基线**：避免从零开始在更困难的 partial blackout 模式上训练导致的不稳定性[^src-sadi]
- **混合暴露防止灾难性遗忘**：交替训练防止模型在适应 partial blackout 时丧失随机缺失插补能力
- **多样化缺失块增强泛化**：随机化的块参数（特征数、步数、位置）使模型见过各种严重程度的缺失模式[^src-sadi]

## 实验证据

在所有 4 个数据集上，SADI-MPB 一致优于 SADI-RM（纯随机缺失训练）[^src-sadi]：

| 数据集 | SADI-RM MSE | SADI-MPB MSE | 改善 |
|--------|------------|-------------|------|
| AgAID (3 缺失特征) | $5.98\times10^{-4}$ | $2.93\times10^{-4}$ | **~51%** |
| Air Quality (5 缺失特征) | $1.09\times10^{-3}$ | $1.07\times10^{-3}$ | ~2% |
| Electricity (10 缺失特征) | 0.137 | 0.107 | **~22%** |
| NACSE (10 缺失特征) | $5.93\times10^{-3}$ | $4.88\times10^{-3}$ | **~18%** |

Partial blackout 缺失特征数越多（AgAID 3→11），MPB 的优势越显著，验证了显式暴露于 partial blackout 模式的价值[^src-sadi]。

## 与相关训练策略对比

| 策略 | 方法 | 适用场景 |
|------|------|---------|
| [[csdi\|CSDI]] 自监督掩码 | Random/Historical/Mix/Test pattern | 随机缺失、已知缺失模式 |
| **SADI-MPB** | RM 预训练 → RM+PB 混合微调 | **未知的 partial blackout 场景** |
| [[lscd\|LSCD]] 两阶段训练 | 标准分数匹配 → 频谱一致性损失微调 | 频谱保真度优化 |

## 关联页面

- [[sadi]] — SADI，MPB 的原生模型
- [[partial-blackout]] — MPB 针对的缺失模式
- [[two-stage-imputation]] — SADI 的双阶段插补（与两阶段训练不同）
- [[csdi]] — CSDI 的自监督训练策略（对比）

[^src-sadi]: [[source-sadi]]
