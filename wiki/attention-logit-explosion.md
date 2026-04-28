---
title: "Attention Logit Explosion"
type: technique
tags:
  - transformer
  - attention
  - stability
  - training-dynamics
created: 2026-04-28
last_updated: 2026-04-28
source_count: 1
confidence: high
status: active
---

# Attention Logit Explosion

注意力 logit 爆炸是指在 Transformer 训练过程中，注意力分数的值无限增长，导致训练不稳定的现[^src-quest]。

## 现象描述

标准注意力计算：
$$A = \text{softmax}\left(\frac{1}{\sqrt{d}}QK^T\right)$$

当查询和键的范数 $\|q_i\|$ 和 $\|k_j\|$ 任意增长时：
$$\frac{1}{\sqrt{d}}\|q_i\|\|k_j\| \rightarrow \infty$$

导致：
- softmax 输入值极大
- 注意力分布趋于 one-hot（熵崩溃）
- 梯度爆炸或消失
- 训练崩溃[^src-quest]

## 根本原因

### 1. 虚假模式学习
当数据中存在易于学习的虚假相关性时：
- 模型学习"捷径"（位置 + 偏差向量）
- 相关键的范数增长
- 注意力集中于少数 token
- 形成正反馈循环[^src-quest]

### 2. Q-K 交叉依赖
- 高键范数 → 高注意力 → 大梯度 → 相关查询范数增长
- 查询范数增长 → 所有 logit 放大 → 进一步强化注意力集中
- 最终导致注意力熵崩溃[^src-quest]

### 3. 缺乏约束
- 标准注意力没有约束查询/键的范数
- LayerNorm 仅作用于输入，不直接约束 Q、K
- 训练动态中范数可以无限增长[^src-quest]

## 解决方案对比

| 方法 | 机制 | 效果 |
|------|------|------|
| QKNorm | Q、K 均 ℓ2 归一化 | 稳定训练，但限制表达能力 |
| RMSNorm | 均方根归一化 | 部分稳定 |
| 初始化策略 | 精心设计初始化 | 延迟但不解决 |
| **键归一化** | **仅归一化 K** | **稳定 + 保持锐度控制** |

## 训练不稳定的信号

1. **注意力熵下降**：注意力分布趋于尖锐
2. **键/查询范数增长**：数值爆炸
3. **Loss 发散**：训练 loss 突然上升
4. **梯度异常**：梯度范数异常大/小[^src-quest]

## 预防方法

### 1. 归一化策略
- QKNorm：完全归一化
- QUEST：仅归一化键
- 保留每个查询独立控制锐度的能力[^src-quest]

### 2. 架构改进
- 限制注意力机制的结构
- 引入正则化项
- 使用更稳定的激活函数[^src-quest]

### 3. 训练技巧
- 渐进式学习率
- 梯度裁剪
- 早停策略[^src-quest]

## 引用

[^src-quest]: [[source-quest]]