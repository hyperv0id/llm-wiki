---
title: "QUEST Attention"
type: entity
tags:
  - transformer
  - attention
  - robustness
  - vision-transformer
  -iclr-2026
created: 2026-04-28
last_updated: 2026-04-28
source_count: 1
confidence: high
status: active
---

# QUEST Attention

QUEST（Query-modulated Spherical Attention）是一种 Transformer 注意力机制，通过仅对键（keys）进行 ℓ2 归一化来实现稳定的训练和更强的鲁棒性[^src-quest]。

## 数学定义

标准注意力：
$$A = \text{softmax}\left(\frac{1}{\sqrt{d}}QK^T\right)$$

QUEST 注意力：
$$A = \text{softmax}(Q\bar{K}^T)$$

其中 $\bar{K}$ 是归一化后的键向量，$C=1$（无额外缩放因子）[^src-quest]。

## 设计原则

### 1. 键归一化
- 将键约束到超球面（unit sphere）
- 消除键范数对全局注意力的"窃取"效应
- 注意力排名完全由余弦相似度决定[^src-quest]

### 2. 查询未归一化
- 每个查询独立控制其注意力锐度
- 高查询范数 → 更尖锐的注意力分布
- 低查询范数 → 更均匀的注意力分布[^src-quest]

### 3. 无额外缩放
- 不使用 $\frac{1}{\sqrt{d}}$ 缩放因��
- 简化实现，提高可解释性[^src-quest]

## 与其他注意力变体的对比

| 方法 | Q 归一化 | K 归一化 | 缩放 | 锐度控制 |
|------|---------|---------|------|---------|
| 标准注意力 | 否 | 否 | $1/\sqrt{d}$ | 通过 Q 范数 |
| QKNorm-HS | 是 | 是 | 每头可学习 | 全局统一 |
| QKNorm-DS | 是 | 是 | 每维可学习 | 全局统一 |
| **QUEST** | **否** | **是** | **无** | **每查询独立** |

## 核心洞见

### 查询范数的作用
- 缩放所有注意力 logit
- 控制该查询的注意力分布锐度
- 高范数 → 聚焦少数 token
- 低范数 → 聚合更多信息[^src-quest]

### 键范数的问题
- 高键范数会不成比例地吸引注意力
- 即使向量对齐（余弦相似度）不高，高范数键也能获得高注意力
- 这导致"全局注意力窃取"现象[^src-quest]

### 训练不稳定的根源
1. 键范数增长 → 注意力集中于少数 token
2. 高注意力 → 梯度更新增大相关查询的范数
3. 查询范数增长 → 注意力 logit 爆炸
4. 熵崩溃 → 训练不稳定[^src-quest]

## 实验验证

### 训练稳定性
- 标准注意力在 ViT-Base/Large 上训练崩溃
- QUEST 可稳定训练所有规模[^src-quest]

### 虚假模式实验
- 构建玩具数据集：biased（50%）+ unbiased（50%）
- 标准注意力：25% 成功率学习正确解
- QUEST：58% 成功率学习正确解
- QKNorm 方法：接近 0% 成功率（完全丢弃键范数信息）[^src-quest]

### 鲁棒性提升
- 对抗攻击：FGSM、PGD、SPSA、Auto 上均优于标准注意力
- 数据损坏（IN-C）：更低的 MCE
- 注意力更均匀分布在目标区域[^src-quest]

## 应用场景

- **视觉 Transformer**：ViT、DeiT、DeiT-3、CrossViT
- **语言模型**：GPT 风格模型
- **图 Transformer**：GraphGPS
- **点云模型**：PointTransformer
- **时序模型**：时间序列预测[^src-quest]

## 局限性

- 仅在视觉任务上进行了充分验证
- 线性注意力变体尚未探索
- 与其他归一化方法（如 RMSNorm）的交互待研究[^src-quest]

## 引用

[^src-quest]: [[source-quest]]