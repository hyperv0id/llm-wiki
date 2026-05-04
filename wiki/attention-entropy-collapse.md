---
title: "Attention Entropy Collapse"
type: technique
tags:
  - transformer
  - attention
  - stability
  - training-dynamics
created: 2026-04-28
last_updated: 2026-05-04
source_count: 2
confidence: medium
status: active
---

# Attention Entropy Collapse

注意力熵崩溃是指 Transformer 训练过程中注意力分布趋于极端（one-hot），导致模型无法有效利用全局信息的现象[^src-quest]。

## 现象描述

注意力分布的熵：
$$H(A_i) = -\sum_j A_{ij} \log A_{ij}$$

- **高熵**：注意力均匀分布，聚合更多信息
- **低熵**：注意力集中于少数 token，视野受限[^src-quest]

熵崩溃时：
- $A_{ij} \approx 1$ 对于某个 $j$，其余接近 0
- 模型仅关注少数"捷径"特征
- 难以学习长程依赖[^src-quest]

## 触发机制

## 根本原因：点积方差膨胀

熵崩溃的数学根源是高维点积的方差膨胀性质。设 $q_i, k_i \sim N(0,1)$ 且独立，则点积 $Z = \sum_{i=1}^{d_k} q_i k_i$ 的方差为 $d_k$，标准差为 $\sqrt{d_k}$[^src-bluuuuue-scaling-factor-intuition]。当 $d_k$ 较大时，Softmax 输入的极差期望急剧扩大，饱和（输出趋近 One-hot）成为高概率事件[^src-bluuuuue-scaling-factor-intuition]。标准注意力中的缩放因子 $1/\sqrt{d_k}$（参见 [[scaling-factor-sqrt-dk]]）正是为了将该方差归一化至 1，从根本上预防熵崩溃[^src-bluuuuue-scaling-factor-intuition]。

### 1. 键范数增长
高键范数吸引更多注意力：
$$A_{ij} \propto \|q_i\|\|k_j\|(\bar{q}_i \cdot \bar{k}_j)$$

导致：
- 少数高范数键主导注意力
- 注意力分布趋于尖锐
- 相关查询范数进一步增长（正反馈）[^src-quest]

### 2. 查询范数增长
高查询范数放大所有 logit：
$$\|q_i\| \cdot \|k_j\| \cdot \cos\theta \rightarrow \text{large}$$

softmax 在大输入值下趋于 one-hot：
$$\text{softmax}(x)_j \xrightarrow{x_j \gg x_{k \neq j}} 1_j$$

### 3. 虚假模式学习
当数据中存在虚假相关性时：
- 模型学习"捷径"特征
- 注意力集中于携带虚假信号的 token
- 难以转向学习真正的语义特征[^src-quest]

## 解决方案

### 1. 键归一化（QUEST）
- 消除键范数的影响
- 注意力排名由余弦相似度决定
- 打破 Q-K 交叉依赖[^src-quest]

### 2. 查询归一化
- 限制查询范数增长
- 但会限制每个查询独立控制锐度的能力[^src-quest]

### 3. 温度缩放
- 调整 softmax 温度
- 但需要预先设定，难以自适应[^src-quest]

## 实验观察

### 玩具实验（QUEST 论文）
- 标准注意力：biased 样本的键范数增长
- QNorm：完全丢弃键范数信息，导致退化解
- QUEST：保持键范数信息，但消除其"窃取"效应[^src-quest]

### ImageNet 实验
- 标准注意力：注意力集中于少数目标区域
- QUEST：注意力更均匀分布��整个目标上
- 对抗攻击下：QUEST 更鲁棒[^src-quest]

## 与 Attention Logit Explosion 的关系

| 现象 | 原因 | 结果 |
|------|------|------|
| Attention Logit Explosion | Q、K 范数无限增长 | 数值不稳定 |
| Attention Entropy Collapse | 注意力分布趋于 one-hot | 表示能力退化 |

两者经常同时发生：
- Logit 爆炸 → softmax 趋于 one-hot → 熵崩溃
- 熵崩溃 → 少数 token 主导 → 相关 Q、K 范数增长[^src-quest]

## 引用

[^src-quest]: [[source-quest]]
[^src-bluuuuue-scaling-factor-intuition]: [[source-bluuuuue-scaling-factor-intuition]]

## 相关页面

- [[cbsa]] — CBSA 通过子空间压缩缓解注意力熵崩溃，压缩目标鼓励 token 向子空间聚集
- [[mcr2]] — MCR² 目标通过编码率度量紧凑程度，直接驱动 token 压缩行为
- [[scaling-factor-sqrt-dk]] — 缩放因子 $1/\sqrt{d_k}$ 将点积方差归一化，是防止熵崩溃的根本性预防机制
- [[spurious-patterns]] — Spurious patterns in attention that can trigger entropy collapse through shortcut learning