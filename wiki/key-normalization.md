---
title: "Key Normalization"
type: technique
tags:
  - transformer
  - attention
  - normalization
  - stability
created: 2026-04-28
last_updated: 2026-10-26
source_count: 3
confidence: medium
status: active
---

# Key Normalization

键归一化（Key Normalization）是一种注意力机制改进技术，通过对键向量进行 ℓ2 归一化来消除键范数对注意力的不成比例影响[^src-quest]。

## 背景：键范数的问题

在标准注意力机制中：
$$A_i = \text{softmax}\left(C\|q_i\|\|\bar{k}_j\|(\bar{q}_i \cdot \bar{k}_j)\right)$$

缩放因子 $1/\sqrt{d_k}$（参见 [[scaling-factor-sqrt-dk]]）在初始化阶段将点积方差归一化至 1，预防 Softmax 饱和[^src-bluuuuue-scaling-factor-intuition]。然而，缩放因子不约束 Q/K 的范数——训练过程中键范数仍可不受控制地增长，导致注意力"窃取"效应[^src-quest]。键范数 $\|k_j\|$ 直接影响注意力分数，导致：
- 高范数键"窃取"全局注意力
- 即使余弦相似度不高，高范数键也能获得高注意力
- 训练时键范数不受控制地增长[^src-quest]

## 键归一化的效果

通过 $\ell2$ 归一化键向量 $\bar{k}_j = k_j / \|k_j\|$：
$$A_i = \text{softmax}\left(C\|q_i\|(\bar{q}_i \cdot \bar{k}_j)\right)$$

效果：
- 注意力排名完全由余弦相似度决定
- 消除键范数的"窃取"效应
- 每个查询独立控制锐度（通过查询范数）[^src-quest]

## 与查询归一化的对比

| 方法 | 效果 | 局限性 |
|------|------|--------|
| Q 归一化 | 稳定训练 | 限制表达能力 |
| K 归一化 | 稳定训练 + 保持锐度控制 | 每个查询可独立调整 |
| Q、K 均归一化（[[qk-norm|QK-Norm]]） | 打分变为余弦相似度乘可学习全局标量 $g$[^src-henry-2020-qk-norm] | 锐度与单个查询无关，由全局 $g$ 统一设定 |

## 数学性质

### 注意力分布
- 归一化后，注意力分数仅取决于向量对齐
- 消除了幅度差异带来的偏差
- 保持余弦相似度的排序性质[^src-quest]

### 梯度影响
- 键的梯度不再受键范数影响
- 打破 Q-K 之间的交叉依赖
- 减少注意力熵崩溃的风险[^src-quest]

## 应用

- **QUEST 注意力**：核心机制是仅归一化键[^src-quest]
- **QKNorm**：同时归一化查询与键，再乘可学习标量 $g$（初值 $\log_2(L^2-L)$，$L$ 为训练序列长度的 97.5 分位），见 [[qk-norm]][^src-henry-2020-qk-norm]
- **Elliptical Attention**：结合椭球度量和键归一化[^src-quest]

## 只归一 K 与两侧都归一的分工

键归一化保留 $\lVert q_i\rVert$ 作为每个查询的锐度控制器，归一化后注意力排名完全由余弦相似度决定[^src-quest]。QK-Norm 把 Q 一并归一化，锐度不再随查询变化，改由单一全局可学习标量承担，该标量的初值由训练序列长度而非 head 维 $d$ 决定[^src-henry-2020-qk-norm]。两条路线约束的对象不同：这里约束键范数对打分的"窃取"，QK-Norm 把每一对打分的取值整体搬进固定区间。

## 引用

[^src-quest]: [[source-quest]]
[^src-bluuuuue-scaling-factor-intuition]: [[source-bluuuuue-scaling-factor-intuition]]
[^src-henry-2020-qk-norm]: [[source-henry-2020-qk-norm]]

## 相关页面

- [[scaling-factor-sqrt-dk]] — 缩放因子 $1/\sqrt{d_k}$ 控制初始化阶段的点积方差，键归一化则约束训练中的 K 范数增长，两者互补
- [[cbsa]] — CBSA 通过子空间压缩实现稳定训练，与 QUEST 的键归一化是两种不同的稳定性机制
- [[qk-norm]] — 同时归一 Q 与 K、尺度交给全局可学习标量的路线，锐度不随查询变化