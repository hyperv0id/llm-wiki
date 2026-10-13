---
title: "Non-Contrastive SSL"
type: technique
tags:
  - self-supervised-learning
  - representation-learning
  - energy-based-model
  - collapse
created: 2026-10-13
last_updated: 2026-10-13
source_count: 1
confidence: medium
status: active
---

# Non-Contrastive SSL

非对比自监督学习是 2022 年论文四项贡献之一："一族非对比自监督学习范式，产生同时有信息量且可预测的表征"[^src-autonomous-machine-intelligence]。它针对能量式训练的核心结构性困难：压低训练样本能量容易，确保其他样本能量更高则困难。

## 塌缩：按架构分型

若仅压低训练样本能量，能量函数易塌缩为平坦常数，对所有 y 给出近似相同的能量。论文按架构梳理了塌缩条件（Figure 10）[^src-autonomous-machine-intelligence]：

| 架构 | 是否可塌缩 | 条件 |
|------|-----------|------|
| 确定性预测/回归 | 不能 | 给定 x 只产生一个 $\tilde{y}$，只有 $y=\tilde{y}$ 能量为零 |
| 生成式隐变量架构 | 可 | 隐变量信息容量过大，例如 z 维度 ≥ y 维度时可能给整个 y 空间零能量 |
| 自编码器 | 可 | $s_y$ 容量过大时学成恒等映射，重建误差在整个 y 空间为零 |
| 简单联合嵌入架构 | 可 | 编码器忽略输入、输出恒定且相等（$s_x=s_y$），整个空间零能量 |

## 两条路线

**对比方法**压低训练样本能量，同时抬高人工构造的对比样本 $(x,\hat{y})$ 能量。论文指出其两大局限：一是必须设计如何生成或挑选合适的 $\hat{y}$；二是当 y 处于高维空间且 EBM 足够柔性时，为确保能量在所有无数据方向上均更高，所需对比样本数在最坏情况下随表征维度呈指数增长（维度灾难）[^src-autonomous-machine-intelligence]。论文对该谱系的划分较宽：通过蒙特卡洛/MCMC/对比散度生成 $\hat{y}$ 的概率模型、由可训练生成器产出 $\hat{y}$ 的 GAN，以及通过损坏 y 获得 $\hat{y}$ 的去噪自编码器与掩码自编码器（[[mae|MAE]]、[[videomae]]）均属对比方法[^src-autonomous-machine-intelligence]。

**正则化（非对比）方法**在压低训练样本能量的同时显式最小化低能量区域体积，使低能量区像收缩包装（shrink-wrap）般紧裹高数据密度区域。该路线更具长期潜力的理由是规避对比方法的维度灾难；既有范例包括稀疏建模、稀疏自编码器以及带噪声隐变量的自编码器（VAE）[^src-autonomous-machine-intelligence]。两条路线并不互斥，可组合使用。

在 [[jepa|JEPA]] 架构中，正则化体现为四条判据：最大化 $s_x$ 关于 x 的信息量、最大化 $s_y$ 关于 y 的信息量、使 $s_y$ 容易由 $s_x$ 预测、最小化隐变量 z 的信息量；前两条与第四条共同防止塌缩[^src-autonomous-machine-intelligence]。

## VICReg

论文将 VICReg（Bardes et al. 2021）作为非对比判据的具体实现引用（Figure 14）[^src-autonomous-machine-intelligence]：

- 先将 $s_x$、$s_y$ 经可训练 expander 非线性映射至更高维嵌入 $v_x$、$v_y$；
- **方差项**：hinge loss，维持 batch 内每个分量的标准差高于阈值；
- **协方差项**：将 $v_y$ 不同分量两两间的协方差推向零以去相关，进而使 $s_y$ 的分量近似独立；
- 第三项是表征预测误差 $D(s_y,\tilde{s}_y)$。最简实现将 predictor 取为恒等映射，表征对"把 x 变成 y 的变换"保持不变，此时 $D(s_y,\tilde{s}_y)=D(s_y,s_x)=\|s_y-s_x\|^2$。

论文指出关键对照：传统对比方法要求 batch 内不同输入的表征彼此不同（contrastive over vectors），依赖大量对比样本；VICReg 则要求 batch 内表征的不同分量彼此不同（contrastive over components），从而摆脱了同样的样本量约束[^src-autonomous-machine-intelligence]。

论文将 Barlow Twins、whitening 类方法与最大编码率下降（MCR²）归为同类，即均依靠最大化嵌入信息量防止塌缩；BYOL 同属非对比范式，但因机制不同而被单独列出[^src-autonomous-machine-intelligence]。

## 隐变量信息量的控制手段

针对判据 4，论文提出通过离散、低维、稀疏或带噪来限制 z 的容量，并归纳为四类方法：离散化/量化（VQ-VAE）、维度/秩最小化（Implicit Rank-Minimizing AE）、稀疏化（线性稀疏建模、LISTA、非线性稀疏建模）、模糊化（noisy AE、VAE 及其在控制问题中的变体）[^src-autonomous-machine-intelligence]。以离散 z 为例，若 z 有 K 个取值，则给定 x 至多 K 个 $s_y$ 可取零能量，能量面在表征空间中即为 K 个二次能量阱的最小值；若 z 维度为 d 且 $\mathrm{Pred}(s_x,z)$ 对 z 光滑，可达预测集至多构成 $s_y$ 空间中的一个 d 维流形[^src-autonomous-machine-intelligence]。

## 范围

以上均为论文的立场性论述与谱系归类，除 VICReg 机制描述外无配套实验。非对比路线在具体任务上的收益与边界，见 [[jepa]] 中 Sobal 等人报告的反例：非对比目标本身不阻止模型聚焦时间上最慢的特征。

## Related Pages

- [[jepa]] — 四条判据的架构落点与固定噪声下的失效证据
- [[contrastive-learning]] — 对比学习的一般形式与本页所述批评的对照
- [[energy-based-model]] — 能量函数、塌缩与自由能的形式化背景

[^src-autonomous-machine-intelligence]: [[source-autonomous-machine-intelligence]]
