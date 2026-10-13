---
title: "Sliced Wasserstein Distance"
type: technique
tags:
  - optimal-transport
  - wasserstein
  - distribution-alignment
  - representation-learning
created: 2026-10-13
last_updated: 2026-10-13
source_count: 1
confidence: medium
status: active
---

# Sliced Wasserstein Distance (SWD)

**定义**：两个高维分布的 Sliced Wasserstein 距离是其在随机一维方向上投影后 1D Wasserstein 距离的平均（方法引入见 Bonneel et al. 2015，此处依据 VISReg 论文的表述）。1D 情形有闭式解：p-Wasserstein 距离等价于分位数函数之间的 Lp 距离；对 N 个经验样本，即排序后逐点距离的平均：

$$W_p^p(\hat{\mu},\hat{\nu})=\frac{1}{N}\sum_{i=1}^{N}\lVert x_{(i)}-y_{(i)}\rVert^p$$

其中 $x_{(i)}$、$y_{(i)}$ 分别为两侧经验样本的第 i 个顺序统计量[^src-visreg]。

**为何够用：Cramér-Wold 定理**。Radon 变换在所有单位球方向上的取值唯一决定一个概率测度（单射），因此 µ=ν 当且仅当二者在所有方向上的 1D 投影分布相同。高维分布对齐由此化简为「对 K 个随机 1D 投影做对齐」，每个投影只需一次排序（$O(N\log N)$）加逐点比较，避免了求解高维最优传输耦合问题的开销[^src-visreg]。

**与仓库内其它 Wasserstein 工具的分工**：

- [[bures-wasserstein|Bures-Wasserstein]] 给出两个高斯之间 W2 的闭式解，但只匹配到均值与协方差（二阶矩），无法约束高阶分布形状；
- SWD 经随机投影匹配任意形状的完整边缘分布，不假设高斯性；
- [[entropic-optimal-transport|熵正则 OT]] 与一般 [[optimal-transport|OT]] 需要迭代求解耦合问题，SWD 用「投影 + 排序」绕开。

**在 VISReg 中的用法**：把 stop-gradient 归一化后的嵌入 $\tilde{Z}\in\mathbb{R}^{N\times D}$ 投影到 K 个归一化随机方向，逐方向排序后与标准正态的固定分位数 $q_N$（$u_i=i/(N+1)$）做 L2 回归，构成 shape 正则项。由 Cramér-Wold，这严格强于协方差去相关：不仅约束维间相关性，还约束每个投影方向上的完整边缘分布形状[^src-visreg]。代价是切片数 K 与嵌入维度 D 耦合——最优精度要求 K 超过 D 一个因子 C>1（K–D 关系的实证见 [[visreg]]）[^src-visreg]。

## Related Pages

- [[visreg]] — 以 SWD 为 shape 正则的 JEPA 训练方法
- [[bures-wasserstein]] — 高斯假设下的 W2 闭式解（二阶矩对齐）
- [[joint-distribution-wasserstein-alignment]] — 时序预测中用 Wasserstein 对齐联合分布的另一用法
- [[optimal-transport]] — Wasserstein 距离的理论基础

[^src-visreg]: [[source-visreg]]
