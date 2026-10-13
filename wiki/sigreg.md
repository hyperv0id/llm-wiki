---
title: "SIGReg: Sketched Isotropic Gaussian Regularization"
type: technique
tags:
  - self-supervised-learning
  - representation-learning
  - statistics
  - hypothesis-testing
created: 2026-10-13
last_updated: 2026-10-13
source_count: 1
confidence: medium
status: active
---

# SIGReg: Sketched Isotropic Gaussian Regularization

SIGReg 将「嵌入分布对齐各向同性高斯」（见 [[isotropic-gaussian-embeddings]]）实现为具备可微性、线性复杂度并支持跨 GPU 同步的分布匹配正则：将单个 batch 的嵌入沿 M 个随机单位方向投影为 M 组标量，对每组利用统计检验度量其与标准高斯的偏离，再聚合为标量损失[^src-lejepa]。该正则项在 [[lejepa|LeJEPA]] 中负责防止表示塌缩。

## 问题

将高维嵌入分布约束到目标分布时，常规做法是估计某种散度，但其计算代价随维度上升且难以稳定进行梯度训练。SIGReg 的设计动机是引入假设检验框架充当判别准则——由检验统计量判定分布匹配是否成立——再借助 1D 投影将高维检验拆解为低维检验，以此绕开维数灾难[^src-lejepa]。

## 机制

**方向检验的充分性。** 基于超球 Cramér-Wold 引理（lemma 3），两多元分布相同当且仅当其在所有 1D 投影上的分布相同[^src-lejepa]。SIGReg 据此对单位方向 $a$ 定义单变量方向检验，并聚合 M 个方向构成全局统计量。Theorem 2 证明：对方向族取 max 的统计量是原多元零假设（$H_0: P_\theta = Q$）的有效检验，其 level 与 power 的渐近性质均成立；工程实现中将 max 替换为平均，以避免梯度稀疏地集中在个别方向[^src-lejepa]。

**检验统计量的三族对比。**

- 矩检验（Jarque-Bera 及其扩展）：匹配有限 $K$ 阶矩无法推出分布一致（Theorem 3），嵌入仍可能停留在退化分布上；若提高矩阶数则会引发训练不稳定——$k$ 阶矩的梯度范数按 $O(k)$ 增长，蒙特卡洛梯度估计方差按 $O(k^2m_{k-1}^2)$ 增长[^src-lejepa]。
- CDF 检验（Cramér-von Mises、Watson、Anderson-Darling）：需要排序导致复杂度达 $O(N\log N)$ 且破坏 SGD 的多卡并行特性，同时排序与次序统计量不可微；Kolmogorov-Smirnov 的 $\ell_\infty$ 范数会产生稀疏梯度；Shapiro-Wilk 实测不稳定[^src-lejepa]。
- 特征函数（CF）检验：经验特征函数作为复指数的样本均值天然可微，跨卡同步仅需 all_reduce 操作，且其函数值与梯度对任意输入分布均保持有界[^src-lejepa]。

论文采用 **Epps-Pulley** 统计量：$\mathrm{EP}=N\int|\hat\varphi_X(t)-\varphi(t)|^2\,w(t)\,dt$，通过加权 $\ell_2$ 范数比较经验特征函数与目标特征函数，权重取高斯 $w(t)=e^{-t^2/2}$[^src-lejepa]。该统计量享有三项理论保证：Theorem 4 证明 EP 的损失、梯度与曲率均有界；Theorem 5 给出随嵌入分布 Sobolev 光滑度 $\alpha$ 变化的统一误差上界，目标高斯分布的光滑性促使嵌入的 $\alpha$ 随训练快速增大，使 SIGReg 免受维数灾难影响；Theorem 6 指出小批量估计使期望损失包含显式 $O(1/N)$ 偏差项，其对应梯度亦具同阶偏差，实测表明 batch 规模即使小到 16 该偏差也可忽略[^src-lejepa]。

**实现。** 方向矩阵从标准正态采样后按列归一化；每个 global_step 均重新采样，并通过同步种子保证各 DDP 进程取到同一组方向；数值积分在 $t\in[-5,5]$ 上选取均匀分布的 17 点执行梯形积分[^src-lejepa]。算法全程保持 $O(N)$ 时间与内存复杂度：在 V100 上，$N=32768$、$M=512$ 且取 16 积分点时耗时约 26.4 ms（Table 6）[^src-lejepa]。

**方向数 M 与退化分布。** 在投影方向固定的情况下，增大 M 会按照 Theorem 5 上界降低期望损失；而在训练中每步重采样方向，则能在不增加计算代价的前提下获得额外的空间覆盖（Figure 7）[^src-lejepa]。受控实验表明（Figure 5）：构造各边缘均为标准高斯、协方差为单位阵的「X」形分布时，所有低维矩均与目标分布吻合，导致矩检验完全失效，但合适的投影方向仍会使统计量出现尖峰；对其进行梯度训练，SIGReg 能够捕捉退化子空间并将样本分布拉回各向同性高斯[^src-lejepa]。

## 谱系与退化情形

- 切片与分布匹配相结合的处理流程此前见于生成模型与最优传输（如 Sliced Score Matching 与 [[sliced-wasserstein-distance|Sliced Wasserstein 距离]]）；将其引入并用作 JEPA 正则是该文的做法。若对每条切片计算精确积分而非梯形数值近似，EP 值将退化为 kernel MMD 并招致平方复杂度，而这正是 SIGReg 所致力避开的计算开销[^src-lejepa]。
- 若将检验统计量替换为矩形式 $T=\mathrm{mean}^2+(\mathrm{std}-1)^2$，SIGReg 在期望意义下仅强制 $\mathbb{E}[Z]=0$ 与 $\mathrm{Cov}(Z)=I_d$，并在大切片数极限下恢复为 VICReg；论文依据 Theorem 3 明确反对该设置，指出有限矩匹配会导向 shortcut 解（见 [[vicreg]]）[^src-lejepa]。

## Related Pages

- [[lejepa]] — SIGReg 与预测损失组合成的训练目标
- [[isotropic-gaussian-embeddings]] — 目标分布的最优性来源
- [[vicreg]] — 被该框架恢复并批评的矩统计正则
- [[visreg]] — 同以 1D 切片做分布对齐的 JEPA 正则，scale/shape 解耦视角
- [[sliced-wasserstein-distance]] — 同谱系的切片式分布距离
- [[jepa]] — JEPA 架构背景

[^src-lejepa]: [[source-lejepa]]
