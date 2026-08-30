---
title: "Positive Random Features (PRF)：softmax 核的正随机特征无偏估计"
type: technique
tags:
  - kernel-methods
  - random-features
  - linear-attention
  - softmax
  - iclr-2021
created: 2026-08-30
last_updated: 2026-08-30
source_count: 2
confidence: medium
status: active
---

# Positive Random Features (PRF)

Positive Random Features（PRF，正随机特征）是 Performer 论文为 softmax 核提出的随机特征估计器：只用非负特征值得到无偏估计，且在被近似核值趋 0 的区域方差趋 0，与 sin/cos 三角特征形成对照（Performer 论文 Sec 2.3）[^src-performer]。它是 [[performer|Performer]] 的 FAVOR+ 机制中 OR+ 部分的数学核心。

## 问题：softmax 核的随机特征估计

核化注意力（Performer 论文 Sec 2.2）需要把 softmax 核 SM(x,y)=exp(xᵀy) 写成期望形式 E[φ(x)ᵀφ(y)]。论文 Sec 2.3 给出的通用构造 φ(x)=(h(x)/√m)·(f₁(ω₁ᵀx),…,fₗ(ωₘᵀx))，其中 ωᵢ 从各向同性分布（通常高斯）采样；取 f₁=sin、f₂=cos 配合高斯分布即得经典 shift-invariant 核随机特征（Rahimi & Recht, 2007），加缩放 h(x)=exp(‖x‖²/2) 后构成 softmax 核的无偏估计器 SMm^trig（Sec 2.3）[^src-performer]。

注意力场景对核估计器有两个额外约束：其一，注意力是对 value 向量的凸组合，核分数必须非负，否则归一化对角阵 D⁻¹ 可出现负值（Sec 2.3）[^src-performer]；其二，注意力矩阵大量条目对应低相关 token，核值接近 0，恰恰要求估计器在小值区域低方差（Sec 2.3）[^src-performer]。

## trig 特征的失效模式

Performer 论文 Lemma 2 给出三方 MSE 闭式（独立采样时）[^src-performer]：

- MSE(SMm^trig) = (1/2m)·exp(‖x+y‖²)·SM(x,y)⁻²·(1−exp(−‖x−y‖²))²
- MSE(SMm^+) = (1/m)·exp(‖x+y‖²)·SM(x,y)²·(1−exp(−‖x+y‖²))
- MSE(SMm^hyp+) = (1/2)·(1−exp(−‖x+y‖²))·MSE(SMm^+)

推论：SM(x,y)→0 时 trig 的 MSE → ∞，而正特征的 MSE → 0（Lemma 2 推论）[^src-performer]。trig 估计器在核值小、需要仔细近似的临界区域方差大，导致训练异常甚至完全无法训练（Sec 2.3；实验中三角特征在 PG-19 上高度不稳定，见 [[performer|Performer 页]]实验节）[^src-performer]。

## PRF 构造与两个变体

Lemma 1 的恒等式：对 z=x+y，SM(x,y)=E_ω[exp(ωᵀx−‖x‖²/2)·exp(ωᵀy−‖y‖²/2)]=Λ·E_ω[cosh(ωᵀz)]，其中 Λ=exp(−(‖x‖²+‖y‖²)/2)，ω~N(0,I_d)（Lemma 1）[^src-performer]。由此得到两个正特征无偏估计器（Sec 2.3）[^src-performer]：

- **SMm^+**：h(x)=exp(−‖x‖²/2)、f₁=exp，单个指数特征；
- **SMm^hyp+**：h(x)=(1/√2)exp(−‖x‖²/2)、f₁(u)=exp(u)、f₂(u)=exp(−u)，双指数特征，用于进一步降方差——Lemma 2 显示其 MSE 严格小于 SMm^+ 以两倍特征数达到的水平（Sec 2.3, Lemma 2）[^src-performer]。

## SMREG：正则化 softmax 核

把采样分布从 N(0,I) 换成半径 √d 球面上的均匀分布（Haar 测度），即把 ω 换成 √d·ω/‖ω‖，得到正则化 softmax 核 SMREG 的估计器（Sec 2.3）[^src-performer]。Theorem 1：在注意力矩阵 L∞ 范数 ≤ C 的条件下，SMREG 注意力矩阵 Areg 与 A 的逐元素比值满足 inf ≥ 1−2/d^(1/3)+o(d^(−1/3))、sup ≤ 1；且对 d≥2 无需该条件即有 Areg ≤ A——SMREG 是 softmax 核的通用下界，其正特征可用来近似 softmax 核（Theorem 1）[^src-performer]。

## 正交化（ORF）与方差控制

对各向同性分布，把 m 个样本 ω₁,…,ωₘ 经 Gram-Schmidt 严格正交化后边缘分布不变、估计保持无偏，要求 m ≤ d（Sec 2.4）[^src-performer]。论文称首次证明正交化降低 softmax/高斯核估计 MSE 对任意维度 d>0 成立（此前仅知大 d 渐近结果），并给出显式 gap：MSE(正交) ≤ MSE(独立) − (2(m−1)/(m(d+2)))·(SM(x,y)−exp(−(‖x‖²+‖y‖²)/2))²（Theorem 2）[^src-performer]。尾概率界同样严格更小且为指数级（Theorem 3；附录以"beautiful functions"框架给出一般定理 Theorems 5/6，Theorem 2/3 为其特例，Appendix F.4）[^src-performer]。附录 B.2 还列出 R-ORF（高斯正交矩阵，O(md) 存储、O(md²) 一次性预处理）与 H/G-ORF（Hadamard/Givens，O(m) 或 O(m log d) 存储、小偏差趋 0）两类实现变体（Appendix B.2）[^src-performer]。

## 与后续工作的关系

随机特征线性注意力的共同形态是：核分解 φ(q)ᵀφ(k) + 结合律重排（先算 Σφ(k)v 与 Σφ(k)），复杂度 O(Lrd)（Performer 论文 Sec 2.2）[^src-performer]。下游实例：[[long-sequence-feature-extractor|BigST 的 LSFE]] 与 [[linearized-spatial-convolution|LSC]] 用 PRF 线性化注意力与自适应邻接；[[urbanpg|UrbanPG]] 的 STCA 用随机特征映射（该论文将其描述为 sin/cos 编码——对应论文中的 trig 无偏估计器，而 Performer 论文本身主张以正特征替代 sin/cos 以避免不稳定，两处转述按各自来源归因）[^src-urbanpg][^src-performer]。固定维随机特征的秩上限（≤ 特征维）是该路线的表达力约束，[[spectral-kernel-linear-attention]] 在图核语境下有讨论。

## 相关页面

- [[performer]] — 使用 PRF 的线性注意力 Transformer（FAVOR+）
- [[linear-attention-unified-framework]] — 线性注意力核形式与 Mamba 对应
- [[long-sequence-feature-extractor]] — PRF 用于时间自注意力线性化（BigST）
- [[linearized-spatial-convolution]] — PRF 用于自适应邻接分解（BigST）
- [[urbanpg]] — STCA 的随机特征线性注意力
- [[spectral-kernel-linear-attention]] — 旋转作为随机特征的图核读法
- [[source-performer]] — 论文源摘要

[^src-performer]: [[source-performer]]
[^src-urbanpg]: [[source-urbanpg]]
