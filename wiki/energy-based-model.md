---
title: "Energy-Based Model (EBM)"
type: concept
tags:
  - generative-model
  - energy-function
  - score-function
created: 2026-05-04
last_updated: 2026-10-13
source_count: 3
confidence: medium
status: active
---

# Energy-Based Model (EBM)

**Energy-Based Model（EBM）**通过未归一化的能量函数（energy function）$E_\theta(\mathbf{x})$ 定义概率分布[^src-energy-based-model]：

$$p_\theta(\mathbf{x}) = \frac{\exp(-E_\theta(\mathbf{x}))}{Z_\theta}$$

其中 $Z_\theta = \int \exp(-E_\theta(\mathbf{x})) d\mathbf{x}$ 为难以直接计算的配分函数（partition function）。在 [[score-based-generative-modeling|score-based models]] 中，score function $\nabla_\mathbf{x} \log p(\mathbf{x}) = -\nabla_\mathbf{x} E(\mathbf{x})$ 绕过配分函数 $Z_\theta$ 的计算，通过估计能量梯度生成样本。

## 能量函数作为基本对象

LeCun 2022 年立场论文将能量函数本身作为基本对象，训练直接塑造能量地形（energy surface），无需假定其表示概率分布的未归一化对数；该定义脱离概率建模，尽管多数 EBM 可通过 Gibbs 分布转化为概率模型[^src-autonomous-machine-intelligence]。自监督学习在此表述为学习标量函数 $F(x,y)$：x 与 y 兼容时能量低、不兼容时能量高，用以建模一对多的依赖关系[^src-autonomous-machine-intelligence]。

## 隐变量 EBM

当兼容性判断需要额外信息时引入隐变量 $z$：推断对应关于 z 的能量最小化，消去 z 后得到仅依赖 x、y 的自由能[^src-autonomous-machine-intelligence]。

$$\check{z} = \arg\min_{z \in Z} E_w(x,y,z), \qquad F_w(x,y) = \min_{z \in Z} E_w(x,y,z) = E_w(x,y,\check{z})$$

论文注明 $F_w$ 技术上对应零温自由能。[[jepa|JEPA]] 为该形式的具体实例：能量取表征空间的预测误差 $D(s_y, \mathrm{Pred}(s_x,z))$[^src-autonomous-machine-intelligence]。

## 塌缩与两条训练路线

若仅压低训练样本能量，能量面易塌缩（collapse）为平坦常数。论文按架构梳理可塌缩条件：确定性预测/回归不可塌缩；生成式隐变量架构在隐变量信息容量过大时可塌缩；自编码器在表征容量过大时可塌缩（学成恒等映射）；简单联合嵌入架构在编码器忽略输入、输出恒定且相等时可塌缩[^src-autonomous-machine-intelligence]。

防塌缩的两条路线：**对比方法**压低训练样本能量、抬高对比样本能量，其局限在于需设计对比样本生成策略，且高维柔性 EBM 所需对比样本数在最坏情况下随表征维度呈指数增长；**正则化（非对比）方法**（[[non-contrastive-ssl]]）在压低训练样本能量的同时最小化低能量区域体积以包裹高数据密度区，该路线规避了维度灾难[^src-autonomous-machine-intelligence]。

## Related Pages
- [[score-function]]
- [[score-based-generative-modeling]]
- [[diffusion-model]]
- [[timegrad]] — 用于多变量概率时间序列预测的自回归扩散 EBM[^src-timegrad]
- [[jepa]] — 把能量定义为表征空间预测误差的自监督架构
- [[non-contrastive-ssl]] — 防塌缩的正则化路线与 VICReg

[^src-energy-based-model]: [[source-energy-based-model]]
[^src-timegrad]: [[source-timegrad]]
[^src-autonomous-machine-intelligence]: [[source-autonomous-machine-intelligence]]