---
title: "Diffusion Memorization Geometry"
type: concept
tags:
  - flow-matching
  - memorization
  - ode-dynamics
  - privacy
  - theory
  - icml-2025
created: 2026-09-09
last_updated: 2026-09-09
source_count: 1
confidence: medium
status: active
---

# 扩散/流匹配记忆现象的几何理论

记忆（memorization）指生成模型在训练数据上完全拟合、丧失泛化的现象：扩散模型可被提取训练图像（Carlini et al. 2023），重复样本会加剧复制行为（Somepalli et al. 2023）[^src-2412-18730]。Wan、Wang、Mishne、Wang 的 ICML 2025 论文把这一现象与 FM ODE 的终端阶段几何动力学直接联系起来：在经验（离散）分布下，终端阶段每个训练点构成强吸引子，吸引强度可由显式量 $\sigma_0(V_i^\epsilon)$ 刻画；对经验分布渐近最优的 denoiser 必然重现训练数据[^src-2412-18730]。

## 终端阶段：每个训练点都是吸引子

设经验分布 $p = \sum_{i=1}^n a_i \delta_{x_i}$，支撑 $\Omega = \{x_1,\dots,x_n\}$。对 $\epsilon>0$ 定义 $\epsilon$-收缩 Voronoi 单元

$$
V_i^\epsilon := \{x : \|x - x_i\|^2 \le \|x - x_j\|^2 - \epsilon^2,\ \forall x_j \neq i\},
$$

即到 $x_i$ 的优势超过 $\epsilon$ 的点集；$\epsilon\to 0$ 时恢复经典 Voronoi 划分[^src-2412-18730]。引入吸引时刻

$$
\sigma_0(V_i^\epsilon) = \begin{cases} \infty, & C_{i,\epsilon} \le 1 \\ \epsilon\, C_{i,\epsilon}^{-1/2}, & C_{i,\epsilon} > 1 \end{cases}, \qquad C_{i,\epsilon} = \sqrt{\frac{\mathrm{sep}(x_i)^2 - \epsilon^2}{2\,\mathrm{sep}(x_i)}} \cdot \frac{1-a_i}{a_i}\cdot \mathrm{diam}(\Omega)
$$

其中 $\mathrm{sep}(x_i) = d_{\Omega\setminus\{x_i\}}(x_i)$ 是样本点与其他训练点的分隔度。**Proposition 5.9**：$\sigma < \sigma_0(V_i^\epsilon)$ 时 $V_i^\epsilon$ 吸收，从单元内出发的轨迹收敛到 $x_i$；权重 $a_i$ 越大、分隔度越高，$\sigma_0$ 越大，即该训练点越早锁定轨迹[^src-2412-18730]。

## 对经验观察的解释

- **重复样本**：重复把经验权重 $a_i$ 抬高，$\sigma_0$ 随之增大，重复图像的记忆风险上升——论文以此解释 Somepalli et al. (2023) 观察到的复制现象[^src-2412-18730]。
- **何时锁定**：CIFAR-10 上取 $\epsilon = 1.0$ 时，全训练集平均 $\sigma_0(V_i^\epsilon) \approx 0.17$（作者计算），对应 EDM 采样步的最后一个四分之一区间——即轨迹大体只在采样的最后阶段才被个别训练点锁定[^src-2412-18730]。

## 渐近最优 denoiser 必然记忆

**Proposition 5.10**：设神经 denoiser $m_\theta^\sigma$ 与经验最优 $m_\sigma$ 的误差 $\|m_\theta^\sigma(x) - m_\sigma(x)\| \le \phi(\sigma)$，$\phi(\sigma) \to 0$（"渐近最优"）。则对每个训练点，存在 $\sigma_0(V_i^\epsilon, \phi)$，当采样在 $\sigma$ 低于该阈值时从 $V_i^\epsilon$ 内任意点出发，轨迹收敛到训练点 $x_i$[^src-2412-18730]。

结论的形状值得注意：问题不在中间步骤的误差，而在终端渐近行为——只要终端逼近的是经验最优（对训练点做投影），轨迹就收敛到训练点。论文由此推出泛化条件：**要泛化，终端 denoiser 本身必须泛化**，即逼近真实底层数据流形上的投影，而非训练点的投影；作者据此建议对终端时间训练做针对性正则化（例如正则化 denoiser 的 Jacobian 以避免塌缩到局部常值映射，论文 Remark C.3 的讨论方向）[^src-2412-18730]。

## 实验

- **合成（Appendix J.1）**：给经验最优 denoiser 加 $\tilde m_\sigma = m_\sigma + \sigma\epsilon$（$\epsilon \sim \mathcal{N}(0, 3^2 I)$ 固定采样）构造渐近最优扰动。扰动在早期大幅改道轨迹（一条轨迹横穿数据区域），但随扰动强度随 $\sigma$ 衰减，所有轨迹最终仍收敛到具体训练点——验证记忆的终端鲁棒性[^src-2412-18730]。
- **CIFAR-10（Appendix J.2）**：同型扰动下，denoiser 输出在前 14 步几乎不可辨认（被大幅噪声损坏），但 ODE 轨迹仍收敛到与训练图像距离约 1.0 的位置（相对数据集均值范数 27.2 很小；1 万随机种子平均，18 步 EDM 调度；作者报告）——终端收敛对中间损坏不敏感，训练正则化的着力点应是终端阶段[^src-2412-18730]。

## 范围与边界

- 以上为经验分布（离散支撑）下的结果；流形支撑下的记忆行为论文未给出对应刻画。
- "终端阶段不应以经验最优为目标训练"是论文的理论推论与建议方向，正则化方案本身（如 Jacobian 正则）在文中是讨论而非实验验证的对象[^src-2412-18730]。
- 论文的良性推论（Proposition 5.10 后段）：若轨迹极限与 $m_\theta^\sigma(x_\theta^\sigma)$ 的极限都存在，则二者差的极限为 0——即终端 denoiser 输出与轨迹终点在渐近意义下一致[^src-2412-18730]。

## 相关页面

- [[fm-ode-trajectory-stages]] — 前终端阶段的均值/簇动力学
- [[fm-ode-terminal-convergence]] — 终端收敛定理与流映射等变性
- [[consistency-models]] — 一步生成与终端阶段的关系
- [[flow-matching]] — FM 框架基础
- [[classifier-free-guidance]] — 固定引导尺度与 memorization 的另一条关联线索

[^src-2412-18730]: [[source-2412-18730]]
