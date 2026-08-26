---
title: "Uncertainty-Aware Rectification"
type: technique
tags:
  - flow-matching
  - consistency-models
  - uncertainty
  - gradient-conflict
  - curriculum-learning
  - kdd-2026
created: 2026-08-26
last_updated: 2026-08-26
source_count: 1
confidence: medium
status: active
---

# Uncertainty-Aware Rectification

**Uncertainty-Aware Rectification**（不确定性感知矫正）是 [[loft|LOFT]]（KDD 2026）提出的训练机制，用于化解流匹配精度目标与轨迹线性化目标之间的梯度冲突，使生成轨迹线性化、支持少步推理[^src-loft]。

## 梯度冲突问题

论文报告：直接联合最小化 $L_{CFM}+L_{CT}$ 时，$\nabla L_{CFM}$ 与 $\nabla L_{CT}$ 的余弦相似度在训练全程为负（Fig 3 蓝线）——精确拟合局部复杂向量场的优化方向，与拉直全局轨迹的方向相反。附录 C 进一步把训练样本按先验不确定性分为高（σ̃>Q_0.75）与低（σ̃≤Q_0.75）两组：高不确定性组中，精确流匹配目标 v_FM 与一致性教师速度 v_teacher 的余弦相似度更低。论文据此论证：对高不确定性样本施加统一轨迹线性化约束与数据保真目标相抵触，均匀约束是次优的[^src-loft]。

## 矫正速度目标

用样本级矫正系数 α∈[η,1]（η>0 为下限超参）在两个目标间插值：

$$v_R=\alpha\,v_{FM}+(1-\alpha)\,\mathrm{sg}(v_\theta(z_s,s))$$

- $v_{FM}=z_1-z_0$ 为条件 OT 方向；sg 为 stop-gradient，教师速度作稳定 bootstrap 目标[^src-loft]
- 教师评估时刻随 α 调制：$s=\alpha\cdot 1.0+(1-\alpha)\cdot t$。α→1 时 s→终点 1.0，抑制中间高频噪声干扰并使教师预测对齐流匹配目标；α 减小时 s→t，纳入局部一致性约束以拉直轨迹[^src-loft]

最终损失 $L_R$ 仅在该次采样的观测掩码 M 上计算平方误差并以 ‖M‖₁ 归一化，论文表述此举可防止数据泄漏[^src-loft]。

## 样本级调度

α 由数据不确定性与训练进度共同决定：

1. **不确定性聚合**（Eq 13）：对观测掩码内的元素级估计 Σ 做温度 τ 缩放的 softmax 加权（以观测内最大值 σ_max 平移），得到样本级度量 σ̃。权重向高不确定条目倾斜，防止局部估计误差被大量可信观测平均掉[^src-loft]
2. **有界映射**（Eq 14）：$f_\sigma=\eta+(1-\eta)\tanh(\lambda\tilde\sigma)\in[\eta,1)$[^src-loft]
3. **课程进度**（Eq 15）：warm-up $e_{warm}$ 之后，$c(e)=\tfrac{1}{2}\bigl(1+\cos\pi\tfrac{\max(0,e-e_{warm})}{E_{total}-e_{warm}}\bigr)$ 从 1 余弦退火至 0[^src-loft]
4. **合成**（Eq 16）：$\alpha=c(e)\cdot 1.0+(1-c(e))\cdot f_\sigma$[^src-loft]

行为归纳（论文表述）：训练早期（c≈1）或高不确定性样本（σ̃ 大）时 α 接近 1.0，目标遵循 v_FM 并跳过轨迹线性化约束；随训练推进且样本不确定性降低，α 向 η 平滑衰减，逐步强制一致性以拉直生成轨迹[^src-loft]。

## 实证支撑（作者报告）

- 加入该机制后，两目标梯度的负相关得到缓解（Fig 3 橙线 vs 直接联合优化的蓝线）[^src-loft]
- 消融 Wo-U（移除不确定性感知、仅按训练进度施加统一一致性）在 PeMS04/PeMS08 的 SC-TC 与 SR-TC 设置下误差高于完整 LOFT（Fig 5，均为 2 步推理）[^src-loft]
- 测试集按先验不确定性分组（σ̃ 与 Q_0.33/Q_0.66 分位比较）：两类模型在高不确定分位上线性度均减弱（EPE 升高、VMR 偏离 1.0），LOFT 整体 EPE 更低且 VMR 集中于 1.0（Fig 4）[^src-loft]

## 与静态矫正的关系

Consistency-FM 与 AlphaFlow 对轨迹线性化施加静态约束。在 LOFT 的匹配预算实验中，即使配低秩先验初始化，两者精度仍低于 LOFT；论文将其归因于高度稀疏训练目标下分布匹配与轨迹线性化的梯度冲突。本机制的区别是以训练进度和样本不确定性为条件动态仲裁两个目标的权衡[^src-loft]。

## 相关页面

- [[loft]] — 使用该机制的模型
- [[low-rank-prior-estimation]] — 提供机制所消费的不确定性估计 Σ
- [[consistency-models]] — 一致性模型源头工作（映射轨迹点至起点）
- [[rectified-flow]] — 迭代 reflow 式轨迹直线化路线

[^src-loft]: [[source-loft]]
