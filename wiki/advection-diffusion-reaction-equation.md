---
title: "Advection-Diffusion-Reaction Equation"
type: concept
tags:
  - pde
  - physics-informed
  - advection-diffusion
  - atmospheric-science
  - chemical-transport
created: 2026-07-14
last_updated: 2026-07-14
source_count: 1
confidence: medium
status: active
---

# Advection-Diffusion-Reaction (ADR) Equation

ADR 方程是描述污染物在大气中时空演化的核心偏微分方程，也是化学传输模型（CTM）的数学基础。其连续形式为：[^src-ctenet]

$$\frac{\partial X}{\partial t} + \vec{W} \cdot \nabla X = k_\theta \cdot \nabla^2 X + R(X) + S$$

四项分别对应：

1. **平流（Advection）**：$\vec{W} \cdot \nabla X$，风场驱动的大尺度定向传输。CTENet 用风矢量场（东西/南北分量）显式计算，这对强风场或长距离传输场景至关重要。[^src-ctenet]
2. **扩散（Diffusion）**：$k_\theta \cdot \nabla^2 X$，分子运动或湍流引起的随机扩散。CTENet 使用可学习的扩散系数 kθ，使模型能适应不同环境条件。[^src-ctenet]
3. **化学反应（Reaction）**：$R(X)$，光化学反应、二次气溶胶生成等非线性化学转化。CTENet 不试图精确建模数千个化学方程，而是以 sigmoid 门控气象特征作为环境引导的软注意力。[^src-ctenet]
4. **源项（Source）**：$S$，排放源和汇。[^src-ctenet]

## 数值离散化

CTENet 采用 FTCS（Forward-Time Central-Space）有限差分法：[^src-ctenet]

- **空间**：中心差分近似梯度 $\nabla X$ 和拉普拉斯 $\nabla^2 X$[^src-ctenet]
- **时间**：显式 Euler 方法，更新 $X_{T+1} = X_T + \Delta t(-\vec{W}_T \cdot \nabla X_T + k_\theta \cdot \nabla^2 X_T + R(X_T) + S_T)$[^src-ctenet]

附录 B 证明了 3×3 卷积核可近似离散拉普拉斯算子（当核权重接近特定形式时），为 CNN 隐式学习扩散提供了理论依据。[^src-ctenet]

## 在 CTENet 中的角色

CTENet 的欧拉 ADR 解码器将 ADR 三项作为独立的计算模块嵌入多层网络，每层对应一个时间步的 PDE 离散更新。消融实验表明：移除平流项性能下降最显著，证明风驱动传输是标准深度网络最难以隐式学习到的物理机制。[^src-ctenet]

## 相关页面

- [[ctenet]] — 嵌入 ADR 的模型
- [[physics-informed-neural-network]] — 架构嵌入型 PINN
- [[air-quality-forecasting]] — ADR 的主要应用领域

[^src-ctenet]: [[source-ctenet]]
