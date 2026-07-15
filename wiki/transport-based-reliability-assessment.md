---
title: "Transport-Based Reliability Assessment"
type: technique
tags:
  - multimodal-fusion
  - schrodinger-bridge
  - rectified-flow
  - reliability
  - trustworthy-ai
created: 2026-07-18
last_updated: 2026-07-18
source_count: 1
confidence: medium
status: active
---

# Transport-Based Reliability Assessment

**基于传输的可靠性评估**是一种将可靠性视为潜在空间中外在几何属性而非分类器内在预测结果的方法论。其核心思想是：通过 Diffusion Schrödinger Bridge / Rectified Flow 在潜在空间中学习的传输代价，能够反映数据质量、语义一致性和流形偏离程度，且这些信号与分类器的内部状态相互独立[^src-gmf]。

## 两类传输代价

[[gmf|GMF]] 框架区分了两种互补的传输度量[^src-gmf]：

### 模态内传输能量 $E_{\text{intra}}$

从类无关先验分布到模态表示的 rectified flow 的初始速度平方范数：

$$E_{\text{intra}}^{(m)} = \|v_\theta^{(m)}(z^{(m)}, 0)\|_2^2$$

- **直觉**：干净的表示靠近先验分布，仅需小速度即可到达；损坏/异常表示需要大速度修正
- **作用**：度量单个模态的内在质量，检测噪声、损坏、不完整数据
- **关键性质**：仅依赖编码器输出的几何位置，与分类器输出解耦

### 模态间传输代价 $E_{\text{inter}}$

跨模态速度场 $\Phi_{a \to b}$ 的端点残差平方范数：

$$E_{\text{inter}}^{(a \to b)} = \|\Phi_{a \to b}(z^{(a)}) - z^{(b)}\|_2^2$$

- **直觉**：语义一致的模态对在跨模态映射后彼此靠近；冲突对则相距遥远
- **作用**：度量跨模态语义一致性，是[[geometric-barrier-principle|几何屏障原理]]的基础
- **关键性质**：独立于任一单模态分类器的置信度

## 与统计方法的对比

| 维度 | 统计方法 | 传输方法 |
|------|---------|---------|
| 信号来源 | 分类器输出（置信度、熵） | 潜在空间几何（速度场） |
| 循环依赖 | 存在 | 不存在 |
| 对过度自信的鲁棒性 | 差 | 好 |
| 计算开销 | 几乎为零 | 轻量速度网络 |
| 互信息与置信度 | 高（~0.67） | 低（~0.08） |

## 与相关概念的关系

- [[rectified-flow]]：传输代价通过 Rectified Flow 速度网络估计，利用 rectification 使轨迹线性化以便单步推理
- [[schrodinger-bridge]]：整体框架植根于 SB 理论，GMF 使用单步 Rectified Flow 作为 DSB 近似
- [[optimal-transport]]：传输代价本质上度量了概率分布间的几何变换难度

[^src-gmf]: [[source-gmf]]
