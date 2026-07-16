---
title: "STPDE"
type: entity
tags:
  - spatio-temporal-forecasting
  - pde
  - ood-generalization
  - framework
created: 2026-07-23
last_updated: 2026-07-23
source_count: 1
confidence: medium
status: active
---

# STPDE

STPDE（Spatio-Temporal Partial Differential Equation）是 Liu, Wei & Zhang（Tongji University）在 ICML 2026 提出的**物理启发式时空预测框架**。其核心思想是将时空动态重新形式化为非齐次偏微分方程（inhomogeneous PDE）的演化，将动力学显式分解为**普适规律**与**环境异质性**两个正交维度[^src-stpde]。

## 设计动机

现有 STGNN 方法在 ID 场景下表现良好，但在 OOD 泛化、跨城市迁移和持续学习等非平稳场景下急剧退化。已有 OOD 方法（CauSTG、STONE、STOP 等）主要将时空演化建模为**离散图推理**，忽略了时空场的连续物理本质，且难以有效解耦通用演化机制与环境特定变化[^src-stpde]。

## 架构 (Encode–Solve–Decode)

1. **Encode**：线性投影 $\phi_\text{in}$ 将历史窗口 $\mathbf{X} \in \mathbb{R}^{T \times N \times C}$ 压缩为潜在场 $\mathbf{h} \in \mathbb{R}^{N \times D}$
2. **Solve**（核心）：物理学启发的 PDE Solver 模拟非齐次扩散方程 $\frac{\partial u}{\partial \tau} = \alpha(\Phi, \xi) \nabla^2 u + S(\Phi, \xi)$，其中：
   - [[invariant-diffusion-operator|Invariant Diffusion Operator]]（$\nabla^2$）：共享的不变拉普拉斯算子，由线性注意力实现
   - [[environment-basis-manifold|Environment Basis Manifold]]（$\Phi$）：K 个可学习基底 + 稀疏 Top-K 路由 + 随机扰动，生成 $\alpha$ 和 $S$
3. **Decode**：$\phi_\text{out}$ 将演化后状态映射回预测 $\hat{\mathbf{Y}}$

关键创新：Green 函数核传播 ⟷ 全局注意力之间的理论等价性（Theorem 4.1），以及可分离核近似下 O(N²) → O(ND²) 的线性复杂度转化（Theorem 4.2）[^src-stpde]。STPDE 代表了 [[pde-driven-spatio-temporal-forecasting|PDE-driven 时空预测]] 范式的典型实例。

## 评估场景

| 场景 | 数据配置 | 核心挑战 |
|------|----------|----------|
| ID 预测 | 同分布训练/测试 | 基础精度 |
| OOD 泛化 | 1 月训练 → 8 月测试 | 季节性分布偏移 |
| Few-shot 迁移 | 多城预训练 → 10% 目标数据微调 | 跨域样本效率 |
| 持续学习 | PEMS-Stream 7 增量周期 (655→871 节点) | 灾难性遗忘 + 拓扑扩展 |

## 关键消融发现

- **w/o PDE**（MLP 替换 PDE solver）：退化最大，OOD 下尤为显著——证明 PDE 显式建模提供了关键的泛化归纳偏置
- **w/o E**（移除环境调制）：跨环境泛化能力下降
- **w/o D**（移除扩散算子）：倾向记忆化，OOD 下失效
- **w/o P**（移除扰动正则）：OOD 退化最大——随机扰动是应对非平稳偏移的必要正则
- **w/ GCN**（用局部 GCN 替换扩散算子）：不及全局扩散——局部消息传递无法在可比深度下实现等效长程耦合[^src-stpde]

[^src-stpde]: [[source-stpde]]
