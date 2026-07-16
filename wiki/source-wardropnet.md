---
title: "WardropNet: Traffic Flow Predictions via Equilibrium-Augmented Learning"
type: source-summary
tags:
  - traffic-prediction
  - combinatorial-optimization
  - equilibrium-learning
  - wardrop-equilibrium
  - coaml
created: 2025-07-25
last_updated: 2025-07-25
source_count: 1
confidence: high
status: active
---

# WardropNet: Traffic Flow Predictions via Equilibrium-Augmented Learning

WardropNet 提出了一种将组合优化（CO）嵌入机器学习的混合管道（COAML），用于交通流预测[^src-wardropnet]。核心思路是用神经网络作为统计模型学习延迟函数（latency function）的参数化 θ = φ_w(x)，然后将 Wardrop 均衡（WE）作为可微的组合层（CO-layer）嵌入管道末端，通过模仿学习实现端到端训练。

## 核心贡献

1. **广义 Wardrop 均衡的凸特征化**：将 WE 推广到非可分解延迟函数（即弧上延迟可依赖全网流量），并证明当延迟函数从一个势函数 Φ 导出时，WE 等价于凸优化问题 min Φ(ȳ)[^src-wardropnet]。

2. **Fenchel-Young 损失与端到端学习**：利用 Fenchel-Young 损失作为监督信号，将 Bregman 散度最小化转化为凸代理损失，其梯度 ∇_θ L = ∇Ω*(θ) − ȳ 可解析计算[^src-wardropnet]。

3. **三种均衡层实现**：(a) 欧几里得正则化（ER）：均衡为 θ 在 Ȳ 上的正交投影；(b) 扰动正则化（perturbation）：通过蒙特卡洛估计梯度，Bregman 散度与 FY 损失等价；(c) 分段常数延迟（CL）：通过扩展网络 G^exp 将分段常数函数编码为多图并行弧[^src-wardropnet]。

4. **实验验证**：在 4 个风格化场景和 2 个真实柏林场景上，CL 管道较纯 ML 基线（FNN/GNN）平均 MAE 降低 60–75%，即使在纯 ML 有利的高熵场景也保持优势[^src-wardropnet]。

## 局限性

- ER 管道因仅学习 y-intercept 而性能受限；CL 管道虽最优，但需要求解扩展网络上的 MCFP
- 尚未扩展到大规模真实城市网络
- 多项式延迟仅探索了 k={0,1}，更丰富的延迟函数族有待研究

[^src-wardropnet]: [[source-wardropnet]]
