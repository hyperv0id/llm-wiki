---
title: "Combinatorial Optimization-Augmented Machine Learning (COAML)"
type: concept
tags:
  - combinatorial-optimization
  - machine-learning
  - end-to-end-learning
  - hybrid-modeling
created: 2025-07-25
last_updated: 2025-07-25
source_count: 1
confidence: medium
status: active
---

# Combinatorial Optimization-Augmented Machine Learning (COAML)

COAML 是一种将组合优化（CO）问题嵌入机器学习管道的混合架构范式[^src-wardropnet]。核心思想是：用统计模型（如神经网络）从上下文信息中预测 CO 问题的参数，再将 CO 求解器作为可微层嵌入管道末端，实现端到端训练。

## 与纯 ML 的区别

纯 ML 管道直接学习从上下文 x 到目标 y 的映射 f(x) = y，忽略了目标的组合结构约束。COAML 管道利用 CO 层强制预测满足组合可行性（如在交通流预测中满足流守恒和非负性），从而在组合结构重要的场景中获得显著精度提升[^src-wardropnet]。

## 管道结构

```
上下文 x → 统计模型 φ_w → 参数 θ → CO层 solver → 预测 ŷ
```

训练需要 CO 层可微或可计算有意义的梯度。[[wardropnet|WardropNet]] 利用 [[fenchel-young-loss|Fenchel-Young 损失]] 和 Danskin 引理获得梯度 ∇_θ L = y*(θ) − ȳ，使得通过组合层的反向传播成为可能[^src-wardropnet]。

## 与其他概念的关联

- **Predict-then-Optimize**：COAML 的端到端训练（end-to-end）对应 Elmachtoub & Grigas (2022) 的 Smart Predict-then-Optimize 框架，但后者通常针对线性目标，COAML 可处理非线性延迟函数和均衡问题[^src-wardropnet]。
- **Deep Equilibrium Models**：Bai et al. (2019) 的隐式深度均衡模型与 COAML 的均衡层有精神相似性，但 DEQ 求解的是不动点方程而非组合优化问题。
- **Differentiable Optimization Layers**：Agrawal et al. (2019) 和 Amos & Kolter (2017) 的可微凸优化层可视为 COAML 的早期形式。

## 正则化策略

COAML 的均衡层需要正则化以保证可微性。[[wardropnet|WardropNet]] 探索了三种策略[^src-wardropnet]：
1. **欧几里得正则化**：Ω(y) = ½‖y‖²，均衡为 θ 在可行流多面体上的正交投影
2. **扰动正则化**：Ω = F*，其中 F(θ) = E[max(θ+Z)^⊤ y]，通过蒙特卡洛采样估计梯度
3. **分段常数扩展**：通过扩展网络 G^exp 将非平滑延迟函数转化为多商品流问题

[^src-wardropnet]: [[source-wardropnet]]
