---
title: "Benamou-Brenier 算法（博客笔记）"
type: source-summary
tags:
  - optimal-transport
  - benamou-brenier
  - wasserstein-geodesic
  - convex-optimization
  - augmented-lagrangian
created: 2026-06-16
last_updated: 2026-06-16
source_count: 0
confidence: medium
status: active
---

# Benamou-Brenier 算法（博客笔记）

**来源**：Better_Yu, 博客园 (2021-05-09), [最优传输算法——Benamou Brenier算法](https://www.cnblogs.com/langdayu/p/14745914.html)

## 概述

该博客是对经典最优传输数值方法——Benamou-Brenier 算法的介绍性笔记，内容主要来自最优传输教材的相关章节（推测为 Santambrogio 的 *Optimal Transport for Applied Mathematicians*，第 5-6 章），用更通俗的语言阐述了该算法的数学思想和数值实现。

## 核心贡献

1. **动态 OT 公式化**：将最优传输问题（寻找代价为 $|x-y|^p$ 的最优映射）转化为 Wasserstein 空间 $\mathbb{W}_p$ 中寻找常速测地线的问题——等价于在时空 $(t,x)$ 上最小化动能 $\int_0^1 \int_\Omega |\mathbf{v}_t|^p d\varrho_t dt$，受连续性方程约束 $\partial_t \varrho_t + \nabla \cdot (\mathbf{v}_t \varrho_t) = 0$。

2. **凸化变换**：通过变量替换 $E_t = \mathbf{v}_t \varrho_t$，将原本非凸、非线性约束的问题转化为 **凸优化问题**（以 $(\varrho, E)$ 为变量），约束变为线性：$\partial_t \varrho_t + \nabla \cdot E_t = 0$。目标函数变为 $\mathscr{B}_p(\varrho, E) = \int f_p(\varrho, E)$，其中 $f_p$ 是一个 1-齐次凸函数。

3. **增广拉格朗日数值求解**：由于 $f_p$ 是 1-齐次（非严格凸、不可微），使用 **增广拉格朗日方法 (augmented Lagrangian)** 迭代求解，每次迭代包含三步：求解时空 Laplace 方程更新 $\phi$（$O(n\log n)$）、逐点投影到凸集 $K_q$ 更新 $\xi$（$O(N)$）、对偶上升更新 $m$。

4. **与 Hamilton-Jacobi 方程的联系**：在 $p=2$ 时，交换 inf-sup 运算可得 Hamilton-Jacobi 方程 $\partial_t \phi + \frac{1}{2}|\nabla \phi|^2 = 0$，且可从最优 $\phi$ 恢复 Kantorovich 势。

## 方法优势

- **处理消失密度**：不对密度支集做光滑性假设
- **通用代价函数**：除 $|x-y|^p$ 外，适用于任意凸代价 $h(x-y)$ 及黎曼流形
- **灵活约束**：可在密度上施加凸约束（上下界）
- **可扩展**：适用于平均场博弈、多种群交互等动态问题

## 局限性

- 每次迭代需解全局 Laplace 方程，复杂度 $O(N\log N)$
- 1-齐次函数不可微，标准梯度下降法效率低
- 原文为教材笔记形式，非独立原创性贡献
