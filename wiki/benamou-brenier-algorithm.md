---
title: "Benamou-Brenier 算法"
type: technique
tags:
  - optimal-transport
  - benamou-brenier
  - wasserstein-geodesic
  - convex-optimization
  - augmented-lagrangian
  - dynamic-optimal-transport
created: 2026-06-16
last_updated: 2026-07-13
source_count: 2
confidence: medium
status: active
---

# Benamou-Brenier 算法

**Benamou-Brenier 算法**是一种求解**最优传输**（Optimal Transport, OT）问题的连续数值方法，由 Benamou 和 Brenier 于 2000 年提出[^src-benamou-brenier-blog]。其核心思想是将 OT 问题转化为 Wasserstein 空间中的**动态公式化**（dynamic formulation），然后在 $(d+1)$ 维时空上求解一个凸变分问题。

## 核心思想：从静态映射到动态测地线

传统 OT（Monge/Kantorovich 公式化）寻找最小代价的映射 $T(x)$。Benamou-Brenier 方法换了一个角度：与其直接找映射，不如找连接 $\mu$ 到 $\nu$ 的 **Wasserstein 测地线** $\mu_t$（$t \in [0,1]$），即分布随时间连续演化的最优路径[^src-benamou-brenier-blog]。

在代价函数 $c(x,y) = |x-y|^p$（$p>1$）下，这一转化等价于求解以下**动能最小化问题**：

$$
\min_{(\varrho_t, \mathbf{v}_t)} \left\{ \int_0^1 \int_\Omega |\mathbf{v}_t|^p \, d\varrho_t \, dt \;:\; \partial_t \varrho_t + \nabla \cdot (\mathbf{v}_t \varrho_t) = 0,\; \varrho_0 = \mu,\; \varrho_1 = \nu \right\}
$$

其中：
- $\varrho_t$ 是 $t$ 时刻的密度（$\varrho_0 = \mu$, $\varrho_1 = \nu$）
- $\mathbf{v}_t$ 是速度场
- $\partial_t \varrho_t + \nabla \cdot (\mathbf{v}_t \varrho_t) = 0$ 是质量守恒（连续性方程）

> 直观理解：你想把一堆沙（$\mu$）以最小总"努力"搬到目标形状（$\nu$）。每粒沙沿某条路径运动，$\mathbf{v}_t$ 是速度，$|\mathbf{v}_t|^p$ 是瞬时功率。积分就是总动能。

## 凸化变换：Benamou-Brenier 公式

上述问题有两个麻烦：约束 $\mathbf{v}_t \varrho_t$ 是非线性的，且目标函数是非凸的。Benamou 和 Brenier 的关键洞察是**变量替换** $E_t = \mathbf{v}_t \varrho_t$（通量变量），将问题转化为：

$$
\boxed{\min_{(\varrho, E)} \left\{ \mathscr{B}_p(\varrho, E) \;:\; \partial_t \varrho_t + \nabla \cdot E_t = 0,\; \varrho_0 = \mu,\; \varrho_1 = \nu \right\}}
$$

其中 **Benamou-Brenier 泛函** $\mathscr{B}_p$ 定义为：

$$
\mathscr{B}_p(\varrho, E) = \int_0^1 \int_\Omega f_p(\varrho_t(x), E_t(x)) \, dx \, dt
$$

而 $f_p$ 是一个 1-齐次凸函数（由对偶表示定义）：

$$
f_p(t, x) = \sup_{(a, b) \in K_q} (a t + b \cdot x) = \begin{cases}
\frac{1}{p} \frac{|x|^p}{t^{p-1}} & t > 0 \\
0 & t = 0, x = 0 \\
+\infty & t = 0, x \neq 0 \text{ 或 } t < 0
\end{cases}
$$

其中 $K_q = \{(a, b) \in \mathbb{R} \times \mathbb{R}^d : a + \frac{1}{q} |b|^q \leq 0\}$（与对偶范数相关）。

> 经过这一变换后，**约束变为线性**，**目标函数变为凸**——这是一个经典的凸优化问题！

### 时空发散视角

约束 $\partial_t \varrho_t + \nabla \cdot E_t = 0$ 可以统一理解为 $(d+1)$ 维时空中的发散约束：

$$
\nabla_{t,x} \cdot (\varrho, E) = \delta_0 \otimes \mu - \delta_1 \otimes \nu
$$

这本质上是一个时空上的 **Beckmann 问题**（最优传输的动态推广）。

## 数值求解：增广拉格朗日方法

尽管问题已转化为凸优化，$f_p$ 仍然是**1-齐次**（非严格凸、不可微），直接梯度下降效率低下。Benamou-Brenier 算法采用**增广拉格朗日方法**（augmented Lagrangian）迭代求解[^src-benamou-brenier-blog]。

### 问题重写

通过引入对偶变量（势函数 $\phi$ 和凸约束变量 $\xi$），原始问题改写为 saddle-point 问题：

$$
\min_m \sup_{\xi, \phi: \xi \in K_q} \left\{ \langle \xi - \nabla_{t,x} \phi, m \rangle + G(\phi) - \frac{\tau}{2} |\xi - \nabla_{t,x} \phi|^2 \right\}
$$

其中 $m = (\varrho, E)$，$G(\phi) = \int_\Omega \phi(1,x) d\nu(x) - \int_\Omega \phi(0,x) d\mu(x)$。

### 三步迭代

从初始 $(m_k, \xi_k, \phi_k)$ 出发：

| 步骤 | 操作 | 复杂度 |
|------|------|--------|
| 1. 更新 $\phi_{k+1}$ | 求解时空 Laplace 方程 $\tau \Delta_{t,x} \phi = \nabla \cdot (\tau \xi_k - m_k)$（带 Neumann 边界 + 非齐次时间边界） | $O(N \log N)$ |
| 2. 更新 $\xi_{k+1}$ | 对每个时空点逐点投影到凸集 $K_q$ | $O(N)$ |
| 3. 更新 $m_{k+1}$ | $m_{k+1} \leftarrow m_k - \tau(\xi_{k+1} - \nabla_{t,x} \phi_{k+1})$ | $O(N)$ |

其中 $N$ 是时空离散化格点总数。

### 与 Hamilton-Jacobi 方程的联系

在 $p=2$ 的特殊情况下，交换 inf-sup 运算后可推导出 Hamilton-Jacobi 方程：

$$
\partial_t \phi + \frac{1}{2} |\nabla \phi|^2 = 0 \quad \text{（在 } \varrho > 0 \text{ 上）}
$$

求解最优 $\phi$ 后，可以通过 $\psi(x) = \phi(1, x)$ 和 $\varphi(x) = -\phi(0, x)$ 恢复原始 OT 问题的 **Kantorovich 势**。

## 与其他方法比较

| 方法 | 特点 | 限制 |
|------|------|------|
| Angenent-Hacker-Tannenbaum | 利用 OT 映射应为梯度的性质，移除非梯度项 | 需要光滑、非消失密度 |
| Loeper-Rapetti | 基于 Monge-Ampère 方程解析 | 需要特殊域和边界条件 |
| **Benamou-Brenier** | 动态 + 时空凸优化 | 每次迭代需解全局 Laplace 方程 |

## 关键优势

1. **处理消失密度**：不需要对密度支集做光滑性假设，是唯一轻松处理密度消失的数值方法
2. **通用代价**：不仅限于 $|x-y|^p$，可推广到任意凸代价 $h(x-y)$ 及黎曼流形
3. **灵活约束**：可在密度上施加凸约束（如上下界）
4. **动态扩展**：适用于平均场博弈、多种群交互等变体

## 在生成模型中的意义

Benamou-Brenier 的**动态 OT 公式化**与现代生成模型有深层联系：

- **Flow Matching** 中的 OT 路径（直线插值 $\mu_t = (1-t)x_0 + t x_1$）可视为离散形式的 Wasserstein 测地线
- 连续性方程 $\partial_t \varrho + \nabla \cdot (\mathbf{v} \varrho) = 0$ 在 [[flow-matching|Flow Matching]] 中对应 CNF 的概率密度演化（Fokker-Planck 方程的无噪声特例）
- 动能 $\int \int |\mathbf{v}|^2 d\varrho dt$ 与 Flow Matching 目标 $\|v_t(x) - u_t(x)\|^2$ 有相同的变分结构
- [[source-stochasticinterpolants|Stochastic Interpolants]] 将 $\min_{\hat v} G$ 的极小值等同于负路径动能，并对插值做 max-min 以逼近本算法的动态 OT 解（Proposition 2）[^src-stochasticinterpolants]

## 相关页面

- [[optimal-transport]] — 最优传输理论基础
- [[flow-matching]] — Flow Matching 中的 OT 路径
- [[wasserstein-distance]] — Wasserstein 距离
- [[continuous-normalizing-flow]] — 连续归一化流的概率密度演化
- [[stochastic-interpolant]] — 随机插值 max-min 通往动态 OT
- [[interflow]] — InterFlow

## 引用

[^src-benamou-brenier-blog]: [[source-benamou-brenier-blog]]
[^src-stochasticinterpolants]: [[source-stochasticinterpolants]]
