---
title: "生成扩散模型漫谈（六）：一般框架之ODE篇"
type: source-summary
tags:
  - diffusion-models
  - ode
  - fokker-planck
  - ddim
created: 2026-06-22
last_updated: 2026-06-22
source_count: 1
confidence: medium
status: active
---

# 生成扩散模型漫谈（六）：一般框架之 ODE 篇

苏剑林于 2022 年 8 月发表的博客文章，从 **Fokker-Planck 方程**的等价变换出发，系统推导了 SDE 框架下的概率流 ODE，并证明 DDIM 是其线性漂移特例[^src-ddim-ode-spaces-ac-cn]。

## 核心论述路径

文章按以下逻辑链展开：

### 1. Dirac 函数作为分布-期望桥梁

将概率密度 $p(x)$ 表示为 Dirac δ 函数的期望：$p(x) = \mathbb{E}_y[\delta(x-y)]$。利用 δ 函数的泰勒展开可将 SDE 离散形式的随机关系转化为分布的偏微分方程[^src-ddim-ode-spaces-ac-cn]。

### 2. Fokker-Planck 方程的推导

从前向 SDE 的离散形式 $x_{t+\Delta t} = x_t + f_t(x_t)\Delta t + g_t\sqrt{\Delta t}\,\varepsilon$ 出发，将 $\delta(x - x_{t+\Delta t})$ 做泰勒展开至 $\mathcal{O}(\Delta t)$，两边取期望后除以 $\Delta t$ 并取极限，得到[^src-ddim-ode-spaces-ac-cn]：

$$\frac{\partial p_t}{\partial t} = -\nabla_x \cdot (f_t p_t) + \frac{1}{2}g_t^2 \nabla_x^2 p_t$$

这是描述 $p_t(x)$ 如何随时间演化的 F-P 方程。

### 3. F-P 方程的等价变换：核心洞见

对于任意满足 $\sigma_t^2 \le g_t^2$ 的函数 $\sigma_t$，F-P 方程可等价改写为[^src-ddim-ode-spaces-ac-cn]：

$$\frac{\partial p_t}{\partial t} = -\nabla_x \cdot \left[\left(f_t - \frac{1}{2}(g_t^2 - \sigma_t^2)\nabla_x\log p_t\right) p_t\right] + \frac{1}{2}\sigma_t^2 \nabla_x^2 p_t$$

这意味着**存在一族不同的前向 SDE，它们产生完全相同的边缘分布 $p_t(x)$**。这比 DDIM 论文中"训练目标只依赖边缘分布"的论述更深刻——它从 PDE 层面直接保证了等价性。

### 4. 概率流 ODE：$\sigma_t=0$ 的极端情形

令 $\sigma_t=0$，SDE 退化为确定性 ODE[^src-ddim-ode-spaces-ac-cn]：

$$dx = \left(f_t(x) - \frac{1}{2}g_t^2 \nabla_x \log p_t(x)\right) dt$$

这就是 **Probability Flow ODE**。由于前向过程是确定性的，反向求解 ODE 得到从 $x_T$ 到 $x_0$ 的唯一映射——使扩散模型具备了 flow 模型的可逆性。

### 5. DDIM 作为特例

当漂移项 $f_t(x)$ 为线性函数 $f_t x$ 时（对应 DDPM/VP SDE），代入相关参数化关系后，概率流 ODE 化简为 DDIM 的连续形式[^src-ddim-ode-spaces-ac-cn]：

$$\frac{d}{ds}\left(\frac{x(s)}{\bar{\alpha}(s)}\right) = \epsilon_\theta(x(s), t(s)) \frac{d}{ds}\left(\frac{\bar{\beta}(s)}{\bar{\alpha}(s)}\right)$$

从而在数学上严格确立了：**DDIM = 概率流 ODE 在线性漂移下的 Euler 离散化**。

## 方法论贡献

文章展示了一种强有力的分析范式：不直接处理 SDE，而是通过 F-P 方程研究边缘分布的演化，再通过等价变换揭示自由度（不同的 $\sigma_t$ 选择）。这套方法可推广到任意扩散模型的分析。

## 与论文视角的对比

| 维度 | DDIM 论文 | 苏剑林博客 |
|------|----------|-----------|
| 出发点 | 变分推断、非马尔可夫过程 | Fokker-Planck 方程 |
| 工具 | 概率图模型、KL 散度 | δ 函数、PDE 等价变换 |
| 核心论证 | 训练目标只依赖 marginals | F-P 方程在 $(f,g)\to(f',\sigma)$ 下不变 |
| DDIM 位置 | $\sigma=0$ 的隐式模型 | ODE 的 Euler 离散化 |

博客视角更**一般化**：它不预设前向过程的具体形式（高斯或离散），也不依赖变分下界的结构，仅从 F-P 方程出发统一处理。

## 引用

[^src-ddim-ode-spaces-ac-cn]: [[source-ddim-ode-spaces-ac-cn]]
