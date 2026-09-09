---
title: "FM ODE Trajectory Stages"
type: concept
tags:
  - flow-matching
  - ode-dynamics
  - denoiser
  - theory
  - icml-2025
created: 2026-09-09
last_updated: 2026-09-09
source_count: 1
confidence: medium
status: active
---

# FM ODE 轨迹三阶段动力学

Flow Matching 采样是把一个 ODE 从高斯先验积分到数据分布，但单条样本轨迹如何被数据几何塑造、每一段在做什么，此前缺乏定量刻画。Wan、Wang、Mishne、Wang 的 ICML 2025 论文以 denoiser 为中心回答了这个问题：**denoiser 是 FM ODE 向量场中唯一的数据依赖成分，它与数据几何的交互决定轨迹全程的动力学**；论文将采样轨迹划分为三个阶段——初始阶段向数据均值移动，中间阶段被局部簇吸引并吸收，终端阶段收敛到数据支撑[^src-2412-18730]。

## Denoiser 视角下的 ODE

给定调度 $(\alpha_t, \beta_t)$（$\alpha_0=\beta_1=0$，$\alpha_1=\beta_0=1$），FM 向量场有闭式分解：

$$
u_t(x) = \frac{\dot\beta_t}{\beta_t}\,x + \frac{\dot\alpha_t\beta_t - \alpha_t\dot\beta_t}{\beta_t}\,\underbrace{\mathbb{E}[X\mid X_t = x]}_{\text{denoiser } m_t(x)}
$$

即 $u_t$ 完全由后验均值 $m_t(x)=\mathbb{E}[X\mid X_t=x]$ 决定[^src-2412-18730]。论文用 noise-to-signal ratio $\sigma_t := \beta_t/\alpha_t$ 把任意调度统一到同一坐标系：经重参数化 $x_\sigma = x_{t(\sigma)}/\alpha_{t(\sigma)}$ 后，任何 FM 调度下的 ODE 都化为同一个方程（论文 Proposition 2.2）[^src-2412-18730]：

$$
\frac{dx_\sigma}{d\sigma} = -\frac{1}{\sigma}\big(m_\sigma(x_\sigma) - x_\sigma\big), \qquad q_\sigma = p * \mathcal{N}(0,\sigma^2 I)
$$

直读该式：轨迹始终朝当前 denoiser 输出移动，且 $\sigma \to 0$ 时 $1/\sigma$ 因子放大步长。用 denoiser 而非向量场做训练目标也更稳定——$m_t(x)$ 对一切 $x$ 有界，而 $u_t(x)$ 在 $t\to 1$ 处可发散[^src-2412-18730]。

## 两个元定理：吸引与吸收

分析工具是两条元定理（Theorem 3.1 / 3.2）。对闭集 $\Omega$，若轨迹方向与投影方向成锐角

$$
\langle m_\sigma(x_\sigma) - x_\sigma,\ \mathrm{proj}_\Omega(x_\sigma) - x_\sigma\rangle > 0,
$$

则到 $\Omega$ 的距离沿轨迹下降并趋于 0（**attracting**，要求轨迹避开 $\Omega$ 的 medial axis $\Sigma_\Omega$）；若邻域 $B_r(\Omega)$ 与 $\Sigma_\Omega$ 不相交且边界上满足锐角条件，则 $B_r(\Omega)$ 是**absorbing** 的——进入后不再离开，吸收性质可把锐角条件传播给区域内所有轨迹[^src-2412-18730]。由于 denoiser 输出总落在 $\mathrm{conv}(\mathrm{supp}(p))$ 内，凸包对一切轨迹自动满足锐角条件：轨迹被凸包吸引，距离按 $d_{\mathrm{conv}}(x_\sigma) \le d_{\mathrm{conv}}(x_{\sigma_1})\cdot\sigma/\sigma_1$ 衰减（Proposition 3.4）[^src-2412-18730]。

作为基础，论文还证明了 FM ODE 在 $[0,1)$ 上的良定性（Theorem 4.1）：只要 $p$ 有有限 2-moment，对每个 $x_0$ 存在唯一解。证明通过后验协方差的细致控制建立局部 Lipschitz 性与 denoiser 的线性增长界；此前结果（Lipman et al. 2022；Gao et al. 2024）的假设排除数据支撑在低维子空间或子流形上的情形[^src-2412-18730]。

## 初始阶段：向数据均值移动

$\sigma=\infty$（纯噪声）处 $m_\sigma(x) \equiv \mathbb{E}[X]$，直觉上轨迹应先奔向数据均值。Proposition 4.2 对有界支撑（允许高斯平滑）分布定量验证：从距离均值 $R_0$ 处出发，当 $\sigma > \sigma_{\mathrm{init}}(\Omega,\zeta,R_0) := \frac{2R_0\,\mathrm{diam}(\Omega)}{\sqrt{\log(1+\zeta R_0/\mathrm{diam}(\Omega))}}$ 时，轨迹向均值吸引，速率为 $\|x_\sigma - \mathbb{E}[X]\| < R_0(\sigma^2+\delta^2)^{\frac{1-\zeta}{2}} / (\sigma_1^2+\delta^2)^{\frac{1-\zeta}{2}}$；$\zeta \to 1$ 时适用范围扩大但速率变弱[^src-2412-18730]。实验上，CIFAR-10（经验最优 denoiser 与 EDM 预训练 denoiser，各 1 万随机种子）在前 4 个采样步内轨迹方向与"指向均值"方向的相对误差约低于 1%（作者报告）[^src-2412-18730]。

## 中间阶段：局部簇吸收

粗尺度几何（聚类结构）塑造中段。论文定义局部簇假设：$S$ 闭有界、直径 $D$，且 $\mathrm{supp}(p)\setminus S$ 中所有点到 $\mathrm{conv}(S)$ 的距离超过 $2D$[^src-2412-18730]。在此假设下：

- Proposition 4.3：簇权重 $a_S = p(S) > 0$ 时，对靠近 $\mathrm{conv}(S)$ 的点，denoiser 以 $\mathrm{diam}(\Omega)\frac{1-a_S}{a_S}e^{-3D\epsilon/2\sigma^2}$ 量级的误差被压向 $\mathrm{conv}(S)$；
- Proposition 4.4：当 $\sigma < \sigma_0(S,\epsilon)$ 时 $B_{D/2-\epsilon}(\mathrm{conv}(S))$ 吸收，从中出发的轨迹收敛到 $\mathrm{conv}(S)$；簇权重大时该结果对一切 $\sigma$ 成立[^src-2412-18730]。

轨迹被系统性吸收进局部簇，论文将此"locking"性质视为 FM 有效特征分离与模式覆盖的理论基础（呼应 Georgiev et al. 2023 的经验观察），并建议把数据嵌入到按类别或属性组织、几何结构清晰的潜在空间以增强特征学习[^src-2412-18730]。

## 证据与假设之外的鲁棒性

合成三簇数据上的定量对照（作者报告）：$\sigma_{\mathrm{init}}=13$（$\zeta=0.5$）、$\sigma_{\mathrm{cluster}}=0.65$、$\sigma_{\mathrm{terminal}}=0.007/0.004$ 的理论预测与单条轨迹的"转向点"逐一吻合；多条轨迹大多先以近乎直线路径奔向均值（个别起点离数据近的轨迹会越过均值再被簇吸收）[^src-2412-18730]。

真实数据不满足局部簇假设时的观察：FFHQ 上按照度维度 t-SNE 着色（无显式簇，但暗/亮两端形成高密度区），从暗区/亮区附近初始化的 EDM 轨迹持续生成对应照度的样本，随机初始化则覆盖整个照度谱——轨迹仍向局部密集区吸收[^src-2412-18730]。理论上 Corollary H.1 部分解释了这种鲁棒性：对满足局部簇假设的测度做高斯卷积得到的分布，密集区仍保持吸引与吸收[^src-2412-18730]。作者同时指出 worst-case 常数不紧（这在理论界中常见），定性阶段划分在实验中稳定成立[^src-2412-18730]。

## 含义

- **采样资源分配**：初始段与终端段轨迹移动有限、中段 denoiser 变化剧烈，与 Esser et al. (2024, SD3) 的经验一致；理论上支持把采样算力向中间阶段倾斜[^src-2412-18730]。
- **潜在空间设计**：同一数据嵌入不同空间会产生不同采样轨迹，可用于优化 latent 空间与稳定微调[^src-2412-18730]。
- 终端阶段的收敛性、等变性与记忆现象另见 [[fm-ode-terminal-convergence]] 与 [[diffusion-memorization-geometry]]。

## 相关页面

- [[fm-ode-terminal-convergence]] — 终端阶段收敛定理（首个覆盖低维子流形支撑的结果）
- [[diffusion-memorization-geometry]] — 终端吸收行为与记忆现象
- [[flow-matching]] — FM 框架基础
- [[meanflow]] — 终端阶段轻微移动是少步生成可行的观测依据之一
- [[x-prediction]] — denoiser 即 $x_1$-prediction 参数化目标

[^src-2412-18730]: [[source-2412-18730]]
