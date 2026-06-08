---
title: "Spectral Trajectory Scheduler (STS)"
type: technique
tags:
  - diffusion-models
  - noise-schedule
  - frequency-domain
  - time-series
  - spectral-analysis
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: medium
status: active
---

# Spectral Trajectory Scheduler (STS)

**STS** (Spectral Trajectory Scheduler) 是 [[stats|StaTS]] 框架中学习数据自适应噪声调度的核心组件，通过频域正则化优化前向腐蚀轨迹[^src-stats]。

## 动机

固定噪声调度（线性、余弦、二次）在不同数据集间迁移效果差——不同函数形式施加不同的腐蚀速率，重塑中间加噪状态的分布和可去噪性[^src-stats]。例如，周期性强的 Traffic 数据集在不适配调度下前向腐蚀剖面与反向去噪修正能力不匹配，导致概率预测退化[^src-stats]。

## 方法

### 优化目标

STS 将调度参数 β = [β₁, ..., β_T] 作为投影梯度下降 (PGD) 的可学习参数[^src-stats]：

$$\mathcal{R}(\beta) = \mathcal{L}_{\text{pred}} + \lambda_{\text{smooth}} \mathcal{L}_{\text{smooth}} + \lambda_{\text{init}} \mathcal{L}_{\text{init}} + \lambda_{\text{end}} \mathcal{L}_{\text{end}} + \lambda_{\text{bar}} \mathcal{L}_{\text{bar}} + \lambda_{\text{flatness}} \mathcal{L}_{\text{flatness}}$$

| 损失项 | 作用 | 权重 |
|--------|------|------|
| $\mathcal{L}_{\text{pred}}$ | 预测导向目标，直接引导调度改进预测性能 | $\lambda_{\text{obj}}=0.01$ |
| $\mathcal{L}_{\text{smooth}}$ | 约束调度平滑性，防止剧烈波动 | $\lambda_{\text{smooth}}=5$ |
| $\mathcal{L}_{\text{init}}$ | 端点初始化条件，确保前向过程从接近干净数据开始 | $\lambda_{\text{init}}=0.5$ |
| $\mathcal{L}_{\text{end}}$ | 终端约束，频谱有意义的腐蚀程度 | $\lambda_{\text{end}}=0.5$ |
| $\mathcal{L}_{\text{bar}}$ | 均值约束，调度平均腐蚀速率 | $\lambda_{\text{bar}}=5\times 10^{-3}$ |
| $\mathcal{L}_{\text{flatness}}$ | 终端频谱平坦度约束，KL 散度到均匀分布 | $\lambda_{\text{flatness}}=0.5$ |

### PGD 优化与收敛性

$$\beta^{k+1} = \text{Proj}_{[\beta_{\min},\beta_{\max}]^T}\left(\beta^k - \eta \nabla \mathcal{R}(\beta^k)\right), \quad 0 < \eta \leq \frac{1}{L}$$

**定理 3.1**：目标单调递减且 $\|G_\eta(\beta^k)\| \to 0$。任何极限点 $\beta^\star$ 满足投影一阶驻点条件[^src-stats]。

**定理 3.2**：当调度更新时，前向漂移是 Lipschitz 稳定的[^src-stats]：
$$D_{\text{KL}}\left(q_{\beta'}(x_t|x_0) \| q_{\beta}(x_t|x_0)\right) \leq C(a,t,x_0) \|\beta' - \beta\|_\infty^2$$

这保证了两阶段交替训练中分布一致性不会崩塌[^src-stats]。

## 学习到的调度特征

不同初始化（线性/余弦/二次）下 STS 收敛到一致的调度模式[^src-stats]：

- **初期急剧上升**：快速注入基础噪声
- **中期平坦**：避免在关键信息区域过度压缩频谱结构
- **末端再次上升**：确保终端充分腐蚀以匹配去噪先验

这种非单调模式与标准模板有本质区别，使中间加噪状态具有更强的逐步可分离性和改进的可逆性[^src-stats]。

## 频谱平坦度控制

STS 通过约束终端频谱平坦度来控制前向过程的频谱破坏程度[^src-stats]：

$$\mathcal{L}_{\text{flatness}}(T) = D_{\text{KL}}\left(p_T(f) \| U(f)\right)$$

其中 $p_T(f)$ 是归一化功率谱，$U(f)$ 是均匀分布。更长的扩散轨迹倾向于产生更平坦的终端频谱，意味着更强的底层频率结构破坏[^src-stats]。STS 学习到的自适应调度改善了调度诱导的频谱衰减与反向重建动态的一致性[^src-stats]。

## 训练细节

- 与 FGD 交替优化
- STS 训练 2–5 个 epoch 即可收敛，对 epoch 数不敏感[^src-stats]
- 在实例归一化空间中运行，与 FGD 共享同分布空间[^src-stats]
- 调度初始化 β₁=10⁻⁵, β_T=0.1[^src-stats]

## 消融

移除 EPO（端点目标 $\mathcal{L}_{\text{init}}+\mathcal{L}_{\text{end}}$）导致退化最严重，尤其在 Electricity 和 Traffic——表明良好初始化和终端约束对形成稳定腐蚀轨迹至关重要[^src-stats]。移除 STS（回退固定线性调度）也降低性能，验证了自适应调度的价值[^src-stats]。

[^src-stats]: [[source-stats]]
