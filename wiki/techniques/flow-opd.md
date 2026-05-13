---
title: "Flow-OPD: On-Policy Distillation for Flow Matching"
type: technique
tags:
  - flow-matching
  - on-policy-distillation
  - text-to-image
  - grpo
  - knowledge-distillation
  - multi-task-learning
created: 2026-05-13
last_updated: 2026-05-13
source_count: 1
confidence: high
status: active
---

# Flow-OPD: On-Policy Distillation for Flow Matching

> Flow-OPD 是首个将 On-Policy Distillation (OPD) 引入 Flow Matching 模型的后训练框架，通过多教师在线蒸馏解决多任务对齐中的标量奖励稀疏性和梯度干扰问题。[^src-2605-08063]

## 动机

### GRPO 在单任务上有效

标准流匹配依赖离线重建损失（L2 速度场回归），无法优化不可微的偏好。[[flow-grpo|Flow-GRPO]] 通过在线探索（组内优势归一化）突破离线 SFT 的性能上限。[^src-2605-08063]

### GRPO 在多任务上失败

单奖励 GRPO 导致灾难性遗忘：优化一个目标（如 OCR）时，其他未监控能力（如 GenEval）严重退化。根源是标量奖励压缩多维度冲突进入零和博弈。简单混合多个奖励同样不可行——梯度干扰导致 seesaw effect。[^src-2605-08063]

## 两阶段框架

### 阶段一：领域专家训练

对每个任务使用单奖励 [[flow-grpo|Flow-GRPO]] 微调独立教师：
- GenEval 教师（组合生成）
- OCR 教师（文字渲染）
- PickScore 教师（人类偏好）
- DeQA 教师（图像质量）

每个教师在其对应指标上达到性能上限。[^src-2605-08063]

### 阶段二：多教师在线蒸馏

#### Cold-Start 初始化

两种策略建立初始策略 θ₀：

1. **SFT-based** — 用专家教师的采样轨迹做监督微调
2. **Model Merging** — 直接合并多个教师参数，将学生置于高能力区域

#### On-Policy Sampling

通过 [[ode-to-sde-conversion|ODE-to-SDE 转换]] 将确定性 ODE 转为等效 SDE，使学生能探索自身分布偏移：

\[
dx_t = \left[v_\theta(x_t,t) + \frac{\sigma_t^2}{2t}\big(x_t + (1-t)v_\theta(x_t,t)\big)\right] dt + \sigma_t dw
\]

Euler-Maruyama 离散化后，学生的转移行为为各向同性高斯策略：[^src-2605-08063]

\[
\pi_\theta(x_{t-\Delta t}|x_t,c) = \mathcal{N}(\mu_\theta(x_t,t), \sigma_t^2 \Delta t I)
\]

#### Task-Routing Labeling

硬路由机制 ⊮_{T(c)=k}，将文本条件 c 映射到唯一领域专家 k，只激活一个教师提供参考速度场：

\[
v_{\text{target}}(x_t,t,c) = v_{\phi_k}(x_t,t,c), \quad \text{where } k = \mathcal{R}(c)
\]

#### Dense KL Reward

由于学生和教师的转移策略共享相同的各向同性协方差 σ_t²ΔtI，Reverse KL 散度可解析计算为均值间的 L2 距离：[^src-2605-08063]

\[
D_{\mathrm{KL}}(\pi_\theta\|\pi_{\text{target}}) = \frac{\|\mu_\theta(x_t,t) - \mu_{\text{target}}(x_t,t)\|^2}{2\sigma_t^2\Delta t}
\]

化简后等价于速度场加权 L2 距离：

\[
r_t^{(i)} = -w(t)\|\bar{v}_\theta^{(i)}(x_t,t,c) - v_{\text{target}}^{(i)}(x_t,t,c)\|^2
\]

#### Clipped Policy Gradient

使用 PPO clip 机制稳定稠密奖励训练，梯度仅通过策略比率 ρ_{t,i,j}(θ) 传播，不经过稠密奖励（严格 detached）。[^src-2605-08063]

### Manifold Anchor Regularization (MAR)

引入与任务无关的审美教师（DeQA 优化）提供全数据 KL 惩罚，将优化锚定在高品质视觉流形上：

\[
\mathcal{L}_{\text{Total}}(\theta) = \mathcal{L}_{\text{Policy}}(\theta) + \lambda \mathbb{E}_{c,t,x_t \sim \rho_\theta^t}\left[w(t)\|v_\theta(x_t,t,c) - v_{\text{aesthetic}}(x_t,t,c)\|^2\right]
\]

详见 [[manifold-anchor-regularization|Manifold Anchor Regularization]]。[^src-2605-08063]

## 关键洞察

### Teacher-Surpassing 效应

Flow-OPD 在部分边缘案例上超越所有单个教师，推测源于知识交叉授粉：多个专家的同时稠密监督迫使学生学习更整体、更平滑的表示，弥合了单一专家的认知盲区。[^src-2605-08063]

### 稠密监督 vs 标量奖励

核心创新在于用**轨迹级稠密监督**替代**标量级奖励**：
- 标量奖励：将多维度冲突压缩为单一数值 → 零和博弈
- 稠密监督：每个时间步的速度场指导 → 可解耦的梯度信号

## 与 Flow-GRPO 的对比

| 维度 | Flow-GRPO | Flow-OPD |
|------|-----------|----------|
| 监督信号 | 标量奖励（稀疏） | 轨迹级速度场（稠密） |
| 多任务策略 | 奖励加权混合 | 多教师路由+蒸馏 |
| 画质保护 | KL 惩罚（锚定预训练模型） | MAR（锚定审美教师） |
| 训练稳定性 | 组内优势归一化 | PPO clip + 稠密奖励 detached |
| GenEval / OCR | 单任务 0.95/0.92 | 多任务 **0.92/0.94** |

## 引用

[^src-2605-08063]: [[source-flow-opd]]