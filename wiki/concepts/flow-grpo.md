---
title: "Flow-GRPO"
type: concept
tags:
  - flow-matching
  - reinforcement-learning
  - text-to-image
  - grpo
created: 2026-05-12
last_updated: 2026-05-12
source_count: 1
confidence: high
status: active
---

# Flow-GRPO

> **Flow-GRPO** 是首个将在线策略梯度强化学习（GRPO）集成到流匹配生成模型的方法，通过 ODE-to-SDE 转换和 Denoising Reduction 实现高效、高质量的对齐训练。[^src-2505-05470]

## 核心思想

将 [[grpo|GRPO]]（Group Relative Policy Optimization）应用于 [[flow-matching|流匹配模型]]（如 Stable Diffusion 3.5、FLUX），使模型通过在线交互从奖励信号中学习改进。根本上解决了三个问题：

1. **确定性障碍**：通过 [[ode-to-sde-conversion|ODE-to-SDE 转换]] 将确定性采样改为随机采样
2. **效率障碍**：通过 [[denoising-reduction|Denoising Reduction]] 将训练去噪步数从 40 降至 10
3. **Reward Hacking**：通过 KL 正则化约束策略不偏离预训练模型太远

## 数学框架

### GRPO 损失函数

Flow-GRPO 优化以下目标：

\[
\mathcal{J}_{\text{Flow-GRPO}}(\theta) = \mathbb{E}_{c \sim C, \{x^i\}_{i=1}^G \sim \pi_{\theta_\text{old}}(\cdot|c)} \left[ \frac{1}{G} \sum_{i=1}^G \frac{1}{T} \sum_{t=0}^{T-1} \min\left( r_t^i(\theta)\hat{A}_i^t,\ \mathrm{clip}(r_t^i(\theta), 1-\varepsilon, 1+\varepsilon)\hat{A}_i^t \right) - \beta D_{\mathrm{KL}}(\pi_\theta\|\pi_{\text{ref}}) \right]
\]

其中：
- \(r_t^i(\theta) = \frac{p_\theta(x_{t-1}^i|x_t^i,c)}{p_{\theta_\text{old}}(x_{t-1}^i|x_t^i,c)}\) 为重要性权重
- \(\hat{A}_i^t\) 为组内归一化优势：\(\hat{A}_i^t = \frac{R(x_0^i,c) - \mathrm{mean}(\{R\}_{i=1}^G)}{\mathrm{std}(\{R\}_{i=1}^G)}\)
- β 为 KL 惩罚系数
- ε 为 clip 范围

### ODE-to-SDE 转换

确定性过程 \(dx_t = v_t dt\) 转换为等效 SDE（Euler-Maruyama 离散化）：

\[
x_{t+\Delta t} = x_t + \left[v_\theta(x_t,t) + \frac{\sigma_t^2}{2t}(x_t + (1-t)v_\theta(x_t,t))\right]\Delta t + \sigma_t\sqrt{\Delta t}\,\epsilon,\quad \epsilon \sim \mathcal{N}(0,I)
\]

噪声调度：\(\sigma_t = a\sqrt{\frac{t}{1-t}}\)，a 控制随机性强度。

### Denoising Reduction

训练时 T_train = 10 步，推理时 T_infer = 40 步。利用少步样本的结构信息进行 RL 优化，在完整步数推理时质量无损。

## 实验表现

### Compositional Generation (GenEval)

| 模型 | Overall | 计数 | 位置 | 颜色 | 属性绑定 |
|---|---|---|---|---|---|
| SD3.5-M | 0.63 | 0.50 | 0.24 | 0.81 | 0.52 |
| SD3.5-M + Flow-GRPO | **0.95** | **0.95** | **0.99** | **0.92** | **0.86** |
| GPT-4o | 0.84 | 0.85 | 0.75 | 0.92 | 0.61 |

Flow-GRPO 在全部子任务上超越 GPT-4o。

### Visual Text Rendering

OCR 准确率从 59% → 92%，文字生成能力大幅提升。

### Human Preference Alignment

PickScore 从 21.72 → 23.31（w/ KL 约束，保持画质和多样性）。

## 与 RLHF/DPO 的对比

| 方法 | 奖励类型 | 样本利用 | 在线 |
|---|---|---|---|
| DPO | 偏好对 | 静态 | 离线 |
| RLHF (PPO) | 标量奖励 | 在线采样 | 在线 |
| **Flow-GRPO** | 标量奖励 | 在线采样（组归一化优势） | **在线** |

相比 DPO 等离线方法，在线交互能持续自我改进（online DPO 也优于 offline DPO，但 Flow-GRPO 稳定性和性能更优）。

## 关键设计选择

| 参数 | 推荐值 | 影响 |
|---|---|---|
| 组大小 G | 24 | 过小→训练崩溃 |
| 噪声水平 a | 0.7 | 过小→探索不足，过大→画质降 |
| KL 系数 β | 0.04（GenEval/OCR）/ 0.01（PickScore） | 过小→Reward Hacking |
| 训练步数 T | 10 | 平衡速度与稳定 |

## 泛化能力

- **未见物体**：在 60 个训练物体类别上训练，在 20 个未见类别上评估 → Overall 从 0.63 提至 0.90
- **未见计数**：训练在 2-4 个物体，评估在 5-6 个（0.13→0.48）和 12 个（0.02→0.12）
- **跨 benchmark**：在 T2I-CompBench++ 上全面超越 SD3.5-M 和 FLUX.1 Dev

## 局限与前景

- 当前仅在 T2I 上验证，视频生成的潜力尚待探索
- 多奖励平衡（真实感/平滑/连贯）需要更精细的调参
- 规模扩展至更大分辨率/更长序列仍需更高效的采样管线

[^src-2505-05470]: [[source-flow-grpo]]
