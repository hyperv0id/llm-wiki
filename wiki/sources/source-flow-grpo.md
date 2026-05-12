---
title: "Flow-GRPO: Training Flow Matching Models via Online RL"
type: source-summary
tags:
  - flow-matching
  - reinforcement-learning
  - text-to-image
created: 2026-05-12
last_updated: 2026-05-12
source_count: 1
confidence: high
status: active
---

# Flow-GRPO: Training Flow Matching Models via Online RL

> **核心贡献**：首次将 GRPO（Group Relative Policy Optimization）在线强化学习引入流匹配生成模型，通过 ODE-to-SDE 转换引入随机性，并采用 Denoising Reduction 策略加速训练。在 GenEval 上将 SD3.5-M 准确率从 63% 提升至 95%，超越 GPT-4o。

**论文标识**：arXiv:2505.05470v5 [cs.CV] (NeurIPS 2025)  
**作者**：Jie Liu, Gongye Liu, Jiajun Liang, Yangguang Li, Jiaheng Liu, Xintao Wang, Pengfei Wan, Di Zhang, Wanli Ouyang  
**代码**：https://github.com/yifan123/flow_grpo

## 核心问题与挑战

流匹配模型虽能生成高质量图像，但在复杂场景组合（多物体、属性绑定、空间关系）和文字渲染上表现不佳。将在线 RL 应用于流匹配面临两个关键挑战：

1. **确定性采样问题**：流匹配基于 ODE（dxt = vt dt），无法随机采样，而 RL 依赖随机性进行探索
2. **采样效率问题**：流匹配生成单张图像需大量迭代步数，在线 RL 数据收集成本过高

## 方法：Flow-GRPO

### 3.1 Flow Matching 基础

定义“加噪”数据 xt：

\[
x_t = (1 - t) x_0 + t x_1, \quad t \in [0,1]
\]

其中 x0 ∼ X0 为真实数据，x1 ∼ N(0,I) 为噪声。训练模型 vθ(xt,t) 回归速度场 v = x1 − x0，优化 Flow Matching 目标：

\[
\mathcal{L}(\theta) = \mathbb{E}_{t, x_0 \sim X_0, x_1 \sim X_1} \| v - v_\theta(x_t, t) \|^2
\]

去噪过程可建模为 MDP：状态 st ≜ (c, t, xt)，动作 at ≜ xt−1，策略 π(at|st) ≜ pθ(xt−1|xt,c)，奖励仅在最终步给定 r(x0,c)。

### 3.2 GRPO 适配流匹配

给定 prompt c，采样 G 张图像 {xi0}G_i=1 及其逆时轨迹，第 i 张图像的 advantage 通过组内归一化计算：

\[
\hat{A}_i^t = \frac{R(x_0^i, c) - \mathrm{mean}(\{R(x_0^i, c)\}_{i=1}^G)}{\mathrm{std}(\{R(x_0^i, c)\}_{i=1}^G)}
\]

GRPO 优化目标：

\[
\mathcal{J}_{\text{Flow-GRPO}}(\theta) = \mathbb{E}_{c \sim C, \{x^i\}_{i=1}^G \sim \pi_{\theta_\text{old}}(\cdot|c)} \left[ \frac{1}{G} \sum_{i=1}^G \frac{1}{T} \sum_{t=0}^{T-1} \min\left( r_t^i(\theta)\hat{A}_i^t,\ \mathrm{clip}(r_t^i(\theta), 1-\varepsilon, 1+\varepsilon)\hat{A}_i^t \right) - \beta D_{\mathrm{KL}}(\pi_\theta\|\pi_{\text{ref}}) \right]
\]

其中重要性权重 \( r_t^i(\theta) = \frac{p_\theta(x_{t-1}^i | x_t^i, c)}{p_{\theta_\text{old}}(x_{t-1}^i | x_t^i, c)} \)。

### 3.3 ODE-to-SDE 转换（核心理论）

确定性 ODE：dxt = vt dt

构造前向 SDE：dxt = fSDE(xt,t)dt + σt dw，要求其边缘分布与 ODE 相同。通过 Fokker-Planck 方程推导，得到：

\[
f_{\text{SDE}} = v_t(x_t) + \frac{\sigma_t^2}{2} \nabla \log p_t(x_t)
\]

即前向 SDE：\( dx_t = \left(v_t(x_t) + \frac{\sigma_t^2}{2} \nabla \log p_t(x_t)\right) dt + \sigma_t dw \)

利用逆向 SDE 公式（[75, 23]），得到逆向 SDE：

\[
dx_t = \left(v_t(x_t) - \frac{\sigma_t^2}{2} \nabla \log p_t(x_t)\right) dt + \sigma_t dw
\]

对于线性插值 xt = (1−t)x0 + t x1，得分函数与速度场的关系为：

\[
\nabla \log p_t(x) = -\frac{x}{t} - \frac{1-t}{t} v_t(x)
\]

代入逆向 SDE 得最终形式：

\[
dx_t = \left(v_t(x_t) + \frac{\sigma_t^2}{2t} (x_t + (1-t)v_t(x_t))\right) dt + \sigma_t dw
\]

Euler-Maruyama 离散化得到更新规则：

\[
x_{t+\Delta t} = x_t + \left[v_\theta(x_t,t) + \frac{\sigma_t^2}{2t}(x_t + (1-t)v_\theta(x_t,t))\right] \Delta t + \sigma_t \sqrt{\Delta t}\,\epsilon,\quad \epsilon \sim \mathcal{N}(0,I)
\]

噪声调度：\( \sigma_t = a \sqrt{\frac{t}{1-t}} \)，a 控制探索程度。

策略 πθ(xt−1|xt,c) 成为各向同性高斯分布，KL 散度可闭式计算：

\[
D_{\mathrm{KL}}(\pi_\theta\|\pi_{\text{ref}}) = \frac{\|x_{t+\Delta t,\theta} - x_{t+\Delta t,\text{ref}}\|^2}{2\sigma_t^2 \Delta t} = \frac{\Delta t \sigma_t (1-t)}{2} \cdot \frac{1}{2t} \cdot \|v_\theta(x_t,t) - v_{\text{ref}}(x_t,t)\|^2
\]

### 3.4 Denoising Reduction

训练时使用少量去噪步数（T_train = 10），推理时保持原有步数（T_infer = 40）。实验表明该策略可加速训练 4 倍以上，且不牺牲最终性能。

## 实验结果

- **GenEval**：SD3.5-M 从 0.63 → 0.95，超越 GPT-4o (0.84)
- **文字渲染**：OCR 准确率从 59% → 92%
- **人类偏好对齐**：PickScore 从 21.72 → 23.31（w/ KL）
- **泛化性**：对未见物体类别（60→20 类）和计数（2-4 → 5-6/12 物体）均表现良好
- **极小 reward hacking**：适当 KL 正则化可保持图像质量和多样性

## 关键消融

- 组大小 G=24 最优，更小组导致优势估计高方差和训练崩溃
- 噪声水平 a=0.7 最佳，过小探索不足，过大毁坏画质
- Denoising Reduction 将 T=40 降至 T=10，速度提升 4 倍，性能无损

## 局限性

- 未在视频生成上验证
- 多奖励平衡需要精细调参
- KL 正则化虽有效但需延长训练时间

[^src-2505-05470]: [[source-flow-grpo]]
