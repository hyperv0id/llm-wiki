---
title: "ODE-to-SDE 转换：流匹配中的随机采样"
type: technique
tags:
  - flow-matching
  - ode-sde-conversion
  - reinforcement-learning
  - stochastic-differential-equation
created: 2026-05-12
last_updated: 2026-05-12
source_count: 1
confidence: high
status: active
---

# ODE-to-SDE 转换：流匹配中的随机采样

> 将确定性 ODE 采样转换为等效的随机微分方程（SDE），在保持边缘分布的前提下引入随机性，使得在线强化学习（GRPO）可应用于流匹配模型。[^src-2505-05470]

## 问题背景

[[flow-matching|流匹配模型]] 使用确定性 ODE 进行推理：

\[
dx_t = v_t\,dt
\]

这种确定性映射意味着给定初始噪声，输出图像完全确定——不满足 RL 对随机探索的需求。将 GRPO 应用于流匹配面临两个技术障碍：

1. 策略概率比 \(r_t^i(\theta)=\frac{p_\theta(x_{t-1}|x_t,c)}{p_{\theta_\text{old}}(x_{t-1}|x_t,c)}\) 在确定性动力学下需估计散度，计算成本极高
2. RL 依赖探索多样性，确定性采样（除初始种子外无随机性）严重降低训练效率

## 完整数学推导

### 步骤 1：构造等边缘分布的 SDE

考虑通用 SDE 形式：

\[
dx_t = f_{\text{SDE}}(x_t, t)dt + \sigma_t dw
\]

其边缘概率密度 pt(x) 按 Fokker-Planck 方程演化：

\[
\partial_t p_t(x) = -\nabla \cdot [f_{\text{SDE}}(x_t,t)p_t(x)] + \frac{1}{2}\nabla^2 [\sigma_t^2 p_t(x)]
\]

而 ODE 对应的边缘密度演化为：

\[
\partial_t p_t(x) = -\nabla \cdot [v_t(x_t)p_t(x)]
\]

令两者相等，得到漂移系数关系：

\[
-\nabla \cdot [f_{\text{SDE}}\,p_t] + \frac{1}{2}\nabla^2 [\sigma_t^2 p_t] = -\nabla \cdot [v_t\,p_t]
\]

利用恒等式 \(\nabla^2[\sigma_t^2 p_t] = \sigma_t^2 \nabla \cdot [p_t \nabla \log p_t]\)，解得：

\[
f_{\text{SDE}} = v_t + \frac{\sigma_t^2}{2} \nabla \log p_t(x)
\]

### 步骤 2：从 Forward SDE 推导 Reverse SDE

前向 SDE 的完整形式：

\[
dx_t = \left(v_t + \frac{\sigma_t^2}{2} \nabla \log p_t(x)\right) dt + \sigma_t dw
\]

根据逆向 SDE 的标准关系（前向 \(dx_t = fdt + gdw\) → 逆向 \(dx_t = (f - g^2 \nabla \log p_t)dt + gdw\)），代入 g(t)=σt：

逆向 SDE：

\[
dx_t = \left(v_t + \frac{\sigma_t^2}{2}\nabla \log p_t - \sigma_t^2 \nabla \log p_t\right) dt + \sigma_t dw
\]

\[
dx_t = \left(v_t - \frac{\sigma_t^2}{2}\nabla \log p_t(x)\right) dt + \sigma_t dw
\]

### 步骤 3：链接速度场与得分函数

对流匹配的线性插值 \(x_t = (1-t)x_0 + t x_1\)，条件分布为：

\[
p_{t|0}(x_t|x_0) = \mathcal{N}(x_t\,|\,(1-t)x_0,\,t^2 I)
\]

条件得分：

\[
\nabla \log p_{t|0}(x_t|x_0) = -\frac{x_t - (1-t)x_0}{t^2} = -\frac{x_1}{t}
\]

边际得分（通过条件期望）：

\[
\nabla \log p_t(x_t) = -\frac{1}{t}\mathbb{E}[x_1|x_t]
\]

速度场可通过 x0、x1 的条期望表达：

\[
v_t(x) = \mathbb{E}[\dot{\alpha}_t x_0 + \dot{\beta}_t x_1\,|\,x_t=x]
\]

代入 αt = 1−t，βt = t(故 α̇t = −1，β̇t=1) 并利用 \(x_t = (1-t)x_0 + t x_1\)：

\[
\begin{aligned}
v_t(x) &= (-1)\mathbb{E}[x_0|x_t=x] + (1)\mathbb{E}[x_1|x_t=x] \\
&= -\mathbb{E}\left[\frac{x_t - t x_1}{1-t}\Big|x_t=x\right] + \mathbb{E}[x_1|x_t=x] \\
&= -\frac{x}{1-t} + \frac{t}{1-t}\mathbb{E}[x_1|x_t=x] + \mathbb{E}[x_1|x_t=x] \\
&= -\frac{x}{1-t} - \frac{t}{1-t}\nabla \log p_t(x)
\end{aligned}
\]

解出得分函数的显式表达：

\[
\boxed{\nabla \log p_t(x) = -\frac{x}{t} - \frac{1-t}{t}v_t(x)}
\]

### 步骤 4：最终 SDE 形式

将得分函数代入逆向 SDE（Eq. 21）：

\[
\begin{aligned}
dx_t &= \left(v_t - \frac{\sigma_t^2}{2}\left(-\frac{x_t}{t} - \frac{1-t}{t}v_t\right)\right) dt + \sigma_t dw \\
&= \left(v_t + \frac{\sigma_t^2}{2t}x_t + \frac{\sigma_t^2(1-t)}{2t}v_t\right) dt + \sigma_t dw
\end{aligned}
\]

整理得最终 SDE：

\[
\boxed{dx_t = \left[v_t(x_t) + \frac{\sigma_t^2}{2t}\big(x_t + (1-t)v_t(x_t)\big)\right] dt + \sigma_t dw}
\]

### 步骤 5：Euler-Maruyama 离散化

将连续 SDE 离散化为可执行的采样步骤：

\[
\boxed{x_{t+\Delta t} = x_t + \left[v_\theta(x_t,t) + \frac{\sigma_t^2}{2t}\big(x_t + (1-t)v_\theta(x_t,t)\big)\right]\Delta t + \sigma_t \sqrt{\Delta t}\,\epsilon}
\]

其中 ε ∼ N(0,I) 注入高斯随机性。

### 步骤 6：噪声调度

实验中使用 \(\sigma_t = a\sqrt{\frac{t}{1-t}}\)，其中 a 为超参数控制噪声（探索）水平：
- a=0.1：探索不足，奖励增长缓慢
- a=0.7：最优，探索充分
- a=1.0：噪声过多，画质下降

### KL 散度闭式解

策略 πθ(xt−1|xt,c) 退化为各向同性高斯分布，KL 散度可解析计算：

\[
D_{\mathrm{KL}}(\pi_\theta\|\pi_{\text{ref}}) = \frac{\|x_{t+\Delta t,\theta} - x_{t+\Delta t,\text{ref}}\|^2}{2\sigma_t^2\Delta t} = \frac{\Delta t\,\sigma_t(1-t)}{2} \cdot \frac{1}{2t} \cdot \|v_\theta(x_t,t) - v_{\text{ref}}(x_t,t)\|^2
\]

## 关键性质

- **边缘分布保持**：SDE 与 ODE 在所有时间步具有完全相同的边缘概率分布
- **计算高效**：无需估计散度或重训练模型，仅需在采样时注入高斯噪声
- **已集成到框架中**：当前 SOTA 的 T2I 流匹配模型（SD3.5、FLUX）均可直接使用

[^src-2505-05470]: [[source-flow-grpo]]
