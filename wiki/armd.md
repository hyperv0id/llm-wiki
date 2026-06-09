---
title: "ARMD"
type: entity
tags:
  - diffusion-models
  - time-series
  - forecasting
  - arma
  - sliding-window
  - aaai-2025
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: medium
status: active
---

# ARMD

**ARMD**（Auto-Regressive Moving Diffusion，自回归移动扩散）是首个**连续序列扩散** (continuous sequential diffusion) 时间序列预测模型，由 Gao、Cao、Chen 提出（arXiv:2412.09328，2024；AAAI 2025）[^src-armd]。它把扩散过程从"加噪—去噪"重新诠释为时间序列的"演化—去演化" (evolution–devolution)，用[[sliding-window-diffusion|滑动窗口]]产生的确定性中间态取代白高斯噪声，从而让扩散采样过程与预测目标天然对齐[^src-armd]。

## 动机：扩散机制与 TSF 目标的失配

传统扩散类预测器（如 [[timegrad|TimeGrad]]、CSDI、Diffusion-TS）把真实序列前向扩散成白高斯噪声，再以历史序列为条件去噪生成未来——这是一种条件生成范式[^src-armd]。ARMD 主张这忽视了时间序列连续序列演化的本质，造成扩散机制与 TSF 目标的根本失配，并丢弃了序列演化途中有价值的中间信息[^src-armd]。

## 核心设计

### 状态重定义

ARMD 把**未来序列**设为扩散初始状态 $X^0_{1:T}$、**历史序列**设为最终状态 $X^T_{-T+1:0}$，上标表扩散状态、下标表时间步；历史长度被设为等于预测长度 $T$，故最大扩散步数 $T$ 等于待预测序列长度[^src-armd]。

### 前向演化：滑动而非加噪

中间态由未来序列按扩散步滑动得到，而非加噪：$X^t_{1-t:T-t}=\mathrm{Slide}(X^0_{1:T}, t)$，其中 $\mathrm{Slide}(X,k)$ 表序列窗口朝历史方向移动 $k$ 步[^src-armd]。借用 DDPM 的形式可写成 $X^t_{1-t:T-t}=\sqrt{\bar\alpha_t}X^0_{1:T}+\sqrt{1-\bar\alpha_t}\,z^t$，其中**演化趋势** $z^t$ 扮演 DDPM 中"噪声"的角色，但因每步中间态确定，$z^t$ 可由 $X^t$ 与 $X^0$ 闭式解出，作为优化目标的 ground truth[^src-armd]。这是一个确定性过程，消除了加噪引入的不确定性[^src-armd]。详见[[sliding-window-diffusion]]。

### 反向去演化：基于距离的线性网络

反向过程用一个**线性骨干**的去演化网络 $R(\cdot)$ 从历史序列迭代生成未来序列[^src-armd]。它采用[[distance-based-devolution|基于距离的方法]]：先用线性层估计中间态到目标的距离 $D=\mathrm{Linear}(X^t_{1-t:T-t})$，再以随 $t$ 递减的权重 $W(t)\in[0,1]$（以 DDPM 的 $\bar\alpha_t$ 初始化并随训练更新）自适应平衡中间态与 $D$：$t$ 大时更依赖 $D$，$t$ 小（接近目标）时输出更接近输入[^src-armd]。训练目标为演化趋势的 L1 损失 $L_\theta=\mathbb{E}_t[|z^t-\hat z(t,\theta)|]$[^src-armd]。

### 采样：去掉噪声项的 DDIM

采样沿 DDIM 形式，把预测噪声 $\epsilon_\theta$ 替换为预测演化趋势 $\hat z(t,\theta)$，并因演化确定性而**移除随机项** $\sigma_t\epsilon_t$，支持每步跳过 $k$ 步以加速[^src-armd]。最终从历史序列出发逐步变换为未来序列，无需条件生成，得到无条件扩散预测器[^src-armd]。

## 与 ARMA 的理论联系

[[arma-inspired-diffusion|ARMA 启发的扩散]]为 ARMD 提供理论依据：滑动步 $X^t\to X^{t+k}$ 等价于按 ARMA 假设向序列注入噪声（移动平均/MA 视角），而线性去演化网络把每个点建模为前 $k$ 个时间步的线性组合（自回归/AR 视角），两者都与 ARMA 假设一致[^src-armd]。

## 结果与效率

7 个数据集（Solar Energy、Exchange、Stock、四个 ETT，历史=预测=96）上 ARMD 在 14 设置中 12 个最优，显著超越 Diffusion-TS、MG-TSD、TSDiff、[[d3vae|D³VAE]]、TimeGrad；ETTm1 上较 D³VAE 降 47.7% MSE[^src-armd]。线性骨干 + 极少采样步（{1,2,3,4,6,8,12} grid search，对手用 100 步）带来逾十倍训练/推理加速[^src-armd]。给定同一历史做 10 次预测，ARMD 比 Diffusion-TS 更稳定[^src-armd]。

## 消融要点

- 滑动法 > 插值法（保持序列连续性）[^src-armd]
- 距离法 > t-embedding 法[^src-armd]
- 线性骨干 > Transformer 骨干（14 设置中 11 个）[^src-armd]
- 训练加微小 deviation 可防过拟合；采样加噪反损性能[^src-armd]

## 局限

历史长度被绑定为等于预测长度；确定性采样使其偏点预测而非完整概率分布；$b,c,d$ 需逐数据集 grid search；评测限于 96-96[^src-armd]。

## 相关工作

ARMD 与同样用"中间态/退化"替代噪声的[[sliding-window-diffusion|广义扩散]]思路相关：[[cold-sampling|Cold Sampling]]/Cold Diffusion 用任意退化算子、[[dyffusion|DYffusion]]用时间插值作为扩散步、mr-Diff 用季节-趋势分解、MG-TSD 用多粒度。ARMD 的独特之处是用时间轴上的**滑动窗口**直接把"历史→未来"映射成扩散链[^src-armd]。

[^src-armd]: [[source-armd]]
