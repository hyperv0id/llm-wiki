---
title: "SCALE"
type: entity
tags:
  - conformal-prediction
  - spectral-graph-theory
  - graph-wavelet
  - icml-2026
  - time-series
created: 2026-05-07
last_updated: 2026-05-07
source_count: 1
confidence: medium
status: active
---

# SCALE

**Spectral Conformal prediction via wAveLEt transform** — 由 Guo, Han, Luo, Liu, Gong & Wang（上海交通大学，ICML 2026）提出的图结构多变量时间序列共形预测方法。[^src-scale]

## 架构

SCALE 包含三个主要通道：

1. **谱分解（SGWT）**：通过谱图小波变换将残差快照分解为低频和高频分量。使用归一化拉普拉斯矩阵 $\Delta = I - D^{-1/2}AD^{-1/2}$ 的谱分解，带通核 $\{g_s\}_{s=1}^S$ 和低通核 $h$。截止尺度 k 通过自动化 SGWT 诊断选择。[^src-scale]

2. **低频通道（可学习）**：STID 风格编码器，将低频历史 $\hat{L}_{t-W+1:t}$ 通过 MLP、空间嵌入 $E_s(v)$ 和周期时间嵌入 $E_p(t)$ 映射为条件嵌入 $C_t$。[^src-scale]

3. **高频通道（参数免费）**：计算每个节点的标准差和 RMS 作为时间统计特征 $M_t$。设计上避免可学习网络，防止重新引入空间耦合。[^src-scale]

**自适应门控融合**：$G_{t+1:t+K} = \text{Sigmoid}(\text{Proj}_{\text{gate}}(C_t))$，$\hat{Q}^\alpha = Z^L + G \odot Z^H$。网络端到端训练，最小化分位数回归损失。[^src-scale]

## 理论

- **SGCE（谱图条件可交换性）**：高频序列 $\{H_t\}$ 在条件化于低频序列 $\{L_t\}$ 后可交换。
- **有限样本覆盖**：在 SGCE 假设下满足。
- **近似覆盖**：当考虑小波近似误差时，覆盖间隙 $\delta = \delta_{leak} + \delta_{dep}$ 可控。[^src-scale]

[^src-scale]: [[source-scale]]
