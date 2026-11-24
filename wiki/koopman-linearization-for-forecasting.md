---
title: "Koopman Linearization for Forecasting"
type: concept
tags:
  - koopman-operator
  - time-series
  - dynamical-systems
  - linearization
  - forecasting
created: 2026-06-08
last_updated: 2026-11-24
source_count: 4
confidence: high
status: active
---

# Koopman Linearization for Forecasting

> **编排者备注（去重）**：本页是从 [[k2vae|K²VAE]] ingest 中提议的通用概念页。仓库中已有 [[micro-macro-coupled-koopman-modeling]] 包含一节 "Koopman Operator Theory"，并存在 `koopman-operator` 标签（无同名页面）。请将本页与 MMCKM 的 Koopman 理论段落合并/去重，作为多个 Koopman 时间序列模型（K²VAE、MMCKM）共享的概念枢纽。

**Koopman 线性化**是一类把非线性时间序列预测转化为**线性动力系统**建模的范式。其理论基础是 Koopman 理论 (Koopman, 1931)：对非线性系统 $x_{k+1}=f(x_k)$，存在一个（理论上无限维的）**测量函数空间** $\psi$，系统状态映射到该空间后，其演化可由一个**线性 Koopman 算子** $\mathcal{K}$ 描述[^src-k2vae]：

$$\psi(x_{k+1}) = \psi(f(x_k)) = \mathcal{K}\circ\psi(x_k)$$

## 为什么对预测有用

非线性使概率/确定性模型都难以写出简洁的状态转移方程，长期预测尤其困难。把系统提升 (lift) 到测量空间后获得线性演化，带来三个好处[^src-k2vae]：

1. **建模简化**：线性系统的状态转移、不确定性传播都有闭式或可微形式；
2. **长期稳健**：线性外推比逐步非线性回归更不易累积误差；
3. **高效**：避免扩散/流模型的多步迭代采样。

## 有限维近似

无限维 Koopman 算子需有限维近似。常见做法[^src-k2vae]：

- **DMD / eDMD (Dynamic Mode Decomposition / extended DMD)**：通过矩阵计算（伪逆）从观测数据拟合线性算子。K²VAE 用 **one-step eDMD** 拟合局部算子 $K_{loc}=X_{fore}(X_{back})^{\dagger}$，并加全局可学习项 $K_{glo}$ 兜底数值不稳定[^src-k2vae]。
- **神经网络测量函数**：用 MLP 等学习 $\psi$（及其逆 $\psi^{-1}$ 作为解码器），数据驱动地构造测量空间。
- **酉谱参数化一步演化**：[[source-sonnet|Sonnet]] 对复矩阵 $S\in\mathbb{C}^{K\times K}$ 每次前向做 `QR` 分解得酉矩阵 $U$，构造 $\mathcal{K}=UDU^\dagger$（$D=\operatorname{diag}(e^{ip_k})$）端到端一次性演化整段频谱特征以规避逐步递归误差；但每次前向需执行 `QR` 分解且 AAAI 正文将可学习小波与酉约束的独立消融推至附录[^src-sonnet]。

## 关键性质与代表模型

- **Markov 性 / history-free**：当测量函数时不变时，预测仅依赖当前状态而非历史轨迹（[[mmckm|MMCKM]] 据此实现 history-free 交通预测）[^src-mmckm]。
- **控制输入扩展（同槽位互斥分歧）**：传统 Koopman-with-input 允许控制矩阵依赖状态（$B(z,u)$）或通过残差/注意力注入——[[mmckm|MMCKM]] 用 CrossAttention 注入宏观流影响；[[k2vae|K²VAE]] 用 KalmanNet 注入非线性残差以精炼不确定性[^src-k2vae][^src-mmckm]。与此**同槽位互斥**的是 [[slide|SLiDE]]（NeurIPS 2026）：它是目前唯一以外生纯加性推力（放弃状态相关 $B(z,u)$，改用跨步共享 $\operatorname{MLP}_u(u_\tau)$）驱动共享分块线性算子 $W$、并在输出端改用分块非线性 MLP 替代线性观测矩阵 $C$ 的预测口径，借严格容量约束换取分布外（OOR）与跨驱动迁移泛化[^src-slide]。

| 模型 | 线性算子构造 | 控制 / 外生输入形式 | 输出映射 | 目标任务与预测性质 |
|---|---|---|---|---|
| [[k2vae\|K²VAE]] | one-step eDMD $K_{loc} + K_{glo}$ | 无外生变量；KalmanNet 注入非线性残差[^src-k2vae] | 双 MLP 逆映射解码 | 长期概率分布预测（VAE 一步生成） |
| [[mmckm\|MMCKM]] | 5 模态参数有界 Koopman MoE | CrossAttention 状态耦合控制 $B_z u_t$[^src-mmckm] | 线性 / MLP 解码 | history-free 微观交通轨迹预测 |
| [[slide\|SLiDE]] | 历史编码与未来滚动共享 $N=2$ 分块 $W$ | 纯加性外生推力 $\operatorname{MLP}_u(u_\tau)$[^src-slide] | $N=2$ 分块 $\operatorname{MLP}_{\hat{y}}$ 求和（非 $C$） | 外生条件确定性点预测（侧重 OOR 与迁移） |
| [[source-sonnet\|Sonnet]] | `QR` 分解酉参数化 $\mathcal{K}=UDU^\dagger$ | 小波嵌入后沿变量轴 `FFT` 互谱相干注意力（MVCA）[^src-sonnet] | $[5,3,3]$ 三层 1D 卷积 + 线性头 | 窗末单点 $\hat{y}_{t+H}$ 确定性预测（外生变量） |

## 关联页面

- [[slide]] — 编码与滚动共享线性算子 $W$ + 纯加性外生驱动的轻量预测架构
- [[source-sonnet]] — 可学习小波 + 变量轴 `FFT` 频谱相干 + 酉约束 Koopman 一步频谱演化（特例对照）
- [[k2vae]] — Koopman 线性化 + Kalman 精炼的概率预测
- [[mmckm]]、[[micro-macro-coupled-koopman-modeling]] — 微观/宏观双尺度 Koopman 交通模型
- [[kalmannet-uncertainty-modeling]] — 在 Koopman 线性系统上做 Kalman 精炼
- [[generative-time-series-forecasting]] — 生成式预测范式

[^src-k2vae]: [[source-k2vae]]
[^src-mmckm]: [[source-mmckm]]
[^src-slide]: [[source-slide]]
[^src-sonnet]: [[source-sonnet]]

