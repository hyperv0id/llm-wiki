---
title: "SLiDE: Structured Linear Dynamics for Forecasting with Exogenous Inputs"
type: source-summary
tags:
  - source
  - time-series-forecasting
  - exogenous
  - koopman-operator
  - linear-dynamics
  - ood-generalization
  - neurips-2026
created: 2026-11-24
last_updated: 2026-11-24
source_count: 1
confidence: medium
status: active
---

# SLiDE: Structured Linear Dynamics for Forecasting with Exogenous Inputs

S. Pütz、T. Glavas、B. Schäfer（KIT / Helmholtz AI）、B. N. Oreshkin（Amazon Science）、M. Coates（McGill, Mila），NeurIPS 2026[^src-slide]。

## 核心问题与机制

多数外生预测模型采用交叉注意力或深层 MLP 做柔性特征融合，当测试期外生变量超出训练分布范围（OOR）或驱动机制切换时容易过拟合虚假相关[^src-slide]。SLiDE 在潜空间构建结构化线性动力学系统：先用跨步共享编码器将外生与目标输入映射为 $p_\tau = \operatorname{MLP}_u(u_\tau)$ 与 $d_\tau = \operatorname{MLP}_y(y_\tau)$，再以 $N=2$ 分块对角线性矩阵 $W$ 执行滞后不变循环编码得到初态 $z_k = \pi_k$（$\pi_\tau = W\pi_{\tau-1} + d_\tau + p_\tau$，$\pi_0 = 0$）[^src-slide]。未来滚动直接复用同一线性矩阵 $W$ 与同一外生编码器 $\operatorname{MLP}_u$ 执行纯加性推动 $z_{k,h} = W z_{k,h-1} + p_{k,h}$，最终经分块非线性解码求和输出 $\hat{y}_{k,h} = \sum_{b=1}^{N} \operatorname{MLP}_{\hat{y}}(z_{k,h}^b)$，放弃状态相关控制项 $B(z,u)$ 与线性观测矩阵 $C$[^src-slide]。详细维度流与结构对照见 [[slide]]。

## 证据

在 EPF-5、Favorita 与 Bouc-Wen 共 8 项任务中 8 次进前二、平均排名 1.87 居首，推理速度接近 TSMixer，在 EPF-DE OOR 与跨驱动迁移分别降错约 20% 与 30%；非线性替换消融将 $W$ 换为两层 ReLU MLP 后，Bouc-Wen 同分布精度更佳但迁移崩溃且 Favorita OOR 落后约 18%，证实线性转移专门服务于分布偏移泛化[^src-slide]。

## 局限

依赖近似有限维潜线性假设且不支持连续谱，循环稳定性仅经经验检验；缺失或时间错位的外生输入会直接带入滚动，且 OOR 验证样本量偏小[^src-slide]。

[^src-slide]: [[source-slide]]
