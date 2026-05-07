---
title: "Spectral Graph Wavelet Transform"
type: technique
tags:
  - spectral-graph-theory
  - graph-wavelet
  - signal-processing
  - graph-neural-networks
created: 2026-05-07
last_updated: 2026-05-07
source_count: 1
confidence: medium
status: active
---

# Spectral Graph Wavelet Transform (SGWT)

SGWT 是一种在图信号上进行多尺度谱分解的方法，由 Hammond, Vandergheynst & Gribonval（2011）提出。

## 工作原理

1. 计算归一化拉普拉斯矩阵 $\Delta = I - D^{-1/2}AD^{-1/2}$ 的谱分解 $\Delta = U\Lambda U^\top$。
2. 使用一组带通核 $\{g_s\}_{s=1}^S$ 和低通核 $h$ 在谱域进行滤波。
3. 小波系数通过谱卷积 $W_{s,t} = U g_s(\Lambda) U^\top X_t$ 计算。
4. 低频分量：低通输出 $V_t$ + 大尺度小波系数（$s \ge k$）。
5. 高频分量：小尺度小波系数（$s < k$）。[^src-scale]

## 在 SCALE 中的应用

SCALE 使用 SGWT 将图结构 MTS 残差分解为低/高频分量。低频包含全局趋势（强耦合、不可交换），高频包含局部变化（弱耦合、近似可交换）。通过 SGCE 概念，在条件化低频后对高频进行共形校准。[^src-scale]

## 关联方法

- [[scale]] — 基于 SGWT 的共形预测方法
- [[spectral-graph-conditional-exchangeability]] — 谱域条件可交换性

[^src-scale]: [[source-scale]]
