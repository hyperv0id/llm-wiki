---
title: "SCALE: Spectral Conformal Prediction via Wavelet Transform (ICML 2026)"
type: source-summary
tags:
  - conformal-prediction
  - spectral-graph-theory
  - graph-wavelet
  - spatio-temporal
  - graph-neural-networks
  - icml-2026
created: 2026-05-07
last_updated: 2026-05-07
source_count: 1
confidence: medium
status: active
---

# SCALE: Spectral Conformal Prediction via Wavelet Transform

**Authors**: Ruichao Guo*, Xingyao Han*, Wenshui Luo, Zhe Liu, Chen Gong, Hesheng Wang (Shanghai Jiao Tong University)
**Venue**: ICML 2026, Seoul, South Korea
**Link**: https://arxiv.org/abs/2605.04957v1

## Summary

SCALE (Spectral Conformal prediction via wAveLEt transform) 提出了一种用于图结构多变量时间序列共形预测的新框架。核心洞见：图结构时间序列中的跨节点耦合破坏了可交换性假设，但这种耦合主要存在于全局趋势（低频分量）中，而高频分量近似满足可交换性。[^src-scale]

## 核心贡献

1. **谱图条件可交换性（SGCE）**：一个新的概念形式化——高频分量在条件化于低频分量后近似可交换。即 $p(H_{t_1}, \ldots, H_{t_n} | L_{t_1}, \ldots, L_{t_n})$ 在置换下不变。[^src-scale]

2. **SCALE 方法**：
   - 使用谱图小波变换（SGWT）将原始 MTS 分解为低频和高频分量。
   - 低频通道：通过 STID 风格编码器（空间嵌入 + 周期时间嵌入 + 低频嵌入）生成条件嵌入 $C_t$。
   - 高频通道：无参数的时间统计提取器，计算标准差和 RMS，避免重新引入空间耦合。
   - 自适应门控融合：低频嵌入生成门控映射 $G_{t+1:t+K}$，调制高频特征到最终分位数输出。[^src-scale]

3. **理论保证**：
   - **定理 5.1**：在完美 SGCE 假设下，SCALE 预测区间满足有限样本覆盖保证。
   - **定理 5.2**：在 SGWT 近似误差下，覆盖间隙 $\delta = \delta_{leak} + \delta_{dep}$（谱泄漏 + 空间耦合）。
     - 当谱滤波器趋近理想带通且高频耦合消失时，$\delta \to 0$。[^src-scale]

4. **实证结果**：在 METR-LA、PEMS04/07/08 四个交通数据集上，SCALE 在几乎所有设置中取得最优或接近最优的 Winkler Score，在 METR-LA（$\alpha=0.05$）上相比 EnbPI 区间宽度减少约 14.4%。高频耦合分析验证了谱分解的有效性。[^src-scale]

## 关键洞察

- 原始的交叉耦合分析（METR-LA）表明：低频分量具有高相关性（密集耦合），高频分量相关性显著更低（稀疏耦合）。这验证了高频分量更接近可交换性。
- SGCE 的关键设计：不丢弃低频信息，而是将其作为条件用于指导高频通道的分位数校准。[^src-scale]

## 消融实验

- w/o SGWT：失去谱分离导致覆盖违规。
- w/o LF：去除低频条件导致覆盖违规。
- SCALE 的标准版本在所有设置中表现最优。[^src-scale]

[^src-scale]: [[source-scale]]
