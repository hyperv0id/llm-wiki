---
title: "TimeBridge: Non-Stationarity Matters for Long-term Time Series Forecasting"
type: source-summary
tags:
  - time-series-forecasting
  - long-term-forecasting
  - transformer
  - non-stationarity
  - cointegration
  - icml-2025
created: 2026-11-24
last_updated: 2026-11-27
source_count: 1
confidence: medium
status: active
---

# TimeBridge: Non-Stationarity Matters for Long-term Time Series Forecasting

多变量时序预测里，非平稳性有两副相反的面孔：在单变量的短程窗口内，它让无关的随机波动被误读成相关，诱发伪回归；在跨变量的长期尺度上，它又是变量共同漂移、协同变动（协整）关系的载体。既有方法把非平稳性当单一对象处置——要么在入口处全局抹平，要么原样保留——两头都会伤到一边。[[timebridge|TimeBridge]] 的核心巧思是按作用域分工：变量内的时间建模去掉非平稳性以压住伪相关，跨变量的长期建模保留它以承载协整[^src-timebridge]。

## 核心做法

前向流程由标准算子两阶段串行。先把每条变量序列切成互不重叠的 patch 并线性嵌入；变量内阶段做通道独立的时间注意力，但查询与键取自「减去滑动平均趋势」后的稳态分量，值向量保留原始 patch——输出因此天然保持原趋势尺度、无需反归一化，同时抑制短程伪相关（论文称 Integrated Attention）。随后用前馈网络沿时间块轴压缩 patch 数，在保留非平稳性的前提下，对同一时间槽的跨变量切片直接做注意力以捕获长期协整（论文称 Cointegrated Attention），最后展平线性映射出预测[^src-timebridge]。

## 证据

1. **定位与主结论**：论文报告 9 个长期基准中 8 项最优，较 TimeMixer 平均降低 MSE $13.66\%$，短程交通与两大股指金融任务亦占优，属真实精度超越而非低开销逼近[^src-timebridge]。
2. **真实涨点归因**：消融中掉点最狠的是放反去平稳/留平稳作用域与颠倒两阶段顺序，单独移除去趋势算子影响很小——收益来自作用域分工加阶段顺序的主故事，而非平滑算子本身[^src-timebridge]。
3. **证据缺口与负结果**：ETTh2 上落后于 ModernTCN 与 PatchTST，金融任务收益上升但波动与回撤更差；协整假说缺正文形式化证明[^src-timebridge]。

## 相关页面

- [[timebridge]] — 本模型机制页，含分阶段维度流与消融判词
- [[patchtst]] — 瓶颈接力 $\Delta$：继承其 patch 化通道独立时间注意力骨架，关键改动是 $Q/K/V$ 非对称去趋势分工并补入跨变量协作
- [[itransformer]] — 同槽位互斥：对方把整条变量序列压成单一 variate token 交互，本模型在压缩后的 $M$ 个时间槽上逐槽交互、保留更多局部时间分辨率
- [[source-pesd-tsf]]、[[source-renf]] — 下游瓶颈接力与复现对齐：前者直接继承非对称去趋势注意力，后者沿用其时频加权 MAE 损失并重跑对照

---

作者 Peiyuan Liu、Beiliang Wu、Yifan Hu 等（清华大学深圳国际研究生院、深圳大学），ICML 2025（PMLR 267）。

[^src-timebridge]: [[source-timebridge]]
