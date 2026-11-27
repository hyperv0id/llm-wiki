---
title: "DSR-TSF: Spectrum-Driven Dynamic Routing for Efficient Long-Horizon Time Series Forecasting"
type: source-summary
tags:
  - time-series-forecasting
  - periodicity
  - dynamic-routing
  - low-rank
  - lightweight
  - neurips-2026
created: 2026-11-24
last_updated: 2026-11-27
source_count: 1
confidence: medium
status: active
---

# DSR-TSF: Spectrum-Driven Dynamic Routing for Efficient Long-Horizon Time Series Forecasting

SparseTSF 把周期当常量：固定周期折段、跨段共享头预测。但同一变量在不同历史窗口的主导周期会漂移（ETTh1 油温依次 720、480、206 小时），固定步长便与当前频谱错配。DSR-TSF 不重采样也不逐窗对齐：用输入频谱摘要做软路由，加权组合多条可学习时域响应分支并残差加回，再走原有分段与共享头。主结论是千参数预算下相对 SparseTSF 小幅降 MSE[^src-dsr-tsf]。

## 核心做法

骨架不动，只在分段前插入谱条件软路由[^src-dsr-tsf]。

**频谱门控**：去均值后做实数傅里叶变换并排除直流分量，取幅值最大的前 $K$ 个非零频率，用其对数幅值与归一化频率位置拼成短向量；它不预测，只经小门控配合静态先验（与输入无关的可学习分支基础权重）给出分支权重[^src-dsr-tsf]。

**时域响应分支**：每条是固定核长的一维可学习响应算子，不估周期也不绑定物理周期，只作分段前的响应基，加权后残差加回输入[^src-dsr-tsf]。

**低秩共享头**：序列按固定周期折成段内位置与历史段数两维，各位置共用一个跨段预测头；低秩在此限制映射自由度、注入先验，而非压缩参数[^src-dsr-tsf]。

## 证据

1. **定位与主结论**：千参数预算下相对 SparseTSF 小幅降 MSE（七集均值约 $2.4\%$），逼近 MixLinear 等强基线[^src-dsr-tsf]。
2. **真实涨点归因**：将响应算子核长对齐至同感受野的对照变体与 SparseTSF 持平，排除感受野解释；去掉低秩约束与改用输入无关静态路由的掉点幅度相当[^src-dsr-tsf]。
3. **证据缺口**：高维 Traffic 无明显优势、MAE 更弱；分支数与周期敏感性未报告[^src-dsr-tsf]。

## 相关页面

- [[sparsetsf|SparseTSF]] · [[cross-period-sparse-forecasting|Cross-Period Sparse Forecasting]] — 瓶颈接力 $\Delta$：折段骨架与跨段共享头沿用，仅在分段前补谱条件软路由。
- [[frequency-pathway-routing|Frequency Pathway Routing]] — 同构对比：同以频域门控，对象是时域响应基而非独立频带。
- [[mixture-of-experts|Mixture of Experts]] — 软 MoE 特例：静态先验加频谱门控的加权求和，无稀疏截断。
- [[cyclenet|CycleNet]] · [[hybrid-periodicity-decoupling|Hybrid Periodic Decoupling]] — 跨槽位正交：显式周期表查表 vs 单窗谱驱动响应修正。

*Hua Wang、Bing Li（鲁东大学），Fan Zhang（山东工商学院）；NeurIPS 2026。*

[^src-dsr-tsf]: [[source-dsr-tsf]]
