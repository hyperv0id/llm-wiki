# Ingest 报告：TimeMixer

**日期**：2026-05-31
**源文件**：Wang et al. (2024) - TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting (ICLR 2024)

## 创建

- `wiki/source-timemixer.md` — 源文件摘要，PDM + FMM 双模块全 MLP 多尺度混合架构
- `wiki/timemixer.md` — 实体页面，TimeMixer 模型、架构细节、实验性能、消融分析及与其他模型的全面关系

## 新建交叉链接

- `timemixer` ↔ `autoformer` — 继承 SeriesDecomp 分解模块，扩展到多尺度混合
- `timemixer` ↔ `fedformer` — vs MOEDecomp 分解策略对比
- `timemixer` ↔ `dlinear` — TimeMixer 将分解嵌入深度网络 vs DLinear 的预处理分解
- `timemixer` ↔ `patchtst` — vs CI 策略在多变量场景下的退化
- `timemixer` ↔ `itransformer` — MLP 混合 vs attention 机制在时间维度建模上的对比
- `timemixer` ↔ `timesnet` — 多尺度下采样混合 vs 多周期 1D→2D 卷积
- `timemixer` ↔ `cyclenet` — 多尺度隐式周期性建模 vs 显式全局周期残差
- `timemixer` ↔ `lstf` — LSTF benchmark SOTA 归属

## 关键洞察

TimeMixer 是时序预测领域重要的**MLP-only**标杆。不同于 PatchTST（Transformer + CI + patching）和 iTransformer（inverted attention），TimeMixer 证明精心设计的多尺度 MLP 混合架构可以在没有任何注意力机制的情况下达到 SOTA。其 24.7% Solar-Energy MSE 降幅和 M4 全面领先尤为显著。

Seasonal bottom-up / Trend top-down 的双向混合设计源于对时序的本质理解：季节性由小周期聚合为大周期，趋势由宏观支配微观。这一设计选择不是经验试出来的，而是由时序本身的性质推导的。
