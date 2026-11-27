---
title: "CATS: Enhancing Multivariate Time Series Forecasting by Constructing Auxiliary Time Series as Exogenous Variables"
type: source-summary
tags:
  - time-series-forecasting
  - exogenous
  - channel-independence
  - mlp
  - plug-and-play
  - icml-2024
created: 2026-11-24
last_updated: 2026-11-27
source_count: 1
confidence: medium
status: active
---

# CATS: Enhancing Multivariate Time Series Forecasting by Constructing Auxiliary Time Series as Exogenous Variables

多变量时序预测有个反直觉现象：忽略序列间关系的单变量模型常胜过多变量模型——真实基准的跨序列关系偏弱，强行混合会把伪相关当模式学进去[^src-cats]。CATS（Constructing Auxiliary Time Series）的巧思是不动预测器主体（哪怕只是 2 层 MLP），从原始序列内部派生「虚拟外生辅助序列」承载跨序列交互，并行预测后以残差折回原通道，相当于自适应的时间×通道二维注意力[^src-cats]。连续性平滑与双维稀疏截断让辅助通道按需介入；论文报告固定结构即可在主流基准领先，开销远低于 Transformer 类模型[^src-cats]。

## 核心做法

多个构造器从原始序列（OTS，Original Time Series）并联生成辅助序列（ATS，Auxiliary Time Series）；全部序列拼接送入共享预测器（默认 2 层 MLP）并行预测；ATS 预测经线性投影折回原通道、以残差加回[^src-cats]。这些异构构造器均由标准算子组成，涵盖多尺度一维卷积、非重叠步长卷积（等价 patch 聚合）、按通道分组卷积、跨通道线性投影、恒等复制与时间步 embedding[^src-cats]。三原则：连续性（Continuity）以平滑损失滤高频噪声；稀疏性（Sparsity）通道维可学习门控抑制无用辅助序列，时间维软截断经直通估计器（straight-through estimator, STE）训练，弱耦合时退回近似单变量；多样性（Variability）异构构造器覆盖不同范围依赖[^src-cats]。OTS 残差捷径保结果不差于基座[^src-cats]。

## 证据

1. **定位与主结论**：固定 2 层 MLP 在 36 个设定中 28 个最优，强跨序列合成集优势最大，开销远低于 Transformer 类基线[^src-cats]。
2. **真实涨点归因**：强耦合集去双维稀疏掉点最狠、弱耦合集去 OTS 捷径最狠；缺逐类构造器消融与等算力对照[^src-cats]。
3. **证据缺口与负结果**：对最强基线的平均优势不大，且缺乏与真实外部外生变量来源的同台对照[^src-cats]。

## 局限

论文自述 ATS 条数固定、训练中无法自动扩展[^src-cats]。

## 相关页面

- [[channel-independence]] / [[patchtst]] — 瓶颈接力 $\Delta$：通道独立丢弃跨序列信息换稳健，CATS 以可稀疏截断的辅助通道受控补回。
- [[source-timexer]] / [[crosslinear]] — 同槽位互斥：依赖真实外部外生变量辅助预测，CATS 全部自造虚拟外生序列。
- [[ltsf-linear]] — 复现对齐：以 DLinear 为基线与插件宿主；CATS 固定长输入配时间截断，DLinear 逐设定调优。
- [[autoformer]] — 跨槽位正交：CATS 可作为前置虚拟外生构造插件接入 Autoformer 等 Transformer 预测骨干。

> 来源：Jiecheng Lu, Xu Han, Yan Sun, Shihao Yang, "CATS: Enhancing Multivariate Time Series Forecasting by Constructing Auxiliary Time Series as Exogenous Variables", ICML 2024 (PMLR 235), Georgia Tech / AWS。

[^src-cats]: [[source-cats]]
