---
title: "TimeBase: The Power of Minimalism in Efficient Long-term Time Series Forecasting"
type: source-summary
tags:
  - time-series-forecasting
  - linear-model
  - periodicity
  - low-rank
  - orthogonal-basis
  - plug-and-play
  - icml-2025
created: 2026-11-24
last_updated: 2026-11-27
source_count: 1
confidence: medium
status: active
---

# TimeBase: The Power of Minimalism in Efficient Long-term Time Series Forecasting

长时序预测惯以大参数拟合逐点依赖，而一维时间序列高度规则、近似低秩，真实自由度很少。TimeBase 的巧思是把「逐点映射」改写为「先提基、再重构」：按主周期把历史窗口切段，一个小线性层压出少数互不冗余的时间基，另一个线性层从时间基直接生成未来各段。预测器只剩两层极小线性层，以百参数量级逼近强基线；同一提基投影还可前置嵌入 PatchTST，在注意力编码前大幅削减 token 数与计算量[^src-timebase]。

## 核心做法

TimeBase 不引入新算子，把三步标准操作组合成纯线性预测器，各变量通道独立[^src-timebase]：

- **周期分段与基提取**：按主周期（先验或 FFT 主波长）把历史窗口重排为「段数 × 段内步长」矩阵，用一个线性投影压出极少数可学习时间基，近似整个窗口的低秩结构[^src-timebase]。
- **段级重构**：第二个线性层从时间基直接生成未来各周期段，展平为预测序列，全程无非线性激活或门控[^src-timebase]。
- **正交约束**：训练时惩罚不同时间基间的两两重叠，令各基各司其职、抑制冗余[^src-timebase]。

## 证据

1. **定位与主结论**：以百参数级逼近强基线——17 个常规基准 16 个进 Top-2（例外 ETTh2）；作为插件可削减 PatchTST 约八至九成乘加计算且精度大体持平[^src-timebase]。
2. **真实涨点归因**：唯一消融是基底数——过少明显掉点、四个以上趋稳；缺组件级消融[^src-timebase]。
3. **证据缺口与负结果**：插件化在约四分之一指标轻微退回；依赖周期先验、通道独立[^src-timebase]。

## 相关页面

- [[ltsf-linear]] — 瓶颈接力 $\Delta$：把点级全连接映射改写为段级低秩基映射，承接「线性即够用」并大幅收紧参数。
- [[sparsetsf]] — 同槽位互斥：同为周期切分极简线性路线，SparseTSF 共享单一线性头映射周期内子序列，TimeBase 改为少量可学习正交基的分段低秩重构。
- [[cyclenet]] — 同槽位互斥：同以周期为先验，CycleNet 按墙钟相位查可学习周期表做残差剥离加回，TimeBase 不查全局相位、直接从输入分段提基。
- [[patchtst]] — 跨槽位正交：TimeBase 作为前置基提取压缩层嵌入 PatchTST，编码前削减 token 数与乘加量，骨干不变。

> 出版：ICML 2025（PMLR 267）；Qihe Huang、Zhengyang Zhou 等，中国科学技术大学等机构。

[^src-timebase]: [[source-timebase]]
