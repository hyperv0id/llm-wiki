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
last_updated: 2026-11-24
source_count: 1
confidence: medium
status: active
---

# CATS: Enhancing Multivariate Time Series Forecasting by Constructing Auxiliary Time Series as Exogenous Variables

**来源**：Jiecheng Lu、Xu Han、Yan Sun、Shihao Yang（Georgia Tech / AWS）；ICML 2024（PMLR 235）[^src-cats]。

纯多变量模型容易因跨序列伪相关而污染单序列时序建模，于是单变量通道独立（CI）骨干常胜过多变量模型。CATS 不改动轻量化预测器主体，而是先从原始多变量序列（OTS）内部派生一组虚拟辅助外生序列（ATS），把跨序列交互搬运到辅助通道上与原序列一同送入共享的 2 层 MLP 并行预测，再通过 OTS 残差捷径、连续性平滑损失与通道和时间维稀疏截断抑制噪声[^src-cats]。

## 去黑话算子配方与图谱关系

模型由标准卷积、线性投影与门控截断组合而成，既可独立作预测器，也可作外围插件接入 [[ltsf-linear|DLinear]] 与 [[autoformer|Autoformer]] 等骨干[^src-cats]：

- **八类异构 ATS 构造器**：从原始多变量序列并联抽取八类辅助特征并接 `GELU`，包括可变卷积核 `Conv1d`、等价 patch 聚合的非重叠步长卷积、按通道分组的 `Conv1d` 单序列独立滤波、跨通道 `1×1 Linear`、恒等映射（OTS 复制）与可学习 Embedding 等，覆盖局部多尺度与全局混合先验[^src-cats]。
- **共享预测器与 OTS shortcut**：固定 2 层 MLP 并行预测原序列与全部辅助序列得到 $[\hat{A}, \hat{X}]$，再经线性投影 $P$ 将 $\hat{A}$ 投回 $C$ 维与 $\hat{X}$ 相加；当强耦合信号有效时由 $\hat{A}$ 贡献增量，弱耦合时由直通支路维持基准预测。代价是增加 $N$ 条 ATS 前向与投影开销（$L_P=720$ 时为 $645\text{M}$ FLOPs、$4.26\text{M}$ 参数）[^src-cats]。
- **连续性正则与双维稀疏**：连续性损失 $L_{\text{cont}}$ 惩罚一阶差分二次变差（等效低通滤波）；通道稀疏门控 $\gamma\in(0,1)$ 抑制无用辅助通道；时间稀疏用可学习斜率配合直通估计（STE）截断历史窗口，使弱耦合场景可退回近似纯单变量 MLP[^src-cats]。
- **图谱关系**：与 [[source-timexer|TimeXer]]、[[crosslinear|CrossLinear]] 属**同槽位互斥**（不依赖真实外部外生变量而全由内部自造虚拟外生序列）；相对 [[channel-independence|Channel Independence]] 属**演进与折衷**，通过可稀疏截断的辅助通道补回 CI 丢弃的跨序列信息；与 [[patchtst|PatchTST]] 属**同槽位对比**（虚拟辅助通道 + 2 层 MLP vs 通道独立局部 patch）；与 [[ltsf-linear|DLinear]] 兼具**复现口径对照与插件宿主**关系（CATS 固定输入 $L_I=720$ 配时间截断，而基线从 $\{96,192,336,720\}$ 调优）[^src-cats]。

## 证据

1. **定位与主结论**：9 集 36 设定 28/36 最优，平均相对 MSE 仅略优 ARM（43.9% vs 44.9%，PatchTST 55.4%）；优势在自建强耦合合成集 Multi20–200（56%–63% vs ARM 64%–78%），CATS 固定 $L_I=720$ 配时间截断、基线按 $\{96,192,336,720\}$ 调优[^src-cats]。
2. **真实涨点归因**：强弱耦合消融反转——Multi 去通道+时间稀疏最狠（158%）、去构造器异质性（146%）、去连续性损失（142%）；ETTh2 去 OTS shortcut 最狠（110%），其余 103%–105%；Mean pooling+CATS 亦有效，增益在构造/稀疏/残差而非 MLP 容量[^src-cats]。
3. **证据缺口与负结果**：ATS 数量固定无法训练扩展；无真实外部外生源；缺等算力对照且主表只报平均 MSE/MAE[^src-cats]。

## 相关页面

- [[source-timexer]] — 同槽位互斥（真实外部外生 variate token vs 内部自造 ATS）
- [[crosslinear]] — 同槽位互斥（真实外生 1D 卷积嵌入 vs 内部自造 ATS）
- [[channel-independence]] — 演进与折衷（可稀疏退回单变量的跨通道补充）
- [[patchtst]] — 同槽位对比（通道独立局部 patch vs 虚拟辅助通道 + 2 层 MLP）
- [[ltsf-linear]] — 复现口径对照与插件宿主（DLinear 接入 CATS 包装）

[^src-cats]: [[source-cats]]
