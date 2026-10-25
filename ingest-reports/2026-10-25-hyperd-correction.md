# Ingest 报告：HyperD（arXiv:2511.09275v2）与 CycleNet 数据流更正

## 创建
- wiki/correction-hyperd-secondary-summary.md — WHY：`raw/hyperd-hybrid-periodicity-decoupling.md` 与原文在作者、模块、数据流、消融数字上系统性冲突，需要一个可引用的更正页把「哪些说法不能再用」固定下来。

## 修改
- wiki/hyperd.md — WHY：按原文重写架构与数据流；明确「periodic 支 = 日/周参数表按时间戳查表」，删除三频段、双路径、fDMLP 三类错误表述；补 Table 1/2/4 数字与边界。
- wiki/source-hyperd-hybrid-periodicity-decoupling.md — WHY：源摘要改为原文口径，补 Stage A/B 维度块与二手摘要警告。
- wiki/frequency-aware-residual-representation.md — WHY：FARR 只处理残差（$X-S$），STFE 是空间/时间 FFT → 复值 MLP → IFFT；删除 RFFT 三频段与「学习阈值」。
- wiki/spatial-temporal-attentive-encoder.md — WHY：STAE 输入是周期参数表，日/周表独立同构精炼；删除「双路径 / STFE / 原始信号」描述。
- wiki/dual-view-alignment-loss.md — WHY：DVA 是预测频谱按 $F_{low}$ 分段的 MSE 对齐；删除 Soft-DTW 与余弦相似度。
- wiki/hybrid-periodicity-decoupling.md — WHY：概念改为 periodic/residual 解耦与周期表查表，删除短/长周期与三频段。
- wiki/demlp-decoder.md — WHY：原文不存在该模块，整页删除（git rm）。
- wiki/periodicity-modeling-in-time-series.md — WHY：HyperD、CycleNet 两节补分阶段维度块；修正三频段表、认知层级、分解粒度、路线四与时间线；CycleNet 明写「不是检索」。
- wiki/residual-cycle-forecasting.md — WHY：补「不是检索（RAG）」警告与对齐/复制的维度标注。
- wiki/cyclenet.md — WHY：周期分量生成改为「按绝对相位滚动 + 复制（查表，不是检索）」。
- wiki/fedformer.md、wiki/timesnet.md、wiki/dualsformer.md、wiki/pdformer.md、wiki/fgti.md、wiki/source-autoformer.md、wiki/traffic-forecasting.md — WHY：清除「短/长周期双编码器」「高频映射残差」「HyperD 在 PeMS 上超越 PDFormer」等由二手摘要派生的错误陈述。
- wiki/index.md — WHY：更新 6 条描述、删除 demlp 条目、新增更正页条目。

## 新建交叉链接
- [[hyperd]] ↔ [[correction-hyperd-secondary-summary]]
- [[source-hyperd-hybrid-periodicity-decoupling]] ↔ [[correction-hyperd-secondary-summary]]
- [[periodicity-modeling-in-time-series]] ↔ [[correction-hyperd-secondary-summary]]
- [[hybrid-periodicity-decoupling]] ↔ [[cyclenet]]

## 数值备注
- `raw/` 为二手摘要：作者、模块名、数据流、基线数、消融数字、效率对照均与原文冲突；逐条差异见更正页。`raw/` 未改动。
- Table 1 内部不一致点：PEMS03 MAPE 为 15.76%，DGCRN 为 15.42%，故「四数据集三指标全 SOTA」不成立；页面已并列写出。
- 原文式 (9) 后未印时间轴 FFT 的频点数，本页按共轭对称取 $T_1'=\lfloor T_1/2\rfloor+1$，标 [INFERENCE]。
- Table 4 的 `Reduction` 行未注明基准；按每列最优对照复算为 6.27×（10.28/1.64）、6.79×（49.65/7.31）、4.77×（7.10/1.49），标 [INFERENCE]。
- STAE 的 $W_o$ 在 PDF 转写中为 `2N*N`，正文按 $[2N,N]$ 理解。
- PDF 文本公式下标断裂（如 $R^s_f$、$P_D$），已按 §4.1–4.4 上下文复原；未发现正文与表格的其它数字冲突。
