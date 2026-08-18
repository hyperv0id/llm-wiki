---
title: "SARAF"
type: entity
tags:
  - time-series-forecasting
  - retrieval-augmented
  - stationarity
  - diversity-based-retrieval
  - kdd-2026
created: 2026-08-19
last_updated: 2026-08-19
source_count: 1
confidence: medium
status: active
---

# SARAF

**SARAF**（Stationarity-Aware Retrieval-Augmented Time Series Forecasting）是一个平稳性感知的检索增强时间序列预测框架，由 University of Birmingham 和 Siemens AG 联合提出，发表于 KDD 2026[^src-stationarity-aware-retrieval-augmented-forecasting-kdd26]。

## 动机

现有检索增强预测器（如 [[source-raf|RAFT]]、[[pir|PIR]]）隐含假设"相似的过去意味着相似的未来"，仅依赖输入相似度进行 Top-K 检索。SARAF 通过诊断实验证明这一假设的可靠性与数据集平稳性强相关：平稳数据集上 Spearman ρ=1.000，非平稳数据集上 ρ 降至 0.285[^src-stationarity-aware-retrieval-augmented-forecasting-kdd26]。此外，相似度检索的 Top-K 结果常高度冗余，在非平稳条件下会将不匹配案例聚合为误导性共识[^src-stationarity-aware-retrieval-augmented-forecasting-kdd26]。

现有方法（如 [[source-raf|RAFT]]、[[pir|PIR]]）在检索证据不可靠时采用降权策略，但此策略有两点局限：(i) 未解决根本原因——输入检索与未来相关性的失配；(ii) 会过度抑制部分有用的邻居，使模型退化为纯参数预测器[^src-stationarity-aware-retrieval-augmented-forecasting-kdd26]。

## 架构

SARAF 由四个核心组件构成[^src-stationarity-aware-retrieval-augmented-forecasting-kdd26]：

1. **时间对齐相似度检索**（[[time-aligned-retrieval-enhancement]]）：在 Pearson 相关相似度基础上叠加时间对齐奖励（hour-of-day, day-of-week, month-of-year, minute-of-hour），抑制"形态相似但时间错位"的候选。取 Top-M（M≫K）构成候选池。
2. **平稳性控制的多样性选择**（[[diversity-based-retrieval-selection]]）：基于 [[dataset-stationarity-estimation|数据集平稳性分数]] s̄ 调节 MMR 平衡系数 λ(s̄) = λ_min + s̄(λ_max − λ_min)，通过随机 MMR 从 Top-M 中选择 Top-K。低平稳性 → 较小 λ → 更强多样化。
3. **自适应 Gaussian 加权聚合**：Gaussian 核带宽 σ(s̄) = σ_min + (1−s̄)(σ_max − σ_min)，低平稳性 → 较大 σ → 更平滑的权重分配。
4. **轻量融合**：直接线性预测与检索加权预测取平均，经线性投影输出。

## 实验结果

在 ETTh1/h2、ETTm1/m2、Exchange、Solar、Electricity、Traffic 八个数据集上评估（look-back=720, H∈{96,192,336,720}）[^src-stationarity-aware-retrieval-augmented-forecasting-kdd26]：

- 8 个数据集中 5 个取得最优平均 MSE 和 MAE[^src-stationarity-aware-retrieval-augmented-forecasting-kdd26]。
- 相比 [[source-raf|RAFT]] 平均 MSE 降 3.85%、MAE 降 1.87%；相比 DUET 平均 MSE 降 4.05%、MAE 降 0.75%[^src-stationarity-aware-retrieval-augmented-forecasting-kdd26]。
- 在非平稳数据集 Exchange 上优势明显（论文报告 SARAF avg MSE 0.394 vs RAFT 0.449 vs DUET 0.469）[^src-stationarity-aware-retrieval-augmented-forecasting-kdd26]。
- 消融实验表明：时间对齐增强贡献最一致（移除后所有数据集 MSE 上升）；多样性和平稳性估计主要在非平稳数据集上产生增益[^src-stationarity-aware-retrieval-augmented-forecasting-kdd26]。
- 检索器可即插即用：PatchTST + 检索器在 ETTh1 上 MSE 降 14.96%，DLinear + 检索器降 20.73%[^src-stationarity-aware-retrieval-augmented-forecasting-kdd26]。
- 效率：0.088M 参数，0.334 ms/iter，模型内存仅 0.335 MiB[^src-stationarity-aware-retrieval-augmented-forecasting-kdd26]。

## 与其他方法的关系

- 与 [[gtr|GTR]]（ICLR 2026）：同属检索增强预测，但 GTR 检索全局周期参数矩阵，SARAF 检索历史窗口段并引入平稳性控制的多样性选择[^src-stationarity-aware-retrieval-augmented-forecasting-kdd26]。
- 与 [[pir|PIR]]（NeurIPS 2025）：同为即插即用检索增强框架，但 PIR 是后处理修订（检索训练目标加权求和作为修订项），SARAF 在预测前检索并融合未来段[^src-stationarity-aware-retrieval-augmented-forecasting-kdd26]。
- 与 [[ratd|RATD]]（NeurIPS 2024）：RATD 用 k-NN 参照引导扩散去噪，SARAF 直接加权聚合检索到的未来段[^src-stationarity-aware-retrieval-augmented-forecasting-kdd26]。
- 与 [[source-raf|RAFT]]（ICML 2025）：RAFT 仅用相似度检索后拼接增强输入，SARAF 增加时间对齐和多样性控制以应对非平稳性[^src-stationarity-aware-retrieval-augmented-forecasting-kdd26]。
- 与 [[nsdiff|NsDiff]]（ICML 2025）：均关注非平稳性，但 NsDiff 从概率扩散角度建模非平稳性，SARAF 从检索质量角度应对非平稳性[^src-stationarity-aware-retrieval-augmented-forecasting-kdd26]。

## 局限

1. 使用全局多变量相似度，未实现通道级或组级相似度[^src-stationarity-aware-retrieval-augmented-forecasting-kdd26]。
2. 密集滑动窗口数据库对长序列/大数据集内存密集[^src-stationarity-aware-retrieval-augmented-forecasting-kdd26]。
3. 平稳性控制在数据集级别，未实现实例级 regime shift 感知[^src-stationarity-aware-retrieval-augmented-forecasting-kdd26]。

[^src-stationarity-aware-retrieval-augmented-forecasting-kdd26]: [[source-stationarity-aware-retrieval-augmented-forecasting-kdd26]]
