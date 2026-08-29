---
title: "Imputation Consistency（插补一致性）"
type: concept
tags:
  - data-imputation
  - time-series
  - consistency
  - self-supervised-learning
created: 2026-08-29
last_updated: 2026-08-29
source_count: 2
confidence: medium
status: active
---

# Imputation Consistency（插补一致性）

**插补一致性（imputation consistency）**是 MTSCI（CIKM 2024）提出的插补质量概念：插补结果应在两个层面与真实数据结构保持一致——窗口内（intra）与相邻窗口之间（inter）[^src-mtsci]。

## 两类一致性

- **intra-consistency（窗口内一致性）**：插补值在观测值引导下应能反过来帮助重构观测值，使插补值与观测值保持一致、降低插补偏差（Sec. 1）[^src-mtsci]。
- **inter-consistency（相邻窗口一致性）**：插补单个窗口样本时应考虑相邻窗口样本，使完整样本与相邻窗口保持时序一致；论文称该想法与相邻窗口时序样本"良好连续性"（good continuity）的思想吻合（Sec. 1，论文所引为 Tonekaboni et al. 的时序邻域表示学习工作）[^src-mtsci]。

论文自述：此前没有插补方法在多元时间序列插补任务中处理插补一致性问题（Sec. 1，论文自述）[^src-mtsci]。

## 实现载体（MTSCI）

- **intra**：complementary mask 把同一窗口切成两个互补掩码视图，用 InfoNCE 式对比损失约束两视图的编码器表示相似——即插补目标与条件观测互相重构，见 [[mtsci]]（Sec. 4.2、4.3.1）[^src-mtsci]。
- **inter**：mixup 机制在训练期把相邻 "context" 窗口的条件信息与当前窗口观测混合，推理期该窗口不可得、退化为单窗口条件（Sec. 4.3.2）[^src-mtsci]。
- **度量**：论文用 [[crps|CRPS]] 衡量插补结果与观测值在整个数据集上的 imputation consistency，作者报告 MTSCI 在三数据集 × point/block 六组设置均低于 CSDI（Table 5）[^src-mtsci]。

## 与其他"一致性"概念的辨析

- **[[consistency-models|Consistency Models]] 的自一致性**：CM 的 consistency 指同一生成轨迹上的点经一致性函数映射到同一起点（用于少步生成），[[costi|CoSTI]] 将该机制用于插补加速；MTSCI 的 consistency 是插补值相对观测值/相邻窗口的统计一致性，与采样步数无关。两个 "consistency" 同形不同义，跨文献检索时注意区分。
- **[[contrastive-learning|对比学习]]中的多视图一致性**：[[nuwats|NuwaTS]] 同样使用"掩码多视图 + InfoNCE"，但其目标是学 mask-invariant patch 表示以支撑跨域零样本插补；MTSCI 用互补掩码视图约束插补值与观测值互相重构，目标直接指向插补精度[^src-nuwats][^src-mtsci]。
- **与 [[self-supervised-imputation-training|自监督插补训练]]的关系**：MTSCI 的掩码生成建立在 CSDI 式自监督掩码训练之上（point/block 两种目标策略），一致性约束是叠加在该训练范式之上的额外损失项 $L = L_\epsilon + \lambda L_{CL}$，不是对它的替代[^src-mtsci]。

## 关联页面

- [[mtsci]] — 提出并实现该概念的论文
- [[csdi]] — 被论文归入"未处理插补一致性"的条件扩散插补代表
- [[crps]] — 论文采用的一致性度量指标
- [[consistency-models]] / [[costi]] — 另一种"一致性"（少步生成自一致性）
- [[contrastive-learning]] / [[nuwats]] — 掩码多视图对比的其他用法

[^src-mtsci]: [[source-mtsci]]
[^src-nuwats]: [[source-nuwats]]
