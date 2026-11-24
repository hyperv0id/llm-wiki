---
title: "TabPFN-TS"
type: entity
tags:
  - time-series
  - data-imputation
  - foundation-model
  - zero-shot
  - tabpfn
  - in-context-learning
created: 2026-06-08
last_updated: 2026-11-24
source_count: 3
confidence: medium
status: active
---

# TabPFN-TS

**TabPFN-TS**（Hoo et al. 2025）是一种零样本时间序列插补/分析方法，把时间序列任务**重构为标准表格回归问题**，从而直接调用预训练的 **TabPFN** 表格基础模型[^src-time-indexed-imputation]。在 EDF R&D 的 TMLR 2026 [[time-indexed-foundation-model|时间索引基础模型]]基准中，它以 **NMAE 0.293（平均排名 1.35）统计显著地优于所有对手**，是零样本插补的最佳方法——代价是推理慢。

## 设计：简单特征 + 强回归器

TabPFN-TS 的哲学与 [[motm|MoTM]] **互逆**——用手工特征搭配高表达力回归器[^src-time-indexed-imputation]：

### (i) 手工时间表征 $H(t)$
对每个时间戳 $t$，把**归一化时间索引**与一组**预定义 Fourier 基函数**（正余弦对，捕获日/周等季节性）组合成固定特征集，显式编码时间位置与周期性。

### (ii) 基于 TabPFN 的 in-context 插补
核心表达力来自 **TabPFN**——一个在**数亿合成表格回归任务**上预训练的大型 Transformer。其 in-context learning 特性使它在推理时：
- 把观测点 $(H(t_{obs}), x(t_{obs}))$ 作为一个"prompt"集合输入
- 在注意力层内推断特征与序列值之间的函数关系
- 对缺失时间戳的查询特征 $H(t_{miss})$ 预测 $x(t_{miss})$——**单次前向、无梯度微调**[^src-time-indexed-imputation]

天然支持**协变量集成**（拼接，无需重训）与**不确定性量化**（TabPFN 返回输出分布）。

## 性能与代价

- **精度**：33 个域外数据集、1.3M+ 缺失窗口、4 种缺失场景下 NMAE 0.293，零样本最佳，在**块状缺失**下尤其稳健[^src-time-indexed-imputation]。
- **推理成本**（关键局限）：H100 上对 672 步 chunk 的一次前向约 **1 秒**；比 [[motm|MoTM]] **慢约两个数量级**[^src-time-indexed-imputation]。可在离线批量插补中摊销，但对实时或数千并发序列场景过于昂贵。
- **部署建议**：有 GPU/离线批处理时用 TabPFN-TS；资源受限或高吞吐时用 MoTM[^src-time-indexed-imputation]。

## 与相关方法的关系

- **vs [[motm|MoTM]]**：互逆设计（强回归器+简单特征 vs 学习表征+简单回归器）；TabPFN-TS 更准但慢两个数量级。
- **vs [[nuwats|NuwaTS]]**：两者都做零样本插补，但 NuwaTS 走 PLM 重编程 + patch；基准中 **NuwaTS 在所有设定下显著落后于 TabPFN-TS**[^src-time-indexed-imputation]。
- **基座 TabPFN**：Hollmann et al. 2025 的表格基础模型，prior-fitted network，靠合成先验 + in-context learning 实现零样本表格回归。
- **vs [[zeus|Zeus]]**（范式层面对比，课程组织，非直接基准对照）：Zeus（ICML 2026）走掩码重建路线——以 MOTM 多目标掩码预训练 + 重建误差做零样本插补；TabPFN-TS 与 [[motm|MoTM]] 走时间索引 + in-context 回归路线。两者是不同范式。
- **同槽位互斥与实证复现对齐（vs [[baguan-ts|Baguan-TS]]）**：两者同属无需梯度的零样本时序上下文学习，但在表征接口与延迟–校准权衡上互斥。TabPFN-TS 依靠人工时间与傅里叶特征将预测转为表格回归，而 Baguan-TS（ICML 2026）采用序列原生时间–变量–上下文 3D 轴分解并辅以 Y-space 近邻检索，单次前向约 $0.43\text{ s}$（约为 TabPFN-TS 延迟的 $1/10$）[^src-baguan-ts]。在 [[fev-bench|fev-bench-cov]] 30 项含协变量任务上，Baguan-TS 点预测（SQL $0.7997$）领先于 TabPFN-TS（$0.8404$），但概率校准指标 WQL 仍由 TabPFN-TS 占优（$0.1643$ vs $0.1724$）[^src-baguan-ts]。
- **实证复现对齐（vs [[fev-bench]]）**：在 [[fev-bench]] 针对 42 项动态协变量任务的受控开关消融中，引入动态协变量将 TabPFN-TS 的 SQL 技能分从 $34.1$ 拉升至 $42.5$，实证了表格化 ICL 拼接外生协变量的实际增益（全基准在单卡 A10G 24 GB 显存下运行）[^src-fev-bench]。

## 关联页面

- [[fev-bench]] — 面向协变量与多变量的零样本统一基准（动态协变量开关增益实证）
- [[baguan-ts]] — 序列原生 3D ICL 对照（点预测与推理延迟占优，概率校准 WQL 逊于 TabPFN-TS）
- [[time-indexed-foundation-model]] — TabPFN-TS 所属范式
- [[motm]] — 互逆设计的时间索引模型（更快的替代）
- [[nuwats]] — PLM 重编程零样本插补（基准中落后）
- [[source-time-indexed-imputation]] — 评估 TabPFN-TS 的 TMLR 2026 基准
- [[missing-not-at-random]] — 缺失机制谱系
- [[zeus]] — 掩码重建路线的零样本插补基础模型（对比范式）

[^src-time-indexed-imputation]: [[source-time-indexed-imputation]]
[^src-baguan-ts]: [[source-baguan-ts]]
[^src-fev-bench]: [[source-fev-bench]]
