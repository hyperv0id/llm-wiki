---
title: "TimesFM"
type: entity
tags:
  - time-series
  - foundation-model
  - decoder-only
  - iclr2024
created: 2026-04-29
last_updated: 2026-11-24
source_count: 15
confidence: high
status: active
---

# TimesFM

## 概述

**TimesFM** (Time-Series Forecasting) 是 Google 提出的首个用于时间序列预测的解码器-only基础模型，发表于 ICLR 2024[^src-timesfm]。

## 核心贡献

1. **大规模预训练**：在超过 10 亿时间点的大规模数据上预训练
2. **Patch 级别的处理**：将时间序列划分为重叠的 patch，作为 token
3. **Decoder-only 架构**：类似 LLM 的自回归预测
4. **强零样本能力**：在未见过的数据集上表现出色

## 架构

```
Input Patches → Linear Embedding → Transformer Decoder → Output Projection
                                                        ↑
                                            [Autoregressive]
```

### 关键设计

- **Patch Size**：128 个时间点
- **Context Length**：2048 个时间点（512 个 patch）
- **预���长度**：可变
- **模型规模**：200M 参数

## 与 UniCA 的结合

TimesFM 可以通过 UniCA 框架适配到协变量感知预测任务：

- **Pre-Fusion**：融入历史协变量
- **Post-Fusion**：融入未来已知协变量
- **性能提升**：+12.7% MAPE (单模态)，+6.5% MAE (多模态图像)[^src-unca]

## 相关页面

- [[unified-covariate-adaptation]] — UniCA 框架
- [[timesnet]] — 另一个时间序列基础模型
- [[chronos]] — Chronos 时间序列模型
- [[most]] — MoST 多模态时空基础模型（不同领域：TS vs ST）
- [[aurora]] — Aurora 多模态生成式基础模型（TimesFM 为单模态，Aurora 支持多模态）
- [[uniextreme]] — UniExtreme 极端天气基础模型（TimesFM: 通用 TS；UniExtreme: 天气极端事件）
- [[timecap]] — TimeCAP LLM agent 框架，用 LLM 做时间序列上下文理解（AAAI 2025 Oral）
- [[timedit]] — TimeDiT (KDD 2025)，将 DiT 扩散 Transformer 作为时间序列基础模型，支持预测/插补/异常检测/数据生成四合一，区别于 TimesFM 的纯预测架构[^src-timedit]
- [[sundial]] — Sundial (ICML 2025)，原生 Flow Matching TS 基础模型。Sundial 在 TSLib 零样本预测上全面超越 TimesFM（Table 9，TimesFM 在所有 24 个指标中仅 1 个获胜），并支持概率预测（TimesFM 仅点估计）[^src-sundial]
- [[probts|ProbTS]] — 将 TimesFM 作为 **AR 解码基础模型** 纳入零样本分析：短 horizon 有竞争力，长 horizon 相对 MOIRAI 等 NAR 基础模型劣势扩大（误差累积）[^src-probts]
- [[moirai-moe|Moirai-MoE]] — ICML 2025 的 MoE 时间序列基础模型。在 10 个零样本数据集上，Moirai-MoE-B (86M activated) 在 CRPS 总评（Avg all: 0.478）上超越 TimesFM (0.488)，排除 Electricity/Solar 含泄露数据集后 Avg non-leak 两者持平（0.439 vs 0.439）。Moirai-MoE-S (11M) 则在 Solar、Power、ETT2、Traffic 上 CRPS 优于 TimesFM，但总体 Avg 不及[^src-moirai-moe]
- [[ts-memory]] — TS-Memory (KDD 2026)：以 TimesFM 为冻结 backbone 验证参数记忆蒸馏，ETTm2 上 MSE 降 16.0%、Weather 降 7.6%[^src-ts-memory-time-series-foundation-models-kdd26]
- [[tirex]] — **同槽位互斥对照**：TiRex (NeurIPS 2025) 用 35M 纯 sLSTM + 连续块掩码（CPM）替换 Transformer 注意力与自回归点估计回喂，在 GiftEval-ZS（81 项设置）与 Chronos-ZS 上排名居首，且推理速度较 TimesFM-2.0 快 11 倍以上[^src-tirex]
- [[timer-xl]] — **瓶颈接力 $\Delta$**：针对 TimesFM / 原版 Timer 等单变量自回归基础模型丢失跨通道交互的局限，Timer-XL 将非重叠时间块展平至 $NT$ 维上下文并引入 Kronecker 因果图掩码与置换等变偏置 $(u, v)$，统一单/多变量与协变量长上下文预测[^src-timer-xl]
- [[timesx]] — **实证复现对齐**：在合成基准 CiK 上，TimesFM 排名落后于通用 LLM 与代码修订代理 `CodeRev`；而在 [[timesx|TimesX]] 的 19 域真实数值、真实时钟上下文且严格隔离 2024-07-01 模型知识截断的评测中，TimesFM 转为稳健领先于 Gemini 与 `CodeRev`[^src-timesx]
- [[baguan-ts]] — **实证复现对齐与范式对照**：[[baguan-ts|Baguan-TS]]（ICML 2026）将 TimesFM-2.5 纳入 [[fev-bench|fev-bench-cov]]（30 项含协变量任务）同台对比，报告 Baguan-TS 点预测（SQL $0.7997$）领先于 TimesFM-2.5（$0.8276$）；机制上 TimesFM 需靠 UniCA 或 TimesFMX 等外部适配器扩展协变量，而 Baguan-TS 通过 3D 轴分解与 Y-space 检索在推理期实现免梯度多示例 ICL[^src-baguan-ts]
- [[source-posttime|PostTime]] — **下游后修订上游先验依赖**：PostTime（NeurIPS 2026）以冻结 TimesFM-2.5 作为数值先验生成器 $\tilde{y}=\operatorname{TimesFM}(x)$，再由 4B LLM 结合文本上下文判别修正、维持或回退该先验；在刷新版 [[timesx|TimesX]] 基准（88 ID / 11 OOD）上整体优于未修订 TimesFM-2.5 先验（而直接 SFT 拟合真值反劣于先验 $0.788/0.729$），但论文仅验证强 TimesFM-2.5 先验，尚未检验弱先验下的退化边界[^src-posttime]
- [[fev-bench]] — **实证复现对齐与统计不可分证据**：在涵盖 7 域 96 唯一集（100 项任务、46 项含协变量）并采用 $B=1000$ 任务级配对 Bootstrap 95% 置信区间的 [[fev-bench]] 统一评测中，Chronos-2 双指标居首，而 TimesFM-2.5 与 [[tirex|TiRex]] 的 95% 置信区间大幅重叠、第二名在统计上不可区分[^src-fev-bench]

## DynaMix 对比

[[dynamix|DynaMix]] (NeurIPS 2025) 在 DSR 任务上对 TimesFM 进行了评估[^src-dynamix]：

- **DSR 全面失败**：TimesFM 无法重建动力系统的长期行为——在 54 个测试系统上，Dstsp 和 DH 均显著差于 DynaMix[^src-dynamix]
- **短期预测仍有竞争力**：在 MASE 上 TimesFM 与 DynaMix 接近，因为 TS 基础模型正是为短期预测优化的[^src-dynamix]
- **多变量耦合缺失**：TimesFM 将各维度独立处理，无法捕获非线性动力系统中维度间的耦合关系——这是 DSR 失败的根本原因[^src-dynamix]
- **6D Lorenz-96 失败**：TimesFM 完全无法重建更高维的动力学结构，MASE=4.82 vs DynaMix 的 1.02[^src-dynamix]

## TimesFMX（协变量适配）

[[chronosx|ChronosX]] 将同一 IIB/OIB 式适配推广到 patch 输入与点预测骨干，得到 **TimesFMX**：协变量与目标同步 patch 后进注入块；OIB 在点预测 $\hat z_t$ 上加残差。合成/真实评测中 TimesFMX 相对零样本 TimesFM 明显受益，但真实集聚合 WQL 通常仍落后 ChronosX[^src-chronosx]。

---

## 引用

[^src-timesfm]: [[source-timesfm]]
[^src-unca]: [[source-unica]]
[^src-timedit]: [[source-timedit]]
[^src-sundial]: [[source-sundial]]
[^src-probts]: [[source-probts]]
[^src-dynamix]: [[source-dynamix]]
[^src-moirai-moe]: [[source-moirai-moe]]
[^src-chronosx]: [[source-chronosx]]
[^src-ts-memory-time-series-foundation-models-kdd26]: [[source-ts-memory-time-series-foundation-models-kdd26]]
[^src-tirex]: [[source-tirex]]
[^src-timer-xl]: [[source-timer-xl]]
[^src-timesx]: [[source-timesx]]
[^src-baguan-ts]: [[source-baguan-ts]]
[^src-posttime]: [[source-posttime]]
[^src-fev-bench]: [[source-fev-bench]]