---
title: "FusionBench"
type: entity
tags:
  - deep-model-fusion
  - model-merging
  - benchmark
  - library
  - multi-task-learning
  - jmlr-2025
created: 2026-07-31
last_updated: 2026-07-31
source_count: 1
confidence: medium
status: active
---

# FusionBench

**FusionBench** 是 Tang et al.（武大 / 中山大学深圳 / 北理工 / 南洋理工，JMLR 2025）提出的**深度模型融合**统一库与综合基准：把 ensemble / merging / mixing（及配套 compression）放进同一套可配置管线，提供标准化模型池、任务池与评测协议[^src-jmlr-25-1243]。

- 代码：`https://github.com/tanganke/fusion_bench`（`pip install fusion-bench`）
- 文档：`https://tanganke.github.io/fusion_bench`
- 模型与数据：HuggingFace `tanganke` 集合
- 源摘要：[[source-jmlr-25-1243]]

## 核心主张

1. 融合方法评测长期碎片化（任务、微调设定、实现不一致）；MergeKit / MergeBench 偏 LLM 且范围窄[^src-jmlr-25-1243]。
2. 需要**算法–模型池–任务池**解耦的研究平台，而不仅是合并脚本。
3. 在统一尺子下：自适应与 mixing 最强；简单平均是弱基线；未见任务与腐蚀分布会暴露负迁移与过拟合[^src-jmlr-25-1243]。

## 三模块架构

| 模块 | 职责 |
|------|------|
| Algorithm | `BaseAlgorithm.run(model_pool)`；CLI 或库调用 |
| Model Pool | 加载/保存；CLIP / ResNet / GPT-2 / Flan-T5 / CausalLM / ConvNeXt / DINOv2 等 |
| Task Pool | 数据集 + 指标；可选评测 |

配置：Hydra YAML。流水线：选算法与模型 → 融合 → 可选存盘 → 可选评测报告。LazyStateDict 支持大模型按需载参[^src-jmlr-25-1243]。

## 方法覆盖（Table 1 摘要）

- **Ensemble**：Simple / Weighted / Max-Model
- **Merging**：Soup 平均、Fisher、RegMean/++、Task Arithmetic、Ties-Merging、AdaMerging（task/layer）、Concrete Subspace、Representation Surgery、TALL mask、TSV、Isotropic、OPCM、FW-Merging、RanDeS 等
- **Mixing**：MoE upscaling/merging、Depth Upscaling、Recombination、WE-MoE/V2、Pareto-Driven、SMILE
- **Compression**：BitDelta、Magnitude / Wanda / SparseGPT / Expert Pruning

每条标注是否需要超参搜索、标数据、测试时适配（TTA）、校准数据——本身是选型清单[^src-jmlr-25-1243]。

## 基准设定

| 骨干 | 任务 | 备注 |
|------|------|------|
| CLIP-ViT B/32·B/16·L/14 | 最多 20 图像分类；主表 8 核任务 | 只微调视觉塔 |
| ResNet-50 | NYUv2 分割/深度/法向 | 并骨干、头分开 |
| GPT-2 | GLUE 7 分类 | 全参微调 |
| Flan-T5 base/large | GLUE 8 text-to-text | LoRA → 合入再融 |
| Mistral-7B ×3 | MMLU / TruthfulQA / GSM8K / ARC | SMILE 升尺度示例 |

## 关键数字（CLIP-ViT-B/32，八任务 AVG）

- 预训练 48.2 → 权重平均 66.5 → Ties 72.2 → RegMean++ 84.4 → 层间 AdaMerging 82.6 → WEMoE/SMILE **89.2/89.3**
- 传统 MTL 88.6；单任务专家上界 90.3
- L/14 上 mixing 93.6–93.8，逼近 STL 94.3
- 层间 AdaMerging（82.6）≫ 任务级（68.7）
- 泛化：未见任务可出现**负迁移**；腐蚀下自适应更稳但仍会崩[^src-jmlr-25-1243]

完整表与分类学见 [[deep-model-fusion]]、[[model-merging]]、[[task-arithmetic]]。

## 谱系位置

| 对照 | 关系 |
|------|------|
| MergeKit | 实用 LLM 合并工具；无统一多域评测 |
| MergeBench | LLM 域合并评测 |
| 传统 MTL | 联合重训上界/强基线；融合换数据与算力效率 |
| [[mixture-of-experts|MoE]]（Time-MoE / Moirai-MoE / TiMi MMoE） | *训时*路由专家；FusionBench 的 WEMoE/SMILE 是*事后*多检查点升混合 |
| 多模态时序「融合」 | 模态特征融合 ≠ 多模型参数融合；词同义不同 |

## 局限

- 主文短、硬结果在附录 E
- LLM/MLLM 协议仍在扩展
- 合并默认**同构 + 共享预训练**
- 固定 λ 等超参可能偏置；排行榜 seen 准确率不等于 OOD 部署

## 相关页面

- [[source-jmlr-25-1243]] — 源摘要
- [[deep-model-fusion]] · [[model-merging]] · [[task-arithmetic]] · [[mixture-of-experts]]

[^src-jmlr-25-1243]: [[source-jmlr-25-1243]]
