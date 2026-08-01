---
title: "FusionBench: A Unified Library and Comprehensive Benchmark for Deep Model Fusion"
type: source-summary
tags:
  - deep-model-fusion
  - model-merging
  - model-ensemble
  - model-mixing
  - multi-task-learning
  - benchmark
  - jmlr-2025
created: 2026-07-31
last_updated: 2026-07-31
source_count: 0
confidence: low
status: active
---

# FusionBench 源文件摘要

**来源**: Anke Tang, Li Shen, Yong Luo, Enneng Yang, Han Hu, Lefei Zhang, Bo Du, Dacheng Tao (Wuhan University / SYSU Shenzhen / BIT / NTU). *FusionBench: A Unified Library and Comprehensive Benchmark for Deep Model Fusion.* Journal of Machine Learning Research **26** (2025) 1–38. Submitted 5/25; Revised 11/25; Published 11/25. Editor: Zeyi Wen. raw: `raw/jmlr-25-1243.pdf`. 代码：`https://github.com/tanganke/fusion_bench`；文档：`https://tanganke.github.io/fusion_bench`；HuggingFace：`https://huggingface.co/tanganke`。

## 核心论点

深度模型融合把多个网络的**预测或参数**合成一个更强/更省的模型，属 “learn from model” 范式。既有方法爆炸，但评测不一致：任务与微调设定不统一，MergeKit 等工具偏 LLM 且缺评测框架。FusionBench 是**首个**专为深度模型融合设计的综合基准 + 统一库：算法模块 / 模型池 / 任务池，Hydra + CLI，覆盖 ensemble / merging / mixing（及配套 compression），并在 CV+NLP、多尺度、泛化与鲁棒设定上给出可复现对照。

## 架构与管线

- **Algorithm Module**：`BaseAlgorithm` 子类，独立可调用。
- **Model Pool Module**：统一加载/保存；架构无关，新架构只需新 Pool。
- **Task Pool Module**：数据集 + 指标；可选。
- **主流程**：载配置 → 实例化 → `run` 得 merged_model → 可选 `evaluate` → 存报告。
- **工程**：LazyStateDict 按需载参；多数实验可在单卡 RTX 3090 24GB 复现。

## 分类学（Appendix A）

| 类别 | 形式 | 特点 |
|------|------|------|
| Model Ensemble | 多模型预测聚合 | 准；推理/存储贵 |
| Model Merging | 同构参数 → 单组参数 | 单模型足迹；数据高效 |
| Model Mixing | 组件重组 / 常扩维 | 层重组、MoE 升尺度；常需再训或 TTA |
| Compression（扩展） | BitDelta / Wanda / SparseGPT 等 | 融前/融后降本 |

Table 1 覆盖 Simple/Weighted Ensemble、Max-Model、Model Soup、Fisher、RegMean/++、Concrete Subspace、Task Arithmetic、Ties-Merging、AdaMerging（task/layer）、Representation Surgery、TALL mask、TSV、Isotropic、OPCM、FW-Merging、RanDeS、MoE upscaling/merging、Depth Upscaling、Recombination、WE-MoE/V2、Pareto-Driven、SMILE 及多种剪枝，并标注超参搜索 / 标数据 / TTA / 校准数据等需求。

## 模型集合（Appendix D）

1. **CLIP-ViT**（B/32, B/16, L/14）：最多 20 图像分类任务；主文 8 核任务 SUN397 / Cars / RESISC45 / EuroSAT / SVHN / GTSRB / MNIST / DTD。仅训视觉编码器。Adam lr=1e-5，4000 step，bs=32。
2. **ResNet-50**：NYUv2 分割 / 深度 / 法向。
3. **GPT-2**：GLUE 七分类。
4. **Flan-T5 base/large**：GLUE 八任务 text-to-text（含 STSB），**LoRA**；prompt 见 Appendix F。

任务向量（微调权重 − 预训练权重）在 CLIP 八任务上**近正交**（Figure 4）。单任务微调存在正/负迁移（如 Flan-T5-base 的 STSB 专家在 MNLI 上 1.7% vs 预训练 56.5%）。

## 关键实证（Appendix E）

### CLIP 八任务（Table 8–9，B/32 AVG）

| 方法 | AVG |
|------|-----|
| Pre-trained | 48.2 |
| Individual STL | 90.3 |
| Traditional MTL | 88.6 |
| Weight Averaging | 66.5 |
| Fisher | 70.6 |
| RegMean / ++ | 82.4 / 84.4 |
| Task Arithmetic | 68.0 |
| Ties-Merging | 72.2 |
| task-wise AdaMerging | 68.7 |
| **layer-wise AdaMerging** | **82.6** |
| **WEMoE / SMILE**（mixing） | **89.2 / 89.3** |

L/14：STL 94.3；MTL 92.4；WEMoE 93.8 / SMILE 93.6；layer-wise AdaMerging 与 RegMean++ ≈ 91.0。观察：mixing 与自适应最强；层间适配 ≫ 任务级；大模型更易并。

### 其他设定

- **NYUv2 ResNet（Table 10）**：专家跨任务崩溃；Weight Avg / Task Arithmetic / Ties 更均衡（分割 mIoU 52 → ~34–39，深度/法向大幅好于非专家）。
- **GPT-2（Table 11）**：Simple Avg 56.1；RegMean 68.8；Task Arithmetic / Ties **70.0**；STL 82.0。
- **Flan-T5-base（Table 12）**：Weight Avg 78.2；SMILE **84.0**（STL 84.6）。**large（Table 13）**：Task Arithmetic 87.3 / Ties 87.4 / layer AdaMerging 87.6（STL 89.6）。LoRA 先合入基座再融。
- **泛化（Table 14–15）**：六 seen + 两 unseen；存在对 RESISC45 等的**负迁移**（融后 < 预训练）。
- **鲁棒（Table 16）**：六类腐蚀；中等腐蚀下 AdaMerging/WEMoE 更稳，极端腐蚀全员重创。
- **Mistral-7B ×3 → SMILE upscale 11.2B（Table 17）**：相对单专家更均衡；LM-Evaluation-Harness。

### 成本直觉（E.2）

无参平均 / 任务算术：入门性价比；有标数据：RegMean 系；要极限且有测试分布：AdaMerging / WE-MoE。

## 与相关工具

- **MergeKit**：偏 Transformer LLM 合并，缺统一评测。
- **MergeBench**：仅 LLM 域。
- FusionBench：CV+NLP、完整三分类、研究向模型集合与协议。

## 局限

主文短、结果在附录；LLM 覆盖仍薄；合并默认同构共享预训练；部分 λ 固定；负迁移与腐蚀过拟合提示排行榜 ≠ 部署。

## 相关页面

- [[fusionbench]] — 实体
- [[deep-model-fusion]] · [[model-merging]] · [[task-arithmetic]] · [[mixture-of-experts]]
