---
title: "Model Merging"
type: technique
tags:
  - model-merging
  - deep-model-fusion
  - task-arithmetic
  - multi-task-learning
  - parameter-space
created: 2026-07-31
last_updated: 2026-07-31
source_count: 1
confidence: medium
status: active
---

# Model Merging（模型合并）

**Model merging** 是 [[deep-model-fusion|深度模型融合]] 的核心子类：在**同构**网络之间把多组参数 \(\theta_1,\ldots,\theta_N\) 合成单组 \(\theta\)，使推理成本与单模型相同，同时尽量保留各任务能力[^src-jmlr-25-1243]。

## 机制族（FusionBench 覆盖）

| 族 | 代表 | 额外需求 |
|----|------|----------|
| 无参/少参平均 | Model Soup / Weight Averaging | 无或超参 |
| 重要性加权 | Fisher Merging、Weighted Average | 标数据或搜索 |
| 数据对齐闭式 | RegMean / RegMean++ | 标数据上统计 |
| 任务向量算术 | [[task-arithmetic\|Task Arithmetic]]、Ties-Merging | 通常要缩放系数 λ |
| 测试时自适应 | AdaMerging（task-wise / **layer-wise**） | TTA |
| 子空间 / 掩码 | Concrete Subspace、TALL mask、TSV、Isotropic | 视方法 |
| 其他 | Representation Surgery、OPCM、FW-Merging、RanDeS | 视方法 |

与 **ensemble** 的分界：合并后只存/跑一个网络。与 **mixing** 的分界：合并保持架构与参数量量级；mixing 可升 MoE、重组层、扩参[^src-jmlr-25-1243]。

## FusionBench 关键数字（CLIP-ViT-B/32，八任务 AVG）

- Weight Averaging 66.5；Task Arithmetic 68.0；Ties-Merging 72.2
- Fisher 70.6；RegMean 82.4；RegMean++ **84.4**
- task-wise AdaMerging 68.7 vs **layer-wise 82.6**
- 对照：MTL 88.6；mixing WEMoE/SMILE 89.2/89.3；STL 90.3[^src-jmlr-25-1243]

语言侧（GPT-2 七任务）：简单平均 56.1，Task Arithmetic / Ties 70.0，仍低于 STL 82.0——合并缝更大[^src-jmlr-25-1243]。

## 预设与风险

1. **同构 + 共享预训练**：置换对称、异构骨干不在主表默认设定内。
2. **任务向量近正交**有利于相加；强冲突任务会负迁移（未见任务上融后 < 预训练已出现）[^src-jmlr-25-1243]。
3. **TTA 方法**在干净测试集强，腐蚀下可能对单任务过拟合。
4. 成本：无参平均最便宜；有标数据优先 RegMean 系；要极限再上 AdaMerging[^src-jmlr-25-1243]。

## 相关页面

- [[deep-model-fusion]] · [[task-arithmetic]] · [[fusionbench]] · [[source-jmlr-25-1243]]
- [[mixture-of-experts]] — 对照事后 mixing（WE-MoE/SMILE）vs 训时 MoE

[^src-jmlr-25-1243]: [[source-jmlr-25-1243]]
