---
title: "Task Arithmetic"
type: technique
tags:
  - task-arithmetic
  - model-merging
  - task-vector
  - deep-model-fusion
  - editing
created: 2026-07-31
last_updated: 2026-07-31
source_count: 1
confidence: medium
status: active
---

# Task Arithmetic（任务算术）

**Task Arithmetic**（Ilharco et al., 2023；FusionBench 系统评测）是 [[model-merging|模型合并]] 的代表性技术：把每个微调模型相对预训练的位移写成**任务向量**，再在参数空间做加减与缩放，以编辑或组合能力[^src-jmlr-25-1243]。

## 定义

任务向量：

\[
\tau_t = \theta_t^{\mathrm{ft}} - \theta^{\mathrm{pre}}
\]

多任务合并的常用形式：

\[
\theta \leftarrow \theta^{\mathrm{pre}} + \lambda \sum_t \tau_t
\]

其中 \(\lambda\) 为缩放超参（FusionBench 部分表固定 \(\lambda=0.3\)）[^src-jmlr-25-1243]。

直觉：每个任务把权重往一个方向推了一小步；若这些方向近似正交，相加不易完全互相拆台。

## 在 FusionBench 中的证据

- CLIP 八任务上，任务向量余弦相似度热图**近对角**（Figure 4），支撑「方向可加」叙事[^src-jmlr-25-1243]。
- CLIP-ViT-B/32 八任务 AVG：Task Arithmetic **68.0**（权平均 66.5；Ties 72.2；RegMean 82.4；层间 AdaMerging 82.6）[^src-jmlr-25-1243]。
- GPT-2 七任务 AVG：**70.0**（与 Ties 并列；简单平均 56.1）[^src-jmlr-25-1243]。
- Flan-T5-large：Task Arithmetic **87.3**（STL 89.6；Ties 87.4）[^src-jmlr-25-1243]。
- NYUv2：\(\lambda=0.3\) 的 Task Arithmetic 在三任务间均衡但弱于专家本行[^src-jmlr-25-1243]。

**Ties-Merging** 在任务向量上增加修剪与符号冲突消解，常作为 Task Arithmetic 的强化基线（B/32：72.2 vs 68.0）[^src-jmlr-25-1243]。

## 何时用

- 只要检查点、几乎无数据、要快速多任务原型 → Task Arithmetic / Ties 是默认起点。
- 有标数据 → RegMean 系通常明显更强。
- 有测试分布可适配 → 层间 AdaMerging 往往再上一档。
- 可接受扩结构 → WEMoE/SMILE 等 mixing 逼近 MTL[^src-jmlr-25-1243]。

## 风险

- \(\lambda\) 敏感；固定 0.3 可能偏置比较。
- 负迁移：组合向量可能伤害未见任务。
- 依赖共享预训练与同构；不解释置换对齐问题。

## 相关页面

- [[model-merging]] · [[deep-model-fusion]] · [[fusionbench]] · [[source-jmlr-25-1243]]

[^src-jmlr-25-1243]: [[source-jmlr-25-1243]]
