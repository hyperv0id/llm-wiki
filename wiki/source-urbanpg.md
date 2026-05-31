---
title: "UrbanPG: Personalized Context + General Backbone for Urban Spatio-Temporal Learning (AAAI 2026)"
type: source-summary
tags:
  - spatial-temporal
  - foundation-model
  - prompt-learning
  - linear-attention
  - large-scale
  - few-shot
  - continual-learning
  - traffic-forecasting
created: 2026-06-01
last_updated: 2026-06-01
source_count: 1
confidence: high
status: active
---

# UrbanPG: Personalized Context + General Backbone for Urban Spatio-Temporal Learning

**Authors**: Aoyu Liu, Yaying Zhang* (Tongji University, Shanghai)
**Venue**: AAAI 2026
**Code**: https://github.com/Aoyu-Liu/UrbanPG

## 核心贡献

UrbanPG 提出一个高效的、可扩展的城市时空学习框架，通过解耦"个性化上下文提示"（personalized context prompts）和"通用时空骨干"（general backbone），同时解决了大规模预测、小样本泛化和持续学习三个挑战[^src-urbanpg]。核心贡献包括：（1）提出线性时空上下文注意力（STCA）模块，基于 Performers 的随机特征映射将自注意力复杂度从 O(N²) 降为 O(N·d²)[^src-urbanpg]；（2）个性化提示通过可学习的时间/空间 embedding 捕获场景特有模式，并通过随机扰动正则化防止大规模场景下的过拟合[^src-urbanpg]；（3）支持预训练-微调和持续学习两种范式，通过冻结骨干并仅重建/扩展提示实现跨场景泛化[^src-urbanpg]。

## 架构设计

UrbanPG = 个性化上下文提示 + 通用骨干 + 提示调整门控[^src-urbanpg]：

**（1）个性化上下文提示**：时间上下文提示 Pt = index(Etod, X) + index(Edow, X)，其中 Etod (time-of-day) 和 Edow (day-of-week) 是可学习的时间 embedding。空间上下文提示 Ps 由可学习空间 embedding Es ∈ R^(N×d) 经随机扰动正则化构造——以概率 p 随机替换节点为共享 embedding Ed 或随机噪声 Md^(n)，训练时引入"被迫泛化"效应防止大 N 过拟合，推理时使用完整 Es。最优 p=0.1[^src-urbanpg]。

**（2）通用时空骨干**：场景无关的轻量骨干，核心是 STCA（Spatio-Temporal Context Attention）模块。STCA 将输入特征 H 映射为 Q, K, V ∈ R^(N×d)，执行三项线性注意力：Hst = φ(Q)·(φ(K)^T·V + φ(Pt)^T·V + φ(Ps)^T·V)，其中 d×d 矩阵乘法替代了标准注意力的 N×N 矩阵，复杂度 O(N·d²)。φ 是随机特征映射（来自 Performers），正交投影方向 w_i ∼ N(0, I_d) 后做 sin/cos 编码[^src-urbanpg]。

**（3）提示调整门控**：Hpst = (Hst · (1 + Pt) + Ps) · Pt。Pt 充当时间门控缩放，Ps 提供空间偏移，最终再经 Pt 时间调制。提示不单参与注意力查询，还直接参与特征的非线性调制[^src-urbanpg]。

**（4）多学习范式**：预训练-微调范式中冻结骨干 M，仅重建和训练下游个性化提示 P'；持续学习范式中首次训练后冻结 M，每次增量阶段仅扩展 Ps 参数（新增节点加行，旧节点不动），实现零遗忘[^src-urbanpg]。

## 实验结果

**大规模预测**（LargeST 四子集，12→12）：在所有 4 个数据集 SOTA。CA（8600 节点）MAE=17.23, RMSE=29.08, MAPE=12.49%，训练时间比 PatchSTG 少 48.96%，推理时间少 72.44%，内存少 45.72%[^src-urbanpg]。

**小样本预测**（CA-D3/CA-D5，仅 10% 训练数据）：CA-D3 MAE=18.28（FlashST=18.91, STD-MAE=20.09），CA-D5 MAE=12.70（FlashST=13.47, STD-MAE=13.93）[^src-urbanpg]。

**持续学习**（PEMS-Stream 7 增量期 / AIR-Stream 4 增量期）：PEMS-Stream 平均 MAE=10.77（EAC=13.49，↑20.2%），AIR-Stream 平均 MAE=19.67（EAC=20.77）[^src-urbanpg]。

**消融**：移除空间上下文提示（w/o SC）性能退化最严重，其次为移除时间上下文提示（w/o TC）> 移除随机扰动正则化（w/o RPR）> 移除 STCA（w/o STCA）。t-SNE 可视化显示 Etod 呈环形周期结构，Es 将模式相似的节点聚成簇[^src-urbanpg]。

## 局限性

（1）不支持多任务并行训练——个性化提示与骨干耦合训练，无法像 [[urbanfm|UrbanFM]] 在同一训练循环中优化多个任务，这是 UrbanPG 走向"时空基础模型"的核心障碍[^src-urbanpg]。（2）预训练数据依赖——通用骨干的知识上限 = 预训练语料的覆盖度，未在预训练中出现的极端时空模式无法被迁移[^src-urbanpg]。（3）线性注意力近似误差——φ 随机特征映射在 d<128 时近似误差显著，d=256 是最优平衡点。可学习的投影方向可能进一步提升低 d 精度[^src-urbanpg]。

[^src-urbanpg]: [[source-urbanpg]]
