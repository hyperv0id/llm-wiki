---
title: "TS-Memory"
type: entity
tags:
  - time-series-forecasting
  - foundation-model
  - knowledge-distillation
  - retrieval-free
  - plug-and-play
  - kdd-2026
created: 2026-08-19
last_updated: 2026-08-19
source_count: 1
confidence: medium
status: active
---

# TS-Memory

**TS-Memory** 是 HKUST(GZ) / Tencent / Squirrel AI 团队提出的即插即用记忆适配器，发表于 KDD 2026 [^src-ts-memory-time-series-foundation-models-kdd26]。它通过 [[parametric-memory-distillation|参数化记忆蒸馏]] 范式，将在线检索的分布知识离线蒸馏为轻量模块 PlugMem，使冻结 [[time-series-foundation-model|TSFM]] 获得检索增强的域适应能力，推理时 $O(1)$ 复杂度且无需外部数据库 [^src-ts-memory-time-series-foundation-models-kdd26]。

## 核心设计

### 两阶段流程

**Stage I — 特权监督构建** [^src-ts-memory-time-series-foundation-models-kdd26]：
- 在训练集上构建泄漏安全 kNN 知识库 $\mathcal{K} = \{(X^{(i)}, Y^{(i)})\}$，用冻结 TSFM 编码器 $f_{\text{enc}}$ 提取嵌入
- 欧氏距离检索 $K$ 个最近邻，显式排除索引匹配窗口防自检索
- 对每个候选做 shift alignment：尾部 $m$ 步均值偏移校正绝对尺度差异
- 按 $\ell_1$ 距离重排，softmax 加权聚合邻居未来轨迹为加权经验分位数 $\hat{Q}_t^{\text{teach}}$
- 检索置信度 $\text{Conf}_t = \max_k w_k$ 反映检索权重集中度

**Stage II — 置信门控记忆蒸馏** [^src-ts-memory-time-series-foundation-models-kdd26]：
- PlugMem $g_\phi$：轻量 encoder-decoder Transformer，接收原始 $X_t$，Instance Norm → patch → 编码器 → $H$ 个可学习 horizon query 解码 → 分位数头 → Instance Norm 逆变换
- 复合损失 $\mathcal{L} = \mathcal{L}_{\text{task}} + \lambda_{\text{align}}\mathcal{L}_{\text{align}} + \lambda_{\text{reg}}\mathcal{L}_{\text{reg}}$
  - **任务损失**：pinball loss 对真值回归
  - **对齐损失**：仅当 advantage gate $\chi_t$ 判定教师优于 backbone 时，以置信加权 $\omega_t$ 蒸馏 Huber loss
  - **稳定性正则**：锚定中位数至 backbone $(1-\omega_t)$ 权重 + 分位数交叉惩罚
- 参见 [[confidence-gated-distillation]]

### 推理融合

$$\hat{Q}_t^{\text{final}} = (1-\alpha)\hat{Q}_t^{\text{base}} + \alpha\hat{Q}_t^{\text{mem}}, \quad \alpha \in [0,1]$$

$\alpha$ 在验证集调优；仅需两次前向传播（backbone + PlugMem），无检索索引 [^src-ts-memory-time-series-foundation-models-kdd26]。

## 范式定位

TS-Memory 提出第三条 TSFM 适配路线——区别于参数适配（微调/LoRA）和非参数检索（RAG）[^src-ts-memory-time-series-foundation-models-kdd26]：

| 范式 | 检索? | 存储成本 | 推理复杂度 | Backbone |
|------|-------|---------|-----------|----------|
| 参数适配 (STF) | 否 | 高（每域模型副本） | $O(1)$ | 更新 |
| 非参数检索 (RAG) | 是 | 高（向量库） | $O(|\mathcal{D}|)$ | 冻结 |
| **参数记忆蒸馏 (TS-Memory)** | **否** | **低（轻量模块）** | $O(1)$ | **冻结** |

## 实验要点

- 四种冻结 backbone（[[chronos|ChronosBolt]]、Chronos2、[[sundial|Sundial]]、[[timesfm|TimesFM]]）× 8 数据集，平均 MSE 降 5.8%、MAE 降 2.1% [^src-ts-memory-time-series-foundation-models-kdd26]
- 唯一在全部 8 数据集同时降低 MSE/MAE/CRPS 的适配方法 [^src-ts-memory-time-series-foundation-models-kdd26]
- 推理延迟仅增加 3.8–4.7%，远低于 TS-RAG 的 95–198% [^src-ts-memory-time-series-foundation-models-kdd26]
- PlugMem 可跨 backbone 迁移（不同 TSFM 架构间），且可跨模型规模复用（205M 蒸馏 → 9M 使用）[^src-ts-memory-time-series-foundation-models-kdd26]

## 与相关方法的关系

- **[[gtr|GTR]]**（ICLR 2026）：同为即插即用检索增强，但 GTR 通过可学习参数矩阵按绝对时间位置检索全局周期模式，推理时仍含检索操作；TS-Memory 将检索完全离线化为参数记忆 [^src-ts-memory-time-series-foundation-models-kdd26]。
- **[[pir|PIR]]**（NeurIPS 2025）：同为后处理检索修订，但 PIR 在推理时仍需检索训练库做全局修订；TS-Memory 将检索蒸馏为参数模块，推理时检索免除 [^src-ts-memory-time-series-foundation-models-kdd26]。
- **[[retrieval-augmented-spatio-temporal-forecasting|RAG-STF]]**：在线检索范式，推理时 kNN 搜索导致延迟随数据库线性增长；TS-Memory 将此成本移至训练阶段 [^src-ts-memory-time-series-foundation-models-kdd26]。
- **[[zero-initialized-adaptation|零初始化适配]]**：TS-Memory 的 $\alpha$ 融合在 $\alpha=0$ 时等价于冻结 backbone，且 PlugMem 的 anchor loss 约束其在不确定时回归 backbone，共享"从冻结 backbone 起点渐进偏离"的设计哲学 [^src-ts-memory-time-series-foundation-models-kdd26]。
- **[[tsfm-covariate-adaptation-comparison|TSFM 适配方法对比]]**：CoRA/UniCA/ChronosX 聚焦协变量注入，TS-Memory 聚焦分布知识蒸馏，两者正交可叠加 [^src-ts-memory-time-series-foundation-models-kdd26]。

## 局限性

1. 仅覆盖长程概率预测 [^src-ts-memory-time-series-foundation-models-kdd26]
2. Shift alignment 仅处理加性偏移，不适用于时序变形或 regime 突变 [^src-ts-memory-time-series-foundation-models-kdd26]
3. 检索语料与部署域不匹配时教师噪声增大 [^src-ts-memory-time-series-foundation-models-kdd26]
4. 融合权重 $\alpha$ 需逐 dataset–backbone 验证集调优 [^src-ts-memory-time-series-foundation-models-kdd26]

[^src-ts-memory-time-series-foundation-models-kdd26]: [[source-ts-memory-time-series-foundation-models-kdd26]]
