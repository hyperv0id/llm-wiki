---
title: "Location-Aware Adaptive Normalization (LOAN)"
type: technique
tags:
  - normalization
  - conditioning
  - spatio-temporal
  - geospatial
created: 2026-07-21
last_updated: 2026-07-21
source_count: 1
confidence: medium
status: active
---

# Location-Aware Adaptive Normalization (LOAN)

**LOAN**（Location-Aware Adaptive Normalization）是一种条件归一化层，通过注入静态地理/物理属性来调制深度神经网络的特征表示。最初由 Shams Eddin et al. (2023) 在野火危险预测中提出，RiverMamba 将其适配用于河流流量预报中的集水区特征条件化[^src-rivermamba]。

## 公式

给定输入特征 X 和静态属性 X_static，LOAN 的计算为[^src-rivermamba]：

$$\text{LOAN}(X) = \left( \frac{X - \mu}{\sigma} \right) + \text{GELU}(\text{Linear}(X_{\text{static}}))$$

其中：
- μ、σ 分别是 X 沿通道维度的均值和标准差
- X_static ∈ R^(B×1×P×Vs) 被线性投影到 R^(B×1×P×K)，然后沿时间维度复制
- 归一化项 (X − μ)/σ 提供标准化特征
- GELU(Linear(X_static)) 提供位置相关的系统偏差（bias）

## 在 RiverMamba 中的角色

每个 Hindcast 块包含两个 LOAN 层（Mamba 块前后各一），每个 Forecast 块同样包含 LOAN 层[^src-rivermamba]。静态河流属性 X_static 来自 LISFLOOD 模型的集水区形态数据，包括流域面积、坡度、土壤类型等——这些信息直接影响排水和洪水行为。LOAN 确保模型在每个空间位置都"知道"该点的水文地理特征，即使该点没有实测流量数据。

## 设计直觉

与普通的 Layer Norm 或条件 Batch Norm 不同，LOAN 的关键洞察是：**归一化消除分布偏移，静态属性偏置注入先验知识**。在水文场景中，两个地理上接近的网格点可能有完全不同的洪水响应（例如一个在陡坡、一个在平原），仅靠空间坐标编码不足以区分——LOAN 通过显式的属性偏置弥补这一不足[^src-rivermamba]。

## 相关页面

- [[rivermamba|RiverMamba]] — LOAN 在洪水预报中的应用
- [[mamba-block-design|Mamba Block Design]] — Mamba 块内部结构
- [[instance-normalization|Instance Normalization]] — 相关归一化技术

[^src-rivermamba]: [[source-rivermamba]]
