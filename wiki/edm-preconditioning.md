---
title: "EDM Preconditioning"
type: technique
tags:
  - diffusion
  - training
  - preconditioning
created: 2026-04-28
last_updated: 2026-04-28
source_count: 1
confidence: high
status: active
---

# EDM Preconditioning

EDM 预处理技术是 Karras 等人在 2022 年提出的网络预处理方法[^src-edm]，通过设计输入输出缩放函数来解决扩散模型训练中的数值问题。

## 核心公式

去噪器表示为[^src-edm]：
$$D_\theta(x; \sigma) = c_{skip}(\sigma) \cdot x + c_{out}(\sigma) \cdot F_\theta(c_{in}(\sigma) \cdot x; c_{noise}(\sigma))$$

其中：
- $c_{skip}(\sigma) = \sigma_{data}^2 / (\sigma^2 + \sigma_{data}^2)$
- $c_{out}(\sigma) = \sigma \cdot \sigma_{data} / \sqrt{\sigma^2 + \sigma_{data}^2}$
- $c_{in}(\sigma) = 1 / \sqrt{\sigma^2 + \sigma_{data}^2}$
- $c_{noise}(\sigma)$ 将噪声水平映射为网络条件输入

## 设计原则

1. **输入缩放**：使网络输入在不同噪声水平下保持单位方差
2. **输出缩放**：使训练目标保持单位方差
3. **Skip 连接**：在高噪声时让网络预测接近输入，减少误差放大

## 损失权重

$$\lambda(\sigma) = (\sigma^2 + \sigma_{data}^2) / (\sigma \cdot \sigma_{data})^2$$

配合对数正态噪声分布 $\ln(\sigma) \sim N(-1.2, 1.2^2)$，实现各噪声水平的平衡训练。

## 链接

- [[edm]] — EDM 论文
- [[diffusion-model]] — 扩散模型基础
- [[score-function]] — 分数函数

[^src-edm]: [[source-edm]]