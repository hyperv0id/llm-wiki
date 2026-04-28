---
title: "无分类器引导"
type: technique
tags:
  - diffusion
  - conditional-generation
  - guidance
created: 2026-04-28
last_updated: 2026-04-28
source_count: 2
confidence: high
status: active
---

# 无分类器引导

无分类器引导（Classifier-Free Guidance, CFG）是一种用于扩散模型的条件生成技术，它不需要独立的分类器来引导生成过程。取而代之的是，它联合训练条件模型和无条件模型，在推理时通过插值来控制条件信号的影响强度。[^src-understanding-diffusion-models]

## 核心思想

传统条件扩散模型需要额外的分类器在推理时提供梯度信号来引导生成（即分类器引导）。无分类器引导则通过训练一个**单一扩散模型**，使其既能接受条件信息 $y$，也能接受一个空标记 $\varnothing$（表示无条件）。这样，同一个模型既可以做条件预测，也可以做无条件预测。[^src-understanding-diffusion-models]

在训练时，以一定概率将条件信息 $y$ 替换为空标记 $\varnothing$，使模型学会在有无条件两种模式下工作。在推理时，同时计算条件预测和无条件预测，然后对二者进行插值。

## 数学形式

设 $\epsilon_\theta(x_t, y)$ 为给定条件 $y$ 时的噪声预测，$\epsilon_\theta(x_t, \varnothing)$ 为无条件噪声预测。无分类器引导的修正预测为：

$$
\tilde{\epsilon} = \epsilon_\theta(x_t, \varnothing) + w \cdot \big(\epsilon_\theta(x_t, y) - \epsilon_\theta(x_t, \varnothing)\big)
$$

其中 $w$ 是**引导尺度**（guidance scale），控制条件信号的影响强度。[^src-understanding-diffusion-models]

## 引导尺度 $w$ 的行为

- **$w = 0$**：退化为无条件生成，完全忽略条件信息。
- **$w = 1$**：标准的条件生成，相当于直接使用 $\epsilon_\theta(x_t, y)$。
- **$w > 1$**：放大条件信号的影响，使生成结果更严格地遵循条件信息，但会降低多样性。典型取值在 $3$ 到 $15$ 之间，具体取决于任务和模型。[^src-understanding-diffusion-models]

随着 $w$ 增大，模型会更"听话"地遵循条件（如文本提示），但样本多样性下降，可能出现伪影或过度饱和。

## 与分类器引导的对比

| 特性 | 分类器引导 | 无分类器引导 |
|------|-----------|-------------|
| 是否需要额外分类器 | 是，需要训练一个噪声鲁棒的分类器 | 否，仅需扩散模型本身 |
| 训练复杂度 | 需要额外训练分类器 | 只需在训练时随机丢弃条件 |
| 推理计算量 | 每次采样需额外计算分类器梯度 | 需两次前向传播（条件+无条件） |
| 实际效果 | 较好，但受限于分类器质量 | 更优，广泛用于主流模型 |

无分类器引导在实践中表现更优，已成为现代扩散模型的标准技术。它被广泛应用于 [[stable-diffusion]]、[[dall-e-2]]、[[imagen]] 等主流文本到图像模型。[^src-understanding-diffusion-models]

## 应用

- **文本到图像生成**：CFG 是 Stable Diffusion、DALL-E 2、Imagen 等模型的核心组件，用于控制文本提示的遵循程度。
- **文本到视频/3D 生成**：扩散模型在视频和 3D 生成中同样采用 CFG 技术。
- **其他条件生成任务**：任何需要条件控制（如类别条件、布局条件）的扩散模型都可以使用 CFG。

## 局限性

- 两次前向传播使推理成本翻倍。
- 高引导尺度（$w$ 过大）会导致样本质量下降、颜色饱和度过高、出现伪影。
- 无法像分类器引导那样提供显式的分类置信度信号。

## LDM 中的应用

LDM 成功将无分类器引导应用于文本到图像生成[^src-rombach-ldm-2022]。在 MS-COCO 数据集上，CFG 将 FID 从 23.31 提升到 12.63（引导尺度 s=1.5）。典型引导尺度在 1.5 到 10.0 之间。

[^src-understanding-diffusion-models]: [[source-understanding-diffusion-models]]
[^src-rombach-ldm-2022]: [[source-rombach-ldm-2022]]
