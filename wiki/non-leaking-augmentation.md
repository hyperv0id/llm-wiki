---
title: "Non-Leaking Augmentation"
type: technique
tags:
  - diffusion
  - training
  - augmentation
created: 2026-04-28
last_updated: 2026-04-28
source_count: 1
confidence: high
status: active
---

# Non-Leaking Augmentation

非泄漏增强是 Karras 等人在 EDM 论文中引入的技术[^src-edm]，源自 GAN 训练中的数据增强，用于防止扩散模型在小数据集上过拟合。

## 核心思想

传统数据增强会导致生成图像也包含增强痕迹（泄漏）。EDM 的解决方案是：将增强参数作为条件输入提供给网络[^src-edm]，推理时设为零以保证生成非增强图像。

## 实现方式

1. 对训练图像应用几何变换（翻转、旋转、缩放等）
2. 将变换参数（角度、尺度、位移）编码为额外条件输入 $c_{aug}$
3. 网络输出变为 $D_\theta(x; \sigma, c_{aug})$
4. 推理时设置 $c_{aug} = 0$

## 效果

在 CIFAR-10 和 FFHQ 数据集上[^src-edm]：
- FID 提升约 0.1-0.3
- 对小数据集效果更显著

## 链接

- [[edm]] — EDM 论文
- [[diffusion-model]] — 扩散模型基础

[^src-edm]: [[source-edm]]
