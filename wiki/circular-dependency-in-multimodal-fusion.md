---
title: "Circular Dependency in Multimodal Fusion"
type: concept
tags:
  - multimodal-fusion
  - reliability
  - trustworthy-ai
  - uncertainty-quantification
created: 2026-07-18
last_updated: 2026-07-18
source_count: 1
confidence: medium
status: active
---

# Circular Dependency in Multimodal Fusion

**循环依赖**是多模态融合中一个根本性的可靠性问题：当系统依赖分类器自身的输出（如预测置信度、熵、信念质量）来评估输入质量或分配融合权重时，若分类器本身在面对损坏或 OOD 数据时产生过度自信的预测，可靠性信号与需要检测的错误就来自同一来源，形成自我指涉的循环[^src-gmf]。

## 问题机制

传统统计融合方法（如 [[gmf|GMF]] 论文中评估的 QMF、PDF、DBF、UAW-EEF）的工作流程是：

1. 各模态编码器提取特征
2. 分类器对各模态产生预测及置信度
3. 基于置信度或不确定性（entropy、belief mass）分配融合权重
4. 加权融合产生最终预测

问题出现在步骤 2–3 之间：深度神经网络在噪声或分布偏移下常产生**校准不良**的高置信度预测[^src-gmf]。当分类器对损坏输入仍给出 90%+ 置信度时，基于置信度的可靠性评估完全失效。

## 经验证据

GMF 论文通过可靠性图（reliability diagram）在 NYU Depth V2 噪声场景下量化了这一现象[^src-gmf]：

- 统计基线（QMF、PDF）的预测置信度在高噪声（σ=2.0）下仍保持高位，而准确率已大幅下降，形成明显的校准缺口
- GMF 的传输能量与分类器置信度的互信息仅 0.08，而统计方法的置信度-正确性互信息为 0.67，证实了几何信号与分类器输出的独立性

## 解决方向

GMF 提出的解决方案是将可靠性评估从**内在统计度量**转向**外在几何度量**：通过 Diffusion Schrödinger Bridge / Rectified Flow 在潜在空间中估计传输代价，该代价仅依赖编码器输出的几何结构，与分类器状态无关[^src-gmf]。

更一般的解决思路包括：

- **几何方法**：使用流形距离、传输代价等与分类器解耦的信号
- **校准方法**：通过 temperature scaling、conformal prediction 等后处理修复分类器校准
- **外部验证**：引入独立的质量评估模型

[^src-gmf]: [[source-gmf]]
