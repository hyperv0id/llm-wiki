---
title: "Mixed-Modal Masked Image Modeling (MMIM)"
type: technique
tags:
  - weather-forecasting
  - masked-image-modeling
  - in-context-learning
  - vision-transformer
  - iclr-2025
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: medium
status: active
---

# Mixed-Modal Masked Image Modeling (MMIM)

**混合模态掩码图像建模（Mixed-Modal Masked Image Modeling, MMIM）** 是 [[weathergfm|WeatherGFM]] 的核心训练-推理范式，将多样天气理解任务统一为视觉提问-回答（Visual Question Answering, VQA）问题[^src-weathergfm]。

## 核心思想

由于重新定义后各天气任务的输入和输出均为图像形态，Transformer 可将图像数据视为 token 集合统一处理[^src-weathergfm]。借鉴 VQA 与 [[mae|MAE]] 式掩码建模，MMIM 在多种天气模态上施加混合模态掩码，把"完成任务"转化为"重建被掩码的目标"[^src-weathergfm]。

## 训练目标

给定提示对 $(P_{in}, P_{target})$ 与 query $(X_{in}, X_{target})$，对 prompt target 与 ground-truth target 按掩码比例随机施加掩码算子 $M(\cdot)$，**保留** prompt input $P_{in}$ 与 query input $X_{in}$：

$$P'_{target}, X'_{target} = F_\tau(P_{in}, M(P_{target}), X_{in}, M(X_{target}); \theta)$$

优化目标为重建两个被掩码 target 的 MSE 损失之和[^src-weathergfm]：

$$\mathcal{L}_{total} = L_2(P'_{target}, P_{target}) + L_2(X'_{target}, X_{target})$$

（实现中主损失采用 L1）。被掩码区域用可学习 token 向量替换，采用**块状掩码（block-wise masking）策略，掩码比例 75%**[^src-weathergfm]。

## 推理：目标全掩码

推理阶段保持 $P_{in}, P_{target}, X_{in}$ 完整，将目标图像**完全掩码**。这一"目标全掩码（target full masking）"策略使通用基础模型通过 VQA 形式直接生成对应目标[^src-weathergfm]。

## 输入格式处理

气象数据物理变量数随数据集/任务变化（区别于 RGB 固定 3 通道）。给定 $(C,H,W)$ 输入，模型用**任务特定 patch embedding 层**将其 token 化（每 patch 大小 $C\times p^2$），再用一个 MLP 层把不同任务的嵌入对齐到同一空间：

$$z_C = \text{PatchEmbed}_C(x),\quad z_0 = \text{MLP}_C(\text{LN}(z_C))$$

从而让单一 ViT 适配可变通道数的多任务输入[^src-weathergfm]。

## 与图像 MAE 的区别

- [[mae|MAE]] 在单图内随机掩码重建以做自监督预训练；MMIM 则在 **prompt 对 + query** 的拼接结构上掩码 target，把任务语义编码进 prompt，从而支持 [[weather-prompt|天气提示]]驱动的多任务与 OOD 泛化[^src-weathergfm]。
- MMIM 的掩码对象是"目标输出"而非随机 patch，使掩码重建等价于"按 prompt 指定的任务完成 query"[^src-weathergfm]。

## 相关页面

- [[weathergfm]] — WeatherGFM 主模型
- [[weather-prompt]] — 提供 MMIM 输入结构的天气提示设计
- [[mae]] — 掩码图像建模的来源
- [[in-context-learning]] — in-context learning 范式
- [[source-weathergfm]] — 源文件摘要

[^src-weathergfm]: [[source-weathergfm]]
