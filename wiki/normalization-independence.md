---
title: "Normalization Independence"
type: technique
tags:
  - diffusion-models
  - time-series
  - normalization
  - distribution-shift
created: 2026-04-28
last_updated: 2026-04-28
source_count: 1
confidence: high
status: active
---

# Normalization Independence (N.I.)

Normalization Independence 是 SimDiff 提出的扩散模型专属技术，用于更好地捕获数据分布并缓解时间序列的分布漂移问题 [^src-simdiff]。

## 核心思想

过去和未来的时间序列段落很少共享相同的水平或尺度 — 现实中的时间序列在历史窗口和未来窗口之间往往存在显著的分布漂移（distribution drift）。然而，以往的扩散模型对两者都使用过去的统计量进行归一化，隐含假设平稳性，这会引入偏差 [^src-simdiff]。

## 方法

### 训练阶段
1. 对过去序列 X：用其自身统计量 (μ_X, σ_X) 归一化，然后通过可学习的仿射参数 (γ, β) 进行重缩放：
   - X_norm = γ · (X - μ_X) / σ_X + β
2. 对未来目标 Y：用其自身统计量 (μ_Y, σ_Y) 独立归一化：
   - Y_norm = (Y - μ_Y) / σ_Y
3. 在 Y_norm 上添加扩散噪声并优化损失

### 推理阶段
1. 从标准高斯噪声 Ŷ_K ~ N(0, I) 开始
2. 通过条件去噪（基于 X_norm）逐步去噪
3. 反归一化时仅使用过去统计量和学习的仿射参数：
   - Ŷ = σ_X · Ŷ_norm - β + μ_X
   - （注意：这里 γ 被吸收到 β 的处理中）

## 效果

N.I. 仅增加一个轻量级仿射层，计算成本可忽略，但显著提升了模型对分布漂移的鲁棒性 [^src-simdiff]。消融实验表明，N.I. 在所有数据集上一致提升性能，对分布漂移严重的数据集（如 Weather, NorPool）效果尤为显著 [^src-simdiff]。

## 理论依据

通过在训练时对每个段落用其自己的统计量进行"高斯化"，N.I. 稳定了优化过程，使网络能够从过去推断未来的尺度变化，而不存在数据泄露的风险 [^src-simdiff]。这更好地将训练数据与扩散的高斯先验对齐 [^src-simdiff]。

## 与其他归一化方法的对比

- **RevIN (Instance Normalization)**：用于处理分布漂移，但设计为可逆实例归一化 [^src-simdiff]
- **标准 Z-Score**：假设过去和未来共享统计量，无法处理分布漂移 [^src-simdiff]
- **N.I.**：解耦归一化，仅在训练时独立归一化未来，推理时完全避免未来泄露 [^src-simdiff]

[^src-simdiff]: [[source-simdiff]]