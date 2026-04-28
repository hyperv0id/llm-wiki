---
title: "Consistency Models"
type: entity
tags:
  - diffusion-models
  - fast-inference
  - one-step-generation
  - icml-2023
created: 2026-04-28
last_updated: 2026-04-28
source_count: 1
confidence: high
status: active
---

# Consistency Models

**Consistency Models** 是扩散模型加速采样的里程碑工作，由 OpenAI 的 Yang Song 等人于 2023 年发表在 ICML。该模型支持单步生成，同时保留多步采样和零样本编辑能力[^src-consistency-models]。

## 核心创新

1. **单步生成**：直接映射噪声到数据，无需迭代
2. **自一致性**：同一 PF ODE 轨迹上的点映射到相同起点
3. **双训练模式**：蒸馏模式 (CD) + 独立训练模式 (CT)
4. **零样本编辑**：无需显式训练即可执行多种图像编辑任务

## 技术细节

### 定义

给定 PF ODE 的解轨迹 $\{x_t\}_{t \in [\epsilon, T]}$，一致性函数定义为：
$$
f: (x_t, t) \mapsto x_\epsilon
$$

### 参数化

使用 skip connection 参数化：
$$
f_\theta(x, t) = c_{\text{skip}}(t) \cdot x + c_{\text{out}}(t) \cdot F_\theta(x, t)
$$

其中 $c_{\text{skip}}(\epsilon) = 1, c_{\text{out}}(\epsilon) = 0$（边界条件）

### 采样

**单步采样**：
$$
x_0 = f_\theta(x_T, T), \quad x_T \sim \mathcal{N}(0, T^2 I)
$$

**多步采样**：通过交替去噪和噪声注入提升质量

## 实验结果

- **CIFAR-10**: 1 步 FID 3.55, 2 步 FID 2.93
- **ImageNet 64×64**: 1 步 FID 6.20, 2 步 FID 4.70

## 相关页面

- [[diffusion-model]] — 扩散模型概念
- [[score-based-sde]] — Score-Based SDE，理论基础
- [[probability-flow-ode]] — 概率流 ODE
- [[dpm-solver]] — DPM-Solver，另一快速采样方法
- [[progressive-distillation]] — 渐进蒸馏，对比方法

## 引用

[^src-consistency-models]: [[source-consistency-models]]