---
title: "Manifold Anchor Regularization (MAR)"
type: technique
tags:
  - flow-matching
  - regularization
  - text-to-image
  - reward-hacking
  - aesthetic-preservation
created: 2026-05-13
last_updated: 2026-05-13
source_count: 1
confidence: high
status: active
---

# Manifold Anchor Regularization (MAR)

> MAR 是 Flow-OPD 框架中的一种任务无关正则化方法，通过审美教师提供全数据 KL 惩罚，将生成过程锚定在高品质视觉流形上，有效缓解纯 RL 对齐中的审美退化。[^src-2605-08063]

## 问题背景

在 [[flow-grpo|Flow-GRPO]] 等纯 RL 对齐方法中，针对功能目标（如精确文字渲染、严格空间布局）的激进优化频繁触发 **reward hacking**，表现为：[^src-2605-08063]
- **Background Mode Collapse** — 模型过拟合到单调背景环境
- **Semantic Redundancy** — 多个实体间出现雷同特征（如多个物体共享相同纹理）
- **Aesthetic Degradation** — 视觉质量显著下降

## 核心思想

不同于 Flow-GRPO 使用 KL 惩罚锚定通用预训练模型，MAR 维护一个**冻结的审美教师**（经 DeQA 优化）来提供高保真正则化速度场。[^src-2605-08063]

### 数学形式

利用 SDE 框架下 Reverse KL 散度到速度场加权 L2 距离的解析关系：

\[
\mathcal{L}_{\text{MAR}} = \lambda \mathbb{E}_{c,t,x_t \sim \rho_\theta^t}\left[w(t)\|v_\theta(x_t,t,c) - v_{\text{aesthetic}}(x_t,t,c)\|^2\right]
\]

总损失为策略损失与 MAR 损失的直接求和：[^src-2605-08063]

\[
\mathcal{L}_{\text{Total}}(\theta) = \mathcal{L}_{\text{Policy}}(\theta) + \lambda \mathbb{E}_{c,t,x_t \sim \rho_\theta^t}\left[w(t)\|v_\theta(x_t,t,c) - v_{\text{aesthetic}}(x_t,t,c)\|^2\right]
\]

### 关键性质

| 性质 | 描述 |
|------|------|
| 任务无关 | 审美教师不关注功能对齐，只提供视觉质量指导 |
| 全数据监督 | 覆盖整个训练数据集，不限于特定任务子集 |
| 连续弹性锚 | 在策略贪婪吸收多教师功能智能的同时，严格约束在高品质视觉流形内 |
| 功能-审美解耦 | 将功能对齐（任务性能）与风格保留（视觉质量）分离 |

## 实验验证

### 定性效果

MAR 有效解决了背景模式崩溃和语义冗余问题。纯 GRPO 优化往往导致单调背景和雷同实体特征，而加入 MAR 后恢复了结构多样性和精确语义遵循。[^src-2605-08063]

### 定量结果

| 模型 | Aesthetic | UnifiedReward | HPS-v2.1 | QwenVL Score |
|------|-----------|---------------|----------|-------------|
| SD3.5-M | 5.87 | 3.339 | 0.2982 | 3.45 |
| w/o MAR | 5.89 | 3.518 | 0.2998 | 3.82 |
| **Ours (MAR)** | **6.23** | **3.659** | **0.3302** | **4.05** |

MAR 在所有画质和人类偏好指标上带来显著提升。[^src-2605-08063]

### 超参数

MAR 的 KL 惩罚系数 λ（论文中记为 β）设为 **0.02**。[^src-2605-08063]

## 引用

[^src-2605-08063]: [[source-flow-opd]]