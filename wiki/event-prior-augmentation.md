---
title: "Event Prior Augmentation (EPA)"
type: technique
tags:
  - extreme-weather
  - memory-network
  - attention-mechanism
  - event-prior
  - weather-forecasting
created: 2026-05-03
last_updated: 2026-05-03
source_count: 1
confidence: medium
status: active
---

# Event Prior Augmentation (EPA)

事件先验增强（EPA）是 [[uniextreme|UniExtreme]] 的核心模块之一，通过分类极端事件记忆池和双层注意力融合网络，捕获极端事件的层次化模式结构和复合效应[^src-uniextreme]。

## 动机

现有记忆/提示学习方法无法适配极端天气预测任务[^src-uniextreme]：

1. **记忆/提示来源问题**：全可学习或从模型特征提取，无法自动捕获真实极端先验
2. **粒度问题**：像素级原始输入添加无法处理区域尺度事件；patch 级特征序列无法感知原始极端细节
3. **扁平架构**：缺乏层次化建模能力，无法同时捕获类间差异和类内模式变化

## 技术细节

### 记忆构建

从包含极端事件记录的训练子集中构建分类记忆池[^src-uniextreme]：

1. 对每个区域网格标注多热类型向量 $\mathbf{c}_r \in \{0,1\}^{M'}$（M' = M 极端类型 + 1 "正常"类型）
2. 数量平衡：正常区域采样数量与极端区域匹配
3. **KMeans 聚类**：将每个类型的记忆标准化为固定容量 U=5，解决类型间数量不均
4. 零填充 + 掩码机制：处理样本不足的类型

最终记忆池 $\mathcal{M}^* = \{\mathcal{M}^*_m\}_{m=0}^{M'-1}$，每个 $\mathcal{M}^*_m$ 包含 U 个聚类中心[^src-uniextreme]。

### 双层记忆融合

#### 类内融合（Intra-type）

每个区域独立查询各类型记忆[^src-uniextreme]：

- **Query**：区域天气状态 → CNN + MeanPool2D → $\mathbf{Q}_r^t \in \mathbb{R}^C$
- **Key**：类型记忆 → CNN + MeanPool2D → $\mathbf{K}_m \in \mathbb{R}^{U \times C}$
- **Value**：类型记忆原始网格 $\mathbf{M}_m \in \mathbb{R}^{U \times a_h \times a_w \times C}$
- **输出**：$\mathbf{P}_{r;m}^t = \text{Attention}(\mathbf{Q}_r^t, \mathbf{K}_m, \mathbf{M}_m) \in \mathbb{R}^{a_h \times a_w \times C}$

类内融合聚合同一类型内的不同模式变体。

#### 类间融合（Inter-type）

将所有类内融合结果整合为混合类型记忆[^src-uniextreme]：

- 拼接 $\{\mathbf{P}_{r;m}^t\}_{m=0}^{M'-1}$ → $\mathbf{O}_r^t \in \mathbb{R}^{M' \times a_h \times a_w \times C}$
- 通过类似的注意力机制（$\mathbf{Q}_r^t$ 为 query，$\mathbf{O}_r^t$ 为 key+value）得到混合记忆 $\mathbf{P}_r^t$

类间融合捕获不同极端类型间的复合效应。

### 残差增强

混合记忆通过残差连接增强原始输入[^src-uniextreme]：

$$\tilde{\mathbf{X}}_r^t = (\mathbf{X}_r^t + \mathbf{P}_r^t)\mathbf{W}_m + \mathbf{b}_m$$

## 消融实验验证

- 移除 EPA：极端预测性能显著下降[^src-uniextreme]
- 随机初始化记忆替代真实事件记忆（w/o MC）：性能下降，证明真实极端先验不可替代
- 移除类内融合（w/o MF）：性能下降，证明层次化建模（类内+类间）优于扁平架构
- 注意力权重可视化显示与实际事件类型高度一致，验证 EPA 能正确利用事件先验[^src-uniextreme]

## 与相关技术的对比

| 技术 | 记忆来源 | 架构 | 粒度 | 层次化 |
|------|----------|------|------|--------|
| Visual Prompt Tuning | 全可学习 | 扁平 | Patch 级 | 无 |
| End-to-End Memory Networks | 全可学习 | 扁平 | Token 级 | 无 |
| UniST (traffic) | 模型特征提取 | 扁平 | Patch 级 | 无 |
| **EPA (UniExtreme)** | 真实极端事件 | 双层 | 区域级 | 有（类内+类间） |

## 相关页面

- [[uniextreme]] — UniExtreme 模型
- [[adaptive-frequency-modulation]] — AFM 模块（EPA 的互补模块，EPA 输出作为 AFM 输入）
- [[extreme-weather-forecasting]] — 极端天气预测概念
- [[mixture-of-experts]] — MoE 路由（EPA 使用注意力而非 MoE 路由）
- [[most]] — MoST 使用 SNR 模态选择（EPA 使用事件类型记忆）

[^src-uniextreme]: [[source-uniextreme]]
