---
title: "Self-Supervised Deviation Learning"
type: technique
tags:
  - self-supervised
  - representation-learning
  - spatiotemporal
  - contrastive-learning
  - prototype
created: 2026-07-21
last_updated: 2026-07-21
source_count: 1
confidence: medium
status: active
---

# Self-Supervised Deviation Learning (SSDL)

**SSDL** 是 [[st-ssdl|ST-SSDL]]（NeurIPS 2025）提出的自监督学习方法，旨在系统化地建模时空数据中当前输入与历史模式之间的动态偏差，无需任何外部标注[^src-st-ssdl]。

## 动机

现有方法通常将偏差视为二值事件（超过阈值即标记异常），然而现实中偏差是连续变化的：城市中心的传感器方差大，乡村道路则相对稳定。关键挑战：**如何在连续潜在空间中量化动态时空偏差并利用其增强预测**[^src-st-ssdl]。

## 三个组件

### 1. 历史锚点（Historical Anchor）

将完整训练序列按周分割为 $S$ 个非重叠段落，对对齐的时间步取平均，得到每周锚点 $\bar{X}^w$。对于当前输入 $X^c$，检索时间对齐的历史锚点 $X^a$——这为偏差建模提供了无外部标签的自监督参考[^src-st-ssdl]。

### 2. 潜在空间离散化（Latent Space Discretization）

引入 $M$ 个可学习原型 $P_1, \dots, P_M \in \mathbb{R}^{M \times d}$ 作为典型时空模式的代表。通过 query-prototype 交叉注意力计算输入与各原型的相似度：

$$\alpha_i = \frac{\exp(Q \cdot P_i^\top / \sqrt{d})}{\sum_{j=1}^M \exp(Q \cdot P_j^\top / \sqrt{d})}$$

按注意力分数排序，选择最相关原型为**正原型** $P^c$，次相关为**负原型** $N^c$。这使连续潜在空间被划分成以原型为中心的离散区域[^src-st-ssdl]。

### 3. 两个自监督目标

**对比损失**（Triplet Loss 变体）：

$$L_{\text{Con}} = \max\left( \|\nabla(Q^c) - P^c\|_2^2 - \|\nabla(Q^c) - N^c\|_2^2 + \delta, 0 \right)$$

鼓励查询靠近正原型、远离负原型，增强原型间可区分性。$\nabla$ 表示 stop-gradient，防止模型把所有表示坍缩到同一原型[^src-st-ssdl]。

**偏差损失**（Deviation Loss）：

$$L_{\text{Dev}} = \left| \nabla(\|Q^c - Q^a\|_1) - \|P^c - P^a\|_1 \right|_1$$

这是 SSDL 的核心创新：强制物理空间中的距离 $D = \nabla(\|Q^c - Q^a\|_1)$ 与潜在空间中正原型间的距离 $\tilde{D} = \|P^c - P^a\|_1$ 保持一致。这实现了 [[relative-distance-consistency|相对距离一致性]] 原则：$D_1 > D_2 \Rightarrow \tilde{D}_1 > \tilde{D}_2$[^src-st-ssdl]。

## 与预测模型集成

SSDL 的两个损失与主预测损失（MAE）联合优化：$L = L_{\text{MAE}} + \lambda_{\text{Con}} L_{\text{Con}} + \lambda_{\text{Dev}} L_{\text{Dev}}$。原型查询的加权和 $V = \sum_i \alpha_i P_i$ 作为增强表示注入 GCRU decoder 的自适应图生成[^src-st-ssdl]。

## 相关页面

- [[st-ssdl]] — ST-SSDL 框架
- [[relative-distance-consistency]] — 相对距离一致性原则
- [[spatiotemporal-deviation]] — 时空偏差概念
- [[contrastive-learning]] — 对比学习方法论

[^src-st-ssdl]: [[source-st-ssdl]]
