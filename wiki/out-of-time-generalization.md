---
title: "Out-of-Time (OOT) Generalization"
type: concept
tags:
  - traffic-forecasting
  - spatio-temporal
  - out-of-distribution
  - distribution-shift
  - long-term-deployment
created: 2026-09-16
last_updated: 2026-09-16
source_count: 2
confidence: low
status: active
---

# Out-of-Time (OOT) Generalization

**Out-of-time（OOT）泛化**是 DynaSTar 论文提出的部署期泛化设定：将连续时间线划分为不重叠的 temporal phase（数月或一年），模型在历史 phase $\{P_1,\dots,P_M\}$ 上训练，目标是 minimized 未来未见 phase 的期望预测误差——且不仅是数据分布 $P(\mathbf{X},\mathbf{Y})$，图结构 $\mathbf{A}_t$ 本身也在 phase 间漂移[^src-dynastar]。

## 与既有 OOD 分类的关系

[[spatio-temporal-ood-learning|ST-OOD]] 文献通常把分布漂移分为时间 OOD（T-OOD：信号统计量漂移）与结构 OOD（S-OOD：节点增删导致的图变化）两轴[^src-stop]。DynaSTar 的 OOT 设定与二者的区别在于：漂移不是一次性的场景切换，而是**持续演化的拓扑**——论文将 OOT 与"conventional OOD scenarios"区分开来，其核心论点是既有 OOD 方法用静态邻接建模漂移、忽视拓扑动态[^src-dynastar]。

## 评测协议

DynaSTar 将 OOT 操作化为两阶段测试：LargeST 的 SD/SGBA 子集以 2019 年数据训练（7:3 训/验），2020 年上半年为**近期部署**（near-term）、下半年为**长期部署**（long-term）测试；论文观察多数方法长期误差高于近期，GNN 基线与纯时间 OOD 模型随时间显著退化[^src-dynastar]。这与 [[st-ood|ST-OOD]] 基准的"同年日历窗 IN vs 次年 OUT"协议同属年度漂移评测，但 DynaSTar 的设定假设拓扑也在变。

## 相关页面

- [[dynastar]] — 提出 OOT 设定并给出解法的模型
- [[momentum-updated-probabilistic-graph]] — 演化拓扑的建模机制
- [[node-heterogeneous-invariant-learning]] — 环境级不变学习机制
- [[spatio-temporal-ood-learning]] — ST-OOD 解法全景
- [[ood-generalization]] — 一般 OOD 概念

## 引用

[^src-dynastar]: [[source-dynastar]]
[^src-stop]: [[source-stop]]
