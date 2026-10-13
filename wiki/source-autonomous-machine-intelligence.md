---
title: "Source: A Path Towards Autonomous Machine Intelligence"
type: source-summary
tags:
  - source
  - cognitive-architecture
  - world-model
  - self-supervised-learning
  - energy-based-model
  - position-paper
created: 2026-10-13
last_updated: 2026-10-13
source_count: 1
confidence: medium
status: active
---

# Source: A Path Towards Autonomous Machine Intelligence

- **作者**：Yann LeCun（NYU Courant Institute of Mathematical Sciences；Meta - Fundamental AI Research）
- **版本与时间**：Version 0.9.2，2022-06-27，62 页
- **证据源**：raw/ 中无此文件。全文取自 v0.9.2 PDF（1641225 字节，md5 253ede84d0f7bb1d2883e719e83aab8b），经 pdftotext -layout 抽取 2780 行；图号与节号均按该版本。

## 核心论点

论文定位为立场论文，不为单一构想主张优先权，核心是将现有设想组装为全模块可微、兼容梯度学习的自主智能架构[^src-autonomous-machine-intelligence]。开篇提出三个挑战：主要靠观察学习表征、预测与行动；以兼容梯度学习的方式实现推理与规划；在多抽象层级与多时间尺度上表示感知及行动计划[^src-autonomous-machine-intelligence]。

## 贡献

四项：全模块可微、多数可训练的可配置认知架构（[[autonomous-agent-architecture]]）；学习层级表征的非生成式预测世界模型 JEPA 与 Hierarchical JEPA（[[jepa]]、[[hierarchical-jepa]]）；产生兼具信息量与可预测性表征的非对比自监督范式（[[non-contrastive-ssl]]）；以 H-JEPA 为底座在不确定下进行分层规划的路径[^src-autonomous-machine-intelligence]。

## 关键机制

架构包含感知、世界模型、代价（intrinsic cost + trainable critic）、短期记忆、actor 与 configurator；感知-动作循环分反应式 Mode-1 与经世界模型规划的 Mode-2，后者是滚动时域模型预测控制，区别在于世界模型与代价均由学习获得（[[mode-1-mode-2-planning]]、[[intrinsic-cost-and-critic]]、[[world-model]]）。推理即能量最小化；预测随机性推进隐变量而非输出分布[^src-autonomous-machine-intelligence]。

## 局限

8.1 节列出开放问题：能否从视频构建并训练 H-JEPA 是第一个开放问题；隐变量正则化方式未定；actor 推断隐变量实例与最优动作序列未作规定，动作空间离散或代价不平滑时梯度法可能失效；Mode-2 中枚举多隐变量配置的机制未描述；预测器动态路由、门控电路与短期记忆等微架构模糊；configurator 最不透明，其分解复杂任务为子目标未解决[^src-autonomous-machine-intelligence]。8.3 节反对"规模足够"、"reward 足够"与"符号操作是推理必要补丁"三种立场[^src-autonomous-machine-intelligence]。

## Related Pages

- [[jepa]] · [[hierarchical-jepa]] · [[non-contrastive-ssl]]
- [[autonomous-agent-architecture]] · [[intrinsic-cost-and-critic]] · [[mode-1-mode-2-planning]] · [[world-model]]
- [[source-jepa-focus-on-slow-features]] — 对该架构中 JEPA 的实证检验

[^src-autonomous-machine-intelligence]: [[source-autonomous-machine-intelligence]]
