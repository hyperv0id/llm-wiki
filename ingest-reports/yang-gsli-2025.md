# Ingest 报告：GSLI (AAAI 2025)

## 创建
- wiki/source-yang-gsli-2025.md — WHY：源文件摘要，GSLI 多尺度图结构学习填补框架
- wiki/gsli.md — WHY：GSLI 实体页面，AAAI 2025 时空数据填补模型
- wiki/node-scale-graph-structure-learning.md — WHY：节点尺度图结构学习技术页面，GSLI 的核心模块之一
- wiki/feature-scale-graph-structure-learning.md — WHY：特征尺度图结构学习技术页面，GSLI 的核心模块之一
- wiki/prominence-modeling-gsl.md — WHY：显著度建模技术页面，GSLI 的权重增强机制

## 修改
- wiki/imputeformer.md — WHY：添加 GSLI 交叉引用和对比，同为时空填补方法
- wiki/cofill.md — WHY：添加 GSLI 到对比表格和关联页面，同为时空填补方法
- wiki/traffic-forecasting.md — WHY：添加 Spatial-Temporal Imputation 子章节，填补与预测相关
- wiki/index.md — WHY：添加新页面条目

## 新建交叉链接
- [[gsli]] ↔ [[imputeformer]] — 同为时空填补方法，GSLI 处理特征异质性，ImputeFormer 引入低秩偏置
- [[gsli]] ↔ [[cofill]] — 同为时空填补方法，GSLI 用图结构学习，CoFILL 用条件扩散
- [[gsli]] ↔ [[traffic-forecasting]] — 填补是预测的前置步骤
- [[gsli]] ↔ [[node-scale-graph-structure-learning]] — GSLI 的核心模块
- [[gsli]] ↔ [[feature-scale-graph-structure-learning]] — GSLI 的核心模块
- [[gsli]] ↔ [[prominence-modeling-gsl]] — GSLI 的权重机制
- [[node-scale-graph-structure-learning]] ↔ [[feature-scale-graph-structure-learning]] — 互补的双尺度学习
- [[prominence-modeling-gsl]] ↔ [[adaptive-graph-agent-attention]] — 不同图权重机制
- [[prominence-modeling-gsl]] ↔ [[embedded-attention]] — 不同空间建模方式
