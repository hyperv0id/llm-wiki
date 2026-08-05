---
title: "Mutual Information Disentanglement"
type: technique
tags:
  - mutual-information
  - disentanglement
  - representation-learning
  - incomplete-multimodal-learning
  - variational-inference
created: 2026-08-05
last_updated: 2026-08-05
source_count: 1
confidence: medium
status: active
---

# Mutual Information Disentanglement

**MI minimax 解耦（Mutual Information Disentanglement）** 是 [[midas|MIDAS]] 的核心技术：把每个模态的表示分解为共享（shared）与独有（exclusive）潜因子后，最小化二者之间的互信息 $I(Z^s_m; Z^e_m)$ 以清除模态独有噪声，同时最大化跨模态共享空间互信息 $I(Z^s_{m_1}; Z^s_{m_2})$ 以强化语义对齐（[^src-midas]）。论文主张互信息作为统计依赖的直接度量，比正交性约束或对抗目标这类间接几何分离更适合不完全多模态条件——后者容易不稳定或产生退化解（[^src-midas]）。

## 解耦项：最小化共享与独有互信息

共享与独有潜变量的互信息无法直接计算。论文用 interaction information 将其拆解（[^src-midas]）：

$$I(Z^s_m; Z^e_m) = I(Z^s_m; X_m) + I(Z^e_m; X_m) - I(X_m; Z^s_m, Z^e_m)$$

在条件独立假设（给定输入 $X_m$ 后 $Z^s_m \perp Z^e_m$）下第三项 $I(Z^s_m; Z^e_m | X_m) = 0$，再引入变分编码器 $q(Z^s_m|X_m)$、$q(Z^e_m|X_m)$ 与变分解码器 $p(X_m|Z^s_m, Z^e_m)$，推导出由两项 KL 散度与重构似然组成的**变分上界**（[^src-midas]）：

$$I(Z^s_m; Z^e_m) \leq \mathbb{E}_{p(X_m)}\left[ D_{KL}(q(Z^s_m|X_m) \| p(Z^s_m)) + D_{KL}(q(Z^e_m|X_m) \| p(Z^e_m)) \right] - \mathbb{E}_{p(Z^s_m,Z^e_m)} \mathbb{E}_{q(Z^s_m|X_m)q(Z^e_m|X_m)} \log p(X_m | Z^s_m, Z^e_m)$$

最小化该上界既抑制共享与独有空间的信息重叠，又通过重构似然保留输入信息，实现"干净的解耦"（[^src-midas]）。

## 对齐项：最大化跨模态共享互信息

跨模态共享互信息同样难以精确估计，论文采用 Deep InfoMax 的 **JSD 估计器**作为变分下界（softplus 判别器 $f_\phi$），并构造**硬负样本**——用另一模态的独有特征 $Z^e_{m_2}$ 替换负样本，迫使判别器区分共享语义与模态独有信息（[^src-midas]）。论文选择 JSD 而非 InfoNCE 的理由是其对负样本数量不敏感、更稳健，适合本设置（[^src-midas]）。

## 辅助目标

纯 MI 目标无法保证共享特征对下游任务语义有效，需配合两类辅助目标（[^src-midas]）：

1. **辅助预测损失 $L_{aux}$**：每个模态的共享表示接线性分类器做情感标签粗分类（二元）弱监督，锚定任务相关语义。
2. **重构损失 $L_{rec}$**：共享/独有拼接后经全连接层映射回输入空间，与原始完整特征算 MSE，保证两类因子都贡献于内容重建。

消融（论文表 V）显示 $L_{pred}$ 最关键：去掉后 MOSI Acc-2 从 71.88 降至 61.96（[^src-midas]）。

## 与其他解耦手段的对比

| 解耦手段 | 原理 | 稳定性 | 代表方法 |
|----------|------|--------|----------|
| 互信息 minimax | 直接最小化共享/独有统计依赖（变分上界） | 有变分下界与重构正则 | [[midas\|MIDAS]]（[^src-midas]） |
| 正交性/对抗约束 | 几何分离或判别器博弈 | 易不稳定/退化（论文观点） | MISA 等（[^src-midas]） |
| 时间/空间双流分解 | 按数据维度显式分流 | 依赖维度可分性 | [[dual-dimension-feature-disentanglement\|RAST]] |

论文自述 MI 驱动的解耦优于损失或约束式分解（以持续优于 MISA 为据）；该结论的适用范围是随机缺失率实验设置下的不完全 MSA（[^src-midas]）。

## 相关页面

- [[midas]] — 该技术的载体模型
- [[source-midas]] — 源文件摘要
- [[mutual-information]] — 互信息概念与定义
- [[contrastive-learning]] — JSD 估计器与 InfoNCE 的关系
- [[information-bottleneck-principle]] — 另一类互信息驱动的表示学习目标

[^src-midas]: [[source-midas]]
