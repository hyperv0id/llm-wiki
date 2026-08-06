---
title: "Multi-Objective Temporal Masking (MOTM)"
type: technique
tags:
  - masked-modeling
  - self-supervised-pretraining
  - time-series-foundation-model
  - pretraining-objective
  - icml-2026
created: 2026-08-06
last_updated: 2026-08-06
source_count: 1
confidence: medium
status: active
---

# Multi-Objective Temporal Masking (MOTM)

**Multi-Objective Temporal Masking（MOTM）** 是 [[source-2607-01918|Zeus]]（arXiv:2607.01918, ICML 2026）的掩码预训练策略：按「掩码比例 → 时序范围 → 掩码策略」三级采样，把外推、插值、全局一致性等多类归纳偏置组合进同一个自监督目标[^src-2607-01918]。

> [!warning] 同名消歧
> 与 [[motm|MoTM（Mixture of TimeFlow Models，TMLR 2026）]] 同名不同物：后者是零样本插补的 INR 混合方法，本页指 Zeus 的多目标掩码策略。

## 问题：单一掩码目标覆盖不了异构任务

论文指出现有 TSFM 的训练困境：归纳偏置异质——预测需要外推（extrapolation），插补和异常检测需要插值（interpolation），分类需要全局抽象（global abstraction）；单一 BERT 式掩码重建或 GPT 式自回归目标无法同时赋予所有能力[^src-2607-01918]。论文提出 MOTM 作为对该训练困境的回应：用不同形状的掩码分别训练不同能力，让模型在同一个重建目标下同时获得三类归纳偏置[^src-2607-01918]。

## 机制

### 三级采样 pipeline

每次预训练样本按三级顺序生成[^src-2607-01918]：

1. **掩码比例**：$p \sim U(0, 0.5)$，期望 0.25。可变比例用于防过拟合单一缺失率，并模拟短时预测（低比例）与严重缺失（高比例）两类场景[^src-2607-01918]。
2. **时序范围**：序列长度分段均匀采样——0.2 概率采 [64, 512]、0.2 概率采 [513, 2048]、0.6 概率采 [2049, 4096]，随后随机裁剪并 padding 到 4096[^src-2607-01918]。
3. **掩码策略**：从下列策略中采样。

### 四种掩码策略与混合

- **Predictive Mask**：掩掉序列尾部 $\lfloor Tp \rfloor$ 步，训练外推（预测）能力[^src-2607-01918]。
- **Point Mask**：随机掩单个时间步，训练逐点插值与局部连续性[^src-2607-01918]。
- **Multi-Block Mask**：采样多个连续块，块长 $\ell_k \sim U(1, 24)$、总长约 $\lfloor Tp \rfloor$；论文指出块长用均匀分布而非语言建模常用的 Poisson 分布。该策略训练结构化缺失下的插值，受 span corruption 启发[^src-2607-01918]。
- **Single-Block Mask**：在任意位置移除一个长连续段，训练全局一致性，服务分类与上下文异常检测[^src-2607-01918]。
- **Mixed Mask**：简单（multi-block / point）与困难（predictive / single-block）策略组合[^src-2607-01918]。

### 训练目标

损失只在掩码位置计算：quantile loss（pinball loss，论文式 5）[^src-2607-01918]。

## 与 BERT / GPT 式目标的对比

论文认为单一 BERT 式掩码重建或 GPT 式自回归目标无法同时赋予外推、插值与全局抽象三类能力，MOTM 的定位是以掩码形状的组合把这些归纳偏置并入同一目标[^src-2607-01918]。四种掩码与三类能力一一对应：predictive 对应外推，point / multi-block 对应插值，single-block 对应全局一致性；论文以此避免为每种任务单独设计预训练目标[^src-2607-01918]。

## 证据

论文图 6 的消融实验报告[^src-2607-01918]：

- 去掉 predictive mask → GIFT-Eval 明显下降（外推能力）
- 去掉 multi-block mask → 插补下降
- 去掉 single-block mask → 异常检测与分类一致下降（全局一致性）

论文在该消融设置下只报告方向性结果，未给出具体数值[^src-2607-01918]。

## 相关页面

- [[source-2607-01918]] — Zeus 论文源文件摘要
- [[motm]] — 同名歧义：MoTM（Mixture of TimeFlow Models）
- [[time-indexed-foundation-model]] — 不同范式：时间索引表征 vs 掩码重建
- [[patch-based-tokenization]] — patch 与 point tokenization 的对比（Zeus 采用 point-wise）

[^src-2607-01918]: [[source-2607-01918]]
