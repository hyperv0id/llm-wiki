---
title: "Variable-wise Partitioning (变量维度划分基准)"
type: concept
tags:
  - time-series
  - data-imputation
  - benchmarking
  - generalization
  - evaluation-protocol
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: medium
status: active
---

# Variable-wise Partitioning（变量维度划分基准）

**Variable-wise Partitioning** 是 [[nuwats|NuwaTS]] (arXiv 2024) 提出的时间序列插补评测协议：将多元时间序列沿**变量（传感器）维度**而非时间维度划分 train/validation/test 集合（默认 1:1:1 比例），从而严格检验模型对**未见变量**和**未见领域**的泛化能力[^src-nuwats]。

## 动机：时间维度划分的盲区

传统插补/预测评测沿**时间维度**切分——模型在历史记录上训练，在同一批变量的未来观测上测试[^src-nuwats]。这只衡量"同变量、未来时刻"的插补能力，掩盖了部署中两类更现实的泛化需求：

- **Cross-variable（跨变量）**：同领域、不同变量、相似模式。例如工厂新上线一条生产线产生新传感器序列，模型能否泛化到这些训练时未见的变量？
- **Cross-domain（跨领域）**：不同领域、不同模式。例如缺乏某领域训练数据时，只能依赖在其他领域（如交通）训练的模型去插补该领域（如工厂）的数据[^src-nuwats]。

时间维度划分对这两点均无能为力，因为测试变量在训练时已被模型"见过"。

## 协议

把多元序列的变量集合（而非时间轴）按 1:1:1 划分到 train/val/test[^src-nuwats]：

| 划分 | 内容 |
|------|------|
| Train | 一部分变量（模拟"已有较完整数据、低缺失率的少数传感器"） |
| Val | 另一批**不相交**变量 |
| Test | 又一批**不相交**变量（模拟"高缺失率的其他传感器/领域"） |

这模拟真实部署：用少数传感器的较完整数据训练模型，再去插补**其他**传感器/变量的高缺失数据。NuwaTS 在此协议下用输入长度 96、随机缺失率 0.1–0.9 训练，并在 0.1…0.9 共 9 档缺失率上分别测试[^src-nuwats]。

> [!note] 与 channel-independence 的协同
> 变量维度划分天然偏好 [[channel-independence|channel-independent]] 模型——CI 模型（如 NuwaTS、PatchTST）对变量数无固定要求，可在变量维不同的数据集间迁移；而 channel-dependent 模型（TimesNet、GPT4TS）需固定输入维度，只能在同变量数数据集间零样本[^src-nuwats]。论文为兼容 CD 基线才采用固定划分比例。

## 意义

该协议把"是否为基础模型"的检验标准从"拟合域内未来"提升到"泛化到未见变量/领域"，与 SAM、GPT 等基础模型的评测哲学一致[^src-nuwats]。它也使**插补任务的 scaling law** 可观测：在多域融合数据上训练后，NuwaTS 与 PatchTST 的跨变量/跨域泛化均提升[^src-nuwats]。

## 关联页面

- [[nuwats]] — 提出此基准的插补基础模型
- [[channel-independence]] — 支撑跨变量泛化的处理策略
- [[spatio-temporal-foundation-model]] — 跨城市/跨域零样本的相关泛化范式
- [[imputeformer]] — 其零样本迁移实验同样关注跨数据集泛化

[^src-nuwats]: [[source-nuwats]]
