---
title: "Multimodal Time Series Forecasting"
type: concept
tags:
  - time-series
  - multimodal
  - forecasting
  - covariate
created: 2026-04-29
last_updated: 2026-05-03
source_count: 6
confidence: high
status: active
---

# Multimodal Time Series Forecasting (多模态时间序列预测)

## 定义

**多模态时间序列预测**是指同时利用多种数据模态（如数值时间序列、图像、文本）进行未来值预测的任务[^src-unca]。

## 背景与挑战

传统时间序列预测方法主要依赖数值型历史数据。然而，在许多实际应用场景中，外部信息以多种形式存在：

| 模态 | 示例 | 挑战 |
|------|------|------|
| 数值 | 温度、销量、价格 | 标准处理 |
| 分类 | 商品ID、门店类型 | 需要 embedding |
| 图像 | 卫星云图、工业检测 | 维度高、语义异构 |
| 文本 | 新闻、天气报告、社交媒体 | 序列长、语义复杂 |

## 现有方法分类

### 1. 文本增强预测

利用 LLM 的强大时间编码能力：
- **Time-LLM**：将时间序列 reprogramming 为 LLM 输入
- **ChatTime**：结合时间感知提示
- **LLM4TS**：零样本 LLM 预测

局限性：通常处理静态文本，难以利用动态文本信息。

### 2. 图像-时间序列预测

- **FusionSF**：专为卫星场景设计
- **MMSP 数据集**：多模态太阳能预测

### 3. 时间序列基础模型方法

| 方法 | 多模态支持 | 特点 |
|------|-----------|------|
| Moirai | 有限 | 展平为联合序列 |
| Chronos | 无 | 仅支持数值 |
| TimesFM | 无 | 仅支持数值 |
| **UniCA** | 完全支持 | 同质化 + 融合 |

## UniCA 的解决方案

UniCA 通过**协变量同质化**将不同模态转换为统一表示：

1. **模态专用编码器**：CNN（图像）、GIST（文本）
2. **线性投影层**：将特征映射到同质时间序列空间
3. **统一融合框架**：Pre-Fusion + Post-Fusion 处理所有模态[^src-unca]

## 数据集

### MMSP (Multimodal Solar Power)

- 太阳能发电预测
- 输入：历史发电量 + 卫星云图
- 评估指标：MAE（MAPE 不稳定）

### Time-MMD

- 多模态时间序列数据集
- 输入：数值序列 + 文本描述
- 评估指标：MAPE

## 实验结果

UniCA 在多模态场景下的表现：

| 数据集 | 基线模型 | +UniCA | 提升 |
|--------|---------|--------|------|
| MMSP | TimesFM | TimesFM | -6.5% MAE |
| MMSP | Chronos-Bolt | Chronos-Bolt | -5.9% MAE |
| Time-MMD | TimesFM | TimesFM | -5.9% MAPE |
| Time-MMD | Chronos-Bolt | Chronos-Bolt | -13.0% MAPE |

## ChannelMTS：高铁信道预测

**ChannelMTS** (KDD 2026) 是首个将**环境信息**（位置、K因子、RMS延迟）融入高铁信道预测的多模态框架[^src-channelmts]。

## MoST：多模态时空交通基础模型

**[[most|MoST]]** (KDD 2026) 是首个多模态时空交通预测基础模型，支持卫星图像、POI文本、位置坐标和时间序列四种模态的任意组合输入[^src-most]。与 UniCA（适配现有 TSFMs）和 ChannelMTS（高铁信道专用）不同，MoST 从零训练为原生多模态基础模型，通过 SNR 自适应模态选择和 MoE 空间专家实现零样本跨城市泛化[^src-most]。

## Aurora：多模态时间序列基础模型

**[[aurora|Aurora]]** (arXiv 2026) 是首个多模态时间序列基础模型，支持文本、图像和数值时间序列的多模态输入和零样本推理[^src-aurora]。与 UniCA（适配现有 TSFMs）、MoST（判别式 ST 预测）和 VoT（LLM 推理式）不同，Aurora 是**生成式**多模态基础模型，通过 Modality-Guided Self-Attention 和 Prototype-Guided Flow Matching 实现概率预测[^src-aurora]。

## TaTS：文本作为辅助变量

**[[tats|TaTS (Texts as Time Series)]]** (ICLR 2026) 是一个即插即用的多模态时间序列框架，由 Li et al. (UIUC/Meta/IBM) 提出[^src-language-in-the-flow-of-time]。TaTS 基于 **[[chronological-textual-resonance|Chronological Textual Resonance (CTR)]]** 现象——时间序列配对的文本天然展现出与数值序列一致的周期性——将文本编码后作为辅助变量拼接到原始时间序列中，无需修改任何现有模型架构[^src-language-in-the-flow-of-time]。在 18 个数据集和 9 个模型上验证，预测和插补任务均取得一致提升。TaTS 的核心优势在于极简设计：仅需 MLP 降维 + 拼接操作，与 Transformer-based、线性、频域模型均兼容[^src-language-in-the-flow-of-time]。

### 与其他多模态模型的对比

| 维度 | Aurora | UniCA | MoST | VoT | TaTS | ChannelMTS |
|------|--------|-------|------|-----|------|------------|
| 范式 | 生成式基础模型 | 适配框架 | 判别式基础模型 | LLM 推理 | 即插即用框架 | 任务专用 |
| 模态 | 文本 + 图像 + TS | 分类 + 图像 + 文本 | 图像 + 文本 + 位置 + TS | 文本 + TS | 文本 + TS | 环境 + TS |
| 零样本 | ✓ | ✓ (via TSFM) | ✓ | ✗ | ✗ | ✗ |
| 生成方式 | Flow Matching | N/A | N/A | LLM 生成 | N/A | N/A |
| 概率预测 | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| 架构修改 | 需要 | 需要 (fusion module) | 需要 | 需要 (dual-branch) | **不需要** | 需要 |

### 与 VoT 的区别

**[[vot|VoT (Value of Text)]]** (ICLR 2026) 是另一个多模态时间序列预测模型，同样来自 ECNU 团队。与 ChannelMTS 不同，VoT 专注于利用 LLM 的推理能力从外生文本（新闻、政策文件）中提取预测信号，并通过双分支架构融合文本推理与数值预测。

| 维度 | ChannelMTS | VoT |
|------|-----------|-----|
| 目标 | 高铁信道预测 | 通用多模态时间序列预测 |
| 文本类型 | 环境参数 (K因子、RMS延迟) | 外生文本 (新闻、政策) |
| LLM 使用 | 无 | 推理 + 特征提取 |
| 对齐方式 | 自适应动态权重 | 多级对齐 (表示级 + 预测级) |
| 数据集 | HSR/VSR (通信) | 10 个真实世界数据集 |

### 与 UniCA 的区别

| 维度 | ChannelMTS | UniCA |
|------|-----------|-------|
| 目标 | 高铁信道预测 | TSFM 协变量适应 |
| 输入 | 环境信息 + 时间序列 | 分类/图像/文本协变量 |
| 核心方法 | RAGC + 未来环境信息 | 协变量同质化 |
| 融合策略 | 自适应动态权重 | Pre-Fusion + Post-Fusion |
| 部署 | 离线训练 | 即插即用适配器 |

### 关键创新

1. **检索增强统计信道 (RAGC)**：从预缓存的高铁地图中检索相似环境对应的历史统计信道
2. **未来环境信息利用**：利用铁路轨迹预定义特性，使用未来环境信息提升预测
3. **线上部署验证**：真实 5G NR 系统上 A/B 测试 MSE 降低 82%-92%

### 实验结果

| 数据集 | ChannelMTS MSE | 最佳基线 MSE | 提升 |
|--------|---------------|-------------|------|
| HSR I | 0.0722 | 0.0859 (ChatTime) | 16% |
| VSR I | 0.1675 | 0.2569 (ChatTime) | 35% |

---

## 相关概念

- [[heterogeneous-covariates]] — 异构协变量
- [[covariate-homogenization]] — 协变量同质化
- [[unified-covariate-adaptation]] — UniCA 框架
- [[channelmts]] — 高铁多模态信道预测框架
- [[most]] — 多模态时空交通基础模型
- [[timesnet]] — 时间序列基础模型
- [[multimodal-time-series-anomaly-detection]] — 多模态时间序列异常检测（MindTS, ICLR 2026）
- [[mindts]] — MindTS 多模态异常检测模型
- [[vot]] — VoT 多模态时间序列预测模型 (ICLR 2026)
- [[aurora]] — Aurora 多模态生成式基础模型 (arXiv 2026)
- [[tats]] — TaTS 即插即用多模态框架 (ICLR 2026)
- [[chronological-textual-resonance]] — CTR 现象
- [[texts-as-auxiliary-variables]] — 文本作为辅助变量概念
- [[generative-time-series-forecasting]] — 生成式时间序列预测概念
- [[event-driven-reasoning]] — 事件驱动推理范式
- [[multi-level-alignment]] — 多级对齐概念

---

## 引用

[^src-unca]: [[source-unca]]
[^src-channelmts]: [[source-channelmts]]
[^src-most]: [[source-most]]
[^src-event-driven-ts-forecasting]: [[source-event-driven-ts-forecasting]]
[^src-aurora]: [[source-aurora]]
[^src-language-in-the-flow-of-time]: [[source-language-in-the-flow-of-time]]