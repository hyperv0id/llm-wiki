---
title: "WeatherPEFT"
type: entity
tags:
  - weather-forecasting
  - foundation-model
  - parameter-efficient-fine-tuning
  - peft
  - prompt-tuning
  - fisher-information
  - iclr-2026
created: 2026-07-25
last_updated: 2026-08-08
source_count: 1
confidence: medium
status: active
---

# WeatherPEFT

**WeatherPEFT** 论文自称是首个探索天气基础模型（WFMs）参数高效微调的框架，由 Cao, Lin, Cheng 等（中山大学、港科广、港科大、港中文、深圳国家超算中心、华为、清华）提出，发表于 ICLR 2026[^src-weatherpeft]。它通过两个协同模块——前向传播的 [[task-adaptive-dynamic-prompting|TADP]] 和反向传播的 [[stochastic-fisher-guided-adaptive-selection|SFAS]]——使 WFM 能以极少可训练参数适配异构天气下游任务，实现与全量微调相当甚至更优的性能[^src-weatherpeft]。

## 动机

天气基础模型（如 Aurora 的 1.3B 参数、Prithvi WxC 的 2.3B 参数）规模持续增长，全量微调的计算和存储成本不可持续[^src-weatherpeft]。然而，现有 PEFT 方法（LoRA、DoRA、AdaptFormer、SSF、VPT、APrompt）均设计于视觉/语言领域，面临三大气象特有挑战[^src-weatherpeft]：

1. **变量异质性**：温度、湿度等不同物理量受不同流体力学方程支配，变量间耦合关系随任务变化
2. **分辨率多样性**：分辨率变化（如 5.625°→0.25°）从根本上改变主导物理机制（静力→非静力对流尺度）
3. **时空覆盖差异**：全球与区域任务对特征层级的诉求截然不同

此外，大多数 PEFT 对所有下游任务统一应用相同的可训练参数，无法区分不同参数在不同任务中的不同角色[^src-weatherpeft]。

## 架构

WeatherPEFT 包含两个分阶段运作的模块[^src-weatherpeft]：

### TADP（前向传播）

[[task-adaptive-dynamic-prompting|Task-Adaptive Dynamic Prompting]] 从编码器嵌入权重 $E \in \mathbb{R}^{D\times V\times P_h\times P_w}$ 提取任务特定信息，通过三个专用适配器（逐层提取空间/分辨率、变量间、气象特征内部模式）和自注意力（建模物理量与空间特征的外部耦合），生成任务自适应软提示注入预训练骨干[^src-weatherpeft]。

### SFAS（反向传播）

[[stochastic-fisher-guided-adaptive-selection|Stochastic Fisher-Guided Adaptive Selection]] 利用 Fisher 信息矩阵量化参数对学习目标的敏感度，结合退火随机分量（线性衰减 Uniform 噪声）稳定训练，选择 Top-k 最关键参数更新，其余冻结以保留预训练知识[^src-weatherpeft]。

## 实验

以 Aurora（Bodnar et al., 2025, 1.3B, 3D Swin Transformer U-Net）为骨干，在三类下游任务上验证[^src-weatherpeft]：

| 任务 | 关键结果 |
|------|----------|
| **全球降尺度** 5.625°→1.40625°, 68 变量 | WeatherPEFT 3.48M 参数（默认设置）下优于全部 PEFT 基线；论文报告在 ∼4% 参数预算（52.47M）下逼近 Full-Tuning 1239.94M；DoRA (3.75M) 的 T2m RMSE 比 Full-Tuning 高 ∼36% |
| **集合预报后处理** ENS-10, 10 成员→ERA5 | WeatherPEFT 3.18M 参数；Z500 上 CRPS/EECRPS 均超越 Full-Tuning（72.701 vs 73.760） |
| **中国区域降水预报** ERA5-CH 0.25°, TP-6hr, 12/24/36h | WeatherPEFT 3.38M 参数全面超越所有 PEFT 基线（含 SCT、Child-Tuning、SAM 等 task-selective 方法）；∼4% 参数下超越 Full-Tuning |

消融实验表明 TADP 和 SFAS 各自有效，但组合使用效果最优，验证了前向/反向双阶段的协同增益[^src-weatherpeft]。

## 与相关工作的关系

- **[[weather-foundation-model|天气基础模型]]**：WeatherPEFT 不是 WFM，而是 WFM 的微调适配层，与 Aurora、Prithvi WxC、[[weathergfm|WeatherGFM]] 等是互补关系——WeatherPEFT 为这些大模型的实用部署提供高效微调方案[^src-weatherpeft]。
- **[[weather-prompt|WeatherGFM 的天气提示]]**（本课程对比）：TADP 的"提示"概念与 WeatherGFM 的 weather prompt 不同：前者从编码器嵌入提取任务自适应软提示注入骨干，后者是视觉 in-context learning 的示例对格式。
- **通用 PEFT（LoRA/DoRA/AdaptFormer）**：WeatherPEFT 在三个天气任务上全面超越这些方法，论文认为天气领域 PEFT 需要任务自适应机制[^src-weatherpeft]。

## 相关页面

- [[source-weatherpeft]] — 源文件摘要
- [[task-adaptive-dynamic-prompting]] — TADP 技术详解
- [[stochastic-fisher-guided-adaptive-selection]] — SFAS 技术详解
- [[weather-foundation-model]] — 天气基础模型概念
- [[weathergfm]] — WeatherGFM，天气通用基础模型
- [[weather-prompt]] — WeatherGFM 的天气提示设计

[^src-weatherpeft]: [[source-weatherpeft]]
