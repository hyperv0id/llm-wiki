---
title: "TSFM 协变量适配方法全景对比"
type: analysis
tags:
  - tsfm-adaptation
  - covariate-aware-forecasting
  - time-series-foundation-model
  - iclr-2026
created: 2026-07-04
last_updated: 2026-07-04
source_count: 6
confidence: high
status: active
---

# TSFM 协变量适配方法全景对比

时间序列基础模型（TSFMs）的一个核心张力在于：**预训练于单变量序列的模型，如何有效利用真实预测场景中的多元协变量。** 2024-2026 年间涌现出至少六种不同的适配方案，形成了两条根本对立的设计路线。

## 设计哲学分歧：前置注入 vs 后置注入

> [!important] 核心分歧
> 所有 TSFM 协变量适配方法的根本区别在于**协变量信息在哪个阶段进入模型**。这决定了预训练空间的保护程度和适配的渐进性。

**路线 A：前置注入（Pre-Encoder Injection）**——在 TSFM 编码器之前修改输入结构，将协变量嵌入与目标序列嵌入混合后送入 backbone。代表方法：ChronosX、AdaPTS、UniCA、Gen-P-Tuning。

**路线 B：后置注入（Post-Backbone Injection）**——冻结 TSFM backbone 不变，将协变量信息作为外部条件注入预测头。代表方法：CoRA（adaLN 后置注入）、DiTS（MM-DiT 双流 cross-attention）。

前置注入的根本问题在于：**协变量嵌入扰乱了预训练的嵌入空间分布，触发灾难性遗忘**。UniCA 试图通过 CAP + 自注意力缓解这一问题，但仍未采用零初始化[^src-unica]。CoRA 的立场最激进：预训练嵌入空间应当**绝对保护**，协变量仅作为预测头的条件调制信号[^src-cora]。

## 六方法系统对比

| 维度 | CoRA | UniCA | DiTS | ChronosX | AdaPTS | Gen-P-Tuning |
|------|------|-------|------|----------|--------|-------------|
| **注入位置** | 预测头（adaLN） | 编码器前+后 | 双流 cross-attn | 编码器前 | 编码器前 | 上下文前缀 |
| **backbone 处理** | 完全冻结 | 冻结 | 双流并行 | 冻结 | 冻结 | 冻结 |
| **协变量选择** | Causality Embedding（可解释） | CAP 注意力池化（黑盒） | Token-level conditioning | 无显式选择 | 无显式选择 | Prompt 学习 |
| **零初始化** | ✅ 全部新增参数 | ❌ | ❌（MM-DiT 标准初始化） | ❌ | ❌ | 部分 |
| **多模态支持** | TS + 文本 + 图像 | TS 为主 | TS 协变量流 | TS 协变量 | TS 协变量 | TS 协变量 |
| **核心优势** | 渐进式融合、可解释选择 | 双阶段融合、同质化处理 | Token 级细粒度控制 | 首个 TSFM 协变量适配 | 概率化多元适配 | 最小结构改动 |
| **核心局限** | 协变量数量可扩展性未讨论 | 前置注入扰乱嵌入空间 | 计算开销（双流 attention） | 无零初始化 | 无零初始化 | 仅前缀级信息注入 |
| **backbone 兼容** | Sundial/TimesFM/Chronos-Bolt/FlowState/Moirai | Chronos-Bolt/TimesFM/Moirai | 自有 MM-DiT backbone | Chronos | 自有 | TimesFM/Lag-Llama |

## 性能对比

在统一使用 Sundial 作为 backbone 的公平对比中（TSLib 基准，7 数据集，预测长度 {96, 192, 336, 720}）[^src-cora]：

| 方法 | avg MSE | 相对 CoRA 差距 |
|------|---------|---------------|
| **CoRA** | **0.068** | — |
| UniCA | 0.084 | +23.5% |
| AdaPTS | 0.084 | +23.5% |
| ChronosX | 0.134 | +97.1% |
| TimeXer（监督） | 0.090 | +32.4% |

多模态场景下差距同样显著：RT-1（图像协变量）上 CoRA 比最佳端到端模型降低 12.7% MSE；Time-MMD（文本协变量）上比 UniCA 降低 1.9% MSE[^src-cora]。

## 为什么后置注入 + 零初始化更优

CoRA 的设计建立在三个相互增强的支柱上：

1. **预训练知识保护**：冻结 backbone 确保 TSFM 在大规模预训练中习得的通用时间模式完好无损。消融实验表明，移除协变量（退化为纯 SFT）仅导致 6.5% MSE 退化——说明预训练知识本身就贡献了大部分性能[^src-cora]。

2. **零初始化消除冷启动劣势**：所有新增参数从零开始，适配起点与预训练模型完全等价。这意味着模型从零样本能力出发渐进提升，而不会出现"适配后反而不如零样本"的尴尬。替换零初始化为 Xavier 初始化导致性能退化 4.3%[^src-cora]。

3. **可解释的协变量门控**：Causality Embedding 学习的权重与传统 Granger-Geweke 因果检验高度相关，赋予适配过程统计可解释性。这与 UniCA 的黑盒注意力池化形成鲜明对比[^src-cora][^src-unica]。

## DiTS：第三条路线

DiTS 代表了与前两者不同的范式：它不使用现有 TSFM，而是从头构建 MM-DiT（多模态 Diffusion Transformer），将目标序列和协变量作为两个独立的 modality stream 通过 dual-stream attention 交互[^src-dits]。这避免了"适配"问题本身，但也放弃了利用预训练 TSFM 的优势。DiTS 的 token-level 细粒度条件控制是 CoRA 的 series-level adaLN 调制所不具备的能力[^src-dits]。

## 开放问题

截至 2026 年中，该领域仍有若干未解决问题：

- **协变量数量可扩展性**：CoRA 的 Causality Embedding 维度为 N（协变量总数），当 N → ∞ 时的行为未被讨论。UniCA 的 CAP 同样面临注意力复杂度挑战。
- **零初始化的理论分析**：零初始化的有效性在经验上被反复验证（LoRA、DiT、CoRA），但缺乏严格的理论解释——为什么零比小随机值更好？
- **模态间结构性先验**：adaLN 将所有模态压缩为统一的 scale/shift 参数，损失了模态特有的结构性信息。DiTS 的 token-level 双流交互可能是更好的方向，但计算代价更高。
- **协变量因果 vs 相关**：Granger 因果 ≠ 真实因果。当协变量与目标变量共享潜在驱动因子时，Granger 因果检验可能高估协变量的预测价值。

## 相关页面

- [[cora-tsfm|CoRA]] — 后置注入 + 零初始化 + 因果嵌入的协变量适配框架
- [[unica|UniCA]] — 前置+后置双阶段融合的协变量适配框架
- [[dits|DiTS]] — MM-DiT 双流架构，协变量作为独立 modality stream
- [[mm-dit-for-time-series]] — MM-DiT 在时间序列中的应用范式
- [[zero-initialized-adaptation]] — 零初始化适配的设计原理与跨领域实例
- [[heterogeneous-covariates]] — 异构协变量的分类与挑战
- [[sundial|Sundial]] — CoRA 和对比实验的主要 backbone
- [[timesfm|TimesFM]] — CoRA 兼容的 decoder-only TSFM
- [[chronos|Chronos]] — CoRA 兼容的 tokenized TSFM
- [[multimodal-time-series-forecasting]] — 多模态时间序列预测的总体概念

[^src-cora]: [[source-cora]]
[^src-unica]: [[source-unica]]
[^src-dits]: [[source-dits]]
[^src-timesfm]: [[source-timesfm]]
[^src-chronos]: [[source-chronos]]
[^src-sundial]: [[source-sundial]]