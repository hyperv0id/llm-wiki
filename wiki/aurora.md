---
title: "Aurora"
type: entity
tags:
  - multimodal-time-series
  - foundation-model
  - generative-forecasting
  - flow-matching
  - zero-shot
  - arxiv-2026
created: 2026-05-03
last_updated: 2026-08-05
source_count: 5
confidence: high
status: active
---

# Aurora

论文将 **Aurora** 定位为首个多模态时间序列基础模型（Multimodal Time Series Foundation Model），由 Wu, Jin, Qiu, Chen, Shu, Yang 和 Guo 提出（ICLR 2026, arXiv:2509.22295v6）[^src-aurora]。Aurora 支持多模态输入（文本、图像、数值时间序列）和零样本推理，通过生成式概率预测实现跨域泛化。代码与权重开源（GitHub `decisionintelligence/Aurora`、HuggingFace `DecisionIntelligence/Aurora`）。

## 核心能力

Aurora 填补了现有时间序列基础模型的两个关键空白[^src-aurora]：

1. **单模态 TSFMs**（如 [[timesfm|TimesFM]]、[[chronos|Chronos]]）缺乏对领域特定知识（文本、图像）的显式利用
2. **端到端多模态监督模型**不支持零样本推理

Aurora 在 Cross-domain Multimodal Time Series Corpus（>10 亿时间点，ERA5/IoT/Monash/UEA/UCR/PEMS 等来源，域覆盖 Nature 24.2%/Energy 18.8%/Health 15.9%/Weather 13.3% 等）上预训练，能够自适应提取和聚焦于文本或图像模态中包含的关键领域知识[^src-aurora]。预训练与下游基准严格不重叠；文本描述由 GPT-4 生成并经"GPT-4 粗查 + 人工抽检"双重质检；预训练阶段随机遮挡文本以支持模态缺失场景。

## 架构

### 编码阶段

1. **Tokenization**：时序（Channel-Independence）经 RevIN + 非重叠 Patching（patch size 48）切块；内生图像由 FFT 主周期折叠 2D 后渲染为 3 通道图并经 ImagePatching 切块；文本经 Bert tokenizer 分词[^src-aurora]
2. **Encoding + Distillation**：ViT/Bert 编码后，以 $K$ 个可学习语义质心为 query 的交叉注意力蒸馏出少量精华 token，过滤冗余[^src-aurora]
3. **Modality-Guided Multi-head Self-Attention**：以 VisionGuider/TextGuider 捕获的跨模态相关桥接出 $\text{Corr}$ 矩阵注入自注意力打分，将多模态领域知识注入时间表示建模，最后经 Cross-Attention Fuser 融合为 $X^{\text{fuse}}$[^src-aurora]

### 解码阶段

1. **ConditionDecoder**（Causal-Transformer + Cross-Transformer，DiT 启发）：由融合表示解码未来 token 的多模态条件
2. **Prototype-Guided Flow Matching**：1000 个周期/趋势原型（三角/指数/对数/多项式基初始化）经 PrototypeRetriever 加权合成未来原型，作为流匹配起点（$y^{(0)}=\tilde{P}_i+\epsilon_i$），沿条件 OT 路径生成概率预测[^src-aurora]；推理按 Algorithm 1 离散积分，多次采样得到预测分布

**规模**：Encoder 1 层、Decoder 9 层、Flow-Matching Net 3 层；Model Dim 256、FFN Dim 512；共 **210.8M 参数**。推理（Environment，horizon 336）：MACs 18.329G、显存 1265MB、83.5ms/样本。

## 与现有模型的对比

| 维度 | Aurora | [[simdiff|SimDiff]] | [[most|MoST]] | [[timesfm|TimesFM]] | [[chronos|Chronos]] |
|------|--------|---------------------|---------------|---------------------|---------------------|
| 类型 | 多模态基础模型 | 单模态扩散模型 | 多模态 ST 基础模型 | 单模态基础模型 | 单模态基础模型 |
| 模态 | 文本 + 图像 + 数值 | 仅数值 | 图像 + 文本 + 位置 + TS | 仅数值 | 仅数值 |
| 生成方式 | Flow Matching | Diffusion (DDPM) | 判别式 | 自回归 | 自回归 |
| 零样本 | ✓ | ✗ | ✓ | ✓ | ✓ |
| 跨域泛化 | ✓ | ✗ | ✓ (跨城市) | ✓ (跨数据集) | ✓ (跨数据集) |
| 概率预测 | ✓ (生成式) | ✓ (扩散式) | ✗ | ✗ | ✗ |

### 与 SimDiff 的对比

两者都是生成式方法，但：
- **SimDiff** 使用扩散模型（DDPM）进行点预测，仅支持单模态数值输入[^src-simdiff]
- **Aurora** 使用 Flow Matching 进行概率预测，支持多模态输入和零样本推理[^src-aurora]

### 与 MoST 的对比

两者都是多模态基础模型，但：
- **MoST** 是判别式模型，通过 SNR 自适应模态选择进行时空交通预测[^src-most]
- **Aurora** 是生成式模型，通过 Modality-Guided Attention 和 Prototype-Guided Flow Matching 进行通用时间序列预测[^src-aurora]

### 与 VoT 的对比

两者都利用多模态文本信息，但：
- **[[vot|VoT]]** 使用 LLM 进行事件驱动推理，通过多级对齐融合文本与时间序列[^src-event-driven-ts-forecasting]
- **Aurora** 通过 tokenization-encoding-distillation 提取领域知识，使用注意力引导注入时间建模[^src-aurora]

## 实验

在 5 个基准上评估：TimeMMD、TSFM-Bench、ProbTS、TFB 和 EPF，作者报告单模态和多模态场景均取得 SOTA[^src-aurora]：

- **TimeMMD 多模态零样本**：平均 MSE 较 Sundial 降 27.0%、VisionTS 降 31.2%；**10% few-shot 训练**即超越 full-shot 多模态监督模型（GPT4MTS −12.8%、CALF −24.5%），Climate/Environment 上零样本即优于 full-shot 基线
- **TSFM-Bench 单模态零样本**：平均 MSE 较 Time-MoE 降 15.1%、ROSE 降 22.9%（1st count 27/21，即在基准各指标上取得最优结果的次数）
- **ProbTS 概率预测**：平均 CRPS 较 CSDI 降 21.5%、MOIRAI 降 38.3%
- **TFB（8068 单变量集）**：平均 MASE 2.13、平均 msMAPE 19.96（论文 Figure 5 报告的 Aurora 值），均低于全部对照模型；**EPF 短时预测**与 full-shot 监督模型竞争

**消融**（TimeMMD 九域）：去掉模态引导注意力（退化为普通 MSA）后 Economy MSE 0.033→0.277（约 8.4 倍）；去掉原型引导流匹配（退回高斯噪声起点）后 Social Good MSE 0.838→1.425；论文报告两者同时移除时性能大幅下降（cascading effect）[^src-aurora]。

**采样可扩展性**：采样数 20→100 时 CRPS/NMAE 持续改善，100 次达良好性能。

## 相关页面

- [[source-aurora]] — 源文件摘要
- [[modality-guided-self-attention]] — 模态引导自注意力技术
- [[prototype-guided-flow-matching]] — 原型引导流匹配技术
- [[generative-time-series-forecasting]] — 生成式时间序列预测概念
- [[multimodal-time-series-forecasting]] — 多模态时间序列预测概念
- [[flow-matching]] — Flow Matching 理论基础
- [[simdiff]] — 扩散式生成预测对比
- [[most]] — 判别式多模态基础模型对比
- [[vot]] — LLM 推理式多模态预测对比
- [[tats]] — TaTS 即插即用多模态框架（Aurora 为生成式基础模型，TaTS 为轻量级插件）
- [[timesfm]] — 单模态 TSFM 对比
- [[chronos]] — 单模态 TSFM 对比
- [[uniextreme]] — UniExtreme 极端天气基础模型（Aurora: 通用 TS 多模态生成式；UniExtreme: 天气领域极端事件判别式）

[^src-aurora]: [[source-aurora]]
[^src-simdiff]: [[source-simdiff]]
[^src-most]: [[source-most]]
[^src-event-driven-ts-forecasting]: [[source-event-driven-ts-forecasting]]
