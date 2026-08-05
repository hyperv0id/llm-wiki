---
title: "Aurora: Towards Universal Generative Multimodal Time Series Forecasting"
type: source-summary
tags:
  - multimodal-time-series
  - foundation-model
  - generative-forecasting
  - flow-matching
  - zero-shot
  - arxiv-2026
created: 2026-05-03
last_updated: 2026-08-05
source_count: 1
confidence: high
status: active
---

# Aurora: Towards Universal Generative Multimodal Time Series Forecasting

**Wu, Jin, Qiu, Chen, Shu, Yang & Guo (2026), ICLR 2026, arXiv:2509.22295v6**

完整论文（40 页，ICLR 2026 camera-ready）：`raw/2509.22295.pdf`。代码：https://github.com/decisionintelligence/Aurora ；权重：https://huggingface.co/DecisionIntelligence/Aurora

## 核心论题

论文提出 Aurora，并将其定位为首个多模态时间序列基础模型（Multimodal Time Series Foundation Model），在 Cross-domain Multimodal Time Series Corpus 上预训练，支持多模态输入（文本、图像、数值）和零样本推理，通过生成式概率预测实现跨域泛化[^src-aurora]。核心论点：单模态 TSFMs（Sundial、VisionTS、Time-MoE 等）只在裸数字里找域间共性、缺乏对文本/图像中领域知识的显式利用；端到端多模态监督模型（GPT4MTS、CALF 等）能融合文本知识但"训练什么域用什么域"，不支持零样本跨域推理——论文认为 Aurora 填补了"预训练 + 多模态 + 零样本"同时成立的空白[^src-aurora]。

## 方法

### 编码阶段：多模态领域知识提取

Aurora 采用 Channel-Independence（PatchTST 式）时序骨干，通过 tokenization、encoding 和 distillation 三阶段从多模态输入中提取领域知识[^src-aurora]：

1. **Tokenization**：时序经 RevIN + 非重叠 Patching（参考 patch size $p=48$）得到时间 token；内生图像通过 FFT 求主周期 $F=\arg\max|Amp(FFT(X))|$，按周期折叠为 2D 矩阵、沿通道重复并 resize 成 3 通道图像，再经 ImagePatching 得到图像 token；文本经 Bert tokenizer 得到文本 token[^src-aurora]
2. **Encoding**：图像/文本 token 分别过预训练 ViT 与 Bert 编码器得到模态表示[^src-aurora]
3. **Distillation**：以 $K^{\text{text}}$、$K^{\text{image}}$ 个可学习向量为 query（语义聚类质心），用交叉注意力把文本/图像 token 压缩为少量精华 token，过滤冗余信息（关键描述往往只值几个词）[^src-aurora]

### Modality-Guided Multi-head Self-Attention

先通过 VisionGuider / TextGuider 捕获时间模态与其他模态的相关性 $V_{\text{Attn}}$、$T_{\text{Attn}}$，再以可学习度量 $W$ 桥接出时间内部相关性 $\text{Corr} = V_{\text{Attn}} \cdot W \cdot T_{\text{Attn}}^\top$，将 $\text{Corr}$ 注入自注意力打分 $S = (QK^\top + \text{Corr})/\sqrt{d}$——外部领域知识直接调节时间 token 之间的注意力权重[^src-aurora]。之后经 Cross-Attention Fuser 把三模态表示融合为 $X^{\text{fuse}} = X^{\text{time}} + \tilde{X}^{\text{image}} + \tilde{X}^{\text{text}}$。

### Prototype-Guided Flow Matching

解码阶段由三部分构成[^src-aurora]：

1. **ConditionDecoder**（DiT 启发，Causal-Transformer + RoPE 的 Cross-Transformer）：把融合表示解码为未来 token 的多模态条件 $X^{\text{cond}}$
2. **Prototype Bank + PrototypeRetriever**：1000 个以三角函数/指数/对数/多项式基初始化的可学习"未来雏形"向量；Transformer 检索器根据文本/图像表示 + 未来位置 Sinusoidal Embedding，经 Softmax 加权合成针对当前样本的未来原型 $\tilde{P} = D \cdot P$
3. **Flow-Matching Network**（MLP + AdaLN 条件注入）：从"原型 + 高斯噪声" $y^{(0)} = \tilde{P}_i + \epsilon_i$ 出发而非纯噪声，沿条件 OT 路径拟合速度场，token-wise 优化 $\mathcal{L}(\theta, h_i) = \mathbb{E}\left[\|v_t^\theta(y_i^{(t)}|h_i) - (y_i^{(1)} - y_i^{(0)})\|^2\right]$；推理时按 Algorithm 1 离散积分采样，多次采样得到概率预测

## 实验

在 5 个公认基准（TimeMMD、TSFM-Bench、ProbTS、TFB、EPF）上评估，预训练语料与下游严格不重叠[^src-aurora]：

| 场景 | 基准 | 结果 |
|------|------|------|
| 多模态零样本 | TimeMMD | 平均 MSE 较 Sundial 降 27.0%、较 VisionTS 降 31.2%（1st count 31/26） |
| 多模态 few-shot（10% 数据） | TimeMMD | 较 GPT4MTS 降 12.8%、较 CALF 降 24.5%；Climate/Environment 上零样本即优于 full-shot 基线 |
| 单模态零样本（确定性） | TSFM-Bench | 平均 MSE 较 Time-MoE 降 15.1%、较 ROSE 降 22.9%（1st count 27/21） |
| 单模态零样本（概率） | ProbTS | 平均 CRPS 较 CSDI 降 21.5%、较 MOIRAI 降 38.3%（1st count 19/24） |
| 短时预测 | EPF + TFB（8068 单变量集） | 平均 MASE 2.13、平均 msMAPE 19.96（论文 Figure 5 报告的 Aurora 值），均低于全部对照模型 |

**消融**（TimeMMD 九域平均，表 5）：去掉 Modality-Guided Attention（退化为普通 MSA）后 Economy 域 MSE 从 0.033 升至 0.277（0.277/0.033≈8.4 倍，本课程换算）；去掉 Prototype-Guided Flow Matching（退回高斯噪声起点）后 Social Good 域 MSE 从 0.838 升至 1.425；论文报告两模块同时移除时性能大幅下降（cascading effect）[^src-aurora]。

**效率**（Environment，horizon 336，batch 1）：参数 210.8M、MACs 18.329G、GPU 显存 1265MB、推理 83.5ms/样本。采样数从 20 增至 100 时 CRPS/NMAE 持续改善，论文报告采样数达 100 时性能良好[^src-aurora]。

**训练**：8×A800 80GB 上从头训练约 30 天；AdamW + StepLR，初始 lr $5\times10^{-5}$，batch size 8192；11 个历史 token + 4 个预测 token[^src-aurora]。

## 关键贡献

1. 论文将 Aurora 定位为首个支持多模态输入和零样本推理的多模态时间序列基础模型
2. Modality-Guided Multi-head Self-Attention 以跨模态相关桥接时间内部相关，将领域知识注入时间建模主流程
3. Prototype-Guided Flow Matching 以"未来原型 + 噪声"而非纯噪声为生成起点
4. 作者报告在 5 个基准上单模态和多模态场景均取得 SOTA

## 预训练语料

- **构成**：Cross-Domain Multimodal Time Series Corpus，来源包括 ERA5（1.34 亿点）、IoT（9941 万点）、Monash（Wind Farms 1.72 亿点、London Smart Meters 1.67 亿点、Web Traffic 1.16 亿点等）、UEA/UCR（MotorImagery 7258 万点、TDBrain 7923 万点等）、PEMS（5423 万点）等，总计 **>10 亿时间点**；域分布：Nature 24.2%、Energy 18.8%、Health 15.9%、Weather 13.3%、Web 11.6%、Cloud 9.9%、Transport 5.4%、Economy 0.9%
- **文本生成**：真实多模态时序数据稀缺，采用 GPT-4 按"领域描述 + 样本曲线"启发式生成逐样本文本说明（≤200 词）；生成后先由另一个 GPT-4 agent 粗查质量，低质则重置，批次内随机抽样人工复核并据此调整 prompt
- **模态缺失训练**：预训练阶段随机遮挡文本，使模型在无文本输入时仍可预测（内生图像始终可从时序自身获得）

## 边界与课程评估

论文正文未单设局限性章节；以下结合论文自述与本课程评估，交代适用边界。

- **论文自述**：预训练与评测文本均为 GPT-4 生成的模拟文本，模型学的是"读 GPT 风格描述"；论文未在真实、更脏、时序对齐更弱的下游文本上正面验证分布外表现。**本课程评估**：这是该模型最需要真实多模态数据复核的边界。
- **论文自述**：Corr 桥接依赖"跨模态相关能重构时间内部结构"的假设。**本课程评估**：文本与图像同时信息不足时 Corr 可能注入噪声；论文未讨论"知识缺失但未被遮挡"的中间态。
- **论文自述**：原型库以周期与趋势基（三角/指数/对数/多项式）为骨架。**本课程评估**：对突变、混沌、长尾事件（如金融极端事件）的覆盖，论文未展开说明。

[^src-aurora]: [[source-aurora]]
