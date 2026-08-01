---
title: "Multimodal Time Series Forecasting"
type: concept
tags:
  - time-series
  - multimodal
  - forecasting
  - covariate
  - satellite-imagery
created: 2026-04-29
source_count: 18
last_updated: 2026-08-01
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
- **[[time-llm|Time-LLM]]**：将时间序列 reprogramming 为 LLM 输入
- **Time-LLM + CVPE**：在 [[time-llm|Time-LLM]] 的 patch embedding 层注入跨变量上下文，Weather ↓4.6% MSE, Traffic ↓6.7% MSE [^src-cvpe-2025]
- **ChatTime**：结合时间感知提示
- **LLM4TS**：零样本 LLM 预测

局限性：通常处理静态文本，难以利用动态文本信息。

### 2. 图像-时间序列预测

- **FusionSF**：专为卫星场景设计
- **MMSP 数据集**：多模态太阳能预测
- **[[time-vlm|Time-VLM]]**（ICML 2025）：时序自生成图像 + 结构化文本 + 检索记忆，经冻结 VLM 桥接三模态（无外生新闻/卫星）[^src-time-vlm]

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

### [[time-mmd|Time-MMD]]

- NeurIPS 2024 D&B 多领域**数值–文本**时序数据集（9 主域；日/周/月；cutoff 至 2024-05）[^src-time-mmd]
- 设计要点：细粒度目标变量对齐；报告+检索互补；LLM 拆分 **fact vs prediction** 控污染；二元时间戳多任务切片[^src-time-mmd]
- 配套 [[time-mmd|MM-TSFlib]]：20+ TSF 骨干 × 开源 LLM，分路建模 + 投影融合；>1000 实验约 95% 多模态更优，MSE 平均降 >15%（富文本域可达 ~40%）[^src-time-mmd]
- 下游评测锚点：[[cora-tsfm|CoRA]] / [[unica|UniCA]] 文本协变量、[[source-gpt4mts|DP-GPT4MTS]]、[[vot|VoT]]、[[timi|TiMi]] 等[^src-time-mmd]

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

## TiMi：Non-Fusion Guidance 多模态预测

**[[timi|TiMi]]** (ICML 2026) 提出第三种多模态预测范式——**[[non-fusion-guidance|Non-Fusion Guidance]]**：不再尝试在表示层对齐或融合文本与数值模态，而是让冻结 LLM 独立推理文本中的未来趋势因果知识，通过 **[[mmoe|Multimodal Mixture-of-Experts (MMoE)]]** 门控机制注入 Transformer backbone 的时序建模过程[^src-timi]。MMoE 包含 TMoE（基于文本的路由）和 SMoE（基于序列全局趋势的路由）两个互补专家系统，可在不修改 backbone 架构的情况下即插即用[^src-timi]。在 16 个多模态基准上一致 SOTA，PatchTST+MMoE 平均 MSE 提升 18.2%[^src-timi]。与 VoT 同为 LLM 推理式方法但放弃特征融合，与 TaTS 同为即插即用但引入因果推理而非简单特征拼接[^src-timi]。

## TESS：离散原语瓶颈的 Non-Fusion 实现

**[[tess|TESS]]**（Li et al., arXiv:2603.12664v2）是 Non-Fusion Guidance 的第二条实现。先用半合成实验（FNSPID 真实序列 + GPT-5.2 生成文本、token 级标注）定位两个瓶颈：冗余 token 分散注意力（焦点比 $R_t<0$）、删冗余后语义仍难解码为数值信号（Signal-Only ≪ Numerical）；再冻结 LLM 将外生新闻分类为四类离散 [[temporal-semantic-primitives|时间演化原语]]（mean shift/volatility/shape/lag），以 top-1/top-2 margin 为不确定度信号经置信门控过滤，最后以 prefix token 条件化 PatchTST[^src-tess]。与 TiMi 的 MoE 路由相比：知识形态从自由文本变为受限类别、引导从路由变为条件化+门控；信息瓶颈定理（4.1）保证预测互信息不损，gating 误差按 $g^2$ 衰减（A.5）。四数据集上相对最强基线最高 +29.1% MSE 降幅（Bitcoin）。其半合成诊断与 [[constrained-text-fusion|CFA]] 的 >20K 实证共同构成「naive 融合有害」的证据链[^src-tess]。

## Time-VLM：VLM 桥接时序 / 视觉 / 文本

**[[time-vlm|Time-VLM]]** (Zhong et al., ICML 2025, arXiv:2502.04395) 用冻结预训练 VLM（默认 ViLT；亦支持 CLIP / BLIP-2）统一 **时序 · 视觉 · 文本**：[[time-vlm|RAL]] 做 patch + local/global 检索记忆，[[time-vlm|VAL]] 将时序经 FFT/周期编码与多尺度卷积渲染为图像，[[time-vlm|TAL]] 生成统计与域描述 prompt；跨模态注意力 + 门控融合后预测。**不依赖外生文本/图像**，仅由原始时序自增强——相对 [[time-mmd|Time-MMD]]/[[vot|VoT]] 的外生文本路线，以及 UniCA/CoRA 的协变量适配路线，是一条 **内生多模态 + 检索** 路径[^src-time-vlm]。约 143.6M 参数（≈1/20 Time-LLM）；5% few-shot 上 ETTh1 相对 Time-LLM MSE 约 −29.5%；Weather 消融显示去 RAL +35.6% MSE、去 VAL +9.0%、去 TAL 仅 +2.1%（文本 token 稀疏）[^src-time-vlm]。

## 对齐极限：独立预训练三模态近正交

**[[ts-vl-alignment|TS–VL Alignment]]**（Yashwante & Yu, arXiv:2602.19367）用 34 组冻结编码器 + 共享投影头、对称 InfoNCE，系统探测时序–折线图–文本的对比空间几何。**无显式耦合时跨模态表示近正交（MAD≈90°）**；后验对齐随尺度改善但不均匀——**TS–IMG ≫ TS–TXT**，全局 cosine/Procrustes 可强而 mutual kNN 弱；文本 information density 仅在低–中段抬升后**饱和**；图像可作 TS–TXT 中介；间接临床文本 / 跨语报告进一步削弱对齐。对外生多模态 ST 的含义：不能默认 Chronos/CLIP/LLM 表示已共享 latent；**轻量投影有上限，需显式耦合与匹配的语义显式性**（与 [[time-vlm|Time-VLM]] 的内生图文桥接、Time-MMD/VoT 的任务侧融合形成互补的诊断层）[^src-ts-vl-alignment]。

## Constrained Text Fusion：Naive 常伤、约束才稳

**[[constrained-text-fusion|Constrained Text Fusion / CFA]]**（Lee et al., LG AI Research, KDD ’26 MILETS, arXiv:2603.22372）在 [[time-mmd|Time-MMD]] 九域上做 **>20K** 设定对照：冻结文本编码器（BERT / GPT-2 / Llama3 / Doc2Vec）× 14 TS 骨干 × first/middle/last × add/concat，发现 **naive 融合经常低于 unimodal TS**（甚至 Div.：MSE>单模态 10×），归因于辅助文本的无关/冲突信号无控注入。**Constrained** 族——Gating、FiLM 调制、正交分量注入、以及 **CFA**（低秩瓶颈残差 \(z_{\mathrm{TS}}+W_{\mathrm{up}}\phi(W_{\mathrm{down}}z_{\mathrm{Text}})\)，\(r=8\)，近零 init）——系统优于 naive；CFA 在 9 域全胜 unimodal、7/9 rank-1、13/14 骨干提升，参数仅约 +0.61%。相对 [[tats|TaTS]]（first-add naive plug-in）与 [[timi|TiMi]] 的 [[non-fusion-guidance|Non-Fusion Guidance]]（完全不融特征），CFA 是 **plug-in + 受控特征融合** 的中间路线[^src-constrained-text-fusion]。

## Cross-Modal Misalignment：缓解 vs 利用

**[[cross-modal-misalignment|Cross-modal misalignment]]**（Cai, Liu et al., NeurIPS 2025, arXiv:2504.10143）在 CLIP 式 **MMCL** 下形式化 **selection bias**（文本省略语义）与 **perturbation bias**（选中语义被改写）：对比表示 **只 block-identify 无偏共享语义子集**，省略/扰动与模态噪声一律剔除。对多模态时序的含义：（1）外生文本若只描述部分动力学因子或含错误展望，对齐空间天然变窄，不能指望规模 alone 补回；（2）若目标是 OOD 稳健，可**有意**让文本省略/扰动环境敏感因子；（3）与 [[ts-vl-alignment|TS–VL]] 的几何上限、[[constrained-text-fusion|CFA]] 的任务侧约束融合形成「预训练可辨识 → 表示几何 → 预测融合」三层互补[^src-cross-modal-misalignment]。

### 与其他多模态模型的对比

| 维度 | Aurora | TiMi | UniCA | MoST | VoT | TaTS | ChannelMTS | PIPE | TESS |
|------|--------|------|-------|------|-----|------|------------|------|------|
| 范式 | 生成式基础模型 | Non-Fusion Guidance | 适配框架 | 判别式基础模型 | LLM 推理 | 即插即用框架 | 任务专用 | 位置编码注入 | Non-Fusion Guidance（原语瓶颈） |
| 模态 | 文本 + 图像 + TS | 文本 + TS | 分类 + 图像 + 文本 | 图像 + 文本 + 位置 + TS | 文本 + TS | 文本 + TS | 环境 + TS | 卫星图像 + TS | 文本 + TS |
| 零样本 | ✓ | ✗ | ✓ (via TSFM) | ✓ | ✗ | ✗ | ✗ | ✓ (跨洋区) | ✗ |
| 生成方式 | Flow Matching | N/A | N/A | N/A | LLM 生成 | N/A | N/A | N/A | N/A |
| 概率预测 | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| 架构修改 | 需要 | **仅 FFN 替换** | 需要 (fusion module) | 需要 | 需要 (dual-branch) | **不需要** | 需要 | **仅位置编码** | 需 PatchTST prefix 注入 |

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

## MTP：多模态交通状态分类

与本章侧重**预测**的多模态方法不同，[[mtp|MTP]] (Xiang et al., AAAI 2026) 将多模态方法扩展到交通状态的**分类**任务（畅通/缓行/拥堵）[^src-mtp]。MTP 通过频域模态增强：FFT 转换数值序列为频率图像 + 周期性图像（视觉模态），LLM 生成描述性文本（文本模态），三大编码器均在频域处理，通过分层对比学习（监督 + InfoNCE + JS 散度分布对齐）融合三模态[^src-mtp]。6 个数据集 8 个基线 SOTA，首次证明多模态增强对时间序列分类同样有效[^src-mtp]。

## AllSpark：多模态时空通用智能模型

**[[allspark|AllSpark]]** (Shao et al., 2024) 是一个统一 10 种时空模态的通用智能模型，涵盖语言、代码、表格（1D）、RGB、SAR、多光谱、高光谱、图、轨迹（2D）和点云（3D）[^src-allspark]。其核心设计原则是 **[[language-as-reference-framework|Language as Reference Framework (LaRF)]]**：将异质模态特征通过 modal bridge 映射到统一的语言特征空间，实现跨模态联合解释[^src-allspark]。虽然 AllSpark 侧重遥感/地理空间智能而非纯时间序列预测，但其轨迹和 graph 模态直接覆盖时空预测任务，且 training-free 的 few-shot 能力（RGB 5-way 1-shot 达 95.58%）代表了多模态模型的新方向[^src-allspark]。

## STReasoner：时空推理 TS-LM

**[[streasoner|STReasoner]]** (Ni et al., 2026) 是首个面向时间序列时空推理的 TS-LM，通过 S-GRPO 空间感知强化学习实现多步 CoT 推理[^src-streasoner]。与本章所有其他模型不同，STReasoner 的核心任务不是预测数值，而是回答自然语言查询（如 "哪个节点导致了 Node 2 在 9:00 的拥堵？"），要求模型显式追踪传播路径、识别延时，整合图结构、时序数据和文本语义[^src-streasoner]。在 ST-Bench 四个任务上，STReasoner-8B 以 0.004× proprietary model 成本实现 17-135% 平均提升[^src-streasoner]。

## ST-Vision-LLM：时间序列即图像

**[[st-vision-llm|ST-Vision-LLM]]** (Yang et al., arXiv 2025) 代表多模态预测中一条独特路线：它不引入外部模态，而是将数值交通矩阵本身渲染为灰度伪RGB图像，用 Vision-LLM (Qwen2.5-VL-7B) 的原生视觉编码器感知整个网格的全局时空场景，再逐格生成数值 token 预测[^src-st-vision-llm]。与 MoST（使用真实卫星图像作为模态）不同，ST-Vision-LLM 的"图像"是数值场的视觉表示，选择视觉编码器纯粹为利用其 2D 网格归纳偏置，并配合 SFT + GRPO 两阶段训练直接优化预测精度[^src-st-vision-llm]。与 [[time-vlm|Time-VLM]] 同属“时序渲染为图像 + VLM”，但 Time-VLM 面向通用 LTSF 基准、冻结编码器 + 门控融合且自带检索记忆与文本自描述；ST-Vision-LLM 面向网格交通、生成式 Vision-LLM + 数值 token / GRPO[^src-time-vlm][^src-st-vision-llm]。

## PIPE：物理知情位置编码的台风预测

**[[pipe|PIPE]]** (Li et al., HKUST, NeurIPS 2025) 首次提出将物理元数据（时间戳、经纬度）嵌入 VLM 位置编码的多模态台风预测方法[^src-pipe]。基于 Qwen-2.5-VL，PIPE 通过两个核心机制在位置编码层注入物理知识：（1）[[physics-informed-position-encoding#1. 物理知情位置索引|物理知情位置索引]]——将图像 token 的位置 ID 替换为物理量（年日、小时、纬度、经度），并映射到负值以避免与文本 token 冲突；（2）[[variant-frequency-positional-encoding|变频率位置编码]]——为不同物理变量分配不同波长的正弦函数[^src-pipe]。在 [[digital-typhoon-dataset|Digital Typhoon]] 数据集上达到 SOTA，台风强度预测 MAE 比此前最优的无视觉方法（TiDE）提升 12%[^src-pipe]。消融实验显示：视觉数据贡献约 8% 改善，物理知情编码额外贡献约 6%[^src-pipe]。PIPE 代表了多模态融合的一条独特路线：不通过额外的融合模块或适配器，仅通过**位置编码层的物理知识注入**实现跨模态对齐，训练成本极低（PIPE-3B 仅需 2.1 小时 4×H800）[^src-pipe]。

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
- [[cvpe]] — 跨变量 Patch Embedding (CI+CD 折中策略)
- [[allspark]] — AllSpark 10 模态时空通用智能模型
- [[language-as-reference-framework]] — LaRF 以语言为参考框架的多模态统一原理

- [[tess]] — TESS 实体：离散原语瓶颈的 Non-Fusion 实现（arXiv:2603.12664v2）
- [[temporal-semantic-primitives]] — 时间演化原语技术页
- [[e2-cstp]] — E²-CSTP 因果多模态时空预测框架
- [[streasoner]] — STReasoner 时空推理 TS-LM
- [[spatio-temporal-reasoning]] — 时空推理概念
- [[pipe]] — PIPE 物理知情位置编码台风预测模型 (NeurIPS 2025)
- [[physics-informed-position-encoding]] — 物理知情位置编码技术
- [[variant-frequency-positional-encoding]] — 变频率正弦编码
- [[digital-typhoon-dataset]] — Digital Typhoon 台风卫星数据集
- [[time-mmd]] — Time-MMD 多领域数值–文本时序数据集与 MM-TSFlib（NeurIPS 2024 D&B）
- [[time-vlm]] — Time-VLM：冻结 VLM 桥接时序/视觉/文本 + RAL/VAL/TAL（ICML 2025）
- [[source-time-vlm]] — Time-VLM 源摘要
- [[ts-vl-alignment]] — 时序–视觉–语言对比对齐极限诊断（arXiv:2602.19367）
- [[source-ts-vl-alignment]] — TS–VL Alignment 源摘要
- [[constrained-text-fusion]] — Constrained Text Fusion / CFA：naive 常伤、低秩受控融合（KDD ’26 MILETS）
- [[source-constrained-text-fusion]] — CFA 源摘要
- [[cross-modal-misalignment]] — 跨模态 selection/perturbation bias 与 MMCL 可辨识性（NeurIPS 2025）
- [[source-cross-modal-misalignment]] — 源摘要

---

## 引用

[^src-unca]: [[source-unca]]
[^src-channelmts]: [[source-channelmts]]
[^src-most]: [[source-most]]
[^src-aurora]: [[source-aurora]]
[^src-language-in-the-flow-of-time]: [[source-language-in-the-flow-of-time]]
[^src-cvpe-2025]: [[source-cvpe-2025]]
[^src-allspark]: [[source-allspark]]
[^src-streasoner]: [[source-streasoner]]
[^src-mtp]: [[source-mtp]]
[^src-st-vision-llm]: [[source-st-vision-llm]]
[^src-pipe]: [[source-pipe]]
[^src-timi]: [[source-timi]]
[^src-time-mmd]: [[source-time-mmd]]
[^src-time-vlm]: [[source-time-vlm]]
[^src-ts-vl-alignment]: [[source-ts-vl-alignment]]
[^src-constrained-text-fusion]: [[source-constrained-text-fusion]]
[^src-cross-modal-misalignment]: [[source-cross-modal-misalignment]]
[^src-tess]: [[source-tess]]
