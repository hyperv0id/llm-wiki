---
title: "TRACE"
type: entity
tags:
  - time-series
  - multimodal
  - retrieval
  - cross-modal
  - contrastive-learning
  - channel-biased-attention
  - neurips-2025
created: 2026-08-19
last_updated: 2026-08-19
source_count: 1
confidence: medium
status: active
---

# TRACE

TRACE（Time-series Retriever with Aligned Context Embedding）是首个多模态时序检索器，由 Chen et al.（Yale / McGill / UTRGV）在 NeurIPS 2025 提出[^src-trace-neurips2025]。TRACE 将时间序列嵌入与对齐的文本上下文锚定到共享语义空间，支持 Text→TS、TS→Text 和 TS→TS 跨模态检索，同时作为独立编码器服务下游预测与分类。

## 架构

### Stage 1：时序编码器预训练

Encoder-only Transformer，核心组件：

- **[[channel-identity-token|Channel Identity Tokens (CITs)]]**：每通道一个唯一可学习 token，置于该通道 patch 序列首位，充当通道级摘要锚点。CITs 引导模型关注各通道独特行为，学习通道解耦表示，弥补传统 decoder-only 基础模型嵌入缺乏判别力的问题[^src-trace-neurips2025]。
- **[[channel-biased-attention|Channel-biased Attention (CbA)]]**：通过偏置注意力掩码 M 防止异质通道间语义纠缠——CIT 仅关注本通道 token，非 CIT token 可自由关注全局序列[^src-trace-neurips2025]。
- **RoPE**：在每通道内的 T̂ 个时间 token 上独立施加旋转位置编码，不施加于 CIT（位置无关聚合器）。使用原始时间差 Δt_ij 而非展平序号，确保展平多通道序列后位置编码与真实时间结构一致[^src-trace-neurips2025]。
- **[[patch-based-tokenization|Patch tokenization]]**：非重叠 patch 切分 + 线性投影 + RevIN 预处理 + 随机掩码重建（MSE loss）[^src-trace-neurips2025]。

Token 序列结构：`H = [CLS]; [CIT]₁; X₁^patch; [CIT]₂; X₂^patch; ...; [CIT]_C; X_C^patch`，总长度 L = C(T̂+1)+1[^src-trace-neurips2025]。

### Stage 2：跨模态对齐

冻结 Stage 1 编码器，用冻结 Sentence-Transformer（默认 nomic）编码文本，通过 [[dual-level-hard-negative-mining|双级硬负采样]] + 双向 InfoNCE 进行对齐：

- **Sample-level**：[CLS] 嵌入 h_CLS ↔ 样本级上下文文本嵌入 z_cxt。硬负样本为 batch 内 TopK 相似但非配对样本[^src-trace-neurips2025]。
- **Channel-level**：CIT 嵌入 h_c ↔ 通道级文本嵌入 z_c。硬负样本包括 intra-instance（同实例其他通道）和 inter-instance（不同实例同索引通道）两类 distractor[^src-trace-neurips2025]。
- 交叉注意力模块在时序与文本嵌入间做最终融合[^src-trace-neurips2025]。
- 总损失 L_align = (L_global + λ_ch · L_channel) / 2，λ_ch=1.0[^src-trace-neurips2025]。

### RAG 框架

检索 top-R 多模态对 → 各经线性投影压缩为 soft prompt P → 前置于冻结 TSFM 输入。支持 decoder-only（Time-MoE / Timer-XL，P 追加到自回归上下文）和 encoder-only（Moment / TRACE，P 作为 encoder 前缀）[^src-trace-neurips2025]。仅 Proj + Head 可训练。

## 数据集

- 自建天气多模态数据集（扩展 MTBench [6]）：NOAA Storm Events 事件报告 + GHCN-h 气象时序，74,337 实例，7 通道，3 种时间分辨率（7 日小时 / 28 日 4 小时 / 180 日日）[^src-trace-neurips2025]。
- [[time-mmd|Time-MMD]] 公开数据集（Health H=12 / Energy H=12 / Environment H=48）[^src-trace-neurips2025]。

## 实验结果

### 预测

| 数据集 | TRACE | 最佳基线 | 指标 |
|--------|-------|---------|------|
| Weather (M, H=7) | MSE 0.576 | 0.581 (TimesNet) | MSE |
| Health (U, H=12) | MSE 0.547 | 0.656 (PatchTST) | MSE |
| Environment (U, H=48) | MSE 0.389 | 0.398 (TimesNet) | MSE |
| Energy (U, H=48) | MSE 0.455 | 0.462 (iTransformer) | MSE |

TRACE 在 11 个预测设置中取得最优，长程预测优势尤为显著[^src-trace-neurips2025]。

### 分类

9 类天气事件分类：TRACE w/ RAG 89.76% Acc / 72.36% F1，无 RAG 85.20% / 69.98%[^src-trace-neurips2025]。基线 TSFM 微调后表现弱于从头训练模型（如 FSCA 85.62%），TRACE 兼具判别结构与语义对齐。

### RAG

分类准确率 +4.56%，预测误差 −4.55%（论文自述）。decoder-only TSFM（Time-MoE / Timer-XL）对检索模态丰富度更敏感，TS+Text > TS-only；encoder-only（Moment / TRACE）更稳健[^src-trace-neurips2025]。

### 效率

| 模型 | 总参数 | 激活参数 | 训练时间(s/epoch) |
|------|--------|---------|-------------------|
| TRACE | 10.78M | 0.12M | 6.054 |
| FSCA | 82.35M | 22.68M | 1249.701 |
| Moment_base | 109.87M | 0.24M | 11.706 |
| Time-MoE_small | 113.49M | 113.49M | 106.308 |

TRACE 微调激活参数 ~200× 少于 FSCA、~700× 少于 Time-MoE_small[^src-trace-neurips2025]。

## 消融

- **CIT 移除**：Avg MSE 0.670→0.713，Acc 85.20→85.04[^src-trace-neurips2025]。
- **CbA → Full Attention**：MSE 0.713，Acc 84.18[^src-trace-neurips2025]。
- **CbA → Causal Attention**：MSE 0.705，Acc 83.72[^src-trace-neurips2025]。
- **Cross-attention 移除**：检索精度在小 K 时显著下降[^src-trace-neurips2025]。
- **Channel-level 对齐移除**：检索精度一致下降[^src-trace-neurips2025]。
- **文本编码器**：nomic > bge > MiniLM[^src-trace-neurips2025]。
- **模型规模**：d=384 vs 768 无显著差异，架构设计和 PE 选择比参数缩放更重要[^src-trace-neurips2025]。

## 局限性

依赖训练时可用对齐 TS–文本对；channel-level 对齐增加训练开销[^src-trace-neurips2025]。

## 与其他方法的关系

- 与 [[rast|RAST]] / [[ratd|RATD]] / [[gtr|GTR]] / [[pir|PIR]] 同属检索增强预测，但 TRACE 是唯一做**跨模态**检索的方法——此前方法仅用单模态时序嵌入[^src-trace-neurips2025]。
- 与 [[time-llm|Time-LLM]] / [[ts-vl-alignment|TS–VL Alignment]] / [[constrained-text-fusion|CFA]] 同属多模态时序对齐，但 TRACE 引入通道级细粒度对齐和双级硬负采样，此前方法仅做全局文本对齐[^src-trace-neurips2025]。
- 与 [[time-vlm|Time-VLM]] 不同：Time-VLM 用冻结 VLM 桥接三模态（内生自增强），TRACE 用文本作为外部语义锚定（外生对齐）[^src-trace-neurips2025]。
- 与 [[mae|MAE]] / [[patchtst|PatchTST]] 的关系：共享掩码重建和 patch+CI 设计，但 TRACE 用 CIT + CbA 弥补纯 CI 的通道信息缺失[^src-trace-neurips2025]。
- 与 [[roformer|RoFormer]] 的关系：TRACE 在通道内独立施加 RoPE，用真实时间差而非展平序号[^src-trace-neurips2025]。

## 相关页面

- [[source-trace-neurips2025]] — 源摘要
- [[channel-identity-token]] — Channel Identity Tokens 技术
- [[channel-biased-attention]] — Channel-biased Attention 技术
- [[dual-level-hard-negative-mining]] — 双级硬负采样技术
- [[retrieval-augmented-spatio-temporal-forecasting]] — RAG-for-STF 范式
- [[multimodal-time-series-forecasting]] — 多模态时序预测
- [[contrastive-learning]] — 对比学习
- [[patch-based-tokenization]] — Patch tokenization
- [[channel-independence]] — 通道独立
- [[time-mmd]] — Time-MMD 数据集
- [[gtr]] / [[rast]] / [[ratd]] / [[pir]] — 其他检索增强预测方法

[^src-trace-neurips2025]: [[source-trace-neurips2025]]
