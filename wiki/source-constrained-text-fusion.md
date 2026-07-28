---
title: "Rethinking Multimodal Fusion for Time Series: Text Modalities Need Constrained Fusion"
type: source-summary
tags:
  - multimodal-time-series
  - text-fusion
  - constrained-fusion
  - time-mmd
  - plug-in-adapter
  - kdd-2026
created: 2026-07-28
last_updated: 2026-07-28
source_count: 1
confidence: high
status: active
---

# Constrained Text Fusion 源文件摘要

**来源**: Seunghan Lee, Jun Seo, Jaehoon Lee, Sungdong Yoo, Minjae Kim, Tae Yoon Lim, Dongwan Kang, Hwanil Choi, SoonYoung Lee, Wonbin Ahn (LG AI Research). *Rethinking Multimodal Fusion for Time Series Forecasting: Text Modalities Need Constrained Fusion.* KDD ’26 MILETS Workshop, arXiv:2603.22372v3 (15 Jul 2026). raw: `raw/rethinking-multimodal-fusion-for-time-series-text-modalities-need-constrained-fusion.pdf`. 代码：`https://github.com/seunghan96/cfa`[^src-constrained-text-fusion]

## 核心论点

多模态时序预测常把文本当**辅助模态**，却默认用 **naive fusion**（简单 **add / concat**，在 first / middle / last 层注入）。作者在 **Time-MMD** 九域上做超大规模对照后主张：**无约束的文本注入经常低于单模态 TS 基线**，因为辅助文本可含与时序动力学无关或冲突的信息，无控融合会破坏时间表示。**Constrained fusion**（门控、FiLM、正交注入、以及提出的 **CFA**）系统优于 naive；其中 **Controlled Fusion Adapter (CFA)** 用 **LoRA 式低秩瓶颈残差** 过滤无关文本，且可 plug-in 进任意 TS 骨干而不改架构[^src-constrained-text-fusion]。

## 问题设定与 Naive / Constrained

- 输入 lookback \(X\) 与对齐文本序列 \(T\)；文本编码器冻结，仅训 TS 侧与融合模块。  
- **Naive**：\(F(Z_{\mathrm{TS}}, Z_{\mathrm{Text}})\) 为 add 或 concat，位置 first / middle / last 互斥（Algorithm 1）。代表：Time-MMD 式 last add、TaTS 式 first add、若干 concat 即插即用。  
- **Constrained**（Table 2）：  
  - **Gating**：\(z_{\mathrm{TS},t} + g_t \odot z_{\mathrm{Text},t}\)  
  - **FiLM**（特征调制，≠ TS 模型 FiLM）：\(\gamma_t \odot z_{\mathrm{TS},t} + \beta_t\)  
  - **Orthogonal**：\(z_{\mathrm{TS},t} + z_{\mathrm{Text},t}^\perp\)（只注入正交分量）  
  - **CFA**：\(z_{\mathrm{TS},t} + W_{\mathrm{up}}\phi(W_{\mathrm{down}} z_{\mathrm{Text},t})\)，瓶颈 \(D/r\)（默认 \(r=8\)），\(\phi=\) ReLU∘LN；\(W_{\mathrm{up}}\) 近零初始化；残差加在各 encoder 层[^src-constrained-text-fusion]。

Table 1 定位：既有工作要么 **architecture-specific**，要么 **plug-in 但 naive**；**CFA 同时是 plug-in + constrained**[^src-constrained-text-fusion]。

## 实验规模（~20K 配置）

| 轴 | 内容 |
|----|------|
| 数据 | [[time-mmd|Time-MMD]] 9 域（Agriculture…Traffic） |
| TS 骨干 | 14：Transformer 系（Nonstat. Trans.、PatchTST、iTransformer、Cross/FED/Autoformer、Reformer、Informer、Transformer）+ DLinear / TiDE / TSMixer + Koopa / FiLM |
| 文本 | 冻结 BERT、GPT-2、Llama3、Doc2Vec |
| 视界 | 日 {48,96,192,336}；周 {12,24,36,48}；月 {6,8,10,12} |
| 融合 | 10 种（6 naive 位置×算子 + 4 constrained） |
| 调参 | 每设定 10 个 LR 取最优；split 7:1:2；MSE/MAE |

文中亦称 “Average over **2K** settings” 的图级聚合（14×4×9×4 量级）与 “over **20K** experiments”（含融合/LR 全展开）[^src-constrained-text-fusion]。

## 主要实证

1. **Naive 常伤于单模态**：Figure 1 / Table 4 中 add 与 concat 大量蓝（劣化）甚至 **Div.**（MSE > 单模态 10×）；concat-last 等尤其不稳。  
2. **Constrained 更稳、CFA 最稳**：相对 unimodal 的 win rate 上，CFA 在多骨干块常达 **88.9–100%**（如 Nonstat. Trans. / PatchTST / DLinear / TiDE / FiLM 行），而 naive first/last 可低至 0–44%。  
3. **跨设定汇总**：CFA 在 **9 域全部优于 unimodal**，**7/9 域 rank-1**；**14 骨干中 13 个** 优于 unimodal（例外标准 Transformer，其单模态本身很弱）。Constrained 整体压过 naive-add / naive-concat。  
4. **低秩瓶颈必要**：toy 合成（matching / contradicting / irrelevant 文本）：有 bottleneck 相对无瓶颈 MSE 改善约 **+12.2% / +4.6% / +20.0%**；matching 的 text-contribution ratio 显著高于 contradicting（\(t=4.03\)，Cohen’s \(d=0.58\)）。  
5. **机制**：更高 **effective rank** 与更低 MAE 正相关（例 \(\rho=0.6727\)）；CFA 改变输入时间步的 gradient×input 归因分布。  
6. **效率**：相对 unimodal，CFA 参数约 **+0.61%**、FLOPs 约 **+0.04%**（Gating 可 +28% 参数 / +31% FLOPs）[^src-constrained-text-fusion]。

## 与相关工作的对照（文内）

- **TaTS**：可 plug-in，但属 **first-layer additive naive**，本文作基线。  
- **Time-MMD / MM-TSFlib**：数据与 naive 融合实验床；本文在其 9 域上系统否定“无控融合默认有益”。  
- **Time-VLM 等 architecture-specific**：附录相关工作归为特定骨干设计，非通用 constrained plug-in。  
- 局限：仅文本辅助；理论仍偏经验；未来扩到 vision / tabular[^src-constrained-text-fusion]。

实体与方法归纳见 [[constrained-text-fusion]]。

## 相关页面

- [[constrained-text-fusion]] — 实体 / 方法  
- [[time-mmd]] · [[multimodal-time-series-forecasting]] · [[non-fusion-guidance]] · [[timi]] · [[vot]] · [[time-vlm]] · [[ts-vl-alignment]] · [[tats]]

[^src-constrained-text-fusion]: [[source-constrained-text-fusion]]
