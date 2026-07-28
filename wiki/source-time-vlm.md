---
title: "Time-VLM: Exploring Multimodal Vision-Language Models for Augmented Time Series Forecasting"
type: source-summary
tags:
  - multimodal-time-series
  - vision-language-model
  - retrieval-augmented
  - time-series-forecasting
  - icml-2025
created: 2026-07-28
last_updated: 2026-07-28
source_count: 1
confidence: high
status: active
---

# Time-VLM 源文件摘要

**来源**: Siru Zhong, Weilin Ruan, Ming Jin, Huan Li, Qingsong Wen, Yuxuan Liang. *Time-VLM: Exploring Multimodal Vision-Language Models for Augmented Time Series Forecasting.* ICML 2025 (PMLR 267). arXiv:2502.04395v2 (26 May 2025). HKUST (Guangzhou) / Griffith / Zhejiang University / Squirrel Ai Learning. Code: `CityMind-Lab/ICML25-TimeVLM`. raw: `raw/time-vlm-exploring-multimodal-vision-language-models-for-augmented-time-series-forecasting.pdf`[^src-time-vlm]

## 核心论点

文本增强（Time-LLM、GPT4TS 等）有语义但缺细粒度时序；视觉增强（TimesNet、VisionTS 等）能保留连续时空结构却缺语义。两者与数值时序的**三模态统一**此前未系统打通。Time-VLM 主张：用**冻结预训练 VLM** 作为桥，把时序自生成的图像与文本投到同一视觉–语言语义空间，再与检索增强的时序记忆融合，尤其在 few-shot / zero-shot 下借预训练先验补数据稀缺[^src-time-vlm]。

## 三组件：RAL · VAL · TAL

1. **Retrieval-Augmented Learner (RAL)**：输入 patch 化（默认 pl=16, stride=8）+ 位置编码；环形 **memory bank** 存历史 patch。**Local memory** 按余弦相似度 top-k 检索历史 patch 经 MLP 残差回写；**Global memory** 对当前 patch 做 multi-head self-attention 再时序平均；\(M_{\mathrm{fused}}=M_{\mathrm{local}}+M_{\mathrm{global}}\)[^src-time-vlm]。
2. **Vision-Augmented Learner (VAL)**：FFT 频率编码与 \(\sin/\cos\) 周期编码拼到输入，经 1D/2D 多尺度卷积得到三通道图，双线性插值到固定分辨率（默认 64×64）并 min-max 归一到 \([0,255]\)，送入 **冻结 VLM 视觉编码器**[^src-time-vlm]。
3. **Text-Augmented Learner (TAL)**：由统计量（min/max/median/趋势）、任务窗长/视界、域描述与图像说明组成结构化 prompt（可叠专家预定义文本），经 **冻结 VLM 文本编码器**[^src-time-vlm]。

**融合**：RAL 时序记忆作 query、VLM 多模态嵌入作 key/value 的 cross-modal multi-head attention，再 **gated fusion** 动态权衡时序与多模态；仅训 RAL/VAL/预测头，VLM 冻结；默认骨干 **ViLT-b32-finetuned-coco**，亦支持 CLIP / BLIP-2；约 **143.6M** 参数（相对 Time-LLM ~3405M 约 1/20）[^src-time-vlm]。

## 实验要点

评测 ETT×4、Weather、ECL、Traffic 长程 + M4 短程。**5% few-shot**：ETTh1 上 MSE **0.442** vs Time-LLM **0.627**（约 −29.5% MSE / −16.6% MAE）；Weather 相对 Time-LLM MSE −7.7%。**零样本** ETT 跨子集迁移与 Time-LLM 互有胜负、整体紧贴。**M4** 加权平均 SMAPE/MASE/OWA 优于 Time-LLM 等。**全量长程** 多数集有竞争力，ECL/Traffic 等仍可落后专用单模态。Weather 消融：去 RAL **+35.6%** MSE，去 VAL **+9.0%**，去 TAL 仅 **+2.1%**（ViLT 文本 token 稀疏）。独立 ViT+BERT 模块组合弱于预训练 VLM 对齐空间；UMAP 显示 COCO 图文对与时序图文对有交叉，纯图/纯文簇分离——支撑“投到多模态空间而非单模态投影”的设计[^src-time-vlm]。

## 局限与定位

自述：TAL 受现有 VLM 时序语义理解与短文本限制，增益小；全量设定上部分高维集弱于专用单模态；波动/非平稳突变场景视觉变换仍弱。**不依赖外生新闻/卫星**，仅由原始时序**自增强**视觉与文本——相对 [[time-mmd|Time-MMD]] 外生文本路线、[[vot|VoT]] 事件推理路线，Time-VLM 是 **VLM 桥接 + 检索记忆** 的内生多模态范式；谱系细节见 [[time-vlm]][^src-time-vlm]。

## 相关页面

- [[time-vlm]] — 实体与机制
- [[multimodal-time-series-forecasting]] · [[source-raf]] · [[source-solar-vlm]] · [[st-vision-llm]] · [[vot]] · [[time-mmd]] · [[source-time-llm]]

[^src-time-vlm]: [[source-time-vlm]]
