---
title: "HORAI: Empowering Time Series Analysis with Large-Scale Multimodal Pretraining"
type: source-summary
tags:
  - time-series-forecasting
  - multimodal-time-series
  - foundation-model
  - frequency-domain
  - mixture-of-experts
  - zero-shot
created: 2026-11-24
last_updated: 2026-11-24
source_count: 1
confidence: medium
status: active
---

# HORAI: Empowering Time Series Analysis with Large-Scale Multimodal Pretraining

**来源**：Peng Chen、Siyuan Wang、Shiyan Hu、Xingjian Wu、Yang Shu、Zhongwen Rao、Meng Wang、Yijie Li、Bin Yang、Chenjuan Guo（华东师范大学 + 华为）；核心切片未带会议水印、仅引文含 2026 年条目（本仓库 OpenReview CCF-A 调研记录其录用于 NeurIPS 2026 Poster）[^src-horai]。

多模态时间序列预训练常面临两难：直接拼接文本与图像容易混入噪声，而用同一套机制对齐全频段又忽视了平滑走势与局部波动的表达差异。HORAI 在涵盖六个领域、规模超过十亿时间点的 MM-TS 多模态语料上预训练，核心巧思是把频带当作跨模态对齐接口：利用傅里叶变换将历史序列拆分为对应趋势与宏观叙事的低频分支、以及对应局部突变形态的中高频分支，分别与文本和折线图做注意力对齐，经门控融合后再送入时频混合专家网络自回归解码预测。需要严格界定模态真实性边界：预训练中的折线图由数值历史直接渲染而成，内生文本由 GPT-4o 读取同一数值窗口生成，两者均属输入重编码；仅过滤后的 GDELT 新闻以及下游的 Time-MMD 与 FNSPID 数据集才提供真实外部文本[^src-horai]。

## 去黑话算子配方与路线差异

模型未引入改变数学复杂度的新底层算子，核心由频带切分、既有跨模态注意力、门控融合与时频专家解码器组合而成[^src-horai]：

- **无参数频带分离**：对历史序列 $X$ 做 `FFT` 变换，按截止阈值 $\tau=\alpha(L/2+1)$（默认 $\alpha=0.05$）切出低频分量 $X_{\text{low}}$ 与中高频分量 $X_{\text{mh}}$，再经 `iFFT` 还原回时域[^src-horai]。
- **跨模态对齐与门控融合**：在 `Flow-Attention` 中以低频表示作为 query 对齐 `Qwen-0.5B` 文本嵌入 $E_{\text{text}}$，以中高频表示作为 query 对齐 `ViT-Base` 折线图嵌入 $E_{\text{img}}$；随后计算门控 $G=\sigma(W_g[E'_{\text{img}}, E'_{\text{text}}])$，以 $E_{\text{fused}}=G\odot E'_{\text{img}}+(1-G)\odot E'_{\text{text}}+E_{\text{ts}}$ 与原始时序表征相加融合[^src-horai]。
- **时频混合专家自回归解码**：因果 Transformer 自回归预测下一 patch，并将 FFN 替换为时频路由混合专家——对隐藏状态 $H$ 及其 `FFT` 变换结果各自通过 MLP 后按门控混合 $H_r = g\odot \text{MLP}(H) + (1-g)\odot \text{MLP}(\text{FFT}(H))$，再执行 Top-$K$ 专家加权输出[^src-horai]。
- **有向关系对照**：相对 [[aurora|Aurora]]（**同槽位互斥 + 实证复现对齐**），两者同为十亿级多模态预训练模型，Aurora 采用语义质心蒸馏与原型引导 Flow Matching 概率生成，HORAI 改为固定频带分工对齐与确定性自回归点预测，失去概率分布输出且需调 $\alpha$ 与 $K$，Time-MMD 九域零样本中 Environment 与 Traffic 仍逊于 Aurora；相对 [[time-vlm|Time-VLM]]（**瓶颈接力 $\Delta$**），门控融合形式相似，但从冻结 VLM 检索记忆转向轻编码器全量预训练与频带对齐；相对 [[frequency-pathway-routing|Frequency Pathway Routing]] 与 [[mixture-of-experts|Mixture of Experts]]（**架构特例对照**），输入端不做 [[adafre|AdaFre]] 式稀疏选带而采固定二值配对，仅在解码器 FFN 内部将时频门控接入 Top-$K$ MoE；并与 [[time-mmd|Time-MMD]] 构成**下游评测域对照**[^src-horai]。

## 证据

1. **定位与主结论**：Time-MMD 九域零样本取得 5 域最优（Agriculture 逊于 Chronos-2、Environment 与 Traffic 逊于 Aurora、Social Good 逊于 TimesFM），5 集异常检测 15 项指标 12 项第一[^src-horai]。
2. **真实涨点归因**：Band–Modality 互换消融（中高频配文本、低频配折线图）在 Social Good 上从 $0.819$ 恶化至 $1.237$（差于删图文模态的 $1.058$），证实核心涨点来自「低频配文本 + 中高频配折线图」的配对先验与大语料预训练，时频 MoE 为次级增益；随机文本替换劣于删文本，表明文本支确被读取，但 GPT-4o 重编码与外部真实语义增益尚未彻底分离[^src-horai]。
3. **证据缺口与负结果**：仅测 0.5B 文本与 Base 视觉小编码器，缺更大规模骨干与主文系统效率对照；预训练图像与内生文本属输入重编码；自述局限仅置于附录而未在主文展开[^src-horai]。

## 相关页面

- [[aurora]] — 同槽位互斥 + 实证复现对齐（流匹配概率生成 vs 自回归频带点预测）
- [[time-vlm]] — 瓶颈接力 $\Delta$（冻结 VLM 检索记忆 vs 轻编码器全量预训练）
- [[time-mmd]] — 下游九域外部文本零样本与互换消融评测基准
- [[multimodal-time-series-forecasting]] — 多模态基础模型与模态真实性边界二维矩阵
- [[frequency-pathway-routing]] — 静态频带配对与解码端时频门控架构特例对照
- [[mixture-of-experts]] — 解码器时频门控前置 Top-$K$ MoE 架构特例对照

[^src-horai]: [[source-horai]]
