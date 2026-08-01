---
title: "From Text to Forecasts: Bridging Modality Gap with Temporal Evolution Semantic Space"
type: source-summary
tags:
  - time-series
  - multimodal
  - llm
  - forecasting
  - non-fusion-guidance
  - iclr-2026
created: 2026-08-01
last_updated: 2026-08-01
source_count: 0
confidence: low
status: active
---

# From Text to Forecasts: Bridging Modality Gap with Temporal Evolution Semantic Space

**Authors**: Lehui Li, Yuyao Wang, Jisheng Yan, Wei Zhang, Jinliang Deng, Haoliang Sun, Zhongyi Han, Yongshun Gong（山东大学、波士顿大学、北航、南科大）。**Venue**: 标注 ICLR 2026。arXiv:2603.12664v2 [cs.CL]。

## 核心论点

文本与数值序列之间存在模态鸿沟：文本对事件影响的描述是隐式、定性、时间上弱锚定的，而预测模型需要显式、定量的信号。论文用可控实验定位两个具体瓶颈，再以中间语义空间绕过它们。

## 两个诊断（半合成实验）

用 FNSPID 真实序列 + GPT-5.2 按未来窗口统计特征生成时间对齐文本，自动标注信号 token（$T_{sig}$）与冗余 token（$T_{red}$）：

1. **注意力分散**：焦点比 $R_t=\log(\bar\alpha_{sig}/\bar\alpha_{red})$ 显示，即使文本带来正增益的样本，多数仍 $R_t<0$——模型系统性过度关注冗余 token。
2. **表征不匹配**：Full / Signal-Only / Numerical 三变体对比，Signal-Only 优于 Full 但仍显著劣于 Numerical——删光冗余后，文本语义仍难解码为数值信号。

## 方法：TESS

两阶段框架：

1. **文本 → 语义空间**：冻结 LLM 按结构化 prompt 将文本分类为四类离散 [[temporal-semantic-primitives|时间演化原语]]（mean shift / volatility / shape / lag-decay），每类小候选集；以 top-1/top-2 log 概率 margin 作不确定度信号，经可学习门控 $g_{t,k}=\sigma(w_k^\top[h_{t,k};W_m m_{t,k}]+b_k)$ 过滤。门控用 BCE 训练，标签 $y_{t,k}=\mathbb{1}[\hat v_{t,k}=\psi_k(Y_t)]$ 由原语数值可验证性自动获得（无需人工标注）。
2. **语义空间 → 数值空间**：门控后的原语 embedding 作 prefix token 与 PatchTST patch embedding 拼接（$Z^{(0)}=[P;E_{patch}]$），全程参与自注意力条件化预测。

理论支撑：定理 4.1（语义充分性下信息瓶颈不损预测互信息、降文本依赖、泛化不劣化）；定理 A.5（原语错误的影响按 $g_{t,k}^2$ 衰减）；定理 A.6（复杂度 $\sqrt{M/n}$ vs token 级 $\sqrt{\log|A_T|/n}$）。

## 实验

四个数据集（Bitcoin、FNSPID、Electricity、Environment）。相对最强基线：Bitcoin（vs NewsForecasting）MAE/MSE/RMSE +18.2%/+29.1%/+15.8%；FNSPID（vs TimesNet）+3.3%/+20.0%/+9.9%；Electricity 全指标最优（+0.4%~5.0%）；Environment 次优（<1% 差距）。非平稳子集（shape/volatility/mean shift）上 MSE 较多模态基线降 21–52%。

消融：去 TESS 使 MSE 升 46.2%/29.4%/22.8%，去 gating 仅升 3.7%/2.6%/7.5%；单独移除 mean shift 原语 +33% MSE；gating 对正确提取样本给 0.65–0.78 中位权重、错误样本 0.21–0.40。

## 局限

- 无独立 Limitations 章节；代码接收后发布（未开源）；提取所用冻结 LLM 型号未披露。
- 正文消融百分比（46.2% 等）与 Table 2 数值无法完全对上，引用以表内数字为准。
- 附录 Table 3 中 FNSPID 粒度栏印 "5 days"，与表题 "daily stock price data" 矛盾（原文即如此）。

## 范式定位

归入 [[non-fusion-guidance|Non-Fusion Guidance]] 范式（[[timi|TiMi]] 后第二成员）：LLM 独立处理文本、输出离散原语而非特征 embedding、数值预测完全由 PatchTST 完成，不做表示级对齐。
