---
title: "TADiff: Counterfactual Time Series Forecasting with Textual Conditions"
type: source-summary
tags:
  - time-series-forecasting
  - counterfactual
  - diffusion-model
  - ddim
  - multimodal-time-series
  - contrastive-learning
created: 2026-11-24
last_updated: 2026-11-27
source_count: 1
confidence: medium
status: active
---

# TADiff: What if Tomorrow is the World Cup Final? Counterfactual Time Series Forecasting with Textual Conditions

预测不只由历史走势决定：「若明天是世界杯决赛」这类假设无法写成固定类别干预变量、只能以自由文本表达，反事实未来又无真值可监督或评分[^src-tadiff]。TADiff 的巧思在初始噪声：把历史序列沿扩散前向链逆推成剥离了文本条件影响的固有演化隐状态作去噪起点，再由目标文本引导去噪生成未来[^src-tadiff]。对无真值的反事实样本，另训对比评分器 DTTC 评分预测的语义一致性，直接以其分数微调模型[^src-tadiff]。论文报告其在合成与多数真实集上全面领先，反事实设定下优势最明显[^src-tadiff]。

## 核心做法

基座为去噪扩散隐式模型（DDIM）条件去噪。文本归因（Text-Attribution, TA）：训练时在预测噪声估计损失外加一路以历史文本为条件的归因损失，两路共享去噪网络；推理先沿扩散前向链逆推，把观测历史剥离成与外生条件无关的固有特征隐态（论文证明其与历史文本条件独立），再以其为初始噪声态、回填真实历史前缀，由目标文本引导去噪出未来[^src-tadiff]。解耦时序-文本一致性（Disentangled Time series and Text Consistency, DTTC）：序列编码器把序列拆成固有/外生两路特征，与文本编码器做对比学习，DTTC-I、DTTC-E 分别度量预测守住历史固有特征、兑现目标语义，支持无真值评估[^src-tadiff]。反事实微调（counterfactual finetuning, CF）：真实数据一条历史只配一个确定未来，按文本相似度给每条历史挑一条不冲突的候选未来作无真值反事实样本，直接最大化 DTTC 分数微调，论文报告不损害事实精度[^src-tadiff]。

## 证据

1. **定位与主结论**：多数数据集上数值与语义双领先，反事实设定优势最明显；正文称 Weather 上 MSE 最佳，与主表另一基线更低的排名定性不符[^src-tadiff]。
2. **真实涨点归因**：拿掉 TA 掉点最狠（Traffic 事实精度大幅回退），拿掉 CF 该集事实数值反而略好——增益主要来自归因初始化；消融仅这两个开关[^src-tadiff]。
3. **证据缺口与负结果**：反事实真值仅合成集有，真实集评估靠启发式构造的无真值样本；DTTC 既是评测指标又是微调目标，评测与训练同源[^src-tadiff]。

## 局限

论文自述变长预测的模型化指标评估仍是局限[^src-tadiff]。

## 相关页面

- [[diffusion-model]] — 同构/特例：继承 DDIM 条件去噪，特例化在初始噪声从随机高斯换成文本归因隐态、推理多走一条条件剥离前向链。
- [[doflow]] — 同槽位互斥：同为反事实时序预测生成模型；DoFlow 依赖显式因果 DAG 与流模型溯因输出单轨迹，TADiff 不建 SCM、以自由文本为条件。
- [[causal-counterfactual-recovery]] — 同槽位互斥：都把「外生条件从序列中剥离」当核心一步；前者研究剥离何时可识别，TADiff 用扩散逆变换启发式剥离并证明隐态与历史条件独立。
- [[source-whatif-tsf]] / [[time-mmd]] — 跨槽位正交：WIT 以专家手工确定性未来事件做条件基准，TimeMMD 用历史对齐的真实文本，TADiff 面向未发生的自由文本反事实假设，三者互斥于文本条件的不同子槽位且可互补。

> 来源：Shuqi Gu, Yongxiang Zhao, Baoyu Jing, Kan Ren, "What if Tomorrow is the World Cup Final? Counterfactual Time Series Forecasting with Textual Conditions", ICML 2026 (PMLR 306), ShanghaiTech University / Meta Platform Inc.

[^src-tadiff]: [[source-tadiff]]
