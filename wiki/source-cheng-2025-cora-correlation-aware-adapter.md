---
title: "CoRA: Boosting TSFMs for Multivariate Forecasting through Correlation-aware Adapter"
type: source-summary
tags:
  - time-series-foundation-model
  - multivariate-forecasting
  - channel-correlation
  - plug-and-play-adapter
  - iclr-2026
created: 2026-07-28
last_updated: 2026-07-28
source_count: 1
confidence: high
status: active
---

# CoRA (Correlation-aware Adapter) 源文件摘要

> **命名消歧**：本页对应 Cheng et al. (ECNU / Huawei Noah's Ark, ICLR 2026) 的 **CoRrelation-aware Adapter**——为 TSFM 下游微调补通道相关。仓库中既有 [[source-cora|source-cora]] / [[cora-tsfm]] 是另一篇 **Covariate-awaRe Adaptation**（协变量注入），二者同缩写、不同问题。

**来源**: Hanyin Cheng, Xingjian Wu, Yang Shu, Zhongwen Rao, Lujia Pan, Bin Yang, Chenjuan Guo. *CoRA: Boosting Time Series Foundation Models for Multivariate Forecasting through Correlation-aware Adapter.* ICLR 2026. Code: https://github.com/decisionintelligence/CoRA[^src-cheng-2025-cora-correlation-aware-adapter].

## 核心论点

### 1. 问题：TSFM 多为通道独立，相关建模不完整

多数 TSFM（TimesFM、Timer、Moment、GPT4TS 等）走 [[channel-independence|通道独立]]，强在时序泛化、弱在通道交互。即便 TTM / UniTS / Moirai 做了通道混合或注意力，也未同时覆盖三类相关[^src-cheng-2025-cora-correlation-aware-adapter]：

| 类型 | 含义 |
|------|------|
| **DCorr**（动态相关） | 通道关系随时间段变化 |
| **HCorr**（异质相关） | 同段内同时存在正/负相关 |
| **PCorr**（部分相关） | 仅部分通道对显著，全通道交互易引入噪声 |

相关形态跨数据集差异大，预训练阶段难学通用相关，更适合**下游微调插件**而非重预训练[^src-cheng-2025-cora-correlation-aware-adapter]。既有插件（CCM、C-LoRA、LIFT）多面向端到端骨干，且 few-shot 下易伤 TSFM[^src-cheng-2025-cora-correlation-aware-adapter]。

### 2. 方法：CoRA 四段流水线

CoRA 吃 TSFM 的输入、中间表示 \(\tilde X\) 与原始预测 \(\hat Y\)，输出增强预测 \(\hat Y^*\)[^src-cheng-2025-cora-correlation-aware-adapter]：

1. **[[dynamic-correlation-estimation|DCE]]**：可学习相关 = 低秩 **Time-Varying** \(Q_t\) + **Time-Invariant** \(V\)，再加 Pearson 规则项 \(R\)：\(M_t^{\mathrm{corr}} = R + Q_t V Q_t^\top\)。\(Q_t\) 用 **[[time-aware-polynomial-correlation|可学习时间多项式]]** 拟合趋势/周期；\(V=\mathrm{Sigmoid}(\mathrm{ReLU}(E_1 E_2^\top))\)[^src-cheng-2025-cora-correlation-aware-adapter]。
2. **Heterogeneous Division**：SE 风格 channel-aware 投影 \(P_1,P_2\) 把表示映到正/负相关空间（本身不直接解耦，靠对比学习引导）[^src-cheng-2025-cora-correlation-aware-adapter]。
3. **[[heterogeneous-partial-contrastive-learning|HPCL]]**：用 \(M^{\mathrm{pos/neg}}\)（阈值 \(\epsilon\) 切分正/负相关）在两空间做对比损失 \(L_{\mathrm{aux}}=L_{\mathrm{pos}}+L_{\mathrm{neg}}\)，自适应学 PCorr；**仅训练期**[^src-cheng-2025-cora-correlation-aware-adapter]。
4. **Heterogeneous Fusion**：\(P_3,P_4\) 投影后凸组合门控 \(\beta\) 与原预测融合：\(\hat Y^*=\beta\,\mathrm{Linear}(\tilde X^{\mathrm{pos}}+\tilde X^{\mathrm{neg}})+(1-\beta)\hat Y\)[^src-cheng-2025-cora-correlation-aware-adapter]。

**复杂度**：训练 DCE/HPCL 为 \(O(N^2)\)；推理只跑投影，**\(O(N)\)**[^src-cheng-2025-cora-correlation-aware-adapter]。

### 3. 理论

- **Theorem 1**：局部平稳下 \(Q_t V Q_t^\top\) 可写成时不变 + 时变两项之和，功能等价加性分解、参数更省[^src-cheng-2025-cora-correlation-aware-adapter]。
- **Theorem 2**：相关关于基 \(q\) 充分光滑时，K 阶时间多项式逼近误差随 K 增大以 Maclaurin 余项下降[^src-cheng-2025-cora-correlation-aware-adapter]。

## 主要实验

设定：TSFM-Bench，5% few-shot，10 数据集（ETT×4、Electricity、Traffic、Solar、Weather、AQShunyi、ZafNoo）；骨干 GPT4TS / CALF / UniTime / Moment / Timer / TTM；\(H\in\{96,192,336,720\}\)，MSE/MAE[^src-cheng-2025-cora-correlation-aware-adapter]。

- **主表**：六骨干 × 十数据集上，加 CoRA 的平均 MSE 几乎全面优于仅微调骨干（Table 1）[^src-cheng-2025-cora-correlation-aware-adapter]。
- **TTM CI+CoRA vs CD**：同预训练参数下，CI+CoRA 优于 CD 微调，说明三类相关比朴素全通道混合更有效[^src-cheng-2025-cora-correlation-aware-adapter]。
- **插件对比**（Fig.4，H=96）：LIFT / C-LoRA 在 few-shot 常伤骨干；CoRA 稳定降 MSE[^src-cheng-2025-cora-correlation-aware-adapter]。
- **消融**（Table 2）：DCE + HD + HPCL 全开最好；单 HPCL 有限，补 DCE 或 HD 再升[^src-cheng-2025-cora-correlation-aware-adapter]。
- **效率**（Fig.5）：N=7/21/321 时参数与训练/推理时间相对骨干增量小，推理尤其稳[^src-cheng-2025-cora-correlation-aware-adapter]。
- **数据量**（Table 3）：3% 数据仍有温和提升（如 TTM ETTm2 0.264→0.260）[^src-cheng-2025-cora-correlation-aware-adapter]。
- **超参**：K≈3–4，M 不必随 N 猛增，投影层 l1/l2 常取 3 或 5[^src-cheng-2025-cora-correlation-aware-adapter]。
- **可视化**（Weather 四通道三段）：正/负空间相似度随时间变化，对齐 DCorr/HCorr/PCorr 叙事[^src-cheng-2025-cora-correlation-aware-adapter]。

## 局限性

1. 训练仍含 \(O(N^2)\) 相关估计与对比；超大 N（如 Traffic 862）训练代价未充分剖析[^src-cheng-2025-cora-correlation-aware-adapter]。
2. Theorem 1–2 依赖**局部平稳**与相关关于基光滑；强非平稳/突变相关下误差界未必成立[^src-cheng-2025-cora-correlation-aware-adapter]。
3. HPCL 的正负对由 DCE 矩阵阈值产生——DCE 偏差会系统性污染对比监督（训练期自指）[^src-cheng-2025-cora-correlation-aware-adapter]。
4. 正文基线写 “Moment, Chronos, Timer”，主表实为 Moment / Timer / TTM，Chronos 未进 Table 1[^src-cheng-2025-cora-correlation-aware-adapter]。
5. 与仓库另一 CoRA 同名，检索与引用易混。

## 相关页面

- [[cora-correlation-aware-adapter]] — 实体页
- [[dynamic-correlation-estimation]] — DCE 技术
- [[heterogeneous-partial-contrastive-learning]] — HPCL 技术
- [[time-aware-polynomial-correlation]] — 时间多项式相关
- [[channel-independence]] — 多数 TSFM 默认策略
- [[cora-tsfm]] / [[source-cora]] — **另一篇**协变量 CoRA（消歧）
- [[crossformer]] / [[itransformer]] — 端到端通道交互对照

## 引用

[^src-cheng-2025-cora-correlation-aware-adapter]: [[source-cheng-2025-cora-correlation-aware-adapter]]
