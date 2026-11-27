---
title: "VolaRM: Volatility-Whitened Probabilistic Residual Modeling for Long-Term Time Series Forecasting"
type: source-summary
tags:
  - time-series-forecasting
  - probabilistic-forecasting
  - diffusion-model
  - ddim
  - residual-modeling
  - non-stationary
  - neurips-2026
created: 2026-11-24
last_updated: 2026-11-27
source_count: 1
confidence: medium
status: active
---

# VolaRM: Volatility-Whitened Probabilistic Residual Modeling for Long-Term Time Series Forecasting

多数扩散预测直接在原始未来序列上加噪去噪，高波动通道的幅值主导噪声匹配损失，平稳通道的精细结构欠监督；两阶段「基准加残差精修」把残差原样加回，观测与未来交界处又容易跳变。VolaRM 把「未来落在哪」和「还剩多少随机性」拆开：边界概率基准给出逐点均值与标准差，残差按该标准差白化后再扩散，最后用门控保守放回。八个数据集中七个领先，高维 Traffic 是例外[^src-volarm]。

## 核心做法

**边界概率基准。** 提取输入窗口末端统计量（值、一二阶差分与局部波动）归一池化为接缝条件；将二阶多项式外推与近期周期插值逐通道加权融合，叠加历史编码与位置偏移，输出未来逐点均值与截断为正的标准差[^src-volarm]。

**白化残差与八步 DDIM。** 将目标残差按该标准差标准化白化并截断后加噪，以历史编码、接缝条件与扩散步为条件执行八步截断 DDIM（去噪扩散隐式模型）确定性采样，计算开销低但样本多样性受限[^src-volarm]。

**门控融合与接缝正则。** 最终预测以带上限的门控将去噪残差保守放回基准，联合点预测、白化噪声匹配、高斯负对数似然与交界连续正则训练。代价是需同时训练三个头、依赖手工外推锚点，且标准差估计误差会被白化放大[^src-volarm]。

## 证据

1. **定位与主结论**：八个数据集七个点误差最优，较 iTransformer、Pathformer 及重构式自监督类基线约降 4%–13%，较 TimeDART 约降 3%；Traffic 不及后者[^src-volarm]。
2. **真实涨点归因**：未白化直接扩散掉点最重，仅留概率基准退化较小，接缝正则边际贡献偏弱，主收益在概率基准与白化坐标[^src-volarm]。
3. **证据缺口**：缺 [[nsdiff|NsDiff]]、[[k2vae|K²VAE]]、[[csdi|CSDI]]、[[timegrad|TimeGrad]] 等同生态位概率基线同台，且无 CRPS（连续分级概率评分）指标[^src-volarm]。

## 相关页面

- [[nsdiff|NsDiff]] — 同槽位互斥：它做时变位置—尺度噪声调度，VolaRM 挪进残差白化坐标。
- [[location-scale-noise-model|LSNM]] — 同构对接：基准均值与标准差即该位置—尺度参数化，作用于残差而非噪声。
- [[k2vae|K²VAE]] — 同槽位互斥：同占「基准加残差精修」槽位，它以 Koopman 与一步 VAE 替代八步 DDIM。
- [[instance-normalization|RevIN]] — 跨槽位正交：它在输入端做整窗归一化，VolaRM 在预测端做逐点残差白化。

Fan Zhang、Shiming Fan、Zexuan Ma、Shijun Chen、Meijia Wang（山东工商学院），Hua Wang（鲁东大学，通讯）；NeurIPS 2026。

[^src-volarm]: [[source-volarm]]
