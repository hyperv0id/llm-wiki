---
title: "Sonnet: Spectral Operator Neural Network for Multivariable Time Series Forecasting"
type: source-summary
tags:
  - time-series-forecasting
  - exogenous
  - frequency-domain
  - wavelet
  - koopman-operator
  - aaai-2026
created: 2026-11-24
last_updated: 2026-11-24
source_count: 1
confidence: medium
status: active
---

# Sonnet: Spectral Operator Neural Network for Multivariable Time Series Forecasting

**来源**：Yuxuan Shu、Vasileios Lampos（University College London）；AAAI 2026（arXiv:2505.15312，代码 `ClaudiaShu/Sonnet`）[^src-sonnet]。

在包含外生变量的多元时间序列预测中，各外生信号的预测价值会随频率和时间动态变化，常规时域点积注意力容易被局部幅度起伏或非平稳共漂误导。Sonnet 先利用可学习调制小波抽取多频带时频特征，再沿变量轴做傅里叶变换并计算归一化互谱相干作为注意力权重来筛选跨变量关联；随后通过正交分解约束为酉矩阵的 Koopman 算子一次性推进整段频谱演化，最后用三层一维卷积与线性头输出预测窗口末端的单点目标值[^src-sonnet]。

## 去黑话算子配方与路线差异

模型未引入改变复杂度的底层新算子，而是将可学习小波嵌入、变量维互谱相干注意力（MVCA）与酉约束 Koopman 一步演化串行拼接[^src-sonnet]：

- **可学习小波嵌入**：以调制核 $M_k = \exp(-w_\alpha t^2)\cos(w_\beta t + w_\gamma t^2)$ 将输入投影为 $K$ 个频带原子特征 $P\in\mathbb{R}^{K\times L\times d}$，并以 $K$ 个小波原子充当 $K$ 个注意力头；该设计保留完整时间长度 $L$，同时带来 $K$ 倍特征存储放大且频带数 $K$ 固定[^src-sonnet]。
- **变量轴频谱相干注意力（MVCA）**：沿变量维对 $Q_h, K_h$ 执行 `FFT`，计算互谱 $P_{qk} = Q_f\odot K_f^*$ 与归一化频谱相干 $|P_{qk}|^2/(P_{qq}P_{kk}+\epsilon)$，沿频带取均值后经 `softmax` 加权 $V_h$，从而用变量轴频谱相干打分替代常规时域或通道点积[^src-sonnet]。
- **酉约束 Koopman 一步演化与解码**：对复矩阵 $S\in\mathbb{C}^{K\times K}$ 在每次前向传播中执行 `QR` 分解得到酉矩阵 $U$，构造 $\mathcal{K}=UDU^\dagger$（其中 $D=\operatorname{diag}(e^{ip_k})$）对特征张量做一次性前向频谱演化（非逐波步递归，规避步进累积误差），最后经 $[5,3,3]$ 三层 1D 卷积与线性头输出窗末单点 $\hat{y}_{t+H}$（该主表评测口径与常规全窗模型不同，整窗轨迹对比推至 arXiv 附录 E.8）[^src-sonnet]。
- **有向关系对照**：相对 [[source-timexer|TimeXer]] 属**同槽位互斥**（patch 与 variate 双粒度时域交叉注意力 vs 变量轴 `FFT` 频谱相干注意力）；相对 [[itransformer|iTransformer]] 属**同槽位互斥**（整序列压缩为 variate token 做点积 vs 保留完整时间长度 $L$ 的变量轴频谱相干）；相对 [[crosslinear|CrossLinear]] 属**同槽位对比**（1D 卷积注入时不变跨相关 vs 可学习小波与互谱相干）；相对 [[koopman-linearization-for-forecasting|Koopman 线性化]] 中 eDMD 局部拟合属**特例对照**（端到端可微酉谱参数化一步推进）；与 [[frequency-pathway-routing|Frequency Pathway Routing]] 属**跨槽位正交**（节点级稀疏 Top-$K$ 频带路由 vs 全原子频谱相干加权）；并与 [[fedformer|FEDformer]] 存在**复现口径疑点对照**[^src-sonnet]。

## 证据

1. **定位与主结论**：主表预测窗末单点 $\hat{y}_{t+H}$（整窗推至 arXiv 附录 E.8），12 集 47 项任务 34 项最优（较 DeformTime 降 MAE $2.2\%$），优势在多外生天气与 ILI 场景[^src-sonnet]。
2. **真实涨点归因**：MVCA 替换 [[patchtst|PatchTST]]、Samformer、[[itransformer|iTransformer]] 点积在 ILI 平均降 MAE $10.7\%$（PatchTST 降 $15.1\%$），但去原注意力不掉点甚至个别步长更好，涨点在频谱相干打分而非注意力容量；小波与酉 Koopman 独立消融推至 arXiv 附录 E.3，正文无法判别谁掉点最狠[^src-sonnet]。
3. **证据缺口与负结果**：6 外生短跨度 ETT 偏弱（ETTh2 $H=720$ $0.4335$ vs $0.3640$）；内部替换 [[fedformer|FEDformer]] MAE 异常升至约 $6$（作者报告存疑）；漏测 [[crosslinear|CrossLinear]] 且缺显存耗时实测[^src-sonnet]。

## 相关页面

- [[source-timexer]] — patch/variate 交叉注意力 vs 变量轴 `FFT` 频谱相干（同槽位互斥）
- [[itransformer]] — 整序列 variate token 点积 vs 保留长度 $L$ 的变量轴频谱相干（同槽位互斥）
- [[crosslinear]] — 1D 卷积时不变跨相关 vs 小波与频谱相干（同槽位对比与漏测指针）
- [[koopman-linearization-for-forecasting]] — eDMD 局部拟合 vs 酉矩阵 $UDU^\dagger$ 一步谱演化（特例对照）
- [[frequency-pathway-routing]] — 稀疏 Top-$K$ 频带路由 vs 全原子频谱相干加权（跨槽位正交）
- [[fedformer]] — 作者报告内部替换异常退化注记（复现口径疑点）

[^src-sonnet]: [[source-sonnet]]
