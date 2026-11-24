---
title: "DMMV: Multi-Modal View Enhanced Large Vision Models for Long-Term Time Series Forecasting"
type: source-summary
tags:
  - time-series-forecasting
  - multimodal-time-series
  - masked-autoencoder
  - decomposition
  - vision-foundation-model
  - neurips-2025
created: 2026-11-24
last_updated: 2026-11-24
source_count: 1
confidence: medium
status: active
---

# DMMV: Multi-Modal View Enhanced Large Vision Models for Long-Term Time Series Forecasting

**来源**：ChengAo Shen、Wenchao Yu、Ziming Zhao、Dongjin Song、Wei Cheng、Haifeng Chen、Jingchao Ni（Univ. of Houston、NEC Labs America、UConn）；NeurIPS 2025；代码仓库 `D2I-Group/dmmv`[^src-dmmv]。

大型视觉模型用于时间序列预测时，通常将序列按周期折叠成二维图像再交给掩码自编码器补全。论文指出这一做法偏爱周期整数倍的输入长度：一旦段长落在周期的非整数倍处，预测便会被跨周期一致性主导，全局趋势遭到压制。为此，论文让视觉模型只承担周期成分预测，把原序列减去视觉模型回测重建得到的残差交给线性模型预测趋势，再由可学习标量门控合并两路结果[^src-dmmv]。

## 去黑话算子配方与路线差异

两套变体共享同一影像化入口：先以 FFT 主周期 $P$ 折叠序列，经标准差归一化与三通道复制后双线性缩放到 $224^2$，送入预训练 MAE 重建掩码区并反变换回一维序列[^src-dmmv]：

- **$\mathrm{D_{MMV}\text{-S}}$（经典分解）**：用窗长 $2\lfloor P/2\rfloor+1$ 的滑动平均拆分趋势与周期分量，分别送入线性层与 MAE 预测后再由门控融合[^src-dmmv]。
- **$\mathrm{D_{MMV}\text{-A}}$（回测残差分解）**：采用两趟半幅掩码回测（右半可见重建左半、左半可见重建右半）拼出历史重建 $\hat{x}$，将残差 $\Delta x = x - \hat{x}$ 送入线性层预测趋势，全图送入 MAE 预测周期，最后通过可学习标量门控 $g=\sigma(w_g)$ 计算 $\hat{y} = g \odot \hat{y}_{\text{season}} + (1-g) \odot \hat{y}_{\text{trend}}$；训练期视觉分支仅微调归一化层[^src-dmmv]。

在谱系定位上，$\mathrm{D_{MMV}}$ 与近邻形成两处明确对照：**瓶颈接力 $\Delta$（vs VisionTS）**在于把 MAE 从最终预测器改在输出前一步充当周期分解器，代价是通道独立下每变量需额外生成两张 $224^2$ 图像并多跑两趟 MAE 前向；**同槽位互斥（vs [[time-vlm|Time-VLM]]）**在于两者同属无外生模态的伪多模态路径，但 $\mathrm{D_{MMV}}$ 主动舍弃文本与检索记忆，改在分解-融合槽位显式注入周期归纳偏置[^src-dmmv]。

## 证据

1. **定位与主结论**：8 项基准中 $\mathrm{D_{MMV}\text{-A}}$ 获 6 项平均 MSE 最优，无文本下整体优于 [[time-vlm|Time-VLM]][^src-dmmv]。
2. **真实涨点归因**：去掉回测残差分解（Illness 退化约 21.7%）、冻结视觉分支（Weather 退化约 13.4%）或用加和替代门控掉点最重；线性分支换 [[patchtst|PatchTST]] 几乎不掉点[^src-dmmv]。
3. **证据缺口与负结果**：Time-VLM 缺 Illness 未纳对比；VisionTS 重跑至 $T=336$（原设 $1152/2304$）口径不利；数值分支仅测两类；ETTm1/Illness 仍受周期敏感性影响[^src-dmmv]。

## 相关页面

- [[multimodal-time-series-forecasting]] — 多模态与伪多模态视觉视图的二维决策矩阵
- [[time-vlm]] — 伪多模态路线的同槽位互斥与实证对照
- [[patchtst]] — 数值分支替换消融的跨槽位正交参考
- [[traceable-image-conversion]] — 周期折叠成像的逆向可追溯性对照

[^src-dmmv]: [[source-dmmv]]
