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
last_updated: 2026-11-27
source_count: 1
confidence: medium
status: active
---

# DMMV: Multi-Modal View Enhanced Large Vision Models for Long-Term Time Series Forecasting

把序列按周期折叠成图像、交给掩码自编码器 MAE 预测，是当前视觉路线的主流做法；但这种成像让模型偏爱周期信号，窗长不是周期整数倍时全局趋势被压制[^src-dmmv]。已有的 Time-VLM 把视觉、文本、数值三视图简单拼接，没接住各视图的归纳偏置，文本编码还贵、收益小[^src-dmmv]。DMMV 的巧思是分工而非拼接：MAE 只预测擅长的周期分量，趋势交给线性分支，可学习标量门控融合两路[^src-dmmv]。论文报告 8 项基准中 6 项平均 MSE 最优，整体优于 Time-VLM[^src-dmmv]。

## 核心做法

两套变体共享同一成像入口：先用 FFT 找出主周期，窗内序列按周期折叠成多行图像并缩放，送入预训练 MAE 补全掩码区域，再反变换回一维；两路输出由可学习标量门控加权合并[^src-dmmv]。

- **DMMV-S（简单分解）**：固定窗长的滑动平均拆出趋势与周期分量，分别给线性分支与 MAE[^src-dmmv]。
- **DMMV-A（自适应分解）**：让 MAE 先做两趟半幅掩码回测——各遮住窗内一半、用可见的另一半重建被遮部分（论文称 BC MASK，即回测阶段的整段半幅掩码），拼出对历史的重建；原序列减去这份重建的残差作为趋势分支输入，完整序列仍交 MAE 预测周期[^src-dmmv]。训练时视觉分支先冻结、仅微调其归一化层[^src-dmmv]。

## 证据

1. **定位与主结论**：论文报告 8 项基准对 14 个基线取得 6 项平均 MSE 居首、共 43 个单项第一，无文本条件下整体优于 Time-VLM[^src-dmmv]。
2. **真实涨点归因**：消融中拿掉回测残差分解掉点最狠（Illness 上 MSE 退化约 22%），换掉 MAE、门控改加和、冻结视觉分支均明显退化；数值分支换 PatchTST 骨干几乎不掉点——涨点来自分解与融合设计，而非更强数值骨干[^src-dmmv]。
3. **证据缺口与负结果**：Time-VLM 缺 Illness 对比；数值分支消融仅覆盖两类实现；ETTm2、Electricity 上数值基线仍具竞争力，8 项中 2 项平均 MSE 未居首[^src-dmmv]。

## 相关页面

- **同槽位互斥**：[[time-vlm]] — 同列 [[multimodal-time-series-forecasting]] 伪多模态槽位的两条路线；DMMV 舍弃文本与检索记忆、把周期归纳偏置显式放进分解与门控，Time-VLM 靠 embedding 级拼接并保留文本编码器。
- **瓶颈接力 $\Delta$**：[[traceable-image-conversion]] — 针对同一折叠成像偏周期瓶颈的两种接续：TIC 重做成像算子追求逆过程无损可追溯，DMMV 保留成像、让 MAE 从终点预测器退为周期分解器，把趋势接力给线性分支。
- **跨槽位正交**：[[patchtst]] — 数值槽位换用 PatchTST 化骨干几乎不掉点，说明视觉-分解组件与 patch 化数值模型的贡献正交、可互换。
- **同构（显式周期分解）**：[[cyclenet]] — 同以显式周期分解为核心构件但纯数值、无视觉视图；论文平均排名显示固定分解的 DMMV-S 与其相当。

---

Shen et al.（University of Houston、NEC Laboratories America、University of Connecticut）, *DMMV: Multi-Modal View Enhanced Large Vision Models for Long-Term Time Series Forecasting*, NeurIPS 2025；代码仓库 `D2I-Group/dmmv`。

[^src-dmmv]: [[source-dmmv]]
