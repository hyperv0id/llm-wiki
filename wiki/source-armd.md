---
title: "ARMD: Auto-Regressive Moving Diffusion Models for Time Series Forecasting"
type: source-summary
tags:
  - diffusion-models
  - time-series
  - forecasting
  - arma
  - sliding-window
  - aaai-2025
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: medium
status: active
---

# ARMD: Auto-Regressive Moving Diffusion Models for Time Series Forecasting

**ARMD**（Auto-Regressive Moving Diffusion）由 Jiaxin Gao、Qinglong Cao、Yuntian Chen（上海交通大学 / 宁波东方理工数字孪生研究院）提出，arXiv:2412.09328（2024-12），发表于 AAAI 2025[^src-armd]。它是[[armd|首个"连续序列扩散" (continuous sequential diffusion) 时间序列预测模型]]，旨在弥合扩散机制与预测目标之间的失配。

## 核心问题

现有扩散类 TSF 方法（[[csdi|CSDI]]、[[timegrad|TimeGrad]]、Diffusion-TS 等）沿用图像领域的"加噪—去噪"范式，把预测当作以历史序列为条件、从白高斯噪声生成未来序列的条件生成任务[^src-armd]。作者指出这忽视了时间序列固有的连续序列演化性质，导致扩散机制与 TSF 目标根本性失配，且直接把真实序列扩散成白噪声会丢弃序列演化中有价值的中间信息[^src-armd]。

## 核心思想与方法

受经典 [[arma-inspired-diffusion|ARMA 理论]]启发——时间序列被视为前序数据点连续演化加随机噪声的链式过程——ARMD 把"未来序列"设为扩散初始状态 $X^0_{1:T}$、"历史序列"设为最终状态 $X^T_{-T+1:0}$（历史长度与预测长度相等）[^src-armd]。关键创新是：中间状态不靠加噪生成，而是把未来序列按扩散步**滑动** $t$ 步得到，即 $X^t_{1-t:T-t}=\mathrm{Slide}(X^0_{1:T}, t)$，构成确定性的[[sliding-window-diffusion|滑动窗口扩散]]过程[^src-armd]。反向"去演化"(devolution) 网络是一个**线性骨干**，通过[[distance-based-devolution|基于距离的方法]]预测演化趋势 $z^t$：先用线性层估计中间态到目标的距离 $D$，再以随扩散步 $t$ 自适应的权重 $W(t)$ 平衡中间态与 $D$[^src-armd]。采样借用 DDIM 形式但去掉随机噪声项，从历史序列出发迭代生成未来序列，构成无条件 (unconditional) 扩散预测器[^src-armd]。

## 主要结果

在 7 个常用数据集（Solar Energy、Exchange、Stock、四个 ETT）上，历史与预测长度均设为 96，ARMD 在 14 个设置中的 12 个取得最优，显著超越 Diffusion-TS、MG-TSD、TSDiff、[[d3vae|D³VAE]]、TimeGrad 等扩散基线[^src-armd]。在 ETTm1 上较次优的 D³VAE 降低 47.7% MSE、30.1% MAE[^src-armd]。因线性骨干 + 极少采样步（grid search 从 {1,2,3,4,6,8,12} 选取，其他扩散模型用 100 步），ARMD 在 ETTm1 上训练/推理较 Diffusion-TS 等加速逾十倍[^src-armd]。给定相同历史序列做 10 次独立预测，ARMD 比 Diffusion-TS 更稳定准确[^src-armd]。消融显示滑动法优于插值法、距离法优于 t-embedding 法、线性骨干优于 Transformer 骨干、训练加微小 deviation 防过拟合、采样加噪反而损害性能[^src-armd]。

## 局限性

历史长度被强制等于预测长度（受滑动扩散步数 $T$ 与序列长度绑定所限），未报告变长配置[^src-armd]。确定性采样使其本质偏向点预测而非完整概率分布，与 TimeGrad/CSDI 的概率预测目标不同[^src-armd]。超参数 $b,c,d$ 需逐数据集 grid search[^src-armd]。评测集中在 96-96 单一长度，未覆盖更长预测视野[^src-armd]。

## 关联页面

- [[armd]] — 方法实体页
- [[sliding-window-diffusion]] — 滑动窗口替代加噪的中间态生成
- [[arma-inspired-diffusion]] — ARMA 启发的扩散范式
- [[distance-based-devolution]] — 基于距离的线性去演化网络
- [[timegrad]]、[[d3vae]]、[[diffusion-models]] — 对比与背景

[^src-armd]: [[source-armd]]
