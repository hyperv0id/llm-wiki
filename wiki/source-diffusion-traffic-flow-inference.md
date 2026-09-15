---
title: "Diffusion Probabilistic Modeling for Fine-Grained Urban Traffic Flow Inference with Relaxed Structural Constraint"
type: source-summary
tags:
  - spatio-temporal
  - diffusion-model
  - traffic-inference
  - urban-computing
  - super-resolution
created: 2026-09-15
last_updated: 2026-09-15
source_count: 1
confidence: medium
status: active
---

# DP-TFI: Diffusion Probabilistic Modeling for Fine-Grained Urban Traffic Flow Inference

**Authors**: Xovee Xu, Yutao Wei, Pengyu Wang, Xucheng Luo, Fan Zhou (UESTC), Goce Trajcevski (Iowa State)
**Venue**: ICASSP 2023 (doi: 10.1109/ICASSP49357.2023.10096169)

## 问题定义

论文提出 DP-TFI，处理 fine-grained urban traffic flow inference (FUFI)：不是预测未来，而是从粗粒度流量图推断细粒度流量图——粗粒度图的每个 superregion 对应 $N\times N$ 个 subregion（scaling factor $N$），目标是生成 $NI\times NJ$ 的细粒度图以减少传感器部署成本[^src-diffusion-traffic-flow-inference]。FUFI 有一个 structural constraint：superregion 流量必须等于其 $N^2$ 个 subregion 流量之和（$N^2$-Normalization 层强制执行）[^src-diffusion-traffic-flow-inference]。

## 方法机制

DP-TFI 三个模块：(1) **Disentangled Flow Feature Learning**——流量与外部因子（天气、日期）各走一条 CNN 流（3 层 3x3 conv + ReLU，PixelShuffle 上采样），解耦学习让流量动态仅由内部空间依赖决定[^src-diffusion-traffic-flow-inference]。(2) **Diffusion Probabilistic Augmentor (DPA)**——基于 DDPM（Ho et al., NeurIPS 2020）的forward Markovian process $q(X_f^t|X_f^{t-1})=\mathcal{N}(\sqrt{\alpha_t}X_f^{t-1},(1-\alpha_t)I)$ 对细粒度图逐步加噪，再经 reverse process $p_\theta$ 生成带不确定性的新流量图实例，生成后仍过 $N^2$-Normalization 保证不偏离原图，用作数据增强提升鲁棒性[^src-diffusion-traffic-flow-inference]。(3) **Relaxed Distributional Upsampling (RDU)**——放宽 structural constraint：用 relax matrix $R^f = 2\mu\,\mathrm{Sigmoid}(\mathrm{Conv2d}(H_{ef})) - (\mu+1)J$ 允许 subregion 分得比严格均分更多或更少的流量，relax coefficient $\mu$ 控制灵活度，损失为 MSE $\|\hat{X}_f - X_f\|_2$[^src-diffusion-traffic-flow-inference]。

## 实验结果

数据为北京出租车流量 2013-2016，分 4 个时段 P1-P4，30 分钟间隔，粗粒度 32x32 上采样至 128x128（$N=4$），外部因子含温度（[-24.6, 41.0]°C）、风速（[0, 48.6] mph）、16 类天气[^src-diffusion-traffic-flow-inference]。对比 10 个基线（SRCNN/ESPCN/VDSR/DeepSD/SRResNet/LapSRN 六个 SISR 方法 + UrbanFM/FODE/UrbanODE/UrbanPy 四个 FUFI 方法），指标 RMSE/MAE/MAPE。DP-TFI 在全部 4 个时段三项指标均为最优：P4 上 RMSE 3.429 / MAE 1.759 / MAPE 0.292，优于次优 UrbanPy（3.470 / 1.801 / 0.313）和最佳 SISR 基线 LapSRN（3.997 / 1.841 / 0.315）[^src-diffusion-traffic-flow-inference]。消融显示外部因子融合贡献最大；$\mu$ 在 0 附近（约 0.02-0.03）最优——完全严格约束（$\mu=0$）与过度放宽都更差[^src-diffusion-traffic-flow-inference]。

## 局限性

- relax matrix 分析显示模型对多数 subregion 都倾向打破结构约束，重流量区域获得更多分配，但缺乏对约束违反量的理论界[^src-diffusion-traffic-flow-inference]
- DPA 生成的流量图仅用作训练增强，推理时未将外部因子与粗粒度图作为 diffusion 的条件输入（论文自述为 future work）[^src-diffusion-traffic-flow-inference]
- 仅在单一城市（北京）单一数据模态（出租车流量）上验证

## 与 [[diffusion-model]] 的关联

DP-TFI 是 [[diffusion-model]] 在城市计算中的早期应用案例：不把 diffusion 当主生成器，而是作为数据增强器（augmentor）建模 FUFI 的不确定性；同一时期 diffusion 开始进入时空领域（如 DiffSTG）。与 wiki 中 [[flow-matching]]、[[consistency-models]] 等后续生成范式形成对照。

[^src-diffusion-traffic-flow-inference]: [[source-diffusion-traffic-flow-inference]]
