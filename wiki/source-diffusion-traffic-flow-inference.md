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

论文提出 DP-TFI 处理 FUFI：从粗粒度流量图 $X_c \in \mathbb{R}^{I\times J}_+$ 推断细粒度图 $X_f \in \mathbb{R}^{NI\times NJ}_+$，粗粒度每个 superregion 对应 $N\times N$ 个 subregion（$N$ 为 scaling factor），以减少传感器部署[^src-diffusion-traffic-flow-inference]。structural constraint 要求 superregion 流量等于其 $N^2$ 个 subregion 之和[^src-diffusion-traffic-flow-inference]。

## 方法机制

三模块[^src-diffusion-traffic-flow-inference]：

1. **Disentangled Flow Feature Learning**：流量与外部因子各走一条 CNN 流（流量侧 3 层 3×3 conv + ReLU，外部因子经 embedding/dense 层编码），各自经 PixelShuffle + conv 上采样；解耦使推断仅依赖流内部空间特性。
2. **Relaxed Distributional Upsampling (RDU)**：先在 $N^2$ 个 subregion 上归一化得 $\tilde{H}_f^l \in [0,1]$，再以 relax matrix $R^f = 2\mu\,\mathrm{Sigmoid}(\mathrm{Conv2d}(\cdot)) - (\mu+1)J$（$J$ 为全 1 矩阵，$\mu$ 控制灵活度）放宽上采样，得 $\hat{X}_f = X_c^{\text{relax}} \otimes \tilde{H}_f^l$，损失为 MSE $\|\hat{X}_f - X_f\|_2$。原文文本丢失了 Eq.(3) 中 Conv2d 自变量的上标，此处不指定。
3. **Diffusion Probabilistic Augmentor (DPA)**：基于 DDPM（Ho et al., NeurIPS 2020），forward $q(X_f^t \mid X_f^{t-1}) = \mathcal{N}(\sqrt{\alpha_t}X_f^{t-1}, (1-\alpha_t)I)$ 逐步加噪，reverse $p_\theta(X_f^{t-1} \mid X_f^t, X_f) = \mathcal{N}(\sigma_\theta(X_f, X_f^t, \gamma_t), \sigma_t^2 I)$ 生成带不确定性的新流量图，生成后过 $N^2$-Normalization，仅作训练增强。

## 证据

数据为 TaxiBJ 北京出租车流量，粗粒度 $32\times32$ 上采样至 $128\times128$（$N=4$），外部因子含温度（$[-24.6, 41.0]$°C）、风速（$[0, 48.6]$ mph）、16 类天气[^src-diffusion-traffic-flow-inference]。年份原文即不一致：正文写 "from 2013 to 2015"，Table 1 的 P4 却至 Mar 31, 2016。基线 10 个：6 SISR（SRCNN/ESPCN/VDSR/DeepSD/SRResNet/LapSRN）+ 4 FUFI（UrbanFM/FODE/UrbanODE/UrbanPy）[^src-diffusion-traffic-flow-inference]；Adam，lr $1\times10^{-3}$，$F=128$，batch 16，RTX 3090。

按 RMSE/MAE/MAPE，DP-TFI 在 4 个时段均最优：P1 3.853/1.926/0.298、P2 4.240/2.144/0.292、P3 4.404/2.227/0.287、P4 3.429/1.759/0.292[^src-diffusion-traffic-flow-inference]。P4 次优 UrbanPy（3.470/1.801/0.313）；P4 最佳 SISR 基线 LapSRN 的 MAE 1.841、MAPE 0.315，但其 P4 RMSE 原文印作 0.351，与同列 UrbanFM 3.514 量级不符，疑为排版笔误，本页不引。消融设 w/o EF、w/o DPA、w/o RDU，论文称外部因子融合贡献最大（"external factor fusion contributes the most"）[^src-diffusion-traffic-flow-inference]。$\mu$ 从 0 扫到 0.1，论文称 "µ around 0.02 leads to better performance"；"$\mu=0$ 与过度放宽都更差"系据 Fig. 2(c) 曲线的推断，原文未明说。Fig. 2(b) 显示模型对多数 subregion 倾向打破结构约束，重流量区域分得更多（relax coefficient 取 3%）[^src-diffusion-traffic-flow-inference]。

## 局限性

- 仅在 TaxiBJ（北京、出租车流量）上验证。
- 论文未给出约束违反量的理论界（页面评述）。
- DPA 生成图仅作训练增强，推理时不以外部因子与粗粒度图为 diffusion 条件。

## 论文自述 future work 与页面评述

- **论文自述**：future work 可将外部因子与粗粒度图作为生成条件，并以 loss regularization 优化推理[^src-diffusion-traffic-flow-inference]。
- **页面评述**：RDU 未论证放宽后的分配何时严格优于严格归一化；约束违反量无理论界。

## 与 [[diffusion-model]] 的关联（页面评述）

DP-TFI 不把 diffusion 当主生成器，而是作 augmentor 为 FUFI 建模不确定性；同期 DiffSTG（引文 [14]）已将 denoising diffusion 用于时空图预测。

[^src-diffusion-traffic-flow-inference]: [[source-diffusion-traffic-flow-inference]]
