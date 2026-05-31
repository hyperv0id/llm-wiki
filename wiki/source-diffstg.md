---
title: "DiffSTG: Probabilistic Spatio-Temporal Graph Forecasting with Denoising Diffusion Models"
type: source-summary
tags:
  - diffusion-models
  - spatio-temporal-graph
  - probabilistic-forecasting
  - traffic-forecasting
created: 2026-05-31
last_updated: 2026-05-31
source_count: 1
confidence: medium
status: active
---

# Source: DiffSTG

**作者**: Haomin Wen, Youfang Lin, Yutong Xia, Huaiyu Wan, Qingsong Wen, Roger Zimmermann, Yuxuan Liang
**发表**: AAAI 2023 (arXiv:2301.13629v4, 2024年3月10日最终归档)
**机构**: 北京交通大学 / NUS / DAMO Academy Alibaba / HKUST(GZ)
**代码**: https://github.com/wenhaomin/DiffSTG

## 核心论点

DiffSTG 首次将去噪扩散概率模型（[[ddpm|DDPM]]）推广到时空图（STG）预测领域，提出了非自回归的条件扩散框架[^src-diffstg]。现有 STGNN 虽然在确定性预测中表现出色，但无法建模数据中的内在不确定性；而 [[timegrad|TimeGrad]] 等扩散时序模型又缺乏空间依赖建模能力[^src-diffstg]。DiffSTG 连接了这两条线，证明了扩散模型可以在非欧几里得的图空间中进行概率预测。

## 关键创新

### 广义条件扩散模型

将历史 $\mathcal{X}^h$ 和未来 $\mathcal{X}^p$ 统一为 $\mathcal{X}^{\text{all}} = [\mathcal{X}^h, \mathcal{X}^p]$，通过 mask 未来位置得到条件信号 $\mathcal{X}^{\text{all}}_\text{msk}$，使去噪网络 $\varepsilon_\theta(\mathcal{X}^{\text{all}}_n, n \mid \mathcal{X}^{\text{all}}_\text{msk}, \mathcal{G})$ 同时在历史重建和未来预测上接受训练[^src-diffstg]。这一公式统一了预测、生成、插补三个任务。

### UGnet 去噪网络

专为 STG 设计的异构去噪架构[^src-diffstg]：
- **时间维度**：TCN 门控因果卷积（Gate TCN: $P \odot \sigma(Q)$）+ Unet 结构的多尺度聚合
- **空间维度**：vanilla GCN（$A_\text{gcn} = D^{-1/2}(A+I)D^{-1/2}$）
- **噪声条件**：Transformer 正弦位置编码注入每个 ST-Residual Block

### 非自回归一次全窗口生成

不同于 [[timegrad|TimeGrad]] 的自回归生成（$T_p=12$ 需 $12\times 100 = 1200$ 次前向），DiffSTG 通过一次反向扩散直接输出全部 $T_p$ 个未来时间步[^src-diffstg]。配合 DDIM 子集采样（$M=40$）和尾步样本复用，推理速度比 [[timegrad|TimeGrad]] 快约 40 倍。

## 实验与结果

在三个真实数据集上验证：PEMS08（170 节点交通流量）、AIR-BJ（34 站 PM2.5）、AIR-GZ（41 站 PM2.5）[^src-diffstg]。CRPS 在三个数据集上分别降低了 5.6%、4.3%、14.3%；MAE 分别降低 4.1%、1.5%、7.0%[^src-diffstg]。消融实验表明 U-structure ≥ Temporal > Spatial 三个组件的贡献度递减。

## 局限性

- 确定性精度落后于最佳 STGNN（GMSDR、STGNCDE）约 5–10%——扩散模型的 ELBO 优化不会产生和直接 RMSE 优化同样锐利的后验[^src-diffstg]
- GCN 过于朴素（vanilla），缺乏注意力机制和动态图学习能力[^src-diffstg]
- 非自回归架构要求预测长度 $T_p$ 在训练时固定[^src-diffstg]
- 噪声二次调度（quadratic schedule）是手工设计的经验选择[^src-diffstg]

## 后续影响

DiffSTG 开创了"用扩散模型做 STG 概率预测"的范式，后续工作包括 [[specstg|SpecSTG]]（将扩散移至谱域）、D3、DiffLoad 和 UrbanDiT 等[^src-diffstg]。

[^src-diffstg]: [[source-diffstg]]
