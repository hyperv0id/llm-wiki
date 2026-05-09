---
title: "DCW: Differential Correction in Wavelet Domain"
type: technique
tags:
  - diffusion-model
  - snr-t-bias
  - wavelet
  - wavelete-domain
  - training-free
  - post-processing
created: 2026-05-09
last_updated: 2026-05-09
source_count: 1
confidence: high
status: active
---

# DCW: Differential Correction in Wavelet Domain

**DCW（Differential Correction in Wavelet Domain）** 是一种针对扩散模型 SNR-t bias 的无需训练、即插即用的后处理方法。通过对每一步去噪结果与重建样本之间的差分信号进行小波域分解和频率分量级校正，有效缓解信噪比-时间步错配问题。[^src-2604.16044]

## 动机

SNR-t bias 使逆过程预测样本 $\hat{x}_{t-1}$ 偏离理想的前向样本 $x_{t-1}$。然而，差分信号 $\hat{x}_{t-1} - x_\theta^0(\hat{x}_t, t)$ 天然包含将 $\hat{x}_{t-1}$ 拉向 $x_{t-1}$ 的方向信息（势信息）。[^src-2604.16044]

## 核心公式

### 像素空间差分校正

$$\hat{x}'_{t-1} = \hat{x}_{t-1} + \lambda_t (\hat{x}_{t-1} - x_\theta^0(\hat{x}_t, t)) \tag{Eq. 17}$$

其中 $\lambda_t$ 是标量引导系数。该方法不需要增加 NFE，因为 $x_\theta^0(\hat{x}_t, t)$ 在去噪过程 $x_{t-1} \leftarrow x_t$ 中已经计算完毕。

### 小波域差分校正

对 $\hat{x}_{t-1}$ 和 $x_\theta^0(\hat{x}_t, t)$ 分别做离散小波变换（DWT），得到四个子带 $(ll, lh, hl, hh)$：

- $ll$：低频子带，表示图像的形状轮廓（如人脸、房屋）
- $lh, hl, hh$：不同方向的高频子带，表示纹理细节（如皱纹、叶脉）

对每个子带分别校正：

$$\hat{x}^f_{t-1} = \hat{x}^f_{t-1} + \lambda^f_t (\hat{x}^f_{t-1} - x^f_\theta(\hat{x}_t, t)), \quad f \in \{ll, lh, hl, hh\} \tag{Eq. 18}$$

校正后通过逆离散小波变换（iDWT）映射回像素空间：

$$\tilde{x}_{t-1} = \text{iDWT}(\hat{x}^f_{t-1} | f \in \{ll, lh, hl, hh\}) \tag{Eq. 19}$$

## 动态权重调度

动机：去噪过程遵循**从粗到细**的范式——早期重建低频轮廓，后期细化高频细节。[^src-2604.16044]

利用逆过程方差 $\sigma_t$ 作为去噪进度的自然指示器（去噪早期 $\sigma_t$ 大，后期 $\sigma_t$ 小）：

**低频校正系数**（早期大后期小）：

$$\lambda^l_t = \lambda_l \cdot \sigma_t \tag{Eq. 20}$$

**高频校正系数**（早期小后期大）：

$$\lambda^h_t = (1 - \lambda_h) \cdot \sigma_t \tag{Eq. 21}$$

其中 $\lambda_l, \lambda_h$ 是标量超参数。通过两阶段搜索法确定：先粗搜索（步长 0.01）确定大致区间，再细搜索（步长 0.001）精确确定最优值。

## 算法流程

1. 对每个去噪步 $t$（从 $T$ 到 1），执行标准 DPM 单步去噪得到 $\hat{x}_{t-1}$
2. 从去噪网络获取重建样本 $x_\theta^0(\hat{x}_t, t)$（零额外 NFE）
3. DWT 分解 $\hat{x}_{t-1}$ 和 $x_\theta^0(\hat{x}_t, t)$ 到四子带
4. 对每个子带 $f$ 应用 Eq. 18，系数 $\lambda^f_t$ 由 Eq. 20-21 给出
5. iDWT 重构回像素空间得到 $\tilde{x}_{t-1}$
6. 以 $\tilde{x}_{t-1}$ 作为下一步的输入

## 关键性质

| 性质 | 说明 |
|------|------|
| **无需训练** | 纯推理时后处理，不修改模型参数 |
| **即插即用** | 适用于所有主流 DPM 架构 |
| **零额外 NFE** | 校正利用已计算的 $x_\theta^0$，不增加网络推理次数 |
| **计算开销极低** | DWT/iDWT 仅增加 0.08%~0.47% 的推理时间 |
| **可叠加** | 可在 ADM-ES、DPM-FR 等偏置校正模型之上进一步改进 |

## 实验结果（节选）

| 模型 | 数据集 | 步数/NFE | FID 降幅 |
|------|--------|-----------|----------|
| IDDPM | CIFAR-10 | 20 步 | 42.6% |
| IDDPM | CIFAR-10 | 50 步 | 25.0% |
| EDM | CIFAR-10 | 13 NFE | 42.9% |
| ADM | ImageNet 128 | 20 步 | 15.8% |
| IDDPM | LSUN Bedroom 256 | 20 步 | 41.0% |

## 超参数敏感性

$\lambda_l$ 和 $\lambda_h$ 对最终 FID 的影响呈先降后升的单峰形态，存在较宽的最优区间，敏感性较低。在两阶段搜索中，先固定 $\lambda_h$ 搜索 $\lambda_l$，再固定最优 $\lambda_l$ 搜索 $\lambda_h$，可快速确定最优值。[^src-2604.16044]

## 与同类方法的区别

相比于 ADM-ES 和 DPM-FR 等 exposure bias 校正方法：

- DCW 的理论基础是 SNR-t bias（有严格推导）而非 exposure bias（经验观察）
- DCW 在小波域操作而非像素域，可利用频率分量的去噪阶段性特征
- DCW 使用 $\sigma_t$ 动态调度权重，而非固定系数
- DCW 可在这些方法之上进一步改进（证明其捕捉到了不同/互补的偏置信号）

[^src-2604.16044]: [[source-snr-t-bias]]
