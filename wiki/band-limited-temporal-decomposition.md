---
title: "Band-Limited Temporal Decomposition"
type: technique
tags:
  - frequency-domain
  - fourier
  - time-series
  - decomposition
  - spatio-temporal
created: 2026-09-16
last_updated: 2026-09-16
source_count: 1
confidence: medium
status: active
---

# Band-Limited Temporal Decomposition

**Band-limited temporal decomposition** 是把序列按频率切带、再用逆变换把每个频带单独投回时域的做法，使各频带成为一条可直接送入下游模型处理的时域序列，而不是停留在复数谱系数上[^src-adafre]。[[adafre|AdaFre]]（AAAI 2026）用它构造多频视图[^src-adafre]。

## 机制

给定输入 $X\in\mathbb{R}^{T\times N\times C}$，沿时间轴做 DFT[^src-adafre]：

$$\hat{X}=\mathrm{FFT}(X,\dim=0)\in\mathbb{C}^{T\times N\times C}$$

实信号的谱具有 Hermitian 对称性，$T$ 个 bin 中只有前 $\lfloor T/2\rfloor+1$ 个独立，这部分被定义为有效频率范围。把有效范围切成 $P$ 个不重叠的连续索引集[^src-adafre]：

$$I_p=\left(\frac{(p-1)(\lfloor T/2\rfloor+1)}{P},\ \dots,\ \frac{p(\lfloor T/2\rfloor+1)}{P}\right],\quad p=1,\dots,P$$

每个频带约覆盖 $\frac{\lfloor T/2\rfloor+1}{P}$ 个频率分量。对每个频带，把 $I_p$ 之外的谱系数置零，再逆变换回时域[^src-adafre]：

$$H^{(p)}=\mathrm{Re}\left(\mathrm{IFFT}\left(\hat{X}_{I_p}\right)\right)\in\mathbb{R}^{T\times N\times C}$$

得到 $P$ 条时域序列 $H=\{H^{(1)},\dots,H^{(P)}\}$。$I_1$ 覆盖最低频，对应长期趋势；$I_P$ 覆盖最高频，对应噪声或快速变化；中间频带对应中程波动[^src-adafre]。

因为 DFT 与 IFFT 互为可逆变换，各频带在时域上互补，理想情况下逐分量相加可还原输入 [INFERENCE]；论文本身没有讨论重构误差，只在实验中以 $P=4$ 作为默认设置[^src-adafre]。

## 论文自述的设计取舍

AdaFre 选择 DFT 分带而非小波或学习式分解的理由是：DFT 把不同分量放进彼此独立的频率通道，可直接解释和操作，便于抑制高低频之间的干扰并对不同时间模式做专门处理[^src-adafre]。这条理由属于论文自述，论文未给出 DFT 与 wavelet 分解在同一模型内的对照实验[^src-adafre]。

## 与相关频域技术的区别

| 技术 | 频率划分方式 | 划分是否可学习 | 产出的表示 |
|------|--------------|----------------|------------|
| **Band-limited temporal decomposition**（AdaFre） | DFT 有效范围均匀切成 $P$ 个连续频带 | 否（边界固定） | $P$ 条时域序列 |
| [[spectral-graph-wavelet-transform|SGWT]] / [[source-stwave|STWave]] 的 DWT 解耦 | 低通/高通滤波 + 下采样，一级只分两带 | 否 | 两条下采样时域序列 |
| [[fedformer|FEDformer]] | 随机选择固定数量频率分量 | 否（随机） | 频谱子集 |
| [[adaptive-frequency-modulation|AFM]] | 对数尺度划分 + 可学习 Beta 滤波核 | 展布可学习 | 空间域滤波结果 |
| [[adaptive-frequency-fusion|AFF]]（VoT） | 三带掩码分割预测谱 | 否（权重可学习） | 频域分量+可学习权重 |
| [[xcpd|XCPD]] | 可学习频率边界切低/中/高 | 是 | 频谱节点分组 |

## 使用边界

- 频带边界由均匀切分给定，不参与梯度更新；若要按数据分布切带需另加边界学习机制[^src-adafre]。
- 频带数 $P$ 是显式超参。AdaFre 报告 $P$ 从 2 增到 5 时 $P=4$ 最优，$P$ 过小使每个子带内部语义混合，过大则子带碎片化、单带信息不足[^src-adafre]。
- 该分解与"哪些频带真正被使用"是两件事：AdaFre 中 $P$ 个频带全部生成，只有被路由选中的 $K$ 个进入 backbone[^src-adafre]。

## 相关页面

- [[adafre]] — 使用该分解的模型
- [[frequency-pathway-routing]] — 频带之上的选择与融合
- [[frequency-specific-spatial-embedding]] — 与该分解配对的空间侧分组
- [[spectral-graph-wavelet-transform]] — 谱图小波变换
- [[graph-frequency-decomposition]] — 图频率分解

[^src-adafre]: [[source-adafre]]
