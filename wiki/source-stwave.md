---
title: "Spatio-Temporal meets Wavelet: Disentangled Traffic Flow Forecasting via Efficient Spectral Graph Attention Network (STWave)"
type: source-summary
tags:
  - spatio-temporal
  - traffic-prediction
  - wavelet
  - graph-attention
  - frequency-decomposition
created: 2026-09-15
last_updated: 2026-09-15
source_count: 1
confidence: medium
status: active
---

# STWave: Disentangled Traffic Flow Forecasting via Efficient Spectral Graph Attention Network

**Authors**: Yuchen Fang, Yanjun Qin, Haiyong Luo, Fang Zhao, Bingbing Xu, Chenxing Wang, Liang Zeng (BUPT, ICT-CAS, THU)
**Venue**: arXiv:2112.02740 (Dec 2021), published as ICDE 2023 (pp. 517-529)；期刊扩展版 STWave+ 发表于 TKDE 2023 (DOI 10.1109/TKDE.2023.3324501)
**Code**: https://github.com/LMissher/STWave

## 问题

交通流序列把短期波动（雷暴式突变）与长期趋势（daily trend）纠缠在一起，既有端到端网络用单一模块同时建模两类模式，全图 GAT 复杂度 O(N²T) 且缺图结构先验[^src-stwave]。

## 方法机制

- **Disentangling Flow Layer**：one-level DWT（discrete wavelet transform）用低通滤波 g、高通滤波 h 分解出低频趋势与高频波动分量，↓2 下采样使时间步减半，逆滤波 g^T、h^T 上采样复原[^src-stwave]。
- **Dual-Channel Encoder**：高频通道用 dilated causal convolution（kernel J=2）捕获短期波动，低频通道用 masked temporal self-attention 捕获长期趋势，两通道都经 ESGAT（Efficient Spectral Graph Attention Network）做全局空间建模[^src-stwave]。
- **ESGAT**：Query Sampling 用 GAT 传消息打分、topk-pooling 采 ⌈logN⌉ 个活跃节点作 sparse query，未采样节点复用最相似采样节点的注意力权重，复杂度降至 O(TN logN)；graph positional encoding 取图小波基 ρ=Φ·G_s^{1/2}（Laplacian 特征向量 Φ 加扩散缩放），同时注入空间图与时序图的结构信息和局部性，缩放 s 可学习——对比中 Laplacian 特征向量（EV）注意力过密、node2vec（N2V）过疏[^src-stwave]。
- **Frequency-Specific Decoder**：fusion attention 以低频分量为 query 融合低/高频预测，multi-supervision 用 L1 loss 同时监督交通流与低频分量[^src-stwave]。

## "Long-term" 的语义

指**长期依赖而非长时程预测**：输入输出仍是 12 步预测 12 步（1 小时），长期趋势来自低频分量，经 multi-supervision 注入[^src-stwave]。TKDE 2023 扩展版 STWave+ 用 multi-level DWT 从 1 小时外长历史提取长期趋势，以 contrastive loss 对齐长短趋势表征，位置编码升级为 multi-scale graph wavelet positional encoding（MS-ESGAT，路级/区域级多尺度）[^src-stwave]。

## 实验证据

4 个 Caltrans PeMS 数据集（PeMSD3 358 / PeMSD4 307 / PeMSD7 883 / PeMSD8 170 节点）对比 15 个基线，MAE 全面最优：PeMSD3 14.93（AGCRN 15.98、STGODE 16.50），PeMSD4 18.50（AGCRN 19.83），PeMSD7 19.94（AGCRN 22.37），PeMSD8 13.42（AGCRN 15.95）；PeMSD4 RMSE 30.39、MAPE 12.43%[^src-stwave]。消融中 -S（去 ESGAT）降幅大于 -T（去时间模块），空间维度贡献更大[^src-stwave]。

## 局限

one-level DWT 只分两个频段；图小波位置编码依赖 Laplacian 特征分解，大图开销大；基线截至 2021（AGCRN、STGODE、STFGNN），未含后续频域模型[^src-stwave]。

[^src-stwave]: [[source-stwave]]
