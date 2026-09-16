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
last_updated: 2026-09-16
source_count: 2
confidence: medium
status: active
---

# STWave: Disentangled Traffic Flow Forecasting via Efficient Spectral Graph Attention Network

**Authors**: Yuchen Fang¹, Yanjun Qin¹*, Haiyong Luo²†, Fang Zhao¹†, Bingbing Xu³, Chenxing Wang¹, Liang Zeng⁴（¹BUPT；²中科院计算所；³中科院计算所网数重点实验室；⁴清华 IIIS）
**Venue**: 依据 arXiv:2112.02740v2（2022-01-17）文本；文本未出现 ICDE/TKDE 等出版信息，旧页所记 ICDE 2023 页码与 TKDE 扩展版内容无法核验，已删除。[^src-stwave]代码链接文本未提供。[^src-stwave]

**书目信息（第三方佐证）**: [[source-adafre|AdaFre]]（AAAI 2026）的参考文献列出 Fang et al. 2023a "When spatio-temporal meet wavelets: Disentangled traffic forecasting via efficient spectral graph attention networks"，In ICDE, 517–529，作者序 Fang, Qin, Luo, Zhao, Xu, Zeng, Wang，与本页作者列表一致；同组的 2023b 为 STWave+（TKDE, 2671–2685）。该条只佐证出版信息，不改变本页对 arXiv 文本内容的记录。[^src-adafre]

## 问题
交通序列把短期波动与长期日趋势纠缠在单一方法里建模；全图 attention 时间与空间复杂度 O(N²)，vanilla GAT 又把感受野限死在邻居、缺图结构先验[^src-stwave]。

## 方法机制

- **Disentangling Flow Layer**：DWT 以低通滤波 g、高通滤波 h 分解低频（稳定、长趋势）与高频（波动、短影响）分量；↓2 下采样后步数减半，逆滤波 g^T、h^T 上采样复原，FC 升维[^src-stwave]。
- **双通道编码器**：高频走 dilated causal convolution（核 J=2），低频走 masked temporal self-attention；两通道各自再做 full GAT 空间建模，即 ESGAT[^src-stwave]。
- **ESGAT**：①Query Sampling——GAT 传消息打分，topk-pooling 取 ⌈logN⌉ 个最大流量节点作稀疏 query，未采样节点的注意力权重**从与它之间注意力权重最高的采样节点拷贝**（功能相似仅为解释性论证）；②图位置编码 ρ=Φ·G_s^{1/2}，Laplacian 特征向量经扩散缩放矩阵（σ(sλ_i)=e^{λ_i s}），空间/时序图各一份，尺度 s 可学习；对比 EV 注意力更密、N2V 更疏，图小波在结构性与局部性间平衡[^src-stwave]。

## "Long-term" 的语义

指**时间模式（依赖）的类型**，非长时程预测：输入输出仍是前 12 步预测后 12 步[^src-stwave]。

## 实验证据
4 个 Caltrans PeMS 数据集（358/307/883/170 节点）对比 15 基线：STWave MAE 14.93/18.50/19.94/13.42 全面最优（AGCRN 15.98/19.83/22.37/15.95；PeMSD3 上 STGODE 16.50）；PeMSD4 RMSE 30.39、MAPE 12.43%。[^src-stwave]消融：去 ESGAT（-S）降幅远超去时间模块（-T），空间维度贡献更大。[^src-stwave]效率（RQ4）：速度与性能折中较好、内存合理，LSGCN 各方面最差。[^src-stwave]复杂度记法原文如此：编码器 O(L(TNJ+NT²+TN logN))，ESGAT 为 O(TN logN)——论文只称与 GCN 系框架"相当"[^src-stwave]。

## 范围与局限（页面评述，论文无 Limitations/Future Work 章节，结论仅正面陈述）
one-level DWT 只分两个频段，无中间尺度；权重拷贝依赖功能相似假设，论文未分析失效行为；基线截至 2021，不含后续频域路线。[^src-stwave]与 [[source-patchstg]] 同属稀疏 query 降注意力开销路线（本页评述，非论文自述）。

[^src-stwave]: [[source-stwave]]
[^src-adafre]: [[source-adafre]]
