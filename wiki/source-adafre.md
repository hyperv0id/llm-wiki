---
title: "AdaFre: Adaptive Frequency Pathways for Spatiotemporal Forecasting (Qin et al., AAAI 2026)"
type: source-summary
tags:
  - spatio-temporal
  - traffic-prediction
  - frequency-decomposition
  - spectral-embedding
  - adaptive-routing
created: 2026-09-16
last_updated: 2026-09-16
source_count: 1
confidence: medium
status: active
---

# AdaFre: Adaptive Frequency Pathways for Spatiotemporal Forecasting

**作者**：Yanjun Qin¹、Yuchen Fang²（两人共同一作）、Xinke Jiang²、Hao Miao³（通讯）、Xiaoming Tao¹⁴。¹新疆大学，²电子科技大学（成都），³香港理工大学，⁴清华大学。[^src-adafre]
**Venue**：AAAI-26（PDF 页眉 "The Fortieth AAAI Conference on Artificial Intelligence"，页码 15653–15661，共 9 页）。[^src-adafre]
**源文件**：`raw/adaptive-frequency-pathways-spatiotemporal-forecasting-aaai26.pdf`。
Yuchen Fang 同为 [[source-stwave|STWave]] 一作；本文参考文献含 Fang et al. 2023a（STWave, ICDE, 517–529）与 2023b（STWave+, TKDE, 2671–2685）。[^src-adafre]

## 问题

论文把现有频域时空模型的不足归为两类失配[^src-adafre]：

1. **Temporal Frequency Dynamics**：频率分量的重要性随上下文变化，而多数模型对所有频率做固定权重的融合。证据是 Figure 1(b)——PeMSD4 单个传感器的频率能量在一天内变化，清晨低频主导，傍晚高频能量出现尖峰。
2. **Frequency-Specific Spatial Dependency**：不同时间频率对应不同尺度的空间交互，低频反映全局相干现象，高频对应局部事故或传感器故障；多数模型仍在所有时间分量上共享同一张空间图。证据是 Figure 1(c) 的时-空频率索引联合分布，时间高频分量与更高的空间频率索引共现。

## 方法机制

- **分带时域分解**：对时间轴做 DFT（$\hat{X}=\mathrm{FFT}(X,\dim=0)$），实信号 Hermitian 对称，只有前 $\lfloor T/2\rfloor+1$ 个 bin 独立有效；把有效范围切成 $P$ 个连续频带 $I_p$，带外系数置零后 IFFT 回时域得 $H^{(p)}=\mathrm{Re}(\mathrm{IFFT}(\hat{X}_{I_p}))$，共得 $P$ 个 band-limited 时域视图。论文自述相对 wavelet 或学习式分解的优势是把分量分到独立频率通道，可直接解释与操作。[^src-adafre] 见 [[band-limited-temporal-decomposition]]。
- **频率感知图谱嵌入**：归一化拉普拉斯 $L=U\Lambda U^\top$，按特征值从小到大把特征向量切成 $P$ 组 $U^{(p)}=U[:,\hat{I}_p]$，与时间频带对齐；$U^{(1)}$ 捕获全局空间模态，$U^{(P)}$ 捕获高频局部模式。[^src-adafre] 见 [[frequency-specific-spatial-embedding]]。
- **多频路由**：$R=g_\phi(X)\in\mathbb{R}^{N\times P}$ 按节点对 $P$ 个频率打相关性分，温度 softmax 得 $\hat{r}_{ip}$，在频率维取 top-$K$（默认 $K=2$）索引 $\tilde{I}$；每个被选频率把 $(H^{(\tilde{I}_k)}, U^{(\tilde{I}_k)})$ 送入对应 backbone，得 $\hat{Y}^{(\tilde{I}_k)}$。[^src-adafre] 见 [[frequency-pathway-routing]]。
- **频率感知融合**：softmax 权重 $\alpha_k$ 加权聚合所选分支，$\sum\alpha_k=1$。[^src-adafre]
- **backbone 与损失**：backbone 用 [[stid]]（STID），论文自述目的是凸显频率自适应机制的增益、避开复杂 backbone 的混杂效应。总目标 $L_{\text{total}}=L_{\text{pred}}+L_{\text{bal}}$，其中 $L_{\text{pred}}$ 作用于融合后的最终预测，$L_{\text{bal}}$ 惩罚各频率在 mini-batch 上的平均选择概率偏离 $1/P$。[^src-adafre] 见 [[frequency-balance-loss]]。

## 实验证据

四个 Caltrans PeMS 数据集（PeMSD3/4/7/8，358/307/883/170 节点，5 min，26208/16992/28224/17856 步）；输入 288 步（1 天）预测 12 步（1 小时），60/20/20 时序切分，Z-Score 归一化，邻接矩阵由站点物理距离 + Gaussian kernel 构造；100 epochs、Adam、初始 lr 0.001，第 2/50/80 epoch 减半；$P=4$、$K=2$。[^src-adafre]

对 22 个基线，AdaFre 在四个数据集上 MAE 与 RMSE 均最低；MAPE 在 PeMSD3/4/7 最低，在 PeMSD8 与 STAEFormer 同值 8.88%（表中 AdaFre 标为 best、STAEFormer 标为 second best）。三指标为 14.27 / 24.22 / 15.04%、17.89 / 29.76 / 11.97%、18.58 / 31.55 / 7.85%、12.99 / 22.32 / 8.88%。次优 MAE 分别为 STDN 14.89（PeMSD3，STWave 14.93、PDFormer 14.94）、HimNet 18.14（PeMSD4，STAEFormer 18.22）、STAEFormer 19.14（PeMSD7，HimNet 19.21）、STWave 13.42（PeMSD8，STAEFormer 13.46）。[^src-adafre]

消融三项分别去掉频率分解（w/o FD）、让所有频率共享同一谱嵌入（w/o FSE）、去掉自适应路径选择与重要性加权（w/o AP）。全模型在四个数据集上均优于三个变体，PeMSD3 MAE 14.27 对 14.51 / 14.44 / 14.37，PeMSD8 12.99 对 13.50 / 13.26 / 13.41。[^src-adafre]

超参分析（Figure 3）：输入长度 $T\in\{144,288,576,864\}$ 中 288 最优；频带数 $P$ 从 2 到 5 中 $P=4$ 最优，论文解释 $P$ 过小使每个子带语义混合、过大则子带碎片化且信息不足。$K$ 虽在同句列为 "systematic study" 的对象，正文与 Figure 3 标题均只给出输入长度与 $P$ 的结论。[^src-adafre]

效率（Table 3，PeMSD4）：337K 参数、14 ms/iter、23 s/epoch、2137 MB；对照 STID 123K/10ms/15s/1679MB、GWN 309K/48ms/69s/2046MB、AGCRN 763K/63ms/102s/2679MB、HimNet 11B/103ms/189s/6784MB、STDN 3B/73ms/118s/3533MB。[^src-adafre]

## 正文与公式、表格的不一致（如实记录）

论文存在若干记法与自述不一致处，此处分别归因，不代为择定[^src-adafre]：

- 式(13) 写作 $L_{\text{pred}}=\|\hat{Y}-Y\|^2$（平方误差），同一段正文写 "we apply the mean absolute error (MAE) loss"。
- 式(10) 的温度 softmax 分母写作 $\sum_{j=1}^{K}$，而该式语义是对 $P$ 个频率的归一化。
- 式(14) 对 batch 与节点双重求和（$\sum_i\sum_j$），归一化因子只除以 $B$。
- 式(8) 的说明文字写 "each index set $\hat{I}_p$ covers $P$ consecutive eigenvectors"，按 $P$ 组覆盖 $N$ 个特征向量应为 $N/P$。
- 正文称 "STWave (Fang et al. 2023b)"，但参考文献中 2023b 是 STWave+（TKDE），STWave 为 2023a（ICDE）。
- 正文称相对 STAEFormer、PDFormer、STFGNN "requires significantly less computational resources"，Table 3 并未列出这三个方法的数据。

## 范围与边界

论文在结论中以 "We believe" 归因地陈述 AdaFre 是可推广到时空预测之外更广时序任务的范式，这属于论文自述，本文未提供时空预测之外的实证[^src-adafre]。评估局限在 4 个交通速度数据集与单一输入/输出长度设定；消融未把 $P$ 与 $K$ 的作用分开，也未检验图频率分组数是否必须与 $P$ 相等；频带边界由均匀切分给定，不参与学习（对照 [[xcpd|XCPD]] 的可学习频率边界）[^src-adafre]。论文无独立 Limitations 章节。

## 交叉链接

- [[adafre]] — 模型页
- [[band-limited-temporal-decomposition]] — 分带时域分解
- [[frequency-specific-spatial-embedding]] — 频率特异谱嵌入
- [[frequency-pathway-routing]] — 频率路径路由
- [[frequency-balance-loss]] — 频率均衡损失
- [[source-stwave]] — 同作者谱系的 DWT 解耦路线
- [[traffic-forecasting]] — 交通预测总览

[^src-adafre]: [[source-adafre]]
