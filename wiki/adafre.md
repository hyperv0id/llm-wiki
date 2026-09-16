---
title: "AdaFre"
type: entity
tags:
  - spatio-temporal
  - traffic-prediction
  - frequency-decomposition
  - spectral-embedding
  - adaptive-routing
  - aaai-2026
created: 2026-09-16
last_updated: 2026-09-16
source_count: 1
confidence: medium
status: active
---

# AdaFre

**AdaFre**（Adaptive Frequency Pathways）是 Qin、Fang 等提出的时空预测模型，发表于 AAAI 2026，把输入序列按时间频率分带，为每个频带配对一张相应尺度的图谱嵌入，再用可学习的路由在频带间做稀疏选择与加权聚合[^src-adafre]。

## 定位

AdaFre 处理两类它认为既有频域时空模型未处理的失配：频率重要性随时间上下文变化，模型却对所有频率固定加权；空间依赖随频率变化，模型却在所有时间分量上共享同一张空间图[^src-adafre]。它的做法是把"频率"从分解层一直贯穿到空间建模层与模型选择层：时间侧分带、空间侧按图频率分组、路径侧按上下文稀疏激活。

## 架构

1. **分带时域分解**：DFT 后把独立频带 $I_p$ 之外置零并 IFFT 回时域，得到 $P$ 个 band-limited 视图 $H^{(1)},\dots,H^{(P)}$[^src-adafre]。
2. **频率感知图谱嵌入**：拉普拉斯特征向量按特征值切分为 $P$ 组，$U^{(p)}$ 与 $H^{(p)}$ 一一配对[^src-adafre]。
3. **多频路由与稀疏激活**：$g_\phi(X)$ 给出每节点对 $P$ 个频率的相关性，温度 softmax 后取 top-$K$（默认 $K=2$），仅被选中的频率分支参与前向计算[^src-adafre]。
4. **频率特异 backbone**：每个被选频率用独立 backbone 编码 $(\hat{H}^{(\tilde{I}_k)}, U^{(\tilde{I}_k)})$；backbone 取 [[stid|STID]]，论文自述是为了凸显频率机制增益、避免复杂 backbone 的混杂效应[^src-adafre]。
5. **频率感知融合**：softmax 权重对所选分支加权求和[^src-adafre]。
6. **训练目标**：$L_{\text{total}}=L_{\text{pred}}+L_{\text{bal}}$，均衡项抑制路由坍缩到少数频率[^src-adafre]。

## 结果

PeMSD3/4/7/8 四数据集、输入 288 步预测 12 步、22 个基线下 MAE 与 RMSE 全部最低，MAPE 除 PeMSD8 与 STAEFormer 并列 8.88% 外均最低（MAE 14.27 / 17.89 / 18.58 / 12.99）[^src-adafre]。PeMSD4 上参数量 337K、23 s/epoch，低于 AGCRN（763K/102s）与 HimNet（11B/189s）[^src-adafre]。三项消融（去分带、共享谱嵌入、去自适应路径）均掉点，论文据此归因于两个机制各自有效[^src-adafre]。论文自述其可推广性到更广的时序任务，未在本文以外验证[^src-adafre]。

## 关联页面

- [[source-adafre]] — 源文件摘要（含正文与公式不一致的记录）
- [[band-limited-temporal-decomposition]] — 分带时域分解
- [[frequency-specific-spatial-embedding]] — 频率特异谱嵌入
- [[frequency-pathway-routing]] — 频率路径路由
- [[frequency-balance-loss]] — 频率均衡损失
- [[graph-frequency-decomposition]] — 图频率分解（HiFiNet 的层次粗化路线）
- [[spectral-graph-wavelet-transform]] — 谱图小波变换
- [[source-stwave]] — STWave 的 DWT 解耦 + ESGAT 路线
- [[xcpd]] — 谱域 channel-patch 依赖路由（时间序列，非时空图）
- [[traffic-forecasting]] — 交通预测总览

[^src-adafre]: [[source-adafre]]
