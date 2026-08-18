---
title: "Predicting the Future by Retrieving the Past (PFRP)"
type: source-summary
tags:
  - time-series-forecasting
  - retrieval-augmented
  - contrastive-learning
  - univariate
  - plug-and-play
  - aaai-2026
created: 2026-08-19
last_updated: 2026-08-19
source_count: 1
confidence: low
status: active
---

# Predicting the Future by Retrieving the Past (PFRP)

**Dazhao Du, Tao Han, Song Guo（2026），AAAI 2026（第 40 届）**。作者单位：HKUST CSE。

## 核心论点

深度学习时序预测模型（MLP、Transformer、TCN）在训练时隐式压缩历史信息到参数中，但推理时仅依赖固定长度回溯窗口（local context），无法显式访问全局历史模式。PFRP 提出通过构建 Global Memory Bank（GMB）显式存储历史样本，并在推理时检索相似历史模式生成全局预测，再与任意局部预测模型动态融合，提升单变量时序预测精度。[^src-predicting-the-future-by-retrieving-the-past-aaai2026]

## 方法

两阶段框架：

1. **GMB 构建**：提出 Predictive Contrastive Learning（PCL）训练 MLP 编码器——正样本选取基于预测区间序列的 MSE 最小而非回溯窗口相似度，使得"未来更相似"的回溯窗口在特征空间更接近。编码后对全部训练样本做 K-medoids 聚类，仅保留 K 个聚类中心（medoids）及其对应预测区间序列存入 GMB，降低冗余并提高检索效率。[^src-predicting-the-future-by-retrieving-the-past-aaai2026]

2. **检索与融合**：推理时编码当前回溯窗口为 query，与 GMB 中 K 个 key 计算余弦相似度取 top-k。引入 confidence gate（MLP+sigmoid 评估 query-value 拼接序列的存在概率，调制注意力权重）和 output gate（MLP 输出 scale α 和 shift β 调节全局预测的尺度偏移），生成全局预测 y₁。再通过 Dynamic Fusion 将 y₁ 与任意局部模型的 y₂ 加权融合，权重由调制后的 top-k 权重经 MLP+Softmax 生成。[^src-predicting-the-future-by-retrieving-the-past-aaai2026]

## 实验结果

在 Traffic、Electricity、Weather、ETTh1/h2/m1/m2 七个数据集上评估，L=96，H={96,192,336,720}，MSE/MAE 指标。PFRP 对 SparseTSF 和 DLinear 的平均提升分别为 8.4% 和 7.1%，对 PatchTST 和 TimesNet 提升略小。在周期性强的 Traffic 和 Electricity 上分别平均提升 17.4% 和 10.1%。[^src-predicting-the-future-by-retrieving-the-past-aaai2026]

与 RATD、RAFT 对比，PFRP 在各预测长度上 MAE 更优，推理速度优于 RAFT（固定大小 GMB 检索 vs 全训练集遍历）和 RATD（多步扩散采样）。PFRP 也可增强大型时序模型（TimeCMA、Moirai、Sundial），冻结预训练参数仅微调 PFRP 组件。[^src-predicting-the-future-by-retrieving-the-past-aaai2026]

## 局限性

1. 仅针对单变量 TSF，未扩展到多变量场景。[^src-predicting-the-future-by-retrieving-the-past-aaai2026]
2. 在周期性弱的数据集（ETT 系列）上提升有限（1.6%–3.4%），全局预测权重与数据周期性正相关。[^src-predicting-the-future-by-retrieving-the-past-aaai2026]
3. GMB 构建需要 PCL 训练 + K-medoids 聚类（论文报告 Electricity 数据集上固定 186 秒），且 K 值需调参。[^src-predicting-the-future-by-retrieving-the-past-aaai2026]

[^src-predicting-the-future-by-retrieving-the-past-aaai2026]: [[source-predicting-the-future-by-retrieving-the-past-aaai2026]]
