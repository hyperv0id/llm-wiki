---
title: "UniST: A Prompt-Empowered Universal Model for Urban Spatio-Temporal Prediction"
type: source-summary
tags:
  - spatio-temporal
  - foundation-model
  - prompt-learning
  - masked-autoencoder
  - traffic-forecasting
  - zero-shot
created: 2026-05-31
last_updated: 2026-11-07
source_count: 1
confidence: high
status: active
---

# UniST: A Prompt-Empowered Universal Model for Urban Spatio-Temporal Prediction

Yuan Yuan, Jingtao Ding, Jie Feng, Depeng Jin, Yong Li（清华电子工程系 + BNRist），KDD 2024（DOI 10.1145/3637528.3671662；arXiv:2402.11838；raw/unist-yuan-2024.pdf）[^src-unist]。论文自述 universal 时空预测的 first attempt：单一 one-for-all 模型覆盖 21 个数据集（14 城 6 域）。

## 问题与机制

现有方法逐数据集训练且依赖大量域内数据；朴素联合训练失效——PatchTST(one-for-all) 在全部短期数据集劣于专用版（Table 2：Crowd RMSE $10.25\to 13.51$、BikeNYC $5.27\to 9.97$）。UniST 两阶段：①掩码重建预训练——channel-independent 拆分后 Conv3D patch 化（patch $(2,2,2)$、掩码率 0.5、隐藏维 128），每 mini-batch 随机选 random/tube/block/temporal 四种掩码之一，各练一种能力，被掩 patch 上 MSE 重建；预测即 temporal 掩码重建（掩码率 pred/(his+pred)）。②知识引导 prompt——从输入提取空间近邻/层级、时间近邻/周期四类知识表示，softmax 查询两个 512×128 的 key-value memory pool 生成逐样本 prompt，逐元素加到 decoder 输入端的未来 token；微调冻结骨干、只训 prompt 网络（Adam，lr 3e-4/5e-5）。

## 证据

短期 6→6（Table 2，Crowd/Cellular ×10³）：TaxiBJ 26.84 vs STID 27.36；Crowd 3.00 vs PromptST 3.52（↓14.8%）；Cellular 14.29 vs 15.74；BikeNYC 3.50 vs ACFM 3.93；TrafficSH 0.665 vs 0.742。长期 64→64（Table 3）：TaxiNYC 19.83 vs SimVP 20.18，平均改进 10.1%。few-shot（arXiv App.）：Crowd 1% 13.952 超最强基线 MetaST 14.242；zero-shot Crowd 14.665 超 ACFM 1%-shot 21.166。完整基准中 TrafficBJ、TrafficWH（RMSE）、BikeNYC-2、TDrive 输给基线（arXiv App.）；BikeNYC/TaxiBJ 的 zero-shot 分别输给多个 1% 基线。效率 6.71M 参数、1.4 min/epoch（21 数据集一轮），训练时长约为 STID 的 26 倍。

## 局限

仅网格数据（grid/sequence/graph 统一列为 future work）；参数 2M→30M 边际收益递减（归因时空数据多样性低）；知识提取网络朴素、可替换（如 Fourier 方法）。

机制细节与全部数字见 [[unist]]。

[^src-unist]: [[source-unist]]
