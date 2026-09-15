---
title: "Spatio-Temporal Self-Supervised Learning for Traffic Flow Prediction"
type: source-summary
tags:
  - spatio-temporal
  - self-supervised-learning
  - traffic-prediction
  - heterogeneity
  - graph-neural-network
  - contrastive-learning
created: 2026-09-15
last_updated: 2026-09-15
source_count: 1
confidence: medium
status: active
---

# ST-SSL: Spatio-Temporal Self-Supervised Learning for Traffic Flow Prediction

**作者**: Jiahao Ji, Jingyuan Wang 等（北航、京东城市研究院、HKU）
**发表**: AAAI 2023（arXiv:2212.04475）
**代码**: https://github.com/Echo-Ji/ST-SSL

## 核心论点

论文针对交通流预测中两类被共享参数空间忽略的异质性：空间异质性（不同区域流量分布偏斜，模型偏向高流量区域）与时间异质性（不同时段模式不同，但传统方法对所有时段共享参数）[^src-st-ssl]。解法是将两个辅助自监督任务加入主预测任务联合训练——注意这是单阶段联合训练范式，不是 [[std-mae]]（IJCAI 2024）那种"掩码预训练+微调"两阶段设计。

## 方法机制

1. **ST Encoder**: gated 1D 因果时间卷积 (TC) + 图卷积 (SC)，按 TC→SC→TC "sandwich" 块堆叠（继承 STGCN 结构），嵌入维度 D=64，卷积核 3。
2. **自适应图增强**: 用区域聚合嵌入的余弦相似度 $q_{m,n}$ 度量区域间异质性。流量级增强按 $\mathrm{Bern}(1-p_{\tau,n})$ 掩码与区域整体规律相关性低的时间步流量；拓扑级增强按 $\mathrm{Bern}(1-q_{m,n})$ 删低相关邻接边、按 $\mathrm{Bern}(q_{m,n})$ 加非邻接长程边。扰动比例均 0.1。
3. **空间 SSL**: 软聚类任务——增广图生成 K 个聚类嵌入作为伪标签，原始图嵌入做 cross-entropy 预测（温度 $\gamma$）；用单纯形约束 + 最大熵正则（Eq.7-8）避免所有区域塌缩到同一聚类。
4. **时间 SSL**: 对比任务——同一时间步的区域级与城市级嵌入为正对，不同时间步为负对，判别函数 $g=\sigma(v_{t,n}^\top W_3 s_t)$。
5. 联合损失 $L_{joint}=L_p+L_s+L_t$，$L_p$ 用 $\lambda$ 平衡 inflow/outflow。

## 实验结果

4 个数据集（7:1:2 划分，输入前 2 小时 + 前 3 天流量，预测下一时间步）：NYCBike1（2014.04-09，30min，16×8 网格）、NYCBike2（2016，30min，10×20）、NYCTaxi（2015.01-03，30min）、BJTaxi（2015.03-06，1h，32×32）。对比 8 个基线（ARIMA、SVR、ST-ResNet、STGCN、GMAN、AGCRN、STSGCN、STFGNN），LibCity 平台，5 seeds，t-test p<0.01 [^src-st-ssl]。

| 数据集 | ST-SSL MAE (In/Out) | 最优基线 |
|---|---|---|
| NYCBike1 | 4.94 / 5.26 | AGCRN 5.17 / 5.47 |
| NYCBike2 | 5.04 / 4.71 | AGCRN 5.18 / STGCN 4.79 |
| NYCTaxi | 11.99 / 9.78 | AGCRN 12.13 / 9.87 |
| BJTaxi | 11.31 / 11.40 | AGCRN 12.30 / 12.38 |

MAPE 同样全面占优（BJTaxi In 15.03% vs AGCRN 15.61%）。消融（随机增强替换自适应增强、去掉空间/时间 SSL 共 4 个变体）均劣于完整模型；误差可视化显示郊区区域提升显著，验证空间异质性建模在全局相似区域间迁移信息。

## 局限性

- 只预测下一时间步 $t+1$，无长时程预测实验[^src-st-ssl]
- 城市为固定 $I\times J$ 网格划分，依赖邻接矩阵，未在路网图数据集上验证
- 自监督任务作为辅助损失联合训练，无法像 STD-MAE 那样预训练后迁移到少样本/跨数据集场景
- 自适应增强引入超参（扰动比例 0.1）且未分析对超参敏感性

[^src-st-ssl]: [[source-st-ssl]]
