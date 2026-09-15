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

**作者**: Jiahao Ji, Jingyuan Wang 等（北航、鹏城实验室、HKU）
**发表**: AAAI 2023（arXiv:2212.04475）
**代码**: https://github.com/Echo-Ji/ST-SSL

## 核心问题

共享参数空间抹平两类异质性：空间上，区域流量分布偏斜，模型偏向高流量区域（L86-88）；时间上，所有时段共享一套参数，分参数策略又假设时段模式静态（L100-108）[^src-st-ssl]。论文自称首次以自监督框架建模交通流预测的时空异质性（L119-120）[^src-st-ssl]。

## 方法机制

1. **ST encoder**：门控 1D 因果时间卷积沿用 STGCN（L167-168）+ 图消息传递 SC，TC→SC→TC sandwich 块（L187-188）；D=64，核均为 3（L664-665）[^src-st-ssl]。
2. **自适应图增强**：区域聚合嵌入余弦相似度 $q_{m,n}$ 度量异质性（Eq.4）[^src-st-ssl]。流量级按 $\mathrm{Bern}(1-p_{\tau,n})$ 掩掉低相关时间步流量（L315-321）；拓扑级按 $\mathrm{Bern}(1-q_{m,n})$ 删相邻低相关边、按 $\mathrm{Bern}(q_{m,n})$ 加非邻接边（L327-334）[^src-st-ssl]。扰动比例均 0.1（L665）[^src-st-ssl]。
3. **空间 SSL**：增广图嵌入生成 K 个软聚类伪标签，原图嵌入经温度 γ softmax 交叉熵预测聚类分配（Eq.5-6）；单纯形约束 + 最大熵正则防聚类塌缩（Eq.7-8）[^src-st-ssl]。
4. **时间 SSL**：同一时间步的区域级与城市级嵌入为正对、跨时间步为负对，判别函数 $g=\sigma(v_{t,n}^\top W_3 s_t)$（Eq.11）[^src-st-ssl]。
5. 联合损失 $L_{joint}=L_p+L_s+L_t$（L552），$L_p$ 用 λ 平衡 inflow/outflow[^src-st-ssl]。

## 实验结果

前 2 小时 + 前 3 天流量预测下一时间步，7:1:2 划分（L580-584）[^src-st-ssl]。NYCBike1（16×8，6.8k+）、NYCBike2（10×20，2.6m+）、NYCTaxi（10×20，22m+）均 30 min，BJTaxi（32×32，34k+）1h（L573-579；Table 1 的 interval 行与之矛盾，以正文为准）[^src-st-ssl]。8 基线三类（ARIMA/SVR；ST-ResNet/STGCN/GMAN；AGCRN/STSGCN/STFGNN），LibCity、5 seeds、t-test 0.01 显著（L673-676）[^src-st-ssl]。

Table 2 核算（列序 ARIMA|SVR|ST-ResNet|STGCN|GMAN|AGCRN|STSGCN|STFGNN|ST-SSL，已逐列扫描）：

| 数据集 | ST-SSL MAE (In/Out) | 最优基线 |
|---|---|---|
| NYCBike1 | 4.94 / 5.26 | AGCRN 5.17 / 5.47[^src-st-ssl] |
| NYCBike2 | 5.04 / 4.71 | AGCRN 5.18 / 4.79（STGCN Out 4.92 次优）[^src-st-ssl] |
| NYCTaxi | 11.99 / 9.78 | AGCRN 12.13 / 9.87[^src-st-ssl] |
| BJTaxi | 11.31 / 11.40 | ST-ResNet 12.12 / 12.16（AGCRN 12.30 / 12.38 次优）[^src-st-ssl] |

MAPE-In 最优基线：NYCBike1=SVR 25.39、NYCBike2=AGCRN 27.14、NYCTaxi=AGCRN 18.78、BJTaxi=ST-ResNet 15.50（AGCRN 15.61 非最优，L775）[^src-st-ssl]。

消融：散文写 "five variants"（L847）但仅枚举 4 个（-sa 随机删加边、-ta 随机流量掩码、-sh/-th 去空间/时间异质性，L848-856），系论文笔误；四变体均劣于完整模型[^src-st-ssl]。误差可视化显示郊区提升显著（L682-684）；RQ4 展示删边/加边实例（L985-988）[^src-st-ssl]。

## 范式对照（页面评述）

与 [[source-2312-00516-std-mae]] 对照：ST-SSL 单阶段联合训练，SSL 损失全程在线约束表示空间；STD-MAE（IJCAI 2024）两阶段掩码预训练+微调，可迁移[^src-st-ssl]。切入点不同：前者显式建模区域/时段聚类与对比结构，后者重建被掩码流量[^src-st-ssl]。

## 局限性

- **论文自认**：无独立局限小节；未来工作仅提框架 model-agnostic 化（L1021），暗示绑定该 ST encoder[^src-st-ssl]。
- **页面评述**：仅预测下一时间步（L149-150）；固定 I×J 网格 + 邻接矩阵，未在路网图验证（L134-148）；超参敏感性未报告；联合训练无法跨数据集迁移[^src-st-ssl]。

[^src-st-ssl]: [[source-st-ssl]]
