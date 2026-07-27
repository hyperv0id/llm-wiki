---
title: "ST-OOD Benchmark"
type: entity
tags:
  - spatio-temporal
  - out-of-distribution
  - traffic-forecasting
  - urban-computing
  - benchmark
  - generalization
created: 2026-07-27
last_updated: 2026-07-27
source_count: 1
confidence: medium
status: active
---

# ST-OOD Benchmark

**ST-OOD**（Spatiotemporal Out-of-Distribution）是 Wang 等人在 *IEEE TMC* 2025 提出的城市时空**跨年泛化**评测基准：六个真实城市场景，统一 **同年 IN / 次年 OUT** 同日历窗协议，用于回答“SOTA 时空模型换一年还能否用”[^src-st-ood]。代码与数据协议见 `github.com/Dreamzz5/ST-OOD`[^src-st-ood]。

## 与现有基准的差异

DL-Traff、BasicTS+、LargeST、LibCity、UCTB 等侧重标准化与短跨度公平比较；即便 LargeST 覆盖多年，也**不显式**定义自然时间 OOD 协议[^src-st-ood]。ST-OOD 的主张是：评测必须把**年际演化**（基建、政策、疫情、出行习惯）当成一等公民，而不是默认图关系在数周内静止[^src-st-ood]。

在 [[ood-generalization|OOD]] 二分法下，ST-OOD 主测的是 **T-OOD（时间分布偏移）**，并辅以 Kendall’s τ / DTW 刻画跨年空间秩与时间形状变化；邻接由 haversine 阈值（或行政区拓扑）重建，**不是** [[stop|STOP]] / [[stunet|STUNet]] 主叙事的传感器增删或跨城零样本图迁移[^src-st-ood]。

## 协议要点

- **六场景：** Zurich 行人、Chicago 出租车需求、Chicago 共享单车、NYC 车速、PEMS08 流量、NYC 311 请求[^src-st-ood]。
- **时间切分：** 年 *Y* 1/1–10/19 训练（内 6:2:2）；*Y* 10/20–12/31 = IN；*Y+1* 同日历窗 = OUT——对齐季节、隔离年际漂移[^src-st-ood]。
- **图：** 单位间 haversine ≤500 m 连边（对 300–700 m 敏感分析：相对排序与退化模式稳定，最优点约 450–550 m）；311 用社区区地理邻接[^src-st-ood]。
- **指标：** 12 步预测的 MAE / RMSE / MAPE[^src-st-ood]。

## 评测结论（压缩）

| 观察 | 证据摘要 |
|------|----------|
| 全模型跨年崩 | OUT RMSE 相对 IN 升 **40%–116%**[^src-st-ood] |
| 简单 > 复杂（OUT） | STID / MLP 平均 OUT 排名优于多数 GNN/Transformer[^src-st-ood] |
| 专用 OOD 方法 | CaST / CauSTG / STONE：相对退化更小，但 IN+OUT 绝对误差更高 → underfitting 假象[^src-st-ood] |
| 自适应图 / 节点 embedding | 抬 IN、伤或绑架 OUT[^src-st-ood] |
| 正则 | STID 上 dropout **0.2–0.3** 显著抬 OUT（FlowSC 上 RMSE 约 −55%），对 IN 几乎无伤[^src-st-ood] |

标准模型平均归一化 MAE：IN 1.07 → OUT 1.68（**×1.57**）；专用 OOD：IN 1.47 → OUT 1.87（**×1.27** 但全程更高）[^src-st-ood]。

## 在方法谱系中的位置

- 相对 [[source-cast|CaST]] / CauSTG / STONE：ST-OOD 是**评测器**，实证质疑“时空因果分离 / 不变学习”在真实多年城市数据上的交付——它们往往用降容量换稳健，而非不变表示[^src-st-ood]。
- 相对 [[stop|STOP]]：STOP 主攻**切断 node-to-node messaging** 以抗结构/时间耦合脆弱性；ST-OOD 不提出新架构，而是用自然跨年协议量出“复杂 STGNN 过拟合训练年”的基线事实[^src-st-ood]。
- 相对 [[stunet|STUNet]]：STUNet 主攻**跨网络零样本**（显式邻接 token + 冻结）；ST-OOD 主攻**同城跨年**。两者都指向“过参数化 + 隐式结构/时间纠缠”的泛化风险，但设定与指标不同[^src-st-ood]。

## 设计启示

论文收束为：最小归纳偏置、适度 dropout、物理/地理约束、稳定特征、简单集成；并呼吁发展**时空纠缠**的 OOD 理论，而非把图 OOD 与时序 OOD 硬拆开[^src-st-ood]。生产侧含义是持续再训练与稳健性监控优先于历史 IN 上的边际精度[^src-st-ood]。

## 相关页面

- [[source-st-ood]] — 源摘要
- [[ood-generalization]] · [[spatio-temporal-ood-learning]] · [[traffic-forecasting]]
- [[source-cast]] · [[stop]] · [[stunet]] · [[gwnet]] · [[stgcn]]

[^src-st-ood]: [[source-st-ood]]
