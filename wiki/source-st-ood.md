---
title: "ST-OOD: Evaluating the Generalization Ability of Spatiotemporal Models in Urban Scenarios"
type: source-summary
tags:
  - spatio-temporal
  - out-of-distribution
  - traffic-forecasting
  - urban-computing
  - benchmark
  - generalization
  - ieee-tmc
created: 2026-07-27
last_updated: 2026-07-27
source_count: 0
confidence: low
status: active
---

# ST-OOD: Evaluating the Generalization Ability of Spatiotemporal Models in Urban Scenarios

**Authors:** Hongjun Wang, Jiyuan Chen, Tong Pan, Zheng Dong, Renhe Jiang, Xuan Song (SUSTech / UTokyo / Jilin University) · **Venue:** *IEEE Transactions on Mobile Computing*, Vol. 24, No. 12, Dec 2025 · **DOI:** 10.1109/TMC.2025.3590606 · **Code:** `github.com/Dreamzz5/ST-OOD` · **raw:** `raw/st-ood-evaluating-generalization-spatiotemporal-urban.pdf`

## 核心问题

现有城市时空基准（DL-Traff、BasicTS+、LargeST、LibCity 等）多在训练后数周到数月的近 IID 区间评测，不显式做**跨年自然分布偏移**。城市基础设施、政策与出行行为持续演化，SOTA STGNN 是否真正泛化仍未知。论文提出 **[[st-ood|ST-OOD]]**——首个面向**自然时间 OOD** 的多场景城市时空基准，并系统评测 12 类模型。

## 基准协议

六个城市场景（同日历窗控制季节性）：

| 场景 | 数据源 | 训练年 / OOD 年（示意） |
|------|--------|-------------------------|
| Pedestrians | Zurich 自动计数站 | 2019 → 2020 |
| Taxi demand | Chicago 社区区 pickup | 2013 → 2014 |
| Bike-sharing | Chicago Divvy | 2013 → 2014 |
| Traffic speed | NYC TMC 传感器 | 2019 → 2020 |
| Traffic flow | PEMS08 (CA) | 2016 → 2017 |
| 311 services | NYC 社区区请求量 | 2019 → 2020 |

统一划分：年 *Y* 的 1/1–10/19 训练（内再 6:2:2 训/验/测）、*Y* 的 10/20–12/31 为 **IN** 测、*Y+1* 同日历窗为 **OUT** 测。邻接默认 haversine ≤500 m（311 用行政区拓扑）。用 Kendall’s τ（空间秩相关）与 DTW（时间形状距离）刻画跨年偏移：如 COVID 下 Zurich 行人时间模式相对稳、空间差增大；NYC 311 空间稳、时间量级波动大。

## 主要发现

1. **跨年退化剧烈**：OUT 相对 IN 的 RMSE 升幅约 **40.06%（Speed NYC）–116.44%（Flow SC）**；多数复杂模型 OUT 上甚至不如简单 **MLP**。
2. **简单架构更耐 OOD**：STID 式时空 identity + 纯 MLP 在多数 OUT 集上平均排名靠前；复杂图/注意力（含 D²STGNN、[[staeformer|STAEformer]]）IN 强、OUT 弱——作者归因于参数过拟合训练期伪相关。
3. **专用 ST-OOD 方法未胜出**：[[source-cast|CaST]]、CauSTG、STONE 的相对退化比常更小，但 **IN/OUT 绝对误差更高**——更像通过 **underfitting** 换“稳健”，而非学到分布不变表示。论文主张合格 OOD 方法应同时满足强 IN + 稳 OUT。
4. **结构组件的双刃剑**：消融显示 STID 去空间 embedding 会伤 IN、反助 OUT；GWNet 自适应邻接助 IN、OUT 误差约 +9.7%。细粒度时空组件易学到**训练年特有**模式。
5. **轻度 dropout 有效**：STID 上 0.2–0.3 dropout 普遍改善 OUT，对 IN 影响小；FlowSC 上 0.3 dropout 可将 OUT RMSE 从 54.35 降至 24.12（约 −55.6%）；>0.3 则双降。

## 设计原则与局限

作者提出简单有效 OOD 模型的原则：最小归纳偏置、适度正则（dropout 0.2–0.3）、保留物理/地理约束（Tobler 第一定律类拓扑）、稳定特征与因果筛选、简单模型集成。未来方向含统一图+时序 OOD 理论（反对时空硬分离）、LLM 辅助（如 [[urbangpt|UrbanGPT]] 式零样本）等。

**局限：** 主设定是**跨年时间偏移**（图由距离阈值重建，非传感器大规模增删的纯结构 OOD）；速度/流量等子窗较短；专用 OOD 基线仅 CaST/CauSTG/STONE 三家；COVID 年份混入部分场景，偏移来源不完全是“常规城市化”。

## 相关页面

- [[st-ood]] — 基准实体
- [[ood-generalization]] · [[spatio-temporal-ood-learning]] · [[traffic-forecasting]]
- [[source-cast]] · [[stop]] · [[stunet]] · [[gwnet]] · [[stgcn]]
