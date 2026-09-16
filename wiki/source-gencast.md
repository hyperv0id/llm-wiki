---
title: "GenCast: Generalising Traffic Forecasting to Regions Without Traffic Observations"
type: source-summary
tags:
  - traffic-forecasting
  - unobserved-regions
  - physics-informed
  - lwr
  - external-signals
  - aaai-2026
created: 2026-09-16
last_updated: 2026-09-16
source_count: 1
confidence: medium
status: active
---

# GenCast

Xinyu Su, Majid Sarvi, Feng Liu, Egemen Tanin, Jianzhong Qi（墨尔本大学 / HKUST-GZ），AAAI 2026, pp. 15761–15769。代码 github.com/suzy0223/GenCast；扩展版 arXiv:2508.08947。源文件 `raw/gencast-generalising-traffic-forecasting-without-observations-aaai26.pdf`。

> [!note] 同名
> 与 Google DeepMind 用于中期天气集合预报的扩散模型 GenCast 无关。

## 问题

预测大片连续无传感器区域（相邻或外围有观测）的未来交通。克里金/外推（IGNNK、INCREASE）与虚拟节点方法（KITS）面向零星散落的无观测点；STSM 用静态 POI 与坐标选取观测区中的相似位置做掩码对比，静态特征跟不上动态交通演化[^src-gencast]。

## 方法

沿用 STSM 的掩码子图对比骨干（伪观测 + 图级对比损失），加三类外部知识[^src-gencast]：

1. **物理**：LWR 守恒律经 Greenshields 关系 $x=x_{fspd}(1-\rho/\rho_{max})$ 化为速度形式残差 $R=\partial\hat X/\partial TE_{enc}+(2\hat X-X_{fspd})\odot\partial\hat X/\partial L_{enc}$，以 Huber 损失惩罚，$\delta$ 取热身一轮后残差的 $\tau$-分位数。求偏导要求位置编码连续可微，故须另设可微空间/时间嵌入（见 [[differentiable-spatial-embedding]]）。
2. **外部信号**：节点按最近邻对齐 ERA5-Land 气象站，取过去 12 小时温度、太阳辐射、降水、地表径流，经时序 cross-attention 与 sigmoid 门控融合。
3. **空间分组**：每层把通道分组重排后软聚类到 $s_g\times c_g$ 个原型，对分配权重做熵最小化，压制节点局部特异特征。

总损失 $\mathcal{L}=\mathcal{L}_{pred}+\lambda\mathcal{L}_{cl}+\mu\mathcal{L}_{spg}+\theta\mathcal{L}_{phy}$；除 $\mathcal{L}_{cl}$ 外均在掩码图上计算。

## 结果

5 个数据集（PEMS07/PEMS08/PEMS-Bay/METR-LA + Melbourne CBD），空间按 4:1:5 切分，测试区全程无观测，2h→2h，四组切分取平均。全部指标优于 GE-GAN/IGNNK/INCREASE/STSM/KITS[^src-gencast]：

| 数据集 | 变体 | 与 STSM 对比 |
|---|---|---|
| PEMS07 | L | RMSE 8.253 / 8.390 |
| PEMS08 | L | RMSE 7.863 / 7.925 |
| PEMS-Bay | H | RMSE 8.683 / 8.773 |
| METR-LA | H | RMSE 12.720 / 12.952 |
| Melbourne | H | MAE 7.083 / 7.308；$R^2$ 0.061 / 0.027 |

误差最多降 3.1%，Melbourne $R^2$ 相对提升 125.6%；配对 $t$ 检验与 Wilcoxon 均为 $p\ll10^{-8}$；环形切分 PEMS-Bay $R^2$ 提升 27.51%；未观测比例 0.2→0.8 时衰减最慢；NREL 太阳能数据（去物理与空间嵌入的 w/wx 变体）仍提升 1.46%~5.83%[^src-gencast]。

SE-L 在较新的 PEMS07/08 更好，SE-H 在年代久远的 PEMS-Bay/METR-LA 与范围小、同质的 Melbourne CBD 更好；论文归因于 SE-L 依赖当期 OpenStreetMap，与旧数据的物理环境不匹配，且 CBD 区域缺乏可区分的空间特征[^src-gencast]。

## 边界

- Greenshields 线性关系是简化本构，论文未给出极端拥堵下的误差分析。
- 论文自述不假定系统闭合，实际约束的是相邻连通路段间的局部流量守恒[^src-gencast]。
- 论文自述 Melbourne 的 $R^2$ 绝对值仍低，跨区域泛化与本地保真度的权衡留作后续工作[^src-gencast]。

[^src-gencast]: [[source-gencast]]
