---
title: "GraphCast"
type: technique
tags:
  - weather-forecasting
  - medium-range-forecasting
  - graph-neural-network
  - multi-mesh
  - autoregressive
  - mlwp
  - era5
created: 2026-08-30
last_updated: 2026-08-30
source_count: 1
confidence: medium
status: active
---

# GraphCast

**GraphCast** 是 Google DeepMind 团队提出的机器学习天气预测（MLWP）方法：直接从 ERA5 再分析数据训练，以 encode-process-decode 配置的 GNN 学习 6 小时一步的全球大气演化，自回归滚动生成 10 天、0.25° 分辨率的全球预报[^src-graphcast]。

## 问题：中期预报与 NWP 的可扩展性

中期天气预报指预测未来至 10 天的大气变量。传统数值天气预报（NWP）通过求解大气控制方程计算预报，其精度随算力投入可扩展，但无法直接利用历史天气数据改进模式本身——改进依赖专家长期迭代模式与算法，论文将此视为 NWP 的结构性限制[^src-graphcast]。论文将机器学习天气预测（MLWP）定位为另一条路线：直接从历史数据训练，利用深度学习硬件而非超算，寻求更有利的速度-精度权衡[^src-graphcast]。

论文以 ECMWF 的 HRES（High RESolution forecast，IFS 的确定性分量）为对照：论文称其为世界上最准确的确定性业务系统，0.1° 分辨率生成 10 天全球预报约需 1 小时（补充材料注明其运行于 11,664 核集群，Supp Sec 3.2）[^src-graphcast]。

## 机制

### 预报格式与自回归滚动

GraphCast 输入当前时刻与 6 小时前两个天气状态，预测 6 小时后的下一状态；将自身预测回灌为输入即可滚动生成任意长度的状态序列（Figure 1b–c）[^src-graphcast]。天气状态定义在 0.25° 经纬网格（721 × 1440 = 1,038,240 个格点，赤道约 28 × 28 km），每点 5 个表面变量 + 6 个大气变量 × 37 个气压层 = 227 个变量，单个状态共 235,680,480 个数值（Figure 1a、Table 1）[^src-graphcast]。输入特征还包括解析可得的强迫项（TOA 入射太阳辐射、地方时/年进程的正余弦）与静态常数（海陆掩码、地表位势、经纬度），合计每格点 474 个输入特征（Supp Sec 3.3）[^src-graphcast]。

### 编码-处理-解码 GNN

架构基于 GNN 的 "encode-process-decode" 配置（论文归因于既有 learned simulator 谱系，参考文献 [1][31][26]），共 36.7M 参数（Figure 1d–f）[^src-graphcast]：

- **编码器**：单层 GNN，经有向 Grid2Mesh 边将格点特征映射到内部 mesh 表征（Supp Sec 3.4）[^src-graphcast]；
- **处理器**：16 层不共享参数的 GNN，在 [[multi-mesh-representation|multi-mesh]] 上做消息传递，借助长短边混合实现少步数下的局部与长程信息传播（Figure 1e）[^src-graphcast]；
- **解码器**：单层 GNN 经有向 Mesh2Grid 边映射回经纬网格，输出作为对最近输入状态的残差更新：$\hat{X}_{t+k+1} = \hat{X}_{t+k} + \hat{Y}_{t+k}$（Supp Sec 3.2/3.3）[^src-graphcast]。

### 训练目标与课程

训练数据为 ERA5 1979–2017 共 39 年。损失为按垂直层加权的 MSE，在 N 个自回归步上累计；N 随训练进度从 1 递增至 12（对应 6 小时至 3 天）的课程训练（Supp Sec 4）[^src-graphcast]。训练在 32 块 Cloud TPU v4 上约 4 周（batch parallelism，Supp Sec 4）[^src-graphcast]。单块 TPU v4 生成 0.25°、10 天（6 小时步长）预报用时不足 60 秒（Supp Sec 3.2）[^src-graphcast]。

## 证据：与 HRES 的验证

### 验证协议

对照指标为 RMSE 与 ACC；在 227 个变量-层组合中按 ECMWF Scorecard 与 WeatherBench 的 13 个气压层选取 69 个评估，覆盖 20 个预报时效，合计 1380 个验证目标；总降水因 ERA5 降水数据已知偏差被排除（正文 Verification methods；另见 Supp Sec 1.2/5）[^src-graphcast]。公平性处理：GraphCast 以 ERA5 为真值，HRES 以其预报初值场 HRES-fc0 为真值（保证第 0 步零误差）；GraphCast 仅评估 06z/18z 起报（+3h 观测前瞻与 HRES 输入一致），目标仅每 12h 评估一次（Supp Sec 5.2）[^src-graphcast]。HRES 06z/18z 起报的预报仅 3.75 天，其后改与 00z/12z 起报对比（图中虚线，Figure 2）[^src-graphcast]。

### 作者报告的结果（2018 年留出数据）

- **总体**：GraphCast 在 1380 个目标中的 90.3% 上 RMSE 优于 HRES；显著性检验（p ≤ 0.05，名义样本量 n ∈ {729, 730}）下为 89.9%（Figure 2d）[^src-graphcast]。
- **头条变量 z500**：全时效领先，RMSE skill score 改善约 7%–14%（Figure 2a–c）[^src-graphcast]。
- **分层结构**：HRES 占优区域集中于平流层（训练损失权重最低的层级）；排除 50 hPa 层后显著领先比例升至剩余 1280 目标的 96.9%，再排除 100 hPa 后为 1180 目标的 99.7%（正文；平流层定位分析见 Supp Sec 7.2.2）[^src-graphcast]。
- **模糊化对照**：自回归训练步数增多会促使模型以空间平滑（更模糊）的输出表达不确定性，而 HRES 的物理方程不产生模糊预报；论文为两个系统分别拟合最优模糊滤波器后再比，模糊化后的 GraphCast 仍在 1380 目标的 88.0% 上领先（Supp Sec 7.4）[^src-graphcast]。
- **与 Pangu-Weather 对比**：在 Pangu-Weather 论文给出的 252 个目标上，GraphCast 99.2% 领先（Supp Sec 6）[^src-graphcast]。

### 极端事件（非专门训练）

- **热带气旋路径**：基于 ECMWF 公开协议实现追踪算法，作用于 z、10u/10v、u/v、msl 预报，以 IBTrACS 再分析为真值。2018–2021 年 GraphCast 中位路径误差低于 HRES；配对分析显示 18 小时至 4.75 天时效显著更好（Figure 3a–b）[^src-graphcast]。
- **大气河流**：由 u、v、q 非线性组合计算垂直积分水汽输送 ivt；在北美西海岸至东太平洋冷季（10–4 月）评估中，短时效改进约 25%、长时效约 10%（Figure 3c）[^src-graphcast]。
- **极端温度**：对 2t 超过气候态 top 2% 事件的 precision-recall 曲线（Figure 3d），5 天与 10 天时效 GraphCast 高于 HRES；12 小时时效 HRES 更好，论文注这与 2t skill score 接近零一致（Figure 2d）[^src-graphcast]。

### 训练数据时效

以终止年份 2017/2018/2019/2020 训练四个变体，在 2021 年测试数据上比较 z500：训练至 2020 年前的变体 skill score 进一步高于训练至 2017 年前的变体；作者以 speculate 口径推测近期数据使模型捕获 ENSO 等变化的天气趋势（Figure 4）[^src-graphcast]。

## 论文自述的局限与范围

- **不确定性**：GraphCast 为确定性预报；MSE 训练使其以空间模糊表达不确定性，而非如 IFS 集合系统（ENS）那样以多次随机预报建模未来天气的经验分布；论文将更显式的不确定性建模列为关键下一步（Conclusions）[^src-graphcast]。
- **分辨率与规模差距**：0.25°/37 层/6h 步长低于 HRES 的 0.1°/137 层/1h，受制于 ERA5 原生分辨率与硬件显存约束；36.7M 参数是工程约束下可容纳的规模，论文称 GraphCast 应被理解为一个模型家族（Conclusions）[^src-graphcast]。
- **数据依赖**：数据驱动 MLWP 依赖 NWP 同化产生的高质量再分析数据；论文明确该工作不应被视为传统天气预测方法的替代，而是对现有最好方法的补充（Conclusions）[^src-graphcast]。

## 相关页面

- [[multi-mesh-representation]] — 多分辨率 icosahedron mesh 表征，GraphCast 处理器的核心设计
- [[spherical-geometry-inductive-bias]] — GraphCast 的 mesh 路线与 CirT/PanguWeather 等几何偏置路线的对照
- [[storminsight]] — 其大气环境编码器采用「GraphCast 风格」multi-mesh 消息传递
- [[cirt]] — CirT 对 GraphCast 的对比：mesh 局部消息传递 vs 频域全局周期性
- [[precipitation-nowcasting]] — GraphCast 等中期模型与临近预报之间的尺度不匹配问题
- [[subseasonal-to-seasonal-forecasting]] — 自回归迭代扩展至 S2S 尺度的累积误差路线
- [[weather-foundation-model]] — 天气模型谱系与基础模型范式背景

[^src-graphcast]: [[source-graphcast]]
