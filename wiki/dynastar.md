---
title: "DynaSTar"
type: entity
tags:
  - traffic-forecasting
  - spatio-temporal
  - dynamic-graph
  - invariant-learning
  - out-of-distribution
created: 2026-09-16
last_updated: 2026-09-16
source_count: 1
confidence: medium
status: active
---

# DynaSTar

**DynaSTar**（Dynamic Spatio-Temporal Graph Invariant learning model）是 Hao et al.（IJCAI 2026 camera-ready）提出的鲁棒交通预测模型，面向拓扑持续演化下的 out-of-time（[[out-of-time-generalization|OOT]]）预测；作者自述其为首个显式建模拓扑动态以解决该问题的形式化工作[^src-dynastar]。

## 问题定位

论文的诊断是：既有时空 OOD 方法（CaST、CauSTG、STONE、STEVE）依赖静态邻接关系建模分布漂移，而真实路网拓扑持续演化，学到的知识随时间退化，损害 OOT 泛化[^src-dynastar]。观察证据：GNN 基线在长期部署中误差显著高于近期部署；时空 OOD 模型整体优于纯时间 OOD 模型，说明空间聚合比仅处理时间漂移更关键[^src-dynastar]。

## 架构

### 动态拓扑追踪（4.1 节）

1. **时空注意力编码**：$L{=}3$ 个堆叠块，每块为原始邻接阵上的图注意力（SA）+ 窗口化时间注意力（TA），各块输出经 FC 投影后拼接为动态表示 $\mathbf{Z}^{(t)}$[^src-dynastar]。
2. **memory 增强的动态相关图**：32×32 可学习 meta memory bank 经 cross-attention 将节点表示投影到关系原型空间，得到图生成嵌入 $\mathbf{g}_i$；节点对相关分数 $s^t_{ij}=(\mathbf{g}^t_i W_\phi)(\mathbf{g}^t_j W_\psi)^\top$ 构成瞬时相关图[^src-dynastar]。
3. **动量更新的概率图**（见 [[momentum-updated-probabilistic-graph]]）：瞬时图缺历史记忆、对噪声敏感，故维护动量更新的图原型并 sigmoid 化为逐边 Bernoulli 概率图[^src-dynastar]。
4. **可微稀疏图生成**：Straight-Through Gumbel-Softmax（温度 10）从全连接概率原型采样 2 张稀疏二值图，前向离散、反向可导；推理直接以边概率 $>0.5$ 阈值化[^src-dynastar]。

### 节点异质不变学习（4.2 节）

见 [[node-heterogeneous-invariant-learning]]。核心是：按节点邻域 Bernoulli 分布采样邻域环境集模拟异质拓扑漂移；条件向量 $c_i^t=\mathrm{MLP}(z_i^t+e_{time}(t))$（64 维）经 MLP 生成逐层 FiLM 仿射参数，调制与编码器同构的共享预测器；优化目标 $L_{total}=L_{pred}+0.01\,L_{node}+1\,L_{inv}$[^src-dynastar]。

## 结果

- **多阶段部署**（LargeST SD/SGBA，2019 训练 → 2020 测试，5 次运行平均）：两个数据集、近期与长期部署全部指标（MAE/MAPE，3/6/12 步平均）对 10 个基线最优；SD 长期 60min MAE 27.94（STONE 38.22、STEVE 37.54），SGBA 长期 60min MAE 29.23（STEVE 32.45、STONE 34.25）[^src-dynastar]。
- **消融**：去 memory 增强（w/o ME）、去动量图（w/o MG）、去稀疏采样（w/o SG）、去节点调制（w/o NM）、去邻域环境（w/o NE）均掉点；w/o MG 降幅最大，作者以此论证演化且稳定的图对长期部署的重要性[^src-dynastar]。
- **未见图零样本**：以 SGBA 训练、GBA 其余传感器划分的 NWGBA/NEGBA 测试，优于 STONE 与 STEVE[^src-dynastar]。
- **效率**：稀疏图聚合避免稠密图操作，样本规模与图规模增长下运行时间最低、吞吐最高（对比 STEVE/STONE）[^src-dynastar]。

## 与本 wiki 其他页面的关系

- 与 [[stop|STOP]] 的对照：STOP 在架构层阻断节点间消息传递以解耦结构耦合，DynaSTar 则保留消息传递但让图本身演化并施加环境级不变约束[^src-dynastar]。
- 代码：https://github.com/ShaunHao/dynastar

## 引用

[^src-dynastar]: [[source-dynastar]]
