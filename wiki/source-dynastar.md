---
title: "DynaSTar: Dynamic Graph Invariant Learning for Out-of-Time Spatio-Temporal Prediction (Hao et al., IJCAI 2026)"
type: source-summary
tags:
  - traffic-forecasting
  - spatio-temporal
  - out-of-distribution
  - dynamic-graph
  - invariant-learning
created: 2026-09-16
last_updated: 2026-09-16
source_count: 1
confidence: medium
status: active
---

# DynaSTar: Dynamic Graph Invariant Learning for Out-of-Time Spatio-Temporal Prediction

**DynaSTar**（Hao, Wan, Guo, Wang, Lin；北京交通大学 + 中科院南京，IJCAI 2026 camera-ready）针对论文所称的 **out-of-time（OOT）泛化**问题：模型部署后面对的不仅是信号分布漂移，邻接拓扑本身也在持续演化，而既有时空 OOD 方法依赖静态图结构，论文认为这限制了长期部署可靠性[^src-dynastar]。作者自述这是首个显式建模拓扑动态以解决时空交通预测 OOT 泛化的形式化工作[^src-dynastar]。

## 方法

两段式：**动态拓扑追踪**（4.1 节）先用 $L$ 层时空注意力块（图注意力 + 窗口化时间注意力）得到动态表示，经 32×32 可学习 memory bank 的 cross-attention 提纯为图生成嵌入，计算节点对的实时相关分数得瞬时相关图；再以动量规则 $\mathcal{G}^{proto}_t \leftarrow \beta\mathcal{G}^{proto}_{t-1}+(1-\beta)\mathcal{G}^p_t$ 平滑更新图原型，sigmoid 化为逐边 Bernoulli 概率图；训练时以 Straight-Through Gumbel-Softmax（温度 10，每 batch 采样 2 张）实例化稀疏二值邻接阵，推理时直接以边概率 >0.5 阈值化[^src-dynastar]。**节点异质不变学习**（4.2 节）将每个节点的邻域参数化为多元 Bernoulli 分布并采样邻域环境集，条件向量 $c_i^t=\mathrm{MLP}(z_i^t+e_{time}(t))$ 生成逐层 FiLM 仿射参数调制共享预测器；损失为 $L_{pred}+\gamma_1 L_{node}+\gamma_2 L_{inv}$：InfoNCE 节点对比（同节点跨时刻为正对，$\gamma_1=0.01$）加跨环境预测风险的方差惩罚（IRM 风格，$\gamma_2=1$）[^src-dynastar]。

## 实验证据

LargeST 子集 SD（716 传感器）与 SGBA（1278 传感器），2019 年训练（7:3 训/验）、2020 年测试；2020 上半年为近期部署、下半年为长期部署，输入 12 步预测 12 步，MAE/MAPE 取 3/6/12 步平均，5 次运行平均[^src-dynastar]。论文报告 DynaSTar 在两个数据集、两个阶段对 3 类 10 个基线（STGCN/GWNet/AGCRN；AdaRNN/Diversify/RevIN；CaST/CauSTG/STONE/STEVE）全指标最优，例如 SD 长期部署 60min MAE 27.94（次优 STONE 38.22）、SGBA 长期 60min MAE 29.23（次优 STEVE 32.45）[^src-dynastar]。消融中去动量图更新（w/o MG）掉点最大；零样本迁移到未见图结构（NWGBA/NEGBA）优于 STONE/STEVE；稀疏图聚合使其在规模扩展时运行时间最低、吞吐最高[^src-dynastar]。图 5 可视化显示边概率在约 4 天与 9 天两个时段显著下降，与对应节点对序列模式的分化期吻合[^src-dynastar]。

## 局限与边界

实验限于 LargeST 交通流同构传感器图，未见对异构城市任务（如 [[st-ood|ST-OOD]] 基准的六类城市场景）的验证。代码：https://github.com/ShaunHao/dynastar[^src-dynastar]。

## 引用

[^src-dynastar]: [[source-dynastar]]
