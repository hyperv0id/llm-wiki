---
title: "Momentum-Updated Probabilistic Graph"
type: technique
tags:
  - dynamic-graph
  - graph-structure-learning
  - traffic-forecasting
  - spatio-temporal
created: 2026-09-16
last_updated: 2026-09-16
source_count: 1
confidence: low
status: active
---

# Momentum-Updated Probabilistic Graph

**动量更新的概率图**是 [[dynastar|DynaSTar]] 追踪演化拓扑的机制：不直接用瞬时相关图做预测，而是维护一个带历史记忆、逐边 Bernoulli 参数化的图原型，从而把图结构学习变成一个平滑、历史有据的演化过程[^src-dynastar]。

## 机制

1. **瞬时相关图**：节点表示经可学习 memory bank（32×32，cross-attention 投影到关系原型空间）提纯后，节点对相关分数 $s^t_{ij}=(\mathbf{g}^t_i W_\phi)(\mathbf{g}^t_j W_\psi)^\top$ 构成瞬时图 $\mathcal{G}^p_t$。论文的诊断：瞬时图缺乏历史记忆，对噪声敏感[^src-dynastar]。
2. **动量更新**：
$$\mathcal{G}^{proto}_t \leftarrow \beta\cdot\mathcal{G}^{proto}_{t-1} + (1-\beta)\cdot\mathcal{G}^p_t$$
3. **概率化**：$\hat{A}^t_{ij}=\sigma(\mathcal{G}^{proto,t}_{ij})\in(0,1)$，每条边成为 Bernoulli 随机变量，刻画节点间依赖的波动[^src-dynastar]。
4. **稀疏实例化**：训练时以 Straight-Through Gumbel-Softmax（温度 10）从全连接概率原型采样稀疏二值邻接阵（见 [[reparameterization-trick]] 的离散情形）；推理直接阈值化 $\hat{A}_{ij}>0.5$，避免采样随机性，保证确定性预测[^src-dynastar]。

## 证据与边界

消融中去掉动量图（w/o MG，退回瞬时图）是五项消融中降幅最大的一项：SD 长期部署 60min MAE 从全模型 26.87 升至 28.49（+1.62，其余消融为 w/o NE +1.09、w/o NM +1.38、w/o SG +1.26、w/o ME +0.93），论文以此论证"演化且稳定的图"对长期部署的贡献大于 memory 增强或稀疏采样本身[^src-dynastar]。图 5 可视化显示边概率在节点对序列模式分化的两个时段（约 4 天与 9 天）显著下降，作为该机制捕捉依赖变化的定性证据[^src-dynastar]。该机制的 $\beta$ 敏感性、与其它动态图方法（如 [[evolving-rn-traffic-forecasting|EvolveRN]]）的对照未见报告。

## 引用

[^src-dynastar]: [[source-dynastar]]
