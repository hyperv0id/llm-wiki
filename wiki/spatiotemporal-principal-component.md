---
title: "Spatiotemporal Principal Component（时空主成分预测）"
type: technique
tags:
  - uncertainty-estimation
  - pca
  - traffic-flow
  - probabilistic-forecasting
created: 2026-09-16
last_updated: 2026-09-16
source_count: 1
confidence: low
status: active
---

# Spatiotemporal Principal Component（时空主成分预测）

## 问题

概率预测要给出未来流量的协方差结构。对时空数据，全协方差是二阶张量 $\Sigma = X \otimes X \in \mathbb{R}^{T \times N \times T \times N}$，维度随预测步 $T$ 与节点数 $N$ 二次增长，直接监督需 $O((TN)^3)$ 存储与分解。且流量相关性高度动态，用历史数据估计的协方差难以泛化到未来[^src-ripcn]。

## 机制

对去均值数据 $X = X^P - \hat{X}^P$ 的协方差做低秩近似：

$$\Sigma \approx \sum_{k=1}^{K} \lambda_k \boldsymbol{w}_k \otimes \boldsymbol{w}_k$$

前几个特征向量即可捕捉大部分方差（PEMS08 相邻路段的协方差谱显示方差集中于少量主成分）。与经典 PCA 降维观测数据不同，这里是**直接预测未来流量的时空主成分**：网络输出 $\boldsymbol{d}_k(X^H, \hat{X}^P)$，经 Schmidt 正交化得 $\boldsymbol{w}_k$[^src-ripcn]。

理论约束两条，各配一个损失：

- **方向约束** $\mathcal{L}_D = -\langle \boldsymbol{w}_k, X \rangle_F^2$：拉格朗日乘子法可证最优解 $\boldsymbol{w}_k$ 是 $\Sigma$ 的时空特征向量（$\Sigma \boldsymbol{w}_k = \theta_k \boldsymbol{w}_k$），该损失使预测方向对齐真实涨落方向[^src-ripcn]。
- **方差幅值** $\mathcal{L}_V = \sum_k (\langle \boldsymbol{w}_k, X \rangle_F^2 - \|\tilde{\boldsymbol{w}}_k\|_F^2)^2$：约束正交化输出范数匹配方差，测试时 $\sigma_k^2 = \|\tilde{\boldsymbol{w}}_k\|_F^2$ 直接读出各方向方差[^src-ripcn]。

**样本构造**：$\mathcal{S} = \{ \hat{X}^P + t_k \sigma_k \boldsymbol{w}_k \}_{k=1}^K$，$t_k$（可正可负，涨落方向对齐或相反）在验证集上二分搜索；样本均值与方差作为最终预测。论文称由此得到的均值常比 $\hat{X}^P$ 更准（理论论证 + 附录 A 实证）[^src-ripcn]。

## 证据与边界

PEMS08 上 $K{=}3$ 最优，$K{=}1$ 已给出合理不确定性，过多低方差主成分引入噪声致过拟合；ST-Graph 块最优 16 个[^src-ripcn]。消融中 w/o $\mathcal{L}_D$ 使 CRPS 0.0955→0.1247、MIS 近乎翻倍，w/o $\mathcal{L}_V$ 使预测区间变宽（MIS 1297.79→1356.27），说明方向与幅值两级约束缺一不可[^src-ripcn]。效率：一次前向得全部 $K$ 个主成分即可构造多样本，PEMS08 推理 16.77s、GPU 3952 MB，对比直接监督全协方差的变体（229.94s / 8346 MB）与扩散模型的多步去噪采样均显著更低[^src-ripcn]。证据限于 RIPCN 单一源文件；与 [[crps|CRPS]] 等区间/分布指标的配合、在其他领域的可迁移性未验证。

## 引用

[^src-ripcn]: [[source-ripcn]]
