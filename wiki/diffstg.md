---
title: "DiffSTG"
type: entity
tags:
  - diffusion-models
  - spatio-temporal-graph
  - probabilistic-forecasting
  - traffic-forecasting
created: 2026-05-31
last_updated: 2026-06-01
source_count: 1
confidence: medium
status: active
---

# DiffSTG

**DiffSTG** 是首个将去噪扩散概率模型（[[ddpm|DDPM]]）推广到时空图（STG）预测领域的非自回归条件扩散框架，由 Wen 等人于 AAAI 2023 提出[^src-diffstg]。它将扩散模型的概率建模能力与 STGNN 的时空表示学习能力结合，能够在给定图结构和历史信号的条件下，一次生成整个未来窗口的联合概率分布。

## 问题与动机

传统确定性 STGNN（如 [[dcrnn|DCRNN]]、[[stgcn|STGCN]]、[[gwnet|GWNet]]）在交通流量、空气质量等预测任务上已将 RMSE 和 MAE 刷至极低水平，但它们仅输出点估计——无法提供预测不确定性[^src-diffstg]。在交通疏散、公共安全管理等决策场景中，知道"预测有多不确定"可能比精确的点估计更重要。

另一方面，扩散时序模型（[[timegrad|TimeGrad]]、CSDI）已证明了扩散在时间序列上的可行性，但它们各有盲点[^src-diffstg]：

- **[[timegrad|TimeGrad]]**：自回归 LSTM 编码历史，扩散模型独立建模每步的多维分布——完全没有空间依赖概念，相邻传感器的关联被当作全连接处理；且自回归生成导致推理极慢（$T_p=12$ 需 1200 次去噪前向）
- **CSDI**：非自回归但本为插补设计，如果强行 mask 所有未来时间步做预测，训练（随机 mask）和推理（连续段全 mask）之间存在分布不匹配

DiffSTG 要做的事是填补裂缝：把扩散模型从欧几里得的图像/序列空间迁移到非欧几里得的图空间，同时解决自回归速度瓶颈[^src-diffstg]。

## 核心设计

### 广义条件扩散模型（3.2 节）

DiffSTG 最具创造性的公式化[^src-diffstg]。不是分别处理历史和未来，而是定义 $\mathcal{X}^{\text{all}} = [\mathcal{X}^h, \mathcal{X}^p] \in \mathbb{R}^{F \times V \times T}$，其中 $T = T_h + T_p$。构造 $\mathcal{X}^{\text{all}}_\text{msk}$：历史部分保留原始值，未来部分全部 mask。条件扩散模型变为：

$$p_\theta(\mathcal{X}^{\text{all}}_{0:N} \mid \mathcal{X}^{\text{all}}_\text{msk}, \mathcal{G}) = p(\mathcal{X}^{\text{all}}_N) \prod_{n=N}^{1} p_\theta(\mathcal{X}^{\text{all}}_{n-1} \mid \mathcal{X}^{\text{all}}_n, \mathcal{X}^{\text{all}}_\text{msk}, \mathcal{G})$$

训练目标是对整个全时信号的单步 MSE 去噪[^src-diffstg]：

$$\min_\theta \mathbb{E}_{\mathcal{X}^{\text{all}}_0, \varepsilon, n} \left[ \| \varepsilon - \varepsilon_\theta(\mathcal{X}^{\text{all}}_n, n \mid \mathcal{X}^{\text{all}}_\text{msk}, \mathcal{G}) \|^2 \right]$$

这个公式的精妙之处：历史重建和未来预测被装进同一个损失里。历史数据提供高确定性的梯度锚点，稳定整个数据分布流形的学习——相当于用"已知答案的题目"做训练正则化[^src-diffstg]。

### UGnet 去噪网络（4.1 节）

专为 STG 设计的异构去噪架构，设计规则是"时间维度 Unet 化，空间维度 GCN 化"[^src-diffstg]：

- **ST-Residual Block**：每个块内部依次执行四步：(1) TCN 门控因果卷积，输出劈为 $P$ 和 $Q$ 后做门控激活 $P \odot \sigma(Q)$；(2) Reshape 将 $C_\text{out} \times V \times T_i$ → $V \times (T_i \times C_\text{out})$；(3) Vanilla GCN 图卷积 $A_\text{gcn} = D^{-1/2}(A+I)D^{-1/2}$；(4) 残差连接 + 层归一化

- **时间 Unet**：下采样块逐步缩小时间维度（$T \to T/2 \to T/4 \ldots$），抓长周期模式；上采样块逐步恢复，跳跃连接防止梯度消失。关键区别：空间维度从头到尾保持 $V$ 不变——空间建模完全交给 GCN

- **噪声条件**：Transformer 正弦位置编码 $\mathbf{e}(n) = [\dots, \cos(n/r^{2d/D}), \sin(n/r^{2d/D}), \dots]^\top$（$D=32, r=10000$）注入每个 ST-Residual Block[^src-diffstg]

三个条件各走各的路径：$\mathcal{X}^{\text{all}}_\text{msk}$ 与 $\mathcal{X}^{\text{all}}_n$ 在时间维度拼接后送入 UGnet；$\mathcal{G}$ 通过 GCN 层作用；$n$ 通过位置编码注入每个 Block[^src-diffstg]。

### 非自回归采样与加速（4.2 节）

DiffSTG 的关键工程优势[^src-diffstg]：

1. **非自回归**：一次反向扩散过程输出全部 $T_p$ 个未来时间步，而非 [[timegrad|TimeGrad]] 的 $T_p$ 次独立扩散
2. **DDIM 子集采样**：从 $N=100$ 步中只取 $M=40$ 步子集做采样，速度提升 2.5 倍且几乎无损
3. **尾步样本复用**：需要 $S$ 条轨迹时，前 $N-k$ 步各独立采样 1 条路径，最后 $k$ 步每条路径分叉出 $k$ 个样本——总扩散步数从 $S \times N$ 降至 $(S/k) \times N$

在 $S=32, M=40, k=2$ 配置下，推理仅需 0.21 秒——[[timegrad|TimeGrad]] 需要 672 秒（超过 11 分钟），加速约 3200 倍[^src-diffstg]。

## 实验结果

### 概率预测

在三个数据集上评估 CRPS、MAE、RMSE（$T_h = T_p = 12, N=100$）[^src-diffstg]：

| 数据集 | 节点数 | 数据类型 | CRPS 提升 | MAE 提升 |
|--------|--------|----------|----------|----------|
| PEMS08 | 170 | 交通流量 | -14.3% | -7.0% |
| AIR-BJ | 34 | PM2.5 | -5.6% | -4.1% |
| AIR-GZ | 41 | PM2.5 | -4.3% | -1.5% |

PEMS08 上空间依赖最强的数据集中优势最显著——GCN 真正发挥了作用[^src-diffstg]。AIR-GZ 上优势最小——空气质量空间变化平缓，空间信息贡献相对弱。

### 消融实验

三个消融变体的贡献度排序为：U-structure ≥ Temporal > Spatial。去掉 Unet 结构（仅一层 TCN）退化最严重——说明多时间粒度的层级特征对捕捉"15 分钟波动叠加在 1 小时趋势上"的组合模式至关重要[^src-diffstg]。

### 与确定性方法对比

DiffSTG 的 RMSE/MAE 落后于最佳确定性方法（GMSDR、STGNCDE）约 5–10%[^src-diffstg]。这是扩散模型 ELBO 优化的系统性局限——训练目标是噪声预测 MSE，不是直接的预测误差。但 DiffSTG 在预测出错时"知道自己不确定"——预测区间会变宽——这个元认知能力可能比精确的点估计更有决策价值[^src-diffstg]。

## 噪声调度与超参数

DiffSTG 采用二次噪声调度（区别于 DDPM 的线性调度）[^src-diffstg]：

$$\beta_n = \left( \frac{n-1}{N-1} \sqrt{\beta_N} + \frac{N-n}{N-1} \sqrt{\beta_1} \right)^2$$

其中 $\beta_1=0.0001, \beta_N \in [0.1, 0.4], N=100$。前期 $\beta$ 增长更平缓（低噪声区更精细），后期增长更陡峭。关键敏感性：在 $N=50, \beta_N=0.1$ 时性能崩溃——因为 $\bar{\alpha}_N$ 不够接近 0，正向终点的分布不够高斯化[^src-diffstg]。

核心超参数：隐维度 $C=32$，批大小 8，学习率 0.002（Adam，每 5 epoch 减半），TCN kernel size $K=3$。

## 局限性

1. **确定性精度天花板**：落后最佳 STGNN 约 5–10%，ELBO 优化不产生和直接 RMSE 优化同样锐利的后验[^src-diffstg]
2. **GCN 过于朴素**：vanilla GCN 无注意力、无动态图学习。当图结构有噪声或节点关系随时间变化（如早晚高峰拥堵模式变化），简单 GCN 可能不够用[^src-diffstg]
3. **预测长度固定**：$T_p$ 训练时硬编码，无法在推理时灵活选择预测步数[^src-diffstg]
4. **噪声调度手工设计**：二次调度是经验选择，无理论最优性保证[^src-diffstg]
5. **图结构输入质量依赖**：邻接矩阵 $A$ 的质量直接影响性能，模型没有机制修正或绕过错误的图结构[^src-diffstg]

## 后续影响

DiffSTG 开创了"用扩散模型做 STG 概率预测"的范式，证明了三个关键结论[^src-diffstg]：

1. 图结构信息可以作为扩散条件自然注入
2. Unet 架构在时间维度上的多尺度聚合同样适用于非图像信号
3. 非自回归扩散在时空预测上比自回归更高效

后续工作包括 [[specstg|SpecSTG]]（将扩散移至图谱域）、D3（扩散+去噪+解耦）、DiffLoad（电力负荷预测不确定性量化）、[[ustd|USTD]]（统一预测与插值的预训练编码器+扩散 decoder 两阶段框架，首次让 diffusion STG 在预测上超越 deterministic baseline，SIGSPATIAL 2024）、UrbanDiT（扩散 Transformer + Rectified Flow）等[^src-diffstg]。

## 关联页面

- [[generative-time-series-forecasting]] — 生成式时间序列预测范式
- [[traffic-forecasting]] — 时空图交通预测总览
- [[ddpm]] — DDPM 扩散模型基础
- [[specstg]] — 谱域扩散时空图预测
- [[std-mae]] — 时空解耦掩码预训练
- [[simdiff]] — 端到端扩散时间序列点预测
- [[ustd]] — USTD，解耦预训练的统一时空扩散预测与插值框架（SIGSPATIAL 2024）

[^src-diffstg]: [[source-diffstg]]
