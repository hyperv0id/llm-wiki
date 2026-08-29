---
title: "GiFlow"
type: entity
tags:
  - flow-matching
  - spatio-temporal-imputation
  - graph-signal-processing
  - prior-distribution
  - icml-2026
created: 2026-08-29
last_updated: 2026-08-29
source_count: 2
confidence: medium
status: active
---

# GiFlow (Graph-Informed Flow Matching)

**GiFlow** 是 Zhang、Einizade、Giraldo、Fink（EPFL IMOS、Télécom SudParis、Télécom Paris）发表于 ICML 2026 的时空插补模型：以时空图滤波构造的 [[graph-informed-prior|图信息先验]] 替代问题无关的高斯先验，向量场由空间注意力、时间注意力与时空传播混合参数化[^src-giflow]。论文将 GiFlow 定位为首个把图信息先验引入流匹配时空插补的框架（论文表述）[^src-giflow]。

## 问题设定与动机

时空数据表示为矩阵 X∈R^{N×R}（N 个节点、R 个时间步），二值掩码 M 标记观测，插补目标是估计缺失部分[^src-giflow]。论文对两条既有路线的批评：RNN/GNN 方法（BRITS、GRIN、SPIN、OPCR）依赖逐时空位置的迭代传播，误差累积且在高缺失率下出现信息瓶颈；扩散类方法（PriSTI 等）依赖问题无关的高斯先验，且需要多步去噪与多次采样取平均[^src-giflow]。流匹配允许任意源分布、推理为确定性 ODE 积分，为构造任务定制先验提供了空间[^src-giflow]。

## 核心机制

### 图信息先验

详见 [[graph-informed-prior]]。要点：将 vec(X₁ᴹ) 视为空间图与时间图笛卡尔积上的图信号，经指数滤波 X_τ = e^{−τηLη}X₁ᴹe^{−τξLξ} 生成先验样本；滤波因子 (τη, τξ) 由"信号对齐 + α_τ 加权拉普拉斯平滑"的最小化问题在训练数据上优化；Theorem 3.2 证明该先验的期望二次传输代价不高于高斯先验[^src-giflow]。

### 线性条件流

从先验到真实信号的线性条件路径 ϕt(X|Z) = (1−t)·e^{−τηLη}X₁ᴹe^{−τξLξ} + t·X₁，论文称线性路径在动能误差界意义下最优（引 Lipman et al. 2023）[^src-giflow]。由于先验是观测信号的确定函数，推理无需扩散方法的多次采样平均；需要不确定性量化时可向先验额外注入高斯噪声[^src-giflow]。

### 向量场模型

三组件混合参数化[^src-giflow]：

- **空间注意力**：节点嵌入经 GNN（Morris et al. 2019）传播后作为 key/query，对 X_t 经 MLP 的 value 做节点间自注意力；
- **时间注意力**：位置编码 + 可选时间戳嵌入（引 Informer），对时间步做自注意力；
- **时空传播**：两路注意力消息与原始特征、流步 t 的时间嵌入拼接投影后，做 L_MP 层 GNN 消息传递（Wu et al. 2019 的简化图卷积）——空间域逐时刻、时间域逐节点。

消融显示去除时空传播的性能损失最大；先验消融的影响整体大于架构消融[^src-giflow]。

## 实验结果（作者报告）

**设置**：合成数据（50 节点 KNN 图，谱域低频生成）、Air-36（36 站 PM2.5）、AQI（437 站、43 城市）、PeMS08（170 传感器）；point missing（随机遮蔽比例 ρ）与 block missing（节点内连续段遮蔽）两种注入方式，主表 ρ=20%，Air-36 另测 20–60%；窗口长度 24，70/10/20 划分，5 次重复取平均；默认 Euler 20 步[^src-giflow]。基线：非参数法（Mean-S/Mean-T/Linear/KNN/FP）、BRITS、SAITS、SPIN、GRIN、OPCR、PriSTI、CoSTI[^src-giflow]。

| 实验 | 设置 | 结果（作者报告） |
|------|------|------|
| 总体性能 | 3 个真实数据集 × point/block × MAE/RMSE/MAPE | 各设置总体最优，如 Air-36 point MAE 9.54（次优 GRIN 9.94）、AQI point MAE 7.83（次优 GRIN 7.97）、PeMS08 point MAE 12.66（次优 OPCR 12.77）、PeMS08 block MAE 18.70（次优 PriSTI 18.94） |
| 缺失率鲁棒性 | Air-36 point 20–60% | 论文报告各缺失模式与指标下总体最优（图 2 展示 MAE/MAPE），对比 SPIN/GRIN/OPCR/PriSTI |
| 先验消融（Table 4） | Air-36 point 20% | 传输代价 FM-Gauss 299.62 / TFM 123.39 / GFM 115.05 / GiFlow 104.29；MAE 12.79 / 10.12 / 9.75 / 9.54；FM-Gauss 劣于 Table 2 多个基线 |
| 架构消融（Table 5） | 同上 | 去时空传播降幅最大（MAE 10.40），其次去双注意力（10.22） |
| 图质量敏感性（Table 6） | 二值化阈值 0.02–0.6 | 0.05–0.4 内稳定（最佳 0.1），0.02 与 0.6 明显退化 |
| 推理时间（Table 7） | A100，测试集整体 | Air-36 0.28 min（PriSTI 9.30 / CoSTI 0.37）；AQI 2.47（43.12 / 8.41）；PeMS08 0.99（7.46 / 3.63） |
| Euler 步数（Table 9） | 1–20 步 | 20 步最佳；5 步 MAE 9.81 / RMSE 18.95 仍优于次优基线；1 步 MAE 9.87 |
| 滤波因子优化开销（Table 8） | 100 epochs，batch 64 | Air-36 0.19 min；5 万节点合成图 69.85 min / 19.88 GB |

> [!note] CoSTI 数字口径
> Table 7 中 CoSTI 的推理时间为 GiFlow 论文在 A100 环境的复测口径，非 CoSTI 原文报告的数字；[[costi|CoSTI]] 原文（Table 3）在 RTX A5000 上报告 AQI-36 0.005 h、METR-LA 0.06 h 等，且其数据集与缺失设置与 GiFlow 不完全相同，两套数字不可混用[^src-costi]。

## 定位与相关方法

| 方法族 | 代表 | 先验 | 与 GiFlow 的关系 |
|--------|------|------|------|
| 条件扩散插补 | [[csdi\|CSDI]]、[[pristi\|PriSTI]]、[[cofill\|CoFILL]] | 高斯 | PriSTI 为基线：多步去噪 + 多次采样平均；GiFlow 确定性 ODE 积分 |
| 一致性插补 | [[costi\|CoSTI]] (KBS 2025) | 高斯 | CoSTI 为基线：加速采样但先验仍是高斯[^src-costi] |
| 信息先验流匹配 | [[tsflow\|TSFlow]]（GP 先验，预测）、[[loft\|LOFT]]（低秩先验 + 轨迹一致性，交通插补） | 任务定制 | 同属"先验对齐目标分布"路线；GiFlow 的先验由图滤波闭式给出 |
| 时空 GNN/Transformer | GRIN、SPIN、OPCR、SAITS、BRITS | — | 迭代传播或逐序列建模基线 |

论文指出一致性模型思想已被扩展到流匹配框架（Liu et al. 2025b，CVPR 2025），但该扩展在时空插补中的有效性截至其写作（2026-06）尚未被研究（论文表述）[^src-giflow]——[[trajectory-consistency-flow-matching|LOFT 的轨迹一致性]] 正是沿这一方向的同期工作，但两篇论文互未对比。

## 局限

- 滤波因子优化依赖含完整真值的训练数据，推理时固定；α_τ 为验证集调参项（搜索空间 0.1–0.0001）[^src-giflow]
- Theorem 3.2 限于期望二次传输代价，与下游插补精度仅在消融中经验相关[^src-giflow]
- 需要图结构输入；空间图由训练数据经高斯核 + 阈值二值化构造，图质量极端偏差时性能退化[^src-giflow]
- 实验为 point/block 注入缺失，未讨论 MNAR 场景[^src-giflow]

## 相关页面

- [[source-giflow]] — 源文件摘要
- [[graph-informed-prior]] — 图信息先验：时空图滤波、传输代价定理与自适应感受野
- [[loft]] — LOFT，低秩先验 + 轨迹一致性的流匹配插补（KDD 2026）
- [[tsflow]] — TSFlow，GP 先验流匹配（预测任务对照）
- [[flow-matching]] — Flow Matching 理论基础
- [[pristi]] — PriSTI，GiFlow 的扩散基线
- [[grin]] — GRIN，GiFlow 的时空 GNN 基线
- [[cofill]] — CoFILL，GiFlow 相关工作中引用的扩散插补
- [[message-passing-imputation]] — 消息传递插补范式（GiFlow 批评的迭代传播路线）
- [[costi]] — CoSTI（KBS 2025），一致性训练插补，GiFlow 的基线之一（Table 7 推理时间对比）[^src-costi]
- [[mts-imputation-taxonomy]] — MTSI 综述（arXiv:2402.04059）分类框架页，含流匹配插补路线相对该综述框架的定位讨论

[^src-giflow]: [[source-giflow]]
[^src-costi]: [[source-costi]]
