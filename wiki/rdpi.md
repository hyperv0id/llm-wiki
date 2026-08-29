---
title: "RDPI: A Refine Diffusion Probability Generation Method for Spatiotemporal Data Imputation"
type: technique
tags:
  - diffusion-models
  - spatio-temporal-imputation
  - conditional-diffusion
  - residual-learning
  - two-stage
  - traffic
  - air-quality
  - aaaai-2025
created: 2026-08-29
last_updated: 2026-08-29
source_count: 2
confidence: medium
status: active
---

# RDPI

**RDPI**（论文题目：*RDPI: A Refine Diffusion Probability Generation Method for Spatiotemporal Data Imputation*）是 Liu、Zhao 与 Song（北京航空航天大学，You Song 为通讯作者）提出的两阶段时空插补框架，主张"确定性初值 + 残差条件扩散精炼"。PDF 为 arXiv v1（2024-12-17，双栏 AAAI 模板排版），首页含 AAAI 2025 版权声明（Copyright © 2025, AAAI），但无会议论文集页眉等标识；AAAI 2025 会议著录来自用户，与该版权声明一致，未在 PDF 内以 proceedings 形式核实[^src-rdpi]。论文对缩写的展开不统一：摘要作 "Refined Diffusion Probability Impuation"（原文拼写如此），Introduction 作小写的 "refined diffusion probability imputation"，Method 节作 "Refine Diffusion Probability Imputation"；本页沿用题目口径[^src-rdpi]。

## 问题与动机

论文把时空插补方法分为确定性、概率与扩散三类[^src-rdpi]。作者的批评指向两类方法：确定性/自回归方法缺少不确定性建模、易误差累积；以 [[csdi|CSDI]] 为代表的条件扩散模型只在反向去噪训练中使用观测条件，"在前向与插补过程中忽略条件"，难以充分捕捉观测与缺失数据之间的时空依赖（论文表述，Introduction 与 Related Work 两处重申）[^src-rdpi]。对应地，RDPI 的两个改造点是：把观测值写入扩散前向过程（见 [[forward-process-conditioning]]），以及把扩散目标从缺失值本身换成初始估计与真值之间的残差[^src-rdpi]。

## 方法

### 两阶段框架

- **初始阶段**：确定性插补模型 $f_\theta$ 从观测值 $x_0^c$ 生成粗插值。论文强调 $f_\theta$ 原则上可取任意插补方法（包括扩散模型），但两阶段都用扩散会增加随机性与迭代采样开销，故实验取确定性方法——具体用 [[grin|GRIN]]（Initial Stage 节）[^src-rdpi]。
- **精炼阶段**：条件扩散模型 $g_\theta$ 以残差为扩散目标。按论文 Algorithm 1 第 4 行，残差取 $z_0^m = f_\theta(x_0^c) - x_0^m$（初值减真值），初始损失 $L_{init} = \|z_0^m\|$；采样时最终插补为 $\hat{x}_0^m = x_{init}^m - z_0^m$，即从初值中减去预测残差（Algorithm 2 第 8 行）[^src-rdpi]。
- **联合训练**：$L_{joint} = L_{simple} + \lambda L_{init}$（Eq 14），Algorithm 1 第 9-10 行对 $\nabla_\theta L_{joint}$ 做梯度下降。论文说明初始模型并非必须预训练（联合训练的梯度会流入初始模型侧，论文此处引 Whang et al. 2022），但为稳定扩散模型的训练推荐预训练（Initial Stage 节；Algorithm 1 第 1 行）[^src-rdpi]。

### 残差目标与 stochastic refinement 的关系

残差扩散精化与图像复原中的 stochastic refinement（Whang et al. 2022，CVPR）直接相关：论文在 Initial Stage 节引用该方法，并称与它的关键差异在损失与目标的方向——RDPI 取 $\|f_\theta(x_0^c) - x_0^m\|$ 而非 $\|x_0^m - f_\theta(x_0^c)\|$；消融中把扩散目标反向为 $x_0^m - f_\theta(x_0^c)$（$-f_\theta$ 变体）后性能显著下降（AQI36 In-sample MAE 9.13 vs RDPI 7.98；METR-LA Block 2.63 vs 1.96，Table 5），作者解释为目标反向后去噪损失下降时初值模型损失上升、残差失稳（消融分析 (5)，论文口径）[^src-rdpi]。

### 前向过程条件化

详见 [[forward-process-conditioning]]。要点：前向转移以观测值 $z_0^c$ 为条件（论文 Eq 3），并给出边际与重参数化形式（Eq 4-5）；作者据此推导含前向条件的 ELBO（Eq 6，推导在附录）与 ε-预测训练目标（Eq 11-13）[^src-rdpi]。

### 去噪网络与加速采样

去噪网络四个组件（Figure 2）：1D 卷积特征嵌入（拼接观测值与残差后卷积，论文写作 $E_f = \mathrm{CNN}(z_0^c \| z_0^m)$，Eq 16）加可学习时间窗嵌入与空间嵌入（Eq 17）、时间自注意力（Eq 18）、带邻接矩阵 $A$ 的图神经网络（Eq 19）、空间自注意力；反向方差 $\Sigma_\theta$ 固定为常数（Reverse Process 节）[^src-rdpi]。采样可采用 DDIM 式非马尔可夫加速方案（Eq 15、Algorithm 3）；论文强调与原版 DDIM 不同，RDPI 保留加速方案中的全部随机项，认为去除随机性的版本可能损害插补质量（Imputation 节，论文原文 "a version without randomness may compromise the quality of final data imputation"）[^src-rdpi]。

## 实验结果（作者报告）

### 设置

- **数据集**（Table 1）：PEMS-BAY 325 节点 / 52128 时间片 / 0.023% 缺失，METR-LA 207 / 34272 / 8.11%，AQI36 36 / 8759 / 13.24%，AQI 437 / 8760 / 25.67%。注意：正文数据集段落把 PEMS-BAY 写作 207 节点、METR-LA 写作 325 节点，与 Table 1 互换，且该段称 "Table 1 shows ... for three datasets" 而表列四个数据集；论文未作说明，本页统计引自 Table 1[^src-rdpi]。
- **协议**（Settings 节，沿用 GRIN 与 PriSTI (Liu et al. 2023) 的设置）：交通数据按 70%/10%/20% 划分，缺失模式为 Point missing（随机遮蔽 25% 观测值）与 Block missing（先遮蔽 5%，再对每个节点以 0.15% 概率遮蔽 1-4 小时连续数据）；空气质量数据分 In-sample（按数据集标注的缺失位置训练）与 Out-of-sample（随机生成新缺失位置）两种训练场景，第 3/6/9/12 月为测试集[^src-rdpi]。
- **基线**（Baseline 节）：MEAN、KNN、MICE、VAR、BRITS、GRIN、rGAIN、GP-VAE、CSDI、MIDM（Wang et al. 2023，KDD 2023）；不含 PriSTI。指标为 MAE/MSE/MRE，全部实验运行 5 次[^src-rdpi]。

### 主结果

- **Table 3（AQI36/AQI，In/Out-of-sample）**：RDPI 的 MAE 与 MSE 在全部数据集×模式下最优。AQI36 In-sample：RDPI 7.98±0.24 / 238.25±13.22，MIDM 9.41±0.20 / 361.28±21.33，CSDI 9.60±0.14 / 372.49±16.90；AQI In-sample：RDPI 9.10±0.33 / 266.81±13.65，MIDM 10.06±0.11 / 562.84±12.01，CSDI 11.37±0.12 / 589.31±11.20。作者称 In-sample 下 MSE 相对最近基线在 AQI36 降低超过 34%、在 AQI 降低超过 50%（Results 节；按 Table 3 数字计算分别为约 34.1% 与 52.6%，与该表述一致）[^src-rdpi]。
- **MRE 例外**：AQI In-sample 上 MIDM 的 MRE 16.87% 低于 RDPI 的 17.17%（Table 3）；作者称 MRE 不能充分反映模型表现，归因于概率采样对残差的平滑作用主要改善 MSE（Results 节，论文口径）[^src-rdpi]。
- **Table 4（PEMS-BAY/METR-LA，Block/Point missing）**：RDPI 在两组数据集的两类缺失下 MAE/MSE/MRE 全部最低，如 PEMS-BAY Block MAE 0.90±0.01（MIDM 1.03、GRIN 1.14、CSDI 1.16）、METR-LA Block MAE 1.96±0.01（CSDI 1.98、GRIN 2.03）[^src-rdpi]。

### 敏感性、消融、概率插补与超参数

- **RQ2（Figure 3）**：METR-LA 缺失率 10%-90%，对比 BRITS、GRIN、CSDI。作者报告 RDPI 在各缺失率下保持最好，且扩散类（RDPI/CSDI）优于确定性类（BRITS/GRIN）[^src-rdpi]。
- **RQ3 消融（Table 5，AQI36 In-sample + METR-LA Block）**：六个变体按 MAE 均劣于完整 RDPI——w/o cond-forw（前向不用观测条件）、w/o residual（直接扩散缺失值本身）、w/o joint（冻结初始模型）、w/o pre-train（不预训练初始模型）、predicting $x_\theta$（预测数据而非噪声）、$-f_\theta(x_0^c)$（残差目标反向）。作者的解释（消融分析 (1)-(5)，论文口径）：前向无条件时无法利用观测-缺失关系；仅扩散模型自身已有较强生成力；初始模型的状态影响残差稳定性，w/o joint 的退化被归因于扩散模型对训练集残差的过拟合；残差均值小导致信噪比低、预测噪声更难；目标反向使初值损失与去噪损失此消彼长[^src-rdpi]。
  - > [!note] 数字与正文总结的分立记录
    > predicting $x_\theta$ 变体在 AQI36 In-sample 的 MSE 为 153.66±15.90，低于完整 RDPI 的 238.25±13.22（MAE 8.29 vs 7.98、MRE 11.69 vs 11.67 则 RDPI 略优）；论文正文总结"x 预测不是最优选择"，该总结与 Table 5 在 AQI36 上的 MSE 数字不完全一致，按口径分立记录，不作取舍[^src-rdpi]。
- **RQ4 概率插补与 Kriging**：AQI36 上 50 次采样取中位数与 5/95 分位做可视化（Figure 4）；全节点遮蔽设定下遮蔽连接度最高（节点 14）与最低（节点 31）的节点并全网重建：RDPI MAE 9.50 / 15.28 vs GRIN 13.75 / 20.55（Table 6），作者称 MAE 分别降低 31% 与 26%[^src-rdpi]。
- **RQ5 超参数（Figure 6）**：AQI36 上对扩散步数、$\lambda$ 与加速步数做敏感性分析，作者结论"更大不总是更好"。超参数（Table 2）：PEMS-BAY/METR-LA 扩散 50 步、加速 10 步、$\lambda=0.2$；AQI36/AQI 扩散 100 步、加速 40 步、$\lambda=0.5$[^src-rdpi]。

## 口径与注意

- "state-of-the-art imputation accuracy"（摘要）与 "achieving state-of-the-art performance"（贡献列表）为论文自述；本页全部数字出自该论文自己的 Table 1-6，未经第三方复现核实。
- 摘要另称 RDPI "significantly reduces sampling computational costs"（论文自述）；正文无采样耗时或计算成本的对比数据（Results 节仅有机制性说明：初始模型提供粗插值，去噪模型只需估计残差分布），该主张没有对应的实验表格支撑。
- Table 3/4 中部分基线单元格为空：Table 3 的 Out-of-sample 中，CSDI 与 GP-VAE 在 AQI 整行未报告、MIDM 在 AQI36/AQI 整行未报告、GP-VAE 与 CSDI 在 AQI36 的 MRE 未报告；Table 4 中 CSDI 与 GP-VAE 在 METR-LA 两种缺失下的 MRE 未报告。相应"最优"是在已报告的基线内比较。
- 基线不含 PriSTI。RDPI 的划分与缺失协议自述沿用 GRIN 与 PriSTI（Settings 节）[^src-rdpi]；[[pristi|PriSTI]] 的实验数据集为 PEMS-BAY、METR-LA、AQI36（未含 437 节点的 AQI）[^src-pristi]。两篇论文各自报告各自设置下的结果，数字不可混用。
- 论文未设独立局限性章节；MRE 的说明（Results 节）与未来工作（时空解耦、面向图结构与多维时间序列的条件扩散，Conclusion 节）为论文自述的范围边界。

## 关联页面

- [[source-rdpi]] — 源文件摘要
- [[forward-process-conditioning]] — RDPI 自述的核心机制改造：观测值进入前向过程
- [[csdi]] — RDPI 论文的主要批评对象与扩散基线
- [[grin]] — RDPI 实验中的确定性初始模型
- [[pristi]] — 扩散插补方法，RDPI 的划分与缺失协议自述沿用其设置（数据集重叠 PEMS-BAY/METR-LA/AQI36），但未被 RDPI 列为基线
- [[two-stage-imputation]] — 双阶段插补范式（网络级与框架级两种用法辨析）
- [[ddpm]] / [[ddim]] — RDPI 的扩散训练与加速采样基础

[^src-rdpi]: [[source-rdpi]]
[^src-pristi]: [[source-pristi]]
