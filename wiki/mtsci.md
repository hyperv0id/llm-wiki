---
title: "MTSCI"
type: technique
tags:
  - diffusion-models
  - data-imputation
  - conditional-diffusion
  - contrastive-learning
  - time-series
  - cikm-2024
created: 2026-08-29
last_updated: 2026-08-29
source_count: 1
confidence: medium
status: active
---

# MTSCI

**MTSCI**（Multivariate Time Series Consistent Imputation）是 Zhou、Li、Zheng（通讯作者）、Wang（上海交通大学）与 Zhou（中国科学院）发表的多元时间序列插补模型，ACM 出版信息为 CIKM '24（DOI 10.1145/3627673.3679532；raw PDF 为 arXiv:2408.05740v1 水印 + ACM CIKM'24 排版，著录核实见 [[source-mtsci]]）[^src-mtsci]。论文将其定位为针对插补一致性（imputation consistency）的条件扩散模型：以互补掩码对比损失处理窗口内一致性（intra-consistency），以 mixup 机制融合相邻窗口条件信息处理相邻窗口一致性（inter-consistency）[^src-mtsci]。

> [!note] 命名辨析
> 名称中的 "MT" 指 Multivariate Time Series，不是 multi-task：论文的任务设定是单一的多元时间序列插补（学习 $p_\theta(X^m \mid X^o)$，Sec. 3.1 Problem 1），不含预测任务，也没有预测与插补的联合训练[^src-mtsci]。同理，论文的 "consistent" 指插补结果与观测值/相邻窗口的统计一致性，与 [[consistency-models|Consistency Models]] 的少步生成自一致性不同义（辨析见 [[imputation-consistency]]）。

## 问题：插补一致性

论文将插补一致性分为两类（Sec. 1，论文提出的划分）[^src-mtsci]：

- **intra-consistency（窗口内一致性）**：插补值在观测值引导下应能反过来帮助重构观测值，使插补值与观测值保持一致、降低插补偏差；
- **inter-consistency（相邻窗口一致性）**：插补单个窗口样本时应考虑相邻窗口样本，使完整样本与相邻窗口保持时序一致。

论文认为既有方法只依赖人工模拟插补目标的归纳偏置（inductive bias）引导学习过程、忽略了插补一致性（Sec. 1，作者观点），并自述此前没有插补方法处理插补一致性问题（Sec. 1，论文自述）。选择扩散模型作为载体的理由是作者认为扩散模型倾向于生成与观测分布一致的数据、与插补一致性的概念契合（Sec. 1，作者表述）[^src-mtsci]。

## 机制

MTSCI 的整体流程：对 "sampled" 窗口 $x_{t-L:t}$ 在前向加噪阶段做对比互补掩码生成双视图，在去噪阶段用 intra contrastive module 与 inter-consistency condition network 保证两类一致性，总损失为去噪损失与对比损失的加权和（Sec. 4.1，Fig. 2）[^src-mtsci]。

### 前向：Contrastive Complementary Mask（Sec. 4.2）

基于自监督掩码建模（self-supervised mask modeling）产生的随机掩码矩阵 $m$，把 sampled 窗口切成两个互补视图：$\hat{x}_{1,t-L:t} = m \odot x_{t-L:t}$ 与 $\hat{x}_{2,t-L:t} = (1-m) \odot x_{t-L:t}$（Sec. 4.2.1）。每个视图按标准 DDPM 闭式加噪（式 11）：视图内的"插补目标"与"条件观测"互补——一个视图里被遮住的位置在另一个视图里恰好可观测[^src-mtsci]。

### 反向：Consistency-Assured Denoising（Sec. 4.3）

- **Intra contrastive module（Sec. 4.3.1）**：两个加噪视图经同一去噪网络的编码器得到表示 $z_1, z_2$，按 InfoNCE 形式的对比损失（式 12，余弦相似度、温度 $\tau$）约束两视图表示相似。论文表述：由此模型学到插补值在插补过程中也能重构观测值。
- **Inter-consistency condition network（Sec. 4.3.2）**：训练时以 "context" 窗口 $x_{t:t+L}$ 提供补充条件信息，用 mixup 机制（式 13）将其与 sampled 窗口观测混合：$x^{mix} = m_k \odot x^{co}_1 + (1-m_k) \odot F(x^{co}_2)$，其中 $F$ 为 1×1 卷积、混合系数矩阵 $m_k \sim U(0,1)$（mixup 机制沿用 Zhang et al. 与 Shen & Kwok 的工作）。推理时 context 窗口不可得，$m_k$ 全部置 1，条件退化为 sampled 窗口自身的观测值（Sec. 4.3.2、Sec. 4.5、Algorithm 2）[^src-mtsci]。

### 去噪网络与训练目标（Sec. 4.3.3、Sec. 4.4）

输入经线性嵌入加正弦扩散步嵌入；编码器每层由一个单层 vanilla transformer block（时间依赖）与一个单层 inverted transformer block（变量依赖）组成，论文称后者受 [[itransformer|iTransformer]] 启发（式 18）；解码器拼接各层输出、经两层全连接（ReLU）预测噪声[^src-mtsci]。总损失 $L = L_\epsilon + \lambda L_{CL}$（式 22），去噪损失只在缺失位置计算（式 21）[^src-mtsci]。

参数化选择上，作者报告在相同去噪网络架构下 ε-预测优于 $x_0$-预测（Table 4：ETT point RMSE 0.358 vs 0.775，Weather point 16.162 vs 22.739，METR-LA point 3.076 vs 4.111），作者推测原因是噪声服从高斯分布、利于互补掩码视图预测噪声并互相重构，而观测值与缺失值的分布未必相同（Sec. 5.3，作者解释）[^src-mtsci]。

## 实验（作者报告）

**设置**（Sec. 5.1）：数据集为 ETT（论文称 electricity 数据集；69,680 步、15 min、7 特征、窗口 24、原始缺失 0%）、Weather（52,696 步、10 min、21 特征、0.017%）与 METR-LA（34,272 步、5 min、207 特征、8.6%）（Table 1）。13 个基线：Mean、KNN、MICE、TRMF；BRITS、mTAN、SAITS、TimesNet、Non-stationary Transformer（表中缩写 Stationary）；GP-VAE、rGAIN、CSBI、[[csdi|CSDI]]。指标 MAE/RMSE/MAPE，5 次运行平均。人工缺失两种模式：point（随机遮 20% 数据点）与 block（在随机遮 5% 观测的基础上以 0.15% 概率遮长度 $[L/2, 2L]$ 的段）；训练用 point（随机选 $r \in [0\%, 100\%]$ 观测值为目标）与 block（以 $r \in [0\%, 15\%]$ 概率选长度 $[L/2, L]$ 的段）策略[^src-mtsci]。数据划分：论文 5.1 节称 ETT 取 2016/07–2016/10 四个月为测试集、2016/11–2017/02 为验证集、2017/03–2018/06 为训练集；Weather 与 METR-LA 按 70%/10%/20% 划分[^src-mtsci]。

**总体性能（Table 2/3，RQ1）**：Table 2（point）与 Table 3（block）各自的 9 个数据集 × 指标列（3 数据集 × MAE/RMSE/MAPE，两表共 18 格）数值均由 MTSCI 取得最低值。point 缺失下 MTSCI MAE 为 ETT 0.214、Weather 1.955、METR-LA 1.655（次优均为 CSDI：0.225 / 2.084 / 1.733）；block 缺失下为 ETT 0.642、Weather 3.092、METR-LA 1.982（CSDI：0.967 / 4.648 / 2.582）。作者报告相对基线平均提升 17.88% MAE、15.09% RMSE、13.64% MAPE，相对确定性方法平均提升 42.07% MAE、24.15% RMSE、39.76% MAPE（Sec. 5.2）[^src-mtsci]。

**一致性度量（Table 5，RQ2）**：论文用 CRPS 衡量插补结果与观测值在整个数据集上的 imputation consistency，六组设置（3 数据集 × point/block）MTSCI 数值均低于 CSDI（如 ETT point 0.0206 vs 0.0220、METR-LA block 0.0265 vs 0.0383）（Sec. 5.4）[^src-mtsci]。

**缺失率敏感性（Table 6，RQ3）**：在 Weather 上测 10%/30%/50%/70% 缺失率、对比 CSDI 与 CSBI（表中列序经 `pdftotext -layout` 复核为 MTSCI/CSDI/CSBI，流式抽取会打乱列序）。作者报告 MTSCI 在两种掩码模式的各缺失率下优于两个基线（Sec. 5.5）；按表分立：MAE 与 MAPE 在全部 8 组设置（2 模式 × 4 缺失率）低于两基线，RMSE 在 block 全部 4 档与 point 70% 低于两基线，而 point 10%/30%/50% 档 CSDI 的 RMSE 低于 MTSCI（15.524 / 14.952 / 16.923 vs 16.831 / 15.711 / 18.417）——论文的"优于基线"是总结性表述，非全指标最低。同时作者观察到插补性能不随缺失率上升而单调下降，将其归因于不完整时序中的分布偏移（distribution shift），并称一致性策略在一定程度上缓解了该问题（Sec. 5.5）[^src-mtsci]。

**跨缺失模式泛化（Table 7，RQ3）**：Point→Block 与 Block→Point（训练与测试用不同缺失模式）两种交叉设置下，论文的表述是 "achieves relatively better performance"。该表并非 MTSCI 全指标占优，本页按表分立记录：ETT Block→Point 上 CSDI 三项指标均低于 MTSCI（MAE 0.229 vs 0.345、RMSE 0.385 vs 0.679、MAPE 2.893 vs 4.360）；ETT Point→Block 上 CSBI 的 MAE（0.705 vs 0.707）与 MAPE（8.385 vs 9.191）低于 MTSCI、RMSE（2.881 vs 2.857）高于 MTSCI；METR-LA Point→Block 上 CSDI 的 RMSE（6.457 vs 6.485）略低于 MTSCI（其余指标 MTSCI 更低）[^src-mtsci]。

**消融（Fig. 3，RQ2）**：w/o intra（去互补掩码与对比损失）、w/o inter（去 mixup）、w/o cons（仅单窗口观测条件）三个变体均劣于完整模型（图示比较，论文未给出对应数值表）[^src-mtsci]。**超参（Fig. 5，RQ4）**：隐维度 $d$、编码器层数 $l$、最大噪声 $\beta_T$、对比损失权重 $\lambda$；论文称增大 $l$ 提升性能但同时增加参数量与计算复杂度，$\lambda$ 需小范围调节（Sec. 5.6）[^src-mtsci]。

## 与相关方法的关系

- **vs [[csdi|CSDI]]**：MTSCI 的条件扩散形式属 CSDI 一脉（Sec. 3.3 的条件扩散插补定义；CSDI 是其主扩散基线）。论文对 CSDI/CSBI 的批评是：它们仅靠自监督掩码策略生成插补目标、直接以插补目标上的归纳偏置引导去噪网络，因而误差明显（Sec. 5.2 观察 4，作者观点）[^src-mtsci]。
- **vs [[pristi|PriSTI]]**：MTSCI 相关工作节将 PriSTI 概括为"利用地理数据为时空插补提取条件信息"的扩散方法，归入未处理插补一致性的概率生成方法（Sec. 2，转述口径）[^src-mtsci]。
- **被后续工作引为基线/相关工作**：[[fence|FENCE]]（AAAI 2026）相关工作节将其与 CSBI、DSDI、[[costi|CoSTI]] 并列；[[loft|LOFT]]（KDD 2026）与 [[prdim|PRDIM]] 把 MTSCI 列入各自实验基线。
- **数字口径警示**：LOFT/FENCE/PRDIM 表格中的 MTSCI 数字是各论文自己的复现/适配口径，且任务设置不同（LOFT/FENCE 为 PEMS 交通高缺失插补，PRDIM 为 [[missing-not-at-random|MNAR]] 插补），与 MTSCI 原文在 ETT/Weather/METR-LA point/block 设置下的 Table 2/3 数字不可混用。

## 论文自述边界

- 结论节自述未来工作是将方法扩展到更复杂的缺失数据场景（Sec. 6）；论文未设独立局限性章节[^src-mtsci]。
- 机制层面：context 窗口只在训练期可用，推理期条件退化为单窗口观测（Sec. 4.3.2 的设计本身）[^src-mtsci]。
- 对比损失权重 $\lambda$ 需小范围调节以平衡训练目标（Sec. 5.6）[^src-mtsci]。

## 关联页面

- [[source-mtsci]] — 源文件摘要（raw/zhou-mtsci-arxiv-2024.pdf）
- [[imputation-consistency]] — 插补一致性概念（intra/inter），本页机制所服务的目标
- [[self-supervised-imputation-training]] — 自监督掩码训练范式，MTSCI 的 complementary mask 建立在其上
- [[contrastive-learning]] — 对比学习；MTSCI 的 intra loss 为 InfoNCE 式双视图对比
- [[csdi]] — CSDI，条件扩散插补源头，MTSCI 的主扩散基线
- [[pristi]] — PriSTI，被 MTSCI 相关工作归入未处理一致性的扩散插补
- [[fence]] / [[loft]] / [[prdim]] — 将 MTSCI 列为基线或相关工作的后续论文（数字口径不同，见上文警示）
- [[crps]] — CRPS，论文用于度量插补一致性的指标
- [[consistency-models]] — 另一种"一致性"（少步生成自一致性），与本文概念不同义
- [[mts-imputation-taxonomy]] — MTSI 综述分类框架，MTSCI 归入生成式-扩散类

[^src-mtsci]: [[source-mtsci]]
