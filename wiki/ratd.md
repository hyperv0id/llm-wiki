---
title: "RATD"
type: entity
tags:
  - diffusion-models
  - time-series-forecasting
  - retrieval-augmented-generation
  - probabilistic-forecasting
  - reference-guidance
  - neurips-2024
created: 2026-06-08
last_updated: 2026-08-06
source_count: 2
confidence: medium
status: active
---

# RATD (Retrieval-Augmented Time series Diffusion)

**RATD** 是首个检索增强的时间序列扩散预测模型，由 Jingwei Liu、Ling Yang、Hongyan Li、Shenda Hong（北京大学）发表于 NeurIPS 2024[^src-ratd]。其核心思想是：从一个外部数据库中检索与历史序列最相似的 k 个近邻样本作为「参照（references）」，并用这些参照引导扩散模型的去噪过程，从而在数据不足、类别不平衡的场景下提升复杂预测任务的精度[^src-ratd]。

## 动机

RATD 针对现有时间序列扩散模型（如 [[timegrad|TimeGrad]]、[[csdi|CSDI]]）的两个限制因素[^src-ratd]：

1. **缺乏有意义的引导（absence of guidance）**：与图像扩散可用文本/标签引导不同，多数时间序列缺乏语义或标签对应关系，导致去噪过程缺少引导信号，模型潜力受限[^src-ratd]。
2. **数据集规模不足且不平衡（insufficient & imbalanced）**：时间序列数据集规模通常远小于图像数据集（如 LAION-400M 含 4 亿样本对，而多数时序数据集仅数万数据点），且现实数据严重不平衡（例如 MIMIC-IV 中预激综合征 PS 记录占比 < 0.025%）。这导致模型倾向于生成「常见」预测，难以处理罕见的复杂样本[^src-ratd]。

RATD 的解决思路是**最大化利用已有数据集**——通过检索机制把训练库中与历史序列最相关的样本调出来当参照，既补偿了引导的缺失，又在一定程度上缓解了数据不平衡问题[^src-ratd]。

## 框架

RATD 由两部分组成：嵌入式检索（embedding-based retrieval）+ 参照引导的扩散模型（reference-guided diffusion model）[^src-ratd]。整体流程建立在 DiffWave 的扩散框架与 2D Transformer 结构之上（与 [[csdi|CSDI]] 的双轴 Transformer 架构同源）[^src-ratd]。

```
历史序列 x^H ──► 预训练编码器 E_φ ──► 查询向量 v^H
                                          │
                  数据库 D^R ──► E_φ ──► 嵌入库 ──► L2 最近邻检索 (Top-k)
                                          │
                                          ▼
                                  参照 x^R (k 条)
                                          │
[x^H, x_t] + 侧信息 I_s + 参照 x^R ──► RMA ──► μ_θ 去噪 ──► x_{t-1}
```

### 1. 检索数据库构建

针对两类不同数据集采用两种数据库定义[^src-ratd]：

| 数据集类型 | 例子 | 数据库定义 |
|-----------|------|-----------|
| 规模不足、难单类别标注 | 电力时序 | 直接以整个训练集为库 $D^R = \{x_i \mid \forall x_i \in D^{train}\}$ |
| 有完整标签但类别不平衡 | 医疗时序 (MIMIC) | 取含全部类别样本的子集 $D^R = \{x^c_i, \dots \mid \forall c \in C\}$ |

实验中 MIMIC 用第二种策略，其余四个数据集用第一种[^src-ratd]。

### 2. 嵌入式检索机制

理想参照是「其前 n 个点与历史序列最相关」的样本。RATD 用**预训练编码器** $E_\phi$（参数冻结，在表示学习任务上训练，实验中默认用 TCN）将序列嵌入，再用嵌入间的 L2 距离衡量相似度[^src-ratd]：

$$\text{index}(v^H) = \arg\min_{x_i \in D^R_{emb}}{}^{k} \| v^H - E_\phi(x_{i[0:n]}) \|^2$$

检索到 k 个最小距离对应样本的**未来段** $x^R = \{x_{j[n:n+h]}\}$ 作为参照[^src-ratd]。为降低训练成本，检索过程被预处理——把每个训练样本的参照索引存进字典，训练时直接查表，避免重复检索[^src-ratd]。

### 3. Reference Modulated Attention (RMA)

RMA 是 RATD 的关键模块，用于把参照信息注入去噪网络[^src-ratd]。与普通注意力不同，RMA 通过矩阵点积**融合三种特征**：当前时序特征 $[x^H, x_t]$、侧信息 $I_s$、参照特征 $x^R$（所有参照拼接后统一提取特征）[^src-ratd]。RMA 被放在每个残差模块的开头（消融实验证明放在双向 Transformer **前面**效果最佳）[^src-ratd]。

> [!note] RMA 与 MiDDiR 检索引导的本质区别
> RATD 把检索结果作为**条件特征输入**，经 RMA 注意力融合进去噪网络（类似文-图扩散的 cross-attention 条件化）；而 [[retrieval-guidance|MiDDiR 的检索引导]]（ICLR 2026）是**分析性地修改采样过程的得分函数**（指数倾斜 + 能量梯度）。两者机制不同，因此 RATD「首个检索增强时序扩散」与 MiDDiR「首个将检索结合到扩散的分析性引导」的「首创」声明并不矛盾。

### 训练与去噪目标

RATD 网络**预测干净数据 $x_0$**（[[x-prediction|x-prediction]]）而非噪声 $\epsilon$，损失为[^src-ratd]：

$$L_{t-1} = \gamma_t \| x_0 - \hat{x}_0 \|^2$$

消融实验显示 $x_0$-prediction 优于 $\epsilon$-prediction（Wind MSE 0.784 vs 0.841），作者推测因为参照与 $x_0$ 的关系更直接，使去噪任务更容易[^src-ratd]。这与基线 [[csdi|CSDI]]/[[timegrad|TimeGrad]] 采用的 $\epsilon$-prediction 形成对比。采样用非自回归 Transformer 框架，扩散步 $T=100$[^src-ratd]。

## 实验结果

在 5 个真实数据集上评测（Exchange、Wind、Electricity、Weather + 医疗 ECG 数据集 MIMIC-IV-ECG），指标为 MSE/MAE/CRPS，历史长度 168，预测长度取 (96, 192, 336) 平均[^src-ratd]。

- **四个日常数据集**：RATD 超越现有时序扩散模型（CSDI、TimeDiff、mr-Diff、D³VAE），在 4 个数据集中 3 个超越全部基线、剩余 1 个有竞争力；在缺乏明显短期周期性的 **Wind** 数据集上优势最突出（MSE 0.784 vs CSDI 1.066）[^src-ratd]。
- **MIMIC-IV-ECG**：在完整测试集上与 iTransformer 相近，但在**罕见病例子集**（占比 < 2%）上显著领先，验证了检索增强对复杂/罕见任务的针对性优势[^src-ratd]。
- **推理效率**：尽管多了检索模块，得益于非自回归 Transformer，RATD 采样效率不低，甚至略快于 TimeGrad、MG-TSD、SSSD 等基线（TimeGrad 因自回归解码最慢）[^src-ratd]。

## 消融发现

- **检索机制有效性**：移除检索（无参照）或用随机参照作基线，所有合理检索方法都正面提升预测；说明合理参照对引导生成高度有效[^src-ratd]。
- **检索方式**：embedding-based 检索显著优于 correlation-based（DTW / Pearson）——后者无法捕获关键特征[^src-ratd]。不同编码器（DLinear、Informer、TimesNet、TCN）差异不大，TCN 在计算成本与性能间最平衡[^src-ratd]。
- **数据库超参 n, k**：增大 n（每类样本数）因多样性提升而改善精度；单纯增大 k（参照数）改善不明显，过多参照反引入噪声。最终设 n=256, k=3[^src-ratd]。
- **RMA 优于其他融合**：以 CSDI 为基线，CSDI+RMA 全面优于 CSDI+Linear、CSDI+CrossAttention——RMA 能更有效整合表示时间/特征相关性的边信息矩阵[^src-ratd]。

## 局限性

1. **Transformer 框架的计算开销**：处理变量过多的时序时消耗大量算力[^src-ratd]。
2. **额外预处理成本**：训练前的检索预处理增加约十小时训练时间[^src-ratd]。

## 关联页面

- [[source-ratd]] — 完整源文件摘要
- [[diffusion-models]] — 扩散模型概念总览
- [[csdi|CSDI]] — RATD 的架构基础（双轴 Transformer），同时是主要扩散基线 (NeurIPS 2021)
- [[timegrad|TimeGrad]] — 首个时序扩散预测模型，RATD 引用的奠基工作 (ICML 2021)
- [[generative-time-series-forecasting]] — 生成式时间序列预测范式
- [[x-prediction]] — RATD 采用的 $x_0$-预测参数化
- [[retrieval-augmented-spatio-temporal-forecasting]] — RAG-for-时空预测范式
- [[retrieval-guidance|Retrieval Guidance]] — MiDDiR 的分析性检索引导（与 RMA 机制不同）
- [[middir|MiDDiR]] — 后续检索增强扩散模型 (ICLR 2026)
- [[craft|CRAFT]] — 跨城市检索增强扩散生成 (NeurIPS 2025)
- [[rast|RAST]] — 检索增强时空交通预测 (AAAI 2026)
- [[gtr|GTR]] — 全局时序检索模块 (ICLR 2026)
- PIR（Liu et al., NeurIPS 2025）— 检索增强预测家族的确定性后处理成员：全局修订用实例检索 + top-K 加权求和，不经扩散去噪[^src-pir]

[^src-ratd]: [[source-ratd]]
[^src-pir]: [[source-pir]]
