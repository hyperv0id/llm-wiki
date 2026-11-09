---
title: "Joint-Distribution Wasserstein Alignment (DistDF)"
type: technique
tags:
  - time-series-forecasting
  - learning-objective
  - optimal-transport
  - wasserstein
  - distribution-alignment
created: 2026-07-13
last_updated: 2026-11-09
source_count: 3
confidence: high
status: active
---

# 联合分布 Wasserstein 对齐（Joint-Distribution Wasserstein Alignment）

**是什么**：一种训练多步时序预测模型的目标构造方式。它不去估计标签序列的条件似然，而是最小化预测序列与标签序列在给定历史 $x$ 下的条件分布差异；由于条件差异本身无法从有限观测估计，改用联合分布 $(\mathbb{P}_{x,y},\mathbb{P}_{x,\hat{y}})$ 之间的 Wasserstein 差异作为上界代理。该构造由 [[distdf|DistDF]] 提出（ICLR 2026），用于替换 [[direct-forecast|DF]] 的 MSE 目标[^src-distdf]。

## 起点：条件分布差异无法直接估计

训练目标是让 $\mathbb{P}_{\hat{y}\mid x}$ 与 $\mathbb{P}_{y\mid x}$ 一致。最直接的做法是最小化 $W_p(\mathbb{P}_{y\mid x},\mathbb{P}_{\hat{y}\mid x})$，但给定 $x$ 时数据集通常只提供一条标签序列 $y$，模型也只输出一条 $\hat{y}$，于是两侧经验集 $S_{y\mid x}$、$S_{\hat{y}\mid x}$ 各只有一个样本，经验分布退化为单点质量，差异值不携带信息[^src-distdf]。

论文的 Footnote 3 明确这一点不是 Wasserstein 的缺陷：任何分布差异度量在样本量为一的情况下都会退化[^src-distdf]。

## 联合分布 Wasserstein 上界（Definition 3.2、Lemma 3.3、Theorem 3.4）

Definition 3.2 取离散 Kantorovich 形式：对经验样本 $S_\alpha=[\alpha_1,\dots,\alpha_n]$、$S_\beta=[\beta_1,\dots,\beta_m]$ 与样本权重 $a$、$b$，$p$-Wasserstein 差异为[^src-distdf]

$$
W_p(\mathbb{P}_\alpha,\mathbb{P}_\beta):=\min_{P\in\Pi(\alpha,\beta)}\langle D,P\rangle,\qquad d_{i,j}=\lVert\alpha_i-\beta_j\rVert_p^p,
$$

其中 $\Pi(\alpha,\beta)=\{P\in\mathbb{R}^{n\times m}_{+}\mid P\mathbf{1}_m=a,\ P^{\top}\mathbf{1}_n=b\}$ 是行和为 $a$、列和为 $b$ 的传输计划集合，$D$ 为成对代价矩阵。求解与近似路径见 [[optimal-transport]]。

Lemma 3.3（Kim et al. 2022, Theorem 2）给期望条件差异一个可由整批样本估计的上界[^src-distdf]：

$$
\int W_p\!\left(\mathbb{P}_{y\mid x},\mathbb{P}_{\hat{y}\mid x}\right)d\mathbb{P}(x)\ \leq\ W_p\!\left(\mathbb{P}_{x,y},\mathbb{P}_{x,\hat{y}}\right),
$$

当 $p=1$，或条件 Wasserstein 项对 $x$ 为常数时取等。

Theorem 3.4：若 $W_p(\mathbb{P}_{x,y},\mathbb{P}_{x,\hat{y}})=0$，则几乎处处 $\mathbb{P}_{y\mid x}=\mathbb{P}_{\hat{y}\mid x}$[^src-distdf]。Lemma 3.3 左端的被积函数非负，右端为零时 $W_p(\mathbb{P}_{y\mid x},\mathbb{P}_{\hat{y}\mid x})=0$ 几乎处处成立；再由 Wasserstein 度量的正定性得两个条件分布相等。Appendix A.3 对离散经验测度给出这一步的逐项论证：最优计划满足 $p^*_{i,j}\lVert\alpha_i-\beta_j\rVert_p^p=0$，即质量只在同一点之间搬运。

## Lemma 3.5 与式 (6)：可微实现

Theorem 3.4 只保证「最小到零则对齐」，不给出可计算形式。加高斯假设后，$\mathbb{P}_{x,y}=\mathcal{N}(\mu_{x,y},\Sigma_{x,y})$、$\mathbb{P}_{x,\hat{y}}=\mathcal{N}(\mu_{x,\hat{y}},\Sigma_{x,\hat{y}})$，平方 $W_2$ 取 [[bures-wasserstein|Bures–Wasserstein]] 闭式解（Lemma 3.5，证明见 Peyré & Cuturi 2019 Remark 2.31）[^src-distdf]：

$$
\mathrm{BW}(\mu_{x,y},\mu_{x,\hat{y}},\Sigma_{x,y},\Sigma_{x,\hat{y}})=\lVert\mu_{x,y}-\mu_{x,\hat{y}}\rVert_2^2+\mathcal{B}(\Sigma_{x,y},\Sigma_{x,\hat{y}}),
$$

$$
\mathcal{B}(\Sigma_1,\Sigma_2)=\operatorname{Tr}\!\left(\Sigma_1+\Sigma_2-2\left(\Sigma_1^{1/2}\Sigma_2\Sigma_1^{1/2}\right)^{1/2}\right).
$$

矩统计丢掉了 batch 内样本与标签的逐元素对应，这项信息对训练预测模型是必需的，因此加入 MSE 守住点对点对齐[^src-distdf]：

$$
\mathcal{L}_{\mathrm{DistDF}}:=\gamma\cdot\mathcal{L}_{\mathrm{Dist}}+(1-\gamma)\cdot\mathcal{L}_{\mathrm{MSE}},\qquad 0\leq\gamma\leq1 .
$$

实际计算走 Algorithm 1：$Z=[X,Y]$、$\hat{Z}=[X,\hat{Y}]$ 沿时间轴拼接，取 batch 维的均值与协方差，代入上式。完整步骤与开销见 [[distdf]]。

## 与其他分布差异度量的对比（Table 3）

在相同拼接与矩估计框架下将 BW 替换为 EMD、MMD（线性核 / RBF 核）或 KL 散度后，各分布差异度量均不劣于纯 MSE，其中 BW 在 2 个骨干 × 4 个数据集的 16 组指标中取得 14 组最优（完整数值见 [[distdf]] Table 3）[^src-distdf]。

## 权重 $\gamma$ 的行为（Table 4、Table 5）

$\gamma=0$ 退化为纯 DF，$\gamma=1$ 表示只用矩统计、丢弃样本级对应。扫描 $\gamma\in\{0,0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.5,1\}$ 后：TimeBridge 在 ECL 上 $\gamma=0.001\text{--}0.01$ 落在 $0.172\text{--}0.173/0.267$，$\gamma=1$ 退化到 $0.197/0.282$；Weather 从 $\gamma=0.01$ 的 $0.249/0.276$ 退到 $\gamma=1$ 的 $0.265/0.286$；ETTh2 的最优点在 $\gamma=0.05\text{--}0.1$（$0.375/0.399$）。Fredformer 在 ECL 上 $\gamma=0$ 为 $0.191/0.284$、$\gamma=0.002$ 为 $0.181/0.274$、$\gamma=1$ 为 $0.239/0.326$[^src-distdf]。

结论是分布项起正则作用、不能单独使用：按数据集不同最优 $\gamma$ 落在 $0.001\text{--}0.1$，跨两个数量级，论文在 §4.1 因此要求同时搜索学习率 $\eta$[^src-distdf]。

## 与领域自适应中联合 OT 的区别

联合分布最优传输在领域自适应里用于对齐输入侧分布以改善跨域泛化（Courty et al. 2017）。论文明确指出本工作的对齐对象是模型输出与标签的条件分布，服务于多任务监督学习下的训练目标设计，而非跨域迁移（§3.2）[^src-distdf]。落到构造上，DistDF 的 $Z=[X,Y]$ 与 $\hat{Z}=[X,\hat{Y}]$ 共享同一段历史，差异只来自 $\hat{y}$ 与 $y$，对齐的是输出侧而非输入侧（Algorithm 1）[^src-distdf]。

## 同一诊断下的其他路线

MSE 的似然偏差还有两种处理方式，可作为对照：

- [[fredf|FreDF]]：把标签变换到 Fourier 正交基，在频域分量上算似然，保证的是分量间的边缘去相关（DML 测得约 $3.6\%$ 非对角元超过 $0.1$，Weather，$T=192$）[^src-fredf]。技术细节见 [[frequency-enhanced-direct-forecast]]。
- [[qdf|QDF]]：保留二次型似然 $L_\Sigma=\lVert Y-\hat{Y}\rVert_{\Sigma^{-1}}^2$，用双层学习得到 PSD 矩阵 $\Sigma$，由 $\Sigma^{-1}$ 的非对角元建模残余条件依赖、对角元建模 [[heterogeneous-task-weights|异构任务权重]][^src-qdf]。技术细节见 [[quadratic-form-weighted-objective]]。

与两者相比，本方法的差别在于不做标签变换、也不做似然因式分解，而是用联合分布差异的上界直接约束条件分布；代价是需要 $(H+T)\times(H+T)$ 的协方差与矩阵平方根[^src-distdf]。

## 相关页面

- [[distdf]] — 完整框架、算法步骤与实验表
- [[bures-wasserstein]] — 高斯闭式解与矩对齐的含义
- [[autocorrelation-bias]] — 被替代的偏差来源
- [[label-autocorrelation]] — 偏差来源的统计现象
- [[direct-forecast]] — 所在的预测范式
- [[optimal-transport]] — Kantorovich 形式与求解方法
- [[sliced-wasserstein-distance]] — 另一种绕开高维耦合的 Wasserstein 近似

---

[^src-distdf]: [[source-distdf]]
[^src-fredf]: [[source-fredf]]
[^src-qdf]: [[source-qdf]]
