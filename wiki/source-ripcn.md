---
title: "RIPCN: A Road Impedance Principal Component Network for Probabilistic Traffic Flow Forecasting"
type: source-summary
tags:
  - probabilistic-forecasting
  - traffic-flow
  - uncertainty-estimation
  - pca
  - domain-knowledge
created: 2026-09-16
last_updated: 2026-09-16
source_count: 1
confidence: high
status: active
---

# RIPCN: A Road Impedance Principal Component Network for Probabilistic Traffic Flow Forecasting

Lv Haochen、Lin Yan（共同一作）、Guo Shengnan（通讯）等，北京交通大学 + Aalborg University。KDD 2026（PDF 首页印有接收声明）[^src-ripcn]。代码：https://github.com/LvHaochenBANG/RIPCN.git[^src-ripcn]。

## 问题

概率交通流预测（PTFF）要求同时输出点预测与不确定性估计。论文指出现有方法的两个缺口：(1) 未建模不确定性的成因——交通流涨落来自道路本身的拥堵与流量变异性，现有三类范式（近似贝叶斯、扩散/流生成式、分布参数化）都不触及这一层面；(2) 逐节点独立估计不确定性，忽略不确定性的时空相关结构[^src-ripcn]。

## 方法

两段式。**阻抗演化模块**：以 BPR 阻抗函数为基础，加入流量变异系数因子 $\sigma_a/\mu_a$ 刻画波动对行程时间的影响；给出仅用历史流量（+速度/占有率）的道路容量简化估计；用 temporal attention 从历史阻抗外推未来阻抗，并以未来真实流量算出的阻抗做 MSE 监督（$\mathcal{L}_R$）；按邻接关系取相邻路段阻抗差生成动态阻抗图，方向为"阻抗高处流出、低处流入"[^src-ripcn]。**主成分网络**：对去均值流量的时空协方差张量做低秩近似 $\Sigma \approx \sum_{k=1}^K \lambda_k \boldsymbol{w}_k \otimes \boldsymbol{w}_k$，用带阻抗图 GCN 的 ST-Graph 块直接预测未来流量的前 $K$ 个时空主成分（Schmidt 正交化），配合方向约束损失 $\mathcal{L}_D$ 与方差幅值损失 $\mathcal{L}_V$；推理时沿主成分构造样本 $\hat{X}^P + t_k \sigma_k \boldsymbol{w}_k$ 取均值与方差[^src-ripcn]。

## 实验与结果

PEMS03/04/08 + Seattle，6:2:2 时间切分，12 步预测 12 步，对比 9 个概率基线（DeepAR、Latent ODE、MC Dropout、STGNF、PriSTI、CSDI、DiffSTG、DER、SDER），指标 MAE/RMSE/MAPE + CRPS/MIS：论文报告四个数据集全指标最优，如 PEMS08 MAE 15.14（DER 16.66）、CRPS 0.0565；PEMS03 MAE 15.28（DER 15.66）；Seattle MAE 94.97（SDER 101.73）[^src-ripcn]。消融（Seattle）：去 ST-Graph 块降幅最大（MAE 94.97→104.94），去方向损失 $\mathcal{L}_D$ 使 CRPS 从 0.0955 恶化到 0.1247、MIS 几乎翻倍；效率上推理只需一次前向构造样本，对比扩散基线 GPU 内存与推理时间最低（PEMS08 推理 16.77s，扩散类 CSDI/PriSTI/DiffSTG 数百秒）[^src-ripcn]。

## 局限与边界

实验为同构交通流传感器图（PEMS/Seattle），未见跨城市或异构网络验证；容量估计 Eq. 5 依赖流量峰值时点的速度/占有率观测，数据稀缺场景退化为仅用 $X^{max}$[^src-ripcn]。

## 第 4 章全文中文翻译（模型）

> [!note] 本节为 `raw/ripcn.pdf` 第 4 章（原 PDF 第 3–6 页）的全文中文翻译，非摘要。

### 4.1 概览
在本研究中，我们提出 RIPCN，一种用于概率交通流预测的新架构，如图 2 所示。RIPCN 包含两个网络：阻抗演化网络（impedance evolution network）与主成分网络（principal component network）。阻抗演化网络通过一个时间注意力模块（temporal attention block）将历史道路阻抗 $\boldsymbol{R}^H$ 演化为未来道路阻抗 $\boldsymbol{R}^P$，随后生成动态阻抗图 $\boldsymbol{A}_r$。该图随后被输入主成分网络，以引导模型理解交通转移模式。主成分网络通过 $L$ 个时空图模块（ST-Graph Block）处理输入流量来预测时空主成分（PCs），这些模块将 TCN 与 GCN 同动态阻抗图结合，之后进行正交化。RIPCN 由阻抗 MSE 损失与 PCA 损失训练。在推理阶段，通过将主成分与 $\hat{\boldsymbol{X}}^P$ 组合，生成一系列样本用于概率预测。

**图 2**：RIPCN 的整体架构。阻抗演化网络以历史流量为输入演化道路阻抗，生成动态阻抗图。主成分网络利用该图预测主成分。

后续各节分别介绍阻抗演化模块与主成分模块，每个模块都给出其理论框架与相应的网络实现。

## 4.2 阻抗演化模块

为揭示并建模交通流不确定性的成因，我们引入道路阻抗理论中的领域知识。该理论通过刻画交通如何跨路段传播，来描述流量波动，而流量波动正是不确定性的直接来源。这样的引导使模型能够学习由道路拥堵与流量变异性驱动的转移模式，从而改进不确定性估计并增强可解释性。

### 4.2.1 含交通流不确定性的道路阻抗

BPR 函数（式 2）基于道路通行能力与自由流旅行时间估计道路阻抗。然而，它没有考虑除流量的绝对水平之外，交通流波动的强度也会显著影响实际旅行时间。实践中，交通流变异性更高的路段往往面临更大的旅行时间不确定性。为此，我们在阻抗函数中引入一个基于流量的不确定性因子：

$$T_a(X) = t_a \times \left[1 + \alpha\left(\frac{X}{C_a}\right)^{\beta}\right] \times \left(1 + \frac{\sigma_a}{\mu_a}\right), \qquad (4)$$

其中 $T_a(X)$ 表示路段 $a$ 上的旅行时间。$\sigma_a/\mu_a$ 是历史 $\tau$ 个时间步上交通流的变异系数（coefficient of variation），刻画相对波动。特征 $t_a$、$C_a$、$\mu_a$ 与 $\sigma_a$ 被视为与各路段相关的静态或缓变属性，对应式 1 中定义的 $\boldsymbol{P} \in \mathbb{R}^{N \times 4}$。

重要的是，获得道路通行能力 $C_a$ 具有挑战性，因为理想情况下应来自道路建设手册的详细信息常常无法获得。诸如 [5, 24] 的交通研究通常基于车头时距（headway）、交通计数等因素，用复杂模型估计 $C_a$。为使我们的方法在数据稀缺场景下仍然适用，我们提出一种简化的道路通行能力估计：

$$C_a = X^{\max} \times \left[1 + \frac{1}{2}\left(\frac{s_a}{s^{\text{mean}}} + \frac{o^{\text{mean}}}{o_a}\right)\right], \qquad (5)$$

其中 $X^{\max}$ 是路段 $a$ 上观测到的最大交通流，$s_a$ 与 $o_a$ 表示记录 $X^{\max}$ 时刻的车辆速度与占有率（occupancy），$s^{\text{mean}}$ 与 $o^{\text{mean}}$ 表示所有路段的平均速度与平均占有率。

这一表述意味着，观测到的最大流量可以作为通行能力的基线估计。若相应速度 $s_a$ 高于平均值，或占有率 $o_a$ 低于平均值，则意味着该路段可能支持高于 $X^{\max}$ 的通行能力。反之，速度更低或占有率更高说明道路已接近其通行能力上限。

此外，在仅有历史交通流数据的情况下，仅 $X^{\max}$ 本身即可作为 $C_a$ 的代理。我们通过实验验证了这些近似策略的有效性。

### 4.2.2 阻抗演化网络

尽管阻抗函数式 4 纳入了变异因子，它在很大程度上仍是静态的。因此，为引导对未来交通动态的预测，我们提出阻抗演化网络，利用历史观测建模道路阻抗的时间演化。

具体地，首先用式 4 计算历史道路阻抗，随后将其送入全连接编码器，把原始序列变换为适合建模其时间演化的紧凑表示。

$$\boldsymbol{R}^H = \text{ImpedanceFunc}(\boldsymbol{X}^H, \boldsymbol{P}), \quad \boldsymbol{R}_h^H = \text{Encoder}(\boldsymbol{R}^H), \qquad (6)$$

其中 $\boldsymbol{X}^H \in \mathbb{R}^{\tau \times N \times 1}$ 表示历史交通流数据，$\boldsymbol{R}^H \in \mathbb{R}^{\tau \times N \times 1}$ 表示相应计算出的道路阻抗，$\boldsymbol{R}_h^H \in \mathbb{R}^{\tau \times N \times F}$ 是编码后的表示。

接下来，我们采用时间注意力 [32]，基于历史值推断未来道路阻抗。跨时间步的注意力权重计算如下：

$$\alpha_{i,j} = \frac{\exp\left(\left(\boldsymbol{W}^Q\boldsymbol{R}_i^H\right)^{\top}\boldsymbol{W}^K\boldsymbol{R}_j^H\right)}{\sum_{t=1}^{T}\exp\left(\left(\boldsymbol{W}^Q\boldsymbol{R}_i^H\right)^{\top}\boldsymbol{W}^K\boldsymbol{R}_j^H\right)}, \quad \boldsymbol{R}_i^P = \sum_{j=1}^{T}\alpha_{i,j}\left(\boldsymbol{W}^V\boldsymbol{R}_i^H\right), \qquad (7)$$

其中 $\boldsymbol{W}^Q, \boldsymbol{W}^K, \boldsymbol{W}^V \in \mathbb{R}^{F \times F}$ 是可学习参数，$\boldsymbol{R}_i^P$ 表示第 $i$ 个未来时间步的阻抗。所有时间步的道路阻抗可表示为 $\boldsymbol{R} = [\boldsymbol{R}^H, \boldsymbol{R}^P]$。

按照道路阻抗理论，阻抗越高表示拥堵越严重，车辆更难进入路段，而更容易驶出。因此，路段之间的阻抗差刻画了交通流的方向性倾向，并成为其重新分布的驱动力 [33, 34]。这一性质自然地为道路图的构建提供了依据，使交通流转移模式能够被有效捕捉。因此，基于实际网络连通性，我们在每个时间步计算各路段与其邻居之间的阻抗差，以生成反映方向性流量倾向随时间演化的动态阻抗图：

$$\hat{\boldsymbol{A}}_r^{a,b} = -\left[\text{FC}(\boldsymbol{R}_a) - \text{FC}(\boldsymbol{R}_b)\right] \times \mathbb{I}_{\boldsymbol{A}^{a,b}}, \qquad (8)$$

其中 $\text{FC}(\cdot)$ 是一个全连接网络，将阻抗表示 $\boldsymbol{R}_a \in \mathbb{R}^{(\tau+T)\times F}$ 与 $\boldsymbol{R}_b \in \mathbb{R}^{(\tau+T)\times F}$ 解码为阻抗值，$\boldsymbol{A}^{a,b}$ 表示路段 $a$ 与 $b$ 之间的连通性，$\mathbb{I}_{\boldsymbol{A}^{a,b}}$ 是一个指示量：若 $\boldsymbol{A}^{a,b}$ 为正则返回 1，否则返回 0。

训练时，我们使用均方误差（MSE）损失，促使预测阻抗 $\hat{\boldsymbol{R}}^P$ 逼近由实际交通流导出的真实阻抗 $\boldsymbol{R}^P$：

$$\mathcal{L}_R = \left\|\boldsymbol{R}^P - \hat{\boldsymbol{R}}^P\right\|_2^2 \qquad (9)$$

总结起来，阻抗演化网络学习生成动态阻抗图 $\hat{\boldsymbol{A}}_r \in \mathbb{R}^{(\tau+T)\times N \times N}$，该图随后被融入主成分网络，以指导时空交通转移模式的建模。

## 4.3 主成分模块

如图 1 所示，忽略时空相关性、独立建模不确定性，往往导致估计过于发散。为此，我们采用基于 PCA 的方法，预测未来不确定性的主成分，从而给出协方差结构的紧凑表示。这种低秩近似捕捉主要的时空变化，降低直接估计协方差的代价与噪声，并同时改进不确定性量化与点预测精度。

### 4.3.1 时空主成分预测

为捕捉交通流不确定性的时空相关性，我们引入协方差。协方差是一种基本统计量，量化两个随机变量之间的联合变异性，刻画它们共同变化的程度。

对于目标交通流 $\boldsymbol{X}^P \in \mathbb{R}^{T \times N}$，我们定义中心化数据为 $\boldsymbol{X} = \boldsymbol{X}^P - \hat{\boldsymbol{X}}^P$，其中 $\hat{\boldsymbol{X}}^P$ 是网络预测的均值。中心化过程去除偏置，并突出底层波动。协方差结构随后由二阶张量 $\boldsymbol{\Sigma} = \boldsymbol{X} \otimes \boldsymbol{X} \in \mathbb{R}^{T \times N \times T \times N}$ 表示，其中 $\otimes$ 表示在时间与空间两个维度上的外积。该表述捕捉交通流波动中的时空依赖。

然而，（1）在交通流预测中，潜在相关性高度动态，使由历史数据导出的协方差结构难以很好地泛化到未来场景；（2）此外，由于交通数据维度高，精确估计完整的时空协方差张量既困难又计算昂贵。

如图 3 所示，交通流的协方差随时间变化，并随预测时域 $T$ 与路段数 $N$ 二次增长。所幸我们观察到，经特征值分解后，前若干个特征向量捕捉了协方差信息的大部分。

**图 3**：左图展示 PEMS08 数据集中相邻路段 7、13、41 在 7:00 至 16:00 的协方差结构。右图展示相应的特征值谱，说明大部分方差由少数几个主成分捕捉。

基于上述观察，我们提出预测协方差张量的时空特征向量，称为时空主成分（PCs）；利用它们可以高效近似底层协方差结构，同时有效抑制噪声及其他无关变化：

$$\boldsymbol{\Sigma} \approx \sum_{k=1}^{K}\lambda_k \boldsymbol{w}_k \otimes \boldsymbol{w}_k, \qquad (10)$$

其中 $\boldsymbol{w}_k \in \mathbb{R}^{T \times N}$ 表示第 $k$ 个主成分，$\lambda_k$ 是相应特征值，$K$ 是保留的成分个数。

与传统 PCA 对被观测数据降维不同，由于交通数据的动态特性，我们的目标是预测未来交通流的主成分。因此，我们用一个主成分网络对这些时空主成分进行参数化，详见 4.3.2 节。网络第 $k$ 个输出记为 $\boldsymbol{d}_k(\boldsymbol{X}^H, \hat{\boldsymbol{X}}^P) \in \mathbb{R}^{T \times N}$。为确保预测成分满足主成分的正交性与单位范数性质，我们采用 Schmidt 正交化过程。所得正交输出记为 $\boldsymbol{w}_k$，由下式给出：

$$\tilde{\boldsymbol{w}}_k = \boldsymbol{d}_k - \sum_{i=1}^{k-1}\langle \boldsymbol{d}_k, \boldsymbol{w}_i \rangle_F \cdot \boldsymbol{w}_i, \quad \boldsymbol{w}_k = \frac{\tilde{\boldsymbol{w}}_k}{\left\|\tilde{\boldsymbol{w}}_k\right\|_F}, \qquad (11)$$

其中 $\langle \cdot, \cdot \rangle_F$ 表示 Frobenius 内积，$\|\cdot\|_F$ 是相应的 Frobenius 范数。

基于式 3 所示的主成分性质，我们引入方向约束损失（Directional Constraint Loss），促使 $\boldsymbol{w}_k$ 对齐交通流中变化最大的方向：

$$\mathcal{L}_D = -\sum_{k=1}^{K}\langle \boldsymbol{w}_k, \boldsymbol{X} \rangle_F^2, \qquad (12)$$

其中 $\langle \boldsymbol{w}_k, \boldsymbol{X} \rangle_F^2$ 表示 $X$ 在 $\boldsymbol{w}_k$ 方向上的方差。在这一表述的基础上，我们进一步揭示其与协方差结构的联系，从而为该设计的有效性给出理论依据。为施加单位范数约束 $\|\boldsymbol{w}_k\|_F^2 = 1$，我们引入 Lagrange 乘子 $\theta_k$，并构造如下 Lagrangian：

$$\mathcal{L}(\boldsymbol{w}_k, \theta_k) = -\langle \boldsymbol{w}_k, \boldsymbol{X} \rangle_F^2 + \theta_k\left(\|\boldsymbol{w}_k\|_F^2 - 1\right) \qquad (13)$$

对 $\mathcal{L}$ 关于 $\boldsymbol{w}_k$ 求导，我们得到：

$$\frac{\partial \mathcal{L}}{\partial \boldsymbol{w}_k} = -2\langle \boldsymbol{w}_k, \boldsymbol{X} \rangle_F \cdot \boldsymbol{X} + 2\theta_k \boldsymbol{w}_k \qquad (14)$$

令梯度为零，得到下式：

$$\langle \boldsymbol{w}_k, \boldsymbol{X} \rangle_F \cdot \boldsymbol{X} = (\boldsymbol{X} \otimes \boldsymbol{X}) \cdot \boldsymbol{w}_k = \boldsymbol{\Sigma} \cdot \boldsymbol{w}_k = \theta_k \boldsymbol{w}_k \qquad (15)$$

这确认了理论上最优解 $\boldsymbol{w}_k$ 是协方差结构 $\boldsymbol{\Sigma}$ 的时空特征向量，满足式 10。

主成分给出交通流变化的方向，我们还需要量化这些方向上的波动水平，该水平体现在各主成分上数据的方差。为此，我们采用方差量级损失（Variance Magnitude Loss），通过均方误差约束正交化输出 $\tilde{\boldsymbol{w}}_k$ 的范数，使其与方差 $\langle \boldsymbol{w}_k, \boldsymbol{X} \rangle_F^2$ 相匹配：

$$\mathcal{L}_V = \sum_{k=1}^{K}\left(\langle \boldsymbol{w}_k, \boldsymbol{X} \rangle_F^2 - \left\|\tilde{\boldsymbol{w}}_k\right\|_F^2\right)^2 \qquad (16)$$

于是，测试时预测的方差 $\sigma_k^2$ 由 $\|\tilde{\boldsymbol{w}}_k\|_F^2$ 给出。注意这反映的是沿主成分方向的方差，而非数据的整体方差。

由于主成分捕捉了数据中联合时空变化的主导方向，它们可用于建模预测不确定性。为此，我们通过沿各主方向扩展预测均值 $\hat{\boldsymbol{X}}^P$ 来构造一组样本：

$$\mathcal{S} = \left\{\hat{\boldsymbol{X}}^P + t_k\sigma_k\boldsymbol{w}_k \mid k = 1, \ldots, K\right\}, \qquad (17)$$

其中 $\mathcal{S}$ 表示样本集，$t_k$ 是一个标量系数，用于调节沿 $\boldsymbol{w}_k$ 的幅度与方向。这考虑到每个 $\boldsymbol{w}_k$ 既可能与真实变化方向一致，也可能与之相反。我们在验证集上通过二分搜索确定 $t_k$ 的取值。我们计算 $\mathcal{S}$ 中样本的均值与方差，得到最终的点预测 $\boldsymbol{\mu}$ 与相关的不确定性估计 $\boldsymbol{\sigma}$。

值得注意的是，由 $\mathcal{S}$ 导出的均值往往比 $\hat{\boldsymbol{X}}^P$ 更准确。这一改进归因于时空主成分的良好性质与我们采样机制。我们在 5.6 节与附录 A 中为该观察提供理论依据与经验证据。

### 4.3.2 主成分网络

上一节我们关注预测时空主成分所需的损失约束。除损失设计之外，合适的网络架构同样关键。为此，我们构建了一个面向交通流数据的专用主成分网络。

在该网络中，我们首先部署一个预训练均值预测器，基于历史数据预测未来交通流。预测均值随后与历史交通流数据拼接，并输入一个编码器；该编码器由全连接层实现，用于抽取隐藏表示：

$$\hat{\boldsymbol{X}}^P = \text{MeanPredictor}(\boldsymbol{X}^H, G), \quad \boldsymbol{h}_C = \text{Encoder}(\boldsymbol{X}^H \| \hat{\boldsymbol{X}}^P), \qquad (18)$$

其中 $\hat{\boldsymbol{X}}^P \in \mathbb{R}^{T \times N}$ 表示预测未来交通数据的均值，$\boldsymbol{h}_C \in \mathbb{R}^{T \times N \times F}$ 是拼接数据的隐藏表示。随后实现 $L$ 个时空图模块（ST Graph Block），在道路阻抗的引导下学习交通转移模式。每个模块利用时间卷积网络（TCN）[4] 建模时间依赖。TCN 实现为一维因果卷积。该操作可表示为：

$$\boldsymbol{h}_T = \text{TCN}(\boldsymbol{W}^T, \boldsymbol{h}_C), \qquad (19)$$

其中 $\boldsymbol{W}^T \in \mathbb{R}^{k \times F \times F'}$ 表示核大小为 $k$ 的卷积核，$\boldsymbol{h}_T \in \mathbb{R}^{T \times N \times F'}$ 是时间特征表示。

空间依赖由图卷积网络（GCN）[43] 基于动态阻抗图 $\boldsymbol{A}_r$ 捕捉：

$$\boldsymbol{h}_G = \sigma\left((\boldsymbol{A} \| \boldsymbol{A}_r)\boldsymbol{h}_T\boldsymbol{W}^G\right), \qquad (20)$$

其中 $\boldsymbol{A}$ 是路网的归一化邻接矩阵，$\boldsymbol{W}^G \in \mathbb{R}^{F' \times F'}$ 是可学习参数，$\boldsymbol{h}_G \in \mathbb{R}^{T \times N \times F'}$。除 ST 图模块之间的直接连接之外，还采用了跳跃连接。主成分网络中的解码器由 TCN 与全连接层实现，输出 $\boldsymbol{d} \in \mathbb{R}^{K \times T \times N}$，其中 $K$ 是待预测主成分的个数。最后，经 Schmidt 正交化过程得到主成分 $\boldsymbol{w} \in \mathbb{R}^{K \times T \times N}$：

$$\boldsymbol{d} = \text{Decoder}(\boldsymbol{h}_G), \quad \boldsymbol{w} = \text{Orthogonalization}(\boldsymbol{d}) \qquad (21)$$

## 4.4 模型训练

为训练 RIPCN 网络，我们将总体损失函数定义为：

$$\mathcal{L} = \lambda_1\mathcal{L}_R + \lambda_2(\mathcal{L}_D + \mathcal{L}_V), \qquad (22)$$

其中 $\mathcal{L}_R$ 监督动态阻抗，而 $\mathcal{L}_D$ 与 $\mathcal{L}_V$ 共同构成 PCA 损失，引导模型学习主成分以及沿这些成分的方差。系数 $\lambda_1$ 与 $\lambda_2$ 是超参数。训练过程中，我们先固定 $\lambda_1 = 1$，以优先学习道路阻抗。随后，$\lambda_2$ 以线性调度从 0 逐步增至 1，促使模型在阻抗学习的引导下捕捉主成分及其方差。

## 引用
[^src-ripcn]: [[source-ripcn]]
