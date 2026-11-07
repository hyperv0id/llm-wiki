---
title: "STONE"
type: entity
tags:
  - spatio-temporal
  - out-of-distribution
  - traffic-forecasting
  - invariant-learning
  - graph-intervention
  - frechet-embedding
created: 2026-11-07
last_updated: 2026-11-07
source_count: 2
confidence: medium
status: active
---

# STONE

**STONE**（Spatio-temporal OOD learning framework）是 Binwu Wang、Jiaming Ma、Pengkun Wang、Xu Wang、Yudong Zhang、Zhengyang Zhou、Yang Wang（中国科学技术大学 + 中科大苏州高等研究院）在 KDD 2024 提出的时空 OOD 框架（pp. 2948–2959，DOI 10.1145/3637528.3671680）[^src-stone]。它同时处理两类漂移：**时间漂移** $P(X_{te})\neq P(X_{tr})$ 与**空间漂移** $G_{te}\neq G_{tr}$（新传感器出现、旧传感器消失），目标是学到跨环境不变的节点依赖[^src-stone]。代码：github.com/PoorOtterBob/STONE-KDD-2024。

## 问题定位

论文把现有工作归为两条路线[^src-stone]：

1. **不处理 OOD 的 STGNN**（STGCN、GWNet、STNN、D2STGNN、PDFormer）：在预定义训练图上做聚合，知识绑定于该图，节点增删后原聚合路径失效，也无法为训练期未见节点生成表示——论文称 *inflexible scalability* 与 *unreliable generalization*。
2. **只处理时间漂移**（CauSTG、CaST、STEVE 等）：忽略图结构演化。

STONE 的判断是：两类漂移的共同本质是「节点间依赖关系随环境改变」，因此不必在原始时空数据上做增强，而是直接扰动已学到的节点依赖（语义图）来构造多个环境，再抽取跨环境不变的依赖作为聚合路径；空间侧再用对结构变化不敏感的 Fréchet 嵌入给新节点赋初值[^src-stone]。

## 数据流与维度

形状记作 $[\text{行}, \text{列}]$。**维度符号**：$N$ 节点数；$T$ 输入窗口（代码默认 12）；$T_p$ 预测步（12）；$d$ 输入特征维（代码 3）；$d_e=\lceil\log N\rceil R$ Fréchet 嵌入维（$R$ 为采样轮数，SD 10 / GBA 30）；$d_s$ 空间分支隐维；$C_t$ 时间分支通道；$d_o$ 扩散卷积输出维；$Z$ 扩散步数；$K_M$ 干预环境数（2）。

### Stage 1 · 并行构造两条语义图

**支路 A：空间侧（Fréchet 嵌入 → 门控 Transformer → 空间语义图）**

锚点集按「第 $j$ 个锚点集中任意节点以 $2^{-j}$ 概率入选」采样，每轮 $\lceil\log N\rceil$ 个锚点集、共 $R$ 轮：

$$\begin{aligned}
f_i(u) &= \alpha_i\cdot\min_{v\in V_i}d_V(u,v) && \text{// } [N]\xrightarrow{\text{每个节点到第 }i\text{ 个锚点集的 K 阶 Manhattan 距离}} [N, d_e]\ \text{——Bourgain 定理保证嵌入失真 } O(\log N)\\
S_{in} &= \mathrm{Noise}_\gamma\!\left(\left[f_1(u);\dots;f_{d_e}(u)\right]\right) && \text{// } [N, d_e]\xrightarrow{+\ \gamma\sim\mathcal{N}(0,1)} [N, d_e]\ \text{——加噪扰动节点空间依赖，模拟 spatial shift}\\
D_s &= \mathrm{SA}\!\left(\mathrm{GatedTransformer}_2(S_{in})\right) && \text{// } [N, d_e]\to[N, d_s]\xrightarrow{\text{节点两两打分}} [N, N]\ \text{——空间语义图：节点间空间依赖强度}
\end{aligned}$$

**支路 B：时间侧（时间位置嵌入 → 门控空洞因果 TCN → 时间语义图）**

$$\begin{aligned}
X_T^{(l+1)} &= \sigma_T^l\odot\theta_{k_t}*_t X_T^{(l)} && \text{// } [T_l, N, d_t]\xrightarrow{\text{5 层不同感受野的门控因果卷积}} [T_{l+1}, N, d_t]\ \text{——逐节点提取长短趋势，Sigmoid 门逐通道选择}\\
X_t &= \mathrm{ReLU}\!\left(X_T^{(0:L_t)}\times_2 W_{t1}\times_2 W_{t2}\right) && \text{// } [T, N, d]\xrightarrow{\text{拼接各层 + 时间维压到 1}} [N, 1, C_t]\\
D_t &= \mathrm{SA}(X_t) && \text{// } [N, C_t]\xrightarrow{\text{节点两两打分}} [N, N]\ \text{——时间语义图：节点流量分布的相关性}
\end{aligned}$$

### Stage 2 · 图干预：按可学习分布删除节点，构造 $K_M$ 个环境

两个掩码生成器（空间/时间各一）各维护一份节点采样分布 $B_s,B_t\in\mathbb{R}^{K_M\times N}$，每步采样 $1\%$ 的节点、把这些节点在自适应邻接中的整列置零；第 0 个环境保留原图，其余环境是编辑副本：

$$\begin{aligned}
p_s &= \mathrm{softmax}(B_s) && \text{// } B_s\in\mathbb{R}^{K_M\times N}\ \text{可学习——每个环境一行节点采样分布}\\
S_k &\sim \mathrm{Multinomial}\!\left(p_s[k],\ \lfloor 0.01N\rfloor\right) && \text{// } [K_M, N]\xrightarrow{\text{每行抽 }1\%\text{ 节点}} \text{被删节点下标 } S_k\\
M_s[k,j] &= \begin{cases}1, & k=0\\ \mathbb{1}[j\notin S_k], & k\ge 1\end{cases} && \text{// } [K_M, N]\to\{0,1\}^{K_M\times N}\ \text{——第 0 行恒为 1（原图），其余行删去抽样节点}\\
\hat D_s^{(k)} &= M_s[k]\odot D_s && \text{// 节点掩码按列缩放邻接，得到第 }k\text{ 个环境的空间依赖}
\end{aligned}$$

时间侧同构：$\hat D_t^{(k)}=M_t[k]\odot D_t$。$K_M=2$ 即「原图 + 一个编辑副本」。掩码矩阵在初始化时建一次，训练中只把抽样节点置零、不重置，编辑环境的删点集合随迭代累积。

### Stage 3 · 交叉扩散图卷积

$$\begin{aligned}
X_o^{(k)} &= \sum_{i=1}^{Z}\theta_s *_s \left(I+D_s^{(k)}\right)^{i} X_t && \text{// } [N,N]^{i}\times[N,C_t]\to[N,d_o]\ \text{——时间特征沿空间语义图扩散聚合}\\
S_o^{(k)} &= \sum_{i=1}^{Z}\theta_t *_s \left(I+D_t^{(k)}\right)^{i} S_s && \text{// } [N,N]^{i}\times[N,d_s]\to[N,d_o]\ \text{——空间特征沿时间语义图扩散聚合（交叉使用）}
\end{aligned}$$

两支路各自带着对方的语义图做聚合：时间特征走空间图、空间特征走时间图，再汇合到解码器。

### Stage 4 · 门控解码与交替优化的对抗目标

$$\begin{aligned}
\sigma_{out} &= \mathrm{Sigmoid}\!\left(S_oW^{s}_{out}+X_oW^{i}_{out}\right) && \text{// } [N,d_o]\times[d_o,d_{out}]\to[K_M,N,d_{out}]\ \text{——每个环境的门控}\\
\hat Y &= \sigma_{out}\odot\left(X_oW^{o}_{out}+b_{out}\right) && \text{// } [K_M,N,d_{out}]\odot[N,d_o]\to[K_M,N,T_p]\ \text{——门控调制后的预测}
\end{aligned}$$

训练时两套参数交替更新，各自一个 Adam 优化器[^src-stone]：

$$\begin{aligned}
\bar L &= \frac{n_{ob}\bar L_{ob}+n_{un}\bar L_{un}}{n_{ob}+n_{un}} && \text{// 观测/未观测两组节点按样本数加权的 L1}\\
\Theta^* &= \arg\min_{\Theta}\ \bar L + \mathrm{Var}\!\left(\left|\hat Y-y\right|\right) && \text{// 主网络：加权 L1 + 逐元素绝对误差的方差}\\
\{B_s,B_t\}^* &= \arg\max_{B_s,B_t}\ \log p\!\left(B_s,B_t\right)\cdot \operatorname{sg}\!\left[\mathrm{Var}\!\left(\left|\hat Y-y\right|\right)\right] && \text{// 掩码生成器：让采样分布偏向误差方差大的节点（$\operatorname{sg}$ 停梯度）}
\end{aligned}$$

主网络压低误差并压缩误差离散度，掩码生成器同时把干预推向模型最不稳的节点集合——这是 explore-to-extrapolate 的对抗实现，论文用 IRM 语言把 $\mathrm{Var}$ 记为跨环境方差。推理期不加掩码、不加噪声，单次前向输出 $[N,T_p]$。

## 实验协议

- **数据**：LargeST 的 California 子集，PeMS 流量。SD（San Diego）716 传感器、训练用 550 节点；GBA（Greater Bay Area）2352 传感器、训练用 1809 节点[^src-stone]。
- **划分**：训练 2019-01→08（21,010 时间戳），验证 2019-09→10（7,003），测试 **2020-11→12**（7,022）——中间 14 个月被丢弃，构成时间漂移；测试图在训练节点基础上加入 10%/15%/20% 新节点（训练期从未出现）、并随机遮蔽 10% 节点模拟设备消失，构成空间漂移。同城同路网，不是跨城市迁移[^src-stone]。
- **基线**：HL、GWNet、STGCN、STNN、CaST、CauSTG；D²STGNN 与 PDFormer 因「核心组件参数与图规模耦合」被排除[^src-stone]。
- **指标**：MAE / RMSE / MAPE，3/6/12 步。训练一次直接测试，不做微调。

## 结果

**全节点泛化（Table 2）与仅新节点可扩展性（Table 3），MAE**：

| 数据 | 新节点比例 | 步长 | 全节点 STONE / GWNet | 仅新节点 STONE / GWNet |
|---|---|---|---|---|
| SD | 10% | 3 | 18.16 / 18.86 | 17.22 / 19.58 |
| SD | 15% | 3 | 19.36 / 21.58 | 18.92 / 19.58 |
| SD | 20% | 3 | 19.32 / 19.12 | 18.51 / 19.58 |
| SD | 10% | 6 | 26.32 / 27.15 | 24.91 / 28.48 |
| SD | 15% | 6 | 26.23 / 27.01 | 26.31 / 28.33 |
| SD | 20% | 6 | 26.24 / 27.62 | 25.71 / 28.55 |
| SD | 10% | 12 | 41.05 / 39.31 | 39.53 / 41.98 |
| SD | 15% | 12 | 36.17 / 39.14 | 37.78 / 41.40 |
| SD | 20% | 12 | 36.19 / 40.32 | 36.74 / 42.12 |
| GBA | 10% | 3 | 17.67 / 19.23 | 17.64 / 21.64 |
| GBA | 15% | 3 | 18.73 / 17.56 | 18.25 / 18.00 |
| GBA | 20% | 3 | 18.83 / 18.53 | 19.59 / 21.39 |
| GBA | 10% | 6 | 25.35 / 28.10 | 25.73 / 33.01 |
| GBA | 15% | 6 | 25.16 / 25.24 | 24.87 / 25.89 |
| GBA | 20% | 6 | 25.37 / 26.71 | 25.46 / 33.46 |
| GBA | 10% | 12 | 36.35 / 39.91 | 38.24 / 49.10 |
| GBA | 15% | 12 | 36.04 / 38.94 | 36.50 / 39.92 |
| GBA | 20% | 12 | 36.51 / 39.24 | 37.60 / 53.09 |

**边界**：GWNet 是每一格的最强基线；STONE 在全节点表 36 格中 5 格 MAE 落后（SD 20%/3 步、SD 10%/12 步、GBA 15%/3 步、GBA 20%/3 步，以及 SD 15%/3 步的 RMSE）。优势集中在 6/12 步与仅新节点场景，GBA 20% 新节点 12 步从 53.09 降到 37.60（−29.2%）；3 步小比例场景多处打平或落后[^src-stone]。

**只保留时间漂移的对照（Table 5，SD）**：MAE 18.26 vs 18.38（3 步）、25.06 vs 25.84（6 步）、34.46 vs 34.62（12 步）——去掉空间漂移后 STONE 与 GWNet 基本持平，增益几乎全部来自节点增删这一维度[^src-stone]。论文用这一对照论证「空间漂移才是难点」。

**消融（Table 4，SD-10%、3 步，MAE/RMSE/MAPE）**：

| 变体 | 全节点 | 仅新节点 |
|---|---|---|
| 完整模型 | 18.17 / 29.67 / 15.51 | 17.23 / 26.85 / 14.93 |
| 去 Fréchet 嵌入 | 21.91 / 32.24 / 26.79 | 21.64 / 31.84 / 25.30 |
| 去干预损失 | 19.66 / 31.29 / 17.83 | 20.49 / 32.23 / 20.49 |
| 去输入噪声 | 19.36 / 31.43 / 16.69 | 18.92 / 30.43 / 18.67 |

Fréchet 嵌入贡献最大（MAPE 退化 72.7%），干预损失次之（15.0%），输入噪声最小（7.6%）。消融只覆盖一个数据集、一个步长、一个节点比例[^src-stone]。

## 与相邻方法的差异

- **[[cast|CaST]] / CauSTG**：以结构因果模型 + 后门调整从**时间**环境中分离不变成分；STONE 不变量的载体是**节点依赖图**，环境由语义图干预生成，并把结构漂移（新节点）纳入范围[^src-stone]。
- **STEVE**：两个解耦表示 + 时空环境自监督，同样以时间漂移为主。
- **[[stop|STOP]]**（ICML 2025，同组后续）：改在 message-passing 层面切开 node-to-node 连接，主张耦合本身是脆弱源；STONE 则保留节点间聚合，只让聚合路径跨环境不变[^src-stone]。

## 边界与外部核查

- 仅在交通数据上验证；作者自述掩码算子增多时训练代价指数增长、语义图学习组件（gated transformer）可用更强的图结构学习替代[^src-stone]。
- 有效边界集中在空间结构演化：纯时间跨年漂移（Table 5）下与 GWNet 持平，增益主要来自新增节点（最多 −29.2%）[^src-stone]。这与 [[st-ood|ST-OOD]] 基准「专用因果/不变方法在自然跨年 T-OOD 上难以击败常规强基线」的观察方向一致；同批表里欠拟合显著的是 CaST/CauSTG，12 步 MAE 达 60 以上，而 GWNet/STONE 多在 34–41[^src-st-ood][^src-stone]。

## Related Pages

- [[source-stone]] — source summary
- [[spatio-temporal-ood-learning]] — ST-OOD 问题与解法谱系
- [[stop]] — 同组后续，message-passing 路线的 OOD 方法
- [[cast]] — CaST，因果/不变学习路线的代表
- [[ood-generalization]] — OOD 泛化总概念
- [[st-ood]] — ST-OOD 跨年评测基准与 underfitting 批评
- [[dynastar]] — 把 STONE 当基线的动态拓扑 OOT 方法
- [[distributionally-robust-optimization]] — 另一条鲁棒目标路线（STOP 采用）

[^src-stone]: [[source-stone]]
[^src-st-ood]: [[source-st-ood]]
