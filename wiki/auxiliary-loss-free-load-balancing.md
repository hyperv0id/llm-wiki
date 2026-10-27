---
title: "Auxiliary-Loss-Free Load Balancing"
type: technique
tags:
  - mixture-of-experts
  - load-balancing
  - routing
  - deepseek
created: 2026-10-27
last_updated: 2026-10-27
source_count: 1
confidence: medium
status: active
---

# Auxiliary-Loss-Free Load Balancing

Loss-Free Balancing（Wang et al., DeepSeek-AI 与北京大学, arXiv:2408.15664, 2024-08 preprint）将负载均衡移出损失函数，改为每个 routed expert 维护一个标量 bias $b_i$。bias 仅参与 top-$K$ 的选择比较，训练后按上一 batch 的负载偏差 sign 更新；它不加到路由权重上，也不进入任何损失项，模型参数的梯度只来自语言建模目标[^src-wang-2024-aux-free]。方法在 [[deepseekmoe]] 的细粒度专家加共享专家上验证，主实验将门控从 softmax 换成 sigmoid（§4.1）[^src-wang-2024-aux-free]。

## 问题：辅助损失的两难

主流做法在主损失中引入 $\mathcal{L}_{\mathrm{Balance}}=\alpha\sum_{i=1}^{N}f_iP_i$，其中 $f_i$ 是分给 expert $i$ 的 token 比例（乘 $N/K$ 归一，完全均匀时为 $f_i=1$），$P_i$ 是该 expert 在序列上的平均路由打分（式 2）[^src-wang-2024-aux-free]。$f_i$ 来自不可微的 top-$K$ 计数，实际反传仅走 $P_i$：$\alpha$ 越大，路由打分受该梯度的牵引越强。这构成两难代价：$\alpha$ 大则负载受控但训练受损，$\alpha$ 小或不加则出现路由塌缩（Shazeer et al. 2017 的同一问题）[^src-wang-2024-aux-free]。

图 2 在 1B 上扫描 $\alpha\in\{10^{-2},10^{-3},10^{-4},0\}$：$\alpha=10^{-2}$ 的困惑度为 9.70、MaxVio 接近 0；$\alpha=0$ 的困惑度为 9.54，MaxVio 约 5.0；$\alpha=10^{-3}$ 落在 9.56/0.72。该点与表 2 的 1B 辅助损失基线完全一致，确认图 2 的扫描采用 1B 设置[^src-wang-2024-aux-free]。Loss-Free 在同图的点为 9.50/0.04。

表 1 对比三类方法的性质：强辅助损失均衡但有干扰梯度；弱辅助损失干扰小但失衡；Expert Choice 均衡且无干扰梯度，却让未来 token 参与当前 token 的专家分配，破坏因果性；Loss-Free 均衡、无干扰梯度且无泄漏[^src-wang-2024-aux-free]。

## 机制：bias 只改选择，不改权重

每个 expert 设一个 bias $b_i$，按加 bias 的打分取 top-$K$，但写回路由权重的是**未加 bias** 的原打分（式 3）[^src-wang-2024-aux-free]：

$$
g_{i,t}=
\begin{cases}
s_{i,t}, & s_{i,t}+b_i\in\operatorname{Top}_K\big(\{s_{j,t}+b_j\}_{j=1}^{N}\big),\\
0, & \text{otherwise}.
\end{cases}
$$

$b_i$ 仅用于 top-$K$ 选择，不加到给专家输出加权的 $g_{i,t}$ 上[^src-wang-2024-aux-free]。这是该方法与辅助损失的分界：模型前向数值路径与无 bias 时一致，改变的仅是「选中谁」。

bias 的更新是一条离散控制回路（Algorithm 1）[^src-wang-2024-aux-free]：

1. 初始化 $b_i=0$；
2. 用式 (3) 的路由在 batch 上正常训练一步；
3. 统计每个 expert 实际分到的 token 数 $c_i$ 及其平均值 $\bar c_i$；
4. 计算负载违例 $e_i=\bar c_i-c_i$；
5. 更新 $b_i\leftarrow b_i+u\cdot\operatorname{sign}(e_i)$。

超载 expert（$c_i>\bar c_i$）的 $e_i<0$，bias 下调；欠载则上调。$u$ 是更新率，主实验取 $0.001$。等价写法是 $b_i\leftarrow b_i-\gamma\operatorname{sign}(c_i-\bar c_i)$，$\gamma=u$；论文采用前者。论文没有 $\gamma$ 的上升或衰减调度，$u$ 全程固定，且只在 1B 上调参，3B 直接继承（§4.1、§4.3）[^src-wang-2024-aux-free]。

两个设计约束值得单独记：

更新依据是上一个 batch 的统计量，不用当前序列的负载；使用当前序列会破坏语言建模的因果约束、泄漏未来 token 信息[^src-wang-2024-aux-free]。更新采用 sign 而非线性：换成 $b_i\leftarrow b_i+u\cdot e_i$ 会让偏离大的 expert 调得更快、负载更均衡，但困惑度没有改善（表 3），因此保留 sign[^src-wang-2024-aux-free]。

## 数据流与维度变换

> **维度符号**：$T$ 一个 batch 内的 token 数（最大序列长 2048）；$N$ 路由专家数（64）；$K$ 每 token 激活的路由专家数（6）；$N_s$ 共享专家数（2）；$d$ 隐层宽（1B 为 1024，3B 为 1280）；$b_i$ 是第 $i$ 个专家的标量 bias，其统计窗口为一个 batch，更新时读上一个 batch 的计数。形状按 $[\text{token},\text{expert}]$ 记。

### Stage 1 · sigmoid 门控打分

$$\begin{aligned}
z_{i,t} &= u_t^{\top}e_i,\qquad s_{i,t}=\sigma(z_{i,t}) && \text{// } [T,\bcancel{d}]\times[\bcancel{d},N]\to[T,N]\ \text{——消去 } d\text{，每个 token 对 64 个路由专家各得一个 }(0,1)\text{ 打分；主实验用 sigmoid，专家之间不归一化}
\end{aligned}$$

$e_i$ 是第 $i$ 个专家的质心，$G$ 是非线性门控（式 1）；主实验取 $G=\sigma$，附录 C 另给 softmax 结果[^src-wang-2024-aux-free]。模型没有 affine gating，门控仅在 $u_t^{\top}e_i$ 上施加逐元素非线性。

### Stage 2 · 加 bias 后取 top-K

$$\begin{aligned}
\mathcal{I}_t &= \operatorname{Top}_K\big(\{s_{j,t}+b_j\}_{j=1}^{N}\big) && \text{// } [T,N]\xrightarrow{\text{沿 } N \text{ 维逐 token 取 top-}K}[T,K]\ \text{——bias 只在这次比较中进入，决定选中哪 } K \text{ 个专家}
\end{aligned}$$

### Stage 3 · 路由权重取未加 bias 的打分

$$\begin{aligned}
g_{i,t} &= \begin{cases}s_{i,t}, & i\in\mathcal{I}_t\\ 0, & \text{otherwise}\end{cases} && \text{// } [T,N]\xrightarrow{\text{按 } \mathcal{I}_t \text{ 每行保留 } K \text{ 个位置、其余置零}}[T,N]_{\text{稀疏}}\ \text{——返回原打分，选中位不重归一}
\end{aligned}$$

bias 不进入 $g_{i,t}$：在选中集合 $\mathcal{I}_t$ 不变的分段上，$g$ 对 $u_t$ 与 $e_i$ 的梯度与没有 bias 时逐项相同——这就是「无干扰梯度」在计算图上的落点[^src-wang-2024-aux-free]。softmax 门控下调整 bias 更难：归一化让两个专家的分数差随其他专家的打分变化，固定步长的 bias 不易稳定，附录 C 改用线性更新 $b_i\leftarrow b_i+u\cdot e_i$（$u=10^{-3}$）[^src-wang-2024-aux-free]。

### Stage 4 · 共享专家与路由专家汇合

$$\begin{aligned}
h_t &= u_t+\sum_{i=1}^{N_s}\mathrm{FFN}_i^{(s)}(u_t)+\sum_{i=1}^{N}g_{i,t}\,\mathrm{FFN}_i^{(r)}(u_t) && \text{// } [d]+N_s\times[d]+K\times[d]\to[d]\ \text{——2 个共享专家常开、6 个路由专家按未加 bias 的 } s \text{ 加权，再叠输入残差}
\end{aligned}$$

附录 A 式 (6) 中，$s$ 表示共享专家、$r$ 表示路由专家，$g_{i,t}$ 只有 $K$ 个非零[^src-wang-2024-aux-free]。

### Stage 5 · 训练后更新 bias（不进计算图）

$$\begin{aligned}
c_i &= \big|\{t: i\in\mathcal{I}_t\}\big|,\qquad \bar c_i=\frac{KT}{N} && \text{// } [T,K]\xrightarrow{\text{逐专家计数}}[N]\ \text{——本 batch 每个 expert 实收 token 数，对照完全均匀时的 } KT/N\\
e_i &= \bar c_i-c_i,\qquad b_i\leftarrow b_i+u\,\operatorname{sign}(e_i) && \text{// } [N]\ \text{——超载则 } e_i<0\text{、bias 降；欠载则升；sign 把步长截成固定 } u
\end{aligned}$$

论文在 Algorithm 1 第 3 行仅写「平均分配数 $\bar c_i$」而未给闭式，top-$K$ 下均匀分配即 $KT/N$，上式按此理解 [INFERENCE]。该步在优化器更新后执行，不产生梯度，也不进入反向图[^src-wang-2024-aux-free]。

## 负载度量

论文以最大违例 MaxVio 衡量均衡程度（式 4）[^src-wang-2024-aux-free]：

$$
\mathrm{MaxVio}=\frac{\max_i \mathrm{Load}_i-\overline{\mathrm{Load}_i}}{\overline{\mathrm{Load}_i}},
$$

$\mathrm{Load}_i$ 是分给 expert $i$ 的 token 数，$\overline{\mathrm{Load}_i}$ 是完全均衡时的期望负载。两个变体中，$\mathrm{MaxVio}_{\text{global}}$ 在整个验证集上统计，反映专家利用率与效率上限；$\mathrm{MaxVio}_{\text{batch}}$ 在每个训练 batch 上统计，更贴近训练效率。正文报告各层平均值[^src-wang-2024-aux-free]。

## 实验

**设置**（§4.1、附录 B、表 5）：骨干为 [[deepseekmoe]]，设 1B 与 3B 两个规模。1B：hidden 1024、8 个注意力头、9 层 MoE、粒度 $d_{ff}/d_{expert}=16/3$、2 个共享专家、64 个路由专家、激活 6 个；3B：hidden 1280、10 头、11 层 MoE、粒度 4，其余相同。1B 训 100B token（batch 1152、40000 步、cosine 调度），3B 训 200B token（batch 1728、56514 步、多步调度）；验证集从训练语料留出约 71M token。基线辅助损失系数 $\alpha=0.001$[^src-wang-2024-aux-free]。

### 主结果（表 2、图 3）

| 规模 | 方法 | 验证困惑度 | $\mathrm{MaxVio}_{\text{global}}$ |
|---|---|---|---|
| 1B | 辅助损失（$\alpha=0.001$） | 9.56 | 0.72 |
| 1B | Loss-Free | 9.50 | 0.04 |
| 3B | 辅助损失（$\alpha=0.001$） | 7.97 | 0.52 |
| 3B | Loss-Free | 7.92 | 0.04 |

两个规模上 Loss-Free 的困惑度均低 0.05–0.06，全局最大违例从 0.52–0.72 降至 0.04[^src-wang-2024-aux-free]。图 3 的 $\mathrm{MaxVio}_{\text{batch}}$ 曲线（100 步滑动平均）显示其在训练大部分时间均更低[^src-wang-2024-aux-free]。

### 更新率与更新式（图 4、表 3、表 4）

图 4 在 1B 上扫 $u$：$10^{-4}$ 前期收敛慢，$10^{-2}$ 后期 bias 抖动、均衡变差，两者困惑度均为 9.51；$10^{-3}$ 的 9.50 最好[^src-wang-2024-aux-free]。

| 变体 | 困惑度 | $\mathrm{MaxVio}_{\text{global}}$ |
|---|---|---|
| $b_i+u\,\mathrm{sign}(e_i)$，$u=10^{-3}$ | 9.50 | 0.044 |
| $b_i+u\,e_i$，$u=10^{-2}$ | 9.53 | 0.028 |
| $b_i+u\,e_i$，$u=10^{-3}$ | 9.51 | 0.036 |
| $b_i+u\,e_i$，$u=10^{-4}$ | 9.51 | 0.040 |

线性更新把最大违例压到 0.028，但困惑度升至 9.53；论文折中选取 sign 版本的 0.044 与 9.50（表 3；表 2 将同一配置记作 0.04）[^src-wang-2024-aux-free]。乘性 bias（$s_{i,t}\cdot b_i$，初值 1）在 $u=10^{-2}/10^{-3}/10^{-4}$ 下困惑度为 9.52/9.52/9.54，均衡与加性相当（表 4），因而选用加性[^src-wang-2024-aux-free]。

### 门控形式（附录 C，表 6）

softmax 门控下辅助损失基线取 $\alpha=0.0003$（该门控下困惑度最低），Loss-Free 采用线性更新 $u=10^{-3}$：困惑度为 9.604 对 9.599，$\mathrm{MaxVio}_{\text{global}}$ 为 0.937 对 0.027[^src-wang-2024-aux-free]。softmax 下困惑度仅差 0.005，均衡差距与 sigmoid 下同量级。图 7 另显示：同等均衡程度下 softmax 基线困惑度更高，且对失衡更敏感[^src-wang-2024-aux-free]。

### 专家并行下的 batch 级均衡（§5.1、图 5）

一个计算步实际处理 `micro_batch_size × ep_data_parallel_size` 个样本，记为 computation batch。$\mathrm{MaxVio}_{\text{computation-batch}}$ 随 computation batch 增大，Loss-Free 持续改善，辅助损失基线在 batch 较大时基本走平；图 5 两栏分别标注 18 与 36 的 computation batch[^src-wang-2024-aux-free]。专家并行把 computation batch 放大 `ep_data_parallel_size` 倍，方法与大规模 MoE 训练天然相容[^src-wang-2024-aux-free]。

### Expert Choice 的未来 token 泄漏（§5.2、附录 D）

EC 分配给每个 expert 完全相同数量的 token，但当前 token 的专家分配取决于同 batch 的未来 token。稀疏率 $R=K/N$ 时，单层至少泄漏 $K\log_2\frac{1-R}{R}$ bit/token；9 层 MoE、16 个专家、每 token 平均 2 个专家（$R=1/8$）合计泄漏超过 50 bit/token（附录 D.1 式 7）[^src-wang-2024-aux-free]。实测在 2B MoE、100B token 上，将 top-$K$ 的 chunk 从 8192 缩到 512 会出现约 10% 的异常 loss 下降，打乱 token 后再切 chunk 则削弱了该降幅（附录 D.2、图 9）[^src-wang-2024-aux-free]。

## 与辅助损失的对照

两者的差别不在「均衡得好不好」，而在均衡信号走哪条路：

| | 辅助损失（式 2） | Loss-Free bias（式 3） |
|---|---|---|
| 均衡信号 | 标量损失项 $\alpha\sum_i f_iP_i$ | 每个 expert 一个标量 $b_i$ |
| 进入前向的位置 | 无（只进损失） | top-$K$ 比较，不进加权 |
| 对参数的梯度 | 经 $P_i$ 回传到路由打分，强度随 $\alpha$ | 无，bias 由 sign 规则更新 |
| 可调强度 | $\alpha$，与困惑度直接冲突 | $u$ 与统计窗口，不改主梯度方向 |
| 因果性 | 满足 | 满足（只用历史 batch） |

[[deepseekmoe]] 的 expert-level 与 device-level 两项辅助损失都属左列：直接计入总损失，由 $\alpha_1,\alpha_2$ 权衡均衡与性能（公式见该页）；该方法移除了全部辅助损失，改用独立回路替代[^src-wang-2024-aux-free]。论文未做「bias 与辅助损失同时开启」的组合实验，也未扫描 $\alpha$ 与 $u$ 的二维组合；§4 的对照始终是二选一[^src-wang-2024-aux-free]。

## 边界

训练规模限于 1B/3B 与 100B/200B token，bias 更新率仅在 1B 调优并由 3B 直接继承，未验证更大模型的可迁移性（§4.1）[^src-wang-2024-aux-free]。评测仅报告验证困惑度，缺少下游 benchmark，均衡指标仅用 MaxVio，未给各专家负载分布或熵[^src-wang-2024-aux-free]。训练 token 数口径不一：§4.1、摘要与附录 B 均记录 3B 训 200B token，§6 结论则写「100B and 300B tokens respectively」[^src-wang-2024-aux-free]。门控主实验为 sigmoid，softmax 仅置于附录 C 且换用线性而非 sign 更新式；softmax 下两法困惑度仅相差 0.005（9.599 对 9.604），收益主要体现在均衡[^src-wang-2024-aux-free]。bias 数值范围与动态、专家专业化程度、与容量因子/丢 token 策略的交互，以及 EC 的完整性能对照（仅做泄漏实验）均未报告[^src-wang-2024-aux-free]。方法虽不产生辅助损失梯度，但 bias 更新仍依赖手工设定的控制率 $u$，并非零超参，论文未列为局限[^src-wang-2024-aux-free]。

## 谱系

辅助损失形态见 [[sparse-gated-moe-layer]] 与 [[gshard]]，式 (2) 为论文采用的通用写法；backbone、共享专家及细粒度切分源自 [[deepseekmoe]]。该方案保留 DeepSeekMoE 架构，去除辅助损失，由路由侧 bias 回路接管均衡[^src-wang-2024-aux-free]。实验止于 1B/3B 自训验证，未涉及后续模型及更大规模 MoE 的实际部署。

[^src-wang-2024-aux-free]: [[source-wang-2024-aux-free]]
