---
title: "Linear Attention 的 RNN 形式（Katharopoulos et al., 2020）"
type: technique
tags:
  - linear-attention
  - kernel-methods
  - rnn
  - autoregressive
  - efficiency
created: 2026-10-27
last_updated: 2026-10-27
source_count: 1
confidence: medium
status: active
---

# Linear Attention 的 RNN 形式（Katharopoulos et al., 2020）

本页是线性注意力原始论文（Katharopoulos et al., ICML 2020, arXiv:2006.16236）的机制页：核化推导、$\operatorname{elu}(x)+1$ 特征映射、因果掩码下的前缀和递推与 RNN 重写、梯度与复杂度，逐表实验数字收在证据一节。与后续工作的关系见文末；Mamba 对照与六项组件差异见 [[linear-attention-unified-framework]]。

## 第一步：广义注意力与非负性约束

抽象掉 softmax 注意力中的 $\operatorname{sim}(q,k)=\exp(q^{\top}k/\sqrt{D})$，注意力的一般形式为（式 3）[^src-katharopoulos-2020-linear-attention]：

$$V'_i=\frac{\sum_{j=1}^{N}\operatorname{sim}(Q_i,K_j)\,V_j}{\sum_{j=1}^{N}\operatorname{sim}(Q_i,K_j)}$$

论文对 $\operatorname{sim}(\cdot)$ 只加一条约束：非负。原文："the only constraint we need to impose to sim(·), in order for equation 3 to define an attention function, is to be non-negative. This includes all kernels $k(x,y):\mathbb{R}^{2\times F}\to\mathbb{R}^+$"（§3.2）[^src-katharopoulos-2020-linear-attention]。任何正定核都能充当注意力打分，softmax 只是取指数核的特例。

## 关键推导：结合律消掉 $N^2$

取非负核特征映射 $\phi$ 使 $\operatorname{sim}(q,k)=\phi(q)^{\top}\phi(k)$，代入式 3 再利用矩阵乘法结合律（式 4→6）[^src-katharopoulos-2020-linear-attention]：

$$(\phi(Q)\phi(K)^{\top})V=\phi(Q)\,(\phi(K)^{\top}V)$$

左式按 vanilla 顺序先物化 $N\times N$ 注意力矩阵；右式先算 $[C,N]\times[N,M]$ 的 $[C,M]$ 中间量供各 query 复用，复杂度从 $O(N^2)$ 降为 $O(N)$（§3.2）[^src-katharopoulos-2020-linear-attention]。该结合律重排是后续所有核化线性注意力（随机特征、agent token 等）共用的骨架；Performer 在此基础上回答"$\phi$ 能否逼近 softmax 核"，见 [[performer]]。

与 softmax 的数学差异在于归一化：softmax 采用行内指数归一化，分母耦合整行；线性化后退化为显式的 $\phi(Q_i)^{\top}\sum_j\phi(K_j)$，与分子共享同一组充分统计量，指数被核相似度替代。精确 softmax 线性化不可行，因为指数核的精确特征映射为无限维；多项式核（2 次）具有精确有限维特征映射，代价为 $O(ND^2M)$，仅在 $N>D^2$ 时占优（§3.2.1）[^src-katharopoulos-2020-linear-attention]。

## 特征映射：$\phi(x)=\operatorname{elu}(x)+1$

论文实验选用 $\phi(x)=\operatorname{elu}(x)+1$（式 7）[^src-katharopoulos-2020-linear-attention]：逐元素非负、维数不变（$C=D$）、处处有定义。选 elu 而非 relu 的原因见原文："We prefer elu(·) over relu(·) to avoid setting the gradients to 0 when x is negative"（§3.2.1）[^src-katharopoulos-2020-linear-attention]。该选择没有定理保证，结论基于经验；§5 将 $\phi$ 的选择列为待探索的开放方向。

## 因果掩码：前缀和递推

因果注意力只允许位置 $j\le i$ 参与打分（式 8–9）。线性化后分子分母均为前缀和，定义（式 10–11）[^src-katharopoulos-2020-linear-attention]：

$$S_i=\sum_{j=1}^{i}\phi(K_j)V_j^{\top},\qquad Z_i=\sum_{j=1}^{i}\phi(K_j)$$

即得 $V'_i=\phi(Q_i)^{\top}S_i/(\phi(Q_i)^{\top}Z_i)$（式 12）。由于 $S_i$、$Z_i$ 可由 $S_{i-1}$、$Z_{i-1}$ 常数时间更新，因果线性注意力对序列长度线性（§3.3）[^src-katharopoulos-2020-linear-attention]。

## 数据流与维度变换

**维度符号**：$N$ 序列长度（token 数）；$F$ 输入特征维；$D$ query/key 投影维；$M$ value 维；$C$ 特征映射维（$\operatorname{elu}+1$ 下 $C=D$）。形状 $[\text{行},\text{列}]$；论文按单头书写，本页照单头口径。$q_i,k_i\in\mathbb{R}^C$、$v_i\in\mathbb{R}^M$ 为列向量。

### Stage 1 · 投影与特征映射

$$\begin{aligned}
Q &= \phi(XW_Q) && \text{// } [N,F]\times[F,D]\to[N,D]\xrightarrow{\text{逐行 } \phi} [N,C]\ \text{——query 投影后逐行过特征映射，非负化}\\
K &= \phi(XW_K) && \text{// } [N,F]\times[F,D]\to[N,D]\xrightarrow{\text{逐行 } \phi} [N,C]\ \text{——key 同样非负化，可预先算好}\\
V &= XW_V && \text{// } [N,F]\times[F,M]\to[N,M]\ \text{——value 只做线性投影，不过 } \phi
\end{aligned}$$

### Stage 2 · 线性化打分（非因果双向形式，结合律重排）

$$\begin{aligned}
\widehat{V} &= \phi(K)^{\top}V && \text{// } [C,\bcancel{N}]\times[\bcancel{N},M]\to[C,M]\ \text{——消去 } N\text{：全部 key 的特征加权值和，所有 query 共享}\\
z &= \phi(K)^{\top}\mathbf{1}_N && \text{// } [C,\bcancel{N}]\times[\bcancel{N},1]\to[C]\ \text{——特征映射列和，充当归一化分母的充分统计量}\\
V' &= \phi(Q)\,\widehat{V}\oslash(\phi(Q)z) && \text{// } [N,\bcancel{C}]\times[\bcancel{C},M]\to[N,M]\ \text{——消去 } C\text{，逐行除以归一化；全程无 } N\times N \text{ 矩阵}
\end{aligned}$$

对照 vanilla 顺序：$(\phi(Q)\phi(K)^{\top})V$ 中 $\phi(Q)\phi(K)^{\top}$ 为 $[N,\bcancel{C}]\times[\bcancel{C},N]\to[N,N]$——正是结合律绕开的二次方项。

### Stage 3 · RNN 形式的因果解码（式 16–20）

初始 $s_0=0\in\mathbb{R}^{C\times M}$、$z_0=0\in\mathbb{R}^{C}$，逐 token 递推：

$$\begin{aligned}
s_i &= s_{i-1}+\phi(k_i)\,v_i^{\top} && \text{// } [C,M]+[C,1]\times[1,M]\to[C,M]\ \text{——注意力记忆：累加当前 token 的特征-值外积，常数时间更新}\\
z_i &= z_{i-1}+\phi(k_i) && \text{// } [C]+[C]\to[C]\ \text{——归一化记忆：特征映射的运行和}\\
y_i &= f_l\!\left(\frac{\phi(q_i)^{\top}s_i}{\phi(q_i)^{\top}z_i}+x_i\right) && \text{// } [1,\bcancel{C}]\times[\bcancel{C},M]\to[1,M]\ \text{——读状态得第 } i \text{ 步输出，加残差后过逐位置 } f_l
\end{aligned}$$

$s_i$ 是前 $i$ 个 key-value 对的核加权充分统计量，$z_i$ 是对应配重；解码第 $i$ 步只读这两个状态，每步时间与内存都是常数——softmax 自回归解码每步要重读全部历史 K/V，代价随 $i$ 增长（§3.3/§3.4）。论文以这组递推构造性证明了"任何因果 masked transformer 层都是 RNN"，且递归在时间维而非深度维（区别于 Universal Transformers，§3.4）[^src-katharopoulos-2020-linear-attention]。前向只需存 $s,z$ 两个状态；反向梯度推导（式 13–15）用同一递推沿 $1..N$ 与 $N..1$ 两个方向计算累积和（类比 BPTT），整体 $O(NCM)$ 时间，避免朴素实现存储全部 $S_i$ 带来的 $\max(D,M)$ 倍内存（§3.3.1，附录 A）[^src-katharopoulos-2020-linear-attention]。

### Stage 4 · 复杂度对比

$$\begin{aligned}
\text{softmax 训练} &= O(N^2\max(D,M)) && \text{// 时间：物化 } [N,N]\text{ 注意力矩阵并为其存梯度}\\
\text{linear 训练} &= O(NCM) && \text{// 时间：} C=D \text{ 时即 } O(NDM)\text{；内存 } O(N\max(C,M))\text{（§3.2.1/§3.3.1）}\\
\text{linear 解码} &= O(CM)\text{/步} && \text{// 每步常数：只更新 } s_i,z_i\text{；softmax 有状态实现每步仍 } O(i\max(D,M))
\end{aligned}$$

Fig 1 实测 $N\in\{2^9,\dots,2^{16}\}$：softmax 在 11 GB 显存上限内仅达 $N=4096$，linear 与 lsh 可达 16,384 且时间内存均随 $N$ 呈线性（§4.1.2）[^src-katharopoulos-2020-linear-attention]。

## 证据（作者报告口径）

基线包含 softmax（标准 Transformer）、Reformer（PyTorch 复现，设 lsh-1/lsh-4 两档哈希轮数，不用可逆层，仅对比注意力）与自实现的 stateful-softmax（缓存 K/V 的 RNN 式 softmax 解码，§C.1）；语音任务另加 3 层 320 宽 Bi-LSTM。

| 实验 | 设置 | linear | softmax | 倍率 | 出处 |
|---|---|---|---|---|---|
| MNIST 图像生成 | 8 层 8 头、250 epochs、batch 10 | 0.644 bpd / 142.8 img/s | 0.621 bpd / 0.45 img/s | 317× | Table 1 |
| CIFAR-10 图像生成 | 16 层、固定训 7 天、softmax batch 1 / linear batch 4 | 3.40 bpd / 17.85 img/s | 3.47 bpd / 0.004 img/s | 4,462× | Table 2 |
| 对 stateful-softmax（CIFAR-10） | 同上 | 17.85 img/s | 0.32 img/s | >50× | Table 4，§C.1 |
| 单图延迟（CIFAR-10, batch 1） | CPU / GPU | 45.1 s / 61.3 s | 8,651.4 s / 300.1 s | CPU 191.8× / GPU 4.9× | Table 5 |
| WSJ 语音识别（CTC） | 9 层 6 头，80h WSJ | PER 8.08 / 824 s/epoch | PER 5.12 / 2,711 s/epoch | >3× 每 epoch | Table 3 |

Table 3 中 Bi-LSTM 为 10.94 / 1,047 s，LSH-4 为 9.33 / 2,250 s。序列复制任务（§4.1.1，4 层 8 头，最大长度 128）中 linear 收敛稳定，最终损失与 softmax 相同且低于 lsh，论文归因于无哈希噪声（Fig 2）。摘要称 "up to 4000x faster on autoregressive prediction of very long sequences"，对应 Table 2 的 4,462×；Fig 5b 显示 CIFAR-10 上 linear 与 lsh-1 单位 GPU 时间完成的 epoch 远多于 softmax/lsh-4。

## 边界

- 语音识别质量：softmax PER 5.12 显著优于 linear 8.08，附录 B 自述该任务上 softmax 收敛显著更好；linear 的优势只在每 epoch 时间（824 对 2,711 s）[^src-katharopoulos-2020-linear-attention]。
- 图像生成质量：MNIST 上 0.644 对 0.621 bpd，linear 略差；CIFAR-10 上 3.40 对 3.47 略好，但 caption 自述这是因为 linear 多完成约 3 倍 epoch（固定 7 天），"the main point of this experiment is not the final perplexity"（§4.2.2）[^src-katharopoulos-2020-linear-attention]。
- 单图延迟口径：§C.2 正文称 CIFAR-10 单图生成 "almost 6.6× faster than softmax"，但 Table 5 数字为 GPU 300.1 s → 61.3 s（约 4.9×）、CPU 8,651.4 s → 45.1 s（约 191.8×），两个实测倍率都不等于 6.6×，属论文正文与表格不一致；本页照录 Table 5 数字，6.6× 不作为可用口径[^src-katharopoulos-2020-linear-attention]。
- $\phi$ 的选择无理论保证：作者把特征映射设计列为未来工作，并举例 RBF 随机傅里叶特征可能允许复用 softmax 预训练权重（§5）[^src-katharopoulos-2020-linear-attention]。
- 指数核精确特征映射无限维，本方案不是精确 softmax（§3.2.1）；论文同时声明该形式"can be used for representing any transformer model, in theory even the ones using softmax attention"（§3.4，指表达能力而非效率）[^src-katharopoulos-2020-linear-attention]。

## 在本 wiki 中的位置

[[linear-attention-unified-framework]] 对齐本页递推与 Mamba 选择性 SSM 公式并消融六项组件差异，本页循环形式在该页作为已知结论引用。[[performer]] 与 [[positive-random-features]] 沿本页骨架把 $\phi$ 换为 softmax 核的随机特征估计；Performer 附录报告 elu+1 线性注意力在其 ProGen 设置中训练早期梯度爆炸。[[fast-weight-mlp-view-of-attention]] 从 fast-weight 视角指出 $s_i$ 正是逐 token 即时写入的矩阵权重。[[lightning-attention]] 在基础模型规模复用本页的右积递推，用分块（块内左积 + 块间右积）解决 causal 场景 cumsum 不可并行的工程问题。[[source-katharopoulos-2020-linear-attention]] 为论文源摘要。

[^src-katharopoulos-2020-linear-attention]: [[source-katharopoulos-2020-linear-attention]]
