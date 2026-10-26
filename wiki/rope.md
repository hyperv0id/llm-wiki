---
title: "Rotary Position Embedding (RoPE)"
type: technique
tags:
  - position-encoding
  - transformer
  - attention
  - rotary-position-embedding
  - llm
created: 2026-06-22
last_updated: 2026-10-26
source_count: 7
confidence: high
status: active
---

# Rotary Position Embedding (RoPE)

RoPE（Rotary Position Embedding）把 token 的绝对位置表示为旋转角，对 query 与 key 施加分块正交旋转，使二者内积只通过相对位移 $n-m$ 依赖位置[^src-roformer]。它作用于自注意力内部的投影结果而非词嵌入：位置以乘法进入 $W_q x_m$ 与 $W_k x_n$，全程不改变向量范数。论文 2021-04 首次提交（arXiv v1），实验以 2023-11 的 v5 为准[^src-roformer]。

## 问题：自注意力本身不含位置

打分 $q_m^\top k_n$ 与 token 的排列无关，位置信息必须从外部注入。要求内积只依赖相对位移，形式化写成

$$\langle f_q(x_m,m),\,f_k(x_n,n)\rangle = g(x_m,\,x_n,\,m-n)$$

即位置只能以差值形式出现在内积里。既有做法都在这个等式上做加法替换：绝对位置编码把 $p_m$ 加到 $x_m$；Shaw 式相对编码把 key 的输入改成 $x_n+\tilde p_{m-n}$ 并对相对距离做 clip；Transformer-XL / T5 / DeBERTa 把 $q_m^\top k_n$ 展开成四项，逐项替换位置项或直接加可学习偏置 $b_{m,n}$[^src-roformer]。RoPE 则不改展开项，直接改变 $f_q$、$f_k$ 自身，让位置以旋转形式乘在投影结果上。

## 二维情形：解出旋转

加法型方案没有走到这一步：基线一侧（绝对位置嵌入）把位置直接加到输入上，另一侧（XLNet 式相对偏置）把相对偏移的可学习向量加到 key 与 value 上——Shaw 等（2018）给出的正是后者的原始形式，位置项与内容项在 softmax 前相加[^src-shaw-2018-relative-position]。两条路径都让位置停留在表示层的加法上，而 RoPE 要问的是能否把它变成内积中的乘法。

## 形式化

设 $f_q(x_q,m)=R_q(x_q,m)e^{\mathrm i\Theta_q(x_q,m)}$、$f_k$ 同构，将内积的模与辐角分开，得两条约束[^src-roformer]：

模条件为 $R_q(x_q,m)R_k(x_k,n)=R_g(x_q,x_k,n-m)$。令 $m=n$ 并代入初始条件 $f_{\{q,k\}}(\cdot,0)=W_{\{q,k\}}x$，得 $R_q=\|q\|$、$R_k=\|k\|$、$R_g=\|q\|\|k\|$，即径向部分与位置无关。辐角条件中 $\Theta_k-\Theta_q$ 只依赖 $n-m$，且差值与词嵌入无关。记位置部分为 $\phi(m)$，取 $n=m+1$ 得 $\phi(m+1)-\phi(m)$ 为与 $m$ 无关的常数，于是 $\phi(m)=m\theta+\gamma$，$\theta\ne 0$。

取 $\gamma=0$ 后得到

$$f_q(x_m,m)=(W_q x_m)\,e^{\mathrm im\theta},\qquad f_k(x_n,n)=(W_k x_n)\,e^{\mathrm in\theta}$$

即把投影后的二维向量按位置转角的倍数旋转。二维时 $\theta$ 可以是任意非零常数，其自由度由高维推广中的频率分配固定。

## 高维：$d/2$ 个二维子空间并行旋转

$d$ 维时将相邻两维 $(2i-1,\,2i)$ 配对，第 $i$ 对旋转 $m\theta_i$，整体是分块对角正交矩阵 $R^d_{\Theta,m}$，频率向量取

$$\Theta=\Big\{\theta_i=10000^{-2(i-1)/d},\ i=1,\dots,d/2\Big\}$$

该序列与原始 Transformer 正弦位置编码同源（二者的第 $i$ 对维度共用 $10000^{-2(i-1)/d}$），差别仅在相乘而非相加[^src-roformer]。$i=1$ 一对旋转最快（每步 1 弧度），$i=d/2$ 一对最慢。§3.4.3 在 0 基索引下将同一 schedule 写作 $\theta_i=10000^{-2i/d}$（$i=0,\dots,d/2-1$），两种编号只差一个下标平移。

代入注意力打分：

$$
\begin{aligned}
q_m^\top k_n &= (R^d_{\Theta,m}W_q x_m)^\top (R^d_{\Theta,n}W_k x_n) \\
&= x_m^\top W_q^\top R^d_{\Theta,n-m}\,W_k x_n, \qquad R^d_{\Theta,n-m}=(R^d_{\Theta,m})^\top R^d_{\Theta,n}
\end{aligned}
$$

分块对角旋转矩阵的转置即反向旋转，同角度两次旋转相乘消去绝对位置，只留下角度差 $n-m$；正交性同时保证 $\|R^d_{\Theta,m}v\|=\|v\|$。位置信息的相对性来自旋转群同态 $R_m^\top R_n=R_{n-m}$，而非将 $p_m$ 替换为 $\tilde p_{m-n}$，这是它与既有相对编码的分界[^src-roformer]。

## 数据流与维度变换

**维度符号**：$N$ 序列长度（token 数）；$d$ 单头通道维，偶数，$d/2$ 为二维子空间个数；$d_{\text{model}}$ 隐层维；$H$ 头数；$d_v$ 值向量维；$m,n\in[1,N]$ 绝对位置。矩阵按 $[\text{行}=N,\ \text{列}=d]$ 记，与论文单头记号一致；多头展开处标 [INFERENCE]。

### Stage 1 · 造带位置信息的 query 与 key

两条支路并行：A 读输入嵌入做投影，B 只用位置索引算角度，汇合时按对施加旋转。

**支路 A：投影**

$$\begin{aligned}
Q &= XW_Q &&\text{// }[N,\bcancel{d_{\text{model}}}]\times[\bcancel{d_{\text{model}}},d]\to[N,d]\ \text{——第 } m \text{ 行是 } x_m^\top W_Q\text{，此刻仍与位置无关}\\
K &= XW_K &&\text{// }[N,\bcancel{d_{\text{model}}}]\times[\bcancel{d_{\text{model}}},d]\to[N,d]\ \text{——key 侧同形，后续与 } Q \text{ 共用同一旋转算子}
\end{aligned}$$

**支路 B：位置只用来算角度**

$$\begin{aligned}
\Phi &= \mathbf m\,\Theta^\top &&\text{// }[N,1]\times[1,d/2]\to[N,d/2]\ \text{——}\Phi_{m,i}=m\theta_i\text{，第 } m \text{ 行的 } d/2 \text{ 个角度沿 } i \text{ 几何递减}\\
C &= \cos\Phi,\quad S=\sin\Phi &&\text{// }[N,d/2]\xrightarrow{\text{逐元素}}\ [N,d/2]\ \text{——只由位置与频率决定，与输入无关，可在前向中一次算定并复用}
\end{aligned}$$

**汇合：按对做 $2\times2$ 旋转**

$$\begin{aligned}
\bar Q &= \operatorname{reshape}(Q) &&\text{// }[N,d]\xrightarrow{\text{末维按相邻两维成组}}[N,d/2,2]\ \text{——每行拆成 } d/2 \text{ 个二维子向量}\\
\tilde Q_{m,i,:} &= \begin{pmatrix}C_{m,i}&-S_{m,i}\\ S_{m,i}&C_{m,i}\end{pmatrix}\begin{pmatrix}\bar Q_{m,i,0}\\ \bar Q_{m,i,1}\end{pmatrix} &&\text{// }[N,d/2,\bcancel{2}]\times[\bcancel{2},2]\to[N,d/2,2]\ \text{——子向量 } i \text{ 旋转 } m\theta_i\text{，组间不混合}\\
\tilde Q &= \operatorname{flatten}(\tilde Q) &&\text{// }[N,d/2,2]\to[N,d]\ \text{——回到原形状，逐行范数不变}
\end{aligned}$$

$\tilde K$ 用同一个算子。实现上不构造 $d\times d$ 矩阵，改写成逐元素乘加[^src-roformer]：

$$\begin{aligned}
\tilde Q &= Q\odot C_{\text{dup}}+\operatorname{rot}(Q)\odot S_{\text{dup}} &&\text{// }[N,d]\odot[N,d]\to[N,d]\ \text{——}C_{\text{dup}}\text{ 是每列重复两次的 }[N,d]\text{，}\operatorname{rot}(Q)=(-q_2,q_1,-q_4,q_3,\dots)\text{ 为组内旋转 }90^\circ\text{；代价 }O(Nd)\\
\operatorname{rot}(Q) &= Q\,P_{\text{rot}} &&\text{// }[N,\bcancel{d}]\times[\bcancel{d},d]\to[N,d]\ \text{——}P_{\text{rot}}\text{ 是按 }(2i,2i-1)\text{ 换位并取负的置换，工程上直接按索引取，不落地成矩阵}
\end{aligned}$$

### Stage 2 · 打分只留相对位移

$$\begin{aligned}
S^{\text{att}} &= \tilde Q\tilde K^\top/\sqrt d &&\text{// }[N,\bcancel{d}]\times[\bcancel{d},N]\to[N,N]\ \text{——}S^{\text{att}}_{mn}\sqrt d=x_m^\top W_Q^\top R^d_{\Theta,n-m}W_Kx_n\text{，位置只在 } n-m \text{ 上出现}\\
S^{\text{att}}_{mn}\sqrt d &= \operatorname{Re}\sum_{i=1}^{d/2} h_{m,n,i}\,e^{\mathrm i(m-n)\theta_i} &&\text{// }[N,N,d/2]\xrightarrow{\text{沿 } d/2 \text{ 维求和}}\ [N,N]\ \text{——}h_{m,n,i}\in\mathbb{C}\text{ 为第 } i \text{ 对的复数内积，第 } i \text{ 项以周期 }2\pi/\theta_i\text{ 随位移振荡}
\end{aligned}$$

第二行是 Stage 1 复数写法的等价形式，把打分拆成 $d/2$ 个不同频率的振荡项之和，长距离衰减的讨论全部落在这组项上；逐项写开只是分析记号，计算上仍是 $[N,N]$ 的一次矩阵乘。

### Stage 3 · 加权取值与输出投影

$$\begin{aligned}
A &= \operatorname{softmax}(S^{\text{att}}) &&\text{// }[N,N]\xrightarrow{\text{沿列维（key 维）逐行 softmax}}[N,N]\ \text{——位置信息到此已全部进入权重}\\
O &= AV &&\text{// }[N,\bcancel{N}]\times[\bcancel{N},d_v]\to[N,d_v]\ \text{——}V=XW_V\text{ 不经旋转，值通路与位置无关}\\
O^{\text{cat}} &= \operatorname{concat}_h\big(O^{(h)}\big) &&\text{// }H\times[N,d_v]\to[N,Hd_v]\ \text{——各头用同一条频率 schedule 旋转自己的 }Q,K\\
O^{\text{out}} &= O^{\text{cat}}W_O &&\text{// }[N,\bcancel{Hd_v}]\times[\bcancel{Hd_v},d_{\text{model}}]\to[N,d_{\text{model}}]\ \text{——回到隐层维}
\end{aligned}$$

多头时每头按头维 $d_h=d/H$ 重取 $\Theta$ 是通行实现[INFERENCE]：论文按单头 $d$ 书写，未规定 $d$ 取模型维还是头维。位置只进 $Q,K$、不进 $V$ 源自式 (14) 仅定义 $f_{\{q,k\}}$，而 Shaw 式编码同时改写 $f_v$[^src-roformer]。

### Stage 2′ · 线性注意力下的同一构造

线性注意力用特征映射后的点积替代 softmax 核，RoPE 的旋转分别乘在两侧特征映射输出上[^src-roformer]：

表中的「XLNet 式相对偏置」即 [[relative-position-representation]]：$2k+1$ 个可学习向量按 $\operatorname{clip}(j-i,k)$ 取用，分别加到 $K$ 与 $V$，位置信息停在表示层的加法上；RoPE 则用旋转把它写进内积，表示里不额外携带位置向量[^src-shaw-2018-relative-position]。

## 在 LLM 中的统治地位

$$\begin{aligned}
u_m &= R^d_{\Theta,m}\,\phi(q_m) &&\text{// }[N,\bcancel{d}]\times[\bcancel{d},d]\to[N,d]\ \text{——query 侧先旋转，再与 key 侧内积}\\
z_n &= R^d_{\Theta,n}\,\varphi(k_n) &&\text{// }[N,\bcancel{d}]\times[\bcancel{d},d]\to[N,d]\ \text{——key 侧同角度旋转，两侧角度同源}\\
\text{num}_m &= u_m^\top\Big(\sum_n z_n v_n^\top\Big) &&\text{// }[N,\bcancel{d}]\times[\bcancel{d},d_v]\to[N,d_v]\ \text{——状态 }\sum_n z_nv_n^\top\in\mathbb{R}^{d\times d_v}\text{ 与 } m \text{ 无关，可前缀累加}\\
\text{den}_m &= \phi(q_m)^\top\sum_n\varphi(k_n) &&\text{// }[N,\bcancel{d}]\times[\bcancel{d},1]\to[N,1]\ \text{——分母不旋转，避免出现零}
\end{aligned}$$

由 $R_m^\top R_n=R_{n-m}$ 得 $u_m^\top z_n=\phi(q_m)^\top R^d_{\Theta,n-m}\varphi(k_n)$，打分仍只依赖 $n-m$；位置因子在累加前乘入 $z_n$，状态与 query 位置解耦，保留 $O(N)$ 的累加形式，每步只多一次 $O(d)$ 旋转。旋转后的 $z_n$ 含负分量，$\text{num}_m$ 内各项可相互抵消，权重不再严格概率归一；论文为此保留未旋转的分母[^src-roformer]。

## 长期衰减：可核对的界

用 Abel 变换改写 Stage 2 的振荡求和。记 $h_i=q_{[2i:2i+1]}k^*_{[2i:2i+1]}$（$q=W_qx_m$ 与 $k=W_kx_n$ 取旋转前投影，旋转量已计入相位）、$S_j=\sum_{i<j}e^{\mathrm i(m-n)\theta_i}$、$S_1=0$、$h_{d/2+1}=0$[^src-roformer]：

$$\begin{aligned}
\sum_i h_i e^{\mathrm i(m-n)\theta_i} &= \sum_i h_i(S_{i+1}-S_i) = -\sum_i S_{i+1}(h_{i+1}-h_i) &&\text{// 复数和恒等变形，不含近似}\\
\Big|\sum_i h_i e^{\mathrm i(m-n)\theta_i}\Big| &\le \sum_i |S_{i+1}|\,|h_{i+1}-h_i| \le \Big(\max_i|h_{i+1}-h_i|\Big)\sum_i|S_{i+1}| &&\text{// 三角不等式后把随数据变化的因子提到求和之外，剩下纯相位项}
\end{aligned}$$

两个因子中仅后一个含位移 $\Delta=m-n$。前一个因子由范数界控制：

$$|h_i|\le \|q_{[2i:2i+1]}\|\,\|k_{[2i:2i+1]}\|\le \|q_m\|\,\|k_n\|,\qquad |h_{i+1}-h_i|\le 2\|q_m\|\|k_n\|$$

于是内积的模满足

$$|S^{\text{att}}_{mn}|\sqrt d\le 2\|q_m\|\|k_n\|\cdot B(\Delta),\qquad B(\Delta)=\sum_{i=1}^{d/2}|S_{i+1}|$$

$B$ 是唯一随相对位移变化的部分，衰减性质就是关于 $B$ 的陈述。

**$B$ 随 $\Delta$ 变化的两段形态。** 位移为零时所有相位相同，$|S_{j+1}|=j$，$B$ 取最大值 $\sum_{j=1}^{d/2}j\approx\frac{1}{2}(d/2)^2$。位移增大后，第 $i$ 项每次前进 $\Delta\theta_i$；$\theta_i$ 沿 $i$ 几何递减，前几对迅速转过整圈，中段各对相位在 $\Delta$ 增大时近似独立且均匀分布在单位圆上，部分和退化为随机游走 $|S_j|\approx\sqrt j$，$B$ 因而降至 $\sum_{j\le d/2}\sqrt j\approx\frac{2}{3}(d/2)^{3/2}$ 量级并基本持平。衰减即 $B$ 从上界 $\Theta(d^2)$ 降至 $\Theta(d^{3/2})$ 后不再下降，并不趋于零。论文 Figure 2 给出归一化后的 $\frac{1}{d/2}B(\Delta)$：曲线自相对位移 1 附近的约 20 降至 250 处的约 8，伴随周期起伏且不归零，与两段形态一致[^src-roformer]。

**成立前提。** 这条界依赖三个条件，缺一条就不成立：

1. $d$ 为偶数，且通道按相邻两维成对分组 $(2i-1,2i)$；分组方式不同，复数分解不成立。
2. 频率是比值为 $10000^{-2/d}$ 的几何序列；换成等频或可学习频率，相位不再逐对拉开，随机相位那一步的论证失效。
3. $\|q_m\|$、$\|k_n\|$ 有界且不随位置增长——旋转保范使这一点成立，$h_i$ 的幅值因此与 $m,n$ 只通过位移相关；若位置以加法进入向量本身，$q_m$ 就带位置依赖，$\max_i|h_{i+1}-h_i|$ 不再是常数。

该界约束单个打分模的上包络，而非每个 $(m,n)$ 的单调性：特定频率组合可在某些 $\Delta$ 处重新同相并抬高内积。上界本身不紧（随机游走为量级估计而非证明），且 softmax 归一化不受逐项界直接控制，「远距离打分被压低」无法由此导出。论文在 §4.5.5 将「长文本上表现更好的原因」列为未解释项[^src-roformer]。

## 与相对位置编码的关系

各类方案的位置注入点不同，是 RoPE 区别于其他位置编码的关键：

| 方案 | 注入点 | 形式 | 新增参数 |
|---|---|---|---|
| 绝对位置编码 | 词嵌入 | $W_q(x_m+p_m)$ | 无（正弦）或有（可学习） |
| Shaw 式相对编码 | key（及值）的输入 | $W_k(x_n+\tilde p_{m-n})$，相对距离 clip | 有 |
| T5 式偏置 | 打分 | $x_m^\top W_q^\top W_kx_n+b_{m,n}$ | 有 |
| Transformer-XL 式 | 打分的四项展开 | $q_m^\top k_n$ 拆成四项，$p_n$ 换成 $\tilde p_{m-n}$，$p_m$ 换成两个可训练向量 | 有 |
| DeBERTa 式 | 打分的中段两项 | 把 $p_m,p_n$ 换成 $\tilde p_{m-n}$ 后与内容项分别相乘 | 有 |
| RoPE | 投影结果 | $R^d_{\Theta,m}W_qx_m$ | 无 |

前五行均向张量添加位置项，位置与内容在加法中混合；RoPE 将内容保留在 $W_qx$ 中，仅用正交变换改变方向，位置与内容在分数上以乘法组合[^src-roformer]。由此可直接得出三条性质：

RoPE 保范：$\|R^d_{\Theta,m}v\|=\|v\|$，位置不改变 $q,k$ 的尺度，旋转矩阵正交，多层叠加不放大数值；无参数且 $O(Nd)$，位置由位置索引与固定频率算出，前向只多逐元素乘加；可与线性注意力组合，位置以保范旋转乘在非线性特征映射之后，$\phi(\cdot)$、$\varphi(\cdot)$ 的值域不受破坏，状态累加仍与 query 位置无关（见 Stage 2′）[^src-roformer]。

论文将「给线性自注意力配上相对位置编码」作为 RoPE 区别于既有相对编码方案的主区分点：Shaw、T5、DeBERTa 一类方法把位置作为加到表示上的项，无法在核分解后的线性注意力里保持相对形式[^src-roformer]。

## 外推：性质表述与实测的分界

$R^d_{\Theta,m}$ 对任意整数 $m$ 均有定义，位置以旋转进入，公式里没有超出训练长度的项，论文据此将「序列长度灵活」列为 RoPE 的性质[^src-roformer]。但 ALiBi 评测显示，rotary 位置编码在超出训练长度的序列上困惑度显著劣化，不做插值或缩放时无法直接外推[^src-alibi]。后续上下文扩展（如 [[yarn|YaRN]] 把 Llama 2 的 4k 窗口扩到 128k）均建立在后者的插值与缩放之上[^src-yarn]。

## 后续发展

- **上下文扩展**：[[ntk-aware-interpolation]]、[[ntk-by-parts-interpolation]]、[[dynamic-scaling]]、[[yarn]]
- **时间序列适配**：[[temporal-rotation]]（SIREN-RoPE，把固定序数索引换成时间条件化旋转）、[[learnable-frequency-scaling]]（可学习频率缩放）
- **物理知情位置编码**：[[physics-informed-position-encoding|PIPE]] 把位置 ID 换成物理量（经纬度 + 时间），配合 [[variant-frequency-positional-encoding|变频率正弦编码]]，用于台风预测[^src-pipe]
- **统一理论框架**：[[generalized-positional-encoding-framework]]
- **图结构扩展**：[[wire|WIRE]] 用图 Laplacian 谱特征作旋转坐标，在网格图上退回标准 RoPE[^src-2509-22259]
- **注意力温度**：[[attention-temperature-scaling]]（等价于缩放旋转嵌入）
- **收敛性分析**：[[convergent-normalization]]
- **预旋转空间几何**：[[qk-concentration]] 观察到 pre-RoPE 的 Q/K 向量高度集中，此时注意力 logit 近似为 Q-K 距离的 [[attention-distance-preference|三角级数]]，据此做 [[kv-cache-compression|KV cache 压缩]][^src-triattention]

## 相关页面

- [[source-roformer]] — 原始论文摘要
- [[roformer]] — RoFormer 模型实体
- [[siren-rope]] — 时间条件化 RoPE 扩展
- [[alibi]] — 替代位置编码方案
- [[relative-position-representation]] — 加法型相对位置表示，旋转类编码的前身
- [[scaling-factor-sqrt-dk]] — 注意力 $\frac{1}{\sqrt{d_k}}$ 缩放因子的数值稳定性
- [[alibi]] — 加性线性偏置的替代方案
- [[scaling-factor-sqrt-dk]] — 注意力 $1/\sqrt{d_k}$ 缩放因子的数值稳定性
- [[linear-attention-bias]] — ALiBi 的偏置注入方式与位置编码对照
- [[physics-informed-position-encoding]] — PIPE：RoPE 的物理知情扩展
- [[pipe]] — PIPE 台风预测模型
- [[qk-concentration]] / [[attention-distance-preference]] — RoPE 注意力在 pre-RoPE 空间的几何结构
- [[triattention]] — 基于该几何结构的 KV cache 压缩

[^src-roformer]: [[source-roformer]]
[^src-alibi]: [[source-alibi]]
[^src-yarn]: [[source-yarn]]
[^src-pipe]: [[source-pipe]]
[^src-2509-22259]: [[source-2509-22259]]
[^src-triattention]: [[source-triattention]]
[^src-shaw-2018-relative-position]: [[source-shaw-2018-relative-position]]
