---
title: "Looped Transformer 的函数逼近率与 modulus of continuity"
type: concept
tags:
  - looped-transformer
  - approximation-theory
  - weight-tying
  - expressivity
  - modulus-of-continuity
created: 2026-10-28
last_updated: 2026-10-28
source_count: 1
confidence: medium
status: active
---

# Looped Transformer 的函数逼近率

Xu & Sato（ICML 2025）给单层 looped Transformer 建立了关于循环次数 $r$ 的逼近率：误差随 $r$ 收缩，但界里比标准 Transformer 多出两项 modulus of continuity，这两项被归因于 weight-tied feed-forward 网络[^src-xu-2025-looped-transformers-timestep-encoding]。同一分析推出通用逼近（无限精度下取 $r\to\infty$），也推出了后来催生 [[timestep-modulated-looped-transformer|TMLT]] 的那项结构缺陷。

## 问题定位

标准 Transformer 的 universality（Yun et al., 2020；Kajitsuka & Sato, 2024）与 rate 结果（Takakura & Suzuki, 2023；Jiang & Li, 2024；Wang & E, 2024）都建立在各层参数彼此独立之上，不能直接搬到只有一个固定参数集的 block 上；weight-tied ReLU 网络的 rate 已由 Zhang et al. (2023) 给出，looped Transformer 的 rate 论文自述此前没有（§1、§2.3）[^src-xu-2025-looped-transformers-timestep-encoding]。Table 1 按 Model Type / Function Class / Approximation Rate / Looped (Weight-Tying) 四个属性列对照八行工作，本文是唯一在 Transformer 类型下同时勾选 Approximation Rate 与 Looped 的一行；同样勾选 weight-tying 的 Zhang et al. (2023) 是 FFN[^src-xu-2025-looped-transformers-timestep-encoding]。论文自述这是首次研究 looped Transformer 的函数逼近能力（§6）[^src-xu-2025-looped-transformers-timestep-encoding]。

## 设定

目标函数类是有界域上的连续、置换等变序列函数（Definition 3.1、§3.1）：

$$
\mathcal{F}_{\mathrm{PE}}(\Omega)=\left\{f:\mathbb{R}^{d\times N}\to\mathbb{R}^{d\times N}\ \middle|\ f\ \text{在}\ \Omega\ \text{上连续},\ f(XP)=f(X)P\ \text{对任意置换矩阵}\ P\right\},\quad \Omega=[0,1]^{d\times N}.
$$

被逼近的假设空间是同一个 TF block 反复施加 $r$ 次的复合（Corollary 3.7）：

$$
\mathcal{H}=\left\{L_2\circ\mathrm{TF}^{\circ r}\circ L_1:[0,1]^{d\times N}\to[0,1]^{d\times N}\ \middle|\ m,q\le Cd,\ h=2,\ s=1,\ r\in\mathbb{N},\ W\in\mathbb{R}^{n_w}\right\}.
$$

Theorem 3.6 给出该族的一个具体实例：$\mathrm{TF}:\mathbb{R}^{(17d+9)\times N}\to\mathbb{R}^{(17d+9)\times N}$，两个头、head size 1、FFN 宽度 $q=49d+25$，$L_1:\mathbb{R}^{d}\to\mathbb{R}^{17d+9}$、$L_2:\mathbb{R}^{17d+9}\to\mathbb{R}^{d}$ 逐 token 作用[^src-xu-2025-looped-transformers-timestep-encoding]。

**维度符号**：$d$ 为 token 维；$N$ 为序列长度（固定，不进 rate 的自由度）；$r$ 为循环次数，定理要求 $r>N$；$M$ 为权重绝对值上界；$\delta$ 是证明里 cube 的边长，由 $r$ 反解，不是超参。参数量的口径是论文自己强调的一条：参数数固定为 $O(d)$，与目标精度和序列长度都无关，误差靠 $r$ 降（§3.3）[^src-xu-2025-looped-transformers-timestep-encoding]。

## 三种 modulus of continuity

只有整体的 modulus 不足以确定 rate，论文按扰动的施加方式拆成三个（§3.2）[^src-xu-2025-looped-transformers-timestep-encoding]。输入侧一律用 $\ell_2$（把 $d\times N$ 矩阵展平），输出侧用 entry-wise $\ell_p$：

**整条序列**（Definition 3.3）：扰动同时落在所有 token 上，

$$
\omega_f(\delta):=\sup\left\{\left\|f(X)-f(X')\right\|_p:\left\|X-X'\right\|_2\le\delta\right\}.
$$

**contextual 连续性**（Definition 3.4）：第 $n$ 个 token 原样不动，只让其余 token 变，

$$
\omega_f^{\mathrm{cont}}(\delta):=\sup_{n,X,X'}\left\{\left\|f(X)_{:,n}-f(X')_{:,n}\right\|_p:\left\|X-X'\right\|_2\le\delta,\ X_{:,n}=X'_{:,n}\right\}.
$$

**token 连续性**（Definition 3.5）：context 原样不动，只让第 $n$ 个 token 变，

$$
\omega_f^{\mathrm{tok}}(\delta):=\sup_{n,X,X'}\left\{\left\|f(X)_{:,n}-f(X')_{:,n}\right\|_p:\left\|X_{:,n}-X'_{:,n}\right\|_2\le\delta,\ X_{:,q}=X'_{:,q}\ (\forall q\ne n)\right\}.
$$

论文用两句最小对照说明为什么要分开（§3.4）：`I write papers.` 与 `I draft papers.` 只换一个词、context 不变，量的是 token 连续性；`I write papers.` 与 `You write books.` 里 `write` 落在不同上下文，量的是 contextual 连续性。作者的说法是，Section 3.2 里任何一个 modulus 单独都界定不了前一例中 `write` 与 `papers` 的差，而后一例中 `write` 的误差可以由 contextual 连续性界定[^src-xu-2025-looped-transformers-timestep-encoding]。

## 主定理

**Theorem 3.6**：对 $f\in\mathcal{F}_{\mathrm{PE}}([0,1]^{d\times N})$ 与 $r>N$，存在上述规格的 looped Transformer 使

$$
\begin{aligned}
\left\|L_2\circ\mathrm{TF}^{\circ r}\circ L_1-f\right\|_{L_p}
&\le (Nd)^{1/p}\left[\omega_f^{\mathrm{tok}}\!\left(\delta\sqrt{d}\right)+\omega_f^{\mathrm{cont}}\!\left(\delta\sqrt{Nd}\right)\right] && \text{// 两份受限 modulus 各自乘 } (Nd)^{1/p}\text{：单 token 半径 } \delta\sqrt d\text{、整序列半径 } \delta\sqrt{Nd}\\
&\quad+\omega_f\!\left(\delta\sqrt{Nd}\right)+O\!\left(N^{2/p}\delta^{d/p}\right)+O\!\left((M\delta)^{-1}dN\right)^{1/p} && \text{// 整体 modulus + 同 token 重复项 + 权重范数项}\\
\delta&=\left(\frac{r-N}{2}\right)^{-1/((N+1)d+1)} && \text{// 由 loop 预算反解，分母的 } +1 \text{ 来自式 16 的放缩}
\end{aligned}
$$

逐项来源（按 §3.4 与式 9 之后的原文归属）[^src-xu-2025-looped-transformers-timestep-encoding]：

- $(Nd)^{1/p}\omega_f^{\mathrm{tok}}(\delta\sqrt d)$：token 维的量化与 weight-tied FF 映射误差。
- $(Nd)^{1/p}\omega_f^{\mathrm{cont}}(\delta\sqrt{Nd})$：sequence ID 的分辨率，即 context 维误差。
- $\omega_f(\delta\sqrt{Nd})$：cube 内部的 piecewise-constant 逼近误差（式 10）。
- $O(N^{2/p}\delta^{d/p})$：序列中出现相同 token 时的情形。
- $O((M\delta)^{-1}dN)^{1/p}$：权重范数受限带来的误差。

定理同时给出 bit complexity $O(\delta^{-(N+1)d})$；$M$ 是模型参数绝对值的上界，不是比特数[^src-xu-2025-looped-transformers-timestep-encoding]。

**Corollary 3.7（universality）**：$\mathcal{H}$ 在 $\mathcal{F}_{\mathrm{PE}}([0,1]^{d\times N})$ 中按 $L_p$ 范数稠密。论文明确这一步需要无限精度权重；有限精度下被权重范数项挡住[^src-xu-2025-looped-transformers-timestep-encoding]。作者还强调 attention 用 hard-max 也成立：weight-tied 自注意力仍能算 contextual mapping 并成为通用逼近器（§3.3 的两条总结）[^src-xu-2025-looped-transformers-timestep-encoding]。

## 证明结构：四段 loop 预算

策略是用分段常值函数 $\bar f$ 逼近 $f$、再用网络 $\tilde f$ 实现 $\bar f$：把 $[0,1]^{d\times N}$ 切成边长 $\delta$ 的 cube，每个 cube 取代表点（式 1）[^src-xu-2025-looped-transformers-timestep-encoding]。下面按前向拓扑拆阶段，右侧注记同时给维度走向与消耗的 loop 数。

### Stage 1 · 抬宽并逐 token 量化

$$
\begin{aligned}
L_1:\ \mathbb{R}^{d}\to\mathbb{R}^{17d+9} && \text{// } [d,N]\xrightarrow{\text{逐 token 仿射}}[17d+9,N]\ \text{——把每个 token 抬到构造用的工作宽度}\\
X_{:,n}\in[0,1]^d\mapsto z\in\{0,1,\dots,\delta^{-d}-1\} && \text{// } [17d+9,N]\to[1,N]\ \text{——每个 token 换成一个整数 ID，cube 编号即 ID（式 2）；消耗 } \delta^{-1}-1 \text{ 次 loop}
\end{aligned}
$$

### Stage 2 · 由 token ID 造 sequence ID

$$
\begin{aligned}
u &= \left(\delta^{-d(N-1)},\dots,\delta^{-d},1\right) && \text{// } [1,N]\ \text{——位置权向量，与 } N \text{ 个 token ID 做内积}\\
u^{\top}z &\ne u^{\top}z'\ \text{且}\ \left|u^{\top}z-u^{\top}z'\right|>1,\quad z\ne z' && \text{// } [1,N]\times[N,1]\to[1]\ \text{——不同的有序 token ID 组合必得不同 sequence ID（式 3）；消耗 } N \text{ 次 loop}
\end{aligned}
$$

关键点是 $u$ 只需 $\delta^{-d}$ 就能表示，因此这套构造能在单个 block 的循环里做完；token ID 与 sequence ID 合起来就是 contextual token ID（Definition 3.2、Fig. 1）[^src-xu-2025-looped-transformers-timestep-encoding]。

### Stage 3 · 把 contextual token ID 映到目标嵌入

$$
\begin{aligned}
k=(\text{token ID},\text{sequence ID}) &\mapsto \tilde y_k\in\mathbb{R}^{d} && \text{// } [1,N]\to[d,N]\ \text{——逐点查表式映射，式 4 把误差上界压到相邻嵌入差的最大值；消耗 } 2\delta^{-(N+1)d}-1 \text{ 次 loop}
\end{aligned}
$$

论文在这一步给了两种计数口径：Step 3 的正文写「用 $K-1$ 次 loop 依次把 $k=0,\dots,K-1$ 映到 $\tilde y_k$」，loop 预算处写 $2\delta^{-(N+1)d}-1$；两处并列，论文没有显式写出 $K$ 与 $\delta$ 的关系（§3.4）[^src-xu-2025-looped-transformers-timestep-encoding]。

### Stage 4 · 三步串成一次前向

$$
\begin{aligned}
r &= \delta^{-1}+2\delta^{-(N+1)d}+N && \text{// 标量——式 14 的四段之和：}(\delta^{-1}-1)+N+(2\delta^{-(N+1)d}-1)+2\\
L_2:\ \mathbb{R}^{17d+9}\to\mathbb{R}^{d} && \text{// } [17d+9,N]\to[d,N]\ \text{——回到输出形状；三个子网络压进同一个 TF block}
\end{aligned}
$$

中间多出的 $2$ 是连接各步的 loop（原文「1 loop required to connect each step」）[^src-xu-2025-looped-transformers-timestep-encoding]。与 Zhang et al. (2023) 需要额外一层不同，论文称单个 Transformer block 就够，细节在附录 A[^src-xu-2025-looped-transformers-timestep-encoding]。

### Stage 5 · 反解 cube 宽度

$$
\begin{aligned}
\delta^{-1}+2\delta^{-(N+1)d} &= r-N && \text{// 式 15：把 Stage 4 的预算移项}\\
\delta &\le\left(\frac{r-N}{2}\right)^{-1/((N+1)d+1)} && \text{// 式 16–17：}\delta\ge2\text{ 下的放缩，分母 } +1 \text{ 的来源}
\end{aligned}
$$

> [!warning] 式 16 按字面无法复现
> PDF 把这一步排成 $\delta^{-1}\cdot2\delta^{-(N+1)d}\ge r-N$，而式 15 的左边是同一对因子的**和**。$\delta\ge2$ 只保证 $\delta^{-1}\le\tfrac12$，不足以说明把和换成积后不等号方向不变；论文未给这一步的依据，按字面它与式 15 的等式冲突。
> 下游推导（代入式 9/10/13 得到 Theorem 3.6）用的是式 17 的 $\delta\le\left((r-N)/2\right)^{-1/((N+1)d+1)}$，本页只引用式 17，把式 16 当过渡行处理[^src-xu-2025-looped-transformers-timestep-encoding]。
> 本页自查（非论文内容）：取 $N=d=1$，只代入论文允许的网格 $\delta^{-1}\in\mathbb{N},\ \delta^{-1}\ge2$。$\delta=0.5$（$\delta^{-1}=2$）时式 14 给出 $r=2+8+1=11$，式 17 右端为 $(10/2)^{-1/3}=0.585$；$\delta=0.1$（$\delta^{-1}=10$）时 $r=10+200+1=211$，右端为 $(210/2)^{-1/3}=0.212$。两例都满足 $\delta\le$ 右端，「$\delta$ 被 $r$ 上界」的用法方向正确。这只是核对引用方向，不构成对式 16 的补证。

把式 17 代回 $\int\|\tilde f-\bar f\|_p$（式 13）与 cube 内误差（式 10），经三角不等式（式 9）即得 Theorem 3.6[^src-xu-2025-looped-transformers-timestep-encoding]。

## 多出来的两项从哪来

标准 Transformer 的 rate 界里没有 $\omega^{\mathrm{tok}}$ 与 $\omega^{\mathrm{cont}}$；它们的出现是 weight-tying 的直接后果。机制在 Lemma 4.1：给定 $y_k\in\mathbb{R}^d$、$k=0,\dots,K-1$，且每步差有界 $\left|(y_k-y_{k-1})_i\right|\le\varepsilon_i$，则存在宽度 $18d$ 的 FF 层与两个仿射映射，使

$$
\left|\left(L_2\circ(\mathrm{id}+\mathrm{FF})^{\circ(K-1)}\circ L_1(k)-y_k\right)_i\right|\le\varepsilon_i .
$$

同一个 $\mathrm{FF}$ 被迭代 $K-1$ 次，各步增量由同一组权重决定，无法让某一步走大步、另一大步走小步；误差只能被最坏的相邻 gap 卡住（式 4）[^src-xu-2025-looped-transformers-timestep-encoding]。相比之下，参数独立的 Transformer 可以用不同层做逐点精确记忆（论文举 Kim et al., 2023 用 $N$ 层存 $N$ 个参数），looped 结构只有一个固定参数集[^src-xu-2025-looped-transformers-timestep-encoding]。

从公式直接读出的两条推论（论文未展开，本页标注为推断）[INFERENCE]：

- 除权重范数项外，其余各项在 $f$ 连续时随 $\delta\to0$ 趋于 $0$；$O((M\delta)^{-1}dN)^{1/p}$ 在 $M$ 固定时按 $\delta^{-1/p}$ 反向增长。要让这一项也收缩，$M$ 必须随 $r$ 增大，而 bit complexity 是 $O(\delta^{-(N+1)d})$，即以 $r$ 的近线性增长换精度。
- 因此这条 rate 的读法是「精度用权重幅度与数值精度买」，不是免费获得。

## 边界

- 单层、固定长度 $N$、$r>N$、定义域 $[0,1]^{d\times N}$；不涉及变长输入，论文没有给出长度泛化的理论主张[^src-xu-2025-looped-transformers-timestep-encoding]。
- 分析排除 LayerNorm；构造依赖 softmax 逼近 hardmax，与 Yun et al. (2020)、Kim et al. (2023) 同一路线（§2）[^src-xu-2025-looped-transformers-timestep-encoding]。
- 结论是逼近率（approximation），不是估计误差或训练稳定性；论文把后两者列为后续挑战（§6）[^src-xu-2025-looped-transformers-timestep-encoding]。
- 多层情形自述为 open question：作者认为多层可能改进精度，但改进体现在常数而非渐近阶，难以理论刻画；若允许随精度取对数层数，现有构造也可能绕过该限制，但那已偏离「只用 loop 数刻画 rate」的目标（附录 D）[^src-xu-2025-looped-transformers-timestep-encoding]。

## 口径备注

Theorem 3.6 的 PDF 排版里 $(Nd)^{1/p}$ 后没有可见括号界定作用范围。由式 13（$(Nd)^{1/p}\left[\omega^{\mathrm{tok}}+\omega^{\mathrm{cont}}\right]$）与式 10（$\omega_f$ 项不带该因子）推回，本页按「$(Nd)^{1/p}$ 乘两项 modulus、$\omega_f$ 独立成项」转写[^src-xu-2025-looped-transformers-timestep-encoding]。式 16 的排版与式 15 冲突，处理方式见 Stage 5 的提示框：只引用式 17。

[^src-xu-2025-looped-transformers-timestep-encoding]: [[source-xu-2025-looped-transformers-timestep-encoding]]
