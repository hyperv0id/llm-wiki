---
title: "Pre-LN vs Post-LN：层归一化放置与梯度尺度"
type: technique
tags:
  - transformer
  - layer-normalization
  - optimization-stability
  - initialization
  - gradient-flow
created: 2026-10-26
last_updated: 2026-10-26
source_count: 1
confidence: medium
status: active
---

# Pre-LN vs Post-LN：层归一化放置与梯度尺度

Pre-LN 与 Post-LN 仅在层归一化置于残差相加前还是后有所不同。这一顺序决定了初始化梯度尺度是否随深度失衡，以及训练是否需要预热（warmup）。Post-LN 下靠近输出的层梯度量级为 $\mathcal{O}(d\sqrt{\ln d})$，与层数无关；Pre-LN 下同一梯度被出口归一化除以 $\sqrt{L}$[^src-xiong-2020-layernorm-placement]。

## 两种放置

Post-LN 采用原版 Transformer 与 BERT 的顺序：子层 → 加残差 → LN。Pre-LN 则将 LN 移至残差连接内部、所有非线性变换之前：LN → 子层 → 加残差[^src-xiong-2020-layernorm-placement]。

Pre-LN 包含**出口归一化**：最后一层输出 $x^{pre}_{L+1}$ 先过 LN 再送入 softmax 前的投影 $W^{emb}$（Table 1 右列末行）[^src-xiong-2020-layernorm-placement]。由于 Pre-LN 残差流不经任何归一化，$\mathbb{E}\|x^{pre}_{l}\|_2^2$ 随深度线性增长，落在 $\left[(1+\frac{l}{2})d,\ (1+\frac{3l}{2})d\right]$（Lemma 2）；若缺少出口 LN，logits 尺度将随 $L$ 放大。Post-LN 无需该步骤：其每层出口已做 LN，第 $L$ 层的出口 LN 即送入 $W^{emb}$ 前的最后一次[^src-xiong-2020-layernorm-placement]。

## 数据流与维度变换

**维度符号**：$n$ 序列长度（token 数，编解码实验取该侧长度）；$d$ 隐层维；$d_{ff}$ FFN 中间维；$H$ 头数，单头宽 $d_K=d_V=d/H$；$L$ 层数。序列级张量记作 $[n,d]$；推导梯度时将整条序列展平为 $[nd]$，层间雅可比为 $[nd,nd]$。LN 沿 $d$ 维逐 token 归一化。

### Stage 1 · 取子层一的输入：Post-LN 直取原流，Pre-LN 先归一化

两支路的注意力计算同形，输入不同。

$$\begin{aligned}
U^{post}_1 &= X^{post}_l && \text{// } [n,d]\to[n,d]\ \text{——原样进入子层，残差流本身不被归一化}\\
U^{pre}_1 &= \mathrm{LN}(X^{pre}_l) && \text{// } [n,d]\xrightarrow{\text{沿 } d \text{ 维逐 token 归一化}}[n,d]\ \text{——每个 token 的 } d \text{ 个通道压到零均值单位方差，尺度与深度脱钩}
\end{aligned}$$

### Stage 2 · 子层一：多头自注意力

$$\begin{aligned}
S^h &= \frac{(U_1W^Q_h)(U_1W^K_h)^{\top}}{\sqrt{d_K}} && \text{// } [n,\bcancel{d_K}]\times[\bcancel{d_K},n]\to[n,n]\ \text{——消去 } d_K\text{，得到 } n\times n \text{ 的 token 间打分}\\
P^h &= \operatorname{softmax}_{\text{key}}\!\left(S^h\right) && \text{// } [n,n]\xrightarrow{\text{沿 key 维 softmax}}[n,n]\ \text{——每个 query token 对全部 } n \text{ 个 key 的权重和归一为 } 1\\
O^h &= P^h\,(U_1W^V_h) && \text{// } [n,\bcancel{n}]\times[\bcancel{n},d_V]\to[n,d_V]\ \text{——消去 key 维，按权重对 value 加权求和}\\
A_1 &= \operatorname{Concat}(O^1,\dots,O^H)\,W^O && \text{// } [n,\bcancel{d}]\times[\bcancel{d},d]\to[n,d]\ \text{——} H \text{ 个头拼成 } [n,Hd_V]=[n,d]\text{，再投影回 } d
\end{aligned}$$

### Stage 3 · 残差相加与归一化的次序（两支路在此分叉）

$$\begin{aligned}
X^{post,3}_l &= \mathrm{LN}\!\left(X^{post}_l + A_1\right) && \text{// } [n,d]\xrightarrow{\text{残差相加后沿 } d \text{ 维逐 token 归一化}}[n,d]\ \text{——LN 在残差和之外，恒等路径每层被重设为 } \sqrt{d} \text{ 量级}\\
X^{pre,3}_l &= X^{pre}_l + A_1 && \text{// } [n,d]+[n,d]\to[n,d]\ \text{——LN 已在 } A_1 \text{ 的输入侧，残差和原样下传，幅值逐层累积}
\end{aligned}$$

### Stage 4 · 子层二的输入：Pre-LN 在这里补第二次归一化

$$\begin{aligned}
U^{post}_2 &= X^{post,3}_l && \text{// } [n,d]\to[n,d]\ \text{——Post-LN 的 FFN 直接吃上一步已归一化的输出}\\
U^{pre}_2 &= \mathrm{LN}(X^{pre,3}_l) && \text{// } [n,d]\xrightarrow{\text{沿 } d \text{ 维逐 token}}[n,d]\ \text{——Pre-LN 的第二次 LN，插在 FFN 之前}
\end{aligned}$$

### Stage 5 · 子层二：逐位置 FFN，以及残差与出口

$$\begin{aligned}
H &= \operatorname{ReLU}\!\left(U_2W^1+b^1\right) && \text{// } [n,\bcancel{d}]\times[\bcancel{d},d_{ff}]\to[n,d_{ff}]\ \text{——消去 } d\text{，逐位置升维}\\
F &= HW^2+b^2 && \text{// } [n,\bcancel{d_{ff}}]\times[\bcancel{d_{ff}},d]\to[n,d]\ \text{——消去 } d_{ff}\text{，回到 } d \text{ 维}\\
X^{post}_{l+1} &= \mathrm{LN}\!\left(X^{post,3}_l + F\right) && \text{// } [n,d]\xrightarrow{\text{残差相加后归一化}}[n,d]\ \text{——每层出口幅值恒为 } \sqrt{d}\text{ 量级}\\
X^{pre}_{l+1} &= X^{pre,3}_l + F && \text{// } [n,d]+[n,d]\to[n,d]\ \text{——纯加法，没有归一化}
\end{aligned}$$

### Stage 6 · 出口归一化：Pre-LN 独有的一步

$$\begin{aligned}
X^{pre}_{Final} &= \mathrm{LN}(X^{pre}_{L+1}) && \text{// } [n,d]\xrightarrow{\text{沿 } d \text{ 维逐 token}}[n,d]\ \text{——把随 } L \text{ 线性增长的残差流压回 } \sqrt{d}\text{，logits 尺度与深度解耦}\\
\ell_i &= -\log \operatorname{softmax}_{y_i}\!\left(W^{emb}x^{pre}_{Final,i}\right) && \text{// } [1,\bcancel{d}]\times[\bcancel{d},|V|]\to[1,|V|]\ \text{——消去 } d\text{，在词表上取出正确 token 的负对数概率}
\end{aligned}$$

## 梯度：层间雅可比连乘

将第 $l$ 层输入展平为 $x_l\in\mathbb{R}^{nd}$。Post-LN 下第 $l$ 层 FFN 第二层权重的梯度须依次穿过 $l+1$ 到 $L$ 各层的雅可比 [^src-xiong-2020-layernorm-placement]：

$$\begin{aligned}
\frac{\partial\tilde{\mathcal{L}}}{\partial W^{2,l}} &= \frac{\partial\tilde{\mathcal{L}}}{\partial x_{L+1}}\left(\prod_{j=l+1}^{L}\frac{\partial x_{j+1}}{\partial x_j}\right)\frac{\partial x_{l+1}}{\partial W^{2,l}} && \text{// } [1,\bcancel{nd}]\cdot\prod_{j}\left[\bcancel{nd},nd\right]\cdot\left[\bcancel{nd},d^2\right]\to[1,d^2]\ \text{——每个因子消去上一层传来的 } nd\text{，最后得到 } d^2 \text{ 个标量}
\end{aligned}$$

单层雅可比拆成四项，两个 LN 雅可比留在连乘里（附录 F）：
$$\begin{aligned}
\frac{\partial x_{j+1}}{\partial x_j} &= \underbrace{\mathrm{diag}\big(\mathbf{J}_{LN}(x^{post,5}_{j,i})\big)}_{\text{出口 LN}}\Big(I+\mathrm{diag}(W^{2,j})\,\mathrm{diag}(\mathbf{J}^{\sigma}_{j})\,\mathrm{diag}(W^{1,j})\Big)\underbrace{\mathrm{diag}\big(\mathbf{J}_{LN}(x^{post,2}_{j,i})\big)}_{\text{子层一后的 LN}}\Big(I+\tfrac{1}{n}\mathbf{1}\mathbf{1}^{\top}\!\otimes W^{V,j}\Big) && \text{// } [nd,\bcancel{nd}]\times[\bcancel{nd},nd]\ \text{——四项都是 } nd\times nd\text{，注意力退化为均匀加权}
\end{aligned}$$

$\mathbf{J}_{LN}$ 的尺度由输入幅值决定，Post-LN 固定每层残差和的幅值，使每层收缩同一常数：

$$\begin{aligned}
\|\mathbf{J}_{LN}(x)\|_2 &= \mathcal{O}\!\left(\frac{\sqrt{d}}{\|x\|_2}\right) && \text{// } [1,\bcancel{d}]\times[\bcancel{d},d]\to[1,d]\ \text{——} \mathbf{J}_{LN} \text{ 把 } d \text{ 维 token 映射回 } d \text{ 维，谱范数只由 } \|x\|_2 \text{ 决定}\\
\mathbb{E}\|x^{post,5}_{j,i}\|_2^2 &= \tfrac{3}{2}d\ \Longrightarrow\ \|\mathbf{J}_{LN}(x^{post,5}_{j,i})\|_2\approx\sqrt{\tfrac{2}{3}} && \text{// } \text{幅值恒为 } \sqrt{d} \text{ 量级，与 } j \text{ 无关，故每层因子小于 } 1\\
\prod_{j=l+1}^{L}\left\|\frac{\partial x_{j+1}}{\partial x_j}\right\|_2 &\approx \left(\tfrac{2}{3}\right)^{(L-l)/2} && \text{// } L-l \text{ 个因子连乘}\ \text{——越往浅层指数衰减}
\end{aligned}$$

Pre-LN 的单层雅可比仍是「恒等 + 扰动」，LN 雅可比移入括号内，成为扰动项的一部分：

$$\begin{aligned}
\frac{\partial x_{j+1}}{\partial x_j} &= \Big(I+\mathrm{diag}(W^{2,j})\,\mathrm{diag}(\mathbf{J}^{\sigma}_{j})\,\mathrm{diag}(W^{1,j})\,\mathrm{diag}\big(\mathbf{J}_{LN}(x^{pre,3}_{j,i})\big)\Big)\Big(I+\tfrac{1}{n}\mathbf{1}\mathbf{1}^{\top}\!\otimes W^{V,j}\,\mathrm{diag}\big(\mathbf{J}_{LN}(x^{pre}_{j,i})\big)\Big) && \text{// } [nd,\bcancel{nd}]\times[\bcancel{nd},nd]\ \text{——恒等项不再被 LN 雅可比乘，连乘不收缩}
\end{aligned}$$

$\|x^{pre}_{j,i}\|_2^2\approx(1+\frac{3j}{2})d$ 使 $\|\mathbf{J}_{LN}(x^{pre}_{j,i})\|_2=\mathcal{O}(1/\sqrt{j})$，扰动随层号递减，$j$ 较大时两因子均趋近恒等，连乘整体近似恒等；深度带来的 $1/\sqrt{L}$ 仅在出口[^src-xiong-2020-layernorm-placement]出现一次：

$$\begin{aligned}
\frac{\partial x_{Final}}{\partial x_{L+1}} &= \mathrm{diag}\big(\mathbf{J}_{LN}(x^{pre}_{L+1,i})\big) && \text{// } [1,\bcancel{nd}]\times[\bcancel{nd},nd]\to[1,nd]\ \text{——整条梯度只经过这一次归一化}\\
\left\|\frac{\partial x_{Final}}{\partial x_{L+1}}\right\|_2 &\approx \sqrt{\frac{d}{\|x^{pre}_{L+1}\|_2^2}} = \mathcal{O}\!\left(\frac{1}{\sqrt{L}}\right) && \text{// } \|x^{pre}_{L+1}\|_2^2\in\left[(1+\tfrac{L}{2})d,\ (1+\tfrac{3L}{2})d\right]\ \text{——整条梯度被除以 } \sqrt{L}
\end{aligned}$$

两条路径的梯度式均为连乘形式，差别在于各因子的谱是否偏离 $1$。Post-LN 将 $L-l$ 个 $\sqrt{d/\|x\|_2^2}$ 量级的因子连乘：在 Xavier 初始化与 $\gamma=1$ 下 $\|x^{post,5}\|_2^2=\frac{3}{2}d$，因子为 $\sqrt{2/3}<1$，浅层梯度指数缩小，输出侧不收缩且保持 $\mathcal{O}(d\sqrt{\ln d})$ 的大值。若初始化尺度使 $\sqrt{d/\|x\|_2^2}$ 大于 $1$，连乘即转为放大 [INFERENCE]。Pre-LN 的连乘因子近似恒等，全模型共享出口处的 $1/\sqrt{L}$，这是浅层与深层梯度同阶的来源。

## 定理 1：条件与结论

结论仅覆盖最后一层 FFN 的第二层权重 $W^{2,L}$[^src-xiong-2020-layernorm-placement]：

$$\begin{aligned}
\text{Post-LN:}\quad & \left\|\frac{\partial\tilde{\mathcal{L}}}{\partial W^{2,L}}\right\|_F \leq \mathcal{O}\!\left(d\sqrt{\ln d}\right) && \text{// } d^2 \text{ 个元素各带 } \mathcal{O}(\ln d) \text{ 的界}\ \text{——与 } L \text{ 无关}\\
\text{Pre-LN:}\quad & \left\|\frac{\partial\tilde{\mathcal{L}}}{\partial W^{2,L}}\right\|_F \leq \mathcal{O}\!\left(d\sqrt{\frac{\ln d}{L}}\right) && \text{// } \text{同一组元素再乘出口 LN 的 } 1/\sqrt{L}\ \text{——随深度下降}
\end{aligned}$$

前提逐条列出：

注意力线性化设为单头，$W^{Q,l},W^{K,l},W^{V,l},W^{1,l},W^{2,l}$ 全为 $d\times d$，$W^{Q,l}=W^{K,l}=\mathbf{0}$ 使注意力在初始化时均匀，$\mathrm{MultiHeadAtt}$ 退化为 $\frac{1}{n}\sum_{j}x_{l,j}W^{V,l}$。初始化采用 Xavier，$W_{uv}\sim N\!\left(0,\frac{2}{n_{in}+n_{out}}\right)$，bias 为零，$\gamma=1$、$\beta=0$，输入向量同分布高斯。假设所有导数有界，由此导出 $\|\partial\mathcal{L}/\partial x^{post}_{L+1,i}\|_2^2=\mathcal{O}(1)$；$\|x^{post,5}_{L,i}\|_2^2$ 与 $\|x^{pre}_{L+1,i}\|_2^2$ 为 $(\epsilon,\delta)$-bounded，即偏离期望超过 $\epsilon$ 的比例不超过 $\delta$。结论成立概率至少为 $0.99-\delta(\epsilon)-\frac{\epsilon}{1+\epsilon-\alpha_0}$，其中 $\delta=\delta(\epsilon)$ 来自 $\|x\|_2^2$ 的 $(\epsilon,\delta)$-bounded，$\alpha_0=0.1$ 对应定理正文所写 $\frac{\epsilon}{0.9+\epsilon}$；$0.99$ 来自逐元素 Chernoff 界 $\mathrm{ReLU}([xW^{1,L}]_p)^2\leq 2\ln(100d)$ 对全部 $p\in\{1,\dots,d\}$ 同时成立。

$\mathcal{O}(d\sqrt{\ln d})$ 由 $d^2$ 个元素乘单元素界 $\mathcal{O}(\ln d/\alpha_0)$ 再开方构成；Pre-LN 只把 $\|x\|_2^2$ 换成随 $L$ 增长的量，其余步骤同形。

论文未给出学习率上限表达式。定理给出梯度尺度，而「尺度大 + 学习率大 → 发散」由两项实验支撑：warmup 后的 Post-LN 梯度变小，可用大学习率继续训练；固定 $10^{-4}$ 的小步长虽能使 Post-LN 正常训练，但收敛明显变慢（附录 H、Figure 6）[^src-xiong-2020-layernorm-placement]。

## 实验证据

**IWSLT14 De-En**：6 层编码器 + 6 层解码器，$d=512$，FFN 隐层 1024，4 头，dropout 0.1，label smoothing 0.1，batch 4096 token，Adam $\beta=(0.9,0.98)$（附录 A.1）[^src-xiong-2020-layernorm-placement]。

| 设置 | 对照 | 结果 | 出处 |
|---|---|---|---|
| Post-LN，Adam，无 warmup | 有 warmup（$T=4000$） | BLEU 8.45 vs 约 34 | §3.2 正文（未点名 lr）；Figure 2a |
| Post-LN，$T_{\mathrm{warmup}}=500$，$\mathrm{lr}_{\max}=5\times10^{-4}$ | 同上 | BLEU 31.16 | §3.2 正文；Figure 2a |
| Post-LN，$T_{\mathrm{warmup}}=500$，$\mathrm{lr}_{\max}=10^{-3}$ | 同上 | BLEU 2.77 | §3.2 正文；Figure 2a |
| Post-LN，SGD，$\mathrm{lr}_{\max}=5\times10^{-3}/10^{-3}$，无 warmup | 有 warmup | 15 个 epoch 内 BLEU 仅略高于 0 | §3.2 正文；Figure 2b |
| Pre-LN，无 warmup，$\mathrm{lr}=5\times10^{-4}$，第 8 epoch 衰减 0.1 | Post-LN 带 warmup | BLEU 约 34、验证损失约 4；第 9 个 checkpoint 追平 Post-LN 第 15 个 | §4.2 正文；Figure 4a–b |
| Pre-LN，无 warmup，$\mathrm{lr}=7.5\times10^{-4}$，8000 步后衰减，dropout 0.3，batch 8192 | — | 35.5 test BLEU | 脚注 1 |
| Post-LN，固定 $\mathrm{lr}=10^{-4}$，无 warmup | 标准 warmup 流程 | 可训练但收敛显著变慢 | 附录 H、Figure 6 |

**WMT14 En-De**（base 设置：6+6 层、$d=512$、8 头）：Pre-LN 不用 warmup，设初始 $\mathrm{lr}=7\times10^{-4}$ 与 $1.5\times10^{-3}$ 两组，第 6 epoch 起按平方根倒数衰减；两组配置均作为正式结果报告，终值仅见于 Figure 4c–d 曲线[^src-xiong-2020-layernorm-placement]。

**BERT base 预训练**（12 层、$d=768$、12 头，Wikipedia + BookCorpus 约 3.4B 词）：Post-LN 按原配置采用 10k 步 warmup 与 $\mathrm{lr}_{\max}=10^{-4}$；Pre-LN 省去 warmup，从 $3\times10^{-4}$ 线性衰减。正文记 Post-LN 在 500k 步达到 1.69 验证损失、Pre-LN 在 700k 步达到同等水平，同句称其为 40% 加速，同段记二者步数差为 200k，但步数与「加速」方向相反（Figure 5a）[^src-xiong-2020-layernorm-placement]。Post-LN 换用 $3\times10^{-4}$ 时优化发散，基线因此仍取 $10^{-4}$（§4.1）[^src-xiong-2020-layernorm-placement]。

**梯度尺度的直接测量**（§3.4、Figure 3）：6-6 / 8-8 / 10-10 / 12-12 / 14-14 五个规模上，Post-LN 最后一层 $W^{2}$ 的梯度范数随规模基本不变，Pre-LN 随规模下降；6-6 模型逐层统计中，Post-LN 的期望梯度范数随层号递增，Pre-LN 各层基本持平。隐藏态范数实测符合 $(0.1,0.125)$-bounded。每组实验用 10 个随机种子重复[^src-xiong-2020-layernorm-placement]。

## 边界

定理 1 给出 $\mathcal{O}(d\sqrt{\ln d})$ 与 $\mathcal{O}(d\sqrt{\ln d/L})$ 两个界及其成立概率。浅层按 $(2/3)^{(L-l)/2}$ 衰减与 Pre-LN 各层梯度同阶，出自附录 F 对单层雅可比范数的近似估计（$\|\mathbf{J}_{LN}\|_2\approx\sqrt{2/3}$、因子趋近恒等），无定理保证。warmup 与稳定性的因果、Pre-LN 收敛更快以及两变体的终值对比，均来自上述四个任务的曲线。

定理设定的 $W^{Q}=W^{K}=0$ 均匀注意力、单头、$d\times d$ 参数、Xavier 与 $\gamma=1$ 在真实训练中均不成立，且定理仅刻画初始化瞬间，未涉及训练中注意力与权重的演化。$(\epsilon,\delta)$-bounded 的实测仅在 6-6 模型上记录了 $(\epsilon,\delta)=(0.1,0.125)$，未跨规模验证。

学习率缺乏解析结论：论文仅给出梯度尺度，$\mathrm{lr}$ 多大算「大」由实验界定（IWSLT 上 $\mathrm{lr}_{\max}=10^{-3}$ 无 warmup 只得 BLEU 8.45、$T_{\mathrm{warmup}}$ 压到 500 时掉到 2.77；BERT 上 Post-LN 用 $3\times10^{-4}$ 时优化发散），不存在由 $d$、$L$、初始化方差推出的 $\mathrm{lr}$ 上限。

终值口径：IWSLT 上两者 comparable，未给出 Pre-LN 更低的结论；脚注 1 的 35.5 test BLEU 采用另一套配置（dropout 0.3、attention dropout 0.1、batch 8192、8000 步后衰减），与表中 $5\times10^{-4}$ 的约 34 不同口径。WMT 与 BERT 的终值只以曲线呈现，正文不提供数值表。

放置位置更细致的架构后果——Post-LN 每层出口终止恒等路径、Pre-LN 保留纯加法残差高速公路——见 [[identity-mapping-property]]；Pre-LN 作为独立架构的沿革见 [[pre-ln-transformer]]。

## 相关页面

[[pre-ln-transformer|Pre-LN Transformer]]、[[identity-mapping-property|残差网络的恒等映射性质]]、[[source-xiong-2020-layernorm-placement|源摘要]]。

[^src-xiong-2020-layernorm-placement]: [[source-xiong-2020-layernorm-placement]]
