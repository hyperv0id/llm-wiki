---
title: "Position Interpolation (PI)"
type: technique
tags:
  - position-encoding
  - rotary-position-embedding
  - context-extension
  - interpolation
created: 2026-10-26
last_updated: 2026-10-26
source_count: 2
confidence: medium
status: active
---

# Position Interpolation (PI)

PI 将 RoPE 模型的位置下标除以 $s = L'/L$ 压回预训练区间 $[0, L)$，再在扩展窗口上短程微调；模型结构、权重集合与注意力计算都不改，唯一变动是位置下标进入旋转相位的写法[^src-pi]。2048 窗口的 LLaMA 7B/13B 经 Pile 上 1000 步全参数微调扩到 32768，33B 扩至 16384，65B 扩至 8192，passkey 检索的有效窗口在 200 步后即达到目标长度[^src-pi]。

## 问题：外推为什么失效

RoPE 下 query 与 key 的打分是 $d/2$ 个三角基的加权和：

$$a(s) = \mathrm{Re}\left[\sum_{j=0}^{d/2-1} h_j e^{\mathrm{i} s \theta_j}\right], \quad \theta_j = 10000^{-2j/d}, \quad h_j = (q_{2j}+\mathrm{i}q_{2j+1})(k_{2j}-\mathrm{i}k_{2j+1})$$

$s$ 是两 token 的位置差，$h_j$ 由 query/key 决定，因此打分只依赖相对位置[^src-pi]。三角基族在 $d$ 足够大时构成通用逼近器：总存在一组 $h_j$ 让 $a(s)$ 在 $[0,L]$ 内取值很小、在区间外取值很大。LLaMA 7B 的单头维 $d=128$ 下，最小二乘拟合出的 $a(s)$ 在 $[0,2048]$ 内限于 $[-1,1]$，出区间后可达 $8000$ 以上（Fig. 2）。直接外推导致 2048 模型在 4096 以上评估窗口的 perplexity 超过 $10^3$，与未训练模型相当（Table 1、§2.2）。RoPE 原文的衰减上界虽随 $|s|$ 衰减，但界中含 $\max_j|h_j|$ 因子且可以很松，无法约束区间外的放大（§2.2）。

## 机制：位置下标的线性压缩

$$f'(x, m) = f\left(x, \frac{mL}{L'}\right) = f(x,\, m/s)$$

下标从整数 $m\in[0,L')$ 换为实数 $m/s\in[0,L)$。RoPE 的相位 $e^{\mathrm{i}m\theta_j}$ 对实数下标有定义，压缩后的下标仍为合法旋转角，是插值可行的前提[^src-pi]。等价表述是将每对逆频率变为 $\theta_j' = \theta_j/s$：$d/2$ 对共用 $s$，波长 $\lambda_j = 2\pi/\theta_j$ 按同一比例拉长。

## 数据流与维度变换

**维度符号**：$N$ 序列长度（token 数）；$d$ 单头通道维；$j\in[0,d/2)$ 二维子空间编号，共 $d/2$ 对；$\theta_j=10000^{-2j/d}$ 第 $j$ 对的逆频率；$L$ 预训练窗口、$L'$ 目标窗口、$s=L'/L$；$m$ 与 $i$ 分别是 query 与 key 的位置下标。$X\in\mathbb{R}^{d\times N}$ 每列一个 token 的单头向量，与 [[rope]] 的列向量写法一致。

### Stage 1 · 重标位置下标

$$\begin{aligned}
p_m &= \frac{mL}{L'} = \frac{m}{s} && \text{// } [N] \xrightarrow{\text{逐元素除以 } s} [N]\ \text{——整数下标变成实数坐标，取值域从 } [0,L') \text{ 收到 } [0,L)\\
\Delta_{mi} &= p_m - p_i = \frac{m-i}{s} && \text{// } [N]\times[N] \xrightarrow{\text{逐对相减}} [N,N]\ \text{——最大相对距离由 } L'-1 \text{ 降到 } (L'-1)/s
\end{aligned}$$

### Stage 2 · 每对乘上压缩后的相位

$$\begin{aligned}
f'(x,m)_j &= (x_{2j} + \mathrm{i}x_{2j+1})\,e^{\mathrm{i}p_m\theta_j} && \text{// } [d,N] \xrightarrow{\text{分 } d/2 \text{ 对，第 } j \text{ 对乘 } e^{\mathrm{i}(m/s)\theta_j}} [d,N]\ \text{——每对模长不变，只改相位}\\
&= (x_{2j} + \mathrm{i}x_{2j+1})\,e^{\mathrm{i}m\theta_j'},\quad \theta_j' = \frac{\theta_j}{s} && \text{// 同一相位的两种读法：下标除以 } s\text{，或频率除以 } s\ \text{——波长整体放大 } s \text{ 倍}
\end{aligned}$$

### Stage 3 · 打分只剩压缩后的距离

$$\begin{aligned}
a'(m,i) &= \mathrm{Re}\left[\sum_{j=0}^{d/2-1} h_j\, e^{\mathrm{i}(p_m-p_i)\theta_j}\right] && \text{// } [d/2] \xrightarrow{\text{逐对共轭内积后沿 } j \text{ 求和}} \text{标量}\ \text{——} h_j \text{ 与位置无关，只由 query/key 决定}\\
&= \mathrm{Re}\left[\sum_{j} h_j e^{\mathrm{i}(m-i)\theta_j/s}\right] = a\!\left(\frac{m-i}{s}\right) && \text{// 绝对下标 } m, i \text{ 双双抵消，只剩 } m/s-i/s\ \text{——打分只依赖压缩后的距离差}
\end{aligned}$$

### Stage 4 · 在新窗口上短程微调

$$\begin{aligned}
\mathcal{L} &= -\sum_{t=1}^{L'} \log p_\theta\!\left(x_t \mid x_{<t};\, p_1,\dots,p_t\right) && \text{// 序列展到 } L' \text{ 个 token，位置 } t \text{ 的相位取 } p_t = t/s\ \text{——梯度只更新原有权重，位置映射本身不带参数}
\end{aligned}$$

微调数据取 Pile，训练 1000 步（直接微调对照 10000 步）；7B/13B/33B 扩至 8192 的模型用 32 张 A100、global batch 64，其余配置用 128 张、128（§3.1）。

## 相对位置量变成 $(m-i)/s$ 之后

Query 在 $m$、key 在 $i$ 时，PI 下的相位差是 $(m-i)\theta_j/s$，即原模型在距离 $(m-i)/s$ 上的相位差[^src-pi]。由此得到三条性质。

一是顺序不变：$s>0$ 使 $m\mapsto m/s$ 严格单调，任意两 token 的先后次序与位置差符号都不变，$i=m$ 的对角项仍是零距离。二是距离曲线被拉长而非替换：同一真实距离的相位差缩成 $1/s$，模型在预训练中学到的「距离—打分」曲线整体放大 $s$ 倍，形状保留。三是绝对位置不进打分：相位只以差的形式出现，扩展窗口不会把模型推到没见过的绝对位置上；被压回的是下标取值域，不是注意力的相对结构。

改动只发生在位置下标进入 RoPE 的入口，$W_q$、$W_k$、MLP 与注意力计算都不动，因此既不加权重也不改架构，参数保持不变，原有训练与推理设施可直接复用[^src-pi]。

### 高频分量被同比例压缩

$d/2$ 对共用一个 $s$，是 PI 与后来按波长分档方法在频率处理上的分界。波长最短的一对 $\theta_0 = 1\ \mathrm{rad/token}$ 波长为 $2\pi\approx 6.28$ token，在 $s=16$ 下变成约 $100$ token，相邻 token 在该对的相位差从 $1$ rad 降至 $1/16$ rad，使 token 级邻接落到该对可分辨的相位差之下[INFERENCE]。预训练窗口内转不满一圈的长波长对，原本以相位角承载长程与绝对位置信息，压缩后转过角度按同比例缩小；按波长分档观点，这一档才是必须插值的部分[^src-yarn]。PI 的界对所有对统一取 $\max_j|h_j|$，不按频率区分（式 5）。按波长分档见 [[ntk-aware-interpolation]]、[[ntk-by-parts-interpolation]]，其「高频被移除」的论证出自 YaRN [^src-yarn]。PI 未与这两者对比，仅提及并行的 blogpost（kaiokendev）同样插值 RoPE 位置；PI 的增量是 65B 规模全参数微调结果与插值上界推导（§1）。

## 理论上界：插值界比外推界小约 600 倍

设 $s\in[s_1,s_2]$，$s_1,s_2$ 为预训练中表现良好的两个整数位置，$a_{\mathrm{linear}}$ 为两点的线性插值。Theorem 2.1 给出

$$|a(s) - a_{\mathrm{linear}}(s)| \le d\max_j|h_j|\,\frac{(s-s_1)(s_2-s)}{8\ln c}$$

其中 $c=10000$。由 $(s-s_1)(s_2-s)\le 1/4$，界化为 $d\max_j|h_j|/(32\ln c)\approx d\max_j|h_j|/294.73$（式 7）。推导分两步：先给出二阶导界 $|a''(s)|\le \max_j|h_j|\sum_j\theta_j^2 \le \max_j|h_j|\,d/(4\ln c)$，再代回线性插值余项（Appendix A）。

对照 RoPE 原文的外推界 $|a(s)|\le 2\max_j|h_j|B(s)$，$B(s)=\sum_{k=0}^{d/2-1}|A_{k+1}(s)|$ 数值上几乎处处不小于 $d$、多处远大于 $d$（Appendix B）。两式均含 $\max_j|h_j|$，余下系数一边为 $d/294.73$、另一边至少 $2d$，故插值界至少小 $2\times 294.73\approx 600$ 倍（LLaMA 7B 的 $d=128$、$c=10000$）。两界约束的量不同：式 5 对应相对线性插值的偏差，式 8 对应打分绝对值本身，论文比较的是系数。

## 证据

以下数值取自 LLaMA 7B/13B/33B/65B（预训练窗口 2048），Pile 上 1000 步全参数微调，7B/13B 学习率 $2\times10^{-5}$、33B/65B $10^{-5}$，AdamW、weight decay $0$，评估窗口 2048–32768（§3.1）[^src-pi]。

**语言建模（Table 1，PG19）**。扩展至 32768 的 7B/13B 随评估窗口单调下降（7.23→6.77、6.54→6.09）；扩展至 8192 的 33B/65B 最后一列回升（5.69→5.71、5.32→5.37）。未扩展的 7B 模型在 4096 以上超过 $10^3$（13B/33B/65B 未扩展行只测到 2048），直接微调 10000 步的模型反而升高。

| 模型 | 方法 | 2048 | 4096 | 8192 | 16384 | 32768 |
|---|---|---|---|---|---|---|
| 7B | 未扩展 | 7.20 | $>10^3$ | $>10^3$ | $>10^3$ | $>10^3$ |
| 7B | 直接微调，8192 | 7.21 | 7.34 | 7.69 | – | – |
| 7B | PI，32768 | 7.23 | 7.04 | 6.91 | 6.80 | 6.77 |
| 13B | PI，32768 | 6.54 | 6.40 | 6.28 | 6.18 | 6.09 |
| 33B | 直接微调，8192 | 5.88 | 5.99 | 6.21 | – | – |
| 33B | PI，8192 | 5.82 | 5.69 | 5.71 | – | – |
| 65B | PI，8192 | 5.42 | 5.32 | 5.37 | – | – |

**Proof-pile（Table 2）**。7B 的 PI 模型在 8192/16384/32768 三档目标窗口上分别给出 2.79/2.57/2.39、2.79/2.57/2.37/2.25、2.82/2.59/2.39/2.24/2.48；13B 扩到 32768 时为 2.68→2.35，33B 扩到 16384 时为 2.53→2.07。代价是 2048 窗口上 perplexity 退化 0.01–0.05（§3.2）。

**微调步数（Table 3，7B，PG19）**。扩至 8192：第 0 步 16.10，200 步 7.12，1000 步 6.95；扩至 16384：第 0 步 112.13，200 步 7.05，1000 步 6.83。200 步已低于原模型 2048 窗口的 7.20。

**Passkey 检索（Table 4）**。$k_{\max}$ 定义为对所有 $k'\le k$ 成功率不低于 20% 的最大距离，每个目标窗口取 32 个均匀间隔的 $k$，各测 10 次，passkey 为随机 5 位数（§3.3）。PI 的 7B/33B 模型在 200 步后 $k_{\max}=L'$，8192、16384、32768 各档均达标（7B 的 32768 模型 600 步时记为 18432）；直接微调在 8192 目标上从 2048 只扩到 2560，其中 1000 步为 2304，10000 步为 2560。

**原窗口基准（Table 5）**。7B 扩至 8192/16384/32768 后，BoolQ 76.1→73.2/69.8/64.7，PIQA 78.9→78.2/77.6/77.2，WinoGrande 69.6→69.0/67.8/66.9，Race-M 55.7→53.8/53.3/50.1。§3.4 称 8192 模型退化最多 2%，而 Table 5 中 7B 的 BoolQ 从 76.1 降到 73.2，差距为 2.9 点。

**长文档摘要（Table 6，GovReport）**。16384 窗口的 LLaMA-7B PI 模型 ROUGE-1/2/L 为 60.0/28.0/29.5，同表 CoLT5 Base 为 58.7/29.6/31.4，CoLT5 XL 为 61.3/32.2/33.8。

§3 称扩窗「up to 32 times of the original size」，但表中最大仅至 32768，相对 2048 为 16×；倍数按表取 16×。

## 边界

该变换仅适用 RoPE，依赖相位结构 $e^{\mathrm{i}m\theta_j}$；OPT 等可学习绝对位置嵌入模型未作实验，列为待做方向（§4）[^src-pi]。模型扩展需要微调，权重不动时 7B 扩到 8192/16384 的 PG19 perplexity 为 16.10/112.13（Table 3 第 0 步），离原模型 7.20 仍有距离，200 步才回到可用区间。原窗口内评测也有代价：Proof-pile 上 2048 窗口退化 0.01–0.05（§3.2）；基准退化随扩展倍数上升，33B 扩到 8192 时仍在 2% 以内，7B 扩到 32768 时 BoolQ 掉 11.4 点（Table 5）。此外，一个 checkpoint 绑定一个 $L'$，固定 $s$ 对短序列仍按 $L'$ 插值，按实际长度自适应见 [[dynamic-scaling]]；方法未作频率分档，所有对共用 $s$，最短波长对的 token 级分辨率被压缩（见上）。

## 相关页面

[[source-position-interpolation-2306-15595]] 为源摘要，[[rope]] 对应被改写的相位结构，[[context-window-extension]] 标定扩窗方法谱系中的位置；[[ntk-aware-interpolation]]、[[ntk-by-parts-interpolation]] 与 [[yarn]] 为按波长分档的后续方法，[[dynamic-scaling]] 对应推理时按序列长度取 $s$，[[learnable-frequency-scaling]] 将每维频率改为可学习参数。

## 引用

[^src-pi]: [[source-position-interpolation-2306-15595]]
[^src-yarn]: [[source-yarn]]
