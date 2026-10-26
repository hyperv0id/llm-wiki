---
title: "Universal Transformer"
type: technique
tags:
  - architecture
  - recurrence
  - adaptive-computation
  - weight-tying
  - language-model
created: 2026-10-27
last_updated: 2026-10-27
source_count: 1
confidence: medium
status: active
---

# Universal Transformer

Universal Transformer（UT，Dehghani et al., ICLR 2019）将 [[source-attention-is-all-you-need-1706-03762|标准 Transformer]] 的固定层栈换成**同一组权重沿深度方向施加 $T$ 步**的循环：循环不沿序列位置展开，而是在深度方向连续修订各位置的表示，计算量由修订次数而非序列长度决定[^src-dehghani-2019-universal-transformers]。其直接动机有二：标准 Transformer 缺少 RNN 的迭代归纳偏置，在超出训练长度的任务（copy、简单逻辑推理）上泛化差；有限精度下固定深度无法实现计算通用（附录 B）[^src-dehghani-2019-universal-transformers]。动态停机版引入 ACT 式逐位置停机，让不同符号分配不同步数[^src-dehghani-2019-universal-transformers]。

## 两种变体与 fixed-depth Transformer 的对照

| 设定 | 等价物 / 含义 | 出处 |
|---|---|---|
| 循环步取 1 | 标准 Transformer | §4 |
| 固定 $T$ 步 | 层间权重绑定的多层 Transformer（自注意力与 transition 都绑定，脚注 5） | §4 |
| 自注意力+残差取恒等、transition 取卷积、$T$ = 输入长度 | 恰为 Neural GPU | §4 |
| 步数取输入长度的（可确定性的）函数 | 有限精度下解除「串行步数与输入无关」的限制，计算通用的前提 | 附录 B |

与 RNN 的关键差别（§4）：时间维递归模型在递归步内只能访问固定大小的状态向量，计算上更接近自动机；UT 每步递归对上一层**全部位置状态**计算注意力，相当于在递归步内可直接访问内存[^src-dehghani-2019-universal-transformers]。

## 数据流与维度

**维度符号**：$m$ 序列位置数；$d$ 表示维；$k$ 头数，每头通道维 $d/k$；$T$ 循环步数上限；$V$ 词表大小；$H^t\in\mathbb{R}^{m\times d}$ 为第 $t$ 步全体位置状态矩阵；停机参数设阈值 $\epsilon\in(0,1)$ 与最大步数 $T_{\max}$（论文仅说明是超参，未给数值）。形状记 $[\text{行},\text{列}]$，与论文记号一致。

### Stage 1 · 嵌入并叠加位置-时间坐标编码

$$\begin{aligned}
H^0 &= \mathrm{Embed}(x) && \text{// } [m]\xrightarrow{\text{逐位置查嵌入表}}[m,d]\ \text{——每行是第 } i \text{ 个符号的 } d \text{ 维嵌入，作为初态}\\
P^t_{i,2j} &= \sin(i/10000^{2j/d})+\sin(t/10000^{2j/d}) && \text{// } [m,d]\ \text{——位置 } i \text{ 与循环步 } t \text{ 各自的正弦编码逐维相加（式 6），固定常数、无可学参数}\\
P^t_{i,2j+1} &= \cos(i/10000^{2j/d})+\cos(t/10000^{2j/d}) && \text{// } [m,d]\ \text{——奇数维用余弦（式 7），同一套频率铺开}
\end{aligned}$$

$P^t$ 对任意位置下标 $i$ 与步数下标 $t$ 直接计算，不含随训练长度定死的查找表参数；算法任务训练时位置还从随机偏移开始，鼓励学到位置相对的变换（§3.4）[^src-dehghani-2019-universal-transformers]。

### Stage 2 · 跨位置自注意力（每步）

$$\begin{aligned}
\mathrm{Attention}(Q,K,V) &= \mathrm{softmax}\!\left(QK^{\top}/\sqrt{d}\right)V && \text{// } [m,\bcancel{d/k}]\times[\bcancel{d/k},m]\to[m,d/k]\ \text{——消去每头通道维；式 1 的 } \sqrt{d} \text{ 中 } d \text{ 按式内定义是 } Q \text{ 的列数，softmax 沿 key 轴}\\
\mathrm{head}_i &= \mathrm{Attention}\!\left((H^{t-1}+P^t)W_i^Q,\ (H^{t-1}+P^t)W_i^K,\ (H^{t-1}+P^t)W_i^V\right) && \text{// } [m,\bcancel{d}]\times[\bcancel{d},d/k]\to[m,d/k]\ \text{——式 3 的逐头仿射投影作用于加坐标编码的输入（与下行式 5 一致），} W\in\mathbb{R}^{d\times d/k}\text{，共 } k \text{ 头}\\
A^t &= \mathrm{LayerNorm}\!\left((H^{t-1}+P^t)+\mathrm{MultiHeadSelfAttention}(H^{t-1}+P^t)\right) && \text{// } [m,d]\ \text{——式 5：坐标编码同时加在残差支与注意力输入上，多头输出拼回 } d \text{ 维后逐行 LayerNorm}
\end{aligned}$$

### Stage 3 · 逐位置 transition 与状态更新

$$\begin{aligned}
H^t &= \mathrm{LayerNorm}\!\left(A^t+\mathrm{Transition}(A^t)\right) && \text{// } [m,d]\ \text{——式 4：Transition 逐行独立作用（separable 卷积或两层仿射夹 ReLU），本步不再跨位置混合信息}
\end{aligned}$$

### Stage 4 · 解码与输出

解码器复用同一循环结构，在自注意力之后加对编码器末态 $H^T$ 的 cross-attention（query 来自解码器状态、key/value 来自编码器），自注意力掩 causal；训练用 teacher-forcing、目标右移一位（§2.1）[^src-dehghani-2019-universal-transformers]。

$$\begin{aligned}
p_{y^{pos}} &= \mathrm{softmax}\!\left(O\,H^T\right) && \text{// } [m,\bcancel{d}]\times[\bcancel{d},V]\to[m,V]\ \text{——式 8：末态过仿射 } O\in\mathbb{R}^{d\times V} \text{ 逐行归一；式中 } T \text{ 是时间步上标而非转置（脚注 1）}
\end{aligned}$$

### Stage 5 · ACT 动态停机（逐位置标量控制）

机制归 ACT（Graves, 2016）：每步预测标量停机概率，按阈值累积停机，余量（remainder）计入 ponder time（§2.2）[^src-dehghani-2019-universal-transformers]。论文未把该节写成数学式，只给实现（附录 C，Listing 1–2）；下列公式为该伪代码的逐行转写，逐位置独立、在 $m$ 个位置上并行。记 $\pi_t$ 为累积停机概率、$\rho_t$ 为余量、$h^t\in\mathbb{R}^d$ 为单个位置的状态，均初始化为零：

$$\begin{aligned}
p_t &= \sigma\!\left(u^{\top}h^{t}+b\right) && \text{// } [m]\ \text{——每位置一个停机概率：当前状态过单层仿射加 sigmoid（Listing 2 L4–5）}\\
\rho_t &\mathrel{+}= \mathbb{1}\!\left[\pi_{t-1}+p_t>\epsilon\right]\odot\mathbb{1}\!\left[\pi_{t-1}<1\right]\odot\left(1-\pi_{t-1}\right) && \text{// } [m]\ \text{——首次跨过阈值的位置记下余量，按本步之前的累积值计算，本步 } p_t \text{ 不计入}\\
\pi_t &\mathrel{+}= p_t\odot\mathbb{1}\!\left[\pi_{t-1}+p_t\le\epsilon\right]+\rho_t\odot\mathbb{1}\!\left[\pi_{t-1}+p_t>\epsilon\right]\odot\mathbb{1}\!\left[\pi_{t-1}<1\right] && \text{// } [m]\ \text{——继续运行的位置累加 } p_t\text{，本步停机的位置补余量使累积恰为 1，已停机位置不再变动}\\
w_t &= p_t\odot\mathbb{1}\!\left[\pi_{t-1}+p_t\le\epsilon\right]+\rho_t\odot\mathbb{1}\!\left[\pi_{t-1}+p_t>\epsilon\right]\odot\mathbb{1}\!\left[\pi_{t-1}<1\right] && \text{// } [m]\ \text{——状态混合权重：继续运行的用 } p_t\text{，本步停机的用余量 } \rho_t\text{，已停机的为 0}\\
h^{t+1} &= w_t\odot\tilde{h}^{t}+\left(1-w_t\right)\odot h^{t},\quad \tilde{h}^{t}=\mathrm{Transition}\!\left(\mathrm{SelfAttn}\!\left(h^{t}\right)\right) && \text{// } [m,d]\ \text{——新旧状态逐位置凸组合；已停机位置 } w_t=0\text{，状态原样复制}
\end{aligned}$$

循环继续的条件为「仍存在位置满足 $\pi<\epsilon$ 且更新次数 $<T_{\max}$」（Listing 1）[^src-dehghani-2019-universal-transformers]。正文中「停机后状态直接复制到下一步、直到全部停机或到最大步数」的描述与该实现一致（§2.2）[^src-dehghani-2019-universal-transformers]。Graves (2016) 原始 ACT 的 ponder-cost 损失项在论文正文与附录 C 实现中均未出现，「ponder time」仅作为测量量报告；Graves 原文的停机概率与平均状态更新公式不在本源内，不作转引[^src-dehghani-2019-universal-transformers]。

## 为什么可变深度能处理任意长度（按论文）

UT 的计算不受序列符号数约束，仅由每个表示被修订的次数约束（§2.1）[^src-dehghani-2019-universal-transformers]。附录 B 论证指出，标准 Transformer 的串行步数恒定且与输入长度无关，有限精度下无法达到计算通用；对任意深度 $T$ 均能构造出长度 $N>T$、无法被正确处理的输入。UT 将步数取为输入长度的函数即解除该限制；该论证不依赖是否使用 ACT，只要求步数非常数，而训练后动态调整步数依靠权重跨步共享[^src-dehghani-2019-universal-transformers]。正弦坐标编码对任意 $i,t$ 直接计算（Stage 1）；算法任务训练时位置从随机偏移开始（§3.4）[^src-dehghani-2019-universal-transformers]。train 40 → test 400 的算法任务外推明显改善但未达完美（Table 4：UT Copy char-acc 0.91 / seq-acc 0.35）；报告满分 1.0 的 Neural GPU 采用了专门 curriculum，其余模型均未使用（§3.4）[^src-dehghani-2019-universal-transformers]。

## 证据

### bAbI 问答（Table 1，平均错误率，括号内为失败任务数 >5% / 20）

| 模型 | 10K single | 10K joint | 1K single | 1K joint |
|---|---|---|---|---|
| QRNet / Sparse DNC / GA+MAGE / MemN2N（此前最好值） | 0.3 (0/20) | 2.9 (1/20) | 8.7 (5/20) | 12.4 (11/20) |
| Transformer | 15.2 (10/20) | 22.1 (12/20) | 21.8 (5/20) | 26.8 (14/20) |
| Universal Transformer | 0.23 (0/20) | 0.47 (0/20) | 5.31 (5/20) | 8.50 (8/20) |
| UT w/ dynamic halting | 0.21 (0/20) | 0.29 (0/20) | 4.55 (3/20) | 7.78 (5/20) |

bAbI 任务中，每条事实的词嵌入先乘可学习的位置掩码再求和，问题同法编码（§3.1）[^src-dehghani-2019-universal-transformers]。模型从 10 次不同初始化中选取验证集最优结果；附录 E 给出逐任务分解与 10 seed 平均（10K single/joint：$0.73\pm0.89$ / $1.12\pm1.62$；1K：$6.34\pm3.32$ / $11.21\pm6.65$），1K joint 下任务 17（位置推理）best seed 42.9、任务 19（路径查找）68.2 仍失败[^src-dehghani-2019-universal-transformers]。注意力分布从均匀逐步锐化到支撑事实附近；平均 ponder 时间按支撑事实数 1/2/3 递增为 $2.3\pm0.8$ / $3.1\pm1.1$ / $3.8\pm2.2$，多数位置 1–2 步即停（§3.1、附录 F）[^src-dehghani-2019-universal-transformers]。

### 主谓一致（Table 2，ranking accuracy）

Transformer 总计 0.962，UT 与 UT w/ ACT 均 0.992，与此前最好值持平；差异在难度轴上：agreement attractors（干扰名词）数 0→5 时，UT w/ ACT 相对最强前模型的差从 0 变到 +0.027，干扰名词越多优势越大（§3.2）[^src-dehghani-2019-universal-transformers]。

### LAMBADA（Table 3，LM 困惑度 / RC 准确率）

| 模型 | LM test PPL (acc) | RC test acc |
|---|---|---|
| Transformer | 7321 (0.0) | 0.3988 |
| LSTM | 5174 (0.0) | 0.2007 |
| UT base, 6 steps (fixed) | 319 (0.17) | 0.5216 |
| UT w/ dynamic halting | 142 (0.19) | 0.5625 |

固定步版 6 步最好；动态停机版在测试上的平均步数是 $8.2\pm2.1$。补训的 8 步（dev 192 / test 202）与 9 步（214/239）固定模型好于 6 步但不及动态版，作者推测停机通过让部分符号少算、部分多算起了正则作用（§3.3）[^src-dehghani-2019-universal-transformers]。RC 设置目标词 81% 出现在上下文中、19% 情形不可解，口径比 LM 设置松（§3.3）[^src-dehghani-2019-universal-transformers]。

### 算法任务（Table 4，train 长度 40 → test 长度 400，char-acc / seq-acc）

| 模型 | Copy | Reverse | Addition |
|---|---|---|---|
| LSTM | 0.45 / 0.09 | 0.66 / 0.11 | 0.08 / 0.0 |
| Transformer | 0.53 / 0.03 | 0.13 / 0.06 | 0.07 / 0.0 |
| Universal Transformer | 0.91 / 0.35 | 0.96 / 0.46 | 0.34 / 0.02 |
| Neural GPU（专门 curriculum） | 1.0 / 1.0 | 1.0 / 1.0 | 1.0 / 1.0 |

### Learning to Execute（Tables 5–6）

Memorization 组（copy/double/reverse，最大长度 55）UT 全部 char-acc 与 seq-acc 1.0；program evaluation 组（nesting 2、长度 5）UT 为 Program 0.89/0.63、Control 1.0/1.0、Addition 1.0/1.0，Transformer 为 0.71/0.29、0.93/0.66、1.0/1.0（§3.5）[^src-dehghani-2019-universal-transformers]。

### WMT14 En-De 翻译（Table 7，BLEU，8×P100）

| 模型 | BLEU |
|---|---|
| Universal Transformer small | 26.8 |
| Transformer base | 28.0 |
| Weighted Transformer base | 28.4 |
| Universal Transformer base | 28.9 |

最优 UT 采用全连接 transition（非 separable 卷积）且**不用 ACT**；在参数量相同的各 base 模型中，比 Transformer base 高 0.9、比 Weighted Transformer 高 0.5（§3.6）[^src-dehghani-2019-universal-transformers]。

## 边界（论文自述）

ACT 在机器翻译上轻度降低结果（§1）[^src-dehghani-2019-universal-transformers]。标准 Transformer 在 bAbI 上调整超参数与网络规模始终过拟合（§3.1 脚注 2），是该基准对照成立的前提[^src-dehghani-2019-universal-transformers]。算法任务长度外推有所改善但不完美（Table 4）；Neural GPU 达满分依赖 curriculum，UT 未用 curriculum（§3.4）[^src-dehghani-2019-universal-transformers]。bAbI 1K 联合训练下难任务仍失败（附录 E：任务 17 best seed 42.9、任务 19 best seed 68.2）[^src-dehghani-2019-universal-transformers]。图灵完备建立在「under certain assumptions」下：需要足够内存与非常数的串行步数；有限精度与恒定深度下标准 Transformer 无法通用计算（§4、附录 B）[^src-dehghani-2019-universal-transformers]。动态停机收益集中在小而结构化的任务，大规模 MT 上不用 ACT 更好（§1、§3.6）[^src-dehghani-2019-universal-transformers]。

## 谱系与后续

UT 定位于与 Neural GPU 及 Neural Turing Machine 同族的「并行 RNN 块」（§4）：与仅绑定自注意力层的 Recursive Transformer 相比，UT 连同 transition 一并绑定，同时在自然语言任务上具备竞争力[^src-dehghani-2019-universal-transformers]。后续循环深度路线见 [[recurrent-depth]]：Huginn 将 UT 列为 recurrent depth 重复发现的一环；两者差异在于 Huginn 采用块级共享、无停机且每轮重注入输入嵌入，UT 采用全层共享、ACT 逐位置停机并在每步重新叠加正弦位置-时间编码（对照见该页谱系段）。逐位置编码见 [[sinusoidal-positional-encoding]]；长度外推见 [[position-extrapolation]]；基础对照见 [[transformer-architecture]]。

[^src-dehghani-2019-universal-transformers]: [[source-dehghani-2019-universal-transformers]]
