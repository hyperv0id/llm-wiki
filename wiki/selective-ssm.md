---
title: "Selective SSM（S6）"
type: technique
tags:
  - mamba
  - state-space-model
  - selective-scan
  - hardware-aware
created: 2026-10-27
last_updated: 2026-10-27
source_count: 4
confidence: medium
status: active
---

# Selective SSM（S6）

选择性状态空间模型是 Mamba 论文（Gu & Dao, 2023）的核心序列变换层：将 S4 的 SSM 参数 $(\Delta, B, C)$ 设为输入函数，沿序列维度选择性记忆或遗忘信息；代价是丧失线性时不变性（LTI）与卷积形式，改用硬件感知的并行 scan 计算[^src-gu-dao-2023-mamba]。论文缩写 S6 = S4 + Selection + Scan（Remark 3.1）[^src-gu-dao-2023-mamba]。

## 问题：压缩视角下的 LTI 困境

论文将序列建模的基本问题归结为把上下文压缩至更小状态：Attention 完全不压缩（自回归推理显式保存全上下文，即 KV cache），有效但训练 $O(L^2)$、推理每步线性变慢；循环模型维持有限状态，高效但效果受限于状态压缩质量[^src-gu-dao-2023-mamba]。LTI 模型（S4 及此前所有结构化 SSM）的 $(\Delta, A, B, C)$ 在全部时间步固定，无法根据输入改变隐状态：递推视图下无法选择正确信息，卷积视图下无法处理变间距——全局卷积能解普通 Copying（只要求时间感知），解不了 Selective Copying（要求内容感知）[^src-gu-dao-2023-mamba]。两个驱动任务是 Selective Copying（在随机间距下记住数据 token、滤除噪声）与 Induction Heads（基于上下文关联召回）[^src-gu-dao-2023-mamba]。

## 数据流与维度变换

张量形状记作 $[\text{行}, \text{列}]$，其中 $B$ 为 batch、$L$ 为序列长度、$D$ 为通道维（模型维）、$N$ 为 SSM 状态扩展维。论文按单通道书写（$A\in\mathbb{R}^{N\times N}$ 对角、$B\in\mathbb{R}^{N\times 1}$、$C\in\mathbb{R}^{1\times N}$ 各可用 $N$ 个数表示）；对角结构下全通道并行时 $A$ 记作 $[D, N]$。每个输入的隐状态总量为 $D\times N$，朴素物化需 $O(BLDN)$ 时间与显存，是效率瓶颈的根源[^src-gu-dao-2023-mamba]。

### Stage 1 · S4 基线：连续参数与离散化

$$\begin{aligned}
h'(t) &= A h(t) + B x(t),\quad y(t) = C h(t) && \text{// } [N]\ \text{——单通道连续线性 ODE（式 1），S4 的出发点}\\
\bar{A} &= \exp(\Delta A) && \text{// } [N, N]\ \text{——ZOH 离散化的转移矩阵（式 4），对角时每通道 } N \text{ 个数}\\
\bar{B} &= (\Delta A)^{-1}\left(\exp(\Delta A) - I\right)\cdot \Delta B && \text{// } [N, 1]\ \text{——ZOH 对零阶保持输入的精确积分}\\
h_t &= \bar{A} h_{t-1} + \bar{B} x_t,\quad y_t = C h_t && \text{// } [N]\ \text{——离散线性递推（式 2），自回归推理形式}\\
K &= (C\bar{B},\ C\bar{A}\bar{B},\ \dots,\ C\bar{A}^{k}\bar{B},\ \dots),\quad y = x * K && \text{// } [L]\ \text{——LTI 下展开递推得全局卷积核（式 3），并行训练形式}
\end{aligned}$$

离散化 $(\Delta, A, B) \mapsto (\bar{A}, \bar{B})$ 后有两条等价路径：训练用卷积（整段输入可见）、推理用递推（逐步输入），二者等价的前提是 LTI[^src-gu-dao-2023-mamba]。论文正文仅给出 ZOH 公式并提及可换其他离散化规则；双线性变换 $\bar{A}=(I-\Delta A/2)^{-1}(I+\Delta A/2)$ 是该系列文献常用替代，本文未给出公式[^src-gu-dao-2023-mamba][INFERENCE]。

### Stage 2 · 选择机制：参数变为输入的函数

Algorithm 2（S6）相对 Algorithm 1（S4）的全部改动（输入 $x\in[B,L,D]$ 不变）[^src-gu-dao-2023-mamba]：

$$\begin{aligned}
A &\in [D, N]\ (\text{常量参数}) && \text{// 对角结构的 } N\times N \text{ 转移，保持静态}\\
B &= s_B(x) = \operatorname{Linear}_N(x) && \text{// } [B, L, D] \xrightarrow{\text{线性投影}} [B, L, N]\ \text{——S4 中 } [D, N] \text{ 常量变逐位置向量}\\
C &= s_C(x) = \operatorname{Linear}_N(x) && \text{// } [B, L, D] \to [B, L, N]\ \text{——控制状态到输出的读出}\\
\Delta &= \tau_\Delta\!\left(\operatorname{Parameter} + \operatorname{Broadcast}_D(\operatorname{Linear}_1(x))\right) && \text{// } [B, L, 1] \xrightarrow{\operatorname{softplus}\text{ 后广播}} [B, L, D]\ \text{——通道特异步长}\\
\bar{A}, \bar{B} &\in [B, L, D, N] && \text{// } \exp(\Delta\, A)\ \text{按 } [B,L,D]\otimes[D,N]\ \text{广播，时变对角转移}\\
h_t &= \bar{A}_t \odot h_{t-1} + \bar{B}_t x_t,\quad y_t = C_t h_t && \text{// } [D, N]\ \text{——逐位置参数递推（式 2 输入依赖版），只能 scan}
\end{aligned}$$

维度变化关键：$B, C$ 从每通道常量 $(D, N)$ 变为逐位置的 $(N)$ 向量——batch/length 维展开、通道维 $D$ 消失，同一位置各通道共享 $B/C$；通道特异性全部集中于 $\Delta$ 的 $(B, L, D)$。$A$ 保持静态：它只通过 $\bar{A}=\exp(\Delta A)$ 与 $\Delta$ 交互，$\Delta$ 的选择性已足够，论文假设再加 $A$ 的选择性收益相似，因而省略[^src-gu-dao-2023-mamba]。

**卷积形式为何失效**：卷积核 $K_k = C\bar{A}^k\bar{B}$ 的推导要求 $(\bar{A}, \bar{B}, C)$ 恒定，展开后系数只依赖滞后 $k$；选择机制下参数随位置变化，展开式 $y_t = \sum_{j\le t} \left(C_t \prod_{s=j+1}^{t} \bar{A}_s\right)\bar{B}_j x_j$ 的系数同时依赖区间 $(j, t]$ 而非单看滞后——输入一变整条核就变，不存在可预先计算的核[^src-gu-dao-2023-mamba][^src-demystify-mamba-linear-attention-2024]。选择机制由此失去与卷积 (3) 的等价性[^src-gu-dao-2023-mamba]。

**与 RNN 门控的等价（Theorem 1）**：取 $N=1, A=-1, B=1$、$\tau_\Delta=\operatorname{softplus}$，ZOH 给出 $\bar{A}_t=\exp(-\Delta_t)=1-\sigma(\operatorname{Linear}(x_t))$，递推为 $g_t=\sigma(\operatorname{Linear}(x_t))$、$h_t=(1-g_t)h_{t-1}+g_t x_t$，即经典门控 RNN（leaky integrator）。这解释了 $s_\Delta, \tau_\Delta$ 的取法：$\Delta$ 大则重置状态、聚焦当前输入；$\Delta$ 小则保持状态、忽略输入[^src-gu-dao-2023-mamba]。

选择机制的三种效应（§3.5.2）：变间距（滤除感兴趣输入间的噪声，$g_t\to 0$）、过滤上下文（可随时重置状态，长上下文性能原则上单调改善）、边界重置（拼接的多序列间不串扰；Transformer 靠 attention mask 实现，LTI 模型则会 bleed）[^src-gu-dao-2023-mamba]。

消融（≈350M 语言模型、Chinchilla token 口径）：Table 7——全不选 10.93 ppl，只选 $\Delta$ 9.81，$\Delta+B+C$ 全选 8.71，$\Delta$ 最关键（Theorem 1 的直接后果）；Table 9——$\Delta$ 投影到 1 维即从 9.12 降到 8.97，加宽到 64 维再降至 8.71；Table 10——状态维 $N$ 从 1 增到 16 只在 $B, C$ 也可选时有超过 1.0 ppl 的改善（9.73→8.71），参数仅 +1%（367.1M→371.5M），验证了"状态扩展 + 选择"的组合动机[^src-gu-dao-2023-mamba]。

### Stage 3 · 硬件感知 scan：kernel 融合与重计算

选择机制把 $\bar{A}, \bar{B}$ 变成 $[B, L, D, N]$，比输入 $[B, L, D]$ 大 $N$ 倍（$N\approx 10\text{–}100$），既不能物化到 HBM 也不能走卷积快路。论文的三个经典技术[^src-gu-dao-2023-mamba]：

$$\begin{aligned}
(\Delta, A, B, C) &: \text{HBM} \to \text{SRAM} && \text{// } O(BLD + DN)\ \text{——只读小参数，不搬运扩展态}\\
(\bar{A}, \bar{B}) &\gets \operatorname{discretize}(\Delta, A, B)\ \text{于 SRAM} && \text{// } [B, L, D, N]\ \text{——扩展态只存在于最快一层显存}\\
h &\gets \operatorname{assoc\_scan}(\bar{A}, \bar{B}x) && \text{// 沿 } L\ \text{work-efficient 并行前缀组合，} O(BLDN)\ \text{FLOPs 线性于 } L\\
y &= \sum_{N} C \odot h\ \text{写回 HBM} && \text{// } [B,L,N]\odot[B,L,D,N] \xrightarrow{\text{沿 } N\ \text{求和}} [B, L, D]\ \text{——IO 共减少 } O(N)\ \text{倍}
\end{aligned}$$

递推的时序串行问题由并行关联扫描（parallel associative scan，Blelloch 1990 一系）解决：逐步转移虽依赖输入，但满足结合律，可按树形归并[^src-gu-dao-2023-mamba]。在 FLOPs 口径下，朴素递推 $O(BLDN)$ 对比卷积 $O(BLD\log L)$，长序列且 $N$ 不大时递推常数更低[^src-gu-dao-2023-mamba]。$L$ 超出 SRAM 时分 chunk 逐段融合扫描，携带中间 scan 状态即可跨 chunk 续算[^src-gu-dao-2023-mamba]。反向传播不存 $[B,L,D,N]$ 中间态而直接重算：HBM 读写从 $O(BLND)$ 降至 $O(BLN + DN)$，且重算比读回更快；整个 selective SSM 层的显存与 FlashAttention 优化的 Transformer 相当（每层约 16 bytes/token，两层 ≈ attention 12 + MLP 20 bytes）[^src-gu-dao-2023-mamba]。实测中，融合 scan 比标准 PyTorch scan 快 20–40×；序列长度超过 2K 后快于 FlashAttention-2，32K 处至多 7×（§4.5、Appendix D）[^src-gu-dao-2023-mamba]。

论文未逐条给出实现细节：初始状态从 $h_0=0$ 起步、scan 沿 head/batch 维并行、chunk 内长度对齐 SRAM 容量等，均为实现约定[^src-gu-dao-2023-mamba][INFERENCE]。

### Stage 4 · block 结构：与 Transformer block 的替换点

Mamba block 受 GAU 启发，将 H3 块与 MLP 块合并为同构块（Figure 3）[^src-gu-dao-2023-mamba]：

$$\begin{aligned}
u &\to \operatorname{S6}\!\left(\operatorname{SiLU}\left(\operatorname{Conv}(\operatorname{Linear}_{ED}(x))\right)\right) && \text{// } [D] \xrightarrow{2ED^2\ \text{输入投影之一}} [ED]\ \text{——SSM 主支路}\\
g &= \operatorname{SiLU}\!\left(\operatorname{Linear}_{ED}(x)\right) && \text{// } [D] \to [ED]\ \text{——门支路}\\
y &= \operatorname{Linear}_D(u \odot g) && \text{// } [ED]\odot[ED]\to[ED] \xrightarrow{ED^2\ \text{输出投影}} [D]\ \text{——乘性门后投回}
\end{aligned}$$

多数参数集中于投影（输入 $2ED^2$、输出 $ED^2$，共 $3ED^2$），SSM 本体（$\Delta, B, C$ 投影与 $A$）占比很小；实验固定 $E=2$，用两块堆叠对齐 Transformer 的 MHA+MLP 的 $12D^2$ 参数[^src-gu-dao-2023-mamba]。采用 SiLU/Swish 使 Gated MLP 成为 SwiGLU 变体；块间配标准归一化与残差，另有可选 normalization（取 LayerNorm，对齐 RetNet 同位置用法）[^src-gu-dao-2023-mamba]。Figure 3 给出拓扑，支路拆分与 SiLU 位置按参数与 SwiGLU 动机整理；短卷积核宽等实现细节论文未写[^src-gu-dao-2023-mamba][INFERENCE]。

Transformer 层由 MHA 与 MLP 子块交错构成，Mamba 则由单一同构块齐次堆叠。按参数对齐口径，一对 (MHA, MLP) 对应两个 Mamba block[^src-gu-dao-2023-mamba]。同 S6 层下 Mamba block 为 8.69 ppl，对比 H3 block 为 8.95（Table 6）；将其与普通 MLP 或 MHA 块交错变化不大（Figure 9）[^src-gu-dao-2023-mamba]。

## 与其他页面

[[mamba]] 为实体页，本页记录原始论文口径细节。[[linear-attention-unified-framework]] 中，Han et al. 将本页递推与线性注意力对齐为同一公式（$C_t\leftrightarrow Q_t$、$B_t\leftrightarrow K_t^\top$、$\Delta_t\odot x_t\leftrightarrow V_t$），差异落在六项可消融组件上；本页给出递推的原始出处。Kimi Linear 报告把 Mamba2 的更新归入「只有乘性衰减、无 delta 修正项」一类（Table 7），其合成任务对照（palindrome/MQAR/stack）中 Mamba2 全部失败，作为 [[kimi-delta-attention|KDA]] 引入 delta rule 的反面证据（§5.1）[^src-moonshot-2025-kimi-linear]。关于 [[fast-weight-mlp-view-of-attention]]，论文 Appendix A 将选择机制与 fast weights 区分：对角层 $y=\sigma(Wx)\circ x$ 虽符合 gating、hypernetwork 与 data-dependence 的通俗定义，实质仅为 GLU；论文只把沿序列维交互的门控（以 RNN 门控为特例）归入选择机制[^src-gu-dao-2023-mamba]。在 [[deep-state-space-model]] 的术语谱系中，DeepState 属于统计 SSM（线性高斯与 Kalman 解析推断），本页 SSM 专指结构化 S4 家族，论文 §2 对二者做了明确切割[^src-gu-dao-2023-mamba]。[[mamba-block-design]] 记录 Han et al. 对 Mamba 块的视觉重构消融（子块交换、拼接与统一归一化），而非原始 Mamba block 本身；原始块见本页 Stage 4。[[flash-attention]] 中 Stage 3 的 kernel 融合与重计算沿用 IO-aware 原则对齐显存目标；论文直接以 FlashAttention 实现作为显存与速度参照。后续谱系见 [[gated-delta-rule]]：Gated DeltaNet 把本页「数据依赖遗忘」的标量门塞进 delta rule 的 Householder 转移矩阵，其论文 §5 指出现代 forget gate 与 LSTM 设计的区别正是去掉对前一隐状态的依赖、保住跨长度并行（Yang et al., ICLR 2025）[^src-yang-2024-gated-delta]。

## 边界（论文自述）

选择机制通过 scan 保持训练并行，但失去了卷积的 FFT 快速路径，这也是此前 SSM 坚持 LTI 的根本原因[^src-gu-dao-2023-mamba]。在连续与离散信号上，选择机制适用于文本与 DNA 等离散模态，但在均匀采样且平滑的连续信号上表现相反：长音频波形预训练中 S4 改为 S6 效果显著变差；靠近原始信号的层需保持 LTI，经外层压缩为 token 后的内层则不需要[^src-gu-dao-2023-mamba]。数域设置以实数为默认且普遍适用，唯一例外是音频实验改用复数，作者推测复数利于连续模态[^src-gu-dao-2023-mamba]。模型规模上，缩放律覆盖至 1.3B，下游评测达到 2.8B（300B token），7B+ 规模下的表现与所需工程调整留待后续验证[^src-gu-dao-2023-mamba]。

[^src-gu-dao-2023-mamba]: [[source-gu-dao-2023-mamba]]
[^src-yang-2024-gated-delta]: [[source-yang-2024-gated-delta]]
[^src-demystify-mamba-linear-attention-2024]: [[source-demystify-mamba-linear-attention-2024]]
[^src-moonshot-2025-kimi-linear]: [[source-moonshot-2025-kimi-linear]]
