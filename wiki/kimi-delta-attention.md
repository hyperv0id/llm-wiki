---
title: "Kimi Delta Attention（KDA）"
type: technique
tags:
  - attention
  - linear-attention
  - delta-rule
  - gated-recurrence
  - long-context
  - llm
created: 2026-10-27
last_updated: 2026-10-27
source_count: 1
confidence: medium
status: active
---

# Kimi Delta Attention（KDA）

KDA 是 Kimi Linear（Moonshot AI，arXiv:2510.26692）提出的门控线性注意力模块：将 [[gated-delta-rule|GDN]] 的每头标量遗忘门细化为逐通道对角门 $\operatorname{Diag}(\alpha_t)$，并由 DPLR（Diagonal-Plus-Low-Rank）转移矩阵的特化变体导出定制 chunkwise 算法，避免通用 DPLR 的计算代价（摘要、§3）[^src-moonshot-2025-kimi-linear]。论文将其定位为首个在同数据同配方公平对照下，于短上下文、长上下文与 RL scaling 三个 regime 全面超过 full attention 的混合线性注意力架构（摘要）[^src-moonshot-2025-kimi-linear]。

## 问题定位

纯线性注意力受有限状态容量约束，长序列精确检索与复制在理论上困难；将少量全局注意力层混入线性层（hybrid）是折中质量与效率的实用方案，但此前混合模型往往规模受限或评测不全（§1）[^src-moonshot-2025-kimi-linear]。GDN 与 Mamba2 均采用粗粒度每头标量遗忘门，头内所有通道共享同一遗忘速率，缺少类似 RoPE 的逐维频率多样性（§1、§6.1）[^src-moonshot-2025-kimi-linear]。KDA 则将门控细化至通道级：为每个特征维维护独立遗忘率，更精细地调控有限状态的 RNN 记忆（§1）[^src-moonshot-2025-kimi-linear]。

## 机制

**递推形式**（式 1）：在 GDN 的 $S_t = \alpha_t(I - \beta_t k_t k_t^\top)S_{t-1} + \beta_t k_t v_t^\top$ 中将标量 $\alpha_t$ 替换为对角矩阵：

$$S_t = \big(I - \beta_t k_t k_t^\top\big)\operatorname{Diag}(\alpha_t)\,S_{t-1} + \beta_t k_t v_t^\top,\qquad o_t = S_t^\top q_t$$

**DPLR 特化**（§6.2）。通用 DPLR 转移为 $S_t = (D - a_t b_t^\top)S_{t-1} + k_t v_t^\top$（S4 采用静态版本，计算时通常联合对角化到复平面，表达力受限）。KDA 设 $D = \operatorname{Diag}(\alpha_t)$、$a_t = \beta_t k_t$、$b_t = k_t \odot \alpha_t$，将 $a$、$b$ 绑定到 $k$，提取对角衰减公因式，先做 GLA 式细粒度乘性衰减，再做 DeltaNet 式 Householder 修正。其表达能力对齐通用 DPLR（兼具细粒度衰减），且计算量大幅下降：二级分块矩阵计算从 4 个减至 2 个，消去 3 个矩阵乘，算子效率提升约 100%（§3.2、§6.2）[^src-moonshot-2025-kimi-linear]。

**参数化**（§4）。逐头输入经 ShortConv + Swish，$q$、$k$ 再经 L2Norm 保持特征值稳定（式 10 前的参数化块）[^src-moonshot-2025-kimi-linear]；遗忘门 $\alpha_t = f(W_\alpha^\uparrow W_\alpha^\downarrow x_t)$ 与输出门均采用秩等于头维的低秩投影（旨在对齐参数量，性能与全秩门相当并缓解 Attention Sink）；输出侧为 $o_t = W_o\big(\operatorname{Sigmoid}(W_g^\uparrow W_g^\downarrow x_t) \odot \operatorname{RMSNorm}(\cdot)\big)$（式 10）[^src-moonshot-2025-kimi-linear]。消融显示 Sigmoid 输出门优于 GDN 采用的 Swish 门（验证 PPL 5.65 对 Swish 门的 5.81），去掉卷积层则效果变差（5.70，Table 1）[^src-moonshot-2025-kimi-linear]。

## 数据流与维度

**维度符号**（论文实配，§4、§6.3）：$L$ 为序列长度，切成 $L/C$ 个长 $C$ 的 chunk（$C=64$）；$n_h$ 为头数；$d_k = d_v = 128$ 为 key/value 单头维；$d$ 为模型维；单头记忆状态 $S_t \in \mathbb{R}^{d_k \times d_v}$ 与序列长度无关；$\Box^{[t]} \in \mathbb{R}^{C \times d}$ 为第 $t$ 个 chunk 内各向量按行堆叠的矩阵；$\gamma^{i\to j} := \prod_{k=i}^{j}\alpha_k$ 为累积衰减，$\Gamma^{[t]} \in \mathbb{R}^{C\times d_k}$ 为逐元素衰减矩阵栈（§2.1）；$(K^{[t]}_{1\to C}\Gamma^{[t]})$ 表示 $\Gamma^{[t]}$ 栈与 $K^{[t]}$ 按行配对做衰减归一（行 $i$ 为 $k_i/\gamma^{1\to i}$，即论文的 $K^{[t]}/\Gamma^{1\to C}_{[t]}$）。以下按单头书写，多头并行独立执行（门控线性注意力标准设定为头间独立，论文给出单头形式）。

### Stage 1 · 逐头参数化：短卷积、低秩门

四条支路并行读同一 $x_t \in \mathbb{R}^d$（§4）：

$$\begin{aligned}
q_t, k_t &= \operatorname{L2Norm}\!\big(\operatorname{Swish}(\operatorname{ShortConv}(W_{q/k} x_t))\big) && \text{// } [d] \xrightarrow{\ [d, d_k]\ \text{投影 + 短卷积 + Swish}\ } [d_k] \xrightarrow{\text{沿 head 维 }\ell_2\text{ 归一}} [d_k]\ \text{——L2Norm 保证特征值稳定}\\
v_t &= \operatorname{Swish}(\operatorname{ShortConv}(W_v x_t)) && \text{// } [d] \xrightarrow{\ [d, d_v]\ }\ [d_v]\ \text{——value 不归一，保留幅值信息}\\
\alpha_t &= f\!\left(W_\alpha^\uparrow W_\alpha^\downarrow x_t\right) \in [0,1]^{d_k} && \text{// } [d] \xrightarrow{\ [d, d_k]\ \text{降秩}}\ [d_k] \xrightarrow{\ [d_k, d_k]\ \text{升回}\ }\ [d_k]\ \text{——秩} = d_k \text{ 的低秩投影，逐通道遗忘率}\\
\beta_t &= \operatorname{Sigmoid}(W_\beta x_t) && \text{// } [d] \xrightarrow{\ [d, 1]\ }\ [1]\ \text{——每头标量写入强度，delta rule 的学习率}
\end{aligned}$$

### Stage 2 · 递推语义：对角门控 delta rule（式 1）

$$\begin{aligned}
S_t &= \big(I - \beta_t k_t k_t^\top\big)\operatorname{Diag}(\alpha_t) S_{t-1} + \beta_t k_t v_t^\top && \text{// } [d_k, d_k]\times[d_k, \bcancel{d_k}]\times[\bcancel{d_k}, d_v] + [d_k, 1]\times[1, d_v] \to [d_k, d_v]\ \text{——先逐通道衰减旧记忆，再秩 1 修正并写入新键值对}\\
o_t &= S_t^\top q_t && \text{// } [d_v, \bcancel{d_k}]\times[\bcancel{d_k}, 1] \to [d_v, 1]\ \text{——转置状态读出，用当前 query 检索联想记忆}
\end{aligned}$$

### Stage 3 · 块内预处理：UT 变换造 $W$、$U$（式 6–7）

把 chunk 内 $C$ 步秩 1 Householder 乘积用 WY 表示压成单个下三角逆，forward substitution 求逆（§3.1）：

$$\begin{aligned}
M^{[t]} &= \Big(I + \operatorname{StrictTril}\big(\operatorname{Diag}(\beta^{[t]})\,\big(\Gamma^{1\to C}_{[t]} \odot K^{[t]}\big)\big(K^{[t]}_{1\to C}\Gamma^{[t]}\big)^{\!\top}\big)\Big)^{-1}\operatorname{Diag}(\beta^{[t]}) && \text{// } [C, \bcancel{d_k}]\times[\bcancel{d_k}, C] \to [C, C] \xrightarrow{\ \text{StrictTril + 前代求逆}\ }\ [C, C]\ \text{——块内因果耦合收进一张小矩阵}\\
W^{[t]} &= M^{[t]}\big(\Gamma^{1\to C}_{[t]} \odot K^{[t]}\big) && \text{// } [C, \bcancel{C}]\times[\bcancel{C}, d_k] \to [C, d_k]\ \text{——衰减加权后的"擦除"方向}\\
U^{[t]} &= M^{[t]} V^{[t]} && \text{// } [C, \bcancel{C}]\times[\bcancel{C}, d_v] \to [C, d_v]\ \text{——修正后的"伪 value"（pseudo-value 项）}
\end{aligned}$$

### Stage 4 · 块间递推：状态跨 chunk 更新（式 8）

$$\begin{aligned}
S_{[t+1]} &= \operatorname{Diag}(\gamma^{C}_{[t]})\,S_{[t]} + \big(\Gamma^{i\to C}_{[t]} \odot K^{[t]}\big)^{\!\top}\big(U^{[t]} - W^{[t]} S_{[t]}\big) && \text{// } [d_k, d_k]\ \text{对角缩放}\ [d_k, d_v] \ +\ [d_k, \bcancel{C}]\times[\bcancel{C}, d_v] \to [d_k, d_v]\ \text{——每 chunk 一次矩阵乘：整块历史按通道指数衰减，整块新关联一次写入}
\end{aligned}$$

### Stage 5 · 块内并行：inter/intra 两段输出（式 9）

以块间递推、块内并行打满 Tensor Core 的矩阵乘吞吐（§3.1）：

$$\begin{aligned}
O_{\text{inter}} &= \Gamma^{1\to C}_{[t]} \odot Q^{[t]} S_{[t]} && \text{// } [C, \bcancel{d_k}]\times[\bcancel{d_k}, d_v] \to [C, d_v]\ \text{——块内全部 query 读进入本块前的历史状态}\\
O_{\text{intra}} &= \operatorname{Tril}\!\big(\big(\Gamma^{1\to C}_{[t]} \odot Q^{[t]}\big)\big(K^{[t]}_{1\to C}\Gamma^{[t]}\big)^{\!\top}\big)\big(U^{[t]} - W^{[t]} S_{[t]}\big) && \text{// } [C, \bcancel{d_k}]\times[\bcancel{d_k}, C] \xrightarrow{\ \text{下三角因果掩码}\ }\ [C, \bcancel{C}]\times[\bcancel{C}, d_v] \to [C, d_v]\ \text{——块内因果注意力规模受 } C \text{ 约束}\\
O^{[t]} &= O_{\text{inter}} + O_{\text{intra}} \in \mathbb{R}^{C \times d_v} && \text{// } [C, d_v] \oplus [C, d_v] \to [C, d_v]\ \text{——历史检索与块内修正相加}
\end{aligned}$$

### Stage 6 · 输出门控（式 10）

$$\begin{aligned}
o_t &= W_o\big(\operatorname{Sigmoid}\!\left(W_g^\uparrow W_g^\downarrow x_t\right) \odot \operatorname{RMSNorm}\!\big(\operatorname{KDA}(q_t, k_t, v_t, \alpha_t, \beta_t)\big)\big) && \text{// } [d_v\,n_h] \odot [d] \xrightarrow{\ [d, d]\ \text{输出投影}\ }\ [d]\ \text{——数据依赖逐通道门控，缓解 Attention Sink}
\end{aligned}$$

推理时切换 kernel：prefill 用 Stage 3–5 的 FLOP 密集 chunk kernel，自回归解码换用逐递推 kernel（式 2），状态保持 $[d_k, d_v]$ 不随序列增长（§6.3）[^src-moonshot-2025-kimi-linear]。

## 细粒度门控的表达力论证（论文机制分析，归因见各条）

**位置编码视角**（§6.1）。RoPE 的细粒度位置编码源于给每对维度分配不同旋转频率，类似沿特征维的非均匀傅里叶变换；标准 GDN 每头仅用标量衰减，缺少逐维差异，促使 KDA 引入通道级可学习门（§6.1）[^src-moonshot-2025-kimi-linear]。门控 delta rule 的打分展开成 $\sum_i q_t^\top\big(\prod_{j=i+1}^t A_j(I - \beta_j k_j k_j^\top)\big)k_i\, v_i$ 后与 RoPE 的累积旋转同构，GDN/KDA 因而可视为「转移矩阵数据依赖的乘性位置编码」，放宽了 RoPE 的正交约束（§6.1、式 11–12、Table 6）[^src-moonshot-2025-kimi-linear]。

**合成任务**（§5.1、Figure 4）。在 2 层 2 头、头维 128 的统一配置下，KDA、GDN、Mamba2 评测 palindrome（整串逆序复制）、MQAR（多查询联想召回）、stack（64 个独立栈的 LIFO 状态跟踪）：序列长 256→2048 全程 KDA 准确率最高；palindrome 与 MQAR 上 KDA 收敛显著快于 GDN；Mamba2 只有乘性衰减而无 delta rule，三个任务全部失败（§5.1）[^src-moonshot-2025-kimi-linear]。

**组件消融**（Table 1，16 头 16 层、同 FLOPs 同超参）：hybrid 比例 3:1 验证 PPL 5.65 最低（0:1 纯 MLA 5.77、1:1 5.66、7:1 5.70、15:1 5.82）；7:1 训练 PPL 与 3:1 同为 9.23 但验证明显更差，15:1 两项都差——细粒度门控支撑了更激进的线性层占比（§5.2）[^src-moonshot-2025-kimi-linear]。

## Kimi Linear 架构与成本账

**3:1 层间混合**。KDA 与 Full MLA 层按 3:1 重复交错（层间而非层内混合，基础设施简单、训练稳定）；MLA 层全部用 NoPE，位置信息与近因偏置完全交由 KDA 承载（§4）[^src-moonshot-2025-kimi-linear]。NoPE 的实际好处：MLA 可转成纯 MQA 推理、免去 RoPE base 调整或 YaRN 类外推处理，长上下文训练更简单（§4）[^src-moonshot-2025-kimi-linear]。消融支持 NoPE：加 RoPE 的 Kimi Linear (RoPE) 短上下文接近、长上下文变差（Table 5 平均 51.8 对 54.5），论文归因于全局层显式位置信号过强、与线性层隐式信号错配（§5.2）[^src-moonshot-2025-kimi-linear]。

**模型配置**。骨干沿 Moonlight：48B 总参 / 3B 激活 MoE，8/256 专家 + 1 共享专家（稀疏度 32 对 Moonlight 的 3 倍），首层稠密（§5.4）[^src-moonshot-2025-kimi-linear]。对照实验统一 1.4T token（MuonClip、WSD、lr $1.1\times10^{-3}$、batch 32M、上下文 4096、K2 语料）；发布 checkpoint 扩到 5.7T、上下文 1M（§5.4.1）[^src-moonshot-2025-kimi-linear]。

**注意力 FLOPs**（§6.3，式 13–14）：

$$\begin{aligned}
\mathrm{FLOPs}_{\mathrm{KDA}}(T; C, d_h) &= 6T d_h^2 + 3TCd_h + TC^2 && \text{// 全部项与 } T \text{ 一次方成正比}\\
\mathrm{FLOPs}_{\mathrm{Attn}}(T; d_h) &= 2T^2 d_h && \text{// 主导项随 } T \text{ 二次增长，即 full attention 的代价}
\end{aligned}$$

**3:1 混合**：KDA 状态固定 $[d_k, d_v]$ 且无 KV cache，长序列 KV cache 用量最多降低 75%（摘要）[^src-moonshot-2025-kimi-linear]；I/O 受限的解码时间随长度趋近 full attention 的 3:1 混合效率上限，batch=1 实测 1M 上下文 TPOT 达 2.3×（Figure 7b）；节省的显存换用更大 batch 后，1M 处解码理论加速至多 6.3×（1.84ms 对 11.48ms，Figure 1b）[^src-moonshot-2025-kimi-linear]。prefill 在 512k 处加速 2.3×、1M 处 2.9×（Figure 7a）；RULER 128k 处 Kimi Linear 得 84.3 分（MLA 为 81.3）且加速 3.98×，落在 Pareto 前沿（Figure 1a）[^src-moonshot-2025-kimi-linear]。相比 DPLR 通用实现，KDA kernel 在 64k 长度内提速约 2×（Figure 2，batch 1、16 头）[^src-moonshot-2025-kimi-linear]。

**Scaling law**（Table 2、Figure 5）：在 653M–1.7B 激活参数、38.8B–128B token 的五个 MoE 尺寸（8/64 专家、Muon、上下文 4096）上拟合，Kimi Linear $2.2879 \times C^{-0.0527}$ 对比 MLA $2.3092 \times C^{-0.0536}$，同算力最优训练下计算效率约为 1.16×（§5.3）[^src-moonshot-2025-kimi-linear]。

## 证据

**预训练 1.4T base（Table 3）**：Kimi Linear 对比 MLA 与 GDN-H：MMLU 73.8/71.6/72.2，MMLU-Pro 51.0/47.2/47.9，BBH 72.9/71.6/70.6，HellaSwag 82.9/81.7/82.2，GSM8K 83.9/83.7/81.7；仅 EvalPlus 处 60.2 低于 GDN-H 的 63.1（高于 MLA 的 59.5）（§5.5.1）[^src-moonshot-2025-kimi-linear]。

**SFT instruct（Table 4）**：MMLU 77.0/75.7/75.6，MMLU-Pro 67.4/65.7/64.8，GPQA-Diamond 62.1/57.1/58.6，AIME 2025 21.3/20.6/21.1，HMMT 2025 12.5/11.3/11.3，PolyMath-en 43.6/41.3/41.5，LiveCodeBench v6 26.0/25.1/25.4；例外：MATH500 81.2 低于 GDN-H 83.0、EvalPlus 61.0 低于 62.5/62.6、LiveBench 45.2 低于 MLA 45.7（§5.5.1）[^src-moonshot-2025-kimi-linear]。

**长上下文 128k（Table 5）**：RULER 84.3 对 MLA 81.3 / GDN-H 80.5 / Kimi Linear (RoPE) 78.8；MRCR 29.6 对 22.6/23.9/22.0；HELMET-ICL 90.0 对 88.0/85.5/88.0；RepoQA 68.5 对 63.0/63.0/66.5；八项平均 54.5 对 MLA 52.2 / GDN-H 51.2 / RoPE 变体 51.8。低于 MLA 的两项：LongBench V2 35.0 对 36.1、Frames 58.8 对 60.5（§5.5.1）[^src-moonshot-2025-kimi-linear]。

**RL（Figure 6）**：相同 RLVR 算法与超参下，数学 RL 全程 Kimi Linear 训练准确率斜率高于 MLA 且差距扩大，MATH500 与 AIME 2025 测试曲线同步占优（§5.5.1）[^src-moonshot-2025-kimi-linear]。

**5.7T 最终模型（Tables 8–9，附录 D）**：base 对 Moonlight（16B 总参）：TriviaQA 75.2 对 66.2、MMLU-Pro 54.8 对 42.4、GSM8k 86.3 对 77.2、EvalPlus 64.9 对 50.3；instruct 版 RULER@128k 95.4、RULER@1M 94.8，MATH500 94.6 对 58.0，LiveCodeBench v6 45.7 对 11.9（Moonlight 因 8K 上下文上限未测长上下文与 AIME/HMMT/OJBench，表中记 "-"）[^src-moonshot-2025-kimi-linear]。

## 边界（论文自述）

纯线性设计受有限状态容量约束，长上下文检索仍是首要瓶颈，精确复制与细粒度选择落后于 full attention，模型以 3:1 混入 MLA 保留全局信息流（§1、§4、§7.1、§7.2）[^src-moonshot-2025-kimi-linear]。论文指出线性注意力（compression as intelligence）与稀疏注意力路线互补，未来可再混合（§7.1）[^src-moonshot-2025-kimi-linear]。1.4T 对照中，EvalPlus、MATH500、LiveBench、LongBench V2、Frames 落后于至少一个基线（Tables 3–5）[^src-moonshot-2025-kimi-linear]。1.16× 效率在 KDA 直接沿用 MLA 超参（仅固定 3:1）下取得，论文预期细调后曲线更优（§5.3）[^src-moonshot-2025-kimi-linear]。Figure 6 未给逐点数值，RL 比较基于训练与测试准确率曲线的斜率与差距（§5.5.1）[^src-moonshot-2025-kimi-linear]。Table 9 中 Moonlight-Instruct 因 8K 上下文上限多项未测，长上下文对照不完整（附录 D）[^src-moonshot-2025-kimi-linear]。

## 相关页面

[[gated-delta-rule]] 为谱系最近的母页：KDA 即其标量门 $\alpha_t \to \operatorname{Diag}(\alpha_t)$ 的细粒度扩展，TTT 目标函数相同（Table 7）；[[multi-head-latent-attention]] 为混合中的全局注意力支路（NoPE 化）；[[selective-ssm]] 与 [[mamba]]：Mamba2 同用标量衰减但无 delta 修正项，合成任务上全败（Figure 4、Table 7）；[[lightning-attention]]：同为线性注意力的分块实现（块内并行/块间递推同构），但无门控无修正，MiniMax-01 需 7:1 配 softmax 而 KDA 实测 3:1 即可（§5.2 Table 1）；[[minimax-sparse-attention]]：稀疏注意力路线的代表，保留全 KV cache 做 token 选择，与 KDA 的固定状态压缩互为对照（§7.1）；[[linear-attention-rnn-form]] 为结合律重排的原始出处；[[native-sparse-attention]] 与 [[moba-attention]] 为论文 related work 中与 KDA 对照的训练期块稀疏方法（§7.1）。

[^src-moonshot-2025-kimi-linear]: [[source-moonshot-2025-kimi-linear]]
