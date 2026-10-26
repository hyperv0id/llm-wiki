---
title: "FlashAttention: IO-aware 精确注意力"
type: technique
tags:
  - attention
  - efficiency
  - gpu-memory-hierarchy
  - tiling
  - online-softmax
created: 2026-10-27
last_updated: 2026-10-27
source_count: 1
confidence: medium
status: active
---

# FlashAttention

FlashAttention（Dao, Fu, Ermon, Rudra, Ré；NeurIPS 2022，arXiv:2205.14135v2）是 IO-aware 的**精确**注意力算法：数学结果与标准注意力一致（$O = \operatorname{softmax}(QK^{\top})V$）；它通过 tiling 将注意力计算移至片上 SRAM，避免向 HBM 写入 $N\times N$ 注意力矩阵，并用重算代替反向存储[^src-dao-2022-flashattention]。

## 问题：注意力是 memory-bound

GPU 内存层级差异悬殊：A100 的 HBM 容量为 40–80 GB、带宽 1.5–2.0 TB/s；片上 SRAM 每 SM 192 KB、共 108 个 SM，带宽约 19 TB/s——其速度快一个数量级，容量却小若干数量级[^src-dao-2022-flashattention]。随着算力增长快于带宽，多数操作受限于访存而非算术计算，softmax、dropout、mask 等逐元素与规约操作均为 memory-bound[^src-dao-2022-flashattention]。

标准注意力实现将中间矩阵存入 HBM：先写入 $S = QK^{\top}$，经 softmax 读回再写入 $P$，最后计算 $O = PV$（Algorithm 0），占用 $O(N^2)$ 显存；叠加 mask 与 dropout 时开销进一步加重[^src-dao-2022-flashattention]。论文指出，近似注意力虽将 FLOP 降至线性或近线性，但因忽视访存开销，多数并未实现 wall-clock 加速[^src-dao-2022-flashattention]。

## 与稀疏/近似路线的区分

稀疏与近似注意力（Sparse Transformer、Longformer 等改变依赖集合的稀疏方法，低秩近似，以及本库中的 [[logsparse-self-attention]]、[[native-sparse-attention]]、[[moba-attention]]、[[deepseek-sparse-attention]]）通过减少参与注意力的元素数降低计算量，输出原注意力的近似[^src-dao-2022-flashattention]。FlashAttention 不改依赖集合与数学结果，精确算出与标准实现相同的输出，加速来自减少 HBM 读写次数。两条路线正交：论文进一步将 block-sparse FlashAttention 作为稀疏路线的 IO-aware 实现，加速超过所有被测近似注意力方法[^src-dao-2022-flashattention]。

## 机制：tiling、online softmax 重缩放、重算

结合两项既有技术：tiling 将输入分块并增量执行 softmax 规约；recomputation 前向只存 softmax 归一化统计量，反向在 SRAM 里重算注意力块，替代从 HBM 读回 $N^2$ 中间矩阵[^src-dao-2022-flashattention]。整个前向（matmul、softmax、mask、dropout、matmul）融合为一个 CUDA kernel，一次装入、一次写回[^src-dao-2022-flashattention]。

### 前向数据流与维度变换

**维度符号**：$N$ 为序列长度，$d$ 为头维，$M$ 为 SRAM 容量（元素数）；行块 $B_r = \min(\lceil M/(4d)\rceil,\, d)$、列块 $B_c = \lceil M/(4d)\rceil$（Algorithm 1 行 1 取值，证明部分记 $B_c = \Theta(M/d)$、$B_r = \Theta(\min(M/d, d))$）；$T_r = N/B_r$、$T_c = N/B_c$；$\tau$ 为 softmax 缩放（通常 $1/\sqrt{d}$）。形状记 $[\text{行}, \text{列}]$，$\ell$、$m$ 为逐行 softmax 统计量（rowsum 与 running max）。

### Stage 1 · 外层循环装入 K、V 块

第 $j$ 个外层迭代把 $K_j, V_j$ 从 HBM 装入 SRAM，此后整个内层循环不再访问它们[^src-dao-2022-flashattention]。

$$\begin{aligned}
K_j, V_j &\xrightarrow{\text{HBM} \to \text{SRAM}} [B_c, d] \ \text{——外层循环 } j = 1..T_c \text{，每块只从 HBM 读一次}
\end{aligned}$$

### Stage 2 · 内层循环：块打分与局部统计

装入 $Q_i, O_i, \ell_i, m_i$（$[B_r, d]$ 与两个 $[B_r]$ 向量），全部计算在片上完成：

$$\begin{aligned}
S_{ij} &= \tau Q_i K_j^{\top} && \text{// } [B_r, \bcancel{d}]\times[\bcancel{d}, B_c]\to[B_r, B_c]\ \text{——消去 } d\text{，} Q \text{ 块对 } K \text{ 块的打分}\\
\tilde{m}_{ij} &= \operatorname{rowmax}(S_{ij}) && \text{// } [B_r, B_c] \xrightarrow{\text{沿 } B_c \text{ 维逐行取最大}} [B_r]\ \text{——本块的数值稳定基准}\\
\tilde{P}_{ij} &= \exp(S_{ij} - \tilde{m}_{ij}) && \text{// } [B_r, B_c]\ \text{逐元素}\ \text{——减去块内 max 防 exp 上溢}\\
\tilde{\ell}_{ij} &= \operatorname{rowsum}(\tilde{P}_{ij}) && \text{// } [B_r, B_c] \xrightarrow{\text{沿 } B_c \text{ 维逐行求和}} [B_r]\ \text{——本块的未归一化分母}
\end{aligned}$$

### Stage 3 · running 统计量更新与输出重缩放

新块的最大值可能超过旧 running max，旧统计量与旧输出须按比例折算到新基准：

$$\begin{aligned}
m_i^{\text{new}} &= \max(m_i,\ \tilde{m}_{ij}) && \text{// } [B_r]\ \text{逐元素}\ \text{——running max 只增不减}\\
\ell_i^{\text{new}} &= e^{m_i - m_i^{\text{new}}}\ell_i + e^{\tilde{m}_{ij} - m_i^{\text{new}}}\tilde{\ell}_{ij} && \text{// } [B_r]\ \text{——旧分母与新分母统一到新 max 基准后相加}\\
O_i &\leftarrow \operatorname{diag}(\ell_i^{\text{new}})^{-1}\!\left(\operatorname{diag}(\ell_i)\,e^{m_i - m_i^{\text{new}}} O_i + e^{\tilde{m}_{ij} - m_i^{\text{new}}} \tilde{P}_{ij} V_j\right) && \text{// } [B_r, \bcancel{B_c}]\times[\bcancel{B_c}, d]\to[B_r, d]\ \text{——旧输出降权、新块贡献并入，写回 HBM}
\end{aligned}$$

**不变量与保持**（Theorem 1，Appendix C 的归纳证明）：处理完前 $j$ 个 K/V 块后，HBM 中的值满足 $m^{(j)} = \operatorname{rowmax}(S_{:,:j})$、$\ell^{(j)} = \operatorname{rowsum}(\exp(S_{:,:j} - m^{(j)}))$、$O^{(j)} = \operatorname{softmax}(S_{:,:j})V_{:j}$。依据 exp 的平移不变性 $e^{x - m^{\text{new}}} = e^{m - m^{\text{new}}} \cdot e^{x - m}$，旧块统计量乘 $e^{m^{(j)} - m^{(j+1)}}$ 换至新基准，新块乘 $e^{\tilde{m} - m^{(j+1)}}$，相加后除以新分母 $\ell^{(j+1)}$，即得前 $j{+}1$ 块的完整 softmax。$j=0$ 时 $O=0, \ell=0, m=-\infty$ 成立；$j = T_c$ 时 $S_{:,:T_c} = S$，得到 $O = \operatorname{softmax}(QK^{\top})V$[^src-dao-2022-flashattention]。Theorem 1 同时给出 FLOPs 为 $O(N^2 d)$（两个块矩阵乘各 $O(B_r B_c d)$、共 $T_r T_c$ 次）、额外内存 $O(N)$（只存 $\ell, m$）[^src-dao-2022-flashattention]。

### Stage 4 · 反向：不存 $N^2$ 矩阵，块级重算

前向额外保存 $O, \ell, m$ 与伪随机数生成器状态 $R$（$O(N)$）；$S, P$ 与 dropout mask（各 $O(N^2)$）均不保存，反向由 Q、K、V 块在 SRAM 重算[^src-dao-2022-flashattention]。此设计采用选择性 gradient checkpointing：普通 checkpointing 以速度换显存，此处因 HBM 访问减少，即便 FLOPs 增加，反向依然更快（图 2 左）[^src-dao-2022-flashattention]。

$$\begin{aligned}
P_{ij} &= \operatorname{diag}(\ell_i)^{-1}\exp(S^{\text{masked}}_{ij} - m_i) && \text{// } [B_r, B_c]\ \text{——用前向存的 } \ell_i, m_i \text{ 重建归一化概率，免存 } P\\
Z_{ij} &\xrightarrow{\text{由 PRNG 状态 } R \text{ 重放}} [B_r, B_c]\ \text{——dropout mask 按 Same state 重新生成，免存 } O(N^2) \text{ 掩码}\\
dV_j &\leftarrow d\tilde{V}_j + (P^{\text{dropped}}_{ij})^{\top} dO_i && \text{// } [\bcancel{B_r}, B_c]\times[\bcancel{B_r}, d]\to[B_c, d]\ \text{——SRAM 上累加，外层结束才写 HBM}\\
D_i &= \operatorname{rowsum}(dO_i \circ O_i) && \text{// } [B_r, d] \xrightarrow{\text{沿 } d \text{ 维逐行求和}} [B_r]\ \text{——把对长度 } N \text{ 的行规约改写成 } d \text{ 维点积 } do_i^{\top} o_i\\
dS_{ij} &= P_{ij} \circ (dP_{ij} - D_i), \quad dP_{ij} = dO_i V_j^{\top} && \text{// } [B_r, B_c]\ \text{逐元素}\ \text{——softmax 雅可比 } \operatorname{diag}(P)-PP^{\top} \text{ 展开后的精确形式（Appendix B.2）}\\
dQ_i &\leftarrow dQ_i + \tau\, dS_{ij} K_j, \qquad d\tilde{K}_j \leftarrow d\tilde{K}_j + \tau\, dS_{ij}^{\top} Q_i && \text{// } [B_r, \bcancel{B_c}]\times[\bcancel{B_c}, d]\to[B_r, d]\ \text{；}\ [B_c, \bcancel{B_r}]\times[\bcancel{B_r}, d]\to[B_c, d]
\end{aligned}$$

$D_i$ 的改写（观察 2）与 dropout 的 PRNG 重放（观察 1）使反向内存仅随 $N$ 线性增长：前者避免为规约加载整行 $P, dP$，后者避免保存 $O(N^2)$ 掩码（Algorithm 4 前的观察、B.2 式 (4)）[^src-dao-2022-flashattention]。Rabe & Staats 的 memory-efficient attention 以峰值显存为目标，保存每块临时输出（$K$ 份）并用 gradient checkpointing 反传；FlashAttention 则以 HBM 访问次数为目标，增量更新单份输出，反向经解析化简只重算注意力矩阵，因而取得 2–4× 加速，而前者与标准实现相当或更慢（B.5）[^src-dao-2022-flashattention]。

## HBM 访问次数分析

**标准实现的计数**（Theorem 2 证明，Appendix C）分三步：算 $S = QK^{\top}$ 读 Q、K 写 S，$\Theta(Nd + N^2)$；softmax 读 S 写 P，$\Theta(N^2)$；算 $O = PV$ 读 P、V 写 O，$\Theta(Nd + N^2)$。合计 $\Theta(Nd + N^2)$。§1 引言以 $\Omega(Nd + N^2)$ 下界对照 FlashAttention 的 $O(N^2d^2M^{-1})$，Theorem 2 与附录 C 证明对两者均用 $\Theta$；Proposition 3 给出下界「不存在对所有 $M$ 达到 $o(N^2d^2M^{-1})$ 的精确算法」[^src-dao-2022-flashattention]。

**FlashAttention 的计数**：K、V 的每个元素从 HBM 读一次（外层行 6），$\Theta(Nd)$；Q、O（连同 $\ell, m$）在外层各块迭代均完整扫描一遍，共 $T_c$ 遍，每遍 $\Theta(Nd)$，计 $\Theta(Nd\, T_c)$。块大小受 SRAM 约束：$B_c d = O(M)$、$B_r d = O(M)$、$B_r B_c = O(M)$，取 $B_c = \Theta(M/d)$、$B_r = \Theta(\min(M/d, d))$，得 $T_c = N/B_c = \Theta(Nd/M)$，总计

$$\Theta(N d\, T_c) = \Theta\!\left(\frac{N^2 d^2}{M}\right).$$

当 $d = 64\text{–}128$、$M \approx 100\,\text{KB}$ 时 $d^2 \ll M$，远小于 $N^2$（Theorem 2）[^src-dao-2022-flashattention]。反向同量级（Theorem 5）：K、V 各读一次，dK、dV 各写一次，Q、O、dO、dQ 各扫描 $T_c$ 遍，仍为 $\Theta(N^2 d^2 M^{-1})$[^src-dao-2022-flashattention]。

**下界**（Proposition 3）：不存在对所有 $M \in [d, Nd]$ 均以 $o(N^2 d^2 M^{-1})$ 次 HBM 访问计算精确注意力的算法。证明取 $M = \Theta(Nd)$：此时 $o(N^2d^2/M) = o(Nd)$，输入 Q、K、V 与输出 O 共 $Nd$ 个元素本在 HBM，精确计算至少需 $\Omega(Nd)$ 次访问，产生矛盾[^src-dao-2022-flashattention]。

**块稀疏扩展**（Proposition 4）：mask 取块形式 $\mathcal{M} \in \{0,1\}^{N/B_r \times N/B_c}$，跳过零块后 HBM 访存量为 $\Theta(Nd + N^2 d^2 M^{-1} s)$（$s$ 为非零块比例）；$s = N^{-1/2}$ 时为 $\Theta(N\sqrt{N})$，$s = N^{-1}\log N$ 时为 $\Theta(N\log N)$。实验采用可近似任意稀疏模式的固定 butterfly 稀疏（§3.3）[^src-dao-2022-flashattention]。

## 证据

实验 GPU 为 A100（另测 RTX 3090、T4），训练使用 8 卡。

**单次注意力的访存与时间**（图 2 左：GPT-2 medium，seq 1024、head dim 64、16 头、batch 64，前向+反向）：FlashAttention FLOPs 75.2 GFLOPs 对标准 66.6（重算代价 +13%），HBM 读写 4.4 GB 对 40.3 GB（少至多 9×），运行时间 7.3 ms 对 41.7 ms——FLOPs 更高而时间更短，直接证明访存是主因[^src-dao-2022-flashattention]。图 2 中：固定其余设置只增大 $B_c$，HBM 读写与运行时间同降；超过 256 后瓶颈移到算术等其他因素，且更大的块放不进 SRAM[^src-dao-2022-flashattention]。图 1 右：GPT-2 注意力计算较 PyTorch 实现加速 7.6×[^src-dao-2022-flashattention]。

**BERT-large 训练**（Table 1：8×A100-80GB，MLPerf 1.1 同一初始化，训练到 masked LM 验证准确率 72.0%，LAMB lr 3.75e-3、batch 448、至多 7100 步，FP16，10 次平均）：Nvidia MLPerf 1.1 记录 20.0±1.5 分钟，FlashAttention 17.4±1.4 分钟，快 15%，创单卡节点最快 BERT 训练速度[^src-dao-2022-flashattention]。

**GPT-2 训练**（Table 2：8×A100-40GB，OpenWebText，400K 步，AdamW，Megatron 配方）：困惑度与基线持平（模型定义未变）；small 对 HuggingFace 2.7 天对 9.5 天（3.5×）、对 Megatron-LM 2.7 天对 4.7 天（2.0×）；medium 6.9 天对 21.0 天（3.0×）、对 11.5 天（1.8×）。注意口径：摘要只写「3×」；§4 开头的贡献列表写「对 HuggingFace 至多 3×、对 Megatron 1.8×」，§4.1 正文与 Table 2 caption 写「对 Megatron 至多 1.7×」，与表内括号（small 2.0×、medium 1.8×）不完全一致——三处并列照录（详见 [[ingest-reports/2026-10-27-flashattention|WHY 报告]]）[^src-dao-2022-flashattention]。

**Long-Range Arena**（Table 3：vanilla Transformer 替换注意力实现，任务 seq 1024–4096，混合精度；加速比为五任务 wall-clock 几何平均，Appendix E.3；基线取原报告与复现更优者）：FlashAttention 平均准确率 59.8（全表最高，vanilla 59.3）、加速 2.4×；block-sparse FlashAttention 59.6、加速 2.8×（全表最快）；对照 Linformer 54.9/2.5×、Linear Attention 59.6/2.3×、Performer 58.9/1.8×、Local Attention 56.0/1.7×、Reformer 57.6/1.3×、Smyrf 57.9/1.7×。精确计算的 FlashAttention 在平均准确率与速度上均不低于上述近似基线[^src-dao-2022-flashattention]；注 3 补充 LRA 准确率对调参流程敏感[^src-dao-2022-flashattention]。

**长上下文语言建模**（Table 4：GPT-2 small，8×A100-40GB）：Megatron-LM 上下文 1K、ppl 18.2、4.7 天；FlashAttention 上下文 4K、ppl 17.5、3.6 天——序列长 4× 仍快 30%、困惑度好 0.7（1K/2K/4K 分别为 18.2/17.6/17.5）[^src-dao-2022-flashattention]。

**长文档分类**（Table 5：预训练 RoBERTa 重复位置嵌入扩窗，micro F1）：MIMIC-III（平均 2395 token、最长 14562）从 seq 512 的 52.8 升至 16K 的 57.1（+4.3）；ECtHR（平均 2197、最长 49392）从 72.2 升至 8K 的 80.7（+8.5，16K 回落至 79.2）。摘要报告的「6.4 点提升」为两任务平均[^src-dao-2022-flashattention]。

**Path-X / Path-256**（Table 6：先在 Path-64 预训练、位置嵌入按空间插值迁移）：此前所有 Transformer 在这两个任务上要么 OOM 要么处于随机水平（表中基线均记「7」，未注口径）；FlashAttention 在 seq 16K 的 Path-X 上得 61.4，block-sparse FlashAttention 在 seq 64K 的 Path-256 上得 63.1，Transformer 首次在这两个任务上超过随机水平[^src-dao-2022-flashattention]。

**注意力算子基准**（图 3；Appendix E.6 设置：单卡 A100 40GB、8 头 × dim 64、batch 16、dropout 0.1、padding mask、FP16）：运行时间随 seq 二次增长，FlashAttention 在 128–2K 常见长度上比 PyTorch 快至多 3×；近似/稀疏方法运行时间随 seq 线性增长，但在 seq 512–1024 间与 FlashAttention 交叉变快（如 Linformer）；block-sparse FlashAttention 在所有长度上快于所有精确、稀疏与近似实现[^src-dao-2022-flashattention]。显存：FlashAttention 与 block-sparse 版相同、随 seq 线性增长，比精确基线至多省 20×；seq 64K 时除 Linformer 外基线全 OOM，此时 FlashAttention 仍比 Linformer 省 2×（Table 21：seq 4096 处 PyTorch 17,024 MB 对 FlashAttention 836 MB，即 20×；Reformer 高达 12,067 MB；seq 8192 起 PyTorch/Megatron OOM）[^src-dao-2022-flashattention]。

**与 Apex FMHA 对照**（Table 7：A100-SXM4-40GB，BERT-large 尺寸 batch 64、16 头、dim 64，mask+dropout）：FMHA 融合 dropout(softmax(mask(QK^T)))V，但前向将注意力矩阵写入 HBM，不省显存，且仅支持 head dim 64、A100 与 seq $\leq 512$；FlashAttention 前向略快（512 处 0.81 对 1.14 ms），反向因重算略慢（2.00 对 1.81 ms），两者合并后 256 处快 8%、512 处快 5%、128 处慢 4%（E.4）[^src-dao-2022-flashattention]。

**跨硬件**（E.5，对 PyTorch 的加速）：A100 加速 2–4×，含 dropout 与 mask 时因 kernel fusion 加速更高；head dim 128 迫使块变小、加速下降，causal mask 下仍至多 3×；RTX 3090 带宽更低（约 900 GB/s），加速升至 2.5–4.5×；T4 的 SRAM 更小、块更小、加速更低，符合 IO 复杂度分析[^src-dao-2022-flashattention]。

## 边界

**块大小适配 SRAM**：$B_r, B_c$ 由 $M$ 与 $d$ 决定（$B_c = \lceil M/(4d)\rceil$），$d = 128$ 或小 SRAM 硬件上块变小、HBM 访问上升、加速下降（图 2 中、E.5）[^src-dao-2022-flashattention]。**工程边界**：新注意力变体需手写 CUDA kernel，层级低、工程量大且不可跨 GPU 架构移植；作者以此论证需要 Halide 式「高级语言编译到 IO-aware 实现」（§5）[^src-dao-2022-flashattention]。**单 GPU**：分析限于单 GPU 内存层级；多 GPU 带来跨卡 HBM 新层级，论文留作未来方向（§5、D.2）[^src-dao-2022-flashattention]。**近似扩展**：块稀疏要求 mask 为块形式，实验用固定 butterfly 模式；任意注意力模式的 IO-aware 实现不在本文范围（§3.3）[^src-dao-2022-flashattention]。**数值口径**：GPT-2 对 Megatron 的加速在论文内有三种口径——§4 开头列表 1.8×、§4.1 正文与表 2 caption 至多 1.7×、表内括号 small 2.0× / medium 1.8×（表内括号与天数比值 4.7/2.7≈1.7、11.5/6.9≈1.7 亦不一致）；LRA 基线准确率取原报告与复现的较优值；页面均照录原文数字。

## 相关页面

[[source-dao-2022-flashattention]] 为源摘要。改变依赖集合的稀疏路线（近似）包括 [[logsparse-self-attention]]、[[native-sparse-attention]]、[[moba-attention]]、[[deepseek-sparse-attention]]；Stage 4 重算是 [[activation-checkpointing]] 在注意力上的具体实例——以重算换显存，靠减少 HBM 访问兼顾运行速度。节省显存的另一路径是推理期 KV：[[grouped-query-attention]]、[[multi-head-latent-attention]] 压缩 KV 头与维度，[[paged-attention]] 管理 KV cache 放置；三者与 FlashAttention 作用于不同对象（KV cache 对比激活与注意力中间矩阵）。

[^src-dao-2022-flashattention]: [[source-dao-2022-flashattention]]
