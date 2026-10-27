---
title: "Native Sparse Attention (NSA)"
type: technique
tags:
  - attention
  - sparse-attention
  - long-context
  - llm
  - hardware
created: 2026-10-24
last_updated: 2026-10-27
source_count: 5
confidence: medium
status: active
---

# Native Sparse Attention (NSA)

DeepSeek-AI、北京大学与华盛顿大学（arXiv:2502.11089v2，2025-02）提出的稀疏注意力架构，将每个 query 的前序 KV 替换为块压缩（compression）、块选择（selection）与滑动窗口（sliding window）三路重映射，各自 softmax 后按 sigmoid 门控加权求和，稀疏模式在预训练时确立而非推理期插入（式 3–5、§3.3）[^src-2502-11089-nsa]。

## 问题定位

既有做法的失效归为四点（§2.1、§2.2）[^src-2502-11089-nsa]：

其一，阶段受限：H2O 只在解码期稀疏，prefill 仍需算注意力图与建索引；MInference 只在 prefill 稀疏。至少一个阶段的开销与 full attention 相当，prefill 为主的负载（书摘、代码补全）与 decode 为主的负载（长思维链）各自拿不到加速（§2.1）。其二，与 GQA/MQA 不兼容：Quest 一类方法逐 head 独立选 KV 子集，MHA 下计算与访存同时稀疏；GQA 下 KV cache 访存量由组内各 query head 选中集合的并集决定，计算量降而访存量不降（§2.1）。这条共享谱系的端点是 [[multi-query-attention]]：K/V 只留一份、全部 query head 共用，解码期 K/V 重读字节降为 $1/H$ 而乘加数不变（§2.4.1）[^src-shazeer-2019-mqa][^src-2502-11089-nsa]。其三，事后稀疏的质量损失：推理期插入稀疏迫使模型偏离预训练优化轨迹；转引 Chen et al. 2024b 表明，top 20% 注意力只覆盖总注意力分数的 70%（§2.2）。其四，训练期插入的障碍：ClusterKV 的 k-means 与 MagicPIG 的 SimHash 在计算图中造成不连续；HashAttention 一类逐 token 选择需非连续读取 KV cache，与 FlashAttention 依赖连续块访存冲突，退化为低硬件利用率（§2.2）。

## 框架

给定 query $q_t$ 与前序 $\mathbf{k}_{:t},\mathbf{v}_{:t}$，先重映射，再在重映射集合上做注意力（式 3–6）[^src-2502-11089-nsa]：

$$\tilde K_t=f_K(q_t,\mathbf{k}_{:t},\mathbf{v}_{:t}),\qquad \tilde V_t=f_V(q_t,\mathbf{k}_{:t},\mathbf{v}_{:t}),\qquad o^*_t=\operatorname{Attn}\!\left(q_t,\tilde K_t,\tilde V_t\right)$$

$$o^*_t=\sum_{c\in\{cmp,slc,win\}} g^c_t\cdot\operatorname{Attn}\!\left(q_t,\tilde K^c_t,\tilde V^c_t\right),\qquad N_t=\sum_{c}\operatorname{size}\!\left[\tilde K^c_t\right]\ll t$$

$g^c_t\in[0,1]$ 由输入特征经 MLP 与 sigmoid 得到，三支路门控互相独立，式（5）未归一化三者之和（§3.2）[^src-2502-11089-nsa]。$N_t$ 是重映射后的 key 总数，稀疏性由 $N_t\ll t$ 保证。

## 三分支机制

**Compression（§3.3.1）**：每 $l$ 个连续 key 由带块内位置编码的可学习 MLP $\varphi$ 归约为一个块级 key，相邻块按步长 $d$ 滑动，取 $d<l$ 减轻信息碎片化。默认 $l=32$、$d=16$（式 7、§4.1）[^src-2502-11089-nsa]。

**Selection（§3.3.2）**：按空间连续块选择，原因有二：GPU 连续访存吞吐远高于随机下标读取，块式访问方能用满 Tensor Core；注意力分数本身呈空间连续（Figure 8 给出 full attention 注意力图的块聚类形态，§6.2）。**重要性分直接复用压缩支路的注意力分数** $p^{cmp}_t=\operatorname{Softmax}(q_t^{\top}\tilde K^{cmp}_t)$（式 8），不引入独立打分网络与辅助损失；压缩块与选择块不对齐时按空间关系累加（式 9），GQA 组内各 head 分数求和以保证全组选同一批块（式 10），按降序取前 $n$ 块（式 11–12）。默认 $l'=64$、$n=16$，固定激活首块与 2 个局部块（§4.1）[^src-2502-11089-nsa]。

**Sliding Window（§3.3.3）**：保留最近 $w=512$ 个 token 的独立 K/V 以隔离局部模式：局部模式学得更快，不单列会抢占另两支路的梯度；三支路各持独立 K/V，防止局部与长程模式间产生梯度干扰与 shortcut（§3.3.3）[^src-2502-11089-nsa]。

## 数据流与维度变换

**维度符号**：$s$ 为已缓存序列长，$t$ 为当前 query 位置（解码时 $s=t$）；$l=32$ 压缩块长；$d=16$ 压缩块滑动步长；$l'=64$ 选择块长；$n=16$ 选中块数；$w=512$ 窗口；$d_k=d_q=192$、$d_v=128$；$H=16$ 为各 GQA 组的 query head 数，共 4 组（§4.1、§5）[^src-2502-11089-nsa]。形状记作 $[\text{行},\text{列}]$，与论文记号一致；论文按单 query 书写，组内批量展开标注 [INFERENCE]。

### Stage 1 · 压缩前序 KV

$$\begin{aligned}
\tilde K^{cmp}_t &= \left\{\varphi\!\left(k_{id+1:\,id+l}\right)\right\}_{0\le i\le\left\lfloor (t-l)/d\right\rfloor} && \text{// } [d_k, t]\xrightarrow{\text{沿 } t \text{ 维滑窗：窗长 } l \text{、步长 } d} [d_k, \lfloor (t-l)/d\rfloor+1]\ \text{——} t \text{ 个 key 压成约 } t/16 \text{ 个块摘要}\\
\tilde V^{cmp}_t &= \left\{\psi\!\left(v_{id+1:\,id+l}\right)\right\}_{0\le i\le\left\lfloor (t-l)/d\right\rfloor} && \text{// } [d_v, t]\xrightarrow{\text{同构滑窗}} [d_v, \lfloor (t-l)/d\rfloor+1]\ \text{——value 侧同构；原文只称 analogous，} \psi \text{ 记号与是否共享 } \varphi \text{ 均未写明 [INFERENCE]}
\end{aligned}$$

$\varphi$ 是带块内位置编码（intra-block position encoding）的可学习 MLP，使块摘要保留块内顺序信息；value 侧原文仅称 analogous formulation，未设独立记号，未说明是否与 $\varphi$ 共享（§3.3.1）[^src-2502-11089-nsa]。原文对块数有两处口径：式（7）索引上界为 $\lfloor (t-l)/d\rfloor$、张量维写作 $\lfloor (t-l)/d\rfloor$，式（8）分数向量长度为 $\lfloor (t-l)/d\rfloor+1$；此处按 $+1$ 计块数。

### Stage 2 · 压缩支路打分（同时充当块重要性分）

$$\begin{aligned}
p^{cmp}_t &= \operatorname{Softmax}\!\left(q_t^{\top}\tilde K^{cmp}_t\right) && \text{// } [1,\bcancel{d_k}]\times[\bcancel{d_k},\lfloor (t-l)/d\rfloor+1]\to[1,\lfloor (t-l)/d\rfloor+1]\ \text{——式(8)，向量长度即压缩块数（原文记 } \lfloor (t-l)/d\rfloor+1\text{）}\\
o^{cmp}_t &= \tilde V^{cmp}_t\,{p^{cmp}_t}^{\top} && \text{// } [d_v,\bcancel{\lfloor (t-l)/d\rfloor+1}]\times[\bcancel{\lfloor (t-l)/d\rfloor+1},1]\to[d_v,1]\ \text{——压缩支路自身的输出}
\end{aligned}$$

### Stage 3 · 块重要性 → top-$n$ → gather 选中块

$$\begin{aligned}
p^{slc}_t[j] &= \sum_{a=0}^{l'/d-1}\ \sum_{b=0}^{l/d-1} p^{cmp}_t\!\left[\tfrac{l'}{d}j-a-b\right] && \text{// } [\lfloor (t-l)/d\rfloor+1]\xrightarrow{\text{按块对齐累加}}[\lfloor t/l'\rfloor]\ \text{——式(9)，每个选择块汇总 } (l'/d)(l/d)=8 \text{ 个压缩分}\\
p'^{slc}_t &= \sum_{h=1}^{H}p^{slc,(h)}_t && \text{// } [\lfloor t/l'\rfloor]\times H\xrightarrow{\text{沿 } H \text{ 个 head 求和}}[\lfloor t/l'\rfloor]\ \text{——式(10)，组内共用一份选块结果}\\
\mathcal{I}_t &= \left\{i\ \middle|\ \operatorname{rank}\!\left(p'^{slc}_t[i]\right)\le n\right\} && \text{// } [\lfloor t/l'\rfloor]\xrightarrow{\text{降序 rank 取前 } n}[n]\ \text{——式(11)，} |\mathcal{I}_t|=n=16\\
\tilde K^{slc}_t &= \operatorname{Cat}\!\left\{k_{il'+1:\,(i+1)l'}\ \middle|\ i\in\mathcal{I}_t\right\} && \text{// } [d_k,\lfloor t/l'\rfloor l']\xrightarrow{\text{按 } \mathcal I_t \text{ 索引块}}[d_k,nl']\ \text{——式(12)，}16\times64=1024 \text{ 个 key}
\end{aligned}$$

式（9）原文用 $m,n$ 作求和下标，此处改用 $a,b$ 避免与选中块数 $n$ 冲突[^src-2502-11089-nsa]。式（9）适用前提为 $l\le l'$、$d\mid l$、$d\mid l'$（§3.3.2，默认超参满足）。默认 $l'=64,l=32,d=16$ 下，8 项下标落在 $4j-4$ 至 $4j$；因 $d<l$ 相邻压缩窗重叠，实为 5 个不同压缩分，权重为 $1,2,2,2,1$ [INFERENCE]。$\mathcal I_t$ 是逐 query（组内逐位置）变化的离散集合，块下标不参与梯度；选中块的 K/V 作为普通 KV 参与 softmax，梯度按常规路径回传 [INFERENCE]。

### Stage 4 · 滑动窗口支路

$$\begin{aligned}
\tilde K^{win}_t = k_{t-w:\,t},\quad \tilde V^{win}_t = v_{t-w:\,t} && \text{// } [d_k,w]=[192,512]\ \text{——最近 } w \text{ 个 token，无重映射参数}
\end{aligned}$$

### Stage 5 · 三支路各自归一化后门控合并

$$\begin{aligned}
\mathbf{g}_t &= \sigma\!\left(\operatorname{MLP}(x_t)\right) && \text{// } [d,1]\to[3,1]\ \text{——式(5) 的 } g^{cmp}_t,g^{slc}_t,g^{win}_t\text{；原文只说由输入特征经 MLP 与 sigmoid 得到，MLP 结构未给}\\
O^{c}_t &= \tilde V^{c}_t\operatorname{softmax}\!\left(\tilde K^{c\top}_tq_t/\sqrt{d_k}\right) && \text{// } [d_v,\bcancel{N^c_t}]\times[\bcancel{N^c_t},1]\to[d_v,1]\ \text{——} c\in\{cmp,slc,win\}\text{ 各用一套分母}\\
o^*_t &= \sum_{c}g^{c}_t\,O^{c}_t && \text{// } [d_v,1]\times 3\xrightarrow{\text{按门控加权求和}}[d_v,1]\ \text{——输出通道仍为 } d_v
\end{aligned}$$

**每 query 读取 key 数的推导**。解码缓存长度为 $s=t$，三支路各读：压缩支路按式（7）索引集 $0\le i\le\lfloor(s-l)/d\rfloor$ 读 $\lfloor(s-l)/d\rfloor+1$ 个块摘要（§5.2 记作 at most $\lfloor(s-l)/d\rfloor$，未计边界块）；选择支路读 $n$ 块、每块 $l'$ 个 token，共 $nl'$；窗口支路读 $w$ 个 token。三项相加：

$$N_t=\left\lfloor\frac{s-l}{d}\right\rfloor+1+nl'+w .$$

代入 $l=32,d=16,l'=64,n=16,w=512$、$s=65{,}536$ 得 $\lfloor(65{,}536-32)/16\rfloor+1+nl'+w=4095+1536=5631$；§5.2 文字口径（无 $+1$）为 $4094+1536=5630$，Table 4 记 5632。Table 4 四档数值按 $\lfloor s/d\rfloor+nl'+w$ 计算（8k/16k/32k/64k 的压缩项分别为 $512/1024/2048/4096$，选择与窗口两项 $nl'+w=1536$ 三处一致）。差 2 全部来自压缩支路的取整口径：$l/d=32/16=2$ 为整数，恒有 $\lfloor s/d\rfloor=\lfloor(s-l)/d\rfloor+2$，Table 4 按不减 $l$ 的 $\lfloor s/d\rfloor$ 计（4096），比 §5.2 文字口径（4094）多 2、比严格按式（7）索引集的口径（4095）多 1 [^src-2502-11089-nsa]。

## 与 full attention 的差别

式（3）（4）用重映射后的 $\tilde K_t,\tilde V_t$ 替换原始 KV 并要求 $N_t\ll t$，故 NSA 输出不是 full attention 输出，也不存在「选满即退化」的开关（§3.2）[^src-2502-11089-nsa]。三支路各自 softmax 后经 sigmoid 门控加权，与把不同来源的 KV 拼入同一 softmax 分母分属两种权重口径：前者各支路权重仅在支路内竞争，后者全部候选在同一分母下竞争。[[mita-attention]] 属后者（拼接后单次 softmax）[^src-2602-01219-mita]。

两种口径不等价 [INFERENCE]：设两支路、每支路 2 个 key，分数均为 $(2,0)$，$e^2\approx 7.39$。NSA 口径下每支路 softmax 分母为 $7.39+1=8.39$，支路内权重为 $0.881/0.119$，门控全为 1 时总权重和为 2（等于支路数，门控不归一）；拼接口径将 4 个分数放入同一 softmax，分母为 $2\times(7.39+1)=16.78$，相同分数的权重变为 $0.440/0.060$，总和恒为 1。本例两支路分数相同，拼接分母恰为支路内分母的 2 倍，每个权重精确减半。支路内归一化权重不可跨支路比较，跨支路尺度分配由 $g^c_t$ 承担，故「各自 softmax + 门控」与「拼接后单次 softmax」之间不存在重参数化关系。

「与 full attention 可比」只在实验中处理：通用评测中多数样本长度落在各稀疏基线的局部窗口内，各基线在该设置下等价于 full attention，故只比较 NSA 与 Full Attention（§4.2）[^src-2502-11089-nsa]。短序列下（$t\le w$ 时压缩与选择两支路仍照常输出）NSA 的行为，论文未讨论。

## 硬件对齐与 kernel（§3.4、§5）

**优化目标的分野**（§3.1）：训练与 prefill 的批量矩阵乘算术强度高，compute-bound，目标是减计算；自回归解码每步生成一个 token 却要读整个 KV cache，算术强度低，memory-bound，目标是减访存。§1 引用的理论估计：64k 上下文解码时 softmax 注意力占总延迟的 70–80%[^src-2502-11089-nsa]。这条「精确注意力也该减访存」路线的起点是 [[flash-attention]]：不改依赖集合、输出与标准注意力相同，加速只来自减少 HBM 读写次数[^src-dao-2022-flashattention]。

**kernel 结构**（§3.4、Figure 3）：压缩与窗口支路兼容 FlashAttention-2，仅选择支路需专门实现。若按 FlashAttention 将时间连续的 query 块装入 SRAM，块内各 query 选中的 KV 块互不相交，访存低效。NSA 改为**按位置装入组内全部 query head**：外层在 Triton grid 上运行 query/output 循环，内层对位置 $t$ 载入 $Q\in\mathbb{R}^{[H,d_k]}$ 与全组共享的块索引 $\mathcal I_t$，再顺序将 $K\in\mathbb{R}^{[B_k,d_k]}$、$V\in\mathbb{R}^{[B_k,d_v]}$ 载入 SRAM，$B_k\mid l'$。由此带来两项收益：组内共享免去重复 KV 传输；内层长度（正比于 $n$）在 query 块间几乎相同，便于 grid 调度器做负载均衡[^src-2502-11089-nsa]。

**解码访存量**（Table 4，8×A100，$g=4$、$H=16$、$d_k=192$、$d_v=128$）：以「等价 token 数」记每次注意力操作的访存量，预期加速比与访存量近似成反比[^src-2502-11089-nsa]。

| 上下文长度 | Full Attention | NSA | 预期加速比 |
|---|---|---|---|
| 8192 | 8192 | 2048 | 4× |
| 16384 | 16384 | 2560 | 6.4× |
| 32768 | 32768 | 3584 | 9.1× |
| 65536 | 65536 | 5632 | 11.6× |

**实测加速比**（Triton 实现的 NSA kernel vs Triton 实现的 FlashAttention-2，Figure 6）：forward 在 8k/16k/32k/64k 分别为 2.1× / 3.8× / 6.3× / 9.0×；backward 分别为 1.1× / 2.0× / 3.4× / 6.0×；解码在 64k 时为 11.6×（§5.2、Figure 1）[^src-2502-11089-nsa]。

## 训练

**可训练设计选择**（§2.2、§6.1）：选块得分取自压缩支路的注意力分数（式 8），无需额外打分网络与辅助损失；块下标离散且不接收梯度，梯度路径上也没有独立不可微模块。在 3B 模型上对照了两种替代选块方案（Figure 7）：辅助损失方案为每 token 加额外 query、每块加代表 key，以块内注意力均值池化分布的 KL 监督（同 SeerAttention 思路）；无参数启发方案按 Quest 用 query 与 key chunk 逐坐标 min-max 之积选块，另测试了先跑 1000 步 full attention 再切换的 cold start。两者损失均高于 NSA 与 Full Attention，曲线无具体数值[^src-2502-11089-nsa]。ClusterKV 等聚类方案被排除，主因在于动态聚类的计算开销、聚类不均在 MoE 专家并行下的负载失衡，以及强制周期性重聚类与分块顺序训练带来的实现约束（§6.1）[^src-2502-11089-nsa]。

**单一训练阶段**：NSA 与 Full Attention 基线按同配方从头训练，不设 dense warm-up 或 dense→sparse 切换，稀疏选块从第 0 步生效（§4.1）；1000 步 full attention 的 cold start 仅用于上述启发式方案。

**预训练配置**（§4.1）：GQA + MoE 骨干，27B 总参数、3B 激活，30 层，hidden 2560，GQA 4 组共 64 个 head，$d_q=d_k=192$、$d_v=128$；DeepSeekMoE 含 72 个 routed expert 加 2 个 shared expert，top-k 取 6，首层 MoE 换成 SwiGLU MLP 以稳定训练。两模型均在 8k 文本上预训练 $270$B token，再在 32k 文本上用 YaRN 适配长上下文继续训练并经监督微调；两者训练到完全收敛[^src-2502-11089-nsa]。摘要与 §1 写 260B token，§4.1 写 270B，两处不一致，见 [[source-2502-11089-nsa]][^src-2502-11089-nsa]。

**推理能力后训练**（§4.3）：从 R1 蒸馏，用 10B token 的 32k 数学推理轨迹做 SFT，得到 Full Attention-R 与 NSA-R 两个对照模型；AIME 24 上采样温度 0.7、top-$p$ 0.95，每题 16 条回答取均值[^src-2502-11089-nsa]。

## 证据

### 通用基准（Table 1，预训练后）

| 模型 | MMLU 5-shot | MMLU-PRO 5-shot | CMMLU 5-shot | BBH 3-shot | GSM8K 8-shot | MATH 4-shot | DROP F1 1-shot | MBPP Pass@1 3-shot | HumanEval Pass@1 0-shot | Avg. |
|---|---|---|---|---|---|---|---|---|---|---|
| Full Attn | 0.567 | 0.279 | 0.576 | 0.497 | 0.486 | 0.263 | 0.503 | 0.482 | 0.335 | 0.443 |
| NSA | 0.565 | 0.286 | 0.587 | 0.521 | 0.520 | 0.264 | 0.545 | 0.466 | 0.348 | 0.456 |

NSA 在 9 项中 7 项更高，平均 +0.013，仅 MMLU（−0.002）与 MBPP（−0.016）更低；论文点出两项差值：DROP +0.042、GSM8K +0.034（§4.3）[^src-2502-11089-nsa]。

### 长上下文（Table 2，LongBench）

所有稀疏基线的每 query 激活 token 预算统一设为 2560；§4.3 将其记为 NSA 在 32k 序列长度下的平均激活量，含前 128 个 token 与 512 个局部 token[^src-2502-11089-nsa]。Table 4 在 32768 长度下记 NSA 访存量为 3584 个等价 token，与 §4.3 的 2560 不同（2560 与 Table 4 中 16384 长度下的数值相同）；两处记录并列。

| 模型 | MFQA-en | MFQA-zh | Qasper | HPQ | 2Wiki | GovRptDur | （列名缺失） | PassR-en | PassR-zh | LCC | Avg. |
|---|---|---|---|---|---|---|---|---|---|---|---|
| H2O | 0.428 | 0.429 | 0.308 | 0.112 | 0.101 | 0.231 | 0.208 | 0.704 | 0.421 | 0.092 | 0.303 |
| InfLLM | 0.474 | 0.517 | 0.356 | 0.306 | 0.250 | 0.277 | 0.257 | 0.766 | 0.486 | 0.143 | 0.383 |
| Quest | 0.495 | 0.561 | 0.365 | 0.295 | 0.245 | 0.293 | 0.257 | 0.792 | 0.478 | 0.135 | 0.392 |
| Exact-Top | 0.502 | 0.605 | 0.397 | 0.321 | 0.288 | 0.316 | 0.291 | 0.810 | 0.548 | 0.156 | 0.423 |
| Full Attn | 0.512 | 0.623 | 0.409 | 0.350 | 0.305 | 0.324 | 0.294 | 0.830 | 0.560 | 0.163 | 0.437 |
| NSA | 0.503 | 0.624 | 0.432 | 0.437 | 0.356 | 0.307 | 0.341 | 0.905 | 0.550 | 0.232 | 0.469 |

NSA 平均分 0.469，比 Full Attention 高 0.032、比 Exact-Top 高 0.046（§4.3）[^src-2502-11089-nsa]。对 Full Attention 的逐项差值：HPQ +0.087、2Wiki +0.051、PassR-en +0.075、LCC +0.069、未标名的一列 +0.047、Qasper +0.023、MFQA-zh +0.001、MFQA-en −0.009、PassR-zh −0.010、GovRptDur −0.017。该缺失列名的表头在 PDF 文本层缺失，数值列位于 GovRptDur 与 PassR-en 之间，逐词坐标核对见 [[ingest-reports/2026-10-24-2502-11089-nsa|WHY 报告]] 的数值备注[^src-2502-11089-nsa]。

### 检索与推理

NIAH 在 64k 上下文、1K 到 64k 位置扫描中检索准确率全部正确（§4.3、Figure 5）；图内逐格数值不可读，色标范围 0.0–1.0[^src-2502-11089-nsa]。AIME 24（Table 3）：

| 生成长度限制 | 8192 | 16384 |
|---|---|---|
| Full Attention-R | 0.046 | 0.092 |
| NSA-R | 0.121 | 0.146 |

8k 限制下 +0.075，16k 限制下 +0.054（§4.3）[^src-2502-11089-nsa]。

### 训练损失（Figure 4）

在 27B、同步数对比中，NSA 的预训练损失曲线全程低于 Full Attention，两者均单调下降、无尖峰；曲线数值不可读[^src-2502-11089-nsa]。

## 范围与边界

评测仅限 27B 总参 / 3B 激活单一模型，未提供其他规模的 scaling 曲线（§4.1）[^src-2502-11089-nsa]。$l=32$、$d=16$、$l'=64$、$n=16$、$w=512$ 全按一组固定值使用，未给出压缩率、块长与窗口长度的敏感性（§4.1、§5）[^src-2502-11089-nsa]。实验未与同属训练期稀疏的方法同台对比：LongBench 仅对照 H2O、InfLLM、Quest、Exact-Top 四种推理期方法，Table 1 与 Table 3 只有 Full Attention，MoBA 等训练期块稀疏方法未纳入实验[^src-2502-11089-nsa]。加速比出自 Triton 实现而非生产 kernel，基线为同后端 FlashAttention-2；Table 4 按访存量推算期望值，与 Figure 6 实测口径不同（Table 4 只覆盖解码，Figure 6 覆盖 forward/backward）[^src-2502-11089-nsa]。门控采用 sigmoid 输出且未约束三支路和为 1，论文未分析门控值分布（§3.2）[^src-2502-11089-nsa]。门控贡献未与稀疏设计分离：Qwen 的 gated-attention 扫描指出 NSA 整体增益未拆出门控项，其 sigmoid 权重属三支路输出的分支级加权系数，既不逐 head 也不逐通道，与 SDPA 输出侧逐元素门控并非同一粒度（arXiv:2505.06708 §1）[^src-gated-attention-for-large-language-models-2505-06708]。

## 相关页面

- [[sdpa-output-gating]] — 分支级权重与逐 head 逐元素输出门控的粒度对照
- [[source-2502-11089-nsa]] — 源摘要
- [[mita-attention]] — 拼接后单次 softmax 的对照口径
- [[moba-attention]] — 同为训练期块稀疏，单支路 top-k 门控
- [[minimax-sparse-attention]] — 同为 GQA 原生块稀疏，逐组独立 top-k + 专用索引头（KL 对齐、detach、warmup 训练）
- [[deepseek-sparse-attention]] — 逐 latent 选择、独立 indexer 打分
- [[compressed-sparse-attention]] — DSA 框架在 $1/4$ 压缩 KV 上的 V4 实例化
- [[deepseek-v4-hybrid-attention]] — 压缩与选择拆到不同层交错的对照路线
- [[fast-weight-scaling-taxonomy]] — NSA 在五维坐标中的格位
- [[fast-weight-mlp-view-of-attention]] — full attention 的 fast-weight 读法
- [[deformable-fast-weight-experts]] — 刚性块与形变专家的差别
- [[mixture-of-experts]] — 块选择与门控合并借用的 MoE 原则
- [[kv-cache-compression]] — 推理期 KV 淘汰路线，与训练期稀疏正交
- [[multi-query-attention]] — 只留一份 K/V，是组内共用摊薄访存的前提
- [[quest-attention]] — 被列入对照的块打分选择方法
- [[grouped-query-attention]] — 组内共享 K/V 的头结构，§2.1 的「组内并集」读量口径以它为背景
[[sdpa-output-gating]] 对照分支级权重与逐 head 逐元素输出门控的粒度；[[source-2502-11089-nsa]] 为源摘要；[[mita-attention]] 说明拼接后单次 softmax 的对照口径；[[moba-attention]] 同属训练期块稀疏与单支路 top-k 门控；[[deepseek-sparse-attention]] 包含逐 latent 选择与独立 indexer 打分；[[fast-weight-scaling-taxonomy]] 标定 NSA 在五维坐标中的格位；[[fast-weight-mlp-view-of-attention]] 提供 full attention 的 fast-weight 读法；[[deformable-fast-weight-experts]] 区分刚性块与形变专家的差别；[[mixture-of-experts]] 归纳块选择与门控合并借用的 MoE 原则；[[kv-cache-compression]] 属推理期 KV 淘汰路线，与训练期稀疏正交；[[multi-query-attention]] 指明只留一份 K/V 是组内共用摊薄访存的前提；[[quest-attention]] 为列入对照的块打分选择方法；[[grouped-query-attention]] 对应组内共享 K/V 的头结构，§2.1 的「组内并集」读量口径以它为背景；[[lightning-attention]] 是不依赖稀疏选择的另一条减成本路线——线性化打分数学，以 7:1 混合 softmax 补检索。[[yoco-architecture]] 换到层数轴：cross 层保留 dense 全局读取，cache 的层数份数压成一份。

[^src-2502-11089-nsa]: [[source-2502-11089-nsa]]
[^src-dao-2022-flashattention]: [[source-dao-2022-flashattention]]
[^src-gated-attention-for-large-language-models-2505-06708]: [[source-gated-attention-for-large-language-models-2505-06708]]
[^src-2602-01219-mita]: [[source-2602-01219-mita]]
[^src-shazeer-2019-mqa]: [[source-shazeer-2019-mqa]]
