---
title: "PagedAttention（分页 KV cache 内存管理）"
type: technique
tags:
  - llm-serving
  - kv-cache
  - memory-management
  - efficient-inference
  - attention-kernel
created: 2026-10-26
last_updated: 2026-10-27
source_count: 1
confidence: medium
status: active
---

# PagedAttention

PagedAttention 借用操作系统的分页：把一条请求的 KV cache 切成固定大小的逻辑块，物理块可散落在显存任意位置，attention kernel 读时按 block table 现场 gather（§4.1）。配套的 vLLM 在此之上做按需分配、块级引用计数共享与抢占调度（§4.2–§4.6）。它解决的是**放置**问题——同样数量的 KV 存在哪里、被谁引用——而不是每个 token 要存多少字节[^src-pagedattention]。

## 问题定位

现有系统（FasterTransformer、Orca）要求一个请求的 KV cache 是连续张量，因而在请求开始时按该请求的**最大可能长度**预留连续 chunk，与实际输入输出长度无关（§3.1）[^src-pagedattention]。由此产生三类浪费：预留槽位（将来会填，但整段请求生命周期内占住）、内部碎片（超出实际长度的部分永不使用）、外部碎片（buddy allocator 取整与空洞，永不使用）。Fig 3 的两个例子把后两类量化：上限 2048 的请求 A 只有 7 个槽存 prompt 的 KV 状态（外加 1 个已生成槽与 2 个预留槽），2038 个槽永不使用；上限 512 的请求 B 只有 3 个 prompt 槽（外加已生成与预留各 1 槽），507 个槽永不使用[^src-pagedattention]。

三类浪费叠加后，实测有效显存只有 20.4%–38.2%（§6.2 实验的 Fig 2）[^src-pagedattention]。批大小因此被显存卡死，而解码阶段本就是 memory-bound（§2.2）[^src-pagedattention]。

## 机制

**分块与映射**。每块容纳 $B$ 个 token 在该层该头上的 K/V（block size，默认 16）。块表每个 entry 记录该逻辑块对应的物理块号与该块已填槽数，于是 KV cache 可以只在生成时逐块增长，而不预留到 $L^{\max}$（§4.2）[^src-pagedattention]。每个 key/value 向量可以在层与头两个维度各自分块、各用一张块表，与全部打包进一块在性能上等价（§4.1 脚注 1）[^src-pagedattention]。

**块级注意力**。注意力按块改写为对 $B$ 个 token 的块内矩阵乘加跨块求和，块内不再要求连续（§4.1，式 4）[^src-pagedattention]。

**共享与写时复制**。物理块带引用计数 $c_p$：并行采样（parallel sampling）让同一 prompt 的多个样本共享 prompt 块；束搜索（beam search）让候选共享前缀与更早的公共块，随解码推进动态改变共享树；共享前缀（system prompt、few-shot 示例）由服务方预置为常驻物理块，用户请求只把逻辑块映过去（§4.4）[^src-pagedattention]。写共享块时触发块级 copy-on-write，只复制一个块（$B$ 个槽），不是整条序列[^src-pagedattention]。

**抢占**。请求率超容量时按 FCFS 调度，淘汰策略是 all-or-nothing（一条序列的块要么全留要么全换出），同一请求内的多条序列按序列组整体调度，因为块可能在组内共享。恢复手段两种：换出到 CPU（需额外 CPU block allocator；换出总量上界是 GPU 上 KV 区大小，因为被换出的块数不会超过物理块总数）、或重算（把已生成 token 与原 prompt 拼成新 prompt，一次 prefill 重建全部位置的 KV）（§4.5）[^src-pagedattention]。

**分布式**。采用 Megatron-LM 式张量模型并行，注意力算子按头切分；各 worker 处理同一批 token，因此只需要一份集中式 block manager 与一套逻辑→物理映射，各 worker 按同一块表只存自己那些头的 KV。每步调度器把 token id 与各请求的块表广播给 worker，worker 之间只做 all-reduce，不同步内存管理（§4.6）[^src-pagedattention]。

**算子与接口**。三个融合 kernel：block 写入（reshape + 按块表写位融合为一次启动）、block 读出与注意力融合（改自 FasterTransformer，一个 warp 读一块以保证合并访存，并支持批内变长）、block 复制批量融合（替代逐块 `cudaMemcpyAsync`）（§5.1）[^src-pagedattention]。解码算法归纳为 fork / append / free 三个原语实现（§5.2）[^src-pagedattention]。

## 数据流与维度

**维度符号**：$N_{\text{req}}$ 一批内的请求数（含并行采样与束搜索产生的多条序列）；$L_r$ 请求 $r$ 的最终 token 数；$L^{\max}$ 模型上限序列长度（OPT 为 2048）；$B$ block size（默认 16）；$N_{\text{logical}}=\lceil L^{\max}/B\rceil$ 每请求逻辑块数上限；$P$ GPU 上物理块总数；$H$ 每层头数；$d_{\text{head}}$ 单头通道维；$\ell$ 为层号。论文按单层单头描述分块、把层与头各自分块作为实现选择，本页把 $(\ell,h)$ 显式带出[^src-pagedattention][INFERENCE]。

### Stage 1 · 请求到达，按块粒度分配物理块

prefill 只需覆盖 prompt 的 $n^{\text{prompt}}_r$ 个 token，不预留到 $L^{\max}$；块从空闲表任意取，所以相邻逻辑块的物理块号一般不相邻。

$$\begin{aligned}
n^{\text{pf}}_r &= \left\lceil n^{\text{prompt}}_r / B \right\rceil && \text{// } \text{标量} \xrightarrow{\ \lceil\cdot/B\rceil\ } \text{prefill 期只需 } n^{\text{pf}}_r \text{ 块，与 } L^{\max} \text{ 无关}\\
T &\in \{1..P\}^{N_{\text{req}}\times N_{\text{logical}}} && \text{// } [N_{\text{req}}, N_{\text{logical}}]\ \text{——entry 只存物理块号：逻辑块 } j \to \text{物理块 } p\\
F &\in \{0..B\}^{N_{\text{req}}\times N_{\text{logical}}} && \text{// } [N_{\text{req}}, N_{\text{logical}}]\ \text{——与 } T \text{ 配对的已填槽数，} F[r,j]=B \text{ 表示该逻辑块已满}\\
T[r,\,0{:}n^{\text{pf}}_r] &\sim \operatorname{FreeList}(P) && \text{// } [N_{\text{req}}, n^{\text{pf}}_r] \xrightarrow{\text{从 } P \text{ 个同尺寸块任取}} \text{槽 } b \text{ 的地址 } = pB+b\ \text{——块间可任意散落}
\end{aligned}$$

块表读出的就是这两张 $[N_{\text{req}}, N_{\text{logical}}]$ 表；解码阶段所有请求的 token 拼成一条序列送进模型，因此一次前向里每行的有效列数为 $\lceil L_r/B\rceil$（§4.3）[^src-pagedattention]。

### Stage 2 · 写入新 token 的 K/V

新 token $t$（从 0 计）的 K/V 落在逻辑块 $j=\lfloor t/B\rfloor$ 的槽 $b=t \bmod B$，即物理地址 $pB+b$；只有最后一个块可能未填满，填满后下一个 token 才新开逻辑块并从空闲表取新物理块。

$$\begin{aligned}
k^{(\ell,h)}_t, v^{(\ell,h)}_t &= W^{(\ell,h)}_k h_t,\ W^{(\ell,h)}_v h_t && \text{// } [d_{\text{model}},\bcancel{1}]\to[d_{\text{head}}]\ \text{——单 token 的一列 K、一列 V}\\
j &= \lfloor t/B\rfloor,\quad b = t \bmod B,\quad p = T[r,j] && \text{// } \text{逻辑块号} \to \text{块内槽} \to \text{物理块号}\\
K^{(\ell,h)}_{\text{phys}}[p,\,b] &\leftarrow k^{(\ell,h)}_t && \text{// } [P, B, d_{\text{head}}] \xrightarrow{\text{按 }(p,b)\text{ 单槽写}} \text{就地覆写空槽，不搬移已有块}\\
F[r,j] &\leftarrow b+1 && \text{// } \#\text{filled 自增；} b=B-1 \text{ 用满后下一 token 走 } b=0 \text{ 并新取一块}
\end{aligned}$$

### Stage 3 · kernel 按块表 gather 物理块

块表在逻辑块序上给出物理块号，索引张量直接作用在物理 KV 张量的块维上：逻辑上连续的一段 KV 被拉成 $[\lceil L_r/B\rceil, B, d_{\text{head}}]$ 的规整张量，摊平后与逐 token 存储的布局同构。

$$\begin{aligned}
\mathcal{J}_r &= T[r,\,0{:}\lceil L_r/B\rceil] && \text{// } \text{索引张量 } [\lceil L_r/B\rceil]\text{，取值 } 1..P\ \text{——每个逻辑块一个物理块号}\\
K^{(r)} &= K_{\text{phys}}[\mathcal{J}_r] && \text{// } [P, B, d_{\text{head}}] \xrightarrow{\text{按 } \mathcal{J}_r \text{ 取第 0 维}} [\lceil L_r/B\rceil, B, d_{\text{head}}]\ \text{——一次 gather 得到"看似连续"的全序列 KV}\\
K_j &= K^{(r)}[j] && \text{// } [B, d_{\text{head}}]\ \text{——第 } j \text{ 个逻辑块的 key，供 Stage 4 逐块使用}
\end{aligned}$$

gather 的访存粒度是整块：一个 warp 负责一块，块内 $B$ 个相邻槽一次读入，故间接寻址只发生在块级而非 token 级（§5.1）[^src-pagedattention]。

### Stage 4 · 合并注意力输出

单个 query 与每个逻辑块算一个 $B$ 维打分向量，分母跨全部已见块求和；因此对 query $i$ 而言，归一化范围是 $\lceil i/B\rceil$ 块，跨块求和等价于逐 token 求和。

$$\begin{aligned}
S_{ij} &= q_i^{\top}K_j/\sqrt{d_{\text{head}}} && \text{// } [d_{\text{head}},\bcancel{d_{\text{head}}}]\times[\bcancel{d_{\text{head}}},B]\to[B]\ \text{——消去 } d_{\text{head}}\text{，一个 query 对一个块的 } B \text{ 个打分}\\
A_{ij} &= \frac{\exp S_{ij}}{\sum_{t=1}^{\lceil i/B\rceil}\exp S_{it}} && \text{// } [B] \xrightarrow{\text{分母跨 } \lceil i/B\rceil \text{ 块求和}} [B]\ \text{——归一化跨块，块内单独 softmax 会算错权重}\\
o_i &= \sum_{j=1}^{\lceil i/B\rceil} V_j A_{ij}^{\top} && \text{// } \sum_{j} [d_{\text{head}},\bcancel{B}]\times[\bcancel{B}] \to [d_{\text{head}}]\ \text{——块内矩阵乘 + 跨块累加}
\end{aligned}$$

逐块并行的前提是每块只算自己那部分分子与分母，跨块用 $(\text{局部最大值},\ \text{局部指数和})$ 归并成全局分母（online softmax），否则各 warp 的归一化基准不一致；论文只交代"一个 warp 读一块 + 融合读出与注意力"，未展开归并细节[^src-pagedattention][INFERENCE]。

### Stage 5 · 请求结束按引用计数回收（含写时复制分支）

物理块的引用计数在共享时大于 1：fork（并行采样、束搜索分支、前缀共享）只是把新序列的逻辑块指向同一物理块并给计数加一。回收与写入都要看计数。

$$\begin{aligned}
c &\in \mathbb{N}^{P} && \text{// } [P]\ \text{——每物理块一个引用计数，未共享时为 } 1\\
\text{if } c_{T[r,j]} > 1:\quad p_{\text{old}} = T[r,j],\ p' &\sim \operatorname{FreeList}(P) && \text{// } \text{要写共享块时，先取一块新物理块}\\
K_{\text{phys}}[p'] &\leftarrow K_{\text{phys}}[p_{\text{old}}] && \text{// } [B, d_{\text{head}}] \xrightarrow{\text{整块复制}} [B, d_{\text{head}}]\ \text{——只搬一个块，不是整条序列}\\
T[r,j] \leftarrow p',\ c_{p'} &= 1,\ c_{p_{\text{old}}} \mathrel{-}= 1 && \text{// } \text{只改自己这一行；其余共享者仍指旧块，计数减到 1 后它可直接原地写}\\
\text{free}(p) &\iff c_p = 0 && \text{// } [P] \xrightarrow{\text{计数归零}} \text{回 FreeList，立即供别的请求使用}
\end{aligned}$$

请求结束时逐块减计数、归零即释放，因此内存回收是块粒度而非请求粒度；束搜索中被淘汰候选的块同样按此路径释放（如淘汰候选后 blocks 2、4、5、8 归零）（§4.4–§4.5）[^src-pagedattention]。

## 内存浪费的算法

**预分配的浪费**。设请求 $r$ 的上限长度 $L^{\max}$、最终长度 $L_r$、当前已填 $f_r(t)$，现有系统占用 $C_r=L^{\max}$ 个槽。三类浪费按定义拆开即：预留 $=L_r-f_r(t)$，内部碎片 $=L^{\max}-L_r$，外部碎片来自分配器取整；有效比例

$$\eta_{\text{prealloc}}(t) = \frac{\sum_r f_r(t)}{\sum_r L^{\max}},\qquad \eta_{\text{paged}}(t) = \frac{\sum_r f_r(t)}{\sum_r B\lceil L_r/B\rceil}$$

分母就是差别所在：后者按实际长度取整到块，前者一律按 2048 撑满。Fig 2（OPT-13B，§6.2 同一组实验）的实测分解如下，单位 %[^src-pagedattention]：

| 系统 | token states | 预留 | 内部碎片 | 外部碎片及其他 |
|---|---|---|---|---|
| Orca (Max) | 20.4 | 13.3 | 57.3 | 8.9 |
| Orca (Pow2) | 26.8 | 17.9 | 13.6 | 41.6 |
| Orca (Oracle) | 38.2 | — | 25.2 | 36.6 |

vLLM 一侧 96.3% 之外只剩不可读的细窄条（合计约 3.7%），图中不给出三分量的归属。由此复现论文的浪费口径：token states 20.4% / 26.8% / 38.2%（Max / Pow2 / Oracle）对应浪费 79.6% / 73.2% / 61.8%，与 §3.1「只有 20.4%–38.2% 用于存真实 token 状态」逐项对上；vLLM 的 96.3% 对应摘要所称的「近零浪费」。构成随基线策略而变：Orca (Oracle) 已知真实输出长度，预留项为零，剩内部碎片 25.2%（buddy 取整）与外部碎片 36.6%；Orca (Pow2) 按 $2^{\lceil\log_2 L\rceil}$ 过度预留（至多 2×），内部碎片降到 13.6%，但 chunk 大小随请求浮动，外部碎片升到 41.6%；Orca (Max) 一律预留 2048，内部碎片 57.3% 成为最大项，chunk 恒为 $2^{11}$ 无取整，外部碎片只剩 8.9%[INFERENCE]。

代入论文的 OPT-13B 参数可把 20.4% 这一端算出来：单 token KV 为 $2\times5120\times40\times2$ B $=800$ KB（§3 原文即给出该算式），Table 1 的 KV 区 12 GB 须按 GiB 口径（$12\times2^{30}$ B）才能除出 $15{,}728$ 个槽，与表值 15.7K 一致；全按 2048 预留时只能装 $\lfloor 15{,}728/2048\rfloor=7$ 条满长请求，而 ShareGPT 的均值序列长度为 $161.31+337.99=499.3$ 个 token（Fig 11a），$499.3/2048=24.4\%$ 即请求完成时刻的效率上界，与时间平均的 20.4% 同一量级[INFERENCE]。

**分页后的残量**。所有块同尺寸，外部碎片消失（空闲表里没有"大小不合"的块）；内部碎片只剩每条序列最后一个未填满块的尾部：

$$\text{waste}_r = \frac{B\lceil L_r/B\rceil - L_r}{B\lceil L_r/B\rceil} \le \frac{B-1}{B\lceil L_r/B\rceil} \le \frac{B-1}{L_r}$$

$B=16$、$L_r\approx499$ 时上界 $15/499\approx3.0\%$，与 Fig 2 里 vLLM 的 96.3% 相符（实测残量 3.7% 还包含最后一个块当下未填、但将来会填的槽位，那部分属于预留而非永久浪费）[INFERENCE]。这个尾部浪费是分页不可消除的：块大小固定而序列长度任意，最后一块必然可能留下 $1$ 至 $B-1$ 个空槽——要再往下压只能缩小 $B$（代价是访存并行度）或改变每 token 的字节数，后者不属于本机制。

## 证据

设置（§6.1、Table 1）：GCP A2 实例 + A100；OPT-13B（单卡 40 GB，参数 26 GB，KV 区 12 GB，15.7K 槽）、OPT-66B（4×A100，160 GB，参数 132 GB，KV 21 GB，9.7K 槽）、OPT-175B（8×A100-80GB，640 GB，参数 346 GB，KV 264 GB，60.1K 槽），容量均出自 Table 1；评测还含 LLaMA-13B；负载由 ShareGPT（输入均值 161.31、输出均值 337.99）与 Alpaca（19.31、58.45）的真实长度合成，Poisson 到达，1 小时 trace（175B 因成本用 15 分钟）[^src-pagedattention]。基线为 FasterTransformer（自建动态批调度）与三个自建 Orca 版本（Oracle / Pow2 / Max，buddy 分配）。指标为归一化延迟（端到端延迟除输出长度）对请求率。

- **基本采样（§6.2、Fig 12）**：ShareGPT 上维持相近延迟可承受的请求率，vLLM 比 Orca (Oracle) 高 1.7×–2.7×、比 Orca (Max) 高 2.7×–8×，比 FasterTransformer 最高 22×[^src-pagedattention]。同请求率下的并发批（Fig 13）：ShareGPT（OPT-13B，2 req/s）vLLM 30.42 条，Orca (Oracle) 13.62、Orca (Pow2) 9.81、Orca (Max) 7.00，即 2.2× 与 4.3×；Alpaca（30 req/s）132.44 对 72.75 / 43.24，约 1.8× 与 3.1×[^src-pagedattention]。例外是 Fig 12f：175B + Alpaca 时显存充裕、序列又短，系统转成 compute-bound，vLLM 相对 Orca (Oracle)/(Pow2) 的优势明显收窄[^src-pagedattention]。
- **块共享（§6.3、Fig 14–15）**：省下的内存按"共享省下的块数 / 不共享时的总块数"计。OPT-13B + Alpaca：并行采样 2/4/6 条输出分别省 6.09%/8.53%/9.79%，束搜索宽 2/4/6 分别省 37.56%/53.13%/55.16%；同实验换 ShareGPT 为 16.2%–30.5% 与 44.3%–66.3%[^src-pagedattention]。吞吐上，OPT-13B + Alpaca 相对 Orca (Oracle) 从基本采样的 1.3× 升到束宽 6 的 2.3×[^src-pagedattention]。
- **共享前缀（§6.4、Fig 16）**：LLaMA-13B、WMT16 En-De，前缀含 1 个示例（80 token）时吞吐 1.67×，含 5 个示例（341 token）时 3.58×，均对 Orca (Oracle)[^src-pagedattention]。
- **聊天负载（§6.5、Fig 17）**：OPT-13B，prompt 截到 1024 token、最多生成 1024 token，vLLM 可承受的请求率为三个 Orca 基线的 2×；此处三条 Orca 基线几乎重合，因为 buddy 分配下它们都按 1024 token 预留输出空间[^src-pagedattention]。
- **kernel 微基准（§7.1、Fig 18a）**：上下文 64/128/256，vLLM（batch 8 / 32）的注意力 kernel 延迟比 FasterTransformer 高 20%–26%，来源是读块表、额外分支与变长处理；开销只在注意力算子，不含 Linear[^src-pagedattention]。
- **块大小扫描（§7.2、Fig 18b）**：块 1–256 全扫。ShareGPT 上 16–128 最好；Alpaca 上 16、32 可用，更大则显著退化，因为序列比块还短。默认取 16——小到内部碎片可忽略，大到能喂满访存并行[^src-pagedattention]。
- **换出与重算（§7.3、Fig 19）**：块小时换出开销大（CPU–GPU 之间退化为大量小传输，占不满 PCIe 带宽），重算开销与块大小无关。原文称重算开销 "never higher than 20% of swapping's latency"，按字面是「不超过换出延迟的 20%」，但这与同段「块大时换出更高效」矛盾——若重算恒低于换出五分之一，换出不可能更高效，此句更可能想表达「至多比换出慢 20%」；块 16–64 区间两者端到端相当[^src-pagedattention]。

## 边界

- **间接寻址的代价是实在的**：块表读取、分支与变长处理让注意力 kernel 慢 20%–26%（Fig 18a），端到端仍赢是因为省下的显存换来了更大的批[^src-pagedattention]。
- **块大小是超参**，两个方向都会掉性能：太小则单块访存并行不足，太大则内部碎片与共享概率同时恶化；论文的扫描结论是 16–128 在长序列负载上可用、16 在短序列负载上才稳，默认取 16（§7.2）[^src-pagedattention]。
- **收益来自并发度而非单条序列**：分页不缩小单条请求的 KV 足迹（仍随 $L_r$ 线性增长），只是把"预留满长"的浪费换成"一块尾巴"；只有同时驻留更多请求才能反映为吞吐。短请求 + 显存充裕的配置（Tab. 1 的 175B、Alpaca 均值 78 token）下系统转 compute-bound，收益收窄（Fig 12f）[^src-pagedattention]。
- **换出/重算的适用面**：换出受 PCIe 带宽限制、块小时不划算；重算受算力限制，靠把已生成 token 拼回 prompt 一次 prefill 重建。换出空间有界（不超过 GPU 上 KV 区的块数）（§4.5、§7.3）[^src-pagedattention]。
- **不普适**：§8 明确该技术成立的前提是动态分配 + 受显存容量约束；DNN 训练形状静态、非 LLM 的 DNN 服务多为 compute-bound，此时引入间接寻址与非连续块反而变慢[^src-pagedattention]。
- **与压缩不同轴**：paging 不减少每个 token 要存的 KV 字节数，也不淘汰任何 KV，只决定这些 KV 放在哪、由谁共享。压低每 token 字节要靠头共享（MQA/GQA，让多个 query 头共用一组 K/V）与量化，压低总量还要靠淘汰或稀疏化；paging 与它们正交且可叠加——压缩后的 KV 同样按块管理[^src-pagedattention]。

## 相关页面

- [[kv-cache-compression]] — 正交轴：固定预算下留哪些 KV（重要性估计/淘汰），不涉及放置
- [[native-sparse-attention]] — 训练期块稀疏，减少每步读取的 KV 量，也不改变缓存布局
- [[source-kwon-2023-pagedattention]] — 源摘要
- [[attention-sink]] — 长序列上 KV 保留策略的另一端：固定保留初始 token

[^src-pagedattention]: [[source-kwon-2023-pagedattention]]
