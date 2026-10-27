---
title: "YOCO (You Only Cache Once)"
type: technique
tags:
  - llm
  - kv-cache
  - architecture
  - efficient-inference
  - long-context
created: 2026-10-27
last_updated: 2026-10-27
source_count: 3
confidence: medium
status: active
---

# YOCO (You Only Cache Once)

YOCO 是 Microsoft Research 与清华大学（arXiv:2405.05254v2，2024-05）提出的 decoder-decoder 架构：将 Transformer 的 $L$ 个 block 拆为两段，前 $L/2$ 层 self-decoder 用常数推理内存的高效自注意力把整段输入编码成**唯一一份**全局 KV cache，后 $L/2$ 层 cross-decoder 全部以标准 cross-attention 读取该 cache；模型对外仍为自回归 decoder-only（§2）[^src-sun-2024-yoco]。它直接改变「**哪些层需要写全局 KV**」，而非每 token 存储开销或每步读取 token 数，消去了 cache 总量中的层数因子。

## 问题定位

自回归解码通过缓存历史 KV 避免重复编码，但 cache 大小为层数与长度之积：65B 模型（GQA + 8-bit KV）在 512K 上下文占约 86 GB，超出单张 H100-80GB；prefill 注意力呈平方复杂度，7B 模型采用 Flash-Decoding 与 kernel fusion 后，四张 H100 填 450K 上下文需约 110 s、1M 需 380 s（§1）[^src-sun-2024-yoco]。推理由此 memory-bound，限制了长上下文部署。

## 结构

**Self-decoder（§2.1）**。前 $L/2$ 层保持 Transformer 块布局（interleaved attention/FFN、pre-RMSNorm、SwiGLU），差别仅在于注意力替换为常数推理内存模块，实验评测两种：gated retention（gRet，默认）与滑窗注意力（SWA，窗口 $C=1024$）。gRet 在 retention 上叠加数据相关门控 $\gamma=\operatorname{sigmoid}(XW_\gamma)^{1/\tau}$（逐头标量，温度 $\tau$ 将 $\gamma$ 推向 1），并行、循环与块循环三种表达等价（式 4–6、附录 B 证明）：训练与 prefill 采用并行或块循环形式，解码采用循环形式 $S_n=\gamma_n S_{n-1}+K_n^{\top}V_n$，每头仅保留一个 $d\times d$ 状态（§3.1）；多头合并含 GroupNorm 与 swish 门（式 7）。gRet 的 Triton kernel 基于 FLA 实现（附录 A 致谢）。SWA 为窗口 $C$ 内的标准注意力（式 8）。两者 cache 大小均与 $N$ 无关：SWA 保存 $C$ 行、gRet 保存每头 $d\times d$ 状态；脚注 2 表明 "once" 专指全局 cache，self 侧常数开销在长序列下可忽略。

**Cross-decoder（§2.2）**。self-decoder 末端表示 $X^{L/2}$ 经一次投影产出全局 cache $\hat K=\operatorname{LN}(X^{L/2})W_K$、$\hat V=\operatorname{LN}(X^{L/2})W_V$（式 2）；随后每个 cross 层只算自己的 query $\hat Q_l=\operatorname{LN}(X^l)W_Q^l$，对同一份 $\hat K,\hat V$ 做带因果掩码的标准多头注意力（式 3）。cross-attention 与 GQA 兼容，可进一步压减 cache；产出 $X^L$ 后接 softmax 分类器。$L/2$ 个 cross 层的 query 各不相同、读的 KV 完全相同——这是「只缓存一次」的落点。

## 数据流与维度变换

**维度符号**：$N$ 序列长（token 数）；$d_{model}$ 隐藏维（YOCO-3B 为 3072）；$L$ 总层数，self 与 cross 各 $L/2$（YOCO-3B 为 26）；$d=d_h=128$ 单头通道维，论文按单头书写 $W_Q,W_K,W_V\in\mathbb{R}^{d\times d}$（式 4），多头按式（7）组装；$d_{kv}=n_{kv}d_h$ 全局 cache 每行通道数（YOCO-3B 为 $8\times128=1024$）；$C$ 滑窗大小（SWA 取 1024）；$B$ chunk 大小（推理 profile 取 256）；$n_q{=}24$、$n_{kv}{=}8$ 为 YOCO-3B 的 GQA 头数（Table 5），作用于 cross-attention。形状记 $[\text{行},\text{列}]$。

### Stage 1 · self-decoder 块

$$\begin{aligned}
Y^l &= \operatorname{ESA}(\operatorname{LN}(X^l)) + X^l && \text{// } [N, d_{model}]\to[N, d_{model}]\ \text{——式(1)，ESA 为 gRet 或 SWA，带因果掩码}\\
X^{l+1} &= \operatorname{SwiGLU}(\operatorname{LN}(Y^l)) + Y^l && \text{// } [N, d_{model}]\to[N, d_{model}]\ \text{——FFN 残差，与 Transformer 布局相同}
\end{aligned}$$

### Stage 2 · gRet：门控衰减与三种等价表示

$$\begin{aligned}
\gamma &= \operatorname{sigmoid}(XW_\gamma)^{1/\tau} && \text{// } [N, \bcancel{d}]\times[\bcancel{d}, 1]\to[N, 1]\ \text{——逐 token 逐头一个标量门（head-wise，非逐通道），温度 } \tau \text{ 把 } \gamma \text{ 推向 } 1\\
D_{nm} &= \textstyle\prod_{i=m+1}^{n}\gamma_i\ \ (n\ge m\text{，否则 } 0) && \text{// } [N, 1] \xrightarrow{\text{沿位置维前缀累乘}} [N, N]\ \text{——query } n \text{ 看 key } m \text{ 的累积衰减，下三角}\\
\operatorname{gRet}(X) &= (QK^{\top}\odot D)\,V && \text{// } [N, \bcancel{d}]\times[\bcancel{d}, N]\to[N, N]\ \odot\ [N, N]\times[\bcancel{N}, d]\to[N, d]\ \text{——并行表示（式 4）；} Q,K \text{ 先经 } \Theta_n=e^{in\theta} \text{ 复数旋转}
\end{aligned}$$

循环表示将同一计算写成状态递推，是解码期常数内存的来源（式 5）：

$$\begin{aligned}
S_n &= \gamma_n S_{n-1} + K_n^{\top}V_n && \text{// } [1, \bcancel{d}]^{\top}\times[\bcancel{1}, d]\to[d, d]\ \text{——每头一个 } d\times d \text{ 状态，元素数与 } N \text{ 无关}\\
\operatorname{gRet}(X_n) &= Q_n S_n && \text{// } [1, \bcancel{d}]\times[\bcancel{d}, d]\to[1, d]\ \text{——当前 query 读状态，等价于对全部历史的衰减加权}
\end{aligned}$$

块循环表示结合两者，实际用于训练与 prefill（式 6）：块内并行、跨块递推状态 $R_i$，$\beta$ 为 $\gamma$ 的块内前缀积：

$$\begin{aligned}
R_i &= K_{[i]}^{\top}\!\left(V_{[i]}\odot\tfrac{\beta_{iB}}{\beta_{[i]}}\right) + \beta_{iB}R_{i-1} && \text{// } [B, \bcancel{d}]^{\top}\times[\bcancel{B}, d]\to[d, d]\ \text{——块内 KV 聚合进状态，} \beta_{iB} \text{ 衰减旧状态}\\
\operatorname{gRet}(X) &= (Q_{[i]}K_{[i]}^{\top}\odot D_{[i]})V_{[i]} + (Q_{[i]}R_{i-1})\odot\beta_{[i]} && \text{// } [B, d]\ \text{两支相加}\ \text{——内块项 + 跨块项，省 FLOPs 又减迭代次数}
\end{aligned}$$

三种表示的等价性证明见附录 B（式 9–11）[^src-sun-2024-yoco]。

### Stage 3 · 全局 KV cache 只投影一次

$$\begin{aligned}
\hat K &= \operatorname{LN}(X^{L/2})\,W_K,\qquad \hat V = \operatorname{LN}(X^{L/2})\,W_V && \text{// } [N, \bcancel{d_{model}}]\times[\bcancel{d_{model}}, d_{kv}]\to[N, d_{kv}]\ \text{——式(2)，由 self-decoder 末端表示一次产出，此后对全部 } L/2 \text{ 个 cross 层冻结；GQA 下 } \hat K,\hat V \text{ 各 } n_{kv}d_h \text{ 宽，} n_{kv}=n_q \text{ 时即 } d_{model}
\end{aligned}$$

### Stage 4 · cross 层读同一份 cache

$$\begin{aligned}
\hat Q_l &= \operatorname{LN}(X^l)\,W_Q^l && \text{// } [N, d_{model}]\to[N, d_{model}]\ \text{——式(3)，每个 cross 层只算自己的 query}\\
Y^l &= \operatorname{Attn}(\hat Q_l, \hat K, \hat V) + X^l && \text{// } [N, \bcancel{d}]\times[\bcancel{d}, N] \xrightarrow{\text{沿 } N \text{ 维因果 softmax}} [N, N]\times[\bcancel{N}, d]\to[N, d]\ \text{——标准多头注意力，} L/2 \text{ 层共用同一份 } \hat K, \hat V
\end{aligned}$$

### Stage 5 · 解码增量：每 token 只追加一行

$$\begin{aligned}
\hat K_t &= [\hat K_{t-1};\ k_t],\quad \hat V_t = [\hat V_{t-1};\ v_t] && \text{// } [t{-}1, d_{kv}]\oplus[1, d_{kv}]\to[t, d_{kv}]\ \text{——全局 cache 每 token 增一行（} n_{kv} \text{ 组），层数不进 cache 的任何维度}\\
S_t &= \gamma_t S_{t-1} + K_t^{\top}V_t && \text{// } [d, d]\ \text{——self 侧状态原地更新，无常数以外的追加}
\end{aligned}$$

### Stage 6 · cache 记账：层数因子的消除

Table 1 给出两种架构的推理内存复杂度对比 [^src-sun-2024-yoco]：

$$\begin{aligned}
M_{\mathrm{TF}} &= O(LND) && \text{// } L \text{ 层各存一份 } N \text{ 行——层数因子乘进总量}\\
M_{\mathrm{YOCO}} &= O((N+L)D) && \text{// 全局 cache 只有一份 } N \text{ 行，self 侧每层常数；} N \text{ 大时 } \approx O(ND)\\
M_{\mathrm{TF}}/M_{\mathrm{YOCO}} &\approx L && \text{// } N \text{ 足够大时省 } L \text{ 倍：65B 顶到 } 80\times\text{（§1、Figure 8）}
\end{aligned}$$

正文 §2.3 的元素口径取 cache 数 $O(N+CL)$（换算 bytes 乘 $D$），与 Table 1 的 $O((N+L)D)$ 仅差在每层常数记为 $C$ 还是 $D$ 量级，两处并列。65B 模型配 GQA 与 8-bit KV 在 512K 占约 86 GB（§1）；同配置对照下 YOCO 在 65B 用 1 GB 服务 128K token，GQA Transformer 只能服务 1.6K token，比值为 $80\times$（§4.4）[INFERENCE：该比值按两数相除复算]。YOCO-3B 的全局 cache 每 token 为 $2n_{kv}d_h=2048$ 个元素，1M 长度下 8-bit 约 2 GB、16-bit 约 4 GB [INFERENCE：按 Table 5 配置推算，论文未给该数]。

## Prefill early exit

Generation 只依赖 self-decoder 产出的 cache：prefill 跑完 $L/2$ 层 self-decoder 即可生成，cross-decoder 只对**生成的** token 逐个前向（Figure 3）[^src-sun-2024-yoco]。两项节省叠加（§2.3）：层数减半带来至少 $2\times$；self 侧注意力为线性，prefill 注意力复杂度从 $O(LN^2D)$ 降至 $O(LND)$（Table 2）。在 H100-80GB 生成末 1024 token 的实测中（Figure 9），32K/64K/128K/256K/512K/1M 加速比为 2.87×/5.05×/8.36×/15.55×/30.3×/71.82×；Transformer 512K 约 180 s、1M 约 300 s，YOCO 512K 降到 6 s 内（§2.3、§4.4）[^src-sun-2024-yoco]。

## 训练

模型端到端联合训练，不分步训练 self-decoder 与 cross-decoder，层间仅体现为输入依赖（cross 层输入由 self 层产出）。分阶段体现在长度课程：64K → 256K → 1M 递进扩窗，每阶段调整学习率与 RoPE $\theta$（Table 7：LR 8e-5/4e-5/2e-5，$\theta$ 640K/5M/80M，token 6B/4B/1.5B），长文档上采样，不用长指令数据（§4.3、附录 E）[^src-sun-2024-yoco]。

配置（§4.1、Table 5）：YOCO-3B 共 26 层、hidden 3072、FFN 8192、词表 100,288、24 query 头 / 8 KV 头（GQA），非嵌入参数 2.83B；采用 AdamW $\beta=(0.9, 0.95)$，LR 3.2e-4、1000 步 warmup，在 5T token 内线性衰减至 1.28e-5，batch 4M token、序列长 4096、tiktoken-cl100k_base 分词，预算内训练 400K 步（1.6T token）。scaling 曲线设定（§4.2、附录 D、Table 6）：设 160M/400M/830M/1.4B/2.7B/6.8B/13B 七档，各训 10B token，序列长 2048、batch 0.25M，SWA 窗口 1024、gRet 头维 256；对齐参数量时 Transformer FFN 取 $\tfrac{8}{3}d$、YOCO 取 $3d$。分布式长序列训练采用 chunk parallelism（附录 A、Figure 11）：序列按 chunk 切到多卡，self-decoder 仅与相邻卡通信（gRet 传 $S_n$、SWA 传窗口），cross-decoder 的 $\hat K,\hat V$ 全程仅 all-gather 一次，无需逐层通信 [^src-sun-2024-yoco]。

## 证据

**语言建模对照（Table 3，Eval Harness 零样本）** [^src-sun-2024-yoco]：

| 模型 | ARC-C | ARC-E | BoolQ | HellaSwag | OBQA | PIQA | Winogrande | SciQ | Avg |
|---|---|---|---|---|---|---|---|---|---|
| OpenLLaMA-3B-v2（1T） | 0.339 | 0.676 | 0.657 | 0.700 | 0.260 | 0.767 | 0.629 | 0.924 | 0.619 |
| StableLM-base-alpha-3B-v2（1T） | 0.324 | 0.673 | 0.646 | 0.686 | 0.264 | 0.760 | 0.621 | 0.921 | 0.612 |
| StableLM-3B-4E1T（1T / 1.6T） | — | 0.666 / 0.688 | — | — | — | 0.768 / 0.762 | 0.632 / 0.627 | 0.914 / 0.913 | — |
| YOCO-3B（1T） | 0.379 | 0.731 | 0.645 | 0.689 | 0.298 | 0.763 | 0.639 | 0.924 | 0.634 |
| YOCO-3B（1.6T） | 0.396 | 0.733 | 0.644 | 0.698 | 0.300 | 0.764 | 0.631 | 0.921 | 0.636 |
| YOCO-3B-1M | 0.413 | 0.747 | 0.638 | 0.705 | 0.300 | 0.773 | 0.651 | 0.932 | 0.645 |

**规模 scaling（Figure 4）**：同数据与设置下，160M–13B 的 YOCO_gRet 验证损失低于 Llama 式 Transformer 与 YOCO_SWA（纵轴 2.9–3.6，曲线无数值）；论文将 gRet 的增益归因于 attention 与 retention 的互补归纳偏置，并注明 1:3 交错两种模块可获类似增益（§4.2）[^src-sun-2024-yoco]。160M 细粒度对照（Table 8）：验证 PPL 为 YOCO_gRet 3.530 < YOCO_SWA 3.553 < Transformer 3.564 < Hybrid H3 3.591 < gRetNet 3.600 < RetNet 3.633 < Mamba 3.645；AR-Hit（联想召回）1.199，对照 Transformer 1.219、Mamba 1.555（附录 G.1）[^src-sun-2024-yoco]。ZeroSCROLLS 四任务的答案 PPL 随输入变长而降；Figure 12 仅展示 Mamba、Sparse Transformer、Hybrid H3、Transformer 与 YOCO_gRet 五条曲线，YOCO 与 Transformer 在全部任务及长度上一致好于其余架构（附录 G.2）[^src-sun-2024-yoco]。

**长上下文（§4.3）**：NIAH 在 1M 长度接近满分（Figure 5，每格跑 10 次取均值）；128K 多针检索（Table 4）——YaRN-Mistral-128K（7B）0.02/0.12/0.08/0.20，LWM-1M-text（7B）1.00/0.90/0.76/0.62，MiniCPM-128K（2.4B）1.00/1.00/0.54/0.56，ChatGLM3-128K（6B）0.94/0.72/0.52/0.44，YOCO-3B-1M（3B）0.98/0.98/0.84/0.56（$N=1/2/4/8$ 根针）；半数参数量下与 LWM-1M-text 相当，全面高于同长度档的 128K 模型。书与代码验证集（均滤出 >1M token 样本）累计平均 NLL 随上下文增长持续下降，近似幂律（Figure 6）[^src-sun-2024-yoco]。

**部署（§4.4，Figures 7–10，H100-80GB，32K–1M）**：推理总内存 32K 节省约 2×（1.95×），1M 达 9.38×——1M 时 YOCO 全程 12.4 GB；Figure 7b 的分解显示 1M 下 Transformer 的显存被 KV cache 占满（Figure 7a 中间档约 3.01×/4.16×/6.39×，图内标注）。KV cache 每 token 内存随模型规模扩大节省更多：65B 约 $80\times$（Figure 8）。prefill 加速比见上节。吞吐（Figure 10，含 prefill 与生成）：32K/64K/128K/256K/512K 为 2.72×/2.57×/2.77×/4.37×/9.56×；512K 处 Transformer 4.5 token/s 对 YOCO 43.1 token/s（§4.4）[^src-sun-2024-yoco]。对照基线为带 GQA、Flash-Decoding 与 kernel fusion 的 Transformer，gRet 的 prefill 用块循环（chunk 256）、生成用循环形式（§4.4）[^src-sun-2024-yoco]。

## 与其他「省 KV」轴的关系

KV 侧既有方法各压一个维度，YOCO 换掉的是**层数份数**这一轴：

[[multi-query-attention]] 与 [[grouped-query-attention]] 压缩 K/V 头维：cache 从每头一份压至每 $G$ 组一份，但仍为每层一份并保留层数因子；YOCO 的 cross-attention 兼容 GQA（§2.2），YOCO-3B 按 24/8 配置叠加两轴 [^src-sun-2024-yoco]。[[multi-head-latent-attention]] 压缩每 token 缓存字节（KV 联合低秩 latent），cache 公式 $(d_c+d^R_h)l$ 中层数 $l$ 仍是乘积因子；YOCO 保持每行字节，将各层分别存储压为单份共享，两轴正交 [^src-deepseek-v2-2405-04434][^src-sun-2024-yoco]。[[native-sparse-attention]]、[[compressed-sparse-attention]]、[[deepseek-sparse-attention]] 改变每步读取集合（通过压缩、选择或滑窗重映射），cache 存储量仍随层数与长度增长；YOCO 的 cross 层保留对全部前序的 dense 全局注意力，减少存量而非读取集合（§2.2）[^src-sun-2024-yoco]。[[paged-attention]] 负责放置与共享（逻辑到物理块表、引用计数），不减少 KV 字节或份数；YOCO 减少份数，可与分页管理叠加 [^src-kwon-2023-pagedattention]。[[kv-cache-compression]] 汇总的淘汰与量化路线压缩时间维和数值精度，同样未触及层数轴。

全局记忆仅有一份且固定由 $X^{L/2}$ 投影生成，cross 层只读不写，取消了 Transformer 中逐层独立写入 KV 的记忆分化。论文未做 self:cross 层数比例消融，$L/2{:}L/2$ 为论文采用值而非扫描结论 [^src-sun-2024-yoco]。

## 范围与边界

“once” 不包含 self 侧：gRet 状态或滑窗 cache 仍按层存在，与 $N$ 无关且在长序列下可忽略（脚注 2）[^src-sun-2024-yoco]。质量证据止于 3B（1.6T token）与 160M–13B（10B token）两个量级，未评测指令微调、对话或多模态；对照 StableLM 的中间数字取自其技术报告而非重训（Table 3）[^src-sun-2024-yoco]。效率数字来自作者自研 Triton kernel 与特定配置（chunk 256、生成 1024 token），基线已叠加 GQA + Flash-Decoding + kernel fusion，属于优化后对照（§4.4）[^src-sun-2024-yoco]。Figure 4/5/6/7/12 为曲线图且无逐点数值，Figure 7a 的中间长度倍数与 Figure 8 的小模型倍数均取自图内标注。多针检索仅在 128K 长度评测（迁就既有模型），未测试 1M 长度（Table 4）[^src-sun-2024-yoco]。gRet 门控为逐头标量（引 GateLoop），论文未给出 $\tau$ 的取值（§3.1）[^src-sun-2024-yoco]。

## 相关页面

[[source-sun-2024-yoco]] 为源摘要；[[native-sparse-attention]] 与 [[compressed-sparse-attention]] 属「减读取」轴的训练期稀疏路线；[[multi-head-latent-attention]] 属「减每行字节」轴的低秩 latent 路线；[[multi-query-attention]] 与 [[grouped-query-attention]] 属「减头副本」轴；[[paged-attention]] 属「改放置」轴；[[kv-cache-compression]] 汇总各轴；[[flash-attention]] 属不改 cache 的 IO 路线，构成本文效率对照基线。

[^src-sun-2024-yoco]: [[source-sun-2024-yoco]]
[^src-deepseek-v2-2405-04434]: [[source-deepseek-v2-2405-04434]]
[^src-kwon-2023-pagedattention]: [[source-kwon-2023-pagedattention]]
