---
title: "Causal Encoder-Decoder (CED)"
type: technique
tags:
  - attention
  - long-context
  - kv-cache
  - prefill
  - llm
created: 2026-10-27
last_updated: 2026-10-27
source_count: 1
confidence: medium
status: active
---

# Causal Encoder-Decoder (CED)

CED 把 40 层 Transformer 按 $L/2$ 切成因果编码器与解码器两段：解码器的**全局** KV 不由本层隐藏状态产生，而由编码器末层 $H^{L/2}$ 经逐层权重投影出来；滑窗分支仍逐层自算。这使 prefill 只需全长序列穿过前 20 层，复杂度从 $O(NL)$ 降到 $O(NL/2+n_{win}L/2)$（§2.2）[^src-deepseek-v4-1-2609-19969]。

## 定义式

对 $l>L/2$ 的解码器层（§2.2 式 1）[^src-deepseek-v4-1-2609-19969]：

$$C^{l}=H^{L/2}W^{KV}_{l},\qquad Z^{l}=H^{L/2}W^{Z}_{l},\qquad l>\frac{L}{2}$$

$C^{l}$ 为主 KV 条目，$Z^{l}$ 为其压缩权重，$W^{KV}_{l},W^{Z}_{l}$ 逐层不同。滑窗注意力不参与该改写，任意层的局部 key/value 均取自本层隐藏状态 $H_{l}$（§2.2）[^src-deepseek-v4-1-2609-19969]。与 CSA2 组合时，仅指派为 Full 模式的解码器层物化这份全局 KV，Reindex/Reuse 层直接共享（§2.3.1）[^src-deepseek-v4-1-2609-19969]；该理解与 890 bytes/token 自洽（反算见 [[csa2]] 的「层配置与 KV 账」）。

## 动机

agentic 工作流中频繁的工具调用产生大量 prefill 请求，KV cache 未命中时 prefill 成本高昂，输入密集型负载使得 prefill 成为主要开销（摘要、§2.2）[^src-deepseek-v4-1-2609-19969]。前置工作 YOCO 让上半层直接共享下半层生成的 KV。CED 在两点上对其改写：一是解码器 KV 从固定的编码器末层投影，以提升 KV 生成的计算深度；二是 SWA 维持逐层计算，将局部 KV 的表达深度保留在每一层，代价是需要重放（§2.2）[^src-deepseek-v4-1-2609-19969]。按 $N\gg n_{win}$，这两项结合将 prefill 计算近乎减半，同时「与基线性能相当」（§2.2）[^src-deepseek-v4-1-2609-19969]。

## SWA Bounded Replay

逐层 SWA 使得精确重建 $L$ 层的 SWA KV 需要重放 $L\times n_{win}$ 个 token，在短 prompt 的多轮场景下不可接受（§2.2、§3.2.2）[^src-deepseek-v4-1-2609-19969]。做法是只重放最近 $n_{win}$ 个 token 并将 SWA 截断至重放段：重放起点为 $s$ 时，位置 $i$ 的 query 只看 $\max(s,\,i-W+1)$ 到 $i$ 的窗口，得到近似状态（§3.2.2）[^src-deepseek-v4-1-2609-19969]。

Encoder SWA Bounded Replay 在 SWA KV 缺失时只重放已缓存前缀的最后 $n_{win}$ 个 token，重放段仅重建 SWA KV、复用已缓存全局 KV，未缓存后缀同时产出全局 KV 与 SWA KV；此时前缀缓存只依赖全局 KV，SWA KV 得以从持久化缓存中移除（§3.2.2）[^src-deepseek-v4-1-2609-19969]。Decoder SWA Bounded Replay 针对 CED 后阻碍「prefill 停在编码器」的解码器 SWA KV，每次 prefill 只把最后 $n_{win}$ 个 token 送入解码器，产出的 SWA KV 仅供解码使用而不进入前缀缓存（§2.2、§3.2.2）[^src-deepseek-v4-1-2609-19969]。

重放得到的前缀状态是近似的，未缓存后缀的全局 KV 与 SWA KV 随命中位置而变、数学上并不相同（§3.2.2）[^src-deepseek-v4-1-2609-19969]。为降低影响，post-training 阶段模拟同一重放过程做 train-aware 适配（§3.2.2）[^src-deepseek-v4-1-2609-19969]。

## 数据流与维度变换

**维度符号**：$N$ 序列长度；$L=40$ 总层数，编码器与解码器各 $L/2=20$ 层；$d=5120$ 隐藏维；$c=512$ 主 KV 条目维；解码器压缩率 $m=1$；$n_{win}=128$ 滑窗；$k=512$ top-$k$（§2.1、§4.2.1）[^src-deepseek-v4-1-2609-19969]。形状记 $[\text{行},\text{列}]$；论文未给的实现细节标 [INFERENCE]。

### Stage 1 · 编码器吃下全长序列

$$\begin{aligned}
H^{L/2} &= \operatorname{Enc}_{1..L/2}\!\left(H^{(0)}\right) && \text{// } [N,d]\to[N,d]\ \text{——前 2 层 SWA、后 18 层 CSA2}(m{=}2)\text{，prefill 的全长计算只发生在这 20 层}
\end{aligned}$$

### Stage 2 · 从编码器末层投影出解码器全局 KV

$$\begin{aligned}
C^{l} &= H^{L/2}W^{KV}_{l} && \text{// } [N,\bcancel{5120}]\times[\bcancel{5120},512]\to[N,512]\ \text{——式(1)，解码器各层不再从本层隐藏状态算全局 KV}\\
Z^{l} &= H^{L/2}W^{Z}_{l} && \text{// } [N,\bcancel{5120}]\times[\bcancel{5120},512]\to[N,512]\ \text{——同源投影出的压缩权重}
\end{aligned}$$

$m=1$ 故压缩为恒等，每条目对应一个原始 token（§2.3）[^src-deepseek-v4-1-2609-19969]；indexer K 再由 $C^{Comp,l}$ 投影（见 [[csa2]] Stage 3）[^src-deepseek-v4-1-2609-19969]。

### Stage 3 · 解码器局部 KV 逐层自算（只覆盖重放段）

$$\begin{aligned}
\left[K^{win}_{l};V^{win}_{l}\right] &= H^{(win)}_{l}\left[W^{K}_{l};W^{V}_{l}\right] && \text{// } [128,\bcancel{5120}]\times[\bcancel{5120},2c]\to[128,2c]\ \text{——取本层隐藏状态的最后 } n_{win} \text{ 个 token，局部注意力的计算深度不因 CED 下降}
\end{aligned}$$

$H^{(win)}_{l}$ 表示本层隐藏状态在重放段上的切片。重放段之外的解码器 SWA KV 不保存、不重算（§3.2.2）[^src-deepseek-v4-1-2609-19969]。

### Stage 4 · decode 时新 token 的全局 KV 仍来自编码器输出

$$\begin{aligned}
C^{l}_{t} &= h^{L/2}_{t}W^{KV}_{l} && \text{// } [1,\bcancel{5120}]\times[\bcancel{5120},512]\to[1,512]\ \text{——第 } t \text{ 个 token 自己过完 20 层编码器后，由 Full 层投一条全局 KV，其余层共享}
\end{aligned}$$

解码阶段 40 层全部参与，索引与选择仍使用 CSA2 的三模式（§2.3.1）[^src-deepseek-v4-1-2609-19969]。

## 证据

激活参数（§2.1、§1、Table 1 的「# Activated Params」行）：prefill 每 token 8B、decode 16B，同一 checkpoint 在两阶段激活的参数规模相差一倍[^src-deepseek-v4-1-2609-19969]。计算量为 $O(NL)\to O(NL/2+n_{win}L/2)$，prefill 近乎减半（§2.2）[^src-deepseek-v4-1-2609-19969]。端到端效率：4K→1M 的 decode 单 token FLOPs 只增 $1/4$（Figure 2 混合了 CSA2、FP4 与 DSpark，非 CED 单项）[^src-deepseek-v4-1-2609-19969]。质量评测仅有「与基线相当」「对回答质量影响可忽略」两句定性表述，无 CED 开/关对照表（§2.2、§3.2.2）[^src-deepseek-v4-1-2609-19969]。

## 边界

CED 缺少单变量消融，未设「同模型关掉 CED」「解码器 KV 取自本层 vs 取自 $H^{L/2}$」「重放 $n_{win}$ vs 全量」三类对照，使「近乎减半 prefill」与「性能相当」无法分别归因（§2.2）[^src-deepseek-v4-1-2609-19969]。bounded replay 的状态为近似重建，未缓存后缀的 KV 随命中位置变化，论文未给出这种不一致的量化影响（§3.2.2）[^src-deepseek-v4-1-2609-19969]；§6 将「SWA 状态近似重建」列为尚未完全刻画的鲁棒性边界[^src-deepseek-v4-1-2609-19969]。所有解码器全局 KV 同源（均投影自同一 $H^{L/2}$），层间 KV 相关性上升，正文未讨论其表达能力代价 [INFERENCE]。§3.2.2 一处「2$L$ 个解码器层」与解码器实为 $L/2$ 层并置，按 $L/2$ 理解。

## 相关页面

[[csa2]] 是与 CED 配合的 KV 压缩方案（CED 解决 prefill 计算，CSA2 解决 cache 存储与 indexer 计算，§2.1 明确把两者当作互补成本）；[[deepseek-v4-hybrid-attention]] 是被替换的 V4 注意力层，V4 的全局分支要逐层算 KV、没有编码器/解码器分工；[[compressed-sparse-attention]] 与 [[heavily-compressed-attention]] 提供 CSA2 的前身；[[multi-head-latent-attention]] 是「共享小 latent 生成 KV」的近亲思路；[[kv-cache-compression]] 收录缓存压缩的其他路线；[[source-deepseek-v4-1-2609-19969]] 为源摘要。

[^src-deepseek-v4-1-2609-19969]: [[source-deepseek-v4-1-2609-19969]]
