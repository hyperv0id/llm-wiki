---
title: "DeepSeek Sparse Attention (DSA)"
type: technique
tags:
  - attention
  - sparse-attention
  - long-context
  - kv-cache
  - llm
created: 2026-10-24
last_updated: 2026-10-26
source_count: 3
source_count: 2
confidence: medium
status: active
---

# DeepSeek Sparse Attention (DSA)

DeepSeek-V3.2 相对 DeepSeek-V3.1-Terminus 的**唯一架构改动**：在 MLA 的 MQA 模式下，用一个小型 indexer 给每个 query 的前序 token 打分，只对分数最高的 $k$ 个 KV entry 做注意力（§2.1）[^src-2512-02556-deepseek-v3-2]。主注意力的核心复杂度从 $\mathcal{O}(L^2)$ 降到 $\mathcal{O}(Lk)$，indexer 本身仍是 $\mathcal{O}(L^2)$（§2.3）[^src-2512-02556-deepseek-v3-2]。

## 两个组件

**Lightning indexer** 为 query token $\mathbf{h}_t$ 与前序 token $\mathbf{h}_s$ 打一个标量分（式 1）：

$$I_{t,s}=\sum_{j=1}^{H^{I}}w_{t,j}^{I}\cdot\mathrm{ReLU}\left(\mathbf{q}^{I}_{t,j}\cdot\mathbf{k}^{I}_{s}\right)$$

其中 $H^{I}$ 是 indexer 头数，$\mathbf{q}^{I}_{t,j},\mathbf{k}^{I}_{s}\in\mathbb{R}^{d^{I}}$，$w_{t,j}^{I}\in\mathbb{R}$ 由 $\mathbf{h}_t$ 导出（$\mathbf{k}^{I}_{s}$ 无头下标，indexer 的 key 在各头间共用）。ReLU 是吞吐考虑；indexer 头数少、可用 FP8（§2.1）[^src-2512-02556-deepseek-v3-2]。$H^{I}$、$d^{I}$ 与 $d^{I}/d$ 均无数值。

**Fine-grained token selection** 按分数取前 $k$ 个 KV entry $\mathbf{c}_s$，主注意力只在它们上计算（式 2）[^src-2512-02556-deepseek-v3-2]：

$$\mathbf{u}_{t}=\mathrm{Attn}\left(\mathbf{h}_{t},\left\{\mathbf{c}_{s}\mid I_{t,s}\in\operatorname{Top}\text{-}k\left(I_{t,:}\right)\right\}\right)$$

选择粒度是**单个 KV entry**，不是块，也不是原始 token hidden；每个 query 有各自的选中集合 $\mathcal{S}_t$（§2.1）[^src-2512-02556-deepseek-v3-2]。

## 为什么落在 MLA 的 MQA 模式

DSA 的 KV entry 就是 MLA 的 latent vector。kernel 层要求一个 KV entry 被多个 query 共享才能摊薄访存，因此 DSA 用 MLA 的 MQA 模式实例化：每个 latent vector 在该 query token 的**所有 query head 间共享**（§2.1）[^src-2512-02556-deepseek-v3-2]。由此 top-$k$ 只需选一次，对所有 head 共用同一组 latent。这正是 [[multi-query-attention]] 的机制：$H$ 组 query 对同一份 $K,V$ 打分，乘加数不变而 K/V 重读字节降为 $1/H$（§2.4.1）[^src-shazeer-2019-mqa]。

附录 A 只给了一张 MHA / MQA 模式的对比示意图（Figure 7），并说明 DeepSeek-V3.1-Terminus 在训练与 prefill 用 MHA 模式、decode 用 MQA 模式；**没有**给两种模式的公式或维度（附录 A）[^src-2512-02556-deepseek-v3-2]。两种模式的维度形态见 [[multi-query-attention]]。

## 数据流与维度变换

**维度符号**：$L$ 序列长度；$d$ 隐藏维；$d^{I}$ indexer 单头维（无数值）；$H^{I}$ indexer 头数（无数值）；$n_h$ 主注意力头数，$d_h$ 单头维；$d_c$ MLA latent 维（无数值）；$k$ 每 query 选中的 KV entry 数，稀疏训练阶段取 2048。

下面在式 (1)(2) 与「latent 在所有 query head 间共享」之外的部分标 `[INFERENCE]`：投影矩阵与分块实现按 MLA 的通行形式与开源实现补全。

### Stage 1 · 造 index 打分

两条支路都只读隐藏状态 $H\in\mathbb{R}^{L\times d}$。

**支路 A：indexer query 与头权重**

$$\begin{aligned}
Q^{I} &= W^{I}_{q}H && \text{// } [L, \bcancel{d}]\times[\bcancel{d}, H^{I}d^{I}]\to[L, H^{I}d^{I}]\ \text{——每个 token 得到 } H^{I}\text{ 组 indexer query}\\
w^{I} &= W^{I}_{w}H && \text{// } [L, \bcancel{d}]\times[\bcancel{d}, H^{I}]\to[L, H^{I}]\ \text{——每个头的标量权重}
\end{aligned}$$

**支路 B：indexer key（各头共用）**

$$\begin{aligned}
K^{I} &= W^{I}_{k}H && \text{// } [L, \bcancel{d}]\times[\bcancel{d}, d^{I}]\to[L, d^{I}]\ \text{——每个前序 token 一份 key，不带头下标}
\end{aligned}$$

**汇合：逐头打分后按权重求和**

$$\begin{aligned}
I_{t,s} &= \sum_{j=1}^{H^{I}} w^{I}_{t,j}\cdot\mathrm{ReLU}\left(\mathbf{q}^{I}_{t,j}\cdot\mathbf{k}^{I}_{s}\right) && \text{// } [H^{I}, L, \bcancel{d^{I}}]\times[\bcancel{d^{I}}, L]\to[H^{I}, L, L]\xrightarrow{\text{按 } w^{I} \text{ 沿 } H^{I} \text{ 加权求和}}[L, L]\ \text{——每个 (query, 前序 token) 对一个标量分}
\end{aligned}$$

Stage 1 的物理含义：indexer 必须看遍全部前序 token 才能选出 top-$k$，因此这一步的矩阵是 $[L,L]$ 量级，复杂度仍是 $\mathcal{O}(L^2)$（§2.3）[^src-2512-02556-deepseek-v3-2]。

### Stage 2 · 逐 query 取 top-k

$$\begin{aligned}
\mathcal{S} &= \operatorname{Top}_k\left(I_{t,:}\right) && \text{// } [L, L] \xrightarrow{\text{沿前序维逐行 Top-}k} [L, k]\ \text{——每行 } k \text{ 个前序 token 下标（取值 } 1..L\text{）}
\end{aligned}$$

候选按原文限定为「preceding token」，即因果位置之前的条目（§2.1）[^src-2512-02556-deepseek-v3-2]。

### Stage 3 · 在选中的 latent 上做主注意力

主注意力的 $K_{\mathcal{S}}$、$V_{\mathcal{S}}$ 由选中的 latent 投影得到（MLA 的标准做法；投影矩阵与 $d_c$ 无数值，投影与 gather 的实现形态标 [INFERENCE]）。

$$\begin{aligned}
C &= W^{DKV}H && \text{// } [L, \bcancel{d}]\times[\bcancel{d}, d_c]\to[L, d_c]\ \text{——MLA 的 KV latent，} d_c \text{ 无数值 [INFERENCE]}\\
C_{\mathcal{S}} &= \operatorname{Gather}\left(C,\mathcal{S}\right) && \text{// } [L, d_c] \xrightarrow{\text{按 } \mathcal{S}\in[L,k] \text{ 索引行}} [L, k, d_c]\ \text{——每行只留该 query 选中的 } k \text{ 个 latent；变长分段实现 [INFERENCE]}\\
S_{t} &= q_{t}K_{\mathcal{S}_t}^{\top}/\sqrt{d_h} && \text{// } [n_h, \bcancel{d_h}]\times[\bcancel{d_h}, k]\to[n_h, k]\ \text{——} n_h \text{ 个 query head 共用同一组 latent}
\end{aligned}$$

$$\begin{aligned}
A_{t} &= \operatorname{softmax}\left(S_{t}\right) && \text{// } [n_h, k] \xrightarrow{\text{沿 } k \text{ 维归一化}} [n_h, k]\ \text{——只在选中的 } k \text{ 个条目上归一化}\\
o_{t} &= A_{t}V_{\mathcal{S}_t} && \text{// } [n_h, k]\times[k, \bcancel{d_h}]\to[n_h, \bcancel{d_h}]\ \text{——} V\ \text{侧同样来自选中的 latent}
\end{aligned}$$

### Stage 4 · 拼头投影回隐藏维

$$\begin{aligned}
u_{t} &= W^{O}\operatorname{Concat}\left(o_{t,1},\dots,o_{t,n_h}\right) && \text{// } [1, \bcancel{n_h d_h}]\times[\bcancel{n_h d_h}, d]\to[1, d]\ \text{——所有 query 拼成 } [L, d]
\end{aligned}$$

### 成本分布

- 主注意力：$L$ 个 query × $n_h$ 个 head × $k$ 个条目，复杂度 $\mathcal{O}(Lk)$（§2.3）[^src-2512-02556-deepseek-v3-2]。
- indexer：$[L,L]$ 打分无法避免，复杂度仍是 $\mathcal{O}(L^2)$；头数少、维度低、可 FP8，实际计算量远小于 MLA，但两者的 FLOPs 对比未给出（§2.3）[^src-2512-02556-deepseek-v3-2]。
- 复杂度结论只覆盖这两块：gather 的访存、indexer 的 $[L,L]$ 中间量驻留均未计入。

## 训练：两阶段继续预训练

起点是上下文已扩到 128K 的 DeepSeek-V3.1-Terminus base checkpoint，两阶段数据分布与 128K 长上下文扩展数据一致（§2.1.1）[^src-2512-02556-deepseek-v3-2]。

**Dense warm-up（初始化 indexer）**：保持 dense attention，冻结除 indexer 外的全部参数；把各主注意力头的打分按头求和、沿序列维 L1 归一化，得到目标分布 $p_{t,:}$，用 KL 对齐 indexer 输出（§2.1.1）[^src-2512-02556-deepseek-v3-2]：

$$\mathcal{L}^{I}=\sum_{t}\mathbb{D}_{\mathrm{KL}}\left(p_{t,:}\,\middle\|\,\operatorname{Softmax}\left(I_{t,:}\right)\right)$$

学习率 $10^{-3}$，1000 步，每步 16 条 128K 序列，共 2.1B token（§2.1.1）[^src-2512-02556-deepseek-v3-2]。

**Sparse training（适配稀疏模式）**：引入 top-$k$ 选择并优化全部参数，KL 只在选中集合 $\mathcal{S}_t$ 上对齐（§2.1.1）[^src-2512-02556-deepseek-v3-2]：

$$\mathcal{L}^{I}=\sum_{t}\mathbb{D}_{\mathrm{KL}}\left(p_{t,\mathcal{S}_{t}}\,\middle\|\,\operatorname{Softmax}\left(I_{t,\mathcal{S}_{t}}\right)\right)$$

indexer 的输入从计算图中 detach：indexer 只由 $\mathcal{L}^{I}$ 优化，主模型只由语言建模损失优化（§2.1.1）[^src-2512-02556-deepseek-v3-2]。学习率 $7.3\times10^{-6}$，每 query 选 2048 个 KV token，15000 步，每步 480 条 128K 序列，共 943.7B token（§2.1.1）[^src-2512-02556-deepseek-v3-2]。

## 与 NSA 的结构差异

同为训练期稀疏，[[native-sparse-attention|NSA]]（2025-02）与 DSA（2025-12）在选择位置与粒度上不同：

- **打分来源**：NSA 的块重要性分直接取自压缩支路的注意力分数 $p^{cmp}_t=\operatorname{Softmax}(q_t^{\top}\tilde K^{cmp}_t)$（式 8），没有独立打分网络与辅助损失；DSA 用独立的 lightning indexer 为每个前序 token 打标量分，indexer 自身仍是 $\mathcal{O}(L^2)$（§2.3）[^src-2502-11089-nsa][^src-2512-02556-deepseek-v3-2]。
- **选择粒度**：NSA 按 64-token 的块选 top-$n=16$，另设压缩与滑动窗口两支路承担全局概括与局部精度；DSA 逐 MLA latent 选 $k=2048$ 个，无块结构，也无额外的压缩或窗口支路[^src-2502-11089-nsa][^src-2512-02556-deepseek-v3-2]。
- **组内一致性**：NSA 把同组各 query head 的块分数求和（式 10），使整组选同一批块；DSA 在 MLA 的 MQA 模式下所有 query head 共用一个 latent 集合，选一次即全组共用（§2.1）[^src-2502-11089-nsa][^src-2512-02556-deepseek-v3-2]。
- **规模**：NSA 在 27B 总参 / 3B 激活的模型上从头预训练（§4.1）；DSA 在 V3.1-Terminus 的 base checkpoint 上做两阶段继续预训练，原文未给参数量[^src-2502-11089-nsa][^src-2512-02556-deepseek-v3-2]。

## 证据

### 稀疏化前后的质量对照（§2.2）

三组对照都比较 **DeepSeek-V3.2-Exp 的 base checkpoint** 与 DeepSeek-V3.1-Terminus，均无表格、无对照分数（§2.2）[^src-2512-02556-deepseek-v3-2]：

- 标准基准：短上下文与长上下文任务上性能相近，未列出 benchmark 与分数。
- 人类偏好：ChatbotArena（2025-11-10 评测）Elo 分数接近（closely matched），无数值。
- 长上下文：AA-LCR 上 reasoning 模式比 Terminus **高 4 分**（唯一给出的数字，无绝对分）；Fiction.liveBench 上「consistently outperforms」，无数值。

### 推理成本（§2.3）

Figure 3 按 token position 给出 V3.1-Terminus 与 V3.2 的每百万 token 成本，分 prefill 与 decode 两个面板；成本由 H800 上实际部署服务基准、按 2 美元/GPU 小时折算（§2.3）[^src-2512-02556-deepseek-v3-2]。正文未给任何成本数值或加速比，本地转换稿亦丢失该图，因此**无法引用具体减免幅度**。短序列 prefill 用 masked MHA 模式模拟 DSA 以提高短上下文效率，阈值未给（§2.3）[^src-2512-02556-deepseek-v3-2]。

## 边界

- **没有与 NSA、MoBA 等稀疏注意力同台对比**，效率对照只有 V3.1-Terminus 的 MLA（§2.3）[^src-2512-02556-deepseek-v3-2]。
- **没有 $k$ 的敏感性实验**：$k=2048$ 是训练配置，未说明取值依据，也无质量-成本曲线（§2.1.1）[^src-2512-02556-deepseek-v3-2]。
- **没有 indexer 消融**：$H^{I}$、$d^{I}$、ReLU、FP8 均未逐项验证；也未报告 indexer 与主注意力分布的 recall 或对齐度（§2.1）[^src-2512-02556-deepseek-v3-2]。
- **parity 归因混杂**：对照双方同时差着继续预训练数据与稀疏化，无法把质量持平单独归给 DSA；且评测对象是 Exp checkpoint，不是最终 V3.2（§2.2）[^src-2512-02556-deepseek-v3-2]。
- **上下文只到 128K**：训练与评测均未超出，无更长序列的外推结果（§2.1.1、§4.1）[^src-2512-02556-deepseek-v3-2]。
- 复现口径缺失：$p_{t,:}$ 取自 softmax 前还是后的注意力分数、$p_{t,\mathcal{S}_t}$ 是否在选中集合上重新归一化、2048 是否逐层一致，均未说明（§2.1.1）[^src-2512-02556-deepseek-v3-2]。

## 相关页面

- [[deepseek-v3-2]] — 该机制唯一落地的模型与训练配方
- [[moba-attention]] — 块粒度的 top-$k$ 路由对照
- [[native-sparse-attention]] — 同一实验室更早的训练期块稀疏方法（三分支 + 门控合并）
- [[fast-weight-scaling-taxonomy]] — DSA 在五维坐标中的格位
- [[deformable-fast-weight-experts]] — 逐 query top-$k$ 与可复用专家组的差别
- [[kv-cache-compression]] — 该机制不淘汰任何 KV，只减少每次注意力的读取
- [[multi-query-attention]] — 该机制所依附的 K/V 共享形态
- [[grouped-query-attention]] — $G=1$ 端点的头结构背景：MQA 模式下全部 query head 共用一个 latent 集合
- [[source-2512-02556-deepseek-v3-2]] — 源摘要

[^src-2512-02556-deepseek-v3-2]: [[source-2512-02556-deepseek-v3-2]]
[^src-2502-11089-nsa]: [[source-2502-11089-nsa]]
[^src-shazeer-2019-mqa]: [[source-shazeer-2019-mqa]]
