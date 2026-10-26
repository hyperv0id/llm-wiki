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
last_updated: 2026-10-27
source_count: 4
confidence: medium
status: active
---

# DeepSeek Sparse Attention (DSA)

DeepSeek-V3.2 相对 DeepSeek-V3.1-Terminus 的**唯一架构改动**，是在 MLA 的 MQA 模式下用小型 indexer 为每个 query 的前序 token 打分，仅对分数最高的 $k$ 个 KV entry 计算注意力（§2.1）[^src-2512-02556-deepseek-v3-2]。主注意力核心复杂度从 $\mathcal{O}(L^2)$ 降至 $\mathcal{O}(Lk)$，indexer 本身仍为 $\mathcal{O}(L^2)$（§2.3）[^src-2512-02556-deepseek-v3-2]。

## 两个组件

**Lightning indexer** 为 query token $\mathbf{h}_t$ 与前序 token $\mathbf{h}_s$ 计算标量分（式 1）：

$$I_{t,s}=\sum_{j=1}^{H^{I}}w_{t,j}^{I}\cdot\mathrm{ReLU}\left(\mathbf{q}^{I}_{t,j}\cdot\mathbf{k}^{I}_{s}\right)$$

其中 $H^{I}$ 为 indexer 头数，$\mathbf{q}^{I}_{t,j},\mathbf{k}^{I}_{s}\in\mathbb{R}^{d^{I}}$，$w_{t,j}^{I}\in\mathbb{R}$ 由 $\mathbf{h}_t$ 导出（$\mathbf{k}^{I}_{s}$ 无头下标，各头共用 indexer 的 key）。ReLU 旨在提升吞吐；indexer 头数少，可用 FP8（§2.1）[^src-2512-02556-deepseek-v3-2]。$H^{I}$、$d^{I}$ 与 $d^{I}/d$ 均无数值。

**打分的非负性**（[INFERENCE]）：单头贡献 $\mathrm{ReLU}\left(\mathbf{q}^{I}_{t,j}\cdot\mathbf{k}^{I}_{s}\right)\geq 0$ 恒成立：内积为负按 0 计，各头对前序 token 仅产生非负贡献；普通注意力 logit $\mathbf{q}\cdot\mathbf{k}/\sqrt{d_h}$ 可正可负，无此下界。头间权重 $w^{I}_{t,j}\in\mathbb{R}$ 不受约束，加权求和后的 $I_{t,s}$ 仍可能为负，非负性仅在单头成立。该性质契合 warm-up 目标：$p_{t,:}$ 由各主注意力头分数求和、沿序列维 L1 归一化得到（§2.1.1）[^src-2512-02556-deepseek-v3-2]，按后文「softmax 后概率」口径天然非负。原文仅以吞吐为由选用 ReLU，未将单头非负性与该目标关联（§2.1）[^src-2512-02556-deepseek-v3-2]。

**Fine-grained token selection** 按分数选取前 $k$ 个 KV entry $\mathbf{c}_s$，主注意力仅在其上计算（式 2）[^src-2512-02556-deepseek-v3-2]：

$$\mathbf{u}_{t}=\mathrm{Attn}\left(\mathbf{h}_{t},\left\{\mathbf{c}_{s}\mid I_{t,s}\in\operatorname{Top}\text{-}k\left(I_{t,:}\right)\right\}\right)$$

选择粒度为**单个 KV entry**，而非分块或原始 token hidden；每个 query 拥有独立的选中集合 $\mathcal{S}_t$（§2.1）[^src-2512-02556-deepseek-v3-2]。

**Top-$k$ 的语义**（[INFERENCE]）：式 2 直接对原始分 $I_{t,:}$ 取前 $k$ 而不经 softmax；因 softmax 严格单调，在 logits 与概率域上的排序等价。$k$ 是每个 query 选中 KV entry 数的**上限**而非固定配额：候选按因果限定为前缀 token，位置 $t$ 只有 $t-1$ 个候选，$t-1\leq k$ 时全部入选；稀疏训练阶段 $k=2048$（§2.1.1）[^src-2512-02556-deepseek-v3-2]。

## 为什么落在 MLA 的 MQA 模式

DSA 的 KV entry 即 [[multi-head-latent-attention|MLA]] 的 latent vector。kernel 层要求多个 query 共享 KV entry 以摊薄访存，因此 DSA 采用 MLA 的 MQA 模式实例化：每个 latent vector 在该 query token 的**所有 query head 间共享**（§2.1）[^src-2512-02556-deepseek-v3-2]。由此 top-$k$ 只需选一次，所有 head 共用同一组 latent。这正是 [[multi-query-attention]]：$H$ 组 query 对同一份 $K,V$ 打分，乘加数不变，而 K/V 重读字节降为 $1/H$（§2.4.1）[^src-shazeer-2019-mqa]。

附录 A 仅给出 MHA / MQA 模式的对比示意图（Figure 7），说明 DeepSeek-V3.1-Terminus 在训练与 prefill 采用 MHA 模式、decode 采用 MQA 模式，**未给出**两种模式的公式或维度（附录 A）[^src-2512-02556-deepseek-v3-2]。两种模式的维度形态见 [[multi-query-attention]]。

## 数据流与维度变换

**维度符号**：$L$ 序列长度；$d$ 隐藏维；$d^{I}$ indexer 单头维（无数值）；$H^{I}$ indexer 头数（无数值）；$n_h$ 主注意力头数，$d_h$ 单头维；$d_c$ MLA latent 维（无数值）；$k$ 每 query 选中的 KV entry 数，稀疏训练阶段取 2048。

除式 (1)(2) 与「latent 在所有 query head 间共享」外，其余部分标 `[INFERENCE]`：投影矩阵与分块按 MLA 通行形式及开源实现补全。

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

Stage 1 中，indexer 须遍历全部前序 token 才能选出 top-$k$，该步矩阵达 $[L,L]$ 量级，复杂度仍为 $\mathcal{O}(L^2)$（§2.3）[^src-2512-02556-deepseek-v3-2]。

### Stage 2 · 逐 query 取 top-k

$$\begin{aligned}
\mathcal{S} &= \operatorname{Top}_k\left(I_{t,:}\right) && \text{// } [L, L] \xrightarrow{\text{沿前序维逐行 Top-}k} [L, k]\ \text{——每行 } k \text{ 个前序 token 下标（取值 } 1..L\text{）}
\end{aligned}$$

候选限定为因果位置之前的条目，即「preceding token」（§2.1）[^src-2512-02556-deepseek-v3-2]。

### Stage 3 · 在选中的 latent 上做主注意力

主注意力的 $K_{\mathcal{S}}$、$V_{\mathcal{S}}$ 沿用 MLA 由选中的 latent 投影得到；投影矩阵与 $d_c$ 无数值，投影和 gather 的实现形态标 [INFERENCE]。

$$\begin{aligned}
C &= W^{DKV}H && \text{// } [L, \bcancel{d}]\times[\bcancel{d}, d_c]\to[L, d_c]\ \text{——MLA 的 KV latent，} d_c \text{ 无数值 [INFERENCE]}\\
C_{\mathcal{S}} &= \operatorname{Gather}\left(C,\mathcal{S}\right) && \text{// } [L, d_c] \xrightarrow{\text{按 } \mathcal{S}\in[L,k] \text{ 索引行}} [L, k, d_c]\ \text{——每行只留该 query 选中的 } k \text{ 个 latent；变长分段实现 [INFERENCE]}\\
S_{t} &= q_{t}K_{\mathcal{S}_t}^{\top}/\sqrt{d_h} && \text{// } [n_h, \bcancel{d_h}]\times[\bcancel{d_h}, k]\to[n_h, k]\ \text{——} n_h \text{ 个 query head 共用同一组 latent}
\end{aligned}$$

$$\begin{aligned}
A_{t} &= \operatorname{softmax}\left(S_{t}\right) && \text{// } [n_h, k] \xrightarrow{\text{沿 } k \text{ 维归一化}} [n_h, k]\ \text{——只在选中的 } k \text{ 个条目上归一化}\\
o_{t} &= A_{t}V_{\mathcal{S}_t} && \text{// } [n_h, k]\times[k, \bcancel{d_h}]\to[n_h, \bcancel{d_h}]\ \text{——} V\ \text{侧同样来自选中的 latent}
\end{aligned}$$

Gather 落在 latent 而非各头投影后的 $K,V$ 上（[INFERENCE]）：先取该 query 的 $k$ 个 $d_c$ 维 latent 再逐头上投影，成本正比于 $k$；若先对全部 $L$ 个 latent 投影再 gather，成本正比于 $L$。「一个 KV entry 被多个 query 共享」的 kernel 前提（§2.1）[^src-2512-02556-deepseek-v3-2]在实现上对应这种 latent 粒度的省算。

### Stage 4 · 拼头投影回隐藏维

$$\begin{aligned}
u_{t} &= W^{O}\operatorname{Concat}\left(o_{t,1},\dots,o_{t,n_h}\right) && \text{// } [1, \bcancel{n_h d_h}]\times[\bcancel{n_h d_h}, d]\to[1, d]\ \text{——所有 query 拼成 } [L, d]
\end{aligned}$$

### 成本分布

主注意力计算 $L$ 个 query × $n_h$ 个 head × $k$ 个条目，复杂度 $\mathcal{O}(Lk)$（§2.3）[^src-2512-02556-deepseek-v3-2]。indexer 的 $[L,L]$ 打分无法避免，复杂度仍是 $\mathcal{O}(L^2)$；头数少、维度低且可 FP8，实际计算量远小于 MLA，但两者的 FLOPs 对比未给出（§2.3）[^src-2512-02556-deepseek-v3-2]。该复杂度结论只覆盖这两处，gather 访存与 indexer 的 $[L,L]$ 中间量驻留均未计入。

**省算量不等式链**（[INFERENCE]，原文未给数值，保持符号形式）：只数 score 与 value 两步乘加，dense 主注意力约 $2L^2 n_h d_h$；DSA 为 indexer 打分的 $L^2 H^{I}d^{I}$ 加稀疏主注意力的 $2Lk\,n_h d_h$，两者之比

$$\frac{L^{2}H^{I}d^{I}+2Lk\,n_h d_h}{2L^{2}n_h d_h}=\frac{H^{I}d^{I}}{2n_h d_h}+\frac{k}{L}.$$

只要 $H^{I}d^{I}\ll 2n_h d_h$（每对 query–key 比较中，indexer 头数 $\times$ 单头维远小于主注意力头数 $\times$ 单头维）且 $k\ll L$，比值就小于 1。indexer 保留 $\mathcal{O}(L^2)$ 项的节省来自压缩每对系数，而非消除 $L$ 的幂次；投影、softmax、gather 访存与 FP8 的常数因子均未计入。

## 在压缩 KV 上实例化：DeepSeek-V4 的 CSA

DeepSeek-V4 把 DSA 的选择对象从逐 token latent 换成压缩条目：先把每 $m=4$ 个 token 的 KV 压成 1 个 entry（重叠双支路、净压缩率 $1/4$），indexer keys 经同一压缩算子同步压短，再在压缩条目上跑同一套 indexer + top-$k$ + MQA；$k$ 从 2048 降到 512（V4-Flash）/ 1024（V4-Pro），主注意力与 indexer 的 query 共用同一个低秩 latent $c^{Q}_t$（§2.3.1）[^src-deepseek-v4-2606-19348]。indexer 打分矩阵由 $[L,L]$ 量级缩短为 $[L, L/4]$ 量级，是本页「indexer 仍为 $\mathcal{O}(L^2)$」结论在 V4 中的直接放松。机制与维度见 [[compressed-sparse-attention]]。

## 训练：两阶段继续预训练

起点为上下文已扩展至 128K 的 DeepSeek-V3.1-Terminus base checkpoint，两阶段数据分布与 128K 长上下文扩展数据一致（§2.1.1）[^src-2512-02556-deepseek-v3-2]。

**Dense warm-up（初始化 indexer）**：保持 dense attention 并冻结除 indexer 外全部参数；将各主注意力头打分按头求和、沿序列维 L1 归一化得到目标分布 $p_{t,:}$，用 KL 对齐 indexer 输出（§2.1.1）[^src-2512-02556-deepseek-v3-2]：

$$\mathcal{L}^{I}=\sum_{t}\mathbb{D}_{\mathrm{KL}}\left(p_{t,:}\,\middle\|\,\operatorname{Softmax}\left(I_{t,:}\right)\right)$$

「main attention scores」逐头求和（§2.1.1）[^src-2512-02556-deepseek-v3-2]；此处 scores 按 softmax 后的注意力概率理解（[INFERENCE]）：各头概率非负，跨头求和再 L1 归一化只是整体缩放，结果仍与注意力质量成比例；若取 softmax 前的 logits，跨头求和可含负值，L1 归一化须对负分取绝对值，所得权重不再与注意力分数成比例。

学习率 $10^{-3}$，1000 步，每步 16 条 128K 序列，共 2.1B token（复算 $1000\times16\times131072=2.097\times10^9$，与论文口径吻合）（§2.1.1）[^src-2512-02556-deepseek-v3-2]。

**Sparse training（适配稀疏模式）**：引入 top-$k$ 选择并优化全部参数，KL 只在选中集合 $\mathcal{S}_t$ 上对齐（§2.1.1）[^src-2512-02556-deepseek-v3-2]：

$$\mathcal{L}^{I}=\sum_{t}\mathbb{D}_{\mathrm{KL}}\left(p_{t,\mathcal{S}_{t}}\,\middle\|\,\operatorname{Softmax}\left(I_{t,\mathcal{S}_{t}}\right)\right)$$

indexer 的输入从计算图中 detach：indexer 只由 $\mathcal{L}^{I}$ 优化，主模型只由语言建模损失优化（§2.1.1）[^src-2512-02556-deepseek-v3-2]。

**梯度不流经 top-$k$**（[INFERENCE]，论文未提供 straight-through 或替代梯度的实现细节）：下标选择是离散操作，无可导路径；语言建模损失的梯度只落在选中条目的注意力计算上，indexer 完全由 KL 损失驱动。先 dense 对齐、后引入 top-$k$ 的两阶段顺序与此互补：冷启动时 indexer 未初始化，语言建模损失又因选择不可导无法传回梯度；先用 dense KL 将 indexer 对齐到主注意力分布，引入 top-$k$ 时选择才能落在有意义的分数上。

学习率 $7.3\times10^{-6}$，每 query 选 2048 个 KV token；15000 步，每步 480 条 128K 序列，共 943.7B token（复算 $15000\times480\times131072=9.437\times10^{11}$，与论文口径吻合）（§2.1.1）[^src-2512-02556-deepseek-v3-2]。

## 与 NSA 的结构差异

同为训练期稀疏，[[native-sparse-attention|NSA]]（2025-02）与 DSA（2025-12）在选择位置与粒度上不同：

打分来源上，NSA 的块重要性分直接取自压缩支路的注意力分数 $p^{cmp}_t=\operatorname{Softmax}(q_t^{\top}\tilde K^{cmp}_t)$（式 8），没有独立打分网络与辅助损失；DSA 则用独立的 lightning indexer 为每个前序 token 打标量分，indexer 自身仍是 $\mathcal{O}(L^2)$（§2.3）[^src-2502-11089-nsa][^src-2512-02556-deepseek-v3-2]。选择粒度上，NSA 按 64-token 的块选 top-$n=16$，另设压缩与滑动窗口两支路承担全局概括与局部精度；DSA 逐 MLA latent 选 $k=2048$ 个，无块结构，也无额外的压缩或窗口支路[^src-2502-11089-nsa][^src-2512-02556-deepseek-v3-2]。为保持组内一致，NSA 把同组各 query head 的块分数求和（式 10），使整组选同一批块；DSA 在 MLA 的 MQA 模式下所有 query head 共用一个 latent 集合，选一次即全组共用（§2.1）[^src-2502-11089-nsa][^src-2512-02556-deepseek-v3-2]。规模上，NSA 在 27B 总参 / 3B 激活的模型上从头预训练（§4.1）；DSA 在 V3.1-Terminus 的 base checkpoint 上做两阶段继续预训练，原文未给参数量[^src-2502-11089-nsa][^src-2512-02556-deepseek-v3-2]。

## 证据

### 稀疏化前后的质量对照（§2.2）

三组对照都比较 **DeepSeek-V3.2-Exp 的 base checkpoint** 与 DeepSeek-V3.1-Terminus，均无表格、无对照分数（§2.2）[^src-2512-02556-deepseek-v3-2]：

标准基准上，短上下文与长上下文任务性能相近，未列出 benchmark 与分数。ChatbotArena（2025-11-10 评测）人类偏好 Elo 分数接近（closely matched），无数值。长上下文任务中，AA-LCR 上 reasoning 模式比 Terminus **高 4 分**（唯一给出的数字，无绝对分）；Fiction.liveBench 上仅称「consistently outperforms」，无数值。

### 推理成本（§2.3）

Figure 3 分 prefill 与 decode 两个面板，按 token position 给出 V3.1-Terminus 与 V3.2 的每百万 token 成本；成本由 H800 实际部署服务基准按 2 美元/GPU 小时折算（§2.3）[^src-2512-02556-deepseek-v3-2]。正文未给成本数值与加速比；本地转换稿丢失该图，arXiv v1 HTML 中该图亦只剩图题「(a) Prefilling」（LaTeXML 不含图内数值），缺少具体减免幅度。短序列 prefill 用 masked MHA 模式模拟 DSA 提高短上下文效率，未给阈值（§2.3）[^src-2512-02556-deepseek-v3-2]。

## 边界

未与 NSA、MoBA 等稀疏注意力同台对比，效率对照只有 V3.1-Terminus 的 MLA（§2.3）[^src-2512-02556-deepseek-v3-2]。缺少 $k$ 的敏感性实验，$k=2048$ 仅为训练配置且未说明取值依据，亦无质量-成本曲线（§2.1.1）[^src-2512-02556-deepseek-v3-2]。indexer 缺少消融实验，$H^{I}$、$d^{I}$、ReLU、FP8 均未逐项验证，亦未报告 indexer 与主注意力分布的 recall 或对齐度（§2.1）[^src-2512-02556-deepseek-v3-2]。parity 归因混杂，对照双方同时差着继续预训练数据与稀疏化，无法把质量持平单独归给 DSA，且评测对象是 Exp checkpoint 而非最终 V3.2（§2.2）[^src-2512-02556-deepseek-v3-2]。上下文只到 128K，训练与评测均未超出，无更长序列外推结果（§2.1.1、§4.1）[^src-2512-02556-deepseek-v3-2]。复现口径缺失，$p_{t,:}$ 取自 softmax 前还是后的注意力分数、$p_{t,\mathcal{S}_t}$ 是否在选中集合上重新归一化、2048 是否逐层一致，均未说明（§2.1.1）[^src-2512-02556-deepseek-v3-2]。

## 相关页面

- [[deepseek-v3-2]] — 该机制唯一落地的模型与训练配方
- [[compressed-sparse-attention]] — DSA 在 $1/4$ 压缩 KV 上的 V4 实例化
- [[deepseek-v4-hybrid-attention]] — CSA 与 HCA 交错的混合架构
- [[moba-attention]] — 块粒度的 top-$k$ 路由对照
- [[minimax-sparse-attention]] — 同样索引保持 $O(N^2)$、KL 对齐训练的块稀疏对照：逐 GQA 组选块而非全头共享 token 级索引
- [[native-sparse-attention]] — 同一实验室更早的训练期块稀疏方法（三分支 + 门控合并）
- [[fast-weight-scaling-taxonomy]] — DSA 在五维坐标中的格位
- [[deformable-fast-weight-experts]] — 逐 query top-$k$ 与可复用专家组的差别
- [[kv-cache-compression]] — 该机制不淘汰任何 KV，只减少每次注意力的读取
- [[multi-query-attention]] — 该机制所依附的 K/V 共享形态
- [[grouped-query-attention]] — $G=1$ 端点的头结构背景：MQA 模式下全部 query head 共用一个 latent 集合
- [[source-2512-02556-deepseek-v3-2]] — 源摘要
- [[multi-head-latent-attention]] — DSA 选中的 KV entry 即 latent vector，其 $d_c$ 与上投影形态见该页数据流

[^src-2512-02556-deepseek-v3-2]: [[source-2512-02556-deepseek-v3-2]]
[^src-2502-11089-nsa]: [[source-2502-11089-nsa]]
[^src-shazeer-2019-mqa]: [[source-shazeer-2019-mqa]]
[^src-deepseek-v4-2606-19348]: [[source-deepseek-v4-2606-19348]]
