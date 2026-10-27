---
title: "GShard: 600B 参数 MoE 条件计算与自动分片"
type: technique
tags:
  - mixture-of-experts
  - sparse-gating
  - distributed-training
  - machine-translation
  - scaling
created: 2026-10-27
last_updated: 2026-10-27
source_count: 2
confidence: medium
status: active
---

# GShard

GShard（Lepikhin et al., Google, ICLR 2021）将稀疏门控 [[mixture-of-experts|MoE]] 从 Shazeer 2017 的 LSTM 语言模型推进至 Transformer 与 600B 参数规模：以单一非集成模型训练 100→英语多语言翻译，在 2048 块 TPU v3 上 4 天内完成[^src-lepikhin-2020-gshard]。它同时是一套系统，结合轻量 sharding 注解 API 与 XLA 编译器扩展，将整个加速器集群视作单一设备自动分片[^src-lepikhin-2020-gshard]。在 [[mixture-of-experts]] 收录的谱系中，它位于 Shazeer 稀疏门控 MoE 之后、Switch Transformer（尚未入库）之前。后继的 [[deepseekmoe]] 保留 top-$K$ 路由与辅助负载均衡损失的框架，但把每个专家切细并隔离出恒激活的共享专家[^src-dai-2024-deepseekmoe]。

## 问题

论文 §1.1 归纳了四项缩放障碍：通用框架（TensorFlow/PyTorch）缺乏高效模型并行支持；加深加宽使计算成本超线性增长，模型并行引入的通信与设备空转无法靠增加设备解决；数千设备上的巨型计算图节点数 $O(D)$、通信通道可增至 $O(D^2)$，建图与编译时间不可行；手工分区需协调跨设备通信，改动架构即连带重写，工程负担繁重[^src-lepikhin-2020-gshard]。对应三项设计原则：以条件计算带来次线性成本（sub-linear scaling）、将模型描述与分区实现分离（仅注解少量关键张量），以及采用 SPMD 编译使编译时间与设备数无关[^src-lepikhin-2020-gshard]。

## MoE 层与 top-2 gating

在 encoder 与 decoder 中，模型每隔一个 feed-forward 层替换为 position-wise MoE 层，expert 形状与常规 Transformer FFN 相同；扩容仅需调整层数与每层 expert 数量[^src-lepikhin-2020-gshard]。层定义（§2.2，式 1–3）：

$$G_{s,\cdot} = \mathrm{GATE}(x_s), \qquad \mathrm{FFN}_e(x) = w_o^e\,\mathrm{ReLU}(w_i^e\, x), \qquad y_s = \sum_{e=1}^{E} G_{s,e}\,\mathrm{FFN}_e(x_s)$$

$x_s$ 为进入 MoE 层的 token 表示，$w_i^e, w_o^e$ 为 expert $e$ 的输入/输出投影；$G_{s,\cdot}$ 分量非负、绝大多数为零，每个 token 最多派发给 2 个 expert，$y_s$ 是选中 expert 输出的加权平均[^src-lepikhin-2020-gshard]。

Top-2 gating（Algorithm 1）先计算 softmax 门控 $g_{S,E} = \mathrm{softmax}(w_g \cdot x_S)$，再分两个循环处理第一、第二 expert[^src-lepikhin-2020-gshard]：

第一 expert 确定性派发：取 top-2 得 $(g_1, e_1, g_2, e_2)$，归一化 $g_1 \leftarrow g_1/(g_1+g_2)$；只要 $e_1$ 计数未达容量即写入 combine 权重。计数器无条件递增，溢出的派发决策也占容量名额。第二 expert 随机派发：归一化 $g_2 \leftarrow g_2/(g_1+g_2)$，需同时满足容量与随机检验 `2·g2 > uniform(0,1)` 才派发，权重记 $g_2$。

归一化保证两个权重之和为 1（均派发时 $y_s$ 是凸组合）。两个 expert 均未派发成功时 $G_{s,\cdot}$ 为零向量，token 表示经残差连接直通下一层[^src-lepikhin-2020-gshard]。

## 辅助负载均衡损失

总损失与辅助项（§2.2 + Algorithm 1 line 13）：

$$\mathcal{L} = \ell_{\mathrm{nll}} + k\,\ell_{\mathrm{aux}}, \qquad \ell_{\mathrm{aux}} = \frac{1}{E}\sum_{e=1}^{E} \frac{c_e}{S}\, m_e, \qquad m_e = \frac{1}{S}\sum_{s=1}^{S} g_{s,e}$$

$c_e$ 是组内派发给 expert $e$ 的 token 计数，$S$ 是组内 token 数，$m_e$ 是 expert $e$ 的平均门控值，$k$ 为常数乘子[^src-lepikhin-2020-gshard]。设计目标是最小化路由比例 $c_e/S$ 的均方。因 $c_e$ 由 top-2 导出不可微，故用可微的 $m_e$ 作代理，将 $(c_e/S)^2$ 换为乘积 $m_e (c_e/S)$，即可用梯度下降优化[^src-lepikhin-2020-gshard]。

论文在两处说明与 Shazeer 2017 的关系：§2.2 指出 MoE 层 "based on [16] with variations in the sparse gating function and the auxiliary loss"；§6 指出选择扩展 [16] 源于其灵活且易移植至 Transformer。论文未包含 [16] 本身的损失公式与逐项对比[^src-lepikhin-2020-gshard]。

## 随机路由

正文指出，$y_s$ 是所选 expert 输出的加权平均；若第二 expert 权重很小，可直接忽略以节省 expert 总容量，因此 GATE 以正比于权重 $g_2$ 的概率向第二 expert 派发[^src-lepikhin-2020-gshard]。伪代码给出的精确条件为 `2·g2 > rnd`（Algorithm 1 line 17–19，注释注明 "with probability ∝ 2 · g2"），派发权重采用归一化的 $g_2$，不再额外缩放；正文表述为 "∝ $g_2$"，伪代码表述为 "∝ $2 g_2$"。$[\mathrm{INFERENCE}]$ 因子 2 的作用在于：归一化后 $g_2 \leq 1/2$（$g_1 \geq g_2$），$2 g_2 \in [0,1]$ 恰好覆盖均匀分布检验的值域，等价于以概率 $2 g_2$ 派发。

## Expert 容量与溢出

batch 包含 $N$ 个 token，每 token 至多派发至 2 个 expert，expert 容量按 $O(N/E)$ 设定；实现中将 batch 均分为 $G$ 组并行 gating，每组分给每个 expert 的缓冲容量为 $C = \frac{2N}{G \cdot E} = \frac{2S}{E}$[^src-lepikhin-2020-gshard]。若 token 的派发目标均超出容量，则记为 overflowed token，其 MoE 贡献被丢弃，仅经残差直通[^src-lepikhin-2020-gshard]。对于缩放极限，$C = O(1/D)$ 不能无限缩小，论文以 $C \geq 1$ 为界，指出该设定可支持数千 expert（§5.2）。全文未报告具体溢出率或容量因子取值。

## 数据流与维度变换

**维度符号**：$N$ batch 内 token 总数；$G$ 组数，$S = N/G$ 每组 token 数；$E$ 每层 expert 数；$C$ 每组分给每 expert 的缓冲槽数；$M = d_{\mathrm{model}} = 1024$；$H$ FFN 隐层宽度 = 8192；$D$ 设备数。张量按 $[\text{组 } G, \text{token } S, \text{expert } E, \text{缓冲 } C, \text{通道}]$ 记。

### Stage 1 · 组内门控打分

$$\begin{aligned}
g &= \mathrm{softmax}\!\left(\mathrm{einsum}(\text{"GSM,ME}\to\text{GSE"},\, x, W_g)\right) && \text{// } [G, S, \bcancel{M}]\times[\bcancel{M}, E]\to[G, S, E]\ \text{——消去通道维 } M\text{，每个 token 对全部 } E \text{ 个 expert 各得一个非负门控值}
\end{aligned}$$

### Stage 2 · Top-2 路由决策（Algorithm 1，两个 token 循环）

**支路 A：第一 expert（确定性）**

$$\begin{aligned}
(g_{(1)}, e_1, g_{(2)}, e_{2}) &= \operatorname{Top}_2(g_{s,\cdot}) && \text{// } [G, S, E] \xrightarrow{\text{逐 token 沿 } E \text{ 维取 top-2}} [G, S, 2]\ \text{——每 token 取出最高/次高 expert 下标与门控值}\\
\tilde{g}_1 &= g_{(1)}/(g_{(1)}+g_{(2)}) && \text{// } [G, S]\ \text{——归一化第一权重；} c_{e_1} < C \text{ 即写入 } G_{s,e_1}\text{，计数器无条件 } {+}1
\end{aligned}$$

**支路 B：第二 expert（随机 + 容量）**

$$\begin{aligned}
\tilde{g}_2 &= g_{(2)}/(g_{(1)}+g_{(2)}), \qquad \text{派发} \iff c_{e_2} < C \wedge 2\tilde{g}_2 > \mathrm{uniform}(0,1) && \text{// } [G, S]\ \text{——概率派发；}\tilde{g}_2 \leq 1/2 \text{ 使 } 2\tilde{g}_2 \in [0,1]\text{ 铺满均匀检验值域}
\end{aligned}$$

两支路汇成 4-D 张量 combine$\_$weights $[G,S,E,C]$：元素 $(g,s,e,c)$ 非零表示组 $g$ 的 token $s$ 进入 expert $e$ 的缓冲位 $c$，每个 $(g,s)$ 切片至多两个非零；置一得 0/1 dispatch$\_$mask[^src-lepikhin-2020-gshard]。

### Stage 3 · all-to-all dispatch

$$\begin{aligned}
x^{\mathrm{disp}} &= \mathrm{einsum}(\text{"GSEC,GSM}\to\text{EGCM"},\, \mathrm{dispatch\_mask},\, x) && \text{// } [G, \bcancel{S}, E, C]\times[G, \bcancel{S}, M]\to[E, G, C, M]\ \text{——沿 } S \text{ 求和，one-hot 掩码把 token 表示复制进所派发 expert 的缓冲槽}
\end{aligned}$$

通信位置：dispatch einsum 的输出从 $G$ 维分片 reshard 到 $E$ 维分片，经 AllToAll 由每个设备收齐发往所持 expert 的 token 缓冲，此为第一阶段通信[^src-lepikhin-2020-gshard]。

### Stage 4 · expert 前馈

$$\begin{aligned}
h &= \mathrm{ReLU}\!\left(\mathrm{einsum}(\text{"EGCM,EMH}\to\text{EGCH"},\, x^{\mathrm{disp}}, W_i)\right) && \text{// } [E, G, C, \bcancel{M}]\times[\bcancel{M}, H]\to[E, G, C, H]\ \text{——每个 expert 对缓冲内 token 做输入投影}\\
y^{\mathrm{exp}} &= \mathrm{einsum}(\text{"EGCH,EHM}\to\text{GECM"},\, h, W_o) && \text{// } [E, G, C, \bcancel{H}]\times[\bcancel{H}, M]\to[G, E, C, M]\ \text{——输出投影并重排回组优先布局}
\end{aligned}$$

### Stage 5 · all-to-all combine

$$\begin{aligned}
y &= \mathrm{einsum}(\text{"GSEC,GECM}\to\text{GSM"},\, \mathrm{combine\_weights},\, y^{\mathrm{exp}}) && \text{// } [G, S, \bcancel{E}, \bcancel{C}]\times[G, \bcancel{E}, \bcancel{C}, M]\to[G, S, M]\ \text{——消去 } E, C \text{ 两维，按门控权重加权平均回 token 表示；溢出 token 权重全零、经残差直通}
\end{aligned}$$

combine 前执行第二阶段 AllToAll，把 expert 输出从 $E$ 维分片送回 token 所属的 $G$ 维分片[^src-lepikhin-2020-gshard]。

**参数与计算对比**：dense FFN 层参数 $2MH$，MoE 层参数 $2EMH$。每 token 前向计算包含 top-2 次 FFN 与一层门控 softmax，$[\mathrm{INFERENCE}]$ 由式 (2) 可得：FFN 部分与 $E$ 无关（恒为 $2 \times 2MH$ 量级），softmax 门控 $O(ME)$，计算对 $E$ 次线性——即论文 "每个 token 只激活一个小子网络，其规模与 expert 数大致无关" 的代数形式[^src-lepikhin-2020-gshard]。

## 系统：annotation API 与 SPMD 编译

三个注解 API 包括 `replicate(tensor)`（跨分区复制，用于非 MoE 层权重）、`split(tensor, dim, n)`（沿一维均分）与 `shard(tensor, device_assignment)`（多维泛化，`device_assignment` 为与张量同 rank 的设备 ID 数组）[^src-lepikhin-2020-gshard]。注解只添加分区信息，不改变逻辑形状，用户始终操作全形状[^src-lepikhin-2020-gshard]。MoE 模型的具体用法：输入沿 $G$ 维 split、门控权重 replicate，dispatch einsum 输出经 AllToAll 从 $G$ 维 reshard 到 $E$ 维；注意力层采用 batch 分片与权重复制，整个模型在 DP 与 MP 间按层切换[^src-lepikhin-2020-gshard]。

编译器只要求用户注解少数关键算子，其余分片通过迭代数据流分析向邻居传播并对齐，以减少 resharding；自动分配采用启发式，整数规划/学习方法留作 future work[^src-lepikhin-2020-gshard]。SPMD partitioner 把整个计算图转换为在所有设备上运行的单一程序，编译时间与分区数无关——MPMD 为不同分片生成专用程序会导致图节点线性膨胀与编译时间爆炸（§7 lessons）[^src-lepikhin-2020-gshard]。相比 Mesh-TensorFlow 需用 Python 库重写计算以实现逐算子分割，GShard 在编译器内基于注解完成，用户无需重写模型[^src-lepikhin-2020-gshard]。

通信原语扩展表现（§5.3 微基准，8/32 MB 负载，16→2048 分区）：AllReduce 耗时与设备数无关（$O(1)$）；AllToAll 在 2D torus 上 $O(\sqrt{D})$，设备增加 128× 耗时仅增 9×；AllGather 固定输入时 $O(D)$；CollectivePermute 源宿就近时 $O(1)$[^src-lepikhin-2020-gshard]。

## M4 实验：600B 翻译模型

**设定**（§4.1–4.3）：训练语料来自网络挖掘的 100 语言↔英语并行文档，共 25B 训练样本，筛选 X→英语方向后约 13B 样本用于训练；评测覆盖 100 个 X→英语方向（排除 Kazakh、Latin 两对）[^src-lepikhin-2020-gshard]。各模型均训练至 1T 子词 token（250k 步 × 约 4M token/步），未出现 plateau，继续训练 loss 仍下降[^src-lepikhin-2020-gshard]。超参（A.2）：$d_{\mathrm{model}}$ 1024、FFN/MoE 隐层 8192、16 heads、dropout 0.1、Adafactor（$\beta_1 = 0$，lr 1.0 十 k 步后开方衰减）、SentencePiece 源端 102 语言 64000 / 目标端英语 32000[^src-lepikhin-2020-gshard]。

**模型族**（Table 1，$L$ × $E$ 二维扫描）：

| Id | 模型 | 每层 experts | 权重 | TPU v3 cores |
|---|---|---|---|---|
| (1) | MoE(2048E, 36L) | 2048 | 600B | 2048 |
| (2) | MoE(2048E, 12L) | 2048 | 200B | 2048 |
| (3) | MoE(512E, 36L) | 512 | 150B | 512 |
| (4) | MoE(512E, 12L) | 512 | 50B | 512 |
| (5) | MoE(128E, 36L) | 128 | 37B | 128 |
| (6) | MoE(128E, 12L) | 128 | 12.5B | 128 |

（另有 MoE(2048E, 60L) 训练设备 2048、权重 1T 的试跑，见下文边界。）训练设备数固定为每层 expert 数，并非硬性要求[^src-lepikhin-2020-gshard]。

**翻译质量**（Figure 6 内嵌表，100 个 X→英语方向平均）：

| 模型 | BLEU avg. | ∆BLEU avg. | 权重 |
|---|---|---|---|
| MoE(2048E, 36L) | 44.3 | +13.5 | 600B |
| MoE(512E, 36L) | 43.7 | +12.9 | 150B |
| MoE(128E, 36L) | 39.0 | +8.2 | 37B |
| T(96L)（GPipe dense 基线） | 36.9 | +6.1 | 2.3B |
| 100 个单语调优双语基线 | 30.8 | — | 100×0.4B |

96 层 dense Transformer T(96L) 采用 GPipe 流水线并行，2048 核收敛需 6 周以上，预算约 235.5 TPU v3 core years；600B MoE 模型需 4.0 天、22.4 core years[^src-lepikhin-2020-gshard]。§7 口径：训练全部 100 个双语基线合计 29 TPU years，高于单个 600B MoE 的 22[^src-lepikhin-2020-gshard]。

**样本效率**（Table 2，达到给定训练交叉熵所需 token 数，单位 10 亿）：固定 $E$，$L$ 12→36 达到同损失只需 1/2–1/3 的 token（如 MoE(128E, 12L) 995 vs MoE(128E, 36L) 321）；固定 $L$ = 36，experts 128→512 需求从 321 骤降到 66，512→2048 回升到 82，论文把容量瓶颈定位在 128–512 experts 之间[^src-lepikhin-2020-gshard]。

**训练成本**（Table 3）：MoE(2048E, 36L) 0.72 steps/s、4M token/batch、22.4 core years、4.0 天、BLEU 44.3；同核数 dense T(96L) 约 235.5 core years、约 42 天、BLEU 36.9[^src-lepikhin-2020-gshard]。

§4.4 全部消融仅覆盖 $L$×$E$ 扫描，未涉及 capacity factor、gating 变体或辅助损失权重。固定 experts 加深网络（12L→36L）平均提高 2–3 BLEU，高低资源均有提升[^src-lepikhin-2020-gshard]；固定 12 层增加 experts 时，128→512 平均 +3.3 BLEU，512→2048 仅 +1.3，收益递减[^src-lepikhin-2020-gshard]。增加 expert 更利于高资源方向；低资源方向主要受益于共享子网迁移，收益递减[^src-lepikhin-2020-gshard]。深 dense 模型（T(96L)）对低资源方向的正迁移强于浅 MoE 模型[^src-lepikhin-2020-gshard]。

## 系统性能（§5）

SPMD 分割后，replicated 权重、distributed 权重与激活在单设备上的占用恒定，不随 expert 数增加；激活超出显存时编译器自动 rematerialization（重算耗时占 36L 总周期 28%、60L 34%、24L 以下为 0）[^src-lepikhin-2020-gshard]。128 扩至 2048 experts（设备 16×）时执行时间仅增至 1.7×；128 experts 时硬件利用率达 >70% roofline，2048 时为 48%[^src-lepikhin-2020-gshard]。FFN 与投影大矩阵乘达 >85% peak FLOPS，attention 受带宽限制（batch matmul）仅 >30%[^src-lepikhin-2020-gshard]。MoE 算子中，gate 投影 einsum $O(D)$ 占比极小；ArgMax 与 Cumsum 等通用门控算子在 128 experts 时耗时可忽略，2048 时占 MoE+Transformer 总时间 <10%；dispatch/combine einsum 对应 one-hot Gather，成本 $O(GC)=O(1)$ 与 expert 数无关，128→2048 仅约 2×[^src-lepikhin-2020-gshard]。dispatch/combine 的 AllToAll 通信代价为 $O(\sqrt{D})$，16× 扩展时间约 3.75×，占 MoE+Transformer 执行时间从 16% 升至 36%[^src-lepikhin-2020-gshard]。

## 边界

MoE(2048E, 60L)、bfloat16 激活与 1T 权重的试跑因数值稳定问题未收录结果，正式训练采用 float32 权重与激活（§4.3）[^src-lepikhin-2020-gshard]。组容量 $C = O(1/D)$ 受 $C \geq 1$ 约束，无法无限扩展（§5.2）[^src-lepikhin-2020-gshard]。溢出 token 的 MoE 贡献被整体丢弃，§2.2 给出定义但全文未报告溢出率数字[^src-lepikhin-2020-gshard]。参数计数不总与规模下的有效容量对应，模型比较需结合问题属性与运行模式（§7）[^src-lepikhin-2020-gshard]。实验仅覆盖多语言翻译，辅助损失权重 $k$、capacity factor 与 top-1/top-2 等设定直接固定给出，未作消融（§4.4）[^src-lepikhin-2020-gshard]。

[^src-lepikhin-2020-gshard]: [[source-lepikhin-2020-gshard]]
[^src-dai-2024-deepseekmoe]: [[source-dai-2024-deepseekmoe]]
