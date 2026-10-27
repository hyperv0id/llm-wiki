---
title: "DeepSeekMoE：细粒度专家分割与共享专家隔离"
type: technique
tags:
  - mixture-of-experts
  - sparse-gating
  - expert-specialization
  - load-balancing
  - llm
created: 2026-10-27
last_updated: 2026-10-27
source_count: 3
confidence: medium
status: active
---

# DeepSeekMoE

DeepSeekMoE（Dai et al., DeepSeek-AI / 北大 / 清华 / 南大, arXiv:2401.06066v1；ACL 2024 的著录来自用户，PDF 内未核实）在 [[gshard|GShard]] 的 top-2 稀疏 MoE 上调整两处：把每个 FFN 专家沿中间维切成 $m$ 份、激活 $mK$ 份，再隔离 $K_s$ 个不经 router 的共享专家[^src-dai-2024-deepseekmoe]。相比 [[sparse-gated-moe-layer|Shazeer 稀疏门控层]] 与 GShard，分割粒度与共享专家是两个新增自由度，门控本身仍是 softmax 打分加 top-$k$ 选择[^src-dai-2024-deepseekmoe]。

规模扩展结果显示：2B 总参 / 0.3B 激活的验证模型在 100B token 上超过同参数量 GShard，并追平专家参数与计算量各 1.5 倍的 GShard 2.9B（表 1、表 2，后者记 GShard×1.5 的专家参数 2.83B）[^src-dai-2024-deepseekmoe]；16.4B 总参 / 2.8B 激活的 16B 模型在 2T token 上以 40.5% 的计算量与 6.9B dense 的 DeepSeek 7B 相当（表 3），以 39.6% 的计算量对比 LLaMA2 7B（表 4）[^src-dai-2024-deepseekmoe]；145B 在 245B token 的初步训练中以 28.5% 的计算量对标 DeepSeek 67B（表 6）[^src-dai-2024-deepseekmoe]。

## 问题：专家特化被两件事挡住

论文将专家特化定义为每个专家获得互不重叠、聚焦的知识，并指出阻碍特化的两个现象（§1）[^src-dai-2024-deepseekmoe]：

知识杂糅（knowledge hybridity）指常用配置只有 8 或 16 个专家，分到专家的 token 覆盖多种知识，专家只能把难以同时使用的知识塞进同一组参数[^src-dai-2024-deepseekmoe]；知识冗余（knowledge redundancy）指不同专家服务的 token 需要同一部分公共知识，多个专家会在各自参数里重复收敛出它[^src-dai-2024-deepseekmoe]。

两项改动与两种现象一一对应：把专家切细，让同一 token 能组合更多专家；再抽出一组专存公共知识的专家，腾出路由专家的容量[^src-dai-2024-deepseekmoe]。

## 细粒度专家分割

保持专家总参数不变，将每个专家的 FFN 中间维降至 $1/m$、总专家数增至 $mN$，同时将激活数从 $K$ 提至 $mK$（式 6–8）[^src-dai-2024-deepseekmoe]。参数与激活记账式：

$$\begin{aligned}
P_{\text{传统}} &= N\cdot 2dd_{ff}, & C_{\text{传统}} &= K\cdot 2dd_{ff} && \text{// 每 token 只走 } K \text{ 个完整 FFN}\\
P_{\text{细粒度}} &= mN\cdot 2d\frac{d_{ff}}{m} = N\cdot 2dd_{ff}, & C_{\text{细粒度}} &= mK\cdot 2d\frac{d_{ff}}{m} = K\cdot 2dd_{ff} && \text{// 专家参数缩小 } m \text{ 倍、激活个数放大 } m \text{ 倍，两个账都持平}
\end{aligned}$$

细分专家带来组合自由度。$N=16$、top-2 只有 $\binom{16}{2}=120$ 种激活组合；将每个专家切为 4 份后 $mN=64$、激活 $mK=8$，组合数达到 $\binom{64}{8}=4{,}426{,}165{,}368$[^src-dai-2024-deepseekmoe]。参数与算力保持不变，可组合的激活专家集合因此大幅增加[^src-dai-2024-deepseekmoe]。

## 共享专家隔离

隔离 $K_s$ 个专家恒激活，路由侧激活数相应从 $mK$ 减至 $mK-K_s$，算力保持不变；最终路由专家总数为 $mN-K_s$，非零门控数为 $mK-K_s$（式 9–11）[^src-dai-2024-deepseekmoe]。

$$\begin{aligned}
h_t^l &= \sum_{i=1}^{K_s}\operatorname{FFN}_i(u_t^l) + \sum_{i=K_s+1}^{mN} g_{i,t}\operatorname{FFN}_i(u_t^l) + u_t^l && \text{// 第一项不含 } g_{i,t}\text{：共享专家按式 (9) 直接相加，没有 router 权重}
\end{aligned}$$

共享专家不带门控权重，也不参与 top-$k$；这组参数用于吸收跨上下文的公共知识，降低路由专家之间的重复[^src-dai-2024-deepseekmoe]。该策略原型见 Rajbhandari et al. (2022)，区别在于对方出于工程考量，本文则从算法角度出发[^src-dai-2024-deepseekmoe]。

## 数据流与维度变换

> 维度符号：$T$ 为 batch 内 token 数；$d$ 为隐层维；$d_{ff}$ 为标准 FFN 的中间维（细粒度专家为 $d_{ff}/m$）；$N$ 为传统专家数；$m$ 为细粒度倍率，$mN$ 为切分后的专家总数；$K$ 为传统激活数；$K_s$ 为共享专家数；激活的路由专家数为 $mK-K_s$。$u_t^l\in\mathbb{R}^{d}$ 是第 $l$ 个 block 的输入 token 表示，$e_i\in\mathbb{R}^{d}$ 是第 $i$ 个专家的质心。

### Stage 1 · 对全部专家打分

$$\begin{aligned}
s_{i,t} &= \operatorname{softmax}_i\big(u_t^{l\top} e_i\big) && \text{// } [T,\bcancel{d}]\times[\bcancel{d},mN]\to[T,mN]\ \text{——消去 } d\text{，每行是一个 token 在 } mN \text{ 个专家质心上的打分分布}
\end{aligned}$$

打分只用专家质心 $e_i$，与 expert FFN 参数无关；共享专家位于 $i=1,\dots,K_s$，不参与式 (10) 的 top-$k$ 选择[^src-dai-2024-deepseekmoe]。

### Stage 2 · 在路由专家集合内取 top-$(mK-K_s)$

$$\begin{aligned}
g_{i,t} &= \begin{cases} s_{i,t} & s_{i,t}\in\operatorname{Top}_{mK-K_s}\big(\{s_{j,t}\}_{j=K_s+1}^{mN}\big)\\ 0 & \text{otherwise}\end{cases} && \text{// } [T,mN]\xrightarrow{\text{沿专家维取 top-}(mK-K_s)}[T,mK-K_s]\ \text{——只留非零项；下标作用回 } [T,mN] \text{ 得稀疏门控矩阵}
\end{aligned}$$

式 (10) 把未选中门控直接置零，选中项沿用 softmax 原值且不在选中集合上重归一，因此非零门控之和小于 1；GShard 则把 top-2 权重归一到和为 1[^src-dai-2024-deepseekmoe][^src-lepikhin-2020-gshard]。

### Stage 3 · 共享专家支路（恒激活、无门控）

$$\begin{aligned}
h^{s}_t &= \sum_{i=1}^{K_s}\operatorname{FFN}_i(u_t^l) && \text{// } [T,\bcancel{d}]\times[\bcancel{d},d_{ff}/m]\times[\bcancel{d_{ff}/m},d]\to[T,d]\ \text{——}K_s\text{ 个专家各自升维再降回，直接相加}
\end{aligned}$$

每个 token 都走这 $K_s$ 个专家，路由决策不影响该支路；其参数计入激活参数，但不受负载均衡损失约束[^src-dai-2024-deepseekmoe]。

### Stage 4 · 路由专家加权支路

$$\begin{aligned}
h^{r}_t &= \sum_{i=K_s+1}^{mN} g_{i,t}\operatorname{FFN}_i(u_t^l) && \text{// 每个被选专家 }[T,\bcancel{d}]\times[\bcancel{d},d_{ff}/m]\times[\bcancel{d_{ff}/m},d]\to[T,d]\text{，再按 } g_{i,t}\in[T,mN] \text{ 加权}\xrightarrow{\text{沿选中集合求和}}[T,d]
\end{aligned}$$

实现上只对被选中的 $mK-K_s$ 个专家做前向，其余专家的 $g_{i,t}=0$ 不参与计算，稀疏性由此体现[^src-dai-2024-deepseekmoe]。

### Stage 5 · 残差汇合

$$\begin{aligned}
h_t^l &= h^{s}_t + h^{r}_t + u_t^l && \text{// } [T,d]+[T,d]+[T,d]\to[T,d]\ \text{——共享输出、路由输出与输入残差相加，成为该 block 的输出}
\end{aligned}$$

式 (9) 将三项合写为一式，等价于在 MoE 层输出上增加残差连接[^src-dai-2024-deepseekmoe]。

## 负载均衡：expert-level 与 device-level

路由需要两个辅助损失。expert-level 项防路由塌缩，device-level 项防设备间算力瓶颈（式 12–17）[^src-dai-2024-deepseekmoe]：

$$\begin{aligned}
\mathcal{L}_{\mathrm{ExpBal}} &= \alpha_1\sum_{i=1}^{N'} f_i P_i && \text{// } [N']\text{ 维的两个向量逐项相乘求和}\\
f_i &= \frac{N'}{K'T}\sum_{t=1}^{T}\mathbf{1}(\text{token } t \text{ selects expert } i) && \text{// } [T]\xrightarrow{\text{逐专家计数并乘 } N'/K'}[N']\ \text{——归一化选择频率，完全均匀时 } f_i=1\\
P_i &= \frac{1}{T}\sum_{t=1}^{T} s_{i,t} && \text{// } [T,mN]\xrightarrow{\text{沿 token 维取均值}}[mN]\ \text{——batch 内平均亲和度}
\end{aligned}$$

其中 $N'=mN-K_s$、$K'=mK-K_s$，$\alpha_1$ 为 expert-level balance factor[^src-dai-2024-deepseekmoe]。$f_i$ 为选择次数按 $N'/K'$ 归一后的频率（完全均匀时为 $f_i=1$），$P_i$ 为该专家在 batch 内的平均亲和度；二者乘积惩罚被选得多且打分高的专家。这与 GShard 的组级辅助损失 $\frac{1}{E}\sum_e \frac{c_e}{S}m_e$ 同族，区别是 DeepSeekMoE 在路由专家上逐专家计算，并另加设备层一项[^src-dai-2024-deepseekmoe][^src-lepikhin-2020-gshard]。

$$\begin{aligned}
\mathcal{L}_{\mathrm{DevBal}} &= \alpha_2\sum_{i=1}^{D} f_i' P_i' && \text{// } [D]\text{ 维两项相乘求和（}D\text{ 为设备数）}\\
f_i' &= \frac{1}{|\mathcal{E}_i|}\sum_{j\in\mathcal{E}_i} f_j && \text{// } [N']\xrightarrow{\text{按设备分组取均值}}[D]\ \text{——设备内专家选择频率的平均}\\
P_i' &= \sum_{j\in\mathcal{E}_i} P_j && \text{// } [N']\xrightarrow{\text{按设备分组求和}}[D]\ \text{——设备内平均亲和度之和}
\end{aligned}$$

将路由专家分为 $D$ 组 $\{\mathcal{E}_1,\dots,\mathcal{E}_D\}$，每组对应一块设备：$f_i'$ 组内取均值、$P_i'$ 组内求和，以此惩罚设备级而非专家级不均衡[^src-dai-2024-deepseekmoe]。论文设定 $\alpha_1$ 取小值防塌缩、$\alpha_2$ 取较大值均衡设备负载，因过强的专家级约束会损害模型性能[^src-dai-2024-deepseekmoe]。

配置上，2B 验证 $\alpha_1=0.01$，单卡部署，不启用 device-level 损失且不丢弃 token[^src-dai-2024-deepseekmoe]；16B 取 $\alpha_1=0.001$，同样不丢 token 且不用 device-level 损失[^src-dai-2024-deepseekmoe]；145B 将 128 个路由专家均分到 4 块设备，$\alpha_1=0.003$、$\alpha_2=0.05$[^src-dai-2024-deepseekmoe]。门控全文固定为对专家质心的 softmax，无噪声项与 gate bias；论文未报告 sigmoid gate 或其他打分变体的消融[^src-dai-2024-deepseekmoe]。

## 与 GShard、稀疏门控层的机制对照

| 机制 | Shazeer 2017 | GShard | DeepSeekMoE |
|---|---|---|---|
| 打分 | $(xW_g)_i+\epsilon\cdot\operatorname{softplus}((xW_{\text{noise}})_i)$，带可学习噪声 | $\operatorname{softmax}(w_g x)$ | $\operatorname{softmax}_i(u_t^\top e_i)$，无噪声、无 bias |
| 稀疏化 | KeepTopK 后 softmax，选中权重和为 1 | top-2 后按 $g_1/(g_1+g_2)$ 归一，第二专家按概率派发 | softmax 打分后把非 top-$(mK-K_s)$ 置零，不重归一 |
| 专家尺寸 | 与标准 FFN 相同 | 与标准 FFN 相同 | 中间维缩到 $1/m$ |
| 每 token 激活 | $k$（随实验设定，多语模型 $k=2$） | 2 | $mK$，其中 $K_s$ 个为共享专家 |
| 共享专家 | 无 | 无 | $K_s$ 个，恒激活、无门控权重 |
| 均衡损失 | importance $\mathrm{CV}^2$ + load $\mathrm{CV}^2$ | 组级 $\frac{1}{E}\sum_e\frac{c_e}{S}m_e$ | expert-level $f_iP_i$ + device-level $f_i'P_i'$ |
| 容量与丢弃 | capacity factor，溢出丢弃 | 容量 $C$，溢出丢弃 | 2B/16B 明确不丢 token |

三个模型列依次对应 [[sparse-gated-moe-layer]]、[[gshard]] 与本文；表中 DeepSeekMoE 各行的出处为式 5–17、§3.3 与 §4.1.3/§5.1.2[^src-dai-2024-deepseekmoe][^src-lepikhin-2020-gshard][^src-shazeer-2017-sparse-moe]。

## 2B 验证实验

设置：9 层、hidden 1280、10 个 attention head（每头 128 维）、8K BPE 词表、100B token；全部 FFN 替换为 MoE 层，专家总参数为标准 FFN 的 16 倍、激活专家参数（含共享）为 2 倍；共 1 个共享专家 + 63 个路由专家，每个专家为标准 FFN 的 0.25 倍，每 token 激活 1 个共享 + 7 个路由[^src-dai-2024-deepseekmoe]。四个对照模型的规模分别为 Dense 0.2B（总参数与激活参数均为 0.2B）、Hash Layer 与 Switch 各 2.0B 总参 / 0.2B 激活、GShard 2.0B / 0.3B；DeepSeekMoE 为 2.0B / 0.3B，与 GShard 的总参数及激活参数完全对齐[^src-dai-2024-deepseekmoe]。

表 1（100B token 后的最终结果）[^src-dai-2024-deepseekmoe]：

| 指标 | Dense | Hash Layer | Switch | GShard | DeepSeekMoE |
|---|---|---|---|---|---|
| Pile (Loss) | 2.060 | 1.932 | 1.881 | 1.867 | **1.808** |
| HellaSwag | 38.8 | 46.2 | 49.1 | 50.5 | **54.8** |
| PIQA | 66.8 | 68.4 | 70.5 | 70.6 | **72.3** |
| ARC-easy | 41.0 | 45.3 | 45.9 | 43.9 | **49.4** |
| ARC-challenge | 26.0 | 28.2 | 30.2 | 31.6 | **34.3** |
| RACE-middle | 38.8 | 38.8 | 43.6 | 42.1 | **44.0** |
| RACE-high | 29.0 | 30.0 | 30.9 | 30.4 | **31.7** |
| HumanEval | 0.0 | 1.2 | 2.4 | 3.7 | **4.9** |
| MBPP | 0.2 | 0.6 | 0.4 | 0.2 | **2.2** |
| TriviaQA | 4.9 | 6.5 | 8.9 | 10.2 | **16.6** |
| NaturalQuestions | 1.4 | 1.4 | 2.5 | 3.2 | **5.7** |

在与 GShard 参数量、激活量及算力（4.3T FLOPs / 2K token）完全相同时，DeepSeekMoE 在 11 项上均最优（Pile loss 最低）[^src-dai-2024-deepseekmoe]。表 2 扩大对照：GShard×1.5 专家尺寸 1.5 倍、专家参数 2.83B、激活专家参数 0.35B、5.8T FLOPs，Pile loss 为 1.808；DeepSeekMoE 以 1.89B / 0.24B / 4.3T 取得相同的 1.808[^src-dai-2024-deepseekmoe]。Dense×16 列（16 个共享专家均与标准 FFN 同参数，1.89B 专家参数全激活，24.6T FLOPs）Pile loss 为 1.806，论文将其作为 MoE 容量上界，并说明未采用 attention:FFN = 1:2 的常见配比[^src-dai-2024-deepseekmoe]。

## 消融（图 3）

三组消融均保持总参数与激活参数一致（§4.4）[^src-dai-2024-deepseekmoe]：

共享专家隔离在 GShard 上隔离 1 个专家作共享，多数 benchmark 提升[^src-dai-2024-deepseekmoe]。细粒度分割把每个专家切成 2 份或 4 份，得到 32 个（1 共享 + 31 路由）或 64 个（1 共享 + 63 路由）专家，效果随粒度变细单调提升[^src-dai-2024-deepseekmoe]。确定共享与路由比例时，在 64 专家、总参数与激活数不变的前提下隔离 1、2、4 个共享专家，Pile loss 分别为 1.808、1.806、1.811；因 1:3 略好，放大规模时固定共享专家 : 激活路由专家 = 1:3[^src-dai-2024-deepseekmoe]。

图 3 是归一化柱状图，坐标刻度与柱值在 PDF 文本层不可读，正文只给上述结论句，没有逐项数值[^src-dai-2024-deepseekmoe]。

## 专家冗余与知识获取（图 4–6、§4.5）

这组实验用同一套 2B 模型（1 共享 + 7/63 路由激活）检验路由专家的可替代性[^src-dai-2024-deepseekmoe]：

图 4 禁用高分路由专家，对每个 token 屏蔽掉路由概率最高的一部分专家，再从剩余专家中取 top-$K$；DeepSeekMoE 对屏蔽比例比 GShard×1.5 更敏感（两者未屏蔽时 Pile loss 相同），路由专家冗余更低、每个专家更不可替代[^src-dai-2024-deepseekmoe]，正文未给出屏蔽比例与 Pile loss 的具体数值。§4.5 禁用共享专家，去掉共享专家、改多激活 1 个路由专家以保持算力，Pile loss 从 1.808 升到 2.414[^src-dai-2024-deepseekmoe]。图 5 减少激活专家，在 3–7 之间扫描激活路由专家数，只用 4 个路由专家时 Pile loss 已与 GShard 相当[^src-dai-2024-deepseekmoe]。图 6 将激活参数减半，从零训练 1 共享 + 63 路由、仅激活 3 个路由专家的模型，在总专家参数相同、激活专家参数只有一半时仍优于 GShard[^src-dai-2024-deepseekmoe]。

## 16B 模型

模型设置 28 层、hidden 2048、16 个 head（每头 128 维）、2 个共享专家与 64 个路由专家（每个 0.25 倍标准 FFN）、每 token 激活 2 + 6、词表 100K，训练 2T token，最大学习率 $4.2\times10^{-4}$、batch 4608 × 4096（18M token）、$\alpha_1=0.001$。除第一层外全部 FFN 替换为 MoE 层，因第一层负载均衡收敛明显更慢[^src-dai-2024-deepseekmoe]。总参数 16.4B、激活参数 2.8B，每 4K token 对应 74.4T FLOPs[^src-dai-2024-deepseekmoe]。

表 3 对比 DeepSeek 7B（6.9B dense、183.5T FLOPs / 4K token、同语料 2T token）[^src-dai-2024-deepseekmoe]：Pile BPB 0.74 对 0.75，HellaSwag 77.1 对 75.4，PIQA 80.2 对 79.2，ARC-challenge 49.8 对 48.1，GSM8K 18.8 对 17.4，MATH 4.3 对 3.3，TriviaQA 64.8 对 59.7，NaturalQuestions 25.5 对 22.2；落后项包括 MMLU 45.0 对 48.2、CEval 40.6 对 45.0、CMMLU 42.5 对 47.2、DROP 32.9 对 34.9、RACE-middle 61.9 对 63.2。表注标明计算量为 40.5%。

表 4 对比 LLaMA2 7B（6.7B、187.9T FLOPs）[^src-dai-2024-deepseekmoe]：Pile BPB 0.74 对 0.76，HumanEval 26.8 对 14.6，MBPP 39.2 对 21.8，CHID 89.4 对 37.9，CEval 40.6 对 33.9，CMMLU 42.5 对 32.6，CLUEWSC 72.1 对 64.0，MMLU 45.0 对 45.8，NaturalQuestions 25.5 对 25.5。总参数为对方的 245%，计算量为 39.6%。

图 1 的 Open LLM Leaderboard 平均性能—激活参数图显示，16B 高于同激活参数量的开源模型，与激活参数约 2.5 倍的 LLaMA2 7B 相当[^src-dai-2024-deepseekmoe]。§5.2.1 报告在适当算子优化下该模型可单卡 40GB 部署、推理速度约为 7B dense 的 2.5 倍，但未给出测量口径[^src-dai-2024-deepseekmoe]。

附录 C 图 7 给出 16B 与 DeepSeek 7B 在训练过程中的 benchmark 曲线，仅标注 for reference 且无额外结论；全文未拟合 scaling law 曲线或幂律。

## SFT 后的 chat 模型

SFT 数据 1.4M 条、batch 1024、8 epoch、最大长度 4K、常数学习率 $10^{-5}$，评测去除 Pile 与 CHID、加入 BBH（§6）[^src-dai-2024-deepseekmoe]。表 5 在同一份 SFT 数据下微调 LLaMA2 7B、DeepSeek 7B 与 DeepSeekMoE 16B[^src-dai-2024-deepseekmoe]：HumanEval 45.7 对 35.4 / 45.1；MBPP 46.2 对 27.8 / 39.0；DROP 33.8 对 40.0 / 41.7；MMLU 47.2 对 50.0 / 49.7；CEval 40.0 对 35.1 / 44.7；CMMLU 49.3 对 36.9 / 51.2。代码任务优势在 SFT 后得以保留，选择题差距亦然。

## 145B 初步结果

设置：62 层、hidden 4096、32 个 head（每头 128 维）、4 个共享专家 + 128 个路由专家（每个 0.125 倍标准 FFN）、每 token 激活 4 + 12、144.6B 总参 / 22.2B 激活、245B token（13,000 步）、最大学习率 $3.0\times10^{-4}$、$\alpha_1=0.003$、$\alpha_2=0.05$[^src-dai-2024-deepseekmoe]。

表 6 三方对照（均为 245B token，DeepSeek 67B 为 dense 67.4B、2057.5T FLOPs）[^src-dai-2024-deepseekmoe]：Pile loss 1.876 对 GShard 137B 的 1.961、DeepSeek 67B 的 1.905；MMLU 39.4 对 26.3 / 45.1；CEval 37.1 对 26.2 / 40.3；CMMLU 35.9 对 25.4 / 40.6；HumanEval 19.5 对 17.7 / 23.8；MBPP 33.2 对 27.6 / 33.6；TriviaQA 61.1 对 52.5 / 57.2；HellaSwag 75.8 对 72.0 / 74.8。142B（Half Activated，2 共享 + 6/128 路由、142.3B / 12.2B、374.6T FLOPs）Pile loss 1.888，计算量为 DeepSeek 67B 的 18.2%[^src-dai-2024-deepseekmoe]。DeepSeekMoE 145B 总参数比 GShard 137B 多 6%，因专家中间维对齐到 64 的倍数[^src-dai-2024-deepseekmoe]。

附录表 10 在 13B 规模重复同一对照：DeepSeekMoE 13.3B（专家参数 13.3B、激活 2.05B）对 GShard×1.2 15.9B 与 GShard×1.5 19.8B，HellaSwag 69.1 对 66.6 / 67.7，HumanEval 9.8 对 6.1 / 6.1，MBPP 10.6 对 7.0 / 11.6[^src-dai-2024-deepseekmoe]。附录表 8、表 9 给出 2B 规模与更大 GShard、更大 dense 的逐项对照[^src-dai-2024-deepseekmoe]。

## 边界

语言建模评测采用 Pile 测试集交叉熵（2B 与 145B 表行记 Loss），16B 因分词器与 LLaMA2 7B 不同改用 bits per byte[^src-dai-2024-deepseekmoe]，表 1–6 另列有十余项下游准确率 benchmark。16B 在 MMLU、CEval、CMMLU 上低于 DeepSeek 7B（45.0 对 48.2、40.6 对 45.0、42.5 对 47.2）；对比 LLaMA2 7B 仅 MMLU 略低（45.0 对 45.8），CEval 与 CMMLU 更高；论文将这一弱点归因于 attention 参数仅约 0.5B（DeepSeek 7B 为 2.5B）[^src-dai-2024-deepseekmoe]。16B 规模未再使用更细粒度，理由是专家过小会降低计算效率，16B 以上仍可采用更细粒度[^src-dai-2024-deepseekmoe]。145B 属初步结果，仅训练 245B token，且与 GShard 137B 对比时总参数多 6%[^src-dai-2024-deepseekmoe]。门控打分形式（softmax vs sigmoid）、gate bias、两个 balance factor 的单独扫描及设备分组数 $D$ 的影响均未做实验；图 3–6 仅给趋势与结论句，正文未列柱值或屏蔽比例[^src-dai-2024-deepseekmoe]。1:3 比例仅在 64 专家、2B 规模上由 1.808 / 1.806 / 1.811 三个 Pile loss 选出，最大差值 0.005，论文记为 marginally better[^src-dai-2024-deepseekmoe]。

[^src-dai-2024-deepseekmoe]: [[source-dai-2024-deepseekmoe]]
[^src-lepikhin-2020-gshard]: [[source-lepikhin-2020-gshard]]
[^src-shazeer-2017-sparse-moe]: [[source-shazeer-2017-sparse-moe]]
