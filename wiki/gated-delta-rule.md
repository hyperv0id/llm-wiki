---
title: "Gated Delta Rule（Gated DeltaNet 核心层）"
type: technique
tags:
  - linear-rnn
  - delta-rule
  - fast-weights
  - gating
  - chunkwise-parallel
created: 2026-10-27
last_updated: 2026-10-27
source_count: 1
confidence: medium
status: active
---

# Gated Delta Rule（Gated DeltaNet 核心层）

Gated DeltaNet（Yang, Kautz & Hatamizadeh, ICLR 2025, arXiv:2412.06464）的 token mixer 将 Mamba2 的标量遗忘门引入 DeltaNet 的 delta rule 转移矩阵，兼具快速整体清空与定点改写能力，并通过 WY 表示将带门控递推扩展成分块并行算法。本页记录公式推导、chunkwise 算法与逐表证据；源摘要见 [[source-yang-2024-gated-delta]]。

## 问题：两种记忆操作的互补缺口

线性 Transformer 的状态是外积关联记忆 $S_t=S_{t-1}+v_tk_t^\top$，可存的正交 key-value 对数受维度上界约束，序列超过该上界后「记忆碰撞」使精确检索必然失效。两条既有修补路线各缺一半（§1）[^src-yang-2024-gated-delta]：

Mamba2 的门控更新 $S_t=\alpha_tS_{t-1}+v_tk_t^\top$ 借助数据依赖标量 $\alpha_t\in(0,1)$ 每步把全部 key-value 关联按同一比例衰减，能整体遗忘但擦除不分对象，清除特定关联时只能整体衰减；DeltaNet 的 delta rule 把当前 key 下的旧值软替换为新值，定点精确，但一次只更新一个关联，上下文切换时无法快速清空过时信息，真实任务表现中等。

gating 负责「快速整体擦除」，delta rule 负责「定点改写」，二者互补（§1）[^src-yang-2024-gated-delta]。

## 机制

### Delta rule：先擦除再写入

delta rule（Widrow et al., 1960）将状态更新拆为两步（§2.2）[^src-yang-2024-gated-delta]：

$$S_t=S_{t-1}-\underbrace{(S_{t-1}k_t)}_{v^{\text{old}}_t}k_t^\top+\underbrace{\big(\beta_tv_t+(1-\beta_t)S_{t-1}k_t\big)}_{v^{\text{new}}_t}k_t^\top=S_{t-1}(I-\beta_tk_tk_t^\top)+\beta_tv_tk_t^\top$$

先从状态中减去 $k_t$ 对应的旧关联 $v^{\text{old}}_t$（擦除），再按写入强度 $\beta_t\in(0,1)$ 写入新值 $v^{\text{new}}_t$（$\beta_t\to1$ 全量替换，$\beta_t\to0$ 不写）。转移矩阵 $I-\beta_tk_tk_t^\top$ 为广义 Householder 矩阵，其 identity-plus-low-rank 形式比 Mamba2 的对角 $\alpha_tI$ 更具表达力（§5）。论文脚注 2 引用指出，$\beta_t$ 放宽到 $(0,2)$ 允许负特征值，可解锁 DeltaNet 的状态跟踪能力。

### Gated delta rule：衰减乘进转移矩阵

gated delta rule 只改一处（Eq. 10）[^src-yang-2024-gated-delta]：

$$S_t=S_{t-1}\big(\alpha_t(I-\beta_tk_tk_t^\top)\big)+\beta_tv_tk_t^\top$$

标量 $\alpha_t\in(0,1)$ 与 $d_k\times d_k$ 转移矩阵可交换，两端表现为：$\alpha_t\to0$ 一步清空状态（切换上下文），$\alpha_t\to1$ 退化为纯 delta rule（定点改写、其余信息不动）。

**在线学习视角**（Table 1）：将线性 RNN 的状态更新视为在线学习目标的闭式解，LA、Mamba2、Longhorn、DeltaNet、Gated DeltaNet 是同一框架的五个点。LA 的目标 $\|S_t-S_{t-1}\|_F^2-2\langle S_tk_t,v_t\rangle$ 用正则项锁住历史，状态饱和后多段信息叠加导致检索变难；Mamba2 将正则项放宽为 $\|S_t-\alpha_tS_{t-1}\|_F^2$，Longhorn 将损失换为回归 $\|S_tk_t-v_t\|^2$（闭式解与 delta rule 同形）；gated delta rule 两处同时放宽，目标为 $\|S_t-\alpha_tS_{t-1}\|_F^2-2\langle S_tk_t,\beta_t(v_t-\alpha_tS_{t-1}k_t)\rangle$——$\alpha_t$ 允许状态受控偏离历史，即选择性遗忘（§3.1）[^src-yang-2024-gated-delta]。

**Test-time SGD 视角**：delta rule 是对回归目标 $\mathcal{L}(S_t)=\frac12\|S_tk_t-v_t\|^2$ 的一步 SGD（$\beta_t$ 即学习率），gated 版相当于在 SGD 更新里加自适应 weight decay $\alpha_t$（§3.1）[^src-yang-2024-gated-delta]。

## 数据流与维度变换

**维度符号**：$N$ 序列长度（token 数）；$d_k$、$d_v$ 单头 query/key 与 value 维（消融给出 head dim 128 的折中，Table S.1）；论文按单头书写，本页照单头口径。状态 $S_t\in\mathbb{R}^{d_v\times d_k}$；$\beta_t\in(0,1)$ 写入强度、$\alpha_t\in(0,1)$ 衰减门均为数据依赖标量；$q_t,k_t\in\mathbb{R}^{d_k}$、$v_t\in\mathbb{R}^{d_v}$ 为列向量。

### Stage 1 · 逐 token：先擦除再写入

$$\begin{aligned}
v^{\text{old}}_t &= S_{t-1}k_t && \text{// } [d_v,\bcancel{d_k}]\times[\bcancel{d_k},1]\to[d_v]\ \text{——读状态：当前 key 名下的旧关联值}\\
S_t &= S_{t-1}-v^{\text{old}}_tk_t^\top && \text{// } [d_v,d_k]-[d_v,\bcancel{1}]\times[\bcancel{1},d_k]\to[d_v,d_k]\ \text{——擦除：把 } k_t \text{ 名下的旧关联从状态里减掉}\\
S_t &\mathrel{+}= v^{\text{new}}_tk_t^\top && \text{// } [d_v,d_k]+[d_v,\bcancel{1}]\times[\bcancel{1},d_k]\to[d_v,d_k]\ \text{——写入：按强度 } \beta_t \text{ 把新值登记到同一 key 下}
\end{aligned}$$

其中 $v^{\text{new}}_t=\beta_tv_t+(1-\beta_t)v^{\text{old}}_t$，三行合并即 DeltaNet 的递推式（§2.2）。

### Stage 2 · gated 转移：衰减插在擦除之前

$$\begin{aligned}
S_t &= S_{t-1}\,\alpha_t\,(I-\beta_tk_tk_t^\top)+\beta_tv_tk_t^\top && \text{// } [d_v,d_k]\times[d_k,d_k]+[d_v,\bcancel{1}]\times[\bcancel{1},d_k]\to[d_v,d_k]\ \text{——旧状态先整体衰减 } \alpha_t \text{，再做定点擦写；} \alpha_t\to0 \text{ 清空、} \alpha_t\to1 \text{ 退化为纯 delta rule}
\end{aligned}$$

$o_t=S_tq_t\in\mathbb{R}^{d_v}$ 读出（$[d_v,\bcancel{d_k}]\times[\bcancel{d_k},1]\to[d_v]$，§2.1）。衰减位于转移矩阵内侧、擦除之前：被擦除的旧值即衰减后的 $\alpha_tS_{t-1}k_t$，与在线学习目标的回归项一致（Table 1）。

### Stage 3 · chunk 内并行（§3.3）

chunk 大小 $\mathcal{C}$，chunk $t$ 内位置 $r$，chunk 内累计衰减 $\gamma^r_{[t]}=\prod_{j=1}^r\alpha^{[t]j}$，全 chunk 衰减 $\gamma^{\mathcal C}_{[t]}$。沿用 Mamba2 的 chunkwise 约定（Eq. 2）：$\overleftarrow{x}^r=\gamma^r_{[t]}x^r$ 表示衰减到 chunk 首，$\overrightarrow{x}^r=\frac{\gamma^{\mathcal C}_{[t]}}{\gamma^r_{[t]}}x^r$ 表示衰减到 chunk 尾。

chunk 内递推经 partial expansion 拆成转移积与输入累积（§3.3）[^src-yang-2024-gated-delta]：

$$\begin{aligned}
S^r_{[t]} &= S_{[t]}\underbrace{\prod_{i=1}^r\alpha^i_{[t]}\big(I-\beta^i_{[t]}k^i_{[t]}k^{i\top}_{[t]}\big)}_{=:F^r_{[t]}}+\underbrace{\sum_{i=1}^r\beta^i_{[t]}v^i_{[t]}k^{i\top}_{[t]}\prod_{j=i+1}^r\alpha^j_{[t]}\big(I-\beta^j_{[t]}k^j_{[t]}k^{j\top}_{[t]}\big)}_{=:G^r_{[t]}} && \text{// } [d_v,d_k]\times[d_k,d_k]+[d_v,d_k]\ \text{——chunk 内状态 = 初态过门控转移积 + 输入项}
\end{aligned}$$

标量与矩阵可交换，$F^r_{[t]}=\gamma^r_{[t]}P^r_{[t]}=\overleftarrow{P^r_{[t]}}$——DeltaNet 的 WY 表示（$P^r_{[t]}=I-\sum_iw^ik^{i\top}$）得以原样保留，衰减整体移至外侧，转置后并入初态项（§3.3）。输入项经同样的 UT transform 写成矩阵形式，仅 Gram 矩阵带有衰减（§2.2/§3.3）[^src-yang-2024-gated-delta]：

$$\begin{aligned}
\widetilde{U}_{[t]} &= \big[I+\mathrm{strictLower}\big(\mathrm{diag}(\beta_{[t]})(\Gamma_{[t]}\odot K_{[t]}K^\top_{[t]})\big)\big]^{-1}\mathrm{diag}(\beta_{[t]})V_{[t]} && \text{// } [\mathcal C,\bcancel{\mathcal C}]\times[\bcancel{\mathcal C},d_v]\to[\mathcal C,d_v]\ \text{——逆与 } \mathrm{diag}(\beta) \text{ 均为 } [\mathcal C,\mathcal C]\text{，与 } V \text{ 相乘消去一个 } \mathcal C\text{；对比 DeltaNet 的 } T_{[t]}\text{：仅 } KK^\top \text{ 换成衰减感知的 } \Gamma\odot KK^\top
\end{aligned}$$

其中 $\Gamma_{[t]}$ 是 Eq. 1 的衰减感知因果掩码（$\Gamma_{ij}=\gamma^i/\gamma^j$，$i\ge j$）。chunk 输出与跨 chunk 状态（§3.3）[^src-yang-2024-gated-delta]：

$$\begin{aligned}
S_{[t+1]} &= \overrightarrow{S}_{[t]}+(\widetilde{U}_{[t]}-\overleftarrow{W}_{[t]}S^\top_{[t]})^\top\overrightarrow{K}_{[t]} && \text{// } [d_v,d_k]+[d_v,\bcancel{\mathcal C}]\times[\bcancel{\mathcal C},d_k]\to[d_v,d_k]\ \text{——传给下个 chunk：整体衰减后的初态 + 矩阵式擦写修正}\\
O_{[t]} &= \overleftarrow{Q}_{[t]}S^\top_{[t]}+(Q_{[t]}K^\top_{[t]}\odot M)(\widetilde{U}_{[t]}-\overleftarrow{W}_{[t]}S^\top_{[t]}) && \text{// } [\mathcal C,\bcancel{d_k}]\times[\bcancel{d_k},d_v]+[\mathcal C,\bcancel{\mathcal C}]\times[\bcancel{\mathcal C},d_v]\to[\mathcal C,d_v]\ \text{——chunk 输出 = 跨 chunk 读取 + chunk 内因果修正}
\end{aligned}$$

其中 $\overleftarrow{q}^r=\gamma^r_{[t]}q^r$、$\overleftarrow{w}^r=\gamma^r_{[t]}w^r$、$\overrightarrow{k}^r=\frac{\gamma^{\mathcal C}_{[t]}}{\gamma^r_{[t]}}k^r$、$\overrightarrow{S}_{[t]}=\gamma^{\mathcal C}_{[t]}S_{[t]}$。跨 chunk 仅传递 $S_{[t]}\in\mathbb{R}^{d_v\times d_k}$；chunk 内由 $[\mathcal C\times\mathcal C]$ 的 masked matmul 与一次 strictLower 三角系统求解组成，均落在张量核友好的 matmul 上，与 Mamba2 将线性注意力扩展为分块 SSD（Eq. 1）路线一致（§3.3）。论文仅给出正演；gated 版与原版 DeltaNet 吞吐基本持平（Fig 3）[^src-yang-2024-gated-delta]。

### Stage 4 · 块设计（Fig 1 右）

q/k 路线：线性投影 → shortconv → SiLU → L2 归一化（训练稳定性，消融显示 L2 归一化不可省）；v 路线：线性投影 → shortconv → SiLU；α/β 仅线性投影（α 沿用 Mamba2 参数化，细节未给出，脚注 4）；输出经归一化 + 输出门（线性投影 + SiLU）+ 输出投影。token mixer 层与 SwiGLU MLP 按 Llama 宏架构堆叠（§3.4）[^src-yang-2024-gated-delta]。

## 证据（作者报告口径）

全部主对比均在 1.3B 参数、100B token（FineWeb-Edu）、4K 训练长度与同一超参下完成（§4）[^src-yang-2024-gated-delta]。

**合成召回 S-NIAH（RULER，zero-shot，1.3B，Table 2）**——三个观察各对应一段任务：

| 任务 | 长度 | DeltaNet | Mamba2 | Gated DeltaNet |
|---|---|---|---|---|
| S-NIAH-1（passkey，测长期保留） | 1K | 97.4 | 99.2 | 98.4 |
| | 2K | 96.8 | 98.8 | 88.4 |
| | 4K | 99.0 | 65.4 | 91.4 |
| | 8K | 98.8 | 30.4 | 91.8 |
| S-NIAH-2（数字 needle，测过滤） | 1K | 98.4 | 99.4 | 100.0 |
| | 2K | 45.6 | 98.8 | 99.8 |
| | 4K | 18.6 | 56.2 | 92.2 |
| | 8K | 14.4 | 17.0 | 29.6 |
| S-NIAH-3（UUID needle，测复杂记忆） | 1K | 85.2 | 64.4 | 86.6 |
| | 2K | 47.0 | 47.6 | 84.2 |
| | 4K | 22.4 | 4.6 | 27.6 |

读法：衰减伤保留（S-NIAH-1 中 Mamba2 超 2K 即崩，delta rule 接近满分）；门控助过滤（S-NIAH-2/3 中 DeltaNet 长上下文崩于记忆碰撞）；delta rule 助记忆（S-NIAH-3 中 Mamba2 快速掉到 4.6，Gated DeltaNet 27.6）（§3.2）[^src-yang-2024-gated-delta]。

**语言建模与常识推理（1.3B，Table 3）**：Wiki ppl 16.42 / LMB ppl 12.17 / 八项平均 55.32，对 Mamba2 的 16.56 / 12.56 / 54.89、DeltaNet 的 52.14、Transformer++ 的 52.25；混合 Gated DeltaNet-H1（+SWA）/H2（Mamba2+GDN+SWA 交错）达 56.40 / 56.18，高于同表 Samba 的 54.00。

**真实世界检索（输入截断 2K，Table 4）**：SWDE/SQuAD/FDA/TriviaQA/NQ/DROP 平均，纯 RNN 中 Gated DeltaNet 30.6 > Mamba2 29.8 > DeltaNet 26.2；混合 H1/H2 39.0/40.1 > Transformer++ 37.0、Samba 37.3。

**长度外推（Fig 2）**：GovReport、QMSum、NarrativeQA、Qasper、CodeParrot、PG19 六基准从 4K 外推到 20K，Gated DeltaNet 在 RNN 中整体困惑度最低；各基准外推表现参差，混合模型靠 SWA 分担局部建模更稳（§4）。

**LongBench（14 任务，Table 5）**：Gated DeltaNet 平均 16.6 为纯 RNN 最高（Mamba2 13.5、DeltaNet 13.6），H1/H2 17.8/18.4 高于 Transformer++ 的 11.0 与 Samba 的 15.9。

**吞吐（Fig 3，1.3B 单 H100）**：Gated DeltaNet 与 DeltaNet 基本持平，比 Mamba2 慢 2–3K tokens/sec（因转移矩阵表达力更强）；SWA 混合在所有长度上吞吐均高于纯 mixer，H1 在短序列也保持高吞吐（§4）。

**消融（Table S.1，400M/15B）**：将 gated delta rule 换回 naive delta rule，Avg-PPL 27.35→30.87、Avg-Acc 47.26→45.12，体现门控贡献；去掉 shortconv 与输出门后 PPL 分别降至 28.95 与 29.12；L2 归一化必要，特征映射选择影响小，head dim 128 为折中。**混合层序（Table S.2，500M/15B）**：Mamba2 → Gated DeltaNet → SWA 顺序平均 48.73 最优（§B.2）[^src-yang-2024-gated-delta]。

## 边界

序列长度外推更稳仅相对同表 RNN，未测试超 20K 序列（§4）。Table 4 真实检索优势（30.6 对 29.8）远小于合成任务差距，论文归因于未对齐指令小模型的重复错误与更新规则无关（§4）。正文称 Table 3 覆盖「400M 与 1.3B」，但表内仅列 1.3B 行，未载 400M 主对比数值（见 [[ingest-reports/2026-10-27-gated-delta|WHY 报告]]数值备注）[^src-yang-2024-gated-delta]。α 的参数化沿用 Mamba2，因求简省略实现细节（脚注 4）。并发的 RWKV-7 采用 diagonal-plus-low-rank 转移 $S_{t-1}(\mathrm{diag}(d_t)-a_tb_t^\top)+v_tk_t^\top$ 覆盖更一般情形，其 chunkwise 算法可类似适配（§5）[^src-yang-2024-gated-delta]。全部数字出自本论文（confidence medium）。

## 在本 wiki 中的位置

[[mamba]] 的 Mamba2 更新 $S_t=\alpha_tS_{t-1}+v_tk_t^\top$ 是本文去掉擦除项后的退化形式，Table 1 的在线学习对照给出两者的目标差异；[[selective-ssm]] 的选择设计是「数据依赖遗忘」的谱系源头，论文 §5 指出现代 forget gate 与 LSTM 的区别在于去除对前一隐状态的依赖，以维持跨长度并行。[[linear-attention-rnn-form]] 的循环形式 $s_i=s_{i-1}+\phi(k_i)v_i^\top$ 是本文递推去掉擦除与写入强度的 Hebbian 特例，矩阵状态 $S$ 对应其 $s_i$。[[fast-weight-mlp-view-of-attention]] 的 fast-weight 视角与论文 §3.1 一致：$S$ 是按 test-time SGD 在线优化的权重矩阵。[[trace-as-state-reproduction]] 记录的 Qwen 3.7 Max 架构归因「GDN + GA」中，GDN 即此模型，本页为其一手出处。

[^src-yang-2024-gated-delta]: [[source-yang-2024-gated-delta]]
