---
title: "Switch Transformer"
type: technique
tags:
  - mixture-of-experts
  - sparse-activation
  - routing
  - scaling
created: 2026-10-27
last_updated: 2026-10-27
source_count: 1
confidence: medium
status: active
---

# Switch Transformer

Switch Transformer（Fedus, Zoph & Shazeer, Google, JMLR 2022）把 MoE 层的路由简化为 top-1，并用单一乘性辅助损失做负载均衡，在保持每 token FLOPs 不变的前提下把 Transformer 参数量推到万亿级[^src-switch-transformer-2101-03961]。

## 问题与谱系位置

MoE 路线到 2021 年为止已积累两层：[[sparse-gated-moe-layer]]（Shazeer et al., 2017）在 LSTM 层间插入 MoE，路由到 $k \geq 2$ 个专家并叠加 load-balancing 与 importance-weighting 两个辅助损失；[[gshard]]（Lepikhin et al., 2020）把 MoE 层移入 Transformer 的 FFN 位置、以 top-2 路由把机器翻译推到 100 语言规模[^src-switch-transformer-2101-03961]。MoE 的广泛采用受实现复杂度、通信成本与训练不稳定三点阻碍，Switch Transformer 针对这三点做减法：路由只选 1 个专家、辅助损失合并为一条、路由计算局部用 float32 保稳定[^src-switch-transformer-2101-03961]。

设计假设沿 Kaplan et al. (2020) 的缩放定律再开第四条轴：参数量独立于每 token 计算量，是可缩放的维度——MoE 只增加参数与显存，不增加每 token FLOPs[^src-switch-transformer-2101-03961]。

## 数据流与维度变换

**维度符号**：$T$ 为 batch 内 token 数；$d$ 为模型通道维 $d_{model}$；$E$ 为 expert 数；$c_f$ 为 capacity factor；$C = (T/E) \cdot c_f$ 为 expert 容量。专家为 FFN，参数量 $E \times 2 d d_{ff}$。

### Stage 1 · 路由打分（局部 float32）

$$\begin{aligned}
h &= W_r x && \text{// } [T, \bcancel{d}]\times[\bcancel{d}, E]\to[T, E]\ \text{——消去 } d\text{，每个 token 对全部 } E \text{ 个 expert 的 logits}\\
p &= \operatorname{softmax}(h) && \text{// } [T, E] \xrightarrow{\text{沿 } E \text{ 维归一}} [T, E]\ \text{——每 token 一个 expert 分布，行和为 } 1
\end{aligned}$$

训练时在 logits 上引入乘性均匀抖动 $[1-\varepsilon, 1+\varepsilon]$ 做探索（附录 F 伪代码在 softmax 前对 logits 加该噪声，附录 C 正文则记在输入表示上，两处描述不一）[^src-switch-transformer-2101-03961]。附录 C 表明 input jitter 实证最好并全文沿用：Table 11 中 jitter 的 -1.468 为表中最高值，与表头 Quality (↑) 及论文全局约定（Neg. Log Perp. 越高越好）一致；但 Table 11 图注标注 "(lower is better)"，按该口径最低的 sample softmax（-1.570）才是最优——图注与表头、正文矛盾，此处并存记录[^src-switch-transformer-2101-03961]。

### Stage 2 · top-1 选择

$$\begin{aligned}
(g, e) &= \operatorname{Top}_1(p) && \text{// } [T, E] \xrightarrow{\text{沿 } E \text{ 维 argmax}} [T]\ \text{——每 token 一条 expert 下标 } e \text{ 与门控概率 } g\\
m &= \operatorname{onehot}(e) && \text{// } [T] \to [T, E]\ \text{——路由指示矩阵，派发与负载统计共用}
\end{aligned}$$

相比 top-$k$，top-1 带来三重收益（论文 §2.1）：路由计算量降为单专家、expert 容量至少减半（每 token 只占一个槽位）、派发-合并通信实现简化[^src-switch-transformer-2101-03961]。门控概率乘回输出使 router 端到端可微（论文式 (2) 的性质在 $k=1$ 时得以保留）[^src-switch-transformer-2101-03961]。

### Stage 3 · 容量裁剪与派发

$$\begin{aligned}
C &= \frac{T}{E}\cdot c_f && \text{// 标量——expert 容量：均分 batch 再乘 buffer（式 (3)）}\\
\text{pos} &= \operatorname{cumsum}(m)\cdot m && \text{// } [T, E] \to [T, E]\ \text{——同 expert 内按到达顺序排槽位}\\
D &= m \cdot \mathbb{1}\{\text{pos} < C\} && \text{// } [T, E] \to [T, E]\ \text{——槽位超容的 token 被 mask，gate 清零}\\
X_e &= X^{\top} D && \text{// } [\bcancel{T}, d]\times[T, E, C]\to[E, C, d]\ \text{——einsum gather 按 expert 聚 token（式 (7)）}
\end{aligned}$$

溢出 token 不经该层计算，直接走残差连接到下一层[^src-switch-transformer-2101-03961]。实验中 token 丢弃率通常 < 1%，且不依赖 expert 数，靠辅助损失系数即可压住（§2.2）[^src-switch-transformer-2101-03961]。附录 B 的 No-Token-Left-Behind（溢出重路由到次优 expert）未带来实证收益，作者推测打乱网络学定的 token-expert 关联反而有害[^src-switch-transformer-2101-03961]。

### Stage 4 · 专家计算与 all-to-all

$$\begin{aligned}
Y_e &= \operatorname{FFN}_e(X_e) && \text{// } [E, C, \bcancel{d}]\times[\bcancel{d}, d_{ff}]\times[\bcancel{d_{ff}}, d]\to[E, C, d]\ \text{——每 expert 一套独立参数，各自常规 FFN}\\
Y_e &\xrightarrow{\text{all-to-all}} \text{重切 core 布局} && \text{// } [E, C, d]\ \text{——通信张量即 } [E, C, d]\text{，bfloat16；前向派发与合并各一次，反向镜像}
\end{aligned}$$

仅 expert 并行时每个 core 分配一个 expert（Switch-C 配置）；叠加模型并行时每 core 再切分 $d_{ff}/m$，同时承担 all-to-all 与 all-reduce 通信（§5.4–5.5）[^src-switch-transformer-2101-03961]。

### Stage 5 · 合并

$$\begin{aligned}
y_t &= g_t \cdot \operatorname{FFN}_{e_t}(x_t) && \text{// } [T, d]\ \text{——top-1 下无需跨专家求和，乘回门控即完成 combine}
\end{aligned}$$

## 乘性辅助损失

$$\begin{aligned}
f_e &= \frac{1}{T}\sum_{x\in B}\mathbb{1}\{\operatorname{argmax} p(x) = e\} && \text{// dispatch 比例，不可微}\\
P_e &= \frac{1}{T}\sum_{x\in B} p_e(x) && \text{// 概率质量比例，可微，梯度经它回传}\\
\mathcal{L}_{aux} &= \alpha \cdot E \cdot \sum_{e=1}^{E} f_e \cdot P_e && \text{——均匀路由时 } f_e = P_e = \tfrac{1}{E}\text{，和为 } \tfrac{1}{E}\text{，乘 } E \text{ 归一到 } \alpha
\end{aligned}$$

乘以 $E$ 使损失在 expert 数量变化时保持量纲不变（论文式 (4) 说明）；$\sum_e f_e = 1$ 与 $P$ 均为概率分布，在均匀分布处取得最小值。该式是 Shazeer et al. (2017) 双损失（load-balancing + importance-weighting 加性组合）的简化版，此前已在 Shazeer et al. (2018) 与 GShard 中出现，Switch 直接沿用[^src-switch-transformer-2101-03961]。系数 $\alpha$ 在 $10^{-1}$ 到 $10^{-5}$ 扫描后取 $10^{-2}$：足以快速均衡负载，且不压过主交叉熵目标（§2.2）[^src-switch-transformer-2101-03961]。

## 训练与微调技术

**选择性精度（selective precision）**。bfloat16 下 router softmax 数值不稳定，GShard 因此全程用 float32 训练。Switch 把 router 函数体内的输入与计算 cast 成 float32，产出的 dispatch/combine 张量在出口 cast 回 bfloat16，高精度不进入 all-to-all 通信。Table 2：float32 全程 -1.718（1160 ex/s）、纯 bfloat16 -3.780 发散（1390 ex/s）、选择性精度 -1.716（1390 ex/s），速度持平 bfloat16、稳定性持平 float32[^src-switch-transformer-2101-03961]。

**缩小初始化尺度**。截断正态 $\sigma = s/n$ 的 $s$ 从默认 1.0 降到 0.1。Table 3（32 expert，3.5k 步，3 seeds）：0.1x 初始化 -2.72 ± 0.01，1.0x -3.60 ± 0.68，早期质量与方差同时改善；同一方案从 223M 用到 1.6T 参数模型[^src-switch-transformer-2101-03961]。

**expert dropout**。参数量大而微调任务规模小易导致过拟合；微调时仅把 expert 内部 FFN 的 dropout 从 0.1 提高到 0.4，非 expert 层维持 0.1。Table 4：GLUE 85.2 / CNN/DM 19.6 / SQuAD 83.7 / SuperGLUE 73.0——对比均匀 0.1 基线（84.7 / 19.1 / 83.7 / 73.0），GLUE 与 CNN/DM 提升，SQuAD 与 SuperGLUE 持平；SuperGLUE 低于均匀 0.2 的 73.2。论文称四个下游任务均有提升，但表列结果为两升两平[^src-switch-transformer-2101-03961]。

**蒸馏**。叠加两项技巧：student 用 teacher 的非 expert 权重初始化，损失按 0.25 teacher 软目标与 0.75 硬标签混合。Table 6 中，3800M teacher（-1.444）蒸馏至 223M student 得 -1.580，保留 sparse teacher 相对 T5-Base 质量增益的 29%；仅初始化不蒸馏无增益（-1.639）[^src-switch-transformer-2101-03961]。Table 7 验证极端压缩：1.1B→14.7B teacher 蒸馏到同一 223M 架构，压缩 82%–99%，保留增益 37%–28%；微调后的 SuperGLUE 模型同样保留 30%（Table 8：81.3 → 76.6，teacher 7.4B/97% 压缩）[^src-switch-transformer-2101-03961]。

## 证据

### 预训练：速度-质量

Table 1（128 expert、隔层放置、32 核 TPUv3、FLOP-matched，阈值 Neg. Log Perp. -1.50）：T5-Base 100k 步为 -1.731 且未达阈值；Switch-Base 在容量因子 1.0 时 62.8 小时达标（1000 ex/s），同条件 MoE-Base 需 80.1 小时（860 ex/s）；容量因子 1.0 与 1.25 下 Switch 均快于 MoE 对应项。Switch-Base+（hidden 768→896、heads 14→16）增至与 MoE 等速后达 -1.534，仍为最优[^src-switch-transformer-2101-03961]。

Figure 4–6：expert 数 2→256（参数 223M→14.7B）测试损失一致下降；Switch-Base 64e 用 60k 步（T5-Base 450k 步的 1/7.5）达到 T5-Base 需 450k 步的质量（step 口径 7.5 倍加速）、墙钟口径 1/7；对 3.5 倍 FLOPs 的 T5-Large 仍快 2.5 倍且更样本高效[^src-switch-transformer-2101-03961]。

Table 9（250k/500k 步 C4 Neg. Log Perp.）：Switch-XXL（395B，64e，6.3T FLOPs/seq）为 -1.086/-1.008，T5-XXL 为 -1.147/-1.095——250k 步领先超 0.061（T5-XXL 多训 250k 步才涨 0.052），500k 步领先 0.087；Switch-C（1571B 参数、2048 expert、890B FLOPs/seq、15 层均设专家）为 -1.096/-1.043，同算力达固定困惑度比 T5-XXL 快 4 倍[^src-switch-transformer-2101-03961]。

### 微调与多语言

Table 5（FLOP-matched，验证集）：Switch-Base 对 T5-Base SuperGLUE 79.5 vs 75.1、Winogrande 73.3 vs 66.6、CB TriviaQA 30.7 vs 24.5；Switch-Large 对 T5-Large SuperGLUE 84.7 vs 82.7。唯一下滑：ARC Challenge（Switch-Base 32.8 vs 35.5）与 ARC Easy 的 Large 档（66.0 vs 68.8）[^src-switch-transformer-2101-03961]。

Figure 7–8（mC4 101 语言、107 任务、1M 步）：在全部 101 语言中，mSwitch-Base 的 Neg. Log Perp. 均高于 mT5-Base；相对 mT5-Base 平均 step 加速 5 倍，91% 语言 ≥ 4 倍[^src-switch-transformer-2101-03961]。

### 万亿参数档的下游

Switch-XXL 用 503B token（约 T5-XXL 一半数据）多任务微调：SQuAD 89.7（SOTA 91.3）、SuperGLUE 87.5（T5 89.3、SOTA 90.0）、ANLI 65.7（此前最好 49.4）；闭卷知识三项均高于 T5-XXL：Natural Questions 34.4 vs 32.8、Web Questions 41.0 vs 37.2、TriviaQA 47.5 vs 42.9（§5.6）[^src-switch-transformer-2101-03961]。

## 边界（论文自述）

Switch-XXL 间歇不稳定且未训满 1M 步，Switch-C 与 Base/Large 则未观察到不稳定（§5.6、§8）[^src-switch-transformer-2101-03961]。上游困惑度到推理型下游的转化在最大档失效：Switch-C SQuAD 87.7 低于 Switch-XXL 的 89.6（每 token FLOPs 约差 10 倍、参数约差 4 倍），这层依赖 poorly understood（§8、附录 E 图 13）[^src-switch-transformer-2101-03961]。把专家放进 Self-Attention 的 Q/K/V（附录 A）有质量增益，但在 bfloat16 下发散，留作未来工作[^src-switch-transformer-2101-03961]。蒸馏无法完整保留质量，10–100 倍压缩下约保留 30% 增益（§7）[^src-switch-transformer-2101-03961]。注意力稀疏与本方案正交，未在此版本实现（§6）[^src-switch-transformer-2101-03961]。

## 与谱系前作的机制对照

| 维度 | [[sparse-gated-moe-layer]] | [[gshard]] | Switch Transformer |
|---|---|---|---|
| 路由 | top-$k$，$k \geq 2$，加权求和 | top-2（FLOP-matched MoE 对照即此配置） | top-1 |
| 辅助损失 | load-balancing + importance 加性双损失 | 简化为单一点积项 | 同一点积项，乘 $\alpha \cdot E$，$\alpha = 10^{-2}$ |
| 容量因子 | — | — | 论文未陈述预训练默认值；Table 1 消融 2.0/1.25/1.0，结论为低容量因子（1.0、1.25）下表现更优、相对 MoE 优势扩大 |
| 数值精度 | — | 全程 float32 | bfloat16 + router 函数体内 float32 |
| 挂载位置 | LSTM 层间 | Transformer FFN | Transformer FFN（Base/Large/XXL 隔层，Switch-C 每层） |

前两列损失与精度口径来自 Switch 论文 §2.2 的转述（其将 2017 双损失到 GShard 单损失的演化作为自身起点），GShard 侧细节以其原文为准。

[^src-switch-transformer-2101-03961]: [[fedus-2022-switch-transformer]]
