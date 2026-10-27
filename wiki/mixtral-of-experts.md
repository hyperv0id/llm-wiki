---
title: "Mixtral of Experts"
type: technique
tags:
  - mixture-of-experts
  - sparse-activation
  - conditional-computation
  - large-language-models
created: 2026-10-27
last_updated: 2026-10-27
source_count: 1
confidence: medium
status: active
---

# Mixtral of Experts

Mixtral 8x7B 是 Mistral AI 提出的稀疏混合专家（SMoE）语言模型。其架构与 Mistral 7B 基本一致，区别在于每个 Transformer 层的前馈块改为 8 专家 MoE 层，由 router 为每个 token 选 2 个专家处理并加性合并输出；每 token 可访问 47B 参数、推理仅用 13B 激活参数，以 32k 上下文预训练，权重以 Apache 2.0 发布[^src-mistral-2024-mixtral]。相比 Mistral 7B，论文指出的两处改动是支持全 dense 的 32k 上下文与 FFN 换 MoE 层，其余沿用其配置（Table 1：32 层、$d{=}4096$、32 头、8 个 KV 头、head_dim 128、词表 32000）[^src-mistral-2024-mixtral]。

## 问题：参数量与每 token 计算解耦

dense 模型的容量与计算量同步增长，MoE 则将两者解耦：总参数量（sparse parameter count）随专家数 $n$ 增长，每 token 实际使用的激活参数（active parameter count）随 $K$ 增长且至多到 $n$；固定 $K$ 增大 $n$ 时，参数增加而计算成本近似不变（§2.1）[^src-mistral-2024-mixtral]。模型在低 batch 时推理更快、大 batch 时吞吐更高（§1）。但该分析只覆盖与推理算力成正比的激活参数，不含内存与硬件利用率：服务内存按 47B 总参数计（仍低于 Llama 2 70B 的 70B），SMoE 层还有路由开销与每设备驻留多专家的内存负载，更适合能达到足够算术强度的 batched 负载（§3）[^src-mistral-2024-mixtral]。

## 数据流与维度变换

> **维度符号**：$x\in\mathbb{R}^d$ 为单个 token 的隐状态（$d{=}4096$）；$n{=}8$ 为每层专家数，$K{=}2$ 为每 token 激活专家数（Table 1 的 num_experts/top_k_experts）；$h{=}14336$ 为专家前馈块隐层宽（Table 1 hidden_dim）；$W_g\in\mathbb{R}^{d\times n}$ 为 router 权重。

### Stage 1 · router 线性打分

$$\begin{aligned}
s &= x\,W_g && \text{// } [d]\times[\bcancel{d},n]\to[n]\ \text{——消去 } d\text{，每个 token 对 } n \text{ 个专家各得一个路由 logit}
\end{aligned}$$

### Stage 2 · Top-2 稀疏化

$$\begin{aligned}
\operatorname{TopK}(s)_i &= \begin{cases}s_i & s_i\ \text{属于 } s \text{ 的 top-}K\\ -\infty & \text{否则}\end{cases} && \text{// } [n]\xrightarrow{\text{沿 } n \text{ 维保留 top-}K}[n]\ \text{——维度不变，未选中分量的 logit 被 } -\infty\text{ 屏蔽}
\end{aligned}$$

### Stage 3 · softmax 归一成稀疏门控

$$\begin{aligned}
G(x) &:= \operatorname{Softmax}\!\big(\operatorname{TopK}(x\cdot W_g)\big) && \text{// } [n]\xrightarrow{\text{沿 } n \text{ 维归一}}[n]\ \text{——}-\infty\text{ 位 softmax 后恰为 0：门控至多 } K{=}2 \text{ 个非零分量，和为 1}
\end{aligned}$$

论文指出 $G(x)$ 有多种实现，选用在「线性层 Top-K logits 上做 softmax」这一简单高效的方案，并将该门控出处标注为 Shazeer 2017（参考文献 [28]）（§2.1）[^src-mistral-2024-mixtral]。

### Stage 4 · 专家前向与加权汇合

$$\begin{aligned}
E_i(x) &= \operatorname{SwiGLU}_i(x) && \text{// } [d]\xrightarrow{\ d\to h\to d\ }[d]\ \text{——每专家一套独立权重的 SwiGLU 前馈块（}h{=}14336\text{）；SwiGLU 具体式本文未给出，沿其 Mistral 7B 基线}\\
y &= \sum_{i=0}^{n-1} G(x)_i\,E_i(x) && \text{// } K \text{ 个 } [d]\xrightarrow{\text{按门控加权求和}}[d]\ \text{——条件计算语义：} G(x)_i{=}0 \text{ 的专家无需计算，每 token 实际只跑 2 个 SwiGLU 块}
\end{aligned}$$

合并写为论文原式即 $y=\sum_{i=0}^{n-1}\operatorname{Softmax}(\operatorname{Top2}(x\cdot W_g))_i\cdot\operatorname{SwiGLU}_i(x)$（§2.1）[^src-mistral-2024-mixtral]。按 Table 1 复算参数 [INFERENCE]：每专家约 $3\times 4096\times 14336\approx 1.76$ 亿权重，$8$ 专家 $\times$ $32$ 层约 45B，加注意力与嵌入对应 47B 总参；每 token 只取 2 专家约 11B 加注意力，对应 13B 激活。

单卡采用高性能专用核，如 Megablocks 把 MoE 的 FFN 运算改写为大型稀疏矩阵乘，天然容忍各专家分到不同 token 数；跨卡采用标准模型并行与 Expert Parallelism（EP），执行期把 token 按目标专家路由到对应 GPU，算完返回原位，EP 负载须均匀分摊以免个别 GPU 过载（§2.1）[^src-mistral-2024-mixtral]。

## 与前代稀疏 MoE 的机制对照

与 [[sparse-gated-moe-layer]] 相比，两者门控相同，均为线性打分后 KeepTopK（非 top-$k$ 置 $-\infty$）再 softmax，论文注明该门控出自 [28]；差别在于 Shazeer 2017 的打分含可学习噪声项（noisy top-k）并配 importance/load 双均衡辅助损失，而 Mixtral 的 router 无噪声项，未引入辅助负载均衡损失[^src-mistral-2024-mixtral]。与 [[gshard]] 相比，论文指出两者的 formulation 相似但有两点不同：Mixtral 将全部 FFN 子块换成 MoE 层，而 GShard 每隔一个子块替换；且 GShard 对每 token 的第二专家采用更精细的门控，按 [[gshard]] 设计在其余专家中随机路由并加辅助均衡损失（§2.1）[^src-mistral-2024-mixtral]。在没有辅助损失时，负载均衡由 router 学出，§5 显示 The Pile 八个子集在第 0/15/31 层的专家选择比例都贴近均匀线 $1/8$（Figure 7）[^src-mistral-2024-mixtral]。

## 证据

### 与 Llama 系列对比（Table 2）

评测口径（§3 开头）：常识推理 0-shot（HellaSwag、WinoGrande、PIQA、ARC-e/c 等），世界知识 5-shot（NQ、TriviaQA），GSM8K 8-shot maj@8，MATH 4-shot maj@4，HumanEval 0-shot，MBPP 3-shot，MMLU 5-shot；全部基准与 Llama 用同一评测管线重跑[^src-mistral-2024-mixtral]。

Mixtral 8x7B（激活 13B）对 Llama 2 70B：MMLU 70.6% 对 69.9%，ARC-e 83.1% 对 79.9%，ARC-c 59.7% 对 56.5%，PIQA 83.6% 对 82.6%，NQ 30.6% 对 25.4%，HumanEval 40.2% 对 29.3%，MBPP 60.7% 对 49.8%，MATH 28.4% 对 13.8%，GSM8K 74.4% 对 69.6%；HellaSwag 84.4% 对 85.4%、WinoGrande 77.2% 对 80.4%、TriviaQA 71.5% 对 73.0% 三项低于 Llama 2 70B。以约 5 倍少的激活参数在多数基准持平或超出，数学与代码差距最大（Table 2、Figure 2）[^src-mistral-2024-mixtral]。

### 与 GPT-3.5 对照（Table 3，另一组 shot 口径）

| 基准 | Llama 2 70B | GPT-3.5 | Mixtral 8x7B |
|---|---|---|---|
| MMLU（5-shot） | 69.9% | 70.0% | 70.6% |
| HellaSwag（10-shot） | 87.1% | 85.5% | 86.7% |
| ARC Challenge（25-shot） | 85.1% | 85.2% | 85.8% |
| WinoGrande（5-shot） | 83.2% | 81.6% | 81.2% |
| MBPP（pass@1） | 49.8% | 52.2% | 60.7% |
| GSM8K（5-shot） | 53.6% | 57.1% | 58.4% |
| MT-Bench（Instruct 模型） | 6.86 | 8.32 | 8.30 |

Table 2 与 Table 3 同名基准数字存在差异：HellaSwag 84.4（0-shot）对 86.7（10-shot），GSM8K 74.4（8-shot maj@8）对 58.4（5-shot）；论文未解释差异，按两表标注属口径不同 [INFERENCE]。评测协议较 Llama 2 论文有两处不同：MBPP 采用人工核验子集，TriviaQA 不提供 Wikipedia 上下文（§3 Evaluation Differences）[^src-mistral-2024-mixtral]。

### 多语言（Table 4）

预训练显著提高多语言数据上采样比例。ARC-c/HellaSwag/MMLU 在法语（58.2/77.4/70.9 对 49.9/72.5/64.3）、德语（54.3/73.0/71.5 对 47.3/68.7/64.2）、西班牙语（55.4/77.6/72.5 对 50.5/74.5/66.0）、意大利语（52.8/75.1/70.9 对 49.4/70.9/65.1）均高于 Llama 2 70B，同时保持英文成绩（Table 4）[^src-mistral-2024-mixtral]。

### 长上下文（Figure 4）

在 passkey 检索任务中，无论序列长度与 passkey 位置，检索准确率均为 100%；proof-pile 子集困惑度随上下文长度增加单调下降（§3.2）[^src-mistral-2024-mixtral]。

### 偏差基准（Figure 5）

BBQ 准确率 56.0% 对 Llama 2 70B 的 51.5%；BOLD 情感分（avg±std）：gender 0.323±0.045 对 0.293±0.073，profession 0.243±0.087 对 0.218±0.073，political_ideology 0.186±0.146 对 0.149±0.140，race 0.232±0.052 对 0.232±0.049，religious_ideology 0.144±0.089 对 0.188±0.133。Mixtral 偏见更少（BBQ 更高）、组内方差相近，除最后一组均值更低外整体情感更积极（§3.3）[^src-mistral-2024-mixtral]。

### 指令微调（§4）

Mixtral – Instruct 用指令数据 SFT 结合偏好对数据 DPO 训练；MT-Bench 得分 8.30（Table 3），截至 2023 年 12 月为最佳开源权重模型；LMSys 人类评测 Arena Elo 达 1121，高于 Claude-2.1（1117）、GPT-3.5-Turbo 最佳版（1117）、Gemini Pro（1111）、Llama-2-70b-chat（1077）（Figure 6，2023-12-22 截图）[^src-mistral-2024-mixtral]。

### 路由分析（§5）

主题维度无特化：The Pile 八个子集（ArXiv、GitHub、PhilPapers、StackExchange、DM Mathematics、Gutenberg、PubMed Abstracts、Wikipedia (en)）在第 0/15/31 层的专家分配分布几乎相同（Figure 7）；仅 DM Mathematics 略有差异，源于其合成性质与自然语言覆盖有限，该差异在隐状态最贴近输入/输出嵌入的首末层更明显[^src-mistral-2024-mixtral]。句法维度则有明确规律：Python 的 'self' 与英文的 'Question' 等跨多 token 词，各 token 常被路由到同一专家，代码缩进 token 总是分给同一批专家，首末层尤甚（Figure 8）[^src-mistral-2024-mixtral]。时间局部性（Table 5）显示，连续两 token 分到同一专家的比例，第一选择口径下第 0 层各域为 13.6%–14.9%（随机基线 12.5%）、第 15 层 23.6%–28.4%、第 31 层 19.7%–26.3%；第一或第二次选择口径下随机基线约 46%，第 15 层达 61.6%–67.0%。高局部性在 Expert Parallelism 下会造成部分专家过订阅，但也利于缓存（引 Eliseev & Mazur [11]）[^src-mistral-2024-mixtral]。附录 Figure 9 把 Figure 7 拆为第一、第二与任一选择三种口径，Figure 10 给出全部 32 层的连续重复率，形态一致且 DM Mathematics 重复最少[^src-mistral-2024-mixtral]。

## 边界

效率分析只计激活参数，内存与硬件利用率不在其列；路由开销与多专家驻留使其更适合 batched 负载（§3）。预训练数据规模、token 数与硬件配置未公开，训练细节仅交代 32k 上下文与多语言上采样（§1、§3.1）。路由分析只覆盖 The Pile 验证集子集（§5）。论文提出「首个在开源模型中达到 SOTA 的 MoE 网络」，其实测 Instruct 表现优于闭源模型（§1、§4、§6）。

## 谱系

向前：TopK→softmax 稀疏门控始于 [[sparse-gated-moe-layer|稀疏门控 MoE 层]]（论文引 [28]）；top-2 加第二专家精细门控的工程前驱为 [[gshard]]（论文引 [21]）。MoE 谱系及其在时序、时空领域的后续形态见 [[mixture-of-experts]] 与 [[moe-in-time-series-and-spatio-temporal]]。

[^src-mistral-2024-mixtral]: [[source-mistral-2024-mixtral]]
