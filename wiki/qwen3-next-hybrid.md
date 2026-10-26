---
title: "Qwen3-Next Hybrid Architecture"
type: technique
tags:
  - llm
  - architecture
  - hybrid-attention
  - linear-attention
  - mixture-of-experts
  - multi-token-prediction
created: 2026-10-27
last_updated: 2026-10-27
source_count: 2
confidence: high
status: active
---

# Qwen3-Next Hybrid Architecture

Qwen3-Next 是阿里 Qwen 团队 2025-09-11 发布的下一代基础模型架构，首个实现为 80B-A3B（总参 80B、每 token 激活约 3B，占 3.7%）[^src-qwen3-next-blog-2025-09]。发布形态为官方博客与 HF 模型卡，未包含消融表和实验章节。

博客将动机归于两个 scaling 趋势：Context Length Scaling 与 Total Parameter Scaling 下的训练/推理性价比[^src-qwen3-next-blog-2025-09]。

## 架构总览

模型包含 48 层，hidden dim 2048，非嵌入参数 79B（模型卡）[^src-qwen3-next-modelcard-2025-09]。层间交替为 `12 * (3 * (Gated DeltaNet -> MoE) -> 1 * (Gated Attention -> MoE))`（模型卡原文）[^src-qwen3-next-modelcard-2025-09]，即 75% 的层用线性注意力 [[gated-delta-rule]]（GDN）、25% 的层保留 softmax attention（GA）。MoE 采用 512 个 routed 专家（每 token 激活 10 个）外加 1 个 shared expert，专家中间维 512（模型卡）[^src-qwen3-next-modelcard-2025-09]。原生上下文 262,144，经 YaRN（factor 4.0）验证至 1,010,000（模型卡）[^src-qwen3-next-modelcard-2025-09]。

## Hybrid 动机与层设计

博客的取舍在于：线性注意力快但 recall 弱，标准 attention 推理贵且慢；系统实验表明 GDN 的 in-context learning 强于 Sliding Window Attention 与 Mamba2，3:1 混合一致优于任何单一架构[^src-qwen3-next-blog-2025-09]。

GA 层在标准 attention 上做了三处改动[^src-qwen3-next-blog-2025-09]：

1. 输出侧 sigmoid 门控，沿用其先行工作（[[source-gated-attention-for-large-language-models-2505-06708|Gated Attention 论文]]），用于缓解 attention 的低秩问题；
2. 单头维度从 128 提到 256；
3. partial RoPE——只对前 25% 的位置维施加旋转（模型卡给 RoPE 维 64、单头 256）[^src-qwen3-next-modelcard-2025-09]，以改善长序列外推。

输出门控同时消除了 Attention Sink（[[attention-sink]]）与 Massive Activation 两类异常，保证数值稳定[^src-qwen3-next-blog-2025-09]。

## Ultra-Sparse MoE

对比 Qwen3 的 128 专家选 8，Qwen3-Next 扩到 512 选 10 + 1 shared[^src-qwen3-next-blog-2025-09]。在 global load balancing（其引文 [4]）下，固定激活专家数、增大专家总量可持续降低训练损失[^src-qwen3-next-blog-2025-09]。机制谱系见 [[mixture-of-experts]]。

## 训练稳定性设计

Qwen3 用 QK-Norm 时出现部分 norm 权重异常增大；Qwen3-Next 改用 Zero-Centered RMSNorm（出处为其引文 [7]，Gemma），并对 norm 权重加 weight decay 防止无界增长[^src-qwen3-next-blog-2025-09]。MoE router 参数在初始化时归一化，让早期训练各专家被无偏选中（其引文 [8]）[^src-qwen3-next-blog-2025-09]。这些设计让小规模实验更可靠、大规模训练更顺；被 hybrid + 高稀疏 MoE 放大的 RL 训练稳定性问题「已解决」，无细节披露[^src-qwen3-next-blog-2025-09]。

## Multi-Token Prediction

原生 MTP（博客引文 [3][9][10]，即 DeepSeek-V3、Gloeckle et al.、ProphetNet 一系）兼作预训练辅助目标与投机解码 draft：其投机解码接受率高，且针对多步推理采用「训练-推理一致」的多步训练[^src-qwen3-next-blog-2025-09]。详见 [[multi-token-prediction]]，draft 由 MTP 模块供给 [[self-speculative-decoding]]。部署参数：SGLang `NEXTN`（3 步、topk 1、4 个 draft tokens）、vLLM `qwen3_next_mtp`（2 个 speculative tokens）；HF transformers 未提供 MTP（模型代码已进主线）[^src-qwen3-next-modelcard-2025-09]。

## 数据流与维度变换

**维度符号**：$N$ 序列长度，$d=2048$ 隐藏维。GA 含 $h_q{=}16$ 个 Q 头、$h_{kv}{=}2$ 个 KV 头、单头 $d_a{=}256$、RoPE 维 $d_r{=}64$；GDN 含 $h_{qk}{=}16$ 个 QK 头、$h_v{=}32$ 个 V 头、单头 $d_l{=}128$；MoE 含 $E{=}512$ 个 routed 专家、top-$10$ 加 $1$ shared、专家中间维 $d_f{=}512$（均出自模型卡[^src-qwen3-next-modelcard-2025-09]）。

### Stage 1 · 3:1 交替拼成 48 层

$$\begin{aligned}
\mathrm{layer}(\ell) &= \begin{cases} \mathrm{GDN}\!\to\!\mathrm{MoE} & \ell \bmod 4 \in \{0,1,2\} \\ \mathrm{GA}\!\to\!\mathrm{MoE} & \ell \bmod 4 = 3 \end{cases} && \text{// } [48] \xrightarrow{\ \ell \bmod 4\ } [36\ \mathrm{GDN} + 12\ \mathrm{GA}]\ \text{——模型卡 Hybrid Layout 展开即 } 12 \text{ 组「3 GDN + 1 GA」，博客口径 } 75\%/25\%
\end{aligned}$$

每层是「序列混合 + MoE FFN」两子层，残差流输入输出均为 $[N, 2048]$；pre-norm 与残差接法两源未给，按 Qwen 系惯例推断，标 [INFERENCE]。

### Stage 2 · Gated DeltaNet 层：常数状态递推

读 $x \in [N, 2048]$，投影分两路：

$$\begin{aligned}
Q, K &= xW_q,\ xW_k && \text{// } [N, \bcancel{2048}]\times[\bcancel{2048}, 2048]\to[N, 16, 128]\ \text{——}h_{qk}{=}16\ \text{个 QK 头}\\
V &= xW_v && \text{// } [N, \bcancel{2048}]\times[\bcancel{2048}, 4096]\to[N, 32, 128]\ \text{——}h_v{=}32\ \text{个 V 头，宽度为 QK 的两倍}
\end{aligned}$$

逐 token 状态更新（GDN 层取自博客参考文献 [1]；两源未给内部公式，标 [INFERENCE]）：

$$\begin{aligned}
S_i &= \alpha_i S_{i-1} + k_i\big(\beta_i v_i - S_{i-1}^{\!\top} k_i\big)^{\!\top} && \text{// } S_i \in [32, 128, 128] \xrightarrow{\ \text{沿序列逐步更新}\ } S_{i+1}\ \text{——delta rule：检索 } S_{i-1}^{\!\top} k_i \text{ 与目标 } \beta_i v_i \text{ 之差外积改写状态，}\alpha_i \in (0,1) \text{ 为遗忘门}
\end{aligned}$$

回投影（输出侧细节两源未给，标 [INFERENCE]）：

$$\begin{aligned}
o &= \Big[\{S_i^{\!\top} q_i\}_{i=1}^{32}\Big] W_o && \text{// } [N, 32, 128] \xrightarrow{\ \text{拼 32 头}\ } [N, 4096] \times [\bcancel{4096}, 2048] \to [N, 2048]\ \text{——按头检索后拼接回投}
\end{aligned}$$

推理时这 36 层只保留常数大小状态 $S$（每 V 头 $128 \times 128$），KV cache 不随 $N$ 增长。这是线性注意力的一般性质（[[linear-attention-rnn-form]]），也是 75% 的层免于 cache 线性增长的来源。

### Stage 3 · Gated Attention 层：GQA + partial RoPE + 输出门控

$$\begin{aligned}
Q &= xW_q && \text{// } [N, \bcancel{2048}]\times[\bcancel{2048}, 4096]\to[N, 16, 256]\ \text{——单头 } d_a{=}256\text{，Qwen3 的 } 128 \text{ 加倍（博客）}\\
K, V &= xW_k,\ xW_v && \text{// } \to[N, 2, 256]\ \text{——}h_{kv}{=}2\text{，每 KV 头服务 } 16/2 = 8 \text{ 个 Q 头（GQA）}\\
\tilde{Q}, \tilde{K} &= \mathrm{rot}_{64}\!\big(Q_{:64}\big) \oplus Q_{64:} && \text{// } [N, 16, 64] \xrightarrow{\ \text{仅前 64 维旋转}\ } [N, 16, 256]\ \text{——partial RoPE，} 64/256 = 25\% \text{ 位置维（博客「first 25% of position dimensions」+ 模型卡 RoPE 维 64）}\\
A &= \mathrm{softmax}\!\left(\frac{\tilde{Q}\tilde{K}^{\!\top}}{\sqrt{256}}\right) && \text{// } [16, N, \bcancel{256}]\times[16, \bcancel{256}, N]\to[16, N, N]\ \text{——缩放点积打分，softmax 沿 } N \text{ 维}\\
y &= W_o\big[\sigma(xW_g) \odot \mathrm{merge}(AV)\big] && \text{// } [N, 16, 256] \xrightarrow{\ \mathrm{merge}\ } [N, 4096] \odot [N, 4096] \xrightarrow{\ [\bcancel{4096}, 2048]\ } [N, 2048]\ \text{——输出侧逐通道 sigmoid 门控（博客，沿用其 Gated Attention 论文）}
\end{aligned}$$

RoPE 在头内的具体排布（频率铺法）两源未给，标 [INFERENCE]。

### Stage 4 · MoE：512 选 10 + 1 shared

$$\begin{aligned}
s &= x_r W_r && \text{// } [N, \bcancel{2048}]\times[\bcancel{2048}, 512]\to[N, 512]\ \text{——}E{=}512\ \text{个路由得分}\\
g &= \mathrm{Top}_{10}(s) && \text{// } [N, 512] \xrightarrow{\ \text{沿专家维 Top-}10\ } [N, 10]\ \text{——每 token 选 10 个 routed 专家}\\
y &= \mathrm{Shared}(x_r) + \sum_{j \in \mathrm{Top}_{10}} g_j\, E_j(x_r) && \text{// } [N, 2048]\ \text{——再加 1 个 shared expert；激活 } 10{+}1 \text{ 个中间维 } d_f{=}512 \text{ 的专家，约占 } 80\text{B 的 } 3.7\%\ (\approx 3\text{B，博客)}
\end{aligned}$$

$x_r$ 记归一化后的层输入；专家内部激活函数两源未给，标 [INFERENCE]。

### Stage 5 · MTP 模块：训练目标兼投机解码 draft

DeepSeek-V3 式骨架（博客引其 [3]；Qwen3-Next 的模块数与内部结构两源未给，标 [INFERENCE]）。

$$\begin{aligned}
P\big(x_{t+k+1} \mid x_{\le t}\big) &= \mathrm{Head}^{k}\!\Big(\mathrm{Tr}^{k}\big(h_t^{\,k-1} \,\Vert\, \mathrm{Emb}(x_{t+k})\big)\Big) && \text{// } [N, 2048] \oplus [N, 2048] \xrightarrow{\ \mathrm{Tr}^{k}\ } [N, 2048]\ \text{——主模型隐状态拼下一 token 嵌入，经轻量 trunk 预测更远 token}
\end{aligned}$$

## 训练与推理效率（均为博客自述[^src-qwen3-next-blog-2025-09]）

预训练从 Qwen3 的 36T 语料均匀采样 15T tokens，GPU 时少于 Qwen3-30B-A3B 的 80%，计算成本为 Qwen3-32B 的 9.3%。Prefill 吞吐在 4K 上下文约为 Qwen3-32B 的 7 倍，超 32K 后超 10 倍；Decode 在 4K 约 4 倍，超 32K 后仍超 10 倍。Base 版以 Qwen3-32B-Base 约 1/10 的非嵌入激活参数，在其大多数基准上更高。Instruct 版与旗舰 Qwen3-235B-A22B-Instruct-2507 相当，长上下文（至 256K）占优；Thinking 版超 Qwen3-30B-A3B-Thinking-2507 与 Qwen3-32B-Thinking，多个基准超 Gemini-2.5-Flash-Thinking（图表为位图）。

## 评测数字（模型卡表格）[^src-qwen3-next-modelcard-2025-09]

| 基准 | Qwen3-32B 非思考 | Qwen3-30B-A3B-2507 | Qwen3-235B-A22B-2507 | Qwen3-Next-80B-A3B |
|---|---|---|---|---|
| MMLU-Pro | 71.9 | 78.4 | **83.0** | 80.6 |
| GPQA | 54.6 | 70.4 | **77.5** | 72.9 |
| AIME25 | 20.2 | 61.3 | **70.3** | 69.5 |
| LiveCodeBench v6 | 29.1 | 43.2 | 51.8 | **56.6** |
| Arena-Hard v2（GPT-4.1 评审胜率） | 34.1 | 69.0 | 79.2 | **82.7** |

长上下文（RULER 1M 版，13 子任务 × 20 样本/长度）中，Qwen3-Next 平均 91.8（YaRN），Qwen3-235B-A22B-2507 为 92.5（Dual Chunk Attention），Qwen3-30B-A3B-2507 为 86.8[^src-qwen3-next-modelcard-2025-09]。博客称 Instruct 版在 256K 以内优于 235B 版[^src-qwen3-next-blog-2025-09]；模型卡表格显示，256K 档 93.5 高于 91.0，192K 档 94.0 低于 94.5，4K 档 98.5 持平[^src-qwen3-next-modelcard-2025-09]。

## 证据边界

技术发布而非正式论文中，「3:1 一致优于单一架构」「GDN 优于 SWA/Mamba2」「RL 稳定性已解决」均为博客自述的 systematic experiments 结论，未公开实验设置与数据。模型卡注明效率数字依赖具体实现，建议采用 SGLang/vLLM 而非 transformers 推理[^src-qwen3-next-modelcard-2025-09]；静态 YaRN 在短文本上可能损害性能，建议仅在长上下文场景开启并按需调小 factor[^src-qwen3-next-modelcard-2025-09]。Instruct 版只支持非思考模式，输出不含 `<think>` 块[^src-qwen3-next-modelcard-2025-09]。与 [[deepseek-v4-hybrid-attention]] 在注意力层内部交错 CSA/HCA 压缩 KV 相比，Qwen3-Next 在层间把 75% 的注意力整体替换为常数状态线性注意力，两条 hybrid 路线的介入位置不同。

## 相关页面

[[gated-delta-rule]] 对应 36/48 层的线性注意力层机制（在建前向链接），[[mixture-of-experts]] 梳理 512 选 10+1 的 ultra-sparse 形态谱系，[[multi-token-prediction]] 与 [[self-speculative-decoding]] 涵盖 MTP 目标与投机解码，[[source-gated-attention-for-large-language-models-2505-06708]] 为 GA 层输出门控的先行工作，[[attention-sink]] 对应输出门控消除的异常之一，[[linear-attention-rnn-form]] 给出线性注意力常数状态递推的一般形式，[[transformer-architecture]] 作为标准层结构对照。

[^src-qwen3-next-blog-2025-09]: [[source-qwen3-next-blog-2025-09]]
[^src-qwen3-next-modelcard-2025-09]: [[source-qwen3-next-modelcard-2025-09]]
