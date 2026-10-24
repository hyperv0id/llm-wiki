---
title: "DeepSeek-V3.2"
type: entity
tags:
  - llm
  - deepseek
  - attention
  - sparse-attention
  - reinforcement-learning
  - agentic
created: 2026-10-24
last_updated: 2026-10-24
source_count: 1
confidence: medium
status: active
---

# DeepSeek-V3.2

DeepSeek-AI 的开放权重 LLM 系列，技术报告为 arXiv:2512.02556（2025-12-02）[^src-2512-02556-deepseek-v3-2]。谱系上有三个名字：**DeepSeek-V3.2-Exp** 是先行发布的实验版本，架构与最终版完全相同；**DeepSeek-V3.2** 在 Exp 基础上走完整后训练；**DeepSeek-V3.2-Speciale** 是高算力实验变体，只用推理数据并降低长度惩罚（§2.1、§3）[^src-2512-02556-deepseek-v3-2]。

## 架构

相对 DeepSeek-V3.1-Terminus，**唯一架构改动**是通过继续训练引入 [[deepseek-sparse-attention|DSA]]；上下文长度 128K，MLA 的 MQA 模式承载稀疏 KV 选择（§2.1）[^src-2512-02556-deepseek-v3-2]。DSA 随模型进入线上服务：Figure 3 的成本按 H800 上实际部署的服务基准、2 美元/GPU 小时折算（§2.3）[^src-2512-02556-deepseek-v3-2]。参数量与预训练 FLOPs 未披露。

## 后训练配方

**Specialist distillation**：除写作与通用 QA 外共 6 个专门域（数学、编程、通用逻辑推理、通用 agentic、agentic coding、agentic search），每个域从同一个 base checkpoint 微调出专家、各自做大规模 RL，再蒸馏回单一模型；每个域同时覆盖 thinking 与 non-thinking 模式（§3）[^src-2512-02556-deepseek-v3-2]。

**Mixed RL**：把 reasoning、agent、human alignment 三类训练合并为一个阶段，算法仍是 GRPO；reasoning 与 agent 用规则化结果奖励 + 长度惩罚 + 语言一致性奖励，通用任务用带 per-prompt rubrics 的生成式奖励模型（§3）[^src-2512-02556-deepseek-v3-2]。GRPO 上有四项改动（§3.1）[^src-2512-02556-deepseek-v3-2]：

| 改动 | 做法 | 证据 |
|---|---|---|
| Unbiased KL Estimate | 用重要性比率修正 K3 估计：$\frac{\pi_\theta}{\pi_{\mathrm{old}}}\left(\frac{\pi_{\mathrm{ref}}}{\pi_\theta}-\log\frac{\pi_{\mathrm{ref}}}{\pi_\theta}-1\right)$（式 7） | 仅定性称消除系统性偏差；无表 |
| Off-Policy Sequence Masking | 负优势且序列平均策略散度超 $\delta$ 的整条序列掩掉（式 8–9） | 仅定性「improves stability」；$\delta$ 未给 |
| Keep Routing | 训练时复用推理框架记录的 expert 路由 | 称自 DeepSeek-V3-0324 起沿用；无表 |
| Keep Sampling Mask | 把采样时的 top-p/top-k 截断掩码施加到训练策略 | 仅定性称保持语言一致性；无表 |

**Agentic 任务合成**：搜索、代码、代码解释器、通用 agent 四条流水线，合计 85,267 条任务（代码 24,667 / 搜索 50,275 / 通用 4,417 / 代码解释器 5,908），覆盖 1,800 余个环境（Table 1、§1）[^src-2512-02556-deepseek-v3-2]。通用 agent 环境由 environment-synthesis agent 自动构造 1,827 个，并以 pass@100 非零为保留条件（§3.2.3）[^src-2512-02556-deepseek-v3-2]。

## 评测

**主结果（Table 2）**，thinking 模式、128K 上下文[^src-2512-02556-deepseek-v3-2]：

| 指标 | V3.2 | GPT-5 High | Gemini-3.0-Pro | Claude-4.5-Sonnet |
|---|---|---|---|---|
| MMLU-Pro (EM) | 85.0 | 87.5 | 90.1 | 88.2 |
| GPQA Diamond | 82.4 | 85.7 | 91.9 | 83.4 |
| HLE (text) | 25.1 | 26.3 | 37.7 | 13.7 |
| AIME 2025 | 93.1 | 94.6 | 95.0 | 87.0 |
| LiveCodeBench | 83.3 | 84.5 | 90.7 | 64.0 |
| SWE Verified | 73.1 | 74.9 | 76.2 | 77.2 |
| SWE Multilingual | 70.2 | 55.3 | – | 68.0 |
| Terminal Bench 2.0 | 46.4 | 35.2 | 54.2 | 42.8 |

推理能力与 GPT-5-high 相当，低于 Gemini-3.0-Pro（§4.1）[^src-2512-02556-deepseek-v3-2]。若干指标依赖脚手架：Terminal Bench 用 Claude Code（thinking 的上下文管理与 Terminus 不兼容），SWE Verified 用内部框架，四套脚手架复测区间 72–74（§4.1）[^src-2512-02556-deepseek-v3-2]。

**Speciale（Table 3）**：AIME 2025 96.0（23k 输出 token）、HMMT Feb 99.2（27k）、HMMT Nov 94.4（25k）、IMOAnswerBench 84.5（45k）、LiveCodeBench 88.7（27k）、CodeForces 2701（77k）、GPQA 85.7、HLE 30.6；相对 Gemini-3.0-Pro 在 HMMT 两场、IMOAnswerBench、LiveCodeBench 上更高，CodeForces 基本持平（2701 vs 2708），GPQA 与 HLE 仍落后，且输出 token 数显著更多（如 CodeForces 77k vs 22k）（§4.2）[^src-2512-02556-deepseek-v3-2]。

**竞赛（Table 4）**：IMO 2025 35/42（金牌，P6 为 0）、CMO 2025 102/126、IOI 2025 492/600；表注给出 ICPC WF 第 2 名、IOI 第 10 名，ICPC 逐题提交数子表在本地转换稿中丢失。协议为无工具、最大生成 128k，IOI 每题采 500 个候选后取思维链最长的 50 个提交，ICPC 采 32 个（附录 D）[^src-2512-02556-deepseek-v3-2]。

**合成任务与上下文管理（Table 5、Fig. 5–6）**：50 个随机通用合成任务上 pass@1/2/4 为 12%/18%/26%，低于 GPT-5-Thinking 的 62%/75%/82%；只在 search+code 环境做 RL 不提升 MCP/Tau2 基准，加入通用合成任务后才提升（§4.3）[^src-2512-02556-deepseek-v3-2]。搜索 agent 在上下文占用超 80% 时执行摘要或丢弃历史：丢弃全部工具历史使 BrowseComp 从 51.4 升到 67.6，摘要策略平均步数 364（§4.4）[^src-2512-02556-deepseek-v3-2]。

## 边界

局限有三条：预训练总 FLOPs 少于前沿闭源模型，世界知识广度落后；达到同等质量需要更长的生成轨迹，token 效率待优化；复杂任务解决能力仍逊于前沿模型（§5）[^src-2512-02556-deepseek-v3-2]。报告另有三类证据缺口：GRPO 四项改动无消融、无超参数，专家蒸馏只有「marginally below」的定性说法，各表无方差或重复次数；DSA 的效率证据是部署成本曲线而非加速比，且未见超过 128K 的评测（§3.1、§3、§4.1、§2.3）[^src-2512-02556-deepseek-v3-2]。

## 相关页面

- [[deepseek-sparse-attention]] — 架构改动本体
- [[source-2512-02556-deepseek-v3-2]] — 源摘要
- [[moba-attention]] · [[kv-cache-compression]] — 相邻的长上下文稀疏路线
- [[grpo-for-forecasting]] — GRPO 在时空预测上的另一处应用

[^src-2512-02556-deepseek-v3-2]: [[source-2512-02556-deepseek-v3-2]]
