---
title: "Source: DeepSeek-V3.2 — DSA 与后训练扩展 (arXiv:2512.02556)"
type: source-summary
tags:
  - source
  - attention
  - sparse-attention
  - long-context
  - reinforcement-learning
  - agentic
created: 2026-10-24
last_updated: 2026-10-24
source_count: 1
confidence: medium
status: active
---

# Source: DeepSeek-V3.2（arXiv:2512.02556）

- **作者**：DeepSeek-AI（集体署名，Aixin Liu 列首位）
- **时间**：2025-12-02；cs.CL；预印本技术报告，未见 venue
- **链接**：https://arxiv.org/abs/2512.02556
- **本地**：`raw/2512.02556-deepseek-v3-2.md`（LaTeXML 转换的 HTML 全文，约 830 行）
- **实现**：https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp/tree/main/inference

## 核心论点

报告列出三项改动：DeepSeek Sparse Attention（DSA）、规模化的强化学习后训练、大规模 agentic 任务合成（摘要）[^src-2512-02556-deepseek-v3-2]。相对 V3.1-Terminus 的**唯一架构改动**是 DSA，通过继续训练引入，模型结构本身与 DeepSeek-V3.2-Exp 完全一致（§2.1）[^src-2512-02556-deepseek-v3-2]。

## 方法

**DSA** 由 lightning indexer 与 fine-grained token selection 组成：indexer 用少量 FP8 头对每个前序 token 打标量分（ReLU 激活，式 1），主注意力只取分数最高的 $k$ 个 MLA KV latent（式 2）；因 kernel 需要 KV entry 跨 query 共享，DSA 在 MLA 的 MQA 模式下实例化（§2.1）[^src-2512-02556-deepseek-v3-2]。主注意力复杂度 $\mathcal{O}(L^2)\to\mathcal{O}(Lk)$，indexer 仍为 $\mathcal{O}(L^2)$（§2.3）[^src-2512-02556-deepseek-v3-2]。训练分 dense warm-up（冻结除 indexer 外参数，全序列 KL，2.1B token）与 sparse training（全参数，选中集合 KL + LM loss，$k=2048$，943.7B token）两阶段，indexer 输入 detach、单独优化（§2.1.1）[^src-2512-02556-deepseek-v3-2]。

**后训练**沿用 specialist distillation → mixed RL：六个专门域的专家从同一 base checkpoint 微调后再蒸馏；GRPO 上有四项工程改动（无偏 KL 估计、off-policy 序列掩码、保留路由、保留采样掩码），均只给动机与形式化（§3、§3.1）[^src-2512-02556-deepseek-v3-2]。agentic 数据由搜索、代码、代码解释器、通用 agent 四条流水线合成，合计 85,267 条任务、1,827 个环境（Table 1）[^src-2512-02556-deepseek-v3-2]。

## 证据

- 稀疏化 parity 只有定性表述：标准基准「similar performance」、ChatbotArena Elo「closely matched」，唯一数字是 AA-LCR 高 4 分；对照对象是 V3.2-Exp base checkpoint（§2.2）[^src-2512-02556-deepseek-v3-2]。
- 推理成本只给 Figure 3 的每百万 token 成本曲线（H800、2 美元/GPU 小时），无成本数值或加速比（§2.3）[^src-2512-02556-deepseek-v3-2]。
- 主结果（Table 2）：MMLU-Pro 85.0、GPQA Diamond 82.4、AIME 2025 93.1、LiveCodeBench 83.3、SWE Multilingual 70.2；推理总体与 GPT-5-high 相当，低于 Gemini-3.0-Pro（§4.1）[^src-2512-02556-deepseek-v3-2]。
- Speciale 变体（Table 3）：AIME 2025 96.0、HMMT Feb 99.2、IMOAnswerBench 84.5、CodeForces 2701，但 token 消耗显著高于 Gemini-3.0-Pro；竞赛成绩 IMO 35/42、IOI 492/600（Table 4）[^src-2512-02556-deepseek-v3-2]。

## 局限

局限有三条（§5）：世界知识广度受预训练算力限制、token 效率落后 Gemini-3.0-Pro、复杂任务仍弱于前沿模型[^src-2512-02556-deepseek-v3-2]。报告未披露参数量与预训练 FLOPs，GRPO 四项改动无消融与超参数，DSA 无 $k$ 敏感性、无 indexer 消融、无与 NSA/MoBA 的同台对比，评测无方差，§2.2 的 parity 无法把质量持平单独归因于稀疏化。

机制与训练细节见 [[deepseek-sparse-attention]]，模型与评测见 [[deepseek-v3-2]]。

[^src-2512-02556-deepseek-v3-2]: [[source-2512-02556-deepseek-v3-2]]
