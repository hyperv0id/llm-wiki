---
title: "Source: Qwen3-Next 发布博客（Qwen Team, 2025-09-11）"
type: source-summary
tags:
  - source
  - llm
  - architecture
  - hybrid-attention
  - mixture-of-experts
  - multi-token-prediction
created: 2026-10-27
last_updated: 2026-10-27
source_count: 1
confidence: medium
status: active
---

# Source: Qwen3-Next 发布博客

Qwen Team 官方博客《Qwen3-Next: Towards Ultimate Training & Inference Efficiency》于 2025-09-11 发布于 https://qwen.ai/blog?id=4074cca80393150c248e508aa62983f9cb7d27cd ，本地全文见 `raw/qwen3-next-blog-2025-09.md`。qwen.ai 为纯客户端单页应用，正文经无头浏览器渲染提取（2026-10-27）。

## 核心论点

博客将 Context Length Scaling 与 Total Parameter Scaling 定为两大趋势，针对训练与推理成本提出四项改动：Gated DeltaNet 与 Gated Attention 按 3:1 混合的 hybrid 注意力（75% 层线性注意力、25% 层带输出门控的 softmax attention、单头 256、partial RoPE 仅旋转前 25% 位置维）；ultra-sparse MoE（512 选 10 + 1 shared，80B 总参激活约 3B、占 3.7%）；Zero-Centered RMSNorm 配套 norm 权重 weight decay 与 router 初始化归一化两项稳定性设计；原生 Multi-Token Prediction 兼作预训练目标与投机解码 draft[^src-qwen3-next-blog-2025-09]。

## 关键数字（博客自述）

Base 版（15T tokens，取自 Qwen3 的 36T 语料）以不足 Qwen3-32B 十分之一的训练成本达到可比或略优表现，计算成本精确为 9.3%，GPU 时少于 Qwen3-30B-A3B 的 80%；上下文超 32K 后 prefill 与 decode 吞吐均超 Qwen3-32B 的 10 倍[^src-qwen3-next-blog-2025-09]。Instruct 版与 Qwen3-235B-A22B-Instruct-2507 相当且长上下文占优；Thinking 版超过 Qwen3-30B-A3B-Thinking-2507 与 Qwen3-32B-Thinking，并在多个基准超过 Gemini-2.5-Flash-Thinking[^src-qwen3-next-blog-2025-09]。团队解决了 hybrid + 高稀疏 MoE 给 RL 训练带来的稳定性问题，该架构将延续至 Qwen3.5[^src-qwen3-next-blog-2025-09]。

## 局限

该文为技术发布而非正式论文：未提供消融表，「3:1 一致优于单一架构」「GDN 强于 SWA/Mamba2」等关键比较仅有结论而无数据；全部基准图均为位图，图内数字不可读；正文有一处型号笔误（`Qwen3-30A-3B`，应为 Qwen3-30B-A3B）；部署命令代码块未随页面文本导出（以模型卡为准）。评测数字取自[[source-qwen3-next-modelcard-2025-09|模型卡]]表格，完整数据见 [[qwen3-next-hybrid]]。

[^src-qwen3-next-blog-2025-09]: [[source-qwen3-next-blog-2025-09]]
