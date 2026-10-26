---
title: "Source: Qwen3-Next-80B-A3B-Instruct 模型卡（HuggingFace, 2025-09）"
type: source-summary
tags:
  - source
  - llm
  - architecture
  - hybrid-attention
  - mixture-of-experts
  - long-context
created: 2026-10-27
last_updated: 2026-10-27
source_count: 1
confidence: medium
status: active
---

# Source: Qwen3-Next-80B-A3B-Instruct 模型卡

HuggingFace `Qwen/Qwen3-Next-80B-A3B-Instruct` 的 README（Apache-2.0）于 2026-10-27 抓取自 https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct/raw/main/README.md，本地全文见 `raw/qwen3-next-modelcard-2025-09.md`。文件与发布博客同期，是 config 与评测表的第二事实来源。

## 模型配置（模型卡「Model Overview」）

Causal LM，仅 instruct（非思考）模式，输出不含 `<think>` 块；15T tokens 预训练加后训练。总参 80B / 激活 3B，非嵌入 79B，hidden dim 2048；48 层按 `12 * (3 * (Gated DeltaNet -> MoE) -> 1 * (Gated Attention -> MoE))` 交替[^src-qwen3-next-modelcard-2025-09]。Gated Attention：16 个 Q 头、2 个 KV 头、单头 256、RoPE 维 64；Gated DeltaNet：32 个 V 头、16 个 QK 头、单头 128；MoE：512 专家激活 10 个加 1 个 shared、专家中间维 512；原生上下文 262,144、YaRN 后 1,010,000[^src-qwen3-next-modelcard-2025-09]。

## 评测数字（模型卡表格）

与 Qwen3-32B 非思考、Qwen3-30B-A3B-Instruct-2507 及 Qwen3-235B-A22B-Instruct-2507 对照：MMLU-Pro 80.6、GPQA 72.9、AIME25 69.5 均低于 235B 版；LiveCodeBench v6 56.6、Arena-Hard v2 82.7 为同表最高[^src-qwen3-next-modelcard-2025-09]。3B 激活在代码与偏好对齐上追平 22B 激活旗舰，知识、数学与多数 Agent 基准仍有差距。RULER 1M 版平均 91.8（YaRN），对比 235B 版 92.5（Dual Chunk Attention）与 30B-A3B 版 86.8；256K 档 93.5 高于 235B 版的 91.0，192K 档 94.0 低于其 94.5，4K 档持平 98.5；评测每长度 260 样本（13 子任务 × 20）[^src-qwen3-next-modelcard-2025-09]。

## 部署与注意事项

MTP 未在 HF transformers 中提供；实际加速高度依赖实现，推荐 SGLang（`NEXTN`，3 步 / topk 1 / 4 draft tokens，需 `sglang>=0.5.2`）或 vLLM（`qwen3_next_mtp`，2 speculative tokens，需 `vllm>=0.10.2`）[^src-qwen3-next-modelcard-2025-09]。静态 YaRN 的 factor 恒定，可能损害短文本，建议按需开启并调小 factor；采样推荐 Temperature 0.7 / TopP 0.8 / TopK 20[^src-qwen3-next-modelcard-2025-09]。完整参数与数据见 [[qwen3-next-hybrid]]。

[^src-qwen3-next-modelcard-2025-09]: [[source-qwen3-next-modelcard-2025-09]]
