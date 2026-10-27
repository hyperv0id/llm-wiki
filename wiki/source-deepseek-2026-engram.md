---
title: "Source: Conditional Memory via Scalable Lookup — Engram (arXiv:2601.07372)"
type: source-summary
tags:
  - source
  - conditional-memory
  - n-gram
  - sparse-model
  - mixture-of-experts
  - long-context
  - deepseek
created: 2026-10-27
last_updated: 2026-10-27
source_count: 1
confidence: medium
status: active
---

# Source: Conditional Memory via Scalable Lookup（Engram）

DeepSeek-AI 与北京大学，arXiv:2601.07372v2（2026-07-12 版；编号对应 2026-01 首发），预印本。本地 `raw/engram-2601-07372.pdf`，代码 https://github.com/deepseek-ai/Engram。

## 核心论点

论文将语言建模中的静态模式检索抽离为独立稀疏轴，称为条件记忆：条件计算（MoE）以稀疏激活参数处理动态逻辑，条件记忆则用稀疏查表取回固定知识的静态嵌入[^src-deepseek-2026-engram]。Engram 是其实例，在保留 $O(1)$ 查表的同时现代化经典 $N$-gram 嵌入：后缀 $N$-gram 先压缩词表（128k 词表缩小 23.43%），再经 8 个哈希头映射到素数槽位表，每 token 固定读取 16 行并拼成 1280 维向量，由当前隐藏状态作 sigmoid 门控，接 depthwise 因果卷积后残差注入第 2、15 层[^src-deepseek-2026-engram]。查表键为 token ID 而非隐藏状态，单 token 查表量与表大小无关，表可分片训练、按确定性下标预取，推理时可整体卸载到主机内存[^src-deepseek-2026-engram]。

## 证据

论文在固定总参数与激活参数下按比例 $\rho$ 分配 MoE 专家预算与 Engram 槽位，两档计算预算均呈 U 形曲线：约 20%–25% 稀疏预算给 Engram 最优，$\rho\approx40\%$ 时仍与纯 MoE 持平[^src-deepseek-2026-engram]。27B 模型（3.8B 激活、262B token）将 routed experts 从 72 降至 55，换取 5.7B 记忆；Table 1 的 27 项基准全部高于同参同 FLOPs 的 MoE-27B，BBH +5.0、CMMLU +4.0、HumanEval +3.0[^src-deepseek-2026-engram]。32k 长上下文扩展后，Multi-Query NIAH 97.0 对 84.2、Variable Tracking 89.0 对 77.0；100B 表卸载的吞吐损失最高 2.8%[^src-deepseek-2026-engram]。

## 边界

40B 未全面超过 27B，论文推测为欠训练；系统评测仅限 2/3-gram，长上下文止于 32k，未提供哈希碰撞率与表精度，也未对比 NSA/DSA[^src-deepseek-2026-engram]。摘要与 §1 的 MMLU +3.4 与 Table 1 的 +3.0 不一致[^src-deepseek-2026-engram]。

## 相关页面

[[engram]] 涵盖机制、分阶段数据流、参数记账、分配律与全部实验数值。

[^src-deepseek-2026-engram]: [[source-deepseek-2026-engram]]
