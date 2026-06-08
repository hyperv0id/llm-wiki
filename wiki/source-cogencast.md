---
title: "Source: CoGenCast (ICML 2026)"
type: source-summary
tags:
  - time-series
  - llm
  - flow-matching
  - generative-model
  - icml-2026
  - encoder-decoder
  - one-step-generation
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: high
status: active
---

# Source: CoGenCast

**CoGenCast: A Coupled Autoregressive–Flow Generative Framework for Time Series Forecasting**  
Yaguo Liu, Mingyue Cheng, Daoyu Wang, Xiaoyu Tao, Qi Liu (USTC, State Key Laboratory of Cognitive Intelligence)  
ICML 2026 · arXiv:2602.03564 · Code: [github.com/liuyaguo/_CoGenCast](https://github.com/liuyaguo/_CoGenCast)

## 核心论点

CoGenCast 提出**时间序列预测的理想方法应同时具备双重能力**：对上下文条件的语义理解和对连续时间动态的随机建模[^src-cogencast]。现有方法中，基于 LLM 的方法擅长语义理解但缺乏连续随机建模，基于扩散/流匹配的方法能建模不确定性但缺乏语义理解——两者结合无人做到。

## 三大创新

1. **LLM 架构重配置**：将预训练的 decoder-only LLM（Qwen3-0.6B 默认）通过仅修改注意力拓扑重构为原生预测 encoder-decoder 骨架——encoder 采用双向自注意力融合回看窗口和上下文特征，decoder 保持因果自注意力生成未来表示[^src-cogencast]。

2. **LLM 条件化的流匹配机制**：流匹配去噪 decoder 以 LLM decoder 自回归生成的表示作为条件，学习区间条件化的平均速度场，建模连续随机动态[^src-cogencast]。

3. **一步生成**：通过 JVP (Jacobian-Vector Product) 修正的优化目标显式惩罚速度变异性，使学习到的流轨迹趋近直线，支持单步函数求值完成生成，延迟极低[^src-cogencast]。

## 实验结果

在 10 个基准数据集上（Energy, ETTh1/ETTh2, ETTm1/ETTm2, Environment, Exchange, Health, Wind, Solar）对比 8 个基线（LLM 类：LLM4TS, Time-LLM；生成类：FlowTS, CDPM, CSDI；Transformer 类：TimeDART, PatchTST, Autoformer），CoGenCast 在绝大多数数据集上取得最优[^src-cogencast]。平均 MSE 较 LLM 类基线降低约 11%，较强 Transformer 基线降低约 7%。消融实验确认 encoder-decoder 架构、AR 机制和流匹配模块三者均关键[^src-cogencast]。

## 局限

- 仅测试 Qwen3 系列 LLM backbone，其他 LLM 家族的适用性未验证[^src-cogencast]
- 计算开销较纯 Transformer 基线更高（LLM backbone 推理成本）
- 一步生成虽高效，但在少数场景额外采样步数能带来边际收益[^src-cogencast]
- arXiv preprint 状态，peer review 确认中

[^src-cogencast]: [[source-cogencast]]