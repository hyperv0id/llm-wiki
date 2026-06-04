# Ingest 报告：STReasoner (Ni et al., 2026)

## 创建
- wiki/source-streasoner.md — WHY：源文件 `raw/streasoner-ni-2026.pdf` 的 source-summary，STReasoner 首次摄取，必须创建
- wiki/streasoner.md — WHY：STReasoner 是首个时空推理 TS-LM，与现有 Time-LLM/ChatTS/Time-R1 有明确的演化关系（增加 graph + multi-step CoT + spatial-aware RL），需独立 entity 页面
- wiki/spatio-temporal-reasoning.md — WHY：时空推理是一个新的范式——从"预测数值"到"回答需要空间归因的推理问题"，与 ST forecasting 有本质区别，需独立 concept 页面供其他页面引用

## 修改
- wiki/time-llm.md — WHY：STReasoner 是 Time-LLM TS-LM 范式的直接演化（同样 patchify + LLM 架构，增加了 graph awareness 和 S-GRPO），添加演化链接
- wiki/multimodal-time-series-forecasting.md — WHY：新增 STReasoner 小节，这是该概念页面表格中唯一以推理而非预测为核心任务的模型，frontmatter source_count: 7→8
- wiki/index.md — WHY：注册新 source (source-streasoner)、entity (streasoner)、concept (spatio-temporal-reasoning)
- wiki/log.md — WHY：完整记录本次 ingest

## 新建交叉链接
- [[streasoner]] ↔ [[source-streasoner]]
- [[streasoner]] ↔ [[spatio-temporal-reasoning]]
- [[streasoner]] → [[time-llm]]
- [[streasoner]] → [[spatio-temporal-foundation-model]]
- [[streasoner]] → [[vot]]
- [[spatio-temporal-reasoning]] → [[multimodal-time-series-forecasting]]
