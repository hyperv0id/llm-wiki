---
title: "PAPerBench"
type: entity
tags:
  - benchmark
  - privacy
  - personalization
  - long-context
created: 2026-04-28
last_updated: 2026-04-28
source_count: 1
confidence: medium
status: active
---

# PAPerBench

PAPerBench (Privacy And Personalization Benchmark) 是 Gu 等人 (2026) 提出的长上下文评估基准，用于系统性地评估大语言模型在个性化生成和隐私推理任务上的表现[^src-paperbench]。

## 评估范围

- **上下文长度**：1K → 256K tokens[^src-paperbench]
- **任务类型**：个性化生成 + 隐私推理[^src-paperbench]
- **评估模型**：GPT-5.2, Qwen3-235B, Llama-3.3-70B, Llama-4-Scout-109B, Mistral-123B-2512, Mistral-24B-2501, Qwen2.5-14B 等[^src-paperbench]

## 任务设计

### 个性化任务 (Personalization)

1. **长上下文构建**：基于 PersonaHub 记录，重写 persona → 生成初始查询 → 生成 persona-grounded context → 自动扩展到目标长度[^src-paperbench]
2. **信号提取**：从长上下文中提取结构化个性化信号（显式约束 + 隐式偏���）[^src-paperbench]
3. **MCQ 构建**：gold 选项满足所有约束，near-miss 选项故意违反一个约束[^src-paperbench]

**Near-Miss 选项类型**：
- 遗漏关键约束 (Missing-Key)
- 忽略上下文偏好 (Ignore Context)
- 幻觉不支持的内容 (Hallucination)
- 结构/风格不一致 (Bad Structure)[^src-paperbench]

### 隐私任务 (Privacy)

1. **PII 注入**：将敏感信息（电话、邮箱、地址、SSN、信用卡、URL）注入上下文[^src-paperbench]
2. **诱饵注入**：注入诱饵 PII 增加难度，防止表面级检测[^src-paperbench]
3. **问题类型**：
   - **Per-Type 计数**：特定 PII 类型的出现次数[^src-paperbench]
   - **Aggregate 隐私**：跨类别的聚合属性（如泄露类别数、至少 k 种类型是否出现）[^src-paperbench]

## 关键发现

| 发现 | 描述 |
|------|------|
| 统一缩放 gap | 所有模型的个性化与隐私性能均随上下文长度增加而下降[^src-paperbench] |
| 模型容量效应 | 大模型渐进式下降，小模型提前崩溃[^src-paperbench] |
| 失败模式转变 | 短上下文：遗漏关键约束 → 长上下文：结构退化 + 幻觉[^src-paperbench] |
| 类别复杂度敏感 | 多类别隐私推理显著难于单类别[^src-paperbench] |
| 稀疏信号困难 | 敏感线索稀疏时隐私推理准确率急剧下降[^src-paperbench] |

## 数据集质量控制

在短上下文片段 (0.15K tokens) 上评估隐私保护性能，确保数据生成流程的质量[^src-paperbench]：

| 模型 | SSN | Email | Address | URL |
|------|-----|-------|---------|-----|
| Qwen3-235B | 100% | 100% | 100% | 100% |
| Qwen2.5-32B | 99.95% | 100% | 100% | 100% |
| Qwen2.5-14B | 96.30% | 99.80% | 83.05% | 100% |
| Qwen2.5-7B | 98.70% | 99.85% | 22.95% | 99.35% |

## 与现有基准的关系

- **vs. 长上下文理解基准**：现有基准（如 LongBench）主要评估信息检索和理解，PAPerBench 专注于**推理**任务[^src-paperbench]
- **vs. 隐私基准**：传统隐私基准多为短文本分类，PAPerBench 评估**长上下文隐私推理**能力[^src-paperbench]
- **vs. 个性化基准**：现有基准（如 LAMP）主要评估检索增强，PAPerBench 评估**纯上下文条件**下的个性化[^src-paperbench]

## 引用

[^src-paperbench]: [[source-paperbench]]