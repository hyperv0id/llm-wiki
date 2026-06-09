---
title: "Weather Prompt"
type: technique
tags:
  - weather-forecasting
  - in-context-learning
  - prompt
  - multi-modal
  - iclr-2025
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: medium
status: active
---

# Weather Prompt

**Weather Prompt（天气提示）** 是 [[weathergfm|WeatherGFM]] 为统一多模态天气理解任务设计的视觉提示格式，将 LLM/视觉基础模型的 in-context prompt 范式迁移到气象数据[^src-weathergfm]。

## 设计动机

天气理解任务的输入涉及多种模态：单一观测变量、多个不同气象变量、时序气象变量。标准视觉提示（如 Painter 的单图像对）无法覆盖这种模态多样性[^src-weathergfm]。WeatherGFM 因此为不同模态分配不同的提示短语/格式，使单一模型能通过 prompt 识别"当前是什么任务"[^src-weathergfm]。

## 三种提示格式

每种提示由一个示例 (input, target) 对加一个 query 输入组成，模型据示例感知任务语义后对 query 执行对应操作[^src-weathergfm]：

| 格式 | 输入模态 | 典型任务 | 示例 |
|------|----------|----------|------|
| **Weather Prompt1** | 单模态（类视觉提示） | 空间/时序超分、去模糊 | `{example: image1, image2}, query: image3 → image4` |
| **Weather Prompt2** | 多模态（跨通道） | 天气图像翻译 | `{example: image1,image2,image3}, query: image4,image5 → image6`（如 IR069+IR107 → 雷达 VIL） |
| **Weather Prompt3** | 时序 | 天气预报 | `{example: seq1, seq2}, query: seq3 → seq4`（雷达/卫星外推） |

形式化为：给定提示对 $(P_{in}, P_{target})$ 与 query 输入 $X_{in}$，模型输出
$$X_{target} = F_\tau(P_{in}, P_{target}, X_{in}; \theta)$$
通过选择任务特定的 $(P_{in}, P_{target})$ 即可决定对 $X_{in}$ 执行的任务[^src-weathergfm]。

## Prompt 选择的影响

不同 prompt 对性能有显著影响：超分任务对 prompt 随机性不敏感（CSI 标准差极小），而天气预报与图像翻译任务波动较大（约 0.02 CSI）[^src-weathergfm]。WeatherGFM 进一步比较了 random / high-quality / searched 三种 prompt 选择策略——从高值雷达样本构成的高质量库或按 RMSE 相似度检索的 prompt 优于随机 prompt[^src-weathergfm]。这表明 WeatherGFM 可通过 prompt 对特定天气事件做交互式调整，而非黑箱模型[^src-weathergfm]。

## 相关页面

- [[weathergfm]] — WeatherGFM 主模型
- [[mixed-modal-masked-image-modeling]] — 消费天气提示的训练/推理范式
- [[in-context-learning]] — in-context learning 范式
- [[source-weathergfm]] — 源文件摘要

[^src-weathergfm]: [[source-weathergfm]]
