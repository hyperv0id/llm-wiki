---
title: "Lead-Lag Dependency"
type: concept
tags:
  - time-series-forecasting
  - traffic-forecasting
  - causality
  - lead-lag-dependency
created: 2026-09-05
last_updated: 2026-09-05
source_count: 1
confidence: medium
status: active
---

# Lead-Lag Dependency

**Lead–lag dependency（先行-滞后依赖）** 指一个序列/位置的观测在时间上先行、另一序列/位置的观测滞后跟随的跨维度依赖。LagLLM 论文给出形式化定义：位置 $i$ 在 $t_1$ 时刻与位置 $j$ 在 $t_2$ 时刻存在 lead–lag 依赖，当且仅当 $P(X_{t_1}^i \mid X_{t_2}^j) \neq P(X_{t_1}^i)$；时滞定义为 $\tau_{ij}=|t_1-t_2|$。在此定义下，空间依赖（$t_1=t_2$）与时间依赖（$i=j$）是一维退化形式[^src-lagllm]。

## 建模方法谱系

论文按三类归纳既有做法[^src-lagllm]：

1. **统计方法**：Granger 因果、交叉相关、转移熵等，可解释但依赖线性/平稳假设。
2. **领域特定方法**：专家手工指标（如量化金融中的价量曲线），先验强但难泛化。
3. **深度模型**：memory bank（[[pdformer|PDFormer]]、MegaCRN 的原型匹配）、注意力/自适应图（TraverseNet、FCSTGNN）、时空拼接图（STSGCN）、预计算交叉相关 + 神经网络（LIFT、MillGNN）。表达力强但纯数据驱动，对噪声敏感、易学到虚假 lead–lag 模式[^src-lagllm]。

## 为什么难

- **跨维度耦合**：影响同时跨空间与时间（上游 → 下游延迟传播），把空间与时间当作独立组件的模型无法识别[^src-lagllm]。
- **方向性**：需要判断谁 lead 谁 lag，无方向先验的稠密建模有过拟合风险[^src-lagllm]。
- **标注稀缺**：时空数据几乎不带"谁 lead 谁"的文本语义标注，LLM 难以直接推断[^src-lagllm]。

## LLM 引入的契机（LagLLM 的主张）

LagLLM 论文主张：LLM 参数化知识可作为"推理锚"，判断数据驱动图学习候选的 lead–lag 关系是否稳定可解释，从而滤除噪声与虚假相关；再通过 [[structural-token-sorting|token 排序]] 把图结构转译为序列顺序，使自回归注意力天然先见 lead 后见 lag。其案例研究显示，ChinaAQI 上学到的 semantic mask 与污染物 12 小时的真实扩散方向一致[^src-lagllm]。

## 与其他延迟建模的关系

- MegaCRN 通过 meta-node 原型匹配隐式承载延迟效应，LagLLM 将其与 [[pdformer|PDFormer]] 同列为 memory-based 延迟量化代表[^src-lagllm]
- [[patchstg|PatchSTG]] 等共享骨架路线则主张逐对延迟交互可被低维中介替代。
- [[pir|PIR]] 的局部修订模块同样以协变量间领先-滞后依赖为动机，但作用在预测后修订而非图结构学习。

[^src-lagllm]: [[source-lagllm-icml2026]]
