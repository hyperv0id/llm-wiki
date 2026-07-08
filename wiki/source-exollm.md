---
title: "ExoLLM: Exploiting Language Power for Time Series Forecasting with Exogenous Variables"
type: source-summary
tags:
  - llm
  - spatiotemporal
  - exogenous-variables
  - 2025
created: 2026-07-07
last_updated: 2026-07-07
source_count: 1
confidence: medium
status: active
---

# ExoLLM: Exploiting Language Power for Time Series Forecasting with Exogenous Variables

> Qihe Huang, Zhengyang Zhou, Kuo Yang, Yang Wang (USTC). WWW 2025. DOI: [10.1145/3696410.3714793](https://doi.org/10.1145/3696410.3714793)

**ExoLLM** 是首个利用大语言模型（LLM）进行**外生变量时间序列预测**（Forecasting with Exogenous Variables, FEV）的方法。核心洞察：LLM 在海量文本上预训练获得的开放世界知识，能够辅助理解外生变量对内生变量的复杂影响，从而提升预测精度。[^src-exollm]

现实世界中仅关注内生变量（目标变量）的预测方法往往不够准确，因为网页流量、电力负荷等系统容易受到外生变量（如天气、交通流、社会事件）的复杂影响。现有 FEV 方法仅依赖数值型外生序列建模，难以捕获多粒度时间依赖且容易学习到虚假关联——例如天气晴朗时交通流量也可能因管控措施而下降，纯数值模型无法理解这种因果语境。[^src-exollm]

---

## 核心挑战

直接应用 LLM 到 FEV 面临三大障碍：[^src-exollm]

1. **任务激活（Task Activation）**：NLP 与时间序列预测之间存在根本性任务差异，需构造有效的任务指令引导 LLM 实现跨任务知识迁移。
2. **外生知识提取（Exogenous Knowledge Extraction）**：外生变量的影响是多粒度的（自然属性、趋势、周期、稳定性、噪声强度），需要结构化提示来充分挖掘 LLM 中蕴含的外部环境知识。
3. **特征空间对齐（Feature Space Alignment）**：LLM 处理离散文本，而时间序列是连续数值，两者处于不同的特征空间，需设计编码-解码策略实现跨模态对齐。

---

## ExoLLM 框架

### Meta-task Instruction（元任务指令）

包含三个要素：数据集整体描述与领域识别、内外生变量的简要总结、FEV 任务介绍。构造为固定结构的文本描述（`TD = {td_task, td_exo⁽¹⁾, ..., td_exo⁽ᴹ⁾, td_end}`），输入 LLM 后在末尾 `<EOS>` token 处取统一大小的嵌入表示。[^src-exollm]

### Multi-grained Prompt（多粒度提示）

覆盖外生变量的五个维度，每维度提供多个候选模板（共 20 个提示）：自然属性描述（4 种）、趋势相关性（4 种，如总体上升/先升后降/总体下降/先降后升）、周期关系（4 种，如无明显周期/较短周期/清晰周期/较长周期）、稳定性（4 种，如大幅波动/相对稳定/偶发波动/持续稳定）、噪声强度（4 种）。由模型动态选择当前序列最匹配的提示，实现多级外部环境理解。[^src-exollm]

### Temporal-property Preserved Tokenizer（TPT）

将外生变量 E 和内生变量 X 分割为非重叠 patch，使用 Self-Attention 学习 patch 间的时序交互，选取最后一个 patch（与未来最接近）作为整条序列的压缩 token 表示，保留时序语义和局部上下文。[^src-exollm]

### Dual TS-Text Attention（DT² Attention）

双路时序-文本注意力机制：[^src-exollm]
- **TS-Text Attention**（编码前）：以时序空间 token 为 Query，以该变量对应的多粒度提示为 Key/Value，通过 Cross-Attention 将 token 映射到文本空间，同时区分内生和外生 token 防止表示过度平滑。
- **Text-TS Attention**（解码后）：LLM 编码后的内生 token 处于文本空间。以其为 Query、以外生时序 token 为 Key/Value 进行 Cross-Attention，将内生 token 解码回时序空间以生成预测。

最终通过轻量线性预测头将解码后的 token 映射为未来序列 `X̂ ∈ R¹×ᵀ`。LLM 的自注意力和位置嵌入层被冻结以保留预训练语言知识。[^src-exollm]

---

## 实验与发现

在 12 个真实数据集（ECL、Weather、ETTh1/ETTh2、ETTm1/ETTm2、Traffic，以及 PJM、NP、BE、FR、DE 等短期电力数据集）上，与 10 个 SOTA 基线（TimeXer、iTransformer、PatchTST、Crossformer、TiDE、SCINet、Autoformer、GPT4TS、TimeLLM、LLM4TS）对比：[^src-exollm]

| 场景 | 表现 |
|------|------|
| **长期预测** | 56 个设置中 51 个 top-1、5 个 top-2。相对 TimeXer 降低 MSE 9.1%、MAE 4.1%；相对 TimeLLM 降低 MSE 31.2%、MAE 19.8% |
| **短期预测** | 5 个数据集平均 MSE 0.288、MAE 0.251。相对 SCINet 降低 MSE 46.1%、MAE 35.5% |
| **少样本预测** | 10% 训练数据下，相对 GPT4TS 降低 MSE 8.9%、MAE 4.5% |
| **零样本预测** | 跨数据集泛化（如 ETTh1→ETTh2、ETTm1→ETTm2），性能提升超 5% |

### 消融分析

去除 MGP 和 MTI 导致最显著的性能下降（如 Weather 上 MSE 从 0.001 升至 0.003）；DT² Attention 移除后各数据集一致退化；TPT 替换为线性层后精度小幅降低。[^src-exollm]

### 外生变量规模与注意力分析

提示数量从 0 增加到 16 时性能持续提升；外生变量数量从 0% 增至 100% 同样持续改善。Trend 是最关键的外生提示类型（移除后 ETTh1 上 MSE 升至 0.071），HUFT 是最重要的外生变量（移除后 MSE 升至 0.074）。注意力图可视化表明 MTI token 受到各变量的广泛关注，验证了元任务指令成功激活了 LLM 从 NLP 到 FEV 的知识迁移。[^src-exollm]

---

## 与相关工作的关系

ExoLLM 与 [[source-exost]]（时空预测中外生变量的 select-then-balance 范式）和 [[source-e2-cstp]]（因果多模态时空预测）同属外生变量建模方向，但各有侧重：
- ExoST 关注数值型外生变量的选择与平衡，未涉及 LLM 或文本模态。
- E²-CSTP 利用多模态融合（文本+图像+时序）进行因果推断，但其文本来自事件描述而非外生变量的结构化提示。
- ExoLLM 首创性地利用 LLM 的预训练语言知识来理解外生变量对内生变量的多粒度影响，并专门设计了文本-时序特征对齐机制。

---

## 局限性

- 依赖预定义的多粒度提示模板，未能实现提示的完全自动化生成。[^src-exollm]
- 仅使用 LLM 的编码器部分，未探索生成式解码策略（如直接生成未来序列文本描述）。
- 外生变量以数值形式输入，文本知识仅通过提示间接注入——真正意义上的多模态端到端融合仍有提升空间。

---

## 引用

[^src-exollm]: [[source-exollm]]