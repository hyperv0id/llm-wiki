---
title: "TESS (Temporal Evolution Semantic Space)"
type: entity
tags:
  - time-series
  - multimodal
  - llm
  - forecasting
  - semantic-primitives
  - non-fusion-guidance
  - iclr-2026
created: 2026-08-01
last_updated: 2026-08-01
source_count: 1
confidence: medium
status: active
---

# TESS

**TESS**（Temporal Evolution Semantic Space）是 Li et al. 提出的文本-时序多模态预测框架（标注 ICLR 2026，arXiv:2603.12664v2）。核心做法：在文本空间与数值序列空间之间放一个中间瓶颈，把文本压缩成可数值验证的离散时间演化原语，再注入时序模型[^src-tess]。

## 问题

文本对事件的描述（"significant rise"）是定性、隐式、时间弱锚定的；预测模型需要定量、显式、时间对齐的信号。直接融合文本 embedding 有两个可测的失败模式（半合成实验定位）：

1. 注意力被冗余 token 分散（焦点比 $R_t=\log(\bar\alpha_{sig}/\bar\alpha_{red})$ 多为负值）。
2. 即使删光冗余，文本语义仍难解码成数值信号（Signal-Only 显著劣于 Numerical）。

## 两阶段结构

1. **文本 → 原语**：冻结 LLM 做四类 [[temporal-semantic-primitives|时间演化原语]] 的小候选集分类（mean shift / volatility / shape / lag-decay）。每类以 top-1/top-2 log 概率 margin 为不确定度信号，经门控 $g_{t,k}$ 过滤。门控的监督标签来自原语数值可验证性：$\psi_k(Y_t)$ 由未来窗口唯一确定，BCE 训练"判对与否"。
2. **原语 → 预测**：门控后的类别 embedding 作 prefix token，与 PatchTST 的 patch embedding 拼接（$Z^{(0)}=[P;E_{patch}]$），在全部注意力层中条件化，MLP 头出预测。总损失 $L=L_{fcst}+\lambda L_{gate}$，端到端训练、LLM 冻结[^src-tess]。

## 理论依据

- 定理 4.1：语义充分性（$\hat Y_t \perp X_{text}\mid(P_t,X_{time})$）下，原语瓶颈不损预测互信息、降低对文本的依赖、泛化误差不增。
- 定理 A.5：原语提取错误对预测的影响按门控值平方 $g_{t,k}^2$ 衰减。
- 定理 A.6：假设空间复杂度从 token 级 $\sqrt{\log|A_T|/n}$ 降为原语级 $\sqrt{M/n}$（$M=\prod_k |V_k|$）[^src-tess]。

## 结果

四个数据集（Bitcoin、FNSPID、Electricity、Environment）。Bitcoin 相对最强基线 NewsForecasting：MAE/MSE/RMSE +18.2%/+29.1%/+15.8%；FNSPID 相对 TimesNet：+3.3%/+20.0%/+9.9%；Electricity 全指标最优；Environment 次优。非平稳子集 MSE 降 21–52%（vs 多模态基线）[^src-tess]。

消融：TESS 组件贡献远大于门控（去 TESS：+46.2%/29.4%/22.8% MSE；去 gating：+3.7%/2.6%/7.5%）；mean shift 原语单独移除即 +33%；gating 权重与提取正确性相关（正确样本中位 0.65–0.78，错误 0.21–0.40）[^src-tess]。

## 与其他方法的关系

| 方法 | 与 TESS 的关系 | 机制差异 |
|------|--------------|---------|
| [[timi|TiMi]] | 同属 [[non-fusion-guidance|Non-Fusion Guidance]] | TiMi 用 MMoE 连续路由注入自由文本知识；TESS 用离散原语分类 + 置信门控 + prefix 条件化 |
| [[constrained-text-fusion|CFA]] | 同诊断 naive 融合有害 | CFA 在特征层做低秩受控残差注入；TESS 在语义层做离散瓶颈，文本 token 不进预测器 |
| [[vot|VoT]] | 同为事件驱动文本推理 | VoT 生成数值预测 + 推理链并做多级对齐；TESS 只做原语分类，不做任何对齐 |
| [[time-llm|Time-LLM]] | 方向相反（基线对比） | Time-LLM 把 TS reprogram 进 LLM 空间；TESS 把文本压成原语注入 TS 模型空间 |
| [[source-from-news-to-forecast|NewsForecasting]] | Bitcoin 上的最强基线 | NewsForecasting 智能体迭代过滤 + LoRA 微调 LLM 自回归；TESS 单次冻结 LLM 分类 + PatchTST |
| [[tats|TaTS]] | 同为文本辅助输入 | TaTS 整段文本 embedding 拼接（无显式门控，但有池化+MLP+联合训练）；TESS 离散原语 + 门控（有过滤）。判定机制：TT-Wasserstein 语料级预判（训练前、谱距离）vs 置信门控实例级在线过滤（数值可验证 BCE）。两篇在 FNSPID/Bitcoin/Electricity 评测重叠，结论差异源于融合形态而非数据 |
| [[patchtst|PatchTST]] | backbone | 仅两处改动：语义 prefix token + 门控 BCE 监督 |

## 相关页面

- [[source-tess]] · [[temporal-semantic-primitives]] · [[non-fusion-guidance]] · [[timi]] · [[constrained-text-fusion]] · [[vot]] · [[time-llm]] · [[tats]] · [[patchtst]] · [[multimodal-time-series-forecasting]]

[^src-tess]: [[source-tess]]
