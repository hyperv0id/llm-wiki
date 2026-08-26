---
title: "LoFT-LLM"
type: entity
tags:
  - time-series-forecasting
  - frequency-domain
  - llm-calibration
  - few-shot
  - kdd-2026
created: 2026-08-05
last_updated: 2026-08-05
source_count: 1
confidence: medium
status: active
---

# LoFT-LLM

**LoFT-LLM（Low-Frequency Time-series Forecasting with Large Language Models）** 是论文提出的一条面向数据稀缺场景的时序预测流水线，将频域低频学习与 LLM 语义校准组合为三阶段结构（KDD 2026, arXiv:2512.20002v3）[^src-loft-llm]。论文将 LLM 的角色定位为语义校准器（semantic calibrator）：频率模块先给出趋势与高频残差，LLM 再结合辅助领域知识做修正，而非让 LLM 直接承担主预测[^src-loft-llm]。

## 问题与设计动机

论文针对两类现实约束[^src-loft-llm]：

1. **数据稀缺下的高频噪声**：真实金融、能源场景训练样本少，全长度时间窗口的监督信号被高频噪声主导，低频长期趋势难被模型学出。论文依据梯度网络的谱偏置（spectral bias）——低频分量通常更快、用更少样本学到——主张显式分离并监督低频，把高频残留当作独立建模任务。
2. **辅助变量语义被剥离**：页面访问量、利率、云量等辅助信号被归一化成数值向量后，领域语义丢失，few-shot 下尤其严重。论文用 LLM 重新接入这些语义。

## 三阶段机制

1. **PLFM（Patch Low-Frequency forecasting Module）**：低频预训练。目标 Y 经低通滤波（LPF）+ FFT 得频域监督；输入 X 经重叠 patching + 逐 patch DFT（局部谱建模）；双 MLP 拟合实部/虚部；FALoss（与真值傅里叶系数的 L1 距离）训练。与 [[source-fredf|FreDF]] 的频域对齐目标同源，但 FreDF 是模型无关的训练范式，PLFM 是专门的频域低频预测模块[^src-loft-llm]。
2. **Residual Learning**：高频残差。PLFM 冻结后，输入经高通滤波（HPF），轻量骨干（论文采用 [[itransformer|iTransformer]]）拟合高频变化，残差与低频预测相加，用同一 FALoss 对齐[^src-loft-llm]。
3. **LLM Calibration**：语义校准。Qwen3-8B 经 QLoRA 监督微调；PromptBuilder 将低频 token、残差 token、辅助变量与领域知识打包为自然语言 prompt；LLM 输出数值列表作为最终预测。prompt 模板由 ChatGPT-4o 辅助生成，训练用 prompt-to-sequence 对齐（历史 prompt 拼接未来序列）[^src-loft-llm]。

## 实验证据

两个真实数据集（FundAR 日频基金流、Solar 小时级光伏发电），70/10/20 划分，对照 12 个基线，三次运行取平均[^src-loft-llm]：

- **全量数据**：FundAR 30 项评估 26 项最优、平均 MAE 较最佳基线降 26.53%；Solar 30 项中 27 项最优、平均 MAE 降 15.42%（论文表 2、3）
- **few-shot**：FundAR 用约 10% 数据、Solar 用最近 7 天，平均 MAE 较最佳基线降幅超过 40%（论文表 4）
- **消融**（论文表 5）：FundAR 上去掉 LLM 退化更大（作者归因于金融辅助信号的作用），Solar 上去掉频率模块退化更大（作者归因于强周期性与气象依赖）
- **低频有效性**（论文表 6）：FreTS/TimesNet/iTransformer 嵌入流水线后，最低 40% 频谱 MAE 平均改善约 23%

## 与其他方法的对比

| 维度 | LoFT-LLM | [[time-llm|Time-LLM]] | [[source-fstllm|FSTLLM]] | [[source-fredf|FreDF]] |
|------|----------|------------------------|--------------------------|-------------------------|
| LLM 角色 | 后端语义校准器 | 前端提示/重编程（reprogramming） | few-shot LLM 时序预测 | 无 LLM |
| 频域机制 | FALoss + PLFM 频域低频监督 | 无 | 无 | 频域标签对齐（模型无关训练目标） |
| 面向场景 | 数据稀缺 + 辅助变量 | 单变量/多变量预测 | few-shot | 多步直接预测 |

## 相关页面

- [[source-loft-llm]] — 源文件摘要
- [[patch-low-frequency-forecasting]] — PLFM 与 FALoss 技术页
- [[source-fredf]] / [[fredf]] — 频域对齐训练目标 FreDF
- [[itransformer]] — 残差学习骨干
- [[time-llm]] — LLM 作前端提示的对比
- [[source-fstllm]] — few-shot LLM 时序预测
- [[source-frets]] — 频域 MLP 方法 FreTS
- [[loft]] — 同名缩写 LOFT 的另一篇 KDD 2026 论文（低秩先验一致性流匹配交通插补），与本条目无关
- [[generative-time-series-forecasting]] — 生成式时序预测概念

[^src-loft-llm]: [[source-loft-llm]]
