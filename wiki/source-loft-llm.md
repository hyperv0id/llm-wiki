---
title: "LoFT-LLM: Low-Frequency Time-Series Forecasting with Large Language Models"
type: source-summary
tags:
  - time-series-forecasting
  - frequency-domain
  - low-frequency-learning
  - llm-calibration
  - few-shot
  - kdd-2026
created: 2026-08-05
last_updated: 2026-08-05
source_count: 0
confidence: medium
status: active
---

# LoFT-LLM: Low-Frequency Time-Series Forecasting with Large Language Models

**You, Yang, Xie, Wu, Li, Li, Wang, Xu, Zheng & Chen (2026), KDD 2026, arXiv:2512.20002v3**

完整论文（12 页）：`raw/2512.20002.pdf`。代码：https://github.com/yjcGitHub0/LoFT-LLM 。作者主要来自哈尔滨工业大学（深圳），第一作者 Jiacheng You；通讯作者之一 Xinyang Chen。

## 核心论题

论文提出 LoFT-LLM，将频域低频学习与大语言模型语义校准组合成一条三阶段流水线，面向数据稀缺场景的时序预测。论文认为现有深度预测模型存在两个问题：其一，用全长度时间窗口做监督信号，高频噪声会掩盖长期趋势；其二，含领域信息的辅助变量（如页面访问量、利率、云量）被当作普通数值向量归一化后直接输入，语义被剥离，尤其在 few-shot 场景下未能利用。LoFT-LLM 将 LLM 定位为语义校准器（semantic calibrator），而非主预测器。

## 方法

三阶段训练流程，推理时三段冻结、直接组合：

1. **Patch Low-Frequency forecasting Module（PLFM）**：对目标序列 Y 先做低通滤波（LPF），再做 FFT 得到频域监督信号 $\hat{\mathbf{Y}}_p$；输入 X 用重叠 patching（patch 长 P、步长 S）后逐 patch 做 DFT（短时傅里叶变换式局部谱建模）。PLFM 主体是两个双层 MLP，分别拟合频谱的实部与虚部。训练用 Frequency Alignment Loss（FALoss，受 [[source-fredf|FreDF]] 启发）：预测与真值傅里叶系数之差的 L1 均值。
2. **Residual Learning**：PLFM 权重冻结后，对输入 X 做高通滤波（HPF），用轻量骨干（论文采用 [[itransformer|iTransformer]]）拟合高频残差。训练时残差预测加到低频预测上，用同一个 FALoss 对齐。
3. **LLM Calibration**：用 Qwen3-8B 作为骨干 LLM，QLoRA 监督微调。prompt（PromptBuilder 构造）打包三部分——低频 token、残差 token、辅助变量与领域知识，LLM 直接输出数值列表作为最终校准预测。prompt 模板由 ChatGPT-4o 辅助生成；训练用 prompt-to-sequence 对齐（历史 prompt 拼未来序列）。

## 实验

两个真实数据集，70/10/20 划分，MAE/RMSE/MAPE，三次运行取平均（附录 C 报告标准差）：

| 数据集 | 频率 | 时间范围 | 点数 | 目标 | 领域 |
|--------|------|----------|------|------|------|
| FundAR | 日频 | 2021-01-04 至 2022-11-09 | 675 | 申购与赎回量 | 金融（2024 阿里天池基金流竞赛） |
| Solar | 小时级 | 2012-04-01 至 2013-04-01 | 8,760 | 发电功率 | 能源（GEFCom 2014 Region 1） |

对照 12 个基线（Transformer、DLinear、PatchTST、FITS、FreTS、TimesNet、iTransformer、TimeXer、FreDF、TimeKAN、GPT4TS、TimeLLM），论文报告：

- **全量数据**：FundAR 上 30 项评估中 26 项取得最优，平均 MAE 较最佳基线降 26.53%；Solar 上 30 项中 27 项最优，平均 MAE 降 15.42%（表 2、表 3）
- **few-shot**：FundAR 仅用约 10%（约 60 步）训练数据，Solar 用最近 7 天（168 步）；相对各任务最佳基线，平均 MAE 降幅超过 40%（表 4）
- **消融**（表 5）：去掉频率模块（退化为普通 MLP + 剔除 prompt 频域内容）或去掉 LLM（直接求和低/高频输出）均导致明显下降；两个数据集上各模块的相对重要性相反——FundAR 上去掉 LLM 的退化更大（作者归因于金融辅助信号如页面 UV、利率的关键作用），Solar 上去掉频率模块退化更大（作者归因于太阳能的强周期性与气象依赖）
- **低频有效性**（表 6）：把 FreTS、TimesNet、iTransformer 分别嵌入 LoFT-LLM 流水线，在最低 40% 频谱上 MAE 平均改善约 23%

理论部分（附录 B）：Theorem 1 依据 Parseval-Plancherel 恒等式说明时域与频域能量守恒；Theorem 2 论证优化频域系数差异可以降低时域 MAE。附录 E 报告 patch length 消融（8 至 16 区间结果接近，12 为 FundAR 上表现较好的选择之一）。

## 论文自述与课程评估

论文未单设局限性章节；以下分开标注：

- **论文自述**：prompt 模板由 ChatGPT-4o 辅助生成；LLM 微调采用 QLoRA 以降低显存与计算开销。
- **本课程评估**：实验只在两个数据集上进行（一个金融、一个能源），样本量小（FundAR 仅 675 个时间点）；论文未讨论在其他领域（如交通、气象大样本）或面对概念漂移时的表现。论文将 LLM 定位为校准器、数值输出由 prompt 打包的模型结果主导，这种「LLM 在后端做语义修正」的范式与 [[time-llm|Time-LLM]] 的「LLM 作前端推理」、[[source-fstllm|FSTLLM]] 的 few-shot 设计之间的关系，论文未展开讨论。

## 关键术语

- **FALoss（Frequency Alignment Loss）**：预测与真值傅里叶系数之间的 L1 距离，受 FreDF 的频域对齐启发
- **PLFM（Patch Low-Frequency forecasting Module）**：频域监督的低频趋势提取模块
- **Semantic Calibrator**：论文对 LLM 角色的定位——注入领域知识、修正数值模块的系统性偏差

---
