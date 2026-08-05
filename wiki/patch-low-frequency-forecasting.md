---
title: "Patch Low-Frequency Forecasting (PLFM)"
type: technique
tags:
  - time-series-forecasting
  - frequency-domain
  - low-frequency-learning
  - training-loss
  - kdd-2026
created: 2026-08-05
last_updated: 2026-08-05
source_count: 1
confidence: medium
status: active
---

# Patch Low-Frequency Forecasting (PLFM)

**Patch Low-Frequency forecasting Module（PLFM）** 是 [[loft-llm|LoFT-LLM]] 中用于显式学习时序低频趋势的频域监督模块，配套 Frequency Alignment Loss（FALoss）训练[^src-loft-llm]。

## 动机

主流时序预测模型用全长度预测窗口做监督信号。真实数据的窗口包含大量高频成分（振荡、异常、噪声），论文认为这会掩盖低频依赖，且不利于数据稀缺时的泛化[^src-loft-llm]。由于低频分量在频谱上呈显著峰、全局依赖强，论文选择从频域出发，先滤掉高频再监督低频，把「学趋势」从「学噪声」中分离出来。

## 机制

### FALoss（Frequency Alignment Loss）

对预测窗口 Y 与模型输出 Yᵒ，逐通道做离散傅里叶变换（DFT），取对应傅里叶系数之差的 L1 均值[^src-loft-llm]：

$$\mathcal{L}_{FA}(Y, Y^o) = \frac{1}{L \times C} \left\| \hat{Y} - \hat{Y}^o \right\|_1$$

其中 $\hat{Y}$、$\hat{Y}^o$ 分别是真值与预测的傅里叶系数。该目标受 [[source-fredf|FreDF]] 的频域对齐启发，与逐点 MSE/MAE 相比更强调频域一致性，有利于低频长期依赖的建模[^src-loft-llm]。

### 局部谱建模（STFT 式）

PLFM 不在整条序列上做 FFT，而是采用短时傅里叶变换（Short-Time Fourier Transform）式的局部谱建模[^src-loft-llm]：

1. 目标 Y 先经低通滤波（LPF）得到低频监督 $\tilde{Y}$，再 FFT 得频域目标 $\hat{Y}_p$
2. 输入 X 用重叠 patching（patch 长 P、步长 S，S < P），逐 patch 做 DFT 后拼接
3. 双两层 MLP 分别拟合频谱的实部与虚部，堆叠得到复数输出
4. 训练用 FALoss；推理时对预测谱做 IFFT 还原为低频 token $Y^o_{low}$

重叠 patch 保留时间局部结构，比整段 FFT 对局部低频峰更敏感。

## 实验证据

在 [[loft-llm|LoFT-LLM]] 的实验中，PLFM 的效果由消融与迁移验证支撑[^src-loft-llm]：

- **消融**（论文表 5）：去掉频率模块（PLFM + 残差学习退化为普通 MLP，并剔除 prompt 中的频域内容）后，Solar 数据集 MAE 从 0.030 升至 0.043，RMSE 从 0.066 升至 0.089（FundAR 上 MAE 从 0.661 升至 0.692）。Solar 上频率模块的退化大于 LLM 模块，作者归因于太阳能的强周期性与气象依赖。
- **低频有效性**（论文表 6）：把 FreTS、TimesNet、iTransformer 分别嵌入 LoFT-LLM 流水线，在最低 40% 频率谱段上的 MAE 平均改善约 23%。
- **理论**（附录 B）：Theorem 1 用 Parseval-Plancherel 恒等式说明时域能量等于频域能量；Theorem 2 论证优化频域系数差异可降低时域 MAE。

## 与 FreDF 的区分

[[source-fredf|FreDF]] 是模型无关的多步直接预测训练范式——任意骨干加频域损失；PLFM 是专门为低频学习设计的模块——重叠 patch 谱 + 低通监督 + 双 MLP 频域网络，且是 LoFT-LLM 流水线的一段[^src-loft-llm]。两者共享「频域对齐」的思想，但作用层面不同。

## 相关页面

- [[loft-llm]] — 使用 PLFM 的流水线模型
- [[source-fredf]] / [[fredf]] — 频域对齐训练目标 FreDF
- [[itransformer]] — 流水线中的高频残差骨干
- [[source-frets]] — 频域 MLP 方法 FreTS

[^src-loft-llm]: [[source-loft-llm]]
