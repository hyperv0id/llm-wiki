---
title: "Self-Supervised Imputation Training"
type: concept
tags:
  - data-imputation
  - self-supervised-learning
  - time-series
  - diffusion-models
created: 2026-07-14
last_updated: 2026-08-26
source_count: 2
confidence: medium
status: active
---

# Self-Supervised Imputation Training

**Self-Supervised Imputation Training** 是一种训练插补模型的方法论：当训练数据本身不含明确"缺失值→真值"配对标签时，从观测值中人工构造伪缺失目标与伪条件观测，形成自监督训练信号。该范式由 [[csdi|CSDI]] 在时间序列扩散插补中提出[^src-csdi]，灵感来源于 BERT 的掩码语言建模（Masked Language Modeling）。

## 动机

时间序列缺失值插补面临一个结构性困难：真实世界数据中，我们不知道缺失位置的 ground-truth 值。传统的解决方式是使用完整数据人工制造缺失，但这需要假设存在大量完整样本——而很多场景下（如医疗 ICU 数据 ~80% 缺失率），完整样本极少。自监督训练将问题反过来：从已有观测值中"隐藏"一部分，让模型去恢复它们。

## CSDI 的实现

CSDI 将训练样本 $x_0$ 的观测值分离为两部分[^src-csdi]：

- **伪插补目标** $x_0^{\text{ta}}$：被选为"假装缺失"的观测值子集
- **伪条件观测** $x_0^{\text{co}}$：剩余观测值，作为条件输入

在伪目标上加噪 $x_t^{\text{ta}} = \sqrt{\bar{\alpha}_t} x_0^{\text{ta}} + \sqrt{1-\bar{\alpha}_t} \epsilon$，训练去噪网络 $\epsilon_\theta(x_t^{\text{ta}}, t \mid x_0^{\text{co}})$ 恢复原始值。推理时反转：所有观测值作为条件 $x_0^{\text{co}}$，所有真实缺失位置作为目标 $x_0^{\text{ta}}$。

### 四种目标选择策略

| 策略 | 适用场景 | 方法 |
|------|---------|------|
| **Random** | 未知缺失模式 | 从观测值中随机选取 0%–100% 作为目标 |
| **Historical** | 结构性缺失（如传感器连续故障块） | 借用训练集中另一样本的缺失模式作为当前样本的目标模式 |
| **Mix** | 部分结构化缺失 | Random 与 Historical 各以 50% 概率抽取 |
| **Test pattern** | 已知测试缺失模式（如预测任务） | 直接使用测试集的缺失模式作为目标 |

Random 策略的采样比例从 $[0\%, 100\%]$ 均匀抽取，使模型适应测试时的各种缺失率。Historical 策略利用训练集中已有的缺失模式（如连续缺失块），帮助模型学习结构化缺失下的条件分布。Mix 策略在两个极端之间取平衡，既保留 Random 的泛化能力，又利用 Historical 的结构化先验。

## 后续采用

该训练范式被后续扩散插补方法广泛继承：

- [[pristi|PriSTI]]：在 CSDI 自监督框架上添加线性插值增强的条件先验和先验引导注意力，将条件信息使用从"混合输入"升级为"先提取先验、后引导去噪"
- [[cofill|CoFILL]]：沿用 CSDI 的自监督掩码训练，扩展为时域+频域双流架构
- [[fence|FENCE]]：采用两阶段训练（先训练无条件模型学习先验，再以该模型为初始化微调条件插补），将推理时的固定 CFG 引导尺度升级为基于后验似然的动态反馈引导[^src-fence]
- [[lscd|LSCD]]：继承 CSDI 的自监督框架，扩展为频谱条件化——用可微 Lomb–Scargle 周期图替代 FFT
- [[sadi|SADI]]：在自监督训练基础上引入 partial blackout 缺失模式，统一随机缺失、插值、完全停电和预测

## 与相关概念的关系

- **BERT 掩码语言建模**：核心灵感来源——"从已知中人造未知"。区别在于 BERT 掩码 token，CSDI 掩码连续值
- **[[masked-autoencoder|MAE 掩码自编码器]]**：同为掩码自监督，MAE 掩码空间区域（图像 patch），CSDI 掩码时间序列观测值
- **[[score-matching|得分匹配]]**：自监督目标的形式化工具——CSDI 在伪目标上使用 denoising score matching

[^src-csdi]: [[source-csdi]]
[^src-fence]: [[source-fence]]
