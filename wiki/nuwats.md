---
title: "NuwaTS"
type: entity
tags:
  - time-series
  - data-imputation
  - foundation-model
  - pretrained-language-model
  - channel-independence
  - contrastive-learning
created: 2026-06-08
last_updated: 2026-06-08
source_count: 2
confidence: high
status: active
---

# NuwaTS

**NuwaTS**（取名自女娲补天）是 Cheng et al. (Sichuan University / HKUST-GZ / Squirrel Ai, arXiv 2024) 提出的**通用不完整时间序列插补基础模型**——将预训练语言模型（PLM，默认 GPT-2 前 6 层）改造为一个"一统天下"（one-for-all）的插补器，训练一次即可为任意领域、任意变量、任意缺失模式的不完整序列填补缺失值[^src-nuwats]。它是论文作者所称的**首个能够跨域泛化的时间序列插补基础模型**。

## 设计动机

传统插补模型为特定缺失模式/变量/领域定制，且评测沿用**时间维度**的 train/val/test 划分——测试集只是训练变量的未来观测，无法检验模型对**未见变量**或**未见领域**的泛化能力[^src-nuwats]。NuwaTS 借鉴 SAM、GPT 等基础模型的成功，转而追求 cross-variable 与 cross-domain 泛化，并配套提出 [[variable-wise-partitioning|变量维度划分基准]]。

## 架构

模型参数 Φ 用 PLM 权重初始化，仅取前 6 层（兼顾算力，使其可部署到边缘设备）[^src-nuwats]。输入处理与 token 设计如下：

### 实例归一化 + Patching

对每个变量先做可逆实例归一化（[[instance-normalization|RevIN]]，缺失值置零），消除跨域幅度/分布差异；再切分为**非重叠 patch**，经共享可学习线性投影嵌入到隐空间 $Z_{i,(p)} \in \mathbb{R}^{D \times N}$[^src-nuwats]。

### 三类专用嵌入（核心创新）

NuwaTS 摒弃 [[time-llm|Time-LLM]] 式的硬文本提示，转而把统计量与缺失信息直接编码为可学习嵌入[^src-nuwats]：

| 嵌入 | 含义 | 形式 |
|------|------|------|
| **统计嵌入** Statistical | 最小/中位/最大值、趋势，分**整条变量**（series-wise $z_{i,(v_g)}$）与**单 patch**（patch-wise $Z_{i,(v_p)}$）两级 | 共享线性投影 |
| **缺失嵌入** Missing | 捕获每个 patch 的缺失率，乘以该 patch 的 mask ratio $r_i$ | 可学习参数 $z_{i,(m)}$ |
| **领域嵌入** Domain-Specific | 学习领域知识，置于 patch 嵌入之前作为前缀 | 可学习参数 $k \in \mathbb{R}^D$ |

patch 级融合：$E_{i,(p)} = Z_{i,(p)} + Z_{i,(v_p)} + z_{i,(m)} \times r_i$；最终输入序列 $E_i = [k,\; z_{i,(v_g)},\; E_{i,(p)}]$[^src-nuwats]。

### 缺失模式对比学习

为增强对不同缺失模式的适应性，NuwaTS 引入[[contrastive-learning|对比学习]]模块[^src-nuwats]：对每个输入 $x_i$ 生成**两个不同 mask ratio** 的掩码视图送入 PLM，将同一 patch 在不同掩码下的表示视为**正样本对**、其他 patch 与其他序列的表示视为负样本，用 InfoNCE（双线性内积 $q^T W k_+$）配合 MSE 联合优化。这使表示**对缺失模式不变**（mask-invariant）。

### 输出层

经 PLM 后**丢弃**领域嵌入与变量级统计嵌入（前缀仅参与因果注意力计算，不进入输出），保留 $N$ 个 patch 表示，展平后线性映射回原维度得到插补结果 $o_i \in \mathbb{R}^L$[^src-nuwats]。

## 即插即用领域微调

见 [[plug-and-play-prefix-tuning]]。借鉴 **P-tuning v2**：领域迁移层（2 层 MLP）把 $k$ 映射为 $\hat{K}$，与随机初始化的连续 prompt $P$ 组合成每层前缀 $[\text{Key}_p, \text{Value}_p] = P + \beta\hat{K}$（$\beta=0.01$），注入冻结 PLM 的每一层[^src-nuwats]。前缀极轻量——GPT-2 下 <100KB（整模型 331.77MB），移除前缀即退回 one-for-all 模型，故称"即插即用"。还可设计 inter-variable 微调网络，用轻量 Transformer 把变量相关性编入前缀（用于预测任务）。

## 四个版本

| 版本 | 训练数据 | 用途 |
|------|---------|------|
| (a) Specific | 单一领域 | 域内插补基线 |
| (b) One-for-all | 融合数据集（17.6M 段，ETT/Weather/ECL/PEMS） | 通用一统模型 |
| (c) Fine-tuned | (b) 基础上域内微调 | 域特化 |
| (d) Cross-domain | 仅 LargeST（100.1M 段，仅用 2019 数据） | 零样本跨域验证 |

## 关键实验结果

- **主结果**：10 个数据集、9 档缺失率（0.1–0.9），one-for-all NuwaTS 在几乎所有缺失率上**超越域特定 SOTA**（SAITS、BRITS、TimesNet、PatchTST、GPT4TS）[^src-nuwats]。
- **缩放律**：在多域融合数据上训练后，NuwaTS 与 PatchTST(one-for-all) 的泛化能力均进一步提升，支持插补任务存在 scaling law[^src-nuwats]。
- **零样本跨域**：作为 [[channel-independence|channel-independent]] 方法，可推理到变量数不同的数据集（LargeST⇒ECL/Weather），全面优于 PatchTST 零样本；而 channel-dependent 的 TimesNet/GPT4TS 只能在同变量数数据集间零样本[^src-nuwats]。
- **少样本**：在 ETT 上仅用 **10% 数据微调即达 100% 数据的效果**；1% 数据也有效，对数据稀缺领域尤为有用[^src-nuwats]。
- **连续缺失**：在随机缺失上训练的模型直接测试连续缺失，仍鲁棒[^src-nuwats]。
- **真实数据集**：北京多站点空气质量上，NuwaTS 零样本 RMSE 0.370 即超越 BRITS(0.525)、GP-VAE(0.614)，逼近 SAITS(0.518)[^src-nuwats]。
- **赋能预测**：先用 NuwaTS 插补不完整训练数据再训 TimesNet，预测性能优于用 PatchTST 插补；NuwaTS 也可通过**追加 masked padding token** 直接转为预测模型[^src-nuwats]。

## 消融与发现

- 统计嵌入、缺失嵌入、对比学习三者缺一不可；冻结 backbone 表现最差[^src-nuwats]。
- **不加载 NLP 预训练权重（from scratch）会显著削弱零样本跨域能力**——证明跨模态预训练有意义（NLP 任务训练有益于时序任务）[^src-nuwats]。
- 对基础模型而言，**数据量比专门模块引入的归纳偏置更关键**（专用 token 与对比学习在小数据集 ETTh1 上增益更明显，在超大 LargeST 上增益弱）[^src-nuwats]。

> [!note] 与 T1 的范式对照
> [[t1|T1]] (ICLR 2026) 给出相反取向的证据：它主张"鲁棒插补受益于**任务对齐架构**（专门化的时间 CNN + 跨变量注意力 + channel-head 绑定）"，无需预训练即可用单一超参在 11 个数据集上达 SOTA、平均 MSE 较次优降 46%[^src-t1]。NuwaTS 走"复用 NLP 权重 + 大规模多域预训练"的基础模型路线（数据/scale 优先），T1 走"精巧归纳偏置"的 bespoke 架构路线——二者是当前时序插补的两种范式取向。
- backbone：**GPT-2 > BERT > LLaMA2**（LLaMA2 参数太大、特征稀疏、速度慢，不利边缘部署；BERT 双向注意力略逊于 GPT-2 的因果注意力，因时序具因果性）[^src-nuwats]。
- **简单线性嵌入 > 文本对齐**（[[patch-reprogramming|Patch Reprogramming]]）：因不完整 patch 缺失比例高且位置多变，模态对齐难以表征复杂缺失序列（表 14）[^src-nuwats]。

## 局限

模型在固定长度 96 的片段上训练；处理**更长片段**或**整段完全缺失**时可能需进一步微调——这是作者列出的未来方向[^src-nuwats]。

## 关联页面

- [[variable-wise-partitioning]] — NuwaTS 提出的变量维度划分基准
- [[plug-and-play-prefix-tuning]] — NuwaTS 的即插即用领域前缀微调
- [[contrastive-learning]] — NuwaTS 的缺失模式不变对比学习
- [[channel-independence]] — NuwaTS 的 CI 设计，支撑跨变量零样本
- [[instance-normalization]] — NuwaTS 输入端的 RevIN
- [[time-llm]] — 同为 PLM-for-TS，但 NuwaTS 实验证明文本对齐对不完整序列不如线性嵌入
- [[patch-reprogramming]] — Time-LLM 的文本对齐机制，被 NuwaTS 证明不适合插补
- [[model-reprogramming]] — PLM 跨域重用范式（NuwaTS 微调 backbone，非纯重编程）
- [[imputeformer]] — ImputeFormer 在未来工作中设想的"跨域插补基础模型"，由 NuwaTS 实现
- [[t1]] — T1，任务对齐 CNN-Transformer 插补架构（与 NuwaTS 的 scale-优先路线形成范式对照）
- [[csdi]] — 扩散式插补 SOTA，NuwaTS 的对比基线之一
- [[chronos]] — 从零训练的时序语言模型（NuwaTS 复用 NLP 权重而非从零训练）
- [[missing-not-at-random]] — 缺失机制谱系（NuwaTS 假设随机缺失、忽略缺失过程；PRDIM 处理 MNAR）

[^src-nuwats]: [[source-nuwats]]
[^src-t1]: [[source-t1]]
