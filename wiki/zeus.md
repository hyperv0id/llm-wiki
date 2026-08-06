---
title: "Zeus"
type: entity
tags:
  - time-series-foundation-model
  - tuning-free
  - multi-task
  - masked-modeling
  - multi-scale
  - icml-2026
created: 2026-08-06
last_updated: 2026-08-06
source_count: 1
confidence: medium
status: active
---

# Zeus

**Zeus** 是论文提出的 unified tuning-free 时序基础模型（Time Series Foundation Model, TSFM），在点预测、概率预测、插补、异常检测、分类五种下游任务上无需任何任务特定微调即取得有竞争力的性能（ICML 2026, arXiv:2607.01918v1）[^src-2607-01918]。论文将 tuning-free 定义为 inference 时不调参、不重训任何模型参数[^src-2607-01918]。论文声称（非外部定论）：据其所知，ZEUS 是第一个在五个下游任务上无需任务特定适配即达到有竞争力性能的 TSFM[^src-2607-01918]。

## 问题与动机

论文指出现有 TSFM 的两个根本困境[^src-2607-01918]：

1. **架构困境：粒度 vs 可扩展性**。patch-wise tokenization 语义密度高、计算省，但牺牲逐点细节，损害重建类任务（插补、异常检测）；point-wise tokenization 保留细粒度，但信息密度低、长序列计算开销大。
2. **训练困境：归纳偏置异质**。预测需要外推（extrapolation），插补和异常检测需要插值（interpolation），分类需要全局抽象（global abstraction）；单一 BERT 式掩码重建或 GPT 式自回归目标无法同时赋予所有能力。

## 机制

### 多尺度 encoder-only 架构

point-wise tokenization，U 形（fine-to-coarse-to-fine）对称下采样-上采样层级[^src-2607-01918]：

- 尺度配置 scales [1, 8, 32, 8, 1]，层数 [1, 3, 3, 3, 2]，hidden [384, 768, 768, 768, 384]，heads [6, 12, 12, 12, 6]，intermediate [1536, 3072, 3072, 3072, 1536]，约 100M 参数，最大上下文 4096[^src-2607-01918]
- 下采样用 pooling + 可学习线性投影（论文式 2-3），上采样用 unpooling 并加对应尺度的残差跳跃连接（式 4）[^src-2607-01918]
- 细尺度用轻量 Transformer block 保局部细节，粗尺度用更深更宽的 block 建模长程依赖[^src-2607-01918]
- Transformer block：MHA + RoPE + gated FFN（GLU）+ RMSNorm + pre-LN，FlashAttention v2[^src-2607-01918]
- gated embedding：$h_t = W_r x_t + W_d \sigma(W_g x_t) \odot W_u x_t$（式 1）；引入可学习 [MASK] 与 [PAD] token[^src-2607-01918]
- quantile head：每时间步输出 9 个分位数 Q={0.1, ..., 0.9}，实现概率重建；点估计 = 各分位数平均[^src-2607-01918]
- channel-independent 策略处理多变量（引用 [[patchtst|PatchTST]]）；用 instance normalization（RevIN，引用 Kim et al. 2021）去尺度变化[^src-2607-01918]

论文的表征分析（图 7，ETTm1 样本）显示：细尺度对局部变化与极值敏感（捕捉负脉冲），中尺度条纹反映周期结构，粗尺度刻画全局模式变化与上下文异常[^src-2607-01918]。

### MOTM 预训练目标

MOTM（Multi-Objective Temporal Masking）是三级 pipeline：先采掩码比例，再选时序范围，最后采样掩码策略[^src-2607-01918]。掩码比例 p ~ U(0, 0.5)，期望 0.25；时序范围按序列长度分段均匀采样（0.2 概率 [64, 512]，0.2 概率 [513, 2048]，0.6 概率 [2049, 4096]），随机裁剪并 padding 到 4096；四种掩码策略加混合[^src-2607-01918]：

- **Predictive Mask**：掩掉序列尾部 ⌊Tp⌋ 步，训练外推（预测）能力
- **Point Mask**：随机掩单个时间步，训练逐点插值与局部连续性
- **Multi-Block Mask**：采样多个连续块（块长 ℓk ~ U(1, 24)，总长≈⌊Tp⌋，均匀分布而非语言建模常用的 Poisson），训练结构化缺失下的插值（受 span corruption 启发）
- **Single-Block Mask**：任意位置移除一个长连续段，训练全局一致性（服务分类与上下文异常检测）
- **Mixed Mask**：简单（multi-block/point）与困难（predictive/single-block）组合

训练目标：仅在掩码位置计算 quantile loss（pinball loss，论文式 5）[^src-2607-01918]。

> [!note] 同名消歧
> Zeus 的 MOTM 是掩码预训练目标，详见 [[multi-objective-temporal-masking]]；与仓库中 [[motm|MoTM（Mixture of TimeFlow Models）]]（时间索引插补模型）是完全不同的东西。

### 任务统一 formulation

五种任务统一映射到掩码重建框架上[^src-2607-01918]：

- **预测**：对应 Predictive Mask——掩掉序列尾部 ⌊Tp⌋ 步，在掩码位置重建未来值
- **插补**：把缺失位置视为掩码位置重建（Point / Multi-Block / Single-Block Mask 对应逐点、块状与长段缺失）
- **异常检测**：掩目标窗口并用前后文重建，重建误差（默认 MAE；含脉冲模式时用 relative MAE 归一化）作异常分数
- **分类**：表示取自倒数第二尺度（s4=8）或最粗尺度（s3=32），全局池化（默认 max pool）→ flatten → 1-NN 余弦相似度

### 预训练

约 300B observations[^src-2607-01918]：

- 真实数据来自 Chronos datasets（94B 点，不用 TSMixup 增强）与 GiftEvalPretrain（71 单变量 + 17 多变量数据集、450 万条序列、230B 点、7 域 13 频率，与 GIFT-Eval 测试集无重叠）[^src-2607-01918]
- Aegis-Syn 合成数据（约占总采样序列 10%）：扩展 KernelSynth，补充非光滑/不连续模式[^src-2607-01918]
- BLAST 平衡采样（Shao et al. 2025a）缓解模式失衡；所有评测数据集排除在预训练外防泄漏[^src-2607-01918]
- 训练：200k 步，全局 batch 512，AdamW，cosine LR 1e-3，warmup 10k 步，4×H100[^src-2607-01918]

## 实证证据

### 点预测（论文表 1/7）

ETTh1/ETTh2/ETTm1/ETTm2、ECL、Weather，horizon {96, 192, 336, 720}，context 从 {512, 720, 1024, 2048, 3072} 搜索[^src-2607-01918]。24 项中 19 项最优；平均较前 SOTA 模型 MSE −9.0%、MAE −2.3%；与零样本最佳 TSFM（Timer）相比 MSE −40.3%、MAE −25.3%；在多数数据集上超过 full-shot 任务特定模型（ModernTCN/GPT4TS/TimesNet/PatchTST）[^src-2607-01918]。分数据集平均（MSE/MAE）：ETTh1 0.377/0.399、ETTh2 0.320/0.364、ETTm1 0.322/0.359、ETTm2 0.249/0.305、ECL 0.157/0.243、Weather 0.217/0.247[^src-2607-01918]。

### 概率预测（论文表 2）

GIFT-Eval：23 数据集、144,000 序列、177M 点，97 任务（55 短期/21 中期/21 长期）[^src-2607-01918]。Zeus 的 MASE 0.693、CRPS 0.480，在表列出的零样本预训练模型中均最低；对照（MASE/CRPS）：Chronos-2 0.698/0.485、TimesFM2.5 0.705/0.490、Xihe 0.701/0.488、TiRex 0.716/0.488、FlowState 0.726/0.502、Moirai2 0.728/0.516、Kairos 0.742/0.548、Toto 0.750/0.517、Sundial 0.750/0.559；监督 PatchTST 0.849/0.587、DLinear 1.061/0.846、Seasonal Naive 1.000/1.000[^src-2607-01918]。

### 插补（论文表 3/8）

ETT/ECL/Weather，序列长 192，掩码比例 {12.5%, 25%, 37.5%, 50%}，random + block 两种掩码[^src-2607-01918]。block 掩码下缺失段长 ~ Geometric(p=0.125)，期望段长 8，99% 概率小于 35[^src-2607-01918]。零样本下所有数据集、两种掩码全面优于 TSFM（MOMENT/Timer/UniTS）与监督任务特定模型（GPT4TS/ModernTCN/TimesNet/PatchTST/DLinear）；与最强任务特定模型相比，random 掩码 MSE 平均 −24.4%、block 掩码 −18.8%[^src-2607-01918]。论文补充证据（表 6）：MOMENT 从 patch-missing 换到 point-missing 平均 MSE 恶化 −22.4%（ETTm1 −21.7%、ETTh2 −24.8%、Weather −16.9%），论文归因于 patch 预训练目标与逐点缺失分布不匹配（OOD）[^src-2607-01918]。

### 异常检测（论文图 4、表 9）

UCR Anomaly Archive 42 数据集，adjusted F1，窗口从 {64, 256, 512, 1024, 2048} 搜索[^src-2607-01918]。平均 adjusted F1 0.900，在全部基线（含监督）中最高；21 项最优；零样本超过 full-shot 任务特定模型（PatchTST 0.877、TimesNet 0.856、ModernTCN 0.789、Anomaly Transformer 0.651）[^src-2607-01918]。比第二好的 TSFM（UniTS 0.744）F1 高 +21.0%；MOMENT 0.716、Timer 0.598、GPT4TS 0.676[^src-2607-01918]。

### 分类（论文图 5、表 10）

UEA 26 个等长数据集[^src-2607-01918]。tuning-free 设置下 1-NN 平均准确率 0.675：比 MOMENT 1-NN（0.605）高 +7.0pp，与 full-shot TimesNet（0.673）相当；linear probing（骨干冻结）0.728，在对比方法（含监督）中最高（ModernTCN 0.707、SVP-T 0.725、Rocket 0.704）[^src-2607-01918]。论文自述：纯非参数设置下仍无法超越 full-shot SOTA 分类模型[^src-2607-01918]。

## 效率（论文图 8、附录 C.2）

自注意力 FLOPs 为同深度 vanilla Transformer 的约 0.27 倍（约 3.8× 缩减），因为大部分注意力在 L/8、L/32 分辨率计算[^src-2607-01918]。与 [[time-moe|Time-MoE]]base（约 113M，同为 point tokenization）实测对比：L=4096、各 1000 次运行平均、均开 FlashAttention，Zeus 2.1× 更快、GPU 显存省 3.1×[^src-2607-01918]。

## 消融（论文图 6）

去掉 predictive mask → GIFT-Eval 明显下降（外推能力）；去掉 multi-block → 插补下降；去掉 single-block → 异常检测与分类一致下降（全局一致性）[^src-2607-01918]。

## 论文自述局限（附录 F）

- 单变量聚焦：channel-independent 处理多变量，未显式建模变量间相关（论文建议可配合 CoRA 式适配，未来工作）[^src-2607-01918]
- 未支持时序分割、因果发现等任务；不规则时序预测（结构性缺失）未评估[^src-2607-01918]
- 分类：UEA 部分数据集极短（<30 步）且高维（>100 变量）超出模型擅长范围；标签语义异质；尝试对比学习/SwAV 原型对比学习（联合或后训练）聚类表征效果不理想[^src-2607-01918]

## 相关页面

- [[source-2607-01918]] — 源文件摘要
- [[multi-objective-temporal-masking]] — Zeus 的 MOTM 预训练目标（注意与 [[motm|MoTM（Mixture of TimeFlow Models）]] 无关）
- [[patch-based-tokenization]] — 论文批评的 patch 分词范式（附录 C.1：patch 纠缠细粒度变化、patch 级重建过拟合 patch-wise 缺失、周期等于 patch 长度时退化为 FFN）
- [[channel-independence]] — 多变量处理策略
- [[instance-normalization]] — RevIN 去尺度变化
- [[time-moe]] — 效率实测对比对象 Time-MoE
- [[timesfm]] / [[chronos]] / [[sundial]] — 概率预测对照的 TSFM
- [[patchtst]] / [[timesnet]] — 监督对照方法
- [[tabpfn-ts]] / [[time-indexed-foundation-model]] — 与掩码重建不同的插补范式（时间索引 + in-context 回归），Zeus 走掩码重建路线

[^src-2607-01918]: [[source-2607-01918]]
