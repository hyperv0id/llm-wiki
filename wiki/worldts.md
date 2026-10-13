---
title: "WorldTS"
type: entity
tags:
  - time-series
  - forecasting
  - multimodal
  - covariate
  - world-model
  - latent-space
created: 2026-10-13
last_updated: 2026-10-13
source_count: 1
confidence: medium
status: active
---

# WorldTS

**WorldTS**（World Modeling for Multimodal Covariate-aware Time Series Forecasting，Zhu, Qiu, Cheng, Shen, Guo, Yang, Hu 与 Jensen，华东师范大学 + Aalborg University，arXiv:2609.31162v1 [cs.LG]，2026-09-25，preprint）是面向多模态协变量感知时序预测的 world-modeling 框架。观测序列只是系统动力学的部分体现，在观测空间直接拟合历史到未来的映射，会把状态演化与观测生成纠缠在一起；既有 covariate-aware 方法把外部信息用于观测空间预测，既有潜空间方法（LatentTSF、TimeAlign）只从目标序列自身编码——外部多模态信息均不参与状态的形成与演化[^src-worldts]。

## 问题设定

给定历史序列 $x_{1:T} \in \mathbb{R}^{T \times C}$ 与多模态协变量 $C = \{c^m_{1:\tau_m}\},\ m \in \{\mathrm{ts, txt, img}\}$（$\tau_m$ 可为 $T$ 即 future-unknown，或 $T+H$ 即 future-known），预测 $\hat{x}_{T+1:T+H}$。WorldTS 把该任务重构为潜空间状态预测：TS 编码器把观测映射为状态 $Z \in \mathbb{R}^{T \times D}$，各模态专用编码器把协变量映射为模态表征，predictor 在潜空间预测未来状态，解码器把预测状态映射回观测；训练时共享编码器编码真值未来观测，所得状态直接监督预测状态[^src-worldts]。

## 机制

训练分两阶段，把潜动力学学习与观测解码解耦[^src-worldts]：

### Stage 1：Latent Dynamics Learning

- **TS Encoder**：长度 $S_e$、stride 1 的重叠因果 patch（开头 replicate padding，patch 末端只含 $\le t$ 的观测以保持 causality），共享线性投影到 $D$ 维，再经残差两层 FFN（GELU）。同一编码器编码历史与真值未来观测，末尾 $S_e-1$ 个历史观测为首批未来 patch 提供上下文。
- **模态编码**：数值协变量编码器（Cov Encoder）与 TS Encoder 同构、参数独立；文本分支沿用 [[vot|VoT]] 的做法——冻结 GPT-2 的输入 embedding 层提取 token 表征后按时间步 mean pooling；图像分支沿用 [[unified-covariate-adaptation|UniCA]] 的做法——轻量 CNN + 线性投影；两模态各接一个 Modal Adapter MLP 得到时间对齐表征。
- **Predictor**：历史状态与数值协变量表征沿特征维 concat 后过 MLP 得融合历史状态；文本/图像表征经 CFA（Controlled Fusion Adapter，见 [[constrained-text-fusion]]，低秩瓶颈 $W_{down} \to W_{up}$ 加 LN/GELU/Dropout）映射为状态空间残差加到融合状态上；状态转移模块是 patch-based Transformer（PatchTST 式），输出 base 未来状态；若未来协变量可用，再过 fusion MLP 计算状态残差加到 base 上，否则直接用 base。
- **损失**：预测状态与编码器编码真值未来观测所得状态做 state matching——逐元素 L1（stop-gradient 作用于编码状态）+ 展平未来状态序列上的 cosine；state matching 存在表征塌缩风险，故再对预测与编码两组状态加 [[vicreg|VICReg]] 的方差/协方差正则（统计量跨样本与预测步计算）。$\mathcal{L}_{Stage1} = \lambda_{sim}\mathcal{L}_{sim} + \lambda_{latent}\mathcal{L}_{latent} + \lambda_{VIC}\mathcal{L}_{VICReg}$。

### Stage 2：Observation Decoding

冻结编码器与 predictor，只训解码器：把末尾 $S_d-1$ 个历史状态与 $H$ 个预测状态拼接，抽取 $H$ 个长度 $S_d$ 的状态 patch，逐步投影、残差 FFN、输出投影到 $C$ 维。损失为 $\mathcal{L}_{Stage2} = \|x - D_\phi(\hat{Z})\|_1 + \alpha\|x - D_\phi(Z)\|_1$——预测项适配推理时实际可得的预测状态，重建项用编码真值未来状态提供额外监督[^src-worldts]。

## 证据

以下结果均来自论文实验章节：12 个数值协变量数据集沿用 DAG 协议（短程 lookback 168/horizon 24，长程 720/360；Colbun/Rapel 为 60/10 与 180/30），8 个 Time-MMD 文本域沿用 VoT 协议（horizon 12），MMSP 沿用 UniCA 协议（前 10 个光伏电站，lookback 2048/horizon 24）；指标 MSE/MAE；数值协变量基线数字取自 DAG/KITE，文本取自 VoT，图像取自 UniCA[^src-worldts]。

- **数值协变量**（Table 1，短/长 horizon 平均）：WorldTS 在全部 12 个数据集的平均 MSE 与 MAE 均为最低（1st count 12/12）；Energy 上 MSE 0.091 对最强基线 0.111（−18.0%）。Table 5 逐 horizon 上少数设定未排第一：FR@24 与 DE@24 被 KITE 超过，Sdwpfm1/Sdwpfm2@24 被 DAG 超过。
- **文本协变量**（Table 2，8 个 Time-MMD 域，horizon 12）：16 组数据集-指标对比中 14 组第一（MSE 6/8、MAE 8/8）；最大增益在 Health，MSE 与 MAE 相对最强基线分别降 8.9% 与 9.1%。
- **图像协变量**（Table 3，MMSP，horizon 24）：MSE 0.061、MAE 0.171 均最低，相对最强基线 TFT（0.067/0.177）约降 9.0% 与 3.4%。
- **消融**（Table 4，Colbun/Climate/MMSP 三数据集，六变体）：去掉任一损失项、把因果 patch 编解码换成逐点 MLP、退化为单阶段端到端观测空间 MAE 训练、或置空协变量，误差均上升；MMSP 上单阶段变体 MSE 0.085、去协变量 0.101，完整模型 0.061。
- **敏感性**（Figure 4，Colbun/Climate/MMSP 逐超参扫描）：$\lambda_{latent}=10$ 在三个数据集上 MAE 最低，$\lambda_{sim}=15$ 与 $\lambda_{VIC}=0.1$ 为最优或次优；$\lambda_{VIC}$ 升至 0.5 时 MMSP 误差明显上升；潜维 128/256 最好，更大维度无一致增益；patch 长度过短或过长都更差。

## 边界（附录 C）

1. 协变量条件化预测状态学习缺乏严格理论，协变量信息、预测状态质量与预测误差之间的联系有待形式化[^src-worldts]。
2. 目标函数与正则沿用通用设计（LatentTSF 式 state matching、VICReg 正则），缺少面向时间结构的定制正则；论文将其定位为世界模型潜空间预测学习移植到时序预测的初步尝试[^src-worldts]。
3. 只处理规则采样序列、逐数据集单独训练，未与预训练时序基础模型结合，未扩展到不规则采样与异步协变量[^src-worldts]。

## Related Pages

- [[latent-space-future-state-prediction]] — WorldTS 所属的潜空间未来状态预测技术路线
- [[world-model]] — 世界模型概念；WorldTS 是该思想在时序预测中的实例
- [[jepa]] — JEPA：编码未来观测作为预测目标的思想来源之一
- [[vicreg]] — WorldTS 借用的防塌缩正则
- [[constrained-text-fusion]] — WorldTS 沿用的 CFA 低秩残差融合
- [[heterogeneous-covariates]] — 异构协变量；WorldTS 提供非同质化的处理路线
- [[multimodal-time-series-forecasting]] — 所在任务全景
- [[time-mmd]] — 文本协变量评测基准（WorldTS 用其中 8 域）
- [[vot]] — 文本分支沿用其 GPT-2 特征提取方案
- [[urbanfm]] — 注意区分：UrbanFM 的数据组件名为 WorldST，与本工作无关

[^src-worldts]: [[source-worldts]]
