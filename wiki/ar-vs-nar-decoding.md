---
title: "AR vs NAR Decoding for Time-Series Forecasting"
type: concept
tags:
  - time-series
  - forecasting
  - autoregressive
  - decoding
  - foundation-model
created: 2026-07-13
last_updated: 2026-11-24
source_count: 5
confidence: medium
status: active
---

# AR vs NAR Decoding（时间序列预测中的自回归 / 非自回归解码）

**AR vs NAR decoding** 是 [[probts|ProbTS]] 显式区分的核心方法决策之一：给定历史 $x_{t-L:t}$ 与协变量，模型如何产生多步未来 $x_{t+1:t+T}$[^src-probts]。

## 形式化与决策轴对照

设编码器 $f_\phi$ 产生隐状态 $h$，预测头 $p_\theta$ 输出点估计或条件分布采样[^src-probts]；在此之上，[[tirex|TiRex]] 引入「连续块掩码占位（CPM）」作为第三种折中方案：用因果循环状态读入历史，再把未来预测窗整体置为缺失占位进行非自回归延续[^src-tirex]。

| 方案 | 状态演化与编码 | 误差与不确定性传递 | 代表模型 |
|------|------|------|------|
| **Autoregressive (AR)** | $h_t = f_\phi(x_{t-1}, c_t, h_{t-1})$，逐步回喂 | 可捕获递推与强季节性；回喂点估计切断分布传递并累积长程误差 | [[deepar\|DeepAR]]、[[timegrad\|TimeGrad]]、[[timesfm\|TimesFM]]、[[chronos\|Chronos]] |
| **Non-autoregressive (NAR)** | $h_{t+1:t+T} = f_\phi(x_{t-L:t}, c_{t-L:t+T})$ 一次生成 | 并行无逐步反馈；长程概率需依赖全局结构或迭代去噪 | [[patchtst\|PatchTST]]、[[csdi\|CSDI]]、[[manf\|MANF]]、MOIRAI |
| **占位式状态延续（CPM + 循环）** | 历史因果递推 $h_t$，未来窗拼缺失占位 $M=1$ 推进 | 免除点估计回喂，保留隐状态与分位数不确定性连续传递；依赖纯 sLSTM 状态记忆 | [[tirex\|TiRex]]（瓶颈接力 $\Delta$：缓解 AR 回喂失真与长程退化） |

## 路线演进与取舍

长程点预测早期路线（如 Informer、Autoformer、[[patchtst|PatchTST]]、iTransformer）几乎一致转向 NAR，以一次性前向规避长 horizon 误差累积[^src-probts]。在精确似然流上，[[manf|MANF]] 以 NAR 结合条件 RealNVP 替代自回归流，在加倍预测长度与缺失噪声压力下减小 CRPS/MSE 衰减并提速[^src-maf]；视觉集合预测的 [[detr|DETR]] 也以并行 object queries 取代自回归解码作为跨领域对照锚点[^src-detr]。

然而在短程概率与零样本基础模型中，AR 与 NAR 的张力并未消失：[[deepar|DeepAR]] 与 [[timegrad|TimeGrad]] 等 AR 路线在短程与强季节场景仍具竞争力，而 [[timesfm|TimesFM]]、[[chronos|Chronos]] 等基础模型同样延续 AR 分块或逐点回喂[^src-probts][^src-deepar]。针对「既保留因果状态追踪、又避免回喂点估计切断不确定性」的缺口，[[tirex|TiRex]] 用纯 `sLSTM` 替代注意力，并在预训练随机遮蔽连续未来块、推理时以缺失占位块延续状态，实证表明去掉多块延续而退化为自回归滚动会令长程 CRPS 明显恶化[^src-tirex]。

## ProbTS 实证规律

1. **AR 长程痛点是误差累积**：原始归一化下，AR 概率模型（如 TimeGrad）的 CRPS 随 horizon 与趋势强度上升而恶化[^src-probts]。
2. **AR 在强季节性上可占优**：Traffic 等强季节场景中 AR 可超过 PatchTST 等 NAR 点模型；季节越强，AR 相对优势越大[^src-probts]。
3. **归一化可部分“复活”长程 AR**：[[instance-normalization|RevIN]] 显著改善多数数据集上的 AR 长程表现（推测主要通过对冲趋势引起的分布漂移），但对强季节弱趋势的 Traffic 有负作用[^src-probts]。
4. **NAR 概率也非免费午餐**：CSDI 类 NAR 扩散在长 horizon 面临显存与学习效率问题，小数据集上长程表现亦欠佳[^src-probts]。
5. **TSFM 复现同一张力**：短 horizon 上 AR 基础模型有竞争力；horizon 拉长后 NAR（如 MOIRAI）优势扩大[^src-probts]。

## 开放问题

- 如何在保留 AR 对季节/递推结构建模优势的同时系统抑制长程误差累积[^src-probts]；
- 如何设计对长程概率预测既高效又表达充分的 NAR 架构与归一化[^src-probts]；
- 基础模型是否需要 horizon 自适应的混合解码策略[^src-probts]。

## 相关页面

- [[probts]] / [[source-probts]]
- [[generative-style-decoder]] — Informer 的非自回归一次前向解码
- [[timegrad]] — AR + 条件扩散
- [[deepar]] — 全局共享 AR-RNN + 参数化似然
- [[csdi]] — NAR 条件扩散（插补/预测）
- [[timesfm]] / [[chronos]] — AR 系基础模型
- [[tirex]] — 纯 sLSTM + 连续块掩码（CPM）占位延续解码（瓶颈接力 $\Delta$：免去点估计回喂）
- [[instance-normalization]]
- [[generative-time-series-forecasting]]
- [[manf]] — NAR + 条件 RealNVP
- [[multi-scale-attention]] — MANF 编码器技术
- [[detr]] / [[object-queries]] — 视觉集合预测的并行解码（NAR 跨领域锚点）

[^src-probts]: [[source-probts]]
[^src-maf]: [[source-maf]]
[^src-deepar]: [[source-deepar]]
[^src-detr]: [[source-detr]]
[^src-tirex]: [[source-tirex]]
