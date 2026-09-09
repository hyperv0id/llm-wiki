---
title: "TG-MSFM"
type: entity
tags:
  - time-series-imputation
  - flow-matching
  - deterministic-inference
  - multi-scale
  - iclr-2026
created: 2026-09-09
last_updated: 2026-09-09
source_count: 1
confidence: medium
status: active
---

# TG-MSFM（Time-Gated Multi-Scale Flow Matching）

**TG-MSFM** 是 Wang & Sugiyama（NII / SOKENDAI，ICLR 2026）提出的多元时间序列插补框架：用 flow matching 学习数据条件 ODE 的速度场，从高斯噪声沿 linear bridge 积分到编码部分观测的结构化端点；推理为确定性单轨迹——每个窗口按固定随机种子取一个 $z_0$，不做多样本聚合或不确定性校准[^src-tgmsfm]。

## 条件化：结构化端点 + visibility-masked attention

输入 $x\in\mathbb{R}^{T\times D}$ 与观测掩码 $M$。结构化端点 $\tilde{x}=[x\odot M,\; m,\; x^L,\; x^R]\in\mathbb{R}^{T\times(D+3)}$，其中 $m_t=\mathbb{1}\{\exists d: M_{t,d}=1\}$ 是逐时间戳可见性标志，$x^L,x^R$ 为窗口 $w=10$ 的左/右观测移动平均。三个条件通道始终视为已知：不参与监督、推理时被 DC 钳制。骨干为 time-aware Transformer：自注意力以加性 $-\infty$ 偏置实现 visibility masking，查询只聚合可见时间戳，阻止从未观测时间戳泄漏[^src-tgmsfm]。

## 训练：gap-only flow matching

linear bridge $z_t=(1-t)z_0+t\,z_1$，$z_0\sim\mathcal{N}(0,I)$，$z_1=\tilde{x}$，oracle 速度恒为 $z_1-z_0$。FM 损失仅在缺失数据坐标集 $\Omega=\{(t,d)\mid M_{t,d}=0\}$ 上评估（式 1）。论文的理由：观测坐标在推理时由 DC 投影硬约束，再对观测项监督与该约束冗余、可能引入冲突梯度；mini-batch 上子采样 $\Omega_b$ 给无偏估计（附录 A.1）。可选小权重正则：速度的一/二阶时间差分（TV）与固定高通核 $[1,-2,1]$ 的高频抑制——权重 $\ll 1$，论文注明非复现结果的关键项[^src-tgmsfm]。

## 推理：Heun + per-step data consistency

二阶 Heun（显式梯形）积分 $N$ 步（默认 $N=300$，每步 2 次速度评估），可选单调时间扭曲 $t_{\mathrm{eff}}=t^k$（默认 $k=1.5$）；每步后将已知索引集 $K$（观测数据坐标 ∪ 条件通道）覆写回 linear bridge。机制与两条性质（oracle 速度下精确、投影非扩张）见 [[data-consistency-projection]]；速度场的多尺度参数化见 [[time-gated-multi-scale-velocity]]。

## 实验证据（作者报告）

- **主结果**（Table 1，RM 10/30/50/70% 平均、5 seeds、指标仅在缺失条目、13 基线；正文另述含 Sinkhorn OT/TDM 两对齐方法作非生成式对照）：MSE 十数据集全部最低；MAE 7/10 最低，ETTh2/Weather/Traffic 分别以 0.002/0.008/0.002 差距落后最佳基线（iTransformer 0.211 vs 0.213、PriSTI 0.088 vs 0.096、SAITS 0.207 vs 0.209）；论文口径为"十数据集平均意义上最强、无逐数据集调参"[^src-tgmsfm]
- 代表数字：ETTh1 0.120/0.219（MSE 次优 DLinear 0.144）、ETTm2 0.020/0.089（次优 FreTS 0.039）、Exchange 0.029/0.025、Illness 0.064/0.116、Traffic 0.187/0.209、PEMS03 0.047/0.142、Weather 0.102/0.096（MAE 劣于 PriSTI 0.088）[^src-tgmsfm]
- **速度-质量**（Fig 2，ETTh1 同硬件同批大小）：AUPC 0.626 vs CSDI 0.380；$N\approx250$ 后收益递减，$N\lesssim100$ 优雅退化；论文配方 $N\in[200,300]$ 近优、$N\in[80,120]$ 快速验证[^src-tgmsfm]
- **消融**（Table 2，Electricity/ETTh1 MSE）：full 0.101/0.126；单尺度 $s{=}1$ 0.116/0.158；静态混合（去时间门）0.212/0.147；Euler 替代 Heun 0.115/0.143——三个组件中时间门退化最大[^src-tgmsfm]
- **长缺口鲁棒性**（Fig 3a）：12–72h 中心缺口上 MSE 增长最慢、全程最低[^src-tgmsfm]
- **种子方差**（Table E3）：ETTh1 0.123±0.001 vs CSDI 0.154±0.030；Electricity 0.112±0.003 vs 0.568±0.028[^src-tgmsfm]
- 环境：全部实验在 88 核 CPU（Xeon E7-8880 v4 @ 2.20GHz）+ 3TB RAM 上完成（Sec 4.1）；默认超参跨数据集固定（Table A1：6 层、8 头、$d_k{=}64$、$S{=}\{1,2,4\}$、AA taps 5、$N{=}300$、AdamW、峰值 LR $2\times10^{-4}$、warmup 5k、batch 32）[^src-tgmsfm]
- **超参敏感性**（Table E1，Illness 长缺口）：论文报告合理范围内差异 $\lesssim$1–2%（E.3）；表中最大偏差为 $N{=}150$ 的 +5.0% 与 $S{=}\{2,4\}$ 的 +3.5%，其余 ≤1.6%——论文以此支持单配置复用[^src-tgmsfm]

## 定位（论文自述）

- 确定性插补：面向可复现、单轨迹检查与点误差指标（MSE/MAE）优先的场合（临床记录、工业监控、审计流水线）；从概率视角，$z_0$ 采样 + 学到的 ODE 仍诱导条件分布，但本文不采用多样本聚合或校准[^src-tgmsfm]
- 与概率路线的关系：CSDI 类随机扩散也可经 probability-flow ODE / DDIM 式采样器确定性运行；论文目标是提供"更轻量、任务对齐"的替代（Sec 1）[^src-tgmsfm]
- 与图插补的关系：[[grin|GRIN]]/[[imputeformer|ImputeFormer]] 等判别式图插补在可靠图可用时是强基线；TG-MSFM 面向 graph-agnostic、长缺口设置（Sec 2）[^src-tgmsfm]

## 局限（论文自述）

- linear bridge + 全局时间调度，未学数据自适应 bridge / time warp
- 二次自注意力，长序列与流式场景受限
- 确定性设计无校准不确定性；未来方向为速度场 ensemble、ODE 轨迹 conformal band
- 仅公共基准，未测域特定缺失模式（censoring、设备掉线）[^src-tgmsfm]

## 与仓库既有方法的关系

| 方法 | 生成式形式 | 与 TG-MSFM 的关系 |
|---|---|---|
| [[csdi\|CSDI]] (NeurIPS 2021) | 条件扩散随机采样 | 直接基线：Table 1 确定性口径下 MAE 一致更低，Table E3 种子方差小两个数量级；论文称互补而非替代 |
| [[tsflow\|TSFlow]] (ICLR 2025) | CFM + GP 先验，概率预测 | 同为 linear bridge CFM；TSFlow 面向概率预测（GP 先验、多样本 CRPS），TG-MSFM 面向确定性插补（高斯源 + DC 钳制） |
| [[giflow\|GiFlow]] (ICML 2026) | CFM 时空插补 | graph-agnostic 高斯源 vs 时空图滤波信息先验（需图结构输入）；两者源分布选择是同一设计轴的两端 |
| [[loft\|LOFT]] (KDD 2026) | 低秩先验 + 轨迹一致性 FM 插补 | LOFT 论文将 MSFM（时间门控多尺度速度场）列为 FM 插补基线（其复现口径 20 NFE），在 PEMS 80–90% 高缺失交通设置上作者报告低于 MSFM；该口径与 TG-MSFM 原文默认 300 步 Heun+DC 不同，两套设置不可直接混用 |
| [[fgti\|FGTI]] (NeurIPS 2024) | 频域条件扩散插补 | 频率信息放条件（双频域 cross-attention） vs 放速度场参数化并沿 $t$ 调度；TG-MSFM Table 1 中 FGTI 为基线 |
| [[costi\|CoSTI]] (KBS 2025) | 一致性模型 1–2 步采样 | 少步概率采样 vs 多步确定性点估计 |

[^src-tgmsfm]: [[source-time-gated-multi-scale-flow-matching]]
