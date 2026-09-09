---
title: "Time-Gated Multi-Scale Flow Matching for Time-Series Imputation"
type: source-summary
tags:
  - time-series-imputation
  - flow-matching
  - deterministic-ode
  - multi-scale
  - iclr-2026
created: 2026-09-09
last_updated: 2026-09-09
source_count: 0
confidence: medium
status: active
---

# TG-MSFM: Time-Gated Multi-Scale Flow Matching for Time-Series Imputation

**Hangtian Wang & Mahito Sugiyama (2026), ICLR 2026, NII / SOKENDAI**

完整论文（21 页）：`raw/time-gated-multi-scale-flow-matching.pdf`（用户提供：`~/Desktop/20132_Time_Gated_Multi_Scale_F.pdf`，pdftotext 全文 + `-layout` 模式复核 Table 1/2 列对齐；首页水印 "Published as a conference paper at ICLR 2026"，venue 从原文核实）。论文自述以 LLM 润色文本与订正语法（Sec 5 尾注）。

## 核心论题
论文提出 TG-MSFM：把多元时间序列插补形式化为学习数据条件 ODE 的速度场（flow matching），从高斯噪声沿 linear bridge 积分到编码部分观测的结构化端点。三个插补特化组件：(1) 结构化端点 + visibility-masked self-attention——逐时间戳可见性掩码阻止从未观测时间戳泄漏；(2) time-gated multi-scale velocity heads——固定 1D 金字塔 $S=\{1,2,4\}$ 上的尺度专用速度头经 $\mathrm{softmax}(\mathrm{MLP}(t))$ 时间门凸组合，轨迹早期强调粗尺度、后期强调细尺度；(3) Heun 二阶积分 + 每步 data-consistency (DC) 投影——观测坐标与条件通道每步钳回 linear bridge。训练采用 gap-only supervision：FM 损失仅在缺失坐标上评估（Sec 3.2）[^src-tgmsfm]。

## 证据

十数据集（ETTh1/h2/m1/m2、Electricity、Traffic、Weather、Illness、Exchange、PEMS03），RM 10/30/50/70% 平均、5 seeds、MSE/MAE 仅在缺失条目、13 个基线（表 1：含 [[csdi|CSDI]]、PriSTI、Mtsci、Diffusion-TS、[[fgti|FGTI]]、SAITS 等；正文另述含 Sinkhorn OT/TDM 对照）。作者报告：MSE 十列全部最低（Table 1）；MAE 7/10 最低，ETTh2/Weather/Traffic 分别落后最佳基线 0.002/0.008/0.002（iTransformer 0.211 vs 0.213、PriSTI 0.088 vs 0.096、SAITS 0.207 vs 0.209）。速度-质量（Fig 2，ETTh1）：AUPC 0.626 vs CSDI 0.380，$N\approx250$ 后收益递减。消融（Table 2，Electricity MSE）：去时间门退化最大（0.101→0.212），去多尺度头 0.101→0.116，Euler 换 Heun 0.101→0.115。长缺口扫描（Fig 3a，12–72h）增长最慢。种子方差（Table E3）：ETTh1 0.123±0.001 vs CSDI 0.154±0.030，Electricity 0.112±0.003 vs 0.568±0.028。全部实验在 88 核 CPU（Xeon E7-8880 v4）+ 3TB RAM 完成，超参跨数据集固定（Sec 4.1、Table A1）。

## 定位与局限

论文自述定位：确定性、可复现点估计，面向以 MSE/MAE 为主指标的场合（临床记录、工业监控、审计流水线）；不主张替代概率扩散，而是互补（Sec 1）。局限（Sec 5 自述）：linear bridge 与全局时间调度未学数据自适应形式；二次自注意力限制长序列/流式；确定性设计无校准不确定性（未来方向：速度场 ensemble、conformal band）；仅公共基准，未测域特定缺失（censoring、设备掉线）。

[^src-tgmsfm]: [[source-time-gated-multi-scale-flow-matching]]
