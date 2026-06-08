---
title: "Time-Indexed Foundation Model (时间索引基础模型)"
type: concept
tags:
  - time-series
  - data-imputation
  - foundation-model
  - zero-shot
  - continuous-time
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: medium
status: active
---

# Time-Indexed Foundation Model（时间索引基础模型）

**Time-Indexed Foundation Model** 是一类用于零样本时间序列插补的基础模型范式：在**每个时间戳** $t$ 学习一个上下文表示 $H(t)$，再用回归器 $r_\theta(\cdot)$ 把 $H(t)$ 映射到序列值 $x(t)$[^src-time-indexed-imputation]。代表是 [[tabpfn-ts|TabPFN-TS]] 与 [[motm|MoTM]]，由 EDF R&D 的 TMLR 2026 基准研究系统评估。

## 核心思想：连续时间建模

与基于 patch 的预测基础模型（在稠密、完整观测上下文上训练以预测未来 horizon）不同，时间索引模型采用**连续时间建模**——把序列视为时间坐标 $t$ 上的函数 $x(t)$。这使其在推理时天然支持[^src-time-indexed-imputation]：

1. 处理**不规则/未对齐**时间序列
2. 跨**不同采样率**工作
3. 插补**任意缺失区域**（而非固定 horizon）
4. 通过与上下文表示**拼接**集成协变量（无需重训）

> [!note] 为什么 patch-based 预测模型不适合插补
> patch-based 预测器（及 xLSTM 类）需要稠密完整的上下文，难以应对插补任务固有的多样缺失模式；时间索引模型直接在观测时间戳上拟合 $H(t)\to x(t)$，对缺失天然鲁棒[^src-time-indexed-imputation]。

## 两种互逆的实例化

二者哲学相反——一个"简单特征 + 强回归器"，一个"学习表征 + 简单回归器"[^src-time-indexed-imputation]：

| 维度 | [[tabpfn-ts\|TabPFN-TS]] | [[motm\|MoTM]] |
|------|------------|------|
| 表示 $H(t)$ | **手工特征**：归一化时间索引 + Fourier 正余弦基（日/周周期） | **学习表征**：K 个调制 INR 基（超网络生成，每窗口动态调制） |
| 回归器 $r_\theta$ | **TabPFN**（大型 Transformer，在数亿合成表格回归任务上预训练，in-context learning 单次前向） | **Ridge 回归**（在观测上下文上局部拟合） |
| 不确定性 | TabPFN 返回输出分布 | Quantile 回归器 |
| 速度 | 慢（H100 上每 672 步 ~1s） | **快约两个数量级** |
| 精度（基准 NMAE） | **0.293（最佳）** | 0.371（次佳） |

二者均零样本超越所有监督基线（SAITS/BRITS/CSDI/TimesNet）与局部基线（Linear/LOCF），在**块状缺失**下优势尤为明显[^src-time-indexed-imputation]。

## 与其他插补范式的对比

| 范式 | 代表 | 缺失处理 | 泛化 |
|------|------|---------|------|
| **时间索引基础模型** | TabPFN-TS, MoTM | 连续时间 $H(t)\to x(t)$ | **零样本，跨域最强** |
| PLM 重编程 | [[nuwats\|NuwaTS]] | patch + 掩码嵌入 | 零样本，但基准中显著落后于 TabPFN-TS/MoTM |
| 监督专用 | SAITS, [[csdi\|CSDI]], BRITS | 任务特定训练 | 鲁棒性有限，易过拟合 |
| 局部插值 | Linear, LOCF | 时间先验 | 稀疏点缺失下仍有竞争力 |

## 关联页面

- [[tabpfn-ts]] — 手工 Fourier 特征 + TabPFN in-context 回归（基准最佳）
- [[motm]] — 调制 INR 基 + ridge 回归（快两个数量级的可扩展替代）
- [[nuwats]] — PLM 重编程零样本插补（基准中落后于时间索引模型）
- [[source-time-indexed-imputation]] — 系统评估这两个模型的 TMLR 2026 基准
- [[missing-not-at-random]] — 缺失机制谱系

[^src-time-indexed-imputation]: [[source-time-indexed-imputation]]
