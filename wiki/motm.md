---
title: "MoTM (Mixture of TimeFlow Models)"
type: entity
tags:
  - time-series
  - data-imputation
  - foundation-model
  - zero-shot
  - implicit-neural-representation
  - continuous-time
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: medium
status: active
---

# MoTM (Mixture of TimeFlow Models)

**MoTM**（Le Naour et al. 2025）是一种零样本时间序列插补的[[time-indexed-foundation-model|时间索引基础模型]]，扩展自 TimeFlow 架构，核心是用一组**调制隐式神经表示（INR）基**表征任意序列[^src-time-indexed-imputation]。在 TMLR 2026 基准中以 **NMAE 0.371 居次**（仅次于 [[tabpfn-ts|TabPFN-TS]]），且**推理快约两个数量级**，被定位为可扩展的替代方案。

## 设计：学习表征 + 简单回归器

哲学与 [[tabpfn-ts|TabPFN-TS]] **互逆**——丰富的学习表征搭配简单回归器[^src-time-indexed-imputation]：

### (i) 调制 INR 基的表征学习
MoTM 不学单个函数，而是学 **K 个不同的 INR**。每个 INR 是小型神经网络，由**超网络（hypernetwork）**参数化，把连续时间坐标 $t$ 映射到特征向量。这些基被"调制"——参数为每个新窗口动态生成，从而捕获趋势、季节性、高频振荡等多样模式，不受 Fourier 等预定义频率限制。对任意 $t$，上下文表示 $H(t)$ 由 K 个基 INR 在该时刻的输出拼接而成。

### (ii) 局部回归的 in-context 插补
给定含缺失序列，MoTM 先取观测点的上下文窗口，**在线拟合一个 ridge 回归器**学习从高维表征 $H(t)$ 到值 $x(t)$ 的线性映射；再用该局部回归器对任意缺失时刻 $t_{miss}$ 的表征 $H(t_{miss})$ 预测[^src-time-indexed-imputation]。

天然扩展：**协变量集成**（把上下文信息堆叠到 $H(t)$，预训练 INR 基不变）；**不确定性量化**（用 quantile 回归器替代 ridge）。

## 性能与优势

- **精度**：NMAE 0.371（次佳），零样本超越所有监督与局部基线[^src-time-indexed-imputation]。
- **效率**：比 TabPFN-TS **快约两个数量级**，在精度-效率间取得良好平衡——资源受限或高吞吐场景的首选[^src-time-indexed-imputation]。
- 局限：一般不如 TabPFN-TS 准确，对突变的重建有时不够精确（含协变量时）[^src-time-indexed-imputation]。

## 与相关方法的关系

- **vs [[tabpfn-ts|TabPFN-TS]]**：互逆设计；MoTM 更快但略逊精度。论文建议的融合方向：在 MoTM 的调制 INR 特征上换用 in-context 训练的强回归器（替代 ridge），兼取两者之长[^src-time-indexed-imputation]。
- **谱系**：扩展自 TimeFlow（Le Naour et al. 2024），属调制 INR 时序建模一脉。

## 关联页面

- [[time-indexed-foundation-model]] — MoTM 所属范式
- [[tabpfn-ts]] — 互逆设计、更准但更慢的时间索引模型
- [[source-time-indexed-imputation]] — 评估 MoTM 的 TMLR 2026 基准
- [[nuwats]] — PLM 重编程零样本插补（基准中 MoTM 在 10/11 数据集上领先 NuwaTS）

[^src-time-indexed-imputation]: [[source-time-indexed-imputation]]
