---
title: "Rolling Diffusion Models (滚动扩散模型)"
type: concept
tags:
  - diffusion-models
  - sequence-modeling
  - probabilistic-forecasting
  - uncertainty-quantification
created: 2026-07-16
last_updated: 2026-07-16
source_count: 1
confidence: medium
status: active
---

# Rolling Diffusion Models

**Rolling Diffusion Models**（滚动扩散模型，也称 Rolling Sequence Diffusion Models, RSDM）是一类为序列生成设计的扩散模型变体，其核心创新是 snapshot-dependent 的渐进噪声 schedule：序列中越靠后的快照被施加越大的噪声水平，显式建模预测不确定性随时间增长[^src-erdm]。

## 核心机制

传统序列扩散模型对整个窗口施加均匀噪声，无法体现"越远的未来越不确定"的先验。RSDM 为窗口内每个位置 w 分配单调递增的噪声标准差 σ̄1(t) < σ̄2(t) < ... < σ̄W(t)，在扩散时间 t 从 0→1 的演化中：

- t=0 时，首个 snapshot 近乎干净（σ̄1(0) ≈ σmin），末位 snapshot 近乎纯噪声（σ̄W(0) ≈ σmax）
- t=1 时，首个 snapshot 完全去噪（σ̄1(1) = σmin），其余 snapshot 的噪声水平恰等于前一位在 t=0 时的水平（σ̄w(1) = σ̄w-1(0)）

这一"等噪声交接"性质使得窗口可以高效滚动：输出首个去噪 snapshot 后，剩余序列前移一位，末尾追加新纯噪声，循环往复[^src-erdm]。

## 发展脉络

- **AR-Diffusion** (Wu et al., NeurIPS 2023) 和 **Rolling Diffusion Models** (Ruhe et al., ICML 2024)：最早提出滚动扩散概念，基于 DDPM 框架[^src-erdm]。
- **Diffusion Forcing** (Chen et al., NeurIPS 2024)：提出完全随机的 noise schedule，对应"pyramid sampling"训练方案[^src-erdm]。
- **FIFO-Diffusion** (Kim et al., NeurIPS 2024)：将滚动机制应用于无限视频生成[^src-erdm]。
- **[[erdm|ERDM]]** (Cachay et al., NeurIPS 2025)：首次将 RSDM 与 EDM 设计空间统一，引入 EDM 预处理/Heun 采样/损失重加权到滚动框架[^src-erdm]。

## 关键设计选择

ERDM 的消融实验揭示了若干关键设计准则[^src-erdm]：

1. **固定 schedule 优于随机 schedule**：Diffusion Forcing 的随机化训练在混沌系统中导致 ∼2× 退化，因为随机 schedule 与被预测系统的"时间越远越不确定"的先验错配。
2. **噪声曲率 ρ 至关重要**：EDM 默认 ρ=7 不适合滚动设置，ρ=−10 显著更优。
3. **中等噪声水平权重最大**：通过对数正态 PDF 重加权聚焦确定性→随机性的过渡区间。

## 相关页面

- [[erdm]] — ERDM，EDM 框架的滚动扩散实现
- [[diffusion-model]] — 扩散模型总论
- [[edm]] — EDM 设计空间

[^src-erdm]: [[source-erdm]]
