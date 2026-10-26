---
title: "Position Interpolation（arXiv:2306.15595）"
type: source-summary
tags:
  - position-encoding
  - rotary-position-embedding
  - context-extension
  - interpolation
  - 2023
created: 2026-10-26
last_updated: 2026-10-26
source_count: 2
confidence: medium
status: active
---

# Position Interpolation（arXiv:2306.15595）

Meta Platforms（Shouyuan Chen, Sherman Wong, Liangjian Chen, Yuandong Tian）。官方地址 https://arxiv.org/abs/2306.15595，v2 于 2023-06-28 发布。**raw:** `raw/position-interpolation-2306-15595.pdf`（arXiv v2 下载，18 页）。

## 核心论点

预训练 RoPE 模型直接外推超出训练窗口会失效：2048 窗口的 LLaMA 在 4096 以上 perplexity 超过 $10^3$。把位置下标线性压缩回预训练区间（Position Interpolation, PI）即可恢复，且只需极少微调——LLaMA 7B/13B 扩至 32768、33B 至 16384、65B 至 8192，均在 Pile 上以 1000 步全参数微调[^src-pi]。

## 机制

$f'(x,m)=f(x, mL/L')$，即下标除以 $s=L'/L$。由于 RoPE 的相位对实数下标有定义，压缩后仍为合法旋转；注意力打分变为 $a\!\left((m-i)/s\right)$，只依赖压缩后的相对距离。不新增权重、不改架构，位置映射本身没有参数[^src-pi]。

理论部分将打分展开为三角基 $a(s)=\mathrm{Re}[\sum_j h_j e^{\mathrm{i}s\theta_j}]$，用两点线性插值余项给出界 $|a(s)-a_{\mathrm{linear}}(s)|\le d\max_j|h_j|(s-s_1)(s_2-s)/(8\ln c)$，在 $c=10000$ 下最坏为 $d\max_j|h_j|/294.73$；对照 RoPE 原文外推界（含 $B(s)\ge d$），插值界至少小约 600 倍（LLaMA 7B 的 $d=128$ 设置，Theorem 2.1、Appendix A–B）[^src-pi]。

## 证据

PG19 上 7B 扩到 32768 后，2048→32768 各评估窗口 perplexity 为 7.23/7.04/6.91/6.80/6.77，同尺寸直接微调模型为 7.21/7.34/7.69（Table 1）；Proof-pile 上对应为 2.82/2.59/2.39/2.24/2.48（Table 2）。微调 200 步后 passkey 有效窗口 $k_{\max}$ 达到目标长度 $L'$（7B 测到 32768、33B 测到 16384），直接微调 10000 步仅到 2560（Table 4）。7B 扩到 16384 时，第 0 步 perplexity 为 112.13，200 步为 7.05（Table 3）。原窗口基准有退步：7B 扩到 32768 后 BoolQ 76.1→64.7、Race-M 55.7→50.1（Table 5）。

## 边界

方法仅适用于 RoPE，未评测可学习位置嵌入，且须经微调方可使用；窗口内性能随扩展倍数退化，需权衡长窗口收益与可用性；所有 $d/2$ 对频率共用一个 $s$，未按波长分档；分档设计及高频损失论证来自后续 [[ntk-aware-interpolation]]、[[ntk-by-parts-interpolation]]，本文未作比较[^src-yarn]。

## 相关页面

[[position-interpolation]] 涵盖机制、推导与数据流，[[rope]] 讨论被改写的位置编码，[[context-window-extension]] 梳理方法谱系，[[yarn]] 与 [[dynamic-scaling]] 介绍后续扩窗方法。

## 引用

[^src-pi]: [[source-position-interpolation-2306-15595]]
[^src-yarn]: [[source-yarn]]
