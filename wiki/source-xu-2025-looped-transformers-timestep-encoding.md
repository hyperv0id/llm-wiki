---
title: "Source: On Expressive Power of Looped Transformers — Theoretical Analysis and Enhancement via Timestep Encoding (ICML 2025)"
type: source-summary
tags:
  - source
  - looped-transformer
  - approximation-theory
  - weight-tying
  - timestep-encoding
created: 2026-10-28
last_updated: 2026-10-28
source_count: 1
confidence: medium
status: active
---

# Source: On Expressive Power of Looped Transformers（ICML 2025）

- **作者**：Kevin Xu, Issei Sato（The University of Tokyo）
- **会议**：ICML 2025（Proceedings of the 42nd ICML, Vancouver, PMLR 267），正文 8 页 + 附录 A–D
- **代码**：https://github.com/kevin671/tmlt
- **本地**：`raw/xu-2025-looped-transformers-timestep-encoding.pdf`（`pdftotext -layout` 转文本，2543 行）

## 核心论点

论文给单层 looped Transformer 建立函数逼近率：目标函数类是有界域 $[0,1]^{d\times N}$ 上连续且置换等变的序列到序列函数，逼近误差以循环次数 $r$ 为唯一自由度收缩，$\delta=\left((r-N)/2\right)^{-1/((N+1)d+1)}$（Theorem 3.6）[^src-xu-2025-looped-transformers-timestep-encoding]。与标准 Transformer 的既有 rate 相比，界里多出 token 连续性与 contextual 连续性两个 modulus，论文把这两项归因于 weight-tied feed-forward 网络，视为 looped 架构固有的表达限制（§4.1）[^src-xu-2025-looped-transformers-timestep-encoding]。在无限精度权重下取 $r\to\infty$ 得到通用逼近（Corollary 3.7）[^src-xu-2025-looped-transformers-timestep-encoding]。

## 方法

分析沿用 piecewise-constant 逼近路线，但重新设计 contextual token ID 的构造与排序，使相邻嵌入的差能被两个受限 modulus 界定；证明把量化、contextual mapping、值映射三个子网络压进同一个 Transformer block，并给出 loop 预算 $r=\delta^{-1}+2\delta^{-(N+1)d}+N$（式 14）[^src-xu-2025-looped-transformers-timestep-encoding]。据此提出 TMLT：用正弦 timestep 编码经两层 MLP 生成逐 loop 的缩放向量，调制 RMSNorm 增益与两个子层输出，权重仍共享；对应 Theorem 4.2 把同层 FF 的逐点记忆误差从 $\le\varepsilon_i$ 压到 $=0$[^src-xu-2025-looped-transformers-timestep-encoding]。

## 证据

单层、4 头、256 维模型在 LCS / Edit Distance / Sudoku / Countdown 上扫描 $r$：ED（长度 60）由 $r=5$ 的 23.8 升至 $r=100$ 的 47.7（无 timestep 编码）与 88.3（有编码）；LCS（长度 60）$r=100$ 为 96.9 / 97.1，标准 12 层 Transformer 为 70.0（Table 2）[^src-xu-2025-looped-transformers-timestep-encoding]。in-context learning 的测试 MSE 8.6e-03（TF）/ 1.4e-02（looped）/ 1.7e-03（加 timestep）（Table 3）；WikiText-103 测试困惑度 20.5 / 20.6 / 19.6（Table 4）[^src-xu-2025-looped-transformers-timestep-encoding]。

## 局限

理论限于单层、固定长度、$r>N$，排除 LayerNorm，通用性依赖无限精度权重；多层与变长输入被列为 open question，作者认为多层改进可能只体现在常数而非渐近阶（附录 D）[^src-xu-2025-looped-transformers-timestep-encoding]。实验模型规模小，Table 2 无标准差与种子数，部分格子上加 timestep 编码反而略低（如 Sudoku $r=16$ 62.0 vs 65.6），正文未讨论[^src-xu-2025-looped-transformers-timestep-encoding]。用测试精度代理逼近能力，作者以附录 C.1.2 的训练/测试相关性辩护[^src-xu-2025-looped-transformers-timestep-encoding]。

理论与证明结构见 [[looped-transformer-approximation-rate]]，架构与实验见 [[timestep-modulated-looped-transformer]]，路线背景见 [[recurrent-depth]]。

[^src-xu-2025-looped-transformers-timestep-encoding]: [[source-xu-2025-looped-transformers-timestep-encoding]]
