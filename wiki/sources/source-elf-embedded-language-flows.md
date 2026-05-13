---
title: "ELF: Embedded Language Flows"
type: source-summary
tags:
  - diffusion-language-model
  - flow-matching
  - continuous-generation
  - language-modeling
created: 2026-05-13
last_updated: 2026-05-13
source_count: 1
confidence: high
status: active
---

# ELF: Embedded Language Flows

> **arXiv: 2605.10938, MIT, 2026**
> Keya Hu*, Linlu Qiu*, Tianhong Li, Yoon Kim, Yiyang Lu, Hanhong Zhao, Jacob Andreas, Kaiming He
> *Equal contribution; order decided by a coin flip.

## 核心问题

扩散语言模型（DLMs）一直分两个阵营：连续 DLMs（在 embedding 空间去噪）和离散 DLMs（在 token 空间直接扩散）。**离散 DLMs 一直是主流**——MDLM、Duo 等碾压了 Diffusion-LM、CDCD 等连续方法。

ELF 问的是：连续 DLMs 效果差，是因为**语言本质上是离散的**（先天缺陷），还是因为**算法设计没做对**（后天可改）？

ELF 的答案是：**后者**。连续 DLMs 只需要一个关键修正——不要在每一步都往回拉到离散空间[^src-elf]。

## 核心洞察

### 之前的连续 DLM 做了什么
几乎所有之前的方法（Diffusion-LM、CDCD、DiffuSeq、SSD-LM、TESS 等）都在训练时每步加 cross-entropy loss，把中间状态拉回 token 空间。这本质上是"在连续空间里做离散 diffusion"，没有真正利用连续空间的灵活性[^src-elf]。

### ELF 做了什么
ELF 坚持**在整个去噪过程中保持连续**，只在最后一个时间步 t=1 做离散化。这看似简单的改动，有深远影响[^src-elf]：

1. **Flow Matching** 的 t=1 天然就是离散化时机——不需要额外 decoder
2. **共享权重** —— 同一个网络同时做 denoising 和 decoding（靠 mode token 区分）
3. **直接继承图像域 diffusion 的所有技术** —— CFG、self-conditioning、training-time CFG 等直接可用

### 关键设计选择

- **x-prediction 参数化**：预测干净 embedding x（而非 velocity v 或 noise ε）。这让 denoising（MSE loss）和 decoding（CE loss）的权重共享成为可能。v-prediction 在高维 embedding 下性能严重退化，ε-prediction 完全崩溃[^src-elf]

- **80% MSE + 20% CE 混合训练**：两个分支在一个 batch 内通过 masking 并行计算，不增加额外开销[^src-elf]

- **In-context conditioning**：把时间步、CFG scale、model mode 作为 control token prepend 到序列前，而非传统的 adaLN-Zero 加法融合——当条件信号种类多时更有效[^src-elf]

- **SDE sampler**：在每个采样步注入少量高斯噪声，同时回退时间步。在少步数 regime 下显著优于 ODE sampler，说明引入随机性可以有效减少误差累积[^src-elf]

- **Logit-normal 时间调度**：训练和推理都使用 logit-normal 分布采样时间步，在 t≈0 时分配更细的步长（噪声大时需精细处理）[^src-elf]

- **Bottleneck 设计**：T5 encoder 的 512-dim embedding 经线性投影压缩到 128-dim，再投影回 model hidden size——128 是生成质量-多样性的最佳平衡点[^src-elf]

## 实验结果

### 无条件生成（OWT，105M 参数）
- **Gen. PPL 24 @ 32步**，无需蒸馏——超过 MDLM、Duo、FLM、LangFlow（均 170M 参数）
- 与蒸馏变体（MDLM+SDTT、Duo+DCD、FMLM）对比，**不蒸馏的 ELF 照样击败它们**[^src-elf]
- **仅用 45B 训练 token**，其他方法需要 500B+（10x+ 差距）[^src-elf]

### 条件生成
| 任务 | 指标 | ELF | 最佳基线 |
|------|------|-----|----------|
| WMT14 De-En 翻译 | BLEU | **26.4** | 25.2 (MDLM) |
| XSum 摘要 | ROUGE-1 | **36.0** | 33.4 (Duo) |
| XSum 摘要 | ROUGE-2 | **12.2** | 11.6 (Duo) |
| XSum 摘要 | ROUGE-L | **27.8** | 25.8 (Duo) |

### 缩放行为
- ELF-B(105M)、ELF-M(342M)、ELF-L(652M) 三个规模测试
- 模型越大，Gen. PPL-entropy frontier 越优
- SDE sampler 在所有规模上一致优于 ODE sampler[^src-elf]

## 论文定位

ELF 在连续 DLM 设计空间中占据了一个**独特位置**（参见附录 Tab.2 的全面 survey）：

| 维度 | 大多数连续 DLM | ELF |
|------|---------------|-----|
| 训练时每步离散化 | ✅ 是 | ❌ 否 |
| 推理时每步离散化 | ✅ 是 | ❌ 否 |
| 单独 decoder | ✅ 需要 | ❌ 不需要（共享权重） |
| 基础过程 | DDPM / Score-based | **Flow Matching** |
| 连续状态 | 各种（embed/simplex/latent） | **冻结 contextual embedding** |
| CFG 支持 | 困难 | **原生支持** |

## 方法局限性

- 实验规模较小（最大 652M），未验证到更大 scale 的行为
- 仅在英文 OWT 上做无条件生成，多语言未验证
- 条件生成仅测试翻译和摘要，未涉及更复杂的指令跟随
- 依赖 T5 encoder 做 embedding（推理时不用，但训练时是额外模块）
- 生成质量仍然落后于同规模自回归模型（论文未直接对比 GPT-2 等）

## 对领域的影响

ELF 有力地证明了：**连续 DLM 并不天生劣于离散 DLM**。只要设计得当（坚持连续、最后一步才离散化），连续 DLM 可以更高效（10x 训练数据）、更灵活（原生 CFG）、并且直接继承图像 diffusion 领域的所有进步。这意味着 diffusion 语言模型的"离散 vs 连续"之争远未结束——ELF 为连续路线给出了强有力的证据。

[^src-elf]: [[source-elf-embedded-language-flows]]