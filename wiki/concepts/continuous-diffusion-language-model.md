---
title: "Continuous Diffusion Language Model"
type: concept
tags:
  - diffusion-model
  - language-modeling
  - flow-matching
  - continuous-generation
created: 2026-05-13
last_updated: 2026-05-13
source_count: 1
confidence: medium
status: active
---

# Continuous Diffusion Language Model

连续扩散语言模型（Continuous DLM）是一类将 [[diffusion-model|扩散模型]] 应用于语言生成的方法，其核心特征是：将离散 token 映射到连续表示空间（embedding / simplex / latent），然后在连续空间中执行去噪过程。

## 与离散 DLM 的区别

离散 DLM（如 MDLM、Duo）直接在离散 token 空间中定义扩散过程（absorbing state、uniform transition 等）。连续 DLM 则借助连续空间来绕开离散扩散的数学复杂性[^src-elf]。

## 发展脉络

### 第一代：embedding-space 方法
- **Diffusion-LM**（Li et al., 2022）：在 token embedding 上加 Gaussian noise，每步用 rounding loss 拉回离散空间
- **CDCD**（Dieleman et al., 2022）：连续 categorical diffusion
- **DiffuSeq**（Gong et al., 2023）：seq2seq 连续扩散

这些方法的共同特点：**训练时每步做离散化**（cross-entropy loss），本质上没有真正利用连续空间的灵活性[^src-elf]。

### 第二代：simplex / latent 方法
- **SSD-LM**（Han et al., 2023）：半自回归 simplex 扩散
- **TESS**（Mahabadi et al., 2024）：text-to-text self-conditioned simplex diffusion
- **LD4LG**（Lovelace et al., 2023）：潜在扩散，需要额外 decoder

这些方法开始探索不同的连续状态空间，但仍然在训练或推理时做每步离散化，或者依赖单独 decoder[^src-elf]。

### 第三代：Flow Matching 方法（2026 并发）
- **FLM / FMLM**（Lee et al., 2026）：基于 Flow Matching 的语言模型，仍用每步 token 级监督
- **LangFlow**（Chen et al., 2026）：连续 diffusion 语言模型，用 Bregman divergence
- **DFM / CFM**（Potaptchik et al., 2026; Roos et al., 2026）：离散/连续 flow maps

这些方法与 [[ELF]] 是并发工作，但它们仍然在轨迹中每步使用 token 级 cross-entropy 监督[^src-elf]。

### ELF 的定位

[[ELF]] 在这些方法中占据独特位置——它是唯一一个：
- 训练时不做每步离散化
- 推理时不做每步离散化
- 不需要单独 decoder
- 基于连续时间 Flow Matching
- 直接使用冻结的 contextual embedding[^src-elf]

## 设计空间

ELF 论文提供了对连续 DLM 设计空间的系统 survey（附录 Tab.2），涵盖以下设计轴[^src-elf]：

1. **扩散/流过程**：DDPM vs Flow Matching（连续时间 vs 离散时间）
2. **连续状态空间**：embedding vs simplex vs one-hot vs latent
3. **训练时每步离散化**：是否用 token 级监督约束中间步骤
4. **推理时每步离散化**：采样轨迹中是否回退到 token 对齐状态
5. **独立 decoder**：是否需要额外训练 decoder

[^src-elf]: [[source-elf-embedded-language-flows]]