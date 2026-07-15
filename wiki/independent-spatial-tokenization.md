---
title: "Independent Spatial Tokenization"
type: technique
tags:
  - spatial-encoding
  - tokenization
  - swin-transformer
  - spatio-temporal
  - sea-ice-forecasting
created: 2026-07-21
last_updated: 2026-07-21
source_count: 1
confidence: medium
status: active
---

# Independent Spatial Tokenization

**独立空间 tokenization** 是 [[sifusion|SIFusion]] 中提出的一种空间特征提取策略：先将各时间步的海冰密集度（SIC）独立送入共享空间编码器生成 spatial token 表示，再将这些 token 拼接用于序列建模，从而解耦空间编码与时间建模[^src-sifusion]。

## 动机：U-Net 的隐式时空建模缺陷

此前 SIC 预测主流方法（IceNet、SICNet）采用 U-Net 架构，通过 2D 卷积做 channel-wise fusion，隐式地同时处理空间特征和时间依赖。SIFusion 指出这种设计有两个根本问题[^src-sifusion]：

1. **上下采样扰动序列特征**：U-Net 的 channel expansion/contraction 操作在编码-解码过程中不断改变 channel 数和特征维，干扰 SIC 的内在时序结构，使序列相关性捕获复杂化。
2. **多变量混合恶化时空建模**：当联合建模气候变量（如海表温度 SST）时，不同变量的 channel 混合在一起进一步破坏时空相关性的学习。

此外，U-Net 从根本上不是为序列建模设计的——它擅长像素级密集预测（图像分割），而非捕获序列中的时间依赖[^src-sifusion]。

## 实现方案

SIFusion 采用 Swin Transformer V2 作为共享空间编码器和解码器骨干[^src-sifusion]：

1. **Patch partition**：2×2 最小窗口将单通道 SIC 地图分割为 patch token，生成 32 spatial channel 的 patch representation（类似 ViT）。
2. **层级编码**：两层 Swin Transformer block + patch merging，逐层提取多尺度空间特征。
3. **Linear projection**：2D spatial feature → 1D compact spatial token，供后续粒度序列拼接。
4. **空间特征跳跃连接（Spatial Feature Skip Connection）**：编码器最后一对 Swin block 的输出直接加到解码器第一对 block 的输入，最大化保留空间 SIC 信息，防止深层序列编码导致空间特征丢失[^src-sifusion]。
5. **共享解码器**：对称的 Swin Transformer 骨干 + patch expanding（替代 patch merging），从融合后的 granularity variate feature 恢复多粒度 SIC 预测[^src-sifusion]。

## 设计优势

- **解耦**：空间编码与时间序列建模分离，各组件可独立优化
- **一致性**：共享编码器将不同粒度的 SIC 映射到同一嵌入空间，便于跨粒度交互
- **参数高效**：共享权重减少参数量，且隐式促进多粒度一致表示学习

## 相关页面

- [[sifusion]] — SIFusion 模型
- [[granularity-variates]] — 独立 token 的后续使用方式
- [[multi-granularity-sea-ice-forecasting]] — 多粒度海冰预测
- [[sea-ice-concentration-forecasting]] — 海冰预测领域背景

[^src-sifusion]: [[source-sifusion]]
