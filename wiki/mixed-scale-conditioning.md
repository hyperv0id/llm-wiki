---
title: "Mixed-Scale Conditioning"
type: technique
tags:
  - conditioning
  - multi-scale
  - autoregressive
  - climate-forecasting
  - cross-attention
created: 2026-07-16
last_updated: 2026-07-16
source_count: 1
confidence: medium
status: active
---

# Mixed-Scale Conditioning（混合尺度条件控制）

**混合尺度条件控制**是 [[climatear|ClimateAR]] 提出的双层条件注入机制，用于在自回归生成过程中同时捕获尺度内局部一致性和跨尺度全局交互[^src-climatear]。

## 动机

气候系统的核心挑战：ENSO 等大尺度现象（数千公里）由小尺度过程（海洋涡旋、区域对流活动）调制，但标准自回归生成在尺度 $k$ 只能访问条件 token $r'_{\le k}$，无法感知后续更精细尺度的条件信息 $r'_{>k}$[^src-climatear]。此外，直接将所有尺度的条件 token 同时注入会导致高信息密度的收敛困难[^src-climatear]。

## 双层设计

### 1. Intra-Scale Mixed Token（尺度内混合 Token）

在各尺度 $k$，将自回归特征 $\tilde{f}_{k-1}$（由已生成 token $r_{\le k-1}$ 重构）与条件特征 $\tilde{f}'_k$ 在空间上拼接，经下采样对齐分辨率后得到混合 token $R_k$[^src-climatear]：

$$R_k = \text{Concat}\left( \text{down}(\tilde{f}_{k-1}, (w_k, h_k)), \text{down}(\tilde{f}'_k, (w_k, h_k)) \right)$$

这使得条件近似 $p(r_k \mid r_{<k}, r'_{\le K}) \approx p(r_k \mid R_{\le k})$，维持尺度内物理一致性[^src-climatear]。

### 2. Hybrid-Scale Prompt（混合尺度前缀）

通过 cross-attention 将全部 $K$ 个尺度的条件 token 压缩为一个连续的跨尺度上下文向量 $C_{mix}$，作为自回归序列的前缀[^src-climatear]：

$$C_{mix} = \text{Attention}\left(q = q, kv = (r'_1, r'_2, ..., r'_K)\right)$$

其中 $q$ 为可学习的混合尺度查询。最终条件生成：

$$p(r_k \mid r_{<k}, r'_{\le K}) = p(r_k \mid C_{mix}, R_{\le k})$$

## 消融

移除 hybrid-scale prompt 或 intra-scale mix token 均导致 6 月平均 ACC 下降，验证了双层设计的必要性[^src-climatear]。

## 相关页面

- [[climatear]] — ClimateAR 模型
- [[multi-scale-attention]] — 多尺度注意力机制
- [[source-climatear]] — 论文源摘要

[^src-climatear]: [[source-climatear]]
