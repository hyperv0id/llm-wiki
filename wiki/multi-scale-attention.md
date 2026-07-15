---
title: "Multi-scale Attention"
type: technique
tags:
  - attention
  - time-series
  - multi-scale
  - relative-position
  - transformer
created: 2026-07-13
last_updated: 2026-07-21
source_count: 1
confidence: medium
status: active
---

# Multi-scale Attention（多尺度注意力）

**多尺度注意力**在 [[manf|MANF]] 中指：于 Transformer 编码器**不同深度**使用**不同窗口尺度**的局部注意力，并配合**尺度相关的动态相对位置编码**，以同时捕获局部上下文与长程层次模式[^src-maf]。

## 动机

Vanilla 全局自注意力对时间序列有两点不足：(i) 对**局部上下文**不敏感，而局部依赖在多变量预测中往往关键；(ii) 点积注意力把内容与位置混算，且权重易呈长尾，多层重复聚焦“最重要”位置而忽略次要但有用的层次信息[^src-maf]。MANF 用浅层小尺度、深层大尺度形成渐进归纳偏置，并用相对位置在各尺度上单独学习时间信息[^src-maf]。

## 形式化（MANF）

给定序列 $S=(s_1,\ldots,s_N)\in\mathbb{R}^{N\times D}$，第 $l$ 层尺度 $\Theta_l$，多头形式为[^src-maf]：

$$
A(s_i,\Theta_l)=\sum_{h=1}^{H} G^h\!\big(Q^h(s_i),\,W_i(K^h,\Theta_l)\big)\,W_i(V^h,\Theta_l),
$$

其中 $W_i(x,\Theta_l)=[x_{i-\Theta_l},\ldots,x_{i+\Theta_l}]$ 截取局部窗口；$G^h$ 在 query-key 中加入尺度相关偏置 $u_{\Theta_l},v_{\Theta_l}$ 与相对位置 $R_{x-y}$，使同一时间戳在不同尺度可学到不同 timing 表示（不同于固定相对编码设置）[^src-maf]。层更新为 LayerNorm + ReLU 注意力残差 + 位置级 FFN[^src-maf]。

实践中 MANF 取 $\Theta=[L/3,L/2,L]$，$L$ 为 4 倍预测长度，编码器 3 层随深度增大尺度[^src-maf]。

## 作用解读

1. **Locality perception**：小尺度对局部依赖敏感[^src-maf]。
2. **Long-term hierarchy**：堆叠不同尺度近似利用日/周/月等层次结构，在局部知识上构建高阶全局模式[^src-maf]。

消融中，去掉多尺度（换 vanilla Transformer 编码器）或去掉相对位置（改绝对位置）均显著损害 CRPS-sum / MSE，且相对位置与多尺度应作为整体使用[^src-maf]。解码器侧改用多尺度收益有限，故 MANF 默认仅编码器多尺度、解码器 vanilla attention[^src-maf]。

## 与相关技术

| 技术 | 关系 |
|------|------|
| 全局 Transformer 注意力 | 无窗口归纳偏置；MANF 用尺度窗口约束 |
| 绝对正弦 PE | MANF 在多尺度下用可学习相对位置 |
| [[multi-scale-linear-prediction]] 等 | 同属多粒度时序建模，但机制为下采样/线性混合而非注意力窗口 |
| [[multi-granularity-sea-ice-forecasting]] | 多粒度海冰预测：不同时间粒度的 SIC 作为独立 variate 跨粒度建模 |
| [[ar-vs-nar-decoding]] | 多尺度注意力在 MANF 中服务 **NAR** 条件流生成 |

## 相关页面

- [[manf]] / [[source-maf]]
- [[normalizing-flow]]
- [[generative-time-series-forecasting]]
- [[multi-granularity-sea-ice-forecasting]] — 跨粒度注意力与多粒度预测

[^src-maf]: [[source-maf]]
