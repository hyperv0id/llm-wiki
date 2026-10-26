---
title: "On Layer Normalization in the Transformer Architecture"
type: source-summary
tags:
  - transformer
  - layer-normalization
  - optimization-stability
  - initialization
  - warmup
created: 2026-10-26
last_updated: 2026-10-26
source_count: 1
confidence: medium
status: active
---

# On Layer Normalization in the Transformer Architecture

## 论文信息

- **标题**: On Layer Normalization in the Transformer Architecture
- **作者**: Ruibin Xiong, Yunchang Yang, Di He, Kai Zheng, Shuxin Zheng, Chen Xing, Huishuai Zhang, Yanyan Lan, Liwei Wang, Tie-Yan Liu
- **发表**: ICML 2020，PMLR 108（Vienna）
- **arXiv**: 2002.04745v2（2020-06-29；正文 §1–§5 + 附录 A–H）
- **raw**: `raw/on-layer-normalization-in-the-transformer-architecture-2002-04745.pdf`

## 问题

训练 Post-LN Transformer 必须使用学习率预热（warmup），否则优化发散；warmup 拖慢训练，$T_{\mathrm{warmup}}$ 且为敏感超参。论文将原因定位到层归一化的**位置**：初始化时的梯度尺度取决于 LN 位于残差和之前还是之后，与优化器无关[^src-xiong-2020-layernorm-placement]。

## 机制

两种放置仅差一处次序：Post-LN 为「子层 → 加残差 → LN」，Pre-LN 为「LN → 子层 → 加残差」，且 Pre-LN 在最后一层后单独增加一次归一化（output LN）再送入 softmax（Table 1、Figure 1）[^src-xiong-2020-layernorm-placement]。

定理 1 给出初始化时最后一层 FFN 权重 $W^{2,L}$ 的梯度范数：Post-LN 为 $\mathcal{O}(d\sqrt{\ln d})$，与深度 $L$ 无关；Pre-LN 为 $\mathcal{O}(d\sqrt{\ln d/L})$，受 $\sqrt{L}$ 压小[^src-xiong-2020-layernorm-placement]。关键量是 LN 雅可比的谱范数 $\|\mathbf{J}_{LN}(x)\|_2=\mathcal{O}(\sqrt{d}/\|x\|_2)$。Post-LN 每层残差和的幅值维持在 $\sqrt{d}$ 量级，层间雅可比逐层收缩 $\sqrt{2/3}$，梯度沿 $\prod_{j=l+1}^{L}$ 连乘按 $(2/3)^{(L-l)/2}$ 向浅层衰减，靠近输出的一层始终是 $\mathcal{O}(d\sqrt{\ln d})$；Pre-LN 残差流不经归一化，$\mathbb{E}\|x^{pre}_l\|_2^2$ 随 $l$ 线性增长，层间雅可比近似恒等，由出口归一化将整条梯度除以 $\sqrt{L}$（附录 F）[^src-xiong-2020-layernorm-placement]。

## 证据

IWSLT14 De-En（6+6 层、$d=512$、4 头、Adam $\beta=(0.9,0.98)$）：Post-LN 去掉 warmup 后 BLEU 仅 8.45，带 warmup 约 34；$T_{\mathrm{warmup}}$ 降至 500 时，$\mathrm{lr}_{\max}=5\times10^{-4}$ 为 31.16，$1\times10^{-3}$ 降至 2.77（Figure 2a）。SGD 同样无法收敛，无 warmup 下 15 个 epoch 内 BLEU 仅略高于 0，表明 Adam 的方差项并非 warmup 必要的前提（Figure 2b）[^src-xiong-2020-layernorm-placement]。Pre-LN 在无 warmup、$\mathrm{lr}=5\times10^{-4}$ 时 BLEU 约 34、验证损失约 4，第 9 个 checkpoint 追平 Post-LN 第 15 个（Figure 4a–b）；$\mathrm{lr}=7.5\times10^{-4}$、8000 步后衰减、dropout 0.3 配置下达 35.5 test BLEU（脚注 1）[^src-xiong-2020-layernorm-placement]。BERT base（12 层）：Post-LN 采用 10k 步 warmup 与 $\mathrm{lr}_{\max}=10^{-4}$，Pre-LN 无 warmup 并从 $3\times10^{-4}$ 线性衰减；Post-LN 在 500k 步达到 1.69 验证损失，Pre-LN 在 700k 步达到同等水平[^src-xiong-2020-layernorm-placement]。

## 边界

定理条件包括单头注意力、全部参数 $d\times d$、$W^{Q}=W^{K}=0$（注意力退化为全序列均值）、输入同分布高斯、Xavier 初始化、$\gamma=1$、$\beta=0$、导数有界与隐藏态范数 $(\epsilon,\delta)$-bounded，结论仅覆盖最后一层的 $W^{2,L}$；浅层按 $(2/3)^{(L-l)/2}$ 衰减出自附录 F 的初步分析[^src-xiong-2020-layernorm-placement]。从「层末梯度大」到「大学习率发散」缺乏解析 $\mathrm{lr}$ 上限，实验显示：warmup 后梯度变小，固定 $10^{-4}$ 小步长可训练但收敛明显变慢（附录 H、Figure 6）[^src-xiong-2020-layernorm-placement]。IWSLT 上两变体终值表现相当（comparable）；WMT14 En-De 与 BERT 的终值仅展示曲线（Figure 4c–d、Figure 5a）。

完整推导见 [[pre-ln-vs-post-ln-analysis]]。

[^src-xiong-2020-layernorm-placement]: [[source-xiong-2020-layernorm-placement]]
