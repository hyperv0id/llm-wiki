---
title: "Masked Autoregressive Model (MAR)"
type: technique
tags:
  - autoregressive-modeling
  - masked-generative-modeling
  - bidirectional-attention
  - image-generation
  - neurips-2024
created: 2026-10-23
last_updated: 2026-10-23
source_count: 2
confidence: medium
status: active
---

# Masked Autoregressive Model (MAR)

**MAR（Masked Autoregressive model）** 是 MAR 论文提出的广义自回归生成框架：随机顺序 + 双向注意力 + 每步同时预测多个 token，配合 [[diffusion-loss|Diffusion Loss]] 在连续值 token 空间上做图像生成[^src-2406-11838-mar]。

> [!note] 命名消歧
> 此 MAR（masked autoregressive）与 [[manifold-anchor-regularization]]（也缩写 MAR，arXiv 2026）无关联。

## 广义自回归：什么才算"下一个 token"

论文把自回归定义为"基于已知 token 预测未知 token"，并指出它与注意力方向（causal/bidirectional）、预测顺序（raster/random）、每步预测数量（1 个/多个）三者都无关[^src-2406-11838-mar]。masked generative 模型（MaskGIT/MAGE 一系）按随机排列预测 token 子集，可以写成

$$p(x_1,\dots,x_n)=\prod_{k=1}^{K} p(X^k\,|\,X^1,\dots,X^{k-1})$$

即 **next set-of-tokens prediction**——masked generative 建模由此被纳入广义自回归，而非自回归的对立范式。

## 双向注意力如何执行自回归

实现采用 MAE 式非对称结构[^src-2406-11838-mar]：

- **Encoder**：只处理已知 token（序列很短时在开头 padding 64 个 [cls] token 提升稳定性），允许所有已知 token 互相看见；
- **Decoder**：将编码序列与带位置嵌入的 mask token 拼接，未知 token 能看见全部已知 token；损失只在未知 token 上计算；
- **代价与收益**：放弃 causal attention 的 kv-cache，但每步可同时生成多个 token，且全序列注意力带来更强的 token 间通信。

消融链条（MAR-L 400 epochs、ImageNet 256×256、w/o CFG FID，作者报告）[^src-2406-11838-mar]：raster-order 因果 AR 19.23 → 随机顺序 13.07 → 双向注意力 3.43 —— 顺序与注意力方向两步改动的增益都远大于损失函数本身的增益（同设置 cross-entropy 8.79 → Diffusion Loss 3.50）。

## 与 MaskGIT/MAGE 的差异

MaskGIT/MAGE 在推理时按各位置置信度 on-the-fly 选择下一步要预测的位置，温度作用在**位置选择**上，训练与推理的顺序分布不一致；MAR 采用完全随机顺序，温度作用在**每个 token 的分布**上（与 GPT 的 token 级温度采样同构）。作者报告两种顺序策略结果相当（cross-entropy 口径 FID 8.72 vs 8.79），选择随机顺序是为了训练/推理一致性与 token 级温度[^src-2406-11838-mar]。

## 实现要点

- 训练时随机采样 mask ratio ∈ [0.7, 1.0]；encoder 与 decoder 平分全部 Transformer 块（MAR-L 各 16）[^src-2406-11838-mar]。
- 推理按 cosine schedule 将 mask ratio 从 1.0 退火到 0，默认 64 步（系统对比实验用 256 步）[^src-2406-11838-mar]。
- 模型规模：MAR-B/L/H 分别为 24/32/40 块、宽 768/1024/1280（208M/479M/943M）；去噪 MLP 6/8/12 块、宽 1024/1280/1536[^src-2406-11838-mar]。

## 结果与对比（作者报告）

ImageNet 256×256、800 epochs、256 自回归步：MAR-B/L/H w/ CFG FID 2.31/1.78/1.55，对照 DiT-XL/2（675M）2.27、MDTv2-XL/2 1.58、GIVT 3.35、LDM-4 3.60；w/o CFG 报告 2.35[^src-2406-11838-mar]。512×512 上 MAR-L w/ CFG 1.73（DiT-XL/2-G 3.04）。训练成本上，MAR-L 400 epochs 在 16×8 V100 集群约 2.6 天，同设置 DiT-XL/2 需 4.6 天；速度/精度曲线上 MAR（改变自回归步数 8–128）优于 DiT（改变扩散步数 50–250），因为 MAR 的扩散只跑在小 MLP 上，而 DiT 每个扩散步都要过完整 Transformer[^src-2406-11838-mar]。MAR+Diffusion Loss 也能直接建模像素空间（4×4 像素合一个 token，ImageNet 64×64 FID 2.93），但计算代价显著更高[^src-2406-11838-mar]。

## 与其他"改自回归单位"路线的关系

MAR 把自回归单位从"下一个 token"扩为"下一组 token"（随机顺序）；VAR 则同期把自回归单位改为"下一张更细尺度的 token map"（离散 VQ token、由粗到细；ImageNet 256×256 上 2B 模型 FID 1.73、推理约 20× 快于 VQGAN）[^src-2404-02905-var]；[[scale-wise-autoregressive-decoding]] 是 next-scale 思路在轨迹预测的连续坐标移植。三条路线共享"自回归建模组间依赖 + 组内并行生成"的结构，系统对照见 [[next-scale-prediction]]；MAR 与 VAR 同期、不同团队（MIT vs PKU/ByteDance），两文互不引用。

## 相关页面

- [[diffusion-loss]] — per-token 分布建模
- [[source-2406-11838-mar]] — 论文摘要
- [[masked-generative-modeling]] — 掩码生成范式（MAR 将其纳入广义自回归）
- [[mae]] — 双向注意力自回归借用的 encoder/decoder 形式
- [[latent-diffusion-models]] — tokenizer 来源（KL-16/VQ-16）
- [[scale-wise-autoregressive-decoding]] — 另一种扩大自回归单位的思路
- [[next-scale-prediction]] — VAR：尺度分组的姊妹路线（同期不同团队）

[^src-2406-11838-mar]: [[source-2406-11838-mar]]
[^src-2404-02905-var]: [[source-2404-02905-var]]
