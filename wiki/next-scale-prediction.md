---
title: "Next-Scale Prediction"
type: technique
tags:
  - autoregressive-modeling
  - multi-scale
  - vqvae
  - image-generation
  - arxiv-2024
created: 2026-10-23
last_updated: 2026-10-23
source_count: 3
confidence: medium
status: active
---

# Next-Scale Prediction

**Next-scale prediction（下一尺度预测）** 是 VAR 论文提出的视觉自回归范式：自回归的单位从"下一个 token"换成"下一张更细分辨率的 token map"，尺度间由粗到细自回归，尺度内所有 token 并行生成[^src-2404-02905-var]。

## 机制

图像经多尺度 VQVAE 编码为 K 个 token map $(r_1,\dots,r_K)$（分辨率递增，$r_K$ 达到特征图原分辨率），生成似然分解为[^src-2404-02905-var]：

$$p(r_1,\dots,r_K)=\prod_{k=1}^{K}p(r_k\,|\,r_1,\dots,r_{k-1})$$

- **训练**：GPT-2 式 decoder-only Transformer，输入 $([s],r_1,\dots,r_{K-1})$ 预测 $(r_1,\dots,r_K)$，标准 cross-entropy；用**块状因果注意力掩码**保证每个 $r_k$ 只能看见 $r_{\le k}$（同一尺度内 token 互相可见）。
- **推理**：从 1×1 token map 起步逐尺度生成（ImageNet 256×256 共 10 步），可使用 kv-cache。
- **Tokenizer**：VQGAN 架构 + 多尺度残差量化——特征图插值到第 k 尺度分辨率后查共享码本（V=4096），查得向量上采样回原分辨率、经卷积 $\phi_k$ 后从特征图残差中扣除；额外参数约 0.03M。

## 动机：raster-scan next-token AR 的四个问题

论文对"离散化 + 拉平成 1D 序列"的传统图像 AR 给出四点批评[^src-2404-02905-var]：

1. **数学前提违反**：VQVAE 编码器特征图 token 双向相关（附录 A attention 热图证据），与 AR 的单向依赖假设矛盾；next-scale 约束 $r_k$ 只依赖前缀尺度后前提成立。
2. **零样本泛化受限**：单向模型无法做"给下半图补上半图"类双向任务；尺度级 teacher-forcing 天然支持。
3. **结构退化**：flatten 破坏空间局部性；next-scale 不拉平，尺度内 token 全相关。
4. **低效**：n×n 潜图 AR 需 O(n²) 步、O(n⁶) 计算；next-scale 需 O(log n) 步、O(n⁴) 计算（附录 B 引理证明）。

## 证据（作者报告）

- 同设置对比：AR（VQGAN 227M）FID 18.65 → next-scale（VAR-d16 207M）5.22，推理成本降至 0.013×（Table 3 前两行）[^src-2404-02905-var]。
- ImageNet 256×256 上 VAR-d30（2.0B，10 步）FID 1.92、+rejection sampling 1.73（IS 350.2）；论文自称 GPT-style AR 首次超越扩散 Transformer（DiT-XL/2 2.27，45× 推理时间）；相比 VQGAN/ViT-VQGAN 约 20× 加速[^src-2404-02905-var]。
- **Scaling laws**：12 个模型（18.5M–2B）上 test loss 与 token 错误率随 N 与 C_min 幂律下降（$L_{last}=(2.0N)^{-0.23}$、$L_{avg}=(2.5N)^{-0.20}$、$Err_{last}=(4.9\cdot10^2 N)^{-0.016}$ 等），Pearson ≈ −0.998，跨 C_min 六个数量级成立；论文将其定位为视觉模型初步复现 LLM 的 scaling law 性质[^src-2404-02905-var]。
- **零样本泛化**：in-painting / out-painting / class-conditional editing，仅靠 teacher-force mask 外 token，无架构修改、无微调[^src-2404-02905-var]。

## 与其他"改自回归单位"路线的对照

三条已入 wiki 的路线共享"自回归建模组间依赖 + 组内并行"的结构，差别在分组方式与表示：

| 路线 | 分组方式 | token 表示 | 每组分布建模 |
|:-----|:--------|:----------|:------------|
| [[masked-autoregressive-model\|MAR]]（NeurIPS 2024） | 随机顺序的 token 子集（next set-of-tokens） | 连续潜变量（KL-16 等） | per-token 扩散头（[[diffusion-loss\|Diffusion Loss]]） |
| **next-scale（VAR）** | 由粗到细的尺度层级 | 离散 VQ 码本索引 | softmax 分类分布 |
| [[scale-wise-autoregressive-decoding\|TrajAR 解码]] | 轨迹预测的尺度层级 | 连续坐标 | RMSE 回归（无量化） |

VAR 与 MAR 同期、不同团队（PKU/ByteDance vs MIT），两文互不引用；[[trajar|TrajAR]] 明确自述受 VAR 启发并把量化环节换成三次插值 + 连续回归，[[climatear|ClimateAR]] 则沿用 VAR 的离散 VQ + 逐尺度交叉熵路线并加上混合尺度条件控制。

## 移植记录

- [[climatear]]（ICML 2026）：论文自称首个把 VAR 范式引入概率气候预测（VQ 分词器对齐 + [[mixed-scale-conditioning|混合尺度条件控制]]）[^src-climatear]。
- [[scale-wise-autoregressive-decoding]]（TrajAR，IJCAI-26）：next-scale 思路在路口轨迹预测的连续坐标版移植[^src-trajar]。

## 相关页面

- [[source-2404-02905-var]] — 论文摘要
- [[masked-autoregressive-model]] — 同期"随机组"路线（MAR）
- [[diffusion-loss]] — MAR 的连续分布头
- [[scale-wise-autoregressive-decoding]] — 轨迹预测移植
- [[climatear]] — 气候预测移植
- [[dit]] — 被对照的扩散 Transformer

[^src-2404-02905-var]: [[source-2404-02905-var]]
[^src-climatear]: [[source-climatear]]
[^src-trajar]: [[source-trajar]]
