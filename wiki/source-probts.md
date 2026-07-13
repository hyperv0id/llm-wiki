---
title: "ProbTS: Benchmarking Point and Distributional Forecasting across Diverse Prediction Horizons"
type: source-summary
tags:
  - time-series
  - benchmark
  - probabilistic-forecasting
  - point-forecasting
  - foundation-model
  - neurips-2024
created: 2026-07-13
last_updated: 2026-07-13
source_count: 1
confidence: medium
status: active
---

# Source: ProbTS

**作者**: Jiawen Zhang (HKUST(GZ)), Xumeng Wen (MSRA), Zhenwei Zhang (Tsinghua), Shun Zheng (MSRA), Jia Li (HKUST(GZ)), Jiang Bian (MSRA)  
**发表**: NeurIPS 2024 Datasets and Benchmarks Track  
**arXiv**: 2310.07446v5 (2024-10-21)  
**代码**: https://github.com/microsoft/ProbTS  
**领域**: 时间序列点预测与概率预测统一基准

## 核心论点

既有时序深度学习工作常割裂为两条线：长程点预测（强趋势/季节性、定制架构、多为 [[generative-style-decoder|NAR]]）与短程概率估计（复杂分布、生成模型、AR/NAR 并存）[^src-probts]。这种窄场景聚焦会扭曲方法选择，使模型难以迁移到未充分评估的设定。近年时间序列基础模型虽宣称任意 horizon 与零样本能力，但其在点/分布预测与短/长 horizon 上的优势与代价仍缺乏系统刻画[^src-probts]。

[[probts|ProbTS]] 将上述“本质预测需求”置于统一平台：覆盖短程与长程数据集与 horizon，同时报告点指标（NMAE 等）与分布指标（CRPS / CRPS-sum），并量化窗口内 **趋势强度 $F_T$、季节强度 $F_S$、非高斯性（Jensen–Shannon 相对高斯）** 三类数据特征，以解释方法偏好[^src-probts]。方法轴上显式区分：(1) 分布估计（点头 / 预定义分布头 / 神经生成模块）；(2) [[ar-vs-nar-decoding|AR vs NAR 解码]]；(3) 归一化（[[instance-normalization|RevIN]] vs 均值缩放）[^src-probts]。

## 主要发现

1. **定制长程架构在短程失效**：PatchTST / iTransformer / DLinear 等在长程 NMAE 上强势，但短程、尤其高 [[non-gaussianity|非高斯性]] 数据（如 Solar-S）上被概率模型拉开；CRPS 差距更显著[^src-probts]。
2. **既有概率模型长程分布预测崩溃**：TimeGrad、GRU-NVP、CSDI 等在长 horizon 上 CRPS 显著恶化——AR 受误差累积（随 horizon 与趋势增强）困扰，NAR 扩散（CSDI）受显存与学习效率限制[^src-probts]。
3. **AR 在强季节性上出人意料地优**：如 Traffic 上 AR（TimeGrad）可优于 PatchTST；强季节时 AR 优势增大，提示在解决误差累积后 AR 仍有长程价值[^src-probts]。
4. **RevIN 主要利好长程 AR**：RevIN 显著缓解 AR 长程误差累积（ETTh1 上 GRU-NVP+RevIN 甚至超过 PatchTST+RevIN），但对强季节弱趋势的 Traffic 有负作用；短程概率设定中均值缩放更稳，RevIN 不占优[^src-probts]。
5. **基础模型复现同样规律**：零样本下 TimesFM/Timer 等 AR 基础模型短程有竞争力，长 horizon 相对 MOIRAI 等 NAR 模型劣势扩大；MOIRAI/Chronos 等在高非高斯性上相对 CSDI 的 CRPS 落差更大，预定义混合分布头表达力不足[^src-probts]。

## 贡献

1. 发布统一基准工具 [[probts|ProbTS]]，覆盖点+分布、短+长 horizon，并集成经典模型与可复现基础模型[^src-probts]。
2. 以数据特征 × 方法决策（分布头 / 解码 / 归一化）框架解释跨研究线的优劣迁移失败[^src-probts]。
3. 将分析延伸到 TSFM，指出 AR 误差累积与复杂分布建模仍是开放问题[^src-probts]。

## 局限性

- 以经验分析为主，理论深度有限[^src-probts]。
- 主要聚焦 AR/NAR 与归一化等粗粒度决策，可能忽略其他关键因素[^src-probts]。
- 评估数据集虽多样，仍难覆盖真实世界全谱；基础模型预训练数据差异可能扭曲零样本比较[^src-probts]。

## 相关页面

- [[probts]] — 基准工具实体
- [[ar-vs-nar-decoding]] — AR/NAR 解码方法轴
- [[non-gaussianity]] — 非高斯性量化指标
- [[timegrad]] / [[csdi]] / [[patchtst]] / [[timesfm]] / [[chronos]]
- [[instance-normalization]] — RevIN 与均值缩放对照
- [[generative-time-series-forecasting]]

[^src-probts]: [[source-probts]]
