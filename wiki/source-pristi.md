---
title: "PriSTI: A Conditional Diffusion Framework for Spatiotemporal Imputation"
type: source-summary
tags:
  - diffusion-models
  - spatiotemporal-imputation
  - conditional-diffusion
  - air-quality
  - traffic
  - ddpm
created: 2026-05-31
last_updated: 2026-05-31
source_count: 1
confidence: medium
status: active
---

# Source: PriSTI

**作者**: Mingzhe Liu, Han Huang, Hao Feng, Leilei Sun, Bowen Du, Yanjie Fu
**发表**: arXiv:2302.09746, 2023年2月（ICDE 2023）
**机构**: 北京航空航天大学 / University of Central Florida

## 核心论点

PriSTI 提出了一个面向时空缺失值插补的条件扩散框架，核心贡献在于将"条件信息的构建与利用"从 [[csdi|CSDI]] 的简单拼接升级为"先提取条件先验、再引导噪声去噪"的分离式设计[^src-pristi]。[[csdi|CSDI]] 直接将观测值与加噪目标拼接输入 Transformer，噪声信息干扰了时空依赖的学习[^src-pristi]。PriSTI 的回应是：先用线性插值增强条件信息获得粗粒度但干净的全局上下文先验 $H_{pri}$，再以此先验引导噪声估计模块中的注意力计算——Q 和 K 来自干净的 $H_{pri}$，仅 V 来自噪声输入，从而在所有扩散步上隔离了噪声对注意力权重的污染[^src-pristi]。

## 关键创新

### 条件信息增强与提取

对原始观测值进行线性插值得到增强条件信息 $\bar{X}$，再通过条件特征提取模块（单层宽网络，三路并行的时空注意力 + 图消息传递）输出全局上下文先验 $H_{pri}$[^src-pristi]。线性插值的价值不在于插补精度，而在于其确定性、无噪声、保持时序一致性——在连续缺失（block-missing）场景下尤其有效[^src-pristi]。

### 先验引导注意力

噪声估计模块（4 层深网络）的核心设计：时间注意力和空间注意力的 Q、K 均来自干净的 $H_{pri}$，仅 V 来自混合输入 $H_{in}$——"往哪儿看"由先验决定，"看到什么"由当前噪声输入提供[^src-pristi]。这彻底隔离了高噪声扩散步下注意力权重的随机化问题。

### 虚拟节点降采样

将 N 个节点的空间注意力映射到 k 个虚拟节点，复杂度从 $O(N^2 d)$ 降至 $O(Nkd)$（k=16~64）[^src-pristi]。

## 实验与结果

在三个真实数据集上验证：AQI-36（36 站 PM2.5）、METR-LA（207 传感器交通速度）、PEMS-BAY（325 传感器交通速度）[^src-pristi]。MAE 全面超越 [[csdi|CSDI]]：AQI-36 9.03 vs 9.51（-5.0%），METR-LA block-missing 1.86 vs 1.98（-6.1%），PEMS-BAY block-missing 0.78 vs 0.86（-9.3%）[^src-pristi]。CRPS 同样全面优于 [[csdi|CSDI]]。高缺失率（90%）下优势更大（MAE 提升 4.67%-34.11%），传感器完全故障场景下可基于纯空间信息插补[^src-pristi]。成本是训练时间比 [[csdi|CSDI]] 多 25.7%，推理时间多 17.9%[^src-pristi]。

## 局限性

线性插值在高度非线性时序模式（如交通速度的早晚高峰脉冲）中可能错误平滑突变信号[^src-pristi]；邻接矩阵为静态地理距离，无法捕捉动态空间关系（如风向输送、早晚高峰路网连通性变化）[^src-pristi]；三个实验数据集均为单变量数据（每个节点仅一个特征），多变量场景未经验证[^src-pristi]。

## 后续影响

PriSTI 建立了"条件先验与去噪过程分离"的设计范式，成为后续高层次时空扩散模型的标准配置[^src-pristi]。其先验引导注意力的思路影响了 DiffSTG、SpecSTG 等时空扩散方法[^src-pristi]。

[^src-pristi]: [[source-pristi]]
