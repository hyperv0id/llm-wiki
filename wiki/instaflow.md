---
title: "InstaFlow"
type: technique
tags:
  - instaflow
  - rectified-flow
  - one-step-generation
  - diffusion-distillation
  - text-to-image
  - iclr-2024
created: 2026-05-31
last_updated: 2026-05-31
source_count: 1
confidence: medium
status: active
---

**InstaFlow** 是将大规模 [[diffusion-model|Stable Diffusion]] 蒸馏为一步文生图模型的方法，由 Xingchao Liu 等提出（UT Austin / UIUC / Tsinghua，ICLR 2024）。核心发现：**直接蒸馏 SD 完全失败**——reflow 不是可选的优化，而是蒸馏成功的必要前提[^src-instaflow]。

## 核心流程

```
SD (v₁) → Text-Conditioned Reflow (v₂, v₃, ...) → Distill (ṽₖ) → 一步模型
```

### Step 1: Text-Conditioned Rectified Flow (Reflow)

给定预训练 SD 作为 $v_1$，用其 ODE 采样端点作为训练数据训练 $v_2$[^src-instaflow]：

$$v_{k+1} = \arg\min_v \mathbb{E}_{X_0\sim\pi_0, T\sim D_T}\left[ \int_0^1 \|(X_1 - X_0) - v(X_t, t | T)\|^2 dt \right]$$

其中 $X_1 = \text{ODE}[v_k](X_0 | T)$（上一轮 ODE 从噪声生成图像），$X_t = t X_1 + (1-t)X_0$（线性插值），$D_T$ 是文本提示数据集[^src-instaflow]。

Reflow 同时做到三件事：(a) 拉直轨迹（直度 $S(Z)$ 下降），(b) 改善噪声-图像配对（降低凸传输代价 $E[c(Z_1-Z_0)]$），(c) 保持边际分布不变[^src-instaflow]。

### Step 2: 蒸馏

$$\tilde{v}_k = \arg\min_v \mathbb{E}\left[ D(\text{ODE}[v_k](X_0|T), X_0 + v(X_0|T)) \right]$$

蒸馏固定 $t=0$：输入噪声，一步 Euler 得到图像[^src-instaflow]。采用两阶段损失：
- **L2 loss**（21.5K 步，batch 1024）：快速收敛到大体结构
- **LPIPS loss**（18K 步，batch 1024）：精细调优纹理，对视觉质量提升立竿见影[^src-instaflow]

### Step 3: Classifier-Free Guidance

推理时的 CFG 变体[^src-instaflow]：

$$v^\alpha(Z_t, t | T) = \alpha \cdot v(Z_t, t | T) + (1-\alpha) \cdot v(Z_t, t | \text{NULL})$$

最佳 $\alpha \approx 1.5$，远低于 SD 的 5-7.5——拉直后所需引导减弱[^src-instaflow]。

## 关键实验证据

### 直接蒸馏 SD 失败

SD 1.4 网格搜索 9 组超参数（lr ∈ {1e-5,1e-6,1e-7} × wd ∈ {1e-1,1e-2,1e-3}），训练 100K 步、320 万对数据——最佳 FID-5k=40.9（vs 老师 SD 22.8）[^src-instaflow]。放大 lr→NaN 崩溃，缩小 lr→收敛极慢画面模糊。**这不是调参问题，是任务根��太难**——老师 SD 的映射太复杂太不规律，学生学不会[^src-instaflow]。

### Reflow 后蒸馏质变

相同训练预算（50K reflow + 50K distill）[^src-instaflow]：
- SD+Distill: gap = 40.9 − 22.8 = **18.1**
- 2-RF+Distill: gap = 31.0 − 22.1 = **8.9**（gap 缩小一半）
- 视觉对比：同一噪声下，2-RF+Distill 生成的图与老师 2-RF 高度相似，SD+Distill 与老师 SD 完全不是同一张图（Figure 5）[^src-instaflow]

### 规模放大

基于 SD 1.5，batch 扩大到 1024，总计 ~199 A100 GPU days 训练：
- InstaFlow-0.9B: FID-5k=23.4, FID-30k=13.1 @ 0.09s（追平 StyleGAN-T 的 13.9）[^src-instaflow]
- InstaFlow-1.7B (Stacked U-Net): FID-5k=22.4 @ 0.12s, FID-30k=11.83[^src-instaflow]

**首次纯监督学习的一步扩散模型在质量和速度上同时追平 GAN**[^src-instaflow]。

## Stacked U-Net

为扩大容量而不过度增加推理时间，论文比较了三种结构（Figure 12）[^src-instaflow]：

1. 原始 U-Net（0.9B，0.09s）
2. 两 U-Net 串联共享参数（0.9B 但算量翻倍，0.13s）——蒸馏 loss 显著下降
3. **Stacked U-Net**：在结构 2 上删除冗余模块——去除第一个 U-Net 的下采样块 1、上采样块 1、中间 In+Out Block，保留"第二 U-Net 的下采样块和中间块 + 第一 U-Net 的上采样块"。参数 1.7B，推理 0.12s[^src-instaflow]

## 与相关方法的区别

| 方法 | 途径 | 特点 | vs InstaFlow |
|------|------|------|-------------|
| [[consistency-models|Consistency Models]] | 直接学一步模型 | 蒸馏模式(CD)+独立训练(CT) | CM 不依赖教师模型；InstaFlow 走 reflow+distill 路径 |
| [[shortcut-models|Shortcut Models]] | 自一致性+步长调节 | 单阶段训练 | Shortcut 不依赖外部教师；InstaFlow 的 reflow 保证配对更规律 |
| Progressive Distillation | 层层叠蒸馏 | 512→256→128→...→1 步 | PD 一步 FID=37.2 失败；InstaFlow 加 reflow 后降到 23.4 |
| Rectified Flow (原始) | 从零训练+reflow | CIFAR10/小数据集 | InstaFlow 将其扩展到 0.9B 参数 + 文本条件 + 工业数据集 |

## 训练成本

| 模型 | A100 GPU Days |
|------|--------------|
| SD 1.4 (从头训练) | ~6250 |
| GigaGAN | ~4783 |
| StyleGAN-T | ~1792 |
| InstaFlow (SD 1.4, 2-RF+Distill) | ~24.65 |
| InstaFlow-0.9B | ~199 |
| InstaFlow-1.7B | ~199+39.6 |

从 SD 预训练权重微调大幅降低训练成本——这是其大规模可行的关键[^src-instaflow]。

## 生态兼容性

- **ControlNet 兼容**：预训练 InstaFlow 可直接使用预训练 ControlNet 权重，无需任何微调（Figure 27）——reflow+distill 未破坏 SD 潜空间的语义结构[^src-instaflow]
- **SDXL-Refiner 联动**：InstaFlow (0.09s 出 512) → SDXL-Refiner (精修到 1024)，实现"速度+质量"解耦的两阶段管线[^src-instaflow]

## 相关页面

- [[source-instaflow]] — 论文源文件摘要
- [[rectified-flow]] — Rectified Flow 理论
- [[flow-matching]] — Flow Matching 训练框架
- [[diffusion-model]] — 扩散模型概念
- [[ddpm]] — DDPM，被 InstaFlow 超越的扩散基线
- [[probability-flow-ode]] — 概率流 ODE，SD 底层轨迹
- [[dpm-solver]] — DPM-Solver，SD 的 25 步采样器
- [[consistency-models]] — 另一种一步生成方法
- [[shortcut-models]] — 自一致性少步生成
- [[classifier-free-guidance]] — 文本条件方法，InstaFlow 的 CFG 源头
- [[optimal-transport]] — 最优传输，reflow 中降低凸传输代价

[^src-instaflow]: [[source-instaflow]]
