---
title: "MTP: Multimodal Urban Traffic Profiling with Modality Augmentation and Spectrum Fusion"
type: source-summary
tags:
  - multimodal
  - time-series
  - traffic
  - classification
  - frequency-domain
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: high
status: active
---

# MTP: Multimodal Urban Traffic Profiling

## 基本信息

- **作者**: Haolong Xiang, Peisi Wang, Xiaolong Xu, Kun Yi, Xuyun Zhang, Quanzheng Sheng, Amin Beheshti, Wei Fan
- **机构**: Nanjing University of Information Science and Technology / Nanjing University / State Information Center / Macquarie University / University of Auckland
- **来源**: arXiv:2511.10218 (submitted to AAAI 2026)
- **代码**: https://github.com/jorcy3/MTP

## 核心贡献

MTP 提出首个用于城市交通状态分类的多模态框架，通过模态增强（将数值时间序列转化为视觉和文本模态）和频谱融合实现三视角学习。

### 三大编码器

1. **时序模态编码器**：语义嵌入 → FFT → 频域 MLP（复数权重）→ IFFT，在频域中提取多尺度和周期性特征
2. **视觉模态编码器**：将时间序列转化为频率图像 + 周期性图像 → 多尺度卷积 → FFT → FIR 滤波器（Hamming 窗）+ 平均池化去噪 → 跨模态频谱增强 → IFFT
3. **文本模态编码器**：LLM 生成描述性文本（主题 + 背景 + 项目描述）→ 文本编码器 → FFT → 频域去噪 + 跨模态增强 → IFFT

### 分层对比融合

- **监督对比**：同类实例的不同模态嵌入拉近
- **无监督 InfoNCE**：跨模态特征对齐
- **JS 散度分布相似性融合**：基于后验概率分布相似性加权融合

## 实验结果

6 个公开数据集（Chinatown、Melbourne、PEMS-BAY、METR-LA、DodgerLoop、PEMS-SF），8 个基线（TST、ShapeNet、PatchTST、SVP-T、LightTS、ModernTCN、CAFO、InterpGN），MTP 在多数数据集上 SOTA。消融实验验证三模态各有独立贡献，视觉分支对波动性强的数据集（DodgerLoop）贡献最大，去除视觉分支后 F1 从 0.585 骤降至 0.105。

## 局限性

仅处理分类任务，未扩展到预测或生成任务；文本增强依赖 LLM 生成质量；尚未探索细粒度跨模态关系建模。

## 引用

[^src-mtp]: [[source-mtp]]
