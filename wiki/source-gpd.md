---
title: "Source: GPD — Spatio-Temporal Few-Shot Learning via Diffusive Neural Network Generation"
type: source-summary
tags:
  - source
  - spatio-temporal
  - few-shot-learning
  - diffusion-models
  - hypernetwork
created: 2026-06-01
last_updated: 2026-06-01
source_count: 1
confidence: high
status: active
---

# Source: GPD (ICLR 2024)

> **Yuan Yuan\*, Chenyang Shao\*, Jingtao Ding†, Depeng Jin, Yong Li†** — Tsinghua FIB Lab, ICLR 2024. arXiv: 2402.11922.

## 核心贡献

GPD（Generative Pre-training framework based on Diffusion）提出了一个全新的 spaio-temporal few-shot learning 框架。与传统方法直接拟合时空数据不同，GPD 创新性地在**参数空间**上进行生成式预训练[^src-gpd]：

1. **参数空间预训练**：先在多个 source cities 上训练每个 region 的专用预测模型（如 STGCN/GWN/STID），保存优化后的模型参数；然后用 Transformer-based 扩散模型（作为 hypernetwork）学习从 prompt 条件生成模型参数的能力[^src-gpd]。
2. **扩散 hypernetwork**：采用 DDPM 框架训练 Transformer 去噪网络，将模型参数 vectorize 后通过 GCD-based chunking 转为统一 token 序列，以 spatial prompt（基于 UKG 的 KG 嵌入）和 temporal prompt（MAE-style 自监督时序编码）为条件[^src-gpd]。
3. **模型无关（model-agnostic）**：可与 STGCN、GWN、STID 三种不同架构的 base prediction model 兼容[^src-gpd]。
4. **跨城市知识迁移**：预训练完成后的扩散模型被迁移到 target city，利用 target prompt 直接生成该 city 每一个 region 的专属模型参数，仅需极少目标域数据（如 3 天）即可推理[^src-gpd]。

## 关键结果

- 4 个数据集（crowd flow: Washington D.C., Baltimore; traffic speed: METR-LA, Didi-Chengdu）上平均较最优 baseline 降低 7.87% 误差[^src-gpd]
- 长期预测优势明显：Baltimore 第 6 步 MAE 降低 22.1%（vs STGFSL），多 source city 预训练持续提升性能[^src-gpd]
- 训练成本：单卡 RTX 4090 ~3h（DDPM 500 步去噪）[^src-gpd]

## 局限性

- 依赖 region-level 独立训练产生预训练数据，source cities 需有足够数据训练每个 region 的模型[^src-gpd]
- DDPM 500 步采样推理较慢；论文指出可切换到 DDIM 加速但未实验验证[^src-gpd]
- 仅支持 graph-based 时空数据（node=edge=路网传感器），不直接支持 grid-based 数据[^src-gpd]
- 要求 target city 至少拥有少量数据（3 天）用于 prompt 提取，无法实现纯零样本[^src-gpd]

## 区别于其他扩散+时空工作

GPD 的扩散模型用作 **hypernetwork（生成网络参数）**，而非直接对时空序列做扩散预测。这与 [[diffstg|DiffSTG]]（统一历史+未来做条件扩散去噪）和 [[specstg|SpecSTG]]（谱域扩散生成未来傅里叶表示）在应用扩散的方式上本质不同[^src-gpd]。

[^src-gpd]: [[source-gpd]]
