---
title: "Source: MiniTraffic"
type: source-summary
tags:
  - traffic-forecasting
  - pre-training
  - contrastive-learning
  - frequency-domain
  - fine-grained
  - lane-level
created: 2026-07-16
last_updated: 2026-07-19
source_count: 1
confidence: medium
status: active
---

# Source: MiniTraffic

**Full title**: Being More Lightweight and Practical: Mini-sized Contrastive Learning Pre-trained Models for Fine-grained Traffic Task

**Authors**: Shuhao Li, Weidong Yang (Fudan University & Zhuhai Fudan Innovation Research Institute), Ben Fei (CUHK), Yue Cui (Alibaba Tongyi Lab), Lipeng Ma (Fudan), Fan Zhang (Guangzhou University)

**Venue**: ICML 2026 (Proceedings of the 43rd International Conference on Machine Learning, PMLR 306)

**Code**: <https://github.com/ShuhaoLii/Mini-Traffic>

## 核心贡献

MiniTraffic 是首个专为细粒度交通预测（道路级 + 车道级统一预测）设计的轻量级预训练模型，仅 ~119K 可训练参数，可在单张 A100 GPU 上完成预训练[^src-minitraffic]。解决了三个关键挑战：(1) 道路级数据丰富但车道级数据稀缺的不平衡问题；(2) 道路-车道多粒度统一建模；(3) 大型预训练模型部署成本过高[^src-minitraffic]。

## 方法要点

1. **Frequency Domain Stability Augmentation (FDA)**：在频域对道路级数据施加有界扰动（幅值阈值 λ·max A(f) + 选择性频谱掩码 Γ(f)），模拟车道级变异性同时保持频谱一致性，从丰富道路数据生成伪车道模式[^src-minitraffic]。

2. **Contrastive Clustering Graph Partitioning**：基于 InfoNCE 损失学习 patch 间余弦相似度，构建 k-NN 稀疏图。将图注意力复杂度从 O(N²) 降至 O(k·N)，同时从语义相似度（而非空间邻近性）动态构建图结构，保留长程依赖[^src-minitraffic]。

3. **Random Patch Mask**：对时序 patch 做随机掩码（最优比例 40%），结合 Instance Normalization 增强鲁棒性和迁移能力[^src-minitraffic]。

4. **Granularity-Aware Fine-Tuning**：道路级微调用 Extension + Pooling 模块替换 FDA；车道级微调保留 FDA，仅重新训练轻量级 Adaptive Head 和 Reduction Head[^src-minitraffic]。

## 实验与结论

在 6 个细粒度数据集（PeMS-Lane/Road、PeMSF-Lane/Road、HuaNan-Lane/Road）上与 29 个基线模型对比：车道级 MAE 降低 7%–39%，道路级 FLOPs 降低约 85%[^src-minitraffic]。消融实验证实 FDA（MAE +9%–11%）、Contrastive Clustering（参数从 820K 降至 119K）和 Extension & Pooling 的各自贡献[^src-minitraffic]。跨城市迁移实验揭示粒度对齐比同城更重要[^src-minitraffic]。

## 局限性

仅使用速度作为预测目标，未涉及其他交通状态（流量、密度）。跨城市迁移虽有粒度对齐规律，但绝对误差仍然较高，暗示分布偏移问题尚未完全解决[^src-minitraffic]。

[^src-minitraffic]: [[source-minitraffic]]
