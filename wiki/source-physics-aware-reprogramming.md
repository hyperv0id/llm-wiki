---
title: "Language Model Empowered Spatio-Temporal Forecasting via Physics-Aware Reprogramming (REPST)"
type: source-summary
tags:
  - spatio-temporal
  - llm-for-ts
  - model-reprogramming
  - koopman
  - few-shot
created: 2026-09-15
last_updated: 2026-09-15
source_count: 1
confidence: medium
status: active
---

# REPST: Language Model Empowered Spatio-Temporal Forecasting via Physics-Aware Reprogramming

**Authors**: Hao Wang, Jindong Han, Wei Fan, Leilei Sun, Hao Liu (HKUST-GZ, Shandong, Oxford, Beihang)
**Venue**: arXiv 2024（所读版本题为 REPST ... via Semantic-Oriented Reprogramming，结论仍称 physics-aware spatio-temporal decomposer）
**Code**: https://github.com/usail-hkust/REPST
**Local**: downloads/physics-aware-reprogramming.txt

## 核心论点

PLM-based 时序预测器（FPT、Time-LLM）把时空数据当一维序列做过于简化的编码，PLM 的推理能力未被利用。探索实验支持这一诊断：在 reprogrammed GPT-2 上仅加 Fourier 分解，METR-LA MAE 从 2.68 降到 2.34（12.69%），PEMS-BAY 从 6.33 降到 5.82（8.06%）[^src-physics-aware-reprogramming]。

## 方法机制

1. **Koopman 分解器**：RevIN 逐节点归一化后，用演化矩阵 $X_{2:t} = A X_{1:t-1}$ 做 SVD，得 $C$ 个模态 $\omega_i$ 与特征值 $v_i$，重构 $X_{dyn} = \epsilon_i e^{\omega_i t} v_i$；再按能量贡献（初始幅值 $\times \mathrm{Re}(v_i)$）取 top-k 模态重构降噪信号 $X_{rec}$。相比 Fourier 频率强度分解，模态来自数据真实动力学且可解释（红绿灯周期振荡、风向驱动的污染缓变）[^src-physics-aware-reprogramming]。
2. **Patch 编码**：非重叠 patch（conv kernel 3、embedding 维度 64），每 patch 一个 token，区别于 node-as-token 设计，保留细粒度时空语义[^src-physics-aware-reprogramming]。
3. **Selective discrete reprogramming**：学习词掩码 $m = \mathrm{Softmax}(EW)$，用 Gumbel-Softmax（温度 $\tau$，$g_i \sim \mathrm{Gumbel}(0,1)$）可微地 top-K 采样 1000 个词构成扩展时空词汇表 $E'$，patch embedding 作 query 对 $E'$ 做 cross-attention，冻结 GPT-2 backbone（3/6/9/12 层可选）编码，可学习 projection 输出预测。复杂度 $O(N \cdot P \cdot V' + (N \cdot P)^2)$[^src-physics-aware-reprogramming]。

## 实验证据

6 数据集（METR-LA、PEMS-BAY、Beijing Taxi、NYC Bike、Air Quality 35 站 6 指标、Solar Energy 137 光伏站），12 基线（Informer、iTransformer、PatchTST、MTGNN、GWNet、STID、STAEFormer、FPT、Time-LLM、TimesFM、OpenCity 等），BasicTS 复现[^src-physics-aware-reprogramming]。

- Full training（$T = \tau = 24$）：METR-LA MAE 3.63/RMSE 7.43（RMSE 全场最低，MAE 仅次于 STAEFormer 3.60）；PEMS-BAY MAE 1.92/RMSE 4.33 最优；Air Quality MAE 3.27 最优；PLM-based FPT 明显落后（METR-LA 6.03）[^src-physics-aware-reprogramming]
- Few-shot（仅 1 天 <1% 训练数据）：METR-LA MAE 5.63 vs 最优基线 STAEFormer 6.35；PEMS-BAY 3.61 vs 4.55[^src-physics-aware-reprogramming]
- Zero-shot 跨域迁移：NYC→CHI Bike MAE 2.03 vs TimesFM 9.07、Time-LLM 10.32；Solar→Air 31.86 vs Time-LLM 68.44[^src-physics-aware-reprogramming]
- 消融：替换 GPT-2 为 transformer、替换 Koopman 分解为 transformer encoder、去掉扩展词汇表改用 dense mapping，三者均使性能下降[^src-physics-aware-reprogramming]

## 局限

- 注意力 $O((N \cdot P)^2)$ 随节点数增长，需对空间图划分分块训练；大 $N$ 场景成本未消除[^src-physics-aware-reprogramming]
- Koopman 线性算子假设对强非线性系统可能欠拟合；zero-shot 对比池小，Solar→CHI 上输给用交通数据预训练的 OpenCity-base[^src-physics-aware-reprogramming]

## 关联

- 扩展 [[model-reprogramming]] 范式：dense 映射 → 离散词汇采样
- 细化 [[patch-reprogramming]]（[[source-time-llm]] 的 text prototype cross-attention）：加入 Gumbel-Softmax top-K 选择性词汇扩展

[^src-physics-aware-reprogramming]: [[source-physics-aware-reprogramming]]
