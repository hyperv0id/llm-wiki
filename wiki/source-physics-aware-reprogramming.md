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

**Authors**: Hao Wang, Jindong Han, Wei Fan, Leilei Sun, Hao Liu (HKUST-GZ, HKUST, Shandong, Oxford, Beihang)
**Venue**: arXiv 2024（所读版本题为 REPST ... via Semantic-Oriented Reprogramming）
**Code**: https://github.com/usail-hkust/REPST
**Local**: downloads/physics-aware-reprogramming.txt

## 问题

PLM 用于时空预测有两个缺口：现有 PLM-based 时序方法未真正动用 PLM 的推理/生成能力（引 Tan et al. 2024），且 PLM 难以处理图/网格等空间结构。[^src-physics-aware-reprogramming]诊断实验：在 reprogrammed GPT-2 上仅加 Fourier 分解，METR-LA MAE 2.68→2.34（12.69%）、PEMS-BAY 6.33→5.82（8.06%），支持"分解能帮助 PLM 理解时空数据"这一判断[^src-physics-aware-reprogramming]。

## 方法机制

1. **DMD 分解器（Koopman 视角）**：RevIN 逐节点归一化后拟合演化矩阵 $X_{2:t} \approx A X_{1:t-1}$（$A \approx X_{2:t} X_{1:t-1}^+$），特征分解 $A = VDV^{-1}$ 得特征值 $\omega_i$、特征向量 $v_i$，动态分量 $X_{dyn} = \sum_i \epsilon_i e^{\omega_i t} v_i$（$\epsilon_i$ 由 $x_0$ 定）；按能量贡献 $E_i = |\omega_i|^2 \cdot e^{2\mathrm{Re}(v_i)t}$ 取 top-k 主导模态重构降噪信号 $X_{rec}$。[^src-physics-aware-reprogramming]相比 Fourier 频率分解，演化矩阵从系统真实数据导出，可同时捕获周期性与瞬态（指数增/衰减）模态。[^src-physics-aware-reprogramming]
2. **Patch 编码**：每 patch 一个 token（而非 node-as-token），conv kernel 3、embedding 维度 64。[^src-physics-aware-reprogramming]
3. **Selective discrete reprogramming**：词掩码 $m = \mathrm{Softmax}(EW)$，用 Gumbel-Softmax（温度 $\tau$）可微地 top-K 采样 $K{=}1000$ 个词构成扩展词汇表 $E'$，patch embedding 作 query 对 $E'$ 做 cross-attention，冻结 GPT-2 backbone 编码，可学习 projection 出预测。[^src-physics-aware-reprogramming]复杂度 $O(N \cdot P \cdot V' + (N \cdot P)^2)$，冻结块不参与反传；大 $N$ 靠空间子图分块训练。[^src-physics-aware-reprogramming]

## 实验证据

6 数据集：METR-LA、PEMS-BAY、Beijing Taxi、NYC Bike（交通），Air Quality（北京 35 站、6 指标、逐小时），Solar Energy（Alabama 137 光伏站、每 10 分钟）。[^src-physics-aware-reprogramming]Full training 对比基线按 Table 2 表头共 11 个：HI、Informer、iTransformer、PatchTST、MTGNN、GWNet、STNorm、D2STGNN、STID、STAEFormer、FPT。[^src-physics-aware-reprogramming]

- Full training（$T=\tau=24$）：METR-LA MAE 3.63/RMSE 7.43（RMSE 全场最低；MAE 略逊于 STAEFormer 3.60）；PEMS-BAY MAE 1.92 最低、RMSE 4.33 次优（STID 4.31 更低）；Air Quality 3.27/5.12 与 Solar 26.20/39.37 两项均最低。论文自述结果是 "either the best or second-best"。PLM-based FPT 明显落后（METR-LA 6.03）[^src-physics-aware-reprogramming]
- Few-shot（1 天 <1% 训练数据）：METR-LA 5.63 vs 最佳基线 STAEFormer 6.35；PEMS-BAY 3.61 vs PatchTST 4.52（FPT 4.55）[^src-physics-aware-reprogramming]
- Zero-shot（Table 4 未标度量名）：NYC→CHI 2.03（TimesFM 9.07、Time-LLM 10.32）；Solar→Air 31.86（TimesFM 38.62，FPT 68.44；Time-LLM 该设置 OOT——原文解释为其在大空间变量下计算开销过大，只完成小数据集）。Solar→CHI 略输 OpenCity-base（3.96 vs 3.61），论文归因于 OpenCity 预训练含大量交通数据[^src-physics-aware-reprogramming]
- 消融三变体：r/p PLM（transformer 替换 PLM）、r/p Decomposition（transformer encoder 替换分解器）、w/o expanded vocabulary（改用 dense mapping），在 Air Quality/Solar/Beijing Taxi 上均致性能下降[^src-physics-aware-reprogramming]

## 局限

- 论文自认：注意力随 $(N \cdot P)^2$ 增长，大 $N$ 需子图划分训练；zero-shot 可比模型少，因开源的零样本 ST 模型稀缺。[^src-physics-aware-reprogramming]
- 页面评述：DMD 以有限个 top-k 模态截断重构，论文未讨论该有限维截断带来的近似误差。

## 关联

- 扩展 [[model-reprogramming]] 范式：dense 映射 → 可微离散词汇采样
- 细化 [[patch-reprogramming]]（[[source-time-llm]] 的 text prototype cross-attention）：加入 Gumbel-Softmax top-K 选择性词汇扩展

[^src-physics-aware-reprogramming]: [[source-physics-aware-reprogramming]]
