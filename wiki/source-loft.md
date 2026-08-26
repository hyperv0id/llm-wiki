---
title: "LOFT — Low-Rank Prior-Induced Consistency Flow Matching for Efficient Traffic Imputation"
type: source-summary
tags:
  - flow-matching
  - spatio-temporal-imputation
  - consistency-models
  - low-rank
  - traffic
  - kdd-2026
created: 2026-08-26
last_updated: 2026-08-26
source_count: 0
confidence: low
status: active
---

# LOFT — Low-Rank Prior-Induced Consistency Flow Matching for Efficient Traffic Imputation

**作者:** Xiaowei Mao, Tingrui Wu, Yawen Yang, Shengnan Guo(通讯作者), Yan Lin, Shilong Zhao, Haochen Lv, Youfang Lin, Huaiyu Wan（北京交通大学；Yan Lin 属 Aalborg University）
**发表:** KDD 2026（第 32 届，2026-08-09 至 13，韩国济州岛），DOI 10.1145/3770855.3818063，12 页，CC-BY 4.0
**代码:** [github.com/maoxiaowei97/LOFT](https://github.com/maoxiaowei97/LOFT)
**raw:** `raw/loft-low-rank-prior-induced-consistency-flow-matching-efficient-traffic-imputation.pdf`

## 核心论点

论文提出 LOFT，面向高稀疏观测下的交通数据插补。论文将扩散与流匹配方法的推理瓶颈归为两点：学到的向量场诱导弯曲生成轨迹、需多步数值积分；从无信息高斯先验出发学习整个变换带来冗余计算[^src-loft]。方法分三部分：

1. **低秩先验估计**: 先验构造形式化为掩码低秩分解 min‖(X_obs−U_S·W·V_T^⊤)⊙M‖²_F，神经网络单次前向近似其解，借助矩阵乘法结合律等价于线性注意力，复杂度 O(NKd_m)；同时解码先验均值 μ_prior 与不确定性估计 Σ（以 Mean Interval Score 在观测位置监督）。流从 N(μ_prior, I) 初始化，协方差保留单位阵以防生成过程坍缩为确定性先验均值[^src-loft]。
2. **一致性轨迹目标**: L_CT 约束当前状态速度对齐同一线性条件路径上未来状态的速度（stop-gradient 引导）。Lemma 4.1：速度沿轨迹恒定当且仅当轨迹为线性插值。Theorem 4.2：欧拉积分误差 ≤ C(ε_FM+ε_CT/N)，模型偏差项与步数无关，离散化误差按 1/N 缩减[^src-loft]。
3. **不确定性感知矫正**: 论文报告直接联合优化 L_CFM 与 L_CT 时两目标梯度余弦相似度全程为负（梯度冲突），且冲突与数据不确定性相关。矫正系数 α∈[η,1] 按样本不确定性（温度缩放 softmax 聚合）与训练进度（warm-up 后余弦退火）在 v_FM 与自洽目标间插值[^src-loft]。

## 实验结果（作者报告）

PEMS03/04/08，SR-TC 与 SC-TC 两种缺失模式、80% 缺失率，另测 PeMS04 SC-TC 90%；对比 IGNNK/GCASTN/ImputeFormer/LCR/CSDI/PriSTI/FGTI/MTSCI/CoFill/MSFM/FENCE 共 11 个基线。NFE 配置：扩散基线训练与推理均 50，MSFM 均 20，LOFT 训练 10、推理 2[^src-loft]。Table 2 中三个数据集×两种模式的 MAE/RMSE/MAPE 均由 LOFT 取得最低值（如 PeMS04 SR-TC MAE 24.21，次优 FENCE 26.57）；匹配预算实验（Table 1）中基线从 20 NFE 降至 2 NFE 精度退化而 LOFT 保持 RMSE 41.67；90% 缺失下 RMSE 47.01（次优 FENCE 53.61，Table 3）；消融 Wo-P/Wo-C/Wo-U 均劣于完整模型（Fig 5）；效率实验中 LOFT 训练时间最短、论文报告推理约快十倍（Fig 6）；先验构建对整个测试集仅需 2.87 秒，NFE=2 为精度-效率平衡点（Fig 7）[^src-loft]。

## 范围与局限

- 实验仅覆盖 PEMS 系交通数据集，其他时空数据类型未验证。
- 论文未设独立局限性章节；η、λ、τ、γ 等超参数需按设置调节。

## 相关页面

[[loft]] · [[low-rank-prior-estimation]] · [[uncertainty-aware-rectification]]
