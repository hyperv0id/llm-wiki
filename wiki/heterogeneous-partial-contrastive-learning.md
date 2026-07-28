---
title: "Heterogeneous-Partial Contrastive Learning (HPCL)"
type: technique
tags:
  - contrastive-learning
  - channel-correlation
  - partial-correlation
  - tsfm-adapter
created: 2026-07-28
last_updated: 2026-07-28
source_count: 1
confidence: high
status: active
---

# Heterogeneous-Partial Contrastive Learning (HPCL)

**HPCL** 是 [[cora-correlation-aware-adapter|CoRA]] 在**正/负相关两套表示空间**内，用 [[dynamic-correlation-estimation|DCE]] 给出的相关掩码做对比学习，从而同时服务 **HCorr**（异质正负）与 **PCorr**（仅部分通道对显著）的技术[^src-cheng-2025-cora-correlation-aware-adapter]。

## 机制

1. **切分异质相关**（可学习阈值 \(\epsilon\)）：  
   \(M^{\mathrm{pos}}\) 保留 \(m^{\mathrm{corr}}>\epsilon\)，\(M^{\mathrm{neg}}\) 保留 \(m^{\mathrm{corr}}<-\epsilon\)，其余置 0[^src-cheng-2025-cora-correlation-aware-adapter]。  
2. **对比**：在 \(\tilde X^{\mathrm{pos}}\)（及对称的 neg）上，掩码非零为 positive pair，零为 negative pair；损失为温度 \(\tau\) 的余弦相似度 log-softmax 形式（式 11），\(L_{\mathrm{aux}}=L_{\mathrm{pos}}+L_{\mathrm{neg}}\)[^src-cheng-2025-cora-correlation-aware-adapter]。  
3. **作用对象**：训练引导 Heterogeneous Division 的 channel-aware 投影；**推理不计算 HPCL**，故不增加推理复杂度[^src-cheng-2025-cora-correlation-aware-adapter]。

## 相对聚类头插件

相对 CCM 等聚类式部分相关：HPCL 强调细粒度通道对交互，且不在推理挂额外聚类头[^src-cheng-2025-cora-correlation-aware-adapter]。相对全通道 attention：掩码显式压制弱相关，抑制噪声边[^src-cheng-2025-cora-correlation-aware-adapter]。

## 风险

对比监督完全来自 DCE 矩阵——若 \(M^{\mathrm{corr}}\) 系统偏，正负对标签同步偏（训练自指）[^src-cheng-2025-cora-correlation-aware-adapter]。消融显示 HPCL 单独增益有限，需与 DCE、双分支 HD 联用[^src-cheng-2025-cora-correlation-aware-adapter]。

## 相关页面

- [[cora-correlation-aware-adapter]] · [[dynamic-correlation-estimation]]
- [[source-cheng-2025-cora-correlation-aware-adapter]]

## 引用

[^src-cheng-2025-cora-correlation-aware-adapter]: [[source-cheng-2025-cora-correlation-aware-adapter]]
