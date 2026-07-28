---
title: "CoRA: Correlation-aware Adapter for TSFMs"
type: entity
tags:
  - time-series-foundation-model
  - channel-correlation
  - plug-and-play
  - multivariate-forecasting
  - iclr-2026
created: 2026-07-28
last_updated: 2026-07-28
source_count: 1
confidence: high
status: active
---

# CoRA: Correlation-aware Adapter

**CoRA**（CoRrelation-aware Adapter）是面向时间序列基础模型（TSFM）下游微调的轻量**通道相关插件**（Cheng et al., ECNU / Huawei Noah's Ark, ICLR 2026）[^src-cheng-2025-cora-correlation-aware-adapter]。

> [!warning] 同缩写消歧
> 本页 **≠** [[cora-tsfm|CoRA: Covariate-awaRe Adaptation]]。  
> - **本页**：补 **通道间相关**（DCorr / HCorr / PCorr），用 TSFM 表示 + 原预测做增强。  
> - **[[cora-tsfm]]**：补 **外生协变量**（因果嵌入 + 零初始化 adaLN 注入预测头）。  
> 源文件：本页 → [[source-cheng-2025-cora-correlation-aware-adapter]]；协变量篇 → [[source-cora]]。

## 要解决什么

多数 TSFM 默认 [[channel-independence|CI]]，忽略或只粗糙处理通道交互；TTM 通道混合权重不随时变、UniTS/Moirai 注意力仍全通道交互且不显式正负/部分相关。跨数据集相关形态差异大，不宜指望预训练一次学全，需要**可与 TSFM 同训的微调插件**[^src-cheng-2025-cora-correlation-aware-adapter]。

## 流水线（四个过程）

```
TSFM: X → 表示 X̃, 原预测 Ŷ
        │
        ├─ DCE: R (Pearson) + Qt V Qtᵀ → Mcorr
        ├─ HD:  P1/P2 channel-aware 投影 → 正/负空间
        ├─ HPCL: 用 Mpos/Mneg 对比学习（仅训练）
        └─ Fusion: P3/P4 + 门控 β 混合 → Ŷ*
```

1. **[[dynamic-correlation-estimation|Dynamic Correlation Estimation (DCE)]]**  
   可学习相关低秩分解为时变 \(Q_t\) 与时不变 \(V\)，加规则 Pearson \(R\)：\(M_t^{\mathrm{corr}}=R+Q_t V Q_t^\top\)。\(Q_t\) 由共享基 \(q\) 上的 **[[time-aware-polynomial-correlation|K 阶时间多项式]]** + 表示驱动系数 \(C_t=f(\tilde X_t)\) 生成[^src-cheng-2025-cora-correlation-aware-adapter]。

2. **Heterogeneous Division**  
   SE 风格 channel-aware 投影把 \(\tilde X\) 映到 \(\tilde X^{\mathrm{pos}}\) / \(\tilde X^{\mathrm{neg}}\)；解耦靠下一步对比监督，而非投影 alone[^src-cheng-2025-cora-correlation-aware-adapter]。

3. **[[heterogeneous-partial-contrastive-learning|H-PCorr Contrastive Learning (HPCL)]]**  
   阈值 \(\epsilon\) 把 \(M^{\mathrm{corr}}\) 切成正/负相关掩码，在两空间做 InfoNCE 风格对比；显著对拉近、非显著推远 → 自适应 **PCorr**。推理不跑 HPCL[^src-cheng-2025-cora-correlation-aware-adapter]。

4. **Heterogeneous Fusion & Prediction**  
   \(\hat Y^*=\beta\,\mathrm{Linear}(\tilde X^{\mathrm{pos}}+\tilde X^{\mathrm{neg}})+(1-\beta)\hat Y\)，\(\beta\in[0,1]^N\) 按通道学相关 vs 独立的折中[^src-cheng-2025-cora-correlation-aware-adapter]。

## 复杂度与效率

| 阶段 | 主导复杂度 |
|------|------------|
| 训练 DCE / HPCL | \(O(N^2)\)（相对通道数） |
| 推理（仅 HD + Fusion 投影） | \(O(N)\) |

在 ETTm2 (N=7)、Weather (21)、Electricity (321) 上，相对骨干的参数与时间增量有限，推理侧尤其不明显[^src-cheng-2025-cora-correlation-aware-adapter]。

## 实证要点（5% few-shot）

- 六骨干（GPT4TS、CALF、UniTime、Moment、Timer、TTM）× 十数据集：加 CoRA 平均 MSE 普遍优于仅微调[^src-cheng-2025-cora-correlation-aware-adapter]。
- TTM：**CI + CoRA** 优于 **CD 无 CoRA**（同预训练权重）[^src-cheng-2025-cora-correlation-aware-adapter]。
- 对比 LIFT、C-LoRA：few-shot 下后两者常恶化骨干；CoRA 稳定增益[^src-cheng-2025-cora-correlation-aware-adapter]。
- 消融：DCE+HD+HPCL 全开最优；模块可互补不可单打[^src-cheng-2025-cora-correlation-aware-adapter]。
- 3% 数据仍有小幅提升；K 常 3–4，M 不必随 N 线性涨[^src-cheng-2025-cora-correlation-aware-adapter]。

## 与端到端通道交互的定位

| 路线 | 代表 | 与 CoRA |
|------|------|---------|
| 端到端 CD Transformer | [[crossformer]]、[[itransformer]] | 改整网结构，非 TSFM 插件 |
| 端到端相关插件 | CCM、C-LoRA、LIFT | 多需与 E2E 同训/预训练；few-shot 不稳 |
| **TSFM 下游相关插件** | **CoRA（本页）** | 只微调、吃表示与原预测、推理 \(O(N)\) |
| TSFM 协变量适配 | [[cora-tsfm]] | 外生模态，不是通道相关 |

## 相关页面

- [[source-cheng-2025-cora-correlation-aware-adapter]]
- [[dynamic-correlation-estimation]] · [[heterogeneous-partial-contrastive-learning]] · [[time-aware-polynomial-correlation]]
- [[channel-independence]] · [[mixed-channel-dependency]] · [[multivariate-correlation-attention]]
- [[cora-tsfm]] — 协变量 CoRA（消歧）

## 引用

[^src-cheng-2025-cora-correlation-aware-adapter]: [[source-cheng-2025-cora-correlation-aware-adapter]]
