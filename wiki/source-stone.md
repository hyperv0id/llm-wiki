---
title: "STONE: A Spatio-temporal OOD Learning Framework Kills Both Spatial and Temporal Shifts"
type: source-summary
tags:
  - spatio-temporal
  - out-of-distribution
  - traffic-forecasting
  - invariant-learning
  - graph-intervention
created: 2026-11-07
last_updated: 2026-11-07
source_count: 1
confidence: medium
status: active
---

# STONE: A Spatio-temporal OOD Learning Framework Kills Both Spatial and Temporal Shifts

Binwu Wang, Jiaming Ma, Pengkun Wang, Xu Wang, Yudong Zhang, Zhengyang Zhou, Yang Wang（中科大 + 中科大苏州高等研究院），KDD 2024（pp. 2948–2959，DOI 10.1145/3637528.3671680；无 arXiv 版；raw/stone-wang-2024.pdf）[^src-stone]。代码 github.com/PoorOtterBob/STONE-KDD-2024。

## 问题与机制

时空 OOD 含两轴漂移：时间漂移 $P(X_{te})\neq P(X_{tr})$，空间漂移 $G_{te}\neq G_{tr}$（传感器增删）。既有 STGNN 绑定训练图、无法为新节点生成表示；CauSTG/CaST/STEVE 等只处理时间漂移、忽略图演化[^src-stone]。

STONE 的路线是不增强数据、扰动已学到的节点依赖：空间侧用 Fréchet 嵌入（锚点到节点的 K 阶 Manhattan 距离，失真 $O(\log N)$）给新节点直接赋初值，经 2 层门控 Transformer 得空间语义图 $D_s$；时间侧用门控空洞因果 TCN 得节点表示、注意力求时间语义图 $D_t$；训练时两个掩码生成器各维护一份可学习的节点采样分布，每步抽 1% 节点构造「原图 + 编辑图」两个环境；主网络最小化「加权 L1 + 逐元素误差方差」，掩码生成器以独立优化器对抗式最大化 $\log p\cdot$方差（explore-to-extrapolate）。推理不加掩码，单次前向[^src-stone]。

## 证据

LargeST 的 SD（716 传感器）/GBA（2352）：训练 2019-01→08 的 550/1809 节点，测试 2020-11→12、含 10%/15%/20% 新节点并遮蔽 10% 旧节点。基线 HL/GWNet/STGCN/STNN/CaST/CauSTG。全节点 Table 2 与仅新节点 Table 3 共 36 格：STONE 多数 MAE 最优，最大增益在 GBA-20% 新节点 12 步（53.09→37.60，−29.2%）；GWNet 为最强基线，STONE 在 5 格 MAE 落后，优势集中在 6/12 步与新增节点。只保留时间漂移的 Table 5 中与 GWNet 基本持平。消融：去 Fréchet 嵌入退化最大（MAPE +72.7%），去干预损失 +15.0%，去噪声 +7.6%[^src-stone]。

## 局限

仅交通数据；掩码算子多时训练代价指数增长；语义图 gated transformer 可被更强结构学习替代（论文 B.4）[^src-stone]。独立评测的 [[st-ood|ST-OOD]] 跨年基准观察到专用因果/不变方法在自然跨年 T-OOD 上难以击败常规强基线，与上述边界方向一致。

机制细节见 [[stone]]。

[^src-stone]: [[source-stone]]
