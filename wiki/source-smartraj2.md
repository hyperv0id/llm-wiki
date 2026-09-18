---
title: "SMARTraj2: A Stable Multi-City Adaptive Method for Multi-View Spatio-Temporal Trajectory Representation Learning"
type: source-summary
tags:
  - spatio-temporal
  - trajectory-representation
  - gating
  - transfer-learning
  - mixture-of-experts
  - multi-city
created: 2026-09-17
last_updated: 2026-09-17
source_count: 1
confidence: medium
status: active
---

# SMARTraj2: A Stable Multi-City Adaptive Method for Multi-View Spatio-Temporal Trajectory Representation Learning

**Authors**: Tangwen Qian, Junhe Li, Yile Chen, Gao Cong, Zezhi Shao, Jun Zhang, Tao Sun, Fei Wang, Yongjun Xu（中科院计算所 / 南洋理工大学 / 中国科学院大学）。[^src-smartraj2]
**Venue**: NeurIPS 2025（页眉 "39th Conference on Neural Information Processing Systems (NeurIPS 2025)"）。[^src-smartraj2]
**证据源**: `/run/media/jcheng/WD-Data/yjs/Zotero/storage/K3VDNRSH/.zotero-ft-cache`（PDF 全文纯文本缓存）

## 与 MoE 的边界（本页重点）

论文**没有使用 MoE**：对全文缓存检索 expert / MoE / mixture，命中只在参考文献（[12] Dryden & Hoefler, Spatial mixture-of-experts, NeurIPS 2022；[37] Szymanski & Lemmon 1993；[54] Meta-DMoE, NeurIPS 2022）与正文两处引用，即 transfer learning 综述处引用 [16, 12, 5]、city-specific 特征处引用 [32, 54, 37][^src-smartraj2]；正文没有任何 expert 池、专家子网络或 top-k 路由。标题里的 gating 是 Sec 3.3 Personalized Gating Mechanism 的 sigmoid 特征门控：用城市侧/轨迹侧特征生成逐维门向量，对共享编码器的 embedding 做逐元素缩放，门控泛化对象是 city 与 trajectory，不是 expert。机制差异：MoE 在多个独立 expert 输出间做选择与加权，这里的 gate 只调节 domain-invariant 特征的注入强度，输出仍由同一套视图编码器产生。

## 问题

多视图时空轨迹表示学习（GPS view / route view / grid view）在跨城市泛化上遇到两个论文自述挑战[^src-smartraj2]：

1. **multi-city structural heterogeneity**：路网、网格划分、交通规则不同，跨城市的对应视图嵌入空间天然不相交（如各城市的 road ID embedding 处于不同空间）。
2. **amplified seesaw phenomenon**：多城市、多视图、多任务同时优化时，对一个 city/view/task 的改进会不成比例地恶化其它项。

论文自称 "To the best of our knowledge, this is the first work to highlight the importance of multi-city, multi-view trajectory representation learning with a focus on generalization across diverse urban scenes"[^src-smartraj2]。

## 方法

两部分（Figure 2）[^src-smartraj2]：

- **Feature Disentanglement Module**：三个视图各用专属编码器——GPS view 层级双向 GRU（点级→子轨迹级），route view 用 GAT 更新路段嵌入后接 transformer，grid view 用 transformer 编码网格序列并加权融合 POI 语义。之后按 transfer learning 原则（引 [16, 12, 5]）用共享权重提取器 $V$ 取 domain-invariant 特征 $H^i=V(h^p,h^r,h^g)$（Eq.4），每城市单独训练的提取器 $M_k$ 取 domain-specific 特征 $H^s_k$（Eq.5），二者以 soft subspace orthogonality 约束的 difference loss $L_{\mathrm{diff}}=\sum_k\lVert H^{i\top}H^s_k\rVert_F^2$（Eq.6）分离。
- **Personalized Gating Mechanism**（Sec 3.3, Eq.7–12）：city-level 门把 route 嵌入与城市侧特征（平均轨迹长度、速度统计等）拼接（对 $h_r$ 施加 stop-gradient $\bar{\nabla}(\cdot)$ 以免梯度回传污染嵌入），过两层 FFN 后 sigmoid：$\delta^{\mathrm{city}}_r=\gamma\cdot\sigma(W'_{\mathrm{city}}h'_r+b')$（Eq.8），$\gamma=2$ 把门值限制在 $[0,\gamma]$，再 $h^{\mathrm{city}}_r=\delta^{\mathrm{city}}_r\odot h_r$（Eq.9）；trajectory-level 门用起终点 POI 语义等轨迹侧特征，对 $h^{\mathrm{city}}_r$ 同样 stop-gradient 后生成 $\delta^{\mathrm{traj}}_r$（Eq.11）并逐元素相乘（Eq.12）。无 expert、无稀疏路由、无 load balancing loss。
- **训练目标**：$L_{\mathrm{total}}=w_1L_{\mathrm{diff}}+w_2L_{\mathrm{MLM}}+w_3L_{\mathrm{pair}}$（Eq.15），$L_{\mathrm{MLM}}$ 是轨迹 masked language modeling（Eq.13），$L_{\mathrm{pair}}$ 是跨视图正/负轨迹对的对比损失（Eq.14）。预训练 + 微调范式（Fig.3、Fig.5：pre-train 收敛更快、小标注量下更稳，优于端到端从头训练）。

## 实验

DiDi 滴滴的成都、西安各 15 天轨迹（13/1/1 切分），路网来自 OpenStreetMap；四个下游任务：road label classification（Micro/Macro-F1）、travel time estimation（MAE/RMSE）、destination road 与 destination grid prediction（Acc@1/5）[^src-smartraj2]。

- **TABLE 1（Xi'an）**：SMARTraj2 0.8407/0.8298（F1）、35.0689/60.9156（MAE/RMSE）、0.7409/0.9069 与 0.6675/0.8392（两个 Acc）；最强基线 MVTraj 对应为 0.8290/0.8159、54.9044/85.3847、0.6904/0.8550（表中标 ‡）、0.6630/0.8154。正文报告 Chengdu 上相对 MVTraj travel time MAE −29.30%、RMSE −23.75%，并称 Xi'an 有类似提升。注意：基线均按单城市独立训练，SMARTraj2 是多城市联合训练，对照口径论文已明确写出[^src-smartraj2]。
- **TABLE 2（Xi'an 消融）**：w/o diff loss 使 MAE 升至 44.8105（destination road Acc@1 降到 0.6025）；w/o gating 升至 40.3469；w/o invariant+specific 56.5380；w/o gating+specific 54.8681。一处如实记录：w/o diff loss 在 road label 上的 Micro/Macro-F1（0.8500/0.8366）反而高于全模型（0.8407/0.8298），论文未解释该反差[^src-smartraj2]。
- **Fig.4（Xi'an, travel time 敏感性）**：$\gamma\in\{0.5,1,2,4,8\}$ 中 $\gamma=2$ 最优，论文解释为门值范围 $[0,2]$、中心约 1；$w_1{:}w_2$ 在 $\{1{:}3,\dots,3{:}1\}$ 中 $1{:}1$ 最优，偏小削弱正交约束、偏大破坏损失平衡[^src-smartraj2]。

## 局限

论文 Limitation and Future Work 自述[^src-smartraj2]：仅在成都、西安两个代表性城市验证；依赖多视图城市数据，缺失模态或数据质量不一致可能影响性能；未来将扩展到 10+ 城市验证可扩展性，并探索参数共享、模型压缩、分布式训练等效率手段。

## 与其它 MoE 方法的差异

论文正文未与任何具体 MoE 方法做实验对比；参考文献列出的 Spatial mixture-of-experts [12] 与 Meta-DMoE [54] 仅作为 transfer learning 与 city-specific 特征的背景引用出现，论文未讨论其机制。本文的 gate 与 MoE gating network 名字相近、机制不同：MoE 的门控在多个 expert 输出间分配权重，本文的 sigmoid 门控是条件于城市/轨迹侧特征的特征级调制（含 stop-gradient），作用于共享编码器产生的同一嵌入，论文未提及为何不采用 expert 结构[^src-smartraj2]。

[^src-smartraj2]: [[source-smartraj2]]
