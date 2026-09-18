---
title: "TransferTraj: A Vehicle Trajectory Learning Model for Region and Task Transferability"
type: source-summary
tags:
  - trajectory-learning
  - mixture-of-experts
  - transfer-learning
  - spatio-temporal
  - top-k-gating
created: 2026-09-17
last_updated: 2026-09-17
source_count: 1
confidence: medium
status: active
---

# TransferTraj: A Vehicle Trajectory Learning Model for Region and Task Transferability

**Authors**: Tonglong Wei、Yan Lin（两人共同一作）、Zeyu Zhou、Haomin Wen、Jilin Hu、Shengnan Guo（通讯）、Youfang Lin、Gao Cong、Huaiyu Wan（北京交通大学；Aalborg University；Carnegie Mellon University；华东师范大学；南洋理工大学）。
**Venue**: arXiv:2505.12672v1 [cs.LG]（2025-05-19），正文标注 "Preprint. Under review."。
**证据源**: `/run/media/jcheng/WD-Data/yjs/Zotero/storage/TYA64S26/.zotero-ft-cache`（PDF 全文纯文本缓存）

## 问题

车辆轨迹模型通常按区域与任务分别训练。论文目标是**预训练一次、跨区域且跨任务直接使用**：区域差异来自地理尺度与空间上下文（POI、路网）分布不同，移动模式（转弯、加速、方向）随之不同；任务差异来自输入输出结构与所学相关性不同[^src-transfertraj]。

## 方法

RTTE 把轨迹点表示为四种模态（相对首点坐标、时间特征、POI 文本、路段文本），混合成 $e_i$ 后送入 $L$ 个堆叠层，每层由 TRIE 块与 SC-MoE 块串接（Sec 4.1.3）[^src-transfertraj]。

- **插入位置**：SC-MoE 位于**每个 RTTE 层内、TRIE 之后**；共 $L$ 层，最优 $L=2$。
- **expert 设计**：给定 TRIE 输出的隐状态 $h_i$ 与模态向量 $e_i$，$h'_i=\sum_{j=1}^{C}G_j(h_i+e_i)E_j(h_i+e_i)$（Eq 3）。共 $C$ 个参数独立的 expert，**每个 expert 为两层 MLP**；门控网络输出 $C$ 维向量[^src-transfertraj]。
- **路由公式（noisy top-K gating，Eq 4）**：$H(h_i+e_i)_j=((h_i+e_i)\cdot W_g)_j+\mathcal{N}(0,1)\cdot\mathrm{Softplus}(((h_i+e_i)\cdot W_{noise})_j)$；非 top-$k$ 的元素置 $-\infty$，再经 softmax。论文明确标注该门控来自 Shazeer 等 2017（参考文献 [27]）[^src-transfertraj]。
- **稀疏性与超参**：稀疏性由 top-$k$ 截断实现；Appendix G 与 Table 15 给出扫描范围，最优 $k=4$、$C=8$，隐藏维最优 256、层数最优 2。论文解释 expert 过少限制可表达的空间上下文、过多则复杂度上升、训练变差[^src-transfertraj]。
- **额外损失**：预训练损失为空间与时间模态的 MSE（Eq 5）加掩码-恢复目标；论文**未提及负载均衡损失**或 expert 使用率正则项，也未给出路由稀疏度实测比例[^src-transfertraj]。
- **动机**：移动模式受局部空间上下文支配（高速、稀疏 POI 区域多为高速直线行驶，密集 POI 且路网复杂区域多为 stop-and-go），相似局部上下文可跨区域复用，故按空间上下文而非区域路由[^src-transfertraj]。

## 实验

数据为成都、西安、Porto 三个车辆轨迹集，三跳重采样保证间隔至少 6 s，过滤少于 5 或多于 120 点的轨迹[^src-transfertraj]。

- **Table 1（轨迹预测 TP，成都）**：TransferTraj RMSE 187.91 / MAE 144.53 m；对照 START 333.10 / 240.40；去掉预训练的 wo pt 为 289.25 / 218.43，去掉微调的 wo ft 为 223.15 / 176.88。西安 212.62 / 154.86，Porto 196.46 / 149.75[^src-transfertraj]。
- **Table 3（OD 通行时间估计）**：TransferTraj 成都 RMSE 2.861 min / MAE 2.060 min / MAPE 9.360%，西安 3.816 / 2.569 / 8.343，Porto 2.138 / 1.225 / 14.682；wo ft 成都 2.992 / 2.265 / 9.613[^src-transfertraj]。
- **Table 6（消融，成都）**：去掉 SC-MoE 后 OD TTE RMSE 3.240 / MAE 2.585 / MAPE 11.584（全模型 2.861 / 2.060 / 9.360），论文自述该任务性能下降 14.52%；去掉 TRIE 后 TP RMSE 236.18、MAE 185.14[^src-transfertraj]。
- **区域迁移（Table 4、Table 5、Table 10–14）**：论文报告 TP 任务零样本与少样本分别比 SOTA 提升 83.70% 与 33.68%，OD TTE 提升 10.88% 与 13.07%，TR 少样本提升 18.08%[^src-transfertraj]。
- **Figure 2（expert 激活分布）**：按局部 POI 与路段密度分三档——高密度（> 15）、中密度（5–15）、低密度（< 5）——各档高频组合不同：高密度 [1,3,5,6]、[1,3,6,8]、[1,5,7,8]；中密度 [1,2,5,6]、[1,2,7,8]、[1,3,6,7]；低密度 [1,4,5,6]、[2,4,6,8]、[2,4,6,7][^src-transfertraj]。
- **Figure 5 与 Table 16**：$k$、$C$ 的扫描曲线；论文称模型体量与 t2vec、Trembr 相当，明显小于 START、LightPath[^src-transfertraj]。

**论文内部数字不一致**：成都 TP 的 TransferTraj RMSE 在 Table 1 记为 187.91、在 Table 6 记为 197.91（两表 MAE 同为 144.53）；Table 15 的 SC-MoE 配置共列 7 组，为 $k\in\{1,2,4,6\}$ 与 $C\in\{6,8\}$ 的组合（缺 $(k{=}6,C{=}6)$），标签重复出现 $c_3$，而 Figure 5 横轴标为 $c_1$–$c_7$[^src-transfertraj]。

## 局限

Appendix A.1 自述跨区域的分类类任务（trajectory-user linking、目的路段预测）因各区域用户数与路网规模不同而难以处理[^src-transfertraj]。论文未提及 expert 数量的扩展上界与路由偏置分析。

## 与其它 MoE 方法的差异

论文把 SC-MoE 门控归因于 Shazeer 等 2017 的 sparsely-gated MoE（参考文献 [27]），未与其它 MoE 路由变体做实验对比[^src-transfertraj]。论文未提及与其它 MoE 方法在路由粒度或负载均衡上的差异。

[^src-transfertraj]: [[source-transfertraj]]
