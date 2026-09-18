---
title: "DutyTTE: Deciphering Uncertainty in Origin-Destination Travel Time Estimation"
type: source-summary
tags:
  - travel-time-estimation
  - uncertainty-quantification
  - mixture-of-experts
  - spatio-temporal
  - reinforcement-learning
created: 2026-09-17
last_updated: 2026-09-17
source_count: 1
confidence: medium
status: active
---

# DutyTTE: Deciphering Uncertainty in Origin-Destination Travel Time Estimation

**Authors**: Xiaowei Mao、Yan Lin、Shengnan Guo、Yubin Chen、Xingyu Xian、Haomin Wen、Qisen Xu、Youfang Lin、Huaiyu Wan（北京交通大学；Aalborg University；Carnegie Mellon University）。
**Venue**: 缓存未见会议或 arXiv 标注；正文与参考文献采用 AAAI 论文版式与（Author Year）引文格式。
**证据源**: `/run/media/jcheng/WD-Data/yjs/Zotero/storage/LPHZ5BFC/.zotero-ft-cache`（PDF 全文纯文本缓存）

## 问题

给定 origin、destination 与出发时间（ODT 查询），既要预测路径，又要给出通行时间的置信区间。论文认为路径预测的误差会向后传给不确定性量化，因此把"先对齐路径、再估区间"作为两阶段设计的基础[^src-dutytte]。

## 方法

- **路径预测（DRL 阶段）**：用 self-critical policy gradient 优化 LCS/DTW 类整体对齐奖励，梯度以 greedy 解码结果的奖励为 baseline（Eq 3），并与交叉熵联合：$L_\theta=\gamma L_{\pi_\theta}+L_{CE}$（Eq 4）[^src-dutytte]。
- **MoE 插入位置**：在**不确定性量化阶段**对预测路径上的**每个路段**建模。路段输入为该路段在出发时刻前一段窗口内通行时间分布的离散化（top-$m$ 最大可能通行时间 $d_m$ 及其概率 $f_m$）、Node2Vec 路段嵌入与出发时间片嵌入的拼接，再用 wide-deep-recurrent（WDR）编码为含时空聚合信息的 $r'_j$[^src-dutytte]。
- **MoE 层**：$r''_j=\sum_{i=1}^{C}G(r'_j)E_i(r'_j)$（Eq 5），共 $C$ 个参数独立的 expert，门控输出 $C$ 维向量。论文**未给出 expert 的具体网络结构**，也未给出 WDR 的层数等细节[^src-dutytte]。
- **门控（noisy top-k，Eq 6–8）**：$H(r'_j)_i=(r'_j\cdot W_g)_i+\mathcal{N}(0,1)\cdot\mathrm{Softplus}((r'_j\cdot W_{noise})_i)$；top-$k$ 之外的 logit 置 $-\infty$，再对 top-$k$ 做 softmax。稀疏化手段即 top-$k$ 截断，方法来源标注为 Shazeer et al. 2017[^src-dutytte]。
- **聚合与输出**：沿序列长度维对 MoE 输出求和，再由两个独立预测头估计上下界 $\hat u_i=\hat y_i+\hat\sigma^u_i$、$\hat l_i=\hat y_i-\hat\sigma^l_i$；损失为 Mean Interval Score（Eq 9），并附点估计项 $|y_i-\hat y_i|$[^src-dutytte]。
- **超参默认值**：$k=4$、$C=8$（Figure 4(a)）。除路径预测的联合损失与 MIS 外，论文**未提及负载均衡损失**或其他路由正则项。

## 实验

数据集为滴滴成都与西安（2018-11-01 至 11-16，采样 3 s，1,613,355 / 1,951,585 条轨迹，节点 4599 / 5259，Table 2）[^src-dutytte]。

- **Table 1（置信区间，置信水平 90%）**：DutyTTE 成都 RMSE 278.22、MAE 195.70 s、MAPE 23.55%、PICP 91.02%、IW 763.65；西安 270.04 / 186.37 / 22.95 / 91.67 / 774.83。最强对照 T-WDR-MIS 为 306.89 / 221.52 / 27.92 / 84.57 / 829.39（成都），论文报告相对提升 9.34% / 11.65% / 15.65% / 7.63%；IW 列未给提升值[^src-dutytte]。
- **Table 3（点估计）**：DutyTTE 成都 278.22 / 195.70 / 23.55，西安 270.04 / 186.37 / 22.95；对照 TEMP、GBDT、STNN、DeepOD、MWSL-TTE、DOT、T-WDR（T-WDR 成都 306.89 / 221.52 / 27.92，西安 299.13 / 215.61 / 26.81）。论文称 MAPE 相对最强基线提升 15.65% / 14.39%[^src-dutytte]。
- **Table 4（路径预测，LCS↑ / DTW↓）**：DutyTTE 14.52 / 0.017（成都）、14.64 / 0.021（西安）；GDP 13.51 / 0.019、Transformer 13.48 / 0.019；提升 7.48% / 5.24%（LCS）与 10.53% / 12.50%（DTW）[^src-dutytte]。
- **Table 5（消融，成都）**：w/o-M 282.41 / 198.47 / 24.32 / 88.63 / 781.52；w/o-P 302.93 / 219.32 / 27.28 / 87.15 / 836.09；全模型 278.22 / 195.70 / 23.55 / 91.02 / 763.65[^src-dutytte]。
- **超参分析（Figure 4）**：$c_1$–$c_4$ 固定 $C=8$ 取 $k=1,2,4,6$，$k=4$ 与 $C=8$ 组合最优；论文解释 $k$ 或 expert 过少会限制可处理的 context 数量，过多则提高复杂度、加大训练难度。$c_5$–$c_8$ 为策略损失的 $\omega,\beta,\gamma$ 配置，其中 $c_8$ 为 $\omega=0,\beta=0,\gamma=0$[^src-dutytte]。
- **MoE 行为分析（Figure 5、Figure 6）**：随机取 2000 个路段样本，用出发前 10 分钟历史轨迹算通行时间方差；t-SNE 显示含 MoE 时高、低方差路段簇更可分。Figure 6 按方差桶 0–600 / 600–1200 / 1200–2000 给出高频激活组合，低方差档为 [1,3,6,7]、[1,3,7,8]、[1,6,7,8]，中档 [1,3,5,7]、[3,5,6,7]、[1,5,6,7]，高档 [2,4,5,6]、[2,3,4,5]、[1,2,4,5][^src-dutytte]。
- **Table 6（效率，每 batch 128 样本的秒数）**：DutyTTE 成都 0.2706，对照 DOT 1.362、T-WDR 0.2675、MWSL-TTE 0.1289、GDP 5.093；论文将其陈述为效率与效果之间的折中，未给加速比[^src-dutytte]。

## 局限

缓存中**无 Limitations 章节**：正文在 Conclusion 与 Acknowledgment 之间结束，结论段只重述两点主张（DRL 路径对齐改善不确定性量化；MoE 建模路段在复杂 context 下对整体不确定性的贡献）[^src-dutytte]。论文未提及失败场景与 expert 数量的边界条件。

## 与其它 MoE 方法的差异

论文只把 MoE 追溯到 Shazeer et al. 2017 的 sparsely-gated MoE 作为方法来源，未与 MMoE、Switch Transformer 等其它路由设计做实验对比[^src-dutytte]。论文未提及其它 MoE 方法在路由形式、稀疏度或负载均衡上的差异。

[^src-dutytte]: [[source-dutytte]]
