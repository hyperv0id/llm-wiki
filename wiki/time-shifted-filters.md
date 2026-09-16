---
title: "Time-Shifted Filters (TSF)"
type: technique
tags:
  - frequency-domain
  - attention
  - traffic-forecasting
  - federated-learning
created: 2026-09-16
last_updated: 2026-09-16
source_count: 1
confidence: medium
status: active
---

# Time-Shifted Filters (TSF)

[[proxy-node-generation|代理节点生成]] 在注意力之前对 key/value 做的频域滤波。滤波器组 $W_{fil}\in\mathbb{C}^{L\times d_{att}}$ 含 $L$ 行，按当前时间步取其中一行[^src-fedhint]。

## 机制

$$\tilde{K}_t=\mathcal{F}^{-1}\big(\mathcal{F}(K_t)\odot\mathrm{Expand}(W_{fil}^{j},|V_m|)\big),\quad j=t\bmod L,$$

$\mathcal{F}$ / $\mathcal{F}^{-1}$ 是沿时间维的 DFT/IDFT，$W_{fil}^{j}\in\mathbb{C}^{d_{att}}$ 是第 $j$ 行，$\mathrm{Expand}$ 把它扩到 $|V_m|$ 行（每行相同），$\odot$ 逐元素乘。$j=t\bmod L$ 使不同时间步用不同滤波器，同一时间步内整段输入共用一组。$V_t$ 同样处理得到 $\tilde{V}_t$[^src-fedhint]。

## 实证

- $L\in\{1,12,144,288\}$ 中 $L=288$ 最优，论文解释 288 是一天的时间戳数，逐时间戳滤波能抓住一天内不同时段的时序特征[^src-fedhint]。
- 去掉 TSF 后四个数据集一致掉点，但幅度是四个消融项中最小的：PEMS03 MAE 11.95→12.11、RMSE 19.12→19.28；PEMS04 16.11→16.32、PEMS07 17.19→17.42、PEMS08 12.38→12.49[^src-fedhint]。

## 与频域注意力的区别

论文 related work 把频域建模的既有工作列为 FEDformer（在频域做注意力）与 TimesNet（用频率刻画多周期性）[^src-fedhint]。TSF 的位置不同：注意力仍在时域表征上做，频域只用来给 key/value 逐频率加权（wiki 侧对照，非论文自述的差异分析）。

## 相关页面

- [[proxy-node-generation|Proxy Node Generation (GPN)]]
- [[hidden-global-components|Hidden Global Components]]
- [[frequency-enhanced-attention|Frequency Enhanced Attention]]

[^src-fedhint]: [[source-fedhint]]
