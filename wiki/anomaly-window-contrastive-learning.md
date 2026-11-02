---
title: "Anomaly Window Contrastive Learning（AWCL）"
type: technique
tags:
  - time-series
  - anomaly-detection
  - contrastive-learning
  - multimodal
created: 2026-11-02
last_updated: 2026-11-02
source_count: 1
confidence: low
status: active
---

# Anomaly Window Contrastive Learning（AWCL）

异常窗口对比学习（Anomaly Window Contrastive Learning, AWCL）是 [[vetime|VETime]] 框架中针对点异常与上下文异常分别构建多尺度跨模态对比的目标优化机制（arXiv:2602.16681，§3.3）[^src-2602-16681-vetime]。

## 动机与设计思路

在时序异常检测中，两类主要异常对模态感知的需求呈现出互补性：
1. **短程点异常（Point Anomalies）**：表现为瞬时、短寿命的突变数值偏离，持续时间极短，高度依赖 1D 时序分支的高分辨率局部数值敏感度；
2. **长程上下文异常（Context Anomalies）**：表现为长程模式违背或周期结构异化，持续跨度大，高度依赖 2D 视觉分支的宏观全局拓扑视野；
3. **对比学习目标**：AWCL 首先通过双向交叉注意力促成模态深度交互，随后根据异常段长度动态构建自适应异常窗口，在多尺度下分别对短程与长程异常执行跨模态正向拉近与负向推远，增强正常与异常模式的特征可分性（§3.3）[^src-2602-16681-vetime]。

## 机制流程与形式化

### 1. 双向交叉注意力交互（Dual Cross-Attention）

时序特征 $F_{TS} \in \mathbb{R}^{N_{TS} \times D}$ 与对齐视觉特征 $F_V \in \mathbb{R}^{N_{TS} \times D}$ 互为 Query、Key、Value 进行跨模态交互（公式 7–8，附录 B.5）[^src-2602-16681-vetime]：
$$\begin{aligned}
Z_{TS}^{\prime} &= F_{TS} + \operatorname{CrossAttn}(F_{TS}, F_V), \quad Z_V^{\prime} = F_V + \operatorname{CrossAttn}(F_V, F_{TS})\\
Z_{TS} &= Z_{TS}^{\prime} + \operatorname{FFN}(Z_{TS}^{\prime}), \quad Z_V = Z_V^{\prime} + \operatorname{FFN}(Z_V^{\prime})
\end{aligned}$$
其中跨模态注意力算子定义为：
$$\operatorname{CrossAttn}(X, Y) = \operatorname{Softmax}\left(\frac{X W_Q (Y W_K)^\top}{\sqrt{d}}\right) (Y W_V)$$
投影矩阵 $W_Q, W_K, W_V \in \mathbb{R}^{d \times d}$，特征维度 $d = 512$。

### 2. 异常上下文窗口构建（仅限合成预训练阶段）

在源域合成数据离线预训练阶段，AWCL 根据合成异常标签动态构建局部窗口：
- 点级真值标签先转换为 patch 级标签（patch 内存在任一异常点则标为 1）；
- 针对长度为 $L_a$ 个 patch 的连续异常段，向两侧正常上下文对称扩展，最大窗口长度为 $L_w = 2 L_a$；
- 每个窗口包含单个连续异常实例及其邻近正常上下文。在下游测试部署时，双模态特征直接以前馈网络拼接生成异常增强表征 $F_A = \operatorname{FFN}([Z_{TS}; Z_V]) \in \mathbb{R}^{N_{TS} \times 512}$，完全免调窗部署，保证零样本推理的便捷性（附录 B.5）[^src-2602-16681-vetime]。

### 3. 窗内对比学习（Intra-Window Contrastive Learning）

针对短点异常（$L_a \le 1$ patch，如突发离群点），强化视觉特征 $Z_V$ 的细粒度异常对齐：
- **Anchor**：窗内异常视觉特征 $Z_V^A$；
- **正样本**：同窗内对应的时序异常特征 $Z_{TS}^A$；
- **负样本**：同窗内的正常时序特征集合 $\mathcal{N}_{\text{neg}}$。

窗内对比损失定义为（公式 1）[^src-2602-16681-vetime]：
$$\mathcal{L}_{\text{intra}} = - \log \frac{\exp(Z_V^A \cdot Z_{TS}^A / \tau)}{\exp(Z_V^A \cdot Z_{TS}^A / \tau) + \sum_{k \in \mathcal{N}_{\text{neg}}} \exp(Z_V^A \cdot Z_{TS}^k / \tau)}$$
其中 $\cdot$ 表示点积相似度，$\tau = 0.1$ 为温度参数，$\mathcal{N}_{\text{neg}}$ 表示同窗内正常时序特征集合。

### 4. 窗间对比学习（Inter-Window Contrastive Learning）

针对长上下文异常（$L_a \ge 2$ patches，如周期形态畸变），增强时序特征 $Z_{TS}$ 的全局判别力：
沿时间维对窗口内特征做平均池化得到 $\bar{Z}_{TS}$ 与 $\bar{Z}_V$。
- **Anchor**：异常窗口的池化时序特征 $\bar{Z}_{TS}^A$；
- **正样本**：对应异常窗口的池化视觉特征 $\bar{Z}_V^A$；
- **负样本**：正常窗口的池化视觉特征集合 $\mathcal{W}_{\text{neg}}$。

窗间对比损失定义为（公式 2）[^src-2602-16681-vetime]：
$$\mathcal{L}_{\text{inter}} = - \log \frac{\exp(\bar{Z}_{TS}^A \cdot \bar{Z}_V^A / \tau)}{\exp(\bar{Z}_{TS}^A \cdot \bar{Z}_V^A / \tau) + \sum_{j \in \mathcal{W}_{\text{neg}}} \exp(\bar{Z}_{TS}^A \cdot \bar{Z}_V^j / \tau)}$$

### 5. 联合总对比损失与增强表征

为防止跨样本正常与异常模式相互干扰，总对比损失逐样本独立计算归一化（公式 3）[^src-2602-16681-vetime]：
$$\mathcal{L}_{\text{aw}} = \frac{1}{N} \sum_{i=1}^N \left( \frac{1}{W_s} \sum_{s=1}^{W_s} \mathcal{L}_{\text{intra}}^{(i, s)} + \frac{1}{W_l} \sum_{l=1}^{W_l} \mathcal{L}_{\text{inter}}^{(i, l)} \right)$$
其中 $N$ 为样本数量，$W_s$ 与 $W_l$ 分别表示样本中短异常与长异常的窗口数量。
优化后的双模态特征经前馈网络融合，输出异常增强表征供后续 [[task-adaptive-multimodal-fusion|TMF]] 模块使用：
$$F_A = \operatorname{FFN}([Z_{TS}; Z_V]) \in \mathbb{R}^{N_{TS} \times 512}$$

## 超参数与敏感度分析

- **默认超参**：温度参数 $\tau = 0.1$，总损失权重 $\lambda_{\text{aw}} = 0.1$。
- **敏感度验证（§4.3, 图 6）**：
  - $\lambda_{\text{aw}}$ 在 0.1 处达到最佳表现（图 6(a)）；过大的 $\lambda_{\text{aw}}$ 会诱发特征过平滑，模糊正常与异常的判别边界；
  - 温度系数 $\tau$ 在 0.1 处取得最优表现（图 6(c)），偏离经验最优值会导致多模态表征分离度下降。

## 实证证据与消融分析

### 1. 模块组合消融表现（表 4, 附录 D.5 表 17）
- 在基础流水线上引入 AWCL（变体 V3 相对变体 V2）：NAB 数据集上的 VUS-PR 从 27.75% 提升至 29.52%，SED 数据集上的 VUS-PR 从 84.84% 提升至 86.94%；
- 完整模型相比缺失 AWCL 的变体 V4（Base+TIC+PTA+TMF）：NAB 上的 Affiliation-F1 / VUS-PR 从 86.84% / 31.23% 提升至 88.57% / 32.98%；WSD 上的 Affiliation-F1 / VUS-PR 从 91.63% / 29.00% 提升至 94.31% / 33.85%。

### 2. 窗内与窗间对比消融（附录 D.6, 表 18）
在微观子组件消融中，去除窗内或窗间对比均导致性能退化（表 18）[^src-2602-16681-vetime]：
- **去除窗间对比（w/o inter-window contrast）**：NAB 上 Aff-F1 / VUS-PR 降至 86.69% / 31.17%，长异常为主的 WSD 上 VUS-PR 从 33.85% 降至 30.54%，表明长程上下文异常对跨窗口全局拓扑判别的高度依赖；
- **去除窗内对比（w/o intra-window contrast）**：以细粒度点异常为主的 NAB 上 Aff-F1 / VUS-PR 降至 86.15% / 30.86%（完整模型为 88.57% / 32.98%），YAHOO 上为 97.77% / 88.29%。

### 3. 细分异常类型表现（表 3）
在 NAB 与 YAHOO 拆分出的异常类型上，AWCL 促成的多尺度表征带来显著提升（表 3）[^src-2602-16681-vetime]：
- **Point 异常**：VETime 的 Affiliation-F1 / Standard-F1 达到 93.20% / 62.33%（纯时序 TimeRCD 仅为 74.84% / 28.34%，VLM4TS 仅为 67.11% / 23.15%）；
- **Context 异常**：VETime 达到 91.79% / 59.74%（TimeRCD 仅为 73.84% / 32.18%，VLM4TS 仅为 61.94% / 35.90%）。

[^src-2602-16681-vetime]: [[source-2602-16681-vetime]]
