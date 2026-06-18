---
title: "E²-CSTP: Causal Spatio-Temporal Prediction via Multi-Modal Approach"
type: source-summary
tags:
  - spatio-temporal-forecasting
  - causal-inference
  - multi-modal
  - mamba
  - graph-neural-networks
  - backdoor-adjustment
created: 2026-06-18
last_updated: 2026-06-18
source_count: 1
confidence: medium
status: active
---

# E²-CSTP: Causal Spatio-Temporal Prediction via Multi-Modal Approach

该论文提出 E²-CSTP，一个融合多模态信息与因果推理的时空预测框架，解决多模态融合中的混淆偏差和计算效率问题。[^src-causal-st-prediction]

---

## 核心问题

现有时空预测方法面临三个挑战：

1. **多模态融合不足**：文本、图像等辅助模态提供重要的上下文信息，但简单拼接会放大未观测混淆因子（confounders）的影响。
2. **混淆偏差**：辅助模态 $E$（图像）和 $C$（文本）可能同时影响时空信号 $X_{st}$ 和目标 $Y_{st}$，引入后门路径 $X_{st} \leftarrow S \rightarrow Y_{st}$，导致虚假关联。
3. **计算效率**：Transformer 自注意力的复杂度为 $O(B \cdot T^2 \cdot N^2 \cdot d)$，在大规模时空图上不可承受。[^src-causal-st-prediction]

---

## 架构设计

E²-CSTP 由三个核心模块组成：[^src-causal-st-prediction]

### 1. 多模态融合（MMF）

- **特征提取**：BERT 提取文本特征，CNN 提取图像特征，时空序列经归一化处理。
- **跨模态注意力（CMA）**：计算时空序列与文本/图像之间的交叉注意力权重，实现模态间的语义对齐。
- **门控融合**：通过门控机制整合跨模态注意力输出，生成融合表示 $F_{fused}$。
- **图构建**：邻接矩阵 $A = \lambda A^{(0)} + (1 - \lambda)A_{SHAP}$，结合先验空间结构与 DeepSHAP 识别的因果区域。

### 2. 双分支因果推理

基于后门准则的因果干预。给定 SCM：$X_{st} = f_X(S, E, C)$，$Y_{st} = f_Y(X_{st}, S, E, C)$，估计干预分布：

$$P(Y_{st} | do(X_{st} = x), E, C) = \int_S P(Y_{st} | X_{st} = x, S = s_i, E, C) P(S = s_i | E, C) dS$$

干预调整公式为：

$$\hat{x} = x + x \odot W[\alpha_1 h(S) + \alpha_2 p(E) + \alpha_3 q(C)]$$

通过对抗训练使 $\frac{\partial \hat{x}}{\partial S} \to 0$，阻断后门路径，确保 $\hat{x} \perp\!\!\!\perp S \mid E, C$。[^src-causal-st-prediction]

### 3. 时空编解码器（STED）

- **空间编码**：多层 GCN 在自适应邻接矩阵上执行消息传递，捕获空间邻域关系。
- **时间编码**：Mamba（选择性状态空间模型）以线性复杂度 $O(B \cdot T \cdot N \cdot d)$ 替代 Transformer 的二次复杂度，捕获长程时序依赖。
- **融合**：残差连接 + LayerNorm：$X_{encoded} = \text{LayerNorm}(X_{spatial} + X_{temporal})$。
- **解码**：MLP 输出预测。

总复杂度为 $O(B \cdot T \cdot N^2 \cdot d)$（GCN 主导），显著优于 Transformer 的 $O(B \cdot T^2 \cdot N^2 \cdot d)$。

### 双分支训练

损失函数：$\mathcal{L}_{all} = \mathcal{L}_{pred} + \beta\mathcal{L}_{st} + (1 - \beta)\mathcal{L}_{mm}$，其中 $\mathcal{L}_{st}$ 监督原始 ST 分支，$\mathcal{L}_{mm}$ 监督因果调整后的多模态分支。[^src-causal-st-prediction]

---

## 实验

在 4 个数据集（Terra、BjTT、GreenEarthNet、BikeNYC）上与 9 个 SOTA 基线比较。[^src-causal-st-prediction]

### 主要发现

- **整体性能**：E²-CSTP 在所有数据集和指标上一致最优，MAE 相对第二好的方法提升 1.61%-9.66%。
- **消融实验**：去除文本特征、图像特征、DeepSHAP、因果推理、GCN 或 Mamba 中任一组件均导致性能下降。图像和文本对提供关键上下文，因果推理应对混淆条件，GCN 和 Mamba 分别负责空间和时间建模。
- **效率**：训练时间与单模态方法相当，远优于 UniST 和基于 LLM 的 FNF。STED 模块相比 Informer、Autoformer、FEDformer、iTransformer 等 Transformer 变体，精度提升 1.78%-5.45%，每 epoch 时间减少 17.37%-56.11%。
- **参数敏感性**：$\lambda$（图融合因子）在遥感任务上取 0.25（更多依赖因果图），在城市交通任务上取 0.5（两者平衡）；$\beta$（损失平衡因子）根据数据集外生影响强弱在 0.5-0.75 之间。

---

## 贡献与局限

### 贡献
1. 首个将因果干预整合到多模态时空预测中的框架
2. GCN+Mamba 混合架构实现线性复杂度的高效时空建模
3. 在 4 个数据集上全面超越 9 个 SOTA 基线

### 局限
- 仅处理文本、图像和时空序列三种模态，未涉及音频、LiDAR 等
- 因果推理依赖 DeepSHAP 近似，非严格因果发现
- 多模态数据对齐在实际部署中可能困难

## 引用

[^src-causal-st-prediction]: [[source-causal-st-prediction]]