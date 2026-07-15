---
title: "Query-Conditioned Gaussian Splatting (QCGS)"
type: technique
tags:
  - gaussian-splatting
  - precipitation-nowcasting
  - implicit-neural-representation
  - satellite-station-fusion
  - radar-free
  - weather
created: 2026-07-16
last_updated: 2026-07-16
source_count: 1
confidence: medium
status: active
---

# Query-Conditioned Gaussian Splatting (QCGS)

QCGS（也称为 Station2Radar）是 Kim et al.（KAIST）在 ICLR 2026 提出的框架，融合自动气象站（AWS）稀疏观测与卫星亮温（BT）图像，生成无需雷达的高分辨率连续降水场[^src-qcgs]。

## 核心洞察

传统气象客观分析（Barnes 插值、Kriging）使用固定各向同性高斯核对站点观测加权求和。这恰好是 [[gaussian-splatting|Gaussian Splatting]] 的特殊情况——GS 将每个观测建模为具有可学习参数（振幅 + 各向异性协方差）的"高斯 blob"，支持分辨率无关渲染[^src-qcgs]。

$$f_\mathrm{GW}(\mathbf{x}) = \frac{\sum_i K_\sigma(\mathbf{x}-\mu_i) y_i}{\sum_j K_\sigma(\mathbf{x}-\mu_j)} \quad\text{vs}\quad f_\mathrm{GS}(\mathbf{x}) = \sum_i a_i K_{\Sigma_i}(\mathbf{x}-\mu_i)$$

## 三阶段流水线

### 1. Radar Point Proposal Network

- **卫星通路**：ConvNeXt U-Net（4 层编解码 + 跳跃连接）处理 GK2A IR 10.5 µm 亮温图像（2 km）
- **AWS 通路**：3 层 8 头 GAT 从稀疏、含缺失值的站点观测中提取鲁棒表示
- **融合**：AWS 表示通过交叉注意力注入 U-Net 解码器
- **输出**：粗糙代理降水场 $\hat{R}^t$ 和候选降雨位置集

### 2. Rainfall-Aware Point Sampling

仅在降雨掩码 $\mathcal{S}_t = \{\mathbf{x} \mid \hat{R}^t(\mathbf{x}) > \tau\}$ 内采样，概率分布为三项凸组合：

$$P_\mathrm{init}(\mathbf{x}) = \alpha G_{\mathcal{S}_t}(\mathbf{x}) + \beta U_{\mathcal{S}_t}(\mathbf{x}) + \gamma H(\mathbf{x})$$

- $G$：边缘梯度项（增强降水边界）
- $U$：均匀覆盖项（保证空间覆盖）
- $H$：温度 softmax 强降水项（优先关注高影响事件）
- 权重 0.3/0.4/0.3，NMS 去冗余，K=6000 最优

### 3. INR-based Gaussian Parameter Estimator

对每个查询点，5 层 MLP [[implicit-neural-representation|INR]]（hidden=128，正弦位置编码）通过交叉注意力预测高斯参数：

$$\theta^{(n)} = \{\sigma_x^{(n)}, \sigma_y^{(n)}, \rho^{(n)}, \alpha^{(n)}\}$$

- $(\sigma_x, \sigma_y, \rho)$ 定义各向异性协方差 $\Sigma^{(n)} \in \mathbb{S}^2_{++}$
- $\alpha^{(n)}$ 控制高斯振幅
- **AWS 锚定**：在有非零降雨的 AWS 站点，直接设 $\alpha^{(n)} = s^{(n)}$，作为硬约束

### 渲染与训练

最终降水场通过可微 2D Gaussian Splatting 合成：

$$\tilde{R}^t(\mathbf{x}) = \sum_{n=1}^N \alpha^{(n)} \exp\!\Big(-\tfrac{1}{2}(\mathbf{x}-\mu^{(n)})^\top \Sigma^{(n)^{-1}}(\mathbf{x}-\mu^{(n)})\Big)$$

损失 = MSE + $\lambda_\sigma\sum(\sigma_x+\sigma_y)$ + $\lambda_\alpha\sum\alpha^{(n)}$（正则化防止过度平滑），Adam lr=1e-4，余弦调度，8×H200。

## 关键性能

- 2 km 训练，0.5 km 评估：CSI 0.76 vs Kriging 0.50，FSS 0.96 vs 0.69
- 日累计降水：RMSE 6.68 vs IMERG 14.08，CC 0.95
- AWS 融合贡献 CSI +0.11，GS 额外 +0.03
- PSD 跨尺度最接近雷达，运营产品丢失高波数方差

## 局限性

依赖 AWS 站点存在，仅韩国区域验证，两阶段训练非端到端，无时序一致性（帧间），代码未开源[^src-qcgs]。

[^src-qcgs]: [[source-qcgs]]
