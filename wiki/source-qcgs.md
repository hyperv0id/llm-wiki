---
title: "Station2Radar: Query-Conditioned Gaussian Splatting for Precipitation Field"
type: source-summary
tags:
  - precipitation-nowcasting
  - gaussian-splatting
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

# Station2Radar: QCGS

Doyi Kim, Minseok Seo, Changick Kim (KAIST), ICLR 2026, arXiv:2603.00418。

## 核心贡献

提出 **Query-Conditioned Gaussian Splatting (QCGS)**，第一个将自动气象站（AWS）观测与卫星图像融合生成降水场、且无需雷达输入的框架。核心洞察：传统气象学中常用的高斯加权插值在数学上与 Gaussian Splatting 等价——前者使用固定各向同性核，后者允许可学习的各向异性核、自适应振幅和分辨率无关渲染[^src-qcgs]。

## 方法概要

QCGS 为三阶段流水线（两阶段训练）：

1. **Radar Point Proposal Network**：ConvNeXt U-Net 编码器-解码器处理卫星亮温图像，GAT 处理稀疏 AWS 观测，通过交叉注意力融合，输出粗糙代理降水场和候选降雨位置。
2. **Rainfall-Aware Point Sampling**：梯度项 + 均匀覆盖项 + 强降水温度 softmax 项的凸组合采样，仅在选择区域放置高斯核，避免非降雨区的无效计算。
3. **INR-based Gaussian Parameter Estimator**：5 层 MLP INR（含正弦位置编码）+ 交叉注意力，为每个查询点预测各向异性高斯参数 {σx, σy, ρ, α}。在 AWS 站点直接锚定 α 至观测值，作为硬约束。

## 关键结果

- 相比传统网格化降水产品（IMERG/MSWEP/GSMaP），RMSE 降低超 50%
- 2 km 训练、0.5 km 评估时，CSI 0.76 vs Kriging 0.50，FSS 0.96 vs 0.69
- AWS 融合贡献最大（CSI +0.11），Gaussian Splatting 额外 +0.03
- 消融：K=6000 查询点为精度-效率最优平衡
- PSD 分析：跨尺度最接近雷达频谱，运营产品在高波数处丢失方差

## 局限性

- 依赖 AWS 站点数据，站网稀疏区域（非洲、海洋）适用性有限
- 实验限于韩国区域（480×480 网格），全球扩展仍是开放挑战
- 两阶段训练，尚未端到端联合训练
- 当前仅处理"生成"而非时序预报，帧间缺乏时序一致性
- 未开源代码

[^src-qcgs]: [[source-qcgs]]
