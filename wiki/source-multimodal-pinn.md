---
title: "A Multimodal Physics-Informed Neural Network Approach for Mean Radiant Temperature Modeling"
type: source-summary
tags:
  - physics-informed
  - multimodal
  - pinn
  - urban-climate
  - 2025
created: 2026-07-07
last_updated: 2026-07-14
source_count: 1
confidence: medium
status: active
---

# Multimodal PINN for Mean Radiant Temperature Modeling

**Authors:** Pouya Shaeri, Saud AlKhaled, Ariane Middel (Arizona State University, Kuwait University)

**Year:** 2025 | **arXiv:** 2503.08482

## 核心贡献

本文提出一种多模态物理信息神经网络（PINN）框架，用于估算平均辐射温度（Mean Radiant Temperature, Tmrt）——评估户外热舒适度的关键参数。该框架融合数值气象特征、鱼眼图像视觉特征和辐射传输物理方程，结合物理约束与深度学习，在城市热环境建模中实现高精度与可解释性。[^src-multimodal-pinn]

## 方法论

研究基于 MaRTy 数据集（亚利桑那州坦佩市 2016–2019 年期间 159 个地点的 1,130 条观测），包含气象参数、六方向辐射通量、建筑环境描述和鱼眼图像。

**架构设计：**
1. **多模态融合**：数值元数据（气温、湿度、风速、太阳角度等）经全连接层处理；六方向立方图经球面扭曲转为半球鱼眼投影，通过 SegFormer 天空遮罩 + ResNet-50（解冻最后 30 层微调）提取空间特征。
2. **短波辐射 MLP**：融合图像特征（2048 维）和元数据特征，3 隐藏层（128-256-128）。
3. **长波辐射 MLP**：仅处理数值元数据，3 隐藏层（128-256-128）。
4. **物理损失函数**：基于六方向 Tmrt 公式直接编码辐射热传输方程，惩罚偏离热力学平衡的预测。[^src-multimodal-pinn]

## 实验结果

- **阴影预测准确率**：94%，验证了鱼眼图像阴影估计的可靠性。
- **最佳配置（Multimodal PINN）**：RMSE = 3.50，R² = 0.88。
- **消融实验**：使用鱼眼预测阴影替代实测阴影后性能几乎不变，证明视觉特征可作为建筑环境元数据的有效替代。
- 在元数据不完全的场景下，PINN 显著优于纯神经网络（NN）和传统机器学习方法（XGBoost、Random Forest 等）。[^src-multimodal-pinn]

## 意义

该工作桥接了传统物理模型（ENVI-met、SOLWEIG）与纯数据驱动方法的鸿沟，在不依赖昂贵传感器和详细场地数据的情况下实现高精度 Tmrt 估算。鱼眼图像作为阴影信息的可靠代理，提升了模型的可扩展性和实际部署价值。[^src-multimodal-pinn]

## 局限

数据集限于沙漠干旱气候（亚利桑那），向温带/湿润气候的泛化性需进一步验证。PINN 框架对 GPU 资源要求较高，可能限制实时部署。六方向图像输入和多气候区域扩展是未来方向。[^src-multimodal-pinn]

## 交叉链接

- [[source-pi-mfm]] — PI-MFM：物理信息多模态基础模型用于 PDE 求解，与本工作同为 PINN 多模态扩展方向
- [[physics-informed-neural-network]] — PINN 概念（损失约束型 vs 架构嵌入型）
- [[source-ctenet]] — CTENet：架构嵌入型 PINN，将 ADR 方程直接嵌入网络前向传播（对比参照）

[^src-multimodal-pinn]: [[source-multimodal-pinn]]
