---
title: "Physics-Informed Position Encoding"
type: technique
tags:
  - position-encoding
  - physics-informed
  - transformer
  - multimodal
  - time-series
created: 2025-07-14
last_updated: 2026-08-08
source_count: 1
confidence: medium
status: active
---

# Physics-Informed Position Encoding

**Physics-Informed Position Encoding（物理知情位置编码）**是一种将物理元数据（时间戳、地理坐标等）嵌入 Transformer 位置编码的方法，由 [[pipe|PIPE]] 提出[^src-pipe]。

## 动机

传统位置编码（绝对、相对、RoPE、ALiBi 等）仅编码序列内 token 的位置关系，无法捕获跨实例共享的全局物理知识。在多模态时间序列预测中，卫星图像的每个像素天然关联着特定的时间和地理位置——这些物理上下文是全局知识（所有样本共享地球的经纬度关系和季节周期），但现有方法将其完全忽略[^src-pipe]。

## 两个核心机制

### 1. 物理知情位置索引

将图像 token 的 Position ID 从传统的序列/3D 索引替换为物理量：

- 时间维度：$t = t_{\text{day}} \times 24 + t_{\text{hour}}$，其中 $t_{\text{day}} \in [0,365]$，$t_{\text{hour}} \in [0,23]$
- 空间维度：图像 patch 中心对应的纬度和经度坐标

为避免与文本 token 的序列位置 ID 冲突，图像 token 的物理位置 ID 被映射到负值范围[^src-pipe]。

### 2. 变频率位置编码

修改标准正弦位置编码的波长参数，使不同物理变量使用不同的频率范围：

$$PE(pos, 2i) = \sin\left(\frac{2\pi \cdot pos}{p \cdot 10000^{2i/d_{\text{model}}}}\right)$$

其中 $p$ 为物理变量对应的波长：$p_{\text{day}}=366$、$p_{\text{hour}}=24$、$p_{\text{latitude}}=180$、$p_{\text{longitude}}=360$。这使得模型能通过频率区分不同物理变量的贡献[^src-pipe]。

## 与传统位置编码的对比

| 方法 | 编码对象 | 跨实例共享 | 物理信息 |
|------|----------|-----------|---------|
| 正弦 PE | 序列位置 | ❌ | ❌ |
| RoPE | 序列相对位置 | ❌ | ❌ |
| ALiBi | 序列偏置 | ❌ | ❌ |
| 3D PE (Qwen-VL) | 空间-时间位置 | ❌ | ❌ |
| **Physics-Informed PE** | **物理量（经纬度+时间）** | **✅** | **✅** |

## 与传统 PINN 的区别

Physics-Informed Position Encoding 代表了将物理知识注入神经网络的第三种范式[^src-pipe]：

| 范式 | 机制 | 代表方法 |
|------|------|---------|
| 损失约束型 | PDE 残差作为额外损失 | PI-MFM, Raissi et al. |
| 架构嵌入型 | PDE 离散形式作为网络层 | CTENet |
| **编码知情型** | **物理量注入位置编码** | **PIPE** |

编码知情型无需修改损失函数或架构，仅通过位置编码层的改动即可实现物理知识注入，论文称其为 a lightweight method[^src-pipe]。

## 应用场景

目前仅应用于台风预测（[[digital-typhoon-dataset|Digital Typhoon 数据集]]），但其设计原则上适用于任何视觉数据承载物理上下文的任务：气象预测、农业遥感、海洋监测等[^src-pipe]。

## 相关页面

- [[pipe]] — PIPE 模型
- [[source-pipe]] — 论文摘要
- [[rope]] — RoPE 位置编码（PIPE 的基础编码机制）
- [[physics-informed-neural-network]] — PINN 概述
- [[variant-frequency-positional-encoding]] — 变频率正弦编码详解

[^src-pipe]: [[source-pipe]]
