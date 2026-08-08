---
title: "Physics-Informed Neural Network"
type: concept
tags:
  - physics-informed
  - pinn
  - pde
  - deep-learning
created: 2026-07-14
last_updated: 2026-08-08
source_count: 4
confidence: high
status: active
---

# Physics-Informed Neural Network (PINN)

Physics-Informed Neural Network（PINN）是将物理定律（通常以 PDE 形式）融入深度学习的一类方法，旨在提升模型在数据稀疏场景下的泛化能力和物理一致性。根据融入方式，PINN 可分两大范式。[^src-pi-mfm][^src-ctenet]

## 两大范式

### 1. 损失约束型（Loss-Constrained PINN）

将 PDE 残差作为额外的损失项加入训练目标，软约束预测结果满足物理方程。这是 Raissi et al.（2019）开创的经典范式。[^src-pi-mfm][^src-multimodal-pinn]

- **PI-MFM**：在预训练和微调阶段均施加 PDE 残差损失，实现跨 PDE 族的迁移。[^src-pi-mfm]
- **Multimodal PINN (Tmrt)**：六方向辐射传输方程作为物理损失函数。[^src-multimodal-pinn]

优势是实现简单，可直接加入任意网络；劣势是约束强度依赖损失权重调参，且物理知识本身若有误差会引入有害归纳偏置。[^src-pi-mfm][^src-multimodal-pinn]

### 2. 架构嵌入型（Architecture-Embedded PINN）

将 PDE 的离散形式直接构建为网络层或模块，物理约束是**硬编码**进前向传播的。CTENet 是这一范式的代表。[^src-ctenet]

- **CTENet**：将 ADR 方程以 FTCS 有限差分离散化为 Advection/Diffusion/Reaction 三个计算模块，嵌入欧拉 ADR 解码器。平流项用风矢量场显式计算，扩散项含可学习扩散系数 kθ，反应项以 sigmoid 门控气象特征实现软注意力式调制。[^src-ctenet]

优势是物理约束强、可解释性高；劣势是架构设计受限于特定 PDE 形式，不够通用。[^src-ctenet]

### 3. 编码知情型（Encoding-Informed PINN）

将物理元数据（时间戳、地理坐标等）注入模型的位置编码中，无需修改损失函数或架构[^src-pipe]。[[pipe|PIPE]] 是这一范式的代表：将图像 token 的 RoPE 位置 ID 替换为物理量（年日、小时、纬度、经度），配合[[variant-frequency-positional-encoding|变频率正弦编码]]实现物理知识注入[^src-pipe]。

优势是极轻量（仅改动位置编码层）[^src-pipe]；论文提出未来工作将融入物理定律或约束（如 PDE 约束）以提升可解释性与鲁棒性[^src-pipe]。（「训练开销几乎不变」「表达能力受限于位置编码容量」为 wiki 分析，论文未作断言。）


## 三种范式对比

| | 损失约束型 | 架构嵌入型 | 编码知情型 |
|---|---|---|---|
| 灵活性 | 高 | 低 | **最高** |
| 物理一致性 | 软约束 | 硬约束 | 软约束（频率域对齐） |
| 实现成本 | 中 | 高 | **极低** |
| 适用场景 | 已知 PDE | 已知 PDE 离散形式 | 已知物理量周期性 |
| 代表方法 | PI-MFM, Raissi | CTENet | **PIPE** |
## 相关页面

- [[ctenet]] — 架构嵌入型 PINN 实例
- [[advection-diffusion-reaction-equation]] — CTENet 嵌入的核心 PDE
- [[source-pi-mfm]]、[[source-multimodal-pinn]] — 损失约束型 PINN 实例

- [[pipe]] — PIPE 编码知情型实例
- [[physics-informed-position-encoding]] — 物理知情位置编码
- [[variant-frequency-positional-encoding]] — 变频率正弦编码

[^src-pi-mfm]: [[source-pi-mfm]]
[^src-multimodal-pinn]: [[source-multimodal-pinn]]
[^src-ctenet]: [[source-ctenet]]
[^src-pipe]: [[source-pipe]]
