---
title: "SLiDE: Structured Linear Dynamics for Forecasting with Exogenous Inputs"
type: source-summary
tags:
  - source
  - time-series-forecasting
  - exogenous
  - koopman-operator
  - linear-dynamics
  - ood-generalization
  - neurips-2026
created: 2026-11-24
last_updated: 2026-11-27
source_count: 1
confidence: medium
status: active
---

# SLiDE: Structured Linear Dynamics for Forecasting with Exogenous Inputs

带外生输入的时间序列预测里，主流深度模型靠交叉注意力或深层 MLP 把外生信号柔性融合进预测，但外生变量由外部过程产生，测试期一旦超出训练见过的取值范围、或驱动机制整个切换，高自由度的融合模块容易顺着历史里的虚假相关走错[^src-slide]。SLiDE 的核心巧思是把潜空间演化收紧成一个编码与预测共享的线性递推系统：过去阶段用一枚分块线性矩阵把历史观测与外生信号递推成当前潜状态，未来阶段复用同一枚矩阵、以加性外生推力向前滚动，非线性只留在输入嵌入与输出解码两端[^src-slide]。压低自由度换来偏移下的稳健：电力价格与零售基准上平均排名居首、推理开销与轻量 MLP 相当，在外生越界与换驱动迁移设置中降幅最大[^src-slide]。

## 核心做法

SLiDE（Structured Linear Dynamics with Exogenous inputs）可完全拆成标准算子组合：跨步共享 MLP 嵌入 + 分块对角线性循环编码 + 共享算子纯加性滚动 + 分块 MLP 解码求和[^src-slide]。三个设计决定：

1. **编码与滚动共享同一枚线性矩阵**：历史端把外生与目标观测逐时间步嵌入后，用少数互不交互子块（论文取 2 块）的分块对角线性矩阵循环递推出当前潜状态；预测端不做二次动力学拟合，直接复用同一枚矩阵滚动。论文自述这是 Koopman 思想的实用化——不做精确有限维 Koopman 模型，只借「线性潜演化」作归纳偏置[^src-slide]。
2. **外生信号纯加性注入**：过去步与未来步的外生变量走同一份逐时间步 MLP 嵌入，以加性推力进入递推；放弃随潜状态变化的控制输入，外生通道的自由度不随状态放大[^src-slide]。
3. **输出端非线性**：线性递推限制了中间潜态容量，于是放弃线性观测矩阵，各子块分别经 MLP 解码后求和回目标维度；长序列可先做不重叠 patching 压有效长度[^src-slide]。

## 证据

1. **定位与主结论**：8 项电力、零售与非线性系统任务上平均排名居首，越界外生样本较最强基线降错约 20%，换驱动迁移降约 30%[^src-slide]。
2. **真实涨点归因**：线性转移约束掉点最狠——换成两层 MLP 同分布更优但跨驱动与越界显著劣化，说明其专服务分布偏移；「编码与滚动共享」本身未被单独消融[^src-slide]。
3. **证据缺口**：偏移评估仅限小样本越界子集与单一迁移；合成多正弦激励（multisine）同分布任务不及传统系统辨识方法，且缺概率校准指标[^src-slide]。

## 局限

论文自述：适用性取决于目标系统能否被近似有限维潜线性动力学刻画，对低维 Koopman 近似难以覆盖或含连续谱的系统未做系统研究；外生输入缺失、噪声或时间错位会直接影响滚动；偏移评估仅限小样本越界子集与单一合成迁移[^src-slide]。

## 相关页面

- [[slide]] — 本文机制主页（分阶段数据流与设计轴对照矩阵）。
- [[mmckm]] / [[intent-discriminator-koopman]] — 同构/特例：SLiDE 是状态相关多模态 Koopman 族的线性加性简化特例。
- [[koopman-linearization-for-forecasting]] — 同槽位互斥：既有 Koopman 预测保留状态耦合控制项，SLiDE 仅用纯加性推力并强制共享算子。
- [[k2vae]] — 跨槽位正交：同属 Koopman 潜线性谱系，K²VAE 做无外生概率长期预测，SLiDE 做外生确定性点预测。
- [[tide]] — 同槽位互斥：TiDE 将历史与协变量展平进 dense residual MLP，SLiDE 以共享线性递推约束演化。

出处：Sebastian Pütz、Theodore Glavas、Benjamin Schäfer、Boris N. Oreshkin、Mark Coates（KIT / Helmholtz AI、McGill / Mila、Amazon Science），NeurIPS 2026，代码公开。

[^src-slide]: [[source-slide]]
